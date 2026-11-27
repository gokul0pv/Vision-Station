//! ONNX Runtime, loaded from its DLL at start (no link-time dependency):
//! DirectML on any DirectX 12 graphics card, the CPU otherwise. Sessions are
//! loaded on first use, cached, and run one at a time each (DirectML does not
//! take concurrent runs of one session).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use ort::execution_providers::{CPUExecutionProvider, DirectMLExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

static INIT: OnceLock<Result<String, String>> = OnceLock::new();
/// A loaded model. On the CPU, ONNX Runtime runs one session from many
/// threads at once; DirectML takes one run at a time, so `lock` serialises.
pub struct Model {
    pub s: Session,
    pub gpu: bool,
    lock: Mutex<()>,
}

type Cache = Mutex<HashMap<PathBuf, Arc<Model>>>;
static SESSIONS: OnceLock<Cache> = OnceLock::new();

fn dll() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("VS_ORT_DLL") {
        return Some(PathBuf::from(p));
    }
    crate::paths::find(&["onnxruntime.dll", "runtime/onnxruntime.dll"])
}

/// The CPU unless asked otherwise (VS_DEVICE=gpu): a job's checks run in
/// parallel and ONNX Runtime serves them all at once on the CPU, while
/// DirectML takes one run at a time. Measured on the 172-check board job:
/// CPU 0.65 s an inspection, DirectML 1.2 s.
pub fn use_gpu() -> bool {
    match std::env::var("VS_DEVICE").as_deref() {
        Ok("gpu") | Ok("GPU") | Ok("dml") | Ok("directml") => true,
        Ok(_) => false,
        Err(_) => GPU.load(std::sync::atomic::Ordering::Relaxed),
    }
}

static GPU: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Settings > General > Run AI models on (read once at start).
pub fn set_gpu(on: bool) {
    GPU.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Threads inside one model run (measured: 8 beat 1, 2 and 4 on 28 cores).
fn intra_threads() -> usize {
    std::env::var("VS_ORT_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8))
}

/// Initialise once. Ok(device) or the reason inference is unavailable.
pub fn init() -> Result<String, String> {
    INIT.get_or_init(|| {
        let p = dll().ok_or("onnxruntime.dll not found beside the program (runtime/)")?;
        if let Some(dir) = p.parent() {
            // DirectML.dll beside it must be found by the loader
            let path = std::env::var("PATH").unwrap_or_default();
            std::env::set_var("PATH", format!("{};{}", dir.display(), path));
        }
        ort::init_from(p.to_string_lossy().to_string()).with_name("vision-next").commit().map_err(|e| e.to_string())?;
        Ok(if use_gpu() { "gpu (DirectML)".to_string() } else { "cpu".to_string() })
    }).clone()
}

pub fn session(path: &Path) -> Result<Arc<Model>, String> {
    init()?;
    let cache = SESSIONS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(s) = cache.lock().unwrap().get(path) {
        return Ok(s.clone());
    }
    if !path.exists() {
        return Err(format!("model not found: {}", path.display()));
    }
    let build = |gpu: bool| -> Result<Session, ort::Error> {
        let mut b = Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;
        if gpu {
            b = b.with_memory_pattern(false)?.with_parallel_execution(false)?
                .with_execution_providers([DirectMLExecutionProvider::default().build()])?;
        } else {
            b = b.with_execution_providers([CPUExecutionProvider::default().build()])?
                .with_intra_threads(intra_threads())?;
        }
        b.commit_from_file(path)
    };
    let (s, gpu) = match build(use_gpu()) {
        Ok(s) => (s, use_gpu()),
        Err(_) if use_gpu() => (build(false).map_err(|e| e.to_string())?, false),
        Err(e) => return Err(e.to_string()),
    };
    let s = Arc::new(Model { s, gpu, lock: Mutex::new(()) });
    cache.lock().unwrap().insert(path.to_path_buf(), s.clone());
    Ok(s)
}

/// One float input, first float output: (shape, data).
pub fn run(m: &Model, shape: &[usize], data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>), String> {
    let t = ort::value::Tensor::from_array((shape.to_vec(), data)).map_err(|e| e.to_string())?;
    let _one_at_a_time = if m.gpu { Some(m.lock.lock().unwrap()) } else { None };
    let sess = &m.s;
    let inputs = ort::inputs![t].map_err(|e| e.to_string())?;
    let out = sess.run(inputs).map_err(|e| e.to_string())?;
    let (shp, d) = out[0].try_extract_raw_tensor::<f32>().map_err(|e| e.to_string())?;
    Ok((shp.iter().map(|&v| v.max(0) as usize).collect(), d.to_vec()))
}

/// One float input, every float output: [(shape, data)] in the model's order.
pub fn run_all(m: &Model, shape: &[usize], data: Vec<f32>) -> Result<Vec<(Vec<usize>, Vec<f32>)>, String> {
    let t = ort::value::Tensor::from_array((shape.to_vec(), data)).map_err(|e| e.to_string())?;
    let _one_at_a_time = if m.gpu { Some(m.lock.lock().unwrap()) } else { None };
    let inputs = ort::inputs![t].map_err(|e| e.to_string())?;
    let out = m.s.run(inputs).map_err(|e| e.to_string())?;
    let mut v = vec![];
    for i in 0..out.len() {
        let (shp, d) = out[i].try_extract_raw_tensor::<f32>().map_err(|e| e.to_string())?;
        v.push((shp.iter().map(|&x| x.max(0) as usize).collect(), d.to_vec()));
    }
    Ok(v)
}

/// The first input's dimensions and the "names" metadata (YOLO class names).
pub fn describe(m: &Model) -> (Vec<i64>, Option<String>) {
    let sess = &m.s;
    let dims = sess.inputs.first().and_then(|i| i.input_type.tensor_dimensions().cloned()).unwrap_or_default();
    let names = sess.metadata().ok().and_then(|m| m.custom("names").ok().flatten());
    (dims, names)
}

/// Free the models nobody has used (memory back after a heavy job closes).
pub fn release_all() {
    if let Some(c) = SESSIONS.get() {
        c.lock().unwrap().clear();
    }
}

pub fn loaded() -> Vec<String> {
    SESSIONS.get().map(|c| c.lock().unwrap().keys().map(|k| k.file_name().unwrap_or_default().to_string_lossy().to_string()).collect()).unwrap_or_default()
}
