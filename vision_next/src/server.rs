//! The application's state and its local HTTP interface (127.0.0.1 only).
//! The window shows the same pages; a line PC can run it without a window.

use crate::camera::{self, CameraSettings, Hub};
use crate::engine::{self, Inspection, Project, Store};
use crate::img::Img;
use crate::station::{self, Clients, Light, PictureWriter};
use rayon::prelude::*;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use tiny_http::{Header, Method, Request, Response, Server};

const HISTORY: usize = 300;
const IMAGE_EXT: &[&str] = &["png", "jpg", "jpeg", "bmp", "tif", "tiff"];

pub struct Side {
    pub project: Arc<Project>,
    pub mode: String,
    pub files: Vec<PathBuf>,
    pub idx: usize,
    pub camera: usize,
    pub last: Option<Arc<Inspection>>,
    pub last_img: Option<Arc<Img>>,
    pub img_seq: u64,
    pub undo: Vec<Value>,
    pub redo: Vec<Value>,
    pub dirty: bool,
}

pub struct OpenApp {
    pub id: String,
    pub name: String,
    pub folder: PathBuf,
    pub single: bool,
    pub sides: Vec<Side>,
}

/// One part in the history: each side's result, a thumbnail, saved pictures.
pub struct HistItem {
    pub seq: u64,
    pub t: f64,
    pub verdict: String,
    pub ms: f64,
    pub barcode: Option<String>,
    pub sides: Vec<(String, Arc<Inspection>, Option<Arc<Vec<u8>>>, Vec<PathBuf>)>,
}

pub struct State {
    pub store: Store,
    pub open: Option<OpenApp>,
    pub continuous: bool,
    pub busy: bool,
    pub seq: u64,
    pub total: u64,
    pub passed: u64,
    pub failed: u64,
    pub history: VecDeque<HistItem>,
    pub last_cycle_ms: f64,
    pub barcode: Option<String>,
    pub barcode_note: String,
    pub note: String,
    pub locked: bool,
    jpeg: HashMap<(usize, String, usize, u64), Arc<Vec<u8>>>,
}

pub struct Shared {
    pub st: Mutex<State>,
    pub changed: Condvar,
    pub cams: Hub,
    pub started: Instant,
    pub settings: Mutex<Value>,
    pub light: Mutex<Light>,
    pub clients: Clients,
    line_stop: Mutex<Arc<Mutex<bool>>>,
    scan_stop: Mutex<Arc<Mutex<bool>>>,
    pictures: PictureWriter,
    pub line_state: Mutex<String>,
}

pub type App = Arc<Shared>;

fn list_images(folder: &str) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(folder) else { return vec![] };
    let mut v: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| {
        p.extension().map(|x| IMAGE_EXT.contains(&x.to_string_lossy().to_lowercase().as_str())).unwrap_or(false)
    }).collect();
    v.sort();
    v
}

fn side_from(p: Project) -> Side {
    let src = p.spec.get("image_source").cloned().unwrap_or(Value::Null);
    let mode = src.get("mode").and_then(|v| v.as_str()).unwrap_or("reference").to_string();
    let mut files = vec![];
    if mode == "folder" {
        files = list_images(src.get("folder").and_then(|v| v.as_str()).unwrap_or(""));
    } else if mode == "files" {
        files = src.get("paths").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(PathBuf::from)).collect()).unwrap_or_default();
    }
    let camera = src.get("camera").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    Side { project: Arc::new(p), mode, files, idx: 0, camera, last: None, last_img: None, img_seq: 0, undo: vec![], redo: vec![], dirty: false }
}

impl Clone for Project {
    fn clone(&self) -> Project {
        Project { id: self.id.clone(), name: self.name.clone(), folder: self.folder.clone(), spec: self.spec.clone(), reference: self.reference.clone(), ref_version: self.ref_version }
    }
}

pub fn new_app() -> App {
    let store = Store::new(crate::paths::projects());
    let settings = station::load(&crate::paths::settings_file());
    let cams = Hub::default();
    if let Ok(c) = serde_json::from_value::<Vec<CameraSettings>>(settings["cameras"].clone()) {
        *cams.settings.lock().unwrap() = c;
    }
    crate::infer::set_gpu(settings["general"]["device"].as_str() == Some("gpu"));
    let locked = settings["operator"]["enabled"].as_bool().unwrap_or(false);
    let app = Arc::new(Shared {
        st: Mutex::new(State {
            store, open: None, continuous: false, busy: false, seq: 0, total: 0, passed: 0, failed: 0,
            history: VecDeque::new(), last_cycle_ms: 0.0, barcode: None, barcode_note: String::new(), note: String::new(),
            locked, jpeg: HashMap::new(),
        }),
        changed: Condvar::new(),
        cams,
        started: Instant::now(),
        settings: Mutex::new(settings),
        light: Mutex::new(Light::new()),
        clients: Arc::new(Mutex::new(vec![])),
        line_stop: Mutex::new(Arc::new(Mutex::new(true))),
        scan_stop: Mutex::new(Arc::new(Mutex::new(true))),
        pictures: PictureWriter::new(),
        line_state: Mutex::new("off".into()),
    });
    apply_settings(&app);
    app
}

/// Make the settings take effect: light, line interface, barcode scanner, cameras.
fn apply_settings(app: &App) {
    let s = app.settings.lock().unwrap().clone();
    {
        let a = app.clone();
        let light = s["light"].clone();
        std::thread::spawn(move || a.light.lock().unwrap().configure(&light));
    }
    // line interface
    {
        let mut stop = app.line_stop.lock().unwrap();
        *stop.lock().unwrap() = true;
        let fresh = Arc::new(Mutex::new(false));
        *stop = fresh.clone();
        if s["line"]["enabled"].as_bool().unwrap_or(false) {
            let port = s["line"]["port"].as_u64().unwrap_or(3000) as u16;
            let a = app.clone();
            let answer: Arc<dyn Fn(&str) -> String + Send + Sync> = Arc::new(move |cmd: &str| line_command(&a, cmd));
            let msg = match station::line_server(port, app.clients.clone(), fresh, answer) {
                Ok(()) => format!("listening on port {}", port),
                Err(e) => e,
            };
            *app.line_state.lock().unwrap() = msg;
        } else {
            *app.line_state.lock().unwrap() = "off".into();
            for c in app.clients.lock().unwrap().drain(..) {
                let _ = c.shutdown(std::net::Shutdown::Both);
            }
        }
    }
    // serial barcode scanner
    {
        let mut stop = app.scan_stop.lock().unwrap();
        *stop.lock().unwrap() = true;
        let fresh = Arc::new(Mutex::new(false));
        *stop = fresh.clone();
        let b = &s["barcode"];
        if b["enabled"].as_bool().unwrap_or(false) && b["source"].as_str() == Some("serial") {
            let (tx, rx) = mpsc::channel::<String>();
            station::serial_reader(b["port"].as_str().unwrap_or("").to_string(), b["baud"].as_u64().unwrap_or(9600) as u32, tx, fresh);
            let a = app.clone();
            std::thread::spawn(move || {
                for code in rx {
                    let _ = scan(&a, &code);
                }
            });
        }
    }
    if let Ok(c) = serde_json::from_value::<Vec<CameraSettings>>(s["cameras"].clone()) {
        *app.cams.settings.lock().unwrap() = c;
        app.cams.reconfigure();
    }
}

/// A scanned code: checked against the rule, kept for the next part, and
/// (with auto run) the inspection started.
pub fn scan(app: &App, raw: &str) -> Result<Value, String> {
    let cfg = app.settings.lock().unwrap()["barcode"].clone();
    match station::barcode_check(&cfg, raw) {
        Ok(code) => {
            {
                let mut st = app.st.lock().unwrap();
                st.barcode = Some(code.clone());
                st.barcode_note = format!("{} scanned", code);
                st.seq += 1;
            }
            app.changed.notify_all();
            if cfg["auto_run"].as_bool().unwrap_or(true) {
                let a = app.clone();
                std::thread::spawn(move || {
                    let _ = run_once(&a, None);
                });
            }
            Ok(json!({"ok": true, "code": code}))
        }
        Err(why) => {
            {
                let mut st = app.st.lock().unwrap();
                st.barcode = None;
                st.barcode_note = format!("not good: {}", why);
                st.seq += 1;
            }
            app.changed.notify_all();
            Ok(json!({"ok": false, "why": why}))
        }
    }
}

/// The line's text protocol (Vision Station's): one command, one reply.
fn line_command(app: &App, cmd: &str) -> String {
    let mut parts = cmd.splitn(2, char::is_whitespace);
    let word = parts.next().unwrap_or("").to_uppercase();
    let arg = parts.next().unwrap_or("").trim().to_string();
    match word.as_str() {
        "PING" => "PONG\r\n".into(),
        "VERSION" => "VISIONSTATION 1\r\n".into(),
        "STATUS" => if app.st.lock().unwrap().busy { "BUSY\r\n".into() } else { "READY\r\n".into() },
        "CODE" => match scan(app, &arg) {
            Ok(v) if v["ok"].as_bool() == Some(true) => format!("CODE OK {}\r\n", v["code"].as_str().unwrap_or("")),
            Ok(v) => format!("CODE NG {}\r\n", v["why"].as_str().unwrap_or("")),
            Err(e) => format!("CODE NG {}\r\n", e),
        },
        "TRIGGER" => {
            let only = if arg.is_empty() { None } else {
                let st = app.st.lock().unwrap();
                match &st.open {
                    Some(o) => match o.sides.iter().position(|s| s.project.id.eq_ignore_ascii_case(&arg) || s.project.name.eq_ignore_ascii_case(&arg)) {
                        Some(i) => Some(i),
                        None => return format!("ERROR no job {}\r\n", arg),
                    },
                    None => return "ERROR no application open\r\n".into(),
                }
            };
            match run_once(app, only) {
                Ok(_) => results_lines(app, only),
                Err(e) if e.contains("already running") => "BUSY\r\n".into(),
                Err(e) => format!("ERROR {}\r\n", e),
            }
        }
        "GET" => {
            // job.tool.value
            let p: Vec<&str> = arg.splitn(3, '.').collect();
            if p.len() < 3 {
                return "ERROR GET job.tool.value\r\n".into();
            }
            let st = app.st.lock().unwrap();
            let r = st.open.as_ref().and_then(|o| o.sides.iter().find(|s| s.project.id == p[0] || s.project.name == p[0]))
                .and_then(|s| s.last.clone()).and_then(|r| r.tools.iter().find(|t| t.tool_id == p[1]).cloned());
            match r {
                Some(t) => {
                    let v = match p[2] { "status" => json!(t.status), "score" => json!(t.score), k => t.values.get(k).cloned().unwrap_or(Value::Null) };
                    if v.is_null() { "ERROR no such value\r\n".into() } else { format!("VALUE {}\r\n", v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string())) }
                }
                None => "ERROR no such tool or no result yet\r\n".into(),
            }
        }
        _ => "ERROR unknown command\r\n".into(),
    }
}

fn results_lines(app: &App, only: Option<usize>) -> String {
    let st = app.st.lock().unwrap();
    let Some(o) = &st.open else { return "ERROR no application open\r\n".into() };
    let mut out = String::new();
    let mut overall = "PASS";
    let mut n = 0;
    for (i, s) in o.sides.iter().enumerate() {
        if only.map(|x| x != i).unwrap_or(false) {
            continue;
        }
        if let Some(r) = &s.last {
            n += 1;
            if r.verdict != "PASS" {
                overall = if r.verdict == "ERROR" && overall == "PASS" { "INCOMPLETE" } else { "FAIL" };
            }
            let failed = if r.failed_tools.is_empty() { "-".to_string() } else { r.failed_tools.join(",") };
            out += &format!("RESULT {} {} {:.0} {}\r\n", s.project.id, r.verdict, r.total_ms, failed);
        }
    }
    out += &format!("DONE {} {}\r\n", overall, n);
    out
}

// ------------------------------------------------------------------ actions
enum Want {
    File(PathBuf),
    Camera(usize),
    Reference,
}

/// Inspect every side (or one), in parallel; store, save and announce.
pub fn run_once(app: &App, only: Option<usize>) -> Result<Value, String> {
    let t0 = Instant::now();
    let settings = app.settings.lock().unwrap().clone();
    let gate = settings["barcode"]["enabled"].as_bool().unwrap_or(false);
    let (jobs, barcode, app_name) = {
        let mut st = app.st.lock().unwrap();
        if st.busy {
            return Err("an inspection is already running".into());
        }
        if gate && st.barcode.is_none() {
            return Err("scan the part's barcode first".into());
        }
        let barcode = st.barcode.clone();
        let Some(open) = st.open.as_mut() else { return Err("no application open".into()) };
        let name = open.name.clone();
        let mut v = vec![];
        for (i, s) in open.sides.iter_mut().enumerate() {
            if only.map(|o| o != i).unwrap_or(false) {
                continue;
            }
            let want = match s.mode.as_str() {
                "camera" => Want::Camera(s.camera),
                "folder" | "files" if !s.files.is_empty() => {
                    let f = s.files[s.idx % s.files.len()].clone();
                    s.idx = (s.idx + 1) % s.files.len();
                    Want::File(f)
                }
                _ => Want::Reference,
            };
            v.push((i, s.project.clone(), want));
        }
        st.busy = true;
        (v, barcode, name)
    };
    // pictures: each camera grabbed once even when several sides read it,
    // with the light on for the picture
    let mut grabbed: HashMap<usize, Result<Arc<Img>, String>> = HashMap::new();
    let needs_camera = jobs.iter().any(|j| matches!(j.2, Want::Camera(_)));
    // the camera sides, for the light: their own level before, Keep adjusted after
    let cam_sides: Vec<(usize, String, Option<u32>, Option<Arc<Img>>)> = jobs.iter().filter(|j| matches!(j.2, Want::Camera(_)))
        .map(|j| (j.0, j.1.id.clone(), j.1.spec["light_level"].as_u64().map(|v| v as u32), j.1.reference.clone())).collect();
    if let Some((_, id, taught, _)) = cam_sides.first() {
        app.light.lock().unwrap().use_side_level(id, *taught);
    }
    // the light switched on just now: the picture waits one more frame to settle
    let switched = needs_camera && app.light.lock().unwrap().hold();
    for (_, _, w) in &jobs {
        if let Want::Camera(c) = w {
            grabbed.entry(*c).or_insert_with(|| app.cams.grab_fresh(*c, if switched { 2 } else { 1 }).map(Arc::new));
        }
    }
    if needs_camera {
        let a = app.clone();
        let after = settings["light"]["off_after_ms"].as_u64().unwrap_or(300);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(after));
            a.light.lock().unwrap().release();
        });
    }
    let one = |(i, p, w): (usize, Arc<Project>, Want)| -> (usize, Option<Arc<Img>>, Inspection) {
        let (img, src): (Result<Arc<Img>, String>, String) = match w {
            Want::File(f) => (Img::load(&f).map(Arc::new), f.file_name().unwrap_or_default().to_string_lossy().to_string()),
            Want::Camera(c) => (grabbed.get(&c).cloned().unwrap_or(Err("camera".into())), format!("camera {}", c)),
            Want::Reference => (p.reference.clone().ok_or("no picture: set an image source or a reference".to_string()), "reference".into()),
        };
        match img {
            Ok(im) => {
                let r = p.inspect(im.clone(), &src, None, barcode.clone());
                (i, Some(im), r)
            }
            Err(e) => (i, None, Inspection {
                id: engine::new_id(), project: p.id.clone(), verdict: "ERROR".into(), source: src, timestamp: engine::now(),
                total_ms: 0.0, image_size: [0, 0], aborted: Some(e), failed_tools: vec![], not_ported: 0, tools: vec![],
            }),
        }
    };
    let out: Vec<(usize, Option<Arc<Img>>, Inspection)> = if settings["general"]["parallel"].as_bool().unwrap_or(true) {
        jobs.into_par_iter().map(one).collect()
    } else {
        jobs.into_iter().map(one).collect()
    };
    let cycle = (t0.elapsed().as_secs_f64() * 1000.0 * 10.0).round() / 10.0;
    // Keep adjusted: the light's level for the next part, from this part's picture
    if settings["light"]["auto"].as_bool().unwrap_or(false) && settings["light"]["enabled"].as_bool().unwrap_or(false) {
        for (i, id, taught, reference) in &cam_sides {
            if let Some(img) = out.iter().find(|o| o.0 == *i).and_then(|o| o.1.clone()) {
                keep_adjusted(app, id, *taught, &img, reference.as_deref());
            }
        }
    }
    // thumbnails and saved pictures, off the lock
    let pic_cfg = settings["results"].clone();
    let mut extra: Vec<(Option<Arc<Vec<u8>>>, Vec<PathBuf>)> = vec![];
    for (_, img, r) in &out {
        let thumb = img.as_ref().map(|im| Arc::new(im.to_jpeg(480, 80)));
        let mut paths = vec![];
        if let Some(im) = img {
            if station::wanted(&pic_cfg, &r.verdict) {
                let (im, r2, cfg, job) = (im.clone(), r.clone(), pic_cfg.clone(), format!("{} {}", app_name, r.project));
                let root = crate::paths::projects().join("_results");
                let code = barcode.clone();
                // where they will be: Show in folder and Open picture use them
                let (a, o) = station::picture_paths(&cfg, &r2, &job, &root, code.as_deref());
                let planned: Vec<PathBuf> = a.into_iter().chain(o).collect();
                if app.pictures.submit(Box::new(move || { station::write_pictures(&cfg, &im, &r2, &job, &root, code.as_deref()); })) {
                    paths.extend(planned);
                }
            }
        }
        extra.push((thumb, paths));
    }
    let mut st = app.st.lock().unwrap();
    st.busy = false;
    let code_now = st.barcode.clone();
    let Some(open) = st.open.as_mut() else { return Err("closed while inspecting".into()) };
    let mut overall = "PASS";
    let mut per = vec![];
    let mut hist_sides = vec![];
    for ((i, img, r), (thumb, paths)) in out.into_iter().zip(extra) {
        let s = &mut open.sides[i];
        engine::log_result(&s.project.folder, &r, &paths, code_now.as_deref());
        match r.verdict.as_str() {
            "FAIL" | "ERROR" => overall = "FAIL",
            "WARN" if overall == "PASS" => overall = "WARN",
            _ => {}
        }
        per.push(json!({"side": s.project.id, "verdict": r.verdict, "ms": r.total_ms}));
        if img.is_some() {
            s.last_img = img;
            s.img_seq += 1;
        }
        let r = Arc::new(r);
        hist_sides.push((s.project.id.clone(), r.clone(), thumb, paths));
        s.last = Some(r);
    }
    st.total += 1;
    if overall == "FAIL" { st.failed += 1 } else { st.passed += 1 }
    st.seq += 1;
    st.last_cycle_ms = cycle;
    let seq = st.seq;
    let used_code = st.barcode.clone();
    st.history.push_back(HistItem { seq, t: engine::now(), verdict: overall.into(), ms: cycle, barcode: used_code, sides: hist_sides });
    while st.history.len() > HISTORY {
        st.history.pop_front();
    }
    if gate {
        st.barcode = None; // one scan, one part
        st.barcode_note = "scan the next part".into();
    }
    drop(st);
    app.changed.notify_all();
    if settings["line"]["broadcast"].as_bool().unwrap_or(false) {
        let lines = results_lines(app, only);
        station::broadcast(&app.clients, &lines);
    }
    Ok(json!({"seq": seq, "verdict": overall, "ms": cycle, "sides": per}))
}

pub fn continuous_loop(app: App) {
    std::thread::spawn(move || {
        let mut holding = false;
        loop {
            let on = app.st.lock().unwrap().continuous;
            if !on {
                if holding {
                    app.light.lock().unwrap().release();
                    holding = false;
                }
                std::thread::sleep(Duration::from_millis(120));
                continue;
            }
            if !holding {
                let _ = app.light.lock().unwrap().hold(); // on for the whole continuous run
                holding = true;
            }
            let t0 = Instant::now();
            if let Err(e) = run_once(&app, None) {
                let mut st = app.st.lock().unwrap();
                if e == "no application open" {
                    st.continuous = false;
                }
                st.note = e;
                drop(st);
                std::thread::sleep(Duration::from_millis(300));
            }
            let spent = t0.elapsed();
            if spent < Duration::from_millis(100) {
                std::thread::sleep(Duration::from_millis(100) - spent);
            }
        }
    });
}

fn side_index(open: &OpenApp, v: &Value) -> Result<usize, String> {
    let s = v.get("side").and_then(|x| x.as_str()).unwrap_or("");
    if s.is_empty() && open.sides.len() == 1 {
        return Ok(0);
    }
    open.sides.iter().position(|x| x.project.id == s).ok_or(format!("no side '{}'", s))
}

/// Change a side's recipe (copy-on-write: a running inspection keeps its own).
fn edit<F: FnOnce(&mut Value) -> Result<Value, String>>(app: &App, body: &Value, f: F) -> Result<Value, String> {
    let mut st = app.st.lock().unwrap();
    if st.locked {
        return Err("Operator mode: unlock to change inspections".into());
    }
    let open = st.open.as_mut().ok_or("no application open")?;
    let i = side_index(open, body)?;
    let side = &mut open.sides[i];
    let before = side.project.spec.clone();
    let mut p = (*side.project).clone();
    let out = f(&mut p.spec)?;
    side.undo.push(before);
    if side.undo.len() > 80 {
        side.undo.remove(0);
    }
    side.redo.clear();
    side.project = Arc::new(p);
    side.dirty = true;
    Ok(out)
}

fn tools_mut(spec: &mut Value) -> Result<&mut Vec<Value>, String> {
    let o = spec.as_object_mut().ok_or("bad project")?;
    if !o.contains_key("tools") {
        o.insert("tools".into(), json!([]));
    }
    o.get_mut("tools").and_then(|t| t.as_array_mut()).ok_or("bad tools".into())
}

fn base_params() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("enabled".into(), json!(true));
    m.insert("critical".into(), json!(true));
    m.insert("angle".into(), json!(0.0));
    m.insert("shape".into(), json!("rect"));
    m.insert("capture".into(), json!("default"));
    m
}

fn state_json(app: &App) -> Value {
    let st = app.st.lock().unwrap();
    let open = st.open.as_ref().map(|o| {
        json!({
            "id": o.id, "name": o.name, "single": o.single,
            "sides": o.sides.iter().map(|s| {
                let p = &s.project;
                let ref_size = p.reference.as_ref().map(|r| json!([r.w, r.h])).unwrap_or(Value::Null);
                json!({
                    "id": p.id, "name": p.name,
                    "tools": p.spec.get("tools").cloned().unwrap_or(json!([])),
                    "source": {"mode": s.mode, "count": s.files.len(), "camera": s.camera,
                               "folder": p.spec.pointer("/image_source/folder").cloned().unwrap_or(json!("")),
                               "paths": p.spec.pointer("/image_source/paths").cloned().unwrap_or(json!([]))},
                    "reference": ref_size, "ref_version": p.ref_version,
                    "img_seq": s.img_seq, "dirty": s.dirty, "can_undo": !s.undo.is_empty(), "can_redo": !s.redo.is_empty(),
                    "mm_per_px": p.spec.get("mm_per_px").cloned().unwrap_or(json!(0.0)),
                    "advanced": crate::advanced::is_trained(&p.spec, &p.folder),
                })
            }).collect::<Vec<_>>()
        })
    });
    let settings = app.settings.lock().unwrap();
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "apps": st.store.list(),
        "open": open,
        "continuous": st.continuous, "busy": st.busy, "seq": st.seq,
        "counts": {"total": st.total, "pass": st.passed, "fail": st.failed},
        "cycle_ms": st.last_cycle_ms, "barcode": st.barcode, "barcode_note": st.barcode_note, "note": st.note,
        "barcode_rule": if settings["barcode"]["enabled"].as_bool().unwrap_or(false) { json!(station::barcode_rule(&settings["barcode"])) } else { Value::Null },
        "barcode_keyboard": settings["barcode"]["enabled"].as_bool().unwrap_or(false) && settings["barcode"]["source"].as_str() != Some("serial"),
        "locked": st.locked, "operator": settings["operator"]["enabled"],
        "general": settings["general"],
        "ported": crate::tools::PORTED,
        "projects_folder": st.store.root.display().to_string(),
        "legacy": crate::paths::legacy_projects().is_some(),
    })
}

fn results_json(app: &App) -> Value {
    let st = app.st.lock().unwrap();
    let sides: Vec<Value> = st.open.as_ref().map(|o| o.sides.iter().map(|s| json!({
        "side": s.project.id, "img_seq": s.img_seq,
        "result": s.last.as_ref().map(|r| serde_json::to_value(&**r).unwrap_or(Value::Null)),
    })).collect()).unwrap_or_default();
    let hist: Vec<Value> = st.history.iter().rev().take(60).map(|h| json!({"seq": h.seq, "verdict": h.verdict, "ms": h.ms, "t": h.t})).collect();
    json!({"seq": st.seq, "continuous": st.continuous, "busy": st.busy, "cycle_ms": st.last_cycle_ms,
           "counts": {"total": st.total, "pass": st.passed, "fail": st.failed},
           "barcode": st.barcode, "barcode_note": st.barcode_note, "locked": st.locked,
           "history": hist, "sides": sides})
}

/// The side's recipe run on every training picture of a check: its verdict
/// against the picture's label.
fn evaluate_training(p: &Project, id: &str, okd: &Path, ngd: &Path) -> Vec<Value> {
    let mut rows = vec![];
    for (label, d) in [("ok", okd), ("ng", ngd)] {
        for f in list_images(&d.display().to_string()) {
            let Ok(img) = Img::load(&f) else { continue };
            let r = p.inspect(Arc::new(img), "validation", None, None);
            let (status, score, msg) = match r.tools.iter().find(|t| t.tool_id == id) {
                Some(t) => (t.status.clone(), t.score, t.message.clone()),
                None => ("fail".to_string(), 0.0, "did not run - an earlier step stopped the inspection".to_string()),
            };
            let passed = status == "pass";
            rows.push(json!({"name": f.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), "label": label,
                "status": status, "score": score, "message": msg, "correct": if label == "ok" { passed } else { !passed }}));
        }
    }
    rows
}

/// (text, others) a Verify text check should expect, from its box on the teach
/// picture: the clearest whole line -- lines cut by the box's edge and single
/// characters are passed over while a clear line exists (Vision Station's taught_text).
fn taught_text(p: &Project, id: &str) -> Result<(String, bool), String> {
    let img = p.reference.clone().ok_or("no reference picture")?;
    let mut tools = p.tools();
    let t = tools.iter_mut().find(|t| t.id == id).ok_or("no such check")?;
    let strip = t.p.get("strip_spaces").and_then(|x| x.as_bool()).unwrap_or(true);
    t.p.insert("expect_text".into(), json!(""));
    t.p.insert("expect_regex".into(), json!(""));
    t.p.insert("expect_contains".into(), json!(""));
    let r = p.inspect(img, "teach", Some(&tools), None);
    let tr = r.tools.iter().find(|x| x.tool_id == id).ok_or("the check did not run on the teach picture")?;
    if tr.status == "error" {
        return Err(tr.message.clone());
    }
    let bbox = |pts: &[[f64; 2]]| pts.iter().fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, q| [b[0].min(q[0]), b[1].min(q[1]), b[2].max(q[0]), b[3].max(q[1])]);
    let region = tr.graphics.iter().find(|g| g.role == "region").map(|g| bbox(&g.points));
    let norm = |x: &str| if strip { x.chars().filter(|c| !c.is_whitespace()).collect::<String>() } else { x.trim().to_string() };
    let lines: Vec<(String, bool)> = tr.graphics.iter().filter(|g| g.role != "region" && !g.label.trim().is_empty()).map(|g| {
        let b = bbox(&g.points);
        let cut = region.map(|r| b[0] <= r[0] + 1.5 || b[1] <= r[1] + 1.5 || b[2] >= r[2] - 1.5 || b[3] >= r[3] - 1.5).unwrap_or(false);
        (norm(&g.label), cut)
    }).collect();
    let mut cands: Vec<&(String, bool)> = lines.iter().filter(|l| l.0.chars().count() >= 2).collect();
    if cands.is_empty() {
        cands = lines.iter().filter(|l| !l.0.is_empty()).collect();
    }
    let Some(best) = cands.into_iter().max_by_key(|l| (!l.1, l.0.chars().count())) else { return Ok((String::new(), false)) };
    Ok((best.0.clone(), lines.len() > 1))
}

/// The part finder's network candidates for the last area searched: a new
/// sensitivity refilters them without running the network again.
static FIND_CACHE: Mutex<Option<(String, Arc<Vec<(crate::parts::Bx, f32)>>)>> = Mutex::new(None);

/// The model teaching run's state, as the page shows it.
static TEACH: Mutex<Value> = Mutex::new(Value::Null);
static TEACH_CANCEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn teach_set(f: impl FnOnce(&mut Value)) {
    let mut t = TEACH.lock().unwrap();
    if !t.is_object() {
        *t = json!({});
    }
    f(&mut t);
}

/// The Python that has Vision Station's packages (VS_PYTHON, else Python 3.12, else python).
fn teach_python() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("VS_PYTHON") {
        return Some(PathBuf::from(p));
    }
    let local = std::env::var("LOCALAPPDATA").ok().map(|l| PathBuf::from(l).join("Programs").join("Python").join("Python312").join("python.exe"));
    if let Some(p) = local.filter(|p| p.is_file()) {
        return Some(p);
    }
    Some(PathBuf::from("python"))
}

/// teach.py: beside the program, or in vision_next/ above target/release.
fn teach_script() -> Option<PathBuf> {
    let mut d = std::env::current_exe().ok()?.parent()?.to_path_buf();
    for _ in 0..4 {
        if d.join("teach.py").is_file() {
            return Some(d.join("teach.py"));
        }
        d = d.parent()?.to_path_buf();
    }
    None
}

/// The model teaching run: the pictures taken here (light stepped a little
/// brighter and darker, as Vision Station does), then Vision Station's own
/// training session on them (teach.py), its progress relayed to the page.
#[allow(clippy::too_many_arguments)]
fn teach_run(app: App, app_id: String, sid: String, folder: PathBuf, cam: usize, refsize: (usize, usize), taught: Option<u32>,
             python: PathBuf, script: PathBuf, size: String) {
    const N: usize = 16;
    const STEPS: [f64; 5] = [0.0, -0.08, 0.08, -0.04, 0.04];
    let work = folder.join("advanced").join("capture");
    let control = folder.join("advanced").join("control");
    let _ = std::fs::remove_dir_all(&work);
    let _ = std::fs::remove_dir_all(&control);
    let cancelled = || TEACH_CANCEL.load(std::sync::atomic::Ordering::SeqCst);
    let finish = |v: Value| {
        let _ = std::fs::remove_dir_all(&work);
        let _ = std::fs::remove_dir_all(&control);
        // the advanced folder, if only ours
        let adv = folder.join("advanced");
        if std::fs::read_dir(&adv).map(|mut d| d.next().is_none()).unwrap_or(false) {
            let _ = std::fs::remove_dir(&adv);
        }
        teach_set(|t| { if let (Some(o), Some(n)) = (t.as_object_mut(), v.as_object()) { for (k, x) in n { o.insert(k.clone(), x.clone()); } } t["running"] = json!(false); t["done"] = json!(true); });
    };
    let captured: Result<(), String> = (|| {
        std::fs::create_dir_all(&work).map_err(|e| e.to_string())?;
        std::fs::create_dir_all(&control).map_err(|e| e.to_string())?;
        let cfg = app.settings.lock().unwrap()["light"].clone();
        let base = { app.light.lock().unwrap().side_level(&sid, taught) };
        let lit = cfg["enabled"].as_bool().unwrap_or(false);
        if lit {
            app.light.lock().unwrap().hold();
        }
        let r = (|| -> Result<(), String> {
            let mut fails = 0;
            let mut k = 0;
            while k < N {
                if cancelled() {
                    return Err("cancelled".into());
                }
                if let (true, Some(b)) = (lit, base) {
                    let lv = ((b as f64) * (1.0 + STEPS[k % STEPS.len()])).round().clamp(1.0, 255.0) as u32;
                    let _ = app.light.lock().unwrap().send_level(lv);
                }
                let img = match app.cams.grab_fresh(cam, 2) {
                    Ok(i) => i,
                    Err(e) => {
                        fails += 1;
                        if fails >= 3 {
                            return Err(format!("The camera gave no picture: {}", e));
                        }
                        continue;
                    }
                };
                if (img.w, img.h) != refsize {
                    return Err(format!("The pictures are {}x{}, the reference {}x{}: the camera's settings changed since it was taken.", img.w, img.h, refsize.0, refsize.1));
                }
                img.save_png(&work.join(format!("good_{:02}.png", k + 1)))?;
                k += 1;
                teach_set(|t| { t["progress"] = json!(0.12 * k as f64 / N as f64); t["detail"] = json!(format!("{} of {}", k, N)); });
            }
            Ok(())
        })();
        if lit {
            let mut l = app.light.lock().unwrap();
            if let Some(b) = base {
                let _ = l.send_level(b);
            }
            l.release();
        }
        r
    })();
    if let Err(e) = captured {
        let c = e == "cancelled";
        finish(json!({"ok": false, "cancelled": c, "error": if c { Value::Null } else { json!(e) },
                      "message": if c { "Cancelled - everything it made was deleted" } else { "Not trained" }}));
        return;
    }
    // Vision Station's own session on the pictures
    let mut cmd = std::process::Command::new(&python);
    cmd.arg(&script).arg(&folder).arg(&work).arg(&control).arg(&size)
        .current_dir(script.parent().unwrap_or(Path::new(".")))
        .stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => { finish(json!({"ok": false, "error": format!("could not start {}: {}", python.display(), e), "message": "Not trained"})); return; }
    };
    let out = child.stdout.take().unwrap();
    let watch = {
        let control = control.clone();
        std::thread::spawn(move || {
            // Cancel: the session is told, and deletes what it made
            while !TEACH.lock().unwrap()["done"].as_bool().unwrap_or(false) {
                if TEACH_CANCEL.load(std::sync::atomic::Ordering::SeqCst) {
                    let _ = std::fs::write(control.join("cancel"), b"1");
                }
                std::thread::sleep(Duration::from_millis(300));
            }
        })
    };
    let mut last = json!({});
    for line in std::io::BufRead::lines(std::io::BufReader::new(out)).map_while(Result::ok) {
        if let Ok(v) = serde_json::from_str::<Value>(&line) {
            // the session's own capture (from the pictures) is quick: progress never goes back
            teach_set(|t| {
                let p = v["progress"].as_f64().unwrap_or(0.0).max(t["progress"].as_f64().unwrap_or(0.0));
                for k in ["stage", "message", "detail", "eta_s", "epoch", "epochs", "can_cancel", "error", "cancelled", "ok", "result"] {
                    if let Some(x) = v.get(k) { t[k] = x.clone(); }
                }
                t["progress"] = json!(p);
            });
            last = v;
        }
    }
    let _ = child.wait();
    let ok = last["ok"].as_bool().unwrap_or(false);
    if ok {
        let _ = open_app(&app, &app_id); // the job as the training left it: its model in use
    }
    finish(json!({"ok": ok, "cancelled": last["cancelled"].as_bool().unwrap_or(false),
                  "error": if ok { Value::Null } else { last.get("error").cloned().unwrap_or(json!("the training stopped")) },
                  "message": last["message"].clone()}));
    let _ = watch.join();
}

/// Change the light settings in place and keep them (the light follows, without switching).
fn light_setting(app: &App, f: impl FnOnce(&mut Value)) {
    let mut cur = app.settings.lock().unwrap();
    f(&mut cur["light"]);
    let _ = station::save(&crate::paths::settings_file(), &cur);
    let l = cur["light"].clone();
    drop(cur);
    app.light.lock().unwrap().sync(&l);
}

/// One Keep adjusted step after a side's picture (as Vision Station's auto_step):
/// the side's own level when it has one, else the Brightness setting.
fn keep_adjusted(app: &App, side: &str, taught: Option<u32>, img: &Img, reference: Option<&Img>) {
    let cfg = app.settings.lock().unwrap()["light"].clone();
    let Some((mean, clipped)) = station::light_measure(img) else { return };
    let target = station::light_target(&cfg, reference);
    let limit = station::light_clip_limit(&cfg, reference);
    let own = taught.is_some() || cfg["job_levels"][side].is_u64();
    let now = {
        let l = app.light.lock().unwrap();
        if own { l.side_level(side, taught) } else { Some(cfg["brightness"][0].as_u64().unwrap_or(255) as u32) }
    }.unwrap_or(255);
    let new = station::light_next_level(now, mean, clipped, target, limit);
    if new == now {
        return;
    }
    let side = side.to_string();
    light_setting(app, |l| {
        if own {
            if !l["job_levels"].is_object() {
                l["job_levels"] = json!({});
            }
            l["job_levels"][side.as_str()] = json!(new);
        } else {
            let mut b = l["brightness"].as_array().cloned().unwrap_or_else(|| vec![json!(255); 4]);
            for x in b.iter_mut() {
                *x = json!(new);
            }
            l["brightness"] = json!(b);
            l["set_brightness"] = json!(true);
        }
    });
}

/// (id, name, folder) of every side of the open application.
fn open_sides(app: &App) -> Vec<(String, String, PathBuf)> {
    let st = app.st.lock().unwrap();
    st.open.as_ref().map(|o| o.sides.iter().map(|s| (s.project.id.clone(), s.project.name.clone(), s.project.folder.clone())).collect()).unwrap_or_default()
}

/// Where result pictures are saved (Settings > Result pictures).
fn results_root(app: &App) -> PathBuf {
    let f = app.settings.lock().unwrap()["results"]["folder"].as_str().unwrap_or("").trim().to_string();
    if f.is_empty() { crate::paths::projects().join("_results") } else { PathBuf::from(f) }
}

/// Vision Station's own results folder (Settings > Results there), from the
/// registry where it keeps it -- read, never written.
fn vs_results_folder() -> Option<PathBuf> {
    let v = station::vs_picture_settings()?;
    let p = PathBuf::from(v["folder"].as_str()?.trim());
    p.is_dir().then_some(p)
}

/// Every folder saved result pictures may be in: Vision Next's own, Vision
/// Station's results folder, and each Vision Station job's own archive
/// (results/images), so pictures of parts inspected before are found too.
fn picture_roots(app: &App) -> Vec<PathBuf> {
    let mut roots = vec![results_root(app)];
    if let Some(p) = vs_results_folder() {
        roots.push(p);
    }
    let mut jobs = vec![crate::paths::projects()];
    if let Some(l) = crate::paths::legacy_projects() {
        jobs.push(l);
    }
    for base in jobs {
        let Ok(apps) = std::fs::read_dir(&base) else { continue };
        for a in apps.flatten() {
            let a = a.path();
            let mut sides = vec![a.clone()];
            if let Ok(rd) = std::fs::read_dir(a.join("sides")) {
                sides.extend(rd.flatten().map(|e| e.path()));
            }
            for s in sides {
                let im = s.join("results").join("images");
                if im.is_dir() {
                    roots.push(im);
                }
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    roots.retain(|r| r.is_dir() && seen.insert(r.canonicalize().unwrap_or_else(|_| r.clone())));
    roots
}

fn dirs_documents() -> PathBuf {
    std::env::var("USERPROFILE").map(|h| PathBuf::from(h).join("Pictures")).ok().filter(|p| p.is_dir())
        .unwrap_or_else(crate::paths::projects)
}

fn stamp() -> String {
    let t = engine::now();
    let (d, h, m, s, _) = engine::local_time(t);
    format!("{}_{:02}{:02}{:02}", d, h, m, s)
}

/// A logged run's first result picture that is still on disk.
fn logged_picture(r: &Value) -> Option<String> {
    let mut c: Vec<String> = r["pictures"].as_array().into_iter().flatten().filter_map(|p| p.as_str().map(String::from)).collect();
    if let Some(i) = r["image"].as_str().filter(|s| !s.is_empty()) {
        c.push(i.to_string());
    }
    c.into_iter().find(|p| Path::new(p).is_file())
}

fn history_json(app: &App, q: &HashMap<String, String>) -> Value {
    let sides = open_sides(app);
    let verdict = q.get("verdict").cloned().unwrap_or_default();
    let text = q.get("q").cloned().unwrap_or_default().to_lowercase();
    let limit: usize = q.get("limit").and_then(|v| v.parse().ok()).unwrap_or(120);
    let want_verdict = |v: &str| verdict.is_empty() || (verdict == "fail" && v != "PASS") || (verdict == "pass" && v == "PASS");
    let (mut items, session_ids, session_total): (Vec<Value>, std::collections::HashSet<String>, usize) = {
        let st = app.st.lock().unwrap();
        let ids = st.history.iter().flat_map(|h| h.sides.iter().map(|s| s.1.id.clone())).collect();
        let items = st.history.iter().rev().filter(|h| {
            want_verdict(&h.verdict)
                && (text.is_empty() || h.barcode.as_deref().unwrap_or("").to_lowercase().contains(&text)
                    || h.sides.iter().any(|s| s.0.to_lowercase().contains(&text) || s.1.source.to_lowercase().contains(&text)
                        || s.1.failed_tools.iter().any(|f| f.to_lowercase().contains(&text))))
        }).take(limit).map(|h| json!({
            "seq": h.seq, "t": h.t, "verdict": h.verdict, "ms": h.ms, "barcode": h.barcode,
            "sides": h.sides.iter().map(|s| json!({"side": s.0, "verdict": s.1.verdict, "ms": s.1.total_ms, "source": s.1.source,
                "failed": s.1.failed_tools, "thumb": s.2.is_some()})).collect::<Vec<_>>()
        })).collect();
        (items, ids, st.history.len())
    };
    // parts inspected before this session: Vision Next's own log and
    // Vision Station's run log, one card per side's run
    let mut logged = 0;
    let mut older: Vec<Value> = vec![];
    for (i, (id, name, folder)) in sides.iter().enumerate() {
        for r in crate::report::runs(folder, None) {
            if r["from"] == "next" && session_ids.contains(r["id"].as_str().unwrap_or("")) {
                continue;
            }
            logged += 1;
            let v = r["verdict"].as_str().unwrap_or("");
            let failed: Vec<String> = r["failed_tools"].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(String::from)).collect();
            let barcode = r["barcode"].as_str().unwrap_or("");
            let hay = [barcode, id, name, r["source"].as_str().unwrap_or(""), r["image"].as_str().unwrap_or("")];
            if !want_verdict(v) || !(text.is_empty() || hay.iter().any(|x| x.to_lowercase().contains(&text))
                || failed.iter().any(|f| f.to_lowercase().contains(&text))) {
                continue;
            }
            let t = r["timestamp"].as_f64().unwrap_or(0.0);
            older.push(json!({"log": {"side": i, "t": t, "from": r["from"]}, "t": t, "verdict": v, "ms": r["total_ms"],
                "barcode": if barcode.is_empty() { Value::Null } else { json!(barcode) },
                "app": if r["from"] == "vs" { "Vision Station" } else { "Vision Next" },
                "sides": [{"side": id, "verdict": v, "ms": r["total_ms"], "source": r["source"], "failed": failed, "thumb": false,
                           "picture": Value::Null, "_r": r}]}));
        }
    }
    items.extend(older);
    items.sort_by(|a, b| b["t"].as_f64().unwrap_or(0.0).partial_cmp(&a["t"].as_f64().unwrap_or(0.0)).unwrap_or(std::cmp::Ordering::Equal));
    items.truncate(limit);
    for it in items.iter_mut() {
        if let Some(s) = it["sides"].get_mut(0) {
            if let Some(r) = s.as_object_mut().and_then(|o| o.remove("_r")) {
                s["picture"] = json!(logged_picture(&r));
            }
        }
    }
    json!({"items": items, "total": session_total + logged})
}

/// One logged run in full, for its card's dialog.
fn history_logged(app: &App, q: &HashMap<String, String>) -> Result<Value, String> {
    let i: usize = q.get("side").and_then(|x| x.parse().ok()).ok_or("which side?")?;
    let t: f64 = q.get("t").and_then(|x| x.parse().ok()).ok_or("which run?")?;
    let from = q.get("from").cloned().unwrap_or_default();
    let (id, name, folder) = open_sides(app).into_iter().nth(i).ok_or("that side is not open")?;
    let r = crate::report::runs(&folder, None).into_iter()
        .find(|r| r["from"].as_str() == Some(from.as_str()) && (r["timestamp"].as_f64().unwrap_or(0.0) - t).abs() < 1e-6)
        .ok_or("that run is no longer in the log")?;
    Ok(json!({"side": id, "name": name, "picture": logged_picture(&r), "result": r}))
}

fn stats_json(app: &App) -> Value {
    let (rss, peak) = memory();
    let light = app.light.lock().unwrap();
    json!({"rss_mb": rss, "peak_mb": peak, "uptime_s": app.started.elapsed().as_secs(),
           "models": crate::infer::loaded(), "inference": crate::infer::init().map(|d| if crate::infer::use_gpu() { d } else { "cpu".into() }).unwrap_or_else(|e| format!("off: {}", e)),
           "camera": Hub::status(), "threads": rayon::current_num_threads(),
           "light": if light.enabled() { light.state.clone() } else { "off".into() },
           "line": app.line_state.lock().unwrap().clone(), "clients": app.clients.lock().unwrap().len()})
}

#[cfg(windows)]
fn memory() -> (f64, f64) {
    #[repr(C)]
    struct Pmc { cb: u32, faults: u32, peak_ws: usize, ws: usize, a: usize, b: usize, c: usize, d: usize, pagefile: usize, peak_pagefile: usize }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(h: isize, p: *mut Pmc, cb: u32) -> i32;
    }
    unsafe {
        let mut p: Pmc = std::mem::zeroed();
        p.cb = std::mem::size_of::<Pmc>() as u32;
        if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut p, p.cb) != 0 {
            return ((p.ws as f64 / 1048576.0 * 10.0).round() / 10.0, (p.peak_ws as f64 / 1048576.0 * 10.0).round() / 10.0);
        }
    }
    (0.0, 0.0)
}

#[cfg(not(windows))]
fn memory() -> (f64, f64) {
    (0.0, 0.0)
}

pub fn open_app(app: &App, id: &str) -> Result<(), String> {
    let mut st = app.st.lock().unwrap();
    let a = st.store.open(id)?;
    st.open = Some(OpenApp {
        id: a.id, name: a.name, folder: a.folder, single: a.single,
        sides: a.sides.into_iter().map(side_from).collect(),
    });
    st.jpeg.clear();
    st.seq += 1;
    drop(st);
    app.changed.notify_all();
    Ok(())
}

fn job_json_path(app: &App) -> Result<(PathBuf, bool), String> {
    let st = app.st.lock().unwrap();
    let o = st.open.as_ref().ok_or("no application open")?;
    Ok((o.folder.join("job.json"), o.single))
}

/// Training pictures for a check: <side>/training/<tool>/ok|ng/*.png
fn training_dir(folder: &Path, tool: &str, label: &str) -> PathBuf {
    folder.join("training").join(tool).join(label)
}

fn count_pictures(d: &Path) -> usize {
    std::fs::read_dir(d).map(|r| r.filter_map(|e| e.ok()).filter(|e| e.path().is_file()).count()).unwrap_or(0)
}

fn require_unlocked(app: &App) -> Result<(), String> {
    if app.st.lock().unwrap().locked { Err("Operator mode: unlock first".into()) } else { Ok(()) }
}

/// One API call: Ok(json) or Err(message).
fn api(app: &App, method: &Method, path: &str, query: &HashMap<String, String>, body: &[u8]) -> Result<Value, String> {
    let v: Value = if body.is_empty() { json!({}) } else { serde_json::from_slice(body).unwrap_or(json!({})) };
    let s = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    match (method, path) {
        (Method::Get, "/api/state") => Ok(state_json(app)),
        (Method::Get, "/api/results") => Ok(results_json(app)),
        (Method::Get, "/api/stats") => Ok(stats_json(app)),
        (Method::Get, "/api/history") => Ok(history_json(app, query)),
        (Method::Get, "/api/history/logged") => history_logged(app, query),
        (Method::Get, "/api/history/item") => {
            let seq: u64 = query.get("seq").and_then(|x| x.parse().ok()).unwrap_or(0);
            let st = app.st.lock().unwrap();
            let h = st.history.iter().find(|h| h.seq == seq).ok_or("no longer in the history")?;
            Ok(json!({"seq": h.seq, "t": h.t, "verdict": h.verdict, "barcode": h.barcode,
                      "sides": h.sides.iter().map(|s| json!({"side": s.0, "result": &*s.1,
                          "files": s.3.iter().filter(|p| p.exists()).map(|p| p.to_string_lossy()).collect::<Vec<_>>()})).collect::<Vec<_>>()}))
        }
        (Method::Get, "/api/report") => {
            let hours = query.get("hours").and_then(|x| x.parse::<f64>().ok()).filter(|h| *h > 0.0);
            Ok(crate::report::summary(&open_sides(app), hours))
        }
        (Method::Post, "/api/export_csv") => {
            let hours = v["hours"].as_f64().filter(|h| *h > 0.0);
            let sides = open_sides(app);
            if sides.is_empty() {
                return Err("open an application first".into());
            }
            let start = dirs_documents();
            let Some(folder) = crate::dialogs::browse("folder", "Export history to folder", "", &start, &[])? else { return Ok(json!({"files": []})) };
            let files = crate::report::export_csv(&sides, hours, &folder)?;
            Ok(json!({"files": files.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(), "folder": folder.to_string_lossy()}))
        }
        (Method::Get, "/api/pictures/find") => {
            let roots = picture_roots(app);
            let limit = query.get("limit").and_then(|x| x.parse().ok()).unwrap_or(1000usize).min(5000);
            let mut out = crate::report::find_pictures_in(&roots, query.get("q").map(|s| s.as_str()).unwrap_or(""), limit);
            out["root"] = json!(results_root(app).to_string_lossy());
            out["roots"] = json!(roots.iter().map(|r| r.to_string_lossy()).collect::<Vec<_>>());
            Ok(out)
        }
        (Method::Post, "/api/reveal") => {
            let p = PathBuf::from(s("path"));
            crate::report::reveal(&p, v["open"].as_bool().unwrap_or(false))?;
            Ok(json!({"ok": true}))
        }
        (Method::Post, "/api/save_picture") => {
            // the picture at full resolution, where the operator chooses
            let what = s("what");
            let side = s("side");
            let (img, name): (Option<Arc<Img>>, String) = match what.as_str() {
                "live" => {
                    let cam = v["camera"].as_u64().unwrap_or(0) as usize;
                    (Some(Arc::new(app.cams.grab(cam)?)), format!("camera{}_{}.png", cam, stamp()))
                }
                "hist" => {
                    let seq = v["seq"].as_u64().unwrap_or(0);
                    let st = app.st.lock().unwrap();
                    let h = st.history.iter().find(|h| h.seq == seq).ok_or("no longer in the history")?;
                    let sd = h.sides.iter().find(|x| x.0 == side).or(h.sides.first()).ok_or("no side")?;
                    // the saved original when there is one, else the saved annotated picture
                    let file = sd.3.iter().find(|p| p.to_string_lossy().contains("_original") && p.exists()).or(sd.3.iter().find(|p| p.exists())).cloned();
                    drop(st);
                    let Some(file) = file else { return Err("this part's picture was not saved: Settings > Result pictures keeps them".into()) };
                    let name = file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    let Some(to) = crate::dialogs::save_as("Save picture", &name, &dirs_documents(), &[])? else { return Ok(json!({"path": null})) };
                    std::fs::copy(&file, &to).map_err(|e| e.to_string())?;
                    return Ok(json!({"path": to.to_string_lossy()}));
                }
                _ => {
                    let st = app.st.lock().unwrap();
                    let o = st.open.as_ref().ok_or("no application open")?;
                    let i = side_index(o, &v)?;
                    let sd = &o.sides[i];
                    let img = if what == "reference" { sd.project.reference.clone() } else { sd.last_img.clone() };
                    (img, format!("{}_{}_{}.png", sd.project.id, what, stamp()))
                }
            };
            let img = img.ok_or("no picture to save yet")?;
            let filters = vec![("PNG picture".to_string(), vec!["png".to_string()]), ("JPEG picture".to_string(), vec!["jpg".to_string()]), ("BMP picture".to_string(), vec!["bmp".to_string()])];
            let Some(to) = crate::dialogs::save_as("Save picture", &name, &dirs_documents(), &filters)? else { return Ok(json!({"path": null})) };
            let ext = to.extension().and_then(|e| e.to_str()).unwrap_or("png").to_lowercase();
            match ext.as_str() {
                "jpg" | "jpeg" => std::fs::write(&to, img.to_jpeg(0, 95)).map_err(|e| e.to_string())?,
                "bmp" => img.save_png(&to)?, // the format follows the extension
                _ => img.save_png(&to)?,
            }
            Ok(json!({"path": to.to_string_lossy()}))
        }
        (Method::Get, "/api/wait") => {
            let since: u64 = query.get("since").and_then(|x| x.parse().ok()).unwrap_or(0);
            let st = app.st.lock().unwrap();
            let (st, _) = app.changed.wait_timeout_while(st, Duration::from_secs(25), |s| s.seq <= since).unwrap();
            drop(st);
            Ok(results_json(app))
        }
        (Method::Post, "/api/open") => {
            open_app(app, &s("app"))?;
            Ok(state_json(app))
        }
        (Method::Post, "/api/close") => {
            let mut st = app.st.lock().unwrap();
            st.open = None;
            st.continuous = false;
            st.jpeg.clear();
            st.history.clear();
            st.seq += 1;
            drop(st);
            crate::infer::release_all();
            app.cams.close_all();
            app.changed.notify_all();
            Ok(state_json(app))
        }
        (Method::Post, "/api/app/new") => {
            require_unlocked(app)?;
            let name = s("name");
            let id = app.st.lock().unwrap().store.create(if name.is_empty() { "New application" } else { &name })?;
            open_app(app, &id)?;
            Ok(state_json(app))
        }
        (Method::Post, "/api/side/new") => {
            require_unlocked(app)?;
            let (root, id) = {
                let st = app.st.lock().unwrap();
                let o = st.open.as_ref().ok_or("no application open")?;
                if o.single {
                    return Err("this application was made with one side: create a new application for several".into());
                }
                (st.store.root.clone(), o.id.clone())
            };
            let n = s("name");
            Store::new(root).add_side(&id, if n.is_empty() { "New side" } else { &n })?;
            open_app(app, &id)?;
            Ok(state_json(app))
        }
        (Method::Post, "/api/side/rename") | (Method::Post, "/api/side/remove") => {
            require_unlocked(app)?;
            let (jp, single) = job_json_path(app)?;
            let sid = s("side");
            if single {
                return Err("an application with one side keeps it".into());
            }
            let mut job: Value = serde_json::from_str(&std::fs::read_to_string(&jp).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            let sides = job["sides"].as_array_mut().ok_or("job.json has no sides")?;
            let i = sides.iter().position(|x| x["id"].as_str() == Some(&sid)).ok_or("no such side")?;
            if path.ends_with("rename") {
                sides[i]["name"] = json!(s("name"));
            } else {
                if sides.len() == 1 {
                    return Err("the last side cannot be removed".into());
                }
                sides.remove(i);
                // kept, not deleted: moved aside where it can be brought back by hand
                let folder = jp.parent().unwrap().join("sides");
                let _ = std::fs::rename(folder.join(&sid), folder.join(format!("_removed_{}_{}", sid, engine::now() as u64)));
            }
            std::fs::write(&jp, serde_json::to_string_pretty(&job).unwrap()).map_err(|e| e.to_string())?;
            let id = app.st.lock().unwrap().open.as_ref().map(|o| o.id.clone()).unwrap_or_default();
            open_app(app, &id)?;
            Ok(state_json(app))
        }
        (Method::Post, "/api/run") => {
            let only = {
                let st = app.st.lock().unwrap();
                match (&st.open, v.get("side").and_then(|x| x.as_str())) {
                    (Some(o), Some(_)) => Some(side_index(o, &v)?),
                    _ => None,
                }
            };
            run_once(app, only)?;
            Ok(results_json(app))
        }
        (Method::Post, "/api/continuous") => {
            let on = v.get("on").and_then(|x| x.as_bool()).unwrap_or(false);
            let mut st = app.st.lock().unwrap();
            st.continuous = on && st.open.is_some();
            st.seq += 1;
            drop(st);
            app.changed.notify_all();
            Ok(json!({"continuous": on}))
        }
        (Method::Post, "/api/reset_counts") => {
            let mut st = app.st.lock().unwrap();
            st.total = 0;
            st.passed = 0;
            st.failed = 0;
            st.history.clear();
            st.seq += 1;
            drop(st);
            app.changed.notify_all();
            Ok(json!({}))
        }
        (Method::Post, "/api/preview") => {
            let (p, img, tools) = {
                let st = app.st.lock().unwrap();
                let o = st.open.as_ref().ok_or("no application open")?;
                let i = side_index(o, &v)?;
                let side = &o.sides[i];
                let img = if s("on") == "last" { side.last_img.clone().or(side.project.reference.clone()) } else { side.project.reference.clone().or(side.last_img.clone()) };
                (side.project.clone(), img, side.project.tools())
            };
            let img = img.ok_or("no picture yet: set a reference or run once")?;
            let r = p.inspect(img, "preview", Some(&tools), None);
            Ok(serde_json::to_value(r).unwrap())
        }
        (Method::Post, "/api/tool/add") => {
            let ty = s("type");
            if ty.is_empty() {
                return Err("no check type".into());
            }
            edit(app, &v, |spec| {
                let tools = tools_mut(spec)?;
                let base = v.get("name").and_then(|x| x.as_str()).filter(|x| !x.is_empty()).map(|x| x.to_string()).unwrap_or(ty.clone());
                let mut n = tools.len() + 1;
                let mut id = format!("{}_{}", base, n);
                while tools.iter().any(|t| t.get("id").and_then(|x| x.as_str()) == Some(&id)) {
                    n += 1;
                    id = format!("{}_{}", base, n);
                }
                let mut params = base_params();
                if let Some(Value::Object(p)) = v.get("params") {
                    for (k, x) in p {
                        params.insert(k.clone(), x.clone());
                    }
                }
                for k in ["roi", "shape", "points", "angle"] {
                    if let Some(x) = v.get(k) {
                        params.insert(k.into(), x.clone());
                    }
                }
                let at = v.get("at").and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(tools.len()).min(tools.len());
                tools.insert(at, json!({"id": id, "type": ty, "params": params}));
                Ok(json!({"id": id}))
            })
        }
        (Method::Post, "/api/tool/duplicate") => {
            // one id or several ("ids"): each copy goes just beside its original
            let mut ids: Vec<String> = v["ids"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
            if ids.is_empty() {
                ids.push(s("id"));
            }
            edit(app, &v, |spec| {
                let tools = tools_mut(spec)?;
                let mut made = vec![];
                for id in &ids {
                    let i = tools.iter().position(|t| t["id"].as_str() == Some(id.as_str())).ok_or("no such check")?;
                    let mut t = tools[i].clone();
                    let mut n = 2;
                    let mut nid = format!("{}_copy", id);
                    while tools.iter().any(|x| x["id"].as_str() == Some(&nid)) {
                        nid = format!("{}_copy{}", id, n);
                        n += 1;
                    }
                    t["id"] = json!(nid);
                    if let Some(r) = t["params"]["roi"].as_array_mut() {
                        if r.len() >= 2 {
                            r[0] = json!(r[0].as_f64().unwrap_or(0.0) + 20.0);
                            r[1] = json!(r[1].as_f64().unwrap_or(0.0) + 20.0);
                        }
                    }
                    if let Some(ps) = t["params"]["points"].as_array_mut() {
                        for q in ps.iter_mut() {
                            if let Some(q) = q.as_array_mut() {
                                if q.len() >= 2 {
                                    q[0] = json!(q[0].as_f64().unwrap_or(0.0) + 20.0);
                                    q[1] = json!(q[1].as_f64().unwrap_or(0.0) + 20.0);
                                }
                            }
                        }
                    }
                    if let Some(p) = t["params"].as_object_mut() {
                        p.remove("locked");
                    }
                    tools.insert(i + 1, t);
                    made.push(nid);
                }
                Ok(json!({"id": made.first().cloned().unwrap_or_default(), "ids": made}))
            })
        }
        (Method::Post, "/api/tool/update_many") => {
            // several checks changed as one step (align, rotate, nudge): one undo
            edit(app, &v, |spec| {
                let tools = tools_mut(spec)?;
                let ups = v["updates"].as_object().ok_or("no updates")?;
                for (id, np) in ups {
                    let t = tools.iter_mut().find(|t| t.get("id").and_then(|x| x.as_str()) == Some(id.as_str())).ok_or(format!("no check '{}'", id))?;
                    let o = t.as_object_mut().unwrap();
                    if !o.contains_key("params") {
                        o.insert("params".into(), json!({}));
                    }
                    let p = o.get_mut("params").and_then(|p| p.as_object_mut()).unwrap();
                    if let Value::Object(np) = np {
                        for (k, x) in np {
                            if x.is_null() { p.remove(k); } else { p.insert(k.clone(), x.clone()); }
                        }
                    }
                }
                Ok(json!({"ids": ups.keys().collect::<Vec<_>>()}))
            })
        }
        (Method::Post, "/api/tool/update") => {
            let id = s("id");
            edit(app, &v, |spec| {
                let tools = tools_mut(spec)?;
                let t = tools.iter_mut().find(|t| t.get("id").and_then(|x| x.as_str()) == Some(&id)).ok_or(format!("no check '{}'", id))?;
                let o = t.as_object_mut().unwrap();
                if !o.contains_key("params") {
                    o.insert("params".into(), json!({}));
                }
                let p = o.get_mut("params").and_then(|p| p.as_object_mut()).unwrap();
                if let Some(Value::Object(np)) = v.get("params") {
                    for (k, x) in np {
                        if x.is_null() { p.remove(k); } else { p.insert(k.clone(), x.clone()); }
                    }
                }
                Ok(json!({"id": id}))
            })
        }
        (Method::Post, "/api/tool/rename") => {
            let (id, new) = (s("id"), s("new_id"));
            let new: String = new.trim().chars().map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' }).collect();
            if new.is_empty() {
                return Err("empty name".into());
            }
            edit(app, &v, |spec| {
                let tools = tools_mut(spec)?;
                if tools.iter().any(|t| t.get("id").and_then(|x| x.as_str()) == Some(&new)) {
                    return Err(format!("'{}' is taken", new));
                }
                // references to it by other checks follow the new name
                for t in tools.iter_mut() {
                    if let Some(p) = t["params"].as_object_mut() {
                        for k in ["source", "first", "second", "third", "fixture", "capture"] {
                            if p.get(k).and_then(|x| x.as_str()) == Some(&id) {
                                p.insert(k.into(), json!(new));
                            }
                        }
                    }
                }
                let t = tools.iter_mut().find(|t| t.get("id").and_then(|x| x.as_str()) == Some(&id)).ok_or("no such check")?;
                t["id"] = json!(new);
                Ok(json!({"id": new}))
            })
        }
        (Method::Post, "/api/tool/delete") => {
            let ids: Vec<String> = v.get("ids").and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|i| i.as_str().map(String::from)).collect()).unwrap_or_default();
            edit(app, &v, |spec| {
                let tools = tools_mut(spec)?;
                tools.retain(|t| !ids.iter().any(|i| t.get("id").and_then(|x| x.as_str()) == Some(i)));
                Ok(json!({"deleted": ids.len()}))
            })
        }
        (Method::Post, "/api/tool/move") => {
            let id = s("id");
            let to = v.get("to").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            edit(app, &v, |spec| {
                let tools = tools_mut(spec)?;
                let i = tools.iter().position(|t| t.get("id").and_then(|x| x.as_str()) == Some(&id)).ok_or("no such check")?;
                let t = tools.remove(i);
                tools.insert(to.min(tools.len()), t);
                Ok(json!({}))
            })
        }
        (Method::Post, "/api/undo") | (Method::Post, "/api/redo") => {
            let mut st = app.st.lock().unwrap();
            if st.locked {
                return Err("Operator mode: unlock first".into());
            }
            let o = st.open.as_mut().ok_or("no application open")?;
            let i = side_index(o, &v)?;
            let side = &mut o.sides[i];
            let undo = path.ends_with("undo");
            let prev = if undo { side.undo.pop().ok_or("nothing to undo")? } else { side.redo.pop().ok_or("nothing to redo")? };
            let mut p = (*side.project).clone();
            let now = std::mem::replace(&mut p.spec, prev);
            if undo { side.redo.push(now) } else { side.undo.push(now) }
            side.project = Arc::new(p);
            side.dirty = true;
            Ok(json!({}))
        }
        (Method::Post, "/api/save") => {
            let mut st = app.st.lock().unwrap();
            let o = st.open.as_mut().ok_or("no application open")?;
            for side in o.sides.iter_mut() {
                if side.dirty {
                    side.project.save()?;
                    side.dirty = false;
                }
            }
            Ok(json!({"saved": true}))
        }
        (Method::Post, "/api/revert") => {
            let id = app.st.lock().unwrap().open.as_ref().map(|o| o.id.clone()).ok_or("no application open")?;
            open_app(app, &id)?;
            Ok(state_json(app))
        }
        (Method::Post, "/api/project/set") => {
            // project-wide values: mm_per_px, name
            edit(app, &v, |spec| {
                if let Some(o) = spec.as_object_mut() {
                    if let Some(Value::Object(set)) = v.get("set") {
                        for (k, x) in set {
                            if ["mm_per_px", "name", "captures", "light_level"].contains(&k.as_str()) {
                                o.insert(k.clone(), x.clone());
                            }
                        }
                    }
                }
                Ok(json!({}))
            })
        }
        (Method::Post, "/api/browse") => {
            // Windows' Open dialog for a Browse button. With "side", a file is
            // given relative to that side's folder; one from elsewhere is
            // copied into its models folder, so packages carry it along.
            let side_dir = {
                let st = app.st.lock().unwrap();
                st.open.as_ref().and_then(|o| side_index(o, &v).ok().map(|i| o.sides[i].project.folder.clone()))
            };
            let fallback = side_dir.clone().unwrap_or_else(crate::paths::projects);
            let filters: Vec<(String, Vec<String>)> = v["filters"].as_array().map(|a| a.iter().filter_map(|f| {
                let name = f[0].as_str()?.to_string();
                let exts = f[1].as_array()?.iter().filter_map(|e| e.as_str().map(String::from)).collect();
                Some((name, exts))
            }).collect()).unwrap_or_default();
            let mut current = s("current");
            if let (Some(d), false) = (&side_dir, current.is_empty()) {
                if std::path::Path::new(&current).is_relative() {
                    current = d.join(&current).to_string_lossy().into_owned();
                }
            }
            let kind = s("kind");
            let title = if s("title").is_empty() { if kind == "folder" { "Choose a folder".to_string() } else { "Choose a file".to_string() } } else { s("title") };
            let picked = crate::dialogs::browse(&kind, &title, &current, &fallback, &filters)?;
            let Some(p) = picked else { return Ok(json!({"path": null})) };
            if kind != "folder" && v["side"].is_string() {
                if let Some(d) = side_dir {
                    if let Ok(rel) = p.strip_prefix(&d) {
                        return Ok(json!({"path": rel.to_string_lossy().replace('\\', "/")}));
                    }
                    let models = d.join("models");
                    std::fs::create_dir_all(&models).map_err(|e| e.to_string())?;
                    let name = p.file_name().ok_or("no file name")?;
                    std::fs::copy(&p, models.join(name)).map_err(|e| format!("could not copy {}: {}", p.display(), e))?;
                    // a .onnx model may keep its weights in a .data file beside it
                    let data = p.with_extension(format!("{}.data", p.extension().and_then(|e| e.to_str()).unwrap_or("")));
                    if data.exists() {
                        let _ = std::fs::copy(&data, models.join(data.file_name().unwrap()));
                    }
                    return Ok(json!({"path": format!("models/{}", name.to_string_lossy()), "copied": true}));
                }
            }
            Ok(json!({"path": p.to_string_lossy()}))
        }
        (Method::Post, "/api/source") => {
            let mode = s("mode");
            let folder = s("folder");
            let camera = v.get("camera").and_then(|x| x.as_u64()).unwrap_or(0);
            let files: Vec<String> = v.get("files").and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|i| i.as_str().map(String::from)).collect()).unwrap_or_default();
            if mode == "folder" && list_images(&folder).is_empty() {
                return Err(format!("no pictures in '{}'", folder));
            }
            edit(app, &v, |spec| {
                spec["image_source"] = json!({"mode": mode, "folder": folder, "paths": files, "camera": camera});
                Ok(json!({}))
            })?;
            let mut st = app.st.lock().unwrap();
            let o = st.open.as_mut().ok_or("no application open")?;
            let i = side_index(o, &v)?;
            let p = (*o.sides[i].project).clone();
            let old = &mut o.sides[i];
            let (last, last_img, seq, undo, redo) = (old.last.take(), old.last_img.take(), old.img_seq, std::mem::take(&mut old.undo), std::mem::take(&mut old.redo));
            let mut ns = side_from(p);
            ns.last = last;
            ns.last_img = last_img;
            ns.img_seq = seq;
            ns.undo = undo;
            ns.redo = redo;
            ns.dirty = true;
            o.sides[i] = ns;
            Ok(json!({}))
        }
        (Method::Post, "/api/reference") => {
            let mut st = app.st.lock().unwrap();
            if st.locked {
                return Err("Operator mode: unlock first".into());
            }
            let o = st.open.as_mut().ok_or("no application open")?;
            let i = side_index(o, &v)?;
            let side = &mut o.sides[i];
            let img = side.last_img.clone().ok_or("no picture yet: run once first")?;
            let mut p = (*side.project).clone();
            p.set_reference(&img)?;
            side.project = Arc::new(p);
            side.dirty = true;
            st.jpeg.clear();
            Ok(json!({}))
        }
        (Method::Post, "/api/reference/upload") => {
            require_unlocked(app)?;
            let im = Img::decode(body)?;
            let mut st = app.st.lock().unwrap();
            let o = st.open.as_mut().ok_or("no application open")?;
            let q = json!({"side": query.get("side").cloned().unwrap_or_default()});
            let i = side_index(o, &q)?;
            let side = &mut o.sides[i];
            let mut p = (*side.project).clone();
            p.set_reference(&im)?;
            side.project = Arc::new(p);
            side.dirty = true;
            st.jpeg.clear();
            Ok(json!({"size": [im.w, im.h]}))
        }
        (Method::Post, "/api/picture/upload") => {
            // inspect a picture dropped on the window, once
            let im = Arc::new(Img::decode(body)?);
            let (p, i) = {
                let st = app.st.lock().unwrap();
                let o = st.open.as_ref().ok_or("no application open")?;
                let i = side_index(o, &json!({"side": query.get("side").cloned().unwrap_or_default()}))?;
                (o.sides[i].project.clone(), i)
            };
            let r = p.inspect(im.clone(), query.get("name").map(|s| s.as_str()).unwrap_or("dropped picture"), None, None);
            let mut st = app.st.lock().unwrap();
            if let Some(o) = st.open.as_mut() {
                let side = &mut o.sides[i];
                side.last_img = Some(im);
                side.img_seq += 1;
                side.last = Some(Arc::new(r));
            }
            st.seq += 1;
            drop(st);
            app.changed.notify_all();
            Ok(results_json(app))
        }
        (Method::Get, "/api/import/list") => {
            let from = crate::paths::legacy_projects().ok_or("Vision Station's projects folder was not found")?;
            Ok(json!({"folder": from.display().to_string(), "apps": engine::list_in(&from)}))
        }
        (Method::Post, "/api/import") => {
            require_unlocked(app)?;
            let from = crate::paths::legacy_projects().ok_or("Vision Station's projects folder was not found")?;
            let id = app.st.lock().unwrap().store.import(&from, &s("id"))?;
            open_app(app, &id)?;
            Ok(state_json(app))
        }
        (Method::Post, "/api/package/import") => {
            require_unlocked(app)?;
            let files = crate::npz::unzip(body)?;
            // the application folder: the one holding job.json or project.json
            let top = files.iter().map(|f| f.0.clone()).filter(|n| n.ends_with("job.json") || n.ends_with("project.json"))
                .min_by_key(|n| n.matches('/').count()).ok_or("no application in this package")?;
            let prefix = top.rsplit_once('/').map(|x| format!("{}/", x.0)).unwrap_or_default();
            let name = prefix.trim_end_matches('/').rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or("imported").to_string();
            let root = app.st.lock().unwrap().store.root.clone();
            let mut id = name.clone();
            let mut n = 2;
            while root.join(&id).exists() {
                id = format!("{}_{}", name, n);
                n += 1;
            }
            for (fname, data) in files.iter().filter(|f| f.0.starts_with(&prefix) && !f.0.ends_with('/')) {
                let rel = &fname[prefix.len()..];
                if rel.split('/').any(|c| c == "..") {
                    continue;
                }
                let dst = root.join(&id).join(rel);
                if let Some(d) = dst.parent() {
                    std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
                }
                std::fs::write(&dst, data).map_err(|e| e.to_string())?;
            }
            open_app(app, &id)?;
            Ok(state_json(app))
        }
        (Method::Get, "/api/settings") => {
            let s = app.settings.lock().unwrap().clone();
            let mut out = s.clone();
            out["operator"]["password_set"] = json!(!s["operator"]["password_sha256"].as_str().unwrap_or("").is_empty());
            out["operator"]["password_sha256"] = json!("");
            out["file"] = json!(crate::paths::settings_file().display().to_string());
            Ok(out)
        }
        (Method::Post, "/api/settings/restore") => {
            // one page's settings back to their defaults, keeping where things are
            require_unlocked(app)?;
            let sec = s("section");
            let key = if sec == "pictures" { "results".to_string() } else { sec.clone() };
            let keep: &[&str] = match key.as_str() {
                "general" => &[],
                "results" => &["folder", "subfolder"],
                "line" => &["enabled", "port"],
                "light" => &["enabled", "port", "channels", "job_levels"],
                "barcode" => &["source", "port", "baud", "folder"],
                "cameras" => &[],
                _ => return Err(format!("no defaults for '{}'", sec)),
            };
            {
                let mut cur = app.settings.lock().unwrap();
                if key == "cameras" {
                    let kept: Vec<Value> = cur["cameras"].as_array().cloned().unwrap_or_default().iter().map(|c| json!({"ip": c["ip"].clone()})).collect();
                    cur["cameras"] = json!(kept);
                } else {
                    let mut fresh = station::defaults()[key.as_str()].clone();
                    for k in keep {
                        if let Some(v) = cur[key.as_str()].get(*k) {
                            fresh[*k] = v.clone();
                        }
                    }
                    cur[key.as_str()] = fresh;
                }
                station::save(&crate::paths::settings_file(), &cur)?;
            }
            apply_settings(app);
            Ok(json!({"restored": sec}))
        }
        (Method::Post, "/api/settings") => {
            require_unlocked(app)?;
            {
                let mut cur = app.settings.lock().unwrap();
                let mut patch = v.clone();
                if let Some(o) = patch.as_object_mut() {
                    o.remove("file");
                    if let Some(op) = o.get_mut("operator").and_then(|x| x.as_object_mut()) {
                        op.remove("password_sha256");
                        op.remove("password_set");
                    }
                }
                // a list replaces a list (cameras)
                if let Some(c) = patch.get("cameras").cloned() {
                    cur["cameras"] = c;
                }
                station::merge(&mut cur, &patch);
                if patch.get("results").map(|r| r.is_object()).unwrap_or(false) {
                    // chosen here: no longer follows Vision Station's
                    cur["results"]["chosen"] = json!(true);
                    cur["results"]["from_vs"] = json!(false);
                }
                station::save(&crate::paths::settings_file(), &cur)?;
            }
            apply_settings(app);
            Ok(json!({"saved": true}))
        }
        (Method::Get, "/api/serial/ports") => Ok(json!(station::serial_ports())),
        (Method::Post, "/api/camera/list") => {
            let ips: Vec<String> = app.cams.settings.lock().unwrap().iter().map(|c| c.ip.clone()).chain(std::iter::once(s("ip"))).collect();
            Ok(json!(camera::list_devices(&ips)?))
        }
        (Method::Post, "/api/camera/test") => {
            let idx = v.get("camera").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let t0 = Instant::now();
            let _ = app.light.lock().unwrap().hold();
            let r = app.cams.grab(idx);
            app.light.lock().unwrap().release();
            let im = r?;
            Ok(json!({"size": [im.w, im.h], "ms": t0.elapsed().as_millis() as u64, "refused": app.cams.missed(idx)}))
        }
        (Method::Get, "/api/camera/values") => {
            let idx = query.get("camera").and_then(|x| x.parse().ok()).unwrap_or(0usize);
            app.cams.values(idx)
        }
        (Method::Post, "/api/camera/balance_once") => {
            let idx = v["camera"].as_u64().unwrap_or(0) as usize;
            let cam = app.cams.get(idx)?;
            cam.lock().unwrap().balance_once()?;
            // the camera balances on the frames that follow
            let _ = app.cams.grab_fresh(idx, 4);
            let vals = app.cams.values(idx).unwrap_or(Value::Null);
            Ok(json!({"ok": true, "white_balance": vals["white_balance"]}))
        }
        (Method::Get, "/api/camera/ip") => {
            let idx = query.get("camera").and_then(|x| x.parse().ok()).unwrap_or(0usize);
            let cam = app.cams.get(idx)?;
            let ip = cam.lock().unwrap().ip_config();
            Ok(json!({"ip": ip}))
        }
        (Method::Post, "/api/camera/keep_ip") => {
            require_unlocked(app)?;
            let idx = v["camera"].as_u64().unwrap_or(0) as usize;
            let cam = app.cams.get(idx)?;
            let ip = cam.lock().unwrap().store_ip()?;
            Ok(json!({"ip": ip}))
        }
        (Method::Post, "/api/light/adjust") => {
            // Adjust now: the level that gives the side's picture its teach picture's
            // brightness, found from the camera with the light held on
            require_unlocked(app)?;
            let (cam, reference, mode) = {
                let st = app.st.lock().unwrap();
                let o = st.open.as_ref().ok_or("no application open")?;
                let i = side_index(o, &v)?;
                (o.sides[i].camera, o.sides[i].project.reference.clone(), o.sides[i].mode.clone())
            };
            if mode != "camera" {
                return Err("this side does not take its pictures from a camera".into());
            }
            let cfg = app.settings.lock().unwrap()["light"].clone();
            if !cfg["enabled"].as_bool().unwrap_or(false) {
                return Err("switch on Control the light first".into());
            }
            let target = station::light_target(&cfg, reference.as_deref());
            let limit = station::light_clip_limit(&cfg, reference.as_deref());
            app.light.lock().unwrap().hold();
            std::thread::sleep(Duration::from_millis(cfg["on_before_ms"].as_u64().unwrap_or(150)));
            let res = station::light_auto_level(
                |lv| app.light.lock().unwrap().send_level(lv),
                || app.cams.grab_fresh(cam, 2),
                target, limit);
            app.light.lock().unwrap().release();
            let (level, mean, clipped, tries) = res?;
            light_setting(app, |l| {
                l["brightness"] = json!([level, level, level, level]);
                l["set_brightness"] = json!(true);
            });
            Ok(json!({"level": level, "grey_pct": mean * 100.0, "white_pct": clipped * 100.0, "target_pct": target * 100.0,
                      "tries": tries.iter().map(|t| json!([t.0, t.1 * 100.0, t.2 * 100.0])).collect::<Vec<_>>()}))
        }
        (Method::Post, "/api/light/forget") => {
            // a side's Keep adjusted level dropped: it starts again from its own
            require_unlocked(app)?;
            let side = s("side");
            light_setting(app, |l| { if let Some(o) = l["job_levels"].as_object_mut() { o.remove(&side); } });
            Ok(json!({"ok": true}))
        }
        (Method::Post, "/api/light") => {
            let cfg = app.settings.lock().unwrap()["light"].clone();
            let mut l = app.light.lock().unwrap();
            l.sync(&cfg);
            match s("action").as_str() {
                "on" => l.switch(true)?,
                "off" => l.switch(false)?,
                "set" => l.set_level(v["channel"].as_u64().unwrap_or(1) as u8, v["value"].as_u64().unwrap_or(255) as u32)?,
                "read" => { l.read()?; }
                _ => return Err("light: on, off, set or read".into()),
            }
            Ok(json!({"state": l.state, "levels": l.levels}))
        }
        (Method::Post, "/api/barcode/scan") => scan(app, &s("code")),
        (Method::Post, "/api/barcode") => {
            let code = s("code");
            let mut st = app.st.lock().unwrap();
            st.barcode = if code.trim().is_empty() { None } else { Some(code.trim().to_string()) };
            st.seq += 1;
            Ok(json!({"barcode": st.barcode}))
        }
        (Method::Post, "/api/operator/lock") => {
            let has = !app.settings.lock().unwrap()["operator"]["password_sha256"].as_str().unwrap_or("").is_empty();
            if !has {
                return Err("set a supervisor password first (Settings > Operator)".into());
            }
            let mut st = app.st.lock().unwrap();
            st.locked = true;
            st.seq += 1;
            drop(st);
            app.changed.notify_all();
            Ok(json!({"locked": true}))
        }
        (Method::Post, "/api/operator/unlock") => {
            let want = app.settings.lock().unwrap()["operator"]["password_sha256"].as_str().unwrap_or("").to_string();
            if !want.is_empty() && station::sha256(s("password").as_bytes()) != want {
                return Err("wrong password".into());
            }
            let mut st = app.st.lock().unwrap();
            st.locked = false;
            st.seq += 1;
            drop(st);
            app.changed.notify_all();
            Ok(json!({"locked": false}))
        }
        (Method::Post, "/api/operator/password") => {
            let mut cur = app.settings.lock().unwrap();
            let have = cur["operator"]["password_sha256"].as_str().unwrap_or("").to_string();
            if !have.is_empty() && station::sha256(s("old").as_bytes()) != have {
                return Err("the present password is wrong".into());
            }
            let new = s("new");
            cur["operator"]["password_sha256"] = json!(if new.is_empty() { String::new() } else { station::sha256(new.as_bytes()) });
            station::save(&crate::paths::settings_file(), &cur)?;
            Ok(json!({"saved": true}))
        }
        (Method::Post, "/api/train/add") | (Method::Post, "/api/train/list") | (Method::Post, "/api/train/clear") | (Method::Post, "/api/train/run")
        | (Method::Post, "/api/train/remove") | (Method::Post, "/api/train/move") | (Method::Post, "/api/train/evaluate") | (Method::Post, "/api/train/tune") => {
            require_unlocked(app)?;
            let (p, last, tool) = {
                let st = app.st.lock().unwrap();
                let o = st.open.as_ref().ok_or("no application open")?;
                let i = side_index(o, &v)?;
                let side = &o.sides[i];
                let tool = side.project.tools().into_iter().find(|t| t.id == s("id")).ok_or("no such check")?;
                (side.project.clone(), side.last_img.clone(), tool)
            };
            let (okd, ngd) = (training_dir(&p.folder, &tool.id, "ok"), training_dir(&p.folder, &tool.id, "ng"));
            match path {
                "/api/train/add" => {
                    let label = if s("label") == "ng" { "ng" } else { "ok" };
                    let img = if s("from") == "reference" { p.reference.clone() } else { last.or(p.reference.clone()) }.ok_or("no picture to add")?;
                    let d = training_dir(&p.folder, &tool.id, label);
                    std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
                    img.save_png(&d.join(format!("{}.png", engine::new_id())))?;
                }
                "/api/train/clear" => {
                    let _ = std::fs::remove_dir_all(p.folder.join("training").join(&tool.id));
                }
                "/api/train/remove" => {
                    // one picture out of the training set (only from this check's own folders)
                    let name = s("name");
                    let label = if s("label") == "ng" { "ng" } else { "ok" };
                    if name.contains(['/', '\\']) || name.contains("..") {
                        return Err("not a training picture".into());
                    }
                    std::fs::remove_file(training_dir(&p.folder, &tool.id, label).join(&name)).map_err(|e| e.to_string())?;
                }
                "/api/train/move" => {
                    // a picture to the other column (OK <-> NG), as Vision Station's "Move to NG"
                    let name = s("name");
                    let (from, to) = if s("label") == "ng" { ("ng", "ok") } else { ("ok", "ng") };
                    if name.contains(['/', '\\']) || name.contains("..") {
                        return Err("not a training picture".into());
                    }
                    let d = training_dir(&p.folder, &tool.id, to);
                    std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
                    std::fs::rename(training_dir(&p.folder, &tool.id, from).join(&name), d.join(&name)).map_err(|e| e.to_string())?;
                }
                "/api/train/evaluate" | "/api/train/tune" => {
                    // the whole recipe on every training picture, as Vision Station's
                    // evaluate_tool: an OK judged NG is a false reject, an NG judged OK an escape
                    let rows = evaluate_training(&p, &tool.id, &okd, &ngd);
                    let false_rejects = rows.iter().filter(|r| r["label"] == "ok" && r["correct"] == false).count();
                    let escapes = rows.iter().filter(|r| r["label"] == "ng" && r["correct"] == false).count();
                    let mut out = json!({"rows": rows, "false_rejects": false_rejects, "escapes": escapes,
                                         "ok": count_pictures(&okd), "ng": count_pictures(&ngd)});
                    if path == "/api/train/tune" {
                        let param = match tool.ty.as_str() { "presence" | "align" => "min_score", _ => return Err(format!("{} has no single score threshold to tune", tool.ty)) };
                        let ok: Vec<f64> = rows.iter().filter(|r| r["label"] == "ok").filter_map(|r| r["score"].as_f64()).collect();
                        let ng: Vec<f64> = rows.iter().filter(|r| r["label"] == "ng").filter_map(|r| r["score"].as_f64()).collect();
                        if ok.is_empty() || ng.is_empty() {
                            return Err("tuning needs at least one OK and one NG picture".into());
                        }
                        let lo_ok = ok.iter().cloned().fold(f64::INFINITY, f64::min);
                        let hi_ng = ng.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let margin = lo_ok - hi_ng;
                        let old = tool.p.get(param).cloned().unwrap_or(Value::Null);
                        let mut new = old.clone();
                        if margin > 0.0 {
                            let v2 = ((lo_ok + hi_ng) / 2.0 * 1000.0).round() / 1000.0;
                            new = json!(v2);
                            let id = tool.id.clone();
                            edit(app, &v, |spec| {
                                let tools = tools_mut(spec)?;
                                let t = tools.iter_mut().find(|t| t["id"].as_str() == Some(id.as_str())).ok_or("no such check")?;
                                t["params"][param] = json!(v2);
                                Ok(json!({}))
                            })?;
                        }
                        out["tune"] = json!({"param": param, "old": old, "new": new, "lowest_ok": lo_ok, "highest_ng": hi_ng,
                                             "margin": margin, "separates": margin > 0.0});
                    }
                    return Ok(out);
                }
                "/api/train/run" => {
                    let load = |d: &Path| -> Vec<Img> { list_images(&d.display().to_string()).iter().filter_map(|f| Img::load(f).ok()).collect() };
                    let (ok, ng) = (load(&okd), load(&ngd));
                    let pp = p.clone();
                    let all = pp.tools();
                    // every picture through the side's own checks up to this one, so the region follows the part
                    let before: Vec<_> = all.iter().take_while(|t| t.id != tool.id).filter(|t| t.ty == "align" || t.ty == "fixture_npoint").cloned().collect();
                    let ctx_for = move |im: &Img| -> crate::model::Ctx {
                        let mut c = crate::model::Ctx::new(pp.reference.clone(), pp.folder.clone(), 0.0);
                        for t in &before {
                            let _ = crate::tools::run(t, im, &mut c);
                        }
                        c
                    };
                    let report = crate::tools::more::train_surface(&tool, &p.folder, &ctx_for, &ok, &ng)?;
                    return Ok(json!({"report": report, "ok": count_pictures(&okd), "ng": count_pictures(&ngd)}));
                }
                _ => {}
            }
            let names = |d: &Path| -> Vec<String> { let mut v: Vec<String> = list_images(&d.display().to_string()).iter().filter_map(|f| f.file_name().map(|n| n.to_string_lossy().into_owned())).collect(); v.sort(); v };
            Ok(json!({"ok": count_pictures(&okd), "ng": count_pictures(&ngd),
                      "files": {"ok": names(&okd), "ng": names(&ngd)},
                      "trained": crate::tools::more::model_path(&tool, &p.folder).exists()}))
        }
        (Method::Post, "/api/tool/teach_text") => {
            // Verify text expects what it reads on the teach picture (Read text: what it contains).
            // force: also over text typed in by hand -- the Expect what each reads now button
            require_unlocked(app)?;
            let force = v["force"].as_bool().unwrap_or(false);
            let ids: Vec<String> = v["ids"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
            let p = {
                let st = app.st.lock().unwrap();
                let o = st.open.as_ref().ok_or("no application open")?;
                o.sides[side_index(o, &v)?].project.clone()
            };
            let mut taught = vec![];
            let mut notes = vec![];
            let mut ups: Vec<(String, String, String)> = vec![];
            for id in &ids {
                let Some(t) = p.tools().into_iter().find(|t| &t.id == id) else { continue };
                if t.ty != "ocv" && t.ty != "ocr" {
                    continue;
                }
                let gs = |k: &str| t.p.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
                if !force && (!gs("expect_regex").is_empty() || (!gs("expect_text").is_empty() && gs("expect_text") != gs("expect_auto"))) {
                    continue; // typed in by hand: never replaced
                }
                match taught_text(&p, id) {
                    Ok((text, others)) if !text.is_empty() => {
                        notes.push(format!("{} expects '{}'{}", id, text, if others { "; other text in the box is ignored" } else { "" }));
                        ups.push((id.clone(), t.ty.clone(), text.clone()));
                        taught.push(json!({"id": id, "text": text}));
                    }
                    Ok(_) => notes.push(format!("No text read in the box of {} - fit the box round the marking, or type the text under Settings", id)),
                    Err(e) => notes.push(format!("{}: could not read the text to teach it: {}", id, e)),
                }
            }
            if !ups.is_empty() {
                edit(app, &v, |spec| {
                    let tools = tools_mut(spec)?;
                    for (id, ty, text) in &ups {
                        if let Some(t) = tools.iter_mut().find(|t| t["id"].as_str() == Some(id.as_str())) {
                            if ty == "ocv" {
                                t["params"]["expect_text"] = json!(text);
                                t["params"]["expect_auto"] = json!(text);
                            } else {
                                t["params"]["expect_contains"] = json!(text);
                            }
                        }
                    }
                    Ok(json!({}))
                })?;
            }
            Ok(json!({"taught": taught, "notes": notes}))
        }
        (Method::Post, "/api/find/parts") | (Method::Post, "/api/find/like") | (Method::Post, "/api/find/text")
        | (Method::Post, "/api/find/read_box") | (Method::Post, "/api/find/add") => {
            // Find parts and text on the side's reference picture (Vision Station's part finder)
            require_unlocked(app)?;
            let (p, img) = {
                let st = app.st.lock().unwrap();
                let o = st.open.as_ref().ok_or("no application open")?;
                let sd = &o.sides[side_index(o, &v)?];
                (sd.project.clone(), sd.project.reference.clone())
            };
            let img = img.ok_or("set a reference picture first")?;
            let rect = |k: &str| -> Option<[f64; 4]> {
                let a = v[k].as_array()?;
                (a.len() == 4).then(|| [a[0].as_f64().unwrap_or(0.0), a[1].as_f64().unwrap_or(0.0), a[2].as_f64().unwrap_or(0.0), a[3].as_f64().unwrap_or(0.0)])
            };
            match path {
                "/api/find/parts" => {
                    if !crate::parts::available() {
                        return Err(format!("the part-finding model is missing ({})", crate::parts::model_path().display()));
                    }
                    // a turned area is searched in an upright copy of it
                    let ang = v["area_angle"].as_f64().unwrap_or(0.0);
                    let turned = rect("area").filter(|_| ang.abs() > 1e-6).map(|a| crate::parts::turned_area(&img, a, ang));
                    let (img, area) = match &turned { Some(t) => (std::sync::Arc::new(t.img.clone()), None), None => (img, rect("area")) };
                    let key = format!("{}:{}:{:?}:{}", p.id, p.ref_version, rect("area"), ang);
                    let t0 = std::time::Instant::now();
                    let cached = FIND_CACHE.lock().unwrap().as_ref().filter(|c| c.0 == key).map(|c| c.1.clone());
                    let raw = match cached {
                        Some(r) => r,
                        None => {
                            let r = std::sync::Arc::new(crate::parts::area_candidates(&img, area, &|_, _| {})?);
                            *FIND_CACHE.lock().unwrap() = Some((key, r.clone()));
                            r
                        }
                    };
                    let mut out = crate::parts::filter(&img, area, &raw, v["sensitivity"].as_f64().unwrap_or(0.5) as f32);
                    if let Some(t) = &turned {
                        t.back_all(&mut out);
                    }
                    out["ms"] = json!(t0.elapsed().as_secs_f64() * 1000.0);
                    out["device"] = json!(crate::infer::init().unwrap_or_default());
                    Ok(out)
                }
                "/api/find/like" => {
                    let ex = rect("example").ok_or("draw a box round one part first")?;
                    let found = crate::parts::find_similar(&img, ex, rect("area"), v["min_score"].as_f64().unwrap_or(0.6) as f32, true);
                    Ok(json!({"found": found}))
                }
                "/api/find/text" => {
                    let ang = v["area_angle"].as_f64().unwrap_or(0.0);
                    match rect("area").filter(|_| ang.abs() > 1e-6) {
                        Some(a) => {
                            let t = crate::parts::turned_area(&img, a, ang);
                            let mut out = crate::parts::find_text(&t.img, None)?;
                            t.back_all(&mut out);
                            Ok(out)
                        }
                        None => crate::parts::find_text(&img, rect("area")),
                    }
                }
                "/api/find/read_box" => {
                    let (text, conf) = crate::parts::read_box(&img, rect("box").ok_or("no box")?, v["angle"].as_f64().unwrap_or(0.0))?;
                    Ok(json!({"text": text, "score": conf}))
                }
                _ => {
                    // the kept boxes as checks, in one step (one Ctrl+Z takes them all back)
                    let mode = s("mode");
                    let existing: Vec<String> = p.tools().iter().map(|t| t.id.clone()).collect();
                    let kept = v["parts"].as_array().cloned().unwrap_or_default();
                    let size = (img.w, img.h);
                    let new = if mode == "parts" {
                        crate::parts::presence_tools(&kept, &v["names"], &v["thresholds"], &existing, size, v["seated"].as_bool().unwrap_or(true))
                    } else {
                        crate::parts::text_tools(&kept, &mode, &existing, size, v["min_confidence"].as_f64(), &v["defaults"])
                    };
                    // the check type's own defaults first, what the finder set over them
                    let defaults = v["defaults"].clone();
                    let new: Vec<Value> = new.into_iter().map(|mut t| {
                        if mode == "parts" {
                            let mut pr = if defaults.is_object() { defaults.clone() } else { json!({}) };
                            if let (Some(d), Some(o)) = (pr.as_object_mut(), t["params"].as_object()) {
                                for (k, x) in o { d.insert(k.clone(), x.clone()); }
                            }
                            t["params"] = pr;
                        }
                        t
                    }).collect();
                    let n = new.len();
                    let ids: Vec<String> = new.iter().filter_map(|t| t["id"].as_str().map(String::from)).collect();
                    edit(app, &v, |spec| {
                        let tools = tools_mut(spec)?;
                        tools.extend(new);
                        Ok(json!({}))
                    })?;
                    Ok(json!({"added": n, "ids": ids}))
                }
            }
        }
        (Method::Get, "/api/teach") => Ok(TEACH.lock().unwrap().clone()),
        (Method::Post, "/api/teach/cancel") => {
            TEACH_CANCEL.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(json!({"ok": true}))
        }
        (Method::Post, "/api/teach/start") => {
            require_unlocked(app)?;
            if TEACH.lock().unwrap()["running"].as_bool() == Some(true) {
                return Err("a teaching run is already going".into());
            }
            let (app_id, sid, folder, cam, mode, dirty, refsize, taught) = {
                let st = app.st.lock().unwrap();
                let o = st.open.as_ref().ok_or("no application open")?;
                let sd = &o.sides[side_index(o, &v)?];
                (o.id.clone(), sd.project.id.clone(), sd.project.folder.clone(), sd.camera, sd.mode.clone(), sd.dirty,
                 sd.project.reference.as_ref().map(|r| (r.w, r.h)), sd.project.spec["light_level"].as_u64().map(|x| x as u32))
            };
            if dirty {
                return Err("Save first (Ctrl+S): the teaching run reads the job as saved".into());
            }
            if mode != "camera" {
                return Err("A model teaching run takes its pictures from the camera: set this side's picture source to the camera".into());
            }
            let refsize = refsize.ok_or("the side has no reference picture")?;
            let python = teach_python().ok_or("Python with Vision Station's packages was not found (VS_PYTHON)")?;
            let script = teach_script().ok_or("teach.py was not found beside Vision Next")?;
            let size = v["model_size"].as_str().map(String::from)
                .or_else(|| app.settings.lock().unwrap()["general"]["model_size"].as_str().map(String::from)).unwrap_or_default();
            TEACH_CANCEL.store(false, std::sync::atomic::Ordering::SeqCst);
            *TEACH.lock().unwrap() = json!({"running": true, "side": sid, "stage": "capture", "progress": 0.0,
                "message": "Taking pictures of the good part", "detail": "0 of 16", "started": engine::now()});
            let a = app.clone();
            std::thread::spawn(move || teach_run(a, app_id, sid, folder, cam, refsize, taught, python, script, size));
            Ok(json!({"started": true}))
        }
        (Method::Post, "/api/teach/remove") => {
            // the advanced model and its good examples gone (Vision Station's advanced.remove)
            require_unlocked(app)?;
            let (app_id, folder) = {
                let st = app.st.lock().unwrap();
                let o = st.open.as_ref().ok_or("no application open")?;
                (o.id.clone(), o.sides[side_index(o, &v)?].project.folder.clone())
            };
            if let Ok(rd) = std::fs::read_dir(folder.join("training")) {
                for e in rd.flatten() {
                    if let Ok(files) = std::fs::read_dir(e.path().join("ok")) {
                        for f in files.flatten() {
                            if f.file_name().to_string_lossy().starts_with("advanced_good_") {
                                let _ = std::fs::remove_file(f.path());
                            }
                        }
                    }
                }
            }
            std::fs::remove_dir_all(folder.join("advanced")).map_err(|e| e.to_string())?;
            open_app(app, &app_id)?;
            Ok(json!({"removed": true}))
        }
        (Method::Post, "/api/firewall/allow") => {
            let ok = crate::firewall::allow()?;
            Ok(json!({"ok": ok, "missing": crate::firewall::missing().iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>()}))
        }
        (Method::Get, "/api/system") => {
            let models = crate::paths::models();
            let fw_missing = crate::firewall::missing();
            let fw_note = if fw_missing.is_empty() { "allowed".to_string() } else {
                format!("not allowed: {}", fw_missing.iter().filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())).collect::<Vec<_>>().join(", "))
            };
            let fw_fix = if fw_missing.is_empty() { Value::Null } else { json!("firewall") };
            let parts_ok = std::fs::read_dir(models.join("parts"))
                .map(|d| d.flatten().any(|e| e.path().extension().map(|x| x == "onnx").unwrap_or(false))).unwrap_or(false);
            let root = results_root(app);
            let test = root.join(".write_test");
            let root_ok = std::fs::create_dir_all(&root).is_ok() && std::fs::write(&test, b"").is_ok() && std::fs::remove_file(&test).is_ok();
            Ok(json!({
                "stats": stats_json(app),
                "checks": [
                    {"what": "ONNX Runtime (AI checks, text reading)", "ok": crate::infer::init().is_ok(), "note": crate::infer::init().unwrap_or_else(|e| e)},
                    {"what": "Text reading models", "ok": models.join("ocr").join("rec_small.onnx").exists() || models.join("ocr").join("rec_medium.onnx").exists(), "note": models.join("ocr").display().to_string()},
                    {"what": "Basler pylon (cameras)", "ok": Hub::status() == "pylon ready", "note": Hub::status()},
                    {"what": "Serial ports", "ok": true, "note": station::serial_ports().join(", ")},
                    {"what": "Line interface", "ok": true, "note": app.line_state.lock().unwrap().clone()},
                    {"what": "Windows Firewall (camera pictures)", "ok": fw_missing.is_empty(), "note": fw_note, "fix": fw_fix},
                    {"what": "Part finder model (Find parts)", "ok": parts_ok, "note": models.join("parts").display().to_string()},
                    {"what": "Result pictures folder", "ok": root_ok, "note": root.display().to_string()},
                ],
                "firewall_explanation": crate::firewall::EXPLANATION,
                "projects": crate::paths::projects().display().to_string(),
                "cores": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            }))
        }
        _ => Err(format!("no such call: {} {}", method, path)),
    }
}

/// A side's picture as JPEG: reference or last, at most `w` wide.
fn picture(app: &App, which: &str, side: &str, w: usize) -> Option<Arc<Vec<u8>>> {
    let (idx, img, ver) = {
        let st = app.st.lock().unwrap();
        let o = st.open.as_ref()?;
        let i = o.sides.iter().position(|s| s.project.id == side).or(if side.is_empty() { Some(0) } else { None })?;
        let s = &o.sides[i];
        let (img, ver) = match which {
            "reference" => (s.project.reference.clone()?, s.project.ref_version),
            _ => (s.last_img.clone().or(s.project.reference.clone())?, s.img_seq * 1000 + s.project.ref_version),
        };
        let key = (i, which.to_string(), w, ver);
        if let Some(j) = st.jpeg.get(&key) {
            return Some(j.clone());
        }
        (i, img, ver)
    };
    let j = Arc::new(img.to_jpeg(w, 88));
    let mut st = app.st.lock().unwrap();
    if st.jpeg.len() > 32 {
        st.jpeg.clear();
    }
    st.jpeg.insert((idx, which.to_string(), w, ver), j.clone());
    Some(j)
}

fn package(app: &App) -> Result<(String, Vec<u8>), String> {
    let (folder, id) = {
        let st = app.st.lock().unwrap();
        let o = st.open.as_ref().ok_or("no application open")?;
        (o.folder.clone(), o.id.clone())
    };
    let mut z = crate::npz::Zip::default();
    fn walk(z: &mut crate::npz::Zip, root: &Path, d: &Path, prefix: &str) -> Result<(), String> {
        for e in std::fs::read_dir(d).map_err(|e| e.to_string())?.filter_map(|e| e.ok()) {
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            if p.is_dir() {
                if name == "results" || name.starts_with("_removed_") {
                    continue;
                }
                walk(z, root, &p, prefix)?;
            } else {
                let rel = p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
                z.add(&format!("{}/{}", prefix, rel), std::fs::read(&p).map_err(|e| e.to_string())?);
            }
        }
        Ok(())
    }
    walk(&mut z, &folder, &folder, &id)?;
    Ok((format!("{}.vnp", id), z.bytes()))
}

// -------------------------------------------------------------- HTTP plumbing
const INDEX: &str = include_str!("../ui/index.html");
const APP_JS: &str = include_str!("../ui/app.js");
const APP_CSS: &str = include_str!("../ui/app.css");
const SCHEMA: &str = include_str!("../ui/schema.json");
const ICON: &str = include_str!("../ui/icon.svg");
const FONTS: &[(&str, &[u8])] = &[
    ("Inter-Regular.ttf", include_bytes!("../ui/fonts/Inter-Regular.ttf")),
    ("Inter-Medium.ttf", include_bytes!("../ui/fonts/Inter-Medium.ttf")),
    ("Inter-SemiBold.ttf", include_bytes!("../ui/fonts/Inter-SemiBold.ttf")),
    ("Inter-Bold.ttf", include_bytes!("../ui/fonts/Inter-Bold.ttf")),
];

fn ui_file(name: &str) -> Option<(String, &'static str)> {
    let (embedded, ct) = match name {
        "/" | "/index.html" => (INDEX, "text/html; charset=utf-8"),
        "/app.js" => (APP_JS, "application/javascript; charset=utf-8"),
        "/app.css" => (APP_CSS, "text/css; charset=utf-8"),
        "/api/schema" | "/schema.json" => (SCHEMA, "application/json"),
        "/icon.svg" | "/favicon.ico" => (ICON, "image/svg+xml"),
        _ => return None,
    };
    // while developing, VS_UI_DIR serves the files from disk (edit, reload)
    if let Ok(dir) = std::env::var("VS_UI_DIR") {
        let f = match name { "/" => "index.html", "/api/schema" => "schema.json", "/favicon.ico" => "icon.svg", n => n.trim_start_matches('/') };
        if let Ok(t) = std::fs::read_to_string(PathBuf::from(dir).join(f)) {
            return Some((t, ct));
        }
    }
    Some((embedded.to_string(), ct))
}

fn header(k: &str, v: &str) -> Header {
    Header::from_bytes(k.as_bytes(), v.as_bytes()).unwrap()
}

fn parse_query(q: &str) -> HashMap<String, String> {
    q.split('&').filter_map(|kv| {
        let mut it = kv.splitn(2, '=');
        let k = it.next()?;
        let v = it.next().unwrap_or("");
        Some((k.to_string(), url_decode(v)))
    }).collect()
}

fn url_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(if b[i] == b'+' { b' ' } else { b[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

/// A 480-wide JPEG of a saved picture (20 MP PNGs are far too big for a
/// card), cached under %LOCALAPPDATA%\VisionNext\thumbs by path, size and time.
fn file_thumb(p: &Path) -> Option<Arc<Vec<u8>>> {
    let md = std::fs::metadata(p).ok()?;
    let t = md.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos()).unwrap_or(0);
    let key = station::sha256(format!("{}|{}|{}", p.display(), md.len(), t).as_bytes());
    let dir = crate::paths::data_dir().join("thumbs");
    let cached = dir.join(format!("{}.jpg", &key[..32]));
    if let Ok(b) = std::fs::read(&cached) {
        return Some(Arc::new(b));
    }
    let j = Img::load(p).ok()?.to_jpeg(480, 80);
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(&cached, &j);
    Some(Arc::new(j))
}

fn send_jpeg(req: Request, j: Option<Arc<Vec<u8>>>, cache: bool) {
    match j {
        Some(j) => {
            let _ = req.respond(Response::from_data((*j).clone()).with_header(header("Content-Type", "image/jpeg"))
                .with_header(header("Cache-Control", if cache { "max-age=3600" } else { "no-store" })));
        }
        None => {
            let _ = req.respond(Response::from_string("no picture").with_status_code(404));
        }
    }
}

fn handle(app: &App, mut req: Request) {
    let url = req.url().to_string();
    let (path, q) = url.split_once('?').unwrap_or((&url, ""));
    let path = path.to_string();
    let query = parse_query(q);
    let method = req.method().clone();
    if method == Method::Get {
        if let Some((body, ct)) = ui_file(&path) {
            let _ = req.respond(Response::from_string(body).with_header(header("Content-Type", ct)).with_header(header("Cache-Control", "no-cache")));
            return;
        }
        if let Some(f) = path.strip_prefix("/fonts/") {
            if let Some((_, b)) = FONTS.iter().find(|x| x.0 == f) {
                let _ = req.respond(Response::from_data(b.to_vec()).with_header(header("Content-Type", "font/ttf")).with_header(header("Cache-Control", "max-age=31536000")));
                return;
            }
        }
        if let Some(rest) = path.strip_prefix("/img/hist/") {
            // /img/hist/<seq>/<side>.jpg
            let mut parts = rest.splitn(2, '/');
            let seq: u64 = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            let side = url_decode(parts.next().unwrap_or("").trim_end_matches(".jpg"));
            let j = app.st.lock().unwrap().history.iter().find(|h| h.seq == seq).and_then(|h| h.sides.iter().find(|s| s.0 == side).and_then(|s| s.2.clone()));
            send_jpeg(req, j, true);
            return;
        }
        if let Some(rest) = path.strip_prefix("/img/live/") {
            let cam: usize = rest.trim_end_matches(".jpg").parse().unwrap_or(0);
            let w = query.get("w").and_then(|x| x.parse().ok()).unwrap_or(1280usize).clamp(64, 6000);
            let r = app.cams.grab_live(cam).map(|im| Arc::new(im.to_jpeg(w, 78)));
            match r {
                Ok(j) => send_jpeg(req, Some(j), false),
                Err(e) => {
                    let _ = req.respond(Response::from_string(e).with_status_code(503));
                }
            }
            return;
        }
        if let Some(rest) = path.strip_prefix("/img/") {
            let mut parts = rest.splitn(2, '/');
            let which = parts.next().unwrap_or("last").to_string();
            let side = url_decode(parts.next().unwrap_or("").trim_end_matches(".jpg"));
            let w = query.get("w").and_then(|x| x.parse().ok()).unwrap_or(1600usize).clamp(64, 12000);
            send_jpeg(req, picture(app, &which, &side, w), true);
            return;
        }
        if path == "/api/train/img" {
            // a check's training picture, small
            let name = query.get("name").cloned().unwrap_or_default();
            let label = if query.get("label").map(|x| x.as_str()) == Some("ng") { "ng" } else { "ok" };
            let folder = {
                let st = app.st.lock().unwrap();
                st.open.as_ref().and_then(|o| o.sides.iter().find(|x| Some(&x.project.id) == query.get("side")).or(o.sides.first()).map(|x| x.project.folder.clone()))
            };
            let j = (!name.contains(['/', '\\']) && !name.contains(".."))
                .then(|| folder.map(|f| training_dir(&f, query.get("id").map(|x| x.as_str()).unwrap_or(""), label).join(&name))).flatten()
                .and_then(|f| Img::load(&f).ok()).map(|im| Arc::new(im.to_jpeg(240, 80)));
            send_jpeg(req, j, true);
            return;
        }
        if path == "/api/file" {
            // a saved result picture, for Find picture: only from a results folder
            let want = PathBuf::from(query.get("path").cloned().unwrap_or_default());
            let ok = want.canonicalize().ok().map(|w| picture_roots(app).iter().filter_map(|r| r.canonicalize().ok()).any(|r| w.starts_with(r))).unwrap_or(false);
            let ext = want.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
            let ct = match ext.as_str() { "jpg" | "jpeg" => "image/jpeg", "png" => "image/png", "bmp" => "image/bmp", _ => "" };
            if ok && !ct.is_empty() && query.get("thumb").is_some() {
                send_jpeg(req, file_thumb(&want), true);
                return;
            }
            match (ok && !ct.is_empty()).then(|| std::fs::read(&want).ok()).flatten() {
                Some(b) => { let _ = req.respond(Response::from_data(b).with_header(header("Content-Type", ct))); }
                None => { let _ = req.respond(Response::from_string("not found").with_status_code(404)); }
            }
            return;
        }
        if path == "/api/package" {
            match package(app) {
                Ok((name, bytes)) => {
                    let _ = req.respond(Response::from_data(bytes).with_header(header("Content-Type", "application/zip"))
                        .with_header(header("Content-Disposition", &format!("attachment; filename=\"{}\"", name))));
                }
                Err(e) => {
                    let _ = req.respond(Response::from_string(e).with_status_code(400));
                }
            }
            return;
        }
    }
    let mut body = vec![];
    let _ = req.as_reader().take(512 * 1024 * 1024).read_to_end(&mut body);
    let res = if path.starts_with("/api/") { api(app, &method, &path, &query, &body) } else { Err("not found".into()) };
    let (code, v) = match res {
        Ok(v) => (200, v),
        Err(e) => (if path.starts_with("/api/") { 400 } else { 404 }, json!({"error": e})),
    };
    let _ = req.respond(Response::from_string(v.to_string()).with_status_code(code)
        .with_header(header("Content-Type", "application/json")).with_header(header("Cache-Control", "no-store")));
}

/// Serve on 127.0.0.1 (the first free port from `port`). Returns the port.
pub fn serve(app: App, port: u16) -> Result<u16, String> {
    let mut last = String::new();
    for p in port..port + 20 {
        match Server::http(("127.0.0.1", p)) {
            Ok(server) => {
                let server = Arc::new(server);
                // a small pool: long polls park a thread each, so keep several
                for _ in 0..16 {
                    let (s, a) = (server.clone(), app.clone());
                    std::thread::spawn(move || loop {
                        match s.recv() {
                            Ok(rq) => handle(&a, rq),
                            Err(_) => break,
                        }
                    });
                }
                continuous_loop(app);
                return Ok(p);
            }
            Err(e) => last = e.to_string(),
        }
    }
    Err(format!("no free port: {}", last))
}
