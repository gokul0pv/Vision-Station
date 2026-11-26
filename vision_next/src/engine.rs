//! A project (one side of a part) and its inspection loop, and the store of
//! applications on disk -- the same folders and files as Vision Station.

use crate::img::Img;
use crate::model::{Ctx, Graphic, ToolResult, ToolSpec};
use crate::tools;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct Project {
    pub id: String,
    pub name: String,
    pub folder: PathBuf,
    pub spec: Value,
    pub reference: Option<Arc<Img>>,
    pub ref_version: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Inspection {
    pub id: String,
    pub project: String,
    pub verdict: String,
    pub source: String,
    pub timestamp: f64,
    pub total_ms: f64,
    pub image_size: [usize; 2],
    pub aborted: Option<String>,
    pub failed_tools: Vec<String>,
    pub not_ported: usize,
    pub tools: Vec<ToolResult>,
}

/// Checks that read only the picture, the reference and the fixture: a run
/// of them can go to every core at once.
const INDEPENDENT: &[&str] = &[
    "presence", "ocr", "ocv", "text_detect", "ocr_auto", "code_read", "count_template", "histogram",
    "sharpness", "color_segment", "color_match", "defect_diff", "measure_length", "find_line",
    "find_circle", "caliper", "yolo_detect", "yolo_defect", "classify",
];

fn is_independent(t: &ToolSpec) -> bool {
    let no_links = t.p.get("links").and_then(|l| l.as_object()).map(|o| o.is_empty()).unwrap_or(true);
    let fx = t.s("fixture", "auto");
    let cap = t.s("capture", "default");
    INDEPENDENT.contains(&t.ty.as_str()) && no_links && (fx == "auto" || fx.is_empty())
        && (cap == "default" || cap.is_empty())
}

/// Centres a check found, from its result's "points" value.
fn point_list(r: &ToolResult) -> Option<Vec<[f64; 2]>> {
    let a = r.values.get("points")?.as_array()?;
    let v: Vec<[f64; 2]> = a.iter().filter_map(|p| {
        let q = p.as_array()?;
        Some([q.first()?.as_f64()?, q.get(1)?.as_f64()?])
    }).collect();
    if v.is_empty() { None } else { Some(v) }
}

/// Every check drawn the same way: its outline, in its result's colour;
/// then its Graphics settings (region, results, labels, search) applied.
fn add_outline(t: &ToolSpec, r: &mut ToolResult, ctx: &Ctx) {
    if !r.graphics.iter().any(|g| g.role == "region") && t.ty != "align" {
        if let Some(o) = ctx.outline(t) {
            let st = r.status.clone();
            r.graphics.insert(0, Graphic { kind: "poly".into(), points: o, status: st, label: String::new(), role: "region".into() });
        }
    }
    for g in r.graphics.iter_mut() {
        if g.role == "region" {
            g.status = r.status.clone();
        }
    }
    if let Some(Value::Object(opt)) = t.p.get("graphics") {
        let on = |k: &str| opt.get(k).and_then(|v| v.as_bool()).unwrap_or(true);
        let (region, results, labels, search) = (on("region"), on("results"), on("labels"), on("search"));
        r.graphics.retain(|g| if g.role == "region" { region } else if g.status == "search" { search } else { results });
        if !labels {
            for g in r.graphics.iter_mut() {
                g.label.clear();
            }
        }
    }
}

pub fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

pub fn new_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let t = (now() * 1000.0) as u64;
    format!("{:x}{:04x}", t, N.fetch_add(1, Ordering::Relaxed) & 0xffff)
}

impl Project {
    pub fn load(folder: &Path, id: &str, name: &str) -> Result<Project, String> {
        let path = folder.join("project.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let spec: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {}", path.display(), e))?;
        let mut p = Project { id: id.into(), name: name.into(), folder: folder.to_path_buf(), spec, reference: None, ref_version: 1 };
        p.load_reference();
        Ok(p)
    }

    pub fn load_reference(&mut self) {
        let r = self.spec.get("reference_image").and_then(|v| v.as_str()).unwrap_or("reference.png").to_string();
        let rp = if Path::new(&r).is_absolute() { PathBuf::from(&r) } else { self.folder.join(&r) };
        self.reference = Img::load(&rp).ok().map(Arc::new);
        self.ref_version += 1;
    }

    pub fn tools(&self) -> Vec<ToolSpec> {
        self.spec.get("tools").and_then(|t| t.as_array()).map(|a| a.iter().map(ToolSpec::from_json).collect()).unwrap_or_default()
    }

    pub fn save(&self) -> Result<(), String> {
        let text = serde_json::to_string_pretty(&self.spec).map_err(|e| e.to_string())?;
        let path = self.folder.join("project.json");
        let tmp = self.folder.join("project.json.tmp");
        std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &path).map_err(|e| e.to_string())
    }

    pub fn set_reference(&mut self, img: &Img) -> Result<(), String> {
        img.save_png(&self.folder.join("reference.png"))?;
        if let Some(o) = self.spec.as_object_mut() {
            o.insert("reference_image".into(), json!("reference.png"));
        }
        self.reference = Some(Arc::new(img.clone()));
        self.ref_version += 1;
        Ok(())
    }

    /// Run every enabled check on `frame`. `tools` overrides the recipe (an
    /// unsaved edit being previewed).
    pub fn inspect(&self, frame: Arc<Img>, source: &str, tools: Option<&[ToolSpec]>, barcode: Option<String>) -> Inspection {
        let t0 = std::time::Instant::now();
        let owned;
        let list: &[ToolSpec] = match tools {
            Some(t) => t,
            None => {
                owned = self.tools();
                &owned
            }
        };
        let mut ctx = Ctx::new(self.reference.clone(), self.folder.clone(),
            self.spec.get("mm_per_px").and_then(|v| v.as_f64()).unwrap_or(0.0));
        ctx.barcode = barcode;
        ctx.fixture_override = test_fixture(source);
        ctx.specs = Arc::new(list.iter().map(|t| (t.id.clone(), t.clone())).collect());
        // checks that make a picture in coordinates of their own
        let geo: std::collections::HashSet<String> = list.iter()
            .filter(|t| t.ty == "affine_transform" || t.ty == "polar_unwrap").map(|t| t.id.clone()).collect();
        let mut results: Vec<ToolResult> = vec![];
        let mut fixtures: HashMap<String, Option<crate::geom::Affine>> = HashMap::new();
        let mut aborted = None;
        let enabled: Vec<&ToolSpec> = list.iter().filter(|t| t.enabled()).collect();
        let mut i = 0;
        'outer: while i < enabled.len() {
            // a run of independent checks goes to every core at once
            let mut j = i;
            while j < enabled.len() && is_independent(enabled[j]) {
                j += 1;
            }
            if j - i >= 2 {
                let shared = &ctx;
                let outs: Vec<ToolResult> = enabled[i..j].par_iter().map(|t| {
                    let mut c = shared.light();
                    let img = shared.images.get(&t.s("capture", "default")).cloned().unwrap_or_else(|| frame.clone());
                    let mut r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tools::run(t, &img, &mut c)))
                        .unwrap_or_else(|_| ToolResult::error(t, "the check failed unexpectedly"));
                    add_outline(t, &mut r, &c);
                    r
                }).collect();
                for r in outs {
                    if let Some(p) = point_list(&r) {
                        ctx.points.insert(r.tool_id.clone(), p);
                    }
                    let ng = r.values.get("barcode_match") == Some(&Value::Bool(false));
                    ctx.results.insert(r.tool_id.clone(), r.clone());
                    results.push(r);
                    if ng {
                        aborted = Some("Barcode not good — remaining checks skipped".to_string());
                        break 'outer;
                    }
                }
                i = j;
                continue;
            }
            let spec = enabled[i];
            i += 1;
            let mut t = spec.clone();
            // linked settings: a value driven by an earlier check's result
            let mut problem = None;
            if let Some(Value::Object(links)) = spec.p.get("links") {
                for (param, r) in links {
                    let name = r.as_str().unwrap_or("");
                    match tools::logic::lookup(&ctx, name) {
                        Some(tools::logic::V::N(n)) => { t.p.insert(param.clone(), json!(n)); }
                        Some(tools::logic::V::S(s)) => { t.p.insert(param.clone(), json!(s)); }
                        Some(tools::logic::V::B(b)) => { t.p.insert(param.clone(), json!(b)); }
                        None => { problem = Some(format!("input {} <- {} has no value", param, name)); break; }
                    }
                }
            }
            // which picture: a processed one by id, else the camera's
            let want = t.s("capture", "default");
            let reads_transformed = geo.contains(&want);
            if reads_transformed && !ctx.images.contains_key(&want) {
                // never the camera picture instead: its coordinates are another's
                problem = Some(format!("reads the picture from {}, which was not produced", want));
            }
            let img: Arc<Img> = ctx.images.get(&want).cloned().unwrap_or_else(|| frame.clone());
            // which fixture: the latest Locate (auto), none, or a named one
            let mut fx_choice = t.s("fixture", "auto");
            let saved = ctx.fixture;
            let saved_ref = ctx.reference.clone();
            if reads_transformed {
                // the transform already followed the part; compare with the
                // reference transformed the same way
                fx_choice = "none".into();
                ctx.reference = ctx.ref_images.get(&want).cloned();
            }
            if fx_choice == "none" {
                ctx.fixture = None;
            } else if fx_choice != "auto" && !fx_choice.is_empty() {
                match fixtures.get(&fx_choice) {
                    Some(Some(f)) => ctx.fixture = Some(*f),
                    Some(None) => problem = Some(format!("fixture {} did not find the part", fx_choice)),
                    None => problem = Some(format!("fixture {} has not run before this check", fx_choice)),
                }
            }
            let mut r = match problem {
                Some(p) => ToolResult::error(&t, p),
                None => {
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tools::run(&t, &img, &mut ctx)));
                    res.unwrap_or_else(|_| ToolResult::error(&t, "the check failed unexpectedly"))
                }
            };
            if !reads_transformed {
                add_outline(&t, &mut r, &ctx);
            }
            ctx.reference = saved_ref;
            if let Some(p) = point_list(&r) {
                ctx.points.insert(t.id.clone(), p);
            }
            if t.ty == "align" || t.ty == "fixture_npoint" {
                fixtures.insert(t.id.clone(), if r.status == "pass" { ctx.fixture } else { None });
            } else if fx_choice != "auto" {
                ctx.fixture = saved;
            }
            let stop_align = t.ty == "align" && r.status != "pass";
            let barcode_ng = r.values.get("barcode_match") == Some(&Value::Bool(false));
            ctx.results.insert(t.id.clone(), r.clone());
            results.push(r);
            if stop_align {
                aborted = Some("alignment failed — remaining checks skipped".to_string());
                break;
            }
            if barcode_ng {
                aborted = Some("Barcode not good — remaining checks skipped".to_string());
                break;
            }
        }
        // the side's own trained network confirms the checks it learned (advanced training)
        if let Some(det) = crate::advanced::load(&self.folder, &self.spec) {
            crate::advanced::apply(&det, list, &mut results, &frame);
        }
        let critical: HashMap<&str, bool> = list.iter().map(|t| (t.id.as_str(), t.critical())).collect();
        let failed: Vec<String> = results.iter()
            .filter(|r| (r.status == "fail" || r.status == "error") && *critical.get(r.tool_id.as_str()).unwrap_or(&true))
            .map(|r| r.tool_id.clone()).collect();
        let noncrit = results.iter().any(|r| (r.status == "fail" || r.status == "error") && !failed.contains(&r.tool_id));
        let barcode_ng = results.iter().any(|r| r.values.get("barcode_match") == Some(&Value::Bool(false)));
        let verdict = if !failed.is_empty() || barcode_ng { "FAIL" } else if noncrit { "WARN" } else { "PASS" };
        Inspection {
            id: new_id(),
            project: self.id.clone(),
            verdict: verdict.into(),
            source: source.into(),
            timestamp: now(),
            total_ms: (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
            image_size: [frame.w, frame.h],
            aborted,
            failed_tools: failed,
            not_ported: results.iter().filter(|r| r.status == "skip").count(),
            tools: results,
        }
    }
}

/// An application: one part, one or more sides, each a project.
pub struct App {
    pub id: String,
    pub name: String,
    pub folder: PathBuf,
    pub sides: Vec<Project>,
    pub single: bool,
}

pub struct Store {
    pub root: PathBuf,
}

#[derive(Serialize)]
pub struct AppInfo {
    pub id: String,
    pub name: String,
    pub sides: usize,
    pub checks: usize,
}

fn read_json(p: &Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()
}

impl Store {
    pub fn new(root: PathBuf) -> Store {
        let _ = std::fs::create_dir_all(&root);
        Store { root }
    }

    pub fn list(&self) -> Vec<AppInfo> {
        list_in(&self.root)
    }

    pub fn open(&self, id: &str) -> Result<App, String> {
        open_in(&self.root.join(id), id)
    }

    pub fn create(&self, name: &str) -> Result<String, String> {
        let mut id: String = name.trim().chars().map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '_' }).collect();
        if id.is_empty() {
            id = "application".into();
        }
        let mut n = 1;
        let base = id.clone();
        while self.root.join(&id).exists() {
            n += 1;
            id = format!("{}_{}", base, n);
        }
        let side = self.root.join(&id).join("sides").join("job_1");
        std::fs::create_dir_all(&side).map_err(|e| e.to_string())?;
        std::fs::write(self.root.join(&id).join("job.json"),
            serde_json::to_string_pretty(&json!({"name": name, "sides": [{"id": "job_1", "name": "Job 1"}]})).unwrap()).map_err(|e| e.to_string())?;
        std::fs::write(side.join("project.json"),
            serde_json::to_string_pretty(&json!({"name": "Job 1", "id": "job_1", "tools": [],
                "image_source": {"mode": "folder", "paths": [], "folder": "", "camera": 0}})).unwrap()).map_err(|e| e.to_string())?;
        Ok(id)
    }

    pub fn add_side(&self, app: &str, name: &str) -> Result<String, String> {
        let jp = self.root.join(app).join("job.json");
        let mut job = read_json(&jp).ok_or("this application has one side only")?;
        let sides = job.get_mut("sides").and_then(|s| s.as_array_mut()).ok_or("job.json has no sides")?;
        let mut n = sides.len() + 1;
        let mut id = format!("job_{}", n);
        while sides.iter().any(|s| s.get("id").and_then(|v| v.as_str()) == Some(&id)) {
            n += 1;
            id = format!("job_{}", n);
        }
        sides.push(json!({"id": id, "name": name}));
        let folder = self.root.join(app).join("sides").join(&id);
        std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
        std::fs::write(folder.join("project.json"), serde_json::to_string_pretty(&json!({"name": name, "id": id, "tools": [],
            "image_source": {"mode": "folder", "paths": [], "folder": "", "camera": 0}})).unwrap()).map_err(|e| e.to_string())?;
        std::fs::write(&jp, serde_json::to_string_pretty(&job).unwrap()).map_err(|e| e.to_string())?;
        Ok(id)
    }

    /// Copy an application from Vision Station's projects folder (left untouched).
    pub fn import(&self, from: &Path, id: &str) -> Result<String, String> {
        let src = from.join(id);
        if !src.is_dir() {
            return Err(format!("{} not found", src.display()));
        }
        let mut dst_id = id.to_string();
        let mut n = 1;
        while self.root.join(&dst_id).exists() {
            n += 1;
            dst_id = format!("{}_{}", id, n);
        }
        copy_dir(&src, &self.root.join(&dst_id))?;
        // where it came from: History reads Vision Station's run log there
        let _ = std::fs::write(self.root.join(&dst_id).join(".imported_from"), id);
        Ok(dst_id)
    }
}

pub fn list_in(root: &Path) -> Vec<AppInfo> {
    let mut out = vec![];
    let Ok(rd) = std::fs::read_dir(root) else { return out };
    let mut dirs: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    for d in dirs {
        let id = d.file_name().unwrap().to_string_lossy().to_string();
        if let Some(job) = read_json(&d.join("job.json")) {
            let sides = job.get("sides").and_then(|s| s.as_array()).cloned().unwrap_or_default();
            let checks = sides.iter().filter_map(|s| s.get("id").and_then(|v| v.as_str()))
                .filter_map(|sid| read_json(&d.join("sides").join(sid).join("project.json")))
                .map(|p| p.get("tools").and_then(|t| t.as_array()).map(|a| a.len()).unwrap_or(0)).sum();
            out.push(AppInfo { id: id.clone(), name: job.get("name").and_then(|v| v.as_str()).unwrap_or(&id).to_string(), sides: sides.len(), checks });
        } else if let Some(p) = read_json(&d.join("project.json")) {
            let checks = p.get("tools").and_then(|t| t.as_array()).map(|a| a.len()).unwrap_or(0);
            out.push(AppInfo { id: id.clone(), name: p.get("name").and_then(|v| v.as_str()).unwrap_or(&id).to_string(), sides: 1, checks });
        }
    }
    out
}

pub fn open_in(folder: &Path, id: &str) -> Result<App, String> {
    if let Some(job) = read_json(&folder.join("job.json")) {
        let mut sides = vec![];
        for s in job.get("sides").and_then(|s| s.as_array()).cloned().unwrap_or_default() {
            let sid = s.get("id").and_then(|v| v.as_str()).unwrap_or("job_1").to_string();
            let sname = s.get("name").and_then(|v| v.as_str()).unwrap_or(&sid).to_string();
            sides.push(Project::load(&folder.join("sides").join(&sid), &sid, &sname)?);
        }
        let name = job.get("name").and_then(|v| v.as_str()).unwrap_or(id).to_string();
        return Ok(App { id: id.into(), name, folder: folder.to_path_buf(), sides, single: false });
    }
    let p = Project::load(folder, "main", "Main")?;
    let name = p.spec.get("name").and_then(|v| v.as_str()).unwrap_or(id).to_string();
    Ok(App { id: id.into(), name, folder: folder.to_path_buf(), sides: vec![p], single: true })
}

fn copy_dir(src: &Path, dst: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dst).map_err(|e| e.to_string())?;
    for e in std::fs::read_dir(src).map_err(|e| e.to_string())? {
        let e = e.map_err(|e| e.to_string())?;
        let p = e.path();
        let name = e.file_name();
        // results and training sets stay behind: big, and not part of the recipe
        if p.is_dir() {
            if name == "results" || name == "training" {
                continue;
            }
            copy_dir(&p, &dst.join(&name))?;
        } else {
            std::fs::copy(&p, dst.join(&name)).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// yyyy-mm-dd of a unix time (UTC), for result file names.
/// Seconds the local clock is ahead of UTC (India: +19800), read once.
pub fn utc_offset() -> f64 {
    static OFF: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        #[cfg(windows)]
        {
            #[repr(C)]
            #[derive(Default)]
            struct St { y: u16, mo: u16, dow: u16, d: u16, h: u16, mi: u16, s: u16, ms: u16 }
            #[link(name = "kernel32")]
            extern "system" {
                fn GetLocalTime(t: *mut St);
                fn GetSystemTime(t: *mut St);
            }
            let (mut l, mut u) = (St::default(), St::default());
            unsafe {
                GetLocalTime(&mut l);
                GetSystemTime(&mut u);
            }
            let mins = |t: &St| (t.d as i64) * 1440 + (t.h as i64) * 60 + t.mi as i64;
            let mut diff = mins(&l) - mins(&u);
            // across a month end the day numbers wrap
            if diff > 1440 * 2 { diff -= 1440 * (l.d as i64); }
            if diff < -1440 * 2 { diff += 1440 * (u.d as i64); }
            // round to the quarter hour (offsets are whole quarters)
            (((diff as f64) / 15.0).round() * 15.0) * 60.0
        }
        #[cfg(not(windows))]
        { 0.0 }
    })
}

/// Local seconds of the day and date for a unix time.
pub fn local_time(ts: f64) -> (String, u32, u32, u32, u32) {
    let t = ts + utc_offset();
    let s = (t.rem_euclid(86400.0)) as u32;
    (day_utc(t), s / 3600, (s / 60) % 60, s % 60, (t.fract().abs() * 1000.0) as u32)
}

/// yyyy-mm-dd of a unix time, local.
pub fn day(ts: f64) -> String {
    day_utc(ts + utc_offset())
}

fn day_utc(ts: f64) -> String {
    let days = (ts / 86400.0).floor() as i64;
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}-{:02}-{:02}", if m <= 2 { y + 1 } else { y }, m, d)
}

/// Test only: VS_TEST_FIXTURES names a JSON file {picture file name: [[a, b, tx], [c, d, ty]]},
/// the reference -> picture transform another engine found, used in place of
/// Locate's own (to compare the checks after it on equal terms).
fn test_fixture(source: &str) -> Option<crate::geom::Affine> {
    static MAP: std::sync::OnceLock<Option<Value>> = std::sync::OnceLock::new();
    let m = MAP.get_or_init(|| std::env::var("VS_TEST_FIXTURES").ok()
        .and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| serde_json::from_str(&t).ok())).as_ref()?;
    let name = Path::new(source).file_name()?.to_string_lossy().to_string();
    let r = m.get(&name)?;
    let g = |i: usize, j: usize| r[i][j].as_f64();
    Some(crate::geom::Affine { a: g(0, 0)?, b: g(0, 1)?, tx: g(0, 2)?, c: g(1, 0)?, d: g(1, 1)?, ty: g(1, 2)? })
}

/// Append one inspection to the side's results file (JSON lines).
/// `pictures`: where its result pictures were saved; `barcode`: the part's.
pub fn log_result(folder: &Path, r: &Inspection, pictures: &[PathBuf], barcode: Option<&str>) {
    let dir = folder.join("results");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let mut slim = serde_json::to_value(r).unwrap_or(Value::Null);
    if let Some(tools) = slim.get_mut("tools").and_then(|t| t.as_array_mut()) {
        for t in tools.iter_mut() {
            if let Some(o) = t.as_object_mut() {
                o.remove("graphics");
            }
        }
    }
    slim["pictures"] = json!(pictures.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>());
    slim["barcode"] = json!(barcode);
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(format!("next_{}.jsonl", day(r.timestamp)))) {
        let _ = writeln!(f, "{}", slim);
    }
}

pub fn summary(r: &Inspection) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("id".into(), json!(r.id));
    m.insert("project".into(), json!(r.project));
    m.insert("verdict".into(), json!(r.verdict));
    m.insert("total_ms".into(), json!(r.total_ms));
    m.insert("timestamp".into(), json!(r.timestamp));
    m
}
