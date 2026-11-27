//! Production history from the results logs (yield per side, the checks
//! rejecting most, CSV export), finding saved result pictures, and showing a
//! file in Explorer -- as Vision Station's History page and Find picture.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Every logged run of a side within the last `hours` (None: all), newest
/// first: Vision Next's own log (next_<day>.jsonl) and Vision Station's
/// (results.db), so parts inspected before Vision Next are counted too.
pub fn runs(folder: &Path, hours: Option<f64>) -> Vec<Value> {
    let cutoff = hours.map(|h| crate::engine::now() - h * 3600.0);
    let mut out = next_runs(folder, cutoff);
    for db in vs_dbs(folder) {
        match vs_runs(&db) {
            Ok(rs) => out.extend(rs.iter().filter(|r| cutoff.map(|c| r["timestamp"].as_f64().unwrap_or(0.0) >= c).unwrap_or(true)).cloned()),
            Err(e) => eprintln!("[history] {}: {}", db.display(), e),
        }
    }
    out.sort_by(|a, b| b["timestamp"].as_f64().unwrap_or(0.0).partial_cmp(&a["timestamp"].as_f64().unwrap_or(0.0)).unwrap_or(std::cmp::Ordering::Equal));
    out
}

fn next_runs(folder: &Path, cutoff: Option<f64>) -> Vec<Value> {
    let first_day = cutoff.map(crate::engine::day);
    let mut out = vec![];
    let Ok(rd) = std::fs::read_dir(folder.join("results")) else { return out };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(d) = name.strip_prefix("next_").and_then(|n| n.strip_suffix(".jsonl")) else { continue };
        if let Some(f) = &first_day {
            if d < f.as_str() {
                continue;
            }
        }
        let Ok(text) = std::fs::read_to_string(e.path()) else { continue };
        for line in text.lines() {
            let Ok(mut v) = serde_json::from_str::<Value>(line) else { continue };
            if let Some(c) = cutoff {
                if v["timestamp"].as_f64().unwrap_or(0.0) < c {
                    continue;
                }
            }
            v["from"] = json!("next");
            out.push(v);
        }
    }
    out
}

/// Vision Station's run logs for a side: results.db in the side's own
/// folder, and in Vision Station's copy of the job when it was imported
/// (the same job folder there; an import named <job>_2 came from <job>).
pub fn vs_dbs(folder: &Path) -> Vec<PathBuf> {
    let db = |side: &Path| side.join("results").join("results.db");
    let mut out = vec![db(folder)];
    if let (Some(legacy), Ok(rel)) = (crate::paths::legacy_projects(), folder.strip_prefix(crate::paths::projects())) {
        let mut parts: Vec<String> = rel.iter().map(|x| x.to_string_lossy().into_owned()).collect();
        if let Some(app) = parts.first().cloned() {
            let marker = std::fs::read_to_string(crate::paths::projects().join(&app).join(".imported_from")).ok()
                .map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
            let base = match marker {
                Some(m) => Some(m),
                None if legacy.join(&app).is_dir() => Some(app.clone()),
                None => app.rsplit_once('_').filter(|(_, n)| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
                    .map(|(b, _)| b.to_string()).filter(|b| legacy.join(b).is_dir()),
            };
            if let Some(b) = base {
                parts[0] = b;
                out.push(db(&parts.iter().fold(legacy.clone(), |p, x| p.join(x))));
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    out.retain(|p| p.is_file() && seen.insert(p.canonicalize().unwrap_or_else(|_| p.clone())));
    out
}

type Cached = (u64, std::sync::Arc<Vec<Value>>);

fn stamp(p: &Path) -> u64 {
    let one = |p: &Path| std::fs::metadata(p).ok().map(|m| {
        let t = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos() as u64).unwrap_or(0);
        t ^ m.len().rotate_left(32)
    }).unwrap_or(0);
    let mut w = p.as_os_str().to_owned();
    w.push("-wal");
    one(p) ^ one(Path::new(&w)).rotate_left(17)
}

/// Every run in a Vision Station results.db, newest first, in Vision Next's
/// run shape (timestamp, verdict, total_ms, source, image, tools,
/// failed_tools). Read again only when the file changed.
pub fn vs_runs(db_path: &Path) -> Result<std::sync::Arc<Vec<Value>>, String> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<PathBuf, Cached>>> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let st = stamp(db_path);
    if let Some((s, v)) = cache.lock().unwrap().get(db_path) {
        if *s == st {
            return Ok(v.clone());
        }
    }
    let db = crate::sqlite::Db::open(db_path)?;
    if !db.has_table("runs") {
        return Ok(Default::default());
    }
    let side = db_path.parent().and_then(|p| p.parent()).map(Path::to_path_buf).unwrap_or_default();
    let mut runs: Vec<(i64, Value)> = vec![];
    db.rows_newest_first("runs", &mut |r| {
        let img = r.get("image_path").str();
        let image = if img.is_empty() { String::new() } else {
            let p = PathBuf::from(img);
            if p.is_absolute() { img.to_string() } else { side.join(p).to_string_lossy().into_owned() }
        };
        // a barcode folder holds the picture when the part had one
        let barcode = Path::new(&image).parent().and_then(|p| p.file_name()).map(|f| f.to_string_lossy().into_owned())
            .filter(|f| !is_day(f) && f != "images" && !f.is_empty());
        runs.push((r.get("id").i64().unwrap_or(r.rowid), json!({
            "id": r.get("run_uid").str(), "project": r.get("project").str(), "timestamp": r.get("ts").f64().unwrap_or(0.0),
            "verdict": r.get("verdict").str(), "total_ms": r.get("total_ms").f64().unwrap_or(0.0),
            "source": r.get("source").str(), "image": image, "barcode": barcode, "operator": r.get("operator").str(),
            "from": "vs", "tools": [], "failed_tools": []})));
        true
    })?;
    if db.has_table("tool_results") && !runs.is_empty() {
        let first = runs.last().map(|r| r.0).unwrap_or(0);
        let mut tools: std::collections::HashMap<i64, Vec<Value>> = Default::default();
        db.rows_newest_first("tool_results", &mut |r| {
            let run = r.get("run_id").i64().unwrap_or(0);
            if run < first {
                return false; // ids only grow: everything older is done
            }
            tools.entry(run).or_default().push(json!({"tool_id": r.get("tool_id").str(), "type": r.get("type").str(),
                "status": r.get("status").str(), "score": r.get("score").f64(), "ms": r.get("ms").f64(), "message": r.get("message").str()}));
            true
        })?;
        for (id, v) in runs.iter_mut() {
            if let Some(mut ts) = tools.remove(id) {
                ts.reverse();
                let failed: Vec<Value> = ts.iter().filter(|t| matches!(t["status"].as_str(), Some("fail") | Some("error")))
                    .map(|t| t["tool_id"].clone()).collect();
                v["failed_tools"] = json!(failed);
                v["tools"] = json!(ts);
            }
        }
    }
    let all = std::sync::Arc::new(runs.into_iter().map(|r| r.1).collect::<Vec<_>>());
    cache.lock().unwrap().insert(db_path.to_path_buf(), (st, all.clone()));
    Ok(all)
}

/// Yield per side and the checks rejecting parts, most first.
pub fn summary(sides: &[(String, String, PathBuf)], hours: Option<f64>) -> Value {
    let mut rows = vec![];
    let mut pareto: Vec<(String, String, u64)> = vec![];
    for (id, name, folder) in sides {
        let rs = runs(folder, hours);
        let total = rs.len() as u64;
        let passed = rs.iter().filter(|r| r["verdict"].as_str() == Some("PASS")).count() as u64;
        let avg = if total > 0 { rs.iter().map(|r| r["total_ms"].as_f64().unwrap_or(0.0)).sum::<f64>() / total as f64 } else { 0.0 };
        rows.push(json!({"side": id, "name": name, "total": total, "passed": passed,
            "yield_pct": if total > 0 { passed as f64 * 100.0 / total as f64 } else { 0.0 }, "avg_ms": avg}));
        let mut counts: std::collections::BTreeMap<String, u64> = Default::default();
        for r in &rs {
            if r["verdict"].as_str() == Some("PASS") {
                continue;
            }
            for t in r["failed_tools"].as_array().into_iter().flatten() {
                if let Some(t) = t.as_str() {
                    *counts.entry(t.to_string()).or_default() += 1;
                }
            }
        }
        pareto.extend(counts.into_iter().map(|(t, n)| (name.clone(), t, n)));
    }
    pareto.sort_by(|a, b| b.2.cmp(&a.2));
    json!({"sides": rows, "pareto": pareto.into_iter().map(|(s, t, n)| json!({"side": s, "tool": t, "count": n})).collect::<Vec<_>>()})
}

fn csv_cell(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}

fn iso(ts: f64) -> String {
    let (d, h, m, s, ms) = crate::engine::local_time(ts);
    format!("{}T{:02}:{:02}:{:02}.{:03}", d, h, m, s, ms)
}

/// One CSV per side in `folder`, Vision Station's columns: a row per check.
pub fn export_csv(sides: &[(String, String, PathBuf)], hours: Option<f64>, folder: &Path) -> Result<Vec<PathBuf>, String> {
    std::fs::create_dir_all(folder).map_err(|e| e.to_string())?;
    let mut out = vec![];
    for (_, name, side_folder) in sides {
        let safe: String = name.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
        let path = folder.join(format!("{}_history.csv", safe));
        let mut text = String::from("timestamp,verdict,cycle_ms,source,image,tool,tool_status,score,message\r\n");
        for r in runs(side_folder, hours) {
            let head = [iso(r["timestamp"].as_f64().unwrap_or(0.0)), r["verdict"].as_str().unwrap_or("").to_string(),
                format!("{}", r["total_ms"].as_f64().unwrap_or(0.0)), r["source"].as_str().unwrap_or("").to_string(),
                r["image"].as_str().unwrap_or("").to_string()];
            let head: Vec<String> = head.iter().map(|x| csv_cell(x)).collect();
            let tools = r["tools"].as_array().cloned().unwrap_or_default();
            if tools.is_empty() {
                text += &format!("{},,,,\r\n", head.join(","));
            }
            for t in tools {
                let score = t["score"].as_f64().map(|s| format!("{}", s)).unwrap_or_default();
                text += &format!("{},{},{},{},{}\r\n", head.join(","), csv_cell(t["tool_id"].as_str().unwrap_or("")),
                    csv_cell(t["status"].as_str().unwrap_or("")), score, csv_cell(t["message"].as_str().unwrap_or("")));
            }
        }
        // with a BOM, so Excel reads names and messages correctly
        std::fs::write(&path, [b"\xEF\xBB\xBF".as_slice(), text.as_bytes()].concat()).map_err(|e| format!("{}: {}", path.display(), e))?;
        out.push(path);
    }
    Ok(out)
}

/// "OK" / "NG" / "" from a saved picture's name: Vision Station's results
/// folder writes ..._ACCEPT / ..._REJECT, a job's own archive ..._PASS_<id> /
/// ..._FAIL_<id>.
fn verdict_of(stem: &str) -> &'static str {
    let s = stem.to_uppercase();
    let tok: Vec<&str> = s.split('_').collect();
    if tok.iter().any(|t| *t == "REJECT" || *t == "FAIL" || *t == "NG") {
        "NG"
    } else if tok.iter().any(|t| *t == "ACCEPT" || *t == "PASS" || *t == "OK") {
        "OK"
    } else {
        ""
    }
}

fn is_day(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 10 && b[4] == b'-' && b[7] == b'-' && s.chars().enumerate().all(|(i, c)| i == 4 || i == 7 || c.is_ascii_digit())
}

/// Saved result pictures whose path holds `q` (a barcode, a job, a day),
/// newest first, from every folder in `roots` (Vision Next's own, and where
/// Vision Station saved them).
pub fn find_pictures_in(roots: &[PathBuf], q: &str, limit: usize) -> Value {
    let mut items: Vec<Value> = vec![];
    let mut total = 0;
    let mut seen = std::collections::HashSet::new();
    for root in roots {
        let r = find_pictures(root, q, usize::MAX);
        total += r["total"].as_u64().unwrap_or(0) as usize;
        for it in r["items"].as_array().cloned().unwrap_or_default() {
            let key = std::fs::canonicalize(it["path"].as_str().unwrap_or("")).unwrap_or_default();
            if seen.insert(key) {
                items.push(it);
            } else {
                total -= 1;
            }
        }
    }
    items.sort_by(|a, b| b["t"].as_f64().unwrap_or(0.0).partial_cmp(&a["t"].as_f64().unwrap_or(0.0)).unwrap_or(std::cmp::Ordering::Equal));
    items.truncate(limit);
    json!({"items": items, "total": total})
}

/// Saved result pictures whose path holds `q` (a barcode, a job, a day), newest first.
pub fn find_pictures(root: &Path, q: &str, limit: usize) -> Value {
    let q = q.trim().to_lowercase();
    let mut found: Vec<(std::time::SystemTime, PathBuf)> = vec![];
    fn walk(dir: &Path, depth: usize, out: &mut Vec<(std::time::SystemTime, PathBuf)>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            let Ok(md) = e.metadata() else { continue };
            if md.is_dir() {
                if depth < 5 {
                    walk(&p, depth + 1, out);
                }
            } else if matches!(p.extension().and_then(|x| x.to_str()).map(|x| x.to_lowercase()).as_deref(), Some("jpg") | Some("jpeg") | Some("png") | Some("bmp")) {
                out.push((md.modified().unwrap_or(std::time::UNIX_EPOCH), p));
            }
        }
    }
    walk(root, 0, &mut found);
    // the day and verdict are searchable too, wherever the folder put them
    found.retain(|(t, p)| {
        if q.is_empty() {
            return true;
        }
        let rel = p.strip_prefix(root).unwrap_or(p).to_string_lossy().to_lowercase();
        let secs = t.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
        let stem = p.file_stem().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
        rel.contains(&q) || crate::engine::day(secs).contains(&q) || verdict_of(&stem).to_lowercase() == q
    });
    let total = found.len();
    found.sort_by(|a, b| b.0.cmp(&a.0));
    found.truncate(limit);
    let items: Vec<Value> = found.into_iter().map(|(t, p)| {
        let rel = p.strip_prefix(root).unwrap_or(&p).to_path_buf();
        let parts: Vec<String> = rel.iter().map(|x| x.to_string_lossy().into_owned()).collect();
        let stem = p.file_stem().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
        let verdict = verdict_of(&stem);
        let time = stem.get(0..6).filter(|s| s.chars().all(|c| c.is_ascii_digit()))
            .map(|s| format!("{}:{}:{}", &s[0..2], &s[2..4], &s[4..6])).unwrap_or_default();
        let secs = t.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
        // <day>/[subfolder/][barcode/]file, or <barcode>/file: the day from
        // its folder (else the file's date), the barcode from the folder
        // holding the file when that is not the day
        let folders = &parts[..parts.len().saturating_sub(1)];
        let day = folders.iter().find(|f| is_day(f)).cloned().unwrap_or_else(|| crate::engine::day(secs));
        let barcode = folders.last().filter(|f| !is_day(f)).cloned().unwrap_or_default();
        json!({"path": p.to_string_lossy(), "rel": rel.to_string_lossy(), "root": root.to_string_lossy(), "day": day,
               "time": time, "barcode": barcode, "verdict": verdict, "original": stem.ends_with("_original"), "t": secs})
    }).collect();
    json!({"items": items, "total": total})
}

/// Explorer on the file's folder with the file selected (`open`: the file
/// in its own program instead).
pub fn reveal(path: &Path, open: bool) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("{} is not there any more (moved or deleted?)", path.display()));
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let p = path.to_string_lossy().replace('/', "\\");
        let mut c = std::process::Command::new("explorer.exe");
        if open || path.is_dir() {
            c.raw_arg(format!("\"{}\"", p));
        } else {
            c.raw_arg(format!("/select,\"{}\"", p));
        }
        c.spawn().map_err(|e| e.to_string())?;
        Ok(())
    }
    #[cfg(not(windows))]
    {
        let _ = open;
        Err("only on Windows".into())
    }
}
