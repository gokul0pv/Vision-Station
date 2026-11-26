//! A job's own trained network (Vision Station's advanced training): it
//! confirms the part and text checks it learned. A learned check passes only
//! when the network also finds its part or text where the check is -- unless
//! the check itself is sure (part 0.85, text 0.80): taught on one board lying
//! one way, the network lost small parts the checks matched at 0.98 when the
//! board was put back 12 px along. A port of core/advanced.py's Detector,
//! plan, find and apply; the files are the ones Vision Station writes:
//! <side>/advanced/model.onnx and model.json.

use crate::img::Img;
use crate::model::ToolResult;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const AI_MIN: f32 = 0.35;
const SURE_PART: f64 = 0.85;
const SURE_TEXT: f64 = 0.80;

fn kind(ty: &str) -> Option<&'static str> {
    match ty {
        "presence" => Some("part"),
        "ocv" | "ocr" => Some("text"),
        _ => None,
    }
}

pub struct Detector {
    path: PathBuf,
    tools: HashMap<String, usize>,
    imgsz: usize,
    src: usize,
    min_conf: f32,
    thresholds: HashMap<String, f32>,
}

type Det = [f32; 6]; // x1 y1 x2 y2 score class

static CACHE: Mutex<Option<HashMap<PathBuf, (std::time::SystemTime, Arc<Detector>)>>> = Mutex::new(None);

pub fn is_trained(spec: &Value, folder: &Path) -> bool {
    spec["advanced"]["trained"].as_bool().unwrap_or(false)
        && folder.join("advanced").join("model.onnx").is_file()
        && folder.join("advanced").join("model.json").is_file()
}

/// The side's Detector, or None (none trained, or its files gone).
pub fn load(folder: &Path, spec: &Value) -> Option<Arc<Detector>> {
    if !is_trained(spec, folder) {
        return None;
    }
    let path = folder.join("advanced").join("model.onnx");
    let mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok()?;
    let mut c = CACHE.lock().unwrap();
    let map = c.get_or_insert_with(HashMap::new);
    if let Some((t, d)) = map.get(&path) {
        if *t == mtime {
            return Some(d.clone());
        }
    }
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(folder.join("advanced").join("model.json")).ok()?).ok()?;
    let zoom = meta["zoom"].as_f64().unwrap_or(1.0);
    let imgsz = meta["imgsz"].as_u64().unwrap_or(1280) as usize;
    let d = Arc::new(Detector {
        path: path.clone(),
        tools: meta["tools"].as_object().map(|o| o.iter().filter_map(|(k, v)| v.as_u64().map(|n| (k.clone(), n as usize))).collect()).unwrap_or_default(),
        imgsz,
        src: (imgsz as f64 / zoom).round() as usize,
        min_conf: meta["min_conf"].as_f64().unwrap_or(AI_MIN as f64) as f32,
        thresholds: meta["thresholds"].as_object().map(|o| o.iter().filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n as f32))).collect()).unwrap_or_default(),
    });
    map.insert(path, (mtime, d.clone()));
    Some(d)
}

fn iou(a: &Det, b: &Det) -> f32 {
    let i = ((a[2].min(b[2]) - a[0].max(b[0])).max(0.0)) * ((a[3].min(b[3]) - a[1].max(b[1])).max(0.0));
    let u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
    i / u.max(1e-6)
}

impl Detector {
    pub fn threshold(&self, id: &str) -> f32 {
        *self.thresholds.get(id).unwrap_or(&self.min_conf)
    }

    fn tile_for(&self, r: [f64; 4], w: usize, h: usize) -> (usize, usize) {
        let src = self.src as f64;
        let step = (self.src / 4).max(1) as f64;
        let place = |c: f64, size: f64, n: usize| -> usize {
            let n = n as f64;
            if n <= src {
                return 0;
            }
            let mut t = (((c - src / 2.0) / step).round() * step).clamp(0.0, n - src);
            if c - size / 2.0 < t || c + size / 2.0 > t + src {
                t = (c - src / 2.0).round().clamp(0.0, n - src);
            }
            t as usize
        };
        (place(r[0] + r[2] / 2.0, r[2], w), place(r[1] + r[3] / 2.0, r[3], h))
    }

    fn contains(&self, tile: (usize, usize), r: [f64; 4], w: usize, h: usize, margin: f64) -> bool {
        let (sw, sh) = (self.src.min(w) as f64, self.src.min(h) as f64);
        let (tx, ty) = (tile.0 as f64, tile.1 as f64);
        r[0] - margin >= tx && r[1] - margin >= ty && r[0] + r[2] + margin <= tx + sw && r[1] + r[3] + margin <= ty + sh
    }

    /// Finds in one tile, in the frame's pixels.
    fn detect_tile(&self, frame: &Img, tx: usize, ty: usize) -> Result<Vec<Det>, String> {
        let src = self.src;
        let crop = frame.crop(tx as i64, ty as i64, src.min(frame.w - tx), src.min(frame.h - ty));
        let crop = if crop.c >= 3 { crop } else { crop.to_rgb() };
        // padded with grey to the whole tile, as the training tiles were
        let mut tile = Img::new(src, src, 3);
        for v in tile.data.iter_mut() {
            *v = 114;
        }
        tile.paste(&crop, 0, 0);
        let tile = if src != self.imgsz { tile.resize(self.imgsz, self.imgsz) } else { tile };
        let n = self.imgsz * self.imgsz;
        let mut x = vec![0f32; 3 * n];
        for i in 0..n {
            for c in 0..3 {
                x[c * n + i] = tile.data[i * 3 + c] as f32 / 255.0;
            }
        }
        let m = crate::infer::session(&self.path)?;
        let (shape, out) = crate::infer::run(&m, &[1, 3, self.imgsz, self.imgsz], x)?;
        let s = self.imgsz as f32 / src as f32;
        let mut res = vec![];
        if shape.len() == 3 && shape[2] == 6 {
            // YOLO26 end to end: (1, N, 6) x1 y1 x2 y2 score class, overlaps removed
            for d in out.chunks(6) {
                if d[4] >= 0.05 {
                    res.push([d[0] / s + tx as f32, d[1] / s + ty as f32, d[2] / s + tx as f32, d[3] / s + ty as f32, d[4], d[5]]);
                }
            }
            return Ok(res);
        }
        // YOLO11: (1, 4 + classes, N)
        let (rows, cnt) = (shape[1], shape[2]);
        let mut by_class: HashMap<usize, Vec<Det>> = HashMap::new();
        for i in 0..cnt {
            let (mut best, mut cls) = (0f32, 0usize);
            for c in 4..rows {
                let v = out[c * cnt + i];
                if v > best {
                    best = v;
                    cls = c - 4;
                }
            }
            if best < 0.05 {
                continue;
            }
            let (cx, cy, w, h) = (out[i], out[cnt + i], out[2 * cnt + i], out[3 * cnt + i]);
            by_class.entry(cls).or_default().push([(cx - w / 2.0) / s + tx as f32, (cy - h / 2.0) / s + ty as f32,
                (cx + w / 2.0) / s + tx as f32, (cy + h / 2.0) / s + ty as f32, best, cls as f32]);
        }
        for (_, mut v) in by_class {
            v.sort_by(|a, b| b[4].partial_cmp(&a[4]).unwrap());
            let mut kept: Vec<Det> = vec![];
            for d in v {
                if kept.iter().all(|k| iou(k, &d) <= 0.5) {
                    kept.push(d);
                }
            }
            res.extend(kept);
        }
        Ok(res)
    }

    /// The fewest tiles holding every region whole: greedily, from origins a
    /// quarter-tile apart, in bands from the top-left uncovered region.
    fn plan(&self, regions: &[(String, [f64; 4])], w: usize, h: usize) -> HashMap<String, (usize, usize)> {
        let src = self.src;
        let step = (src / 4).max(1);
        let grid = |n: usize| -> Vec<usize> {
            let mut v: Vec<usize> = (0..=n.saturating_sub(src)).step_by(step).map(|x| x.min(n.saturating_sub(src))).collect();
            if v.is_empty() { v.push(0); }
            v.push(n.saturating_sub(src));
            v.sort();
            v.dedup();
            v
        };
        let (xs, ys) = (grid(w), grid(h));
        let cand: Vec<(usize, usize)> = ys.iter().flat_map(|&y| xs.iter().map(move |&x| (x, y))).collect();
        let inside: Vec<Vec<bool>> = cand.iter().map(|&t| regions.iter().map(|(_, r)| self.contains(t, *r, w, h, 16.0)).collect()).collect();
        let n = regions.len();
        let mut left = vec![true; n];
        let holdable: Vec<bool> = (0..n).map(|i| inside.iter().any(|row| row[i])).collect();
        let half = (src / 2).max(1) as f64;
        let band: Vec<f64> = regions.iter().map(|(_, r)| (r[1] / half).floor() * 1e6 + r[0]).collect();
        let mut out = HashMap::new();
        loop {
            let anchor = (0..n).filter(|&i| left[i] && holdable[i]).min_by(|&a, &b| band[a].partial_cmp(&band[b]).unwrap());
            let Some(anchor) = anchor else { break };
            let best = (0..cand.len()).filter(|&t| inside[t][anchor])
                .max_by_key(|&t| (0..n).filter(|&i| inside[t][i] && left[i]).count());
            let Some(best) = best else { break };
            for i in 0..n {
                if inside[best][i] && left[i] {
                    out.insert(regions[i].0.clone(), cand[best]);
                    left[i] = false;
                }
            }
        }
        for i in 0..n {
            if left[i] {
                out.insert(regions[i].0.clone(), self.tile_for(regions[i].1, w, h));
            }
        }
        out
    }

    /// (confidence, box) of the best find of class `cls` for the region: about
    /// its size (half to twice its area), centred within `pad` of the region's.
    fn find(&self, frame: &Img, r: [f64; 4], cls: usize, cache: &mut HashMap<(usize, usize), Vec<Det>>, tile: Option<(usize, usize)>, pad: f64) -> Result<f32, String> {
        let tile = match tile {
            Some(t) if self.contains(t, r, frame.w, frame.h, 0.0) => t,
            _ => self.tile_for(r, frame.w, frame.h),
        };
        if !cache.contains_key(&tile) {
            let d = self.detect_tile(frame, tile.0, tile.1)?;
            cache.insert(tile, d);
        }
        let (cx, cy, area) = (r[0] + r[2] / 2.0, r[1] + r[3] / 2.0, (r[2] * r[3]).max(1.0));
        let (tol_x, tol_y) = (pad + 0.25 * r[2], pad + 0.25 * r[3]);
        let mut best = 0f32;
        for d in &cache[&tile] {
            if d[5] as usize != cls || d[4] <= best {
                continue;
            }
            let (dcx, dcy) = (((d[0] + d[2]) / 2.0) as f64 - cx, ((d[1] + d[3]) / 2.0) as f64 - cy);
            if dcx.abs() > tol_x || dcy.abs() > tol_y {
                continue;
            }
            let a = ((d[2] - d[0]) * (d[3] - d[1])) as f64;
            if !(0.5..=2.0).contains(&(a / area)) {
                continue;
            }
            best = d[4];
        }
        Ok(best)
    }
}

/// The learned checks' results, confirmed by the network (in place).
pub fn apply(det: &Detector, specs: &[crate::model::ToolSpec], results: &mut [ToolResult], frame: &Img) {
    let bbox = |r: &ToolResult| -> Option<[f64; 4]> {
        let g = r.graphics.iter().find(|g| g.role == "region")?;
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for p in &g.points {
            x0 = x0.min(p[0]); y0 = y0.min(p[1]); x1 = x1.max(p[0]); y1 = y1.max(p[1]);
        }
        (x1 > x0 && y1 > y0).then(|| [x0, y0, x1 - x0, y1 - y0])
    };
    let spec_of: HashMap<&str, &crate::model::ToolSpec> = specs.iter().map(|t| (t.id.as_str(), t)).collect();
    let learned: Vec<(usize, [f64; 4])> = results.iter().enumerate().filter_map(|(i, r)| {
        let t = spec_of.get(r.tool_id.as_str())?;
        (det.tools.contains_key(&t.id) && kind(&t.ty).is_some() && t.b("ai", true) && r.status != "error").then_some(())?;
        Some((i, bbox(r)?))
    }).collect();
    if learned.is_empty() {
        return;
    }
    let plan = det.plan(&learned.iter().map(|(i, r)| (results[*i].tool_id.clone(), *r)).collect::<Vec<_>>(), frame.w, frame.h);
    let mut cache = HashMap::new();
    for (i, region) in learned {
        let t = spec_of[results[i].tool_id.as_str()];
        let pad = if t.ty == "presence" { t.f("search_pad", 15.0) } else { 8f64.max(0.15 * region[2].max(region[3])) };
        let r = &mut results[i];
        let conf = match det.find(frame, region, det.tools[&t.id], &mut cache, plan.get(&t.id).copied(), pad) {
            Ok(c) => c,
            Err(e) => {
                r.values.insert("ai".into(), json!(format!("error: {}", e)));
                continue;
            }
        };
        let found = conf >= det.threshold(&t.id);
        r.values.insert("ai_confidence".into(), json!((conf as f64 * 1000.0).round() / 1000.0));
        r.values.insert("ai_found".into(), json!(found));
        let text = kind(&t.ty) == Some("text");
        let sure = if text { SURE_TEXT } else { SURE_PART };
        if r.status == "pass" && !found && r.score >= sure {
            r.values.insert("ai_overruled".into(), json!(false));
            r.values.insert("ai_note".into(), json!(format!("AI unsure ({:.2}); the check itself is sure ({:.2})", conf, r.score)));
            continue;
        }
        if r.status == "pass" && !found {
            r.status = "fail".into();
            r.message = format!("AI did not find the {} here ({:.2})", if text { "text" } else { "part" }, conf);
            for g in r.graphics.iter_mut() {
                if g.status == "pass" {
                    g.status = "fail".into();
                }
            }
        }
    }
}
