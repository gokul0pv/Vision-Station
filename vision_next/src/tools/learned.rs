//! Trained models: Detect with YOLO, Classify.

use crate::img::Img;
use crate::infer;
use crate::model::{jnum, Ctx, Graphic, ToolResult, ToolSpec};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn model_path(t: &ToolSpec, ctx: &Ctx, default: &str) -> PathBuf {
    let m = t.s("model", default);
    let p = PathBuf::from(&m);
    if p.is_absolute() { p } else { ctx.folder.join(p) }
}

fn class_names(meta: Option<String>) -> Vec<String> {
    let Some(s) = meta else { return vec![] };
    let rx = regex::Regex::new(r#"(\d+)\s*:\s*['"]([^'"]*)['"]"#).unwrap();
    let mut v: Vec<(usize, String)> = rx.captures_iter(&s).filter_map(|c| Some((c[1].parse().ok()?, c[2].to_string()))).collect();
    v.sort();
    v.into_iter().map(|x| x.1).collect()
}

struct Det {
    b: [f64; 4], // x0 y0 x1 y1 in patch pixels
    score: f32,
    cls: usize,
}

fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let ix = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let iy = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let i = ix * iy;
    let u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
    if u <= 0.0 { 0.0 } else { i / u }
}

pub fn yolo_detect(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let path = model_path(t, ctx, "models/best.onnx");
    let sess = match infer::session(&path) {
        Ok(s) => s,
        Err(e) => return ToolResult::error(t, e),
    };
    let (dims, names_meta) = infer::describe(&sess);
    let names = class_names(names_meta);
    let size = dims.get(2).copied().filter(|&v| v > 0).unwrap_or(640) as usize;
    let (patch, reg) = match ctx.region(t, frame) {
        Some(r) => (r.patch.clone(), Some(r)),
        None => (frame.clone(), None),
    };
    // letterbox, grey 114
    let s = (size as f64 / patch.w as f64).min(size as f64 / patch.h as f64);
    let (nw, nh) = (((patch.w as f64 * s).round() as usize).max(1), ((patch.h as f64 * s).round() as usize).max(1));
    let small = patch.resize(nw, nh);
    let (px, py) = ((size - nw) / 2, (size - nh) / 2);
    let mut x = vec![114.0f32 / 255.0; 3 * size * size];
    for yy in 0..nh {
        for xx in 0..nw {
            let p = small.rgb(xx, yy);
            for c in 0..3 {
                x[c * size * size + (yy + py) * size + xx + px] = p[c] as f32 / 255.0;
            }
        }
    }
    let (shape, out) = match infer::run(&sess, &[1, 3, size, size], x) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(t, e),
    };
    if shape.len() != 3 {
        return ToolResult::error(t, format!("unexpected model output {:?}", shape));
    }
    // [1, 4 + classes, anchors] (YOLOv8 / 11) or [1, anchors, 4 + classes]
    let (rows, n, transposed) = if shape[1] < shape[2] { (shape[1], shape[2], false) } else { (shape[2], shape[1], true) };
    let at = |r: usize, i: usize| if transposed { out[i * rows + r] } else { out[r * n + i] };
    let conf = t.f("confidence", 0.35) as f32;
    let only: Vec<String> = t.strlist("only_classes");
    let min_area = t.f("min_area", 0.0);
    let mut dets: Vec<Det> = vec![];
    for i in 0..n {
        let (mut bc, mut bs) = (0usize, f32::MIN);
        for c in 4..rows {
            let v = at(c, i);
            if v > bs {
                bs = v;
                bc = c - 4;
            }
        }
        if bs < conf {
            continue;
        }
        let (cx, cy, w, h) = (at(0, i) as f64, at(1, i) as f64, at(2, i) as f64, at(3, i) as f64);
        let b = [(cx - w / 2.0 - px as f64) / s, (cy - h / 2.0 - py as f64) / s, (cx + w / 2.0 - px as f64) / s, (cy + h / 2.0 - py as f64) / s];
        if (b[2] - b[0]) * (b[3] - b[1]) < min_area {
            continue;
        }
        let name = names.get(bc).cloned().unwrap_or_else(|| bc.to_string());
        if !only.is_empty() && !only.contains(&name) {
            continue;
        }
        dets.push(Det { b, score: bs, cls: bc });
    }
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let iou_thr = t.f("iou", 0.45);
    let mut kept: Vec<Det> = vec![];
    for d in dets {
        if !kept.iter().any(|k| k.cls == d.cls && iou(&k.b, &d.b) > iou_thr) {
            kept.push(d);
        }
    }
    let mut counts: BTreeMap<String, i64> = BTreeMap::new();
    for d in &kept {
        *counts.entry(names.get(d.cls).cloned().unwrap_or_else(|| d.cls.to_string())).or_default() += 1;
    }
    let mut why = vec![];
    let total = kept.len() as i64;
    if t.b("reject_when_found", false) || t.ty == "yolo_defect" {
        let allowed = t.i("max_defects", 0);
        if total > allowed {
            why.push(format!("{} found (allowed {})", total, allowed));
        }
    } else {
        let et = t.i("expect_total", 0);
        if et > 0 && total != et {
            why.push(format!("{} found, expected {}", total, et));
        }
        let ec: BTreeMap<String, i64> = match t.p.get("expect_counts") {
            Some(Value::Object(o)) => o.iter().filter_map(|(k, v)| Some((k.clone(), v.as_i64()?))).collect(),
            Some(Value::String(s)) => serde_json::from_str::<BTreeMap<String, i64>>(s).unwrap_or_default(),
            _ => BTreeMap::new(),
        };
        for (k, v) in &ec {
            let got = counts.get(k).copied().unwrap_or(0);
            if got != *v {
                why.push(format!("{} {} (expected {})", got, k, v));
            }
        }
    }
    let mut r = ToolResult::new(t, "pass");
    r.pass_if(why.is_empty());
    r.message = why.join("; ");
    r.score = kept.iter().map(|d| d.score as f64).fold(0.0, f64::max);
    r.set("total", total).set("counts", json!(counts));
    let st = r.status.clone();
    for d in &kept {
        let label = format!("{} {:.2}", names.get(d.cls).cloned().unwrap_or_else(|| d.cls.to_string()), d.score);
        let pts = match &reg {
            Some(rg) => vec![rg.map(d.b[0], d.b[1]), rg.map(d.b[2], d.b[1]), rg.map(d.b[2], d.b[3]), rg.map(d.b[0], d.b[3])],
            None => vec![[d.b[0], d.b[1]], [d.b[2], d.b[1]], [d.b[2], d.b[3]], [d.b[0], d.b[3]]],
        };
        r.graphics.push(Graphic::poly(pts, &st, &label));
    }
    r
}

pub fn classify(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(reg) = ctx.region(t, frame) else { return ToolResult::error(t, "no region drawn") };
    let path = model_path(t, ctx, "");
    let sess = match infer::session(&path) {
        Ok(s) => s,
        Err(e) => return ToolResult::error(t, e),
    };
    let (dims, _) = infer::describe(&sess);
    let size = dims.get(2).copied().filter(|&v| v > 0).unwrap_or(224) as usize;
    let small = reg.patch.resize(size, size);
    let imagenet = t.s("normalize", "imagenet") == "imagenet";
    let (mean, std) = if imagenet { ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]) } else { ([0.0; 3], [1.0; 3]) };
    let mut x = vec![0f32; 3 * size * size];
    for y in 0..size {
        for xx in 0..size {
            let p = small.rgb(xx, y);
            for c in 0..3 {
                x[c * size * size + y * size + xx] = (p[c] as f32 / 255.0 - mean[c]) / std[c];
            }
        }
    }
    let (_, logits) = match infer::run(&sess, &[1, 3, size, size], x) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(t, e),
    };
    // softmax unless already probabilities
    let sum: f32 = logits.iter().sum();
    let probs: Vec<f32> = if (sum - 1.0).abs() < 1e-3 && logits.iter().all(|&v| v >= 0.0) {
        logits
    } else {
        let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
        let e: Vec<f32> = logits.iter().map(|v| (v - mx).exp()).collect();
        let s: f32 = e.iter().sum();
        e.iter().map(|v| v / s).collect()
    };
    let (bi, bp) = probs.iter().enumerate().fold((0, f32::MIN), |a, (i, &p)| if p > a.1 { (i, p) } else { a });
    let classes = t.strlist("classes");
    let name = classes.get(bi).cloned().unwrap_or_else(|| bi.to_string());
    let accept = t.strlist("accept");
    let minc = t.f("min_confidence", 0.6);
    let mut r = ToolResult::new(t, "pass");
    let ok = (accept.is_empty() || accept.contains(&name)) && bp as f64 >= minc;
    r.pass_if(ok);
    if !ok {
        r.message = format!("{} at {:.0}%", name, bp * 100.0);
    }
    r.score = bp as f64;
    r.set("class", name.clone()).set("confidence", jnum(bp as f64, 3));
    r.caption = vec![name];
    r
}
