//! Reading print: PP-OCR detection (DB) and recognition (CTC) through ONNX
//! Runtime, the same models and rules as Vision Station's ocr_engine.py.

use crate::img::{self, Img};
use crate::infer;
use std::sync::{Arc, OnceLock};

const TILE: usize = 1280;
const OVERLAP: usize = 192;
const REC_H: usize = 48;
const REC_MAX_W: usize = 3200;
const REC_BATCH: usize = 16;
const MEAN: [f32; 3] = [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0];
const STD: [f32; 3] = [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0];

pub struct Models {
    det: Arc<infer::Model>,
    rec: Arc<infer::Model>,
    chars: Vec<String>,
    pub names: String,
}

static MODELS: OnceLock<Result<Models, String>> = OnceLock::new();

pub fn models() -> Result<&'static Models, String> {
    MODELS.get_or_init(|| {
        let dir = crate::paths::models().join("ocr");
        let pick = |names: &[&str]| names.iter().map(|n| dir.join(n)).find(|p| p.exists());
        let det = pick(&["det_medium.onnx", "det_server.onnx", "det_small.onnx", "det_mobile.onnx"])
            .ok_or(format!("no text detector in {}", dir.display()))?;
        let rec = pick(&["rec_medium.onnx", "rec_server.onnx", "rec_small.onnx", "rec_mobile.onnx"])
            .ok_or(format!("no text recogniser in {}", dir.display()))?;
        let dict = [rec.with_extension("txt"), dir.join("rec_dict.txt")].into_iter().find(|p| p.exists())
            .ok_or("no character dictionary beside the recogniser")?;
        let text = std::fs::read_to_string(&dict).map_err(|e| e.to_string())?;
        let mut chars: Vec<String> = vec!["blank".into()];
        let mut lines: Vec<String> = text.split('\n').map(|l| l.trim_end_matches('\r').to_string()).collect();
        while lines.last().map(|l| l.is_empty()).unwrap_or(false) {
            lines.pop();
        }
        chars.extend(lines);
        chars.push(" ".into());
        let names = format!("{} + {}", det.file_name().unwrap().to_string_lossy(), rec.file_name().unwrap().to_string_lossy());
        Ok(Models { det: infer::session(&det)?, rec: infer::session(&rec)?, chars, names })
    }).as_ref().map_err(|e| e.clone())
}

#[derive(Clone, Debug)]
pub struct Line {
    pub text: String,
    /// rounded to 4 places, as Vision Station reports it
    pub conf: f64,
    pub char_conf: Vec<f64>,
    pub poly: [[f64; 2]; 4],
    pub turn: usize,
    pub core: Option<String>,
    /// Vision Station's whole-number box when it is not the polygon's own
    /// (a rescue read at twice the size keeps int(box / 2))
    pub box_override: Option<[i64; 4]>,
}

impl Line {
    pub fn bbox(&self) -> [f64; 4] {
        let xs = self.poly.iter().map(|p| p[0]);
        let ys = self.poly.iter().map(|p| p[1]);
        let x0 = xs.clone().fold(f64::MAX, f64::min);
        let x1 = xs.fold(f64::MIN, f64::max);
        let y0 = ys.clone().fold(f64::MAX, f64::min);
        let y1 = ys.fold(f64::MIN, f64::max);
        [x0, y0, x1 - x0, y1 - y0]
    }
    fn thickness(&self) -> f64 {
        let b = self.ibox();
        b[2].min(b[3]) as f64
    }
    /// Vision Station's "box": [int(x0), int(y0), int(round(x1 - x0)), int(round(y1 - y0))]
    pub fn ibox(&self) -> [i64; 4] {
        if let Some(b) = self.box_override {
            return b;
        }
        let b = self.bbox();
        [b[0].trunc() as i64, b[1].trunc() as i64, b[2].round_ties_even() as i64, b[3].round_ties_even() as i64]
    }
}

/// round(x, 4) as Python does it (the exact value, ties to even).
pub fn round4(x: f64) -> f64 {
    format!("{:.4}", x).parse().unwrap_or(x)
}

/// numpy's pairwise sum of float64 (what np.mean adds with).
fn np_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0;
        for &x in a {
            r += x;
        }
        r
    } else if n <= 128 {
        let mut r = [0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &a[n - n % 8..] {
            res += x;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        np_sum(&a[..n2]) + np_sum(&a[n2..])
    }
}

// ---------------------------------------------------------------- detection
fn det_blob(img: &Img) -> (Vec<f32>, usize, usize) {
    let (h, w) = (img.h, img.w);
    let hh = h.div_ceil(32) * 32;
    let ww = w.div_ceil(32) * 32;
    let mut x = vec![0f32; 3 * hh * ww];
    for y in 0..hh {
        let sy = y.min(h - 1);
        for xx in 0..ww {
            let sx = xx.min(w - 1);
            let p = img.rgb(sx, sy);
            // BGR order, as the models were trained
            let bgr = [p[2], p[1], p[0]];
            for c in 0..3 {
                x[c * hh * ww + y * ww + xx] = (bgr[c] as f32 - MEAN[c]) / STD[c];
            }
        }
    }
    (x, hh, ww)
}

fn cross(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

fn hull(mut pts: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let mut lo: Vec<[f64; 2]> = vec![];
    for &p in &pts {
        while lo.len() >= 2 && cross(lo[lo.len() - 2], lo[lo.len() - 1], p) <= 0.0 {
            lo.pop();
        }
        lo.push(p);
    }
    let mut hi: Vec<[f64; 2]> = vec![];
    for &p in pts.iter().rev() {
        while hi.len() >= 2 && cross(hi[hi.len() - 2], hi[hi.len() - 1], p) <= 0.0 {
            hi.pop();
        }
        hi.push(p);
    }
    lo.pop();
    hi.pop();
    lo.extend(hi);
    lo
}

/// Minimum-area rectangle: (centre, (along, across), unit direction along).
fn min_area_rect(h: &[[f64; 2]]) -> ([f64; 2], (f64, f64), [f64; 2]) {
    if h.len() == 1 {
        return (h[0], (0.0, 0.0), [1.0, 0.0]);
    }
    let mut best = (f64::MAX, [0.0; 2], (0.0, 0.0), [1.0, 0.0]);
    for i in 0..h.len() {
        let a = h[i];
        let b = h[(i + 1) % h.len()];
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l = (dx * dx + dy * dy).sqrt();
        if l < 1e-9 {
            continue;
        }
        let u = [dx / l, dy / l];
        let v = [-u[1], u[0]];
        let (mut u0, mut u1, mut v0, mut v1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for p in h {
            let pu = p[0] * u[0] + p[1] * u[1];
            let pv = p[0] * v[0] + p[1] * v[1];
            u0 = u0.min(pu);
            u1 = u1.max(pu);
            v0 = v0.min(pv);
            v1 = v1.max(pv);
        }
        let area = (u1 - u0) * (v1 - v0);
        if area < best.0 {
            let cu = (u0 + u1) / 2.0;
            let cv = (v0 + v1) / 2.0;
            let c = [cu * u[0] + cv * v[0], cu * u[1] + cv * v[1]];
            best = (area, c, (u1 - u0, v1 - v0), u);
        }
    }
    (best.1, best.2, best.3)
}

/// Corners as top-left, top-right, bottom-right, bottom-left.
fn order(mut pts: [[f64; 2]; 4]) -> [[f64; 2]; 4] {
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
    let (mut l, mut r) = ([pts[0], pts[1]], [pts[2], pts[3]]);
    l.sort_by(|a, b| a[1].partial_cmp(&b[1]).unwrap());
    r.sort_by(|a, b| a[1].partial_cmp(&b[1]).unwrap());
    [l[0], r[0], r[1], l[1]]
}

fn boxes_from_map(prob: &[f32], w: usize, h: usize) -> Vec<([[f64; 2]; 4], f32)> {
    let mask: Vec<u8> = prob.iter().map(|&p| if p > 0.2 { 255 } else { 0 }).collect();
    let mut out = vec![];
    for b in img::components(&mask, w, h, true).into_iter().take(3000) {
        if b.area < 3 {
            continue;
        }
        // the outline is enough for the hull
        let pts: Vec<[f64; 2]> = b.pixels.iter().filter_map(|&i| {
            let (x, y) = (i as usize % w, i as usize / w);
            let edge = x == 0 || y == 0 || x + 1 >= w || y + 1 >= h
                || mask[i as usize - 1] == 0 || mask[i as usize + 1] == 0
                || mask[i as usize - w] == 0 || mask[i as usize + w] == 0;
            if edge { Some([x as f64, y as f64]) } else { None }
        }).collect();
        let hl = hull(pts);
        let (c, (rw, rh), u) = min_area_rect(&hl);
        if rw.min(rh) < 2.0 {
            continue;
        }
        let score = b.pixels.iter().map(|&i| prob[i as usize]).sum::<f32>() / b.pixels.len() as f32;
        if score < 0.45 {
            continue;
        }
        let grow = (rw * rh) * 1.5 / (2.0 * (rw + rh)).max(1e-6);
        let (rw2, rh2) = (rw + 2.0 * grow, rh + 2.0 * grow);
        if rw2.min(rh2) < 4.0 {
            continue;
        }
        let v = [-u[1], u[0]];
        let corner = |su: f64, sv: f64| [c[0] + su * rw2 / 2.0 * u[0] + sv * rh2 / 2.0 * v[0], c[1] + su * rw2 / 2.0 * u[1] + sv * rh2 / 2.0 * v[1]];
        out.push((order([corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)]), score));
    }
    out
}

fn detect_one(m: &Models, im: &Img, scale: f64) -> Result<Vec<([[f64; 2]; 4], f32)>, String> {
    // cv2.resize: INTER_CUBIC to enlarge (as opencv-python's IPP does it), INTER_AREA to shrink
    let work;
    let src = if scale != 1.0 {
        work = if scale > 1.0 {
            crate::cvx::resize_cubic_scale(im, scale)
        } else {
            im.resize(((im.w as f64 * scale).round_ties_even() as usize).max(1), ((im.h as f64 * scale).round_ties_even() as usize).max(1))
        };
        &work
    } else {
        im
    };
    let (blob, hh, ww) = det_blob(src);
    let (shape, prob) = infer::run(&m.det, &[1, 3, hh, ww], blob)?;
    let (ph, pw) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    // the map cut to the picture (the padding is not print)
    let (w, h) = (src.w.min(pw), src.h.min(ph));
    let mut cropped = vec![0f32; w * h];
    for y in 0..h {
        cropped[y * w..y * w + w].copy_from_slice(&prob[y * pw..y * pw + w]);
    }
    // b / scale in float32, as numpy divides a float32 array by a Python float
    let sc = scale as f32;
    Ok(crate::cvx::boxes_from_map(&cropped, w, h).into_iter().map(|(b, s)| {
        let q: [[f64; 2]; 4] = std::array::from_fn(|i| [(b[i][0] / sc) as f64, (b[i][1] / sc) as f64]);
        (q, s as f32)
    }).collect())
}

fn short_side(b: &[[f64; 2]; 4]) -> f64 {
    let d = |a: [f64; 2], c: [f64; 2]| ((a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2)).sqrt();
    d(b[0], b[3]).min(d(b[0], b[1]))
}

fn rect_of(b: &[[f64; 2]; 4]) -> [f64; 4] {
    let x0 = b.iter().map(|p| p[0]).fold(f64::MAX, f64::min);
    let y0 = b.iter().map(|p| p[1]).fold(f64::MAX, f64::min);
    let x1 = b.iter().map(|p| p[0]).fold(f64::MIN, f64::max);
    let y1 = b.iter().map(|p| p[1]).fold(f64::MIN, f64::max);
    [x0, y0, x1, y1]
}

fn starts(n: usize) -> Vec<usize> {
    if n <= TILE {
        return vec![0];
    }
    let step = TILE - OVERLAP;
    let mut s: Vec<usize> = (0..n - TILE).step_by(step).collect();
    s.push(n - TILE);
    s.sort();
    s.dedup();
    s
}

pub fn detect(m: &Models, im: &Img, detail: &str) -> Result<Vec<([[f64; 2]; 4], f32)>, String> {
    let long = im.w.max(im.h);
    if detail == "fast" || (detail != "fine" && long <= TILE) {
        let mut scale = 1.0;
        if long < 960 && detail != "fast" {
            scale = (960.0 / long as f64).min(3.0);
        } else if long > 1600 {
            scale = 1600.0 / long as f64;
        }
        return detect_one(m, im, scale);
    }
    let up = if detail == "fine" { 2.0 } else { 1.0 };
    let work = if up > 1.0 { im.resize(im.w * 2, im.h * 2) } else { im.clone() };
    let mut found: Vec<([[f64; 2]; 4], f32)> = vec![];
    for y in starts(work.h) {
        for x in starts(work.w) {
            let tile = work.crop(x as i64, y as i64, TILE.min(work.w), TILE.min(work.h));
            for (mut b, s) in detect_one(m, &tile, 1.0)? {
                for p in b.iter_mut() {
                    p[0] = (p[0] + x as f64) / up;
                    p[1] = (p[1] + y as f64) / up;
                }
                // a line cut by a tile edge is found twice: keep the bigger
                let r = rect_of(&b);
                let area = (r[2] - r[0]) * (r[3] - r[1]);
                let c = [(r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0];
                let mut dup = false;
                for f in found.iter_mut() {
                    let q = rect_of(&f.0);
                    let qa = (q[2] - q[0]) * (q[3] - q[1]);
                    let qc = [(q[0] + q[2]) / 2.0, (q[1] + q[3]) / 2.0];
                    let inside_q = c[0] >= q[0] && c[0] <= q[2] && c[1] >= q[1] && c[1] <= q[3];
                    let inside_r = qc[0] >= r[0] && qc[0] <= r[2] && qc[1] >= r[1] && qc[1] <= r[3];
                    if inside_q || inside_r {
                        if area > qa {
                            *f = (b, s);
                        }
                        dup = true;
                        break;
                    }
                }
                if !dup {
                    found.push((b, s));
                }
            }
        }
    }
    // big lettering is cut into pieces by the tiles: a coarse pass sees it whole
    let coarse = (TILE as f64 / long as f64).min(1.0);
    for (bb, bs) in detect_one(m, im, coarse)? {
        if short_side(&bb) <= 0.5 * OVERLAP as f64 {
            continue;
        }
        let r = rect_of(&bb);
        let inside: Vec<usize> = found.iter().enumerate().filter(|(_, f)| {
            let q = rect_of(&f.0);
            let c = [(q[0] + q[2]) / 2.0, (q[1] + q[3]) / 2.0];
            c[0] >= r[0] && c[0] <= r[2] && c[1] >= r[1] && c[1] <= r[3]
        }).map(|x| x.0).collect();
        if inside.iter().any(|&i| short_side(&found[i].0) < 0.4 * short_side(&bb)) {
            continue;
        }
        let keep: Vec<_> = found.iter().enumerate().filter(|(i, _)| !inside.contains(i)).map(|x| *x.1).collect();
        found = keep;
        found.push((bb, bs));
    }
    Ok(found)
}

// -------------------------------------------------------------- recognition
fn crop_line(im: &Img, b: &[[f64; 2]; 4]) -> Img {
    // ocr_engine._crop_line: float32 corners, sizes from their float32 distances
    let p: [[f32; 2]; 4] = std::array::from_fn(|i| [b[i][0] as f32, b[i][1] as f32]);
    let d = |a: [f32; 2], c: [f32; 2]| { let (x, y) = (a[0] - c[0], a[1] - c[1]); (x * x + y * y).sqrt() };
    let w = (d(p[0], p[1]).max(d(p[3], p[2])).round_ties_even() as usize).max(2);
    let h = (d(p[0], p[3]).max(d(p[1], p[2])).round_ties_even() as usize).max(2);
    let dst = [[0.0, 0.0], [w as f32, 0.0], [w as f32, h as f32], [0.0, h as f32]];
    let m = crate::cvx::get_perspective_transform(&p, &dst);
    crate::cvx::warp_perspective_cubic(im, &m, w, h)
}

fn decode(probs: &[f32], t: usize, c: usize, chars: &[String]) -> (String, f64, Vec<f64>) {
    let mut text = String::new();
    let mut confs = vec![];
    let mut prev = 0usize;
    for i in 0..t {
        let row = &probs[i * c..(i + 1) * c];
        let (mut bi, mut bv) = (0usize, f32::MIN);
        for (k, &v) in row.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = k;
            }
        }
        if bi != 0 && bi != prev && bi < chars.len() {
            text.push_str(&chars[bi]);
            confs.push(bv as f64);
        }
        prev = bi;
    }
    let mean = if confs.is_empty() { 0.0 } else { np_sum(&confs) / confs.len() as f64 };
    (text, mean, confs)
}

fn recognise(m: &Models, crops: &[Img]) -> Result<Vec<(String, f64, Vec<f64>)>, String> {
    let mut out = vec![(String::new(), 0f64, vec![]); crops.len()];
    let mut idx: Vec<usize> = (0..crops.len()).collect();
    idx.sort_by(|&a, &b| {
        let ra = crops[a].w as f64 / crops[a].h as f64;
        let rb = crops[b].w as f64 / crops[b].h as f64;
        ra.partial_cmp(&rb).unwrap()
    });
    for chunk in idx.chunks(REC_BATCH) {
        let ratio = chunk.iter().map(|&i| crops[i].w as f64 / crops[i].h as f64).fold(0.0, f64::max);
        let width = ((REC_H as f64 * ratio / 160.0).ceil() as usize * 160).clamp(160, REC_MAX_W);
        let n = chunk.len();
        let mut x = vec![0f32; n * 3 * REC_H * width];
        for (j, &i) in chunk.iter().enumerate() {
            let c = &crops[i];
            let nw = ((REC_H as f64 * c.w as f64 / c.h as f64).ceil() as usize).clamp(8, width);
            let r = crate::cvx::resize_linear(c, nw, REC_H);
            for y in 0..REC_H {
                for xx in 0..nw {
                    let p = r.rgb(xx, y);
                    let bgr = [p[2], p[1], p[0]];
                    for ch in 0..3 {
                        x[((j * 3 + ch) * REC_H + y) * width + xx] = (bgr[ch] as f32 / 255.0 - 0.5) / 0.5;
                    }
                }
            }
        }
        let (shape, probs) = infer::run(&m.rec, &[n, 3, REC_H, width], x)?;
        let (t, c) = (shape[1], shape[2]);
        for (j, &i) in chunk.iter().enumerate() {
            out[i] = decode(&probs[j * t * c..(j + 1) * t * c], t, c, &m.chars);
        }
    }
    Ok(out)
}

fn score(text: &str, conf: f64) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    let n = text.chars().count() as f64;
    let alnum = text.chars().filter(|c| c.is_alphanumeric()).count() as f64 / n;
    conf * (0.5 + 0.5 * alnum)
}

fn turns_for(c: &Img) -> &'static [usize] {
    if c.w as f64 >= 1.5 * c.h as f64 {
        &[0, 2]
    } else if c.h as f64 >= 1.5 * c.w as f64 {
        &[1, 3]
    } else {
        &[0, 2, 1, 3]
    }
}

pub fn looks_like_a_pattern(text: &str) -> bool {
    let t: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if t.len() < 7 {
        return false;
    }
    let mut best = (' ', 0usize);
    for &ch in &t {
        let n = t.iter().filter(|&&x| x == ch).count();
        if n > best.1 {
            best = (ch, n);
        }
    }
    best.0.is_alphabetic() && best.1 as f64 >= 0.6 * t.len() as f64
}

fn strip_confs(text: &str, confs: &[f64]) -> (String, Vec<f64>) {
    let chars: Vec<char> = text.chars().collect();
    let lead = chars.iter().take_while(|c| c.is_whitespace()).count();
    let trail = chars.iter().rev().take_while(|c| c.is_whitespace()).count();
    let n = chars.len().saturating_sub(lead + trail);
    let s: String = chars.iter().skip(lead).take(n).collect();
    // confidences belong to characters as decoded; whitespace chars are in confs too
    let cc: Vec<f64> = confs.iter().skip(lead).take(n).copied().collect();
    (s, cc)
}

fn read_lines(m: &Models, im: &Img, boxes: &[([[f64; 2]; 4], f32)]) -> Result<Vec<Line>, String> {
    let crops: Vec<Img> = boxes.iter().map(|(b, _)| crop_line(im, b)).collect();
    if crops.is_empty() {
        return Ok(vec![]);
    }
    let mut jobs = vec![];
    let mut owner = vec![];
    for (i, c) in crops.iter().enumerate() {
        for &k in turns_for(c) {
            jobs.push(if k == 0 { c.clone() } else { c.rot90(k) });
            owner.push((i, k));
        }
    }
    let read = recognise(m, &jobs)?;
    // candidates: (turn, text, conf, score, confs)
    let mut cands: Vec<Vec<(usize, String, f64, f64, Vec<f64>)>> = vec![vec![]; crops.len()];
    for ((i, k), (t, c, cc)) in owner.into_iter().zip(read) {
        let (ts, ccs) = strip_confs(&t, &cc);
        let sc = score(&ts, c);
        cands[i].push((k, ts, c, sc, ccs));
    }
    // the way up most of the print agrees on (in the order first voted, as a Python dict)
    let mut votes: Vec<(usize, f64)> = vec![];
    for cs in &cands {
        let mut ranked: Vec<_> = cs.iter().collect();
        ranked.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        if ranked.len() < 2 || ranked[0].1.chars().count() < 3 {
            continue;
        }
        let margin = ranked[0].3 - ranked[1].3;
        if margin >= 0.15 {
            let add = margin * ranked[0].1.chars().count().min(12) as f64;
            match votes.iter_mut().find(|v| v.0 == ranked[0].0) {
                Some(v) => v.1 += add,
                None => votes.push((ranked[0].0, add)),
            }
        }
    }
    // max(votes, key=votes.get): the first of equal counts
    let mut common: Option<usize> = None;
    let mut best_v = f64::MIN;
    for (k, v) in &votes {
        if common.is_none() || *v > best_v {
            common = Some(*k);
            best_v = *v;
        }
    }
    let mut out = vec![];
    for ((b, _), cs) in boxes.iter().zip(cands) {
        // max(cs, key=score): the first of equal scores
        let mut bi = 0;
        for (j, v) in cs.iter().enumerate() {
            if v.3 > cs[bi].3 {
                bi = j;
            }
        }
        let mut best = cs[bi].clone();
        if let Some(cm) = common {
            if best.0 != cm {
                if let Some(same) = cs.iter().find(|v| v.0 == cm) {
                    if !same.1.is_empty() && same.3 >= best.3 - 0.15 {
                        best = same.clone();
                    }
                }
            }
        }
        let (k, t, c, _, cc) = best;
        if t.is_empty() || looks_like_a_pattern(&t) {
            continue;
        }
        out.push(Line { text: t, conf: round4(c), char_conf: cc.iter().map(|&v| round4(v)).collect(), poly: *b, turn: k * 90, core: None, box_override: None });
    }
    Ok(out)
}

const TRIM_BELOW: f64 = 0.85;
const TRIM_CORE_ABOVE: f64 = 0.95;
const TRIM_MAX: usize = 2;

/// The line without probable stray marks at its ends (a second reading).
pub fn core_text(l: &Line) -> Option<String> {
    let chars: Vec<char> = l.text.chars().collect();
    if l.char_conf.len() != chars.len() || chars.len() < 3 {
        return None;
    }
    let keep: Vec<(char, f64)> = chars.iter().zip(&l.char_conf)
        .filter(|(ch, c)| ch.is_alphanumeric() || **ch == ' ' || **c >= TRIM_BELOW)
        .map(|(a, b)| (*a, *b)).collect();
    let (mut lo, mut hi, mut n) = (0usize, keep.len(), 0usize);
    while n < TRIM_MAX && hi - lo > 2 && (keep[lo].1 < TRIM_BELOW || keep[lo].0 == ' ') {
        if keep[lo].0 != ' ' { n += 1; }
        lo += 1;
    }
    n = 0;
    while n < TRIM_MAX && hi - lo > 2 && (keep[hi - 1].1 < TRIM_BELOW || keep[hi - 1].0 == ' ') {
        if keep[hi - 1].0 != ' ' { n += 1; }
        hi -= 1;
    }
    let core = &keep[lo..hi];
    let sure: Vec<f64> = core.iter().filter(|x| x.0 != ' ').map(|x| x.1).collect();
    let out: String = core.iter().map(|x| x.0).collect::<String>().trim().to_string();
    if sure.is_empty() || sure.iter().cloned().fold(f64::MAX, f64::min) < TRIM_CORE_ABOVE || out == l.text.trim() {
        return None;
    }
    Some(out)
}

/// Stray fragments dropped, each line given its core reading.
pub fn clean_lines(lines: Vec<Line>) -> Vec<Line> {
    let lines: Vec<Line> = lines.into_iter().map(|mut l| { l.core = core_text(&l); l }).collect();
    // the main print: the first most confident line of three or more characters
    let mut ri: Option<usize> = None;
    for (i, l) in lines.iter().enumerate() {
        if l.text.replace(' ', "").chars().count() >= 3 && ri.map(|r| l.conf > lines[r].conf).unwrap_or(true) {
            ri = Some(i);
        }
    }
    let Some(ri) = ri else { return lines };
    let tall = lines[ri].thickness();
    lines.into_iter().enumerate().filter(|(i, l)| {
        *i == ri || l.text.replace(' ', "").chars().count() > 2 || l.thickness() >= 0.5 * tall
    }).map(|x| x.1).collect()
}

fn reading_order(mut lines: Vec<Line>) -> Vec<Line> {
    if lines.is_empty() {
        return lines;
    }
    let mut hs: Vec<i64> = lines.iter().map(|l| { let b = l.ibox(); b[2].min(b[3]) }).collect();
    hs.sort();
    let row = (0.7 * hs[hs.len() / 2] as f64).max(6.0);
    // key (int((y + h / 2.0) // row), x) on the whole-number box; a stable sort
    lines.sort_by(|a, b| {
        let (ba, bb) = (a.ibox(), b.ibox());
        let ka = ((ba[1] as f64 + ba[3] as f64 / 2.0) / row).floor() as i64;
        let kb = ((bb[1] as f64 + bb[3] as f64 / 2.0) / row).floor() as i64;
        ka.cmp(&kb).then(ba[0].cmp(&bb[0]))
    });
    lines
}

/// At most this many reads at once: each holds a detector input of up to
/// ~11 MB, and 28 of them together were most of a run's peak memory.
struct Gate {
    n: std::sync::Mutex<usize>,
    cv: std::sync::Condvar,
}

fn gate() -> &'static Gate {
    static G: OnceLock<Gate> = OnceLock::new();
    G.get_or_init(|| Gate { n: std::sync::Mutex::new(0), cv: std::sync::Condvar::new() })
}

struct Pass;

impl Pass {
    fn take() -> Pass {
        let max = std::env::var("VS_OCR_PARALLEL").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize).max(1);
        let g = gate();
        let mut n = g.n.lock().unwrap();
        while *n >= max {
            n = g.cv.wait(n).unwrap();
        }
        *n += 1;
        Pass
    }
}

impl Drop for Pass {
    fn drop(&mut self) {
        let g = gate();
        *g.n.lock().unwrap() -= 1;
        g.cv.notify_one();
    }
}

/// Every line of print in `im`, in reading order.
pub fn read(im: &Img, detail: &str) -> Result<Vec<Line>, String> {
    let _pass = Pass::take();
    let m = models()?;
    let mut boxes = detect(m, im, detail)?;
    if detail == "auto" && !boxes.is_empty() && im.w * im.h <= 8_000_000 {
        let mut s: Vec<f64> = boxes.iter().map(|b| short_side(&b.0)).collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if s[s.len() / 2] < 26.0 && im.w.max(im.h) > 480 {
            let closer = detect(m, im, "fine")?;
            if closer.len() >= boxes.len() {
                boxes = closer;
            }
        }
    }
    let mut lines = read_lines(m, im, &boxes)?;
    if !lines.iter().any(|l| l.conf >= 0.5) && im.w * im.h <= 4_000_000 {
        let rescued = rescue(m, im)?;
        if !rescued.is_empty() {
            lines = rescued;
        }
    }
    Ok(reading_order(lines))
}

/// ocr_engine._rescue: nothing read at 0.5 or better from the picture as
/// taken, so read it smoothed; smoothed with the contrast raised; enlarged,
/// smoothed and stretched -- and keep the best (each OpenCV step exact).
fn rescue(m: &Models, im: &Img) -> Result<Vec<Line>, String> {
    let mut best: Vec<Line> = vec![];
    let mut best_score = 0.0f64;
    for k in 0..3 {
        let (v, scale) = match k {
            0 => (crate::cvx::gaussian_blur(im, 1.2), 1.0),
            1 => (crate::cvx::clahe_lab(&crate::cvx::gaussian_blur(im, 1.5)), 1.0),
            _ => (crate::cvx::stretch(&crate::cvx::gaussian_blur(&crate::cvx::resize_cubic_scale(im, 2.0), 2.5)), 2.0),
        };
        let boxes = detect(m, &v, "auto")?;
        let mut lines = read_lines(m, &v, &boxes)?;
        lines.retain(|l| l.conf >= 0.5);
        let mut score = 0.0f64;
        for l in &lines {
            score += l.conf * l.text.chars().count().min(20) as f64;
        }
        if score > best_score {
            for l in lines.iter_mut() {
                let b = l.ibox();
                l.box_override = Some(b.map(|v| (v as f64 / scale).trunc() as i64));
                for p in l.poly.iter_mut() {
                    p[0] /= scale;
                    p[1] /= scale;
                }
            }
            best = lines;
            best_score = score;
        }
    }
    Ok(best)
}

/// The whole picture as one line, every way up.
pub fn read_single_line(im: &Img) -> Result<Option<Line>, String> {
    let m = models()?;
    let (w, h) = (im.w as f64, im.h as f64);
    let b = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    Ok(read_lines(m, im, &[(b, 0.0)])?.into_iter().next())
}
