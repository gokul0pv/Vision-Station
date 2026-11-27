//! Find parts and text: every component (or line of text) in an area of the
//! reference picture, proposed as boxes the user reviews; the kept ones become
//! checks. A port of Vision Station's core/part_finder.py, with its numbers:
//!
//!   tiles      the area searched at full resolution in overlapping 1280-px
//!              tiles at zooms 2, 1 and 0.6, plus a whole-area pass for big
//!              parts; a box cut by a tile's inner edge is left to another pass
//!   merging    duplicates merged; a box that is really a group (round a row
//!              of capacitors) dropped for its members -- the more confident
//!              side wins, container or members
//!   board      a box that is mostly bare board is not a part; nor one with no
//!              board round it (the fixture beside the board)
//!   types      parts grouped by appearance (ResNet-18 features) and size
//!
//! The network is YOLOE-26s prompt-free, cut to box + objectness (models/parts).

use crate::img::Img;
use serde_json::{json, Value};
use std::path::PathBuf;

const SIZE: usize = 1280;
const OVERLAP: usize = 256;
const MAX_SIDE: f32 = 0.45;
const ZOOMS: [f32; 3] = [2.0, 1.0, 0.6];

pub type Bx = [f32; 4]; // x1 y1 x2 y2

pub fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("VS_PART_MODEL") {
        return PathBuf::from(p);
    }
    crate::paths::models().join("parts").join("parts_yoloe26s.onnx")
}

fn backbone_path() -> PathBuf {
    crate::paths::models().join("backbone").join("resnet18_features.onnx")
}

pub fn available() -> bool {
    model_path().exists()
}

fn area(b: &Bx) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

fn inter(a: &Bx, b: &Bx) -> f32 {
    ((a[2].min(b[2]) - a[0].max(b[0])).max(0.0)) * ((a[3].min(b[3]) - a[1].max(b[1])).max(0.0))
}

fn three(img: &Img) -> Img {
    if img.c >= 3 { img.clone() } else { img.to_rgb() }
}

/// Letterboxed to SIZE x SIZE (grey 114), RGB 0-1, CHW. (data, scale)
fn letterbox(img: &Img) -> (Vec<f32>, f32) {
    let s = (SIZE as f32 / img.w as f32).min(SIZE as f32 / img.h as f32);
    let (nw, nh) = (((img.w as f32 * s).round() as usize).max(1), ((img.h as f32 * s).round() as usize).max(1));
    let r = if (nw, nh) != (img.w, img.h) { img.resize(nw, nh) } else { img.clone() };
    let plane = SIZE * SIZE;
    let mut d = vec![114.0f32 / 255.0; 3 * plane];
    for y in 0..nh.min(SIZE) {
        for x in 0..nw.min(SIZE) {
            let i = (y * r.w + x) * r.c;
            for c in 0..3 {
                d[c * plane + y * SIZE + x] = r.data[i + c.min(r.c - 1)] as f32 / 255.0;
            }
        }
    }
    (d, s)
}

/// Raw candidates in img's pixels.
fn infer(img: &Img, max_side: f32) -> Result<Vec<(Bx, f32)>, String> {
    let m = crate::infer::session(&model_path())?;
    let (x, s) = letterbox(&three(img));
    let (shape, out) = crate::infer::run(&m, &[1, 3, SIZE, SIZE], x)?;
    let n = *shape.last().unwrap_or(&0);
    let mut v = vec![];
    for i in 0..n {
        let score = out[4 * n + i];
        let (cx, cy, w, h) = (out[i], out[n + i], out[2 * n + i], out[3 * n + i]);
        if score >= 0.02 && w.max(h) <= max_side * SIZE as f32 {
            v.push(([(cx - w / 2.0) / s, (cy - h / 2.0) / s, (cx + w / 2.0) / s, (cy + h / 2.0) / s], score));
        }
    }
    Ok(v)
}

fn starts(n: usize, tile: usize, overlap: usize) -> Vec<usize> {
    if n <= tile {
        return vec![0];
    }
    let step = (tile - overlap).max(1);
    let mut out: Vec<usize> = (0..n - tile).step_by(step).collect();
    out.push(n - tile);
    out.sort();
    out.dedup();
    out
}

/// Every box the network proposes in img, from tiles at each zoom and a
/// whole-image pass, before any clean-up.
pub fn candidates(img: &Img, conf: f32, progress: &dyn Fn(usize, usize)) -> Result<Vec<(Bx, f32)>, String> {
    let (h, w) = (img.h, img.w);
    let mut tiles = vec![];
    for z in ZOOMS {
        let src = (SIZE as f32 / z).round() as usize;
        let ov = (OVERLAP as f32 / z).round() as usize;
        if z <= 1.0 || h.max(w) as f32 > src as f32 * 0.75 {
            for y in starts(h, src, ov) {
                for x in starts(w, src, ov) {
                    tiles.push((x, y, src));
                }
            }
        }
    }
    let passes = tiles.len() + if h.max(w) > SIZE { 1 } else { 0 };
    let mut all = vec![];
    for (done, (x0, y0, src)) in tiles.into_iter().enumerate() {
        let tile = img.crop(x0 as i64, y0 as i64, src.min(w - x0), src.min(h - y0));
        let (th, tw) = (tile.h, tile.w);
        let z = if th.max(tw) < SIZE { SIZE as f32 / th.max(tw) as f32 } else { 1.0 };
        let t2 = if z > 1.0 { tile.resize((tw as f32 * z).round() as usize, (th as f32 * z).round() as usize) } else { tile };
        let m = 6.0 / z;
        for (b, s) in infer(&t2, MAX_SIDE)? {
            if s < conf {
                continue;
            }
            let b = [b[0] / z, b[1] / z, b[2] / z, b[3] / z];
            // cut by an inner tile edge: another pass sees it whole
            let inner = (b[0] < m && x0 > 0) || (b[1] < m && y0 > 0) || (b[2] > tw as f32 - m && x0 + tw < w) || (b[3] > th as f32 - m && y0 + th < h);
            if !inner {
                all.push(([b[0] + x0 as f32, b[1] + y0 as f32, b[2] + x0 as f32, b[3] + y0 as f32], s));
            }
        }
        progress(done + 1, passes);
    }
    if h.max(w) > SIZE {
        for (b, s) in infer(img, 0.6)? {
            if s >= conf {
                all.push((b, s));
            }
        }
        progress(passes, passes);
    }
    for (b, _) in all.iter_mut() {
        b[0] = b[0].clamp(0.0, w as f32);
        b[2] = b[2].clamp(0.0, w as f32);
        b[1] = b[1].clamp(0.0, h as f32);
        b[3] = b[3].clamp(0.0, h as f32);
    }
    Ok(all)
}

/// Greedy non-maximum suppression by IoU; indices kept, best first.
fn nms(c: &[(Bx, f32)], iou: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..c.len()).collect();
    order.sort_by(|&a, &b| c[b].1.partial_cmp(&c[a].1).unwrap());
    let mut kept: Vec<usize> = vec![];
    for i in order {
        let ok = kept.iter().all(|&k| {
            let it = inter(&c[i].0, &c[k].0);
            it / (area(&c[i].0) + area(&c[k].0) - it).max(1e-6) <= iou
        });
        if ok {
            kept.push(i);
        }
    }
    kept
}

/// Greedy, best first: a box overlapping a kept one by more than `share` of
/// the smaller of the two is dropped (parts on a board do not overlap).
fn no_overlap(c: &[(Bx, f32)], share: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..c.len()).collect();
    order.sort_by(|&a, &b| c[b].1.partial_cmp(&c[a].1).unwrap());
    let mut kept: Vec<usize> = vec![];
    for i in order {
        if kept.iter().any(|&k| inter(&c[i].0, &c[k].0) / area(&c[i].0).min(area(&c[k].0)).max(1e-6) > share) {
            continue;
        }
        kept.push(i);
    }
    kept
}

/// Area of `bx` covered by the union of member boxes (approximate, on a grid).
fn coverage(bx: &Bx, members: &[Bx]) -> f32 {
    const G: usize = 64;
    let (w, h) = ((bx[2] - bx[0]).max(1e-6), (bx[3] - bx[1]).max(1e-6));
    let mut m = vec![false; G * G];
    for b in members {
        let a = (((b[0] - bx[0]) / w * G as f32) as i64).clamp(0, G as i64) as usize;
        let c = (((b[2] - bx[0]) / w * G as f32).ceil() as i64).clamp(0, G as i64) as usize;
        let d = (((b[1] - bx[1]) / h * G as f32) as i64).clamp(0, G as i64) as usize;
        let e = (((b[3] - bx[1]) / h * G as f32).ceil() as i64).clamp(0, G as i64) as usize;
        for y in d..e {
            for x in a..c {
                m[y * G + x] = true;
            }
        }
    }
    m.iter().filter(|&&v| v).count() as f32 / (G * G) as f32 * w * h
}

/// Indices kept: a container with two or more boxes inside is one part with
/// details or a group -- the more confident side wins; a box mostly inside
/// another of about its size is a duplicate. `others` are weaker candidates
/// that count as a group's members but are never kept themselves.
fn drop_groups_and_duplicates(c: &[(Bx, f32)], others: &[(Bx, f32)]) -> Vec<usize> {
    let n = c.len();
    let ar: Vec<f32> = c.iter().map(|x| area(&x.0)).collect();
    let inside = |i: usize, j: usize| if i == j { 0.0 } else { inter(&c[i].0, &c[j].0) / ar[i].max(1e-6) };
    let mut keep = vec![true; n];
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| ar[b].partial_cmp(&ar[a]).unwrap());
    for &i in &order {
        if !keep[i] {
            continue;
        }
        let members: Vec<usize> = (0..n).filter(|&j| keep[j] && inside(j, i) > 0.85 && ar[j] < ar[i] * 0.5).collect();
        let mut cover: f32 = members.iter().map(|&j| ar[j]).sum();
        let mut count = members.len();
        let mut best = members.iter().map(|&j| c[j].1).fold(0.0f32, f32::max);
        let om: Vec<&(Bx, f32)> = others.iter().filter(|o| inter(&o.0, &c[i].0) / area(&o.0).max(1e-6) > 0.85 && area(&o.0) < ar[i] * 0.5).collect();
        if !om.is_empty() {
            count = count.max(om.len());
            cover = cover.max(coverage(&c[i].0, &om.iter().map(|o| o.0).collect::<Vec<_>>()));
            best = best.max(om.iter().map(|o| o.1).fold(0.0f32, f32::max));
        }
        if count < 2 || cover <= 0.30 * ar[i] {
            continue;
        }
        if c[i].1 > best {
            for j in members {
                keep[j] = false; // details of one part
            }
        } else {
            keep[i] = false; // a group: keep its parts
        }
    }
    let mut weakest: Vec<usize> = (0..n).collect();
    weakest.sort_by(|&a, &b| c[a].1.partial_cmp(&c[b].1).unwrap());
    for i in weakest {
        if !keep[i] {
            continue;
        }
        let twin = (0..n).any(|j| j != i && keep[j] && inside(i, j) > 0.75 && ar[j] >= ar[i] * 0.6 && ar[j] <= ar[i] / 0.6 && c[j].1 >= c[i].1);
        if twin {
            keep[i] = false;
        }
    }
    (0..n).filter(|&i| keep[i]).collect()
}

/// OpenCV's 8-bit Lab (L*2.55, a+128, b+128), so the thresholds carry over.
fn lab8(img: &Img, x: usize, y: usize) -> [f32; 3] {
    let p = img.rgb(x, y);
    let l = crate::img::lab([p[0] as f32, p[1] as f32, p[2] as f32]);
    [l[0] * 2.55, l[1] + 128.0, l[2] + 128.0]
}

/// The board's colour: the most common colour among pixels no box covers.
fn board_colour(img: &Img, boxes: &[Bx]) -> [f32; 3] {
    let step = ((img.w * img.h) as f64 / 250000.0).sqrt().max(1.0) as usize;
    let covered = |x: usize, y: usize| boxes.iter().any(|b| x as f32 >= b[0] && (x as f32) < b[2] && y as f32 >= b[1] && (y as f32) < b[3]);
    let mut pts = vec![];
    for y in (0..img.h).step_by(step) {
        for x in (0..img.w).step_by(step) {
            if !covered(x, y) {
                pts.push(lab8(img, x, y));
            }
        }
    }
    if pts.len() < 100 {
        pts.clear();
        for y in (0..img.h).step_by(step) {
            for x in (0..img.w).step_by(step) {
                pts.push(lab8(img, x, y));
            }
        }
    }
    let key = |p: &[f32; 3]| ((p[0] / 12.0) as i64) * 10000 + ((p[1] / 12.0) as i64) * 100 + (p[2] / 12.0) as i64;
    let mut counts: std::collections::HashMap<i64, usize> = Default::default();
    for p in &pts {
        *counts.entry(key(p)).or_default() += 1;
    }
    let top = counts.into_iter().max_by_key(|x| x.1).map(|x| x.0).unwrap_or(0);
    let mut sel: Vec<[f32; 3]> = pts.into_iter().filter(|p| key(p) == top).collect();
    let mut med = [0f32; 3];
    for c in 0..3 {
        sel.sort_by(|a, b| a[c].partial_cmp(&b[c]).unwrap());
        med[c] = sel.get(sel.len() / 2).map(|p| p[c]).unwrap_or(0.0);
    }
    med
}

/// Share of the sampled pixels of a rectangle that differ from the board.
fn differs(img: &Img, board: [f32; 3], x1: i64, y1: i64, x2: i64, y2: i64, delta: f32, budget: f64) -> Option<(usize, usize)> {
    let (x1, y1) = (x1.max(0) as usize, y1.max(0) as usize);
    let (x2, y2) = ((x2.max(0) as usize).min(img.w), (y2.max(0) as usize).min(img.h));
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    let step = (((x2 - x1) * (y2 - y1)) as f64 / budget).sqrt().max(1.0) as usize;
    let (mut n, mut d) = (0usize, 0usize);
    for y in (y1..y2).step_by(step) {
        for x in (x1..x2).step_by(step) {
            let p = lab8(img, x, y);
            if ((p[0] - board[0]).powi(2) + (p[1] - board[1]).powi(2) + (p[2] - board[2]).powi(2)).sqrt() > delta {
                d += 1;
            }
            n += 1;
        }
    }
    Some((d, n))
}

fn foreground_share(img: &Img, b: &Bx, board: [f32; 3]) -> f32 {
    differs(img, board, b[0] as i64, b[1] as i64, b[2] as i64, b[3] as i64, 22.0, 4000.0).map(|(d, n)| d as f32 / n.max(1) as f32).unwrap_or(0.0)
}

/// Share of a ring round the box that is board.
fn board_around(img: &Img, b: &Bx, board: [f32; 3]) -> f32 {
    let (x1, y1, x2, y2) = (b[0] as i64, b[1] as i64, b[2] as i64, b[3] as i64);
    let m = 4.max((0.25 * (x2 - x1).min(y2 - y1) as f32) as i64);
    let (mut d, mut n) = (0, 0);
    for (a, bb, c, e) in [(x1 - m, y1 - m, x2 + m, y1), (x1 - m, y2, x2 + m, y2 + m), (x1 - m, y1, x1, y2), (x2, y1, x2 + m, y2)] {
        if let Some((dd, nn)) = differs(img, board, a, bb, c, e, 40.0, 1000.0) {
            d += dd;
            n += nn;
        }
    }
    if n == 0 { 1.0 } else { 1.0 - d as f32 / n as f32 }
}

/// One appearance vector per box (unit length): ResNet-18 features pooled
/// over the patch, orientation-free. None without the backbone.
fn embed(img: &Img, boxes: &[Bx]) -> Option<Vec<Vec<f32>>> {
    let m = crate::infer::session(&backbone_path()).ok()?;
    let mean = [0.485f32 * 255.0, 0.456 * 255.0, 0.406 * 255.0];
    let std = [0.229f32 * 255.0, 0.224 * 255.0, 0.225 * 255.0];
    let (pw, ph) = (96usize, 64usize);
    let mut out = vec![];
    for chunk in boxes.chunks(32) {
        let mut data = Vec::with_capacity(chunk.len() * 3 * pw * ph);
        for b in chunk {
            let mut p = img.crop(b[0] as i64, b[1] as i64, ((b[2] - b[0]) as usize).max(1), ((b[3] - b[1]) as usize).max(1));
            if p.h as f32 > p.w as f32 * 1.15 {
                p = p.rot90(3);
            }
            let p = three(&p).resize(pw, ph);
            for c in 0..3 {
                for i in 0..pw * ph {
                    data.push((p.data[i * p.c + c] as f32 - mean[c]) / std[c]);
                }
            }
        }
        let outs = crate::infer::run_all(&m, &[chunk.len(), 3, ph, pw], data).ok()?;
        let pooled: Vec<Vec<f32>> = (0..chunk.len()).map(|k| {
            let mut v = vec![];
            for (shape, d) in outs.iter().take(2) {
                let (ch, hw) = (shape[1], shape[2] * shape[3]);
                for c in 0..ch {
                    let base = k * ch * hw + c * hw;
                    v.push(d[base..base + hw].iter().sum::<f32>() / hw as f32);
                }
            }
            v
        }).collect();
        out.extend(pooled);
    }
    for v in out.iter_mut() {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-6;
        v.iter_mut().for_each(|x| *x /= n);
    }
    Some(out)
}

/// A type number per box, 0 for the most common: alike and about the same size.
fn group_types(img: &Img, boxes: &[Bx]) -> Vec<usize> {
    let n = boxes.len();
    if n == 0 {
        return vec![];
    }
    let (vec, similarity) = match embed(img, boxes) {
        Some(v) => (v, 0.86f32),
        None => {
            // no backbone: mean colour only
            let v: Vec<Vec<f32>> = boxes.iter().map(|b| {
                let p = three(&img.crop(b[0] as i64, b[1] as i64, ((b[2] - b[0]) as usize).max(1), ((b[3] - b[1]) as usize).max(1)));
                let mut m = [0f32; 3];
                for i in 0..p.w * p.h {
                    for c in 0..3 {
                        m[c] += p.data[i * p.c + c] as f32;
                    }
                }
                let nn = (m.iter().map(|x| x * x).sum::<f32>()).sqrt() + 1e-6;
                m.iter().map(|x| x / nn).collect()
            }).collect();
            (v, 0.995)
        }
    };
    let wh: Vec<(f32, f32)> = boxes.iter().map(|b| { let (w, h) = (b[2] - b[0], b[3] - b[1]); (w.max(h).max(1e-3), w.min(h).max(1e-3)) }).collect();
    let link = |i: usize, j: usize| {
        let sim: f32 = vec[i].iter().zip(&vec[j]).map(|(a, b)| a * b).sum();
        sim >= similarity && (wh[i].0 / wh[j].0).ln().abs() < 0.30 && (wh[i].1 / wh[j].1).ln().abs() < 0.30
    };
    let degree: Vec<usize> = (0..n).map(|i| (0..n).filter(|&j| link(i, j)).count()).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| degree[b].cmp(&degree[a]));
    let mut label = vec![usize::MAX; n];
    let mut t = 0;
    for i in order {
        if label[i] != usize::MAX {
            continue;
        }
        for j in 0..n {
            if label[j] == usize::MAX && link(i, j) {
                label[j] = t;
            }
        }
        label[i] = t;
        t += 1;
    }
    let mut counts = vec![0usize; t];
    for &l in &label {
        counts[l] += 1;
    }
    let mut rank: Vec<usize> = (0..t).collect();
    rank.sort_by(|&a, &b| counts[b].cmp(&counts[a]));
    let mut remap = vec![0; t];
    for (r, &old) in rank.iter().enumerate() {
        remap[old] = r;
    }
    label.into_iter().map(|l| remap[l]).collect()
}

pub fn type_name(k: usize) -> String {
    let mut s = String::new();
    let mut k = k + 1;
    while k > 0 {
        let r = (k - 1) % 26;
        s.insert(0, (b'A' + r as u8) as char);
        k = (k - 1) / 26;
    }
    s
}

fn clip_rect(img: &Img, rect: Option<[f64; 4]>) -> (usize, usize, usize, usize) {
    match rect {
        None => (0, 0, img.w, img.h),
        Some(r) => {
            let (x, y) = ((r[0].round() as i64).max(0) as usize, (r[1].round() as i64).max(0) as usize);
            let x = x.min(img.w);
            let y = y.min(img.h);
            ((x), (y), (r[2].round().max(0.0) as usize).min(img.w - x), (r[3].round().max(0.0) as usize).min(img.h - y))
        }
    }
}

/// Parts inside rect (None: the whole picture), from candidates already made
/// (`raw`, in the area's own pixels) at a sensitivity 0..1.
pub fn filter(img: &Img, rect: Option<[f64; 4]>, raw: &[(Bx, f32)], sensitivity: f32) -> Value {
    let t0 = std::time::Instant::now();
    let (rx, ry, rw, rh) = clip_rect(img, rect);
    let crop = three(&img.crop(rx as i64, ry as i64, rw, rh));
    let conf = 0.40 - 0.34 * sensitivity.clamp(0.0, 1.0);
    let lim = 0.25 * (rw * rh) as f32;
    let mut c: Vec<(Bx, f32)> = raw.iter().filter(|x| x.1 >= conf && (x.0[2] - x.0[0]).min(x.0[3] - x.0[1]) >= 6.0 && area(&x.0) <= lim).cloned().collect();
    c = nms(&c, 0.55).into_iter().map(|i| c[i]).collect();
    if !c.is_empty() {
        let boxes: Vec<Bx> = c.iter().map(|x| x.0).collect();
        let board = board_colour(&crop, &boxes);
        c.retain(|x| foreground_share(&crop, &x.0, board) >= 0.18 && board_around(&crop, &x.0, board) >= 0.08);
        let mut weak: Vec<(Bx, f32)> = raw.iter().filter(|x| x.1 >= 0.06 && area(&x.0) <= lim).cloned().collect();
        weak = nms(&weak, 0.6).into_iter().map(|i| weak[i]).collect();
        c = drop_groups_and_duplicates(&c, &weak).into_iter().map(|i| c[i]).collect();
    }
    let boxes: Vec<Bx> = c.iter().map(|x| x.0).collect();
    let types = group_types(&crop, &boxes);
    let mut parts: Vec<Value> = c.iter().zip(&types).map(|((b, s), t)| json!({
        "box": [b[0].round() as i64 + rx as i64, b[1].round() as i64 + ry as i64, ((b[2] - b[0]).round() as i64).max(1), ((b[3] - b[1]).round() as i64).max(1)],
        "score": (s * 1000.0).round() / 1000.0, "type": t})).collect();
    parts.sort_by_key(|p| (p["type"].as_u64().unwrap_or(0), p["box"][1].as_i64().unwrap_or(0) / 40, p["box"][0].as_i64().unwrap_or(0)));
    let ntypes = types.iter().max().map(|m| m + 1).unwrap_or(0);
    json!({"parts": parts, "types": ntypes, "filter_ms": t0.elapsed().as_secs_f64() * 1000.0})
}

/// The network's candidates for an area (in the area's own pixels).
pub fn area_candidates(img: &Img, rect: Option<[f64; 4]>, progress: &dyn Fn(usize, usize)) -> Result<Vec<(Bx, f32)>, String> {
    let (rx, ry, rw, rh) = clip_rect(img, rect);
    if rw == 0 || rh == 0 {
        return Ok(vec![]);
    }
    candidates(&three(&img.crop(rx as i64, ry as i64, rw, rh)), 0.05, progress)
}

/// Every copy of the part in box `example` inside rect, turned by 0/90/180/270
/// when `turns`: normalised correlation at a scale where the example's short
/// side is about 32 px; peaks at least half a part apart.
pub fn find_similar(img: &Img, example: [f64; 4], rect: Option<[f64; 4]>, min_score: f32, turns: bool) -> Vec<Value> {
    let (rx, ry, rw, rh) = clip_rect(img, rect);
    let g = img.gray();
    let (ex, ey, ew, eh) = (example[0].round().max(0.0) as usize, example[1].round().max(0.0) as usize, example[2].round() as usize, example[3].round() as usize);
    if ew < 4 || eh < 4 || rw == 0 || rh == 0 {
        return vec![];
    }
    let tpl = g.crop(ex as i64, ey as i64, ew, eh);
    let ar = g.crop(rx as i64, ry as i64, rw, rh);
    let s = (32.0 / tpl.w.min(tpl.h).max(1) as f64).min(1.0);
    let small = if s < 1.0 { ar.resize((ar.w as f64 * s).round() as usize, (ar.h as f64 * s).round() as usize) } else { ar };
    let t0 = if s < 1.0 { tpl.resize(((tpl.w as f64 * s).round() as usize).max(2), ((tpl.h as f64 * s).round() as usize).max(2)) } else { tpl };
    let sp = crate::ncc::plane(&small);
    let mut found: Vec<(Bx, f32)> = vec![];
    for k in if turns { vec![0usize, 3, 2, 1] } else { vec![0] } {
        let t = t0.rot90(k);
        let mean = t.data.iter().map(|&v| v as f64).sum::<f64>() / t.data.len() as f64;
        let sd = (t.data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / t.data.len() as f64).sqrt();
        if t.h > small.h || t.w > small.w || sd < 1.0 {
            continue;
        }
        let res = crate::ncc::match_template(&sp, &crate::ncc::plane(&t));
        // local maxima above the threshold, in a window half a part across
        let win = (t.h.min(t.w) / 2).max(1) | 1;
        let r = (win / 2) as i64;
        for y in 0..res.h {
            for x in 0..res.w {
                let v = res.d[y * res.w + x];
                if v < min_score {
                    continue;
                }
                let mut top = true;
                'w: for dy in -r..=r {
                    for dx in -r..=r {
                        let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                        if nx < 0 || ny < 0 || nx >= res.w as i64 || ny >= res.h as i64 {
                            continue;
                        }
                        if res.d[ny as usize * res.w + nx as usize] > v {
                            top = false;
                            break 'w;
                        }
                    }
                }
                if top {
                    let (fx, fy) = (rx as f64 + x as f64 / s, ry as f64 + y as f64 / s);
                    let (fw, fh) = (t.w as f64 / s, t.h as f64 / s);
                    found.push(([fx as f32, fy as f32, (fx + fw) as f32, (fy + fh) as f32], v));
                }
            }
        }
    }
    let keep = no_overlap(&found, 0.2);
    let mut out: Vec<Value> = keep.into_iter().take(500).map(|i| {
        let (b, sc) = found[i];
        json!({"box": [b[0].round() as i64, b[1].round() as i64, (b[2] - b[0]).round() as i64, (b[3] - b[1]).round() as i64], "score": (sc * 1000.0).round() / 1000.0})
    }).collect();
    out.sort_by(|a, b| b["score"].as_f64().partial_cmp(&a["score"].as_f64()).unwrap());
    out
}

/// Every line of text inside rect, read by the application's own reader.
pub fn find_text(img: &Img, rect: Option<[f64; 4]>) -> Result<Value, String> {
    let t0 = std::time::Instant::now();
    let (rx, ry, rw, rh) = clip_rect(img, rect);
    if rw == 0 || rh == 0 {
        return Ok(json!({"parts": [], "types": 0, "ms": 0.0}));
    }
    let crop = img.crop(rx as i64, ry as i64, rw, rh);
    let lines = crate::ocr::read(&crop, "normal")?;
    let mut parts: Vec<Value> = lines.into_iter().filter(|l| !l.text.trim().is_empty()).map(|l| {
        let b = l.bbox();
        json!({"box": [b[0].round() as i64 + rx as i64, b[1].round() as i64 + ry as i64, b[2].round() as i64, b[3].round() as i64],
               "text": l.text.trim(), "score": ((l.conf as f64) * 1000.0).round() / 1000.0, "type": 0})
    }).collect();
    parts.sort_by_key(|p| { let h = p["box"][3].as_i64().unwrap_or(8).max(8); (p["box"][1].as_i64().unwrap_or(0) / h, p["box"][0].as_i64().unwrap_or(0)) });
    let n = parts.len();
    Ok(json!({"parts": parts, "types": if n > 0 { 1 } else { 0 }, "ms": t0.elapsed().as_secs_f64() * 1000.0,
              "device": crate::infer::init().unwrap_or_default()}))
}

/// An upright copy of area `rect` turned by `angle` degrees about its centre,
/// and the turn to put what is found in it back on the picture: parts and text
/// in a row at an angle get boxes that fit them, not loose upright ones.
pub struct Turned {
    pub img: Img,
    cx: f64,
    cy: f64,
    w: f64,
    h: f64,
    angle: f64,
}

pub fn turned_area(img: &Img, rect: [f64; 4], angle: f64) -> Turned {
    let (w, h) = (rect[2].round().max(1.0), rect[3].round().max(1.0));
    let crop = crate::geom::extract(img, [rect[0], rect[1], w, h], angle, None).patch;
    let mut a = angle % 360.0;
    if a < 0.0 {
        a += 360.0;
    }
    if a > 180.0 {
        a -= 360.0;
    }
    Turned { img: crop, cx: rect[0] + w / 2.0, cy: rect[1] + h / 2.0, w, h, angle: (a * 100.0).round() / 100.0 }
}

impl Turned {
    /// A part found in the upright copy, on the picture: the same size, its
    /// centre turned with the area, `angle` set.
    pub fn back(&self, p: &mut Value) {
        let b: Vec<f64> = (0..4).map(|i| p["box"][i].as_f64().unwrap_or(0.0)).collect();
        let (u, v) = (b[0] + b[2] / 2.0 - self.w / 2.0, b[1] + b[3] / 2.0 - self.h / 2.0);
        let (s, c) = self.angle.to_radians().sin_cos();
        let (px, py) = (self.cx + c * u - s * v, self.cy + s * u + c * v);
        p["box"] = json!([(px - b[2] / 2.0).round() as i64, (py - b[3] / 2.0).round() as i64, b[2] as i64, b[3] as i64]);
        p["angle"] = json!(self.angle);
    }
    pub fn back_all(&self, out: &mut Value) {
        if let Some(a) = out["parts"].as_array_mut() {
            for p in a {
                self.back(p);
            }
        }
    }
}

/// The text inside one box (with a little margin): (text, confidence) -- for a
/// text box the user has just moved, resized or turned (read upright to it).
pub fn read_box(img: &Img, bx: [f64; 4], angle: f64) -> Result<(String, f64), String> {
    let m = (bx[2].max(bx[3]) * 0.15).round() + 2.0;
    let wide = [bx[0] - m, bx[1] - m, bx[2] + 2.0 * m, bx[3] + 2.0 * m];
    let crop = if angle.abs() > 1e-6 {
        turned_area(img, wide, angle).img
    } else {
        let (x0, y0) = (wide[0].round().max(0.0), wide[1].round().max(0.0));
        let x1 = (wide[0] + wide[2]).round().min(img.w as f64);
        let y1 = (wide[1] + wide[3]).round().min(img.h as f64);
        if x1 <= x0 || y1 <= y0 {
            return Ok((String::new(), 0.0));
        }
        img.crop(x0 as i64, y0 as i64, (x1 - x0) as usize, (y1 - y0) as usize)
    };
    if crop.w == 0 || crop.h == 0 {
        return Ok((String::new(), 0.0));
    }
    let mut lines: Vec<_> = crate::ocr::read(&crop, "normal")?.into_iter().filter(|l| !l.text.trim().is_empty()).collect();
    if lines.is_empty() {
        return Ok((String::new(), 0.0));
    }
    lines.sort_by(|a, b| { let (p, q) = (a.bbox(), b.bbox()); (p[1], p[0]).partial_cmp(&(q[1], q[0])).unwrap() });
    let text = lines.iter().map(|l| l.text.trim().to_string()).collect::<Vec<_>>().join(" ");
    let conf = lines.iter().map(|l| l.conf as f64).fold(1.0, f64::min);
    Ok((text, (conf * 1000.0).round() / 1000.0))
}

fn safe_id(s: &str, fallback: &str) -> String {
    let v: String = s.trim().chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    if v.is_empty() { fallback.to_string() } else { v }
}

/// A check's region round a box, with margins: clipped to the picture when
/// upright (a turned one keeps its shape -- clipping it would move its centre
/// and so the turn).
fn place(x: f64, y: f64, w: f64, h: f64, mx: f64, my: f64, angle: f64, size: (usize, usize)) -> [i64; 4] {
    let (mut x0, mut y0, mut x1, mut y1) = (x - mx, y - my, x + w + mx, y + h + my);
    if angle.abs() < 1e-9 {
        x0 = x0.max(0.0);
        y0 = y0.max(0.0);
        x1 = x1.min(size.0 as f64);
        y1 = y1.min(size.1 as f64);
    }
    [x0.round() as i64, y0.round() as i64, (x1 - x0).round() as i64, (y1 - y0).round() as i64]
}

/// "Part present" check specs for the kept parts: named per type ("inductor_01"),
/// a small margin of board round each part, the type's accept threshold.
pub fn presence_tools(parts: &[Value], names: &Value, thresholds: &Value, existing: &[String], size: (usize, usize), seated: bool) -> Vec<Value> {
    let mut used: std::collections::HashSet<String> = existing.iter().cloned().collect();
    let mut count: std::collections::HashMap<String, usize> = Default::default();
    let mut out = vec![];
    for p in parts {
        let t = p["type"].as_u64().unwrap_or(0) as usize;
        let given = names[t.to_string()].as_str().unwrap_or("");
        let base = safe_id(if given.is_empty() { "" } else { given }, &format!("part_{}", type_name(t).to_lowercase()));
        let n = count.entry(base.clone()).or_insert(0);
        let id = loop {
            *n += 1;
            let id = format!("{}_{:02}", base, n);
            if !used.contains(&id) {
                break id;
            }
        };
        used.insert(id.clone());
        let b: Vec<f64> = (0..4).map(|i| p["box"][i].as_f64().unwrap_or(0.0)).collect();
        let (mx, my) = ((b[2] * 0.08).round() + 2.0, (b[3] * 0.08).round() + 2.0);
        let thr = thresholds[t.to_string()].as_f64().unwrap_or(0.65);
        let ang = p["angle"].as_f64().unwrap_or(0.0);
        let mut params = json!({"roi": place(b[0], b[1], b[2], b[3], mx, my, ang, size), "min_score": (thr * 1000.0).round() / 1000.0,
            "search_pad": (0.15 * b[2].max(b[3])).clamp(8.0, 60.0) as i64, "group": base});
        if ang != 0.0 {
            params["angle"] = json!(ang);
        }
        if seated {
            params["check_seated"] = json!(true);
        }
        out.push(json!({"id": id, "type": "presence", "params": params}));
    }
    out
}

/// Text check specs for the kept text boxes: "ocv" expects exactly the text
/// shown (as corrected), "ocr" reads it and expects it contained.
pub fn text_tools(items: &[Value], kind: &str, existing: &[String], size: (usize, usize), min_conf: Option<f64>, defaults: &Value) -> Vec<Value> {
    let kind = if kind == "ocv" { "ocv" } else { "ocr" };
    let mut used: std::collections::HashSet<String> = existing.iter().cloned().collect();
    let mut count: std::collections::HashMap<String, usize> = Default::default();
    let mut out = vec![];
    for it in items {
        let name = safe_id(it["name"].as_str().unwrap_or(""), "text");
        let n = count.entry(name.clone()).or_insert(0);
        let id = loop {
            *n += 1;
            let id = format!("{}_{:02}", name, n);
            if !used.contains(&id) {
                break id;
            }
        };
        used.insert(id.clone());
        let b: Vec<f64> = (0..4).map(|i| it["box"][i].as_f64().unwrap_or(0.0)).collect();
        // a thin margin: enough background for the reader, not the part the text is on
        let (mx, my) = ((0.15 * b[3]).round() + 2.0, (0.22 * b[3]).round() + 2.0);
        let mut params = defaults.clone();
        if !params.is_object() {
            params = json!({});
        }
        let ang = it["angle"].as_f64().unwrap_or(0.0);
        params["roi"] = json!(place(b[0], b[1], b[2], b[3], mx, my, ang, size));
        params["angle"] = json!(ang);
        params["group"] = json!(name);
        if let Some(c) = min_conf {
            params["min_confidence"] = json!((c * 1000.0).round() / 1000.0);
        }
        let text = it["text"].as_str().unwrap_or("").to_string();
        if kind == "ocv" {
            params["expect_text"] = json!(text);
            params["expect_auto"] = json!(text);
        } else if !text.is_empty() {
            params["expect_contains"] = json!(text);
        }
        out.push(json!({"id": id, "type": kind, "params": params}));
    }
    out
}
