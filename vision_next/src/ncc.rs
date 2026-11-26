//! Normalised cross-correlation (OpenCV's TM_CCOEFF_NORMED): the template's
//! mean-removed dot product with each window, over both standard deviations.
//! Window statistics come from integral images; the dot products run row by
//! row on contiguous slices, in parallel over output rows.

use crate::img::{integral, Img, Plane};
use rayon::prelude::*;

pub struct ScoreMap {
    pub w: usize,
    pub h: usize,
    pub d: Vec<f32>,
}

impl ScoreMap {
    pub fn best(&self) -> (usize, usize, f32) {
        let mut bi = 0;
        let mut bv = f32::MIN;
        for (i, &v) in self.d.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = i;
            }
        }
        (bi % self.w.max(1), bi / self.w.max(1), bv)
    }
    /// Sub-pixel peak by a parabola through the neighbours.
    pub fn refine(&self, x: usize, y: usize) -> (f64, f64) {
        let at = |xx: usize, yy: usize| self.d[yy * self.w + xx] as f64;
        let mut fx = x as f64;
        let mut fy = y as f64;
        if x > 0 && x + 1 < self.w {
            let (l, c, r) = (at(x - 1, y), at(x, y), at(x + 1, y));
            let den = l - 2.0 * c + r;
            if den.abs() > 1e-9 {
                fx += (0.5 * (l - r) / den).clamp(-0.5, 0.5);
            }
        }
        if y > 0 && y + 1 < self.h {
            let (u, c, d) = (at(x, y - 1), at(x, y), at(x, y + 1));
            let den = u - 2.0 * c + d;
            if den.abs() > 1e-9 {
                fy += (0.5 * (u - d) / den).clamp(-0.5, 0.5);
            }
        }
        (fx, fy)
    }
}

/// Score of every placement of `tpl` (top-left corner) inside `img`.
pub fn match_template(img: &Plane, tpl: &Plane) -> ScoreMap {
    let (iw, ih, tw, th) = (img.w, img.h, tpl.w, tpl.h);
    if tw > iw || th > ih || tw == 0 || th == 0 {
        return ScoreMap { w: 0, h: 0, d: vec![] };
    }
    let ow = iw - tw + 1;
    let oh = ih - th + 1;
    let n = (tw * th) as f64;
    let tmean = tpl.d.iter().map(|&v| v as f64).sum::<f64>() / n;
    let t0: Vec<f32> = tpl.d.iter().map(|&v| v - tmean as f32).collect();
    let tss: f64 = t0.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let ii = integral(&img.d, iw, ih);
    let sq: Vec<f32> = img.d.iter().map(|&v| v * v).collect();
    let ii2 = integral(&sq, iw, ih);
    let mut out = vec![0f32; ow * oh];
    if tss < 1e-6 {
        return ScoreMap { w: ow, h: oh, d: out };
    }
    out.par_chunks_mut(ow).enumerate().for_each(|(y, row)| {
        // acc[x] += t[k] * row[x + k]: contiguous in x, so the compiler
        // vectorises it (a per-window dot product would not: float sums
        // are not reordered)
        let mut acc = vec![0f64; ow];
        for ty in 0..th {
            let irow = &img.d[(y + ty) * iw..(y + ty) * iw + iw];
            let trow = &t0[ty * tw..ty * tw + tw];
            for (k, &tv) in trow.iter().enumerate() {
                let src = &irow[k..k + ow];
                for (a, &s) in acc.iter_mut().zip(src) {
                    *a += tv as f64 * s as f64;
                }
            }
        }
        let s1 = iw + 1;
        for x in 0..ow {
            let (x1, y1) = (x + tw, y + th);
            let sum = ii[y1 * s1 + x1] - ii[y * s1 + x1] - ii[y1 * s1 + x] + ii[y * s1 + x];
            let sum2 = ii2[y1 * s1 + x1] - ii2[y * s1 + x1] - ii2[y1 * s1 + x] + ii2[y * s1 + x];
            let var = sum2 - sum * sum / n;
            row[x] = if var <= 1e-3 { 0.0 } else { (acc[x] / (tss * var).sqrt()) as f32 };
        }
    });
    ScoreMap { w: ow, h: oh, d: out }
}

/// Shrink a plane by an integer factor (box average).
pub fn shrink(p: &Plane, f: usize) -> Plane {
    if f <= 1 {
        return p.clone();
    }
    let w = (p.w / f).max(1);
    let h = (p.h / f).max(1);
    let mut out = Plane::new(w, h);
    let inv = 1.0 / (f * f) as f32;
    out.d.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let mut s = 0f32;
            for yy in y * f..((y + 1) * f).min(p.h) {
                for xx in x * f..((x + 1) * f).min(p.w) {
                    s += p.d[yy * p.w + xx];
                }
            }
            *o = s * inv;
        }
    });
    out
}

pub fn plane(img: &Img) -> Plane {
    Plane::from_gray(img)
}

/// Peaks at or above `thr` that are local maxima, best first.
pub fn peaks(map: &ScoreMap, thr: f32) -> Vec<(usize, usize, f32)> {
    let mut out = Vec::new();
    for y in 0..map.h {
        for x in 0..map.w {
            let v = map.d[y * map.w + x];
            if v < thr {
                continue;
            }
            let mut is_max = true;
            'n: for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= map.w as i64 || ny >= map.h as i64 {
                        continue;
                    }
                    if map.d[ny as usize * map.w + nx as usize] > v {
                        is_max = false;
                        break 'n;
                    }
                }
            }
            if is_max {
                out.push((x, y, v));
            }
        }
    }
    out.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    out
}
