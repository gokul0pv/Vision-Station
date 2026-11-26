//! Vision Station's Locate part (tools/align.py), step for step, so Vision
//! Next puts the board where Vision Station puts it.
//!
//! method "auto": SIFT features (OpenCV 4.10's SIFT, the brute-force matcher
//! with Lowe's ratio test, estimateAffinePartial2D: RANSAC then
//! Levenberg-Marquardt), an ECC polish kept only when it fits the region
//! better, scored by the region's correlation; the rotated template search
//! (half-size coarse angles, then full size, ECC polish judged by a whole
//! picture fit) when features do not find the part.
//!
//! What has to match exactly is what makes choices: which keypoints exist
//! and in what order (they are sorted, then cut to the best 2000 by MSVC's
//! nth_element), the RANSAC random samples (cv::RNG), the inlier tests. The
//! arithmetic follows OpenCV's order; float sums that OpenCV vectorises may
//! differ in the last bit, which moves nothing (tested: Vision Station's own
//! result does not change with Intel IPP on or off, which changes the
//! pyramid by more than that).

use crate::img::Img;

/// A fused multiply-add for f32, done exactly in f64 (the product of two
/// floats is exact there). f32::mul_add calls the C library's fmaf, which on
/// this GNU toolchain rounds the product first -- not what an FMA does, and
/// not what OpenCV's AVX2 code does.
trait Fma {
    fn fma(self, b: f32, c: f32) -> f32;
}
impl Fma for f32 {
    #[inline(always)]
    fn fma(self, b: f32, c: f32) -> f32 {
        (self as f64 * b as f64 + c as f64) as f32
    }
}

// ------------------------------------------------------------------ images
#[derive(Clone)]
pub struct Fm {
    pub w: usize,
    pub h: usize,
    pub d: Vec<f32>,
}

impl Fm {
    pub fn new(w: usize, h: usize) -> Fm {
        Fm { w, h, d: vec![0.0; w * h] }
    }
    pub fn from_u8(g: &Img) -> Fm {
        Fm { w: g.w, h: g.h, d: g.data.iter().map(|&v| v as f32).collect() }
    }
    #[inline]
    fn at(&self, x: usize, y: usize) -> f32 {
        self.d[y * self.w + x]
    }
}

fn crop_u8(g: &Img, x0: usize, y0: usize, x1: usize, y1: usize) -> Img {
    let (w, h) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    let mut out = Img::new(w, h, 1);
    for y in 0..h {
        let s = (y0 + y) * g.w + x0;
        out.data[y * w..(y + 1) * w].copy_from_slice(&g.data[s..s + w]);
    }
    out
}

/// cvRound / saturate_cast<int>(double): to nearest, ties to even.
#[inline]
fn cv_round(v: f64) -> i32 {
    let r = v.round_ties_even();
    r.clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

#[inline]
fn cv_round_f(v: f32) -> i32 {
    cv_round(v as f64)
}

// ------------------------------------------------------------ INTER_AREA
/// cv::resize(src, dsize, INTER_AREA) for an 8-bit grey picture. `inv` is
/// inv_scale (fx) as OpenCV holds it: given, or dsize / ssize.
fn resize_area_to(src: &Img, dw: usize, dh: usize, inv_x: f64, inv_y: f64) -> Img {
    if dw == src.w && dh == src.h {
        return src.clone();
    }
    let scale_x = 1.0 / inv_x;
    let scale_y = 1.0 / inv_y;
    let (ix, iy) = (cv_round(scale_x), cv_round(scale_y));
    let fast = (scale_x - ix as f64).abs() < f64::EPSILON && (scale_y - iy as f64).abs() < f64::EPSILON;
    if scale_x >= 1.0 && scale_y >= 1.0 && fast {
        return area_fast(src, dw, dh, ix as usize, iy as usize);
    }
    if scale_x < 1.0 || scale_y < 1.0 {
        // never needed here: Vision Station only shrinks
        return crate::cvx::resize_linear(src, dw, dh);
    }
    let xt = area_tab(src.w, dw, scale_x);
    let yt = area_tab(src.h, dh, scale_y);
    let mut out = Img::new(dw, dh, 1);
    let mut buf = vec![0f32; dw];
    let mut sum = vec![0f32; dw];
    let mut prev_dy = yt[0].0;
    let store = |sum: &[f32], out: &mut Img, dy: usize| {
        for (x, &v) in sum.iter().enumerate() {
            out.data[dy * dw + x] = cv_round_f(v).clamp(0, 255) as u8;
        }
    };
    for &(dy, sy, beta) in &yt {
        buf.iter_mut().for_each(|v| *v = 0.0);
        let row = &src.data[sy * src.w..(sy + 1) * src.w];
        for &(dx, sx, alpha) in &xt {
            buf[dx] += row[sx] as f32 * alpha;
        }
        if dy != prev_dy {
            store(&sum, &mut out, prev_dy);
            for x in 0..dw {
                sum[x] = beta * buf[x];
            }
            prev_dy = dy;
        } else {
            for x in 0..dw {
                sum[x] += beta * buf[x];
            }
        }
    }
    store(&sum, &mut out, prev_dy);
    out
}

/// computeResizeAreaTab: (destination, source, weight) runs.
fn area_tab(ssize: usize, dsize: usize, scale: f64) -> Vec<(usize, usize, f32)> {
    let mut tab = vec![];
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell = scale.min(ssize as f64 - fsx1);
        let mut sx1 = fsx1.ceil() as i64;
        let mut sx2 = fsx2.floor() as i64;
        sx2 = sx2.min(ssize as i64 - 1);
        sx1 = sx1.min(sx2);
        if sx1 as f64 - fsx1 > 1e-3 {
            tab.push((dx, (sx1 - 1) as usize, ((sx1 as f64 - fsx1) / cell) as f32));
        }
        for sx in sx1..sx2 {
            tab.push((dx, sx as usize, (1.0 / cell) as f32));
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            tab.push((dx, sx2 as usize, ((fsx2 - sx2 as f64).min(1.0).min(cell) / cell) as f32));
        }
    }
    tab
}

/// resizeAreaFast_Invoker for 8-bit grey: whole cells (2 x 2 rounded by
/// (sum + 2) >> 2, larger by sum * (1 / area) to even), cells cut by the
/// edge averaged over the pixels that exist.
fn area_fast(src: &Img, dw: usize, dh: usize, sx: usize, sy: usize) -> Img {
    let mut out = Img::new(dw, dh, 1);
    let area = sx * sy;
    let scale = 1.0f32 / area as f32;
    let dwidth1 = src.w / sx;
    for dy in 0..dh {
        let sy0 = dy * sy;
        let w = if sy0 + sy <= src.h { dwidth1 } else { 0 };
        if sy0 >= src.h {
            continue;
        }
        for dx in 0..dw {
            let sx0 = dx * sx;
            let v = if dx < w {
                let mut s = 0u32;
                for yy in 0..sy {
                    for xx in 0..sx {
                        s += src.data[(sy0 + yy) * src.w + sx0 + xx] as u32;
                    }
                }
                if sx == 2 && sy == 2 { ((s + 2) >> 2) as i32 } else { cv_round_f(s as f32 * scale) }
            } else {
                let (mut s, mut n) = (0u32, 0u32);
                for yy in 0..sy {
                    if sy0 + yy >= src.h {
                        break;
                    }
                    for xx in 0..sx {
                        if sx0 + xx >= src.w {
                            break;
                        }
                        s += src.data[(sy0 + yy) * src.w + sx0 + xx] as u32;
                        n += 1;
                    }
                }
                if n == 0 { 0 } else { cv_round_f(s as f32 / n as f32) }
            };
            out.data[dy * dw + dx] = v.clamp(0, 255) as u8;
        }
    }
    out
}

/// cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
pub fn resize_area_scale(src: &Img, s: f64) -> Img {
    if let Some(r) = crate::cvlib::resize(src, 0, 0, s, s, crate::cvlib::INTER_AREA) {
        return r;
    }
    let dw = cv_round(src.w as f64 * s).max(1) as usize;
    let dh = cv_round(src.h as f64 * s).max(1) as usize;
    resize_area_to(src, dw, dh, s, s)
}

/// cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
pub fn resize_area_size(src: &Img, dw: usize, dh: usize) -> Img {
    if let Some(r) = crate::cvlib::resize(src, dw, dh, 0.0, 0.0, crate::cvlib::INTER_AREA) {
        return r;
    }
    resize_area_to(src, dw, dh, dw as f64 / src.w as f64, dh as f64 / src.h as f64)
}

// ------------------------------------------------------ float GaussianBlur
fn reflect101(p: i64, len: usize) -> usize {
    crate::cvx::reflect101(p, len as i64)
}

/// getGaussianKernel(n, sigma, CV_32F): the bit-exact double kernel, as float.
pub fn gauss_kernel_f32(n: usize, sigma: f64) -> Vec<f32> {
    if sigma <= 0.0 {
        match n {
            1 => return vec![1.0],
            3 => return vec![0.25, 0.5, 0.25],
            5 => return vec![0.0625, 0.25, 0.375, 0.25, 0.0625],
            7 => return vec![0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
            _ => {}
        }
    }
    let sig = if sigma > 0.0 { sigma } else { n as f64 * 0.15 + 0.35 };
    let scale2x = -0.125 / (sig * sig);
    let n2 = (n - 1) / 2;
    let mut values = vec![0f64; n2];
    let mut sum = 0f64;
    let mut x = 1 - n as i64;
    for v in values.iter_mut() {
        let t = ((x * x) as f64 * scale2x).exp();
        *v = t;
        sum += t;
        x += 2;
    }
    sum *= 2.0;
    sum += 1.0;
    let mul1 = 1.0 / sum;
    let mut res = vec![0f32; n];
    for i in 0..n2 {
        let t = (values[i] * mul1) as f32;
        res[i] = t;
        res[n - 1 - i] = t;
    }
    res[n2] = mul1 as f32;
    res
}

/// cv::GaussianBlur on a float picture (BORDER_REFLECT_101), sepFilter2D's
/// order: the row filter (taps in order, fused), then the symmetric column
/// filter (centre, then each pair of rows added and fused).
pub fn gauss_f32(src: &Fm, ksize: usize, sigma: f64) -> Fm {
    let n = if ksize == 0 { (cv_round(sigma * 4.0 * 2.0 + 1.0) | 1) as usize } else { ksize };
    let k = gauss_kernel_f32(n, sigma);
    let r = (n / 2) as i64;
    let (w, h) = (src.w, src.h);
    let kx: Vec<f32> = if w == 1 { vec![1.0] } else { k.clone() };
    let ky: Vec<f32> = if h == 1 { vec![1.0] } else { k };
    let rx = (kx.len() / 2) as i64;
    let ry = (ky.len() / 2) as i64;
    let _ = r;
    use rayon::prelude::*;
    let mut tmp = vec![0f32; w * h];
    tmp.par_chunks_mut(w.max(1)).enumerate().for_each(|(y, out)| {
        let row = &src.d[y * w..(y + 1) * w];
        // 8-lane vectors (fused) up to the last whole vector, then plain C
        let vec_end = w / 8 * 8;
        for x in 0..w {
            let mut s = row[reflect101(x as i64 - rx, w)] * kx[0];
            for (j, &kv) in kx.iter().enumerate().skip(1) {
                let v = row[reflect101(x as i64 + j as i64 - rx, w)];
                s = if x < vec_end { v.fma(kv, s) } else { s + kv * v };
            }
            out[x] = s;
        }
    });
    let mut out = Fm::new(w, h);
    out.d.par_chunks_mut(w.max(1)).enumerate().for_each(|(y, o)| {
        let c = ry as usize;
        let rows: Vec<usize> = (0..ky.len()).map(|j| reflect101(y as i64 + j as i64 - ry, h)).collect();
        let vec_end = w / 8 * 8;
        for x in 0..w {
            let mut s = ky[c] * tmp[rows[c] * w + x];
            for k in 1..=c {
                let v = tmp[rows[c - k] * w + x] + tmp[rows[c + k] * w + x];
                s = if x < vec_end { v.fma(ky[c + k], s) } else { s + ky[c + k] * v };
            }
            o[x] = s;
        }
    });
    out
}

// ------------------------------------------------------------ fast maths
const ATAN_P1: f32 = 0.9997878412794807f32 * (180.0 / std::f64::consts::PI) as f32;
const ATAN_P3: f32 = -0.3258083974640975f32 * (180.0 / std::f64::consts::PI) as f32;
const ATAN_P5: f32 = 0.1555786518463281f32 * (180.0 / std::f64::consts::PI) as f32;
const ATAN_P7: f32 = -0.04432655554792128f32 * (180.0 / std::f64::consts::PI) as f32;

/// cv::hal::fastAtan2 in degrees, the vectorised form (min/max, fused).
#[inline]
fn fast_atan2(y: f32, x: f32) -> f32 {
    let (ax, ay) = (x.abs(), y.abs());
    let c = ax.min(ay) / (ax.max(ay) + f64::EPSILON as f32);
    let cc = c * c;
    let mut a = cc.fma(ATAN_P7, ATAN_P5).fma(cc, ATAN_P3).fma(cc, ATAN_P1) * c;
    if !(ax >= ay) {
        a = 90.0 - a;
    }
    if x < 0.0 {
        a = 180.0 - a;
    }
    if y < 0.0 {
        a = 360.0 - a;
    }
    a
}

// ------------------------------------------------------------------ SIFT
#[derive(Clone, Copy, Debug)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
}

const N_LAYERS: usize = 3;
const CONTRAST: f64 = 0.04;
const EDGE: f64 = 10.0;
const SIGMA: f64 = 1.6;
const IMG_BORDER: usize = 5;
const ORI_BINS: usize = 36;

/// cv::SIFT_create(nfeatures).detectAndCompute(gray, None)
pub fn sift(gray: &Img, nfeatures: usize) -> (Vec<KeyPoint>, Vec<[f32; 128]>) {
    if let Some((k, oct, d)) = crate::cvlib::sift(gray, nfeatures) {
        let kps = k.iter().zip(&oct).map(|(k, &o)| KeyPoint { x: k[0], y: k[1], size: k[2], angle: k[3], response: k[4], octave: o }).collect();
        return (kps, d);
    }
    let first_octave: i32 = -1;
    // createInitialImage: doubled by resize(INTER_LINEAR) -- cv2.SIFT_create
    // leaves enable_precise_upscale off. Sample k of 2: source k/2 - 1/4,
    // so weights 1/4 and 3/4 (exact in float), the edges repeated.
    let (w, h) = (gray.w, gray.h);
    let tap = |d: usize, n: usize| -> (usize, usize, f32, f32) {
        let f = (d as f64 + 0.5) * 0.5 - 0.5;
        let s = f.floor() as i64;
        let fr = (f - s as f64) as f32;
        if s < 0 {
            (0, 0, 1.0, 0.0)
        } else if s as usize >= n - 1 {
            (n - 1, n - 1, 1.0, 0.0)
        } else {
            (s as usize, s as usize + 1, 1.0 - fr, fr)
        }
    };
    let xt: Vec<_> = (0..w * 2).map(|x| tap(x, w)).collect();
    let hrow = |y: usize| -> Vec<f32> {
        let row = &gray.data[y * w..(y + 1) * w];
        xt.iter().map(|&(a, b, wa, wb)| row[a] as f32 * wa + row[b] as f32 * wb).collect()
    };
    let rows: Vec<Vec<f32>> = (0..h).map(hrow).collect();
    let mut dbl = Fm::new(w * 2, h * 2);
    for y in 0..h * 2 {
        let f = (y as f64 + 0.5) * 0.5 - 0.5;
        let s = f.floor() as i64;
        let fr = (f - s as f64) as f32;
        let (r0, r1) = (s.clamp(0, h as i64 - 1) as usize, (s + 1).clamp(0, h as i64 - 1) as usize);
        for x in 0..w * 2 {
            dbl.d[y * w * 2 + x] = rows[r0][x] * (1.0 - fr) + rows[r1][x] * fr;
        }
    }
    let sigf = SIGMA as f32;
    let sig_diff = (sigf * sigf - 0.5f32 * 0.5 * 4.0).max(0.01f32).sqrt();
    let base = gauss_f32(&dbl, 0, sig_diff as f64);
    let n_oct = (cv_round((base.w.min(base.h) as f64).ln() / 2f64.ln() - 2.0) - first_octave) as usize;
    // Gaussian pyramid
    let mut sig = vec![0f64; N_LAYERS + 3];
    sig[0] = SIGMA;
    let k = 2f64.powf(1.0 / N_LAYERS as f64);
    for i in 1..N_LAYERS + 3 {
        let prev = k.powf((i - 1) as f64) * SIGMA;
        let total = prev * k;
        sig[i] = (total * total - prev * prev).sqrt();
    }
    let per = N_LAYERS + 3;
    let mut gp: Vec<Fm> = Vec::with_capacity(n_oct * per);
    for o in 0..n_oct {
        for i in 0..per {
            let m = if o == 0 && i == 0 {
                base.clone()
            } else if i == 0 {
                let s = &gp[(o - 1) * per + N_LAYERS];
                let (nw, nh) = (s.w / 2, s.h / 2);
                let mut d = Fm::new(nw, nh);
                for y in 0..nh {
                    for x in 0..nw {
                        d.d[y * nw + x] = s.at((2 * x).min(s.w - 1), (2 * y).min(s.h - 1));
                    }
                }
                d
            } else {
                gauss_f32(&gp[o * per + i - 1], 0, sig[i])
            };
            gp.push(m);
        }
    }
    // DoG
    let dper = N_LAYERS + 2;
    let mut dog: Vec<Fm> = Vec::with_capacity(n_oct * dper);
    for o in 0..n_oct {
        for i in 0..dper {
            let a = &gp[o * per + i];
            let b = &gp[o * per + i + 1];
            dog.push(Fm { w: a.w, h: a.h, d: b.d.iter().zip(&a.d).map(|(p, q)| p - q).collect() });
        }
    }
    // extrema
    let mut kps: Vec<KeyPoint> = vec![];
    for o in 0..n_oct {
        for i in 1..=N_LAYERS {
            let idx = o * dper + i;
            let img = &dog[idx];
            let (rows, cols) = (img.h, img.w);
            if rows <= 2 * IMG_BORDER || cols <= 2 * IMG_BORDER {
                continue;
            }
            use rayon::prelude::*;
            let found: Vec<Vec<KeyPoint>> = (IMG_BORDER..rows - IMG_BORDER).into_par_iter().map(|r| {
                let mut out = vec![];
                extrema_row(&dog, &gp, o, i, idx, r, &mut out);
                out
            }).collect();
            for f in found {
                kps.extend(f);
            }
        }
    }
    // removeDuplicatedSorted
    kps.sort_by(|a, b| kp12_cmp(a, b));
    let mut j = 0usize;
    if !kps.is_empty() {
        for i in 1..kps.len() {
            let (p, q) = (kps[j], kps[i]);
            if p.x != q.x || p.y != q.y || p.size != q.size || p.angle != q.angle {
                j += 1;
                kps[j] = q;
            }
        }
        kps.truncate(j + 1);
    }
    // retainBest
    if nfeatures > 0 && kps.len() > nfeatures {
        let greater = |a: &KeyPoint, b: &KeyPoint| a.response > b.response;
        msvc_nth_element(&mut kps, nfeatures - 1, &greater);
        let amb = kps[nfeatures - 1].response;
        let end = msvc_partition(&mut kps[nfeatures..], |k| k.response >= amb) + nfeatures;
        kps.truncate(end);
    }
    // back to the picture's scale
    let scale = 1.0f32 / (1 << (-first_octave)) as f32;
    for kp in kps.iter_mut() {
        kp.octave = (kp.octave & !255) | ((kp.octave + first_octave) & 255);
        kp.x *= scale;
        kp.y *= scale;
        kp.size *= scale;
    }
    // descriptors
    use rayon::prelude::*;
    let desc: Vec<[f32; 128]> = kps.par_iter().map(|kp| {
        let mut octave = kp.octave & 255;
        let layer = ((kp.octave >> 8) & 255) as usize;
        if octave >= 128 {
            octave |= -128;
        }
        let sc = if octave >= 0 { 1.0f32 / (1 << octave) as f32 } else { (1 << (-octave)) as f32 };
        let size = kp.size * sc;
        let (px, py) = (kp.x * sc, kp.y * sc);
        let img = &gp[((octave - first_octave) as usize) * per + layer];
        let mut angle = 360.0f32 - kp.angle;
        if (angle - 360.0).abs() < f32::EPSILON {
            angle = 0.0;
        }
        descriptor(img, px, py, angle, size * 0.5)
    }).collect();
    (kps, desc)
}

fn kp12_cmp(a: &KeyPoint, b: &KeyPoint) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    let lt = |a: &KeyPoint, b: &KeyPoint| -> bool {
        if a.x != b.x {
            return a.x < b.x;
        }
        if a.y != b.y {
            return a.y < b.y;
        }
        if a.size != b.size {
            return a.size > b.size;
        }
        if a.angle != b.angle {
            return a.angle < b.angle;
        }
        if a.response != b.response {
            return a.response > b.response;
        }
        if a.octave != b.octave {
            return a.octave > b.octave;
        }
        false
    };
    if lt(a, b) {
        Less
    } else if lt(b, a) {
        Greater
    } else {
        Equal
    }
}

fn extrema_row(dog: &[Fm], gp: &[Fm], o: usize, layer0: usize, idx: usize, r: usize, out: &mut Vec<KeyPoint>) {
    let img = &dog[idx];
    let prev = &dog[idx - 1];
    let next = &dog[idx + 1];
    let cols = img.w;
    let step = cols;
    let threshold = (0.5 * CONTRAST / N_LAYERS as f64 * 255.0).floor() as f32;
    let cur = &img.d;
    let base = r * step;
    for c in IMG_BORDER..cols - IMG_BORDER {
        let p = base + c;
        let val = cur[p];
        if val.abs() <= threshold {
            continue;
        }
        let nb = |a: &Vec<f32>| [a[p - step - 1], a[p - step], a[p - step + 1], a[p - 1], a[p + 1], a[p + step - 1], a[p + step], a[p + step + 1]];
        let mx = |v: [f32; 8]| v[0].max(v[1]).max(v[2].max(v[3])).max(v[4].max(v[5]).max(v[6].max(v[7])));
        let mn = |v: [f32; 8]| v[0].min(v[1]).min(v[2].min(v[3])).min(v[4].min(v[5]).min(v[6].min(v[7])));
        let calculate = if val > 0.0 {
            val >= mx(nb(cur)) && val >= mx(nb(&prev.d)) && val >= mx(nb(&next.d)) && val >= prev.d[p].max(next.d[p])
        } else {
            val <= mn(nb(cur)) && val <= mn(nb(&prev.d)) && val <= mn(nb(&next.d)) && val <= prev.d[p].min(next.d[p])
        };
        if !calculate {
            continue;
        }
        let (mut r1, mut c1, mut layer) = (r as i32, c as i32, layer0 as i32);
        let Some(mut kpt) = adjust_local_extrema(dog, o, &mut layer, &mut r1, &mut c1) else { continue };
        let scl_octv = kpt.size * 0.5 / (1 << o) as f32;
        let mut hist = [0f32; ORI_BINS];
        let omax = orientation_hist(&gp[o * (N_LAYERS + 3) + layer as usize], c1, r1,
            cv_round_f(4.5 * scl_octv), 1.5 * scl_octv, &mut hist);
        let mag_thr = omax * 0.8;
        let n = ORI_BINS;
        for j in 0..n {
            let l = if j > 0 { j - 1 } else { n - 1 };
            let r2 = if j < n - 1 { j + 1 } else { 0 };
            if hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr {
                let mut bin = j as f32 + 0.5 * (hist[l] - hist[r2]) / (hist[l] - 2.0 * hist[j] + hist[r2]);
                bin = if bin < 0.0 { n as f32 + bin } else if bin >= n as f32 { bin - n as f32 } else { bin };
                kpt.angle = 360.0 - (360.0f32 / n as f32) * bin;
                if (kpt.angle - 360.0).abs() < f32::EPSILON {
                    kpt.angle = 0.0;
                }
                out.push(kpt);
            }
        }
    }
}

/// Matx33f::solve(b, DECOMP_LU): Cramer's rule with the float determinant.
fn solve3(a: [[f32; 3]; 3], b: [f32; 3]) -> [f32; 3] {
    let d = a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2]) - a[0][1] * (a[1][0] * a[2][2] - a[2][0] * a[1][2])
        + a[0][2] * (a[1][0] * a[2][1] - a[2][0] * a[1][1]);
    if d == 0.0 {
        return [0.0; 3];
    }
    let d = 1.0 / d;
    [
        d * (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2]) + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2])),
        d * (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2]) - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0])),
        d * (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1]) - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0]) + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])),
    ]
}

fn adjust_local_extrema(dog: &[Fm], octv: usize, layer: &mut i32, r: &mut i32, c: &mut i32) -> Option<KeyPoint> {
    let img_scale = 1.0f32 / 255.0;
    let deriv_scale = img_scale * 0.5;
    let second = img_scale;
    let cross = img_scale * 0.25;
    let (mut xi, mut xr, mut xc) = (0f32, 0f32, 0f32);
    let dper = N_LAYERS + 2;
    let mut i = 0;
    while i < 5 {
        let idx = octv * dper + *layer as usize;
        let (img, prev, next) = (&dog[idx], &dog[idx - 1], &dog[idx + 1]);
        let at = |m: &Fm, rr: i32, cc: i32| m.d[rr as usize * m.w + cc as usize];
        let (rr, cc) = (*r, *c);
        let d_d = [
            (at(img, rr, cc + 1) - at(img, rr, cc - 1)) * deriv_scale,
            (at(img, rr + 1, cc) - at(img, rr - 1, cc)) * deriv_scale,
            (at(next, rr, cc) - at(prev, rr, cc)) * deriv_scale,
        ];
        let v2 = at(img, rr, cc) * 2.0;
        let dxx = (at(img, rr, cc + 1) + at(img, rr, cc - 1) - v2) * second;
        let dyy = (at(img, rr + 1, cc) + at(img, rr - 1, cc) - v2) * second;
        let dss = (at(next, rr, cc) + at(prev, rr, cc) - v2) * second;
        let dxy = (at(img, rr + 1, cc + 1) - at(img, rr + 1, cc - 1) - at(img, rr - 1, cc + 1) + at(img, rr - 1, cc - 1)) * cross;
        let dxs = (at(next, rr, cc + 1) - at(next, rr, cc - 1) - at(prev, rr, cc + 1) + at(prev, rr, cc - 1)) * cross;
        let dys = (at(next, rr + 1, cc) - at(next, rr - 1, cc) - at(prev, rr + 1, cc) + at(prev, rr - 1, cc)) * cross;
        let x = solve3([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]], d_d);
        xi = -x[2];
        xr = -x[1];
        xc = -x[0];
        if xi.abs() < 0.5 && xr.abs() < 0.5 && xc.abs() < 0.5 {
            break;
        }
        let big = (i32::MAX / 3) as f32;
        if xi.abs() > big || xr.abs() > big || xc.abs() > big {
            return None;
        }
        *c += cv_round_f(xc);
        *r += cv_round_f(xr);
        *layer += cv_round_f(xi);
        if *layer < 1 || *layer > N_LAYERS as i32 || *c < IMG_BORDER as i32 || *c >= img.w as i32 - IMG_BORDER as i32
            || *r < IMG_BORDER as i32 || *r >= img.h as i32 - IMG_BORDER as i32 {
            return None;
        }
        i += 1;
    }
    if i >= 5 {
        return None;
    }
    let idx = octv * dper + *layer as usize;
    let (img, prev, next) = (&dog[idx], &dog[idx - 1], &dog[idx + 1]);
    let at = |m: &Fm, rr: i32, cc: i32| m.d[rr as usize * m.w + cc as usize];
    let (rr, cc) = (*r, *c);
    let d_d = [
        (at(img, rr, cc + 1) - at(img, rr, cc - 1)) * deriv_scale,
        (at(img, rr + 1, cc) - at(img, rr - 1, cc)) * deriv_scale,
        (at(next, rr, cc) - at(prev, rr, cc)) * deriv_scale,
    ];
    let t = d_d[0] * xc + d_d[1] * xr + d_d[2] * xi;
    let contr = at(img, rr, cc) * img_scale + t * 0.5;
    if contr.abs() * (N_LAYERS as f32) < CONTRAST as f32 {
        return None;
    }
    let v2 = at(img, rr, cc) * 2.0;
    let dxx = (at(img, rr, cc + 1) + at(img, rr, cc - 1) - v2) * second;
    let dyy = (at(img, rr + 1, cc) + at(img, rr - 1, cc) - v2) * second;
    let dxy = (at(img, rr + 1, cc + 1) - at(img, rr + 1, cc - 1) - at(img, rr - 1, cc + 1) + at(img, rr - 1, cc - 1)) * cross;
    let tr = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    let e = EDGE as f32;
    if det <= 0.0 || tr * tr * e >= (e + 1.0) * (e + 1.0) * det {
        return None;
    }
    let o = (1 << octv) as f32;
    Some(KeyPoint {
        x: (cc as f32 + xc) * o,
        y: (rr as f32 + xr) * o,
        octave: octv as i32 + (*layer << 8) + (cv_round((xi as f64 + 0.5) * 255.0) << 16),
        size: (SIGMA as f32) * 2f32.powf((*layer as f32 + xi) / N_LAYERS as f32) * o * 2.0,
        response: contr.abs(),
        angle: 0.0,
    })
}

fn orientation_hist(img: &Fm, px: i32, py: i32, radius: i32, sigma: f32, hist: &mut [f32; ORI_BINS]) -> f32 {
    let n = ORI_BINS;
    let expf_scale = -1.0f32 / (2.0 * sigma * sigma);
    let mut xs = vec![];
    let mut ys = vec![];
    let mut ws = vec![];
    for i in -radius..=radius {
        let y = py + i;
        if y <= 0 || y >= img.h as i32 - 1 {
            continue;
        }
        for j in -radius..=radius {
            let x = px + j;
            if x <= 0 || x >= img.w as i32 - 1 {
                continue;
            }
            let (xu, yu) = (x as usize, y as usize);
            xs.push(img.at(xu + 1, yu) - img.at(xu - 1, yu));
            ys.push(img.at(xu, yu - 1) - img.at(xu, yu + 1));
            ws.push((i * i + j * j) as f32 * expf_scale);
        }
    }
    let mut temp = vec![0f32; n + 4];
    for k in 0..xs.len() {
        let w = ws[k].exp();
        let ori = fast_atan2(ys[k], xs[k]);
        let mag = xs[k].fma(xs[k], ys[k] * ys[k]).sqrt();
        let mut bin = cv_round_f((n as f32 / 360.0) * ori);
        if bin >= n as i32 {
            bin -= n as i32;
        }
        if bin < 0 {
            bin += n as i32;
        }
        temp[bin as usize + 2] += w * mag;
    }
    temp[1] = temp[n + 1];
    temp[0] = temp[n];
    temp[n + 2] = temp[2];
    temp[n + 3] = temp[3];
    let t = |i: usize| temp[i + 2];
    let tm = |i: usize, d: usize| temp[i + 2 - d];
    for i in 0..n {
        // 8-lane vectors (fused) for the first 32 bins, plain arithmetic after
        hist[i] = if i < 32 {
            (tm(i, 2) + t(i + 2)).fma(1.0 / 16.0, (tm(i, 1) + t(i + 1)).fma(4.0 / 16.0, t(i) * (6.0 / 16.0)))
        } else {
            (tm(i, 2) + t(i + 2)) * (1.0 / 16.0) + (tm(i, 1) + t(i + 1)) * (4.0 / 16.0) + t(i) * (6.0 / 16.0)
        };
    }
    let mut maxval = hist[0];
    for &h in hist.iter().skip(1) {
        maxval = maxval.max(h);
    }
    maxval
}

fn descriptor(img: &Fm, ptx: f32, pty: f32, ori: f32, scl: f32) -> [f32; 128] {
    const D: usize = 4;
    const N: usize = 8;
    let (px, py) = (cv_round_f(ptx), cv_round_f(pty));
    let mut cos_t = (ori * (std::f64::consts::PI / 180.0) as f32).cos();
    let mut sin_t = (ori * (std::f64::consts::PI / 180.0) as f32).sin();
    let bins_per_rad = N as f32 / 360.0;
    let exp_scale = -1.0f32 / (D as f32 * D as f32 * 0.5);
    let hist_width = 3.0f32 * scl;
    let mut radius = cv_round_f(hist_width * 1.414_213_6_f32 * (D as f32 + 1.0) * 0.5);
    radius = radius.min(((img.w as f64) * img.w as f64 + (img.h as f64) * img.h as f64).sqrt() as i32);
    cos_t /= hist_width;
    sin_t /= hist_width;
    let (rows, cols) = (img.h as i32, img.w as i32);
    let hl = (D + 2) * (D + 2) * (N + 2);
    let mut hist = vec![0f32; hl];
    for i in -radius..=radius {
        for j in -radius..=radius {
            let c_rot = j as f32 * cos_t - i as f32 * sin_t;
            let r_rot = j as f32 * sin_t + i as f32 * cos_t;
            let rbin = r_rot + (D / 2) as f32 - 0.5;
            let cbin = c_rot + (D / 2) as f32 - 0.5;
            let (r, c) = (py + i, px + j);
            if rbin > -1.0 && rbin < D as f32 && cbin > -1.0 && cbin < D as f32 && r > 0 && r < rows - 1 && c > 0 && c < cols - 1 {
                let (ru, cu) = (r as usize, c as usize);
                let dx = img.at(cu + 1, ru) - img.at(cu - 1, ru);
                let dy = img.at(cu, ru - 1) - img.at(cu, ru + 1);
                let w = ((c_rot * c_rot + r_rot * r_rot) * exp_scale).exp();
                let o = fast_atan2(dy, dx);
                let m = dx.fma(dx, dy * dy).sqrt();
                let mut rbin = rbin;
                let mut cbin = cbin;
                let mut obin = (o - ori) * bins_per_rad;
                let mag = m * w;
                let r0 = rbin.floor() as i32;
                let c0 = cbin.floor() as i32;
                let mut o0 = obin.floor() as i32;
                rbin -= r0 as f32;
                cbin -= c0 as f32;
                obin -= o0 as f32;
                if o0 < 0 {
                    o0 += N as i32;
                }
                if o0 >= N as i32 {
                    o0 -= N as i32;
                }
                let v_r1 = mag * rbin;
                let v_r0 = mag - v_r1;
                let v_rc11 = v_r1 * cbin;
                let v_rc10 = v_r1 - v_rc11;
                let v_rc01 = v_r0 * cbin;
                let v_rc00 = v_r0 - v_rc01;
                let v_rco111 = v_rc11 * obin;
                let v_rco110 = v_rc11 - v_rco111;
                let v_rco101 = v_rc10 * obin;
                let v_rco100 = v_rc10 - v_rco101;
                let v_rco011 = v_rc01 * obin;
                let v_rco010 = v_rc01 - v_rco011;
                let v_rco001 = v_rc00 * obin;
                let v_rco000 = v_rc00 - v_rco001;
                let idx = (((r0 + 1) * (D as i32 + 2) + c0 + 1) * (N as i32 + 2) + o0) as usize;
                let (n2, d2) = (N + 2, D + 2);
                hist[idx] += v_rco000;
                hist[idx + 1] += v_rco001;
                hist[idx + n2] += v_rco010;
                hist[idx + n2 + 1] += v_rco011;
                hist[idx + d2 * n2] += v_rco100;
                hist[idx + d2 * n2 + 1] += v_rco101;
                hist[idx + (d2 + 1) * n2] += v_rco110;
                hist[idx + (d2 + 1) * n2 + 1] += v_rco111;
            }
        }
    }
    let mut raw = [0f32; 128];
    for i in 0..D {
        for j in 0..D {
            let idx = ((i + 1) * (D + 2) + (j + 1)) * (N + 2);
            hist[idx] += hist[idx + N];
            hist[idx + 1] += hist[idx + N + 1];
            for k in 0..N {
                raw[(i * D + j) * N + k] = hist[idx + k];
            }
        }
    }
    // the vectorised sum of squares: 8 lanes, fused, then added up
    let lanes = |v: &[f32; 128]| -> f32 {
        let mut acc = [0f32; 8];
        for c in 0..16 {
            for l in 0..8 {
                let x = v[c * 8 + l];
                acc[l] = x.fma(x, acc[l]);
            }
        }
        ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
    };
    let thr = lanes(&raw).sqrt() * 0.2;
    let mut nrm2 = 0f32;
    for v in raw.iter_mut() {
        let x = v.min(thr);
        *v = x;
        nrm2 += x * x;
    }
    let nrm2 = 512.0f32 / nrm2.sqrt().max(f32::EPSILON);
    let mut out = [0f32; 128];
    for k in 0..128 {
        out[k] = (cv_round_f(raw[k] * nrm2)).clamp(0, 255) as f32;
    }
    out
}

// ------------------------------------------------- MSVC nth_element etc.
fn med3<T: Copy>(v: &mut [T], a: usize, m: usize, l: usize, lt: &impl Fn(&T, &T) -> bool) {
    if lt(&v[m], &v[a]) {
        v.swap(m, a);
    }
    if lt(&v[l], &v[m]) {
        v.swap(l, m);
        if lt(&v[m], &v[a]) {
            v.swap(m, a);
        }
    }
}

fn guess_median<T: Copy>(v: &mut [T], first: usize, mid: usize, last: usize, lt: &impl Fn(&T, &T) -> bool) {
    let count = last - first;
    if count > 40 {
        let step = (count + 1) >> 3;
        let two = step << 1;
        med3(v, first, first + step, first + two, lt);
        med3(v, mid - step, mid, mid + step, lt);
        med3(v, last - two, last - step, last, lt);
        med3(v, first + step, mid, last - step, lt);
    } else {
        med3(v, first, mid, last, lt);
    }
}

fn partition_by_median_guess<T: Copy>(v: &mut [T], first: usize, last: usize, lt: &impl Fn(&T, &T) -> bool) -> (usize, usize) {
    let mid = first + ((last - first) >> 1);
    guess_median(v, first, mid, last - 1, lt);
    let mut pfirst = mid;
    let mut plast = pfirst + 1;
    while first < pfirst && !lt(&v[pfirst - 1], &v[pfirst]) && !lt(&v[pfirst], &v[pfirst - 1]) {
        pfirst -= 1;
    }
    while plast < last && !lt(&v[plast], &v[pfirst]) && !lt(&v[pfirst], &v[plast]) {
        plast += 1;
    }
    let mut gfirst = plast;
    let mut glast = pfirst;
    loop {
        while gfirst < last {
            if lt(&v[pfirst], &v[gfirst]) {
            } else if lt(&v[gfirst], &v[pfirst]) {
                break;
            } else if plast != gfirst {
                v.swap(plast, gfirst);
                plast += 1;
            } else {
                plast += 1;
            }
            gfirst += 1;
        }
        while first < glast {
            let gp = glast - 1;
            if lt(&v[gp], &v[pfirst]) {
            } else if lt(&v[pfirst], &v[gp]) {
                break;
            } else {
                pfirst -= 1;
                if pfirst != gp {
                    v.swap(pfirst, gp);
                }
            }
            glast -= 1;
        }
        if glast == first && gfirst == last {
            return (pfirst, plast);
        }
        if glast == first {
            if plast != gfirst {
                v.swap(pfirst, plast);
            }
            plast += 1;
            v.swap(pfirst, gfirst);
            pfirst += 1;
            gfirst += 1;
        } else if gfirst == last {
            glast -= 1;
            pfirst -= 1;
            if glast != pfirst {
                v.swap(glast, pfirst);
            }
            plast -= 1;
            v.swap(pfirst, plast);
        } else {
            glast -= 1;
            v.swap(gfirst, glast);
            gfirst += 1;
        }
    }
}

fn insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, lt: &impl Fn(&T, &T) -> bool) {
    if first == last {
        return;
    }
    let mut mid = first + 1;
    while mid != last {
        let val = v[mid];
        if lt(&val, &v[first]) {
            let mut i = mid;
            while i > first {
                v[i] = v[i - 1];
                i -= 1;
            }
            v[first] = val;
        } else {
            let mut hole = mid;
            while lt(&val, &v[hole - 1]) {
                v[hole] = v[hole - 1];
                hole -= 1;
            }
            v[hole] = val;
        }
        mid += 1;
    }
}

/// std::nth_element as Microsoft's STL does it (the order of the rest is
/// what decides which keypoint RANSAC draws).
pub fn msvc_nth_element<T: Copy>(v: &mut [T], nth: usize, lt: &impl Fn(&T, &T) -> bool) {
    let (mut first, mut last) = (0usize, v.len());
    if nth == last {
        return;
    }
    let mut ideal = last - first;
    loop {
        let len = last - first;
        if len <= 32 {
            insertion_sort(v, first, last, lt);
            return;
        }
        if ideal < (len >> 1) {
            // median-of-medians fallback (newer STLs only): never reached on
            // keypoints in practice; keep going the classic way
        }
        let m = partition_by_median_guess(v, first, last, lt);
        if m.1 <= nth {
            first = m.1;
        } else if m.0 <= nth {
            return;
        } else {
            last = m.0;
        }
        ideal = (ideal >> 1) + (ideal >> 2);
    }
}

/// std::partition (bidirectional), Microsoft's STL: returns the split point.
pub fn msvc_partition<T: Copy>(v: &mut [T], pred: impl Fn(&T) -> bool) -> usize {
    let (mut first, mut last) = (0usize, v.len());
    loop {
        loop {
            if first == last {
                return first;
            }
            if !pred(&v[first]) {
                break;
            }
            first += 1;
        }
        loop {
            last -= 1;
            if first == last {
                return first;
            }
            if pred(&v[last]) {
                break;
            }
        }
        v.swap(first, last);
        first += 1;
    }
}

// ------------------------------------------------------- matching, RANSAC
/// BFMatcher(NORM_L2).knnMatch(query, train, k=2): (train index, distance) x 2.
pub fn knn2(q: &[[f32; 128]], t: &[[f32; 128]]) -> Vec<[(usize, f32); 2]> {
    if let Some(r) = crate::cvlib::knn2(q, t) {
        return r;
    }
    use rayon::prelude::*;
    q.par_iter().map(|a| {
        let mut best = [(usize::MAX, f32::MAX); 2];
        for (j, b) in t.iter().enumerate() {
            let mut acc = [0f32; 8];
            for c in 0..16 {
                for l in 0..8 {
                    let d = a[c * 8 + l] - b[c * 8 + l];
                    acc[l] = d.fma(d, acc[l]);
                }
            }
            let s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
            let d = s.sqrt();
            if d < best[1].1 {
                if d < best[0].1 {
                    best[1] = best[0];
                    best[0] = (j, d);
                } else {
                    best[1] = (j, d);
                }
            }
        }
        best
    }).collect()
}

/// cv::RNG
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = (self.0 as u32 as u64).wrapping_mul(4164903690u64).wrapping_add(self.0 >> 32);
        self.0 as u32
    }
    fn uniform(&mut self, a: i32, b: i32) -> i32 {
        if a == b { a } else { (self.next() % (b - a) as u32) as i32 + a }
    }
}

fn update_num_iters(p: f64, ep: f64, model_points: i32, max_iters: i32) -> i32 {
    let p = p.clamp(0.0, 1.0);
    let ep = ep.clamp(0.0, 1.0);
    let num = (1.0 - p).max(f64::MIN_POSITIVE);
    let denom = 1.0 - (1.0 - ep).powi(model_points);
    if denom < f64::MIN_POSITIVE {
        return 0;
    }
    let num = num.ln();
    let denom = denom.ln();
    if denom >= 0.0 || -num >= max_iters as f64 * (-denom) { max_iters } else { cv_round(num / denom) }
}

fn partial_kernel(f: [[f32; 2]; 2], t: [[f32; 2]; 2]) -> [f64; 6] {
    let (x1, y1, x2, y2) = (f[0][0] as f64, f[0][1] as f64, f[1][0] as f64, f[1][1] as f64);
    let (xx1, yy1, xx2, yy2) = (t[0][0] as f64, t[0][1] as f64, t[1][0] as f64, t[1][1] as f64);
    let d = 1.0 / ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    let s0 = d * ((xx1 - xx2) * (x1 - x2) + (yy1 - yy2) * (y1 - y2));
    let s1 = d * ((yy1 - yy2) * (x1 - x2) - (xx1 - xx2) * (y1 - y2));
    let s2 = d * ((yy1 - yy2) * (x1 * y2 - x2 * y1) - (xx1 * y2 - xx2 * y1) * (y1 - y2) - (xx1 * x2 - xx2 * x1) * (x1 - x2));
    let s3 = d * (-(xx1 - xx2) * (x1 * y2 - x2 * y1) - (yy1 * x2 - yy2 * x1) * (x1 - x2) - (yy1 * y2 - yy2 * y1) * (y1 - y2));
    [s0, -s1, s2, s1, s0, s3]
}

fn find_inliers(src: &[[f32; 2]], dst: &[[f32; 2]], m: &[f64; 6], thr: f64, mask: &mut [u8]) -> usize {
    let f: Vec<f32> = m.iter().map(|&v| v as f32).collect();
    let t = (thr * thr) as f32;
    let mut nz = 0;
    for i in 0..src.len() {
        let (fp, tp) = (src[i], dst[i]);
        let a = f[0] * fp[0] + f[1] * fp[1] + f[2] - tp[0];
        let b = f[3] * fp[0] + f[4] * fp[1] + f[5] - tp[1];
        let e = a * a + b * b;
        let ok = e <= t;
        mask[i] = ok as u8;
        nz += ok as usize;
    }
    nz
}

/// cv::estimateAffinePartial2D(src, dst, RANSAC, thr) with its defaults
/// (2000 iterations, confidence 0.99, 10 refining iterations).
pub fn estimate_affine_partial(src: &[[f32; 2]], dst: &[[f32; 2]], thr: f64) -> Option<([f64; 6], Vec<u8>)> {
    if let Some(r) = crate::cvlib::affine_partial(src, dst, thr) {
        return r;
    }
    let count = src.len();
    let model_points = 2;
    if count < model_points {
        return None;
    }
    let mut rng = Rng(u64::MAX);
    let mut niters = 2000i32;
    let mut max_good = 0usize;
    let mut best = [0f64; 6];
    let mut best_mask = vec![0u8; count];
    let mut mask = vec![0u8; count];
    if count == model_points {
        best = partial_kernel([src[0], src[1]], [dst[0], dst[1]]);
        best_mask.iter_mut().for_each(|v| *v = 1);
        return Some((best, best_mask));
    }
    let mut iter = 0;
    while iter < niters {
        // getSubset: two different indices (the collinearity test needs three)
        let i0 = rng.uniform(0, count as i32) as usize;
        let mut i1 = rng.uniform(0, count as i32) as usize;
        while i1 == i0 {
            i1 = rng.uniform(0, count as i32) as usize;
        }
        let model = partial_kernel([src[i0], src[i1]], [dst[i0], dst[i1]]);
        let good = find_inliers(src, dst, &model, thr, &mut mask);
        if good > max_good.max(model_points - 1) {
            std::mem::swap(&mut mask, &mut best_mask);
            best = model;
            max_good = good;
            niters = update_num_iters(0.99, (count - good) as f64 / count as f64, model_points as i32, niters);
        }
        iter += 1;
    }
    if max_good == 0 {
        return None;
    }
    // refine on the inliers (in their order) with Levenberg-Marquardt
    let (mut s2, mut d2) = (vec![], vec![]);
    for i in 0..count {
        if best_mask[i] != 0 {
            s2.push(src[i]);
            d2.push(dst[i]);
        }
    }
    if !s2.is_empty() && count > 2 {
        let mut h = [best[0], best[3], best[2], best[5]];
        lm_partial(&s2, &d2, &mut h, 10);
        best = [h[0], -h[1], h[2], h[1], h[0], h[3]];
    }
    Some((best, best_mask))
}

fn lm_compute(src: &[[f32; 2]], dst: &[[f32; 2]], h: &[f64; 4], jac: bool) -> (Vec<f64>, Vec<[f64; 4]>) {
    let mut r = Vec::with_capacity(src.len() * 2);
    let mut j = Vec::with_capacity(if jac { src.len() * 2 } else { 0 });
    for i in 0..src.len() {
        let (mx, my) = (src[i][0] as f64, src[i][1] as f64);
        let xi = h[0] * mx - h[1] * my + h[2];
        let yi = h[1] * mx + h[0] * my + h[3];
        r.push(xi - dst[i][0] as f64);
        r.push(yi - dst[i][1] as f64);
        if jac {
            j.push([mx, -my, 1.0, 0.0]);
            j.push([my, mx, 0.0, 1.0]);
        }
    }
    (r, j)
}

fn solve4(a: [[f64; 4]; 4], b: [f64; 4]) -> [f64; 4] {
    let mut m = [[0f64; 5]; 4];
    for i in 0..4 {
        for k in 0..4 {
            m[i][k] = a[i][k];
        }
        m[i][4] = b[i];
    }
    for c in 0..4 {
        let mut p = c;
        for r in c + 1..4 {
            if m[r][c].abs() > m[p][c].abs() {
                p = r;
            }
        }
        m.swap(c, p);
        let d = m[c][c];
        if d.abs() < 1e-300 {
            return [0.0; 4];
        }
        for r in 0..4 {
            if r != c {
                let f = m[r][c] / d;
                for k in c..5 {
                    m[r][k] -= f * m[c][k];
                }
            }
        }
    }
    [m[0][4] / m[0][0], m[1][4] / m[1][1], m[2][4] / m[2][2], m[3][4] / m[3][3]]
}

fn lm_partial(src: &[[f32; 2]], dst: &[[f32; 2]], x: &mut [f64; 4], max_iters: usize) {
    let jtj = |j: &[[f64; 4]]| {
        let mut a = [[0f64; 4]; 4];
        for row in j {
            for p in 0..4 {
                for q in 0..4 {
                    a[p][q] += row[p] * row[q];
                }
            }
        }
        a
    };
    let jtr = |j: &[[f64; 4]], r: &[f64]| {
        let mut v = [0f64; 4];
        for (row, &e) in j.iter().zip(r) {
            for p in 0..4 {
                v[p] += row[p] * e;
            }
        }
        v
    };
    let sq = |r: &[f64]| r.iter().map(|v| v * v).sum::<f64>();
    let (mut r, mut j) = lm_compute(src, dst, x, true);
    let mut s = sq(&r);
    let mut a = jtj(&j);
    let mut v = jtr(&j, &r);
    let dg = [a[0][0], a[1][1], a[2][2], a[3][3]];
    let (rlo, rhi) = (0.25, 0.75);
    let (mut lambda, mut lc) = (1.0f64, 0.75f64);
    let mut iter = 0;
    loop {
        let mut ap = a;
        for i in 0..4 {
            ap[i][i] += lambda * dg[i];
        }
        let d = solve4(ap, v);
        let xd = [x[0] - d[0], x[1] - d[1], x[2] - d[2], x[3] - d[3]];
        let (rd, _) = lm_compute(src, dst, &xd, false);
        let sd = sq(&rd);
        let mut temp = [0f64; 4];
        for p in 0..4 {
            let mut ad = 0.0;
            for q in 0..4 {
                ad += a[p][q] * d[q];
            }
            temp[p] = -ad + 2.0 * v[p];
        }
        let ds: f64 = (0..4).map(|p| d[p] * temp[p]).sum();
        let rr = (s - sd) / (if ds.abs() > f64::EPSILON { ds } else { 1.0 });
        if rr > rhi {
            lambda *= 0.5;
            if lambda < lc {
                lambda = 0.0;
            }
        } else if rr < rlo {
            let t: f64 = (0..4).map(|p| d[p] * v[p]).sum();
            let mut nu = (sd - s) / (if t.abs() > f64::EPSILON { t } else { 1.0 }) + 2.0;
            nu = nu.max(2.0).min(10.0);
            if lambda == 0.0 {
                // invert(A, DECOMP_EIG): the largest diagonal of the inverse
                let mut maxval = f64::EPSILON;
                for i in 0..4 {
                    let mut e = [0f64; 4];
                    e[i] = 1.0;
                    let col = solve4(a, e);
                    maxval = maxval.max(col[i].abs());
                }
                lambda = 1.0 / maxval;
                lc = lambda;
                nu *= 0.5;
            }
            lambda *= nu;
        }
        if sd < s {
            s = sd;
            *x = xd;
            let rj = lm_compute(src, dst, x, true);
            r = rj.0;
            j = rj.1;
            a = jtj(&j);
            v = jtr(&j, &r);
        }
        iter += 1;
        let dinf = d.iter().fold(0f64, |m, v| m.max(v.abs()));
        let rinf = r.iter().fold(0f64, |m, v| m.max(v.abs()));
        let eps = f32::EPSILON as f64;
        if !(iter < max_iters && dinf >= eps && rinf >= eps) {
            break;
        }
    }
}

// ------------------------------------------------------------ warpAffine
const AB_BITS: i32 = 10;
const AB_SCALE: f64 = 1024.0;
const TAB: i32 = 32;

fn bilinear_tab_f() -> &'static Vec<[f32; 4]> {
    static T: std::sync::OnceLock<Vec<[f32; 4]>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let scale = 1.0f32 / TAB as f32;
        let t1: Vec<[f32; 2]> = (0..TAB).map(|i| { let x = i as f32 * scale; [1.0 - x, x] }).collect();
        let mut out = vec![];
        for i in 0..TAB as usize {
            for j in 0..TAB as usize {
                out.push([t1[i][0] * t1[j][0], t1[i][0] * t1[j][1], t1[i][1] * t1[j][0], t1[i][1] * t1[j][1]]);
            }
        }
        out
    })
}

fn bilinear_tab_i() -> Vec<[i32; 4]> {
    // the 8-bit table with OpenCV's fix-up, as cvx builds it for warpPerspective
    let ft = bilinear_tab_f();
    let mut out = vec![];
    for t in ft.iter() {
        let mut v = [0i32; 4];
        let mut isum = 0;
        for k in 0..4 {
            let x = (t[k] * 32768.0).round_ties_even().clamp(-32768.0, 32767.0) as i32;
            v[k] = x;
            isum += x;
        }
        if isum != 32768 {
            let diff = isum - 32768;
            // ksize 2: the fix-up looks at k1, k2 in 1..2 -> only index 3
            let (mk, big) = (3usize, 3usize);
            if diff < 0 {
                v[big] -= diff;
            } else {
                v[mk] -= diff;
            }
        }
        out.push(v);
    }
    out
}

/// invertAffineTransform (double)
fn invert_affine(m: &[f64; 6]) -> [f64; 6] {
    let d = m[0] * m[4] - m[1] * m[3];
    let d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let a11 = m[4] * d;
    let a22 = m[0] * d;
    let a12 = -m[1] * d;
    let a21 = -m[3] * d;
    let b1 = -a11 * m[2] - a12 * m[5];
    let b2 = -a21 * m[2] - a22 * m[5];
    [a11, a12, b1, a21, a22, b2]
}

/// For each destination pixel: the fixed-point source position (X, Y in
/// 1/32 pixel, or whole pixels for nearest), as WarpAffineInvoker makes it.
/// `m` maps destination to source.
fn warp_coords(m: &[f64; 6], w: usize, h: usize, nearest: bool) -> Vec<(i32, i32)> {
    let round_delta: i32 = if nearest { (AB_SCALE / 2.0) as i32 } else { (AB_SCALE as i32) / TAB / 2 };
    let adelta: Vec<i32> = (0..w).map(|x| cv_round(m[0] * x as f64 * AB_SCALE)).collect();
    let bdelta: Vec<i32> = (0..w).map(|x| cv_round(m[3] * x as f64 * AB_SCALE)).collect();
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let x0 = cv_round((m[1] * y as f64 + m[2]) * AB_SCALE).wrapping_add(round_delta);
        let y0 = cv_round((m[4] * y as f64 + m[5]) * AB_SCALE).wrapping_add(round_delta);
        for x in 0..w {
            let xx = x0.wrapping_add(adelta[x]);
            let yy = y0.wrapping_add(bdelta[x]);
            if nearest {
                out.push((xx >> AB_BITS, yy >> AB_BITS));
            } else {
                out.push((xx >> (AB_BITS - 5), yy >> (AB_BITS - 5)));
            }
        }
    }
    out
}

/// cv::warpAffine(src float, M, (w, h), INTER_LINEAR | WARP_INVERSE_MAP), border 0.
fn warp_affine_f32_inv(src: &Fm, m: &[f64; 6], w: usize, h: usize) -> Fm {
    let tab = bilinear_tab_f();
    let xy = warp_coords(m, w, h, false);
    let mut out = Fm::new(w, h);
    let (sw, sh) = (src.w as i32, src.h as i32);
    for (i, &(x, y)) in xy.iter().enumerate() {
        let (sx, sy) = (x >> 5, y >> 5);
        let wt = &tab[((y & 31) * 32 + (x & 31)) as usize];
        if sx >= sw || sx + 1 < 0 || sy >= sh || sy + 1 < 0 {
            continue;
        }
        let at = |xx: i32, yy: i32| if xx >= 0 && yy >= 0 && xx < sw && yy < sh { src.d[(yy * sw + xx) as usize] } else { 0.0 };
        out.d[i] = at(sx, sy) * wt[0] + at(sx + 1, sy) * wt[1] + at(sx, sy + 1) * wt[2] + at(sx + 1, sy + 1) * wt[3];
    }
    out
}

/// cv::warpAffine(mask u8, M, INTER_NEAREST | WARP_INVERSE_MAP), border 0.
fn warp_affine_u8_nearest_inv(src: &[u8], sw: usize, sh: usize, m: &[f64; 6], w: usize, h: usize) -> Vec<u8> {
    let xy = warp_coords(m, w, h, true);
    xy.iter().map(|&(x, y)| if x >= 0 && y >= 0 && (x as usize) < sw && (y as usize) < sh { src[y as usize * sw + x as usize] } else { 0 }).collect()
}

/// cv::warpAffine(src u8, M, (w, h), INTER_LINEAR) with M source -> destination,
/// border constant 0 (or replicate).
pub fn warp_affine_u8(src: &Img, m_fwd: &[f64; 6], w: usize, h: usize, replicate: bool) -> Img {
    let border = if replicate { crate::cvlib::BORDER_REPLICATE } else { crate::cvlib::BORDER_CONSTANT };
    if let Some(r) = crate::cvlib::warp_u8(src, m_fwd, w, h, crate::cvlib::INTER_LINEAR, border) {
        return r;
    }
    let m = invert_affine(m_fwd);
    let tab = bilinear_tab_i();
    let xy = warp_coords(&m, w, h, false);
    let mut out = Img::new(w, h, 1);
    let (sw, sh) = (src.w as i32, src.h as i32);
    for (i, &(x, y)) in xy.iter().enumerate() {
        let (sx, sy) = (x >> 5, y >> 5);
        let wt = &tab[((y & 31) * 32 + (x & 31)) as usize];
        if !replicate && (sx >= sw || sx + 1 < 0 || sy >= sh || sy + 1 < 0) {
            continue;
        }
        let at = |xx: i32, yy: i32| -> i32 {
            if replicate {
                src.data[(yy.clamp(0, sh - 1) * sw + xx.clamp(0, sw - 1)) as usize] as i32
            } else if xx >= 0 && yy >= 0 && xx < sw && yy < sh {
                src.data[(yy * sw + xx) as usize] as i32
            } else {
                0
            }
        };
        let s = at(sx, sy) * wt[0] + at(sx + 1, sy) * wt[1] + at(sx, sy + 1) * wt[2] + at(sx + 1, sy + 1) * wt[3];
        out.data[i] = ((s + (1 << 14)) >> 15).clamp(0, 255) as u8;
    }
    out
}

// ------------------------------------------------------------------- ECC
/// cv::findTransformECC(template, input, warp, MOTION_EUCLIDEAN,
/// (EPS | COUNT, iters, eps), None, 5). None where OpenCV raises.
pub fn ecc_euclidean(templ: &Img, input: &Img, warp: &mut [f32; 6], iters: usize, eps: f64) -> Option<f64> {
    if let Some(r) = crate::cvlib::ecc(templ, input, warp, iters, eps, 5) {
        return r;
    }
    let (ws, hs) = (templ.w, templ.h);
    let (wd, hd) = (input.w, input.h);
    let tf = gauss_f32(&Fm::from_u8(templ), 5, 0.0);
    let imf = gauss_f32(&Fm::from_u8(input), 5, 0.0);
    // gradients: filter2D with (-0.5, 0, 0.5), reflect-101 (mask is all ones)
    let mut gx = Fm::new(wd, hd);
    let mut gy = Fm::new(wd, hd);
    for y in 0..hd {
        for x in 0..wd {
            let l = imf.at(reflect101(x as i64 - 1, wd), y);
            let r = imf.at(reflect101(x as i64 + 1, wd), y);
            gx.d[y * wd + x] = r.fma(0.5, l * -0.5);
            let u = imf.at(x, reflect101(y as i64 - 1, hd));
            let d = imf.at(x, reflect101(y as i64 + 1, hd));
            gy.d[y * wd + x] = d.fma(0.5, u * -0.5);
        }
    }
    let ones = vec![1u8; wd * hd];
    let n = ws * hs;
    let mut rho = -1.0f64;
    let mut last_rho = -eps;
    let mut i = 1;
    while i <= iters && (rho - last_rho).abs() >= eps {
        let m: [f64; 6] = [warp[0] as f64, warp[1] as f64, warp[2] as f64, warp[3] as f64, warp[4] as f64, warp[5] as f64];
        let mut iw = warp_affine_f32_inv(&imf, &m, ws, hs);
        let gxw = warp_affine_f32_inv(&gx, &m, ws, hs);
        let gyw = warp_affine_f32_inv(&gy, &m, ws, hs);
        let mask = warp_affine_u8_nearest_inv(&ones, wd, hd, &m, ws, hs);
        let stats = |v: &Fm| {
            let (mut s, mut sq, mut c) = (0f64, 0f64, 0usize);
            for k in 0..n {
                if mask[k] != 0 {
                    let x = v.d[k] as f64;
                    s += x;
                    sq += x * x;
                    c += 1;
                }
            }
            let c = c.max(1) as f64;
            let mean = s / c;
            (mean, (sq / c - mean * mean).max(0.0).sqrt())
        };
        let (img_mean, img_std) = stats(&iw);
        let (tmp_mean, tmp_std) = stats(&tf);
        let nnz = mask.iter().filter(|&&v| v != 0).count() as f64;
        let (imf_mean, tmf_mean) = (img_mean as f32, tmp_mean as f32);
        let mut tzm = Fm::new(ws, hs);
        for k in 0..n {
            if mask[k] != 0 {
                iw.d[k] -= imf_mean;
                tzm.d[k] = tf.d[k] - tmf_mean;
            }
        }
        let tmp_norm = (nnz * tmp_std * tmp_std).sqrt();
        let img_norm = (nnz * img_std * img_std).sqrt();
        // Jacobian blocks
        // hatX = -(X h1) - (Y h0), hatY = X h0 - Y h1: each an addWeighted,
        // vectorised as fma(X, a, fma(Y, b, 0)) over the whole (continuous) grid
        let (h0, h1) = (warp[0], warp[3]);
        let mut j0 = vec![0f32; n];
        let vec_end = n / 8 * 8;
        for y in 0..hs {
            for x in 0..ws {
                let k = y * ws + x;
                let (xf, yf) = (x as f32, y as f32);
                let (hat_x, hat_y) = if k < vec_end {
                    (xf.fma(-h1, yf * -h0), xf.fma(h0, yf * -h1))
                } else {
                    ((xf as f64 * -(h1 as f64) + yf as f64 * -(h0 as f64)) as f32, (xf as f64 * h0 as f64 + yf as f64 * -(h1 as f64)) as f32)
                };
                j0[k] = gxw.d[k] * hat_x + gyw.d[k] * hat_y;
            }
        }
        let blocks: [&[f32]; 3] = [&j0, &gxw.d, &gyw.d];
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(&p, &q)| p as f64 * q as f64).sum::<f64>();
        let mut hess = [[0f32; 3]; 3];
        for p in 0..3 {
            // (float) pow(norm(block), 2)
            hess[p][p] = dot(blocks[p], blocks[p]).sqrt().powi(2) as f32;
            for q in p + 1..3 {
                let v = dot(blocks[p], blocks[q]) as f32;
                hess[p][q] = v;
                hess[q][p] = v;
            }
        }
        let hinv = inv3_f32(&hess)?;
        let correlation = dot(&tzm.d, &iw.d);
        last_rho = rho;
        rho = correlation / (img_norm * tmp_norm);
        if rho.is_nan() {
            return None;
        }
        let proj = |v: &[f32]| [dot(blocks[0], v) as f32, dot(blocks[1], v) as f32, dot(blocks[2], v) as f32];
        let ip = proj(&iw.d);
        let tp = proj(&tzm.d);
        // gemm of small float matrices: products summed in double
        let mv = |m: &[[f32; 3]; 3], v: [f32; 3]| -> [f32; 3] {
            std::array::from_fn(|r| (m[r][0] as f64 * v[0] as f64 + m[r][1] as f64 * v[1] as f64 + m[r][2] as f64 * v[2] as f64) as f32)
        };
        let iph = mv(&hinv, ip);
        let d3 = |a: [f32; 3], b: [f32; 3]| (0..3).map(|k| a[k] as f64 * b[k] as f64).sum::<f64>();
        let lambda_n = img_norm * img_norm - d3(ip, iph);
        let lambda_d = correlation - d3(tp, iph);
        if lambda_d <= 0.0 {
            return None;
        }
        let lambda = lambda_n / lambda_d;
        let lf = lambda as f32;
        // lambda * templateZM - imageWarped: addWeighted, fma(a, lambda, fma(b, -1, 0))
        let err: Vec<f32> = (0..n).map(|k| if k < vec_end { tzm.d[k].fma(lf, -iw.d[k]) } else {
            (tzm.d[k] as f64 * lambda - iw.d[k] as f64) as f32 }).collect();
        let ep = proj(&err);
        let dp = mv(&hinv, ep);
        let theta = dp[0] as f64 + (warp[3] as f64).asin();
        warp[2] += dp[1];
        warp[5] += dp[2];
        warp[0] = theta.cos() as f32;
        warp[4] = warp[0];
        warp[3] = theta.sin() as f32;
        warp[1] = -warp[3];
        i += 1;
    }
    Some(rho)
}

fn inv3_f32(s: &[[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let g = |r: usize, c: usize| s[r][c] as f64;
    let d = g(0, 0) * (g(1, 1) * g(2, 2) - g(1, 2) * g(2, 1)) - g(0, 1) * (g(1, 0) * g(2, 2) - g(1, 2) * g(2, 0))
        + g(0, 2) * (g(1, 0) * g(2, 1) - g(1, 1) * g(2, 0));
    if d == 0.0 {
        return None;
    }
    let d = 1.0 / d;
    Some([
        [((g(1, 1) * g(2, 2) - g(1, 2) * g(2, 1)) * d) as f32, ((g(0, 2) * g(2, 1) - g(0, 1) * g(2, 2)) * d) as f32, ((g(0, 1) * g(1, 2) - g(0, 2) * g(1, 1)) * d) as f32],
        [((g(1, 2) * g(2, 0) - g(1, 0) * g(2, 2)) * d) as f32, ((g(0, 0) * g(2, 2) - g(0, 2) * g(2, 0)) * d) as f32, ((g(0, 2) * g(1, 0) - g(0, 0) * g(1, 2)) * d) as f32],
        [((g(1, 0) * g(2, 1) - g(1, 1) * g(2, 0)) * d) as f32, ((g(0, 1) * g(2, 0) - g(0, 0) * g(2, 1)) * d) as f32, ((g(0, 0) * g(1, 1) - g(0, 1) * g(1, 0)) * d) as f32],
    ])
}

// -------------------------------------------------------- numpy float32
/// numpy's float32 sum (a.sum()): the reduction runs over chunks of 8192
/// (its buffer), each summed pairwise (8 accumulators, blocks of 128),
/// added in float32 from zero.
fn np_sum_f32(a: &[f32]) -> f32 {
    fn pw(a: &[f32]) -> f32 {
        let n = a.len();
        if n < 8 {
            let mut res = 0f32;
            for &v in a {
                res += v;
            }
            res
        } else if n <= 128 {
            let mut r = [0f32; 8];
            r.copy_from_slice(&a[..8]);
            let mut i = 8;
            while i < n - (n % 8) {
                for k in 0..8 {
                    r[k] += a[i + k];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            while i < n {
                res += a[i];
                i += 1;
            }
            res
        } else {
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            pw(&a[..n2]) + pw(&a[n2..])
        }
    }
    let mut r = 0f32;
    for c in a.chunks(8192) {
        r += pw(c);
    }
    r
}

/// numpy's float32 dot product: OpenBLAS 0.3.28 sdot, Haswell kernel --
/// 32 at a time in 4 x 8 fused accumulators, folded in a fixed order; the
/// rest added in double; rounded to float once.
fn np_dot_f32(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    let n1 = n & !31;
    let mut acc = [[0f32; 8]; 4];
    let mut i = 0;
    while i < n1 {
        for r in 0..4 {
            for l in 0..8 {
                let k = i + 8 * r + l;
                acc[r][l] = x[k].fma(y[k], acc[r][l]);
            }
        }
        i += 32;
    }
    let mydot = if n1 > 0 {
        let h: Vec<[f32; 4]> = acc.iter().map(|a| [a[0] + a[4], a[1] + a[5], a[2] + a[6], a[3] + a[7]]).collect();
        let s = [(h[0][0] + h[1][0]) + (h[2][0] + h[3][0]), (h[0][1] + h[1][1]) + (h[2][1] + h[3][1]),
            (h[0][2] + h[1][2]) + (h[2][2] + h[3][2]), (h[0][3] + h[1][3]) + (h[2][3] + h[3][3])];
        (s[0] + s[1]) + (s[2] + s[3])
    } else {
        0.0
    };
    let mut dot = 0f64;
    for k in n1..n {
        dot += (y[k] * x[k]) as f64;
    }
    dot += mydot as f64;
    dot as f32
}

/// np.linalg.norm of a float32 array: sqrt of its sdot with itself.
fn np_norm_f32(a: &[f32]) -> f32 {
    np_dot_f32(a, a).sqrt()
}

// --------------------------------------------------------------- the tool
#[derive(Clone, Debug, Default)]
pub struct Found {
    /// reference -> picture, float32 as Vision Station keeps it (3 x 3, last row 0 0 1)
    pub m: Option<[f32; 6]>,
    pub score: f64,
    pub method: &'static str,
    pub margin: Option<f64>,
    pub inliers: usize,
    pub debug: serde_json::Value,
}

pub struct Params {
    pub roi: Option<[f64; 4]>,
    pub method: String,
    pub min_score: f64,
    pub n_features: usize,
    pub min_inliers: usize,
    pub feature_px: f64,
    pub search_pad_px: Option<f64>,
    pub refine: bool,
    pub refine_size: f64,
    pub refine_iters: usize,
    pub angle_range: (f64, f64),
    pub angle_step: f64,
    pub pyramid: bool,
}

type Win = (usize, usize, usize, usize);

fn windows(p: &Params, w: usize, h: usize) -> (Win, Win) {
    let Some(roi) = p.roi else { return ((0, 0, w, h), (0, 0, w, h)) };
    let (x, y, rw, rh) = (roi[0] as i64, roi[1] as i64, roi[2] as i64, roi[3] as i64);
    let pad = match p.search_pad_px { Some(v) if v != 0.0 => v as i64, _ => (0.35 * rw.max(rh) as f64) as i64 };
    let (wi, hi) = (w as i64, h as i64);
    let rw_ = (x.max(0), y.max(0), wi.min(x + rw), hi.min(y + rh));
    let iw_ = ((x - pad).max(0), (y - pad).max(0), wi.min(x + rw + pad), hi.min(y + rh + pad));
    let c = |t: (i64, i64, i64, i64)| (t.0.max(0) as usize, t.1.max(0) as usize, t.2.max(0) as usize, t.3.max(0) as usize);
    (c(rw_), c(iw_))
}

type Mat3 = [[f64; 3]; 3];
fn mm(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [[0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}
fn m3(m: &[f32; 6]) -> Mat3 {
    [[m[0] as f64, m[1] as f64, m[2] as f64], [m[3] as f64, m[4] as f64, m[5] as f64], [0.0, 0.0, 1.0]]
}

/// Align._fit_roi: correlation over the fixture region, the picture warped
/// back into reference coordinates.
pub fn fit_roi(p: &Params, rg: &Img, ig: &Img, m: &[f32; 6]) -> f64 {
    let (hh, ww) = (rg.h as i64, rg.w as i64);
    let roi = p.roi.unwrap_or([0.0, 0.0, ww as f64, hh as f64]);
    let (mut x, mut y, w, h) = (roi[0] as i64, roi[1] as i64, roi[2] as i64, roi[3] as i64);
    x = x.max(0);
    y = y.max(0);
    let w = 8.max(w.min(ww - x));
    let h = 8.max(h.min(hh - y));
    let sc = (640.0 / w.max(h) as f64).min(1.0);
    let minv = invert_affine(&[m[0] as f64, m[1] as f64, m[2] as f64, m[3] as f64, m[4] as f64, m[5] as f64]);
    let s: Mat3 = [[sc, 0.0, -(x as f64) * sc], [0.0, sc, -(y as f64) * sc], [0.0, 0.0, 1.0]];
    let b: Mat3 = [[minv[0], minv[1], minv[2]], [minv[3], minv[4], minv[5]], [0.0, 0.0, 1.0]];
    let a = mm(&s, &b);
    let (dw, dh) = (((w as f64) * sc) as usize, ((h as f64) * sc) as usize);
    let back = warp_affine_u8(ig, &[a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2]], dw, dh, false);
    let x1 = ((x + w) as usize).min(rg.w);
    let y1 = ((y + h) as usize).min(rg.h);
    let refc = resize_area_size(&crop_u8(rg, x as usize, y as usize, x1, y1), dw, dh);
    if let Ok(d) = std::env::var("VS_FIT_DUMP") {
        let _ = back.save_png(std::path::Path::new(&format!("{}_back.png", d)));
        let _ = refc.save_png(std::path::Path::new(&format!("{}_ref.png", d)));
        eprintln!("fit A {:?} size {} {}", a, dw, dh);
    }
    corr_f32(&refc.data, &back.data).max(0.0)
}

/// (a - mean)·(b - mean) / (|a - mean| |b - mean|) in numpy float32.
fn corr_f32(a8: &[u8], b8: &[u8]) -> f64 {
    let mean_u8 = |v: &[u8]| v.iter().map(|&x| x as u64).sum::<u64>() as f64 / v.len().max(1) as f64;
    let (ma, mb) = (mean_u8(a8) as f32, mean_u8(b8) as f32);
    let a: Vec<f32> = a8.iter().map(|&v| v as f32 - ma).collect();
    let b: Vec<f32> = b8.iter().map(|&v| v as f32 - mb).collect();
    let d = np_norm_f32(&a) * np_norm_f32(&b);
    if (d as f64) <= 1e-9 {
        return 0.0;
    }
    let prod: Vec<f32> = a.iter().zip(&b).map(|(p, q)| p * q).collect();
    (np_sum_f32(&prod) / d) as f64
}

/// Align._refine_ecc_window
fn refine_ecc_window(p: &Params, rg: &Img, ig: &Img, m: &[f32; 6], rw: Win, iw: Win) -> Option<[f32; 6]> {
    let r = crop_u8(rg, rw.0, rw.1, rw.2, rw.3);
    let i = crop_u8(ig, iw.0, iw.1, iw.2, iw.3);
    let scale = (p.refine_size / (r.w.max(r.h).max(i.w.max(i.h))) as f64).min(1.0);
    let t_r: Mat3 = [[1.0, 0.0, rw.0 as f64], [0.0, 1.0, rw.1 as f64], [0.0, 0.0, 1.0]];
    let t_i: Mat3 = [[1.0, 0.0, -(iw.0 as f64)], [0.0, 1.0, -(iw.1 as f64)], [0.0, 0.0, 1.0]];
    let s: Mat3 = [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]];
    let sinv: Mat3 = [[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]];
    let mc = mm(&mm(&mm(&mm(&s, &t_i), &m3(m)), &t_r), &sinv);
    let mut warp = [mc[0][0] as f32, mc[0][1] as f32, mc[0][2] as f32, mc[1][0] as f32, mc[1][1] as f32, mc[1][2] as f32];
    let (r, i) = if scale < 1.0 { (resize_area_scale(&r, scale), resize_area_scale(&i, scale)) } else { (r, i) };
    ecc_euclidean(&r, &i, &mut warp, p.refine_iters, 1e-5)?;
    let mc: Mat3 = [[warp[0] as f64, warp[1] as f64, warp[2] as f64], [warp[3] as f64, warp[4] as f64, warp[5] as f64], [0.0, 0.0, 1.0]];
    let t_i_inv: Mat3 = [[1.0, 0.0, iw.0 as f64], [0.0, 1.0, iw.1 as f64], [0.0, 0.0, 1.0]];
    let t_r_inv: Mat3 = [[1.0, 0.0, -(rw.0 as f64)], [0.0, 1.0, -(rw.1 as f64)], [0.0, 0.0, 1.0]];
    let out = mm(&mm(&mm(&mm(&t_i_inv, &sinv), &mc), &s), &t_r_inv);
    let o = [out[0][0] as f32, out[0][1] as f32, out[0][2] as f32, out[1][0] as f32, out[1][1] as f32, out[1][2] as f32];
    if o.iter().all(|v| v.is_finite()) { Some(o) } else { None }
}

struct RefFeatures {
    key: (usize, usize, Win, usize, u64, u64),
    pts: Vec<[f64; 2]>,
    desc: Vec<[f32; 128]>,
}

fn ref_cache() -> &'static std::sync::Mutex<Vec<std::sync::Arc<RefFeatures>>> {
    static C: std::sync::OnceLock<std::sync::Mutex<Vec<std::sync::Arc<RefFeatures>>>> = std::sync::OnceLock::new();
    C.get_or_init(|| std::sync::Mutex::new(vec![]))
}

/// Align._features: (M, score); M None when the part was not found.
fn features(p: &Params, rg: &Img, ig: &Img, dbg: &mut serde_json::Map<String, serde_json::Value>) -> (Option<[f32; 6]>, f64, usize) {
    let n = p.n_features.max(1000);
    let (rw, iw) = windows(p, rg.w, rg.h);
    let long = (iw.2 - iw.0).max(iw.3 - iw.1);
    let sc = (p.feature_px / long.max(1) as f64).min(1.0);
    let detect = |g: &Img, win: Win| -> (Vec<[f64; 2]>, Vec<[f32; 128]>) {
        let mut crop = crop_u8(g, win.0, win.1, win.2, win.3);
        if sc < 1.0 {
            crop = resize_area_scale(&crop, sc);
        }
        let (kps, desc) = sift(&crop, n);
        let pts = kps.iter().map(|k| [k.x as f64 / sc + win.0 as f64, k.y as f64 / sc + win.1 as f64]).collect();
        (pts, desc)
    };
    let sum_key = {
        let mut s = 0u64;
        let mut y = 0;
        while y < rg.h {
            let mut x = 0;
            while x < rg.w {
                s += rg.data[y * rg.w + x] as u64;
                x += 89;
            }
            y += 97;
        }
        s
    };
    let key = (rg.w, rg.h, rw, n, sc.to_bits(), sum_key);
    let cached = ref_cache().lock().ok().and_then(|c| c.iter().find(|r| r.key == key).cloned());
    let rf = match cached {
        Some(r) => r,
        None => {
            let (pts, desc) = detect(rg, rw);
            let r = std::sync::Arc::new(RefFeatures { key, pts, desc });
            if let Ok(mut c) = ref_cache().lock() {
                if c.len() > 32 {
                    c.clear();
                }
                c.push(r.clone());
            }
            r
        }
    };
    let (k2, d2) = detect(ig, iw);
    dbg.insert("kp_ref".into(), rf.pts.len().into());
    dbg.insert("kp_img".into(), k2.len().into());
    if rf.pts.len() < 8 || k2.len() < 8 {
        return (None, 0.0, 0);
    }
    let pairs = knn2(&rf.desc, &d2);
    let good: Vec<(usize, usize)> = pairs.iter().enumerate().filter(|(_, p)| p[1].0 != usize::MAX && (p[0].1 as f64) < 0.75 * p[1].1 as f64).map(|(q, p)| (q, p[0].0)).collect();
    dbg.insert("good".into(), good.len().into());
    if good.len() < 8 {
        return (None, 0.0, 0);
    }
    let src: Vec<[f32; 2]> = good.iter().map(|&(q, _)| [rf.pts[q][0] as f32, rf.pts[q][1] as f32]).collect();
    let dst: Vec<[f32; 2]> = good.iter().map(|&(_, t)| [k2[t][0] as f32, k2[t][1] as f32]).collect();
    let Some((m, mask)) = estimate_affine_partial(&src, &dst, 4.0) else { return (None, 0.0, 0) };
    let inliers = mask.iter().map(|&v| v as usize).sum::<usize>();
    dbg.insert("inliers".into(), inliers.into());
    dbg.insert("ransac".into(), serde_json::json!(m));
    if inliers < p.min_inliers {
        return (None, 0.0, inliers);
    }
    let mut mf = [m[0] as f32, m[1] as f32, m[2] as f32, m[3] as f32, m[4] as f32, m[5] as f32];
    if p.refine {
        if let Some(refined) = refine_ecc_window(p, rg, ig, &mf, rw, iw) {
            let (fa, fb) = (fit_roi(p, rg, ig, &refined), fit_roi(p, rg, ig, &mf));
            dbg.insert("ecc".into(), serde_json::json!(refined));
            dbg.insert("fit_refined".into(), fa.into());
            dbg.insert("fit_raw".into(), fb.into());
            if fa >= fb {
                mf = refined;
            }
        }
    }
    let score = fit_roi(p, rg, ig, &mf);
    (Some(mf), score, inliers)
}

// ------------------------------------------------------- template search
/// Smallest 2^a 3^b 5^c at least n (FFT sizes that run fast).
fn good_size(n: usize) -> usize {
    let mut m = n.max(1);
    loop {
        let mut k = m;
        for p in [2, 3, 5] {
            while k % p == 0 {
                k /= p;
            }
        }
        if k == 1 {
            return m;
        }
        m += 1;
    }
}

/// In-place 2-D FFT of a w x h complex grid (rows, then columns).
fn fft2(buf: &mut [rustfft::num_complex::Complex<f32>], w: usize, h: usize, inverse: bool) {
    use rayon::prelude::*;
    use rustfft::{num_complex::Complex, FftPlanner};
    let mut planner = FftPlanner::<f32>::new();
    let (pw, ph) = if inverse { (planner.plan_fft_inverse(w), planner.plan_fft_inverse(h)) } else { (planner.plan_fft_forward(w), planner.plan_fft_forward(h)) };
    buf.par_chunks_mut(w).for_each(|row| pw.process(row));
    let mut t = vec![Complex::new(0f32, 0f32); w * h];
    t.par_chunks_mut(h).enumerate().for_each(|(x, col)| {
        for y in 0..h {
            col[y] = buf[y * w + x];
        }
        ph.process(col);
    });
    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            row[x] = t[x * h + y];
        }
    });
}

/// cv::matchTemplate(img, tpl, TM_CCOEFF_NORMED): the correlation with the
/// mean-free template (by FFT, in tiles), window sums from integral images,
/// and OpenCV's normalisation, clamps and float result.
fn ccoeff_map(img: &Img, tpl: &Img) -> crate::ncc::ScoreMap {
    if let Some((w, h, d)) = crate::cvlib::match_template(img, tpl, crate::cvlib::TM_CCOEFF_NORMED) {
        return crate::ncc::ScoreMap { w, h, d };
    }
    use rustfft::num_complex::Complex;
    let (iw, ih, tw, th) = (img.w, img.h, tpl.w, tpl.h);
    if tw > iw || th > ih || tw == 0 || th == 0 {
        return crate::ncc::ScoreMap { w: 0, h: 0, d: vec![] };
    }
    let (ow, oh) = (iw - tw + 1, ih - th + 1);
    let n = (tw * th) as f64;
    let tsum: f64 = tpl.data.iter().map(|&v| v as f64).sum();
    let tsq: f64 = tpl.data.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let tmean = tsum / n;
    let tvar = (tsq / n - tmean * tmean).max(0.0);
    if tvar.sqrt() < f64::EPSILON {
        return crate::ncc::ScoreMap { w: ow, h: oh, d: vec![1.0; ow * oh] };
    }
    let templ_norm = tvar.sqrt() / (1.0 / n).sqrt();
    // integral images of the picture and its square
    let (sw, _sh) = (iw + 1, ih + 1);
    let mut s1 = vec![0f64; sw * (ih + 1)];
    let mut s2 = vec![0f64; sw * (ih + 1)];
    for y in 0..ih {
        let (mut r1, mut r2) = (0f64, 0f64);
        for x in 0..iw {
            let v = img.data[y * iw + x] as f64;
            r1 += v;
            r2 += v * v;
            s1[(y + 1) * sw + x + 1] = s1[y * sw + x + 1] + r1;
            s2[(y + 1) * sw + x + 1] = s2[y * sw + x + 1] + r2;
        }
    }
    // FFT tiles: about twice the template, never more than the picture
    let fw = good_size((2 * tw).max(256)).min(good_size(iw));
    let fh = good_size((2 * th).max(256)).min(good_size(ih));
    let (tile_w, tile_h) = (fw - tw + 1, fh - th + 1);
    let mut tspec = vec![Complex::new(0f32, 0f32); fw * fh];
    for y in 0..th {
        for x in 0..tw {
            tspec[y * fw + x] = Complex::new((tpl.data[y * tw + x] as f64 - tmean) as f32, 0.0);
        }
    }
    fft2(&mut tspec, fw, fh, false);
    let mut out = vec![0f32; ow * oh];
    let scale = 1.0 / (fw * fh) as f64;
    let mut ty = 0;
    while ty < oh {
        let mut tx = 0;
        while tx < ow {
            let mut buf = vec![Complex::new(0f32, 0f32); fw * fh];
            for y in 0..fh.min(ih - ty) {
                for x in 0..fw.min(iw - tx) {
                    buf[y * fw + x] = Complex::new(img.data[(ty + y) * iw + tx + x] as f32, 0.0);
                }
            }
            fft2(&mut buf, fw, fh, false);
            for (b, t) in buf.iter_mut().zip(&tspec) {
                *b *= t.conj();
            }
            fft2(&mut buf, fw, fh, true);
            for y in 0..tile_h.min(oh - ty) {
                for x in 0..tile_w.min(ow - tx) {
                    let (ox, oy) = (tx + x, ty + y);
                    let num = buf[y * fw + x].re as f64 * scale;
                    let at = |s: &[f64], xx: usize, yy: usize| s[yy * sw + xx];
                    let wsum = at(&s1, ox + tw, oy + th) - at(&s1, ox, oy + th) - at(&s1, ox + tw, oy) + at(&s1, ox, oy);
                    let wsq = at(&s2, ox + tw, oy + th) - at(&s2, ox, oy + th) - at(&s2, ox + tw, oy) + at(&s2, ox, oy);
                    let wmean2 = wsum * wsum / n;
                    let t = (wsq - wmean2).max(0.0).sqrt() * templ_norm;
                    let v = if num.abs() < t {
                        num / t
                    } else if num.abs() < t * 1.125 {
                        if num > 0.0 { 1.0 } else { -1.0 }
                    } else {
                        0.0
                    };
                    out[oy * ow + ox] = v as f32;
                }
            }
            tx += tile_w;
        }
        ty += tile_h;
    }
    crate::ncc::ScoreMap { w: ow, h: oh, d: out }
}

/// Align._search: (best location (subpixel), score, angle, margin)
fn search(tpl: &Img, ig: &Img, angles: &[f64]) -> (Option<(f64, f64)>, f64, f64, Option<f64>) {
    let (w, h) = (tpl.w, tpl.h);
    let (mut best_loc, mut best_score, mut best_ang, mut margin) = (None, -1.0f64, 0.0f64, None);
    for &a in angles {
        let t = if a.abs() < 1e-9 {
            tpl.clone()
        } else {
            // getRotationMatrix2D((w/2, h/2), a, 1)
            let ang = a * std::f64::consts::PI / 180.0;
            let (al, be) = (ang.cos(), ang.sin());
            let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
            let r = [al, be, (1.0 - al) * cx - be * cy, -be, al, be * cx + (1.0 - al) * cy];
            warp_affine_u8(tpl, &r, w, h, true)
        };
        if ig.h < h || ig.w < w {
            continue;
        }
        let res = ccoeff_map(ig, &t);
        if res.d.is_empty() {
            continue;
        }
        let (lx, ly, mx) = res.best();
        let mx = mx as f64;
        if mx > best_score {
            best_score = mx;
            best_ang = a;
            best_loc = Some(subpixel(&res, lx, ly));
            let x0 = lx.saturating_sub(w / 2);
            let y0 = ly.saturating_sub(h / 2);
            let (x1, y1) = ((lx + w / 2).min(res.w), (ly + h / 2).min(res.h));
            let mut rival = f32::MIN;
            for yy in 0..res.h {
                for xx in 0..res.w {
                    let v = if yy >= y0 && yy < y1 && xx >= x0 && xx < x1 { -1.0 } else { res.d[yy * res.w + xx] };
                    rival = rival.max(v);
                }
            }
            margin = Some(mx - rival as f64);
        }
    }
    (best_loc, best_score, best_ang, margin)
}

fn subpixel(res: &crate::ncc::ScoreMap, px: usize, py: usize) -> (f64, f64) {
    let at = |x: usize, y: usize| res.d[y * res.w + x] as f64;
    let (mut fx, mut fy) = (px as f64, py as f64);
    if px > 0 && px + 1 < res.w {
        let (l, c, r) = (at(px - 1, py), at(px, py), at(px + 1, py));
        let d = l - 2.0 * c + r;
        if d.abs() > 1e-12 {
            fx = px as f64 + (0.5 * (l - r) / d).clamp(-1.0, 1.0);
        }
    }
    if py > 0 && py + 1 < res.h {
        let (u, c, dn) = (at(px, py - 1), at(px, py), at(px, py + 1));
        let d = u - 2.0 * c + dn;
        if d.abs() > 1e-12 {
            fy = py as f64 + (0.5 * (u - dn) / d).clamp(-1.0, 1.0);
        }
    }
    (fx, fy)
}

fn arange(lo: f64, hi: f64, step: f64) -> Vec<f64> {
    let n = ((hi - lo) / step).ceil().max(0.0) as usize;
    (0..n).map(|i| lo + i as f64 * step).collect()
}

/// Align._fit: whole-picture correlation of the reference warped by M.
fn fit_whole(rg: &Img, ig: &Img, m: &[f32; 6]) -> f64 {
    let sc = (240.0 / ig.h.max(ig.w) as f64).min(1.0);
    let scf = sc as f32;
    let a = [m[0], m[1], m[2] * scf, m[3], m[4], m[5] * scf];
    let rs = resize_area_scale(rg, sc);
    let isg = resize_area_scale(ig, sc);
    let warped = warp_affine_u8(&rs, &a.map(|v| v as f64), isg.w, isg.h, false);
    let mut av = vec![];
    let mut bv = vec![];
    for k in 0..warped.data.len() {
        if warped.data[k] > 0 {
            av.push(warped.data[k] as f32);
            bv.push(isg.data[k] as f32);
        }
    }
    if av.len() < 50 {
        return -1.0;
    }
    let ma = np_sum_f32(&av) / av.len() as f32;
    let mb = np_sum_f32(&bv) / bv.len() as f32;
    av.iter_mut().for_each(|v| *v -= ma);
    bv.iter_mut().for_each(|v| *v -= mb);
    let d = np_norm_f32(&av) * np_norm_f32(&bv);
    if (d as f64) <= 1e-9 {
        return -1.0;
    }
    (np_dot_f32(&av, &bv) / d) as f64
}

/// Align._refine_ecc (whole picture, for the template method)
fn refine_ecc_whole(p: &Params, rg: &Img, ig: &Img, m: &[f32; 6]) -> Option<[f32; 6]> {
    let scale = (p.refine_size / rg.w.max(rg.h) as f64).min(1.0);
    let mut warp = *m;
    let (r, i) = if scale < 1.0 {
        warp[2] *= scale as f32;
        warp[5] *= scale as f32;
        (resize_area_scale(rg, scale), resize_area_scale(ig, scale))
    } else {
        (rg.clone(), ig.clone())
    };
    ecc_euclidean(&r, &i, &mut warp, p.refine_iters, 1e-5)?;
    if scale < 1.0 {
        warp[2] /= scale as f32;
        warp[5] /= scale as f32;
    }
    if warp.iter().all(|v| v.is_finite()) { Some(warp) } else { None }
}

/// Align._template: (M, score, margin)
fn template(p: &Params, rg: &Img, ig: &Img) -> (Option<[f32; 6]>, f64, Option<f64>) {
    let (hh, ww) = (rg.h as i64, rg.w as i64);
    let roi = p.roi.unwrap_or([0.0, 0.0, ww as f64, hh as f64]);
    let (x, y) = ((roi[0] as i64).max(0), (roi[1] as i64).max(0));
    let w = 8.max((roi[2].min((ww - x) as f64)) as i64) as usize;
    let h = 8.max((roi[3].min((hh - y) as f64)) as i64) as usize;
    let tpl = crop_u8(rg, x as usize, y as usize, (x as usize + w).min(rg.w), (y as usize + h).min(rg.h));
    let (lo, hi) = p.angle_range;
    let fine_step = if p.angle_step != 0.0 { p.angle_step } else { 1.0 };
    let best;
    if (hi - lo).abs() < 1e-9 {
        best = search(&tpl, ig, &[if lo == 0.0 { 0.0 } else { lo }]);
    } else if p.pyramid {
        let sc = 0.5;
        let tpl_s = resize_area_scale(&tpl, sc);
        let img_s = resize_area_scale(ig, sc);
        let coarse_step = fine_step.max((hi - lo) / 16.0);
        let coarse = arange(lo, hi + 1e-9, coarse_step);
        let a0 = if tpl_s.w.min(tpl_s.h) >= 8 {
            let c = search(&tpl_s, &img_s, &coarse);
            if c.0.is_some() { c.2 } else { 0.0 }
        } else {
            0.0
        };
        let fine: Vec<f64> = arange(a0 - coarse_step, a0 + coarse_step + 1e-9, fine_step).into_iter()
            .filter(|&a| lo - 1e-9 <= a && a <= hi + 1e-9).collect();
        best = search(&tpl, ig, if fine.is_empty() { std::slice::from_ref(&a0) } else { &fine });
    } else {
        let coarse_step = fine_step.max((hi - lo) / 16.0);
        let coarse = arange(lo, hi + 1e-9, coarse_step);
        let mut b = search(&tpl, ig, &coarse);
        if b.0.is_some() {
            let a0 = b.2;
            let fine: Vec<f64> = arange(a0 - coarse_step, a0 + coarse_step + 1e-9, fine_step).into_iter()
                .filter(|&a| lo - 1e-9 <= a && a <= hi + 1e-9).collect();
            let cand = search(&tpl, ig, &fine);
            if cand.1 > b.1 {
                b = cand;
            }
        }
        best = b;
    }
    let (loc, score, angle, margin) = best;
    let Some(loc) = loc else { return (None, 0.0, None) };
    let (cx_ref, cy_ref) = (x as f64 + w as f64 / 2.0, y as f64 + h as f64 / 2.0);
    let (cx_cur, cy_cur) = (loc.0 + w as f64 / 2.0, loc.1 + h as f64 / 2.0);
    let rad = angle.to_radians();
    let (ca, sa) = (rad.cos(), rad.sin());
    let mut m = [ca as f32, sa as f32, (cx_cur - (ca * cx_ref + sa * cy_ref)) as f32,
        (-sa) as f32, ca as f32, (cy_cur - (-sa * cx_ref + ca * cy_ref)) as f32];
    if p.refine {
        if let Some(refined) = refine_ecc_whole(p, rg, ig, &m) {
            if fit_whole(rg, ig, &refined) >= fit_whole(rg, ig, &m) {
                m = refined;
            }
        }
    }
    (Some(m), score, margin)
}

/// Align.run's choice of method and result.
pub fn locate(p: &Params, rg: &Img, ig: &Img) -> Found {
    let mut dbg = serde_json::Map::new();
    let mut f = Found::default();
    match p.method.as_str() {
        "auto" => {
            let (m, score, inl) = features(p, rg, ig, &mut dbg);
            f.m = m;
            f.score = score;
            f.inliers = inl;
            f.method = "features";
            if m.is_none() || score < p.min_score {
                let (m2, s2, margin) = template(p, rg, ig);
                dbg.insert("template_score".into(), s2.into());
                if m2.is_some() && s2 > (if m.is_some() { score } else { -1.0 }) {
                    f.m = m2;
                    f.score = s2;
                    f.method = "template";
                    f.margin = margin;
                }
            }
        }
        "features" => {
            let (m, score, inl) = features(p, rg, ig, &mut dbg);
            f.m = m;
            f.score = score;
            f.inliers = inl;
            f.method = "features";
        }
        _ => {
            let (m, score, margin) = template(p, rg, ig);
            f.m = m;
            f.score = score;
            f.margin = margin;
            f.method = "template";
        }
    }
    f.debug = serde_json::Value::Object(dbg);
    f
}

impl Params {
    /// The check's settings, with Vision Station's defaults.
    pub fn from_spec(t: &crate::model::ToolSpec) -> Params {
        let rng = t.p.get("angle_range").and_then(|v| v.as_array()).map(|a| {
            (a.first().and_then(|v| v.as_f64()).unwrap_or(0.0), a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0))
        }).unwrap_or((0.0, 0.0));
        Params {
            roi: t.roi(),
            method: t.s("method", "template"),
            min_score: t.f("min_score", 0.6),
            n_features: t.i("n_features", 2000).max(0) as usize,
            min_inliers: t.i("min_inliers", 12).max(0) as usize,
            feature_px: t.f("feature_px", 1400.0),
            search_pad_px: t.p.get("search_pad_px").and_then(|v| v.as_f64()),
            refine: t.b("refine", true),
            refine_size: t.f("refine_size", 420.0),
            refine_iters: t.i("refine_iters", 30).max(0) as usize,
            angle_range: rng,
            angle_step: t.f("angle_step", 1.0),
            pyramid: t.b("pyramid", true),
        }
    }
}
