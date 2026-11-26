//! Images: one plain struct, 8-bit, 1 (grey) or 3 (RGB) channels, rows packed.
//! Frames are shared as `Arc<Img>` so the camera, the engine, every check and
//! the screen read the same bytes; nothing is copied unless a check needs its
//! own patch.

use rayon::prelude::*;
use std::io::Cursor;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Img {
    pub w: usize,
    pub h: usize,
    pub c: usize,
    pub data: Vec<u8>,
}

/// A float plane, for maths that needs sign or precision.
#[derive(Clone, Debug)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub d: Vec<f32>,
}

impl Plane {
    pub fn new(w: usize, h: usize) -> Plane {
        Plane { w, h, d: vec![0.0; w * h] }
    }
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.d[y * self.w + x]
    }
    pub fn from_gray(g: &Img) -> Plane {
        let g = if g.c == 1 { std::borrow::Cow::Borrowed(g) } else { std::borrow::Cow::Owned(g.gray()) };
        Plane { w: g.w, h: g.h, d: g.data.iter().map(|&v| v as f32).collect() }
    }
    /// Bilinear sample, edge clamped.
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let xf = x.clamp(0.0, (self.w - 1) as f32);
        let yf = y.clamp(0.0, (self.h - 1) as f32);
        let x0 = xf.floor() as usize;
        let y0 = yf.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let fx = xf - x0 as f32;
        let fy = yf - y0 as f32;
        let a = self.d[y0 * self.w + x0];
        let b = self.d[y0 * self.w + x1];
        let c = self.d[y1 * self.w + x0];
        let d = self.d[y1 * self.w + x1];
        a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy
    }
    /// Mean filter of a (2r+1) square, through an integral image.
    pub fn box_blur(&self, r: usize) -> Plane {
        let (w, h) = (self.w, self.h);
        let ii = integral(&self.d, w, h);
        let mut out = Plane::new(w, h);
        out.d.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let y0 = y.saturating_sub(r);
            let y1 = (y + r + 1).min(h);
            for (x, o) in row.iter_mut().enumerate() {
                let x0 = x.saturating_sub(r);
                let x1 = (x + r + 1).min(w);
                let s = rect_sum(&ii, w, x0, y0, x1, y1);
                *o = (s / ((x1 - x0) * (y1 - y0)) as f64) as f32;
            }
        });
        out
    }
}

/// Summed-area table, (w+1) x (h+1), f64.
pub fn integral(d: &[f32], w: usize, h: usize) -> Vec<f64> {
    let mut ii = vec![0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut run = 0f64;
        for x in 0..w {
            run += d[y * w + x] as f64;
            ii[(y + 1) * (w + 1) + x + 1] = ii[y * (w + 1) + x + 1] + run;
        }
    }
    ii
}

#[inline]
pub fn rect_sum(ii: &[f64], w: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
    let s = w + 1;
    ii[y1 * s + x1] - ii[y0 * s + x1] - ii[y1 * s + x0] + ii[y0 * s + x0]
}

impl Img {
    pub fn new(w: usize, h: usize, c: usize) -> Img {
        Img { w, h, c, data: vec![0; w * h * c] }
    }

    pub fn load(path: &Path) -> Result<Img, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        Img::decode(&bytes).map_err(|e| format!("{}: {}", path.display(), e))
    }

    pub fn decode(bytes: &[u8]) -> Result<Img, String> {
        let di = image::load_from_memory(bytes).map_err(|e| e.to_string())?;
        Ok(match di {
            image::DynamicImage::ImageLuma8(g) => {
                let (w, h) = g.dimensions();
                Img { w: w as usize, h: h as usize, c: 1, data: g.into_raw() }
            }
            other => {
                let rgb = other.to_rgb8();
                let (w, h) = rgb.dimensions();
                Img { w: w as usize, h: h as usize, c: 3, data: rgb.into_raw() }
            }
        })
    }

    pub fn save_png(&self, path: &Path) -> Result<(), String> {
        let ct = if self.c == 1 { image::ExtendedColorType::L8 } else { image::ExtendedColorType::Rgb8 };
        image::save_buffer(path, &self.data, self.w as u32, self.h as u32, ct).map_err(|e| e.to_string())
    }

    /// JPEG for the screen, shrunk so its width is at most `max_w`.
    pub fn to_jpeg(&self, max_w: usize, quality: u8) -> Vec<u8> {
        let small;
        let src = if max_w > 0 && self.w > max_w {
            let nh = ((self.h as f64) * (max_w as f64) / (self.w as f64)).round().max(1.0) as usize;
            small = self.resize(max_w, nh);
            &small
        } else {
            self
        };
        let mut out = Vec::with_capacity(src.w * src.h / 4);
        let ct = if src.c == 1 { image::ExtendedColorType::L8 } else { image::ExtendedColorType::Rgb8 };
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(Cursor::new(&mut out), quality);
        let _ = enc.encode(&src.data, src.w as u32, src.h as u32, ct);
        out
    }

    #[inline]
    pub fn px(&self, x: usize, y: usize, ch: usize) -> u8 {
        self.data[(y * self.w + x) * self.c + ch.min(self.c - 1)]
    }

    #[inline]
    pub fn rgb(&self, x: usize, y: usize) -> [u8; 3] {
        let i = (y * self.w + x) * self.c;
        if self.c == 1 {
            let v = self.data[i];
            [v, v, v]
        } else {
            [self.data[i], self.data[i + 1], self.data[i + 2]]
        }
    }

    pub fn gray(&self) -> Img {
        if self.c == 1 {
            return self.clone();
        }
        let mut out = Img::new(self.w, self.h, 1);
        out.data.par_chunks_mut(self.w.max(1)).zip(self.data.par_chunks(self.w.max(1) * 3)).for_each(|(o, s)| {
            for (i, v) in o.iter_mut().enumerate() {
                let r = s[i * 3] as u32;
                let g = s[i * 3 + 1] as u32;
                let b = s[i * 3 + 2] as u32;
                // cv2.cvtColor(BGR2GRAY): 15-bit weights, rounded (OpenCV 4.10, exact)
                *v = ((r * 9798 + g * 19235 + b * 3735 + (1 << 14)) >> 15) as u8;
            }
        });
        out
    }

    pub fn to_rgb(&self) -> Img {
        if self.c == 3 {
            return self.clone();
        }
        let mut out = Img::new(self.w, self.h, 3);
        for (i, &v) in self.data.iter().enumerate() {
            out.data[i * 3] = v;
            out.data[i * 3 + 1] = v;
            out.data[i * 3 + 2] = v;
        }
        out
    }

    /// One channel as grey: 0 red, 1 green, 2 blue.
    pub fn channel(&self, ch: usize) -> Img {
        if self.c == 1 {
            return self.clone();
        }
        let mut out = Img::new(self.w, self.h, 1);
        for i in 0..self.w * self.h {
            out.data[i] = self.data[i * 3 + ch.min(2)];
        }
        out
    }

    /// Bilinear sample of channel `ch`, edge clamped.
    #[inline]
    pub fn sample(&self, x: f32, y: f32, ch: usize) -> f32 {
        let xf = x.clamp(0.0, (self.w - 1) as f32);
        let yf = y.clamp(0.0, (self.h - 1) as f32);
        let x0 = xf.floor() as usize;
        let y0 = yf.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let fx = xf - x0 as f32;
        let fy = yf - y0 as f32;
        let c = self.c;
        let a = self.data[(y0 * self.w + x0) * c + ch] as f32;
        let b = self.data[(y0 * self.w + x1) * c + ch] as f32;
        let cc = self.data[(y1 * self.w + x0) * c + ch] as f32;
        let d = self.data[(y1 * self.w + x1) * c + ch] as f32;
        a + (b - a) * fx + (cc - a) * fy + (a - b - cc + d) * fx * fy
    }

    /// Plain crop, clamped to the image (edge replicated outside).
    pub fn crop(&self, x: i64, y: i64, w: usize, h: usize) -> Img {
        let mut out = Img::new(w, h, self.c);
        let c = self.c;
        let inside = x >= 0 && y >= 0 && (x as usize + w) <= self.w && (y as usize + h) <= self.h;
        for yy in 0..h {
            if inside {
                let sy = y as usize + yy;
                let s = (sy * self.w + x as usize) * c;
                out.data[yy * w * c..(yy + 1) * w * c].copy_from_slice(&self.data[s..s + w * c]);
            } else {
                let sy = (y + yy as i64).clamp(0, self.h as i64 - 1) as usize;
                for xx in 0..w {
                    let sx = (x + xx as i64).clamp(0, self.w as i64 - 1) as usize;
                    for k in 0..c {
                        out.data[(yy * w + xx) * c + k] = self.data[(sy * self.w + sx) * c + k];
                    }
                }
            }
        }
        out
    }

    /// Resize: box average when shrinking, bilinear when growing.
    pub fn resize(&self, nw: usize, nh: usize) -> Img {
        let nw = nw.max(1);
        let nh = nh.max(1);
        let mut out = Img::new(nw, nh, self.c);
        let c = self.c;
        let sx = self.w as f32 / nw as f32;
        let sy = self.h as f32 / nh as f32;
        if sx >= 1.0 && sy >= 1.0 {
            out.data.par_chunks_mut(nw * c).enumerate().for_each(|(y, row)| {
                let y0 = (y as f32 * sy) as usize;
                let y1 = (((y + 1) as f32 * sy) as usize).max(y0 + 1).min(self.h);
                for x in 0..nw {
                    let x0 = (x as f32 * sx) as usize;
                    let x1 = (((x + 1) as f32 * sx) as usize).max(x0 + 1).min(self.w);
                    let n = ((x1 - x0) * (y1 - y0)) as u32;
                    for k in 0..c {
                        let mut s = 0u32;
                        for yy in y0..y1 {
                            let base = (yy * self.w) * c + k;
                            for xx in x0..x1 {
                                s += self.data[base + xx * c] as u32;
                            }
                        }
                        row[x * c + k] = ((s + n / 2) / n) as u8;
                    }
                }
            });
        } else {
            out.data.par_chunks_mut(nw * c).enumerate().for_each(|(y, row)| {
                let fy = (y as f32 + 0.5) * sy - 0.5;
                for x in 0..nw {
                    let fx = (x as f32 + 0.5) * sx - 0.5;
                    for k in 0..c {
                        row[x * c + k] = self.sample(fx, fy, k).round().clamp(0.0, 255.0) as u8;
                    }
                }
            });
        }
        out
    }

    /// Rotate by quarter turns anticlockwise (numpy rot90 semantics).
    pub fn rot90(&self, k: usize) -> Img {
        let k = k % 4;
        if k == 0 {
            return self.clone();
        }
        let (w, h, c) = (self.w, self.h, self.c);
        let (nw, nh) = if k % 2 == 1 { (h, w) } else { (w, h) };
        let mut out = Img::new(nw, nh, c);
        for y in 0..h {
            for x in 0..w {
                let (nx, ny) = match k {
                    1 => (y, w - 1 - x),
                    2 => (w - 1 - x, h - 1 - y),
                    _ => (h - 1 - y, x),
                };
                let s = (y * w + x) * c;
                let d = (ny * nw + nx) * c;
                out.data[d..d + c].copy_from_slice(&self.data[s..s + c]);
            }
        }
        out
    }

    /// Paste `patch` at (x, y), clipped.
    pub fn paste(&mut self, patch: &Img, x: i64, y: i64) {
        let c = self.c;
        for yy in 0..patch.h {
            let ty = y + yy as i64;
            if ty < 0 || ty >= self.h as i64 {
                continue;
            }
            for xx in 0..patch.w {
                let tx = x + xx as i64;
                if tx < 0 || tx >= self.w as i64 {
                    continue;
                }
                let p = patch.rgb(xx, yy);
                let d = (ty as usize * self.w + tx as usize) * c;
                if c == 1 {
                    self.data[d] = if patch.c == 1 { p[0] } else { ((p[0] as u32 * 77 + p[1] as u32 * 150 + p[2] as u32 * 29) >> 8) as u8 };
                } else {
                    self.data[d..d + 3].copy_from_slice(&p);
                }
            }
        }
    }
}

/// Bayer to RGB, bilinear. `pattern`: "RG", "BG", "GB", "GR" (the first row's
/// first two colours).
pub fn demosaic(raw: &[u8], w: usize, h: usize, stride: usize, pattern: &str) -> Img {
    let mut out = Img::new(w, h, 3);
    // colour at (x, y): 0 R, 1 G, 2 B
    let (r_x, r_y) = match pattern {
        "RG" => (0, 0),
        "GR" => (1, 0),
        "GB" => (0, 1),
        _ => (1, 1), // BG
    };
    let colour = |x: usize, y: usize| -> usize {
        let xo = (x + r_x) & 1;
        let yo = (y + r_y) & 1;
        match (xo, yo) {
            (0, 0) => 0,
            (1, 1) => 2,
            _ => 1,
        }
    };
    let get = |x: i64, y: i64| -> u32 {
        let xx = x.clamp(0, w as i64 - 1) as usize;
        let yy = y.clamp(0, h as i64 - 1) as usize;
        raw[yy * stride + xx] as u32
    };
    out.data.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let yi = y as i64;
        for x in 0..w {
            let xi = x as i64;
            let v = get(xi, yi);
            let cross = (get(xi - 1, yi) + get(xi + 1, yi) + get(xi, yi - 1) + get(xi, yi + 1) + 2) / 4;
            let diag = (get(xi - 1, yi - 1) + get(xi + 1, yi - 1) + get(xi - 1, yi + 1) + get(xi + 1, yi + 1) + 2) / 4;
            let horiz = (get(xi - 1, yi) + get(xi + 1, yi) + 1) / 2;
            let vert = (get(xi, yi - 1) + get(xi, yi + 1) + 1) / 2;
            let (r, g, b) = match colour(x, y) {
                0 => (v, cross, diag),
                2 => (diag, cross, v),
                _ => {
                    // green: red shares this row when the row's parity is red's
                    if ((y + r_y) & 1) == 0 { (horiz, v, vert) } else { (vert, v, horiz) }
                }
            };
            row[x * 3] = r.min(255) as u8;
            row[x * 3 + 1] = g.min(255) as u8;
            row[x * 3 + 2] = b.min(255) as u8;
        }
    });
    out
}

/// RGB (0-255) to HSV the OpenCV way: H 0-179, S and V 0-255.
#[inline]
pub fn hsv(p: [u8; 3]) -> (f32, f32, f32) {
    let r = p[0] as f32;
    let g = p[1] as f32;
    let b = p[2] as f32;
    let v = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let d = v - mn;
    let s = if v > 0.0 { 255.0 * d / v } else { 0.0 };
    let mut h = if d == 0.0 {
        0.0
    } else if v == r {
        60.0 * (g - b) / d
    } else if v == g {
        120.0 + 60.0 * (b - r) / d
    } else {
        240.0 + 60.0 * (r - g) / d
    };
    if h < 0.0 {
        h += 360.0;
    }
    (h / 2.0, s, v)
}

/// sRGB (0-255) to CIE Lab (D65).
pub fn lab(p: [f32; 3]) -> [f32; 3] {
    fn lin(c: f32) -> f32 {
        let c = c / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    }
    let (r, g, b) = (lin(p[0]), lin(p[1]), lin(p[2]));
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    fn f(t: f32) -> f32 {
        if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 }
    }
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

pub fn delta_e(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Otsu threshold of a grey histogram.
pub fn otsu(data: &[u8]) -> u8 {
    let mut hist = [0u64; 256];
    for &v in data {
        hist[v as usize] += 1;
    }
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 128;
    }
    let sum: f64 = hist.iter().enumerate().map(|(i, &n)| i as f64 * n as f64).sum();
    let (mut wb, mut sb, mut best, mut thr) = (0f64, 0f64, -1f64, 0u8);
    for t in 0..256 {
        wb += hist[t] as f64;
        if wb == 0.0 {
            continue;
        }
        let wf = total as f64 - wb;
        if wf == 0.0 {
            break;
        }
        sb += t as f64 * hist[t] as f64;
        let mb = sb / wb;
        let mf = (sum - sb) / wf;
        let between = wb * wf * (mb - mf) * (mb - mf);
        if between > best {
            best = between;
            thr = t as u8;
        }
    }
    thr
}

/// Binary morphology on a 0/255 mask with a k x k square (separable).
pub fn morph(mask: &[u8], w: usize, h: usize, k: usize, dilate: bool) -> Vec<u8> {
    if k <= 1 {
        return mask.to_vec();
    }
    let r = k / 2;
    let pick = |a: u8, b: u8| if dilate { a.max(b) } else { a.min(b) };
    let init = if dilate { 0u8 } else { 255u8 };
    let mut tmp = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut v = init;
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            for xx in x0..=x1 {
                v = pick(v, mask[y * w + xx]);
            }
            tmp[y * w + x] = v;
        }
    }
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r).min(h - 1);
        for x in 0..w {
            let mut v = init;
            for yy in y0..=y1 {
                v = pick(v, tmp[yy * w + x]);
            }
            out[y * w + x] = v;
        }
    }
    out
}

pub fn open(mask: &[u8], w: usize, h: usize, k: usize) -> Vec<u8> {
    let e = morph(mask, w, h, k, false);
    morph(&e, w, h, k, true)
}

pub fn close(mask: &[u8], w: usize, h: usize, k: usize) -> Vec<u8> {
    let d = morph(mask, w, h, k, true);
    morph(&d, w, h, k, false)
}

/// A connected region of a mask.
#[derive(Clone, Debug, Default)]
pub struct Blob {
    pub area: usize,
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    pub cx: f64,
    pub cy: f64,
    /// second moments about the centroid
    pub mxx: f64,
    pub myy: f64,
    pub mxy: f64,
    /// pixel indices (y * w + x), kept when asked for
    pub pixels: Vec<u32>,
}

impl Blob {
    /// Major / minor axis ratio from the second moments.
    pub fn elongation(&self) -> f64 {
        let a = self.mxx;
        let b = self.myy;
        let c = self.mxy;
        let t = ((a - b) * (a - b) + 4.0 * c * c).sqrt();
        let l1 = (a + b + t) / 2.0;
        let l2 = ((a + b - t) / 2.0).max(1e-9);
        (l1 / l2).sqrt()
    }
}

/// 8-connected components of a non-zero mask.
pub fn components(mask: &[u8], w: usize, h: usize, keep_pixels: bool) -> Vec<Blob> {
    let mut label = vec![0u32; w * h];
    let mut parent: Vec<u32> = vec![0];
    fn find(p: &mut Vec<u32>, mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if mask[i] == 0 {
                continue;
            }
            let mut ns = [0u32; 4];
            let mut n = 0;
            if x > 0 && label[i - 1] != 0 {
                ns[n] = label[i - 1];
                n += 1;
            }
            if y > 0 {
                let up = i - w;
                if label[up] != 0 {
                    ns[n] = label[up];
                    n += 1;
                }
                if x > 0 && label[up - 1] != 0 {
                    ns[n] = label[up - 1];
                    n += 1;
                }
                if x + 1 < w && label[up + 1] != 0 {
                    ns[n] = label[up + 1];
                    n += 1;
                }
            }
            if n == 0 {
                let l = parent.len() as u32;
                parent.push(l);
                label[i] = l;
            } else {
                let mut m = find(&mut parent, ns[0]);
                for &l in &ns[1..n] {
                    let r = find(&mut parent, l);
                    if r != m {
                        let (a, b) = if r < m { (r, m) } else { (m, r) };
                        parent[b as usize] = a;
                        m = a;
                    }
                }
                label[i] = m;
            }
        }
    }
    let mut index = vec![u32::MAX; parent.len()];
    let mut blobs: Vec<Blob> = Vec::new();
    let mut sums: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if label[i] == 0 {
                continue;
            }
            let r = find(&mut parent, label[i]) as usize;
            if index[r] == u32::MAX {
                index[r] = blobs.len() as u32;
                blobs.push(Blob { x0: x, y0: y, x1: x, y1: y, ..Default::default() });
                sums.push((0.0, 0.0, 0.0, 0.0, 0.0));
            }
            let k = index[r] as usize;
            let b = &mut blobs[k];
            b.area += 1;
            b.x0 = b.x0.min(x);
            b.x1 = b.x1.max(x);
            b.y0 = b.y0.min(y);
            b.y1 = b.y1.max(y);
            if keep_pixels {
                b.pixels.push(i as u32);
            }
            let s = &mut sums[k];
            let (xf, yf) = (x as f64, y as f64);
            s.0 += xf;
            s.1 += yf;
            s.2 += xf * xf;
            s.3 += yf * yf;
            s.4 += xf * yf;
        }
    }
    for (b, s) in blobs.iter_mut().zip(sums) {
        let n = b.area as f64;
        b.cx = s.0 / n;
        b.cy = s.1 / n;
        b.mxx = s.2 / n - b.cx * b.cx;
        b.myy = s.3 / n - b.cy * b.cy;
        b.mxy = s.4 / n - b.cx * b.cy;
    }
    blobs
}
