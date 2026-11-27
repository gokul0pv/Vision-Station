//! 2-D affine transforms and the region a check reads.

use crate::img::{Img, Plane};
use rayon::prelude::*;

/// x' = a x + b y + tx ; y' = c x + d y + ty
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    pub a: f64,
    pub b: f64,
    pub tx: f64,
    pub c: f64,
    pub d: f64,
    pub ty: f64,
}

impl Affine {
    pub const IDENTITY: Affine = Affine { a: 1.0, b: 0.0, tx: 0.0, c: 0.0, d: 1.0, ty: 0.0 };

    pub fn translate(x: f64, y: f64) -> Affine {
        Affine { tx: x, ty: y, ..Affine::IDENTITY }
    }
    /// Rotation by `deg` (image coordinates, y down: positive turns clockwise
    /// on screen) about (cx, cy).
    pub fn rotate_about(deg: f64, cx: f64, cy: f64) -> Affine {
        let r = deg.to_radians();
        let (s, c) = r.sin_cos();
        Affine { a: c, b: -s, tx: cx - c * cx + s * cy, c: s, d: c, ty: cy - s * cx - c * cy }
    }
    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (self.a * x + self.b * y + self.tx, self.c * x + self.d * y + self.ty)
    }
    /// self after other: p -> self(other(p))
    pub fn then_after(&self, other: &Affine) -> Affine {
        Affine {
            a: self.a * other.a + self.b * other.c,
            b: self.a * other.b + self.b * other.d,
            tx: self.a * other.tx + self.b * other.ty + self.tx,
            c: self.c * other.a + self.d * other.c,
            d: self.c * other.b + self.d * other.d,
            ty: self.c * other.tx + self.d * other.ty + self.ty,
        }
    }
    pub fn invert(&self) -> Affine {
        let det = self.a * self.d - self.b * self.c;
        let det = if det.abs() < 1e-12 { 1e-12 } else { det };
        let a = self.d / det;
        let b = -self.b / det;
        let c = -self.c / det;
        let d = self.a / det;
        Affine { a, b, c, d, tx: -(a * self.tx + b * self.ty), ty: -(c * self.tx + d * self.ty) }
    }
    pub fn angle_deg(&self) -> f64 {
        self.c.atan2(self.a).to_degrees()
    }
    pub fn is_integer_shift(&self) -> bool {
        (self.a - 1.0).abs() < 1e-9
            && self.b.abs() < 1e-9
            && self.c.abs() < 1e-9
            && (self.d - 1.0).abs() < 1e-9
            && (self.tx - self.tx.round()).abs() < 1e-6
            && (self.ty - self.ty.round()).abs() < 1e-6
    }
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!([[self.a, self.b, self.tx], [self.c, self.d, self.ty]])
    }
}

/// The region a check reads: an upright patch and where it came from.
pub struct Region {
    pub patch: Img,
    /// patch pixel -> image pixel
    pub to_img: Affine,
    /// 255 inside the drawn shape (ellipse, polygon); None for a rectangle
    pub mask: Option<Vec<u8>>,
}

impl Region {
    pub fn map(&self, x: f64, y: f64) -> [f64; 2] {
        let (a, b) = self.to_img.apply(x, y);
        [a, b]
    }
    pub fn gray(&self) -> Img {
        self.patch.gray()
    }
    pub fn inside(&self, x: usize, y: usize) -> bool {
        match &self.mask {
            Some(m) => m[y * self.patch.w + x] != 0,
            None => true,
        }
    }
    /// The four corners in image coordinates.
    pub fn quad(&self) -> Vec<[f64; 2]> {
        let (w, h) = (self.patch.w as f64, self.patch.h as f64);
        vec![self.map(0.0, 0.0), self.map(w, 0.0), self.map(w, h), self.map(0.0, h)]
    }
}

/// patch -> reference coordinates for a region drawn at `roi` turned by `angle`
pub fn patch_to_ref(roi: [f64; 4], angle: f64) -> Affine {
    let [x, y, w, h] = roi;
    let (cx, cy) = (x + w / 2.0, y + h / 2.0);
    Affine::rotate_about(angle, cx, cy).then_after(&Affine::translate(x, y))
}

/// Cut a region out of `img`. `fixture` maps reference -> image.
pub fn extract(img: &Img, roi: [f64; 4], angle: f64, fixture: Option<&Affine>) -> Region {
    let w = roi[2].round().max(1.0) as usize;
    let h = roi[3].round().max(1.0) as usize;
    let p2r = patch_to_ref(roi, angle);
    let m = match fixture {
        Some(f) => f.then_after(&p2r),
        None => p2r,
    };
    Region { patch: warp(img, &m, w, h), to_img: m, mask: None }
}

/// Resample `img` through `m` (output pixel -> source pixel) into w x h.
pub fn warp(img: &Img, m: &Affine, w: usize, h: usize) -> Img {
    if m.is_integer_shift() {
        return img.crop(m.tx.round() as i64, m.ty.round() as i64, w, h);
    }
    let c = img.c;
    let mut out = Img::new(w, h, c);
    out.data.par_chunks_mut(w * c).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let (sx, sy) = m.apply(x as f64, y as f64);
            for k in 0..c {
                row[x * c + k] = img.sample(sx as f32, sy as f32, k).round().clamp(0.0, 255.0) as u8;
            }
        }
    });
    out
}

pub fn warp_plane(p: &Plane, m: &Affine, w: usize, h: usize) -> Plane {
    let mut out = Plane::new(w, h);
    out.d.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let (sx, sy) = m.apply(x as f64, y as f64);
            *o = p.sample(sx as f32, sy as f32);
        }
    });
    out
}

/// Ellipse mask filling a w x h box.
pub fn ellipse_mask(w: usize, h: usize) -> Vec<u8> {
    let mut m = vec![0u8; w * h];
    let (rx, ry) = (w as f64 / 2.0, h as f64 / 2.0);
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f64 + 0.5 - rx) / rx;
            let dy = (y as f64 + 0.5 - ry) / ry;
            if dx * dx + dy * dy <= 1.0 {
                m[y * w + x] = 255;
            }
        }
    }
    m
}

/// Polygon mask, points in patch coordinates.
pub fn polygon_mask(w: usize, h: usize, pts: &[[f64; 2]]) -> Vec<u8> {
    let mut m = vec![0u8; w * h];
    if pts.len() < 3 {
        return vec![255u8; w * h];
    }
    for y in 0..h {
        let fy = y as f64 + 0.5;
        let mut xs: Vec<f64> = Vec::new();
        for i in 0..pts.len() {
            let a = pts[i];
            let b = pts[(i + 1) % pts.len()];
            if (a[1] <= fy && b[1] > fy) || (b[1] <= fy && a[1] > fy) {
                xs.push(a[0] + (fy - a[1]) / (b[1] - a[1]) * (b[0] - a[0]));
            }
        }
        xs.sort_by(|p, q| p.partial_cmp(q).unwrap());
        for pair in xs.chunks(2) {
            if pair.len() < 2 {
                break;
            }
            let x0 = pair[0].max(0.0).round() as usize;
            let x1 = (pair[1].round() as usize).min(w);
            for x in x0..x1 {
                m[y * w + x] = 255;
            }
        }
    }
    m
}

/// Least-squares line through points: (point on line, unit direction, rms).
pub fn fit_line(pts: &[[f64; 2]]) -> Option<([f64; 2], [f64; 2], f64)> {
    if pts.len() < 2 {
        return None;
    }
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let my = pts.iter().map(|p| p[1]).sum::<f64>() / n;
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for p in pts {
        let dx = p[0] - mx;
        let dy = p[1] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let ang = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let dir = [ang.cos(), ang.sin()];
    let rms = (pts.iter().map(|p| {
        let d = -(p[0] - mx) * dir[1] + (p[1] - my) * dir[0];
        d * d
    }).sum::<f64>() / n).sqrt();
    Some(([mx, my], dir, rms))
}

/// Algebraic circle fit: (cx, cy, r, rms).
pub fn fit_circle(pts: &[[f64; 2]]) -> Option<(f64, f64, f64, f64)> {
    if pts.len() < 3 {
        return None;
    }
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let my = pts.iter().map(|p| p[1]).sum::<f64>() / n;
    let (mut suu, mut svv, mut suv, mut suuu, mut svvv, mut suvv, mut svuu) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for p in pts {
        let u = p[0] - mx;
        let v = p[1] - my;
        suu += u * u;
        svv += v * v;
        suv += u * v;
        suuu += u * u * u;
        svvv += v * v * v;
        suvv += u * v * v;
        svuu += v * u * u;
    }
    let det = suu * svv - suv * suv;
    if det.abs() < 1e-12 {
        return None;
    }
    let b1 = 0.5 * (suuu + suvv);
    let b2 = 0.5 * (svvv + svuu);
    let uc = (b1 * svv - b2 * suv) / det;
    let vc = (suu * b2 - suv * b1) / det;
    let r = (uc * uc + vc * vc + (suu + svv) / n).sqrt();
    let (cx, cy) = (uc + mx, vc + my);
    let rms = (pts.iter().map(|p| {
        let d = ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt() - r;
        d * d
    }).sum::<f64>() / n).sqrt();
    Some((cx, cy, r, rms))
}

/// Points of a circle as a polygon, for drawing.
pub fn circle_poly(cx: f64, cy: f64, r: f64, n: usize) -> Vec<[f64; 2]> {
    (0..n).map(|i| {
        let t = i as f64 / n as f64 * std::f64::consts::TAU;
        [cx + r * t.cos(), cy + r * t.sin()]
    }).collect()
}

/// A fixture as Vision Station holds it: a float32 2 x 3 matrix.
pub fn f32_rows(f: &Affine) -> [[f32; 3]; 2] {
    [[f.a as f32, f.b as f32, f.tx as f32], [f.c as f32, f.d as f32, f.ty as f32]]
}
