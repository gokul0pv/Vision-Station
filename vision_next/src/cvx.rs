//! OpenCV 4.10's own arithmetic, ported operation for operation, so the
//! text reader sees exactly the pixels Vision Station's (Python + cv2) does:
//! warpPerspective with INTER_CUBIC and a replicated border (with
//! getPerspectiveTransform's LU solve and the 3x3 inverse), resize with
//! INTER_LINEAR (the generic fixed-point path), and resize with INTER_CUBIC as
//! the Intel IPP build in opencv-python does it. Each one is checked pixel for
//! pixel against cv2 (vs-run cvx, scratchpad cvx_test.py).

use crate::img::Img;

// ---------------------------------------------------------------- matrices
/// hal::LU64f (LUImpl, eps = DBL_EPSILON*100) on an n x n system, b solved in place.
fn lu_solve(a: &mut [f64], n: usize, b: &mut [f64]) -> bool {
    let eps = f64::EPSILON * 100.0;
    for i in 0..n {
        let mut k = i;
        for j in i + 1..n {
            if a[j * n + i].abs() > a[k * n + i].abs() {
                k = j;
            }
        }
        if a[k * n + i].abs() < eps {
            return false;
        }
        if k != i {
            for j in i..n {
                a.swap(i * n + j, k * n + j);
            }
            b.swap(i, k);
        }
        let d = -1.0 / a[i * n + i];
        for j in i + 1..n {
            let alpha = a[j * n + i] * d;
            for k in i + 1..n {
                a[j * n + k] += alpha * a[i * n + k];
            }
            b[j] += alpha * b[i];
        }
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    true
}

/// cv::getPerspectiveTransform(Point2f src[4], Point2f dst[4]) with DECOMP_LU.
pub fn get_perspective_transform(src: &[[f32; 2]; 4], dst: &[[f32; 2]; 4]) -> [f64; 9] {
    let mut a = [0f64; 64];
    let mut b = [0f64; 8];
    for i in 0..4 {
        let (sx, sy, dx, dy) = (src[i][0], src[i][1], dst[i][0], dst[i][1]);
        a[i * 8] = sx as f64;
        a[(i + 4) * 8 + 3] = sx as f64;
        a[i * 8 + 1] = sy as f64;
        a[(i + 4) * 8 + 4] = sy as f64;
        a[i * 8 + 2] = 1.0;
        a[(i + 4) * 8 + 5] = 1.0;
        // float * float in C++: the product is a float, widened afterwards
        a[i * 8 + 6] = (-sx * dx) as f64;
        a[i * 8 + 7] = (-sy * dx) as f64;
        a[(i + 4) * 8 + 6] = (-sx * dy) as f64;
        a[(i + 4) * 8 + 7] = (-sy * dy) as f64;
        b[i] = dx as f64;
        b[i + 4] = dy as f64;
    }
    let mut m = [0f64; 9];
    if lu_solve(&mut a, 8, &mut b) {
        m[..8].copy_from_slice(&b);
    }
    m[8] = 1.0;
    m
}

/// cv::invert of a 3x3 double matrix (DECOMP_LU: the closed form).
pub fn invert3(s: &[f64; 9]) -> [f64; 9] {
    let m = |y: usize, x: usize| s[y * 3 + x];
    let det = m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
        + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
    if det == 0.0 {
        return [0.0; 9];
    }
    let d = 1.0 / det;
    [
        (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) * d,
        (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * d,
        (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * d,
        (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * d,
        (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * d,
        (m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2)) * d,
        (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)) * d,
        (m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1)) * d,
        (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) * d,
    ]
}

// ------------------------------------------------------- warpPerspective
const INTER_BITS: i32 = 5;
const INTER_TAB_SIZE: i32 = 1 << INTER_BITS;
const REMAP_SCALE: i32 = 1 << 15;

/// cv::interpolateCubic in float.
fn cubic_coeffs(x: f32) -> [f32; 4] {
    let a = -0.75f32;
    let c0 = ((a * (x + 1.0) - 5.0 * a) * (x + 1.0) + 8.0 * a) * (x + 1.0) - 4.0 * a;
    let c1 = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    let c2 = ((a + 2.0) * (1.0 - x) - (a + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    let c3 = 1.0 - c0 - c1 - c2;
    [c0, c1, c2, c3]
}

/// saturate_cast<short>(float): cvRound, ties to even.
fn sat_short(v: f32) -> i32 {
    (v.round_ties_even() as i32).clamp(-32768, 32767)
}

/// BicubicTab_i: the 32 x 32 sub-pixel tables of 4 x 4 integer weights summing to 2^15.
fn bicubic_tab() -> &'static Vec<[i32; 16]> {
    static T: std::sync::OnceLock<Vec<[i32; 16]>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let scale = 1.0f32 / INTER_TAB_SIZE as f32;
        let tab1: Vec<[f32; 4]> = (0..INTER_TAB_SIZE).map(|i| cubic_coeffs(i as f32 * scale)).collect();
        let mut out = Vec::with_capacity((INTER_TAB_SIZE * INTER_TAB_SIZE) as usize);
        for i in 0..INTER_TAB_SIZE as usize {
            for j in 0..INTER_TAB_SIZE as usize {
                let mut it = [0i32; 16];
                let mut isum = 0;
                for k1 in 0..4 {
                    let vy = tab1[i][k1];
                    for k2 in 0..4 {
                        let v = vy * tab1[j][k2];
                        it[k1 * 4 + k2] = sat_short(v * REMAP_SCALE as f32);
                        isum += it[k1 * 4 + k2];
                    }
                }
                if isum != REMAP_SCALE {
                    let diff = isum - REMAP_SCALE;
                    let (mut mk1, mut mk2, mut big1, mut big2) = (2usize, 2usize, 2usize, 2usize);
                    for k1 in 2..4 {
                        for k2 in 2..4 {
                            if it[k1 * 4 + k2] < it[mk1 * 4 + mk2] {
                                mk1 = k1;
                                mk2 = k2;
                            } else if it[k1 * 4 + k2] > it[big1 * 4 + big2] {
                                big1 = k1;
                                big2 = k2;
                            }
                        }
                    }
                    if diff < 0 {
                        it[big1 * 4 + big2] -= diff;
                    } else {
                        it[mk1 * 4 + mk2] -= diff;
                    }
                }
                out.push(it);
            }
        }
        out
    })
}

/// cv::warpPerspective(src, M, (w, h), INTER_CUBIC, BORDER_REPLICATE), M
/// mapping source to destination (it is inverted here, as OpenCV does).
pub fn warp_perspective_cubic(src: &Img, m_fwd: &[f64; 9], w: usize, h: usize) -> Img {
    if let Some(r) = crate::cvlib::warp_u8(src, m_fwd, w, h, crate::cvlib::INTER_CUBIC, crate::cvlib::BORDER_REPLICATE) {
        return r;
    }
    let m = invert3(m_fwd);
    let cn = src.c;
    let mut out = Img::new(w, h, cn);
    if w == 0 || h == 0 || src.w == 0 || src.h == 0 {
        return out;
    }
    let tab = bicubic_tab();
    // the invoker's blocks: X0 is evaluated at each block's first column
    let block = 32usize;
    let mut bh0 = (block / 2).min(h);
    let bw0 = (block * block / bh0).min(w);
    bh0 = (block * block / bw0).min(h);
    let _ = bh0;
    let (sw, sh) = (src.w as i64, src.h as i64);
    for yy in 0..h {
        let mut x = 0usize;
        while x < w {
            let bw = bw0.min(w - x);
            let x0 = m[0] * x as f64 + m[1] * yy as f64 + m[2];
            let y0 = m[3] * x as f64 + m[4] * yy as f64 + m[5];
            let w0 = m[6] * x as f64 + m[7] * yy as f64 + m[8];
            for x1 in 0..bw {
                let mut ww = w0 + m[6] * x1 as f64;
                ww = if ww != 0.0 { INTER_TAB_SIZE as f64 / ww } else { 0.0 };
                let fx = ((x0 + m[0] * x1 as f64) * ww).clamp(i32::MIN as f64, i32::MAX as f64);
                let fy = ((y0 + m[3] * x1 as f64) * ww).clamp(i32::MIN as f64, i32::MAX as f64);
                let xi = fx.round_ties_even() as i64 as i32;
                let yi = fy.round_ties_even() as i64 as i32;
                let sx = ((xi >> INTER_BITS) as i64).clamp(i16::MIN as i64, i16::MAX as i64) - 1;
                let sy = ((yi >> INTER_BITS) as i64).clamp(i16::MIN as i64, i16::MAX as i64) - 1;
                let a = ((yi & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (xi & (INTER_TAB_SIZE - 1))) as usize;
                let wt = &tab[a];
                let xs: [usize; 4] = std::array::from_fn(|i| (sx + i as i64).clamp(0, sw - 1) as usize);
                let ys: [usize; 4] = std::array::from_fn(|i| (sy + i as i64).clamp(0, sh - 1) as usize);
                let o = (yy * w + x + x1) * cn;
                for k in 0..cn {
                    let mut sum: i32 = 0;
                    for (r, &yv) in ys.iter().enumerate() {
                        let row = yv * src.w;
                        for (c, &xv) in xs.iter().enumerate() {
                            sum += src.data[(row + xv) * cn + k] as i32 * wt[r * 4 + c];
                        }
                    }
                    out.data[o + k] = ((sum + (1 << 14)) >> 15).clamp(0, 255) as u8;
                }
            }
            x += bw0;
        }
    }
    out
}

// ------------------------------------------------------------ resize, linear
const COEF_BITS: i32 = 11;
const COEF_SCALE: f32 = (1 << COEF_BITS) as f32;

/// cv::resize(src, (dw, dh), INTER_LINEAR) for 8-bit pictures: the generic
/// fixed-point path (IPP is not used for 8-bit linear). The exact 2x
/// downscale goes through INTER_AREA as OpenCV does.
pub fn resize_linear(src: &Img, dw: usize, dh: usize) -> Img {
    if let Some(r) = crate::cvlib::resize(src, dw, dh, 0.0, 0.0, crate::cvlib::INTER_LINEAR) {
        return r;
    }
    let inv_x = dw as f64 / src.w as f64;
    let inv_y = dh as f64 / src.h as f64;
    let (scale_x, scale_y) = (1.0 / inv_x, 1.0 / inv_y);
    let (ix, iy) = (scale_x.round_ties_even() as i64, scale_y.round_ties_even() as i64);
    if (scale_x - ix as f64).abs() < f64::EPSILON && (scale_y - iy as f64).abs() < f64::EPSILON && ix == 2 && iy == 2 {
        return area_fast2(src, dw, dh);
    }
    let cn = src.c;
    let sw = src.w as i64;
    // x taps: (offset, a0, a1)
    let xt: Vec<(usize, i32, i32)> = (0..dw).map(|dx| {
        let fx0 = ((dx as f64 + 0.5) * scale_x - 0.5) as f32;
        let mut sx = fx0.floor() as i64;
        let mut fx = fx0 - sx as f32;
        if sx < 0 {
            fx = 0.0;
            sx = 0;
        }
        if sx + 1 >= sw && sx >= sw - 1 {
            fx = 0.0;
            sx = sw - 1;
        }
        let a0 = sat_short((1.0 - fx) * COEF_SCALE);
        let a1 = sat_short(fx * COEF_SCALE);
        (sx as usize, a0, a1)
    }).collect();
    let yt: Vec<(i64, i32, i32)> = (0..dh).map(|dy| {
        let fy0 = ((dy as f64 + 0.5) * scale_y - 0.5) as f32;
        let sy = fy0.floor() as i64;
        let fy = fy0 - sy as f32;
        (sy, sat_short((1.0 - fy) * COEF_SCALE), sat_short(fy * COEF_SCALE))
    }).collect();
    let hrow = |y: usize| -> Vec<i32> {
        let row = &src.data[y * src.w * cn..(y + 1) * src.w * cn];
        let mut d = vec![0i32; dw * cn];
        for (dx, &(sx, a0, a1)) in xt.iter().enumerate() {
            for k in 0..cn {
                let s0 = row[sx * cn + k] as i32;
                d[dx * cn + k] = if sx + 1 < src.w { s0 * a0 + row[(sx + 1) * cn + k] as i32 * a1 } else { s0 * (1 << COEF_BITS) };
            }
        }
        d
    };
    let mut out = Img::new(dw, dh, cn);
    let sh = src.h as i64;
    for (dy, &(sy, b0, b1)) in yt.iter().enumerate() {
        let r0 = hrow(sy.clamp(0, sh - 1) as usize);
        let r1 = hrow((sy + 1).clamp(0, sh - 1) as usize);
        let o = &mut out.data[dy * dw * cn..(dy + 1) * dw * cn];
        for x in 0..dw * cn {
            let v = ((b0 * (r0[x] >> 4)) >> 16) + ((b1 * (r1[x] >> 4)) >> 16) + 2;
            o[x] = (v >> 2).clamp(0, 255) as u8;
        }
    }
    out
}

/// INTER_AREA at exactly half size: the mean of each 2 x 2, rounded.
fn area_fast2(src: &Img, dw: usize, dh: usize) -> Img {
    let cn = src.c;
    let mut out = Img::new(dw, dh, cn);
    for y in 0..dh {
        for x in 0..dw {
            for k in 0..cn {
                let p = |yy: usize, xx: usize| src.data[(yy.min(src.h - 1) * src.w + xx.min(src.w - 1)) * cn + k] as u32;
                let s = p(2 * y, 2 * x) + p(2 * y, 2 * x + 1) + p(2 * y + 1, 2 * x) + p(2 * y + 1, 2 * x + 1);
                out.data[(y * dw + x) * cn + k] = ((s + 2) >> 2) as u8;
            }
        }
    }
    out
}

// ------------------------------------------------------------- resize, cubic
fn cubic64(x: f64) -> [f64; 4] {
    let a = -0.75;
    let c0 = ((a * (x + 1.0) - 5.0 * a) * (x + 1.0) + 8.0 * a) * (x + 1.0) - 4.0 * a;
    let c1 = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    let c2 = ((a + 2.0) * (1.0 - x) - (a + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    [c0, c1, c2, 1.0 - c0 - c1 - c2]
}

/// cv::resize(src, None, fx=s, fy=s, INTER_CUBIC) as opencv-python's IPP does it.
/// When the size ratio is not exactly s IPP warps (double arithmetic, matched
/// exactly); at an exact ratio it resizes in float, matched except for a few
/// pixels whose value is exactly half-way between two levels.
pub fn resize_cubic_scale(src: &Img, s: f64) -> Img {
    if let Some(r) = crate::cvlib::resize(src, 0, 0, s, s, crate::cvlib::INTER_CUBIC) {
        return r;
    }
    let dw = ((src.w as f64 * s).round_ties_even() as usize).max(1);
    let dh = ((src.h as f64 * s).round_ties_even() as usize).max(1);
    let ex = ((dw as f64 / src.w as f64) - s).abs() / s;
    let ey = ((dh as f64 / src.h as f64) - s).abs() / s;
    let exact = ex <= 1e-10 && ey <= 1e-10;
    let cn = src.c;
    let taps = |n_src: usize, n_dst: usize| -> Vec<([usize; 4], [f64; 4])> {
        (0..n_dst).map(|d| {
            let f = (d as f64 + 0.5) / s - 0.5;
            let i = f.floor();
            let t = f - i;
            let i = i as i64;
            let idx = std::array::from_fn(|k| (i - 1 + k as i64).clamp(0, n_src as i64 - 1) as usize);
            (idx, cubic64(t))
        }).collect()
    };
    let xt = taps(src.w, dw);
    let yt = taps(src.h, dh);
    let mut out = Img::new(dw, dh, cn);
    if !exact {
        // horizontal then vertical, double throughout, rounded to nearest
        let mut hbuf = vec![0f64; src.h * dw * cn];
        for y in 0..src.h {
            for (dx, (idx, w)) in xt.iter().enumerate() {
                for k in 0..cn {
                    let mut acc = 0.0;
                    for j in 0..4 {
                        acc += src.data[(y * src.w + idx[j]) * cn + k] as f64 * w[j];
                    }
                    hbuf[(y * dw + dx) * cn + k] = acc;
                }
            }
        }
        for (dy, (idx, w)) in yt.iter().enumerate() {
            for x in 0..dw * cn {
                let mut acc = 0.0;
                for j in 0..4 {
                    acc += hbuf[idx[j] * dw * cn + x] * w[j];
                }
                out.data[dy * dw * cn + x] = acc.round_ties_even().clamp(0.0, 255.0) as u8;
            }
        }
    } else {
        // float32 weights; products summed in pairs with fused multiply-add
        let w32 = |w: &[f64; 4]| [w[0] as f32, w[1] as f32, w[2] as f32, w[3] as f32];
        let sum4 = |v: [f32; 4], w: [f32; 4]| -> f32 {
            // a true fused multiply-add (f32::mul_add goes to this toolchain's
            // fmaf, which is not one): the product is exact in f64
            let fma = |a: f32, b: f32, c: f32| (a as f64 * b as f64 + c as f64) as f32;
            let p = fma(v[1], w[1], v[0] * w[0]);
            let q = fma(v[3], w[3], v[2] * w[2]);
            p + q
        };
        let mut hbuf = vec![0f32; src.h * dw * cn];
        for y in 0..src.h {
            for (dx, (idx, w)) in xt.iter().enumerate() {
                let w = w32(w);
                for k in 0..cn {
                    let v = std::array::from_fn(|j| src.data[(y * src.w + idx[j]) * cn + k] as f32);
                    hbuf[(y * dw + dx) * cn + k] = sum4(v, w);
                }
            }
        }
        for (dy, (idx, w)) in yt.iter().enumerate() {
            let w = w32(w);
            for x in 0..dw * cn {
                let v = std::array::from_fn(|j| hbuf[idx[j] * dw * cn + x]);
                out.data[dy * dw * cn + x] = sum4(v, w).round_ties_even().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

// ------------------------------------------------------------ findContours
const DELTAS: [[i32; 2]; 8] = [[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]];

/// cv::findContours(mask, RETR_LIST, CHAIN_APPROX_SIMPLE): OpenCV 4.10's
/// contour scanner (contours_new.cpp) on the 1-pixel bordered picture, in
/// OpenCV's output order (the last found first).
pub fn find_contours(mask: &[u8], w: usize, h: usize) -> Vec<Vec<[i32; 2]>> {
    let step = w + 2;
    let mut im = vec![0i8; step * (h + 2)];
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] != 0 {
                im[(y + 1) * step + x + 1] = 1;
            }
        }
    }
    let delta = |s: i32| -> isize { let d = DELTAS[(s % 8) as usize]; d[0] as isize + d[1] as isize * step as isize };
    const RIGHT: i8 = -126; // MASK8_NEW | MASK8_RIGHT = 0x82
    const NEW: i8 = 2;
    let width = (w + 2) as isize - 1;
    let height = (h + 2) as isize - 1;
    let mut found: Vec<Vec<[i32; 2]>> = vec![];
    let fetch = |im: &mut Vec<i8>, sx: isize, sy: isize, hole: bool| -> Vec<[i32; 2]> {
        let mut pts = vec![];
        let i0 = sy * step as isize + sx;
        let mut pt = [sx as i32 - 1, sy as i32 - 1];
        let mut s_end: i32 = if hole { 0 } else { 4 };
        let mut s = s_end;
        let mut i1;
        loop {
            s = (s - 1) & 7;
            i1 = i0 + delta(s);
            if im[i1 as usize] != 0 || s == s_end {
                break;
            }
        }
        if s == s_end {
            im[i0 as usize] = RIGHT;
            pts.push(pt);
            return pts;
        }
        let mut i3 = i0;
        let mut prev_s = s ^ 4;
        loop {
            s_end = s;
            s = s.min(15);
            let mut i4 = i3;
            while s < 15 {
                s += 1;
                i4 = i3 + delta(s);
                if im[i4 as usize] != 0 {
                    break;
                }
            }
            s &= 7;
            if ((s - 1) as u32) < (s_end as u32) {
                im[i3 as usize] = RIGHT;
            } else if im[i3 as usize] == 1 {
                im[i3 as usize] = NEW;
            }
            if s != prev_s {
                pts.push(pt);
            }
            prev_s = s;
            pt[0] += DELTAS[s as usize][0];
            pt[1] += DELTAS[s as usize][1];
            if i4 == i0 && i3 == i1 {
                break;
            }
            i3 = i4;
            s = (s + 4) & 7;
        }
        pts
    };
    let (mut px, mut py) = (1isize, 1isize);
    'scan: loop {
        let (mut x, mut y) = (px, py);
        let mut prev = im[(y * step as isize + x - 1) as usize] as i32;
        while y < height {
            let mut p: i32 = 0;
            while x < width {
                while x < width {
                    p = im[(y * step as isize + x) as usize] as i32;
                    if p != prev {
                        break;
                    }
                    x += 1;
                }
                if x >= width {
                    break;
                }
                let hole;
                if prev == 0 && p == 1 {
                    hole = false;
                } else if p != 0 || prev < 1 {
                    prev = p;
                    x += 1;
                    continue;
                } else {
                    hole = true;
                }
                let sx = x - if hole { 1 } else { 0 };
                found.push(fetch(&mut im, sx, y, hole));
                px = x + 1;
                py = y;
                continue 'scan;
            }
            let _ = p;
            y += 1;
            x = 1;
            prev = 0;
        }
        break;
    }
    found.reverse();
    found
}

/// cv::boundingRect of integer points: (x, y, w, h).
pub fn bounding_rect(pts: &[[i32; 2]]) -> [i32; 4] {
    let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for p in pts {
        x0 = x0.min(p[0]);
        x1 = x1.max(p[0]);
        y0 = y0.min(p[1]);
        y1 = y1.max(p[1]);
    }
    if pts.is_empty() {
        return [0, 0, 0, 0];
    }
    [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
}

// ------------------------------------------------------------- convexHull
fn sign_i64(v: i64) -> i32 {
    (v > 0) as i32 - (v < 0) as i32
}

/// Sklansky_<int, int64> over sorted point indices; returns the stack length.
fn sklansky(p: &[[i32; 2]], arr: &[usize], start: isize, end: isize, stack: &mut [isize], nsign: i32, sign2: i32) -> usize {
    let incr: isize = if end > start { 1 } else { -1 };
    let (mut pprev, mut pcur) = (start, start + incr);
    let mut pnext = pcur + incr;
    let mut stacksize: usize = 3;
    let pt = |i: isize| p[arr[i as usize]];
    if start == end || (pt(start)[0] == pt(end)[0] && pt(start)[1] == pt(end)[1]) {
        stack[0] = start;
        return 1;
    }
    stack[0] = pprev;
    stack[1] = pcur;
    stack[2] = pnext;
    let end = end + incr;
    while pnext != end {
        let cury = pt(pcur)[1] as i64;
        let nexty = pt(pnext)[1] as i64;
        let by = nexty - cury;
        if sign_i64(by) != nsign {
            let ax = (pt(pcur)[0] - pt(pprev)[0]) as i64;
            let bx = (pt(pnext)[0] - pt(pcur)[0]) as i64;
            let ay = cury - pt(pprev)[1] as i64;
            let convexity = ay * bx - ax * by;
            if sign_i64(convexity) == sign2 && (ax != 0 || ay != 0) {
                pprev = pcur;
                pcur = pnext;
                pnext += incr;
                stack[stacksize] = pnext;
                stacksize += 1;
            } else if pprev == start {
                pcur = pnext;
                stack[1] = pcur;
                pnext += incr;
                stack[2] = pnext;
            } else {
                stack[stacksize - 2] = pnext;
                pcur = pprev;
                pprev = stack[stacksize - 4];
                stacksize -= 1;
            }
        } else {
            pnext += incr;
            stack[stacksize - 1] = pnext;
        }
    }
    stacksize - 1
}

/// cv::convexHull(points, clockwise=false, returnPoints=true) for integer points.
pub fn convex_hull(p: &[[i32; 2]]) -> Vec<[i32; 2]> {
    let total = p.len();
    if total == 0 {
        return vec![];
    }
    let mut arr: Vec<usize> = (0..total).collect();
    arr.sort_by(|&a, &b| p[a][0].cmp(&p[b][0]).then(p[a][1].cmp(&p[b][1])).then(a.cmp(&b)));
    let (mut miny, mut maxy) = (0usize, 0usize);
    for i in 1..total {
        let y = p[arr[i]][1];
        if p[arr[miny]][1] > y {
            miny = i;
        }
        if p[arr[maxy]][1] < y {
            maxy = i;
        }
    }
    let mut hullbuf: Vec<usize> = vec![];
    let mut stack = vec![0isize; total + 2];
    if p[arr[0]] == p[arr[total - 1]] {
        hullbuf.push(0);
    } else {
        let tl_count = sklansky(p, &arr, 0, maxy as isize, &mut stack, -1, 1);
        let tl: Vec<isize> = stack[..tl_count].to_vec();
        let mut st2 = vec![0isize; total + 2];
        let tr_count = sklansky(p, &arr, total as isize - 1, maxy as isize, &mut st2, -1, -1);
        let tr: Vec<isize> = st2[..tr_count].to_vec();
        // counter-clockwise: the two halves swapped
        let (tl, tr) = (tr, tl);
        for i in 0..tl.len().saturating_sub(1) {
            hullbuf.push(arr[tl[i] as usize]);
        }
        for i in (1..tr.len()).rev() {
            hullbuf.push(arr[tr[i] as usize]);
        }
        let stop_idx: isize = if tr.len() > 2 { tr[1] } else if tl.len() > 2 { tl[tl.len() - 2] } else { -1 };
        let mut st3 = vec![0isize; total + 2];
        let bl_count = sklansky(p, &arr, 0, miny as isize, &mut st3, 1, -1);
        let mut bl: Vec<isize> = st3[..bl_count].to_vec();
        let mut st4 = vec![0isize; total + 2];
        let br_count = sklansky(p, &arr, total as isize - 1, miny as isize, &mut st4, 1, 1);
        let mut br: Vec<isize> = st4[..br_count].to_vec();
        if stop_idx >= 0 {
            let check_idx: isize = if bl.len() > 2 { bl[1] } else if bl.len() + br.len() > 2 { br[2 - bl.len()] } else { -1 };
            if check_idx == stop_idx || (check_idx >= 0 && p[arr[check_idx as usize]] == p[arr[stop_idx as usize]]) {
                bl.truncate(2);
                br.truncate(2);
            }
        }
        for i in 0..bl.len().saturating_sub(1) {
            hullbuf.push(arr[bl[i] as usize]);
        }
        for i in (1..br.len()).rev() {
            hullbuf.push(arr[br[i] as usize]);
        }
        let nout = hullbuf.len();
        if nout >= 3 {
            let (mut min_idx, mut max_idx, mut lt) = (0usize, 0usize, 0usize);
            for i in 1..nout {
                let idx = hullbuf[i];
                lt += (hullbuf[i - 1] < idx) as usize;
                if lt > 1 && lt + 2 <= i {
                    break;
                }
                if idx < hullbuf[min_idx] {
                    min_idx = i;
                }
                if idx > hullbuf[max_idx] {
                    max_idx = i;
                }
            }
            let mmdist = (max_idx as isize - min_idx as isize).unsigned_abs();
            if (mmdist == 1 || mmdist == nout - 1) && (lt <= 1 || lt + 2 >= nout) {
                let ascending = (max_idx + 1) % nout == min_idx;
                let i0 = if ascending { min_idx } else { max_idx };
                let mut j = i0;
                if i0 > 0 {
                    let mut tmp = vec![0usize; nout];
                    let mut i = 0;
                    while i < nout {
                        let curr = hullbuf[j];
                        tmp[i] = curr;
                        let next_j = if j + 1 < nout { j + 1 } else { 0 };
                        let next_idx = hullbuf[next_j];
                        if i < nout - 1 && (ascending != (curr < next_idx)) {
                            break;
                        }
                        j = next_j;
                        i += 1;
                    }
                    if i == nout {
                        hullbuf = tmp;
                    }
                }
            }
        }
    }
    hullbuf.into_iter().map(|i| p[i]).collect()
}

// ------------------------------------------------------------ minAreaRect
/// cv::RotatedRect: centre, (width, height), angle in degrees -- all float.
#[derive(Clone, Copy, Debug)]
pub struct RotRect {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub angle: f32,
}

fn calipers(pts: &[[f32; 2]]) -> [f32; 6] {
    let n = pts.len();
    let mut minarea = f32::MAX;
    let mut vect = vec![[0f32; 2]; n];
    let mut inv = vec![0f32; n];
    let (mut left, mut bottom, mut right, mut top) = (0usize, 0usize, 0usize, 0usize);
    let mut pt0 = pts[0];
    let (mut left_x, mut right_x, mut top_y, mut bottom_y) = (pt0[0], pt0[0], pt0[1], pt0[1]);
    for i in 0..n {
        if pt0[0] < left_x {
            left_x = pt0[0];
            left = i;
        }
        if pt0[0] > right_x {
            right_x = pt0[0];
            right = i;
        }
        if pt0[1] > top_y {
            top_y = pt0[1];
            top = i;
        }
        if pt0[1] < bottom_y {
            bottom_y = pt0[1];
            bottom = i;
        }
        let pt = pts[if i + 1 < n { i + 1 } else { 0 }];
        // float - float in C++ is a float, widened to double afterwards
        let dx = (pt[0] - pt0[0]) as f64;
        let dy = (pt[1] - pt0[1]) as f64;
        vect[i] = [dx as f32, dy as f32];
        inv[i] = (1.0 / (dx * dx + dy * dy).sqrt()) as f32;
        pt0 = pt;
    }
    let mut orientation = 0f32;
    {
        let (mut ax, mut ay) = (vect[n - 1][0] as f64, vect[n - 1][1] as f64);
        for v in vect.iter() {
            let (bx, by) = (v[0] as f64, v[1] as f64);
            let c = ax * by - ay * bx;
            if c != 0.0 {
                orientation = if c > 0.0 { 1.0 } else { -1.0 };
                break;
            }
            ax = bx;
            ay = by;
        }
    }
    let mut base_a = orientation;
    let mut base_b = 0f32;
    let mut seq = [bottom, right, top, left];
    let mut buf_i = [0usize; 2];
    let mut buf_f = [0f32; 4]; // base_a, width, base_b, height
    let right_of = |v1: [f32; 2], v2: [f32; 2]| -> bool { let t = [v1[1], -v1[0]]; t[0] * v2[0] + t[1] * v2[1] < 0.0 };
    for _ in 0..n {
        let rv = [
            vect[seq[0]],
            [vect[seq[1]][1], -vect[seq[1]][0]],
            [-vect[seq[2]][0], -vect[seq[2]][1]],
            [-vect[seq[3]][1], vect[seq[3]][0]],
        ];
        let mut main = 0usize;
        for i in 1..4 {
            if right_of(rv[i], rv[main]) {
                main = i;
            }
        }
        let pi = seq[main];
        let lx = vect[pi][0] * inv[pi];
        let ly = vect[pi][1] * inv[pi];
        match main {
            0 => { base_a = lx; base_b = ly; }
            1 => { base_a = ly; base_b = -lx; }
            2 => { base_a = -lx; base_b = -ly; }
            _ => { base_a = -ly; base_b = lx; }
        }
        seq[main] += 1;
        if seq[main] == n {
            seq[main] = 0;
        }
        let dx = pts[seq[1]][0] - pts[seq[3]][0];
        let dy = pts[seq[1]][1] - pts[seq[3]][1];
        let width = dx * base_a + dy * base_b;
        let dx = pts[seq[2]][0] - pts[seq[0]][0];
        let dy = pts[seq[2]][1] - pts[seq[0]][1];
        let height = -dx * base_b + dy * base_a;
        let area = width * height;
        if area <= minarea {
            minarea = area;
            buf_i = [seq[3], seq[0]];
            buf_f = [base_a, width, base_b, height];
        }
    }
    let (a1, b1) = (buf_f[0], buf_f[2]);
    let (a2, b2) = (-buf_f[2], buf_f[0]);
    let c1 = a1 * pts[buf_i[0]][0] + pts[buf_i[0]][1] * b1;
    let c2 = a2 * pts[buf_i[1]][0] + pts[buf_i[1]][1] * b2;
    let idet = 1.0f32 / (a1 * b2 - a2 * b1);
    let px = (c1 * b2 - c2 * b1) * idet;
    let py = (a1 * c2 - a2 * c1) * idet;
    [px, py, a1 * buf_f[1], b1 * buf_f[1], a2 * buf_f[3], b2 * buf_f[3]]
}

/// cv::minAreaRect of integer points.
pub fn min_area_rect(points: &[[i32; 2]]) -> RotRect {
    let hull: Vec<[f32; 2]> = convex_hull(points).into_iter().map(|p| [p[0] as f32, p[1] as f32]).collect();
    let n = hull.len();
    let mut r = RotRect { cx: 0.0, cy: 0.0, w: 0.0, h: 0.0, angle: 0.0 };
    if n > 2 {
        let o = calipers(&hull);
        r.cx = o[0] + (o[2] + o[4]) * 0.5;
        r.cy = o[1] + (o[3] + o[5]) * 0.5;
        r.w = ((o[2] as f64) * (o[2] as f64) + (o[3] as f64) * (o[3] as f64)).sqrt() as f32;
        r.h = ((o[4] as f64) * (o[4] as f64) + (o[5] as f64) * (o[5] as f64)).sqrt() as f32;
        r.angle = (o[3] as f64).atan2(o[2] as f64) as f32;
    } else if n == 2 {
        r.cx = (hull[0][0] + hull[1][0]) * 0.5;
        r.cy = (hull[0][1] + hull[1][1]) * 0.5;
        let dx = (hull[1][0] - hull[0][0]) as f64;
        let dy = (hull[1][1] - hull[0][1]) as f64;
        r.w = (dx * dx + dy * dy).sqrt() as f32;
        r.angle = dy.atan2(dx) as f32;
    } else if n == 1 {
        r.cx = hull[0][0];
        r.cy = hull[0][1];
    }
    r.angle = ((r.angle * 180.0) as f64 / std::f64::consts::PI) as f32;
    r
}

/// cv::boxPoints / RotatedRect::points.
pub fn box_points(r: &RotRect) -> [[f32; 2]; 4] {
    let ang = r.angle as f64 * std::f64::consts::PI / 180.0;
    let b = (ang.cos() as f32) * 0.5;
    let a = (ang.sin() as f32) * 0.5;
    let p0 = [r.cx - a * r.h - b * r.w, r.cy + b * r.h - a * r.w];
    let p1 = [r.cx + a * r.h - b * r.w, r.cy - b * r.h - a * r.w];
    [p0, p1, [2.0 * r.cx - p0[0], 2.0 * r.cy - p0[1]], [2.0 * r.cx - p1[0], 2.0 * r.cy - p1[1]]]
}

// --------------------------------------------------------------- fillPoly
const XY_SHIFT: i64 = 16;
const XY_ONE: i64 = 1 << XY_SHIFT;

/// cv::fillPoly(mask, [pts], 1) with LINE_8 and shift 0, for a closed
/// contour whose sides are straight 8-way runs (CHAIN_APPROX_SIMPLE).
pub fn fill_poly(mask: &mut [u8], w: usize, h: usize, pts: &[[i32; 2]]) {
    let n = pts.len();
    if n == 0 {
        return;
    }
    // the outline itself: each side is a horizontal, vertical or diagonal run
    let mut set = |x: i64, y: i64, mask: &mut [u8]| {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            mask[y as usize * w + x as usize] = 1;
        }
    };
    struct Edge { y0: i64, y1: i64, x: i64, dx: i64, next: usize }
    let mut edges: Vec<Edge> = vec![];
    let mut pt0 = pts[n - 1];
    for i in 0..n {
        let pt1 = pts[i];
        let (sx, sy) = ((pt1[0] - pt0[0]).signum() as i64, (pt1[1] - pt0[1]).signum() as i64);
        let steps = (pt1[0] - pt0[0]).abs().max((pt1[1] - pt0[1]).abs()) as i64;
        for k in 0..=steps {
            set(pt0[0] as i64 + sx * k, pt0[1] as i64 + sy * k, mask);
        }
        let (p0x, p0y) = (((pt0[0] as i64) << XY_SHIFT) + (XY_ONE >> 1), pt0[1] as i64);
        let (p1x, p1y) = (((pt1[0] as i64) << XY_SHIFT) + (XY_ONE >> 1), pt1[1] as i64);
        if p0y != p1y {
            let dx = (p1x - p0x) / (p1y - p0y);
            if p0y < p1y {
                edges.push(Edge { y0: p0y, y1: p1y, x: p0x, dx, next: usize::MAX });
            } else {
                edges.push(Edge { y0: p1y, y1: p0y, x: p1x, dx, next: usize::MAX });
            }
        }
        pt0 = pt1;
    }
    let total = edges.len();
    if total < 2 {
        return;
    }
    let mut y_max = i64::MIN;
    for e in &edges {
        y_max = y_max.max(e.y1);
    }
    edges.sort_by(|a, b| {
        if a.y0 != b.y0 { a.y0.cmp(&b.y0) } else if a.x != b.x { a.x.cmp(&b.x) } else { a.dx.cmp(&b.dx) }
    });
    // index `total` is the sentinel (y0 = MAX); `total + 1` is tmp, the list head
    edges.push(Edge { y0: i64::MAX, y1: 0, x: 0, dx: 0, next: usize::MAX });
    edges.push(Edge { y0: 0, y1: 0, x: 0, dx: 0, next: usize::MAX });
    let tmp = total + 1;
    const NIL: usize = usize::MAX;
    let mut i = 0usize;
    let mut e = 0usize;
    let y_max = y_max.min(h as i64);
    let mut y = edges[e].y0;
    while y < y_max {
        let mut draw = 0;
        let clipline = y < 0;
        let mut prelast = tmp;
        let mut last = edges[tmp].next;
        while last != NIL || edges[e].y0 == y {
            if last != NIL && edges[last].y1 == y {
                let nx = edges[last].next;
                edges[prelast].next = nx;
                last = nx;
                continue;
            }
            let keep_prelast = prelast;
            if last != NIL && (edges[e].y0 > y || edges[last].x < edges[e].x) {
                prelast = last;
                last = edges[last].next;
            } else if i < total {
                edges[prelast].next = e;
                edges[e].next = last;
                prelast = e;
                i += 1;
                e = i;
            } else {
                break;
            }
            if draw != 0 {
                if !clipline {
                    let (x1, x2) = if edges[keep_prelast].x > edges[prelast].x {
                        (edges[prelast].x >> XY_SHIFT, edges[keep_prelast].x >> XY_SHIFT)
                    } else {
                        (edges[keep_prelast].x >> XY_SHIFT, edges[prelast].x >> XY_SHIFT)
                    };
                    if x1 < w as i64 && x2 >= 0 {
                        let x1 = x1.max(0);
                        let x2 = x2.min(w as i64 - 1);
                        for xx in x1..=x2 {
                            mask[y as usize * w + xx as usize] = 1;
                        }
                    }
                }
                let d = edges[keep_prelast].dx;
                edges[keep_prelast].x += d;
                let d = edges[prelast].dx;
                edges[prelast].x += d;
            }
            draw ^= 1;
        }
        // bubble sort of the active list by x
        let mut keep_prelast = NIL;
        loop {
            let mut prelast = tmp;
            let mut last = edges[tmp].next;
            let mut last_exchange = NIL;
            while last != keep_prelast && last != NIL && edges[last].next != NIL {
                let te = edges[last].next;
                if edges[last].x > edges[te].x {
                    edges[prelast].next = te;
                    edges[last].next = edges[te].next;
                    edges[te].next = last;
                    prelast = te;
                    last_exchange = prelast;
                } else {
                    prelast = last;
                    last = te;
                }
            }
            if last_exchange == NIL {
                break;
            }
            keep_prelast = last_exchange;
            if keep_prelast == edges[tmp].next || keep_prelast == tmp {
                break;
            }
        }
        y += 1;
    }
}

// ---------------------------------------------------------- text boxes
/// ocr_engine._boxes_from_map: rotated boxes of print from the detector's
/// probability map (thresh 0.2, box score 0.45, unclip 1.5, smallest side 4),
/// each box's corners (float32) and its score, in Python's order.
pub fn boxes_from_map(prob: &[f32], w: usize, h: usize) -> Vec<([[f32; 2]; 4], f64)> {
    let mask: Vec<u8> = prob.iter().map(|&p| (p > 0.2f32) as u8).collect();
    let mut out = vec![];
    for c in find_contours(&mask, w, h).into_iter().take(3000) {
        let r = min_area_rect(&c);
        let (rw, rh) = (r.w as f64, r.h as f64);
        if rw.min(rh) < 2.0 {
            continue;
        }
        let br = bounding_rect(&c);
        let (bx, by, bw, bh) = (br[0], br[1], br[2] as usize, br[3] as usize);
        let mut sub = vec![0u8; bw * bh];
        let local: Vec<[i32; 2]> = c.iter().map(|p| [p[0] - bx, p[1] - by]).collect();
        fill_poly(&mut sub, bw, bh, &local);
        let (mut s, mut cnt) = (0f64, 0usize);
        for yy in 0..bh {
            for xx in 0..bw {
                if sub[yy * bw + xx] != 0 {
                    s += prob[(by as usize + yy) * w + bx as usize + xx] as f64;
                    cnt += 1;
                }
            }
        }
        // cv::mean (IPP): the sum times the reciprocal of the count
        let score = if cnt > 0 { s * (1.0 / cnt as f64) } else { 0.0 };
        if score < 0.45 {
            continue;
        }
        let grow = (rw * rh) * 1.5 / (2.0 * (rw + rh)).max(1e-6);
        let (rw2, rh2) = (rw + 2.0 * grow, rh + 2.0 * grow);
        if rw2.min(rh2) < 4.0 {
            continue;
        }
        let pts = box_points(&RotRect { cx: r.cx, cy: r.cy, w: rw2 as f32, h: rh2 as f32, angle: r.angle });
        out.push((order_pts(pts), score));
    }
    out
}

/// ocr_engine._order: top-left, top-right, bottom-right, bottom-left (stable sorts).
pub fn order_pts(p: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
    let mut v = p.to_vec();
    v.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
    let mut l = vec![v[0], v[1]];
    let mut r = vec![v[2], v[3]];
    l.sort_by(|a, b| a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal));
    r.sort_by(|a, b| a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal));
    [l[0], r[0], r[1], l[1]]
}

// ------------------------------------------- warpPerspective, linear, constant
/// BilinearTab_i, built as OpenCV 4.10 builds it -- including its fix-up,
/// which for this 2 x 2 kernel looks one block ahead and so only ever
/// corrects a sum that came out short.
fn bilinear_tab() -> &'static Vec<[i32; 4]> {
    static T: std::sync::OnceLock<Vec<[i32; 4]>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let n = (INTER_TAB_SIZE * INTER_TAB_SIZE) as usize;
        let mut flat = vec![0i32; n * 4 + 8];
        let scale = 1.0f32 / INTER_TAB_SIZE as f32;
        let tab1: Vec<[f32; 2]> = (0..INTER_TAB_SIZE).map(|i| { let x = i as f32 * scale; [1.0 - x, x] }).collect();
        for i in 0..INTER_TAB_SIZE as usize {
            for j in 0..INTER_TAB_SIZE as usize {
                let base = (i * INTER_TAB_SIZE as usize + j) * 4;
                let mut isum = 0;
                for k1 in 0..2 {
                    let vy = tab1[i][k1];
                    for k2 in 0..2 {
                        let v = vy * tab1[j][k2];
                        flat[base + k1 * 2 + k2] = sat_short(v * REMAP_SCALE as f32);
                        isum += flat[base + k1 * 2 + k2];
                    }
                }
                if isum != REMAP_SCALE {
                    let diff = isum - REMAP_SCALE;
                    let (mut mk, mut big) = (base + 3, base + 3);
                    for k1 in 1..3 {
                        for k2 in 1..3 {
                            let o = base + k1 * 2 + k2;
                            if flat[o] < flat[mk] {
                                mk = o;
                            } else if flat[o] > flat[big] {
                                big = o;
                            }
                        }
                    }
                    if diff < 0 {
                        flat[big] -= diff;
                    } else {
                        flat[mk] -= diff;
                    }
                }
            }
        }
        (0..n).map(|k| [flat[k * 4], flat[k * 4 + 1], flat[k * 4 + 2], flat[k * 4 + 3]]).collect()
    })
}

/// cv::warpPerspective(src, M, (w, h), INTER_LINEAR) with the default
/// constant (black) border, M mapping source to destination.
pub fn warp_perspective_linear(src: &Img, m_fwd: &[f64; 9], w: usize, h: usize) -> Img {
    if let Some(r) = crate::cvlib::warp_u8(src, m_fwd, w, h, crate::cvlib::INTER_LINEAR, crate::cvlib::BORDER_CONSTANT) {
        return r;
    }
    let m = invert3(m_fwd);
    let cn = src.c;
    let mut out = Img::new(w, h, cn);
    if w == 0 || h == 0 || src.w == 0 || src.h == 0 {
        return out;
    }
    let tab = bilinear_tab();
    let block = 32usize;
    let bh0 = (block / 2).min(h);
    let bw0 = (block * block / bh0).min(w);
    let (sw, sh) = (src.w as i64, src.h as i64);
    for yy in 0..h {
        let mut x = 0usize;
        while x < w {
            let bw = bw0.min(w - x);
            let x0 = m[0] * x as f64 + m[1] * yy as f64 + m[2];
            let y0 = m[3] * x as f64 + m[4] * yy as f64 + m[5];
            let w0 = m[6] * x as f64 + m[7] * yy as f64 + m[8];
            for x1 in 0..bw {
                let mut ww = w0 + m[6] * x1 as f64;
                ww = if ww != 0.0 { INTER_TAB_SIZE as f64 / ww } else { 0.0 };
                let fx = ((x0 + m[0] * x1 as f64) * ww).clamp(i32::MIN as f64, i32::MAX as f64);
                let fy = ((y0 + m[3] * x1 as f64) * ww).clamp(i32::MIN as f64, i32::MAX as f64);
                let xi = fx.round_ties_even() as i64 as i32;
                let yi = fy.round_ties_even() as i64 as i32;
                let sx = ((xi >> INTER_BITS) as i64).clamp(i16::MIN as i64, i16::MAX as i64);
                let sy = ((yi >> INTER_BITS) as i64).clamp(i16::MIN as i64, i16::MAX as i64);
                let wt = &tab[((yi & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (xi & (INTER_TAB_SIZE - 1))) as usize];
                let o = (yy * w + x + x1) * cn;
                if sx >= sw || sx + 1 < 0 || sy >= sh || sy + 1 < 0 {
                    continue; // all black
                }
                let at = |xx: i64, yv: i64, k: usize| -> i32 {
                    if xx >= 0 && yv >= 0 && xx < sw && yv < sh { src.data[(yv as usize * src.w + xx as usize) * cn + k] as i32 } else { 0 }
                };
                for k in 0..cn {
                    let sum = at(sx, sy, k) * wt[0] + at(sx + 1, sy, k) * wt[1] + at(sx, sy + 1, k) * wt[2] + at(sx + 1, sy + 1, k) * wt[3];
                    out.data[o + k] = ((sum + (1 << 14)) >> 15).clamp(0, 255) as u8;
                }
            }
            x += bw0;
        }
    }
    out
}

// ------------------------------------------------------------- Tool.crop
/// cv2.transform of float32 points by a float32 2 x 3 matrix.
pub fn transform_f32(p: [f32; 2], m: &[[f32; 3]; 2]) -> [f32; 2] {
    [m[0][0] * p[0] + m[0][1] * p[1] + m[0][2], m[1][0] * p[0] + m[1][1] * p[1] + m[1][2]]
}

/// numpy's float32 vector norm of 2 values: sqrt of a BLAS dot.
/// OpenBLAS sdot adds a short vector's products in double and rounds once,
/// which for two products is the float sum.
fn norm2_f32(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// Python's round() of a float: to nearest, ties to even.
fn pyround(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// What a Vision Station check works on (tools/base.py Tool.crop, a rect or
/// ellipse region): the plain slice when the part has not turned or moved by
/// a fraction of a pixel, else the region cut upright with warpPerspective.
pub struct ToolCrop {
    pub patch: Img,
    pub ox: i64,
    pub oy: i64,
    /// crop -> image, when the region was cut out by warping
    pub crop_to_img: Option<[f64; 9]>,
}

/// Tool.mapped_roi for a rect region: (x, y, w, h) of where it lies now.
pub fn mapped_roi(roi: [f64; 4], fixture: Option<&[[f32; 3]; 2]>) -> [i64; 4] {
    let (x, y, w, h) = (roi[0] as f32, roi[1] as f32, roi[2] as f32, roi[3] as f32);
    let mut c = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
    if let Some(m) = fixture {
        for p in c.iter_mut() {
            *p = transform_f32(*p, m);
        }
    }
    let xmin = c.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
    let xmax = c.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
    let ymin = c.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
    let ymax = c.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
    [pyround(xmin as f64), pyround(ymin as f64), pyround((xmax - xmin) as f64), pyround((ymax - ymin) as f64)]
}

pub fn tool_crop(img: &Img, roi: [f64; 4], angle: f64, fixture: Option<&[[f32; 3]; 2]>) -> ToolCrop {
    let r = mapped_roi(roi, fixture);
    let mut fix_rot = 0.0;
    let mut shifted = false;
    if let Some(f) = fixture {
        fix_rot = (f[1][0] as f64).atan2(f[0][0] as f64).to_degrees();
        let q = |i: usize, j: usize| f[i][j] as f64;
        let tx = q(0, 0) * roi[0] + q(0, 1) * roi[1] + q(0, 2);
        let ty = q(1, 0) * roi[0] + q(1, 1) * roi[1] + q(1, 2);
        shifted = (tx - tx.round_ties_even()).abs() > 0.01 || (ty - ty.round_ties_even()).abs() > 0.01;
    }
    let rotated = angle.abs() > 1e-6 || fix_rot.abs() > 0.05 || shifted;
    if !rotated {
        let (x, y) = (r[0].max(0), r[1].max(0));
        let w = r[2].min(img.w as i64 - x).max(1);
        let h = r[3].min(img.h as i64 - y).max(1);
        return ToolCrop { patch: img.crop(x, y, w as usize, h as usize), ox: x, oy: y, crop_to_img: None };
    }
    // Tool.mapped_quad: the corners about the centre, turned, in float32, then moved
    let (x, y, w, h) = (roi[0], roi[1], roi[2], roi[3]);
    let (cx, cy) = (x + w / 2.0, y + h / 2.0);
    let a = angle.to_radians();
    let (ca, sa) = (a.cos(), a.sin());
    let mut quad: [[f32; 2]; 4] = [(-w / 2.0, -h / 2.0), (w / 2.0, -h / 2.0), (w / 2.0, h / 2.0), (-w / 2.0, h / 2.0)]
        .map(|(dx, dy)| [(cx + dx * ca - dy * sa) as f32, (cy + dx * sa + dy * ca) as f32]);
    if let Some(m) = fixture {
        for p in quad.iter_mut() {
            *p = transform_f32(*p, m);
        }
    }
    let sub = |a: [f32; 2], b: [f32; 2]| [a[0] - b[0], a[1] - b[1]];
    let ow = (pyround(norm2_f32(sub(quad[1], quad[0])).max(norm2_f32(sub(quad[2], quad[3]))) as f64)).max(1) as usize;
    let oh = (pyround(norm2_f32(sub(quad[3], quad[0])).max(norm2_f32(sub(quad[2], quad[1]))) as f64)).max(1) as usize;
    let (fw, fh) = ((ow - 1) as f32, (oh - 1) as f32);
    let dst = [[0.0, 0.0], [fw, 0.0], [fw, fh], [0.0, fh]];
    let m = get_perspective_transform(&quad, &dst);
    let patch = warp_perspective_linear(img, &m, ow, oh);
    let back = get_perspective_transform(&dst, &quad);
    let ox = pyround(quad.iter().map(|p| p[0]).fold(f32::MAX, f32::min) as f64);
    let oy = pyround(quad.iter().map(|p| p[1]).fold(f32::MAX, f32::min) as f64);
    ToolCrop { patch, ox, oy, crop_to_img: Some(back) }
}

/// numpy's population standard deviation of 8-bit values (float64, pairwise sums).
pub fn np_std_u8(v: &[u8]) -> f64 {
    let a: Vec<f64> = v.iter().map(|&x| x as f64).collect();
    let n = a.len().max(1) as f64;
    let mean = np_pairwise(&a) / n;
    let d: Vec<f64> = a.iter().map(|x| { let t = x - mean; t * t }).collect();
    (np_pairwise(&d) / n).sqrt()
}

/// numpy's pairwise summation (blocks of 128, 8 accumulators).
pub fn np_pairwise(a: &[f64]) -> f64 {
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
        np_pairwise(&a[..n2]) + np_pairwise(&a[n2..])
    }
}

// ------------------------------------------------------------ GaussianBlur
/// cv::borderInterpolate with BORDER_REFLECT_101.
pub fn reflect101(mut p: i64, len: i64) -> usize {
    if p >= 0 && p < len {
        return p as usize;
    }
    if len == 1 {
        return 0;
    }
    loop {
        if p < 0 {
            p = -p - 1 + 1;
        } else {
            p = len - 1 - (p - len) - 1;
        }
        if p >= 0 && p < len {
            return p as usize;
        }
    }
}

/// The 8-bit GaussianBlur kernel: getGaussianKernelBitExact (sigma > 0), then
/// 8 fraction bits by error diffusion (getGaussianKernelFixedPoint_ED); sums to 256.
fn gauss_kernel_fixed(n: usize, sigma: f64) -> Vec<u32> {
    if n == 1 {
        return vec![256];
    }
    let scale2x = -0.125 / (sigma * sigma);
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
    let mut res = vec![0f64; n];
    for i in 0..n2 {
        let t = values[i] * mul1;
        res[i] = t;
        res[n - 1 - i] = t;
    }
    res[n2] = 1.0 * mul1;
    let mut out = vec![0u32; n];
    let mut err = 0f64;
    let mut isum: i64 = 0;
    for i in 0..n2 {
        let adj = res[i] * 256.0 + err;
        let v0 = adj.round_ties_even() as i64;
        err = adj - v0 as f64;
        out[i] = v0 as u32;
        out[n - 1 - i] = v0 as u32;
        isum += v0;
    }
    out[n2] = (256 - 2 * isum) as u32;
    out
}

/// cv2.GaussianBlur(img, (0, 0), sigma) on an 8-bit picture: OpenCV's bit-exact
/// fixed-point filter, BORDER_REFLECT_101.
pub fn gaussian_blur(src: &Img, sigma: f64) -> Img {
    let n = ((sigma * 3.0 * 2.0 + 1.0).round_ties_even() as i64 | 1) as usize;
    let k = gauss_kernel_fixed(n, sigma);
    let kx = if src.w == 1 { vec![256u32] } else { k.clone() };
    let ky = if src.h == 1 { vec![256u32] } else { k };
    let (w, h, cn) = (src.w, src.h, src.c);
    let (rx, ry) = ((kx.len() / 2) as i64, (ky.len() / 2) as i64);
    let mut hb = vec![0u32; w * h * cn];
    for y in 0..h {
        let row = &src.data[y * w * cn..(y + 1) * w * cn];
        for x in 0..w {
            for c in 0..cn {
                let mut s = 0u32;
                for (j, &kv) in kx.iter().enumerate() {
                    let xx = reflect101(x as i64 + j as i64 - rx, w as i64);
                    s += kv * row[xx * cn + c] as u32;
                }
                hb[(y * w + x) * cn + c] = s;
            }
        }
    }
    let mut out = Img::new(w, h, cn);
    for y in 0..h {
        for i in 0..w * cn {
            let mut s = 0u64;
            for (j, &kv) in ky.iter().enumerate() {
                let yy = reflect101(y as i64 + j as i64 - ry, h as i64);
                s += kv as u64 * hb[yy * w * cn + i] as u64;
            }
            out.data[y * w * cn + i] = ((s + (1 << 15)) >> 16).min(255) as u8;
        }
    }
    out
}

// ------------------------------------------------------------ the stretch
/// np.percentile(values, p) (method linear) of a float32 array, as numpy 2.1 computes it.
fn np_percentile_f32(sorted: &[f32], p: f64) -> f32 {
    let n = sorted.len();
    let q = (p / 100.0) as f32;
    let vi = (n as f32 - 1.0) * q;
    let fl = vi.floor();
    let t = vi - fl;
    let i = (fl as usize).min(n - 1);
    let (a, b) = (sorted[i], sorted[(i + 1).min(n - 1)]);
    let d = b - a;
    if t >= 0.5 { b - d * (1.0 - t) } else { a + d * t }
}

/// ocr_engine._stretch: 1st..99th percentile of all values to 0..255 (float32, truncated).
pub fn stretch(img: &Img) -> Img {
    let mut v: Vec<f32> = img.data.iter().map(|&x| x as f32).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = np_percentile_f32(&v, 1.0);
    let hi = np_percentile_f32(&v, 99.0);
    let span = hi - lo;
    let den = if span > 1.0 { span } else { 1.0 };
    let mut out = img.clone();
    for p in out.data.iter_mut() {
        let x = ((*p as f32 - lo) * 255.0f32 / den).clamp(0.0, 255.0);
        *p = x as u8;
    }
    out
}

// ------------------------------------------------------------- Lab, 8-bit
const D65: [f64; 3] = [f64::from_bits(0x3fee6a22b3892ee8), 1.0, f64::from_bits(0x3ff16b8950763a19)];
const SRGB2XYZ: [u64; 9] = [0x3fda65a14488c60d, 0x3fd6e297396d0918, 0x3fc71819d2391d58, 0x3fcb38cda6e75ff6,
    0x3fe6e297396d0918, 0x3fb279aae6c8f755, 0x3f93cc4ac6cdaf4b, 0x3fbe836eb4e98138, 0x3fee68427418d691];
const XYZ2SRGB: [u64; 9] = [0x4009ec804102ff8f, 0xbff8982a9930be0e, 0xbfdfe7ff583a53b9, 0xbfef042528ae74f3,
    0x3ffe040f23897204, 0x3fa546d3f9e7b80b, 0x3fac7de5082cf52c, 0xbfca1e14bdfd2631, 0x3ff0eabef06b3786];

fn apply_gamma(x: f32) -> f32 {
    let xd = x as f64;
    let thr = 809.0 / 20000.0;
    (if xd <= thr { xd / (323.0 / 25.0) } else { ((xd + 11.0 / 200.0) / (1.0 + 11.0 / 200.0)).powf(12.0 / 5.0) }) as f32
}

fn apply_inv_gamma(x: f32) -> f32 {
    let xd = x as f64;
    let thr = 7827.0 / 2500000.0;
    (if xd <= thr { xd * (323.0 / 25.0) } else { xd.powf(1.0 / (12.0 / 5.0)) * (1.0 + 11.0 / 200.0) - 11.0 / 200.0 }) as f32
}

/// OpenCV's softfloat cbrt (f32_cbrt): a rational polynomial in double on the
/// mantissa, its result truncated (not rounded) back to float.
fn ocv_cbrt_f32(x: f32) -> f32 {
    let u = x.to_bits();
    if u & 0x7fff_ffff == 0 {
        return 0.0;
    }
    let s = u >> 31;
    let ex0 = ((u >> 23) & 0xff) as i32 - 127;
    let mut shx = ex0 % 3;
    if shx >= 0 {
        shx -= 3;
    }
    let ex = (ex0 - shx) / 3 - 1;
    let frac = (u & 0x7f_ffff) as u64;
    let mut fr = f64::from_bits((((shx + 1023) as u64) << 52) | (frac << 29));
    let a = f64::from_bits;
    fr = ((((a(0x4046a09e6653ba70) * fr + a(0x406808f46c6116e0)) * fr + a(0x405dca97439cae14)) * fr + a(0x402add70d2827500)) * fr
        + a(0x3fc4f15f83f55d2d))
        / ((((a(0x402d9e20660edb21) * fr + a(0x4062ff15c0285815)) * fr + a(0x406510d06a8112ce)) * fr + a(0x4040fecbc9e2c375)) * fr + 1.0);
    let frac64 = fr.to_bits() & ((1u64 << 52) - 1);
    f32::from_bits((s << 31) | (((ex + 127) as u32) << 23) | ((frac64 >> 29) as u32))
}

struct LabTabs {
    gamma_b: [i32; 256],
    cbrt_b: Vec<i32>,
    inv_gamma_b: Vec<i32>,
    y_f: Vec<i32>,
    ab_xz: Vec<i32>,
    to_lab: [i32; 9],
    to_rgb: [i32; 9],
}

const MIN_AB: i32 = -8145;

fn lab_tabs() -> &'static LabTabs {
    static T: std::sync::OnceLock<LabTabs> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let f255 = 255f32;
        let mut gamma_b = [0i32; 256];
        let int_scale = (255 * 8) as f32;
        for (i, g) in gamma_b.iter_mut().enumerate() {
            let x = i as f32 / f255;
            *g = (int_scale * apply_gamma(x)).round_ties_even() as i32;
        }
        let lthresh = 216f32 / 24389f32;
        let lscale = 841f32 / 108f32;
        let lbias = 16f32 / 116f32;
        let cb_scale = 1f32 / (f255 * 8.0);
        let lshift2 = (1 << 15) as f32;
        let cbrt_b: Vec<i32> = (0..256 * 3 / 2 * 8).map(|i| {
            let x = cb_scale * i as f32;
            let v = if x < lthresh { x.mul_add(lscale, lbias) } else { ocv_cbrt_f32(x) };
            (lshift2 * v).round_ties_even() as i32
        }).collect();
        let inv_scale = 1f32 / 4096f32;
        let inv_gamma_b: Vec<i32> = (0..4096).map(|i| (f255 * apply_inv_gamma(inv_scale * i as f32)).round_ties_even() as i32).collect();
        let lut_base: i32 = 1 << 14;
        let mut y_f = vec![0i32; 512];
        for i in 0..256i32 {
            let (y, ify);
            if i <= 20 {
                y = ((i * lut_base * 20 * 9) as f32 / (17 * 29 * 29 * 29) as f32).round_ties_even() as i32;
                ify = (lut_base as f32 * (16f32 / 116f32 + (i * 5) as f32 / (3 * 17 * 29) as f32)).round_ties_even() as i32;
            } else {
                let fy = (i * 100 * lut_base) as f32 / (255 * 116) as f32 + (16 * lut_base) as f32 / 116f32;
                ify = fy.round_ties_even() as i32;
                y = (fy * fy * fy / (lut_base * lut_base) as f32).round_ties_even() as i32;
            }
            y_f[i as usize * 2] = y;
            y_f[i as usize * 2 + 1] = ify;
        }
        let base = 1i32 << 14;
        let ab_xz: Vec<i32> = (MIN_AB..base * 9 / 4 + MIN_AB).map(|i| {
            if i <= 3390 {
                i * 108 / 841 - lut_base * 16 / 116 * 108 / 841
            } else {
                (((i as i64 * i as i64) / lut_base as i64) * i as i64 / lut_base as i64) as i32
            }
        }).collect();
        // RGB2Lab_b with blueIdx 0: [B, G, R] coefficients per row (X, Y, Z)
        let lshift = 4096.0;
        let mut to_lab = [0i32; 9];
        for i in 0..3 {
            let c: [f64; 3] = std::array::from_fn(|j| f64::from_bits(SRGB2XYZ[i * 3 + j]));
            to_lab[i * 3 + 2] = (lshift * c[0] / D65[i]).round_ties_even() as i32;
            to_lab[i * 3 + 1] = (lshift * c[1] / D65[i]).round_ties_even() as i32;
            to_lab[i * 3] = (lshift * c[2] / D65[i]).round_ties_even() as i32;
        }
        // Lab2RGBinteger with blueIdx 0: rows R, G, B over (x, y, z)
        let mut to_rgb = [0i32; 9];
        for i in 0..3 {
            let c: [f64; 3] = std::array::from_fn(|j| f64::from_bits(XYZ2SRGB[i + j * 3]));
            to_rgb[i] = (lshift * c[0] * D65[i]).round_ties_even() as i32;
            to_rgb[i + 3] = (lshift * c[1] * D65[i]).round_ties_even() as i32;
            to_rgb[i + 6] = (lshift * c[2] * D65[i]).round_ties_even() as i32;
        }
        LabTabs { gamma_b, cbrt_b, inv_gamma_b, y_f, ab_xz, to_lab, to_rgb }
    })
}

fn descale(x: i32, n: i32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// cv2.cvtColor(BGR2LAB) of an 8-bit picture held in RGB order: (L, a, b) per pixel.
pub fn rgb_to_lab(src: &Img) -> Img {
    let t = lab_tabs();
    let c = &t.to_lab;
    let lscale = (116 * 255 + 50) / 100;
    let lsh = -((16 * 255 * (1 << 15) + 50) / 100);
    let mut out = Img::new(src.w, src.h, 3);
    for (o, p) in out.data.chunks_exact_mut(3).zip(src.data.chunks_exact(3)) {
        let (b, g, r) = (t.gamma_b[p[2] as usize], t.gamma_b[p[1] as usize], t.gamma_b[p[0] as usize]);
        let fx = t.cbrt_b[descale(b * c[0] + g * c[1] + r * c[2], 12) as usize];
        let fy = t.cbrt_b[descale(b * c[3] + g * c[4] + r * c[5], 12) as usize];
        let fz = t.cbrt_b[descale(b * c[6] + g * c[7] + r * c[8], 12) as usize];
        let l = descale(lscale * fy + lsh, 15);
        let a = descale(500 * (fx - fy) + 128 * (1 << 15), 15);
        let bb = descale(200 * (fy - fz) + 128 * (1 << 15), 15);
        o[0] = l.clamp(0, 255) as u8;
        o[1] = a.clamp(0, 255) as u8;
        o[2] = bb.clamp(0, 255) as u8;
    }
    out
}

/// cv2.cvtColor(LAB2BGR) of 8-bit (L, a, b), back to RGB order (Lab2RGBinteger).
pub fn lab_to_rgb(src: &Img) -> Img {
    let t = lab_tabs();
    let c = &t.to_rgb;
    let base = 1i32 << 14;
    let mut out = Img::new(src.w, src.h, 3);
    for (o, p) in out.data.chunks_exact_mut(3).zip(src.data.chunks_exact(3)) {
        let (ll, aa, bb) = (p[0] as i32, p[1] as i32, p[2] as i32);
        let y = t.y_f[ll as usize * 2];
        let ify = t.y_f[ll as usize * 2 + 1];
        let adiv = ((5 * aa * 53687 + (1 << 7)) >> 13) - 128 * base / 500;
        let bdiv = ((bb * 41943 + (1 << 4)) >> 9) - 128 * base / 200 + 1;
        let x = t.ab_xz[(ify + adiv - MIN_AB) as usize];
        let z = t.ab_xz[(ify - bdiv - MIN_AB) as usize];
        let ro = descale(c[0] * x + c[1] * y + c[2] * z, 14).clamp(0, 4095);
        let go = descale(c[3] * x + c[4] * y + c[5] * z, 14).clamp(0, 4095);
        let bo = descale(c[6] * x + c[7] * y + c[8] * z, 14).clamp(0, 4095);
        o[0] = t.inv_gamma_b[ro as usize].clamp(0, 255) as u8;
        o[1] = t.inv_gamma_b[go as usize].clamp(0, 255) as u8;
        o[2] = t.inv_gamma_b[bo as usize].clamp(0, 255) as u8;
    }
    out
}

/// cv2.createCLAHE(clip, (tiles, tiles)).apply(gray8)
pub fn clahe(src: &[u8], w: usize, h: usize, clip: f64, tiles: usize) -> Vec<u8> {
    let (ew, eh) = if w % tiles == 0 && h % tiles == 0 { (w, h) } else { (w + tiles - w % tiles, h + tiles - h % tiles) };
    let (ew, eh) = (if w % tiles == 0 && h % tiles == 0 { ew } else { w + (tiles - w % tiles) }, if w % tiles == 0 && h % tiles == 0 { eh } else { h + (tiles - h % tiles) });
    let ext = |x: usize, y: usize| src[reflect101(y as i64, h as i64) * w + reflect101(x as i64, w as i64)];
    let (tw, th) = (ew / tiles, eh / tiles);
    let total = (tw * th) as i32;
    let lut_scale = 255f32 / total as f32;
    let clip_limit = if clip > 0.0 { ((clip * total as f64 / 256.0) as i32).max(1) } else { 0 };
    let mut lut = vec![0u8; tiles * tiles * 256];
    for k in 0..tiles * tiles {
        let (ty, tx) = (k / tiles, k % tiles);
        let mut hist = [0i32; 256];
        for y in ty * th..ty * th + th {
            for x in tx * tw..tx * tw + tw {
                hist[ext(x, y) as usize] += 1;
            }
        }
        if clip_limit > 0 {
            let mut clipped = 0;
            for v in hist.iter_mut() {
                if *v > clip_limit {
                    clipped += *v - clip_limit;
                    *v = clip_limit;
                }
            }
            let batch = clipped / 256;
            let mut residual = clipped - batch * 256;
            for v in hist.iter_mut() {
                *v += batch;
            }
            if residual != 0 {
                let step = (256 / residual).max(1) as usize;
                let mut i = 0usize;
                while i < 256 && residual > 0 {
                    hist[i] += 1;
                    i += step;
                    residual -= 1;
                }
            }
        }
        let mut sum = 0i32;
        for i in 0..256 {
            sum += hist[i];
            lut[k * 256 + i] = (sum as f32 * lut_scale).round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
    let inv_tw = 1.0f32 / tw as f32;
    let inv_th = 1.0f32 / th as f32;
    let xs: Vec<(usize, usize, f32, f32)> = (0..w).map(|x| {
        let txf = x as f32 * inv_tw - 0.5;
        let tx1 = txf.floor() as i64;
        let xa = txf - tx1 as f32;
        let xa1 = 1.0 - xa;
        (tx1.max(0) as usize, ((tx1 + 1) as usize).min(tiles - 1), xa, xa1)
    }).collect();
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let tyf = y as f32 * inv_th - 0.5;
        let ty1 = tyf.floor() as i64;
        let ya = tyf - ty1 as f32;
        let ya1 = 1.0 - ya;
        let p1 = ty1.max(0) as usize * tiles;
        let p2 = ((ty1 + 1) as usize).min(tiles - 1) * tiles;
        for x in 0..w {
            let v = src[y * w + x] as usize;
            let (i1, i2, xa, xa1) = xs[x];
            let l = |plane: usize, t: usize| lut[(plane + t) * 256 + v] as f32;
            let res = (l(p1, i1) * xa1 + l(p1, i2) * xa) * ya1 + (l(p2, i1) * xa1 + l(p2, i2) * xa) * ya;
            out[y * w + x] = res.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// ocr_engine._clahe: CLAHE (3.0, 8 x 8) on the L of Lab, back to colour.
pub fn clahe_lab(src: &Img) -> Img {
    let mut lab = rgb_to_lab(src);
    let l: Vec<u8> = lab.data.chunks_exact(3).map(|p| p[0]).collect();
    let l2 = clahe(&l, lab.w, lab.h, 3.0, 8);
    for (p, v) in lab.data.chunks_exact_mut(3).zip(l2) {
        p[0] = v;
    }
    lab_to_rgb(&lab)
}
