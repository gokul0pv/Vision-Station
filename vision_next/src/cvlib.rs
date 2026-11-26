//! The real OpenCV, when it is there: runtime/vsx_cv.dll (cv/vsx_cv.cpp, a
//! plain C bridge over the official OpenCV 4.10 build -- the library Vision
//! Station's cv2 is). Through it Vision Next gets OpenCV's own results,
//! Intel IPP paths included, which no port can copy exactly.
//!
//! Every function returns None when the bridge is missing or OpenCV raised;
//! callers then use the Rust ports. VS_NO_CV=1 switches the bridge off (to
//! compare the two).

use crate::img::Img;
use libloading::{Library, Symbol};
use std::sync::OnceLock;

struct Lib {
    lib: Library,
}

fn lib() -> Option<&'static Lib> {
    static L: OnceLock<Option<Lib>> = OnceLock::new();
    L.get_or_init(|| {
        if std::env::var("VS_NO_CV").map(|v| v == "1").unwrap_or(false) {
            return None;
        }
        let cands: Vec<std::path::PathBuf> = ["vsx_cv.dll", "runtime/vsx_cv.dll"].iter()
            .filter_map(|n| crate::paths::find(&[n])).collect();
        for c in cands {
            if c.exists() {
                // the OpenCV DLL sits beside it
                if let Some(dir) = c.parent() {
                    let p = std::env::var("PATH").unwrap_or_default();
                    if !p.contains(&dir.display().to_string()) {
                        std::env::set_var("PATH", format!("{};{}", dir.display(), p));
                    }
                }
                if let Ok(lib) = unsafe { Library::new(&c) } {
                    return Some(Lib { lib });
                }
            }
        }
        None
    }).as_ref()
}

/// Is OpenCV in use (and which)?
pub fn version() -> Option<String> {
    let l = lib()?;
    unsafe {
        let f: Symbol<unsafe extern "C" fn() -> *const std::os::raw::c_char> = l.lib.get(b"vsx_version\0").ok()?;
        Some(std::ffi::CStr::from_ptr(f()).to_string_lossy().into_owned())
    }
}

fn last_error() -> String {
    lib().and_then(|l| unsafe {
        let f: Symbol<unsafe extern "C" fn() -> *const std::os::raw::c_char> = l.lib.get(b"vsx_last_error\0").ok()?;
        Some(std::ffi::CStr::from_ptr(f()).to_string_lossy().into_owned())
    }).unwrap_or_default()
}

macro_rules! sym {
    ($l:expr, $name:literal, $t:ty) => {{
        let s: Symbol<$t> = unsafe { $l.lib.get(concat!($name, "\0").as_bytes()).ok()? };
        s
    }};
}

pub const INTER_NEAREST: i32 = 0;
pub const INTER_LINEAR: i32 = 1;
pub const INTER_CUBIC: i32 = 2;
pub const INTER_AREA: i32 = 3;
pub const WARP_INVERSE_MAP: i32 = 16;
pub const BORDER_CONSTANT: i32 = 0;
pub const BORDER_REPLICATE: i32 = 1;
pub const TM_CCOEFF_NORMED: i32 = 5;
pub const TM_SQDIFF: i32 = 0;

/// cv::resize to (dw, dh), or by fx, fy when dw/dh are 0.
pub fn resize(src: &Img, dw: usize, dh: usize, fx: f64, fy: f64, interp: i32) -> Option<Img> {
    let l = lib()?;
    let f = sym!(l, "vsx_resize", unsafe extern "C" fn(*const u8, i32, i32, i32, i32, *mut u8, i32, *mut i32, *mut i32, f64, f64, i32) -> i32);
    let (ew, eh) = if dw > 0 { (dw, dh) } else {
        (((src.w as f64 * fx).round() as usize + 2), ((src.h as f64 * fy).round() as usize + 2))
    };
    let mut buf = vec![0u8; ew * eh * src.c];
    let (mut ow, mut oh) = (dw as i32, dh as i32);
    let r = unsafe { f(src.data.as_ptr(), src.w as i32, src.h as i32, src.c as i32, (src.w * src.c) as i32,
        buf.as_mut_ptr(), buf.len() as i32, &mut ow, &mut oh, fx, fy, interp) };
    if r != 0 {
        log::warn(&format!("cv resize: {}", last_error()));
        return None;
    }
    buf.truncate(ow as usize * oh as usize * src.c);
    Some(Img { w: ow as usize, h: oh as usize, c: src.c, data: buf })
}

/// cv::warpAffine (m: 2 x 3) or warpPerspective (m: 3 x 3) of an 8-bit picture.
pub fn warp_u8(src: &Img, m: &[f64], dw: usize, dh: usize, flags: i32, border: i32) -> Option<Img> {
    let l = lib()?;
    let f = sym!(l, "vsx_warp", unsafe extern "C" fn(*const u8, i32, i32, i32, i32, i32, *mut u8, i32, i32, *const f64, i32, i32, i32, f64) -> i32);
    let mut out = Img::new(dw, dh, src.c);
    if dw == 0 || dh == 0 {
        return Some(out);
    }
    let r = unsafe { f(src.data.as_ptr(), src.w as i32, src.h as i32, src.c as i32, (src.w * src.c) as i32, 0,
        out.data.as_mut_ptr(), dw as i32, dh as i32, m.as_ptr(), (m.len() == 9) as i32, flags, border, 0.0) };
    if r != 0 { None } else { Some(out) }
}

/// cv::matchTemplate on 8-bit grey: (w, h, scores).
pub fn match_template(img: &Img, tpl: &Img, method: i32) -> Option<(usize, usize, Vec<f32>)> {
    let l = lib()?;
    let f = sym!(l, "vsx_match_template", unsafe extern "C" fn(*const u8, i32, i32, i32, *const u8, i32, i32, i32, i32, *mut f32) -> i32);
    if tpl.w > img.w || tpl.h > img.h || img.c != 1 || tpl.c != 1 {
        return None;
    }
    let (ow, oh) = (img.w - tpl.w + 1, img.h - tpl.h + 1);
    let mut out = vec![0f32; ow * oh];
    let r = unsafe { f(img.data.as_ptr(), img.w as i32, img.h as i32, img.w as i32, tpl.data.as_ptr(), tpl.w as i32, tpl.h as i32, tpl.w as i32, method, out.as_mut_ptr()) };
    if r != 0 { None } else { Some((ow, oh, out)) }
}

/// SIFT_create(n).detectAndCompute: keypoints (x, y, size, angle, response, octave) and descriptors.
pub fn sift(gray: &Img, nfeatures: usize) -> Option<(Vec<[f32; 5]>, Vec<i32>, Vec<[f32; 128]>)> {
    let l = lib()?;
    let run = sym!(l, "vsx_sift", unsafe extern "C" fn(*const u8, i32, i32, i32, i32, *mut i32) -> *mut std::ffi::c_void);
    let get = sym!(l, "vsx_sift_get", unsafe extern "C" fn(*mut std::ffi::c_void, *mut f32, *mut f32));
    let free = sym!(l, "vsx_sift_free", unsafe extern "C" fn(*mut std::ffi::c_void));
    let mut n = 0i32;
    let h = unsafe { run(gray.data.as_ptr(), gray.w as i32, gray.h as i32, gray.w as i32, nfeatures as i32, &mut n) };
    if h.is_null() || n < 0 {
        return None;
    }
    let n = n as usize;
    let mut kp = vec![0f32; n * 6];
    let mut d = vec![0f32; n * 128];
    unsafe {
        get(h, kp.as_mut_ptr(), d.as_mut_ptr());
        free(h);
    }
    let k5 = (0..n).map(|i| [kp[i * 6], kp[i * 6 + 1], kp[i * 6 + 2], kp[i * 6 + 3], kp[i * 6 + 4]]).collect();
    let oct = (0..n).map(|i| kp[i * 6 + 5].to_bits() as i32).collect();
    let desc = (0..n).map(|i| { let mut a = [0f32; 128]; a.copy_from_slice(&d[i * 128..(i + 1) * 128]); a }).collect();
    Some((k5, oct, desc))
}

/// BFMatcher(NORM_L2).knnMatch(q, t, k=2): per query [(train, dist); 2] (usize::MAX when absent).
pub fn knn2(q: &[[f32; 128]], t: &[[f32; 128]]) -> Option<Vec<[(usize, f32); 2]>> {
    let l = lib()?;
    let f = sym!(l, "vsx_knn2", unsafe extern "C" fn(*const f32, i32, *const f32, i32, *mut f32) -> i32);
    let mut out = vec![0f32; q.len() * 4];
    let r = unsafe { f(q.as_ptr() as *const f32, q.len() as i32, t.as_ptr() as *const f32, t.len() as i32, out.as_mut_ptr()) };
    if r != 0 {
        return None;
    }
    let ix = |v: f32| if v < 0.0 { usize::MAX } else { v as usize };
    Some((0..q.len()).map(|i| [(ix(out[i * 4]), out[i * 4 + 1]), (ix(out[i * 4 + 2]), if out[i * 4 + 2] < 0.0 { f32::MAX } else { out[i * 4 + 3] })]).collect())
}

/// cv::estimateAffinePartial2D(src, dst, RANSAC, thr): (2 x 3 model, inlier mask).
pub fn affine_partial(src: &[[f32; 2]], dst: &[[f32; 2]], thr: f64) -> Option<Option<([f64; 6], Vec<u8>)>> {
    let l = lib()?;
    let f = sym!(l, "vsx_affine_partial", unsafe extern "C" fn(*const f32, *const f32, i32, f64, *mut f64, *mut u8) -> i32);
    let mut m = [0f64; 6];
    let mut mask = vec![0u8; src.len()];
    let r = unsafe { f(src.as_ptr() as *const f32, dst.as_ptr() as *const f32, src.len() as i32, thr, m.as_mut_ptr(), mask.as_mut_ptr()) };
    match r {
        1 => Some(Some((m, mask))),
        0 => Some(None),
        _ => None,
    }
}

/// cv::findTransformECC(templ, input, warp, MOTION_EUCLIDEAN, ...): Some(Some(rho)),
/// Some(None) when OpenCV raised (Vision Station then keeps the unpolished pose).
pub fn ecc(templ: &Img, input: &Img, warp: &mut [f32; 6], iters: usize, eps: f64, gauss: i32) -> Option<Option<f64>> {
    let l = lib()?;
    let f = sym!(l, "vsx_ecc", unsafe extern "C" fn(*const u8, i32, i32, i32, *const u8, i32, i32, i32, *mut f32, i32, f64, i32, *mut f64) -> i32);
    let mut rho = 0f64;
    let r = unsafe { f(templ.data.as_ptr(), templ.w as i32, templ.h as i32, templ.w as i32, input.data.as_ptr(), input.w as i32, input.h as i32, input.w as i32,
        warp.as_mut_ptr(), iters as i32, eps, gauss, &mut rho) };
    Some(if r == 0 { Some(rho) } else { None })
}

mod log {
    pub fn warn(m: &str) {
        eprintln!("[cv] {}", m);
    }
}
