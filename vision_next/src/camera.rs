//! Basler cameras through pylon's C library, loaded at run time (no SDK is
//! needed to build, and the program starts without one). A GigE camera that
//! discovery cannot see is announced by its address first.

use crate::img::{demosaic, Img};
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_void, CString};
use std::sync::{Arc, Mutex, OnceLock};

type H = *mut c_void;
type R = i32;

#[repr(C)]
struct GrabResult {
    context: *const c_void,
    h_buffer: H,
    p_buffer: *const u8,
    status: c_int,
    payload_type: c_int,
    pixel_type: c_int,
    time_stamp: u64,
    size_x: c_int,
    size_y: c_int,
    offset_x: c_int,
    offset_y: c_int,
    padding_x: c_int,
    padding_y: c_int,
    payload_size: u64,
    error_code: u32,
    block_id: u64,
    _rest: [u8; 256],
}

const GRABBED: c_int = 2;
const MONO8: c_int = 0x0108_0001;
const BAYER_GR8: c_int = 0x0108_0008;
const BAYER_RG8: c_int = 0x0108_0009;
const BAYER_GB8: c_int = 0x0108_000A;
const BAYER_BG8: c_int = 0x0108_000B;
const RGB8: c_int = 0x0218_0014;
const BGR8: c_int = 0x0218_0015;

struct Pylon {
    lib: Library,
}

unsafe impl Send for Pylon {}
unsafe impl Sync for Pylon {}

static PYLON: OnceLock<Result<Pylon, String>> = OnceLock::new();

fn pylon() -> Result<&'static Pylon, String> {
    PYLON.get_or_init(|| unsafe {
        let mut dirs = vec![];
        if let Ok(p) = std::env::var("PYLON_DEV_DIR") {
            dirs.push(std::path::PathBuf::from(p).join("..").join("Runtime").join("x64"));
        }
        dirs.push(std::path::PathBuf::from(r"C:\Program Files\Basler\pylon\Runtime\x64"));
        dirs.push(crate::paths::exe_dir().join("pylon"));
        for d in dirs {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            let dll = rd.filter_map(|e| e.ok()).map(|e| e.path())
                .find(|p| p.file_name().map(|n| n.to_string_lossy().starts_with("PylonC_v")).unwrap_or(false));
            if let Some(dll) = dll {
                let path = std::env::var("PATH").unwrap_or_default();
                std::env::set_var("PATH", format!("{};{}", d.display(), path));
                let lib = Library::new(&dll).map_err(|e| format!("{}: {}", dll.display(), e))?;
                let init: Symbol<unsafe extern "system" fn() -> R> = lib.get(b"PylonInitialize\0").map_err(|e| e.to_string())?;
                let r = init();
                if r != 0 {
                    return Err(format!("PylonInitialize failed ({:#x})", r));
                }
                return Ok(Pylon { lib });
            }
        }
        Err("Basler pylon is not installed".to_string())
    }).as_ref().map_err(|e| e.clone())
}

macro_rules! sym {
    ($p:expr, $name:literal, $t:ty) => {{
        let s: Symbol<$t> = $p.lib.get(concat!($name, "\0").as_bytes()).map_err(|e| format!("{}: {}", $name, e))?;
        s
    }};
}

pub struct Camera {
    dev: H,
    pub name: String,
    /// settings the camera refused when it was opened (absent or not writable)
    pub missed: Vec<String>,
}

unsafe impl Send for Camera {}

impl Drop for Camera {
    fn drop(&mut self) {
        if let Ok(p) = pylon() {
            unsafe {
                if let Ok(close) = p.lib.get::<unsafe extern "system" fn(H) -> R>(b"PylonDeviceClose\0") {
                    close(self.dev);
                }
                if let Ok(destroy) = p.lib.get::<unsafe extern "system" fn(H) -> R>(b"PylonDestroyDevice\0") {
                    destroy(self.dev);
                }
            }
        }
    }
}

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap_or_default()
}

/// One camera's settings, as Settings > Cameras shows them. Empty or absent
/// means "leave the camera's own value" -- reference pictures stay comparable.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct CameraSettings {
    #[serde(default)]
    pub ip: String,
    #[serde(default)]
    pub exposure_us: Option<f64>,
    #[serde(default)]
    pub exposure_auto: Option<String>, // Off | Once | Continuous
    #[serde(default)]
    pub gain: Option<f64>,
    #[serde(default)]
    pub gain_auto: Option<String>,
    #[serde(default)]
    pub white_balance: Option<String>, // Off | Once | Continuous
    #[serde(default)]
    pub pixel_format: Option<String>,
    #[serde(default)]
    pub gamma: Option<f64>,
    #[serde(default)]
    pub black_level: Option<f64>,
    #[serde(default)]
    pub trigger: Option<String>, // Off | On
    #[serde(default)]
    pub trigger_source: Option<String>, // Line1 | Line2 | Software
    #[serde(default)]
    pub trigger_edge: Option<String>, // RisingEdge | FallingEdge
    #[serde(default)]
    pub trigger_delay_us: Option<f64>,
    #[serde(default)]
    pub frame_rate: Option<f64>,
    #[serde(default)]
    pub packet_size: Option<i64>,
    #[serde(default)]
    pub inter_packet_delay: Option<i64>,
    #[serde(default)]
    pub heartbeat_ms: Option<i64>,
    #[serde(default)]
    pub rotation: i32, // 0 | 90 | 180 | 270, done on the picture
    #[serde(default)]
    pub mirror_x: bool,
    #[serde(default)]
    pub mirror_y: bool,
    #[serde(default)]
    pub timeout_ms: Option<u32>,
}

impl Camera {
    pub fn open(index: usize, cfg: &CameraSettings) -> Result<Camera, String> {
        let p = pylon()?;
        unsafe {
            if !cfg.ip.trim().is_empty() {
                let announce = sym!(p, "PylonGigEAnnounceRemoteDevice", unsafe extern "system" fn(*const c_char) -> R);
                let ip = cstr(cfg.ip.trim());
                announce(ip.as_ptr());
            }
            let enumerate = sym!(p, "PylonEnumerateDevices", unsafe extern "system" fn(*mut usize) -> R);
            let mut n = 0usize;
            let r = enumerate(&mut n);
            if r != 0 || n == 0 {
                return Err(if cfg.ip.is_empty() { "no camera found (set its address in Settings)".into() } else { format!("no camera answers at {}", cfg.ip) });
            }
            // the camera with the asked address, else the index'th
            let mut pick = index.min(n - 1);
            if !cfg.ip.trim().is_empty() {
                let info = sym!(p, "PylonGetDeviceInfoHandle", unsafe extern "system" fn(usize, *mut H) -> R);
                let prop = sym!(p, "PylonDeviceInfoGetPropertyValueByName", unsafe extern "system" fn(H, *const c_char, *mut c_char, *mut usize) -> R);
                for i in 0..n {
                    let mut hi: H = std::ptr::null_mut();
                    if info(i, &mut hi) != 0 {
                        continue;
                    }
                    let mut buf = [0u8; 128];
                    let mut len = buf.len();
                    let key = cstr("IpAddress");
                    if prop(hi, key.as_ptr(), buf.as_mut_ptr() as *mut c_char, &mut len) == 0 {
                        let v = String::from_utf8_lossy(&buf[..len.saturating_sub(1).min(127)]).trim_end_matches('\0').to_string();
                        if v == cfg.ip.trim() {
                            pick = i;
                        }
                    }
                }
            }
            let create = sym!(p, "PylonCreateDeviceByIndex", unsafe extern "system" fn(usize, *mut H) -> R);
            let open = sym!(p, "PylonDeviceOpen", unsafe extern "system" fn(H, c_int) -> R);
            let mut dev: H = std::ptr::null_mut();
            let r = create(pick, &mut dev);
            if r != 0 {
                return Err(format!("could not create the camera ({:#x})", r));
            }
            let r = open(dev, 1 | 2);
            if r != 0 {
                let destroy = sym!(p, "PylonDestroyDevice", unsafe extern "system" fn(H) -> R);
                destroy(dev);
                return Err(format!("could not open the camera ({:#x}): is another program using it?", r));
            }
            let mut cam = Camera { dev, name: format!("camera {}", pick), missed: vec![] };
            cam.missed = cam.apply(cfg);
            Ok(cam)
        }
    }

    /// Every setting given, by the feature names of newer and older cameras.
    /// Returns what could not be set.
    pub fn apply(&mut self, c: &CameraSettings) -> Vec<String> {
        let mut missed = vec![];
        let mut set = |names: &[&str], v: String| {
            if !names.iter().any(|n| self.set_enum(n, &v)) {
                missed.push(names[0].to_string());
            }
        };
        if let Some(v) = &c.pixel_format { set(&["PixelFormat"], v.clone()); }
        if let Some(v) = &c.exposure_auto { set(&["ExposureAuto"], v.clone()); }
        if let Some(v) = c.exposure_us {
            if c.exposure_auto.as_deref().unwrap_or("Off") == "Off" {
                set(&["ExposureTime", "ExposureTimeAbs", "ExposureTimeRaw"], format!("{}", v));
            }
        }
        if let Some(v) = &c.gain_auto { set(&["GainAuto"], v.clone()); }
        if let Some(v) = c.gain {
            if c.gain_auto.as_deref().unwrap_or("Off") == "Off" {
                set(&["Gain", "GainRaw"], format!("{}", v));
            }
        }
        if let Some(v) = &c.white_balance { set(&["BalanceWhiteAuto"], v.clone()); }
        if let Some(v) = c.gamma {
            self.set_enum("GammaEnable", "1");
            set(&["Gamma"], format!("{}", v));
        }
        if let Some(v) = c.black_level { set(&["BlackLevel", "BlackLevelRaw"], format!("{}", v)); }
        if let Some(v) = &c.trigger {
            self.set_enum("TriggerSelector", "FrameStart");
            set(&["TriggerMode"], v.clone());
        }
        if let Some(v) = &c.trigger_source { set(&["TriggerSource"], v.clone()); }
        if let Some(v) = &c.trigger_edge { set(&["TriggerActivation"], v.clone()); }
        if let Some(v) = c.trigger_delay_us { set(&["TriggerDelay", "TriggerDelayAbs"], format!("{}", v)); }
        if let Some(v) = c.frame_rate {
            let on = v > 0.0;
            self.set_enum("AcquisitionFrameRateEnable", if on { "1" } else { "0" });
            if on { set(&["AcquisitionFrameRate", "AcquisitionFrameRateAbs"], format!("{}", v)); }
        }
        if let Some(v) = c.packet_size { set(&["GevSCPSPacketSize"], format!("{}", v)); }
        if let Some(v) = c.inter_packet_delay { set(&["GevSCPD"], format!("{}", v)); }
        if let Some(v) = c.heartbeat_ms { set(&["GevHeartbeatTimeout"], format!("{}", v)); }
        missed
    }

    /// A feature's value as text ("" when the camera has none).
    pub fn get(&self, name: &str) -> String {
        let Ok(p) = pylon() else { return String::new() };
        unsafe {
            let Ok(f) = p.lib.get::<unsafe extern "system" fn(H, *const c_char, *mut c_char, *mut usize) -> R>(b"PylonDeviceFeatureToString\0") else { return String::new() };
            let n = cstr(name);
            let mut buf = [0u8; 256];
            let mut len = buf.len();
            if f(self.dev, n.as_ptr(), buf.as_mut_ptr() as *mut c_char, &mut len) != 0 {
                return String::new();
            }
            String::from_utf8_lossy(&buf[..len.saturating_sub(1).min(255)]).trim_end_matches('\0').to_string()
        }
    }

    fn set_enum(&self, name: &str, v: &str) -> bool {
        let Ok(p) = pylon() else { return false };
        unsafe {
            let Ok(f) = p.lib.get::<unsafe extern "system" fn(H, *const c_char, *const c_char) -> R>(b"PylonDeviceFeatureFromString\0") else { return false };
            let (n, val) = (cstr(name), cstr(v));
            f(self.dev, n.as_ptr(), val.as_ptr()) == 0
        }
    }

    fn set_int_or_float(&mut self, int_name: &str, float_name: &str, v: f64) -> bool {
        let Ok(p) = pylon() else { return false };
        unsafe {
            if let Ok(f) = p.lib.get::<unsafe extern "system" fn(H, *const c_char, f64) -> R>(b"PylonDeviceSetFloatFeature\0") {
                let n = cstr(float_name);
                if f(self.dev, n.as_ptr(), v) == 0 {
                    return true;
                }
            }
            if let Ok(f) = p.lib.get::<unsafe extern "system" fn(H, *const c_char, i64) -> R>(b"PylonDeviceSetIntegerFeature\0") {
                let n = cstr(int_name);
                return f(self.dev, n.as_ptr(), v.round() as i64) == 0;
            }
        }
        false
    }

    /// Exposure in microseconds (ExposureTime on newer cameras, ExposureTimeAbs or Raw on older).
    pub fn set_exposure(&mut self, us: f64) -> bool {
        self.set_int_or_float("ExposureTimeRaw", "ExposureTime", us) || self.set_int_or_float("ExposureTimeRaw", "ExposureTimeAbs", us)
    }

    /// White balance once on what the camera sees now, then held (as Vision Station's Balance Once).
    pub fn balance_once(&self) -> Result<(), String> {
        if self.set_enum("BalanceWhiteAuto", "Once") { Ok(()) } else { Err("this camera has no automatic white balance".into()) }
    }

    /// A GenICam integer as pylon writes it: decimal or 0x hex.
    fn int(v: &str) -> u64 {
        let v = v.trim();
        if let Some(h) = v.strip_prefix("0x").or(v.strip_prefix("0X")) { u64::from_str_radix(h, 16).unwrap_or(0) } else { v.parse().unwrap_or(0) }
    }

    fn dotted(v: &str) -> String {
        let v = v.trim();
        if v.matches('.').count() == 3 {
            return v.to_string();
        }
        let n = Self::int(v);
        format!("{}.{}.{}.{}", (n >> 24) & 255, (n >> 16) & 255, (n >> 8) & 255, n & 255)
    }

    /// The address the camera has now and the one it keeps after a power
    /// cycle. None for a camera without IP settings (not GigE).
    pub fn ip_config(&self) -> Option<serde_json::Value> {
        let cur = self.get("GevCurrentIPAddress");
        if cur.is_empty() {
            return None;
        }
        let flag = self.get("GevCurrentIPConfigurationPersistentIP");
        let on = if flag.is_empty() {
            Self::int(&self.get("GevCurrentIPConfiguration")) & 1 == 1
        } else {
            flag == "1" || flag.eq_ignore_ascii_case("true")
        };
        let (ci, cm, cg) = (Self::dotted(&cur), Self::dotted(&self.get("GevCurrentSubnetMask")), Self::dotted(&self.get("GevCurrentDefaultGateway")));
        let (si, sm, sg) = (Self::dotted(&self.get("GevPersistentIPAddress")), Self::dotted(&self.get("GevPersistentSubnetMask")), Self::dotted(&self.get("GevPersistentDefaultGateway")));
        let keeps = on && si == ci && sm == cm;
        Some(serde_json::json!({"current_ip": ci, "current_mask": cm, "current_gateway": cg,
            "stored_ip": si, "stored_mask": sm, "stored_gateway": sg, "stored_on": on, "keeps_current": keeps}))
    }

    /// Store the address it has now in the camera, so it comes back with it after a power cycle.
    pub fn store_ip(&self) -> Result<serde_json::Value, String> {
        let packed = |name: &str| self.get(name);
        if packed("GevCurrentIPAddress").is_empty() {
            return Err("this camera has no IP settings to keep".into());
        }
        for (to, from) in [("GevPersistentIPAddress", "GevCurrentIPAddress"), ("GevPersistentSubnetMask", "GevCurrentSubnetMask"),
                           ("GevPersistentDefaultGateway", "GevCurrentDefaultGateway")] {
            let v = format!("{}", Self::int(&packed(from)));
            if !self.set_enum(to, &v) {
                return Err(format!("the camera did not take {}", to));
            }
        }
        if !self.set_enum("GevCurrentIPConfigurationPersistentIP", "1") {
            let c = Self::int(&self.get("GevCurrentIPConfiguration"));
            if !self.set_enum("GevCurrentIPConfiguration", &format!("{}", c | 1)) {
                eprintln!("[camera] permanent-IP switch not set");
            }
        }
        self.ip_config().ok_or("no IP settings".into())
    }

}

/// A frame as the camera sent it.
#[derive(Clone, Default)]
pub struct Raw {
    pub data: Vec<u8>,
    pub w: usize,
    pub h: usize,
    pub stride: usize,
    pub px: c_int,
}

fn bayer(px: c_int) -> Option<&'static str> {
    match px {
        BAYER_RG8 => Some("RG"),
        BAYER_BG8 => Some("BG"),
        BAYER_GB8 => Some("GB"),
        BAYER_GR8 => Some("GR"),
        _ => None,
    }
}

/// The frame as a picture, full size.
pub fn convert(r: &Raw) -> Result<Img, String> {
    let (w, h, stride, raw) = (r.w, r.h, r.stride, &r.data);
    if let Some(pat) = bayer(r.px) {
        return Ok(demosaic(raw, w, h, stride, pat));
    }
    Ok(match r.px {
        MONO8 => {
            let mut im = Img::new(w, h, 1);
            for y in 0..h {
                im.data[y * w..(y + 1) * w].copy_from_slice(&raw[y * stride..y * stride + w]);
            }
            im
        }
        RGB8 | BGR8 => {
            let mut im = Img::new(w, h, 3);
            for y in 0..h {
                for x in 0..w {
                    let s = y * stride + x * 3;
                    let d = (y * w + x) * 3;
                    if r.px == RGB8 {
                        im.data[d..d + 3].copy_from_slice(&raw[s..s + 3]);
                    } else {
                        im.data[d] = raw[s + 2];
                        im.data[d + 1] = raw[s + 1];
                        im.data[d + 2] = raw[s];
                    }
                }
            }
            im
        }
        other => return Err(format!("pixel format {:#x} is not handled: set Mono8, BayerRG8 or RGB8", other)),
    })
}

/// The frame at half size, for live view: each 2 x 2 Bayer cell is one
/// colour pixel, four times less work than a full demosaic and resize.
pub fn convert_half(r: &Raw) -> Result<Img, String> {
    use rayon::prelude::*;
    let (w2, h2) = (r.w / 2, r.h / 2);
    let Some(pat) = bayer(r.px) else {
        let full = convert(r)?;
        return Ok(full.resize(w2.max(1), h2.max(1)));
    };
    // positions of red and blue in the 2 x 2 cell
    let (rx, ry) = match pat { "RG" => (0, 0), "GR" => (1, 0), "GB" => (0, 1), _ => (1, 1) };
    let (bx, by) = (1 - rx, 1 - ry);
    let mut im = Img::new(w2, h2, 3);
    im.data.par_chunks_mut(w2 * 3).enumerate().for_each(|(y, row)| {
        let r0 = (2 * y) * r.stride;
        let r1 = (2 * y + 1) * r.stride;
        let at = |dx: usize, dy: usize, x: usize| r.data[if dy == 0 { r0 } else { r1 } + 2 * x + dx] as u16;
        for x in 0..w2 {
            let red = at(rx, ry, x);
            let blue = at(bx, by, x);
            let green = (at(bx, ry, x) + at(rx, by, x)) / 2;
            row[x * 3] = red as u8;
            row[x * 3 + 1] = green as u8;
            row[x * 3 + 2] = blue as u8;
        }
    });
    Ok(im)
}

fn orient(mut im: Img, c: &CameraSettings) -> Img {
    if c.mirror_x || c.mirror_y {
        let (w, h, ch) = (im.w, im.h, im.c);
        let src = im.data.clone();
        for y in 0..h {
            for x in 0..w {
                let sx = if c.mirror_x { w - 1 - x } else { x };
                let sy = if c.mirror_y { h - 1 - y } else { y };
                let (d, s) = ((y * w + x) * ch, (sy * w + sx) * ch);
                im.data[d..d + ch].copy_from_slice(&src[s..s + ch]);
            }
        }
    }
    // clockwise quarter turns; Img::rot90 turns anticlockwise
    match c.rotation.rem_euclid(360) {
        90 => im.rot90(3),
        180 => im.rot90(2),
        270 => im.rot90(1),
        _ => im,
    }
}

/// Cameras on the network (and any given by address), with what they say about themselves.
pub fn list_devices(addresses: &[String]) -> Result<Vec<serde_json::Value>, String> {
    let p = pylon()?;
    unsafe {
        let announce = sym!(p, "PylonGigEAnnounceRemoteDevice", unsafe extern "system" fn(*const c_char) -> R);
        for a in addresses.iter().filter(|a| !a.trim().is_empty()) {
            let ip = cstr(a.trim());
            announce(ip.as_ptr());
        }
        let enumerate = sym!(p, "PylonEnumerateDevices", unsafe extern "system" fn(*mut usize) -> R);
        let info = sym!(p, "PylonGetDeviceInfoHandle", unsafe extern "system" fn(usize, *mut H) -> R);
        let prop = sym!(p, "PylonDeviceInfoGetPropertyValueByName", unsafe extern "system" fn(H, *const c_char, *mut c_char, *mut usize) -> R);
        let mut n = 0usize;
        enumerate(&mut n);
        let mut out = vec![];
        for i in 0..n {
            let mut hi: H = std::ptr::null_mut();
            if info(i, &mut hi) != 0 {
                continue;
            }
            let get = |k: &str| -> String {
                let mut buf = [0u8; 256];
                let mut len = buf.len();
                let key = cstr(k);
                if prop(hi, key.as_ptr(), buf.as_mut_ptr() as *mut c_char, &mut len) == 0 {
                    String::from_utf8_lossy(&buf[..len.saturating_sub(1).min(255)]).trim_end_matches('\0').to_string()
                } else {
                    String::new()
                }
            };
            out.push(serde_json::json!({"index": i, "name": get("FriendlyName"), "model": get("ModelName"),
                "serial": get("SerialNumber"), "ip": get("IpAddress"), "class": get("DeviceClass")}));
        }
        Ok(out)
    }
}

/// A camera streaming continuously: the newest frame is kept, a picture
/// request takes a frame that arrives after it. Started on first use, stopped
/// after IDLE_S without requests (the camera is free for other programs).
pub struct Stream {
    st: Mutex<StreamState>,
    cv: std::sync::Condvar,
    stop: std::sync::atomic::AtomicBool,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

#[derive(Default)]
struct StreamState {
    seq: u64,
    /// the last frame live view showed: the next one is any newer frame
    live_seq: u64,
    raw: Raw,
    err: Option<String>,
    running: bool,
    last_use: Option<std::time::Instant>,
}

const IDLE_S: u64 = 20;
const BUFFERS: usize = 4;

unsafe fn stream_body(cam: &Arc<Mutex<Camera>>, s: &Arc<Stream>) -> Result<(), String> {
    let p = pylon()?;
    let dev = cam.lock().unwrap().dev;
    let get_stg = sym!(p, "PylonDeviceGetStreamGrabber", unsafe extern "system" fn(H, usize, *mut H) -> R);
    let open = sym!(p, "PylonStreamGrabberOpen", unsafe extern "system" fn(H) -> R);
    let close = sym!(p, "PylonStreamGrabberClose", unsafe extern "system" fn(H) -> R);
    let get_wait = sym!(p, "PylonStreamGrabberGetWaitObject", unsafe extern "system" fn(H, *mut H) -> R);
    let set_num = sym!(p, "PylonStreamGrabberSetMaxNumBuffer", unsafe extern "system" fn(H, usize) -> R);
    let set_size = sym!(p, "PylonStreamGrabberSetMaxBufferSize", unsafe extern "system" fn(H, usize) -> R);
    let payload = sym!(p, "PylonStreamGrabberGetPayloadSize", unsafe extern "system" fn(H, H, *mut usize) -> R);
    let prepare = sym!(p, "PylonStreamGrabberPrepareGrab", unsafe extern "system" fn(H) -> R);
    let register = sym!(p, "PylonStreamGrabberRegisterBuffer", unsafe extern "system" fn(H, *mut c_void, usize, *mut H) -> R);
    let deregister = sym!(p, "PylonStreamGrabberDeregisterBuffer", unsafe extern "system" fn(H, H) -> R);
    let queue = sym!(p, "PylonStreamGrabberQueueBuffer", unsafe extern "system" fn(H, H, *const c_void) -> R);
    let start_if = sym!(p, "PylonStreamGrabberStartStreamingIfMandatory", unsafe extern "system" fn(H) -> R);
    let stop_if = sym!(p, "PylonStreamGrabberStopStreamingIfMandatory", unsafe extern "system" fn(H) -> R);
    let cancel = sym!(p, "PylonStreamGrabberCancelGrab", unsafe extern "system" fn(H) -> R);
    let finish = sym!(p, "PylonStreamGrabberFinishGrab", unsafe extern "system" fn(H) -> R);
    let retrieve = sym!(p, "PylonStreamGrabberRetrieveResult", unsafe extern "system" fn(H, *mut GrabResult, *mut bool) -> R);
    let wait = sym!(p, "PylonWaitObjectWait", unsafe extern "system" fn(H, u32, *mut bool) -> R);
    let exec = sym!(p, "PylonDeviceExecuteCommandFeature", unsafe extern "system" fn(H, *const c_char) -> R);
    let from_string = sym!(p, "PylonDeviceFeatureFromString", unsafe extern "system" fn(H, *const c_char, *const c_char) -> R);

    let mut stg: H = std::ptr::null_mut();
    if get_stg(dev, 0, &mut stg) != 0 || open(stg) != 0 {
        return Err("the camera's picture stream could not be opened".into());
    }
    let mut wobj: H = std::ptr::null_mut();
    get_wait(stg, &mut wobj);
    let mut size = 0usize;
    if payload(dev, stg, &mut size) != 0 || size == 0 {
        let mut v: i64 = 0;
        let get_int = sym!(p, "PylonDeviceGetIntegerFeature", unsafe extern "system" fn(H, *const c_char, *mut i64) -> R);
        let k = cstr("PayloadSize");
        get_int(dev, k.as_ptr(), &mut v);
        size = v.max(0) as usize;
    }
    if size == 0 {
        close(stg);
        return Err("camera did not report its picture size".into());
    }
    set_num(stg, BUFFERS);
    set_size(stg, size);
    if prepare(stg) != 0 {
        close(stg);
        return Err("the camera's picture stream could not be prepared".into());
    }
    let mut bufs: Vec<Vec<u8>> = (0..BUFFERS).map(|_| vec![0u8; size]).collect();
    let mut handles: Vec<H> = vec![std::ptr::null_mut(); BUFFERS];
    for i in 0..BUFFERS {
        register(stg, bufs[i].as_mut_ptr() as *mut c_void, size, &mut handles[i]);
        queue(stg, handles[i], i as *const c_void);
    }
    let (mode, cont) = (cstr("AcquisitionMode"), cstr("Continuous"));
    from_string(dev, mode.as_ptr(), cont.as_ptr());
    start_if(stg);
    let (startc, stopc) = (cstr("AcquisitionStart"), cstr("AcquisitionStop"));
    exec(dev, startc.as_ptr());
    {
        let mut st = s.st.lock().unwrap();
        st.running = true;
        st.err = None;
    }
    s.cv.notify_all();
    loop {
        if s.stop.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let idle = s.st.lock().unwrap().last_use.map(|t| t.elapsed().as_secs() > IDLE_S).unwrap_or(false);
        if idle {
            break;
        }
        let mut ready = false;
        wait(wobj, 300, &mut ready);
        if !ready {
            continue;
        }
        let mut res: GrabResult = std::mem::zeroed();
        let mut got = false;
        if retrieve(stg, &mut res, &mut got) != 0 || !got {
            continue;
        }
        let i = res.context as usize;
        if res.status == GRABBED && i < BUFFERS {
            let px = res.pixel_type;
            let bpp = if px == RGB8 || px == BGR8 { 3 } else { 1 };
            let (w, h) = (res.size_x.max(0) as usize, res.size_y.max(0) as usize);
            let stride = w * bpp + res.padding_x.max(0) as usize;
            let n = (stride * h).min(size);
            let mut st = s.st.lock().unwrap();
            st.raw.data.clear();
            st.raw.data.extend_from_slice(&bufs[i][..n]);
            st.raw.w = w;
            st.raw.h = h;
            st.raw.stride = stride;
            st.raw.px = px;
            st.seq += 1;
            drop(st);
            s.cv.notify_all();
        }
        queue(stg, res.h_buffer, res.context);
    }
    exec(dev, stopc.as_ptr());
    stop_if(stg);
    cancel(stg);
    loop {
        let mut res: GrabResult = std::mem::zeroed();
        let mut got = false;
        if retrieve(stg, &mut res, &mut got) != 0 || !got {
            break;
        }
    }
    for h in handles {
        deregister(stg, h);
    }
    finish(stg);
    close(stg);
    drop(bufs);
    Ok(())
}

/// Open cameras by index, shared by every side that reads them.
#[derive(Default)]
pub struct Hub {
    cams: Mutex<Vec<Option<Arc<Mutex<Camera>>>>>,
    streams: Mutex<Vec<Option<Arc<Stream>>>>,
    pub settings: Mutex<Vec<CameraSettings>>,
}

impl Hub {
    pub fn get(&self, index: usize) -> Result<Arc<Mutex<Camera>>, String> {
        let mut cams = self.cams.lock().unwrap();
        if cams.len() <= index {
            cams.resize_with(index + 1, || None);
        }
        if let Some(c) = &cams[index] {
            return Ok(c.clone());
        }
        let cfg = self.settings.lock().unwrap().get(index).cloned().unwrap_or_default();
        let cam = Arc::new(Mutex::new(Camera::open(index, &cfg)?));
        cams[index] = Some(cam.clone());
        Ok(cam)
    }

    fn stream(&self, index: usize) -> Result<Arc<Stream>, String> {
        let mut ss = self.streams.lock().unwrap();
        if ss.len() <= index {
            ss.resize_with(index + 1, || None);
        }
        if let Some(s) = &ss[index] {
            let alive = s.thread.lock().unwrap().as_ref().map(|t| !t.is_finished()).unwrap_or(false);
            if alive {
                return Ok(s.clone());
            }
        }
        let cam = self.get(index)?;
        let s = Arc::new(Stream { st: Mutex::new(StreamState::default()), cv: std::sync::Condvar::new(),
            stop: std::sync::atomic::AtomicBool::new(false), thread: Mutex::new(None) });
        s.st.lock().unwrap().last_use = Some(std::time::Instant::now());
        let (c2, s2) = (cam.clone(), s.clone());
        let t = std::thread::spawn(move || {
            let r = unsafe { stream_body(&c2, &s2) };
            let mut st = s2.st.lock().unwrap();
            st.running = false;
            if let Err(e) = r {
                st.err = Some(e);
            }
            drop(st);
            s2.cv.notify_all();
        });
        *s.thread.lock().unwrap() = Some(t);
        ss[index] = Some(s.clone());
        Ok(s)
    }

    /// A frame that arrives after `fresh` - 1 others from now.
    fn frame(&self, index: usize, fresh: u64) -> Result<Raw, String> {
        let cfg = self.settings.lock().unwrap().get(index).cloned().unwrap_or_default();
        let timeout = std::time::Duration::from_millis(cfg.timeout_ms.unwrap_or(5000) as u64);
        let s = self.stream(index)?;
        let mut st = s.st.lock().unwrap();
        st.last_use = Some(std::time::Instant::now());
        let target = st.seq + fresh;
        let t0 = std::time::Instant::now();
        while st.seq < target {
            if let Some(e) = st.err.clone() {
                drop(st);
                self.drop_camera(index);
                return Err(e);
            }
            if t0.elapsed() > timeout {
                return Err(if cfg.trigger.as_deref() == Some("On") { "no picture: the camera is waiting for its trigger".into() } else { "no picture from the camera in time".into() });
            }
            let (g, _) = s.cv.wait_timeout(st, std::time::Duration::from_millis(100)).unwrap();
            st = g;
        }
        Ok(st.raw.clone())
    }

    /// A picture for an inspection: the second frame from now, so light and
    /// exposure switched just before are settled in it.
    pub fn grab(&self, index: usize) -> Result<Img, String> {
        self.grab_fresh(index, 2)
    }

    /// A full picture from the `fresh`-th frame from now.
    pub fn grab_fresh(&self, index: usize, fresh: u64) -> Result<Img, String> {
        let cfg = self.settings.lock().unwrap().get(index).cloned().unwrap_or_default();
        let raw = self.frame(index, fresh.max(1))?;
        Ok(orient(convert(&raw)?, &cfg))
    }

    /// The newest frame not yet shown, at half size, for live view: no
    /// waiting when the camera has already sent one.
    pub fn grab_live(&self, index: usize) -> Result<Img, String> {
        let cfg = self.settings.lock().unwrap().get(index).cloned().unwrap_or_default();
        let s = self.stream(index)?;
        let raw = {
            let mut st = s.st.lock().unwrap();
            st.last_use = Some(std::time::Instant::now());
            let t0 = std::time::Instant::now();
            while st.seq <= st.live_seq || st.raw.data.is_empty() {
                if let Some(e) = st.err.clone() {
                    drop(st);
                    self.drop_camera(index);
                    return Err(e);
                }
                if t0.elapsed() > std::time::Duration::from_millis(cfg.timeout_ms.unwrap_or(5000) as u64) {
                    return Err("no picture from the camera in time".into());
                }
                let (g, _) = s.cv.wait_timeout(st, std::time::Duration::from_millis(100)).unwrap();
                st = g;
            }
            st.live_seq = st.seq;
            st.raw.clone()
        };
        Ok(orient(convert_half(&raw)?, &cfg))
    }

    fn drop_camera(&self, index: usize) {
        if let Some(Some(s)) = self.streams.lock().unwrap().get(index).cloned().map(|x| x) {
            s.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(c) = self.cams.lock().unwrap().get_mut(index) {
            *c = None;
        }
    }

    /// The camera's present values of the settings shown in Settings.
    pub fn values(&self, index: usize) -> Result<serde_json::Value, String> {
        let cam = self.get(index)?;
        let c = cam.lock().unwrap();
        let first = |names: &[&str]| names.iter().map(|n| c.get(n)).find(|v| !v.is_empty()).unwrap_or_default();
        Ok(serde_json::json!({
            "model": first(&["DeviceModelName"]), "serial": first(&["DeviceSerialNumber", "DeviceID"]),
            "width": first(&["Width"]), "height": first(&["Height"]), "pixel_format": first(&["PixelFormat"]),
            "exposure_us": first(&["ExposureTime", "ExposureTimeAbs", "ExposureTimeRaw"]),
            "exposure_auto": first(&["ExposureAuto"]), "gain": first(&["Gain", "GainRaw"]), "gain_auto": first(&["GainAuto"]),
            "white_balance": first(&["BalanceWhiteAuto"]), "gamma": first(&["Gamma"]), "black_level": first(&["BlackLevel", "BlackLevelRaw"]),
            "trigger": first(&["TriggerMode"]), "trigger_source": first(&["TriggerSource"]), "trigger_edge": first(&["TriggerActivation"]),
            "frame_rate": first(&["ResultingFrameRate", "ResultingFrameRateAbs", "AcquisitionFrameRateAbs"]),
            "packet_size": first(&["GevSCPSPacketSize"]), "inter_packet_delay": first(&["GevSCPD"]),
            "temperature": first(&["DeviceTemperature", "TemperatureAbs"]),
        }))
    }

    /// What the camera refused of its settings when it was opened.
    pub fn missed(&self, index: usize) -> Vec<String> {
        self.cams.lock().unwrap().get(index).and_then(|c| c.as_ref()).map(|c| c.lock().unwrap().missed.clone()).unwrap_or_default()
    }

    /// Apply settings to an open camera now (closing it if it is not open).
    pub fn reconfigure(&self) {
        self.close_all();
    }

    pub fn close_all(&self) {
        let streams: Vec<Arc<Stream>> = self.streams.lock().unwrap().drain(..).flatten().collect();
        for s in &streams {
            s.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        for s in streams {
            if let Some(t) = s.thread.lock().unwrap().take() {
                let _ = t.join();
            }
        }
        self.cams.lock().unwrap().clear();
    }

    pub fn status() -> String {
        match pylon() {
            Ok(_) => "pylon ready".into(),
            Err(e) => e,
        }
    }
}
