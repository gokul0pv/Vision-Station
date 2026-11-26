//! What sits around the inspections: settings, the light controller, the
//! line's TCP interface, the barcode rule and scanner, result pictures and
//! the supervisor password. The same behaviour and defaults as Vision Station.

use crate::engine::Inspection;
use crate::img::Img;
use serde_json::{json, Map, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------- settings
pub fn defaults() -> Value {
    json!({
        "general": {"parallel": true, "device": "cpu", "motion": true, "pointer": true, "theme": "light",
                    "accent": "#0A59F7", "queue_failures_only": false, "advanced": true, "model_size": "small"},
        "cameras": [],
        "results": {"which": "none", "folder": "", "subfolder": "", "annotated": true, "labels": true,
                    "banner": true, "original": false, "format": "jpg", "quality": 92,
                    "ok_colour": "#1F6FFF", "ng_colour": "#FF2A2A", "line_scale": 1.0},
        "line": {"enabled": false, "port": 3000, "broadcast": false},
        "light": {"enabled": false, "port": "", "channels": 1, "set_brightness": false,
                  "brightness": [255, 255, 255, 255], "always_on": true, "on_before_ms": 150,
                  "off_after_ms": 300, "auto": false, "target": "reference", "target_percent": 45,
                  "max_clipped": 0.5, "job_levels": {}},
        "barcode": {"enabled": false, "prefix": "", "length": 0, "match_case": false, "auto_run": true,
                    "source": "keyboard", "port": "", "baud": 9600, "folder": ""},
        "operator": {"enabled": false, "password_sha256": ""}
    })
}

/// `over` laid onto `base`, object by object.
pub fn merge(base: &mut Value, over: &Value) {
    match (base, over) {
        (Value::Object(b), Value::Object(o)) => {
            for (k, v) in o {
                match b.get_mut(k) {
                    Some(bv) if bv.is_object() && v.is_object() => merge(bv, v),
                    _ => { b.insert(k.clone(), v.clone()); }
                }
            }
        }
        (b, o) => *b = o.clone(),
    }
}

pub fn load(path: &Path) -> Value {
    let mut s = defaults();
    if let Ok(t) = std::fs::read_to_string(path) {
        if let Ok(v) = serde_json::from_str::<Value>(&t) {
            merge(&mut s, &v);
        }
    }
    // Result pictures follow Vision Station's (same folder, same choice) until
    // they are changed here, so every part's picture is in one place
    s["results"]["from_vs"] = json!(false);
    if !s["results"]["chosen"].as_bool().unwrap_or(false) {
        if let Some(vs) = vs_picture_settings() {
            for k in ["folder", "subfolder", "which", "annotated", "original", "labels", "banner", "format", "quality",
                      "ok_colour", "ng_colour", "line_scale"] {
                if let Some(x) = vs.get(k) {
                    s["results"][k] = x.clone();
                }
            }
            s["results"]["from_vs"] = json!(true);
        }
    }
    s
}

/// Vision Station's Settings > Results (where it keeps them in the
/// registry) when it saves pictures to a folder -- read, never written.
pub fn vs_picture_settings() -> Option<Value> {
    if std::env::var("VS_NO_LEGACY_SETTINGS").is_ok() {
        return None;
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let out = std::process::Command::new("reg")
            .args(["query", r"HKCU\Software\VisionStation\QuickBuild\results", "/v", "saving"])
            .creation_flags(0x0800_0000) // no console window
            .output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        let line = text.lines().find(|l| l.contains("REG_SZ"))?;
        let v: Value = serde_json::from_str(line.split_once("REG_SZ")?.1.trim()).ok()?;
        let f = v["folder"].as_str()?.trim();
        return (!f.is_empty()).then_some(v);
    }
    #[cfg(not(windows))]
    None
}

pub fn save(path: &Path, s: &Value) -> Result<(), String> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(s).unwrap()).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

// ------------------------------------------------------------------ SHA-256
pub fn sha256(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    h.iter().map(|x| format!("{:08x}", x)).collect()
}

// -------------------------------------------------------------- serial port
#[cfg(windows)]
mod com {
    use std::os::windows::io::AsRawHandle;

    #[repr(C)]
    #[derive(Default)]
    struct Dcb {
        len: u32, baud: u32, flags: u32, reserved: u16, xon_lim: u16, xoff_lim: u16,
        byte_size: u8, parity: u8, stop_bits: u8, xon: i8, xoff: i8, err: i8, eof: i8, evt: i8, reserved1: u16,
    }
    #[repr(C)]
    struct Timeouts { interval: u32, read_mult: u32, read_const: u32, write_mult: u32, write_const: u32 }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCommState(h: isize, dcb: *mut Dcb) -> i32;
        fn SetCommState(h: isize, dcb: *const Dcb) -> i32;
        fn SetCommTimeouts(h: isize, t: *const Timeouts) -> i32;
        fn PurgeComm(h: isize, flags: u32) -> i32;
        fn QueryDosDeviceW(name: *const u16, target: *mut u16, max: u32) -> u32;
    }

    pub fn open(name: &str, baud: u32, timeout_ms: u32) -> Result<std::fs::File, String> {
        let path = format!(r"\\.\{}", name.trim());
        let f = std::fs::OpenOptions::new().read(true).write(true).open(&path).map_err(|e| format!("{}: {}", name, e))?;
        let h = f.as_raw_handle() as isize;
        unsafe {
            let mut d = Dcb { len: std::mem::size_of::<Dcb>() as u32, ..Default::default() };
            if GetCommState(h, &mut d) == 0 {
                return Err(format!("{} is not a serial port", name));
            }
            d.baud = baud;
            // binary, DTR and RTS on (as pyserial opens a port: some USB adapters
            // and controllers stay silent without them), no flow control
            d.flags = 0x0001 | 0x0010 | 0x1000;
            d.byte_size = 8;
            d.parity = 0;
            d.stop_bits = 0;
            if SetCommState(h, &d) == 0 {
                return Err(format!("{}: could not set {} baud", name, baud));
            }
            let t = Timeouts { interval: 50, read_mult: 0, read_const: timeout_ms, write_mult: 0, write_const: 500 };
            SetCommTimeouts(h, &t);
            PurgeComm(h, 0x0004 | 0x0008);
        }
        Ok(f)
    }

    pub fn list() -> Vec<String> {
        let mut out = vec![];
        let mut buf = vec![0u16; 1024];
        for i in 1..=64 {
            let name: Vec<u16> = format!("COM{}", i).encode_utf16().chain(std::iter::once(0)).collect();
            if unsafe { QueryDosDeviceW(name.as_ptr(), buf.as_mut_ptr(), buf.len() as u32) } != 0 {
                out.push(format!("COM{}", i));
            }
        }
        out
    }
}

#[cfg(not(windows))]
mod com {
    pub fn open(name: &str, _b: u32, _t: u32) -> Result<std::fs::File, String> {
        std::fs::OpenOptions::new().read(true).write(true).open(name).map_err(|e| e.to_string())
    }
    pub fn list() -> Vec<String> { vec![] }
}

pub fn serial_ports() -> Vec<String> {
    com::list()
}

// ------------------------------------------------- automatic light level
// As Vision Station's light.py: the picture's mean grey aimed at the teach
// picture's (or a fixed share of white), never more than a little of it burnt out.
const CLIP_GREY: u8 = 250;
const DEADBAND: f64 = 0.06;
const MAX_STEP: f64 = 0.35;

/// (mean grey 0-1, share of pixels with any colour at white 0-1).
pub fn light_measure(img: &Img) -> Option<(f64, f64)> {
    if img.w == 0 || img.h == 0 {
        return None;
    }
    let (mut sum, mut hot, mut n) = (0f64, 0usize, 0usize);
    for y in (0..img.h).step_by(4) {
        for x in (0..img.w).step_by(4) {
            let i = (y * img.w + x) * img.c;
            let px = &img.data[i..i + img.c];
            let (g, mx) = if img.c >= 3 {
                (0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64, px[0].max(px[1]).max(px[2]))
            } else {
                (px[0] as f64, px[0])
            };
            sum += g;
            if mx >= CLIP_GREY {
                hot += 1;
            }
            n += 1;
        }
    }
    Some((sum / n as f64 / 255.0, hot as f64 / n as f64))
}

/// The mean grey to aim for: the teach picture's, or the fixed one.
pub fn light_target(cfg: &Value, reference: Option<&Img>) -> f64 {
    if cfg["target"].as_str().unwrap_or("reference") == "reference" {
        if let Some(m) = reference.and_then(light_measure) {
            return m.0;
        }
    }
    (cfg["target_percent"].as_f64().unwrap_or(45.0) / 100.0).clamp(0.02, 0.95)
}

/// The share of the picture allowed at white: as much as the teach picture has (a shiny part).
pub fn light_clip_limit(cfg: &Value, reference: Option<&Img>) -> f64 {
    let base = cfg["max_clipped"].as_f64().unwrap_or(0.5) / 100.0;
    if cfg["target"].as_str().unwrap_or("reference") == "reference" {
        if let Some(m) = reference.and_then(light_measure) {
            return base.max(m.1 * 1.25 + 0.002);
        }
    }
    base
}

/// The level for the next picture from this one (Keep adjusted).
pub fn light_next_level(level: u32, mean: f64, clipped: f64, target: f64, max_clipped: f64) -> u32 {
    let level = level.clamp(1, 255);
    if clipped > max_clipped {
        return ((level as f64 * 0.85) as u32).min(level - 1).max(1);
    }
    if (mean - target).abs() <= DEADBAND * target.max(0.05) {
        return level;
    }
    let ratio = (target / mean.max(0.004)).clamp(1.0 - MAX_STEP, 1.0 + MAX_STEP);
    let mut new = (level as f64 * ratio).round() as i64;
    if new == level as i64 {
        new += if ratio > 1.0 { 1 } else { -1 };
    }
    new.clamp(1, 255) as u32
}

/// Adjust now: a binary search over 1-255, one fresh picture per try.
/// Returns (level, mean, clipped, tries).
pub fn light_auto_level(mut set_level: impl FnMut(u32) -> Result<(), String>, mut grab: impl FnMut() -> Result<Img, String>,
                        target: f64, max_clipped: f64) -> Result<(u32, f64, f64, Vec<(u32, f64, f64)>), String> {
    let (mut lo, mut hi) = (1u32, 255u32);
    let mut tries: Vec<(u32, f64, f64)> = vec![];
    for _ in 0..9 {
        let mid = (lo + hi) / 2;
        set_level(mid)?;
        let m = light_measure(&grab()?).ok_or("the camera gave no picture")?;
        tries.push((mid, m.0, m.1));
        if m.1 > max_clipped || m.0 > target { hi = mid } else { lo = mid }
        if hi - lo <= 1 {
            break;
        }
    }
    // out of reach: the brightest (every try too dark) or the dimmest (every try too bright)
    let too_dark = |t: &(u32, f64, f64)| t.1 < target && t.2 <= max_clipped;
    let too_bright = |t: &(u32, f64, f64)| t.1 > target || t.2 > max_clipped;
    for (end, dark) in [(255u32, true), (1u32, false)] {
        let all = |ts: &Vec<(u32, f64, f64)>| ts.iter().all(|t| if dark { too_dark(t) } else { too_bright(t) });
        if all(&tries) && !tries.iter().any(|t| t.0 == end) {
            set_level(end)?;
            if let Some(m) = light_measure(&grab()?) {
                tries.push((end, m.0, m.1));
            }
            if all(&tries) {
                let l = *tries.last().unwrap();
                return Ok((end, l.1, l.2, tries));
            }
        }
    }
    let ok: Vec<_> = tries.iter().filter(|t| t.2 <= max_clipped).cloned().collect();
    let pool = if ok.is_empty() { tries.clone() } else { ok };
    let best = pool.into_iter().min_by(|a, b| (a.1 - target).abs().partial_cmp(&(b.1 - target).abs()).unwrap()).unwrap();
    set_level(best.0)?;
    Ok((best.0, best.1, best.2, tries))
}

// ----------------------------------------------------------- light controller
/// HIFLY DCS / OPT protocol: `$ cmd channel data(3 hex) checksum(2 hex)`,
/// 9600 8N1. 1 on, 2 off, 3 set brightness, 4 read brightness.
pub fn light_frame(cmd: u8, channel: u8, data: u32) -> Vec<u8> {
    let body = format!("${}{}{:03X}", cmd, channel, data.min(0xFFF));
    let x = body.bytes().fold(0u8, |a, b| a ^ b);
    format!("{}{:02X}", body, x).into_bytes()
}

pub struct Light {
    port: Option<std::fs::File>,
    pub cfg: Value,
    holds: usize,
    lit: bool,
    pub state: String,
    pub levels: Vec<u32>,
    /// The controller's own brightness per channel (1-based index - 1) as it
    /// was before Vision Next first changed anything; handed back at close.
    home: Option<Vec<Option<u32>>>,
    taking_home: bool,
    /// The brightness last sent to every channel (None: not sent yet).
    pub sent: Option<u32>,
}

/// Where the controller's own levels are kept while Vision Next has the
/// light: a session that is killed (VS Code's Stop) never hands back, and
/// the controller keeps whatever brightness it was given, so the next start
/// must not read that as the operator's own level.
fn light_home_path() -> PathBuf {
    crate::paths::projects().join("light_home.json")
}

/// The levels Vision Station noted as the controller's own before it changed
/// the light and has not handed back (its light settings in the registry:
/// in_use, home_levels, home_port), for this port and these channels.
#[cfg(windows)]
fn vision_station_owes(port: &str, channels: u8) -> Option<Vec<Option<u32>>> {
    use std::os::windows::process::CommandExt;
    let out = std::process::Command::new("reg")
        .args(["query", r"HKCU\Software\VisionStation\QuickBuild\light", "/v", "settings"])
        .creation_flags(0x0800_0000).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let json_part = text.split("REG_SZ").nth(1)?.trim();
    let v: Value = serde_json::from_str(json_part).ok()?;
    if v["in_use"].as_bool() != Some(true) || v["home_port"].as_str() != Some(port) {
        return None;
    }
    let lv = v["home_levels"].as_object()?;
    let home: Vec<Option<u32>> = (1..=channels).map(|ch| lv.get(&ch.to_string()).and_then(|x| x.as_u64()).map(|n| n as u32)).collect();
    if home.iter().all(|x| x.is_none()) { None } else { Some(home) }
}

#[cfg(not(windows))]
fn vision_station_owes(_: &str, _: u8) -> Option<Vec<Option<u32>>> {
    None
}

impl Light {
    pub fn new() -> Light {
        Light { port: None, cfg: defaults()["light"].clone(), holds: 0, lit: false, state: "unknown".into(), levels: vec![], home: None, taking_home: false, sent: None }
    }

    fn talk(&mut self, frame: &[u8], answer: usize) -> Result<Vec<u8>, String> {
        let name = self.cfg["port"].as_str().unwrap_or("").to_string();
        if name.is_empty() {
            return Err("choose the light controller's serial port first".into());
        }
        if self.port.is_none() {
            self.port = Some(com::open(&name, 9600, 300).map_err(|e| {
                if e.contains("os error 5") || e.to_lowercase().contains("access is denied") {
                    format!("{} is in use by another program (Vision Station, or a second Vision Next window?) - close it and try again", name)
                } else if e.contains("os error 2") {
                    format!("{} is not on this PC (unplugged?)", name)
                } else {
                    e
                }
            })?);
        }
        let p = self.port.as_mut().unwrap();
        let r = (|| -> Result<Vec<u8>, String> {
            p.write_all(frame).map_err(|e| e.to_string())?;
            let mut got = vec![];
            let t0 = Instant::now();
            let mut b = [0u8; 32];
            while got.len() < answer && t0.elapsed() < Duration::from_millis(400) {
                match p.read(&mut b) {
                    Ok(0) => {}
                    Ok(n) => {
                        got.extend_from_slice(&b[..n]);
                        if got.first() == Some(&b'&') || (answer == 1 && !got.is_empty()) {
                            break;
                        }
                    }
                    Err(e) => return Err(e.to_string()),
                }
            }
            Ok(got)
        })();
        if r.is_err() {
            self.port = None;
        }
        self.port = None; // closed: the next command opens it again
        let got = r.map_err(|e| format!("the light controller on {} stopped answering: {}", name, e))?;
        if got.first() == Some(&b'&') {
            return Err("the controller refused the command".into());
        }
        if got.is_empty() {
            return Err(format!("no answer from a light controller on {} - is it switched on, and is this its port?", name));
        }
        Ok(got)
    }

    fn channels(&self) -> u8 {
        self.cfg["channels"].as_u64().unwrap_or(1).clamp(1, 4) as u8
    }

    /// Before the first change: the controller's own levels, read from it --
    /// unless the last session never handed back, then the levels it kept.
    fn ensure_home(&mut self) {
        if self.home.is_some() || self.taking_home {
            return;
        }
        let port = self.cfg["port"].as_str().unwrap_or("").to_string();
        if port.is_empty() {
            return;
        }
        let saved = std::fs::read_to_string(light_home_path()).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok());
        let owed = saved.as_ref().filter(|v| v["in_use"].as_bool() == Some(true) && v["port"].as_str() == Some(port.as_str()));
        let home: Vec<Option<u32>> = if let Some(v) = owed {
            v["levels"].as_array().map(|a| a.iter().map(|x| x.as_u64().map(|n| n as u32)).collect()).unwrap_or_default()
        } else if let Some(v) = vision_station_owes(&port, self.channels()) {
            // Vision Station was stopped without handing the light back: its
            // note of the operator's level, not what it left in the controller
            v
        } else {
            self.taking_home = true;
            let mut out = vec![];
            for ch in 1..=self.channels() {
                out.push(self.read_one(ch).ok());
            }
            self.taking_home = false;
            out
        };
        if home.iter().all(|x| x.is_none()) {
            return; // nothing readable: nothing to hand back
        }
        let _ = std::fs::write(light_home_path(), serde_json::to_string_pretty(&json!({"port": port, "levels": home, "in_use": true})).unwrap_or_default());
        self.home = Some(home);
    }

    /// The controller back as it was found: its own brightness, light on --
    /// its state whenever the application is not running (as Vision Station).
    pub fn hand_back(&mut self) {
        let Some(home) = self.home.clone() else { return };
        let mut ok = true;
        for (i, lv) in home.iter().enumerate() {
            let ch = i as u8 + 1;
            // a channel that could not be read (the controller has fewer) is left alone
            if let Some(v) = lv {
                ok &= self.talk(&light_frame(3, ch, (*v).min(255)), 1).is_ok();
                ok &= self.talk(&light_frame(1, ch, 0), 1).is_ok();
            }
        }
        if ok {
            self.home = None;
            self.lit = true;
            self.state = "handed back".into();
            let port = self.cfg["port"].as_str().unwrap_or("");
            let _ = std::fs::write(light_home_path(), serde_json::to_string_pretty(&json!({"port": port, "levels": home, "in_use": false})).unwrap_or_default());
        }
    }

    fn read_one(&mut self, ch: u8) -> Result<u32, String> {
        let a = self.talk(&light_frame(4, ch, 0), 8)?;
        let s = String::from_utf8_lossy(&a).to_string();
        s.get(3..6).and_then(|h| u32::from_str_radix(h, 16).ok()).ok_or(format!("unreadable answer '{}'", s))
    }

    pub fn switch(&mut self, on: bool) -> Result<(), String> {
        self.ensure_home();
        for ch in 1..=self.channels() {
            self.talk(&light_frame(if on { 1 } else { 2 }, ch, 0), 1)?;
        }
        self.lit = on;
        self.state = if on { "on".into() } else { "off".into() };
        Ok(())
    }

    pub fn set_level(&mut self, channel: u8, value: u32) -> Result<(), String> {
        self.ensure_home();
        self.talk(&light_frame(3, channel, value.min(255)), 1)?;
        if channel == 1 {
            self.sent = Some(value.min(255));
        }
        Ok(())
    }

    /// Every channel to this brightness now (a side's level, a level being tried).
    /// A brightness sent while the light is off leaves it off (measured on the DCS2).
    pub fn send_level(&mut self, value: u32) -> Result<(), String> {
        for ch in 1..=self.channels() {
            self.set_level(ch, value)?;
        }
        Ok(())
    }

    /// The level a side's pictures are taken at: Keep adjusted's for it, its
    /// recipe's taught level, or the Brightness setting; None leaves the light alone.
    pub fn side_level(&self, side: &str, taught: Option<u32>) -> Option<u32> {
        if let Some(v) = self.cfg["job_levels"][side].as_u64() {
            return Some(v as u32);
        }
        if taught.is_some() {
            return taught;
        }
        if self.cfg["set_brightness"].as_bool().unwrap_or(false) {
            return Some(self.cfg["brightness"][0].as_u64().unwrap_or(255) as u32);
        }
        None
    }

    /// Before a side's picture: its level, sent when the light does not have it already.
    pub fn use_side_level(&mut self, side: &str, taught: Option<u32>) {
        if !self.enabled() {
            return;
        }
        if let Some(lv) = self.side_level(side, taught) {
            if self.sent != Some(lv) {
                let _ = self.send_level(lv);
            }
        }
    }

    pub fn read(&mut self) -> Result<Vec<u32>, String> {
        let mut out = vec![];
        for ch in 1..=self.channels() {
            out.push(self.read_one(ch)?);
        }
        self.levels = out.clone();
        Ok(out)
    }

    pub fn configure(&mut self, cfg: &Value) {
        let changed_port = cfg["port"] != self.cfg["port"];
        // control switched off or moved to another port: the old controller back first
        if changed_port || !cfg["enabled"].as_bool().unwrap_or(false) {
            self.hand_back();
        }
        self.cfg = cfg.clone();
        if changed_port {
            self.port = None;
        }
        if !self.enabled() {
            return;
        }
        self.sent = None;
        if self.cfg["set_brightness"].as_bool().unwrap_or(false) {
            let levels: Vec<u32> = self.cfg["brightness"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap_or(255) as u32).collect()).unwrap_or_default();
            for ch in 1..=self.channels() {
                let v = levels.get(ch as usize - 1).copied().unwrap_or(255);
                if let Err(e) = self.set_level(ch, v) {
                    self.state = e;
                    return;
                }
            }
        }
        let on = self.cfg["always_on"].as_bool().unwrap_or(true);
        if let Err(e) = self.switch(on) {
            self.state = e;
        }
    }

    /// The port and channels from the settings, without switching anything
    /// (a button pressed straight after choosing the port must use it).
    pub fn sync(&mut self, cfg: &Value) {
        if cfg["port"] != self.cfg["port"] {
            self.port = None;
        }
        self.cfg = cfg.clone();
    }

    pub fn enabled(&self) -> bool {
        self.cfg["enabled"].as_bool().unwrap_or(false)
    }

    /// On for a picture (or a continuous run); off again when no one holds it.
    pub fn hold(&mut self) -> bool {
        if !self.enabled() {
            return false;
        }
        self.holds += 1;
        if !self.lit && self.switch(true).is_ok() {
            std::thread::sleep(Duration::from_millis(self.cfg["on_before_ms"].as_u64().unwrap_or(150)));
            return true;
        }
        false
    }

    pub fn release(&mut self) {
        if !self.enabled() {
            return;
        }
        self.holds = self.holds.saturating_sub(1);
        if self.holds == 0 && !self.cfg["always_on"].as_bool().unwrap_or(true) {
            let _ = self.switch(false);
        }
    }
}

impl Default for Light {
    fn default() -> Self {
        Light::new()
    }
}

// ------------------------------------------------------------------ barcode
pub fn barcode_clean(raw: &str) -> String {
    raw.chars().filter(|c| !c.is_control()).collect::<String>().trim().to_string()
}

/// Ok(code as kept) or Err(why it is not good).
pub fn barcode_check(cfg: &Value, raw: &str) -> Result<String, String> {
    let code = barcode_clean(raw);
    if code.is_empty() {
        return Err("empty code".into());
    }
    let prefix = cfg["prefix"].as_str().unwrap_or("");
    let len = cfg["length"].as_u64().unwrap_or(0) as usize;
    let case = cfg["match_case"].as_bool().unwrap_or(false);
    let starts = if case { code.starts_with(prefix) } else { code.to_uppercase().starts_with(&prefix.to_uppercase()) };
    if !prefix.is_empty() && !starts {
        return Err(format!("{} does not start with {}", code, prefix));
    }
    if len > 0 && code.chars().count() != len {
        return Err(format!("{} has {} characters, not {}", code, code.chars().count(), len));
    }
    Ok(code)
}

pub fn barcode_rule(cfg: &Value) -> String {
    let mut parts = vec![];
    if let Some(p) = cfg["prefix"].as_str().filter(|p| !p.is_empty()) {
        parts.push(format!("starts with {}", p));
    }
    if let Some(n) = cfg["length"].as_u64().filter(|&n| n > 0) {
        parts.push(format!("{} characters", n));
    }
    if parts.is_empty() { "any code".into() } else { parts.join(", ") }
}

/// Lines from a serial scanner, sent to `got`, until `stop` is set.
pub fn serial_reader(port: String, baud: u32, got: mpsc::Sender<String>, stop: Arc<Mutex<bool>>) {
    std::thread::spawn(move || {
        while !*stop.lock().unwrap() {
            match com::open(&port, baud, 200) {
                Ok(f) => {
                    let mut r = BufReader::new(f);
                    let mut line = String::new();
                    while !*stop.lock().unwrap() {
                        line.clear();
                        match r.read_line(&mut line) {
                            Ok(n) if n > 0 && line.ends_with(['\n', '\r']) => {
                                let c = barcode_clean(&line);
                                if !c.is_empty() && got.send(c).is_err() {
                                    return;
                                }
                            }
                            Ok(_) => {}
                            Err(_) => break,
                        }
                    }
                }
                Err(_) => std::thread::sleep(Duration::from_secs(2)),
            }
        }
    });
}

// ------------------------------------------------------------ line commands
pub type Clients = Arc<Mutex<Vec<std::net::TcpStream>>>;

pub fn broadcast(clients: &Clients, text: &str) {
    let mut c = clients.lock().unwrap();
    c.retain_mut(|s| s.write_all(text.as_bytes()).is_ok());
}

/// Listen on `port`; each line is answered by `answer` (a command -> reply).
pub fn line_server(port: u16, clients: Clients, stop: Arc<Mutex<bool>>, answer: Arc<dyn Fn(&str) -> String + Send + Sync>) -> Result<(), String> {
    let listener = std::net::TcpListener::bind(("0.0.0.0", port)).map_err(|e| format!("port {}: {}", port, e))?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    std::thread::spawn(move || {
        while !*stop.lock().unwrap() {
            match listener.accept() {
                Ok((s, _)) => {
                    let _ = s.set_nonblocking(false);
                    if let Ok(w) = s.try_clone() {
                        clients.lock().unwrap().push(w);
                    }
                    let (answer, stop) = (answer.clone(), stop.clone());
                    std::thread::spawn(move || {
                        let mut w = match s.try_clone() { Ok(w) => w, Err(_) => return };
                        let r = BufReader::new(s);
                        for line in r.lines() {
                            if *stop.lock().unwrap() {
                                break;
                            }
                            let Ok(line) = line else { break };
                            let cmd = line.trim();
                            if cmd.is_empty() {
                                continue;
                            }
                            let reply = answer(cmd);
                            if w.write_all(reply.as_bytes()).is_err() {
                                break;
                            }
                        }
                    });
                }
                Err(_) => std::thread::sleep(Duration::from_millis(100)),
            }
        }
    });
    Ok(())
}

// ---------------------------------------------------------- result pictures
fn colour(hex: &str, d: [u8; 3]) -> [u8; 3] {
    let h = hex.trim_start_matches('#');
    if h.len() != 6 {
        return d;
    }
    let p = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).unwrap_or(0);
    [p(0), p(2), p(4)]
}

fn font() -> Option<&'static ab_glyph::FontRef<'static>> {
    static F: std::sync::OnceLock<Option<ab_glyph::FontRef<'static>>> = std::sync::OnceLock::new();
    F.get_or_init(|| ab_glyph::FontRef::try_from_slice(include_bytes!("../ui/fonts/Inter-SemiBold.ttf")).ok()).as_ref()
}

fn put(img: &mut Img, x: i64, y: i64, c: [u8; 3], a: f32) {
    if x < 0 || y < 0 || x >= img.w as i64 || y >= img.h as i64 {
        return;
    }
    let i = (y as usize * img.w + x as usize) * 3;
    for k in 0..3 {
        img.data[i + k] = (img.data[i + k] as f32 * (1.0 - a) + c[k] as f32 * a) as u8;
    }
}

pub fn fill_rect(img: &mut Img, x0: i64, y0: i64, w: i64, h: i64, c: [u8; 3], a: f32) {
    for y in y0.max(0)..(y0 + h).min(img.h as i64) {
        for x in x0.max(0)..(x0 + w).min(img.w as i64) {
            put(img, x, y, c, a);
        }
    }
}

pub fn line(img: &mut Img, a: [f64; 2], b: [f64; 2], t: f64, c: [u8; 3]) {
    let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt().max(1.0);
    let n = (len * 1.5) as usize + 1;
    let r = (t / 2.0).max(0.5);
    let ri = r.ceil() as i64;
    for i in 0..=n {
        let f = i as f64 / n as f64;
        let (x, y) = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f);
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                if ((dx * dx + dy * dy) as f64) <= r * r {
                    put(img, x as i64 + dx, y as i64 + dy, c, 1.0);
                }
            }
        }
    }
}

/// Text at (x, y) top-left, `px` high. Returns its width.
pub fn text(img: &mut Img, s: &str, x: f64, y: f64, px: f32, c: [u8; 3]) -> f64 {
    use ab_glyph::{Font, ScaleFont};
    let Some(f) = font() else { return 0.0 };
    let sf = f.as_scaled(px);
    let mut caret = x as f32;
    for ch in s.chars() {
        let id = f.glyph_id(ch);
        let g = id.with_scale_and_position(px, ab_glyph::point(caret, y as f32 + sf.ascent()));
        caret += sf.h_advance(id);
        if let Some(o) = f.outline_glyph(g) {
            let bb = o.px_bounds();
            o.draw(|gx, gy, cov| put(img, bb.min.x as i64 + gx as i64, bb.min.y as i64 + gy as i64, c, cov.min(1.0)));
        }
    }
    caret as f64 - x
}

pub fn text_width(s: &str, px: f32) -> f64 {
    use ab_glyph::{Font, ScaleFont};
    let Some(f) = font() else { return 0.0 };
    let sf = f.as_scaled(px);
    s.chars().map(|ch| sf.h_advance(f.glyph_id(ch)) as f64).sum()
}

/// The picture with every check's outline and result, and a verdict banner.
pub fn annotate(frame: &Img, r: &Inspection, cfg: &Value, title: &str) -> Img {
    let mut img = frame.to_rgb();
    let ok = colour(cfg["ok_colour"].as_str().unwrap_or("#1F6FFF"), [31, 111, 255]);
    let ng = colour(cfg["ng_colour"].as_str().unwrap_or("#FF2A2A"), [255, 42, 42]);
    let warn = [255, 159, 10];
    let long = img.w.max(img.h) as f64;
    let t = (long / 520.0 * cfg["line_scale"].as_f64().unwrap_or(1.0)).max(1.5);
    let px = (long / 90.0).clamp(12.0, 64.0) as f32;
    let col = |s: &str| match s { "pass" => ok, "fail" => ng, "skip" => [150, 160, 175], _ => warn };
    for tr in &r.tools {
        for g in &tr.graphics {
            let c = col(if g.status.is_empty() { &tr.status } else { &g.status });
            match g.kind.as_str() {
                "point" => {
                    let p = g.points[0];
                    fill_rect(&mut img, (p[0] - t * 1.5) as i64, (p[1] - t * 1.5) as i64, (t * 3.0) as i64, (t * 3.0) as i64, c, 1.0);
                }
                "line" if g.points.len() >= 2 => line(&mut img, g.points[0], g.points[1], t, c),
                _ => {
                    let n = g.points.len();
                    for i in 0..n {
                        if n > 1 {
                            line(&mut img, g.points[i], g.points[(i + 1) % n], t, c);
                        }
                    }
                }
            }
        }
        if cfg["labels"].as_bool().unwrap_or(true) {
            if let Some(g) = tr.graphics.iter().find(|g| g.role == "region") {
                let x = g.points.iter().map(|p| p[0]).fold(f64::MAX, f64::min);
                let y = g.points.iter().map(|p| p[1]).fold(f64::MAX, f64::min);
                let word = match tr.status.as_str() { "pass" => "OK", "fail" => "NG", "skip" => "—", _ => "ERROR" };
                let s = match tr.caption.first() { Some(c) if !c.is_empty() => format!("{} {}", word, c), _ => word.to_string() };
                let s: String = s.chars().take(40).collect();
                let w = text_width(&s, px * 0.8) + px as f64 * 0.5;
                let h = px as f64 * 1.1;
                fill_rect(&mut img, x as i64, (y - h - t) as i64, w as i64, h as i64, col(&tr.status), 0.92);
                text(&mut img, &s, x + px as f64 * 0.25, y - h - t + px as f64 * 0.1, px * 0.8, [255, 255, 255]);
            }
        }
    }
    if cfg["banner"].as_bool().unwrap_or(true) {
        let bh = (px * 2.2) as i64;
        let c = if r.verdict == "PASS" { ok } else { ng };
        let iw = img.w as i64;
        fill_rect(&mut img, 0, 0, iw, bh, [12, 14, 18], 0.78);
        let word = if r.verdict == "PASS" { "OK" } else { "NG" };
        let w = text(&mut img, word, px as f64 * 0.6, px as f64 * 0.45, px * 1.3, c);
        text(&mut img, &format!("{}   {:.0} ms   {}", title, r.total_ms, r.source), px as f64 * 1.4 + w, px as f64 * 0.65, px * 0.9, [235, 238, 245]);
    }
    img
}

pub fn wanted(cfg: &Value, verdict: &str) -> bool {
    match cfg["which"].as_str().unwrap_or("none") {
        "all" => true,
        "fail" => verdict != "PASS",
        "pass" => verdict == "PASS",
        _ => false,
    }
}

fn safe(s: &str) -> String {
    s.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

/// Write a result's pictures where Settings > Pictures says. Returns the paths.
/// Where a run's result pictures go: (annotated, original), each None when
/// not wanted. Known before writing, so the history can point at them.
pub fn picture_paths(cfg: &Value, r: &Inspection, job: &str, default_root: &Path, barcode: Option<&str>) -> (Option<PathBuf>, Option<PathBuf>) {
    if !wanted(cfg, &r.verdict) {
        return (None, None);
    }
    let root = match cfg["folder"].as_str() {
        Some(f) if !f.trim().is_empty() => PathBuf::from(f),
        _ => default_root.to_path_buf(),
    };
    let mut dir = root.join(crate::engine::day(r.timestamp));
    if let Some(sub) = cfg["subfolder"].as_str().filter(|s| !s.is_empty()) {
        dir = dir.join(safe(sub));
    }
    if let Some(code) = barcode {
        dir = dir.join(safe(code));
    }
    let (_, hh, mm, ss, ms) = crate::engine::local_time(r.timestamp);
    let stem = format!("{:02}{:02}{:02}_{:03}_{}_{}", hh, mm, ss, ms,
        safe(job), if r.verdict == "PASS" { "ACCEPT" } else { "REJECT" });
    let ext = if cfg["format"].as_str() == Some("png") { "png" } else { "jpg" };
    let a = cfg["annotated"].as_bool().unwrap_or(true).then(|| dir.join(format!("{}.{}", stem, ext)));
    let o = cfg["original"].as_bool().unwrap_or(false).then(|| dir.join(format!("{}_original.{}", stem, ext)));
    (a, o)
}

pub fn write_pictures(cfg: &Value, frame: &Img, r: &Inspection, job: &str, default_root: &Path, barcode: Option<&str>) -> Vec<PathBuf> {
    let mut out = vec![];
    let (a, o) = picture_paths(cfg, r, job, default_root, barcode);
    let Some(dir) = a.as_ref().or(o.as_ref()).and_then(|p| p.parent()) else { return out };
    if std::fs::create_dir_all(dir).is_err() {
        return out;
    }
    let png = cfg["format"].as_str() == Some("png");
    let save = |img: &Img, p: &Path| -> bool {
        if png {
            img.save_png(p).is_ok()
        } else {
            let q = cfg["quality"].as_u64().unwrap_or(92).clamp(30, 100) as u8;
            std::fs::write(p, img.to_jpeg(0, q)).is_ok()
        }
    };
    if let Some(p) = a {
        if save(&annotate(frame, r, cfg, job), &p) {
            out.push(p);
        }
    }
    if let Some(p) = o {
        if save(frame, &p) {
            out.push(p);
        }
    }
    out
}

/// A background writer so saving pictures never holds an inspection up.
pub struct PictureWriter {
    tx: mpsc::SyncSender<Box<dyn FnOnce() + Send>>,
}

impl PictureWriter {
    pub fn new() -> PictureWriter {
        let (tx, rx) = mpsc::sync_channel::<Box<dyn FnOnce() + Send>>(6);
        std::thread::spawn(move || {
            for job in rx {
                job();
            }
        });
        PictureWriter { tx }
    }
    /// false when the queue is full (the disk cannot keep up): dropped, said so
    pub fn submit(&self, f: Box<dyn FnOnce() + Send>) -> bool {
        self.tx.try_send(f).is_ok()
    }
}

impl Default for PictureWriter {
    fn default() -> Self {
        PictureWriter::new()
    }
}

pub fn obj(v: &Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap_or_default()
}
