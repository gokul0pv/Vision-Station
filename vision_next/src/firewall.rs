//! Windows Firewall permission for GigE cameras (as Vision Station's
//! hardware/firewall.py). A GigE camera sends every picture as UDP packets to
//! the program that asked, and Windows lets them in only for programs it has
//! a rule for -- per program file. Without one, on a Public network (a camera
//! port usually is) the program never hears the camera: it opens, then no
//! pictures come. allow() adds the rule through one administrator prompt;
//! has_rule() checks without administrator rights.

use std::path::PathBuf;

pub const RULE_PREFIX: &str = "Vision Next camera";

/// The program files that talk to cameras: the window and the line runner.
pub fn programs() -> Vec<PathBuf> {
    let me = std::env::current_exe().unwrap_or_default();
    let dir = me.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let mut out = vec![me.clone()];
    for n in ["vision-next.exe", "vs-run.exe"] {
        let p = dir.join(n);
        if p.exists() && !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

pub fn rule_name(prog: &std::path::Path) -> String {
    format!("{} ({})", RULE_PREFIX, prog.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default())
}

/// True when the rule exists, false when not, None when it cannot be told.
#[cfg(windows)]
pub fn has_rule(prog: &std::path::Path) -> Option<bool> {
    use std::os::windows::process::CommandExt;
    // netsh's words are in the Windows language; its exit code is not: 0 when a rule of that name exists
    let out = std::process::Command::new("netsh")
        .args(["advfirewall", "firewall", "show", "rule"])
        .arg(format!("name={}", rule_name(prog)))
        .creation_flags(0x0800_0000) // no console window
        .output().ok()?;
    Some(out.status.success())
}

#[cfg(not(windows))]
pub fn has_rule(_: &std::path::Path) -> Option<bool> {
    None
}

/// Every program without its rule.
pub fn missing() -> Vec<PathBuf> {
    programs().into_iter().filter(|p| has_rule(p) == Some(false)).collect()
}

#[cfg(windows)]
#[repr(C)]
#[allow(non_snake_case)]
struct ShellExecuteInfoW {
    cbSize: u32,
    fMask: u32,
    hwnd: isize,
    lpVerb: *const u16,
    lpFile: *const u16,
    lpParameters: *const u16,
    lpDirectory: *const u16,
    nShow: i32,
    hInstApp: isize,
    lpIDList: *mut std::ffi::c_void,
    lpClass: *const u16,
    hkeyClass: isize,
    dwHotKey: u32,
    hIcon: isize,
    hProcess: isize,
}

#[cfg(windows)]
#[link(name = "shell32")]
extern "system" {
    fn ShellExecuteExW(info: *mut ShellExecuteInfoW) -> i32;
}

#[cfg(windows)]
extern "system" {
    fn WaitForSingleObject(h: isize, ms: u32) -> u32;
    fn CloseHandle(h: isize) -> i32;
}

/// Add an inbound allow rule for every program without one, on every network
/// profile, through Windows' administrator prompt. Ok(true) when they are all
/// there afterwards; Ok(false) when the prompt was declined.
#[cfg(windows)]
pub fn allow() -> Result<bool, String> {
    let need = missing();
    if need.is_empty() {
        return Ok(true);
    }
    let cmds: Vec<String> = need.iter().map(|p| format!(
        "netsh advfirewall firewall add rule name=\"{}\" dir=in action=allow program=\"{}\" enable=yes profile=any",
        rule_name(p), p.display())).collect();
    let params = format!("/c {}", cmds.join(" & "));
    let w = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
    let (verb, file, par) = (w("runas"), w("cmd.exe"), w(&params));
    let mut info = ShellExecuteInfoW {
        cbSize: std::mem::size_of::<ShellExecuteInfoW>() as u32, fMask: 0x40, // SEE_MASK_NOCLOSEPROCESS
        hwnd: 0, lpVerb: verb.as_ptr(), lpFile: file.as_ptr(), lpParameters: par.as_ptr(), lpDirectory: std::ptr::null(),
        nShow: 0, hInstApp: 0, lpIDList: std::ptr::null_mut(), lpClass: std::ptr::null(), hkeyClass: 0, dwHotKey: 0, hIcon: 0, hProcess: 0,
    };
    unsafe {
        if ShellExecuteExW(&mut info) == 0 {
            return Ok(false); // declined, or no rights
        }
        if info.hProcess != 0 {
            WaitForSingleObject(info.hProcess, 60_000);
            CloseHandle(info.hProcess);
        }
    }
    Ok(missing().is_empty())
}

#[cfg(not(windows))]
pub fn allow() -> Result<bool, String> {
    Err("only on Windows".into())
}

pub const EXPLANATION: &str = "Windows Firewall keeps the camera's pictures from reaching a program it has not been told to allow. \
A GigE camera sends its pictures over the network; allow Vision Next once (Windows asks for an administrator) and the camera works from then on.";
