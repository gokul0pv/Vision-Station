//! Windows' own Open dialogs (file and folder) for the Browse buttons. The
//! page asks the server; the server shows the dialog over the app's window.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicIsize, Ordering};

/// The app window's handle, so a dialog opens over it (set by main).
static OWNER: AtomicIsize = AtomicIsize::new(0);

pub fn set_owner(hwnd: isize) {
    OWNER.store(hwnd, Ordering::Relaxed);
}

#[cfg(feature = "gui")]
struct Owner(std::num::NonZeroIsize);

#[cfg(feature = "gui")]
impl raw_window_handle::HasWindowHandle for Owner {
    fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        let h = raw_window_handle::Win32WindowHandle::new(self.0);
        Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(h.into()) })
    }
}

#[cfg(feature = "gui")]
impl raw_window_handle::HasDisplayHandle for Owner {
    fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Ok(raw_window_handle::DisplayHandle::windows())
    }
}

/// The folder to open in: the given path's own folder when it exists,
/// else `fallback`.
fn start_dir(current: &str, fallback: &Path) -> PathBuf {
    let p = PathBuf::from(current.trim());
    if !current.trim().is_empty() {
        if p.is_dir() {
            return p;
        }
        if let Some(d) = p.parent() {
            if d.is_dir() {
                return d.to_path_buf();
            }
        }
    }
    fallback.to_path_buf()
}

/// Ask for a folder (kind "folder") or a file. `filters`: (name, extensions).
/// None when cancelled.
#[cfg(feature = "gui")]
pub fn browse(kind: &str, title: &str, current: &str, fallback: &Path, filters: &[(String, Vec<String>)]) -> Result<Option<PathBuf>, String> {
    let mut d = rfd::FileDialog::new().set_title(title).set_directory(start_dir(current, fallback));
    if let Some(h) = std::num::NonZeroIsize::new(OWNER.load(Ordering::Relaxed)) {
        d = d.set_parent(&Owner(h));
    }
    if kind == "folder" {
        return Ok(d.pick_folder());
    }
    for (name, exts) in filters {
        let e: Vec<&str> = exts.iter().map(|s| s.as_str()).collect();
        d = d.add_filter(name, &e);
    }
    if !filters.is_empty() {
        d = d.add_filter("All files", &["*"]);
    }
    let cur = PathBuf::from(current.trim());
    if let Some(n) = cur.file_name().filter(|_| cur.is_file()) {
        d = d.set_file_name(n.to_string_lossy());
    }
    Ok(d.pick_file())
}

#[cfg(not(feature = "gui"))]
pub fn browse(_: &str, _: &str, _: &str, _: &Path, _: &[(String, Vec<String>)]) -> Result<Option<PathBuf>, String> {
    Err("Browse needs the Vision Next window: type the path instead".into())
}

/// Ask where to save a file. None when cancelled.
#[cfg(feature = "gui")]
pub fn save_as(title: &str, name: &str, dir: &Path, filters: &[(String, Vec<String>)]) -> Result<Option<PathBuf>, String> {
    let mut d = rfd::FileDialog::new().set_title(title).set_directory(dir).set_file_name(name);
    if let Some(h) = std::num::NonZeroIsize::new(OWNER.load(Ordering::Relaxed)) {
        d = d.set_parent(&Owner(h));
    }
    for (n, exts) in filters {
        let e: Vec<&str> = exts.iter().map(|s| s.as_str()).collect();
        d = d.add_filter(n, &e);
    }
    Ok(d.save_file())
}

#[cfg(not(feature = "gui"))]
pub fn save_as(_: &str, _: &str, _: &Path, _: &[(String, Vec<String>)]) -> Result<Option<PathBuf>, String> {
    Err("Saving needs the Vision Next window".into())
}
