//! Where things are. Beside the program when packaged; while developing, the
//! models are found in the Vision Station folder above.

use std::path::{Path, PathBuf};

pub fn exe_dir() -> PathBuf {
    std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_else(|| PathBuf::from("."))
}

/// The first of `names` that exists, searched beside the program and in up
/// to four folders above it (target/release/ during development).
pub fn find(names: &[&str]) -> Option<PathBuf> {
    let mut dir = Some(exe_dir());
    for _ in 0..5 {
        let d = dir?;
        for n in names {
            let p = d.join(n);
            if p.exists() {
                return Some(p);
            }
        }
        dir = d.parent().map(Path::to_path_buf);
    }
    None
}

pub fn models() -> PathBuf {
    if let Ok(p) = std::env::var("VS_MODELS") {
        return PathBuf::from(p);
    }
    find(&["models"]).unwrap_or_else(|| exe_dir().join("models"))
}

/// The folder this application's projects live in.
pub fn projects() -> PathBuf {
    if let Ok(p) = std::env::var("VS_PROJECTS") {
        return PathBuf::from(p);
    }
    // packaged: projects/ beside the program; developing: vision_next/projects
    let beside = exe_dir().join("projects");
    if beside.exists() || find(&["Cargo.toml"]).is_none() {
        let _ = std::fs::create_dir_all(&beside);
        return beside;
    }
    let root = find(&["Cargo.toml"]).and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_else(exe_dir);
    let p = root.join("projects");
    let _ = std::fs::create_dir_all(&p);
    p
}

/// Vision Station's own projects folder, to import from (never written).
pub fn legacy_projects() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("VS_LEGACY_PROJECTS") {
        return Some(PathBuf::from(p));
    }
    let cargo = find(&["Cargo.toml"])?;
    let repo = cargo.parent()?.parent()?;
    let p = repo.join("projects");
    if p.join("..").join("vision_station").exists() { Some(p) } else { None }
}

pub fn data_dir() -> PathBuf {
    let base = std::env::var("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(|_| exe_dir());
    let p = base.join("VisionNext");
    let _ = std::fs::create_dir_all(&p);
    p
}

pub fn settings_file() -> PathBuf {
    projects().join("settings.json")
}
