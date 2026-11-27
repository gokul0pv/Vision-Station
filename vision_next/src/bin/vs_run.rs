//! vs-run: Vision Next without a window.
//!
//!   vs-run serve [--port 8765] [--open APP]     the engine and its pages, for a line PC
//!   vs-run inspect FOLDER PICTURES [--json]     run an application or project on pictures
//!   vs-run list                                 the applications in the projects folder

use std::path::{Path, PathBuf};
use std::sync::Arc;
use vision_next::{engine, img::Img, server};

fn usage() -> ! {
    eprintln!("vs-run serve [--port N] [--open APP]\nvs-run inspect <application or project folder> <picture or folder> [--json] [--repeat N]\nvs-run list");
    std::process::exit(2);
}

fn pictures(p: &Path) -> Vec<PathBuf> {
    if p.is_file() {
        return vec![p.to_path_buf()];
    }
    let mut v: Vec<PathBuf> = std::fs::read_dir(p).map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|x| {
        x.extension().map(|e| ["png", "jpg", "jpeg", "bmp", "tif", "tiff"].contains(&e.to_string_lossy().to_lowercase().as_str())).unwrap_or(false)
    }).collect()).unwrap_or_default();
    v.sort();
    v
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    match args.first().map(|s| s.as_str()) {
        Some("serve") => {
            let app = server::new_app();
            let port = server::serve(app.clone(), arg("--port").and_then(|p| p.parse().ok()).unwrap_or(8765)).unwrap_or_else(|e| {
                eprintln!("{}", e);
                std::process::exit(1)
            });
            if let Some(id) = arg("--open") {
                if let Err(e) = server::open_app(&app, &id) {
                    eprintln!("{}", e);
                }
            }
            println!("Vision Next serving on http://127.0.0.1:{}/  (Ctrl+C to stop)", port);
            loop {
                std::thread::park();
            }
        }
        Some("runlog") => {
            // vs-run runlog <results.db>: Vision Station's run log as JSON lines
            let p = std::path::PathBuf::from(args.get(1).cloned().unwrap_or_default());
            match vision_next::report::vs_runs(&p) {
                Ok(rs) => {
                    for r in rs.iter() {
                        println!("{}", r);
                    }
                }
                Err(e) => {
                    eprintln!("{}", e);
                    std::process::exit(1);
                }
            }
        }
        Some("cvinfo") => {
            // vs-run cvinfo: is the OpenCV bridge (runtime/vsx_cv.dll) in use?
            match vision_next::cvlib::version() {
                Some(v) => println!("OpenCV {} via vsx_cv.dll", v),
                None => println!("OpenCV bridge not found: using the Rust ports"),
            }
            return;
        }
        Some("align") => {
            // vs-run align <side folder> <pictures>: Vision Station's Locate part
            // (vsalign) on each picture, every stage as one JSON line
            let folder = PathBuf::from(&args[1]);
            let side = engine::Project::load(&folder, "side", "side").unwrap_or_else(|e| { eprintln!("{}", e); std::process::exit(1) });
            let t = side.tools().into_iter().find(|t| t.ty == "align").expect("no align check");
            let p = vision_next::vsalign::Params::from_spec(&t);
            let rg = side.reference.as_ref().expect("no reference").gray();
            for pic in pictures(Path::new(&args[2])) {
                let im = Img::load(&pic).unwrap().gray();
                let t0 = std::time::Instant::now();
                let f = vision_next::vsalign::locate(&p, &rg, &im);
                println!("{}", serde_json::json!({"pic": pic.file_name().unwrap().to_string_lossy(), "m": f.m, "score": f.score,
                    "method": f.method, "margin": f.margin, "inliers": f.inliers, "ms": t0.elapsed().as_millis() as u64, "debug": f.debug}));
            }
            return;
        }
        Some("fitroi") => {
            // vs-run fitroi <side> <picture> m0..m5: Align._fit_roi for that transform
            let folder = PathBuf::from(&args[1]);
            let side = engine::Project::load(&folder, "side", "side").unwrap();
            let t = side.tools().into_iter().find(|t| t.ty == "align").unwrap();
            let p = vision_next::vsalign::Params::from_spec(&t);
            let rg = side.reference.as_ref().unwrap().gray();
            let im = Img::load(Path::new(&args[2])).unwrap().gray();
            let m: Vec<f32> = args[3..9].iter().map(|v| v.parse().unwrap()).collect();
            println!("{}", vision_next::vsalign::fit_roi(&p, &rg, &im, &[m[0], m[1], m[2], m[3], m[4], m[5]]));
            return;
        }
        Some("cvx") => {
            // vs-run cvx <op> <in.png> <out.png> [numbers]: OpenCV ports, for checking against cv2
            let op = args.get(1).cloned().unwrap_or_default();
            if op == "ocr" {
                // vs-run cvx ocr <crop.png> [detail]: every line the reader finds, as JSON
                let src = Img::load(Path::new(&args[2])).unwrap();
                let detail = args.get(3).cloned().unwrap_or_else(|| "auto".into());
                let m = vision_next::ocr::models().unwrap();
                let boxes = vision_next::ocr::detect(m, &src, &detail).unwrap();
                let lines = vision_next::ocr::read(&src, &detail).unwrap();
                let v: Vec<serde_json::Value> = lines.iter().map(|l| serde_json::json!({"text": l.text, "conf": l.conf, "poly": l.poly, "turn": l.turn})).collect();
                let b: Vec<serde_json::Value> = boxes.iter().map(|(b, s)| serde_json::json!({"box": b, "score": s})).collect();
                println!("{}", serde_json::json!({"lines": v, "boxes": b}));
                return;
            }
            if op == "sift" {
                // vs-run cvx sift <gray.png> <nfeatures>: keypoints and descriptors as JSON
                let g = Img::load(Path::new(&args[2])).unwrap().gray();
                let n: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
                let (k, d) = vision_next::vsalign::sift(&g, n);
                let kv: Vec<serde_json::Value> = k.iter().map(|k| serde_json::json!([k.x, k.y, k.size, k.angle, k.response, k.octave])).collect();
                let dv: Vec<serde_json::Value> = d.iter().map(|d| serde_json::json!(d.to_vec())).collect();
                println!("{}", serde_json::json!({"kp": kv, "desc": dv}));
                return;
            }
            if op == "gaussf" {
                // vs-run cvx gaussf <in.png> <out.f32> <sigma>: GaussianBlur of the picture as float
                let g = Img::load(Path::new(&args[2])).unwrap().gray();
                let s: f64 = args[4].parse().unwrap();
                let o = vision_next::vsalign::gauss_f32(&vision_next::vsalign::Fm::from_u8(&g), 0, s);
                let n = ((s * 8.0 + 1.0).round_ties_even() as i64 | 1) as usize;
                eprintln!("{:?}", vision_next::vsalign::gauss_kernel_f32(n, s).iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                let b: Vec<u8> = o.d.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(&args[3], b).unwrap();
                return;
            }
            if op == "area" {
                // vs-run cvx area <in.png> <out.png> <scale>
                let g = Img::load(Path::new(&args[2])).unwrap().gray();
                let s: f64 = args[4].parse().unwrap();
                vision_next::vsalign::resize_area_scale(&g, s).save_png(Path::new(&args[3])).unwrap();
                return;
            }
            if op == "boxes" {
                // vs-run cvx boxes <prob.f32> <w> <h>: the text boxes from a probability map, as JSON
                let raw = std::fs::read(&args[2]).unwrap();
                let (w, h): (usize, usize) = (args[3].parse().unwrap(), args[4].parse().unwrap());
                let prob: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                let bx = vision_next::cvx::boxes_from_map(&prob, w, h);
                let v: Vec<serde_json::Value> = bx.iter().map(|(b, s)| serde_json::json!({"box": b, "score": s})).collect();
                println!("{}", serde_json::Value::Array(v));
                return;
            }
            let src = Img::load(Path::new(&args[2])).unwrap_or_else(|e| { eprintln!("{}", e); std::process::exit(1) });
            let n: Vec<f64> = args.iter().skip(4).filter_map(|a| a.parse().ok()).collect();
            let out = match op.as_str() {
                "decode" => src,
                "linear" => vision_next::cvx::resize_linear(&src, n[0] as usize, n[1] as usize),
                "cubic" => vision_next::cvx::resize_cubic_scale(&src, n[0]),
                "gray" => src.gray(),
                "blur" => vision_next::cvx::gaussian_blur(&src, n[0]),
                "stretch" => vision_next::cvx::stretch(&src),
                "lab" => vision_next::cvx::rgb_to_lab(&src),
                "lab2rgb" => vision_next::cvx::lab_to_rgb(&src),
                "clahe_lab" => vision_next::cvx::clahe_lab(&src),
                "clahe" => { let g = src.gray(); let d = vision_next::cvx::clahe(&g.data, g.w, g.h, 3.0, 8); Img { w: g.w, h: g.h, c: 1, data: d } }
                "warplin" => {
                    let b: [[f32; 2]; 4] = std::array::from_fn(|i| [n[2 * i] as f32, n[2 * i + 1] as f32]);
                    let (w, h) = (n[8] as usize, n[9] as usize);
                    let d = [[0.0, 0.0], [(w - 1) as f32, 0.0], [(w - 1) as f32, (h - 1) as f32], [0.0, (h - 1) as f32]];
                    let m = vision_next::cvx::get_perspective_transform(&b, &d);
                    vision_next::cvx::warp_perspective_linear(&src, &m, w, h)
                }
                "warp" => {
                    let b: [[f32; 2]; 4] = std::array::from_fn(|i| [n[2 * i] as f32, n[2 * i + 1] as f32]);
                    let (w, h) = (n[8] as usize, n[9] as usize);
                    let d = [[0.0, 0.0], [w as f32, 0.0], [w as f32, h as f32], [0.0, h as f32]];
                    let m = vision_next::cvx::get_perspective_transform(&b, &d);
                    vision_next::cvx::warp_perspective_cubic(&src, &m, w, h)
                }
                _ => { eprintln!("unknown op"); std::process::exit(1) }
            };
            out.save_png(Path::new(&args[3])).unwrap();
        }
        Some("list") => {
            let s = engine::Store::new(vision_next::paths::projects());
            for a in s.list() {
                println!("{:<32} {:>2} side(s) {:>4} checks  {}", a.id, a.sides, a.checks, a.name);
            }
        }
        Some("inspect") => {
            let (Some(folder), Some(pics)) = (args.get(1), args.get(2)) else { usage() };
            let folder = PathBuf::from(folder);
            let id = folder.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            let t_load = std::time::Instant::now();
            let app = engine::open_in(&folder, &id).unwrap_or_else(|e| {
                eprintln!("{}", e);
                std::process::exit(1)
            });
            let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
            let json_out = args.iter().any(|a| a == "--json");
            let repeat: usize = arg("--repeat").and_then(|r| r.parse().ok()).unwrap_or(1).max(1);
            let list = pictures(Path::new(pics));
            if list.is_empty() {
                eprintln!("no pictures in {}", pics);
                std::process::exit(1);
            }
            eprintln!("{}: {} side(s), loaded in {:.0} ms", app.name, app.sides.len(), load_ms);
            let mut times = vec![];
            let mut fails = 0;
            for _ in 0..repeat {
                for p in &list {
                    let im = match Img::load(p) {
                        Ok(i) => Arc::new(i),
                        Err(e) => {
                            eprintln!("{}", e);
                            continue;
                        }
                    };
                    for side in &app.sides {
                        let r = side.inspect(im.clone(), &p.display().to_string(), None, None);
                        times.push(r.total_ms);
                        if r.verdict != "PASS" {
                            fails += 1;
                        }
                        if json_out {
                            println!("{}", serde_json::to_string(&r).unwrap());
                        } else {
                            let bad: Vec<String> = r.tools.iter().filter(|t| t.status == "fail" || t.status == "error")
                                .map(|t| format!("{} ({})", t.tool_id, if t.message.is_empty() { &t.status } else { &t.message })).collect();
                            println!("{:<5} {:>8.1} ms  {}  {}{}", r.verdict, r.total_ms, p.file_name().unwrap().to_string_lossy(),
                                if bad.is_empty() { String::new() } else { bad.join(", ") },
                                if r.not_ported > 0 { format!("  [{} check(s) not in Vision Next yet]", r.not_ported) } else { String::new() });
                        }
                    }
                }
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = times.len().max(1);
            let mean = times.iter().sum::<f64>() / n as f64;
            eprintln!("{} inspections, {} not PASS; mean {:.1} ms, median {:.1} ms, max {:.1} ms",
                times.len(), fails, mean, times[times.len() / 2], times.last().copied().unwrap_or(0.0));
        }
        _ => usage(),
    }
}
