//! Vision Next, the desktop application: the engine's local server and one
//! native window (Windows' own WebView2) showing it.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tao::dpi::LogicalSize;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoop};
use tao::window::{Icon, WindowBuilder};
use vision_next::{paths, server};
use wry::{WebContext, WebViewBuilder};

const ICON_PNG: &[u8] = include_bytes!("../ui/icon.png");

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let port: u16 = arg("--port").and_then(|p| p.parse().ok()).unwrap_or(8765);
    let app = server::new_app();
    let port = match server::serve(app.clone(), port) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Vision Next could not start its server: {}", e);
            std::process::exit(1);
        }
    };
    let url = format!("http://127.0.0.1:{}/", port);

    let el = EventLoop::new();
    let icon = image::load_from_memory(ICON_PNG).ok().and_then(|i| {
        let rgba = i.to_rgba8();
        let (w, h) = rgba.dimensions();
        Icon::from_rgba(rgba.into_raw(), w, h).ok()
    });
    let window = WindowBuilder::new()
        .with_title("Vision Next")
        .with_inner_size(LogicalSize::new(1440.0, 900.0))
        .with_min_inner_size(LogicalSize::new(900.0, 600.0))
        .with_maximized(true)
        .with_window_icon(icon)
        .build(&el)
        .expect("window");
    {
        // Browse dialogs open over this window
        use tao::platform::windows::WindowExtWindows;
        vision_next::dialogs::set_owner(window.hwnd() as isize);
    }
    let mut ctx = WebContext::new(Some(paths::data_dir().join("webview")));
    let _view = WebViewBuilder::with_web_context(&mut ctx)
        .with_url(&url)
        .with_devtools(cfg!(debug_assertions) || std::env::var("VS_DEVTOOLS").is_ok())
        .with_background_color((241, 243, 245, 255))
        // packages and files the page offers are saved to Downloads
        .with_download_started_handler(|_url, path| {
            if path.as_os_str().is_empty() || path.parent().map(|p| p.as_os_str().is_empty()).unwrap_or(true) {
                let dl = std::env::var("USERPROFILE").map(|h| std::path::PathBuf::from(h).join("Downloads")).unwrap_or_default();
                let name = path.file_name().map(|n| n.to_os_string()).unwrap_or_else(|| "vision-next-package.vnp".into());
                *path = dl.join(name);
            }
            true
        })
        .build(&window)
        .expect("webview (is the WebView2 runtime installed?)");

    el.run(move |event, _, flow| {
        *flow = ControlFlow::Wait;
        if let Event::WindowEvent { event: WindowEvent::CloseRequested, .. } = event {
            // the light back as it was found (its own brightness, on), as Vision Station does
            app.light.lock().unwrap_or_else(|e| e.into_inner()).hand_back();
            *flow = ControlFlow::Exit;
        }
    });
}
