//! Every check, by type name.

pub mod gauge;
pub mod learned;
pub mod locate;
pub mod logic;
pub mod more;
pub mod surface;
pub mod text;

use crate::img::Img;
use crate::model::{Ctx, ToolResult, ToolSpec};

/// The check types Vision Next runs. Others load, show and save, and report
/// "not available yet" instead of a verdict.
pub const PORTED: &[&str] = &[
    "align", "presence", "count_template", "count_blob", "color_classify", "led_glow",
    "defect_diff", "histogram", "sharpness", "color_segment", "color_match",
    "measure_length", "calibrate_scale", "find_line", "find_circle", "caliper",
    "distance", "intersect", "results_analysis", "data_analysis", "image_process",
    "ocr", "ocv", "text_detect", "ocr_auto", "code_read", "yolo_detect", "yolo_defect", "classify",
    "trained_surface", "anomaly", "find_ellipse", "fit_shape", "construct", "bead", "fixture_npoint",
    "calibrate_npoint", "affine_transform", "polar_unwrap", "image_combine",
];

pub fn run(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let t0 = std::time::Instant::now();
    let mut r = match t.ty.as_str() {
        "align" => locate::align(t, frame, ctx),
        "presence" => locate::presence(t, frame, ctx),
        "count_template" => locate::count_template(t, frame, ctx),
        "count_blob" => surface::count_blob(t, frame, ctx),
        "color_classify" => surface::color_classify(t, frame, ctx),
        "led_glow" => surface::led_glow(t, frame, ctx),
        "defect_diff" => surface::defect_diff(t, frame, ctx),
        "histogram" => surface::histogram(t, frame, ctx),
        "sharpness" => surface::sharpness(t, frame, ctx),
        "color_segment" => surface::color_segment(t, frame, ctx),
        "color_match" => surface::color_match(t, frame, ctx),
        "measure_length" => gauge::measure_length(t, frame, ctx),
        "calibrate_scale" => gauge::calibrate_scale(t, frame, ctx),
        "find_line" => gauge::find_line(t, frame, ctx),
        "find_circle" => gauge::find_circle(t, frame, ctx),
        "caliper" => gauge::caliper(t, frame, ctx),
        "distance" => gauge::distance(t, frame, ctx),
        "intersect" => gauge::intersect(t, frame, ctx),
        "results_analysis" => logic::results_analysis(t, frame, ctx),
        "data_analysis" => logic::data_analysis(t, frame, ctx),
        "image_process" => logic::image_process(t, frame, ctx),
        "ocr" | "text_detect" | "ocr_auto" => text::ocr_tool(t, frame, ctx),
        "ocv" => text::ocv(t, frame, ctx),
        "code_read" => text::code_read(t, frame, ctx),
        "yolo_detect" | "yolo_defect" => learned::yolo_detect(t, frame, ctx),
        "classify" => learned::classify(t, frame, ctx),
        "trained_surface" | "anomaly" => more::trained_surface(t, frame, ctx),
        "find_ellipse" => more::find_ellipse(t, frame, ctx),
        "fit_shape" => more::fit_shape(t, frame, ctx),
        "construct" => more::construct(t, frame, ctx),
        "bead" => more::bead(t, frame, ctx),
        "fixture_npoint" => more::fixture_npoint(t, frame, ctx),
        "calibrate_npoint" => more::calibrate_npoint(t, frame, ctx),
        "affine_transform" => more::affine_transform(t, frame, ctx),
        "polar_unwrap" => more::polar_unwrap(t, frame, ctx),
        "image_combine" => more::image_combine(t, frame, ctx),
        other => {
            let mut r = ToolResult::new(t, "skip");
            r.message = format!("'{}' is not available in Vision Next yet", other);
            r
        }
    };
    r.ms = (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0;
    if r.caption.is_empty() {
        r.caption = caption(&r);
    }
    r
}

/// The result in a few words, for the picture.
fn caption(r: &ToolResult) -> Vec<String> {
    let v = &r.values;
    let n = |k: &str| v.get(k).and_then(|x| x.as_f64());
    let line = if let Some(c) = n("count") {
        match n("expected") {
            Some(e) => format!("{} / {}", c, e),
            None => format!("{}", c),
        }
    } else if let Some(x) = n("length_mm").or(n("distance")).or(n("diameter_mm")) {
        format!("{:.3}", x)
    } else if let Some(x) = n("length_px") {
        format!("{:.1} px", x)
    } else if let Some(x) = n("defects") {
        format!("{} defect(s)", x)
    } else if let Some(x) = n("percent") {
        format!("{:.1}%", x)
    } else if let Some(x) = n("delta_e") {
        format!("ΔE {:.1}", x)
    } else if r.status == "skip" || r.status == "error" {
        r.message.chars().take(40).collect()
    } else if r.score > 0.0 {
        format!("{:.2}", r.score)
    } else {
        String::new()
    };
    if line.is_empty() { vec![] } else { vec![line] }
}
