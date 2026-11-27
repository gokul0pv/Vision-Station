//! The rest of Vision Station's checks: trained surface, find ellipse, fit
//! shape, create geometry, bead, fixture from points, N-point calibration,
//! affine transform, polar unwrap, two-image operation.

use super::gauge::{self, profile_edges, Feature};
use crate::geom::{self, Affine};
use crate::img::{self, Img, Plane};
use crate::model::{jnum, Ctx, Graphic, ToolResult, ToolSpec};
use crate::npz;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

// ------------------------------------------------------------ geometry out
fn publish(t: &ToolSpec, ctx: &mut Ctx, cur: Img, reference: Option<Img>, extra: Value) -> ToolResult {
    let (w, h) = (cur.w, cur.h);
    ctx.images.insert(t.id.clone(), Arc::new(cur));
    if let Some(r) = reference {
        ctx.ref_images.insert(t.id.clone(), Arc::new(r));
    }
    let mut r = ToolResult::new(t, "pass");
    r.set("width", w as i64).set("height", h as i64);
    if let Value::Object(o) = extra {
        for (k, v) in o {
            r.values.insert(k, v);
        }
    }
    r.caption = vec![format!("{} × {}", w, h)];
    r
}

pub fn affine_transform(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(reg) = ctx.region(t, frame) else { return ToolResult::error(t, "no region drawn") };
    let k = { let v = t.f("scale", 1.0); if v > 0.0 { v } else { 1.0 } };
    let scale = |im: Img| if (k - 1.0).abs() < 1e-9 { im } else {
        let (w, h) = (((im.w as f64) * k).round().max(1.0) as usize, ((im.h as f64) * k).round().max(1.0) as usize);
        im.resize(w, h)
    };
    let refi = ctx.ref_region(t).map(|r| scale(r.patch));
    publish(t, ctx, scale(reg.patch), refi, json!({"scale": k}))
}

fn unwrap_ring(t: &ToolSpec, img: &Img, fixture: Option<&Affine>) -> Option<Img> {
    let roi = t.roi()?;
    let (mut cx, mut cy) = (roi[0] + roi[2] / 2.0, roi[1] + roi[3] / 2.0);
    let mut start = t.f("start_angle", 0.0);
    if let Some(f) = fixture {
        let (a, b) = f.apply(cx, cy);
        cx = a;
        cy = b;
        start += f.angle_deg();
    }
    let r_out = roi[2].min(roi[3]) / 2.0;
    let r_in = r_out * t.f("inner_radius", 0.5);
    let rows = ((r_out - r_in).round() as usize).max(4);
    let width = { let w = t.i("width", 0); if w > 0 { w as usize } else { ((std::f64::consts::PI * (r_in + r_out)).round() as usize).max(16) } };
    let sweep = t.f("sweep", 360.0);
    let sign = if t.s("direction", "clockwise") == "counterclockwise" { -1.0 } else { 1.0 };
    let mut out = Img::new(width, rows, img.c);
    for y in 0..rows {
        let rad = r_out - (r_out - r_in) * y as f64 / ((rows - 1).max(1)) as f64;
        for x in 0..width {
            let a = (start + sign * sweep * x as f64 / width as f64).to_radians();
            let (sx, sy) = (cx + rad * a.cos(), cy + rad * a.sin());
            for c in 0..img.c {
                out.data[(y * width + x) * img.c + c] = img.sample(sx as f32, sy as f32, c).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Some(out)
}

pub fn polar_unwrap(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(cur) = unwrap_ring(t, frame, ctx.fixture.as_ref()) else { return ToolResult::error(t, "no region drawn") };
    let refi = ctx.reference.clone().and_then(|r| unwrap_ring(t, &r, None));
    let mut r = publish(t, ctx, cur, refi, json!({"inner_radius": t.f("inner_radius", 0.5)}));
    if let Some(o) = ctx.outline(t) {
        r.graphics.push(Graphic { kind: "poly".into(), points: o, status: "pass".into(), label: String::new(), role: "region".into() });
    }
    r
}

pub fn image_combine(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(refi) = ctx.reference.clone() else { return ToolResult::error(t, "needs a reference picture") };
    if refi.w != frame.w || refi.h != frame.h {
        return ToolResult::error(t, "the reference and the picture differ in size");
    }
    // the reference moved onto the part first, so the two line up
    let aligned = match ctx.fixture {
        Some(f) => geom::warp(&refi, &f.invert(), frame.w, frame.h),
        None => (*refi).clone(),
    };
    let (a, b) = if frame.c == aligned.c { (frame.clone(), aligned) } else { (frame.to_rgb(), aligned.to_rgb()) };
    let op = t.s("operation", "difference");
    let f: fn(u8, u8) -> u8 = match op.as_str() {
        "subtract" => |x, y| x.saturating_sub(y),
        "subtract_reference_from" => |x, y| y.saturating_sub(x),
        "minimum" => |x, y| x.min(y),
        "maximum" => |x, y| x.max(y),
        "average" => |x, y| ((x as u16 + y as u16 + 1) / 2) as u8,
        _ => |x, y| x.abs_diff(y),
    };
    let mut out = a.clone();
    for (o, (x, y)) in out.data.iter_mut().zip(a.data.iter().zip(&b.data)) {
        *o = f(*x, *y);
    }
    let mut r = ToolResult::new(t, "pass");
    r.set("operation", op);
    ctx.images.insert(t.id.clone(), Arc::new(out));
    r
}

// ---------------------------------------------------------------- ellipse
/// Conic Ax² + Bxy + Cy² + Dx + Ey = 1 by least squares: centre, semi-axes, angle.
pub fn fit_ellipse(pts: &[[f64; 2]]) -> Option<([f64; 2], f64, f64, f64)> {
    if pts.len() < 5 {
        return None;
    }
    let (mx, my) = (pts.iter().map(|p| p[0]).sum::<f64>() / pts.len() as f64, pts.iter().map(|p| p[1]).sum::<f64>() / pts.len() as f64);
    let mut m = [[0f64; 6]; 5];
    for p in pts {
        let (x, y) = (p[0] - mx, p[1] - my);
        let row = [x * x, x * y, y * y, x, y];
        for i in 0..5 {
            for j in 0..5 {
                m[i][j] += row[i] * row[j];
            }
            m[i][5] += row[i];
        }
    }
    // Gaussian elimination
    for c in 0..5 {
        let piv = (c..5).max_by(|&a, &b| m[a][c].abs().partial_cmp(&m[b][c].abs()).unwrap())?;
        m.swap(c, piv);
        if m[c][c].abs() < 1e-12 {
            return None;
        }
        for r in 0..5 {
            if r != c {
                let f = m[r][c] / m[c][c];
                for k in c..6 {
                    m[r][k] -= f * m[c][k];
                }
            }
        }
    }
    let s: Vec<f64> = (0..5).map(|i| m[i][5] / m[i][i]).collect();
    let (a, b, c, d, e) = (s[0], s[1], s[2], s[3], s[4]);
    let den = 4.0 * a * c - b * b;
    if den <= 0.0 {
        return None; // not an ellipse
    }
    let x0 = (b * e - 2.0 * c * d) / den;
    let y0 = (b * d - 2.0 * a * e) / den;
    // about the centre the conic is A u² + B uv + C v² = K
    let mut k = 1.0 - (a * x0 * x0 + b * x0 * y0 + c * y0 * y0 + d * x0 + e * y0);
    let theta = 0.5 * b.atan2(a - c);
    let (ct, st) = (theta.cos(), theta.sin());
    let mut ap = a * ct * ct + b * ct * st + c * st * st;
    let mut cp = a * st * st - b * ct * st + c * ct * ct;
    if k < 0.0 {
        k = -k;
        ap = -ap;
        cp = -cp;
    }
    if ap <= 0.0 || cp <= 0.0 || k <= 0.0 {
        return None;
    }
    let ra = (k / ap).sqrt();
    let rb = (k / cp).sqrt();
    Some(([x0 + mx, y0 + my], ra, rb, theta.to_degrees()))
}

fn ellipse_residual(p: [f64; 2], c: [f64; 2], a: f64, b: f64, ang: f64) -> f64 {
    let (s, co) = ang.to_radians().sin_cos();
    let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
    let u = dx * co + dy * s;
    let v = -dx * s + dy * co;
    let r = (u * u + v * v).sqrt();
    if r < 1e-9 {
        return a.min(b);
    }
    let t = v.atan2(u);
    let re = 1.0 / ((t.cos() / a).powi(2) + (t.sin() / b).powi(2)).sqrt();
    (r - re).abs()
}

pub fn find_ellipse(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(reg) = ctx.region(t, frame) else { return ToolResult::error(t, "no region drawn") };
    let g = match t.s("channel", "auto").as_str() {
        "red" => reg.patch.channel(0),
        "green" => reg.patch.channel(1),
        "blue" => reg.patch.channel(2),
        _ => reg.patch.gray(),
    };
    let p = Plane::from_gray(&g);
    let (cx, cy) = (p.w as f64 / 2.0, p.h as f64 / 2.0);
    let n = t.i("num_calipers", 36).max(8) as usize;
    let (fi, fo) = (t.f("search_inner", 0.45), t.f("search_outer", 1.0));
    let mc = t.f("min_contrast", 8.0);
    let pol = t.s("polarity", "any");
    let mut pts = vec![];
    for i in 0..n {
        let th = i as f64 / n as f64 * std::f64::consts::TAU;
        let (c, s) = (th.cos(), th.sin());
        // the drawn box's own ellipse radius along this ray
        let re = 1.0 / ((c / cx).powi(2) + (s / cy).powi(2)).sqrt();
        let (r0, r1) = (fi * re, fo * re);
        let steps = ((r1 - r0) / 0.5).max(5.0) as usize;
        let prof: Vec<f64> = (0..steps).map(|k| { let r = r0 + k as f64 * 0.5; p.sample((cx + c * r) as f32, (cy + s * r) as f32) as f64 }).collect();
        if let Some(e) = profile_edges(&prof, mc / 2.0, &pol).into_iter().max_by(|a, b| a.strength.abs().partial_cmp(&b.strength.abs()).unwrap()) {
            let r = r0 + (e.pos - 0.5) * 0.5;
            pts.push([cx + c * r, cy + s * r]);
        }
    }
    let ignore = t.i("num_to_ignore", 4).max(0) as usize;
    let min_found = t.i("min_found", 12).max(5) as usize;
    let mut r = ToolResult::new(t, "pass");
    let Some(mut fit) = fit_ellipse(&pts) else {
        r.status = "fail".into();
        r.message = format!("no ellipse through {} edge point(s)", pts.len());
        return r;
    };
    // the worst points left out, then fitted again
    for _ in 0..ignore {
        if pts.len() <= min_found {
            break;
        }
        let (c, a, b, ang) = fit;
        let worst = pts.iter().enumerate().max_by(|x, y| ellipse_residual(*x.1, c, a, b, ang).partial_cmp(&ellipse_residual(*y.1, c, a, b, ang)).unwrap()).map(|x| x.0).unwrap();
        pts.remove(worst);
        match fit_ellipse(&pts) { Some(f) => fit = f, None => break }
    }
    let (c, a, b, ang) = fit;
    let rms = (pts.iter().map(|q| ellipse_residual(*q, c, a, b, ang).powi(2)).sum::<f64>() / pts.len() as f64).sqrt();
    let max_rms = t.f("max_rms", 1.5);
    let mut why = vec![];
    if pts.len() < min_found { why.push(format!("{} points (needs {})", pts.len(), min_found)); }
    if rms > max_rms { why.push(format!("fit error {:.2} px", rms)); }
    r.pass_if(why.is_empty());
    r.message = why.join("; ");
    let ci = reg.map(c[0], c[1]);
    let (major, minor) = (2.0 * a.max(b), 2.0 * a.min(b));
    r.set("cx", jnum(ci[0], 2)).set("cy", jnum(ci[1], 2)).set("major", jnum(major, 3)).set("minor", jnum(minor, 3))
        .set("angle", jnum(ang + reg.to_img.angle_deg(), 2)).set("rms", jnum(rms, 3)).set("found", pts.len() as i64);
    let mm = ctx.mm(t);
    if mm > 0.0 {
        r.set("major_mm", jnum(major * mm, 4)).set("minor_mm", jnum(minor * mm, 4));
    }
    let st = r.status.clone();
    let (s, co) = ang.to_radians().sin_cos();
    let poly: Vec<[f64; 2]> = (0..72).map(|i| {
        let tt = i as f64 / 72.0 * std::f64::consts::TAU;
        let (u, v) = (a * tt.cos(), b * tt.sin());
        reg.map(c[0] + u * co - v * s, c[1] + u * s + v * co)
    }).collect();
    r.graphics.push(Graphic::poly(poly, &st, ""));
    r.graphics.push(Graphic::point(ci, &st, ""));
    r
}

// --------------------------------------------------------------- fit shape
pub fn fit_shape(t: &ToolSpec, _frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let src = t.s("source", "");
    let Some(pts) = ctx.points.get(&src).cloned() else { return ToolResult::error(t, format!("'{}' gave no centres (has it run?)", src)) };
    let fit = t.s("fit", "line");
    let maxdev = t.f("max_deviation", 0.0);
    let mut r = ToolResult::new(t, "pass");
    let devs: Vec<f64>;
    match fit.as_str() {
        "circle" => {
            let Some((cx, cy, rad, _)) = geom::fit_circle(&pts) else { return ToolResult::error(t, "needs three centres or more") };
            devs = pts.iter().map(|p| (((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt() - rad).abs()).collect();
            r.set("cx", jnum(cx, 2)).set("cy", jnum(cy, 2)).set("radius", jnum(rad, 3)).set("diameter", jnum(2.0 * rad, 3));
            r.graphics.push(Graphic::poly(geom::circle_poly(cx, cy, rad, 64), "pass", ""));
        }
        "ellipse" => {
            let Some((c, a, b, ang)) = fit_ellipse(&pts) else { return ToolResult::error(t, "needs five centres or more") };
            devs = pts.iter().map(|p| ellipse_residual(*p, c, a, b, ang)).collect();
            r.set("cx", jnum(c[0], 2)).set("cy", jnum(c[1], 2)).set("major", jnum(2.0 * a.max(b), 3)).set("minor", jnum(2.0 * a.min(b), 3)).set("angle", jnum(ang, 2));
        }
        _ => {
            let Some((c, d, _)) = geom::fit_line(&pts) else { return ToolResult::error(t, "needs two centres or more") };
            devs = pts.iter().map(|p| (-(p[0] - c[0]) * d[1] + (p[1] - c[1]) * d[0]).abs()).collect();
            let ts: Vec<f64> = pts.iter().map(|p| (p[0] - c[0]) * d[0] + (p[1] - c[1]) * d[1]).collect();
            let (t0, t1) = (ts.iter().cloned().fold(f64::MAX, f64::min), ts.iter().cloned().fold(f64::MIN, f64::max));
            let (a, b) = ([c[0] + t0 * d[0], c[1] + t0 * d[1]], [c[0] + t1 * d[0], c[1] + t1 * d[1]]);
            r.set("x1", jnum(a[0], 2)).set("y1", jnum(a[1], 2)).set("x2", jnum(b[0], 2)).set("y2", jnum(b[1], 2))
                .set("angle", jnum(d[1].atan2(d[0]).to_degrees(), 3));
            r.graphics.push(Graphic::line(a, b, "pass", ""));
        }
    }
    let worst = devs.iter().cloned().fold(0.0, f64::max);
    r.set("max_deviation", jnum(worst, 3)).set("count", pts.len() as i64);
    if maxdev > 0.0 && worst > maxdev {
        r.status = "fail".into();
        r.message = format!("a centre is {:.2} off the {} (at most {:.2})", worst, fit, maxdev);
    }
    let st = r.status.clone();
    for g in r.graphics.iter_mut() { g.status = st.clone(); }
    for p in pts { r.graphics.push(Graphic::point(p, &st, "")); }
    r
}

// ---------------------------------------------------------------- construct
pub fn construct(t: &ToolSpec, _frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let op = t.s("operation", "line_through_points");
    let f = |k: &str| gauge::feature(ctx, &t.s(k, ""));
    let (a, b, c) = (f("first"), f("second"), f("third"));
    let pt = |x: Option<Feature>| x.map(gauge::centre);
    let mut r = ToolResult::new(t, "pass");
    let fail = |m: &str| ToolResult::error(t, m.to_string());
    match op.as_str() {
        "midpoint" => {
            let m = match (a, b) {
                (Some(Feature::Line(p, q)), None) => [(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0],
                (Some(x), Some(y)) => { let (p, q) = (gauge::centre(x), gauge::centre(y)); [(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0] }
                _ => return fail("needs two points, or a line"),
            };
            r.set("cx", jnum(m[0], 2)).set("cy", jnum(m[1], 2));
            r.graphics.push(Graphic::point(m, "pass", ""));
        }
        "line_through_points" => {
            let (Some(p), Some(q)) = (pt(a), pt(b)) else { return fail("needs two points") };
            r.set("x1", jnum(p[0], 2)).set("y1", jnum(p[1], 2)).set("x2", jnum(q[0], 2)).set("y2", jnum(q[1], 2))
                .set("angle", jnum((q[1] - p[1]).atan2(q[0] - p[0]).to_degrees(), 3))
                .set("length", jnum(((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt(), 3));
            r.graphics.push(Graphic::line(p, q, "pass", ""));
        }
        "perpendicular" | "parallel" => {
            let (Some(Feature::Line(p, q)), Some(x)) = (a, b) else { return fail("needs a line, then a point") };
            let o = gauge::centre(x);
            let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
            let l = (dx * dx + dy * dy).sqrt().max(1e-9);
            let (ux, uy) = if op == "parallel" { (dx / l, dy / l) } else { (-dy / l, dx / l) };
            let h = l / 2.0;
            let (s, e) = ([o[0] - ux * h, o[1] - uy * h], [o[0] + ux * h, o[1] + uy * h]);
            r.set("x1", jnum(s[0], 2)).set("y1", jnum(s[1], 2)).set("x2", jnum(e[0], 2)).set("y2", jnum(e[1], 2));
            r.graphics.push(Graphic::line(s, e, "pass", ""));
        }
        "circle_through_points" => {
            let (Some(p), Some(q), Some(w)) = (pt(a), pt(b), pt(c)) else { return fail("needs three points") };
            let Some((cx, cy, rad, _)) = geom::fit_circle(&[p, q, w]) else { return fail("the points are on one line") };
            r.set("cx", jnum(cx, 2)).set("cy", jnum(cy, 2)).set("radius", jnum(rad, 3)).set("diameter", jnum(2.0 * rad, 3));
            r.graphics.push(Graphic::poly(geom::circle_poly(cx, cy, rad, 64), "pass", ""));
        }
        other => return fail(&format!("unknown operation {}", other)),
    }
    r
}

// --------------------------------------------------------------------- bead
pub fn bead(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let fx = ctx.fixture.unwrap_or(Affine::IDENTITY);
    let path: Vec<[f64; 2]> = if t.s("shape", "rect") == "polygon" && t.points().len() >= 2 {
        t.points().iter().map(|p| { let (x, y) = fx.apply(p[0], p[1]); [x, y] }).collect()
    } else if let Some(roi) = t.roi() {
        // the long centre line of the region
        let m = fx.then_after(&geom::patch_to_ref(roi, t.angle()));
        let (w, h) = (roi[2], roi[3]);
        let (a, b) = if w >= h { ((0.0, h / 2.0), (w, h / 2.0)) } else { ((w / 2.0, 0.0), (w / 2.0, h)) };
        let (p, q) = (m.apply(a.0, a.1), m.apply(b.0, b.1));
        vec![[p.0, p.1], [q.0, q.1]]
    } else {
        return ToolResult::error(t, "no path drawn");
    };
    let g = Plane::from_gray(&frame.gray());
    let seg: Vec<f64> = path.windows(2).map(|s| ((s[1][0] - s[0][0]).powi(2) + (s[1][1] - s[0][1]).powi(2)).sqrt()).collect();
    let total: f64 = seg.iter().sum();
    if total < 2.0 {
        return ToolResult::error(t, "the path is too short");
    }
    let n = t.i("stations", 30).max(4) as usize;
    let half = t.f("search_half_width", 25.0);
    let (wexp, wtol, maxoff) = (t.f("width", 12.0), t.f("width_tolerance", 4.0), t.f("max_offset", 6.0));
    let mc = t.f("min_contrast", 15.0);
    let dark = t.s("polarity", "dark") == "dark";
    let mut widths = vec![];
    let mut bad = 0;
    let mut r = ToolResult::new(t, "pass");
    for i in 0..n {
        let d = (i as f64 + 0.5) / n as f64 * total;
        let (mut acc, mut k) = (0.0, 0);
        while k + 1 < seg.len() && acc + seg[k] < d { acc += seg[k]; k += 1; }
        let f = ((d - acc) / seg[k].max(1e-9)).clamp(0.0, 1.0);
        let (a, b) = (path[k], path[k + 1]);
        let o = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
        let l = seg[k].max(1e-9);
        let nrm = [-(b[1] - a[1]) / l, (b[0] - a[0]) / l];
        let steps = (2.0 * half / 0.5) as usize;
        let prof: Vec<f64> = (0..steps).map(|s| {
            let u = -half + s as f64 * 0.5;
            g.sample((o[0] + nrm[0] * u) as f32, (o[1] + nrm[1] * u) as f32) as f64
        }).collect();
        let edges = profile_edges(&prof, mc / 2.0, "any");
        // the bead: an edge into it then an edge out of it, strongest pair
        let into = |e: &gauge::Edge| if dark { e.strength < 0.0 } else { e.strength > 0.0 };
        let mut best: Option<(f64, f64, f64)> = None;
        for e1 in edges.iter().filter(|e| into(e)) {
            for e2 in edges.iter().filter(|e| !into(e) && e.pos > e1.pos) {
                let s = e1.strength.abs() + e2.strength.abs();
                if best.map(|b| s > b.2).unwrap_or(true) {
                    best = Some((e1.pos, e2.pos, s));
                }
            }
        }
        let st;
        match best {
            Some((p1, p2, _)) => {
                let w = (p2 - p1) * 0.5;
                let off = ((p1 + p2) / 2.0) * 0.5 - half;
                widths.push(w);
                let ok = (w - wexp).abs() <= wtol && off.abs() <= maxoff;
                st = if ok { "pass" } else { "fail" };
                if !ok { bad += 1; }
            }
            None => {
                bad += 1;
                st = "fail";
            }
        }
        r.graphics.push(Graphic::point(o, st, ""));
    }
    let allowed = t.i("max_bad_stations", 0);
    r.pass_if(bad as i64 <= allowed);
    if bad as i64 > allowed {
        r.message = format!("{} of {} stations bad (gap, width or place)", bad, n);
    }
    let mean = if widths.is_empty() { 0.0 } else { widths.iter().sum::<f64>() / widths.len() as f64 };
    r.set("bad_stations", bad as i64).set("stations", n as i64).set("mean_width", jnum(mean, 2))
        .set("min_width", jnum(widths.iter().cloned().fold(f64::MAX, f64::min).min(1e9), 2))
        .set("max_width", jnum(widths.iter().cloned().fold(0.0, f64::max), 2));
    let st = r.status.clone();
    for w in path.windows(2) {
        r.graphics.insert(0, Graphic { kind: "line".into(), points: vec![w[0], w[1]], status: st.clone(), label: String::new(), role: "region".into() });
    }
    r
}

// ---------------------------------------------------------- N-point fixture
fn point_of(r: &ToolResult) -> Option<[f64; 2]> {
    let g = |k: &str| r.num(k);
    if let (Some(x), Some(y)) = (g("cx"), g("cy")) { return Some([x, y]); }
    if let (Some(x), Some(y)) = (g("x"), g("y")) { return Some([x, y]); }
    None
}

type RefCache = Mutex<HashMap<String, Vec<[f64; 2]>>>;

fn ref_cache() -> &'static RefCache {
    static C: OnceLock<RefCache> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn fixture_npoint(t: &ToolSpec, _frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let ids = t.strlist("points");
    if ids.len() < 2 {
        return ToolResult::error(t, "name at least two feature checks in 'points'");
    }
    let mut cur = vec![];
    for id in &ids {
        match ctx.results.get(id).filter(|r| r.status == "pass").and_then(point_of) {
            Some(p) => cur.push(p),
            None => {
                let mut r = ToolResult::new(t, "fail");
                r.message = format!("no good feature from {}", id);
                return r;
            }
        }
    }
    let Some(refi) = ctx.reference.clone() else { return ToolResult::error(t, "needs a reference picture") };
    // where the features sit on the reference: found once, by running them there
    let key = format!("{}|{}|{:p}", ctx.folder.display(), ids.join(","), Arc::as_ptr(&refi));
    let cached = ref_cache().lock().unwrap().get(&key).cloned();
    let src = match cached {
        Some(v) => v,
        None => {
            let mut rc = Ctx::new(Some(refi.clone()), ctx.folder.clone(), ctx.mm_per_px);
            rc.specs = ctx.specs.clone();
            let mut v = vec![];
            for id in &ids {
                let Some(spec) = ctx.specs.get(id) else { return ToolResult::error(t, format!("no check '{}'", id)) };
                let r = super::run(spec, &refi, &mut rc);
                match point_of(&r).filter(|_| r.status == "pass") {
                    Some(p) => v.push(p),
                    None => return ToolResult::error(t, format!("{} is not found on the reference picture", id)),
                }
            }
            ref_cache().lock().unwrap().insert(key, v.clone());
            v
        }
    };
    // similarity (rotation, scale, shift) by least squares (Umeyama, 2-D)
    let n = src.len() as f64;
    let (msx, msy) = (src.iter().map(|p| p[0]).sum::<f64>() / n, src.iter().map(|p| p[1]).sum::<f64>() / n);
    let (mdx, mdy) = (cur.iter().map(|p| p[0]).sum::<f64>() / n, cur.iter().map(|p| p[1]).sum::<f64>() / n);
    let (mut sa, mut sb, mut ss) = (0.0, 0.0, 0.0);
    for (p, q) in src.iter().zip(&cur) {
        let (x, y) = (p[0] - msx, p[1] - msy);
        let (u, v) = (q[0] - mdx, q[1] - mdy);
        sa += x * u + y * v;
        sb += x * v - y * u;
        ss += x * x + y * y;
    }
    let (a, b) = (sa / ss.max(1e-12), sb / ss.max(1e-12));
    let m = Affine { a, b: -b, c: b, d: a, tx: mdx - (a * msx - b * msy), ty: mdy - (b * msx + a * msy) };
    let rms = (src.iter().zip(&cur).map(|(p, q)| { let (x, y) = m.apply(p[0], p[1]); (x - q[0]).powi(2) + (y - q[1]).powi(2) }).sum::<f64>() / n).sqrt();
    let scale = (a * a + b * b).sqrt();
    let max_rms = t.f("max_rms", 2.0);
    let ok = rms <= max_rms && (scale - 1.0).abs() <= t.f("max_scale_change", 0.05);
    let mut r = ToolResult::new(t, if ok { "pass" } else { "fail" });
    if ok {
        ctx.fixture = Some(m);
    } else {
        r.message = if rms > max_rms { format!("points disagree by {:.2} px", rms) } else { format!("scale changed by {:.1}%", 100.0 * (scale - 1.0)) };
    }
    r.score = (1.0 - rms / (2.0 * max_rms)).max(0.0);
    r.set("shift_x_px", jnum(m.tx, 2)).set("shift_y_px", jnum(m.ty, 2)).set("angle_deg", jnum(m.angle_deg(), 3))
        .set("scale", jnum(scale, 5)).set("rms_px", jnum(rms, 3)).set("fixture", m.to_json());
    let st = r.status.clone();
    for w in cur.windows(2) { r.graphics.push(Graphic::line(w[0], w[1], &st, "")); }
    for p in &cur { r.graphics.push(Graphic::point(*p, &st, "")); }
    r
}

pub fn calibrate_npoint(t: &ToolSpec, _frame: &Img, ctx: &mut Ctx) -> ToolResult {
    // "px,py = wx,wy; ..."
    let mut pairs = vec![];
    for part in t.s("pairs", "").split(';') {
        let Some((a, b)) = part.split_once('=') else { continue };
        let nums = |s: &str| -> Option<[f64; 2]> {
            let v: Vec<f64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            if v.len() == 2 { Some([v[0], v[1]]) } else { None }
        };
        if let (Some(p), Some(w)) = (nums(a), nums(b)) { pairs.push((p, w)); }
    }
    if pairs.len() < 3 {
        return ToolResult::error(t, "give three point pairs or more: 'px,py = wx,wy; ...'");
    }
    // affine pixel -> world by least squares: solve for each world axis
    let mut ata = [[0f64; 3]; 3];
    let (mut atx, mut aty) = ([0f64; 3], [0f64; 3]);
    for (p, w) in &pairs {
        let row = [p[0], p[1], 1.0];
        for i in 0..3 {
            for j in 0..3 { ata[i][j] += row[i] * row[j]; }
            atx[i] += row[i] * w[0];
            aty[i] += row[i] * w[1];
        }
    }
    let solve = |m: [[f64; 3]; 3], v: [f64; 3]| -> Option<[f64; 3]> {
        let det = |m: [[f64; 3]; 3]| m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let d = det(m);
        if d.abs() < 1e-12 { return None; }
        let mut out = [0.0; 3];
        for k in 0..3 {
            let mut mk = m;
            for i in 0..3 { mk[i][k] = v[i]; }
            out[k] = det(mk) / d;
        }
        Some(out)
    };
    let (Some(fx), Some(fy)) = (solve(ata, atx), solve(ata, aty)) else { return ToolResult::error(t, "the points are on one line") };
    let rms = (pairs.iter().map(|(p, w)| (fx[0] * p[0] + fx[1] * p[1] + fx[2] - w[0]).powi(2) + (fy[0] * p[0] + fy[1] * p[1] + fy[2] - w[1]).powi(2)).sum::<f64>() / pairs.len() as f64).sqrt();
    let unit = (fx[0] * fy[1] - fx[1] * fy[0]).abs().sqrt();
    let max_rms = t.f("max_rms", 0.05);
    let mut r = ToolResult::new(t, "pass");
    r.pass_if(rms <= max_rms);
    if rms > max_rms { r.message = format!("fit error {:.4} (at most {:.4})", rms, max_rms); } else { ctx.mm_per_px = unit; }
    r.set("rms", jnum(rms, 5)).set("units_per_px", jnum(unit, 6)).set("pairs", pairs.len() as i64)
        .set("matrix", json!([[fx[0], fx[1], fx[2]], [fy[0], fy[1], fy[2]]]));
    r
}

// --------------------------------------------------------- trained surface
const MODEL_SIDE: usize = 160;

fn model_size(w: usize, h: usize, cap: usize) -> (usize, usize) {
    let long = w.max(h) as f64;
    let side = (cap as f64).min((MODEL_SIDE as f64).max((long / 2.0).round())) as usize;
    if w >= h { (side, ((side as f64 * h as f64 / w as f64).round() as usize).max(8)) } else { (((side as f64 * w as f64 / h as f64).round() as usize).max(8), side) }
}

fn normalise(d: &mut [f32]) {
    let n = d.len() as f64;
    let m = d.iter().map(|&v| v as f64).sum::<f64>() / n;
    let s = (d.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / n).sqrt();
    let s = if s > 1e-3 { s } else { 1.0 };
    for v in d.iter_mut() { *v = ((*v as f64 - m) / s) as f32; }
}

fn prepare(patch: &Img, size: (usize, usize), flatten: bool) -> Plane {
    let g = patch.gray().resize(size.0, size.1);
    let mut p = Plane::from_gray(&g);
    // 3 x 3 Gaussian
    let k = [0.25f32, 0.5, 0.25];
    let (w, h) = (p.w, p.h);
    let mut tmp = p.d.clone();
    for y in 0..h { for x in 0..w { let mut s = 0.0; for (j, kv) in k.iter().enumerate() { let xx = (x as i64 + j as i64 - 1).clamp(0, w as i64 - 1) as usize; s += kv * p.d[y * w + xx]; } tmp[y * w + x] = s; } }
    for y in 0..h { for x in 0..w { let mut s = 0.0; for (j, kv) in k.iter().enumerate() { let yy = (y as i64 + j as i64 - 1).clamp(0, h as i64 - 1) as usize; s += kv * tmp[yy * w + x]; } p.d[y * w + x] = s; } }
    if flatten {
        let r = ((size.0.max(size.1) as f64 / 24.0).max(4.0) * 1.2) as usize;
        let b = p.box_blur(r).box_blur(r);
        for (v, bv) in p.d.iter_mut().zip(b.d) { *v -= bv; }
    }
    normalise(&mut p.d);
    p
}

fn deviation(p: &Plane, mean: &[f32], std: &[f32], k: f32) -> (usize, Vec<u8>) {
    let mask: Vec<u8> = p.d.iter().zip(mean.iter().zip(std)).map(|(v, (m, s))| if (v - m).abs() / s > k { 255 } else { 0 }).collect();
    let mask = img::open(&mask, p.w, p.h, 2);
    let area = img::components(&mask, p.w, p.h, false).iter().map(|b| b.area).max().unwrap_or(0);
    (area, mask)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) }
    fn uni(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.next() }
    fn normal(&mut self) -> f64 { let (u, v) = (self.next().max(1e-12), self.next()); (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos() }
}

fn augment(p: &Plane, rng: &mut Lcg) -> Plane {
    let (w, h) = (p.w, p.h);
    let m = Affine::rotate_about(rng.uni(-1.5, 1.5), w as f64 / 2.0, h as f64 / 2.0).then_after(&Affine::translate(rng.uni(-1.2, 1.2), rng.uni(-1.2, 1.2)));
    let mut a = geom::warp_plane(p, &m, w, h);
    let (gain, ramp) = (rng.uni(0.9, 1.1), rng.uni(-0.15, 0.15));
    for y in 0..h { for x in 0..w { let i = y * w + x; a.d[i] = (a.d[i] as f64 * gain + ramp * (2.0 * x as f64 / (w - 1).max(1) as f64 - 1.0) + rng.normal() * 0.04) as f32; } }
    normalise(&mut a.d);
    a
}

pub fn model_path(t: &ToolSpec, folder: &std::path::Path) -> PathBuf {
    let m = t.s("model", "");
    let m = if m.trim().is_empty() { format!("models/{}.npz", t.id) } else { m };
    let p = PathBuf::from(&m);
    if p.is_absolute() { p } else { folder.join(p) }
}

pub struct SurfaceModel {
    mean: Vec<f32>,
    std: Vec<f32>,
    size: (usize, usize),
    k: f32,
    threshold: f64,
    flatten: bool,
}

fn models() -> &'static Mutex<HashMap<(PathBuf, u64), Arc<SurfaceModel>>> {
    static M: OnceLock<Mutex<HashMap<(PathBuf, u64), Arc<SurfaceModel>>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}

fn load_model(path: &std::path::Path) -> Result<Arc<SurfaceModel>, String> {
    let mt = std::fs::metadata(path).and_then(|m| m.modified()).map(|t| t.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)).unwrap_or(0);
    let key = (path.to_path_buf(), mt);
    if let Some(m) = models().lock().unwrap().get(&key) { return Ok(m.clone()); }
    let z = npz::read(path)?;
    if z.contains_key("kind") {
        return Err("a deep-network model: retrain this check here (Model: statistical)".into());
    }
    let get = |k: &str| z.get(k).ok_or(format!("the model has no '{}'", k));
    let size = get("size")?;
    let m = Arc::new(SurfaceModel {
        mean: get("mean")?.as_f32(), std: get("std")?.as_f32(),
        size: (size.as_f64()[0] as usize, size.as_f64()[1] as usize),
        k: get("k")?.as_f64()[0] as f32, threshold: get("threshold")?.as_f64()[0],
        flatten: z.get("flatten").map(|v| v.as_f64()[0] != 0.0).unwrap_or(false),
    });
    models().lock().unwrap().insert(key, m.clone());
    Ok(m)
}

pub fn trained_surface(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let path = model_path(t, &ctx.folder);
    if !path.exists() {
        return ToolResult::error(t, "not trained yet: add OK pictures and press Train");
    }
    let m = match load_model(&path) { Ok(m) => m, Err(e) => return ToolResult::error(t, e) };
    let Some(reg) = ctx.region(t, frame) else { return ToolResult::error(t, "no region drawn") };
    let clipped = (0..reg.patch.w * reg.patch.h).filter(|&i| { let p = reg.patch.rgb(i % reg.patch.w, i / reg.patch.w); p[0].max(p[1]).max(p[2]) >= 254 }).count() as f64 / (reg.patch.w * reg.patch.h).max(1) as f64;
    let limit = t.f("max_clipped", 0.15);
    if clipped > limit {
        let mut r = ToolResult::error(t, format!("overexposed: {:.0}% of the region is clipped at white - reduce the exposure or lighting", 100.0 * clipped));
        r.set("clipped_pct", jnum(100.0 * clipped, 1));
        return r;
    }
    let p = prepare(&reg.patch, m.size, m.flatten);
    if p.d.len() != m.mean.len() {
        return ToolResult::error(t, "the model does not fit this region: train again");
    }
    let sig = t.f("sensitivity_sigma", 0.0);
    let k = if sig > 0.0 { sig as f32 } else { m.k };
    let (area, mask) = deviation(&p, &m.mean, &m.std, k);
    let over = t.f("max_defect_area", 0.0);
    let thr = if over > 0.0 { over } else { m.threshold };
    let ok = area as f64 <= thr;
    let mut r = ToolResult::new(t, if ok { "pass" } else { "fail" });
    r.score = (1.0 - area as f64 / (2.0 * thr).max(1.0)).max(0.0);
    r.set("defect_area", area as i64).set("threshold", jnum(thr, 1));
    if !ok {
        r.message = format!("defect area {} over {:.0}", area, thr);
        let (sx, sy) = (reg.patch.w as f64 / m.size.0 as f64, reg.patch.h as f64 / m.size.1 as f64);
        let mut blobs = img::components(&mask, p.w, p.h, false);
        blobs.sort_by(|a, b| b.area.cmp(&a.area));
        for b in blobs.iter().take(3) {
            r.graphics.push(Graphic::patch_box(&reg, b.x0 as f64 * sx, b.y0 as f64 * sy, (b.x1 + 1) as f64 * sx, (b.y1 + 1) as f64 * sy, "fail", "defect"));
        }
    }
    r
}

/// Train from OK (and optionally NG) pictures: the statistical model,
/// written as .npz (the same file Vision Station reads).
pub fn train_surface(t: &ToolSpec, folder: &std::path::Path, ctx_for: &dyn Fn(&Img) -> Ctx, ok: &[Img], ng: &[Img]) -> Result<Value, String> {
    if ok.is_empty() {
        return Err("training needs at least one OK picture".into());
    }
    let cap = t.i("resolution", 640).max(160) as usize;
    let flatten = t.b("flatten_lighting", false);
    let k = 4.0f32;
    let mut size = None;
    let mut prep = |im: &Img| -> Option<Plane> {
        let c = ctx_for(im);
        let reg = c.region(t, im)?;
        let s = *size.get_or_insert_with(|| model_size(reg.patch.w, reg.patch.h, cap));
        Some(prepare(&reg.patch, s, flatten))
    };
    let okp: Vec<Plane> = ok.iter().filter_map(&mut prep).collect();
    let ngp: Vec<Plane> = ng.iter().filter_map(&mut prep).collect();
    let size = size.ok_or("no region drawn")?;
    let mut pool = okp.clone();
    let mut rng = Lcg(7);
    if okp.len() < 8 {
        let per = ((24 - okp.len()) as f64 / okp.len() as f64).ceil() as usize;
        for p in &okp { for _ in 0..per { pool.push(augment(p, &mut rng)); } }
    }
    let n = pool.len() as f32;
    let len = pool[0].d.len();
    let mut mean = vec![0f32; len];
    for p in &pool { for (m, v) in mean.iter_mut().zip(&p.d) { *m += v / n; } }
    let mut std = vec![0f32; len];
    for p in &pool { for ((s, v), m) in std.iter_mut().zip(&p.d).zip(&mean) { *s += (v - m).powi(2) / n; } }
    for s in std.iter_mut() { *s = s.sqrt(); }
    // spread dilated (an edge may wander a pixel) and floored
    let (w, h) = (pool[0].w, pool[0].h);
    let mut dil = std.clone();
    for y in 0..h { for x in 0..w {
        let mut m = 0f32;
        for yy in y.saturating_sub(1)..(y + 2).min(h) { for xx in x.saturating_sub(1)..(x + 2).min(w) { m = m.max(std[yy * w + xx]); } }
        dil[y * w + x] = m.max(0.12);
    } }
    let oks: Vec<usize> = okp.iter().map(|p| deviation(p, &mean, &dil, k).0).collect();
    let ngs: Vec<usize> = ngp.iter().map(|p| deviation(p, &mean, &dil, k).0).collect();
    let worst = *oks.iter().max().unwrap_or(&0) as f64;
    let clear = (worst * 1.5).max(worst + 12.0);
    let separable: Vec<f64> = ngs.iter().map(|&v| v as f64).filter(|&v| v > clear).collect();
    let threshold = if let Some(best) = separable.iter().cloned().reduce(f64::min) { (worst + best) / 2.0 } else { clear };
    let path = model_path(t, folder);
    if let Some(d) = path.parent() { std::fs::create_dir_all(d).map_err(|e| e.to_string())?; }
    let mut z = npz::Npz::new();
    z.f32("mean", &mean, &[h, w]);
    z.f32("std", &dil, &[h, w]);
    z.i64("size", &[size.0 as i64, size.1 as i64], &[2]);
    z.f64("k", &[k as f64], &[]);
    z.f64("threshold", &[threshold], &[]);
    z.i64("flatten", &[flatten as i64], &[]);
    z.write(&path)?;
    Ok(json!({"real_ok": okp.len(), "ng": ngp.len(), "augmented": pool.len() - okp.len(),
              "ok_scores": oks, "ng_scores": ngs, "worst_ok": worst, "threshold": threshold,
              "separates": ngs.iter().all(|&v| v as f64 > clear), "model": path.display().to_string()}))
}
