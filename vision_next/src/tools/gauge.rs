//! Edges and geometry: measure length, calibrate scale, find line / circle,
//! caliper pairs, distance, intersect.

use crate::geom::{self};
use crate::img::{Img, Plane};
use crate::model::{jnum, Ctx, Graphic, ToolResult, ToolSpec};
use serde_json::json;

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub pos: f64,
    pub strength: f64, // signed: positive dark -> light along the profile
}

/// Edges of a 1-D profile: smoothed derivative peaks, sub-pixel.
pub fn profile_edges(profile: &[f64], min_contrast: f64, polarity: &str) -> Vec<Edge> {
    let n = profile.len();
    if n < 5 {
        return vec![];
    }
    let k = [0.054, 0.242, 0.399, 0.242, 0.054];
    let s: Vec<f64> = (0..n).map(|i| {
        let mut acc = 0.0;
        for (j, &kv) in k.iter().enumerate() {
            let idx = (i as i64 + j as i64 - 2).clamp(0, n as i64 - 1) as usize;
            acc += kv * profile[idx];
        }
        acc
    }).collect();
    let mut d = vec![0.0; n];
    for i in 1..n - 1 {
        d[i] = (s[i + 1] - s[i - 1]) / 2.0;
    }
    let mut out = vec![];
    for i in 2..n - 2 {
        let a = d[i].abs();
        if a < min_contrast || a < d[i - 1].abs() || a < d[i + 1].abs() {
            continue;
        }
        if a == d[i - 1].abs() && i > 2 {
            continue; // plateau: keep the first
        }
        let ok = match polarity {
            "dark_to_light" => d[i] > 0.0,
            "light_to_dark" => d[i] < 0.0,
            _ => true,
        };
        if !ok {
            continue;
        }
        let (l, c, r) = (d[i - 1].abs(), a, d[i + 1].abs());
        let den = l - 2.0 * c + r;
        let off = if den.abs() > 1e-9 { (0.5 * (l - r) / den).clamp(-0.5, 0.5) } else { 0.0 };
        out.push(Edge { pos: i as f64 + off + 0.5, strength: d[i] });
    }
    out
}

/// Mean of each column (along_x) or row of a plane.
fn projection(p: &Plane, along_x: bool) -> Vec<f64> {
    if along_x {
        (0..p.w).map(|x| (0..p.h).map(|y| p.at(x, y) as f64).sum::<f64>() / p.h as f64).collect()
    } else {
        (0..p.h).map(|y| (0..p.w).map(|x| p.at(x, y) as f64).sum::<f64>() / p.w as f64).collect()
    }
}

fn plane_for(t: &ToolSpec, patch: &Img) -> Plane {
    let g = match t.s("channel", "auto").as_str() {
        "red" => patch.channel(0),
        "green" => patch.channel(1),
        "blue" => patch.channel(2),
        _ => patch.gray(),
    };
    Plane::from_gray(&g)
}

fn two_edges(t: &ToolSpec, frame: &Img, ctx: &Ctx, max_edges: usize)
    -> Result<(crate::geom::Region, bool, Vec<Edge>), ToolResult> {
    let Some(reg) = ctx.region(t, frame) else { return Err(ToolResult::error(t, "no region drawn")) };
    let along_x = t.s("axis", "x") != "y";
    let p = plane_for(t, &reg.patch);
    let prof = projection(&p, along_x);
    let mut edges = profile_edges(&prof, t.f("min_contrast", 2.0), &t.s("polarity", "any"));
    edges.sort_by(|a, b| b.strength.abs().partial_cmp(&a.strength.abs()).unwrap());
    edges.truncate(max_edges.max(2));
    edges.sort_by(|a, b| a.pos.partial_cmp(&b.pos).unwrap());
    Ok((reg, along_x, edges))
}

fn edge_line(reg: &crate::geom::Region, along_x: bool, pos: f64, st: &str) -> Graphic {
    let (w, h) = (reg.patch.w as f64, reg.patch.h as f64);
    if along_x {
        Graphic::line(reg.map(pos, 0.0), reg.map(pos, h), st, "")
    } else {
        Graphic::line(reg.map(0.0, pos), reg.map(w, pos), st, "")
    }
}

pub fn measure_length(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let (reg, along_x, edges) = match two_edges(t, frame, ctx, t.i("max_edges", 2).max(2) as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if edges.len() < 2 {
        let mut r = ToolResult::new(t, "fail");
        r.message = format!("found {} edge(s), needs 2", edges.len());
        return r;
    }
    let px = edges[edges.len() - 1].pos - edges[0].pos;
    let mm = ctx.mm(t);
    let mut r = ToolResult::new(t, "pass");
    r.set("length_px", jnum(px, 3)).set("edges", edges.len() as i64);
    let nominal = t.f("nominal_mm", 0.0);
    let tol = t.f("tolerance_mm", 0.05);
    if mm > 0.0 {
        let v = px * mm;
        r.set("length_mm", jnum(v, 4));
        if nominal > 0.0 && (v - nominal).abs() > tol {
            r.status = "fail".into();
            r.message = format!("{:.3} mm, nominal {:.3} ± {:.3}", v, nominal, tol);
        }
    } else if nominal > 0.0 {
        r.status = "warn".into();
        r.message = "no scale: calibrate first to judge in mm".into();
    }
    let st = r.status.clone();
    for e in &edges {
        r.graphics.push(edge_line(&reg, along_x, e.pos, &st));
    }
    r
}

pub fn calibrate_scale(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let (reg, along_x, edges) = match two_edges(t, frame, ctx, 2) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if edges.len() < 2 {
        let mut r = ToolResult::new(t, "fail");
        r.message = format!("found {} edge(s), needs 2", edges.len());
        return r;
    }
    let px = edges[1].pos - edges[0].pos;
    let known = t.f("known_mm", 10.0);
    let mm = known / px.max(1e-9);
    ctx.mm_per_px = mm;
    let mut r = ToolResult::new(t, "pass");
    r.set("length_px", jnum(px, 3)).set("mm_per_px", jnum(mm, 6));
    for e in &edges {
        r.graphics.push(edge_line(&reg, along_x, e.pos, "pass"));
    }
    r
}

fn strongest(edges: Vec<Edge>) -> Option<Edge> {
    edges.into_iter().max_by(|a, b| a.strength.abs().partial_cmp(&b.strength.abs()).unwrap())
}

/// Drop the worst point until the fit is good or too few are left.
fn prune<F: Fn(&[[f64; 2]]) -> Option<(f64, Vec<f64>)>>(mut pts: Vec<[f64; 2]>, min: usize, max_rms: f64, fit: F) -> (Vec<[f64; 2]>, f64) {
    loop {
        let Some((rms, res)) = fit(&pts) else { return (pts, f64::MAX) };
        if rms <= max_rms || pts.len() <= min {
            return (pts, rms);
        }
        let worst = res.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|x| x.0).unwrap();
        pts.remove(worst);
    }
}

pub fn find_line(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(reg) = ctx.region(t, frame) else { return ToolResult::error(t, "no region drawn") };
    let p = plane_for(t, &reg.patch);
    let horizontal = p.w >= p.h; // the line runs along the region's long side
    let (len, depth) = if horizontal { (p.w, p.h) } else { (p.h, p.w) };
    let n = t.i("num_calipers", 12).max(3) as usize;
    let cw = t.i("caliper_width", 6).max(1) as usize;
    let mc = t.f("min_contrast", 8.0);
    let pol = t.s("polarity", "any");
    let mut pts = vec![];
    for i in 0..n {
        let u = ((i as f64 + 0.5) * len as f64 / n as f64) as usize;
        let u0 = u.saturating_sub(cw / 2);
        let u1 = (u0 + cw).min(len);
        let prof: Vec<f64> = (0..depth).map(|v| {
            (u0..u1).map(|uu| if horizontal { p.at(uu, v) } else { p.at(v, uu) } as f64).sum::<f64>() / (u1 - u0).max(1) as f64
        }).collect();
        if let Some(e) = strongest(profile_edges(&prof, mc, &pol)) {
            pts.push(if horizontal { [u as f64 + 0.5, e.pos] } else { [e.pos, u as f64 + 0.5] });
        }
    }
    let min_found = t.i("min_found", 6).max(2) as usize;
    let max_rms = t.f("max_rms", 1.5);
    let (pts, rms) = prune(pts, min_found, max_rms, |q| {
        let (c, d, rms) = geom::fit_line(q)?;
        Some((rms, q.iter().map(|p| (-(p[0] - c[0]) * d[1] + (p[1] - c[1]) * d[0]).abs()).collect()))
    });
    let mut r = ToolResult::new(t, "pass");
    let found = pts.len();
    r.set("found", found as i64);
    let Some((c, d, _)) = geom::fit_line(&pts) else {
        r.status = "fail".into();
        r.message = format!("found {} edge point(s)", found);
        return r;
    };
    // ends of the line at the region's two ends
    let ends: Vec<[f64; 2]> = if horizontal {
        [0.0, len as f64].iter().map(|&x| { let tt = if d[0].abs() > 1e-9 { (x - c[0]) / d[0] } else { 0.0 }; [x, c[1] + tt * d[1]] }).collect()
    } else {
        [0.0, len as f64].iter().map(|&y| { let tt = if d[1].abs() > 1e-9 { (y - c[1]) / d[1] } else { 0.0 }; [c[0] + tt * d[0], y] }).collect()
    };
    let a = reg.map(ends[0][0], ends[0][1]);
    let b = reg.map(ends[1][0], ends[1][1]);
    let ok = found >= min_found && rms <= max_rms;
    r.pass_if(ok);
    if !ok {
        r.message = format!("{} points agree (needs {}), fit error {:.2} px", found, min_found, rms);
    }
    let ang = (b[1] - a[1]).atan2(b[0] - a[0]).to_degrees();
    r.set("x1", jnum(a[0], 2)).set("y1", jnum(a[1], 2)).set("x2", jnum(b[0], 2)).set("y2", jnum(b[1], 2))
        .set("cx", jnum((a[0] + b[0]) / 2.0, 2)).set("cy", jnum((a[1] + b[1]) / 2.0, 2))
        .set("angle", jnum(ang, 3)).set("rms", jnum(rms, 3));
    let st = r.status.clone();
    r.graphics.push(Graphic::line(a, b, &st, ""));
    for q in &pts {
        r.graphics.push(Graphic::point(reg.map(q[0], q[1]), &st, ""));
    }
    r
}

pub fn find_circle(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(reg) = ctx.region(t, frame) else { return ToolResult::error(t, "no region drawn") };
    let p = plane_for(t, &reg.patch);
    let (cx, cy) = (p.w as f64 / 2.0, p.h as f64 / 2.0);
    let rr = cx.min(cy);
    let (r0, r1) = (t.f("search_inner", 0.45) * rr, t.f("search_outer", 1.0) * rr);
    let n = t.i("num_calipers", 24).max(6) as usize;
    let mc = t.f("min_contrast", 8.0);
    let pol = t.s("polarity", "any");
    let mut pts = vec![];
    for i in 0..n {
        let th = i as f64 / n as f64 * std::f64::consts::TAU;
        let (c, s) = (th.cos(), th.sin());
        let steps = ((r1 - r0) / 0.5).max(5.0) as usize;
        let prof: Vec<f64> = (0..steps).map(|k| {
            let rad = r0 + k as f64 * 0.5;
            p.sample((cx + c * rad) as f32, (cy + s * rad) as f32) as f64
        }).collect();
        // profile in half-pixel steps: contrast per half pixel
        if let Some(e) = strongest(profile_edges(&prof, mc / 2.0, &pol)) {
            let rad = r0 + (e.pos - 0.5) * 0.5;
            pts.push([cx + c * rad, cy + s * rad]);
        }
    }
    let min_found = t.i("min_found", 10).max(3) as usize;
    let max_rms = t.f("max_rms", 1.5);
    let (pts, rms) = prune(pts, min_found, max_rms, |q| {
        let (x, y, rad, rms) = geom::fit_circle(q)?;
        Some((rms, q.iter().map(|p| (((p[0] - x).powi(2) + (p[1] - y).powi(2)).sqrt() - rad).abs()).collect()))
    });
    let mut r = ToolResult::new(t, "pass");
    r.set("found", pts.len() as i64);
    let Some((x, y, rad, _)) = geom::fit_circle(&pts) else {
        r.status = "fail".into();
        r.message = format!("found {} edge point(s)", pts.len());
        return r;
    };
    let c = reg.map(x, y);
    let mut why = vec![];
    if pts.len() < min_found { why.push(format!("{} points agree (needs {})", pts.len(), min_found)); }
    if rms > max_rms { why.push(format!("fit error {:.2} px", rms)); }
    let (rmin, rmax) = (t.f("min_radius", 0.0), t.f("max_radius", 0.0));
    if rmin > 0.0 && rad < rmin { why.push(format!("radius {:.1} below {:.1}", rad, rmin)); }
    if rmax > 0.0 && rad > rmax { why.push(format!("radius {:.1} above {:.1}", rad, rmax)); }
    r.pass_if(why.is_empty());
    r.message = why.join("; ");
    r.set("cx", jnum(c[0], 2)).set("cy", jnum(c[1], 2)).set("radius", jnum(rad, 3))
        .set("diameter", jnum(2.0 * rad, 3)).set("rms", jnum(rms, 3));
    let mm = ctx.mm(t);
    if mm > 0.0 {
        r.set("diameter_mm", jnum(2.0 * rad * mm, 4));
    }
    let st = r.status.clone();
    let poly: Vec<[f64; 2]> = geom::circle_poly(x, y, rad, 64).iter().map(|q| reg.map(q[0], q[1])).collect();
    r.graphics.push(Graphic::poly(poly, &st, ""));
    r.graphics.push(Graphic::point(c, &st, ""));
    r
}

pub fn caliper(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(reg) = ctx.region(t, frame) else { return ToolResult::error(t, "no region drawn") };
    let p = plane_for(t, &reg.patch);
    let prof = projection(&p, true);
    let edges = profile_edges(&prof, t.f("min_contrast", 8.0), "any");
    let dark = t.s("pair_polarity", "dark") == "dark";
    let mut pairs: Vec<(f64, f64)> = vec![];
    let mut open: Option<f64> = None;
    for e in &edges {
        let opens = if dark { e.strength < 0.0 } else { e.strength > 0.0 };
        if opens {
            open = Some(e.pos);
        } else if let Some(o) = open.take() {
            pairs.push((o, e.pos));
        }
    }
    let mm = ctx.mm(t);
    let unit = if mm > 0.0 { mm } else { 1.0 };
    let (nom, tol) = (t.f("nominal_width", 0.0), t.f("tolerance", 0.0));
    pairs.truncate(t.i("max_results", 20).max(1) as usize);
    let widths: Vec<f64> = pairs.iter().map(|(a, b)| (b - a) * unit).collect();
    let mut bad = 0;
    if nom > 0.0 && tol > 0.0 {
        bad = widths.iter().filter(|w| (*w - nom).abs() > tol).count();
    }
    let expect = t.i("expect_count", 0);
    let mut r = ToolResult::new(t, "pass");
    let mut why = vec![];
    if expect > 0 && pairs.len() as i64 != expect { why.push(format!("{} pairs, expected {}", pairs.len(), expect)); }
    if bad > 0 { why.push(format!("{} width(s) out of tolerance", bad)); }
    r.pass_if(why.is_empty());
    r.message = why.join("; ");
    let centres: Vec<f64> = pairs.iter().map(|(a, b)| (a + b) / 2.0).collect();
    let pitch = if centres.len() > 1 { (centres[centres.len() - 1] - centres[0]) / (centres.len() - 1) as f64 * unit } else { 0.0 };
    r.set("count", pairs.len() as i64)
        .set("widths", json!(widths.iter().map(|w| (w * 1000.0).round() / 1000.0).collect::<Vec<_>>()))
        .set("pitch", jnum(pitch, 3))
        .set("unit", if mm > 0.0 { "mm" } else { "px" });
    let st = r.status.clone();
    let h = reg.patch.h as f64;
    for (a, b) in &pairs {
        r.graphics.push(Graphic::patch_box(&reg, *a, 0.0, *b, h, &st, ""));
    }
    r
}

#[derive(Clone, Copy, Debug)]
pub enum Feature {
    Point([f64; 2]),
    Line([f64; 2], [f64; 2]),
    Circle([f64; 2], f64),
}

pub fn feature(ctx: &Ctx, id: &str) -> Option<Feature> {
    let r = ctx.results.get(id)?;
    let g = |k: &str| r.num(k);
    if let (Some(x1), Some(y1), Some(x2), Some(y2)) = (g("x1"), g("y1"), g("x2"), g("y2")) {
        return Some(Feature::Line([x1, y1], [x2, y2]));
    }
    if let (Some(x), Some(y), Some(rad)) = (g("cx"), g("cy"), g("radius")) {
        return Some(Feature::Circle([x, y], rad));
    }
    if let (Some(x), Some(y)) = (g("cx"), g("cy")) {
        return Some(Feature::Point([x, y]));
    }
    if let (Some(x), Some(y)) = (g("x"), g("y")) {
        return Some(Feature::Point([x, y]));
    }
    None
}

pub fn centre(f: Feature) -> [f64; 2] {
    match f {
        Feature::Point(p) => p,
        Feature::Line(a, b) => [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0],
        Feature::Circle(c, _) => c,
    }
}

fn point_line(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l = (dx * dx + dy * dy).sqrt().max(1e-12);
    ((p[0] - a[0]) * dy - (p[1] - a[1]) * dx).abs() / l
}

pub fn distance(t: &ToolSpec, _frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let (f1, f2) = (t.s("first", ""), t.s("second", ""));
    let Some(a) = feature(ctx, &f1) else { return ToolResult::error(t, format!("'{}' gave no point, line or circle", f1)) };
    let mm = ctx.mm_per_px;
    let unit = if mm > 0.0 { mm } else { 1.0 };
    let mut r = ToolResult::new(t, "pass");
    let (val, g): (f64, Option<([f64; 2], [f64; 2])>) = if f2.trim().is_empty() {
        match a {
            Feature::Circle(_, rad) => (2.0 * rad, None),
            Feature::Line(p, q) => (((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt(), Some((p, q))),
            Feature::Point(_) => return ToolResult::error(t, "a point has no size: name a second feature"),
        }
    } else {
        let Some(b) = feature(ctx, &f2) else { return ToolResult::error(t, format!("'{}' gave no point, line or circle", f2)) };
        match (a, b) {
            (Feature::Line(p, q), Feature::Line(u, v)) => {
                let a1 = (q[1] - p[1]).atan2(q[0] - p[0]);
                let a2 = (v[1] - u[1]).atan2(v[0] - u[0]);
                let mut ang = (a1 - a2).to_degrees().abs() % 180.0;
                if ang > 90.0 { ang = 180.0 - ang; }
                r.set("angle", jnum(ang, 3));
                let m = centre(a);
                (point_line(m, u, v), Some((m, m)))
            }
            (x, Feature::Line(u, v)) | (Feature::Line(u, v), x) => {
                let c = centre(x);
                (point_line(c, u, v), Some((c, c)))
            }
            (x, y) => {
                let (p, q) = (centre(x), centre(y));
                (((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt(), Some((p, q)))
            }
        }
    };
    let v = val * unit;
    r.set("distance", jnum(v, 4)).set("unit", if mm > 0.0 { "mm" } else { "px" });
    let (lo, hi) = (t.f("min_value", 0.0), t.f("max_value", 0.0));
    let mut why = vec![];
    if lo > 0.0 && v < lo { why.push(format!("{:.3} below {:.3}", v, lo)); }
    if hi > 0.0 && v > hi { why.push(format!("{:.3} above {:.3}", v, hi)); }
    r.pass_if(why.is_empty());
    r.message = why.join("; ");
    if let Some((p, q)) = g {
        let st = r.status.clone();
        r.graphics.push(Graphic::line(p, q, &st, &format!("{:.3}", v)));
    }
    r
}

fn line_line(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> Option<[f64; 2]> {
    let (r, s) = ([b[0] - a[0], b[1] - a[1]], [d[0] - c[0], d[1] - c[1]]);
    let den = r[0] * s[1] - r[1] * s[0];
    if den.abs() < 1e-12 {
        return None;
    }
    let tt = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / den;
    Some([a[0] + tt * r[0], a[1] + tt * r[1]])
}

pub fn intersect(t: &ToolSpec, _frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let (f1, f2) = (t.s("first", ""), t.s("second", ""));
    let (Some(a), Some(b)) = (feature(ctx, &f1), feature(ctx, &f2)) else {
        return ToolResult::error(t, "both features must be found lines or circles");
    };
    let mut r = ToolResult::new(t, "pass");
    let mut pts: Vec<[f64; 2]> = vec![];
    match (a, b) {
        (Feature::Line(p, q), Feature::Line(u, v)) => {
            if let Some(x) = line_line(p, q, u, v) { pts.push(x); }
            let a1 = (q[1] - p[1]).atan2(q[0] - p[0]);
            let a2 = (v[1] - u[1]).atan2(v[0] - u[0]);
            let ang = ((a1 - a2).to_degrees().rem_euclid(180.0) * 1000.0).round() / 1000.0;
            r.set("angle", ang);
            let (lo, hi) = (t.f("min_angle", 0.0), t.f("max_angle", 0.0));
            if (lo > 0.0 && ang < lo) || (hi > 0.0 && ang > hi) {
                r.status = "fail".into();
                r.message = format!("angle {:.2}°", ang);
            }
        }
        (Feature::Line(p, q), Feature::Circle(c, rad)) | (Feature::Circle(c, rad), Feature::Line(p, q)) => {
            let d = [q[0] - p[0], q[1] - p[1]];
            let f = [p[0] - c[0], p[1] - c[1]];
            let (aa, bb, cc) = (d[0] * d[0] + d[1] * d[1], 2.0 * (f[0] * d[0] + f[1] * d[1]), f[0] * f[0] + f[1] * f[1] - rad * rad);
            let disc = bb * bb - 4.0 * aa * cc;
            if disc >= 0.0 && aa > 0.0 {
                for sgn in [-1.0, 1.0] {
                    let tt = (-bb + sgn * disc.sqrt()) / (2.0 * aa);
                    pts.push([p[0] + tt * d[0], p[1] + tt * d[1]]);
                }
            }
        }
        (Feature::Circle(c1, r1), Feature::Circle(c2, r2)) => {
            let dd = ((c2[0] - c1[0]).powi(2) + (c2[1] - c1[1]).powi(2)).sqrt();
            if dd > 0.0 && dd <= r1 + r2 && dd >= (r1 - r2).abs() {
                let a0 = (r1 * r1 - r2 * r2 + dd * dd) / (2.0 * dd);
                let h = (r1 * r1 - a0 * a0).max(0.0).sqrt();
                let m = [c1[0] + a0 * (c2[0] - c1[0]) / dd, c1[1] + a0 * (c2[1] - c1[1]) / dd];
                pts.push([m[0] + h * (c2[1] - c1[1]) / dd, m[1] - h * (c2[0] - c1[0]) / dd]);
                pts.push([m[0] - h * (c2[1] - c1[1]) / dd, m[1] + h * (c2[0] - c1[0]) / dd]);
            }
        }
        _ => return ToolResult::error(t, "both features must be lines or circles"),
    }
    let expect = t.i("expect_points", 0);
    if pts.is_empty() || (expect > 0 && pts.len() as i64 != expect) {
        r.status = "fail".into();
        r.message = format!("{} intersection(s)", pts.len());
    }
    if let Some(p) = pts.first() {
        r.set("x", jnum(p[0], 2)).set("y", jnum(p[1], 2));
    }
    r.set("points", json!(pts));
    let st = r.status.clone();
    for p in pts {
        r.graphics.push(Graphic::point(p, &st, ""));
    }
    r
}
