//! Brightness, colour and surface checks.

use crate::img::{self, hsv, lab, Blob, Img, Plane};
use crate::model::{jnum, Ctx, Graphic, ToolResult, ToolSpec};
use serde_json::json;

fn channel_of(patch: &Img, space: &str) -> Vec<u8> {
    match space {
        "hsv_v" => (0..patch.w * patch.h).map(|i| { let p = patch.rgb(i % patch.w, i / patch.w); p[0].max(p[1]).max(p[2]) }).collect(),
        "hsv_s" => (0..patch.w * patch.h).map(|i| hsv(patch.rgb(i % patch.w, i / patch.w)).1 as u8).collect(),
        _ => patch.gray().data,
    }
}

pub fn count_blob(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(reg) = ctx.region(t, frame) else { return ToolResult::error(t, "no region drawn") };
    let (w, h) = (reg.patch.w, reg.patch.h);
    let ch = channel_of(&reg.patch, &t.s("space", "hsv_v"));
    let thr_s = t.s("threshold", "otsu");
    let thr = match thr_s.trim().parse::<f64>() {
        Ok(v) => v.clamp(0.0, 255.0) as u8,
        Err(_) => img::otsu(&ch),
    };
    let invert = t.b("invert", false);
    let mut mask: Vec<u8> = ch.iter().enumerate().map(|(i, &v)| {
        let on = if invert { v < thr } else { v > thr };
        if on && reg.inside(i % w, i / w) { 255 } else { 0 }
    }).collect();
    let k = t.i("open_px", 3).max(0) as usize;
    if k > 1 {
        mask = img::open(&mask, w, h, k);
    }
    let (amin, amax) = (t.i("min_area", 40).max(1) as usize, t.i("max_area", 1_000_000) as usize);
    let mut blobs: Vec<Blob> = img::components(&mask, w, h, true).into_iter()
        .filter(|b| b.area >= amin && b.area <= amax).collect();
    // reading order: rows (within row_tol), then left to right
    let tol = t.f("row_tol", 20.0);
    blobs.sort_by(|a, b| a.cy.partial_cmp(&b.cy).unwrap());
    let mut rows: Vec<Vec<Blob>> = vec![];
    for b in blobs {
        match rows.last_mut() {
            Some(r) if (b.cy - r[0].cy).abs() <= tol => r.push(b),
            _ => rows.push(vec![b]),
        }
    }
    let mut blobs: Vec<Blob> = vec![];
    for mut r in rows {
        r.sort_by(|a, b| a.cx.partial_cmp(&b.cx).unwrap());
        blobs.extend(r);
    }
    let n = blobs.len() as i64;
    let expect = t.i("expect_count", 4);
    let mut r = ToolResult::new(t, "pass");
    r.pass_if(n == expect);
    r.score = if expect > 0 { (1.0 - ((n - expect).abs() as f64 / expect as f64)).max(0.0) } else { 1.0 };
    r.set("count", n).set("expected", expect).set("threshold", thr as i64);
    r.set("areas", json!(blobs.iter().map(|b| b.area).collect::<Vec<_>>()));
    r.set("points", json!(blobs.iter().map(|b| { let p = reg.map(b.cx + 0.5, b.cy + 0.5); [(p[0] * 100.0).round() / 100.0, (p[1] * 100.0).round() / 100.0] }).collect::<Vec<_>>()));
    if n != expect {
        r.message = format!("found {}, expected {}", n, expect);
    }
    let st = r.status.clone();
    for (i, b) in blobs.iter().enumerate() {
        r.graphics.push(Graphic::patch_box(&reg, b.x0 as f64, b.y0 as f64, (b.x1 + 1) as f64, (b.y1 + 1) as f64, &st, &format!("{}", i + 1)));
    }
    ctx.blobs.insert(t.id.clone(), (blobs, reg));
    r
}

const COLOURS: &[(&str, [f32; 3])] = &[
    ("red", [255.0, 0.0, 0.0]),
    ("orange", [255.0, 128.0, 0.0]),
    ("yellow", [255.0, 230.0, 0.0]),
    ("green", [0.0, 255.0, 0.0]),
    ("cyan", [0.0, 255.0, 255.0]),
    ("blue", [0.0, 60.0, 255.0]),
    ("purple", [180.0, 0.0, 255.0]),
    ("white", [255.0, 255.0, 255.0]),
];

/// Name a colour by hue (saturated LEDs clip, so hue says more than Lab).
fn name_colour(rgb: [f32; 3]) -> &'static str {
    let (h, s, v) = hsv([rgb[0] as u8, rgb[1] as u8, rgb[2] as u8]);
    if s < 60.0 && v > 80.0 {
        return "white";
    }
    let h = h * 2.0;
    match h {
        x if !(15.0..345.0).contains(&x) => "red",
        x if x < 40.0 => "orange",
        x if x < 70.0 => "yellow",
        x if x < 165.0 => "green",
        x if x < 195.0 => "cyan",
        x if x < 260.0 => "blue",
        _ => "purple",
    }
}

/// Mean colour of a blob, from its unclipped pixels, or a halo round it.
fn blob_colour(b: &Blob, patch: &Img, clip: u8, halo: f64) -> [f32; 3] {
    let w = patch.w;
    let mut acc = [0f64; 3];
    let mut n = 0f64;
    for &i in &b.pixels {
        let p = patch.rgb(i as usize % w, i as usize / w);
        if p[0].max(p[1]).max(p[2]) < clip {
            for k in 0..3 { acc[k] += p[k] as f64; }
            n += 1.0;
        }
    }
    if n < (b.area as f64 * 0.2).max(4.0) {
        // the centre is blown out: sample the glow around it
        let (cx, cy) = (b.cx, b.cy);
        let rw = (b.x1 - b.x0 + 1) as f64 / 2.0;
        let rh = (b.y1 - b.y0 + 1) as f64 / 2.0;
        let (ox, oy) = (rw * halo, rh * halo);
        let x0 = (cx - ox).max(0.0) as usize;
        let x1 = ((cx + ox) as usize).min(patch.w - 1);
        let y0 = (cy - oy).max(0.0) as usize;
        let y1 = ((cy + oy) as usize).min(patch.h - 1);
        acc = [0.0; 3];
        n = 0.0;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let inside = (x as f64 - cx).abs() <= rw && (y as f64 - cy).abs() <= rh;
                if inside { continue; }
                let p = patch.rgb(x, y);
                if p[0].max(p[1]).max(p[2]) < clip {
                    for k in 0..3 { acc[k] += p[k] as f64; }
                    n += 1.0;
                }
            }
        }
    }
    if n == 0.0 {
        return [255.0, 255.0, 255.0];
    }
    [(acc[0] / n) as f32, (acc[1] / n) as f32, (acc[2] / n) as f32]
}

pub fn color_classify(t: &ToolSpec, _frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let src = t.s("source", "");
    let Some((blobs, reg)) = ctx.blobs.get(&src) else { return ToolResult::error(t, format!("reads blobs from '{}', which has not run", src)) };
    let expect = t.strlist("expect");
    let clip = t.i("clip_level", 250).clamp(1, 255) as u8;
    let halo = t.f("halo", 1.8);
    let de_max = t.f("deltaE_max", 28.0) as f32;
    let mut names = vec![];
    let mut des = vec![];
    let mut ok = expect.is_empty() || expect.len() == blobs.len();
    for (i, b) in blobs.iter().enumerate() {
        let c = blob_colour(b, &reg.patch, clip, halo);
        let name = name_colour(c);
        names.push(name.to_string());
        if let Some(e) = expect.get(i) {
            let proto = COLOURS.iter().find(|(n, _)| n == e).map(|x| x.1);
            let de = proto.map(|p| {
                // compare chroma only: brightness depends on exposure
                let (a, b2) = (lab(c), lab(p));
                ((a[1] - b2[1]).powi(2) + (a[2] - b2[2]).powi(2)).sqrt()
            });
            des.push(de.map(|d| (d * 10.0).round() / 10.0));
            let same = name == e.as_str() || de.map(|d| d <= de_max).unwrap_or(false) && name != "white";
            if !same {
                ok = false;
            }
        }
    }
    let mut r = ToolResult::new(t, "pass");
    r.pass_if(ok);
    r.set("colours", json!(names)).set("expected", json!(expect)).set("delta_e", json!(des));
    if !ok {
        r.message = format!("read {}, expected {}", names.join(" "), expect.join(" "));
    }
    let st = r.status.clone();
    for (i, b) in blobs.iter().enumerate() {
        let s = if expect.get(i).map(|e| e == &names[i]).unwrap_or(true) { st.as_str() } else { "fail" };
        r.graphics.push(Graphic::patch_box(reg, b.x0 as f64, b.y0 as f64, (b.x1 + 1) as f64, (b.y1 + 1) as f64, s, &names[i]));
    }
    r
}

pub fn led_glow(t: &ToolSpec, _frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let src = t.s("source", "");
    let Some((blobs, reg)) = ctx.blobs.get(&src) else { return ToolResult::error(t, format!("reads blobs from '{}', which has not run", src)) };
    let v_on = t.f("v_on", 140.0) as f32;
    let s_on = t.f("s_on", 60.0) as f32;
    let expect: Vec<bool> = t.p.get("expect_state").and_then(|v| v.as_array())
        .map(|a| a.iter().map(|x| x.as_bool().unwrap_or(true)).collect()).unwrap_or_default();
    let mut states = vec![];
    for b in blobs {
        let (mut sv, mut ss) = (0f32, 0f32);
        for &i in &b.pixels {
            let (_, s, v) = hsv(reg.patch.rgb(i as usize % reg.patch.w, i as usize / reg.patch.w));
            sv += v;
            ss += s;
        }
        let n = b.pixels.len().max(1) as f32;
        let (v, s) = (sv / n, ss / n);
        // a white or blown-out LED has no saturation to speak of
        states.push(v >= v_on && (s >= s_on || v >= 250.0));
    }
    // a missing LED is Count by brightness's failure; here each found one is judged
    let ok = expect.is_empty() || expect.iter().zip(&states).all(|(e, s)| e == s);
    let mut r = ToolResult::new(t, "pass");
    r.pass_if(ok);
    r.set("states", json!(states)).set("expected", json!(expect));
    if !ok {
        r.message = "lit state differs".into();
    }
    for (i, b) in blobs.iter().enumerate() {
        let s = if expect.get(i).map(|e| *e == states[i]).unwrap_or(true) { "pass" } else { "fail" };
        r.graphics.push(Graphic::patch_box(reg, b.x0 as f64, b.y0 as f64, (b.x1 + 1) as f64, (b.y1 + 1) as f64, s, if states[i] { "on" } else { "off" }));
    }
    r
}

pub fn defect_diff(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(reg) = ctx.region(t, frame) else { return ToolResult::error(t, "no region drawn") };
    let Some(rr) = ctx.ref_region(t) else { return ToolResult::error(t, "no reference picture") };
    let (w, h) = (reg.patch.w, reg.patch.h);
    let cur = Plane::from_gray(&reg.patch);
    let refp = Plane::from_gray(&rr.patch);
    let rad = (t.i("illum_blur", 51).max(3) as usize) / 2;
    let (cb, rb) = (cur.box_blur(rad), refp.box_blur(rad));
    let thr = t.f("threshold", 22.0) as f32;
    let mut mask = vec![0u8; w * h];
    for i in 0..w * h {
        let d = ((cur.d[i] - cb.d[i]) - (refp.d[i] - rb.d[i])).abs();
        if d > thr && reg.inside(i % w, i / w) {
            mask[i] = 255;
        }
    }
    let k = t.i("open_px", 2).max(0) as usize;
    if k > 1 {
        mask = img::open(&mask, w, h, k);
    }
    let amin = t.i("min_area", 25).max(1) as usize;
    let defects: Vec<Blob> = img::components(&mask, w, h, false).into_iter().filter(|b| b.area >= amin).collect();
    let elong = t.f("scratch_elongation", 3.5);
    let max_def = t.i("max_defects", 0);
    let max_area = t.i("max_area_px", 0);
    let largest = defects.iter().map(|b| b.area).max().unwrap_or(0) as i64;
    let mut ok = defects.len() as i64 <= max_def;
    if max_area > 0 && largest > max_area {
        ok = false;
    }
    let mut r = ToolResult::new(t, "pass");
    r.pass_if(ok);
    r.score = 1.0 / (1.0 + defects.len() as f64);
    let kinds: Vec<&str> = defects.iter().map(|b| if b.elongation() >= elong { "scratch" } else { "dent" }).collect();
    r.set("defects", defects.len() as i64).set("largest_px", largest).set("kinds", json!(kinds));
    if !ok {
        r.message = format!("{} defect(s), largest {} px", defects.len(), largest);
    }
    for (b, kd) in defects.iter().zip(&kinds) {
        r.graphics.push(Graphic::patch_box(&reg, b.x0 as f64, b.y0 as f64, (b.x1 + 1) as f64, (b.y1 + 1) as f64, "fail", kd));
    }
    r
}

fn masked_gray(reg: &crate::geom::Region) -> Vec<u8> {
    let g = reg.patch.gray();
    g.data.iter().enumerate().filter(|(i, _)| reg.inside(i % g.w, i / g.w)).map(|(_, &v)| v).collect()
}

pub fn histogram(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(reg) = ctx.region(t, frame) else { return ToolResult::error(t, "no region drawn") };
    let px = masked_gray(&reg);
    let n = px.len().max(1) as f64;
    let mean = px.iter().map(|&v| v as f64).sum::<f64>() / n;
    let std = (px.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
    let (lo, hi) = (t.i("band_low", 0), t.i("band_high", 255));
    let band = 100.0 * px.iter().filter(|&&v| (v as i64) >= lo && (v as i64) <= hi).count() as f64 / n;
    let mut why = vec![];
    let lim = |k: &str| t.f(k, 0.0);
    if lim("mean_min") > 0.0 && mean < lim("mean_min") { why.push(format!("mean {:.1} below {}", mean, lim("mean_min"))); }
    if lim("mean_max") > 0.0 && mean > lim("mean_max") { why.push(format!("mean {:.1} above {}", mean, lim("mean_max"))); }
    if lim("std_min") > 0.0 && std < lim("std_min") { why.push(format!("spread {:.1} below {}", std, lim("std_min"))); }
    if lim("std_max") > 0.0 && std > lim("std_max") { why.push(format!("spread {:.1} above {}", std, lim("std_max"))); }
    if lim("band_pct_min") > 0.0 && band < lim("band_pct_min") { why.push(format!("{:.1}% in band, below {}", band, lim("band_pct_min"))); }
    if lim("band_pct_max") > 0.0 && band > lim("band_pct_max") { why.push(format!("{:.1}% in band, above {}", band, lim("band_pct_max"))); }
    let mut r = ToolResult::new(t, "pass");
    r.pass_if(why.is_empty());
    r.message = why.join("; ");
    r.set("mean", jnum(mean, 2)).set("std", jnum(std, 2)).set("band_pct", jnum(band, 2));
    let g = reg.patch.gray();
    let mn = g.data.iter().min().copied().unwrap_or(0);
    let mx = g.data.iter().max().copied().unwrap_or(0);
    r.set("min", mn as i64).set("max", mx as i64);
    r
}

fn laplacian_var(g: &Img) -> f64 {
    if g.w < 3 || g.h < 3 {
        return 0.0;
    }
    let mut vals = Vec::with_capacity(g.w * g.h);
    for y in 1..g.h - 1 {
        for x in 1..g.w - 1 {
            let c = g.px(x, y, 0) as f64;
            let l = g.px(x - 1, y, 0) as f64 + g.px(x + 1, y, 0) as f64 + g.px(x, y - 1, 0) as f64 + g.px(x, y + 1, 0) as f64 - 4.0 * c;
            vals.push(l);
        }
    }
    let n = vals.len() as f64;
    let m = vals.iter().sum::<f64>() / n;
    vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n
}

pub fn sharpness(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(reg) = ctx.region(t, frame) else { return ToolResult::error(t, "no region drawn") };
    let s = laplacian_var(&reg.patch.gray());
    let mut r = ToolResult::new(t, "pass");
    r.set("sharpness", jnum(s, 1));
    if t.b("relative", true) {
        if let Some(rr) = ctx.ref_region(t) {
            let s0 = laplacian_var(&rr.patch.gray()).max(1e-6);
            let rel = s / s0;
            let need = t.f("min_relative", 0.6);
            r.set("relative", jnum(rel, 3));
            r.pass_if(rel >= need);
            r.score = rel.min(1.0);
            if rel < need { r.message = format!("focus {:.0}% of the reference, needs {:.0}%", rel * 100.0, need * 100.0); }
            return r;
        }
    }
    let need = t.f("min_sharpness", 50.0);
    r.pass_if(s >= need);
    if s < need { r.message = format!("sharpness {:.0}, needs {:.0}", s, need); }
    r
}

pub fn color_segment(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(reg) = ctx.region(t, frame) else { return ToolResult::error(t, "no region drawn") };
    let (h0, h1) = (t.f("hue_min", 0.0) as f32, t.f("hue_max", 179.0) as f32);
    let (smin, vmin) = (t.f("sat_min", 60.0) as f32, t.f("val_min", 40.0) as f32);
    let (w, h) = (reg.patch.w, reg.patch.h);
    let (mut n, mut hit) = (0usize, 0usize);
    let mut mask = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            if !reg.inside(x, y) { continue; }
            n += 1;
            let (hh, s, v) = hsv(reg.patch.rgb(x, y));
            let in_hue = if h0 <= h1 { hh >= h0 && hh <= h1 } else { hh >= h0 || hh <= h1 };
            if in_hue && s >= smin && v >= vmin {
                hit += 1;
                mask[y * w + x] = 255;
            }
        }
    }
    let pct = 100.0 * hit as f64 / n.max(1) as f64;
    let (pmin, pmax) = (t.f("min_pct", 0.0), t.f("max_pct", 0.0));
    let mut ok = true;
    if pmin > 0.0 && pct < pmin { ok = false; }
    if pmax > 0.0 && pct > pmax { ok = false; }
    let mut r = ToolResult::new(t, "pass");
    r.pass_if(ok);
    r.score = pct / 100.0;
    r.set("percent", jnum(pct, 2));
    if !ok { r.message = format!("{:.1}% in the colour range", pct); }
    let st = r.status.clone();
    for b in img::components(&mask, w, h, false).into_iter().filter(|b| b.area >= 30).take(50) {
        r.graphics.push(Graphic::patch_box(&reg, b.x0 as f64, b.y0 as f64, (b.x1 + 1) as f64, (b.y1 + 1) as f64, &st, ""));
    }
    r
}

fn mean_lab(reg: &crate::geom::Region) -> [f32; 3] {
    let (mut acc, mut n) = ([0f64; 3], 0f64);
    for y in 0..reg.patch.h {
        for x in 0..reg.patch.w {
            if !reg.inside(x, y) { continue; }
            let p = reg.patch.rgb(x, y);
            for k in 0..3 { acc[k] += p[k] as f64; }
            n += 1.0;
        }
    }
    let n = n.max(1.0);
    lab([(acc[0] / n) as f32, (acc[1] / n) as f32, (acc[2] / n) as f32])
}

pub fn color_match(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(reg) = ctx.region(t, frame) else { return ToolResult::error(t, "no region drawn") };
    let Some(rr) = ctx.ref_region(t) else { return ToolResult::error(t, "no reference picture") };
    let de = img::delta_e(mean_lab(&reg), mean_lab(&rr)) as f64;
    let lim = t.f("max_delta_e", 10.0);
    let mut r = ToolResult::new(t, "pass");
    r.pass_if(de <= lim);
    r.score = (1.0 - de / (2.0 * lim)).max(0.0);
    r.set("delta_e", jnum(de, 2));
    if de > lim { r.message = format!("colour differs by {:.1} (at most {:.1})", de, lim); }
    r
}
