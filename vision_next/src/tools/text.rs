//! Read text, Verify text, Read code.

use crate::geom::Region;
use crate::img::Img;
use crate::model::{jnum, Ctx, Graphic, ToolResult, ToolSpec};
use crate::ocr::{self, Line};
use serde_json::json;

fn latin_only(t: &str) -> String {
    t.chars().filter(|&c| (c as u32) < 0x250 || "\u{2103}\u{2126}\u{00b5}\u{2122}".contains(c)).collect::<String>().trim().to_string()
}

fn fit_charset(text: &str, charset: &str) -> String {
    if charset.is_empty() {
        return text.to_string();
    }
    let mut t = text.to_string();
    if !charset.chars().any(|c| c.is_lowercase()) {
        t = t.to_uppercase();
    } else if !charset.chars().any(|c| c.is_uppercase()) {
        t = t.to_lowercase();
    }
    t.chars().filter(|&c| c == ' ' || charset.contains(c)).collect::<String>().trim().to_string()
}

pub fn look_alike(t: &str) -> String {
    t.chars().map(|c| match c {
        'O' | 'o' | 'Q' => '0',
        'I' | 'l' | '|' | 'i' => '1',
        'S' | 's' => '5',
        'B' => '8',
        'Z' | 'z' => '2',
        x => x,
    }).collect()
}

struct Read {
    kept: Vec<Line>,
    unsure: Vec<Line>,
    reg: Region,
}

/// The check's region as Vision Station cuts it (Tool.crop): a plain slice,
/// or cut upright by warpPerspective once the part has turned or moved by a
/// fraction of a pixel.
fn crop_region(t: &ToolSpec, frame: &Img, ctx: &Ctx) -> Option<Region> {
    if t.s("shape", "rect") == "polygon" {
        return ctx.region(t, frame);
    }
    let roi = t.roi()?;
    let fx = ctx.fixture.as_ref().map(crate::geom::f32_rows);
    let tc = crate::cvx::tool_crop(frame, roi, t.angle(), fx.as_ref());
    let to_img = match tc.crop_to_img {
        Some(m) => crate::geom::Affine { a: m[0], b: m[1], tx: m[2], c: m[3], d: m[4], ty: m[5] },
        None => crate::geom::Affine { a: 1.0, b: 0.0, tx: tc.ox as f64, c: 0.0, d: 1.0, ty: tc.oy as f64 },
    };
    Some(Region { patch: tc.patch, to_img, mask: None })
}

fn read_region(t: &ToolSpec, frame: &Img, ctx: &Ctx) -> Result<Read, String> {
    let reg = crop_region(t, frame, ctx).ok_or("no region drawn")?;
    let detail = t.s("detail", "auto");
    let mut lines = ocr::read(&reg.patch, &detail)?;
    let min_conf = t.f("min_confidence", 0.5);
    if lines.is_empty() {
        // a box drawn tightly round one marking leaves the detector no background
        if let Some(one) = ocr::read_single_line(&reg.patch)? {
            if one.conf >= min_conf.max(0.6) {
                lines = vec![one];
            }
        }
    }
    if t.b("ignore_stray_marks", true) {
        lines = ocr::clean_lines(lines);
    }
    let charset = t.s("charset", "");
    let min_chars = t.i("min_chars", 1).max(1) as usize;
    let latin = t.s("script", "latin") != "any";
    let fit = |s: &str| fit_charset(&if latin { latin_only(s) } else { s.to_string() }, &charset);
    let (mut kept, mut unsure) = (vec![], vec![]);
    for mut l in lines {
        let tx = fit(&l.text);
        if tx.chars().count() < min_chars {
            continue;
        }
        if tx != l.text {
            l.char_conf.clear();
        }
        l.text = tx;
        l.core = l.core.as_deref().map(&fit).filter(|s| !s.is_empty());
        if l.conf >= min_conf { kept.push(l) } else { unsure.push(l) }
    }
    Ok(Read { kept, unsure, reg })
}

fn line_graphics(r: &Read, st: &str) -> Vec<Graphic> {
    r.kept.iter().map(|l| Graphic::poly(l.poly.iter().map(|p| r.reg.map(p[0], p[1])).collect(), st, &l.text)).collect()
}

/// text._unsure_note: the most confident of the unsure lines (the first of equals).
fn unsure_note(u: &[Line], min_conf: f64) -> String {
    let mut best: Option<&Line> = None;
    for l in u {
        if best.map(|b| l.conf > b.conf).unwrap_or(true) {
            best = Some(l);
        }
    }
    match best {
        Some(l) => format!("text read with confidence {:.2}, below the minimum {:.2}: '{}'", l.conf, min_conf,
            l.text.chars().take(40).collect::<String>()),
        None => "no text read".into(),
    }
}

/// np.mean of the confidences of the lines kept, else of the unsure ones.
fn mean_conf(rd: &Read) -> f64 {
    let ls = if rd.kept.is_empty() { &rd.unsure } else { &rd.kept };
    if ls.is_empty() {
        return 0.0;
    }
    let v: Vec<f64> = ls.iter().map(|l| l.conf).collect();
    crate::cvx::np_pairwise(&v) / v.len() as f64
}

fn barcode_check(t: &ToolSpec, ctx: &Ctx, read: &[String], r: &mut ToolResult) -> bool {
    if !t.b("compare_barcode", false) {
        return true;
    }
    let Some(code) = &ctx.barcode else {
        r.set("barcode_match", serde_json::Value::Null);
        return true;
    };
    let la = t.b("look_alike", true);
    let ok = read.iter().any(|s| {
        let s2 = s.replace(' ', "");
        if la { look_alike(&s2) == look_alike(code) } else { &s2 == code }
    });
    r.set("barcode_match", ok);
    if !ok {
        r.message = format!("barcode {} not on the part", code);
    }
    ok
}

pub fn ocr_tool(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let rd = match read_region(t, frame, ctx) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(t, e),
    };
    let texts: Vec<String> = rd.kept.iter().map(|l| l.text.clone()).collect();
    let text = texts.join(&t.s("join", " "));
    let conf = mean_conf(&rd);
    let mut r = ToolResult::new(t, "pass");
    let want = t.s("expect_contains", "").trim().to_string();
    let mut ok = !rd.kept.is_empty();
    if !ok {
        r.message = if rd.unsure.is_empty() { "no text found in this region".into() } else { unsure_note(&rd.unsure, t.f("min_confidence", 0.5)) };
    } else if !want.is_empty() {
        // squeezed: no spaces, capitals, look-alikes as one; the whole reading or a line's core
        let la = t.b("look_alike", true);
        let squeeze = |s: &str| { let u: String = s.chars().filter(|c| !c.is_whitespace()).collect::<String>().to_uppercase(); if la { look_alike(&u) } else { u } };
        let need = squeeze(&want);
        let mut hay = vec![text.clone()];
        hay.extend(rd.kept.iter().filter_map(|l| l.core.clone()));
        ok = hay.iter().any(|h| squeeze(h).contains(&need));
        if !ok {
            r.message = format!("'{}' not found; read: {}", want, texts.join(" | ").chars().take(200).collect::<String>());
        }
    }
    ok &= barcode_check(t, ctx, &texts, &mut r);
    r.pass_if(ok);
    r.score = conf;
    r.set("text", text.clone()).set("lines", json!(texts)).set("confidence", jnum(conf, 3))
        .set("engine", format!("pp-ocr ({})", crate::infer::init().unwrap_or_default()));
    let st = r.status.clone();
    r.graphics = line_graphics(&rd, &st);
    r.caption = vec![if text.is_empty() { "no read".into() } else { text.chars().take(28).collect() }];
    r
}

pub fn ocv(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let rd = match read_region(t, frame, ctx) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(t, e),
    };
    let strip = t.b("strip_spaces", true);
    let norm = |s: &str| if strip { s.chars().filter(|c| !c.is_whitespace()).collect() } else { s.to_string() };
    let each: Vec<String> = rd.kept.iter().map(|l| norm(&l.text)).collect();
    let cores: Vec<String> = rd.kept.iter().filter_map(|l| l.core.as_deref().map(norm)).collect();
    let text = each.join(if strip { "" } else { " " });
    let conf = mean_conf(&rd);
    let pattern = t.s("expect_regex", "").trim().to_string();
    let exact = norm(t.s("expect_text", "").trim());
    let mut matched: Option<String> = None;
    let cands: Vec<&String> = std::iter::once(&text).chain(each.iter()).chain(cores.iter()).collect();
    let (ok, why) = if rd.kept.is_empty() {
        (false, if rd.unsure.is_empty() { "no text read".to_string() } else { unsure_note(&rd.unsure, t.f("min_confidence", 0.5)) })
    } else if !exact.is_empty() {
        let la = t.b("look_alike", true);
        matched = cands.iter().find(|c| if la { look_alike(c) == look_alike(&exact) } else { **c == &exact }).map(|s| s.to_string());
        (matched.is_some(), format!("read '{}', expected '{}'", text.chars().take(80).collect::<String>(), exact))
    } else if !pattern.is_empty() {
        match regex::Regex::new(&format!("^(?:{})$", pattern)) {
            Ok(rx) => {
                matched = cands.iter().find(|c| rx.is_match(c)).map(|s| s.to_string());
                (matched.is_some(), format!("read '{}' does not match /{}/", text.chars().take(80).collect::<String>(), pattern))
            }
            Err(e) => return ToolResult::error(t, format!("the expected pattern is not valid: {}", e)),
        }
    } else {
        (true, String::new())
    };
    let mut r = ToolResult::new(t, "pass");
    let mut ok = ok;
    if !ok {
        r.message = why;
    }
    ok &= barcode_check(t, ctx, &each, &mut r);
    r.pass_if(ok);
    r.score = conf;
    let shown = matched.clone().unwrap_or_else(|| text.clone());
    r.set("text", shown.clone()).set("lines", json!(rd.kept.iter().map(|l| l.text.clone()).collect::<Vec<_>>()))
        .set("confidence", jnum(conf, 3)).set("expected", if exact.is_empty() { pattern } else { exact })
        .set("quarter_turn", rd.kept.first().map(|l| l.turn).unwrap_or(0) as i64);
    let st = r.status.clone();
    r.graphics = line_graphics(&rd, &st);
    r.caption = vec![if shown.is_empty() { "no read".into() } else { shown.chars().take(28).collect() }];
    r
}

// -------------------------------------------------------------------- codes
fn decode_all(g: &Img) -> Vec<(String, String)> {
    let mut out = vec![];
    if let Ok(found) = rxing::helpers::detect_multiple_in_luma(g.data.clone(), g.w as u32, g.h as u32) {
        for f in found {
            out.push((f.getText().to_string(), format!("{:?}", f.getBarcodeFormat())));
        }
    }
    out
}

pub fn code_read(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(reg) = ctx.region(t, frame) else { return ToolResult::error(t, "no region drawn") };
    let mut g = reg.patch.gray();
    let long = g.w.max(g.h);
    if long > 2400 {
        let f = 2400.0 / long as f64;
        g = g.resize((g.w as f64 * f) as usize, (g.h as f64 * f) as usize);
    }
    let started = std::time::Instant::now();
    let budget = t.i("timeout_ms", 800).max(50) as u128;
    let mut found = decode_all(&g);
    let mut how = "as taken";
    if found.is_empty() && g.w.max(g.h) < 900 && started.elapsed().as_millis() < budget {
        found = decode_all(&g.resize(g.w * 2, g.h * 2));
        how = "enlarged";
    }
    if found.is_empty() && started.elapsed().as_millis() < budget {
        let inv = Img { w: g.w, h: g.h, c: 1, data: g.data.iter().map(|v| 255 - v).collect() };
        found = decode_all(&inv);
        how = "inverted";
    }
    if found.is_empty() && started.elapsed().as_millis() < budget {
        let thr = crate::img::otsu(&g.data);
        let bin = Img { w: g.w, h: g.h, c: 1, data: g.data.iter().map(|&v| if v > thr { 255 } else { 0 }).collect() };
        found = decode_all(&bin);
        how = "thresholded";
    }
    let mut r = ToolResult::new(t, "pass");
    let texts: Vec<String> = found.iter().map(|f| f.0.clone()).collect();
    let mut ok = !found.is_empty();
    if !ok {
        r.message = "no code found".into();
    }
    let pat = t.s("expect_regex", "");
    if ok && !pat.trim().is_empty() {
        match regex::Regex::new(&format!("^(?:{})$", pat.trim())) {
            Ok(rx) => {
                if !texts.iter().any(|s| rx.is_match(s)) {
                    ok = false;
                    r.message = format!("read '{}' does not match /{}/", texts.join(", "), pat.trim());
                }
            }
            Err(e) => return ToolResult::error(t, format!("the expected pattern is not valid: {}", e)),
        }
    }
    if ok && t.b("compare_barcode", false) {
        if let Some(code) = &ctx.barcode {
            let m = texts.iter().any(|s| s == code);
            r.set("barcode_match", m);
            if !m {
                ok = false;
                r.message = format!("barcode {} not on the part", code);
            }
        }
    }
    r.pass_if(ok);
    r.score = if found.is_empty() { 0.0 } else { 1.0 };
    r.set("text", texts.first().cloned().unwrap_or_default())
        .set("codes", json!(texts))
        .set("symbology", found.first().map(|f| f.1.clone()).unwrap_or_default())
        .set("how", how);
    r.caption = vec![texts.first().cloned().unwrap_or_else(|| "no code".into())];
    r
}
