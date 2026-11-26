//! Locate part, Part present, Count by shape: all template matching.

use crate::geom::{self, Affine};
use crate::img::Img;
use crate::model::{jnum, Ctx, Graphic, ToolResult, ToolSpec};
use crate::ncc;
use serde_json::json;

/// The reference picture in grey, kept between runs (a 20-megapixel
/// conversion each cycle costs time for nothing).
fn ref_gray(r: &std::sync::Arc<Img>) -> std::sync::Arc<Img> {
    use std::sync::{Arc, Mutex, OnceLock};
    static C: OnceLock<Mutex<Option<(usize, usize, usize, u64, Arc<Img>)>>> = OnceLock::new();
    let c = C.get_or_init(|| Mutex::new(None));
    let key = (Arc::as_ptr(r) as usize, r.w, r.h, r.data.iter().step_by(4099).map(|&v| v as u64).sum::<u64>());
    if let Ok(g) = c.lock() {
        if let Some((p, w, h, s, img)) = g.as_ref() {
            if (*p, *w, *h, *s) == key {
                return img.clone();
            }
        }
    }
    let g = Arc::new(r.gray());
    if let Ok(mut slot) = c.lock() {
        *slot = Some((key.0, key.1, key.2, key.3, g.clone()));
    }
    g
}

/// Locate part, as Vision Station's (tools/align.py, ported step for step
/// in vsalign.rs): the same method, transform, score and messages.
pub fn align(t: &ToolSpec, img: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(refi) = ctx.reference.clone() else { return ToolResult::error(t, "no reference image loaded for this project") };
    if let Some(fx) = ctx.fixture_override {
        let mut r = ToolResult::new(t, "pass");
        r.score = 1.0;
        r.set("method", "given (test)");
        ctx.fixture = Some(fx);
        return r;
    }
    let p = crate::vsalign::Params::from_spec(t);
    let rg = ref_gray(&refi);
    let ig = img.gray();
    let f = crate::vsalign::locate(&p, &rg, &ig);
    let mut ok = f.m.is_some() && f.score >= p.min_score;
    let mut why = if f.m.is_none() && f.method == "features" && f.inliers < p.min_inliers {
        format!("part not found - only {} matching points agree on a position (needs {}). Move or enlarge the region onto a part of the product with more detail.", f.inliers, p.min_inliers)
    } else if ok {
        String::new()
    } else {
        format!("no match above {:.2} (best {:.2})", p.min_score, f.score)
    };
    let min_margin = t.f("min_uniqueness", 0.06);
    if let Some(m) = f.margin {
        if ok && m < min_margin {
            ok = false;
            why = format!("match is not unique - the fixture feature appears more than once (margin {:.3}, needs {:.3}). Enlarge the region or move it onto something that appears only once on the part.", m, min_margin);
        }
    }
    let fx = f.m.map(|m| Affine { a: m[0] as f64, b: m[1] as f64, tx: m[2] as f64, c: m[3] as f64, d: m[4] as f64, ty: m[5] as f64 });
    let mut r = ToolResult::new(t, if ok { "pass" } else { "fail" });
    r.score = f.score;
    r.message = why;
    let (dx, dy, ang) = match fx { Some(a) => (a.tx, a.ty, a.c.atan2(a.a).to_degrees()), None => (0.0, 0.0, 0.0) };
    r.set("shift_x_px", jnum(dx, 2)).set("shift_y_px", jnum(dy, 2)).set("angle_deg", jnum(ang, 3))
        .set("uniqueness", f.margin.map(|m| jnum(m, 3)).unwrap_or(serde_json::Value::Null))
        .set("inliers", if f.method == "features" { json!(f.inliers) } else { serde_json::Value::Null })
        .set("method", f.method);
    if let (true, Some(m)) = (ok, f.m) {
        let rd = |v: f32| jnum(v as f64, 6);
        r.set("matrix", json!([[rd(m[0]), rd(m[1]), rd(m[2])], [rd(m[3]), rd(m[4]), rd(m[5])]]));
    } else {
        r.set("matrix", serde_json::Value::Null);
    }
    if let Some(a) = fx {
        // where the reference picture lies in this one (Vision Station's "fixture" outline)
        let (w, h) = (refi.w as f32, refi.h as f32);
        let m = f.m.unwrap();
        let tr = |x: f32, y: f32| [(m[0] * x + m[1] * y + m[2]) as f64, (m[3] * x + m[4] * y + m[5]) as f64];
        let st = r.status.clone();
        r.graphics.push(Graphic::poly(vec![tr(0.0, 0.0), tr(w, 0.0), tr(w, h), tr(0.0, h)], &st, "fixture"));
        r.set("fixture", a.to_json());
        if let Some(roi) = t.roi() {
            let (cx0, cy0) = (roi[0] + roi[2] / 2.0, roi[1] + roi[3] / 2.0);
            let (cx, cy) = a.apply(cx0, cy0);
            r.set("x", jnum(cx, 2)).set("y", jnum(cy, 2)).set("dx", jnum(cx - cx0, 2)).set("dy", jnum(cy - cy0, 2)).set("angle", jnum(ang, 2));
        }
        if ok {
            ctx.fixture = Some(a);
        }
    }
    r
}

/// Part present, as Vision Station's (tools/count.py Presence) without
/// Train-tab examples: the reference region (grey) looked for in the region
/// widened by search_pad, cut as Tool.crop cuts it; a region with no
/// texture is judged by brightness instead.
pub fn presence(t: &ToolSpec, img: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(roi) = t.roi() else { return ToolResult::error(t, "roi and reference required") };
    let Some(refi) = ctx.reference.clone() else { return ToolResult::error(t, "roi and reference required") };
    let (x, y, w, h) = (roi[0] as i64, roi[1] as i64, roi[2] as i64, roi[3] as i64);
    // count.upright: the region of the reference, grey (a plain slice when not turned)
    let tpl = if t.angle().abs() < 1e-6 {
        let (x0, y0) = (x.max(0), y.max(0));
        let x1 = (x + w).min(refi.w as i64);
        let y1 = (y + h).min(refi.h as i64);
        refi.crop(x0, y0, (x1 - x0).max(0) as usize, (y1 - y0).max(0) as usize).gray()
    } else {
        match ctx.ref_region(t) { Some(r) => r.patch.gray(), None => return ToolResult::error(t, "no reference picture") }
    };
    let pad = t.f("search_pad", 15.0) as i64;
    let big = [(x - pad) as f64, (y - pad) as f64, (w + 2 * pad) as f64, (h + 2 * pad) as f64];
    let fx = ctx.fixture.as_ref().map(crate::geom::f32_rows);
    let tc = crate::cvx::tool_crop(img, big, t.angle(), fx.as_ref());
    let cg = tc.patch.gray();
    let home = if tc.crop_to_img.is_none() {
        let m = crate::cvx::mapped_roi(roi, fx.as_ref());
        ((m[0] - tc.ox) as f64, (m[1] - tc.oy) as f64)
    } else {
        (pad as f64, pad as f64)
    };
    if (cg.h as i64) < h || (cg.w as i64) < w || tpl.w == 0 || tpl.h == 0 {
        return ToolResult::error(t, "search window too small");
    }
    let texture = crate::cvx::np_std_u8(&tpl.data);
    let tp = ncc::plane(&tpl);
    let sp = ncc::plane(&cg);
    let (mx, loc, mode) = if texture < t.f("min_texture", 4.0) {
        // TM_SQDIFF: the best place by squared difference, scored by its RMS
        let (ow, oh) = (sp.w - tp.w + 1, sp.h - tp.h + 1);
        let (mut best, mut bl) = (f64::MAX, (0usize, 0usize));
        for yy in 0..oh {
            for xx in 0..ow {
                let mut s = 0f64;
                for ty in 0..tp.h {
                    for tx in 0..tp.w {
                        let d = (sp.at(xx + tx, yy + ty) - tp.at(tx, ty)) as f64;
                        s += d * d;
                    }
                }
                if s < best {
                    best = s;
                    bl = (xx, yy);
                }
            }
        }
        let rms = (best.max(0.0) / (w * h) as f64).sqrt();
        ((1.0 - rms / 64.0).max(0.0), bl, "brightness")
    } else {
        let m = ncc::match_template(&sp, &tp);
        if m.d.is_empty() {
            return ToolResult::error(t, "search window too small");
        }
        let (bx, by, s) = m.best();
        (s as f64, (bx, by), "pattern")
    };
    let thr = t.f("min_score", 0.65);
    let ok = mx >= thr;
    let shift = ((loc.0 as f64 - home.0).powi(2) + (loc.1 as f64 - home.1).powi(2)).sqrt();
    let mut r = ToolResult::new(t, "pass");
    r.pass_if(ok);
    r.score = mx;
    if !ok {
        r.message = format!("match {:.2} below {}", mx, thr);
    }
    r.set("match", jnum(mx, 4)).set("mode", mode).set("texture", jnum(texture, 1)).set("shift_px", jnum(shift, 2));
    let st = r.status.clone();
    let (lx, ly) = (loc.0 as f64, loc.1 as f64);
    let (ox, oy) = (tc.ox as f64, tc.oy as f64);
    let map = |px: f64, py: f64| -> [f64; 2] {
        match tc.crop_to_img {
            Some(m) => { let d = m[6] * px + m[7] * py + m[8]; [(m[0] * px + m[1] * py + m[2]) / d, (m[3] * px + m[4] * py + m[5]) / d] }
            None => [px + ox, py + oy],
        }
    };
    let quad = vec![map(lx, ly), map(lx + w as f64, ly), map(lx + w as f64, ly + h as f64), map(lx, ly + h as f64)];
    r.graphics.push(Graphic::poly(quad, &st, ""));
    r
}

pub fn count_template(t: &ToolSpec, img: &Img, ctx: &mut Ctx) -> ToolResult {
    let Some(reg) = ctx.region(t, img) else { return ToolResult::error(t, "no region drawn") };
    let Some(trect) = t.rect("template_roi") else { return ToolResult::error(t, "no sample region drawn") };
    let Some(refi) = ctx.reference.clone() else { return ToolResult::error(t, "no reference picture") };
    let tpl = ncc::plane(&geom::extract(&refi, trect, 0.0, None).patch);
    let sp = ncc::plane(&reg.patch);
    let m = ncc::match_template(&sp, &tpl);
    if m.d.is_empty() {
        return ToolResult::error(t, "the sample is larger than the region");
    }
    let thr = t.f("min_score", 0.7) as f32;
    let overlap = t.f("overlap", 0.3);
    let (tw, th) = (tpl.w as f64, tpl.h as f64);
    let mut kept: Vec<(f64, f64, f32)> = vec![];
    for (x, y, s) in ncc::peaks(&m, thr) {
        let (x, y) = (x as f64, y as f64);
        let clash = kept.iter().any(|&(kx, ky, _)| {
            let ix = (tw - (x - kx).abs()).max(0.0);
            let iy = (th - (y - ky).abs()).max(0.0);
            let inter = ix * iy;
            inter / (2.0 * tw * th - inter) > overlap
        });
        if !clash {
            kept.push((x, y, s));
        }
        if kept.len() > 2000 {
            break;
        }
    }
    let n = kept.len() as i64;
    let expect = t.i("expect_count", 4);
    let mut r = ToolResult::new(t, "pass");
    r.pass_if(n == expect);
    r.score = kept.iter().map(|k| k.2 as f64).sum::<f64>() / (kept.len().max(1) as f64);
    r.set("count", n).set("expected", expect);
    r.set("points", json!(kept.iter().map(|k| { let p = reg.map(k.0 + tw / 2.0, k.1 + th / 2.0); [(p[0] * 100.0).round() / 100.0, (p[1] * 100.0).round() / 100.0] }).collect::<Vec<_>>()));
    r.set("scores", json!(kept.iter().map(|k| (k.2 * 1000.0).round() / 1000.0).collect::<Vec<_>>()));
    if n != expect {
        r.message = format!("found {}, expected {}", n, expect);
    }
    let st = r.status.clone();
    for (i, &(x, y, _)) in kept.iter().enumerate() {
        r.graphics.push(Graphic::patch_box(&reg, x, y, x + tw, y + th, &st, &format!("{}", i + 1)));
    }
    r
}
