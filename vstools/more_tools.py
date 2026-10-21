"""More tools from VisionPro's catalogue.

Find ellipse        an ellipse fitted to an edge (CogFindEllipseTool)
Image sharpness     is the image in focus (CogImageSharpnessTool)
Color segmenter     how much of the region is a given colour (CogColorSegmenterTool)
Classify            a trained ONNX classifier sorts the region into classes
                    (VisionPro Deep Learning's Green Classify)
"""
import math
import time

import cv2
import numpy as np

from .base import Graphic, Tool, ToolResult, register, runtime
from .geometry import _abs_graphic, _profile_edge, _raw_plane, _ToImage, _too_few_points


# ----------------------------------------------------------- find ellipse
def _fit_ellipse_robust(pts, ignore):
    """cv2.fitEllipse, then drop the point farthest from the ellipse and refit,
    `ignore` times -- one gross outlier (a burr, a chip) drags a least-squares
    ellipse badly."""
    keep = np.arange(len(pts))
    for _ in range(int(ignore) + 1):
        (cx, cy), (w, h), ang = cv2.fitEllipse(pts[keep].astype(np.float32))
        a, b = w / 2.0, h / 2.0
        t = math.radians(ang)
        c, s = math.cos(t), math.sin(t)
        d = pts[keep] - [cx, cy]
        u = d[:, 0] * c + d[:, 1] * s
        v = -d[:, 0] * s + d[:, 1] * c
        # algebraic distance scaled to pixels: how far each point is off the ellipse
        r = np.sqrt((u / max(a, 1e-6)) ** 2 + (v / max(b, 1e-6)) ** 2)
        resid = (r - 1.0) * min(a, b)
        if len(keep) <= 6 or _ == int(ignore):
            return (cx, cy, a, b, ang), resid, keep
        keep = np.delete(keep, int(np.argmax(np.abs(resid))))


@register("find_ellipse")
class FindEllipse(Tool):
    """Rays from the region's centre find the edge; an ellipse is fitted
    through the edge points, the worst few discarded. For a hole seen at an
    angle, or a feature that is genuinely oval."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        g, (ox, oy) = _raw_plane(self, image, ctx)
        if g is None:
            return ToolResult(self.id, self.type_name, "error", message="empty region")
        H, W = g.shape
        cx0, cy0 = (W - 1) / 2.0, (H - 1) / 2.0
        ax, ay = W / 2.0, H / 2.0                    # the region's own semi-axes
        f_in = float(self.p.get("search_inner", 0.45))
        f_out = float(self.p.get("search_outer", 1.0))
        n = max(8, int(self.p.get("num_calipers", 36)))
        pol = self.p.get("polarity", "any")
        minc = float(self.p.get("min_contrast", 8.0))
        to_img = _ToImage(self, ox, oy)
        guides = []
        for f in (f_in, f_out):
            ring = [(cx0 + (ax * f - (1 if f == f_out else 0)) * math.cos(a),
                     cy0 + (ay * f - (1 if f == f_out else 0)) * math.sin(a))
                    for a in np.linspace(0, 2 * math.pi, 48)]
            guides.append(_abs_graphic("poly", to_img(ring), "search"))
        steps = 48
        pts = []
        for k in range(n):
            a = 2 * math.pi * k / n
            fs = np.linspace(f_in, f_out, steps)
            xs = (cx0 + ax * fs * math.cos(a)).astype(np.float32)
            ys = (cy0 + ay * fs * math.sin(a)).astype(np.float32)
            xs = np.clip(xs, 0, W - 1)
            ys = np.clip(ys, 0, H - 1)
            prof = cv2.remap(g, xs.reshape(1, -1), ys.reshape(1, -1), cv2.INTER_LINEAR).ravel()
            step = math.hypot(ax * math.cos(a), ay * math.sin(a)) * (f_out - f_in) / (steps - 1)
            e = _profile_edge(prof, pol, minc, step=max(step, 1e-6))
            if e is None:
                continue
            f = f_in + (f_out - f_in) * e[0] / float(steps - 1)
            pts.append((cx0 + ax * f * math.cos(a), cy0 + ay * f * math.sin(a)))
        need = max(6, int(self.p.get("min_found", 12)))
        if len(pts) < need:
            return _too_few_points(self, ctx, len(pts), need, t0, guides)
        P = to_img(np.float32(pts))
        (cx, cy, a, b, ang), resid, keep = _fit_ellipse_robust(
            np.asarray(P, np.float64), self.p.get("num_to_ignore", 4))
        rms = float(np.sqrt(np.mean(resid ** 2)))
        max_rms = float(self.p.get("max_rms", 1.5))
        ok = rms <= max_rms
        st = "pass" if ok else "fail"
        # published as a circle-like feature, so Distance can measure to its centre
        ctx.share[self.id] = {"type": "circle", "center": [float(cx), float(cy)],
                              "radius": float((a + b) / 2.0)}
        t = math.radians(ang)
        ring = [(cx + a * math.cos(u) * math.cos(t) - b * math.sin(u) * math.sin(t),
                 cy + a * math.cos(u) * math.sin(t) + b * math.sin(u) * math.cos(t))
                for u in np.linspace(0, 2 * math.pi, 64)]
        gr = self.roi_graphic(ctx, st) + guides
        gr.append(_abs_graphic("poly", ring, st, "%s %.1f x %.1f" % (self.id, 2 * a, 2 * b)))
        gr.append(_abs_graphic("line", [(cx - 4, cy), (cx + 4, cy)], st))
        gr.append(_abs_graphic("line", [(cx, cy - 4), (cx, cy + 4)], st))
        mm = float(getattr(ctx, "mm_per_px", 0.0) or 0.0)
        vals = {"cx": round(float(cx), 3), "cy": round(float(cy), 3),
                "major_px": round(2 * max(a, b), 3), "minor_px": round(2 * min(a, b), 3),
                "angle_deg": round(float(ang), 2), "rms_px": round(rms, 3),
                "found": len(pts), "used": int(len(keep))}
        if mm > 0:
            vals.update(major_mm=round(2 * max(a, b) * mm, 4), minor_mm=round(2 * min(a, b) * mm, 4))
        return ToolResult(self.id, self.type_name, st, max(0.0, 1.0 - rms / (2 * max_rms)),
                          vals, gr, "" if ok else "fit error %.2f px over %.2f" % (rms, max_rms),
                          (time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------- image sharpness
@register("sharpness")
class Sharpness(Tool):
    """Is the image in focus? A camera drifting out of focus -- a knocked lens,
    a loosened mount, a part at the wrong height -- makes every other tool fail
    in confusing ways; this names the cause. The score is the variance of the
    Laplacian over the region: sharp edges score high, blur scores low. Set the
    minimum from a known-good image (the result reports the score)."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        crop, _off = self.crop(image, ctx)
        if crop is None or crop.size == 0:
            return ToolResult(self.id, self.type_name, "error", message="empty region")
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        lap = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F, ksize=3)
        score = float(lap.var())
        ref_score = None
        if ctx.reference is not None and self.p.get("relative", True):
            saved, ctx.fixture = ctx.fixture, None
            try:
                rc, _o = self.crop(ctx.reference, ctx)
            finally:
                ctx.fixture = saved
            if rc is not None and rc.size:
                rg = cv2.cvtColor(rc, cv2.COLOR_BGR2GRAY) if rc.ndim == 3 else rc
                ref_score = float(cv2.Laplacian(rg.astype(np.float32), cv2.CV_32F, ksize=3).var())
        vals = {"sharpness": round(score, 2)}
        if ref_score:
            rel = score / ref_score
            vals.update(reference=round(ref_score, 2), relative=round(rel, 3))
            limit = float(self.p.get("min_relative", 0.6))
            ok = rel >= limit
            why = "" if ok else "focus %.0f%% of the reference (limit %.0f%%)" % (100 * rel, 100 * limit)
        else:
            limit = float(self.p.get("min_sharpness", 50.0))
            ok = score >= limit
            why = "" if ok else "sharpness %.1f under %.1f" % (score, limit)
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail",
                          min(1.0, vals.get("relative", score / max(limit, 1e-6))), vals,
                          self.roi_graphic(ctx, "pass" if ok else "fail",
                                           "%s %.0f" % (self.id, score)), why,
                          (time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------- colour segmenter
@register("color_segment")
class ColorSegment(Tool):
    """The share of the region within a colour range (hue, saturation,
    brightness): glue coverage, a coloured label present, rust. Hue wraps
    round, so a red range can straddle 0 (e.g. 170 to 10)."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        crop, (ox, oy) = self.crop(image, ctx)
        if crop is None or crop.size == 0 or crop.ndim != 3:
            return ToolResult(self.id, self.type_name, "error", message="needs a colour region")
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h0, h1 = int(self.p.get("hue_min", 0)), int(self.p.get("hue_max", 179))
        s0, v0 = int(self.p.get("sat_min", 60)), int(self.p.get("val_min", 40))
        if h0 <= h1:
            m = cv2.inRange(hsv, (h0, s0, v0), (h1, 255, 255))
        else:                                        # the range wraps past red
            m = cv2.inRange(hsv, (h0, s0, v0), (179, 255, 255)) | \
                cv2.inRange(hsv, (0, s0, v0), (h1, 255, 255))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        region = getattr(self, "_crop_mask", None)
        if region is not None and region.shape == m.shape:
            m &= region
            total = int((region > 0).sum())
        else:
            total = m.size
        area = int((m > 0).sum())
        pct = 100.0 * area / max(1, total)
        lo = float(self.p.get("min_pct", 0) or 0)
        hi = float(self.p.get("max_pct", 0) or 0)
        why = []
        if lo and pct < lo:
            why.append("%.1f%% of the region is that colour, under %.1f%%" % (pct, lo))
        if hi and pct > hi:
            why.append("%.1f%% of the region is that colour, over %.1f%%" % (pct, hi))
        ok = not why
        st = "pass" if ok else "fail"
        self.record(ctx, "LastRun.ColorMask", cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
        g = self.roi_graphic(ctx, st, "%s %.1f%%" % (self.id, pct))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        to_img = _ToImage(self, ox, oy)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:12]:
            if cv2.contourArea(c) < 20:
                continue
            gg = Graphic("poly", [tuple(map(float, p)) for p in to_img(c.reshape(-1, 2).astype(np.float32))],
                         st, "")
            gg.absolute = True
            g.append(gg)
        return ToolResult(self.id, self.type_name, st, 1.0 if ok else 0.0,
                          {"percent": round(pct, 2), "pixels": area, "blobs": len(cnts)}, g,
                          "; ".join(why), (time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------- classify
@register("classify")
class Classify(Tool):
    """A trained image classifier (ONNX -- e.g. exported from the Colab
    notebook or any framework) sorts the region into classes: good / scratch /
    dent, or which variant of a part is present. Passes when the top class is
    one of the accepted classes and confident enough."""

    def run(self, image, ctx):
        import json
        import os
        t0 = time.perf_counter()
        path = self.p.get("model") or ""
        if path and not os.path.isabs(path):
            path = os.path.join(getattr(ctx, "folder", "") or "", path)
        if not path or not os.path.exists(path):
            return ToolResult(self.id, self.type_name, "error",
                              message="model file not found: %s" % (self.p.get("model") or "(none)"))
        crop, _off = self.crop(image, ctx)
        if crop is None or crop.size == 0:
            return ToolResult(self.id, self.type_name, "error", message="empty region")
        try:
            model = runtime().load(path, self.p.get("backend", "auto"),
                                   self.p.get("precision", "accuracy"))
        except ImportError:
            return ToolResult(self.id, self.type_name, "error",
                              message="no inference runtime installed - pip install openvino")
        shape = model.input_shape() or [1, 3, 224, 224]
        hh = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else 224
        ww = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else 224
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.ndim == 3 else \
            cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        x = cv2.resize(rgb, (ww, hh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if self.p.get("normalize", "imagenet") == "imagenet":
            x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32)
        logits = np.asarray(model(x)[0], np.float64).ravel()
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        names = self.p.get("classes") or []
        if isinstance(names, str):
            names = [c.strip() for c in names.split(",") if c.strip()]
        if not names:
            meta = model.metadata()
            try:
                names = json.loads(meta.get("classes", "[]")) if meta.get("classes") else []
            except ValueError:
                names = []
        names = list(names) + ["class_%d" % i for i in range(len(names), len(probs))]
        top = int(np.argmax(probs))
        label, conf = names[top], float(probs[top])
        accept = self.p.get("accept") or ""
        accept = [a.strip() for a in (accept.split(",") if isinstance(accept, str) else accept)
                  if a.strip()] or [names[0]]
        min_conf = float(self.p.get("min_confidence", 0.6))
        ok = label in accept and conf >= min_conf
        why = "" if ok else ("classified as %s (%.0f%%)" % (label, 100 * conf) if label not in accept
                             else "%s only %.0f%% confident (need %.0f%%)" % (
                                 label, 100 * conf, 100 * min_conf))
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail", conf,
                          {"class": label, "confidence": round(conf, 4),
                           "scores": {n: round(float(p), 4) for n, p in zip(names, probs)}},
                          self.roi_graphic(ctx, "pass" if ok else "fail",
                                           "%s %s %.0f%%" % (self.id, label, 100 * conf)),
                          why, (time.perf_counter() - t0) * 1000)
