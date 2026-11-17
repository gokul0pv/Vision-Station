"""Measurement. A caliper projects the ROI along one axis, differentiates the
profile, and fits a parabola to the derivative peak — that parabola fit is what
gives you subpixel repeatability, typically 1/10 pixel on a clean edge. Without
it your measurement quantises to whole pixels and your Gage R&R will fail."""
import time
import cv2
import numpy as np
from .base import Tool, ToolResult, Graphic, register


def _subpixel_peak(prof, i):
    if i <= 0 or i >= len(prof) - 1:
        return float(i)
    a, b, c = float(prof[i - 1]), float(prof[i]), float(prof[i + 1])
    d = a - 2 * b + c
    return float(i) if abs(d) < 1e-9 else float(i) - 0.5 * (c - a) / d


def _edges(crop, axis, polarity, n, min_contrast):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    g = cv2.GaussianBlur(g, (3, 3), 0)
    prof = g.mean(axis=0 if axis == "x" else 1).astype(np.float32)
    d = np.gradient(prof)
    if polarity == "dark_to_light":
        cand = d
    elif polarity == "light_to_dark":
        cand = -d
    else:
        cand = np.abs(d)
    idx = [i for i in range(1, len(cand) - 1)
           if cand[i] >= cand[i - 1] and cand[i] > cand[i + 1] and cand[i] >= min_contrast]
    idx.sort(key=lambda i: -cand[i])
    idx = sorted(idx[:n])
    return [(_subpixel_peak(cand, i), float(cand[i])) for i in idx]


@register("measure_length")
class MeasureLength(Tool):
    """Distance between the first and last edge across the ROI.
    axis: x (horizontal measurement) or y (vertical)."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        crop, (ox, oy) = self.crop(image, ctx)
        axis = self.p.get("axis", "x")
        edges = _edges(crop, axis, self.p.get("polarity", "any"),
                       int(self.p.get("max_edges", 2)),
                       float(self.p.get("min_contrast", 2.0)))
        if getattr(ctx, "want_records", False):
            # drawn from the edges the tool itself found -- a second search
            # with its own settings could disagree with the measurement
            from .plots import profile_plot
            gg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            prof = cv2.GaussianBlur(gg, (3, 3), 0).mean(axis=0 if axis == "x" else 1)
            self.record(ctx, "LastRun.Projection", profile_plot(
                prof, np.gradient(prof.astype(np.float32)),
                [(p, "pass", "") for p, _c in edges],
                [(edges[0][0], edges[-1][0], "pass")] if len(edges) >= 2 else [],
                title="%s - %d edges along %s" % (self.id, len(edges), axis)))
        if len(edges) < 2:
            return ToolResult(self.id, self.type_name, "fail", 0.0,
                              {"edges_found": len(edges)},
                              self.roi_graphic(ctx, "fail"),
                              "fewer than two edges found — check contrast or polarity",
                              (time.perf_counter() - t0) * 1000)

        p0, p1 = edges[0][0], edges[-1][0]
        px = abs(p1 - p0)
        mm_px = float(self.p.get("mm_per_px", 0)) or ctx.mm_per_px
        mm = px * mm_px if mm_px else None

        nom, tol = self.p.get("nominal_mm"), self.p.get("tolerance_mm")
        ok, why = True, ""
        if nom is not None and mm is not None:
            t = float(tol if tol is not None else 0.05)
            ok = abs(mm - float(nom)) <= t
            why = "" if ok else f"{mm:.4f} mm outside {nom} ±{t}"

        r = self.mapped_roi(ctx)
        g = self.roi_graphic(ctx, "pass" if ok else "fail")
        if axis == "x":
            g += [Graphic("line", [(ox + p0, oy), (ox + p0, oy + r[3])], "pass", ""),
                  Graphic("line", [(ox + p1, oy), (ox + p1, oy + r[3])], "pass", ""),
                  Graphic("text", [(ox + (p0 + p1) / 2, oy - 6)], "pass" if ok else "fail",
                          f"{mm:.4f} mm" if mm else f"{px:.2f} px")]
        else:
            g += [Graphic("line", [(ox, oy + p0), (ox + r[2], oy + p0)], "pass", ""),
                  Graphic("line", [(ox, oy + p1), (ox + r[2], oy + p1)], "pass", ""),
                  Graphic("text", [(ox + r[2] / 2, oy + (p0 + p1) / 2)], "pass" if ok else "fail",
                          f"{mm:.4f} mm" if mm else f"{px:.2f} px")]

        vals = {"length_px": round(px, 3), "edges_found": len(edges)}
        if mm is not None:
            vals.update(length_mm=round(mm, 4), nominal_mm=nom, tolerance_mm=tol,
                        deviation_mm=round(mm - float(nom), 4) if nom is not None else None)
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail", 1.0 if ok else 0.0,
                          vals, g, why, (time.perf_counter() - t0) * 1000)


@register("calibrate_scale")
class CalibrateScale(Tool):
    """Set mm-per-pixel for the whole job from one known feature. Put this early
    in the tool order and every downstream measurement inherits real units."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        known = float(self.p.get("known_mm", 0))
        crop, _ = self.crop(image, ctx)
        edges = _edges(crop, self.p.get("axis", "x"), self.p.get("polarity", "any"), 2,
                       float(self.p.get("min_contrast", 2.0)))
        if len(edges) < 2 or known <= 0:
            return ToolResult(self.id, self.type_name, "error",
                              message="need two edges and a positive known_mm")
        px = abs(edges[-1][0] - edges[0][0])
        ctx.mm_per_px = known / px
        return ToolResult(self.id, self.type_name, "pass", 1.0,
                          {"mm_per_px": round(ctx.mm_per_px, 6), "span_px": round(px, 3)},
                          self.roi_graphic(ctx), "", (time.perf_counter() - t0) * 1000)
