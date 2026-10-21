"""Geometry and analysis tools, modelled on VisionPro's.

Find Line and Find Circle work the way VisionPro's do: many small calipers are
laid across the expected edge, each finds one edge point to subpixel precision,
and a line or circle is fitted through them. Outliers -- a caliper that landed on
a scratch, a screw head, a burr -- are discarded before the final fit, the
equivalent of VisionPro's NumToIgnore. Fitting many points is what makes these
tools more accurate and more robust than any single edge measurement.

Results are published in true image coordinates under the tool's id, so an
analysis tool can reference them: the distance between two circle centres, from
a point to a line, the angle between two lines, a circle's diameter. That is how
VisionPro's distance and angle tools work -- they measure between features found
by other tools, not directly on pixels.

The Histogram tool is VisionPro's standard presence/absence check: grey-level
statistics over a region, judged against limits. It is blunt and very robust --
a missing part changes a region's brightness whatever its texture.
"""
import math
import time

import cv2
import numpy as np

from .base import Tool, ToolResult, register, passed_feature, abs_graphic
from .measure import _subpixel_peak


# ----------------------------------------------------------- helpers
def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _plane(img, channel="auto"):
    """The image plane to find edges in.

    Greyscale conversion throws colour contrast away: on the demo board a green
    LED is 142 grey against a 138 background and an orange one 137 -- invisible
    to an edge tool -- while every individual colour channel shows 60 to 90
    levels of contrast. 'auto' picks whichever of grey, blue, green or red
    varies most across the region, which is where the edges are strongest."""
    if img.ndim == 2:
        return img
    planes = {"grey": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
              "blue": img[:, :, 0], "green": img[:, :, 1], "red": img[:, :, 2]}
    if channel in planes:
        return planes[channel]
    return max(planes.values(), key=lambda p: float(p.std()))


def _profile_edge(prof, polarity, min_contrast, step=1.0):
    """Strongest edge in a 1-D profile, subpixel. Returns (position, strength)
    or None. polarity is relative to increasing position along the profile.

    `step` is the spacing of the samples in pixels. The gradient is divided by
    it so strength is always grey levels per pixel: Find Circle samples its rays
    every half pixel, and without this the same edge read half as strong there
    as in Find Line, so the two tools' contrast settings meant different things."""
    d = np.gradient(cv2.GaussianBlur(prof.reshape(1, -1), (1, 5), 0).ravel()) / float(step)
    if polarity == "dark_to_light":
        cand = d
    elif polarity == "light_to_dark":
        cand = -d
    else:
        cand = np.abs(d)
    if cand.size < 3:
        return None
    i = int(np.argmax(cand[1:-1])) + 1
    if cand[i] < min_contrast:
        return None
    return _subpixel_peak(cand, i), float(cand[i])


def _fit_line(pts):
    """Total least squares: minimises perpendicular distance, so it is not
    biased by the line's angle the way y-on-x regression is."""
    c = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - c)
    d = vt[0]                                   # direction
    n = np.array([-d[1], d[0]])                 # normal
    resid = (pts - c) @ n
    return c, d, resid


def _fit_circle(pts):
    """Algebraic (Kasa) circle fit, then one Gauss-Newton refinement on the
    geometric error, which removes the algebraic fit's bias on short arcs."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(1e-9, sol[2] + cx * cx + cy * cy))
    for _ in range(5):
        dx, dy = x - cx, y - cy
        di = np.sqrt(dx * dx + dy * dy) + 1e-12
        J = np.column_stack([-dx / di, -dy / di, -np.ones_like(di)])
        res = di - r
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
    resid = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
    return (cx, cy), r, resid


def _robust(fit, pts, ignore):
    """Fit, drop the worst point, refit -- `ignore` times. Dropping one at a
    time matters: a gross outlier drags the first fit, so the point that looks
    worst after the first fit may not be the real culprit."""
    keep = np.arange(len(pts))
    out = fit(pts)
    for _ in range(int(ignore)):
        if len(keep) <= 3:
            break
        worst = int(np.argmax(np.abs(out[-1])))
        keep = np.delete(keep, worst)
        out = fit(pts[keep])
    return out, keep


class _ToImage:
    """Crop coordinates -> image coordinates, for rotated or plain regions."""

    def __init__(self, tool, ox, oy):
        self.M = tool._crop_M
        self.ox, self.oy = ox, oy

    def __call__(self, pts):
        pts = np.float32(pts).reshape(-1, 1, 2)
        if self.M is None:
            return pts.reshape(-1, 2) + np.float32([self.ox, self.oy])
        return cv2.perspectiveTransform(pts, self.M).reshape(-1, 2)


def _raw_plane(tool, image, ctx):
    """(smoothed image plane, crop offset) for tools that search radially --
    Find circle and Find ellipse. The raw region, not the ellipse-masked patch:
    the mask would paint an artificial edge exactly where the edge is expected.
    (None, (0, 0)) for an empty region -- callers unpack the offset."""
    shape, tool.shape = tool.shape, "rect"
    try:
        crop, off = tool.crop(image, ctx)
    finally:
        tool.shape = shape
    if crop is None or crop.size == 0:
        return None, (0, 0)
    return cv2.GaussianBlur(_plane(crop, tool.p.get("channel", "auto")).astype(np.float32),
                            (3, 3), 0), off


def _too_few_points(tool, ctx, found, need, t0, guides=()):
    """Find Line and Find Circle fail the same way when too few calipers see
    the edge -- one definition, so the message cannot drift between them. The
    search guides come too: a failed tool is when you most need to see where
    it looked."""
    return ToolResult(tool.id, tool.type_name, "fail", 0.0, {"found": found},
                      tool.roi_graphic(ctx, "fail") + list(guides),
                      "edge found by %d calipers, need %d" % (found, need),
                      (time.perf_counter() - t0) * 1000)


_abs_graphic = abs_graphic


# ----------------------------------------------------------- find line
@register("find_line")
class FindLine(Tool):
    """Fits a line to an edge running along the region's length.

    Calipers are strips across the region's width, each searching along the
    region's height for the strongest edge. Draw the region with its long axis
    along the edge; rotate it for an edge at an angle."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        crop, (ox, oy) = self.crop(image, ctx)
        if crop is None or crop.size == 0:
            return ToolResult(self.id, self.type_name, "error", message="empty region")
        g = _plane(crop, self.p.get("channel", "auto")).astype(np.float32)
        H, W = g.shape
        n = max(3, int(self.p.get("num_calipers", 12)))
        width = max(1, int(self.p.get("caliper_width", 6)))
        pol = self.p.get("polarity", "any")
        minc = float(self.p.get("min_contrast", 8.0))
        pts = []
        for i in range(n):
            xc = (i + 0.5) * W / float(n)
            x0 = int(max(0, xc - width / 2.0))
            x1 = int(min(W, x0 + width))
            e = _profile_edge(g[:, x0:x1].mean(axis=1), pol, minc)
            if e is not None:
                pts.append((xc, e[0]))
        to_img = _ToImage(self, ox, oy)
        guides = [_abs_graphic("line", to_img([((i + 0.5) * W / float(n), 0),
                                               ((i + 0.5) * W / float(n), H - 1)]), "search")
                  for i in range(n)]                 # where each caliper searched
        need = max(3, int(self.p.get("min_found", 6)))
        if len(pts) < need:
            return _too_few_points(self, ctx, len(pts), need, t0, guides)

        P = to_img(np.float32(pts))
        (c, d, resid), keep = _robust(_fit_line, P, self.p.get("num_to_ignore", 2))
        # endpoints: project the kept points onto the fitted direction
        t = (P[keep] - c) @ d
        p0, p1 = c + d * t.min(), c + d * t.max()
        if p1[0] < p0[0]:
            p0, p1 = p1, p0
        angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        rms = float(np.sqrt(np.mean(resid ** 2)))
        max_rms = float(self.p.get("max_rms", 1.5))
        ok = rms <= max_rms
        ctx.share[self.id] = {"type": "line", "p0": p0.tolist(), "p1": p1.tolist(),
                              "point": c.tolist(), "direction": d.tolist(),
                              "angle": angle}
        gr = self.roi_graphic(ctx, "pass" if ok else "fail") + guides
        gr.append(_abs_graphic("line", [p0, p1], "pass" if ok else "fail",
                               "%s %.2f deg" % (self.id, angle)))
        for i, q in enumerate(P):
            gr.append(_abs_graphic("rect", [(q[0] - 1.5, q[1] - 1.5), (q[0] + 1.5, q[1] + 1.5)],
                                   "pass" if i in keep else "warn"))
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail",
                          max(0.0, 1.0 - rms / (2 * max_rms)),
                          {"angle_deg": round(angle, 3), "rms_px": round(rms, 3),
                           "found": len(pts), "used": int(len(keep)),
                           "x0": round(float(p0[0]), 2), "y0": round(float(p0[1]), 2),
                           "x1": round(float(p1[0]), 2), "y1": round(float(p1[1]), 2)},
                          gr, "" if ok else "fit error %.2f px over %.2f" % (rms, max_rms),
                          (time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------- find circle
@register("find_circle")
class FindCircle(Tool):
    """Fits a circle to an edge inside the region.

    The expected circle is the one inscribed in the region; calipers are rays
    from its centre, each searching outward between the inner and outer search
    radii for the strongest edge. Draw an ellipse region just around the hole or
    boss."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        g, (ox, oy) = _raw_plane(self, image, ctx)
        if g is None:
            return ToolResult(self.id, self.type_name, "error", message="empty region")
        H, W = g.shape
        cx0, cy0 = (W - 1) / 2.0, (H - 1) / 2.0
        R = min(W, H) / 2.0
        r_in = R * float(self.p.get("search_inner", 0.45))
        r_out = R * float(self.p.get("search_outer", 1.0)) - 1.0
        n = max(6, int(self.p.get("num_calipers", 24)))
        pol = self.p.get("polarity", "any")
        minc = float(self.p.get("min_contrast", 8.0))
        steps = max(8, int((r_out - r_in) * 2))
        rs = np.linspace(r_in, r_out, steps)
        pts = []
        for k in range(n):
            a = 2 * math.pi * k / n
            xs = (cx0 + rs * math.cos(a)).astype(np.float32)
            ys = (cy0 + rs * math.sin(a)).astype(np.float32)
            prof = cv2.remap(g, xs.reshape(1, -1), ys.reshape(1, -1),
                             cv2.INTER_LINEAR).ravel()
            e = _profile_edge(prof, pol, minc, step=(r_out - r_in) / float(steps - 1))
            if e is None:
                continue
            rr = r_in + (r_out - r_in) * e[0] / float(steps - 1)
            pts.append((cx0 + rr * math.cos(a), cy0 + rr * math.sin(a)))
        to_img = _ToImage(self, ox, oy)
        # where it searched: the inner and outer search radii
        guides = []
        for rr in (r_in, r_out):
            ring = [(cx0 + rr * math.cos(a), cy0 + rr * math.sin(a))
                    for a in np.linspace(0, 2 * math.pi, 48)]
            guides.append(_abs_graphic("poly", to_img(ring), "search"))
        need = max(4, int(self.p.get("min_found", 10)))
        if len(pts) < need:
            return _too_few_points(self, ctx, len(pts), need, t0, guides)
        P = to_img(np.float32(pts))
        ((cx, cy), r, resid), keep = _robust(_fit_circle, P, self.p.get("num_to_ignore", 3))
        rms = float(np.sqrt(np.mean(resid ** 2)))
        rmin = float(self.p.get("min_radius", 0) or 0)
        rmax = float(self.p.get("max_radius", 0) or 0)
        max_rms = float(self.p.get("max_rms", 1.5))
        why = []
        if rms > max_rms:
            why.append("fit error %.2f px over %.2f" % (rms, max_rms))
        if rmin and r < rmin:
            why.append("radius %.2f under %.2f" % (r, rmin))
        if rmax and r > rmax:
            why.append("radius %.2f over %.2f" % (r, rmax))
        ok = not why
        ctx.share[self.id] = {"type": "circle", "center": [float(cx), float(cy)],
                              "radius": float(r)}
        gr = self.roi_graphic(ctx, "pass" if ok else "fail") + guides
        circ = [(cx + r * math.cos(a), cy + r * math.sin(a))
                for a in np.linspace(0, 2 * math.pi, 49)]
        gr.append(_abs_graphic("poly", circ, "pass" if ok else "fail",
                               "%s r=%.2f" % (self.id, r)))
        gr.append(_abs_graphic("line", [(cx - 4, cy), (cx + 4, cy)], "pass"))
        gr.append(_abs_graphic("line", [(cx, cy - 4), (cx, cy + 4)], "pass"))
        mm = float(getattr(ctx, "mm_per_px", 0.0) or 0.0)
        vals = {"cx": round(float(cx), 3), "cy": round(float(cy), 3),
                "radius_px": round(float(r), 3), "rms_px": round(rms, 3),
                "found": len(pts), "used": int(len(keep))}
        if mm > 0:
            vals["diameter_mm"] = round(2 * r * mm, 4)
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail",
                          max(0.0, 1.0 - rms / (2 * max_rms)), vals, gr,
                          "; ".join(why), (time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------- histogram
@register("histogram")
class Histogram(Tool):
    """Grey-level statistics over the region, judged against limits.

    Statistics use only the pixels inside the region's shape. The tool base
    fills the outside of an ellipse or polygon with the inside mean so other
    tools see no false edge -- harmless for them, but counting those filled
    pixels here would drag the standard deviation towards zero."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        crop, (ox, oy) = self.crop(image, ctx)
        if crop is None or crop.size == 0:
            return ToolResult(self.id, self.type_name, "error", message="empty region")
        g = _gray(crop)
        mask = getattr(self, "_crop_mask", None)
        px = g[mask > 0] if mask is not None and mask.shape == g.shape else g.ravel()
        if px.size == 0:
            return ToolResult(self.id, self.type_name, "error", message="no pixels in region")
        px = px.astype(np.float32)
        lo_g, hi_g = int(self.p.get("band_low", 0)), int(self.p.get("band_high", 255))
        vals = {"mean": round(float(px.mean()), 2), "std": round(float(px.std()), 2),
                "min": int(px.min()), "max": int(px.max()),
                "median": round(float(np.median(px)), 1),
                "band_pct": round(100.0 * float(((px >= lo_g) & (px <= hi_g)).mean()), 2),
                "pixels": int(px.size)}
        why = []

        def limit(key, lo_key, hi_key):
            lo = self.p.get(lo_key)
            hi = self.p.get(hi_key)
            v = vals[key]
            if lo is not None and float(lo) > 0 and v < float(lo):
                why.append("%s %.1f under %.1f" % (key, v, float(lo)))
            if hi is not None and float(hi) > 0 and v > float(hi):
                why.append("%s %.1f over %.1f" % (key, v, float(hi)))

        limit("mean", "mean_min", "mean_max")
        limit("std", "std_min", "std_max")
        limit("band_pct", "band_pct_min", "band_pct_max")
        ok = not why
        if getattr(ctx, "want_records", False):
            from .plots import histogram_plot
            self.record(ctx, "LastRun.Histogram", histogram_plot(
                px, (lo_g, hi_g), vals["mean"], "pass" if ok else "fail",
                title="%s - %d pixels, mean %.1f, spread %.1f" % (
                    self.id, vals["pixels"], vals["mean"], vals["std"])))
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail",
                          1.0 if ok else 0.0, vals,
                          self.roi_graphic(ctx, "pass" if ok else "fail",
                                           "%s %.0f" % (self.id, vals["mean"])),
                          "; ".join(why), (time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------- distance / angle
def _point_of(f):
    if f["type"] == "circle":
        return np.array(f["center"])
    if f["type"] == "line":
        return (np.array(f["p0"]) + np.array(f["p1"])) / 2.0
    return np.array(f.get("point", [0, 0]))


def _point_line(pt, line):
    c, d = np.array(line["point"]), np.array(line["direction"])
    n = np.array([-d[1], d[0]])
    return abs(float((pt - c) @ n))


@register("distance")
class Distance(Tool):
    """Measures between features found by other tools, as VisionPro's distance
    and angle tools do. Name the tools in 'first' and 'second':

        circle, circle   centre-to-centre distance
        circle, line     perpendicular distance from the centre to the line
        line,   line     angle between them (and the distance between them if
                         they are near parallel)
        circle alone     diameter (leave 'second' empty)

    The features are in image coordinates, fixture already applied by the tools
    that found them, so the measurement is independent of where the part sits.
    """

    def run(self, image, ctx):
        t0 = time.perf_counter()
        a_id, b_id = self.p.get("first", ""), self.p.get("second", "")
        A = passed_feature(ctx, a_id)
        B = passed_feature(ctx, b_id) if b_id else None
        missing = [i for i, f in ((a_id, A), (b_id, B)) if i and f is None]
        if not a_id or missing:
            return ToolResult(self.id, self.type_name, "error", 0.0, {}, [],
                              "no result from %s - it must run earlier and succeed"
                              % (", ".join(missing) or "'first'"),
                              (time.perf_counter() - t0) * 1000)
        mm = float(getattr(ctx, "mm_per_px", 0.0) or 0.0)
        vals, gr = {}, []
        unit = "px"
        if B is None:
            if A["type"] != "circle":
                return ToolResult(self.id, self.type_name, "error",
                                  message="a single feature must be a circle")
            value = 2 * A["radius"]
            what = "diameter"
        elif A["type"] == "line" and B["type"] == "line":
            da, db = np.array(A["direction"]), np.array(B["direction"])
            cosang = abs(float(da @ db)) / (np.linalg.norm(da) * np.linalg.norm(db))
            value = math.degrees(math.acos(min(1.0, cosang)))
            what, unit = "angle", "deg"
            vals["separation_px"] = round(_point_line(_point_of(B), A), 3)
        elif "line" in (A["type"], B["type"]):
            line, other = (A, B) if A["type"] == "line" else (B, A)
            p = _point_of(other)
            value = _point_line(p, line)
            what = "point_to_line"
            c, d = np.array(line["point"]), np.array(line["direction"])
            foot = c + d * float((p - c) @ d)
            gr.append(_abs_graphic("line", [tuple(p), tuple(foot)], "pass",
                                   "%.2f" % value))
        else:
            pa, pb = _point_of(A), _point_of(B)
            value = float(np.linalg.norm(pa - pb))
            what = "point_to_point"
            gr.append(_abs_graphic("line", [tuple(pa), tuple(pb)], "pass", "%.2f" % value))

        shown = value
        if unit == "px" and mm > 0:
            shown, unit = value * mm, "mm"
        vals.update({"measure": what, "value": round(shown, 4), "unit": unit,
                     "value_px": round(value, 3)})
        lo = float(self.p.get("min_value", 0) or 0)
        hi = float(self.p.get("max_value", 0) or 0)
        why = []
        if lo and shown < lo:
            why.append("%s %.3f %s under %.3f" % (what, shown, unit, lo))
        if hi and shown > hi:
            why.append("%s %.3f %s over %.3f" % (what, shown, unit, hi))
        ok = not why
        for g in gr:
            g.status = "pass" if ok else "fail"
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail",
                          1.0 if ok else 0.0, vals, gr, "; ".join(why),
                          (time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------- caliper pairs
def _signed_edges(prof, min_contrast):
    """Every local gradient extremum above min_contrast, subpixel, with its sign:
    +1 for dark to light, -1 for light to dark."""
    d = np.gradient(cv2.GaussianBlur(prof.reshape(1, -1), (1, 5), 0).ravel())
    out = []
    for i in range(1, len(d) - 1):
        a = abs(d[i])
        if a >= min_contrast and a >= abs(d[i - 1]) and a > abs(d[i + 1]):
            sign = 1 if d[i] > 0 else -1
            pos = _subpixel_peak(np.abs(d), i)
            out.append((pos, sign, float(a)))
    return out


@register("caliper")
class Caliper(Tool):
    """Edge pairs, as VisionPro's Caliper measures them.

    A pair is a bar or a gap: an edge, then the next edge of opposite polarity.
    'dark' pairs are dark bars between lighter surroundings (pins in backlight,
    a dark trace); 'light' pairs the reverse. With an expected width the tool
    prefers pairs close to it, which is what lets it pick out one feature among
    clutter. It reports every pair found, left to right, with each width and the
    pitch between their centres -- so it counts pins and checks their spacing as
    well as measuring them.

    The measurement runs along the region's length, averaging across its
    height, so draw the region across the features with its long side along the
    measuring direction."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        crop, (ox, oy) = self.crop(image, ctx)
        if crop is None or crop.size == 0:
            return ToolResult(self.id, self.type_name, "error", message="empty region")
        g = _plane(crop, self.p.get("channel", "auto")).astype(np.float32)
        prof = g.mean(axis=0)                       # along the region's length
        edges = _signed_edges(prof, float(self.p.get("min_contrast", 8.0)))
        want = self.p.get("pair_polarity", "dark")
        first = -1 if want == "dark" else 1         # a dark bar starts light-to-dark
        exp_w = float(self.p.get("expected_width", 0) or 0)
        tol_w = float(self.p.get("width_tolerance", 0) or 0)
        pairs = []
        for i, (p0, s0, c0) in enumerate(edges):
            if s0 != first:
                continue
            # the next edge of opposite polarity closes the pair
            for p1, s1, c1 in edges[i + 1:]:
                if s1 == -s0:
                    w = p1 - p0
                    if exp_w and tol_w and abs(w - exp_w) > tol_w:
                        break
                    pairs.append({"start": p0, "end": p1, "width": w,
                                  "centre": (p0 + p1) / 2.0, "contrast": min(c0, c1)})
                    break
                if s1 == s0:
                    break                           # a same-polarity edge first: not a clean pair
        n_max = int(self.p.get("max_results", 20))
        pairs = sorted(pairs, key=lambda q: -q["contrast"])[:n_max]
        pairs.sort(key=lambda q: q["centre"])
        mm = float(getattr(ctx, "mm_per_px", 0.0) or 0.0)
        if getattr(ctx, "want_records", False):
            from .plots import profile_plot
            d = np.gradient(cv2.GaussianBlur(prof.reshape(1, -1), (1, 5), 0).ravel())
            marks = [(p0, "pass" if any(abs(p0 - q["start"]) < 1e-6 or abs(p0 - q["end"]) < 1e-6
                                        for q in pairs) else "fail", "")
                     for p0, _s, _c in edges]
            self.record(ctx, "LastRun.Projection", profile_plot(
                prof, d, marks, [(q["start"], q["end"], "pass") for q in pairs],
                title="%s - %d edges, %d pairs (green: in a pair)" % (self.id, len(edges),
                                                                    len(pairs))))
        k = mm if mm > 0 else 1.0
        widths = [q["width"] * k for q in pairs]
        centres = [q["centre"] for q in pairs]
        pitch = float(np.mean(np.diff(centres))) * k if len(centres) > 1 else 0.0
        unit = "mm" if mm > 0 else "px"

        why = []
        n_exp = int(self.p.get("expect_count", 0) or 0)
        if n_exp and len(pairs) != n_exp:
            why.append("found %d pairs, expected %d" % (len(pairs), n_exp))
        if not pairs:
            why.append("no edge pair found")
        nom = float(self.p.get("nominal_width", 0) or 0)
        tol = float(self.p.get("tolerance", 0) or 0)
        if nom and tol:
            bad = [i for i, w in enumerate(widths) if abs(w - nom) > tol]
            if bad:
                why.append("%d width%s out of %.3f +- %.3f %s" % (
                    len(bad), "s" if len(bad) > 1 else "", nom, tol, unit))
        ok = not why

        H = g.shape[0]
        to_img = _ToImage(self, ox, oy)
        gr = self.roi_graphic(ctx, "pass" if ok else "fail")
        Wc = g.shape[1]
        mid = H / 2.0
        a, b = to_img([(4, mid), (Wc - 5, mid)])
        head = to_img([(Wc - 5 - min(12, Wc / 6.0), mid - min(6, H / 3.0)), (Wc - 5, mid),
                       (Wc - 5 - min(12, Wc / 6.0), mid + min(6, H / 3.0))])
        gr.append(_abs_graphic("line", [a, b], "search"))       # the measuring direction
        gr.append(_abs_graphic("poly", head, "search"))
        for q in pairs:
            a = to_img([(q["start"], 0), (q["start"], H - 1)])
            b = to_img([(q["end"], 0), (q["end"], H - 1)])
            gr.append(_abs_graphic("line", a, "pass" if ok else "fail"))
            gr.append(_abs_graphic("line", b, "pass" if ok else "fail",
                                   "%.2f" % (q["width"] * k)))
        vals = {"count": len(pairs), "unit": unit,
                "widths": [round(w, 4) for w in widths],
                "pitch": round(pitch, 4)}
        if widths:
            vals["mean_width"] = round(float(np.mean(widths)), 4)
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail",
                          1.0 if ok else 0.0, vals, gr, "; ".join(why),
                          (time.perf_counter() - t0) * 1000)
