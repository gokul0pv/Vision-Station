"""Fixtures, calibration, constructed geometry, bead inspection and statistics,
from VisionPro's catalogue.

Fixture from points   the part's pose from found features -- two fiducial holes
                      (CogFixtureNPointToNPointTool)
N-point calibration   pixels to real-world units from known points, rotation and
                      skew included (CogCalibNPointToNPointTool)
Create geometry       a line through two points, a midpoint, a perpendicular or
                      parallel line, a circle through three points
                      (CogCreateLine / CreateCircle / CreateSegment tools)
Bead inspect          a glue or sealant bead along a path: width, position and
                      gaps at stations along it (CogBeadInspectTool)
Data analysis         one value's statistics over runs -- mean, spread, Cp and
                      Cpk against its limits (CogDataAnalysisTool)
"""
import math
import time

import cv2
import numpy as np

from .base import Tool, ToolResult, passed_feature, register, abs_graphic


_abs = abs_graphic


def _ids(v):
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in str(v or "").split(",") if x.strip()]


def _point(f):
    """A feature as a point: a circle's centre, a point, a line's middle."""
    if not f:
        return None
    if f.get("type") == "circle":
        return np.array(f["center"], float)
    if f.get("type") == "point":
        return np.array(f["point"], float)
    if f.get("type") == "line":
        if "p0" in f and "p1" in f:
            return (np.array(f["p0"], float) + np.array(f["p1"], float)) / 2.0
        return np.array(f["point"], float)
    return None


def _line(f):
    """A feature as (point, unit direction), or None."""
    if not f or f.get("type") != "line":
        return None
    d = np.array(f.get("direction") or (np.array(f["p1"], float) - np.array(f["p0"], float)), float)
    return np.array(f.get("point") or f["p0"], float), d / (np.linalg.norm(d) or 1.0)


def _line_feature(p, d, length=200.0):
    d = d / (np.linalg.norm(d) or 1.0)
    return {"type": "line", "point": p.tolist(), "direction": d.tolist(),
            "p0": (p - d * length / 2).tolist(), "p1": (p + d * length / 2).tolist(),
            "angle": math.degrees(math.atan2(d[1], d[0]))}


# ----------------------------------------------------------- fixture from points
_REF_POINTS = {}


@register("fixture_npoint")
class FixtureNPoint(Tool):
    """The part's pose from features other tools found -- two fiducial holes,
    three corners -- instead of matching the whole image. Where those features
    sit on the reference part is found by running the same tools on the
    reference once; every later tool then follows this fixture."""
    is_fixture = True

    def _reference_points(self, ctx, ids):
        key = (id(ctx.reference), tuple(ids), getattr(ctx, "folder", ""))
        if key in _REF_POINTS:
            return _REF_POINTS[key]
        ref_ctx = ctx.__class__(reference=ctx.reference, folder=ctx.folder)
        pts = []
        for tid in ids:
            t = ctx.tools.get(tid)
            if t is None:
                return None
            r = t.run(ctx.reference, ref_ctx)
            if r.status != "pass":
                return None                 # the feature is not found on the reference itself
            pts.append(_point(ref_ctx.share.get(tid)))
        if any(p is None for p in pts):
            return None
        if len(_REF_POINTS) > 64:
            _REF_POINTS.clear()
        _REF_POINTS[key] = pts
        return pts

    def run(self, image, ctx):
        t0 = time.perf_counter()
        ids = _ids(self.p.get("points"))
        if len(ids) < 2:
            return ToolResult(self.id, self.type_name, "error",
                              message="name at least two feature tools in 'points'")
        cur = [_point(passed_feature(ctx, i)) for i in ids]
        missing = [i for i, p in zip(ids, cur) if p is None]
        if missing:
            return ToolResult(self.id, self.type_name, "fail", 0.0, {}, [],
                              "no good feature from %s - it failed or did not run" % ", ".join(missing),
                              (time.perf_counter() - t0) * 1000)
        if ctx.reference is None:
            return ToolResult(self.id, self.type_name, "error", message="needs a reference image")
        ref = self._reference_points(ctx, ids)
        if ref is None:
            return ToolResult(self.id, self.type_name, "error",
                              message="could not find the features on the reference image")
        src, dst = np.float64(ref), np.float64(cur)
        if len(ids) == 2:                      # a similarity from two points, exactly
            vr, vc = src[1] - src[0], dst[1] - dst[0]
            s = np.linalg.norm(vc) / (np.linalg.norm(vr) or 1.0)
            a = math.atan2(vc[1], vc[0]) - math.atan2(vr[1], vr[0])
            R = s * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            M2 = np.hstack([R, (dst[0] - R @ src[0]).reshape(2, 1)])
        else:
            M2, _inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
            if M2 is None:
                return ToolResult(self.id, self.type_name, "fail", message="points do not agree")
        pred = (M2[:, :2] @ src.T).T + M2[:, 2]
        rms = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
        scale = float(math.hypot(M2[0, 0], M2[1, 0]))
        max_rms = float(self.p.get("max_rms", 2.0))
        ok = rms <= max_rms and abs(scale - 1.0) <= float(self.p.get("max_scale_change", 0.05))
        M = np.vstack([M2, [0, 0, 1]])
        if ok:
            ctx.fixture = M
        ang = math.degrees(math.atan2(M2[1, 0], M2[0, 0]))
        st = "pass" if ok else "fail"
        g = [_abs("line", [dst[i], dst[i + 1]], st) for i in range(len(dst) - 1)]
        for p in dst:
            g.append(_abs("rect", [(p[0] - 5, p[1] - 5), (p[0] + 5, p[1] + 5)], st))
        why = "" if ok else ("points disagree by %.2f px" % rms if rms > max_rms
                             else "scale changed by %.1f%%" % (100 * (scale - 1)))
        return ToolResult(self.id, self.type_name, st, max(0.0, 1.0 - rms / (2 * max_rms)),
                          {"shift_x_px": round(float(M2[0, 2]), 2), "shift_y_px": round(float(M2[1, 2]), 2),
                           "angle_deg": round(ang, 3), "scale": round(scale, 5),
                           "rms_px": round(rms, 3),
                           "matrix": [[round(float(v), 6) for v in row] for row in M2] if ok else None},
                          g, why, (time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------- n-point calibration
def _parse_pairs(text):
    """'px,py = wx,wy; ...' -> (pixel points, world points)."""
    px, wd = [], []
    for part in str(text or "").replace("\n", ";").split(";"):
        if "=" not in part:
            continue
        a, b = part.split("=", 1)
        pa = [float(v) for v in a.replace(" ", "").split(",") if v]
        pb = [float(v) for v in b.replace(" ", "").split(",") if v]
        if len(pa) == 2 and len(pb) == 2:
            px.append(pa)
            wd.append(pb)
    return np.float64(px), np.float64(wd)


@register("calibrate_npoint")
class CalibrateNPoint(Tool):
    """Pixels to real-world units from points whose positions are known --
    marks on a calibration plate, holes at drawing positions. Unlike a single
    scale, the fit includes rotation and skew. Later tools report millimetres,
    and each measured point can be converted with world = M @ [x, y, 1]."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        px, wd = _parse_pairs(self.p.get("pairs"))
        if len(px) < 3:
            return ToolResult(self.id, self.type_name, "error",
                              message="needs at least three 'px,py = wx,wy' pairs")
        A = np.hstack([px, np.ones((len(px), 1))])
        sol, *_ = np.linalg.lstsq(A, wd, rcond=None)
        M = sol.T                                   # 2 x 3, pixel -> world
        pred = (A @ sol)
        rms = float(np.sqrt(np.mean(np.sum((pred - wd) ** 2, axis=1))))
        scale = float(math.sqrt(abs(np.linalg.det(M[:, :2]))))
        ctx.mm_per_px = scale
        ctx.world = M
        limit = float(self.p.get("max_rms", 0.05))
        ok = rms <= limit
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail", 1.0 if ok else 0.0,
                          {"units_per_px": round(scale, 6), "rms": round(rms, 5), "points": len(px),
                           "matrix": [[round(float(v), 8) for v in row] for row in M]},
                          [], "" if ok else "the points fit to %.4f, over %.4f" % (rms, limit),
                          (time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------- create geometry
@register("construct")
class Construct(Tool):
    """Geometry made from features other tools found: a line through two
    points, the midpoint of two points, a line perpendicular or parallel to a
    line through a point, a circle through three points. Published like any
    feature, so Distance, Intersect and Create geometry can use it."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        op = self.p.get("operation", "line_through_points")
        F = [passed_feature(ctx, self.p.get(k, "")) for k in ("first", "second", "third")]
        fail = lambda why: ToolResult(self.id, self.type_name, "error", message=why)
        if op in ("line_through_points", "midpoint"):
            a, b = _point(F[0]), _point(F[1])
            if a is None or b is None:
                return fail("needs two point features as first and second")
            if op == "midpoint":
                m = (a + b) / 2.0
                feat = {"type": "point", "point": m.tolist()}
                vals = {"x": round(float(m[0]), 3), "y": round(float(m[1]), 3)}
                g = [_abs("line", [a, b], "search"), _abs("line", [(m[0] - 6, m[1]), (m[0] + 6, m[1])], "pass"),
                     _abs("line", [(m[0], m[1] - 6), (m[0], m[1] + 6)], "pass", self.id)]
            else:
                d = b - a
                if np.linalg.norm(d) < 1e-9:
                    return fail("the two points coincide")
                feat = _line_feature((a + b) / 2.0, d, float(np.linalg.norm(d)))
                vals = {"angle_deg": round(feat["angle"], 3), "length_px": round(float(np.linalg.norm(d)), 3)}
                g = [_abs("line", [a, b], "pass", self.id)]
        elif op in ("perpendicular", "parallel"):
            ln, p = _line(F[0]), _point(F[1])
            if ln is None or p is None:
                return fail("needs a line as first and a point as second")
            d = ln[1] if op == "parallel" else np.array([-ln[1][1], ln[1][0]])
            feat = _line_feature(p, d)
            vals = {"angle_deg": round(feat["angle"], 3)}
            g = [_abs("line", [feat["p0"], feat["p1"]], "pass", self.id)]
        elif op == "circle_through_points":
            pts = [_point(f) for f in F]
            if any(q is None for q in pts):
                return fail("needs three point features")
            (x1, y1), (x2, y2), (x3, y3) = pts
            dd = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            if abs(dd) < 1e-9:
                return fail("the three points are in a line")
            ux = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1)
                  + (x3 ** 2 + y3 ** 2) * (y1 - y2)) / dd
            uy = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3)
                  + (x3 ** 2 + y3 ** 2) * (x2 - x1)) / dd
            r = math.hypot(x1 - ux, y1 - uy)
            feat = {"type": "circle", "center": [ux, uy], "radius": r}
            vals = {"cx": round(ux, 3), "cy": round(uy, 3), "radius_px": round(r, 3)}
            ring = [(ux + r * math.cos(t), uy + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 64)]
            g = [_abs("poly", ring, "pass", self.id)]
        else:
            return fail("unknown operation %s" % op)
        ctx.share[self.id] = feat
        return ToolResult(self.id, self.type_name, "pass", 1.0, vals, g, "",
                          (time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------- bead inspect
@register("bead")
class BeadInspect(Tool):
    """A bead of glue or sealant along a path, checked at stations along it:
    present (no gaps), the right width, where it should be. The path is the
    polygon drawn for the region (its vertices, in order) or, for a rectangle,
    its long centre line. At each station a profile across the path finds the
    bead's two edges."""

    def roi_graphic(self, ctx, status="pass", label=""):
        """The region of a bead is an open path, not a closed polygon: the
        default outline drew a closing line from the path's end to its start."""
        if self.shape != "polygon" or len(self.points) < 2:
            return super().roi_graphic(ctx, status, label)
        path = self._path(ctx)
        out = []
        for i, (q0, q1) in enumerate(zip(path[:-1], path[1:])):
            g = _abs("line", [q0, q1], status, label if i == 0 else "")
            g.role = "region"
            out.append(g)
        return out

    def _path(self, ctx):
        if self.shape == "polygon" and len(self.points) >= 2:
            return np.float64(self.polygon_points(ctx))
        x, y, w, h = [float(v) for v in self.roi]
        pts = np.float64([[x, y + h / 2], [x + w, y + h / 2]]) if w >= h else \
            np.float64([[x + w / 2, y], [x + w / 2, y + h]])
        M = getattr(ctx, "fixture", None)
        if M is not None:
            pts = (np.asarray(M)[:2, :2] @ pts.T).T + np.asarray(M)[:2, 2]
        return pts

    def run(self, image, ctx):
        t0 = time.perf_counter()
        if not self.roi and not self.points:
            return ToolResult(self.id, self.type_name, "error", message="no path")
        g = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image,
                             (3, 3), 0).astype(np.float32)
        path = self._path(ctx)
        seg = np.diff(path, axis=0)
        lens = np.hypot(seg[:, 0], seg[:, 1])
        total = float(lens.sum())
        n = max(4, int(self.p.get("stations", 30)))
        half = float(self.p.get("search_half_width", 25))
        want = float(self.p.get("width", 12))
        tol = float(self.p.get("width_tolerance", 4))
        max_off = float(self.p.get("max_offset", 6))
        minc = float(self.p.get("min_contrast", 15))
        dark = self.p.get("polarity", "dark") == "dark"
        stations, gfx = [], []
        cum = np.concatenate([[0], np.cumsum(lens)])
        for k in range(n):
            s = (k + 0.5) * total / n
            i = min(len(seg) - 1, int(np.searchsorted(cum, s, side="right") - 1))
            t = (s - cum[i]) / (lens[i] or 1.0)
            c = path[i] + seg[i] * t
            d = seg[i] / (lens[i] or 1.0)
            nrm = np.array([-d[1], d[0]])
            m = int(2 * half) + 1
            u = np.linspace(-half, half, m)
            xs = (c[0] + nrm[0] * u).astype(np.float32).reshape(1, -1)
            ys = (c[1] + nrm[1] * u).astype(np.float32).reshape(1, -1)
            prof = cv2.remap(g, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
            if not dark:
                prof = -prof
            grad = np.gradient(prof)
            a, b = int(np.argmin(grad)), int(np.argmax(grad))  # into the bead, out of it
            ok_edges = a < b and -grad[a] >= minc / 2 and grad[b] >= minc / 2
            if ok_edges:
                width = float(u[b] - u[a])
                off = float((u[a] + u[b]) / 2.0)
                bad = abs(width - want) > tol or abs(off) > max_off
                st = {"station": k, "width": round(width, 2), "offset": round(off, 2),
                      "status": "fail" if bad else "pass",
                      "why": ("width %.1f" % width if abs(width - want) > tol else
                              "offset %.1f" % off) if bad else ""}
                e1, e2 = c + nrm * u[a], c + nrm * u[b]
                gfx.append(_abs("line", [e1, e2], st["status"]))
            else:
                st = {"station": k, "width": 0.0, "offset": 0.0, "status": "fail", "why": "gap"}
                gfx.append(_abs("rect", [c - 4, c + 4], "fail"))
            stations.append(st)
        gaps = sum(1 for s in stations if s["why"] == "gap")
        bad = [s for s in stations if s["status"] == "fail"]
        widths = [s["width"] for s in stations if s["why"] != "gap"]
        allowed = int(self.p.get("max_bad_stations", 0))
        ok = len(bad) <= allowed
        # the path is open: drawn as segments -- a polygon closed it back to its start
        for q0, q1 in zip(path[:-1], path[1:]):
            gfx.append(_abs("line", [q0, q1], "search"))
        why = "" if ok else "%d station(s) out: %s" % (len(bad), ", ".join(
            "#%d %s" % (s["station"], s["why"]) for s in bad[:6]))
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail",
                          1.0 - len(bad) / float(n),
                          {"stations": n, "bad_stations": len(bad), "gaps": gaps,
                           "min_width": round(min(widths), 2) if widths else 0.0,
                           "max_width": round(max(widths), 2) if widths else 0.0,
                           "mean_width": round(float(np.mean(widths)), 2) if widths else 0.0,
                           "profile": stations},
                          self.roi_graphic(ctx, "pass" if ok else "fail", self.id) + gfx, why,
                          (time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------- data analysis
_SERIES = {}


def reset_series(folder=None):
    """Forget the running statistics (all, or one application's)."""
    for k in list(_SERIES):
        if folder is None or k[0] == folder:
            _SERIES.pop(k)


@register("data_analysis")
class DataAnalysis(Tool):
    """One value followed across runs -- a hole spacing, a bead width -- with
    its mean, spread, range and process capability (Cp, Cpk) against lower and
    upper limits. Each run fails if the value is outside the limits; with a
    minimum Cpk set, it also fails when the process has drifted towards a limit
    even though every part so far passed."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        ref = self.p.get("source", "")
        if "." not in ref:
            return ToolResult(self.id, self.type_name, "error", message="set source as tool.value")
        tid, key = ref.split(".", 1)
        r = ctx.results.get(tid)
        v = None
        if r is not None:
            v = float(r.score) if key == "score" else (r.values or {}).get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return ToolResult(self.id, self.type_name, "error", message="%s has no value" % ref)
        series = _SERIES.setdefault((getattr(ctx, "folder", ""), self.id), [])
        series.append(float(v))
        window = int(self.p.get("window", 100))
        del series[:-window]
        a = np.array(series)
        lsl = self.p.get("lower_limit")
        usl = self.p.get("upper_limit")
        lsl = float(lsl) if lsl not in (None, "") else None
        usl = float(usl) if usl not in (None, "") else None
        mean, sd = float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
        vals = {"value": round(float(v), 5), "count": len(a), "mean": round(mean, 5),
                "std": round(sd, 5), "min": round(float(a.min()), 5), "max": round(float(a.max()), 5)}
        why = []
        if lsl is not None and v < lsl:
            why.append("%.4g under %.4g" % (v, lsl))
        if usl is not None and v > usl:
            why.append("%.4g over %.4g" % (v, usl))
        if sd > 0 and lsl is not None and usl is not None:
            vals["cp"] = round((usl - lsl) / (6 * sd), 3)
        if sd > 0 and (lsl is not None or usl is not None):
            cpk = min(x for x in ((usl - mean) / (3 * sd) if usl is not None else None,
                                  (mean - lsl) / (3 * sd) if lsl is not None else None)
                      if x is not None)
            vals["cpk"] = round(cpk, 3)
            min_cpk = float(self.p.get("min_cpk", 0) or 0)
            if min_cpk and len(a) >= int(self.p.get("min_samples", 25)) and cpk < min_cpk:
                why.append("Cpk %.2f under %.2f - drifting towards a limit" % (cpk, min_cpk))
        ok = not why
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail", 1.0 if ok else 0.0,
                          vals, [], "; ".join(why), (time.perf_counter() - t0) * 1000)
