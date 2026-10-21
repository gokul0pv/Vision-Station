"""Analysis tools, modelled on VisionPro's.

Results Analysis    accept or reject from an expression over earlier results
                    (CogResultsAnalysisTool)
Intersect lines     the corner where two fitted lines meet, and their angle
                    (the corner half of CogFindCornerTool)
Fit line / circle   a line or circle through the centres another tool found
                    (CogFitLineTool, CogFitCircleTool)
Color match         how far a region's colour is from the reference's
                    (CogColorMatchTool)

Expressions are evaluated by a small interpreter that allows numbers,
comparisons, arithmetic, and/or/not, abs/min/max and references to results
written tool.value -- nothing else. Python's eval is never used: a job file is
data, and a crafted one must not be able to run code on the line PC.
"""
import ast
import math
import operator
import time

import cv2
import numpy as np

from .base import Tool, ToolResult, passed_feature, register, abs_graphic


# ----------------------------------------------------------- safe expressions
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
        ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}
_FUN = {"abs": abs, "min": min, "max": max, "round": round}


class ExpressionError(ValueError):
    pass


def references(expr):
    """The tool.value names an expression reads."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionError("not a valid expression: %s" % e.msg)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            out.append("%s.%s" % (node.value.id, node.attr))
    return out


def evaluate(expr, lookup):
    """Evaluate `expr` with lookup(tool_id, key) supplying result values."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionError("not a valid expression: %s" % e.msg)

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, bool, str)):
            return n.value
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            v = lookup(n.value.id, n.attr)
            if v is None:
                raise ExpressionError("%s.%s has no value" % (n.value.id, n.attr))
            return v
        if isinstance(n, ast.BoolOp):
            vals = [bool(ev(v)) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.Not):
                return not ev(n.operand)
            if isinstance(n.op, ast.USub):
                return -ev(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +ev(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for op, right_node in zip(n.ops, n.comparators):
                if type(op) not in _CMP:
                    raise ExpressionError("comparison not allowed")
                right = ev(right_node)
                if not _CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUN \
                and not n.keywords:
            return _FUN[n.func.id](*[ev(a) for a in n.args])
        raise ExpressionError("not allowed in an expression: %s" % type(n).__name__)
    return ev(tree)


def _result_lookup(ctx):
    def lookup(tid, key):
        r = ctx.results.get(tid)
        if r is None:
            return None
        if key == "score":
            return float(r.score)
        if key == "status":
            return r.status
        if key == "passed":
            return r.status == "pass"
        v = (r.values or {}).get(key)
        return v if isinstance(v, (int, float, str, bool)) else None
    return lookup


@register("results_analysis")
class ResultsAnalysis(Tool):
    """Accept or reject from an expression over earlier results, e.g.

        hole_spacing.value_px > 115 and panel_screws.count == 7
        abs(hole_a.radius_px - hole_b.radius_px) < 0.5
        lot_code.passed or backup_code.passed

    Every tool offers .score, .status and .passed as well as its measured
    values."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        expr = (self.p.get("expression") or "").strip()
        if not expr:
            return ToolResult(self.id, self.type_name, "error", message="no expression set")
        try:
            val = evaluate(expr, _result_lookup(ctx))
        except ExpressionError as e:
            return ToolResult(self.id, self.type_name, "error", 0.0, {}, [], str(e),
                              (time.perf_counter() - t0) * 1000)
        except (TypeError, ZeroDivisionError) as e:
            return ToolResult(self.id, self.type_name, "error", 0.0, {}, [],
                              "%s: %s" % (type(e).__name__, e),
                              (time.perf_counter() - t0) * 1000)
        ok = bool(val)
        vals = {"value": val if isinstance(val, (int, float, bool)) else str(val)}
        for ref in references(expr):
            tid, key = ref.split(".", 1)
            vals[ref] = _result_lookup(ctx)(tid, key)
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail",
                          1.0 if ok else 0.0, vals, [],
                          "" if ok else "false: %s" % expr, (time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------- intersect lines
_abs = abs_graphic


@register("intersect")
class Intersect(Tool):
    """The point where two fitted lines meet -- a corner -- and the angle
    between them. Publishes the point as a feature, so Distance can measure
    from it."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        A = passed_feature(ctx, self.p.get("first", ""))
        B = passed_feature(ctx, self.p.get("second", ""))
        kinds = {(A or {}).get("type"), (B or {}).get("type")}
        if not A or not B or not kinds <= {"line", "circle"}:
            return ToolResult(self.id, self.type_name, "error",
                              message="needs two Find line / Find circle results as first and second")
        if "circle" in kinds:
            return self._with_circle(A, B, ctx, t0)
        p1, d1 = np.array(A["point"], float), np.array(A["direction"], float)
        p2, d2 = np.array(B["point"], float), np.array(B["direction"], float)
        den = d1[0] * d2[1] - d1[1] * d2[0]
        angle = math.degrees(math.acos(min(1.0, abs(float(d1 @ d2)))))
        if abs(den) < 1e-9:
            return ToolResult(self.id, self.type_name, "fail", 0.0, {"angle_deg": angle}, [],
                              "the lines are parallel: no corner", (time.perf_counter() - t0) * 1000)
        t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
        x, y = p1 + d1 * t
        ctx.share[self.id] = {"type": "point", "point": [float(x), float(y)]}
        lo = float(self.p.get("min_angle", 0) or 0)
        hi = float(self.p.get("max_angle", 0) or 0)
        why = []
        if lo and angle < lo:
            why.append("angle %.2f under %.2f" % (angle, lo))
        if hi and angle > hi:
            why.append("angle %.2f over %.2f" % (angle, hi))
        ok = not why
        st = "pass" if ok else "fail"
        g = [_abs("line", [(x - 8, y), (x + 8, y)], st),
             _abs("line", [(x, y - 8), (x, y + 8)], st, "%s %.2f deg" % (self.id, angle))]
        return ToolResult(self.id, self.type_name, st, 1.0 if ok else 0.0,
                          {"x": round(float(x), 3), "y": round(float(y), 3),
                           "angle_deg": round(angle, 3)}, g, "; ".join(why),
                          (time.perf_counter() - t0) * 1000)


def _circle_points(A, B):
    """Intersections of a line and a circle, or of two circles -- 0, 1 or 2
    points, as VisionPro's line-circle and circle-circle intersect tools give."""
    if A["type"] == "circle" and B["type"] == "line":
        A, B = B, A
    if A["type"] == "line":
        c, r = np.array(B["center"], float), float(B["radius"])
        p, d = np.array(A["point"], float), np.array(A["direction"], float)
        d = d / (np.linalg.norm(d) or 1.0)
        f = p - c
        bq = float(f @ d)
        disc = bq * bq - (float(f @ f) - r * r)
        if disc < 0:
            return []
        rt = math.sqrt(disc)
        ts = sorted({-bq - rt, -bq + rt})
        return [tuple(p + d * t) for t in ts]
    c1, r1 = np.array(A["center"], float), float(A["radius"])
    c2, r2 = np.array(B["center"], float), float(B["radius"])
    dd = float(np.linalg.norm(c2 - c1))
    if dd == 0 or dd > r1 + r2 or dd < abs(r1 - r2):
        return []
    a = (r1 * r1 - r2 * r2 + dd * dd) / (2 * dd)
    h = math.sqrt(max(0.0, r1 * r1 - a * a))
    m = c1 + a * (c2 - c1) / dd
    off = h * np.array([-(c2 - c1)[1], (c2 - c1)[0]]) / dd
    pts = [tuple(m + off), tuple(m - off)] if h > 1e-9 else [tuple(m)]
    return sorted(pts)


def _intersect_with_circle(self, A, B, ctx, t0):
    pts = _circle_points(A, B)
    need = int(self.p.get("expect_points", 0) or 0)
    ok = bool(pts) and (not need or len(pts) == need)
    st = "pass" if ok else "fail"
    vals = {"points": len(pts)}
    g = []
    for i, (x, y) in enumerate(pts, 1):
        vals["x%d" % i], vals["y%d" % i] = round(float(x), 3), round(float(y), 3)
        g.append(_abs("line", [(x - 8, y), (x + 8, y)], st))
        g.append(_abs("line", [(x, y - 8), (x, y + 8)], st, "%s %d" % (self.id, i)))
    if pts:
        ctx.share[self.id] = {"type": "point", "point": [float(pts[0][0]), float(pts[0][1])],
                              "points": [list(map(float, q)) for q in pts]}
    why = "" if ok else ("they do not meet" if not pts else
                         "%d intersection(s), expected %d" % (len(pts), need))
    return ToolResult(self.id, self.type_name, st, 1.0 if ok else 0.0, vals, g, why,
                      (time.perf_counter() - t0) * 1000)


Intersect._with_circle = _intersect_with_circle


# ----------------------------------------------------------- fit through points
def _centres(ctx, source):
    """Centres from an earlier tool: blobs (Count by brightness) or template
    matches (Count by shape)."""
    blobs = ctx.share.get(source)
    if isinstance(blobs, list):
        return np.array([[b["cx"], b["cy"]] if "cx" in b else
                         [b["x"] + b.get("w", 0) / 2.0, b["y"] + b.get("h", 0) / 2.0]
                         for b in blobs], float)
    r = ctx.results.get(source)
    matches = ((r.values or {}).get("matches") if r is not None else None) or []
    return np.array([[m["x"] + m.get("w", 0) / 2.0, m["y"] + m.get("h", 0) / 2.0]
                     for m in matches if isinstance(m, dict) and "x" in m], float)


@register("fit_shape")
class FitShape(Tool):
    """A line or circle through the centres another tool found: are the
    screws in a straight row, do the holes lie on their bolt circle. Reports
    the worst deviation of any centre from the fitted shape."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        pts = _centres(ctx, self.p.get("source", ""))
        need = {"circle": 3, "ellipse": 6}.get(self.p.get("fit", "line"), 2)
        if len(pts) < need:
            return ToolResult(self.id, self.type_name, "fail", 0.0, {"points": int(len(pts))},
                              [], "%d centres from %s, need %d" % (
                                  len(pts), self.p.get("source") or "(no source)", need),
                              (time.perf_counter() - t0) * 1000)
        from .geometry import _fit_circle, _fit_line
        mm = float(getattr(ctx, "mm_per_px", 0.0) or 0.0)
        k = mm if mm > 0 else 1.0
        g = []
        if self.p.get("fit", "line") == "ellipse":
            from .more_tools import _fit_ellipse_robust
            (cx, cy, ea, eb, eang), resid, _k = _fit_ellipse_robust(pts, 0)
            ctx.share[self.id] = {"type": "circle", "center": [float(cx), float(cy)],
                                  "radius": float((ea + eb) / 2.0)}
            vals = {"cx": round(float(cx), 3), "cy": round(float(cy), 3),
                    "major": round(2 * max(ea, eb) * k, 4), "minor": round(2 * min(ea, eb) * k, 4),
                    "angle_deg": round(float(eang), 2)}
            tt = math.radians(eang)
            ring = [(cx + ea * math.cos(u) * math.cos(tt) - eb * math.sin(u) * math.sin(tt),
                     cy + ea * math.cos(u) * math.sin(tt) + eb * math.sin(u) * math.cos(tt))
                    for u in np.linspace(0, 2 * math.pi, 64)]
            shape_g = ("poly", ring)
        elif self.p.get("fit", "line") == "circle":
            (cx, cy), r, resid = _fit_circle(pts)
            ctx.share[self.id] = {"type": "circle", "center": [float(cx), float(cy)],
                                  "radius": float(r)}
            vals = {"cx": round(float(cx), 3), "cy": round(float(cy), 3),
                    "radius": round(float(r) * k, 4)}
            ring = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 64)]
            shape_g = ("poly", ring)
        else:
            c, d, resid = _fit_line(pts)
            t = (pts - c) @ d
            p0, p1 = c + d * t.min(), c + d * t.max()
            ctx.share[self.id] = {"type": "line", "p0": p0.tolist(), "p1": p1.tolist(),
                                  "point": c.tolist(), "direction": d.tolist(),
                                  "angle": math.degrees(math.atan2(d[1], d[0]))}
            vals = {"angle_deg": round(math.degrees(math.atan2(d[1], d[0])), 3)}
            shape_g = ("line", [p0, p1])
        worst = float(np.abs(resid).max()) * k
        vals.update(points=int(len(pts)), max_deviation=round(worst, 4),
                    unit="mm" if mm > 0 else "px")
        limit = float(self.p.get("max_deviation", 0) or 0)
        ok = not (limit and worst > limit)
        st = "pass" if ok else "fail"
        g.append(_abs(shape_g[0], shape_g[1], st, "%s dev %.2f" % (self.id, worst)))
        for (x, y), e in zip(pts, resid):
            g.append(_abs("rect", [(x - 3, y - 3), (x + 3, y + 3)],
                          "fail" if limit and abs(e) * k > limit else "pass"))
        return ToolResult(self.id, self.type_name, st, 1.0 if ok else 0.0, vals, g,
                          "" if ok else "a centre is %.3f %s off the %s" % (
                              worst, vals["unit"], self.p.get("fit", "line")),
                          (time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------- colour match
def _mean_lab(patch, mask=None):
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if mask is not None and mask.shape[:2] == patch.shape[:2]:
        lab = lab[mask.reshape(-1) > 0]
    # OpenCV's 8-bit Lab scales L to 0..255 and offsets a, b by 128
    L, a, b = np.median(lab, axis=0)
    return np.array([L * 100.0 / 255.0, a - 128.0, b - 128.0])


@register("color_match")
class ColorMatch(Tool):
    """How far the region's colour is from the same region in the reference,
    as CIE delta E (about 2 is just noticeable, over 10 clearly different).
    The median colour, so a few specular glints or a small mark do not move
    it. Checks a label is the right one, a cable the right colour, a coating
    present."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        if ctx.reference is None:
            return ToolResult(self.id, self.type_name, "error", message="needs a reference image")
        cur, _off = self.crop(image, ctx)
        mask_cur = getattr(self, "_crop_mask", None)
        saved_fixture, ctx.fixture = ctx.fixture, None        # reference is unmoved
        try:
            ref, _o = self.crop(ctx.reference, ctx)
            mask_ref = getattr(self, "_crop_mask", None)
        finally:
            ctx.fixture = saved_fixture
        if cur is None or cur.size == 0 or ref is None or ref.size == 0:
            return ToolResult(self.id, self.type_name, "error", message="empty region")
        a, b = _mean_lab(cur, mask_cur), _mean_lab(ref, mask_ref)
        de = float(np.linalg.norm(a - b))
        limit = float(self.p.get("max_delta_e", 10.0))
        ok = de <= limit
        vals = {"delta_e": round(de, 2), "L": round(float(a[0]), 1), "a": round(float(a[1]), 1),
                "b": round(float(a[2]), 1), "ref_L": round(float(b[0]), 1),
                "ref_a": round(float(b[1]), 1), "ref_b": round(float(b[2]), 1)}
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail",
                          max(0.0, 1.0 - de / (2 * limit)), vals,
                          self.roi_graphic(ctx, "pass" if ok else "fail",
                                           "%s dE %.1f" % (self.id, de)),
                          "" if ok else "colour differs by dE %.1f (limit %.1f)" % (de, limit),
                          (time.perf_counter() - t0) * 1000)
