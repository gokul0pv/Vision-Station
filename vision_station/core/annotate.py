"""How results are written on the picture -- on screen, in saved result
pictures and in exported packages on a line PC, all the same:

    every check     its outline, in the shape it was drawn (a box, a rotated
                    box, a circle or ellipse, a polygon), where it found the part
    OK / NG         blue for OK, red for NG, orange for a warning (OK / NG
                    colours and line thickness: Settings > Results, set_look)
    a caption       next to the check, in words: OK or NG, the text read, the
                    text expected, the size measured in mm, the count found...

The look is set here (STYLE) rather than in ui/design.py because a package
running on a line PC draws result pictures without the user interface.
"""
import math

import cv2
import numpy as np

# ---------------------------------------------------------------- the look
STYLE = {
    "ok": "#1F6FFF",          # a check that passed: blue
    "ng": "#FF2A2A",          # a check that failed: red
    "warn": "#FF9F0A",        # a warning or an error in the check itself
    "search": "#38BDF8",      # where a tool searched (off by default in pictures)
    "line_screen_px": 3.0,    # outline width on screen
    "line_picture": 1 / 520,  # outline width in a saved picture, as a share of its long side
}

STATUS_KEY = {"pass": "ok", "fail": "ng", "warn": "warn", "error": "warn", "search": "search"}
WORD = {"pass": "OK", "fail": "NG", "warn": "CHECK", "error": "ERROR"}

# Settings > Results may change the OK / NG colours and the line thickness.
# The factory look is kept here so it can always be restored.
DEFAULT_OK, DEFAULT_NG = STYLE["ok"], STYLE["ng"]
DEFAULT_LINE_SCREEN_PX, DEFAULT_LINE_PICTURE = STYLE["line_screen_px"], STYLE["line_picture"]
LINE_SCALES = (0.5, 1.0, 1.5, 2.0, 3.0)      # thin .. heaviest, 1.0 = factory
_listeners = []


def solid_hex(c):
    """'#RRGGBB' upper case for a solid colour (#RGB, RRGGBB accepted), else
    None. Anything with transparency (#AARRGGBB) is refused."""
    if not isinstance(c, str):
        return None
    h = c.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        return None
    try:
        int(h, 16)
    except ValueError:
        return None
    return "#" + h.upper()


def line_scale(v):
    """A thickness setting, clamped to LINE_SCALES' range; 1.0 when unreadable."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 1.0
    if v != v:                                    # NaN
        return 1.0
    return min(max(v, LINE_SCALES[0]), LINE_SCALES[-1])


def set_look(ok=None, ng=None, scale=None):
    """Apply the user's OK / NG colours and line thickness everywhere results
    are drawn. Missing or bad values fall back to the factory look; OK and NG
    may never be the same colour (NG then falls back so the two still
    differ). Returns True when anything changed."""
    ok = solid_hex(ok) or DEFAULT_OK
    ng = solid_hex(ng) or DEFAULT_NG
    if ng == ok:
        ng = DEFAULT_NG if ok != DEFAULT_NG else DEFAULT_OK
    s = line_scale(1.0 if scale is None else scale)
    new = {"ok": ok, "ng": ng, "line_screen_px": DEFAULT_LINE_SCREEN_PX * s,
           "line_picture": DEFAULT_LINE_PICTURE * s}
    if all(STYLE[k] == v for k, v in new.items()):
        return False
    STYLE.update(new)
    for fn in list(_listeners):
        try:
            fn()
        except Exception as e:                    # a closed window must not stop the rest
            print("[annotate] look listener failed: %s" % e)
    return True


def on_look_changed(fn):
    """fn() is called after set_look changes the colours or thickness."""
    if fn not in _listeners:
        _listeners.append(fn)


def colour_hex(status):
    return STYLE[STATUS_KEY.get(status, "ok")]


def colour_bgr(status):
    h = colour_hex(status).lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def ink_for(bgr):
    """Black or white text, whichever reads on a fill of colour bgr."""
    return (0, 0, 0) if (0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]) > 160 \
        else (255, 255, 255)


# ---------------------------------------------------------------- captions
def _num(v, digits=2):
    return ("%.*f" % (digits, v)).rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def _size(values, mm_key, px_key, prefix=""):
    if values.get(mm_key) is not None:
        return "%s%s mm" % (prefix, _num(float(values[mm_key]), 3))
    if values.get(px_key) is not None:
        return "%s%s px" % (prefix, _num(float(values[px_key]), 1))
    return ""


def _text_of(values):
    lines = values.get("lines") or values.get("texts")
    if isinstance(lines, list) and lines:
        return " / ".join(str(x) for x in lines if str(x).strip())
    return str(values.get("text") or values.get("data") or "")


def caption(result, params=None):
    """A check's result as one or two short lines for the picture.

    result: a ToolResult or its dict; params: the check's settings (for the
    text it expects). Returns a list of strings, first line first."""
    get = (lambda k, d=None: result.get(k, d)) if isinstance(result, dict) else \
        (lambda k, d=None: getattr(result, k, d))
    ty, status = get("type_name") or get("type") or "", get("status", "pass")
    v = get("values") or {}
    p = params or {}
    word = WORD.get(status, status.upper())
    ok = status == "pass"

    if status == "error":
        msg = (get("message") or "").split(";")[0]
        return ["ERROR", msg[:48]] if msg else ["ERROR"]

    if ty in ("ocv",):
        read = _text_of(v) or "nothing read"
        if ok:
            # what matched -- not every fragment the box also held
            return [str(v.get("text") or read)]
        want = p.get("expect_text") or (("pattern " + p["expect_regex"]) if p.get("expect_regex")
                                        else v.get("expected") or "")
        out = ["NG  read: %s" % read if _text_of(v) else "NG  nothing read"]
        if want:
            out.append("expected: %s" % want)
        return out
    if ty in ("ocr", "ocr_auto", "text_detect"):
        read = _text_of(v) or "nothing read"
        if ok:
            return [read]
        want = p.get("expect_contains") or ""
        first = "NG  read: %s" % read if _text_of(v) else "NG  nothing read"
        return [first] + (["expected: %s" % want] if want else [])
    if ty == "code_read":
        data = _text_of(v)
        return [data] if ok and data else [word] + ([data] if data else [])
    if ty == "measure_length":
        s = _size(v, "length_mm", "length_px")
        if v.get("deviation_mm") is not None and not ok:
            s += "  (%+.3f mm)" % float(v["deviation_mm"])
        return ["%s  %s" % (word, s)] if s else [word]
    if ty == "distance":
        if v.get("value") is not None:
            return ["%s  %s %s" % (word, _num(float(v["value"]), 3), v.get("unit", ""))]
        return [word]
    if ty == "find_circle":
        s = _size(v, "diameter_mm", None) if v.get("diameter_mm") is not None else \
            ("Ø %s px" % _num(2 * float(v["radius_px"]), 1) if v.get("radius_px") is not None else "")
        if s and not s.startswith("Ø"):
            s = "Ø " + s
        return ["%s  %s" % (word, s)] if s else [word]
    if ty == "find_ellipse" and v.get("major_px") is not None:
        return ["%s  %s x %s px" % (word, _num(float(v["major_px"]), 1), _num(float(v["minor_px"]), 1))]
    if ty == "caliper":
        if v.get("count") is not None:
            unit = v.get("unit", "px")
            s = "%s  %d found" % (word, int(v["count"]))
            if v.get("pitch"):
                s += ", pitch %s %s" % (_num(float(v["pitch"]), 3), unit)
            return [s]
        return [word]
    if ty in ("count_blob", "count_template", "yolo_detect", "yolo_defect"):
        n = v.get("count")
        if n is None:
            return [word]
        exp = v.get("expected")
        return ["%s  %s%s" % (word, n, (" / %s" % exp) if exp not in (None, "") else "")]
    if ty == "led_glow" and v.get("lit_count") is not None:
        return ["%s  %s lit" % (word, v["lit_count"])]
    if ty == "color_match" and v.get("delta_e") is not None:
        return ["%s  ΔE %s" % (word, _num(float(v["delta_e"]), 1))]
    if ty == "color_segment" and v.get("percent") is not None:
        return ["%s  %s%%" % (word, _num(float(v["percent"]), 1))]
    if ty == "classify" and v.get("class"):
        return ["%s  %s" % (word, v["class"])]
    if ty in ("defect_diff",) and not ok and v.get("defect_count"):
        return ["NG  %d defect%s" % (v["defect_count"], "" if v["defect_count"] == 1 else "s")]
    if ty == "calibrate_scale" and v.get("mm_per_px"):
        return ["%s mm/px" % _num(float(v["mm_per_px"]), 4)]
    if ty == "align":
        return ["located" if ok else "NG  part not found"]
    if ty == "color_classify" and not ok:
        msg = (get("message") or "").split(";")[0]
        return ["NG"] + ([msg[:48]] if msg else [])
    return [word]


# ---------------------------------------------------------------- outlines
def outline(result):
    """The graphic that outlines a check: its region (in its shape), or None."""
    for g in result.get("graphics", []):
        if g.get("role") == "region":
            return g
    return None


def _bbox(points):
    a = np.float32(points)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def anchor(result, where="top"):
    """Where a check's caption goes: the top-left (or bottom-left) of its
    outline, else of everything it drew. (x, y) or None."""
    g = outline(result)
    pts = g["points"] if g else [q for x in result.get("graphics", [])
                                 if x.get("status") != "search" for q in x.get("points", [])]
    if not pts:
        return None
    x0, y0, _x1, y1 = _bbox(pts)
    return (x0, y1) if where == "bottom" else (x0, y0)


# ---------------------------------------------------------------- saved pictures
def tag_height(box_h, picture_long):
    """One line of a result tag in image pixels: a third of its box, kept
    between 1/220 and 1/60 of the picture's long side -- the same rule the
    screen uses, so a saved picture reads like the result page."""
    return max(picture_long / 220.0, min(picture_long / 60.0, box_h * 0.32))


def _tag_on_box(out, lines, box, col, picture_long, taken=None):
    """The result in words, attached to its box: a tag in the result colour
    sitting on the box's top-left corner (inside the box when there is no
    room above it)."""
    x0, y0, _x1, y1 = box
    th = tag_height(y1 - y0, picture_long)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (_, ref_h), _ = cv2.getTextSize("Hg", font, 1.0, 1)
    scale = th * 0.52 / max(1, ref_h)
    thick = max(1, int(round(th / 16.0)))
    pad = int(round(th * 0.25))
    w = max(cv2.getTextSize(t, font, scale, thick)[0][0] for t in lines) + 2 * pad
    h = int(round(th * len(lines)))
    H, W = out.shape[:2]
    left = int(min(max(0, x0), max(0, W - w)))
    top = int(round(y0)) - h
    # another check's tag on the same corner: this one goes on top of it
    for _ in range(6):
        hit = [r for r in (taken or []) if abs(r[0] - left) < th and
               top < r[3] and r[1] < top + h]
        if not hit:
            break
        top = min(r[1] for r in hit) - h
    if top < 0:
        top = int(round(y0))
    top = int(min(max(0, top), max(0, H - h)))
    if taken is not None:
        taken.append((left, top, left + w, top + h))
    region = out[top:top + h, left:left + w]
    if region.size:
        tint = np.empty_like(region)
        tint[:] = col
        cv2.addWeighted(tint, 0.92, region, 0.08, 0, region)
    ink = ink_for(col)
    for i, t in enumerate(lines):
        (tw, tht), _b = cv2.getTextSize(t, font, scale, thick)
        base = int(round(top + i * th + (th + tht) / 2.0))
        cv2.putText(out, t, (left + pad, base), font, scale, ink, thick, cv2.LINE_AA)


def draw(image, raw, banner=True, title="", labels=True, only_failures=False, search=False):
    """The result picture: every check outlined in its shape and colour, its
    caption beside it, and the verdict in a banner above (never over) the
    image. labels: True every caption, "failed" only failed checks', False
    none."""
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    long_side = max(h, w)
    # the factory width (never under 2 px), then the thickness setting -- so the
    # setting also thickens / thins lines on small pictures
    scale = STYLE["line_picture"] / DEFAULT_LINE_PICTURE
    lw = max(1, int(round(max(2.0, long_side * DEFAULT_LINE_PICTURE) * scale)))
    thin = max(1, lw // 2)
    taken = []
    for t in raw["tools"]:
        if only_failures and t["status"] == "pass":
            continue
        col = colour_bgr(t["status"])
        for g in t.get("graphics", []):
            st = g.get("status", t["status"])
            if st == "search" and not search:
                continue
            pts = np.array(g.get("points", []), np.float32)
            if pts.size == 0:
                continue
            gcol = colour_bgr(st) if g.get("role") != "region" else col
            width = lw if g.get("role") == "region" else thin
            kind = g.get("kind")
            if kind == "rect" and len(pts) >= 2:
                cv2.rectangle(out, tuple(np.round(pts[0]).astype(int)),
                              tuple(np.round(pts[1]).astype(int)), gcol, width, cv2.LINE_AA)
            elif kind == "poly":
                cv2.polylines(out, [np.round(pts).astype(np.int32)], True, gcol, width, cv2.LINE_AA)
            elif kind == "line" and len(pts) >= 2:
                cv2.line(out, tuple(np.round(pts[0]).astype(int)),
                         tuple(np.round(pts[-1]).astype(int)), gcol, width, cv2.LINE_AA)
            elif kind == "circle" and len(pts) >= 2:
                c, r = pts[0], float(np.hypot(*(pts[1] - pts[0])))
                cv2.circle(out, tuple(np.round(c).astype(int)), int(round(r)), gcol, width,
                           cv2.LINE_AA)
            elif kind == "point":
                cv2.drawMarker(out, tuple(np.round(pts[0]).astype(int)), gcol,
                               cv2.MARKER_CROSS, 6 * lw, thin)
        name_it = labels is True or (labels == "failed" and t["status"] != "pass")
        lines = t.get("caption") or (caption(t) if name_it else [])
        if name_it and lines:
            g = outline(t)
            pts = g["points"] if g else [q for x in t.get("graphics", [])
                                         if x.get("status") != "search" for q in x.get("points", [])]
            if pts:
                _tag_on_box(out, lines, _bbox(pts), col, long_side, taken)
    if banner:
        ok = raw["verdict"] == "PASS"
        fs = max(0.4, long_side / 1600.0)
        bh = int(28 * fs * 1.6)
        strip = np.zeros((bh, w, 3), np.uint8)
        fill = colour_bgr("pass" if ok else "fail")
        strip[:] = fill
        failed = [t["tool_id"] for t in raw["tools"] if t["status"] != "pass"]
        txt = "%s%s   %.0f ms%s" % (title + "   " if title else "", "OK" if ok else "NG",
                                     raw.get("total_ms", 0.0),
                                     ("   NG: " + ", ".join(failed)) if failed else "")
        cv2.putText(strip, txt, (int(10 * fs), int(bh * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, ink_for(fill), max(1, int(round(long_side * DEFAULT_LINE_PICTURE)) // 2),
                    cv2.LINE_AA)   # the words keep their weight whatever the line thickness
        out = np.vstack([strip, out])
    return out


def translate(graphic_points, dx, dy):
    return [(float(x) + dx, float(y) + dy) for x, y in graphic_points]


def ellipse_points(cx, cy, rx, ry, angle_deg=0.0, n=64):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + rx * math.cos(t) * ca - ry * math.sin(t) * sa,
             cy + rx * math.cos(t) * sa + ry * math.sin(t) * ca)
            for t in (2 * math.pi * i / n for i in range(n))]
