"""Counting, colour and defect accuracy audit.

Ground truth is exact here: we place N features and ask how many were found; we
paint a known colour and ask what it is called; we inject a defect of known size
and ask whether it is caught.

The most important column in the defect section is the false-positive row. A
detector that never misses but rejects good parts gets switched off within a
week, so a clean part must come back clean.
"""
import sys, os
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision_station.core.engine import Context
from vision_station.tools import make_tool


def run_tool(ttype, params, image, ctx=None, reference=None):
    base = {"enabled": True, "critical": True, "angle": 0.0, "shape": "rect"}
    spec = {"id": "t", "type": ttype, "params": dict(base, **params)}
    ctx = ctx or Context(reference=reference if reference is not None else image)
    return make_tool(spec).run(image, ctx), ctx


def section(t):
    print("\n" + t)
    print("  " + "-" * 68)


# ------------------------------------------------------------------ blobs
def dots_image(n, w=640, h=200, r=16, gap=None, bright=225, bg=30, noise=0):
    img = np.full((h, w, 3), bg, np.uint8)
    gap = gap or w // (n + 1)
    for i in range(n):
        cv2.circle(img, (gap * (i + 1), h // 2), r, (bright,) * 3, -1)
    if noise:
        img = np.clip(img.astype(np.int16) +
                      np.random.RandomState(0).normal(0, noise, img.shape),
                      0, 255).astype(np.uint8)
    return img


def audit_count_blob():
    section("A. count_blob — does it count what is actually there?")
    print("  %-34s %8s %8s   %s" % ("case", "true", "found", ""))
    bad = 0
    for n in (1, 3, 4, 6, 10):
        img = dots_image(n)
        r, _ = run_tool("count_blob", {"roi": [0, 0, 640, 200], "space": "gray",
                                       "threshold": "otsu", "min_area": 50,
                                       "max_area": 100000, "expect_count": n,
                                       "row_tol": 40, "invert": False,
                                       "open_px": 3}, img)
        got = r.values.get("count")
        ok = got == n
        bad += 0 if ok else 1
        print("  %-34s %8d %8s   %s" % ("%d dots" % n, n, got, "ok" if ok else "BAD"))

    for label, kw in (("with noise sigma 12", {"noise": 12}),
                      ("small dots r=6", {"r": 6}),
                      ("low contrast (bright 90)", {"bright": 90})):
        img = dots_image(4, **kw)
        r, _ = run_tool("count_blob", {"roi": [0, 0, 640, 200], "space": "gray",
                                       "threshold": "otsu", "min_area": 20,
                                       "max_area": 100000, "expect_count": 4,
                                       "row_tol": 40, "invert": False,
                                       "open_px": 3}, img)
        got = r.values.get("count")
        ok = got == 4
        bad += 0 if ok else 1
        print("  %-34s %8d %8s   %s" % (label, 4, got, "ok" if ok else "BAD"))

    img = dots_image(4)
    r, _ = run_tool("count_blob", {"roi": [0, 0, 640, 200], "space": "gray",
                                   "threshold": "otsu", "min_area": 50,
                                   "max_area": 100000, "expect_count": 5,
                                   "row_tol": 40, "invert": False, "open_px": 3}, img)
    print("  %-34s %8s %8s   %s" % ("4 present, 5 expected", "fail", r.status,
                                    "ok" if r.status == "fail" else "BAD"))
    return bad


def audit_count_template():
    section("B. count_template — repeated feature counting")
    print("  %-34s %8s %8s   %s" % ("case", "true", "found", ""))
    bad = 0
    for n in (2, 4, 6):
        img = dots_image(n, r=18)
        gap = 640 // (n + 1)
        r, _ = run_tool("count_template",
                        {"roi": [0, 0, 640, 200],
                         "template_roi": [gap - 22, 78, 44, 44],
                         "expect_count": n, "min_score": 0.7, "overlap": 0.3}, img)
        got = r.values.get("count")
        ok = got == n
        bad += 0 if ok else 1
        print("  %-34s %8d %8s   %s" % ("%d features" % n, n, got,
                                        "ok" if ok else "BAD"))
    return bad


# ----------------------------------------------------------------- colour
def swatch(bgr, w=200, h=120):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = bgr
    return img


def audit_color():
    """A tool that passes whenever you tell it what to expect is worthless, so
    the real test is the full confusion matrix: every painted colour against
    every expected one. The diagonal must pass and everything else must fail.

    deltaE margin matters as much as the verdict. A correct answer sitting at
    deltaE 34 against a threshold of 40 is one lighting change away from being
    wrong, which is why the swatches want recalibrating against the actual
    lamp on the station."""
    section("C. color_classify — confusion matrix and margins")
    names = ["red", "green", "blue", "yellow"]
    paints = {"red": (40, 40, 210), "green": (60, 190, 60),
              "blue": (210, 70, 50), "yellow": (40, 210, 215)}
    bad = 0
    print("      painted \\ expected   " + "  ".join("%-7s" % n for n in names))
    for painted in names:
        row = []
        for expect in names:
            img = swatch(paints[painted])
            ctx = Context(reference=img)
            ctx.share["src"] = [{"cx": 100, "cy": 60, "x": 60, "y": 30,
                                 "w": 80, "h": 60, "area": 4800}]
            r, _ = run_tool("color_classify",
                            {"roi": [0, 0, 200, 120], "source": "src",
                             "expect": [expect], "deltaE_max": 40.0}, img, ctx=ctx)
            want = "pass" if painted == expect else "fail"
            if r.status != want:
                bad += 1
                row.append("WRONG")
            else:
                row.append("pass" if r.status == "pass" else ".")
        print("      %-20s " % painted + "  ".join("%-7s" % c for c in row))

    print("\n      deltaE of the correct answer (threshold 40):")
    for painted in names:
        img = swatch(paints[painted])
        ctx = Context(reference=img)
        ctx.share["src"] = [{"cx": 100, "cy": 60, "x": 60, "y": 30,
                             "w": 80, "h": 60, "area": 4800}]
        r, _ = run_tool("color_classify",
                        {"roi": [0, 0, 200, 120], "source": "src",
                         "expect": [painted], "deltaE_max": 40.0}, img, ctx=ctx)
        led = (r.values.get("leds") or [{}])[0]
        de = led.get("deltaE")
        flag = "" if de is None else ("  <- thin margin" if de > 25 else "")
        print("        %-8s named %-8s deltaE %-6s%s"
              % (painted, led.get("color"), de, flag))
    return bad


# ----------------------------------------------------------------- defect
def panel(w=400, h=300, level=150, noise=0, shift=0, gain=1.0):
    img = np.full((h, w, 3), level, np.uint8)
    cv2.rectangle(img, (40, 40), (w - 40, h - 40), (level + 25,) * 3, -1)
    for i in range(4):
        cv2.circle(img, (90 + i * 70, 150), 18, (level - 45,) * 3, -1)
    if gain != 1.0 or shift:
        img = np.clip(img.astype(np.float32) * gain + shift, 0, 255).astype(np.uint8)
    if noise:
        img = np.clip(img.astype(np.float32) +
                      np.random.RandomState(2).normal(0, noise, img.shape),
                      0, 255).astype(np.uint8)
    return img


def audit_defect():
    section("D. defect_diff — catching real defects, ignoring good parts")
    ref = panel()
    P = {"roi": [20, 20, 360, 260], "threshold": 22, "illum_blur": 51,
         "open_px": 2, "min_area": 25, "scratch_elongation": 3.5,
         "max_defects": 0, "max_area_px": 0}
    bad = 0

    r, _ = run_tool("defect_diff", P, ref.copy(), reference=ref)
    ok = r.status == "pass"
    bad += 0 if ok else 1
    print("  %-40s %-8s %s" % ("identical to reference (must PASS)", r.status,
                               "ok" if ok else "FALSE REJECT"))

    for label, noise, gain, shift in (("sensor noise sigma 4", 4, 1.0, 0),
                                      ("lighting 8% brighter", 0, 1.08, 0),
                                      ("lighting offset +10", 0, 1.0, 10)):
        cur = panel(noise=noise, gain=gain, shift=shift)
        r, _ = run_tool("defect_diff", P, cur, reference=ref)
        ok = r.status == "pass"
        bad += 0 if ok else 1
        print("  %-40s %-8s %s" % ("good part, " + label, r.status,
                                   "ok" if ok else "FALSE REJECT"))

    for label, draw in (
            ("6 px dent", lambda im: cv2.circle(im, (200, 90), 6, (60, 60, 60), -1)),
            ("12 px dent", lambda im: cv2.circle(im, (200, 90), 12, (60, 60, 60), -1)),
            ("thin scratch", lambda im: cv2.line(im, (120, 220), (280, 232),
                                                 (70, 70, 70), 2)),
            ("missing feature", lambda im: cv2.circle(im, (90, 150), 20,
                                                      (175, 175, 175), -1))):
        cur = panel()
        draw(cur)
        r, _ = run_tool("defect_diff", P, cur, reference=ref)
        ok = r.status == "fail"
        bad += 0 if ok else 1
        print("  %-40s %-8s %s" % ("defect: " + label, r.status,
                                   "ok (caught)" if ok else "MISSED"))
    return bad


def audit_presence():
    section("E. presence — feature there or not")
    ref = panel()
    P = {"roi": [70, 130, 40, 40], "min_score": 0.65, "search_pad": 15}
    bad = 0
    r, _ = run_tool("presence", P, panel(), reference=ref)
    ok = r.status == "pass"
    bad += 0 if ok else 1
    print("  %-40s %-8s %s" % ("feature present (must PASS)", r.status,
                               "ok" if ok else "BAD"))
    gone = panel()
    cv2.circle(gone, (90, 150), 22, (175, 175, 175), -1)
    r, _ = run_tool("presence", P, gone, reference=ref)
    print("  %-40s %-8s %s" % ("feature removed", r.status,
                               "ok (caught)" if r.status == "fail" else "MISSED"))
    return bad


if __name__ == "__main__":
    total = 0
    total += audit_count_blob()
    total += audit_count_template()
    total += audit_color()
    total += audit_defect()
    total += audit_presence()
    print("\n  total problems: %d" % total)
