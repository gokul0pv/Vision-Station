"""Measurement accuracy audit.

Same approach that exposed the alignment bugs: synthesise a scene whose true
dimensions are known exactly, ask the tool, compare.

Measurement is the tool worth auditing hardest. A wrong count is visible to
anyone looking at the part. A wrong millimetre reading looks perfectly
plausible, gets written on a report, and can sit there for months.
"""
import sys, os
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Context
from vstools import make_tool


def bar_image(w=800, h=300, x0=150, x1=650, blur=0.0, noise=0.0,
              bg=40, fg=210, subpixel=0.0):
    """A dark field with one bright bar. Edges at exactly x0 and x1.

    subpixel shifts the right edge by a fractional pixel using linear
    interpolation, so we can check whether the tool resolves better than one
    whole pixel or is quietly rounding."""
    img = np.full((h, w), float(bg), np.float64)
    img[:, x0:x1] = fg
    if subpixel:
        frac = subpixel - int(subpixel)
        xi = x1 + int(subpixel)
        if 0 < xi < w - 1:
            img[:, xi] = bg + (fg - bg) * frac   # coverage = frac
    img = img.astype(np.uint8)
    if blur:
        k = int(blur) * 2 + 1
        img = cv2.GaussianBlur(img, (k, k), blur)
    if noise:
        img = np.clip(img.astype(np.float64) +
                      np.random.RandomState(0).normal(0, noise, img.shape),
                      0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def measure(img, roi, extra=None):
    spec = {"id": "m", "type": "measure_length",
            "params": dict({"roi": roi, "axis": "x", "polarity": "any",
                            "min_contrast": 2.0, "max_edges": 2,
                            "mm_per_px": 0.0, "nominal_mm": 0.0,
                            "tolerance_mm": 1e9, "enabled": True,
                            "critical": True, "angle": 0.0, "shape": "rect"},
                           **(extra or {}))}
    t = make_tool(spec)
    ctx = Context(reference=img)
    return t.run(img, ctx)


def section(title):
    print("\n" + title)
    print("  " + "-" * 66)


def audit_pixel_accuracy():
    section("A. Pixel-distance accuracy, clean synthetic edges")
    print("  %-26s %10s %10s %9s" % ("case", "true_px", "measured", "error"))
    worst = 0.0
    for name, x0, x1 in (("500 px span", 150, 650),
                         ("200 px span", 300, 500),
                         ("60 px span",  370, 430),
                         ("edge near ROI wall", 155, 645)):
        img = bar_image(x0=x0, x1=x1)
        r = measure(img, [140, 100, 520, 100])
        got = r.values.get("length_px", float("nan"))
        err = abs(got - (x1 - x0)) if got == got else float("inf")
        worst = max(worst, err if np.isfinite(err) else 99)
        print("  %-26s %10d %10.3f %9.3f  %s"
              % (name, x1 - x0, got, err, "ok" if err < 1.0 else "BAD"))
    print("  worst error: %.3f px" % worst)
    return worst


def audit_subpixel():
    section("B. Subpixel resolution  (does it beat whole-pixel rounding?)")
    print("  %-26s %10s %10s %9s" % ("right edge shift", "true_px", "measured", "error"))
    errs = []
    for shift in (0.0, 0.25, 0.5, 0.75):
        img = bar_image(x0=150, x1=650, subpixel=shift)
        r = measure(img, [140, 100, 520, 100])
        got = r.values.get("length_px", float("nan"))
        true = 500 + shift
        err = abs(got - true) if got == got else float("inf")
        errs.append(err)
        print("  %-26s %10.2f %10.3f %9.3f" % ("+%.2f px" % shift, true, got, err))
    print("  mean error %.3f px  -> %s" % (np.mean(errs),
          "subpixel works" if np.mean(errs) < 0.35 else
          "NOT subpixel (rounding to whole pixels)"))
    return float(np.mean(errs))


def audit_robustness():
    section("C. Robustness: blur and noise")
    print("  %-26s %10s %10s %9s" % ("condition", "true_px", "measured", "error"))
    worst = 0.0
    for name, blur, noise in (("clean", 0, 0), ("blur 2", 2, 0), ("blur 5", 5, 0),
                              ("noise sigma 8", 0, 8), ("blur 3 + noise 8", 3, 8)):
        img = bar_image(x0=200, x1=600, blur=blur, noise=noise)
        r = measure(img, [190, 100, 420, 100])
        got = r.values.get("length_px", float("nan"))
        err = abs(got - 400) if got == got else float("inf")
        worst = max(worst, err if np.isfinite(err) else 99)
        print("  %-26s %10d %10.3f %9.3f  %s"
              % (name, 400, got, err, "ok" if err < 2.0 else "BAD"))
    print("  worst error: %.3f px" % worst)
    return worst


def audit_polarity():
    """Polarity filters to edges of one kind only. A single bar therefore has
    just one edge of each polarity and must fail with a clear message rather
    than inventing a second edge -- that is correct, not a defect. Two bars are
    needed to exercise a same-polarity pair."""
    section("D. Polarity selection")
    single = bar_image(x0=250, x1=550)
    print("  single bar (one edge of each polarity):")
    for pol in ("any", "dark_to_light", "light_to_dark"):
        r = measure(single, [200, 100, 400, 100], {"polarity": pol})
        expect = "pass" if pol == "any" else "fail"
        print("    %-15s status=%-5s len=%-9s  %s"
              % (pol, r.status, r.values.get("length_px"),
                 "ok" if r.status == expect else "UNEXPECTED"))

    two = np.full((300, 800), 40, np.uint8)
    two[:, 200:300] = 210
    two[:, 500:600] = 210
    two = cv2.cvtColor(two, cv2.COLOR_GRAY2BGR)
    print("  two bars, edges at 200/300 and 500/600:")
    for pol, want in (("any", 400.0), ("dark_to_light", 300.0),
                      ("light_to_dark", 300.0)):
        r = measure(two, [180, 100, 460, 100], {"polarity": pol, "max_edges": 4})
        got = r.values.get("length_px")
        ok = got is not None and abs(got - want) < 1.0
        print("    %-15s len=%-9s want %-6.1f  %s"
              % (pol, got, want, "ok" if ok else "BAD"))


def audit_mm_conversion():
    section("E. Millimetre conversion and tolerance verdict")
    img = bar_image(x0=200, x1=600)          # 400 px
    for mm_per_px, nominal, tol in ((0.05, 20.0, 0.1),     # 400px = 20.000 mm
                                    (0.05, 20.5, 0.1),     # out of tolerance
                                    (0.10, 40.0, 0.5)):
        r = measure(img, [190, 100, 420, 100],
                    {"mm_per_px": mm_per_px, "nominal_mm": nominal,
                     "tolerance_mm": tol})
        mm = r.values.get("length_mm")
        expect_mm = 400 * mm_per_px
        expect_status = "pass" if abs(expect_mm - nominal) <= tol else "fail"
        ok = (r.status == expect_status and mm is not None
              and abs(mm - expect_mm) < 0.02)
        print("  scale %.2f nominal %.1f tol %.2f -> %.3f mm  status=%-5s  %s"
              % (mm_per_px, nominal, tol, mm if mm is not None else float("nan"),
                 r.status, "ok" if ok else "BAD (expected %s)" % expect_status))


def audit_calibrate():
    section("F. calibrate_scale feeding measure_length")
    img = bar_image(x0=200, x1=600)          # 400 px known to be 20 mm
    spec = [{"id": "cal", "type": "calibrate_scale",
             "params": {"roi": [190, 100, 420, 100], "known_mm": 20.0, "axis": "x",
                        "polarity": "any", "min_contrast": 2.0, "enabled": True,
                        "critical": True, "angle": 0.0, "shape": "rect"}},
            {"id": "len", "type": "measure_length",
             "params": {"roi": [190, 100, 420, 100], "axis": "x", "polarity": "any",
                        "min_contrast": 2.0, "max_edges": 2, "mm_per_px": 0.0,
                        "nominal_mm": 20.0, "tolerance_mm": 0.05, "enabled": True,
                        "critical": True, "angle": 0.0, "shape": "rect"}}]
    ctx = Context(reference=img)
    tools = [make_tool(s) for s in spec]
    r1 = tools[0].run(img, ctx)
    print("  calibrate -> status=%s  mm_per_px=%s (want 0.0500)"
          % (r1.status, r1.values.get("mm_per_px")))
    print("  ctx.mm_per_px after calibrate: %s" % getattr(ctx, "mm_per_px", None))
    r2 = tools[1].run(img, ctx)
    print("  measure   -> status=%s  length_mm=%s (want 20.000)"
          % (r2.status, r2.values.get("length_mm")))
    good = (r2.values.get("length_mm") is not None
            and abs(r2.values["length_mm"] - 20.0) < 0.05)
    print("  chained calibration: %s" % ("ok" if good else "BAD"))
    return good


if __name__ == "__main__":
    audit_pixel_accuracy()
    audit_subpixel()
    audit_robustness()
    audit_polarity()
    audit_mm_conversion()
    audit_calibrate()
