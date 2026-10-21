"""Every tool, created the way the UI creates it.

This exists because of a bug that only appeared through the property sheet. The
OCV tool checked `expect_text is not None`; the sheet supplies an empty string
by default, so the tool silently demanded that the part be blank and rejected
everything. Recipes hand-written in JSON worked, because they omitted the field.
Every test we had used hand-written JSON.

So: build each tool from `schema.defaults()` exactly as `add_tool` does, point it
at a plausible image, and check it behaves. Anything that errors, or that
rejects a good part with stock settings, is a tool nobody can use out of the box.
"""
import sys, os, traceback
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Context
from vstools import REGISTRY, make_tool
from schema import defaults


def scene(w=640, h=440):
    """A board with the features the stock tools expect to find."""
    img = np.full((h, w, 3), 38, np.uint8)
    cv2.rectangle(img, (30, 30), (w - 30, h - 30), (150, 142, 132), -1)
    for cx, cy in ((70, 70), (w - 70, 70), (70, h - 70), (w - 70, h - 70)):
        cv2.circle(img, (cx, cy), 16, (205, 205, 210), -1)
        cv2.circle(img, (cx, cy), 16, (70, 70, 75), 2)
    # BGR. The third was (215,165,60) which is a cyan, not the yellow the
    # expectation claimed -- the scene was wrong, not the tool.
    for i, col in enumerate([(60, 60, 215), (60, 195, 60),
                             (60, 215, 225), (215, 90, 60)]):
        cv2.circle(img, (170 + i * 90, 160), 22, col, -1)
    cv2.putText(img, "AB12345X", (170, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (248, 248, 248), 3)
    return img


# A workable ROI and any tool-specific wiring each type needs to be meaningful.
SETUP = {
    "find_ellipse": {"roi": [140, 120, 60, 60]},
    # across the LED row: four dark-on-light... the LEDs are darker than the board in grey
    "caliper": {"roi": [120, 138, 400, 24], "pair_polarity": "dark", "channel": "blue"},
    # the board's top edge runs along y = 20; an LED of radius 22 sits at (170, 150)
    "find_line": {"roi": [100, 5, 400, 30]},
    "find_circle": {"roi": [140, 120, 60, 60]},
    "align":           {"roi": [50, 50, 540, 340]},
    "calibrate_scale": {"roi": [40, 50, 560, 60], "known_mm": 20.0},
    "presence":        {"roi": [50, 50, 60, 60]},
    "count_template":  {"roi": [30, 30, 580, 380], "template_roi": [52, 52, 40, 40],
                        "expect_count": 4},
    "count_blob":      {"roi": [120, 120, 400, 90], "expect_count": 4},
    "color_classify":  {"roi": [120, 120, 400, 90], "source": "blobs",
                        "expect": ["red", "green", "yellow", "blue"]},
    "led_glow":        {"roi": [120, 120, 400, 90], "source": "blobs"},
    "defect_diff":     {"roi": [40, 40, 560, 360]},
    "anomaly":         {"roi": [40, 40, 560, 360]},
    "ocr":             {"roi": [150, 280, 320, 60]},
    "ocv":             {"roi": [150, 280, 320, 60],
                        "expect_regex": "^[A-Z0-9]{6,12}$"},
    "code_read":       {"roi": [150, 280, 320, 60]},
    # ROI must span the board edges, otherwise there is nothing to measure.
    "measure_length":  {"roi": [10, 200, 620, 60]},
    "yolo_detect":     {"roi": [40, 40, 560, 360]},
    "yolo_defect":     {"roi": [40, 40, 560, 360]},
    "trained_surface": {"roi": [40, 40, 560, 360]},
}

# Tools that legitimately cannot pass without something we do not have here:
# a trained model file, or a printed code on the part.
NEEDS_EXTERNAL = {"anomaly", "yolo_detect", "yolo_defect", "code_read",
                  "trained_surface",     # needs OK images before it can run
                  "distance",            # needs features found by other tools
                  "intersect", "fit_shape",
                  "results_analysis",    # needs an expression written for the job
                  "classify",            # needs a trained model
                  "fixture_npoint", "construct", "calibrate_npoint", "data_analysis",
                  "bead"}                # need features, point pairs, a value or a bead


def main():
    img = scene()
    print("Every tool, built from schema defaults exactly as the UI builds it")
    print("=" * 74)
    print("  %-17s %-8s %-9s %s" % ("tool", "status", "ms", "note"))
    print("  " + "-" * 72)

    problems = []
    for ttype in sorted(REGISTRY):
        params = defaults(ttype)
        params.update(SETUP.get(ttype, {"roi": [40, 40, 560, 360]}))
        ctx = Context(reference=img)
        # colour/glow read blobs published by a counter
        if ttype in ("color_classify", "led_glow"):
            cb = make_tool({"id": "blobs", "type": "count_blob",
                            "params": dict(defaults("count_blob"),
                                           **SETUP["count_blob"])})
            cb.run(img, ctx)
        try:
            r = make_tool({"id": ttype, "type": ttype, "params": params}).run(img, ctx)
        except Exception:
            problems.append((ttype, "raised", traceback.format_exc().splitlines()[-1]))
            print("  %-17s %-8s %-9s %s" % (ttype, "RAISED", "-",
                                            traceback.format_exc().splitlines()[-1][:34]))
            continue

        note = (r.message or "")[:36]
        flag = ""
        if r.status == "error":
            if ttype in NEEDS_EXTERNAL:
                flag = "(expected - needs a model, code or training)"
            else:
                problems.append((ttype, "error", r.message))
                flag = "<-- UNUSABLE WITH DEFAULTS"
        elif r.status == "fail" and ttype not in NEEDS_EXTERNAL:
            problems.append((ttype, "fail-on-good", r.message))
            flag = "<-- REJECTS A GOOD PART"
        print("  %-17s %-8s %-9.1f %s %s" % (ttype, r.status, r.ms, note, flag))

    print("  " + "-" * 72)
    if problems:
        print("\n  %d tool(s) need attention:" % len(problems))
        for t, kind, msg in problems:
            print("    %-17s %-13s %s" % (t, kind, (msg or "")[:60]))
    else:
        print("\n  every tool works with stock settings.")
    return len(problems)


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
