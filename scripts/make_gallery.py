"""Build the gallery: every tool, detecting, photographed in the application.

    python scripts/make_gallery.py [output folder]          (default: gallery/)

Builds a gallery application of four jobs -- the demo LED panel good and
rejected, the real chassis front-top view, and a label with a QR code --
using every tool that can run without a model you have trained yourself, runs
it in QuickBuild, and captures each tool's Tool Edit Control showing what it
found. Also the QuickBuild workspace, a Job Editor, and the Train and Graphics
tabs. Writes JPEGs and an index.html beside them.

The Classify photo uses a toy three-colour classifier made for the purpose;
YOLO, the AI surface check and your own classifiers need models you train.
"""
import json
import math
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def _tool(tid, ty, **p):
    from vision_station.core.schema import defaults
    q = defaults(ty)
    q.update(p)
    return {"id": tid, "type": ty, "params": q}


def build_app(root):
    """The gallery application's four jobs."""
    from vision_station.core import job as jobmod
    from make_test_onnx import build_color_classifier
    demo = os.path.join(HERE, "projects", "demo_led_panel")
    ref = cv2.imread(os.path.join(demo, "reference.png"))
    j = jobmod.create_job(os.path.join(root, "gallery"), "Tool gallery")
    demo_tools = json.load(open(os.path.join(demo, "project.json"), encoding="utf-8"))["tools"]

    # ---- LED panel: the geometry, colour, text and logic tools
    panel = [t for t in demo_tools] + [
        _tool("screw_present", "presence", roi=[46, 46, 34, 34], search_pad=10),
        _tool("label_histogram", "histogram", roi=[150, 232, 260, 42], mean_min=40),
        _tool("led1_colour", "color_match", roi=[158, 138, 24, 24], max_delta_e=15),
        _tool("red_coverage", "color_segment", roi=[140, 120, 60, 60], hue_min=170, hue_max=10,
              min_pct=10),
        _tool("led_pitch", "caliper", roi=[120, 140, 400, 20], pair_polarity="dark",
              channel="blue"),
        _tool("led1_circle", "find_circle", roi=[142, 122, 56, 56], shape="ellipse"),
        _tool("led3_circle", "find_circle", roi=[342, 122, 56, 56], shape="ellipse"),
        _tool("led2_ellipse", "find_ellipse", roi=[242, 122, 56, 56], shape="ellipse"),
        _tool("top_edge", "find_line", roi=[150, 5, 320, 30]),
        _tool("left_edge", "find_line", roi=[5, 120, 30, 200], angle=90.0),
        _tool("corner", "intersect", first="top_edge", second="left_edge"),
        _tool("led_row_line", "fit_shape", fit="line", source="led_row", max_deviation=2.0),
        _tool("led_spacing", "distance", first="led1_circle", second="led3_circle"),
        _tool("despeckle", "image_process", operator="median", kernel=5),
        _tool("focus", "sharpness", roi=[20, 20, 600, 380]),
        _tool("lot_read", "ocr", roi=[150, 230, 260, 46]),
        _tool("all_text", "ocr", roi=[120, 210, 320, 80]),
        _tool("led1_class", "classify", roi=[155, 135, 30, 30], model="models/colours.onnx",
              classes="", accept="red", normalize="none"),
        _tool("decision", "results_analysis",
              expression="led_row.count == 4 and led_spacing.value_px > 190 and focus.passed"),
        _tool("led_fixture", "fixture_npoint", points="led1_circle,led3_circle"),
        _tool("led_midpoint", "construct", operation="midpoint", first="led1_circle",
              second="led3_circle"),
        _tool("code_upright", "affine_transform", roi=[150, 230, 260, 46], scale=2.0),
        _tool("code_on_upright", "histogram", roi=[20, 10, 480, 72], capture="code_upright"),
        _tool("spacing_stats", "data_analysis", source="led_spacing.value_px",
              lower_limit="198", upper_limit="202"),
        _tool("plate_calibration", "calibrate_npoint",
              pairs="20,20 = 0,0; 620,20 = 60,0; 620,400 = 60,38; 20,400 = 0,38"),
    ]
    s1 = j.add_side("panel", "LED panel", ref, panel)
    os.makedirs(os.path.join(s1.folder, "models"), exist_ok=True)
    build_color_classifier(os.path.join(s1.folder, "models", "colours.onnx"),
                           ["red", "green", "blue"])

    # ---- LED panel, rejected part: what failures look like
    reject = [t for t in demo_tools] + [
        _tool("panel_surface", "trained_surface", roi=[110, 280, 420, 90]),
        _tool("changes", "image_combine", operation="difference")]
    s2 = j.add_side("reject", "LED panel - reject", ref, reject)
    s2.spec["image_source"] = {"mode": "files", "paths": [os.path.join(demo, "bad.png")]}
    s2.save(s2.spec)
    s2.add_training_image("panel_surface", ref, "ok")
    s2.add_training_image("panel_surface", cv2.imread(os.path.join(demo, "good.png")), "ok")
    s2.train_tool("panel_surface")

    # ---- a real chassis, front-top view: only where its photographs are on
    # this machine. They are production photographs and are not published,
    # so a clone of the repository builds the gallery without this side.
    chassis_json = os.path.join(HERE, "projects", "chassis_example", "project.json")
    photo = os.path.join(HERE, "projects", "chassis_example", "images",
                         "FRT_TOP_Corner_09092026_163024.jpeg")
    if os.path.exists(chassis_json) and os.path.exists(photo):
        chassis_tools = json.load(open(chassis_json, encoding="utf-8"))["tools"]
        cimg = cv2.imread(photo)
        chassis = [dict(t) for t in chassis_tools] + [
            _tool("hole_a", "find_circle", roi=[1244, 756, 68, 68], shape="ellipse"),
            _tool("hole_b", "find_circle", roi=[1254, 872, 68, 68], shape="ellipse"),
            _tool("hole_spacing", "distance", first="hole_a", second="hole_b",
                  min_value=115.5, max_value=117.5),
            _tool("serial_ai", "ocr", roi=[1540, 330, 460, 150]),
        ]
        j.add_side("chassis", "Chassis front top", cimg, chassis)

    # ---- a label with a QR code
    label = np.full((360, 560, 3), 235, np.uint8)
    cv2.rectangle(label, (20, 20), (540, 340), (40, 40, 40), 2)
    qr = cv2.QRCodeEncoder.create().encode("SN:VSX30180450G")
    qr = cv2.resize(qr, (220, 220), interpolation=cv2.INTER_NEAREST)
    label[70:290, 50:270] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
    cv2.putText(label, "NX-540", (310, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3)
    cv2.putText(label, "LOT 4471", (310, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2)
    j.add_side("label", "Label", label, [_tool("qr", "code_read", roi=[40, 60, 240, 240])])

    # ---- a glue bead along a path, with a gap
    bead = np.full((300, 600, 3), (190, 195, 200), np.uint8)
    path = np.int32([[60, 220], [200, 90], [380, 90], [540, 220]])
    cv2.polylines(bead, [path], False, (40, 70, 110), 13, cv2.LINE_AA)
    cv2.line(bead, (270, 90), (300, 90), (190, 195, 200), 17)           # a gap
    bead = cv2.GaussianBlur(bead, (3, 3), 0)
    j.add_side("bead", "Glue bead", bead, [
        _tool("glue", "bead", shape="polygon", points=path.tolist(), roi=[60, 90, 480, 130],
              width=13, width_tolerance=4, stations=40, search_half_width=20)])

    # ---- text round a cap, unwrapped
    cap = np.full((360, 360, 3), 60, np.uint8)
    cv2.circle(cap, (180, 180), 150, (215, 215, 215), -1)
    cv2.circle(cap, (180, 180), 80, (60, 60, 60), -1)
    for i, ch in enumerate("BATCH 2471 * OK * "):
        a = math.radians(-90 + i * 20)
        x, y = 180 + 115 * math.cos(a), 180 + 115 * math.sin(a)
        rot = cv2.getRotationMatrix2D((20, 20), -math.degrees(a) - 90, 1.0)
        tile = np.full((40, 40), 255, np.uint8)
        cv2.putText(tile, ch, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 3)
        tile = cv2.warpAffine(tile, rot, (40, 40), borderValue=255)
        ys, xs = int(y) - 20, int(x) - 20
        region = cap[ys:ys + 40, xs:xs + 40]
        region[tile < 128] = (30, 30, 30)
    j.add_side("cap", "Cap ring", cap, [
        _tool("ring", "polar_unwrap", roi=[25, 25, 310, 310], inner_radius=0.55, start_angle=-100,
              direction="clockwise")])
    return os.path.join(root, "gallery")


# tool id -> (job, record to show, caption)
CAPTIONS = {
    "locate_part": "Locate part (fixture) - finds the part and its angle; every later tool follows it.",
    "scale": "Calibrate scale - millimetres per pixel from a known length.",
    "led_row": "Count by brightness (blob) - the four LEDs found and counted.",
    "led_colours": "LED colour - each LED's colour identified.",
    "led_lit": "LED lit state - which LEDs are lit.",
    "corner_screws": "Count by shape - every screw matched against one taught screw.",
    "lot_code": "Verify text (OCV) - the printed code checked against what it should say.",
    "surface": "Dents and scratches - differences from the reference outlined.",
    "body_width": "Measure length - a caliper measuring edge to edge.",
    "screw_present": "Part present - a feature checked by pattern match.",
    "label_histogram": "Histogram - grey-level statistics judged against limits.",
    "led1_colour": "Color match - the region's colour against the reference (delta E).",
    "red_coverage": "Color segmenter - the share of the region within a colour range.",
    "led_pitch": "Caliper (edge pairs) - bars between edge pairs: count, widths, pitch.",
    "led1_circle": "Find circle - calipers on rays, a circle fitted through the edge points.",
    "led2_ellipse": "Find ellipse - an ellipse fitted through radial caliper points.",
    "top_edge": "Find line - calipers along an edge, a line fitted through them.",
    "corner": "Intersect lines - the corner where two fitted lines meet.",
    "led_row_line": "Fit line / circle - a line through the LED centres: are they in a row?",
    "led_spacing": "Distance / angle - measured between two found circles.",
    "despeckle": "Image processing - a filter whose output later tools can read.",
    "focus": "Image sharpness - is the image in focus, against the reference.",
    "lot_read": "Read text (OCR) - the printed characters read.",
    "all_text": "Find and read all text - every line of print located and read.",
    "ai_text": "AI text detection - a trained network finds the print, then it is read.",
    "led1_class": "Classify - a trained classifier sorts the region (toy colour model).",
    "decision": "Results analysis - accept or reject from an expression over results.",
    "panel_surface": "Trained surface check - learned from OK images; the scratch flagged.",
    "hole_a": "Find circle on the real chassis - a fastener hole fitted.",
    "hole_spacing": "Distance on the real chassis - spacing between two holes.",
    "serial_ai": "AI text detection on the real chassis serial label.",
    "panel_screws": "Count by shape on the real chassis - seven screws.",
    "serial_label_present": "Part present on the real chassis - the serial label.",
    "qr": "Read code - the QR code decoded.",
    "led_fixture": "Fixture from points - the part's pose from two found features (the LEDs).",
    "led_midpoint": "Create geometry - the midpoint of two found circles.",
    "changes": "Two-image operation - the difference from the reference, aligned first.",
    "code_upright": "Affine transform - the region resampled upright at twice the scale.",
    "code_on_upright": "A tool reading the transformed image - its region in that image's coordinates.",
    "spacing_stats": "Data analysis - the LED spacing's mean, spread and Cpk over runs.",
    "plate_calibration": "N-point calibration - pixels to millimetres from four known corners.",
    "glue": "Bead inspect - width and position at stations along the path; the gap flagged.",
    "ring": "Polar unwrap - text round a cap unrolled into a straight strip.",
}
EXTRA_RECORDS = {"changes": "LastRun.OutputImage", "code_upright": "LastRun.OutputImage",
                 "ring": "LastRun.OutputImage",
                 "led_pitch": "LastRun.Projection", "label_histogram": "LastRun.Histogram",
                 "red_coverage": "LastRun.ColorMask", "led_row": "LastRun.BlobImage"}
SHOTS = [("panel", t) for t in ("locate_part", "led_row", "led_colours", "led_lit",
                                  "corner_screws", "lot_code", "body_width", "screw_present",
                                  "label_histogram", "led1_colour", "red_coverage", "led_pitch",
                                  "led1_circle", "led2_ellipse", "top_edge", "corner",
                                  "led_row_line", "led_spacing", "despeckle", "focus", "lot_read",
                                  "all_text", "ai_text", "led1_class", "decision",
                                  "led_fixture", "led_midpoint", "code_upright",
                                  "code_on_upright", "spacing_stats", "plate_calibration")] + \
    [("reject", t) for t in ("led_row", "corner_screws", "surface", "panel_surface", "lot_code",
                             "changes")] + \
    [("chassis", t) for t in ("locate_part", "serial_label_present", "panel_screws", "hole_a",
                              "hole_spacing", "serial_ai")] + [("label", "qr"), ("bead", "glue"),
                                                              ("cap", "ring")]


def main(out=None):
    from PySide6.QtCore import QRectF, Qt, QTimer
    from PySide6.QtWidgets import QApplication
    from vision_station.ui import main_window as V
    out = os.path.abspath(out or os.path.join(HERE, "gallery"))
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    root = tempfile.mkdtemp()
    build_app(root)
    app = QApplication.instance() or QApplication(sys.argv)
    V.apply_theme(app)
    qb = V.QuickBuild(root)
    qb.resize(1600, 980)
    qb.show()
    shots = []

    def save(widget, name, caption, group):
        path = os.path.join(out, name + ".jpg")
        widget.grab().save(path, "JPG", 88)
        shots.append((name + ".jpg", caption, group))

    def frame(te, tid, pid):
        """On the tool's region; for a tool without one (Intersect, Distance,
        Fit line) on what it found -- showing the whole image made a corner a
        speck in the photo."""
        spec = next(t for t in qb.station.projects[pid].spec["tools"] if t["id"] == tid)
        roi = spec["params"].get("roi")
        c = te.display.canvas
        if te.display.current_record() not in V.FULL_FRAME_RECORDS:
            c.fit()
            return
        if not roi:
            res = qb.station.last.get(pid) or {}
            t = next((x for x in res.get("tools", []) if x["tool_id"] == tid), None)
            pts = [q for g in (t or {}).get("graphics", []) for q in g.get("points", [])]
            if pts:
                xs, ys = [q[0] for q in pts], [q[1] for q in pts]
                roi = [min(xs), min(ys), max(8, max(xs) - min(xs)), max(8, max(ys) - min(ys))]
        if roi:
            x, y, w, h = roi
            m = max(60, 0.6 * max(w, h))
            c.fitInView(QRectF(x - m, y - m, w + 2 * m, h + 2 * m), Qt.KeepAspectRatio)
            c.fit_mode = False
            c._refresh_items()
        else:
            c.fit()

    def run():
        qb.open_application("gallery")
        for _ in range(3):
            app.processEvents()
        qb.run_once()
        qb.station.wait(240)
        qb.mdi.setActiveSubWindow(qb.runSub)
        qb.runSub.showMaximized()
        app.processEvents()
        save(qb, "00_quickbuild", "QuickBuild - the application: jobs, run view, results queue.",
             "Application")
        pid_of = {j["pid"].split("/")[1]: j["pid"] for j in qb.station.jobs}
        je = qb.open_job_editor(pid_of["panel"])
        je.parent().showMaximized()
        for i in range(je.tree.topLevelItemCount()):
            it = je.tree.topLevelItem(i)
            if it.data(0, Qt.UserRole) in ("led_spacing", "led1_circle"):
                it.setExpanded(True)
        app.processEvents()
        save(je, "01_job_editor", "Job Editor - the job's tools, status, times and terminals.",
             "Application")
        je.close()
        n = 2
        for job, tid in SHOTS:
            if job not in pid_of:
                continue                     # the chassis, without its photographs
            pid = pid_of[job]
            te = qb.open_tool_edit(pid, tid)
            if te is None:
                continue
            te.parent().showMaximized()
            te.display.record.setCurrentText("LastRun.InputImage")
            app.processEvents()
            frame(te, tid, pid)
            app.processEvents()
            tag = "reject" if job == "reject" else "chassis" if job == "chassis" else job
            save(te, "%02d_%s_%s" % (n, tag, tid),
                 CAPTIONS.get(tid, tid) + (" (rejected part)" if job == "reject" else ""),
                 {"panel": "Tools on the demo LED panel", "reject": "A rejected part",
                  "chassis": "The real chassis", "label": "Codes",
                  "bead": "Bead inspection", "cap": "Image transforms"}[job])
            n += 1
            if tid in EXTRA_RECORDS:
                te.display.record.setCurrentText(EXTRA_RECORDS[tid])
                app.processEvents()
                frame(te, tid, pid)
                save(te, "%02d_%s_%s_record" % (n, tag, tid),
                     "%s - the %s record." % (CAPTIONS.get(tid, tid).split(" - ")[0],
                                              EXTRA_RECORDS[tid].split(".")[1]),
                     "Diagnostic records")
                n += 1
            if job == "reject" and tid == "panel_surface":
                te.props.tabs.setCurrentIndex(te.props.tabs.count() - 2)     # Train
                app.processEvents()
                save(te, "%02d_train_tab" % n, "Train tab - OK and NG images; Train, Tune, Test.",
                     "Application")
                n += 1
                te.props.tabs.setCurrentIndex(te.props.tabs.count() - 1)     # Graphics
                app.processEvents()
                save(te, "%02d_graphics_tab" % n, "Graphics tab - what the tool draws.",
                     "Application")
                n += 1
            te.close()
            app.processEvents()
        write_index(out, shots)
        qb.dirty.clear()
        qb.close()
        app.quit()

    QTimer.singleShot(800, run)
    app.exec()
    return out, shots


def write_index(out, shots):
    groups = []
    for f, cap, g in shots:
        if g not in groups:
            groups.append(g)
    parts = ["<!doctype html><meta charset='utf-8'><title>Vision Station - tool gallery</title>",
             "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f4f5f7;"
             "color:#1b1f23}h1{margin:0 0 4px}h2{margin:28px 0 10px;color:#1F6FB2}"
             ".g{display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));gap:14px}"
             "figure{margin:0;background:#fff;border:1px solid #d6dae0;border-radius:6px;"
             "overflow:hidden}figure img{width:100%;display:block}figcaption{padding:8px 10px;"
             "font-size:13px}</style>",
             "<h1>Vision Station - how each tool detects</h1>",
             "<p>Captured from the application, running the gallery application built by "
             "make_gallery.py.</p>"]
    for g in groups:
        parts.append("<h2>%s</h2><div class='g'>" % g)
        for f, cap, gg in shots:
            if gg == g:
                parts.append("<figure><a href='%s'><img src='%s'></a><figcaption>%s"
                             "</figcaption></figure>" % (f, f, cap))
        parts.append("</div>")
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    folder, shots = main(sys.argv[1] if len(sys.argv) > 1 else None)
    print("%d photos in %s" % (len(shots), folder))
