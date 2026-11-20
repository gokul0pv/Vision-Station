"""The camera opened and set up by the application (by IP address, every
parameter, no leaked connection), the display at full quality, result
pictures saved where Settings says, and Find parts.

Run with the rest:  python -m pytest tests.py tests_vp.py tests_package.py tests_settings_parts.py
"""
import importlib
import json
import os
import sys
import time

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _fresh_camera(devices, hidden=()):
    import fake_pylon as F
    F.install(devices, hidden)
    for m in [k for k in sys.modules if k.startswith("camera")]:
        del sys.modules[m]
    from vision_station.hardware import camera as cam_mod
    importlib.reload(cam_mod)
    return F, cam_mod


def _wait(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------- camera
def test_a_camera_discovery_misses_opens_by_its_ip_address():
    """On the user's network GigE discovery lists nothing while the
    acA5472-5gc answers at 192.168.3.6. By number it was "no camera"; by
    its address it opens."""
    import fake_pylon as F
    dev = F.Device("gige_classic", ip="192.168.3.6", model="acA5472-5gc")
    F, cam = _fresh_camera([], hidden=[dev])
    assert cam.list_devices() == []
    s = cam.CameraSettings()
    s.address = "192.168.3.6"
    t = cam.CameraThread(0, s)
    t.start()
    try:
        assert _wait(lambda: t.get_frame() is not None), t.last_error
        assert t.backend == "pylon"
        assert t.info["model"] == "acA5472-5gc"
    finally:
        t.stop()
    assert not t.is_alive()

    s.address = "192.168.3.99"                      # nothing there
    t = cam.CameraThread(0, s)
    t.start()
    t.join(5)
    assert t.backend is None and "192.168.3.99" in t.last_error, t.last_error
    assert "opencv" != t.backend, "fell back to another camera"


def test_without_an_address_the_message_says_to_enter_one():
    F, cam = _fresh_camera([])
    with pytest.raises(cam.CameraError) as e:
        cam.PylonCamera(0).open()
    assert "IP address" in str(e.value)


def test_stopping_while_the_camera_opens_lets_go_of_it():
    """The bug behind "controlled by another application" with nothing else
    running: a camera stopped while it was still opening came back to life
    after Open() and held the camera for good, so every later open in the
    application was refused."""
    import fake_pylon as F
    dev = F.Device("gige_classic", ip="10.0.0.5")
    F, cam = _fresh_camera([dev])
    real_open = dev.__class__
    slow = {"n": 0}
    orig = F._InstantCamera.Open

    def slow_open(self):
        slow["n"] += 1
        time.sleep(0.4)                            # opening takes a while
        return orig(self)
    F._InstantCamera.Open = slow_open
    try:
        s = cam.CameraSettings()
        s.address = "10.0.0.5"
        t = cam.CameraThread(0, s)
        t.start()
        time.sleep(0.1)
        t.stop()                                   # mid-open
        assert not t.is_alive(), "the thread kept running after a stop during open"
        assert t.backend is None or t.alive is False
        # and the next one opens normally
        t2 = cam.CameraThread(0, s)
        t2.start()
        assert _wait(lambda: t2.get_frame() is not None), t2.last_error
        t2.stop()
    finally:
        F._InstantCamera.Open = orig
    assert real_open is not None


def test_a_connection_left_by_a_crashed_program_is_waited_out():
    """A program that ended without closing leaves the GigE camera held
    until its heartbeat runs out (3 s): the open is retried, not refused."""
    import fake_pylon as F
    dev = F.Device("gige_classic", ip="10.0.0.6", fail_open=3)
    F, cam = _fresh_camera([dev])
    s = cam.CameraSettings()
    s.address = "10.0.0.6"
    t = cam.CameraThread(0, s)
    t.start()
    try:
        assert _wait(lambda: t.get_frame() is not None, 8), t.last_error
        assert dev.open_attempts == 4
    finally:
        t.stop()


def test_opens_of_one_camera_in_this_process_wait_for_the_last_holder():
    import fake_pylon as F
    dev = F.Device("gige_classic", ip="10.0.0.7", frame_period_s=0.02)
    F, cam = _fresh_camera([dev])
    s = cam.CameraSettings()
    s.address = "10.0.0.7"
    a = cam.CameraThread(0, s)
    a.start()
    assert _wait(lambda: a.get_frame() is not None)
    b = cam.CameraThread(0, s)
    b.start()
    time.sleep(0.3)
    assert b.backend is None, "a second connection opened while the first held it"
    a.stop()
    assert _wait(lambda: b.get_frame() is not None, 5), b.last_error
    b.stop()


def test_every_setting_reaches_the_camera_and_region_changes_restart_acquisition():
    import fake_pylon as F
    dev = F.Device("gige_classic", ip="10.0.0.8", frame_period_s=0.01)
    F, cam = _fresh_camera([dev])
    s = cam.CameraSettings()
    s.address = "10.0.0.8"
    s.exposure_ms = 20.0
    s.gain_db = 100
    s.auto_white_balance = False
    s.gamma = 0.7
    s.black_level = 12
    s.frame_rate = 4.0
    s.trigger = "hardware"
    s.trigger_edge = "FallingEdge"
    s.trigger_delay_us = 150.0
    s.roi = [101, 51, 2000, 1500]                  # snapped to the camera's steps
    s.reverse_x = True
    s.packet_size = 9000
    s.inter_packet_delay = 2000
    s.heartbeat_ms = 1000
    s.pixel_format = "BayerRG12"
    t = cam.CameraThread(0, s)
    t.start()
    try:
        assert _wait(lambda: t.backend == "pylon")
        n = dev.nodes
        assert _wait(lambda: n["TriggerMode"].value == "On")
        assert n["ExposureTimeAbs"].value == 20000.0
        assert n["GainRaw"].value == 100 and isinstance(n["GainRaw"].value, int)
        assert n["BalanceWhiteAuto"].value == "Off"
        assert n["GammaEnable"].value is True and abs(n["Gamma"].value - 0.7) < 1e-6
        assert n["BlackLevelRaw"].value == 12
        assert n["AcquisitionFrameRateEnable"].value is True
        assert n["AcquisitionFrameRateAbs"].value == 4.0
        assert n["TriggerActivation"].value == "FallingEdge"
        assert n["TriggerDelayAbs"].value == 150.0
        assert n["Width"].value == 2000 and n["Height"].value == 1500
        assert n["OffsetX"].value == 100 and n["OffsetY"].value == 50
        assert n["ReverseX"].value is True
        assert n["GevSCPSPacketSize"].value == 9000
        assert n["GevSCPD"].value == 2000
        assert n["PixelFormat"].value == "BayerRG12"
        assert dev.tl_nodes["HeartbeatTimeout"].value == 1000
        assert t.fresh_frames == 1, "a triggered camera needs no frames let go by"

        d = t.describe()
        assert d["gain_unit"] == "raw" and d["width_max"] == 5496
        assert "BayerRG8" in d["pixel_format"]["options"]

        s2 = s.copy()
        s2.roi = "full"                            # None would leave the region alone
        s2.trigger = "free"
        t.update_settings(s2)
        assert _wait(lambda: n["Width"].value == 5496 and n["TriggerMode"].value == "Off")
        assert n["OffsetX"].value == 0
    finally:
        t.stop()


def test_camera_settings_are_saved_per_pc_and_live_camera_exposure_wins(tmp_path):
    from PySide6.QtCore import QSettings
    from vision_station.hardware import camera
    from vision_station.ui import settings_dialog as sd
    qs = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    assert sd.load_camera_settings(qs, 0) is None
    s = camera.CameraSettings()
    s.address, s.exposure_ms, s.roi, s.trigger = "192.168.3.6", 12.5, [0, 0, 100, 80], "hardware"
    sd.save_camera_settings(qs, 0, s)
    got = sd.load_camera_settings(qs, 0)
    assert (got.address, got.exposure_ms, got.roi, got.trigger) == \
        ("192.168.3.6", 12.5, [0, 0, 100, 80], "hardware")
    qs.setValue("camera0/exposure_ms", 30.0)          # Live Camera writes this
    assert sd.load_camera_settings(qs, 0).exposure_ms == 30.0
    assert sd.load_camera_settings(qs, 0).address == "192.168.3.6"
    from vision_station.ui.live_view import saved_settings
    assert saved_settings(qs, 0).address == "192.168.3.6"


def test_settings_dialog_round_trips_every_field(qapp, tmp_path):
    from PySide6.QtCore import QSettings
    from vision_station.hardware import camera
    from vision_station.ui import settings_dialog as sd
    qs = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    s = camera.CameraSettings()
    s.address, s.exposure_ms, s.gain_db = "192.168.3.6", 15.0, 3.0
    s.trigger, s.trigger_delay_us, s.frame_rate = "hardware", 250.0, 3.5
    s.roi, s.reverse_y, s.packet_size, s.heartbeat_ms = [8, 4, 640, 480], True, 9000, 1500
    s.gamma, s.black_level = 0.8, 5.0
    sd.save_camera_settings(qs, 0, s)
    applied = {}
    dlg = sd.SettingsDialog(None, qs, on_camera=lambda i, x: applied.update(cam=(i, x)),
                            on_results=lambda d: applied.update(res=d))
    got = dlg.cameraPage.settings()
    for k in ("address", "exposure_ms", "gain_db", "trigger", "trigger_delay_us", "frame_rate",
              "roi", "reverse_y", "packet_size", "heartbeat_ms", "gamma", "black_level"):
        assert getattr(got, k) == getattr(s, k), k
    dlg.resultsPage.folder.setText(str(tmp_path / "pics"))
    dlg.resultsPage.which.setCurrentIndex(1)
    dlg.resultsPage.original.setChecked(True)
    assert dlg.apply()
    assert applied["cam"][0] == 0 and applied["cam"][1].address == "192.168.3.6"
    assert applied["res"]["folder"] == str(tmp_path / "pics") and applied["res"]["which"] == "fail"
    assert os.path.isdir(str(tmp_path / "pics"))
    assert sd.load_result_settings(qs)["original"] is True
    dlg.close()


# ---------------------------------------------------------------- display quality
def test_the_display_draws_every_pixel_at_100_percent(qapp):
    """At 100% a 5472-pixel frame was drawn from an 1800-pixel copy: blurred
    everywhere between fit and 300%."""
    from vision_station.ui.canvas import ImageCanvas
    c = ImageCanvas()
    c.resize(1200, 800)
    c.show()
    img = np.zeros((3648, 5472, 3), np.uint8)
    img[::2, ::2] = 255                                # finest possible detail
    c.set_image(img, keep_view=False)
    fitted = c._pix_scale
    assert fitted < 0.6, "the fitted view was not reduced (%.2f)" % fitted
    c.zoom_100()
    assert c._pix_scale == 1.0, "100%% zoom drew a reduced copy (%.2f)" % c._pix_scale
    c.fit()
    assert c._pix_scale < 0.6, "zooming back out kept a full-size pixmap (%.2f)" % c._pix_scale
    # reduced by averaging, not sampling: a 1-pixel checker becomes mid-grey,
    # where sampling picks whole rows of black or white (aliasing)
    pm = c.pix.pixmap().toImage()
    vals = [pm.pixelColor(x, y).red() for x in range(50, 60) for y in range(50, 60)]
    assert 30 < np.mean(vals) < 100 and np.std(vals) < 40, (np.mean(vals), np.std(vals))
    c.close()


# ---------------------------------------------------------------- result pictures
def _result():
    return {"verdict": "FAIL", "total_ms": 12.0, "tools": [
        {"tool_id": "inductor_01", "type": "presence", "status": "fail", "score": 0.2,
         "graphics": [{"kind": "rect", "points": [(20, 20), (60, 60)], "status": "fail",
                       "label": "inductor_01 0.20"}]}]}


def test_result_pictures_are_saved_annotated_and_original(tmp_path):
    from vision_station.core import result_images as ri
    img = np.full((200, 300, 3), 90, np.uint8)
    cfg = {"folder": str(tmp_path), "which": "all", "annotated": True, "original": True,
           "format": "png", "per_day": True, "labels": True, "banner": True}
    paths = ri.write(cfg, img, _result(), "Top side")
    assert len(paths) == 2
    ann = cv2.imread(paths[0])
    orig = cv2.imread(paths[1])
    assert "REJECT" in os.path.basename(paths[0]) and "Top_side" in paths[0]
    assert np.array_equal(orig, img), "the original picture was altered"
    assert ann.shape[0] > img.shape[0], "no verdict banner above the picture"
    assert ann.shape[1] == img.shape[1], "the annotated picture was resized"
    body = ann[ann.shape[0] - 200:]
    assert (np.abs(body.astype(int) - img.astype(int)).sum(axis=2) > 0).any(), "nothing drawn"

    assert ri.wanted({"folder": "x", "which": "fail"}, "FAIL")
    assert not ri.wanted({"folder": "x", "which": "fail"}, "PASS")
    assert not ri.wanted({"folder": "", "which": "all"}, "FAIL")
    assert not ri.wanted({"folder": "x", "which": "all", "annotated": False,
                          "original": False}, "FAIL")


def test_the_station_saves_each_result_picture_in_the_background(tmp_path):
    from vision_station.core import result_images as ri
    w = ri.ResultImageWriter()
    w.configure({"folder": str(tmp_path), "which": "fail", "annotated": True,
                 "format": "jpg", "quality": 95, "per_day": False})
    img = np.full((120, 160, 3), 50, np.uint8)
    ok = dict(_result(), verdict="PASS")
    assert not w.submit(img, ok, "job")              # accepted parts not asked for
    assert w.submit(img, _result(), "job")
    w.flush()
    day = ri.day_folder()
    assert os.listdir(str(tmp_path)) == [day], "pictures not in a folder for the day"
    files = os.listdir(os.path.join(str(tmp_path), day))
    assert len(files) == 1 and files[0].endswith("_job_REJECT.jpg"), files


# ---------------------------------------------------------------- find parts
def test_groups_and_details_are_told_apart_by_confidence():
    from vision_station.core import part_finder as pf
    # three parts in a row, a weaker box drawn round all three (a group)
    parts = np.float32([[10, 10, 40, 40], [50, 10, 80, 40], [90, 10, 120, 40]])
    group = np.float32([[5, 5, 125, 45]])
    boxes = np.vstack([parts, group])
    keep = pf.drop_groups_and_duplicates(boxes, np.float32([0.6, 0.7, 0.65, 0.3]))
    assert sorted(keep.tolist()) == [0, 1, 2], "the group box was kept"
    # one part (an SFP cage) with weaker details on it: the part stays
    cage = np.float32([[0, 0, 100, 300]])
    details = np.float32([[10, 10, 40, 60], [50, 10, 90, 60], [10, 200, 90, 280]])
    keep = pf.drop_groups_and_duplicates(np.vstack([cage, details]),
                                         np.float32([0.9, 0.2, 0.25, 0.3]))
    assert keep.tolist() == [0], "a part's details were kept as parts"


def test_find_all_like_this_finds_every_copy_turned_or_not():
    from vision_station.core import part_finder as pf
    rng = np.random.default_rng(3)
    img = np.full((600, 900, 3), (60, 110, 50), np.uint8)       # board green
    part = rng.integers(0, 255, (40, 70, 3)).astype(np.uint8)
    spots = [(50, 50), (200, 60), (400, 300), (700, 450)]
    for x, y in spots:
        img[y:y + 40, x:x + 70] = part
    turned = np.ascontiguousarray(np.rot90(part))
    img[100:170, 600:640] = turned
    found = pf.find_similar(img, [50, 50, 70, 40], min_score=0.8)
    assert len(found) == 5, found
    assert any(f["box"][2] == 40 and f["box"][3] == 70 for f in found), "turned copy missed"


def test_found_parts_become_presence_checks_named_by_type():
    from vision_station.core import part_finder as pf
    parts = [{"box": [10, 10, 40, 30], "score": 0.9, "type": 0},
             {"box": [100, 10, 40, 30], "score": 0.8, "type": 0},
             {"box": [5, 200, 300, 90], "score": 0.7, "type": 1}]
    tools = pf.presence_tools(parts, {0: "inductor"}, ["inductor_01"], image_size=(400, 300))
    assert [t["id"] for t in tools] == ["inductor_02", "inductor_03", "part_b_01"]
    assert all(t["type"] == "presence" for t in tools)
    x, y, w, h = tools[0]["params"]["roi"]
    assert x < 10 and y < 10 and w > 40 and h > 30, "no margin round the part"
    x, y, w, h = tools[2]["params"]["roi"]
    assert x >= 0 and x + w <= 400 and y + h <= 300, "box left the picture"


needs_model = pytest.mark.skipif(
    not os.path.exists(os.path.join(HERE, "models", "parts", "parts_yoloe26s.onnx")),
    reason="part model not installed")


@needs_model
def test_the_network_finds_parts_on_a_synthetic_board(monkeypatch):
    # the general model: one trained for a real board (parts_trained.onnx,
    # preferred when present) rightly finds nothing on a drawn one
    from vision_station.core import part_finder as pf
    monkeypatch.setattr(pf, "MODEL", pf.GENERAL)
    pf._NET.clear()
    img = np.full((900, 1400, 3), (40, 95, 35), np.uint8)
    rng = np.random.default_rng(0)
    placed = []
    for i in range(6):
        x, y = 100 + i * 200, 150 + (i % 2) * 300
        cv2.rectangle(img, (x, y), (x + 90, y + 90), (25, 25, 25), -1)        # black IC
        cv2.putText(img, "U%d" % i, (x + 15, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (200, 200, 200), 2)
        for k in range(6):                                                    # its pins
            cv2.rectangle(img, (x + 8 + k * 14, y + 90), (x + 14 + k * 14, y + 102),
                          (190, 190, 200), -1)
        placed.append((x, y, 90, 102))
    r = pf.find_parts(img, None, 0.6)
    assert r["device"] in ("GPU", "CPU")
    hits = 0
    for x, y, w, h in placed:
        for p in r["parts"]:
            bx, by, bw, bh = p["box"]
            ix = max(0, min(x + w, bx + bw) - max(x, bx))
            iy = max(0, min(y + h, by + bh) - max(y, by))
            if ix * iy > 0.5 * w * h:
                hits += 1
                break
    pf._NET.clear()
    assert hits >= 4, (hits, r["parts"])
    # refiltering reuses the network's output
    r2 = pf.find_parts(img, None, 0.2, raw=r["candidates"])
    assert len(r2["parts"]) <= len(r["parts"])


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------- rotation
def test_a_camera_mounted_upside_down_is_turned_on_the_sensor():
    """Rotation 180 = both mirror axes on the sensor (no cost), combined with
    any mirroring asked for; 90 / 270 are turned in software."""
    import fake_pylon as F
    scene = np.zeros((40, 60), np.float32)
    scene[:10, :20] = 200                            # a bright corner, top-left
    dev = F.Device("gige_classic", ip="10.0.0.9", scene=scene)
    F, cam = _fresh_camera([dev])
    s = cam.CameraSettings()
    s.address, s.rotation = "10.0.0.9", 180
    t = cam.CameraThread(0, s)
    t.start()
    try:
        assert _wait(lambda: t.get_frame() is not None), t.last_error
        n = dev.nodes
        assert n["ReverseX"].value is True and n["ReverseY"].value is True
        assert t._pylon_cam.turn_code is None, "a half turn was done in software too"

        s2 = s.copy()
        s2.reverse_x = True                          # mirrored as well: X flips back
        t.update_settings(s2)
        assert _wait(lambda: n["ReverseX"].value is False and n["ReverseY"].value is True)

        s3 = s.copy()
        s3.rotation = 90
        g = t.update_settings(s3)
        f = t.frame_after(g, 2, 5)
        assert f is not None and f.shape[:2] == (60, 40), "a quarter turn did not turn the frame"
        assert n["ReverseX"].value is False and n["ReverseY"].value is False
    finally:
        t.stop()


def test_a_half_turn_without_mirror_nodes_is_done_in_software():
    from vision_station.hardware import camera
    import cv2 as _cv2
    assert camera.turn_code(180, done_on_sensor=False) == _cv2.ROTATE_180
    assert camera.turn_code(180, done_on_sensor=True) is None
    assert camera.turn_code(90) == _cv2.ROTATE_90_CLOCKWISE
    assert camera.turn_code(270) == _cv2.ROTATE_90_COUNTERCLOCKWISE
    assert camera.turn_code(0) is None


def test_rotation_is_kept_in_settings(qapp, tmp_path):
    from PySide6.QtCore import QSettings
    from vision_station.hardware import camera
    from vision_station.ui import settings_dialog as sd
    qs = QSettings(str(tmp_path / "r.ini"), QSettings.IniFormat)
    s = camera.CameraSettings()
    s.rotation = 180
    sd.save_camera_settings(qs, 0, s)
    assert sd.load_camera_settings(qs, 0).rotation == 180
    dlg = sd.SettingsDialog(None, qs)
    assert dlg.cameraPage.settings().rotation == 180
    dlg.cameraPage.rotation.setCurrentIndex(dlg.cameraPage.rotation.findData(270))
    assert dlg.apply()
    assert sd.load_camera_settings(qs, 0).rotation == 270
    for key in ("general", "camera", "results", "line", "security"):
        assert dlg.show_page(key) is dlg.stack.currentWidget()
    dlg.close()


# ---------------------------------------------------------------- results in words
def test_every_check_says_its_result_in_words():
    from vision_station.core import annotate
    ocv_ng = {"type": "ocv", "status": "fail", "values": {"text": "AB12345X", "lines": ["AB12345X"]}}
    assert annotate.caption(ocv_ng, {"expect_text": "AB123456"}) == \
        ["NG  read: AB12345X", "expected: AB123456"]
    ocv_ok = {"type": "ocv", "status": "pass", "values": {"text": "AB123456"}}
    assert annotate.caption(ocv_ok, {"expect_text": "AB123456"}) == ["AB123456"]
    m = {"type": "measure_length", "status": "pass", "values": {"length_px": 280.1, "length_mm": 25.0}}
    assert annotate.caption(m) == ["OK  25 mm"]
    m_px = {"type": "measure_length", "status": "pass", "values": {"length_px": 280.12}}
    assert annotate.caption(m_px) == ["OK  280.1 px"]
    assert annotate.caption({"type": "presence", "status": "fail", "values": {}}) == ["NG"]
    assert annotate.caption({"type": "count_blob", "status": "fail",
                             "values": {"count": 3, "expected": 4}}) == ["NG  3 / 4"]
    c = {"type": "find_circle", "status": "pass", "values": {"radius_px": 10.0, "diameter_mm": 3.2}}
    assert annotate.caption(c) == ["OK  Ø 3.2 mm"]


def test_a_round_region_is_outlined_round_and_ok_is_blue():
    import cv2
    from vision_station.core.engine import Project
    from vision_station.core import annotate
    demo = os.path.join(HERE, "projects", "demo_led_panel")
    p = Project(demo)
    p.spec["tools"] = [t for t in p.spec["tools"] if t["type"] == "align"] + [
        {"id": "screw_round", "type": "presence",
         "params": {"roi": [40, 290, 46, 46], "shape": "ellipse", "min_score": 0.5}}]
    p.tools = p.build_tools()
    img = cv2.imread(os.path.join(demo, "good.png"))
    res = p.inspect(img)
    t = [x for x in res["tools"] if x["tool_id"] == "screw_round"][0]
    region = [g for g in t["graphics"] if g.get("role") == "region"]
    assert len(region) == 1 and region[0]["kind"] == "poly" and len(region[0]["points"]) > 20, \
        "a round region was not outlined round"
    assert t["caption"] == ["OK"]
    out = annotate.draw(img, res, banner=False)
    blue = annotate.colour_bgr("pass")
    near = np.abs(out.astype(int) - np.array(blue)).sum(axis=2) < 40
    assert near.sum() > 50, "the OK outline is not blue"


def test_an_area_drawn_first_becomes_the_next_checks_region(qapp, tmp_path):
    import shutil
    from PySide6.QtCore import QSettings
    from vision_station.ui import main_window as V
    shutil.copytree(os.path.join(HERE, "projects", "demo_led_panel"),
                    str(tmp_path / "projects" / "demo_led_panel"),
                    ignore=shutil.ignore_patterns("results", "dataset"))
    ini = str(tmp_path / "s.ini")
    orig = V.QuickBuild._settings
    V.QuickBuild._settings = lambda self: QSettings(ini, QSettings.IniFormat)
    try:
        qb = V.QuickBuild(str(tmp_path / "projects"))
        ed = qb.open_job_editor(qb.station.jobs[0]["pid"])
        ed.tree.setCurrentItem(None)
        ed._canvas_drawn("ellipse", {"roi": [100, 120, 60, 40]})
        assert ed._pending_area is not None
        tid = ed.add_tool("presence")
        params = ed._tool(tid)["params"]
        assert params["roi"] == [100, 120, 60, 40] and params["shape"] == "ellipse"
        assert ed._pending_area is None
        qb.dirty.clear()
        qb.close()
    finally:
        V.QuickBuild._settings = orig


def test_the_camera_keeps_its_address_after_a_restart():
    import fake_pylon as F
    dev = F.Device("gige_classic", ip="192.168.3.6")
    F, cam = _fresh_camera([dev])
    pc = cam.PylonCamera(0, "192.168.3.6")
    pc.open()
    ip = pc.ip_config()
    assert ip["current_mask"] == "255.255.255.0" and ip["stored_mask"] == "255.255.0.0"
    assert not ip["keeps_current"], "a mismatched stored mask was not noticed"
    ip = pc.store_ip()
    assert ip["keeps_current"] and ip["stored_mask"] == "255.255.255.0" and ip["stored_on"]
    assert dev.nodes["GevCurrentIPConfiguration"].value & 0x1
    pc.close()


# ---------------------------------------------------------------- finding text
def _editor(tmp_path):
    import shutil
    from PySide6.QtCore import QSettings
    from vision_station.ui import main_window as V
    shutil.copytree(os.path.join(HERE, "projects", "demo_led_panel"),
                    str(tmp_path / "projects" / "demo_led_panel"),
                    ignore=shutil.ignore_patterns("results", "dataset"))
    ini = str(tmp_path / "s.ini")
    V.QuickBuild._settings = lambda self: QSettings(ini, QSettings.IniFormat)
    qb = V.QuickBuild(str(tmp_path / "projects"))
    qb.resize(1400, 900)
    qb.show()
    ed = qb.open_job_editor(qb.station.jobs[0]["pid"])
    return qb, ed


def _drag(canvas, start, end):
    """A real mouse drag on the canvas, from scene point to scene point."""
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    vp = canvas.viewport()
    a = canvas.mapFromScene(QPointF(*start))
    b = canvas.mapFromScene(QPointF(*end))
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, a)
    for k in range(1, 9):
        p = QPoint(int(a.x() + (b.x() - a.x()) * k / 8), int(a.y() + (b.y() - a.y()) * k / 8))
        QTest.mouseMove(vp, p)
        QApplication.processEvents()
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, b)
    QApplication.processEvents()


from PySide6.QtCore import Qt  # noqa: E402


@pytest.mark.skipif(not __import__("vision_station.tools.ocr_engine", fromlist=["x"]).onnx_available(),
                    reason="text reader models not installed")
def test_text_found_in_an_area_becomes_verify_text_checks(qapp, tmp_path):
    import cv2
    from PySide6.QtWidgets import QApplication
    from vision_station.ui import main_window as V
    orig = V.QuickBuild._settings
    try:
        qb, ed = _editor(tmp_path)
        before = len(ed._tools())
        panel = ed.find_parts()
        panel.set_mode("ocv")
        assert panel.findBtn.text() == "Find text"
        panel.set_area([100, 200, 340, 120])
        panel.find(wait=True)
        QApplication.processEvents()
        assert [p["text"] for p in panel.parts] == ["AB123456"], panel.parts
        x, y, w, h = panel.parts[0]["box"]
        # resize the text box from its bottom-right corner with the mouse
        _drag(panel.canvas, (x + w, y + h), (x + w + 12, y + h + 6))
        nx, ny, nw, nh = panel.parts[0]["box"]
        assert nw > w and nh > h, "dragging the corner did not resize the box"
        assert panel.parts[0]["text"] == "AB123456", "the box was not read again"
        # the expected text and the check's name, from the list
        panel.table.item(0, 2).setText("lot_code")
        new = panel.add_checks()
        assert new == ["lot_code_01"]
        t = ed._tool("lot_code_01")
        assert t["type"] == "ocv" and t["params"]["expect_text"] == "AB123456"
        assert len(ed._tools()) == before + 1
        assert panel.isVisible(), "the panel closed after adding"
        p = qb.station.projects[qb.station.jobs[0]["pid"]]
        p.tools = p.build_tools()
        demo = os.path.join(HERE, "projects", "demo_led_panel")
        ok = {r["tool_id"]: r for r in p.inspect(cv2.imread(os.path.join(demo, "good.png")))["tools"]}
        ng = {r["tool_id"]: r for r in p.inspect(cv2.imread(os.path.join(demo, "bad.png")))["tools"]}
        assert ok["lot_code_01"]["status"] == "pass", ok["lot_code_01"]["message"]
        assert ng["lot_code_01"]["status"] == "fail"
        assert ng["lot_code_01"]["caption"] == ["NG  read: AB12345X", "expected: AB123456"]
        panel.close()
        qb.dirty.clear()
        qb.close()
    finally:
        V.QuickBuild._settings = orig


@pytest.mark.skipif(not os.path.exists(os.path.join(HERE, "models", "parts", "parts_yoloe26s.onnx")),
                    reason="part model not installed")
def test_a_found_part_box_can_be_moved_before_it_becomes_a_check(qapp, tmp_path):
    from PySide6.QtWidgets import QApplication
    from vision_station.ui import main_window as V
    orig = V.QuickBuild._settings
    try:
        qb, ed = _editor(tmp_path)
        panel = ed.find_parts()
        panel.find(wait=True)
        QApplication.processEvents()
        assert panel.parts, "nothing found on the demo"
        x, y, w, h = panel.parts[0]["box"]
        _drag(panel.canvas, (x + w / 2, y + h / 2), (x + w / 2 + 15, y + h / 2 + 10))
        nx, ny, nw, nh = panel.parts[0]["box"]
        # moved by the drag (to the pixel: screen-to-picture rounding), same size
        assert abs(nx - x - 15) <= 1 and abs(ny - y - 10) <= 1 and (nw, nh) == (w, h), \
            panel.parts[0]["box"]
        panel._keep_all(False)
        panel.toggle_part(0)
        new = panel.add_checks()
        assert len(new) == 1 and ed._tool(new[0])["type"] == "presence"
        roi = ed._tool(new[0])["params"]["roi"]
        assert roi[0] < nx and roi[1] < ny, "the check did not take the moved box"
        panel.close()
        qb.dirty.clear()
        qb.close()
    finally:
        V.QuickBuild._settings = orig


# ---------------------------------------------------------------- result colours and lines
@pytest.fixture
def factory_look():
    from vision_station.core import annotate
    annotate.set_look()
    yield annotate
    annotate.set_look()


def test_result_look_is_solid_never_shared_and_reaches_pictures_and_screen(qapp, factory_look):
    import cv2
    A = factory_look
    from vision_station.ui import canvas, modern
    assert A.solid_hex("#80ff0000") is None and A.solid_hex("0f0") == "#00FF00"
    assert A.set_look("#FFB900", "#10893E", 2.0)
    assert A.colour_hex("pass") == "#FFB900" and A.colour_hex("fail") == "#10893E"
    assert canvas.PASS.name().upper() == "#FFB900" and canvas.FAIL.name().upper() == "#10893E"
    assert canvas.RESULT_W == A.DEFAULT_LINE_SCREEN_PX * 2.0
    assert modern._PILL["PASS"][1] == "#FFB900"
    # the same colour for both: NG falls back so the two still differ
    A.set_look("#FF2A2A", "#FF2A2A")
    assert A.colour_hex("pass") != A.colour_hex("fail")
    # nonsense falls back to the factory look
    A.set_look("pink", None, "x")
    assert (A.STYLE["ok"], A.STYLE["ng"]) == (A.DEFAULT_OK, A.DEFAULT_NG)
    assert A.STYLE["line_picture"] == A.DEFAULT_LINE_PICTURE
    # thickness reaches a saved picture; a light banner gets dark words
    img = np.full((400, 600, 3), 128, np.uint8)
    raw = {"verdict": "PASS", "total_ms": 1.0, "tools": [
        {"tool_id": "t", "status": "pass", "caption": [],
         "graphics": [{"kind": "rect", "role": "region", "points": [[100, 100], [300, 300]]}]}]}
    def ink_pixels(scale):
        A.set_look("#FFB900", None, scale)
        out = A.draw(img, raw, banner=False, labels=False)
        return (np.abs(out.astype(int) - np.array(A.colour_bgr("pass"))).sum(axis=2) < 30).sum()
    assert ink_pixels(3.0) > 2 * ink_pixels(1.0)
    out = A.draw(img, raw, banner=True)
    assert (out[: out.shape[0] - 400] < 40).all(axis=2).any(), "no dark words on a yellow banner"


def test_writer_configure_applies_the_look(factory_look):
    from vision_station.core.result_images import ResultImageWriter
    w = ResultImageWriter()
    w.configure({"ok_colour": "#0078D7", "ng_colour": "#E81123", "line_scale": 1.5})
    assert factory_look.STYLE["ok"] == "#0078D7" and factory_look.STYLE["ng"] == "#E81123"
    assert factory_look.STYLE["line_screen_px"] == factory_look.DEFAULT_LINE_SCREEN_PX * 1.5


def test_accent_picker_interlock_greys_out_the_other_colour(qapp):
    from vision_station.ui.accent_picker import AccentPicker, WINDOWS_COLOURS, interlock, set_pair
    ok, ng = AccentPicker("#1F6FFF", name="OK"), AccentPicker("#FF2A2A", name="NG")
    interlock(ok, ng)
    assert len(ok.swatches) == 48
    yellow = [s for s in ok.swatches if s.colour == "#FFB900"][0]
    yellow.click()
    assert ok.colour() == "#FFB900" and yellow.isChecked()
    ng_yellow = [s for s in ng.swatches if s.colour == "#FFB900"][0]
    assert not ng_yellow.isEnabled() and "OK" in ng_yellow.toolTip()
    ng_yellow.click()                                      # greyed out: nothing happens
    assert ng.colour() == "#FF2A2A"
    assert not ng.set_colour("#FFB900")
    # choosing another OK colour frees yellow for NG
    [s for s in ok.swatches if s.colour == "#0078D7"][0].click()
    assert ng_yellow.isEnabled()
    ng_yellow.click()
    assert ng.colour() == "#FFB900"
    assert not [s for s in ok.swatches if s.colour == "#FFB900"][0].isEnabled()
    # saved colours swapped load; the same colour twice cannot
    set_pair(ok, ng, "#FFB900", "#0078D7", "#FF2A2A")
    assert (ok.colour(), ng.colour()) == ("#FFB900", "#0078D7")
    set_pair(ok, ng, "#107C10", "#107C10", "#FF2A2A")
    assert ok.colour() == "#107C10" and ng.colour() == "#FF2A2A"
    # recent colours do not pile up layout stretches
    for c in WINDOWS_COLOURS[:10]:
        if c != ng.colour():
            ok.set_colour(c)
    assert ok.recentRow.count() <= 7


def test_results_page_saves_and_loads_colours_and_thickness(qapp, tmp_path, factory_look):
    from PySide6.QtCore import QSettings
    from vision_station.ui import settings_dialog as S
    qs = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    got = []
    dlg = S.SettingsDialog(qs=qs, open_camera=lambda *a, **k: None, on_results=got.append,
                           page="results")
    p = dlg.resultsPage
    assert p.values()["ok_colour"] == factory_look.DEFAULT_OK
    [s for s in p.okColour.swatches if s.colour == "#00CC6A"][0].click()
    p.lineScale.setCurrentIndex(p.lineScale.findData(2.0))
    d = dlg.apply_results()
    assert d["ok_colour"] == "#00CC6A" and d["line_scale"] == 2.0 and got
    loaded = S.load_result_settings(qs)
    assert loaded["ok_colour"] == "#00CC6A" and loaded["ng_colour"] == factory_look.DEFAULT_NG
    dlg2 = S.SettingsDialog(qs=qs, open_camera=lambda *a, **k: None, page="results")
    assert dlg2.resultsPage.okColour.colour() == "#00CC6A"
    assert dlg2.resultsPage.lineScale.currentData() == 2.0
    dlg2.resultsPage._reset_look()
    assert dlg2.resultsPage.values()["ok_colour"] == factory_look.DEFAULT_OK
    assert dlg2.resultsPage.values()["line_scale"] == 1.0
    dlg.close(); dlg2.close()


def test_every_settings_page_restores_its_defaults_and_keeps_where_things_are(
        qapp, tmp_path, factory_look, monkeypatch):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox
    from vision_station.ui import settings_dialog as S
    qs = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    dlg = S.SettingsDialog(qs=qs, open_camera=lambda *a, **k: None, page="general")
    for key, page in dlg._pages.items():
        assert page.restoreBtn.text() == "Restore defaults", key
        assert not page.restoreBtn.autoDefault(), key
    # asked first: No changes nothing
    cam = dlg.cameraPage
    cam.address.setText("192.168.3.6")
    cam.exposure.setValue(77.0)
    cam.revX.setChecked(True)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    assert not cam.ask_restore()
    assert cam.exposure.value() == 77.0
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    assert cam.ask_restore()
    d = S.CameraSettings()
    assert cam.exposure.value() == float(d.exposure_ms) and not cam.revX.isChecked()
    assert cam.address.text() == "192.168.3.6"
    # results: everything back, the folder kept
    res = dlg.resultsPage
    res.folder.setText(str(tmp_path / "pics"))
    res.which.setCurrentIndex(res.which.findData("fail"))
    res.quality.setValue(80)
    [s for s in res.okColour.swatches if s.colour == "#00CC6A"][0].click()
    res.lineScale.setCurrentIndex(res.lineScale.findData(3.0))
    res.restore_defaults()
    v = res.values()
    assert all(v[k] == S.RESULT_DEFAULTS[k] for k in v if k != "folder"), v
    assert v["folder"] == str(tmp_path / "pics")
    # barcode: rules back, the scanner's connection kept
    from vision_station.core import barcode as rules
    bc = dlg.barcodePage
    bc.form.prefix.setText("FIC")
    bc.form.length.setValue(11)
    before = bc.values()
    bc.restore_defaults()
    after = bc.values()
    assert after["prefix"] == rules.DEFAULTS["prefix"] and after["length"] == rules.DEFAULTS["length"]
    assert (after["source"], after["port"], after["baud"]) == \
        (before["source"], before["port"], before["baud"])
    # light: levels and timing back, the port kept
    from vision_station.hardware import light as L
    lp = dlg.lightPage
    lp.show_values(dict(L.DEFAULTS, port="COM7", channels=2, on_before_ms=999,
                        target_percent=80, auto=True))
    lp.restore_defaults()
    lv = lp.values()
    assert lv["port"] == "COM7" and lv["channels"] == 2
    assert lv["on_before_ms"] == L.DEFAULTS["on_before_ms"]
    assert lv["target_percent"] == L.DEFAULTS["target_percent"] and not lv["auto"]
    # without the main window, pages with nothing to restore say so
    assert not dlg.securityPage.restoreBtn.isEnabled()
    assert dlg.linePage.restore_defaults() is False
    dlg.close()
