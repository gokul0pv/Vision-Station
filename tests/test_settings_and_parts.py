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
    files = os.listdir(str(tmp_path))
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
def test_the_network_finds_parts_on_a_synthetic_board():
    from vision_station.core import part_finder as pf
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
