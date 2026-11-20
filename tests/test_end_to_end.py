"""The whole application, A to Z and back from Z to A, on fake hardware: a
camera whose picture follows a light controller on a fake serial port.

A to Z: an empty folder -> Settings (light, results) -> a new application
whose first picture is the AI reference image -> a check comparing with the
scanned barcode -> saved and reopened -> barcode runs, good and wrong ->
pictures in the day's folder, found by code -> a continuous run -> a second
job from a file -> export and import.

Z to A: the job removed -> the barcode check from Settings only -> the
light handed back -> closed -> started again, and everything is as it was
left.

Then each way a step can fail: cancelled, no camera picture, light not
controlled, an unreadable file, a light controller that stops answering.

    python -m pytest tests/test_end_to_end.py -q
"""
import glob
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from test_light import FakeController, opener_for              # noqa: E402

CODE = "FIC12345678"


# ---------------------------------------------------------------- the fake line
def _qr(text, size):
    q = cv2.QRCodeEncoder.create().encode(text)
    return cv2.resize(q, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255


class Line:
    """The camera, the light and what is under them. The picture is the
    board lit by the light: dark when it is off, brighter with its level."""

    def __init__(self):
        self.dev = FakeController(level=9)
        base = np.full((480, 640), 0.55, np.float32)
        base[30:250, 30:250] = 1.0
        base[40:240, 40:240] = _qr(CODE, 200)
        base[300:380, 300:420] = 0.15                      # a dark part
        base[100:180, 400:560] = 0.9                       # a bright part
        base[400:440, 60:200] = 0.3
        self.base = base
        self.grabs = []                                    # (light on, level) per picture
        self.camera_ok = True
        self.lamp = 1.0                                    # 0.7: the lamp has aged
        self.sensor = False      # True: the raw sensor's colours under the white light
        self.gains = [1.0, 1.0, 1.0]                       # the camera's B, G, R balance

    def picture(self):
        lv = self.dev.level[1] if self.dev.on[1] else 0
        gain = min(1.0, 0.03 + self.lamp * lv / 180.0)
        img = np.clip(self.base * gain * 255, 0, 255).astype(np.uint8)
        self.grabs.append((self.dev.on[1], self.dev.level[1]))
        f = img.astype(np.float32)
        # measured on the line, gains at 1: the white light reads blue 41.7,
        # green 100.4, red 22.4 -- strongly green until white is balanced
        raw = [f * 0.42, f, f * 0.22] if (self.sensor and self.dev.on[1]) else [f, f, f]
        return cv2.merge([np.clip(c * g, 0, 255).astype(np.uint8)
                          for c, g in zip(raw, self.gains)])

    def balance_once(self):
        """What the camera's Once does: every colour to the level of green,
        under whatever light is on -- under the green light, red and blue
        raised many times."""
        img = self.picture()
        m = img[::8, ::8].reshape(-1, 3).mean(0) / np.array(self.gains)
        self.gains = [float(min(15.98, m[1] / max(v, 0.5))) for v in m]


class _FakePylon:
    """The camera's own settings, as the camera thread's calls see them."""

    def __init__(self, line):
        self.line = line

        class _Node:
            @staticmethod
            def GetValue():
                return "Off"

        class _Cam:
            BalanceWhiteAuto = _Node
        self.cam = _Cam

    def set_neutral_balance(self):
        self.line.gains = [1.0, 1.0, 1.0]
        return True

    def white_balance_once(self):
        self.line.balance_once()
        return True


class FakeCam:
    def __init__(self, line):
        self.line = line
        self.backend = "pylon" if line.camera_ok else None
        self.last_error = "" if line.camera_ok else "no camera at 192.168.3.6"
        self.fresh_frames = 3
        self.info = {}
        self.stopped = False

    def is_alive(self):
        return self.line.camera_ok and not self.stopped

    def frame_after(self, gen=None, settle=3, timeout_s=3.0):
        return self.line.picture() if self.line.camera_ok else None

    def get_frame(self):
        return self.frame_after()

    def call(self, fn, timeout_s=5.0):
        return fn(_FakePylon(self.line))

    def stop(self):
        self.stopped = True


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def world(qapp, tmp_path_factory):
    """One line, one settings file and one folder of applications for the
    whole A to Z and Z to A."""
    from PySide6.QtCore import QSettings
    from vision_station.core import light_tuning
    from vision_station.ui import main_window as V
    root = tmp_path_factory.mktemp("e2e")
    line = Line()
    said = []
    mp = pytest.MonkeyPatch()
    ini = str(root / "settings.ini")
    mp.setattr(V.QuickBuild, "_settings", lambda self: QSettings(ini, QSettings.IniFormat))
    mp.setattr(V, "camera_settings_store", lambda: QSettings(ini, QSettings.IniFormat))
    mp.setattr(V, "open_camera", lambda i: FakeCam(line))
    for kind in ("warning", "information", "critical"):
        mp.setattr(V.QMessageBox, kind, staticmethod(
            lambda *a, kind=kind, **k: said.append((kind, a[2] if len(a) > 2 else ""))))
    mp.setattr(V.QMessageBox, "question", staticmethod(lambda *a, **k: V.QMessageBox.No))
    # the AI, quickly: it sees most at a medium level
    mp.setattr(light_tuning, "ai_stats", lambda img: {
        "parts": max(1.0, 40 - abs(line.dev.level[1] - 110) / 4.0), "parts_found": 30,
        "text": 200.0})
    from vision_station.hardware import light
    orig_init = light.Light.__init__

    def init(self, opener=None):
        orig_init(self, opener or opener_for(line.dev))
    mp.setattr(light.Light, "__init__", init)
    w = {"root": str(root), "apps": str(root / "apps"), "ini": ini, "line": line,
         "said": said, "V": V, "mp": mp, "results": str(root / "pictures")}
    os.makedirs(w["apps"])
    yield w
    qb = w.get("qb")
    if qb is not None:
        qb.dirty.clear()
        qb.close()
    mp.undo()


def _pump(qapp, s=0.05):
    t0 = time.time()
    while time.time() - t0 < s:
        qapp.processEvents()
        time.sleep(0.005)


def _start(world):
    w = world["V"].QuickBuild(world["apps"])
    world["qb"] = w
    return w


def _job(qb):
    return qb.station.jobs[0]["pid"]


# ================================================================ A to Z
def test_a01_an_empty_folder_starts(world):
    qb = _start(world)
    assert qb.station.applications() == []
    assert qb.station.jobs == []


def test_a02_settings_light_and_pictures(world, qapp):
    from vision_station.ui.settings_dialog import save_result_settings
    qb, line = world["qb"], world["line"]
    qb.apply_light({"enabled": True, "port": "COM7", "always_on": False,
                    "on_before_ms": 0, "off_after_ms": 0})
    assert not line.dev.on[1], "the light is on with Always on off"
    assert qb.station.light.found == {1: 9}
    cfg = {"folder": world["results"], "which": "all", "annotated": True, "format": "jpg"}
    save_result_settings(qb._settings(), cfg)
    qb.station.result_images.configure(cfg)


def test_a03_a_new_application_from_the_ai_reference_image(world, qapp):
    from vision_station.core import light_tuning
    qb, line = world["qb"], world["line"]
    line.grabs.clear()
    app = qb.new_application("Line A", source="camera")
    assert app == "line_a" and qb.station.app_id == "line_a"
    pid = _job(qb)
    p = qb.station.projects[pid]
    assert p.spec["image_source"]["mode"] == "camera"
    assert p.spec["light_level"] in light_tuning.LEVELS
    tried = [lv for _on, lv in line.grabs]
    assert all(lv in tried for lv in light_tuning.LEVELS), "a level was not tried: %s" % tried
    assert len(line.grabs) <= len(light_tuning.LEVELS) + 4, "too many pictures: %s" % tried
    assert line.grabs[-1] == (True, p.spec["light_level"]), "not taken at the level chosen"
    assert all(on for on, _lv in line.grabs), "a picture was taken in the dark"
    assert not line.dev.on[1], "the light was left on"
    assert qb.station.sources[pid].mode == "camera"
    assert "Light level" in qb.statusBar().currentMessage()
    world["level"] = p.spec["light_level"]


def test_a04_a_check_comparing_with_the_scanned_code_goes_after_the_fixture(world, qapp):
    qb = world["qb"]
    pid = _job(qb)
    ed = qb.open_job_editor(pid)
    ed.add_tool("presence")
    ed.add_tool("code_read")
    p = qb.station.projects[pid]
    qr = next(t for t in p.spec["tools"] if t["type"] == "code_read")
    qr["params"]["roi"] = [20, 20, 240, 240]
    qr["params"]["compare_barcode"] = True
    ed._props_changed()
    ids = [t["id"] for t in p.spec["tools"]]
    assert ids == ["fixture", qr["id"], "presence_1"], ids
    pres = next(t for t in p.spec["tools"] if t["id"] == "presence_1")
    pres["params"]["roi"] = [290, 290, 140, 100]
    ed._props_changed()
    world["qr_id"] = qr["id"]


def test_a05_saved_and_opened_again(world, qapp):
    qb = world["qb"]
    assert qb.save_all() and not qb.dirty
    qb.open_application("line_a")
    p = qb.station.projects[_job(qb)]
    assert [t["id"] for t in p.spec["tools"]] == ["fixture", world["qr_id"], "presence_1"]
    assert p.spec["light_level"] == world["level"]
    assert next(t for t in p.spec["tools"] if t["id"] == world["qr_id"])["params"][
        "compare_barcode"] is True


def test_a06_a_scanned_part_carrying_its_code(world, qapp):
    from vision_station.core.result_images import day_folder
    qb, line = world["qb"], world["line"]
    qb.apply_barcode({"enabled": True, "prefix": "FIC", "length": 11, "auto_run": True,
                      "folder": world["results"]})
    line.grabs.clear()
    assert qb.barcode_scanned(CODE)
    qb.station.wait(60)
    res = qb.station.last[_job(qb)]
    tools = {t["tool_id"]: t for t in res["tools"]}
    assert tools[world["qr_id"]]["values"]["barcode_match"] is True, tools[world["qr_id"]]
    assert res["verdict"] == "PASS", [(t["tool_id"], t["status"], t["message"])
                                      for t in res["tools"]]
    assert line.grabs == [(True, world["level"])], "not lit at the job's level"
    assert not line.dev.on[1]
    qb.station.result_images.flush()
    pics = glob.glob(os.path.join(world["results"], day_folder(), CODE, "*.jpg"))
    assert pics, "no picture in today's folder, in the code's folder"
    dlg = qb.find_picture()
    for _ in range(200):
        _pump(qapp, 0.01)
        if not dlg.index.scanning and len(dlg.index):
            break
    dlg.query.setText(CODE[-6:])
    _pump(qapp, 0.05)
    assert dlg.model.rows and dlg.model.rows[0].code == CODE
    dlg.close()


def test_a07_a_scanned_code_the_part_does_not_carry(world, qapp):
    from vision_station.core.result_images import day_folder
    qb = world["qb"]
    assert not qb.barcode_scanned("ABC")                 # not even a valid code
    assert not qb.station.running
    assert qb.barcode_scanned("FIC00000000")
    qb.station.wait(60)
    res = qb.station.last[_job(qb)]
    assert res["verdict"] == "FAIL" and res["barcode_ng"].startswith("Barcode not good")
    assert [t["tool_id"] for t in res["tools"]] == ["fixture", world["qr_id"]]
    _pump(qapp, 0.05)
    assert "Barcode not good" in qb.barcodeBar.state.text()
    qb.station.result_images.flush()
    assert glob.glob(os.path.join(world["results"], day_folder(), "FIC00000000", "*.jpg"))


def test_a08_a_continuous_run_keeps_the_light_on(world, qapp):
    qb, line = world["qb"], world["line"]
    qb.apply_barcode({"enabled": False})
    qb.station.start_continuous()
    _pump(qapp, 0.4)
    assert line.dev.on[1], "the light went out between the parts"
    n = sum(qb.station.counts.values())
    qb.station.stop()
    qb.station.wait(60)
    assert not line.dev.on[1]
    assert n >= 1


def test_a09_a_second_job_from_a_file(world, qapp):
    qb, line = world["qb"], world["line"]
    path = os.path.join(world["root"], "side_b.png")
    cv2.imwrite(path, line.picture())
    sid = qb.add_job("Side B", image_path=path)
    assert sid and len(qb.station.jobs) == 2
    b = qb.station.projects[sid]
    assert "light_level" not in b.spec and qb.station.sources[sid].mode == "reference"
    line.grabs.clear()
    qb.station.run_once(gate=False)
    qb.station.wait(60)
    assert len(line.grabs) == 1, "the light came on for a job reading a stored picture"
    assert set(qb.station.last) >= {_job(qb), sid}
    world["side_b"] = sid


def test_a10_export_and_import(world, qapp):
    qb = world["qb"]
    vsp = os.path.join(world["root"], "line_a.vsp")
    assert qb.export_package(vsp) and os.path.exists(vsp)
    app = qb.import_package(vsp)
    assert app and app != "line_a"
    qb.open_application(app)
    assert len(qb.station.jobs) == 2
    levels = [qb.station.projects[j["pid"]].spec.get("light_level") for j in qb.station.jobs]
    assert world["level"] in levels, "the job's light level was not carried"
    qb.open_application("line_a")


# ================================================================ Z to A
def test_z01_the_second_job_removed(world, qapp):
    qb = world["qb"]
    assert qb.remove_job(world["side_b"], confirm=False) is not False
    assert [j["pid"] for j in qb.station.jobs] == [_job(qb)]


def test_z02_barcode_from_settings_only_checks_the_scan_and_nothing_more(world, qapp):
    qb = world["qb"]
    pid = _job(qb)
    p = qb.station.projects[pid]
    qr = next(t for t in p.spec["tools"] if t["id"] == world["qr_id"])
    qr["params"]["compare_barcode"] = False
    qb.apply_barcode({"enabled": True, "prefix": "FIC", "length": 11, "auto_run": True})
    assert not qb.barcode_scanned("XYZ"), "an invalid code started an inspection"
    assert qb.barcode_scanned("FIC00000000")             # valid, but not the part's
    qb.station.wait(60)
    res = qb.station.last[pid]
    assert not res["barcode_ng"] and res["verdict"] == "PASS", \
        "compared with the tool's box off"
    assert len(res["tools"]) == 3
    qr["params"]["compare_barcode"] = True
    qb.apply_barcode({"enabled": False})


def test_z03_the_light_handed_back_when_control_is_switched_off(world, qapp):
    qb, line = world["qb"], world["line"]
    qb.apply_light(dict(qb.station.light.cfg, enabled=False))
    assert line.dev.on[1] and line.dev.level[1] == 9, "not handed back as found"


def test_z04_closed_and_started_again(world, qapp):
    qb, line = world["qb"], world["line"]
    qb.apply_light(dict(qb.station.light.cfg, enabled=True))
    assert not line.dev.on[1]
    qb.save_all()
    qb.dirty.clear()
    qb.close()
    world["qb"] = None
    assert line.dev.on[1] and line.dev.level[1] == 9, "closing did not hand the light back"
    qb = _start(world)
    assert {a["id"] for a in qb.station.applications()} >= {"line_a"}
    qb.open_application("line_a")
    p = qb.station.projects[_job(qb)]
    assert p.spec["light_level"] == world["level"]
    assert next(t for t in p.spec["tools"] if t["id"] == world["qr_id"])["params"][
        "compare_barcode"] is True
    assert qb.station.light.enabled and not line.dev.on[1], "light settings not kept"


# ================================================================ when a step fails
@pytest.fixture
def fresh(world, qapp):
    """A started application on the same line, each test on its own."""
    qb = world.get("qb") or _start(world)
    world["qb"] = qb
    world["said"].clear()
    world["line"].camera_ok = True
    yield qb
    world["line"].camera_ok = True


def _apps(world):
    return sorted(d for d in os.listdir(world["apps"])
                  if os.path.isdir(os.path.join(world["apps"], d)))


def test_cancelling_the_choice_creates_nothing(world, fresh, monkeypatch):
    before = _apps(world)
    monkeypatch.setattr(fresh, "choose_reference_source", lambda title: None)
    assert fresh.new_application("Never") is None
    assert _apps(world) == before


def test_no_picture_from_the_camera_creates_nothing_and_says_why(world, fresh):
    before = _apps(world)
    world["line"].camera_ok = False
    assert fresh.new_application("No camera", source="camera") is None
    assert _apps(world) == before
    assert any("camera" in msg for _k, msg in world["said"]), world["said"]
    assert not world["line"].dev.on[1]


def test_without_light_control_the_picture_is_taken_as_it_is(world, fresh):
    lt = fresh.station.light
    cfg = dict(lt.cfg)
    fresh.apply_light(dict(cfg, enabled=False))
    try:
        app = fresh.new_application("Plain", source="camera")
        assert app
        p = fresh.station.projects[_job(fresh)]
        assert p.spec["image_source"]["mode"] == "camera"
        assert "light_level" not in p.spec
        assert "does not control the light" in fresh.statusBar().currentMessage()
    finally:
        fresh.apply_light(cfg)
        fresh.open_application("line_a")


def test_an_unreadable_file_creates_nothing(world, fresh):
    before = _apps(world)
    bad = os.path.join(world["root"], "not_a_picture.png")
    with open(bad, "w") as f:
        f.write("text")
    assert fresh.new_application("Bad file", image_path=bad) is None
    assert _apps(world) == before
    assert any("Could not read" in msg for _k, msg in world["said"])


def test_a_light_controller_that_stops_answering_never_stops_a_run(world, fresh, qapp):
    line = world["line"]
    fresh.open_application("line_a")
    line.dev.dead = True
    try:
        line.grabs.clear()
        fresh.station.run_once(gate=False)
        fresh.station.wait(60)
        assert line.grabs, "no picture was taken"
        assert fresh.station.light.error, "the controller's silence was not reported"
    finally:
        line.dev.dead = False
        fresh.station.light._next_try = 0
    fresh.station.run_once(gate=False)
    fresh.station.wait(60)
    assert not fresh.station.light.error, "did not reconnect"
    assert not line.dev.on[1]


def test_the_good_reference_menu_no_longer_offers_the_ai_picture(world, fresh):
    fresh.open_application("line_a")
    ed = fresh.open_job_editor(_job(fresh))
    texts = [a.text() for a in ed._ref_menu.actions()]
    assert not any("AI" in t for t in texts), texts


# ================================================================ more of the way through
def test_the_choice_is_made_in_the_dialog_itself(world, fresh, qapp):
    """The real question box: the AI reference image button pressed."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QPushButton
    seen = {}

    def press():
        box = QApplication.activeModalWidget()
        if box is None:
            QTimer.singleShot(50, press)
            return
        buttons = box.findChildren(QPushButton)
        # every choice readable: no button narrower than its text
        seen["cut"] = [b.text() for b in buttons if b.width() < b.sizeHint().width()]
        next(b for b in buttons if "AI reference" in b.text()).click()
    QTimer.singleShot(50, press)
    app = fresh.new_application("Dialog line")
    try:
        assert app, "nothing created from the dialog"
        assert seen.get("cut") == [], "buttons cut short: %s" % seen.get("cut")
        assert fresh.station.projects[_job(fresh)].spec["image_source"]["mode"] == "camera"
    finally:
        fresh.open_application("line_a")


def test_keep_adjusted_follows_the_lamp_and_survives_a_restart(world, fresh, qapp):
    line = world["line"]
    fresh.open_application("line_a")
    pid = _job(fresh)
    lt = fresh.station.light
    fresh.apply_light(dict(lt.cfg, auto=True, target="reference"))
    line.lamp = 0.7                                        # the lamp has aged
    try:
        for _ in range(6):
            fresh.station.run_once(gate=False)
            fresh.station.wait(60)
        learnt = lt.cfg["job_levels"].get(pid)
        assert learnt and learnt > world["level"], "the light was not raised: %s" % learnt
        fresh.dirty.clear()
        fresh.close()
        world["qb"] = None
        qb = _start(world)
        qb.open_application("line_a")
        assert qb.station.light.cfg["job_levels"].get(pid) == learnt, "lost on restart"
        line.grabs.clear()
        qb.station.run_once(gate=False)
        qb.station.wait(60)
        assert line.grabs[-1][1] == learnt, "the run did not use the learnt level"
    finally:
        line.lamp = 1.0
        qb = world["qb"]
        qb.apply_light(dict(qb.station.light.cfg, auto=False, job_levels={}))


def test_opening_another_application_stops_a_continuous_run(world, fresh, qapp):
    line = world["line"]
    fresh.open_application("line_a")
    fresh.station.start_continuous(gate=False)
    _pump(qapp, 0.3)
    assert line.dev.on[1]
    other = next(a["id"] for a in fresh.station.applications() if a["id"] != "line_a")
    fresh.open_application(other)
    fresh.station.wait(60)
    _pump(qapp, 0.2)
    assert not fresh.station.continuous, "still running after another application opened"
    assert not line.dev.on[1], "the light stayed on for an application no longer open"
    fresh.open_application("line_a")


def test_a_camera_lost_in_a_continuous_run(world, fresh, qapp):
    line = world["line"]
    fresh.open_application("line_a")
    errors = []
    fresh.station.runError.connect(lambda pid, msg: errors.append(msg))
    fresh.station.start_continuous(gate=False)
    _pump(qapp, 0.2)
    line.camera_ok = False
    _pump(qapp, 0.6)
    fresh.station.stop()
    fresh.station.wait(60)
    _pump(qapp, 0.3)
    assert any("no image" in e for e in errors), errors
    assert not line.dev.on[1], "the light stayed on after Stop"
    line.camera_ok = True




def test_under_a_white_light_white_is_balanced_once(world, fresh, qapp):
    from vision_station.ui.settings_dialog import load_camera_settings
    line = world["line"]
    line.gains = [0.6, 1.0, 1.7]                        # balanced for some other light
    try:
        app = fresh.new_application("White line", source="camera")
        assert app
        p = fresh.station.projects[_job(fresh)]
        assert not p.spec["image_source"].get("mono"), "a white light taken in black and white"
        m = p.reference[::8, ::8].reshape(-1, 3).mean(0)
        assert max(m) - min(m) < 3, "white not balanced: %s" % m
        cs = load_camera_settings(fresh._settings(), 0)
        assert not cs.neutral_balance and not cs.auto_white_balance
    finally:
        line.gains = [1.0, 1.0, 1.0]
        fresh.open_application("line_a")



def test_a_white_light_the_sensor_shows_green_gives_a_colour_picture(world, fresh, qapp):
    """The camera left with its colour gains at 1 shows the white light as
    green. The AI reference image balances white under the lit light and
    takes a true colour picture -- never black and white."""
    from vision_station.ui.settings_dialog import load_camera_settings
    line = world["line"]
    line.sensor = True
    line.gains = [1.0, 1.0, 1.0]
    try:
        app = fresh.new_application("Colour line", source="camera")
        assert app
        p = fresh.station.projects[_job(fresh)]
        assert not p.spec["image_source"].get("mono"), "taken in black and white"
        m = p.reference[::8, ::8].reshape(-1, 3).mean(0)
        assert max(m) - min(m) < 4, "not balanced - green or pink: %s" % m
        assert line.gains[2] > 3 and line.gains[0] > 2, "white not balanced: %s" % line.gains
        cs = load_camera_settings(fresh._settings(), 0)
        assert not cs.neutral_balance and not cs.auto_white_balance, "balance not held"
        fresh.station.run_once(gate=False)
        fresh.station.wait(60)
        r = fresh.station.last[_job(fresh)]["input_image"][::8, ::8].reshape(-1, 3).mean(0)
        assert max(r) - min(r) < 4, "a run's picture is not in true colour: %s" % r
    finally:
        line.sensor = False
        line.gains = [1.0, 1.0, 1.0]
        fresh.open_application("line_a")


def test_the_cameras_automatic_exposure_is_held_while_the_light_is_controlled(world, fresh):
    """Automatic exposure and gain undo every change of the light: with the
    light controlled they are switched off, at the camera's manual values."""
    from vision_station.hardware.camera import CameraSettings
    from vision_station.ui.settings_dialog import load_camera_settings, save_camera_settings
    s = CameraSettings()
    s.address, s.exposure_ms, s.gain_db = "192.168.3.6", 28.0, 37.0
    s.auto_exposure = s.auto_gain = True
    save_camera_settings(fresh._settings(), 0, s)
    lt = fresh.station.light
    fresh.apply_light(dict(lt.cfg, enabled=False))
    assert load_camera_settings(fresh._settings(), 0).auto_exposure, \
        "changed with the light not controlled"
    fresh.apply_light(dict(lt.cfg, enabled=True))
    got = load_camera_settings(fresh._settings(), 0)
    assert not got.auto_exposure and not got.auto_gain
    assert (got.exposure_ms, got.gain_db) == (28.0, 37.0), "the manual values changed"
    assert "automatic exposure and gain switched off" in fresh.statusBar().currentMessage()
    assert fresh.hold_exposure() is False, "switched off twice"
