"""The external light on its controller's serial port: the protocol, when it
is on, automatic brightness, and the Light page in Settings.

    python -m pytest tests/test_light.py -q
"""
import os
import shutil
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vision_station.hardware import light as L            # noqa: E402


class FakeController:
    """A HIFLY DCS2 on a serial port, as measured: '$' for a setting, '&'
    for a bad frame, the frame with the value for a read; on/off keep the
    brightness; a brightness sent while off leaves the light off."""

    def __init__(self, channels=1, level=9):
        self.on = {ch: True for ch in range(1, channels + 1)}
        self.level = {ch: level for ch in range(1, channels + 1)}
        self.sent = []
        self.out = b""
        self.closed = False
        self.dead = False

    # what serial.Serial offers
    def reset_input_buffer(self):
        self.out = b""

    def flush(self):
        pass

    def close(self):
        self.closed = True

    def write(self, data):
        if self.dead:
            return len(data)
        f = data.decode()
        self.sent.append(f)
        cmd, ch, value = f[1], int(f[2]), int(f[3:6], 16)
        if L.frame(cmd, ch, value).decode() != f or ch not in self.on:
            self.out += b"&"
        elif cmd == "1":
            self.on[ch] = True
            self.out += b"$"
        elif cmd == "2":
            self.on[ch] = False
            self.out += b"$"
        elif cmd == "3":
            self.level[ch] = value
            self.out += b"$"
        elif cmd == "4":
            self.out += L.frame("4", ch, self.level[ch])
        else:
            self.out += b""                      # this model does not answer
        return len(data)

    def read(self, n):
        got, self.out = self.out[:n], self.out[n:]
        return got

    def lit(self, ch=1):
        return self.on[ch] and self.level[ch] > 0


def opener_for(dev):
    def opener(port, timeout):
        dev.closed = False
        return dev
    return opener


# ---------------------------------------------------------------- protocol
def test_frames_are_the_manuals():
    assert L.frame("3", 1, 100) == b"$3106414"          # channel 1 to 100
    assert L.frame("4", 2, 0x64) == b"$4206410"         # read channel 2
    assert L.frame("1", 3, 0x64) == b"$1306414"         # channel 3 on
    assert L.frame("2", 2, 0x29).lower() == b"$220291f"  # channel 2 off
    assert L.frame("4", 1) == b"$4100011"                # what the DCS2 answered
    assert L.parse_read(b"$4100918", "4", 1) == 9        # its answer: brightness 9
    assert L.parse_read(b"$4100919", "4", 1) is None, "a bad checksum was believed"


def test_the_controller_switches_and_dims():
    dev = FakeController()
    c = L.LightController("COM9", opener_for(dev))
    assert c.brightness(1) == 9
    c.off(1)
    assert not dev.on[1]
    c.set_brightness(1, 300)
    assert dev.level[1] == 255, "brightness is 0-255"
    c.on(1)
    assert dev.lit()
    with pytest.raises(L.LightError):
        c.on(3)                                          # refused: '&'
    dev.dead = True
    with pytest.raises(L.LightError, match="no answer"):
        c.off(1)


def test_find_port_reads_only():
    dev = FakeController()

    def opener(port, timeout):
        if port != "COM7":
            raise OSError("nothing there")
        return dev
    assert L.find_port(["COM1", "COM7"], opener) == "COM7"
    assert all(f[1] == "4" for f in dev.sent), "finding the port switched something"
    assert L.find_port(["COM1"], opener) is None


# ---------------------------------------------------------------- when it is on
def _light(dev, **cfg):
    lt = L.Light(opener_for(dev))
    lt.configure(dict({"enabled": True, "port": "COM7"}, **cfg))
    return lt


def test_always_on_and_on_only_while_held():
    dev = FakeController()
    lt = _light(dev, always_on=True)
    assert dev.lit() and lt.is_on
    lt.configure(dict(lt.cfg, always_on=False))
    assert not dev.on[1], "off when not always on and nothing runs"
    lt.hold("run")
    assert dev.lit(), "not on for the run's picture"
    lt.hold("live")
    lt.release("run")
    assert dev.lit(), "Live Camera still open: must stay on"
    lt.release("live")
    assert not dev.on[1]


def test_the_controller_is_handed_back_as_found():
    dev = FakeController(level=9)
    lt = _light(dev, always_on=False, set_brightness=True, brightness=[200, 0, 0, 0])
    assert dev.level[1] == 200 and not dev.on[1]
    lt.shutdown()
    assert dev.level[1] == 9 and dev.on[1], "not back to its own brightness, light on"
    assert dev.closed
    lt.shutdown()                                        # twice is harmless


def test_leaving_the_brightness_alone_never_sends_one():
    dev = FakeController(level=9)
    lt = _light(dev, always_on=True, set_brightness=False)
    lt.hold("run")
    lt.release("run")
    lt.shutdown()
    assert not [f for f in dev.sent if f[1] == "3"], "a brightness was sent"
    assert dev.level[1] == 9


def test_a_dead_controller_never_stops_a_run():
    dev = FakeController()
    lt = _light(dev, always_on=False)
    dev.dead = True
    lt.hold("run")                                       # must not raise
    assert lt.error and lt.ctl is None
    lt.release("run")
    dev.dead = False
    lt._next_try = 0
    lt.hold("run")
    assert dev.lit() and not lt.error, "did not reconnect"


def test_a_brightness_while_off_leaves_it_off():
    dev = FakeController()
    lt = _light(dev, always_on=False)
    lt.set_level(120)
    assert dev.level[1] == 120 and not dev.on[1]
    assert lt.level() == 120 and lt.cfg["set_brightness"]
    lt.set_channel(1, 40)
    assert dev.level[1] == 40 and not dev.on[1]


# ---------------------------------------------------------------- automatic
def _picture(grey, clipped=0.0):
    img = np.full((200, 200, 3), int(round(grey * 255)), np.uint8)
    n = int(clipped * 200)
    if n:
        img[:n] = 255
    return img


class Scene:
    """A camera looking at a part lit by the fake light: grey rises with the
    level (a little ambient light, saturating at white)."""

    def __init__(self, dev, gain=0.0012, ambient=0.01):
        self.dev, self.gain, self.ambient = dev, gain, ambient

    def grab(self):
        lv = self.dev.level[1] if self.dev.on[1] else 0
        g = self.ambient + self.gain * lv
        return _picture(min(1.0, g), clipped=max(0.0, min(1.0, (g - 0.9) * 4)))


def test_measure_and_target():
    m, c = L.measure(_picture(0.4))
    assert abs(m - 0.4) < 0.01 and c == 0
    assert L.measure(_picture(0.4, clipped=0.5))[1] == pytest.approx(0.5, abs=0.02)
    assert L.target_of({"target": "reference"}, _picture(0.33)) == pytest.approx(0.33, abs=0.01)
    assert L.target_of({"target": "fixed", "target_percent": 60}) == 0.6
    assert L.target_of({"target": "reference", "target_percent": 45}, None) == 0.45


def test_the_next_level_moves_toward_the_target_without_hunting():
    assert L.next_level(100, 0.40, 0.0, 0.40, 0.005) == 100          # on target
    assert L.next_level(100, 0.41, 0.0, 0.40, 0.005) == 100          # in the dead band
    up = L.next_level(100, 0.20, 0.0, 0.40, 0.005)
    assert 100 < up <= 135, "too dark: up, at most 35 % at once"
    assert L.next_level(100, 0.80, 0.0, 0.40, 0.005) < 100
    assert L.next_level(100, 0.30, 0.02, 0.40, 0.005) < 100, "burnt out: always down"
    assert L.next_level(255, 0.10, 0.0, 0.40, 0.005) == 255
    assert L.next_level(1, 0.9, 0.0, 0.40, 0.005) == 1


def test_adjust_now_finds_the_level_from_the_camera():
    dev = FakeController(level=9)
    lt = _light(dev, always_on=False)
    scene = Scene(dev)
    level, mean, clipped, tries = lt.adjust_now(scene.grab, _picture(0.25))
    assert abs(mean - 0.25) < 0.01 and clipped == 0
    assert dev.level[1] == level and len(tries) <= 9
    assert not dev.on[1], "the light was left on after adjusting"
    # a target it cannot reach: brightest, said so by the mean
    lt.cfg.update(target="fixed", target_percent=95)
    level, mean, *_ = lt.adjust_now(Scene(dev, gain=0.001).grab)
    assert level >= 254 and mean < 0.3


def test_keep_adjusted_follows_a_dimming_light():
    dev = FakeController(level=150)
    lt = _light(dev, always_on=True, set_brightness=True, brightness=[150] * 4, auto=True)
    ref = _picture(0.19)                                  # the teach picture
    scene = Scene(dev)
    scene.gain *= 0.7                                     # the lamp has aged
    for _ in range(8):
        lt.auto_step(scene.grab(), ref)
    assert abs(L.measure(scene.grab())[0] - 0.19) < 0.02, "did not come back to the teach picture"
    changes = sum(1 for f in dev.sent if f[1] == "3")
    for _ in range(5):
        assert lt.auto_step(scene.grab(), ref) is None, "hunting on target"
    assert sum(1 for f in dev.sent if f[1] == "3") == changes
    lt.cfg["auto"] = False
    assert lt.auto_step(_picture(0.01), ref) is None, "adjusted with Keep adjusted off"


# ---------------------------------------------------------------- the application
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def qb(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from vision_station.ui import main_window as V
    shutil.copytree(os.path.join(HERE, "projects", "demo_led_panel"),
                    str(tmp_path / "projects" / "demo_led_panel"),
                    ignore=shutil.ignore_patterns("results", "dataset"))
    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(V.QuickBuild, "_settings",
                        lambda self: QSettings(ini, QSettings.IniFormat))
    w = V.QuickBuild(str(tmp_path / "projects"))
    w.station.open("demo_led_panel")
    yield w
    w.dirty.clear()                      # no 'save changes?' when closing
    w.station.light.configure({"enabled": False})
    w.close()


def _pump(qapp, s=0.05):
    t0 = time.time()
    while time.time() - t0 < s:
        qapp.processEvents()
        time.sleep(0.005)


def test_a_run_lights_the_camera_picture_only(qb, qapp, monkeypatch):
    dev = FakeController()
    lt = qb.station.light
    lt._opener = opener_for(dev)
    qb.apply_light({"enabled": True, "port": "COM7", "always_on": False,
                    "on_before_ms": 0, "off_after_ms": 0}, save=False)
    assert not dev.on[1]
    pid = qb.station.jobs[0]["pid"]
    src = qb.station.sources[pid]
    seen = []
    ref = qb.station.projects[pid].reference

    def grab():
        seen.append(dev.on[1])
        return ref.copy()
    monkeypatch.setattr(src, "grab", grab)
    src.mode = "camera"
    assert qb.station.run_once(gate=False)
    qb.station.wait(60)
    assert seen == [True], "the light was not on while the camera's picture was taken"
    assert not dev.on[1], "the light stayed on after the picture"
    src.mode = "reference"
    qb.station.run_once(gate=False)
    qb.station.wait(60)
    assert seen[-1] is False, "a run on a stored picture switched the light on"

    src.mode = "camera"
    qb.station.start_continuous(gate=False)
    _pump(qapp, 0.3)
    assert dev.on[1], "off between the parts of a continuous run"
    qb.station.stop()
    qb.station.wait(60)
    assert not dev.on[1], "on after Stop"

    qb.station.shutdown = lambda: None                    # closed by the fixture
    qb.station.light.shutdown()
    assert dev.on[1], "not handed back on when the application closes"


def test_the_light_page_round_trips_and_switches(qb, qapp):
    from vision_station.ui.settings_dialog import SettingsDialog
    dev = FakeController()
    qb.station.light._opener = opener_for(dev)
    dlg = qb.show_settings("light")
    assert isinstance(dlg, SettingsDialog) and dlg.current_page() == "light"
    page = dlg.lightPage
    page.enabled.setChecked(True)
    page.port.setEditText("COM7")
    page.alwaysOn.setChecked(False)
    page.setBright.setChecked(True)
    page.sliders[0].setValue(77)
    assert page.levels[0].value() == 77, "slider and box out of step"
    page.levels[0].setValue(88)
    assert page.sliders[0].value() == 88
    assert page.test(True) and dev.on[1]
    _pump(qapp, 0.3)                                      # the light follows the slider
    assert dev.level[1] == 88
    assert page.test(False) and not dev.on[1]
    assert page.read() == {1: 88}
    page.auto.setChecked(True)
    dlg._ok()
    from vision_station.hardware import light
    saved = light.load(qb._settings())
    assert saved["port"] == "COM7" and saved["auto"] and not saved["always_on"]
    assert saved["brightness"][0] == 88 and saved["set_brightness"]
    assert not dev.on[1], "Done did not put the rule back in charge"
    assert not page._level_rows[1].isVisible() or page.channels.value() > 1


def test_the_light_waits_before_and_stays_on_after_the_picture(qb, qapp, monkeypatch):
    dev = FakeController()
    lt = qb.station.light
    lt._opener = opener_for(dev)
    qb.apply_light({"enabled": True, "port": "COM7", "always_on": False,
                    "on_before_ms": 250, "off_after_ms": 400}, save=False)
    pid = qb.station.jobs[0]["pid"]
    src = qb.station.sources[pid]
    ref = qb.station.projects[pid].reference
    lit_for = []

    def grab():
        lit_for.append(time.perf_counter() - lt.on_since)
        return ref.copy()
    monkeypatch.setattr(src, "grab", grab)
    src.mode = "camera"
    t0 = time.perf_counter()
    assert qb.station.run_once(gate=False)
    assert lit_for[0] >= 0.24, "the picture was taken %.0f ms after the light came on" % (
        lit_for[0] * 1000)
    assert dev.on[1], "off the moment the picture was taken"
    while dev.on[1] and time.perf_counter() - t0 < 3:
        _pump(qapp, 0.02)
    assert not dev.on[1] and time.perf_counter() - t0 >= 0.6, "not kept on after the picture"
    qb.station.wait(60)

    # two runs close together: the first run's late release must not put out
    # the light under the second run's picture
    qb.apply_light(dict(lt.cfg, on_before_ms=0, off_after_ms=150), save=False)
    seen = []
    monkeypatch.setattr(src, "grab", lambda: (seen.append(dev.on[1]), ref.copy())[1])
    qb.station.run_once(gate=False)
    qb.station.wait(60)
    qb.station.run_once(gate=False)
    assert seen == [True, True]
    qb.station.wait(60)


def test_settle_waits_only_for_what_is_left():
    dev = FakeController()
    lt = _light(dev, always_on=False, on_before_ms=200)
    lt.hold("run")
    t = time.perf_counter()
    lt.settle()
    assert 0.15 < time.perf_counter() - t < 0.4
    t = time.perf_counter()
    assert lt.settle() == 0.0 and time.perf_counter() - t < 0.05, "waited twice"
    lt.release("run")
    assert lt.settle() == 0.0, "waited with the light off"


def test_a_shiny_teach_picture_does_not_turn_the_light_down():
    ref = _picture(0.43, clipped=0.06)                   # a real job's teach picture: 6 % white
    cfg = {"target": "reference", "max_clipped": 0.5}
    assert L.clip_limit(cfg, ref) > 0.07
    assert L.next_level(200, 0.43, 0.05, 0.43, L.clip_limit(cfg, ref)) == 200
    assert L.clip_limit({"target": "fixed", "max_clipped": 0.5}, ref) == 0.005


def test_a_switched_light_holds_the_cameras_white_balance(qb, qapp, monkeypatch):
    from vision_station.hardware.camera import CameraSettings
    from vision_station.ui import main_window as V
    from vision_station.ui.settings_dialog import load_camera_settings, save_camera_settings
    calls = []

    class Handle:
        backend = "pylon"

        def frame_after(self, *a, **k):
            calls.append("frame")
            return np.zeros((4, 4, 3), np.uint8)

        def call(self, fn):
            class PC:
                class cam:
                    class BalanceWhiteAuto:
                        @staticmethod
                        def GetValue():
                            return "Off"

                @staticmethod
                def white_balance_once():
                    calls.append("once")
                    return True

                @staticmethod
                def set_neutral_balance():
                    calls.append("neutral")
                    return True
            return fn(PC)

        def stop(self):
            calls.append("stop")
    monkeypatch.setattr(V, "open_camera", lambda i: Handle())
    monkeypatch.setattr(V, "apply_camera_settings", lambda i, s: calls.append("off") or True)
    s = CameraSettings()
    s.auto_white_balance = True
    save_camera_settings(qb._settings(), 0, s)
    dev = FakeController()
    qb.station.light._opener = opener_for(dev)
    qb.apply_light({"enabled": True, "port": "COM7", "always_on": True}, save=False)
    assert load_camera_settings(qb._settings(), 0).auto_white_balance,         "changed with the light always on"
    qb.apply_light({"enabled": True, "port": "COM7", "always_on": False,
                    "on_before_ms": 0}, save=False)
    t0 = time.time()
    while "stop" not in calls and time.time() - t0 < 5:
        _pump(qapp, 0.02)
    assert not load_camera_settings(qb._settings(), 0).auto_white_balance
    assert [c for c in calls if c != "frame"] == ["off", "once", "stop"], calls
    assert not dev.on[1], "the light stayed on after balancing"
    calls.clear()
    assert qb._keep_camera_balance() is False and not calls, "balanced again"


# ---------------------------------------------------------------- best light for a teach picture
def _board_rows():
    """The measures taken on the real board, level by level."""
    data = [(16, .09, 7.1, 7, .53, 224), (32, .16, 9.1, 39, .60, 231),
            (64, .24, 11.5, 41, .58, 237), (96, .31, 13.4, 42, .56, 223),
            (128, .35, 14.9, 38, .54, 233), (160, .40, 16.2, 35, .55, 224),
            (192, .43, 17.3, 30, .54, 239), (224, .47, 18.4, 34, .49, 242),
            (255, .49, 19.2, 32, .53, 242)]
    return [{"level": lv, "mean": m, "clip": 0.0, "dark": 0.057 if lv == 16 else 0.0,
             "detail": d, "parts": n * c, "parts_found": n, "text": float(t)}
            for lv, m, d, n, c, t in data]


def _nearest(by_level, lv):
    return by_level[min(by_level, key=lambda k: abs(k - lv))]


def test_measurement_alone_leaves_headroom():
    """The first measurements (taken, it turned out, of the desk, not the
    board): with no AI, detail alone rises to full power -- a level just
    below it that is nearly as good is taken, to leave room to raise the
    light as it ages."""
    from vision_station.core import light_tuning as T
    plain = T.score([dict(r, parts=None, text=None) for r in _board_rows()])
    assert T.choose(plain)["level"] == 224, "headroom: not full power when it ties"


def test_burnt_out_and_crushed_pictures_lose():
    from vision_station.core import light_tuning as T
    rows = [{"level": 100, "mean": .4, "clip": 0.0, "dark": 0.0, "detail": 10, "parts": 10,
             "text": 100},
            {"level": 200, "mean": .8, "clip": 0.25, "dark": 0.0, "detail": 12, "parts": 11,
             "text": 101}]                              # a quarter burnt out: over-exposed
    assert T.choose(T.score(rows))["level"] == 100


def test_find_best_level_tries_each_level_and_leaves_the_best(monkeypatch):
    from vision_station.core import light_tuning as T
    dev = FakeController(level=9)
    scene = Scene(dev, gain=0.0018)
    by_level = {r["level"]: r for r in _board_rows()}
    monkeypatch.setattr(T, "ai_stats", lambda img: {
        k: _nearest(by_level, dev.level[1])[k] for k in ("parts", "parts_found", "text")})
    lt = _light(dev, always_on=False, on_before_ms=0)
    seen = []
    result = lt.tune(scene.grab, progress=lambda i, n, row: seen.append(row["level"]))
    assert seen == list(T.LEVELS)
    assert 64 <= result["level"] <= 176 and dev.level[1] == result["level"] and result["ai"]
    assert not dev.on[1], "the light stayed on after tuning"
    assert "Light level %d" % result["level"] in T.describe(result)
    stops = iter([False, False, True])
    r = lt.tune(scene.grab, stop=lambda: next(stops, True))
    assert len(r["rows"]) == 2, "Stop was not heard"


def test_each_job_takes_its_picture_at_its_own_level():
    dev = FakeController()
    lt = _light(dev, always_on=False, set_brightness=True, brightness=[150] * 4)
    assert lt.use_job_level("a", taught=96) == 96 and dev.level[1] == 96
    assert lt.use_job_level("b") == 150, "a job with no level of its own: the setting"
    sent = len(dev.sent)
    assert lt.use_job_level("b") == 150 and len(dev.sent) == sent, "sent again unchanged"
    lt.cfg["auto"] = True
    ref = _picture(0.40)
    assert lt.auto_step(_picture(0.20), ref, pid="a", taught=96) > 96
    assert lt.job_level("a", 96) > 96 and lt.level() == 150, "the setting moved for a job"
    lt.configure(dict(lt.cfg, job_levels=None) if False else
                 {k: v for k, v in lt.cfg.items() if k != "job_levels"})
    assert lt.job_level("a", 96) > 96, "Settings' Apply lost what Keep adjusted learnt"
    lt.forget_job("a")
    assert lt.job_level("a", 96) == 96


def test_teaching_with_the_best_light(qb, qapp, monkeypatch):
    import json as _json
    from vision_station.core import light_tuning as T
    dev = FakeController(level=9)
    lt = qb.station.light
    lt._opener = opener_for(dev)
    qb.apply_light({"enabled": True, "port": "COM7", "always_on": False,
                    "on_before_ms": 0, "off_after_ms": 0}, save=False)
    pid = qb.station.jobs[0]["pid"]
    src = qb.station.sources[pid]
    ref = qb.station.projects[pid].reference
    src.mode = "camera"
    lit = []

    def grab():
        lit.append((dev.on[1], dev.level[1]))
        img = ref.copy()
        img[:] = np.uint8(min(255, dev.level[1]))
        return img
    monkeypatch.setattr(src, "grab", grab)
    by_level = {r["level"]: r for r in _board_rows()}
    monkeypatch.setattr(T, "ai_stats", lambda img: {
        k: _nearest(by_level, dev.level[1])[k] for k in ("parts", "parts_found", "text")})
    ed = qb.open_job_editor(pid) if hasattr(qb, "open_job_editor") else qb.edit_job(pid)
    got = ed.teach_with_best_light()
    assert got is not None and got["level"] is not None
    assert all(on for on, _lv in lit), "a picture was taken with the light off"
    level = ed._last_best_light["level"]
    assert lit[-1] == (True, level), "the teach picture was not taken at the chosen level"
    p = qb.station.projects[pid]
    assert p.spec["light_level"] == level
    with open(p.path, encoding="utf-8") as f:
        assert _json.load(f)["light_level"] == level, "not kept with the job"
    assert int(p.reference.mean()) == level, "the teach picture was not replaced"
    assert not dev.on[1]

    # every run of the job takes its picture at that level
    lt.send_level(200)
    lit.clear()
    qb.station.run_once(gate=False)
    qb.station.wait(60)
    assert lit == [(True, level)]


def test_a_green_light_burning_out_is_seen():
    from vision_station.core import light_tuning as T
    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :, 1] = 255                                     # green at white, red and blue empty
    mean, clipped = L.measure(img)
    assert mean < 0.6 and clipped > 0.99, "a saturated green was not burnt out"
    st = T.picture_stats(img)
    assert st["clip"] > 0.99 and st["dark"] == 0.0
    img[(np.arange(200) // 4) % 2 == 0, :, 1] = 100        # stripes, only in green
    assert T.picture_stats(img)["detail"] > 50




# ---------------------------------------------------------------- a coloured light


def test_shiny_parts_white_at_every_level_do_not_drag_the_light_down():
    """The real board under the green light: cages white at every level,
    and more of it burning out above 128."""
    from vision_station.core import light_tuning as T
    rows = [{"level": lv, "mean": m, "clip": c, "dark": 0.0, "detail": d, "parts": 30.0,
             "text": 200.0}
            for lv, m, c, d in ((32, .29, .035, 9), (64, .42, .057, 12), (96, .50, .070, 14),
                                (128, .56, .115, 15.5), (176, .63, .187, 16.5),
                                (224, .68, .230, 17.2), (255, .70, .253, 17.6))]
    best = T.choose(T.score(rows))["level"]
    assert 64 <= best <= 128, best          # where the job's 41 text checks read best




def _green_board_rows():
    """The real board, measured level by level (in black and white from
    the green channel, with the camera's gains at 1 -- before it turned out
    the light is white and the camera only unbalanced; the scores' shape is
    what the test needs)."""
    data = [(32, .19, 31.3, .000, .032, 80.9, 392.1), (64, .28, 39.8, .036, .005, 63.4, 530.0),
            (96, .35, 43.1, .046, 0, 67.4, 648.2), (128, .39, 45.2, .053, 0, 68.6, 680.6),
            (176, .45, 47.1, .065, 0, 70.4, 742.8), (224, .49, 48.2, .072, 0, 68.8, 692.3),
            (255, .52, 48.5, .079, 0, 71.1, 599.2)]
    return [{"level": lv, "mean": m, "detail": d, "clip": c, "dark": k, "parts": pt,
             "text": t} for lv, m, d, c, k, pt, t in data]


def test_the_real_green_board_gets_a_well_lit_level():
    from vision_station.core import light_tuning as T
    best = T.choose(T.score(_green_board_rows()))
    assert 128 <= best["level"] <= 224, best["level"]
    assert best["mean"] >= 0.35, "a dark teach picture: %.2f" % best["mean"]



def test_a_job_set_to_black_and_white_reads_one_colour():
    from vision_station.core import light_tuning as T
    img = np.zeros((20, 20, 3), np.uint8)
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = 40, 100, 20
    m = T.to_mono(img, "G")
    assert m.shape == img.shape and (m == 100).all()
    assert T.to_mono(img, None) is img
    assert not hasattr(T, "coloured_channel"), "a light's colour is guessed again"


# ---------------------------------------------------------------- handing back after a kill
class Store:
    """The light's settings as QSettings would keep them across sessions."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)

    def session(self, dev):
        lt = L.Light(opener_for(dev))
        lt.on_level.append(lambda _lv: self.cfg.update(
            {k: (dict(v) if isinstance(v, dict) else v) for k, v in lt.cfg.items()}))
        lt.configure(dict(self.cfg))
        return lt


def test_a_killed_session_does_not_change_what_is_handed_back():
    """The operator leaves the controller at 0. The application sets the
    job's level, is killed (VS Code's Stop), and the controller keeps 224.
    Started again and closed normally, the light goes back to 0 -- it was
    handed back at 224, light on."""
    dev = FakeController(level=0)
    store = Store({"enabled": True, "port": "COM7", "always_on": False})
    a = store.session(dev)
    assert store.cfg["in_use"] and store.cfg["home_levels"] == {"1": 0}
    a.send_level(224)
    a.hold("run")
    assert dev.lit() and dev.level[1] == 224
    # killed here: no shutdown -- the controller keeps 224 and the light on
    b = store.session(dev)
    assert b.found == {1: 0}, "took the killed session's 224 for the controller's own"
    b.shutdown()
    assert dev.level[1] == 0, "handed back at the killed session's level"
    assert not store.cfg["in_use"], "still owed after handing back"
    c = store.session(dev)
    assert c.found == {1: 0}
    c.shutdown()


def test_a_normal_close_reads_the_controller_afresh_next_time():
    """The operator changes the brightness on the panel between sessions:
    after a normal close, the next session takes the new setting."""
    dev = FakeController(level=0)
    store = Store({"enabled": True, "port": "COM7", "always_on": False})
    store.session(dev).shutdown()
    dev.level[1] = 80                                   # set on the panel
    s = store.session(dev)
    assert s.found == {1: 80}, "kept an old setting after a normal close"
    s.shutdown()
    assert dev.level[1] == 80


def test_settings_applied_keep_what_is_owed(qb, qapp):
    """Settings > Light's Apply must not drop the saved controller setting."""
    from vision_station.ui.settings_dialog import SettingsDialog
    from vision_station.hardware import light
    dev = FakeController(level=0)
    qb.station.light._opener = opener_for(dev)
    qb.apply_light({"enabled": True, "port": "COM7", "always_on": False})
    assert qb.station.light.cfg["in_use"]
    dlg = qb.show_settings("light")
    dlg._ok()
    saved = light.load(qb._settings())
    assert saved["in_use"] and saved["home_levels"] == {"1": 0}, saved


@pytest.fixture(autouse=True)
def _no_vision_next_note(tmp_path, monkeypatch):
    """Vision Next's own light note (vision_next/projects/light_home.json)
    never reaches these tests unless a test writes one."""
    monkeypatch.setenv("VS_NEXT_LIGHT_HOME", str(tmp_path / "next_light_home.json"))


def test_vision_next_stopped_without_handing_back(tmp_path, monkeypatch):
    """The operator leaves the controller at 0. Vision Next sets its level
    (128), is stopped without handing back, and the controller keeps 128.
    Vision Station must hand back 0 -- what Vision Next noted down before
    it changed anything -- not the 128 it finds."""
    import json
    note = tmp_path / "owed.json"
    note.write_text(json.dumps({"port": "COM7", "levels": [0, 255], "in_use": True}))
    monkeypatch.setenv("VS_NEXT_LIGHT_HOME", str(note))
    dev = FakeController(level=128)
    store = Store({"enabled": True, "port": "COM7", "always_on": False})
    s = store.session(dev)
    assert s.found == {1: 0}, "took Vision Next's leftover 128 for the operator's level"
    s.shutdown()
    assert dev.level[1] == 0 and dev.on[1]
    # a note for another port, or one handed back, is not owed here
    note.write_text(json.dumps({"port": "COM3", "levels": [0], "in_use": True}))
    dev.level[1] = 128
    assert store.session(dev).found == {1: 128}
