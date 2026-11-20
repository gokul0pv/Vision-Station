"""Advanced training (core/advanced.py, ui/advanced_training.py): a job's own
YOLO model, trained on its first Run from its own checks.

The training itself runs in the training environment on the graphics card;
here a fake worker (a script printing the same JSON lines) or a fake training
step stands in for it, so every way the session can go -- finished,
cancelled, failed, out of graphics memory -- is tried in seconds.

    python -m pytest tests/test_advanced.py -q
"""
import glob
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vision_station.core import advanced as A                       # noqa: E402
from vision_station.core.engine import Project                      # noqa: E402
from vision_station.core import imio                                # noqa: E402
from vision_station.tools.base import ToolResult, Graphic           # noqa: E402


# ---------------------------------------------------------------- a board
def board(dark_missing=False, shift=(0, 0)):
    img = np.full((900, 1400, 3), (40, 110, 40), np.uint8)
    rng = np.random.default_rng(3)
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    for i in range(6):
        x, y = 100 + i * 200, 150 + (i % 2) * 300
        if dark_missing and i == 2:
            continue
        cv2.rectangle(img, (x, y), (x + 90, y + 90), (25, 25, 25), -1)
        cv2.putText(img, "U%d" % i, (x + 15, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (200, 200, 200), 2)
    cv2.putText(img, "LOT 4711", (500, 800), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (230, 230, 230), 3)
    if shift != (0, 0):
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        img = cv2.warpAffine(img, M, (1400, 900), borderMode=cv2.BORDER_REPLICATE)
    return img


def job(tmp_path, tools=None, name="job"):
    folder = tmp_path / name
    folder.mkdir()
    imio.imwrite(str(folder / "reference.png"), board())
    if tools is None:
        tools = [{"id": "u%d" % i, "type": "presence",
                  "params": {"roi": [100 + i * 200 - 8, 150 + (i % 2) * 300 - 8, 106, 106],
                             "min_score": 0.65, "search_pad": 15}} for i in range(6)]
        tools.append({"id": "lot", "type": "ocv",
                      "params": {"roi": [490, 755, 300, 60], "expect_text": ""}})
    spec = {"name": name, "reference_image": "reference.png", "tools": tools,
            "advanced": {"requested": True, "trained": False}}
    with open(folder / "project.json", "w") as f:
        json.dump(spec, f)
    return Project(str(folder))


def grabber(img=None, fail=False):
    def grab():
        if fail:
            return None
        return (board() if img is None else img).copy()
    return grab


def fake_train(onnx_dir, steps=5, cancel_wait=False):
    """A training step: progress, then a dummy ONNX file."""
    def _train(self, data):
        self._set(train_started=time.time(), epochs=steps)
        for e in range(1, steps + 1):
            self._check()
            self._set(epoch=e)
            self._stage("train", e / float(steps), "pass %d of %d" % (e, steps))
            time.sleep(0.05 if not cancel_wait else 0.3)
        p = os.path.join(onnx_dir, "best.onnx")
        with open(p, "wb") as f:
            f.write(b"onnx")
        return p, {"precision": 0.99, "recall": 0.98, "map50": 0.97, "map": 0.8}
    return _train


def fake_validate(self, onnx, pics, fixtures, zoom, items):
    return {"learned": [it["id"] for it in items], "not_learned": [], "found_share": 1.0,
            "missing_caught": 5, "missing_tried": 5, "pictures": len(fixtures),
            "conf_min": {}}


@pytest.fixture
def faked(monkeypatch, tmp_path):
    d = tmp_path / "onnx"
    d.mkdir()
    monkeypatch.setattr(A.Session, "_train", fake_train(str(d)))
    monkeypatch.setattr(A.Session, "_validate", fake_validate)
    monkeypatch.setattr(A, "N_PICTURES", 3)
    return d


def run(sess, timeout=60):
    sess.start()
    assert sess.wait(timeout), "the session did not finish"
    return sess.snapshot()


# ---------------------------------------------------------------- what can be learned
def test_part_and_text_checks_are_learnable_and_the_rest_is_not():
    spec = {"tools": [
        {"id": "fx", "type": "align", "params": {"roi": [0, 0, 100, 100]}},
        {"id": "a", "type": "presence", "params": {"roi": [10, 10, 50, 40]}},
        {"id": "b", "type": "ocr", "params": {"roi": [10, 10, 50, 40], "angle": 30}},
        {"id": "c", "type": "ocv", "params": {"roi": [10, 10, 50, 40], "enabled": False}},
        {"id": "d", "type": "presence", "params": {}},
        {"id": "e", "type": "measure_distance", "params": {"roi": [1, 1, 5, 5]}},
        {"id": "f", "type": "presence", "params": {"roi": [5, 5, 30, 30], "capture": "dark"}},
        {"id": "g", "type": "presence", "params": {"shape": "polygon",
                                                   "points": [[10, 10], [60, 15], [30, 70]]}}]}
    items, skip = A.learnable(spec)
    got = {i["id"]: i for i in items}
    assert set(got) == {"a", "b", "g"}
    assert got["a"]["cls"] == A.CLASSES.index("part") and got["b"]["cls"] == A.CLASSES.index("text")
    # a turned region: its bounding box, wider than the box drawn
    assert got["b"]["ref"][3] > 40
    assert got["g"]["ref"] == pytest.approx((10, 10, 50, 60), abs=1)
    assert skip == {"c": "switched off", "d": "has no region", "f": "reads another picture"}


def test_regions_too_large_or_too_small_are_left_to_the_classic_test():
    items = [{"id": "x%d" % i, "ref": (0, 0, 60, 40)} for i in range(20)]
    items.append({"id": "huge", "ref": (0, 0, 3000, 2000)})
    items.append({"id": "tiny", "ref": (0, 0, 4, 4)})
    z, keep, why = A.choose_zoom(items)
    assert z == 1.0
    assert "huge" in why and "large" in why["huge"]
    assert "tiny" in why and "small" in why["tiny"]
    assert len(keep) == 20


def test_a_synthetic_copy_moves_its_boxes_with_it():
    img = board()
    rng = np.random.default_rng(0)
    v, M = A.variant(img, rng)
    assert v.shape == img.shape
    (c, x, y, w, h), = A.move_boxes([(0, 100, 150, 90, 90)], M, 1400, 900)
    cx, cy = M @ np.array([145.0, 195.0, 1.0])
    assert abs(x + w / 2 - cx) < 2 and abs(y + h / 2 - cy) < 2
    # a box pushed off the picture is dropped
    assert A.move_boxes([(0, -500, -500, 50, 50)], M, 1400, 900) == []


def test_tiles_hold_their_labels_and_a_big_region_gets_its_own_tile(tmp_path):
    img = np.zeros((3000, 4000, 3), np.uint8)
    boxes = [(0, 100, 100, 60, 60), (0, 1000, 500, 800, 300)]       # 800 px: cut by the grid
    oi, ol = tmp_path / "i", tmp_path / "l"
    oi.mkdir()
    ol.mkdir()
    n = A.write_tiles(img, boxes, 1.0, str(oi), str(ol), "t", np.random.default_rng(0), 0.0)
    assert n >= 2
    whole = 0
    for f in glob.glob(str(ol / "*.txt")):
        for ln in open(f).read().splitlines():
            c, x, y, w, h = map(float, ln.split())
            assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
            whole += abs(w * A.IMGSZ - 800) < 2
    assert whole >= 1, "the big region is in no tile whole"


# ---------------------------------------------------------------- the network at run time
class FakeDet(A.Detector):
    """A Detector whose network answers from a list of finds."""

    def __init__(self, finds, tools):
        super().__init__("", {"tools": tools, "zoom": 1.0, "imgsz": 640})
        self.finds = np.asarray(finds, np.float32).reshape(-1, 6)
        self.calls = 0

    def detect_tile(self, frame, tx, ty):
        self.calls += 1
        return self.finds


def test_the_fewest_tiles_cover_every_region_whole():
    det = FakeDet([], {})
    regs = {"r%d" % i: (50 + 37 * i, 40 + 23 * (i % 9), 30, 30) for i in range(60)}
    regs["far"] = (2900, 1900, 40, 40)
    plan = det.plan(regs, 3000, 2000)
    assert set(plan) == set(regs)
    for k, (tx, ty) in plan.items():
        assert det.contains((tx, ty), regs[k], 3000, 2000), k
    assert len(set(plan.values())) <= 6, "a tile per region: %d" % len(set(plan.values()))


def test_a_find_counts_within_the_checks_own_leeway_and_of_about_its_size():
    det = FakeDet([[120, 110, 170, 160, 0.9, 0],     # 20 px right, 10 down of the region
                   [400, 400, 410, 410, 0.95, 0]],   # far away and tiny
                  {"a": 0})
    frame = np.zeros((640, 640, 3), np.uint8)
    region = (100, 100, 50, 50)
    assert det.find(frame, region, 0, pad=15)[0] == pytest.approx(0.9)
    assert det.find(frame, region, 1, pad=15)[0] == 0.0             # another class
    # 20 px off: more than a quarter of the box, within a 10 px search on top
    assert det.find(frame, (100, 100, 50, 50), 0, pad=0)[0] == 0.0
    assert det.find(frame, (100, 100, 50, 50), 0, pad=10)[0] == pytest.approx(0.9)
    assert det.find(frame, (60, 60, 50, 50), 0, pad=0)[0] == 0.0     # 60 px off: not its part
    assert det.find(frame, (100, 100, 200, 200), 0, pad=50)[0] == 0.0  # a quarter of the size


def _result(status="pass", score=0.5):
    """A check's own result -- by default not sure of itself (0.5), so the
    AI may overrule it."""
    return ToolResult("u0", "presence", status, score, {}, [Graphic("rect", [(0, 0), (1, 1)],
                                                                    status, "")], "", 1.0)


def test_a_learned_check_passes_only_when_the_ai_finds_its_part(tmp_path):
    p = job(tmp_path)
    tool = p.tools[0]
    from vision_station.core.engine import Context
    ctx = Context(reference=p.reference)
    ctx.tools = {t.id: t for t in p.tools}
    x, y, w, h = tool.roi
    frame = board()
    p.advanced = FakeDet([[x, y, x + w, y + h, 0.8, 0]], {"u0": 0, "lot": 1})
    r = A.apply(p, tool, _result(), frame, ctx)
    assert r.status == "pass" and r.values["ai_found"] and r.values["ai_confidence"] == 0.8
    # the AI does not find it: the check fails, and says why
    p.advanced = FakeDet([], {"u0": 0})
    ctx.share.clear()
    r = A.apply(p, tool, _result(), frame, ctx)
    assert r.status == "fail" and "AI did not find the part" in r.message
    assert all(g.status == "fail" for g in r.graphics)
    # a failed check stays failed; an error is left alone; an unlearned check too
    assert A.apply(p, tool, _result("fail"), frame, ctx).status == "fail"
    assert A.apply(p, tool, _result("error"), frame, ctx).status == "error"
    p.advanced = FakeDet([], {"other": 0})
    assert A.apply(p, tool, _result(), frame, ctx).status == "pass"
    # switched off for this check
    p.advanced = FakeDet([], {"u0": 0})
    tool.p["ai"] = False
    assert A.apply(p, tool, _result(), frame, ctx).status == "pass"


def test_the_ai_looks_at_each_tile_once_per_run(tmp_path):
    p = job(tmp_path)
    det = FakeDet([], {t.id: 0 for t in p.tools})
    p.advanced = det
    p.inspect(board())
    regs = {t.id: A.region_in(t, A._Ctx()) for t in p.tools}
    tiles = len(set(det.plan(regs, 1400, 900).values()))
    assert det.calls == tiles < len(regs), "%d network runs for %d tiles, %d checks" % (
        det.calls, tiles, len(regs))
    p.inspect(board())
    assert det.calls == 2 * tiles, "the next run did not look again"


# ---------------------------------------------------------------- the session
def test_a_finished_training_installs_the_model_and_the_good_pictures(tmp_path, faked):
    p = job(tmp_path)
    st = run(A.Session(p, grabber()))
    assert st["ok"], st["error"]
    adv = os.path.join(p.folder, "advanced")
    assert sorted(os.listdir(adv)) == ["good", "model.json", "model.onnx"]
    meta = json.load(open(os.path.join(adv, "model.json")))
    assert set(meta["tools"]) == {"u%d" % i for i in range(6)} | {"lot"}
    spec = json.load(open(os.path.join(p.folder, "project.json")))
    assert spec["advanced"]["trained"] is True
    # every check got the pictures as good examples -- one copy on disk
    oks = sorted(glob.glob(os.path.join(p.folder, "training", "u0", "ok", "*")))
    assert len(oks) == 3
    good = sorted(glob.glob(os.path.join(adv, "good", "*.png")))
    assert os.path.samefile(oks[0], good[0]) or open(oks[0], "rb").read() == \
        open(good[0], "rb").read()
    assert st["result"]["examples_added"] == 3 * 7
    assert st["progress"] == 1.0
    assert A.is_trained(spec, p.folder) and not A.is_pending(spec, p.folder)


def test_cancelled_it_deletes_everything_and_the_job_is_as_it_was(tmp_path, monkeypatch):
    d = tmp_path / "o"
    d.mkdir()
    monkeypatch.setattr(A.Session, "_train", fake_train(str(d), steps=100, cancel_wait=True))
    monkeypatch.setattr(A, "N_PICTURES", 3)
    p = job(tmp_path)
    before = open(os.path.join(p.folder, "project.json")).read()
    sess = A.Session(p, grabber()).start()
    t = time.time()
    while sess.snapshot()["stage"] != "train" and time.time() - t < 30:
        time.sleep(0.05)
    assert sess.snapshot()["stage"] == "train"
    assert os.path.isdir(os.path.join(p.folder, "advanced", "new"))
    assert sess.cancel()
    assert sess.wait(30)
    st = sess.snapshot()
    assert st["cancelled"] and not st["ok"]
    assert not os.path.exists(os.path.join(p.folder, "advanced")), os.listdir(p.folder)
    assert not glob.glob(os.path.join(p.folder, "training", "*", "ok", "*"))
    assert open(os.path.join(p.folder, "project.json")).read() == before


def test_cancelled_while_taking_pictures(tmp_path, faked):
    p = job(tmp_path)
    slow = grabber()

    def grab():
        time.sleep(0.2)
        return slow()
    sess = A.Session(p, grab, n_pictures=50).start()
    time.sleep(0.5)
    sess.cancel()
    assert sess.wait(30)
    assert sess.snapshot()["cancelled"]
    assert not os.path.exists(os.path.join(p.folder, "advanced"))


@pytest.mark.parametrize("case,expect", [
    ("no camera", "camera gave no picture"),
    ("no checks", "no part or text checks"),
    ("not the part", "Align step did not find"),
    ("bad part", "GOOD part"),
    ("other size", "camera's settings changed"),
])
def test_every_way_it_cannot_train_says_why_and_leaves_nothing(tmp_path, faked, case, expect):
    from vision_station.core.schema import defaults
    tools = None
    grab = grabber()
    if case == "no camera":
        grab = grabber(fail=True)
    elif case == "no checks":
        tools = [{"id": "fx", "type": "align", "params": dict(defaults("align"),
                                                               roi=[50, 50, 1300, 800])}]
    elif case == "not the part":
        tools = [{"id": "fx", "type": "align",
                  "params": dict(defaults("align"), roi=[50, 50, 1300, 800])},
                 {"id": "u0", "type": "presence", "params": {"roi": [92, 142, 106, 106]}}]
        grab = grabber(np.full((900, 1400, 3), 128, np.uint8))
    elif case == "bad part":
        grab = grabber(np.full((900, 1400, 3), (40, 110, 40), np.uint8))
    elif case == "other size":
        grab = grabber(np.zeros((480, 640, 3), np.uint8))
    p = job(tmp_path, tools)
    st = run(A.Session(p, grab))
    assert not st["ok"] and expect in (st["error"] or ""), st["error"]
    assert not os.path.exists(os.path.join(p.folder, "advanced"))


def test_without_the_training_environment_it_says_so(tmp_path, faked):
    p = job(tmp_path)
    sess = A.Session(p, grabber())
    sess.python = None
    st = run(sess)
    assert "training environment is not installed" in st["error"]


FAKE_WORKER = r'''
import json, os, sys, time
a = sys.argv
out = a[a.index("--out") + 1]
batch = int(a[a.index("--batch") + 1])
mode = os.environ.get("FAKE_MODE", "ok")
def say(**k):
    print(json.dumps(k), flush=True)
if mode == "oom" and batch > 1:
    say(t="error", message="CUDA out of memory", oom=True); sys.exit(1)
if mode == "crash":
    sys.stderr.write("Traceback...\nRuntimeError: driver gone\n"); sys.exit(3)
say(t="start", epochs=3, batches=4, device="Fake GPU")
for e in range(1, 4):
    for b in range(1, 5):
        say(t="batch", epoch=e, batch=b, batches=4)
        time.sleep(0.05)
    say(t="epoch", epoch=e, epochs=3, metrics={"precision": .9 + e / 100, "recall": .95,
                                                "map50": .9, "map": .7})
    if mode == "slow":
        time.sleep(5)
p = os.path.join(out, "best.onnx")
open(p, "wb").write(b"x")
say(t="export", path="")
say(t="done", metrics={"precision": .99, "recall": .98, "map50": .97, "map": .8},
    onnx=p, seconds=1.0, batch=batch)
'''


@pytest.fixture
def worker(tmp_path, monkeypatch):
    w = tmp_path / "fake_worker.py"
    w.write_text(FAKE_WORKER)
    monkeypatch.setattr(A, "WORKER", str(w))
    monkeypatch.setattr(A.Session, "_validate", fake_validate)
    monkeypatch.setattr(A, "N_PICTURES", 2)
    monkeypatch.setattr(A, "base_weights", lambda size=None: ("yolo11n.pt", 4))
    return w


def test_the_workers_progress_reaches_the_window(tmp_path, worker, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "ok")
    p = job(tmp_path)
    sess = A.Session(p, grabber(), python=sys.executable)
    seen = []
    sess.start()
    while not sess.snapshot()["done"]:
        s = sess.snapshot()
        seen.append((s["stage"], round(s["progress"], 3), s["detail"]))
        time.sleep(0.01)
    st = sess.snapshot()
    assert st["ok"], st["error"]
    progress = [x[1] for x in seen]
    assert progress == sorted(progress), "the progress went backwards"
    assert any("pass 2 of 3" in x[2] for x in seen)
    assert st["result"]["metrics"]["precision"] == 0.99


def test_out_of_graphics_memory_tries_again_with_smaller_batches(tmp_path, worker, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "oom")
    p = job(tmp_path)
    st = run(A.Session(p, grabber(), python=sys.executable))
    assert st["ok"], st["error"]


def test_a_worker_that_crashes_is_reported_with_its_last_words(tmp_path, worker, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "crash")
    p = job(tmp_path)
    st = run(A.Session(p, grabber(), python=sys.executable))
    assert not st["ok"] and "driver gone" in st["error"], st["error"]
    assert not os.path.exists(os.path.join(p.folder, "advanced"))


def test_cancel_kills_the_worker_at_once(tmp_path, worker, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "slow")
    p = job(tmp_path)
    sess = A.Session(p, grabber(), python=sys.executable).start()
    t = time.time()
    while sess.snapshot()["epoch"] < 1 and time.time() - t < 60:
        time.sleep(0.05)
    t = time.time()
    sess.cancel()
    assert sess.wait(20)
    assert time.time() - t < 10, "the worker was not stopped"
    assert sess.snapshot()["cancelled"]
    assert not os.path.exists(os.path.join(p.folder, "advanced"))


def test_retraining_replaces_the_pictures_rather_than_adding_more(tmp_path, faked):
    p = job(tmp_path)
    assert run(A.Session(p, grabber()))["ok"]
    p = Project(p.folder)
    assert run(A.Session(p, grabber()))["ok"]
    assert len(glob.glob(os.path.join(p.folder, "training", "u0", "ok", "*"))) == 3
    A.remove(p.folder)
    assert not os.path.exists(os.path.join(p.folder, "advanced"))
    assert not glob.glob(os.path.join(p.folder, "training", "*", "ok", "advanced_*"))


def test_a_users_own_good_pictures_are_never_touched(tmp_path, faked):
    p = job(tmp_path)
    mine = os.path.join(p.folder, "training", "u0", "ok", "ok_123.png")
    os.makedirs(os.path.dirname(mine))
    imio.imwrite(mine, board())
    assert run(A.Session(p, grabber()))["ok"]
    A.remove(p.folder)
    assert os.path.exists(mine)


def test_example_pictures_are_decoded_once_for_all_checks(tmp_path, faked, monkeypatch):
    from vision_station.tools import count
    p = job(tmp_path)
    assert run(A.Session(p, grabber()))["ok"]
    p = Project(p.folder)
    p.advanced = None
    reads = []
    real = imio.imread
    monkeypatch.setattr(imio, "imread", lambda path, *a, **k: (reads.append(path),
                                                                 real(path, *a, **k))[1])
    count._EXAMPLES.clear()
    p.inspect(board())
    shared = [r for r in reads if "advanced_good" in r]
    assert len(shared) == 3, "decoded %d times for 3 pictures" % len(shared)


# ---------------------------------------------------------------- the window
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class FakeSession:
    def __init__(self, end="ok", steps=10, project=None):
        self.end, self.steps = end, steps
        self.state = {"stage": "capture", "progress": 0.0, "message": "Taking pictures",
                      "detail": "", "started": time.time(), "eta_s": None, "done": False,
                      "ok": False, "cancelled": False, "error": None, "result": None,
                      "can_cancel": True, "epoch": 0, "epochs": 3}
        self._stop = threading.Event()
        self.project = project

    def snapshot(self):
        return dict(self.state)

    def start(self):
        def go():
            for i in range(self.steps):
                if self._stop.wait(0.05):
                    self.state.update(cancelled=True, done=True)
                    return
                self.state.update(progress=(i + 1) / float(self.steps + 1),
                                  stage="train" if i > 2 else "capture", eta_s=10 - i)
            if self.end == "ok":
                self.state.update(ok=True, progress=1.0, result={
                    "metrics": {"precision": 0.995, "recall": 0.99, "map50": 0.98},
                    "validation": {"found_share": 1.0, "missing_caught": 29,
                                   "missing_tried": 30},
                    "learned": 40, "checks": 42, "skipped": {"big": "too large for the AI"},
                    "not_learned": ["x"], "pictures": 8, "seconds": 300})
            else:
                self.state.update(error="Training stopped: something")
            self.state["done"] = True
        threading.Thread(target=go, daemon=True).start()
        return self

    def cancel(self):
        self._stop.set()
        return True


def _pump(qapp, cond, timeout=10):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        qapp.processEvents()
        time.sleep(0.01)
    return cond()


def test_the_window_blurs_the_app_shows_progress_and_the_results_in_percent(qapp):
    from PySide6.QtWidgets import QMainWindow, QLabel
    from vision_station.ui.advanced_training import AdvancedTrainingDialog
    win = QMainWindow()
    win.setCentralWidget(QLabel("behind"))
    win.show()
    s = FakeSession(steps=20)
    dlg = AdvancedTrainingDialog(win, s, "Job 1")
    s.start()
    dlg.show()
    qapp.processEvents()
    assert dlg.windowModality() == __import__("PySide6.QtCore").QtCore.Qt.ApplicationModal
    assert win.centralWidget().graphicsEffect() is not None, "not blurred"
    assert not (dlg.windowFlags() & __import__("PySide6.QtCore").QtCore.Qt.WindowCloseButtonHint)
    assert _pump(qapp, lambda: dlg.bar.value() > 0)
    assert "%" in dlg.bar.format() and dlg.elapsed.text().startswith("Time taken")
    assert _pump(qapp, lambda: dlg.outcome == "ok")
    assert dlg.title.text() == "Training completed"
    texts = [w.text() for w in dlg.results.findChildren(QLabel)]
    assert "99.5 %" in texts and "99.0 %" in texts and "100.0 %" in texts
    assert any("29 of 30" in t for t in texts) and any("40 of 42" in t for t in texts)
    assert "big (too large for the AI)" in dlg.extra.text()
    assert dlg.okBtn.isVisible() and not dlg.cancelBtn.isVisible()
    dlg.accept()
    assert win.centralWidget().graphicsEffect() is None, "the blur stayed"
    win.close()


def test_escape_and_closing_ask_to_cancel_and_cancel_deletes(qapp, monkeypatch):
    from PySide6.QtWidgets import QMainWindow, QMessageBox
    from vision_station.ui.advanced_training import AdvancedTrainingDialog
    win = QMainWindow()
    win.show()
    s = FakeSession(steps=400)
    dlg = AdvancedTrainingDialog(win, s)
    s.start()
    dlg.show()
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(a[1]),
                                                                   QMessageBox.No)[1])
    dlg.reject()                                 # Escape
    dlg.close()                                  # Alt+F4
    qapp.processEvents()
    assert len(asked) == 2 and dlg.isVisible() and dlg.outcome is None
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    dlg.cancelBtn.click()
    assert _pump(qapp, lambda: dlg.outcome == "cancelled")
    assert not dlg.isVisible()
    win.close()


def test_a_failed_training_says_why_and_that_nothing_changed(qapp):
    from PySide6.QtWidgets import QMainWindow
    from vision_station.ui.advanced_training import AdvancedTrainingDialog
    win = QMainWindow()
    s = FakeSession(end="fail", steps=3)
    dlg = AdvancedTrainingDialog(win, s)
    s.start()
    dlg.show()
    assert _pump(qapp, lambda: dlg.outcome == "failed")
    assert "something" in dlg.stage.text() and "Nothing was changed" in dlg.detail.text()
    assert dlg.okBtn.text() == "Close"


# ---------------------------------------------------------------- in the application
@pytest.fixture
def qb(qapp, tmp_path, monkeypatch):
    import shutil
    from PySide6.QtCore import QSettings
    from vision_station.ui import main_window as V
    shutil.copytree(os.path.join(HERE, "projects", "demo_led_panel"),
                    str(tmp_path / "projects" / "demo_led_panel"),
                    ignore=shutil.ignore_patterns("results", "dataset"))
    ini = str(tmp_path / "s.ini")
    monkeypatch.setattr(V.QuickBuild, "_settings", lambda self: QSettings(ini, QSettings.IniFormat))
    w = V.QuickBuild(str(tmp_path / "projects"))
    w.resize(1200, 800)
    w.show()
    yield w
    w.dirty.clear()
    w.close()


def test_the_settings_switch_is_saved_and_off_where_training_is_impossible(qb, qapp, monkeypatch):
    from vision_station.ui.settings_dialog import SettingsDialog
    monkeypatch.setattr(A, "can_train", lambda: (True, "on a fake GPU"))
    dlg = SettingsDialog(qb, qb._settings(), qb=qb)
    sw = dlg.generalPage.useAdvanced
    assert not sw.isChecked() and sw.isEnabled()
    sw.setChecked(True)
    assert qb.advanced_training_on()
    dlg.close()
    monkeypatch.setattr(A, "can_train", lambda: (False, "no NVIDIA graphics card"))
    qb.set_advanced_training(False)
    dlg = SettingsDialog(qb, qb._settings(), qb=qb)
    assert not dlg.generalPage.useAdvanced.isEnabled()
    dlg.close()


def test_the_reference_popup_tells_about_advanced_training_only_when_on(qb):
    from vision_station.ui import main_window as V
    d = V.ReferenceSourceDialog(qb, "New application", advanced=True)
    assert d.advancedNote is not None and "Advanced model training" in d.advancedNote.text()
    assert "Run" in d.advancedNote.text()
    assert V.ReferenceSourceDialog(qb, "x").advancedNote is None


def _fake_training(monkeypatch, outcome="ok"):
    """run_training replaced: records the sessions; "ok" leaves the job
    marked trained, as the real one does."""
    from vision_station.ui import advanced_training as AT
    sessions = []

    def fake_run(win, sess, name=""):
        sessions.append(sess)
        if outcome == "ok":
            pj = os.path.join(sess.folder, "project.json")
            spec = json.load(open(pj))
            spec["advanced"] = {"requested": True, "trained": True}
            json.dump(spec, open(pj, "w"))
            os.makedirs(os.path.join(sess.folder, "advanced"), exist_ok=True)
            json.dump({"tools": {}}, open(os.path.join(sess.folder, "advanced", "model.json"),
                                          "w"))
            open(os.path.join(sess.folder, "advanced", "model.onnx"), "wb").write(b"x")
        return outcome
    monkeypatch.setattr(AT, "run_training", fake_run)
    monkeypatch.setattr(A, "can_train", lambda: (True, "fake"))
    return sessions


def test_run_offers_teaching_only_with_the_setting_on_and_normal_is_the_default(qb):
    assert qb.run_mode == "normal" and qb.aRunNormal.isChecked()
    menu = qb.aRunTeach.parent() if hasattr(qb.aRunTeach, "parent") else None
    btn = [w for w in qb.findChildren(__import__("PySide6.QtWidgets").QtWidgets.QToolButton)
           if w.defaultAction() is qb.aRunOnce][0]
    assert btn.menu() is not None, "Run has no dropdown"
    names = [a.text() for a in btn.menu().actions()]
    assert names == ["Normal run", "Model teaching run"]
    qb.set_advanced_training(False)
    assert not qb.aRunTeach.isEnabled()
    assert qb.set_run_mode("teach") == "normal", "teaching chosen with the setting off"
    qb.set_advanced_training(True)
    assert qb.aRunTeach.isEnabled()
    assert qb.set_run_mode("teach") == "teach" and qb.aRunOnce.text() == "Teach model"
    # switching the setting off takes Run back to normal
    qb.set_advanced_training(False)
    assert qb.run_mode == "normal" and qb.aRunOnce.text() == "Run"
    assert menu is not None


def test_a_normal_run_never_trains_even_a_job_made_for_it(qb, qapp, monkeypatch):
    sessions = _fake_training(monkeypatch)
    qb.set_advanced_training(True)
    pid = qb.station.jobs[0]["pid"]
    p = qb.station.projects[pid]
    p.spec["advanced"] = {"requested": True, "trained": False}
    qb.save_job(pid)
    qb.key_run_once()
    qb.station.wait(30)
    assert sessions == [], "a normal run trained"
    assert qb.station.last.get(pid) is not None, "the normal run did not inspect"


def test_a_teaching_run_teaches_and_run_goes_back_to_normal(qb, qapp, monkeypatch):
    sessions = _fake_training(monkeypatch)
    qb.set_advanced_training(True)
    qb.set_run_mode("teach")
    pid = qb.station.jobs[0]["pid"]
    qb.station.last.pop(pid, None)
    out = qb.key_run_once()
    assert out == {pid: "ok"} and len(sessions) == 1
    p = qb.station.projects[pid]
    assert A.is_trained(p.spec, p.folder) and p.advanced is not None
    assert qb.station.last.get(pid) is None, "a teaching run inspected"
    assert qb.run_mode == "normal" and qb.aRunNormal.isChecked()
    qb.key_run_once()                         # the next Run is a normal one
    qb.station.wait(30)
    assert len(sessions) == 1 and qb.station.last.get(pid) is not None


def test_a_cancelled_teaching_run_changes_nothing_and_stops(qb, qapp, monkeypatch):
    sessions = _fake_training(monkeypatch, outcome="cancelled")
    qb.set_advanced_training(True)
    qb.set_run_mode("teach")
    pid = qb.station.jobs[0]["pid"]
    out = qb.key_run_once()
    assert out == {pid: "cancelled"} and len(sessions) == 1
    p = qb.station.projects[pid]
    assert not A.is_trained(p.spec, p.folder)
    assert qb.run_mode == "normal"


def test_where_teaching_is_impossible_it_says_why(qb, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    qb.set_advanced_training(True)
    qb.set_run_mode("teach")
    monkeypatch.setattr(A, "can_train", lambda: (False, "no NVIDIA graphics card"))
    said = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: said.append(a[2]))
    assert qb.key_run_once() == {}
    assert len(said) == 1 and "no NVIDIA graphics card" in said[0]
    assert qb.run_mode == "normal"


def test_no_teaching_while_inspecting_continuously(qb, qapp, monkeypatch):
    sessions = _fake_training(monkeypatch)
    qb.set_advanced_training(True)
    qb.station.continuous = True
    try:
        assert qb.teaching_run() == {}
    finally:
        qb.station.continuous = False
    assert sessions == []


def test_a_run_asked_for_while_teaching_waits(qb, qapp):
    qb._adv_gating = True
    assert qb.station.run_once() is False
    qb._adv_gating = False
    assert qb.station.run_once() is True
    qb.station.wait(30)


def test_each_check_has_its_own_ai_pass_mark(tmp_path):
    p = job(tmp_path)
    tool = p.tools[0]
    from vision_station.core.engine import Context
    ctx = Context(reference=p.reference)
    ctx.tools = {t.id: t for t in p.tools}
    x, y, w, h = tool.roi
    det = FakeDet([[x, y, x + w, y + h, 0.25, 0]], {"u0": 0})
    p.advanced = det
    assert A.apply(p, tool, _result(), board(), ctx).status == "fail"     # 0.25 < 0.35
    det.thresholds = {"u0": A.own_threshold(0.4)}
    ctx.share.clear()
    assert det.threshold("u0") == 0.2
    assert A.apply(p, tool, _result(), board(), ctx).status == "pass"
    assert A.own_threshold(0.05) == A.MIN_THRESHOLD and A.own_threshold(0.99) == A.AI_MIN


def test_the_ai_never_overrules_a_check_that_is_sure(tmp_path):
    """A part matched at 0.98 is there: the AI losing it (a board put back a
    little further along) must not fail the board; a lenient check's 0.3
    match is what it overrules."""
    p = job(tmp_path)
    tool = p.tools[0]
    from vision_station.core.engine import Context
    ctx = Context(reference=p.reference)
    ctx.tools = {t.id: t for t in p.tools}
    p.advanced = FakeDet([], {"u0": 0})
    r = A.apply(p, tool, _result(score=0.98), board(), ctx)
    assert r.status == "pass" and r.values["ai_found"] is False and "unsure" in r.values["ai_note"]
    assert A.apply(p, tool, _result(score=0.3), board(), ctx).status == "fail"


def test_positions_cover_the_whole_range_and_tiles_start_anywhere():
    rng = np.random.default_rng(0)
    ps = A.poses(64, rng)
    dx = np.array([q["dx"] for q in ps])
    dy = np.array([q["dy"] for q in ps])
    tn = np.array([q["turn"] for q in ps])
    assert dx.min() < -0.6 * A.SHIFT and dx.max() > 0.6 * A.SHIFT
    assert dy.min() < -0.6 * A.SHIFT and dy.max() > 0.6 * A.SHIFT
    assert tn.min() <= -0.9 * A.TURN and tn.max() >= 0.9 * A.TURN
    # every quarter of the shift range is used
    q = np.histogram(dx, bins=4, range=(-A.SHIFT, A.SHIFT))[0]
    assert (q > 0).all(), q
    starts = {tuple(A.tile_starts(4000, 1280, o)) for o in (0, 100, 333)}
    assert len(starts) == 3, "the grid does not move"
    for st in starts:
        assert st[0] == 0 and st[-1] == 4000 - 1280


def test_training_tiles_are_lossless(tmp_path):
    img = board()
    oi, ol = tmp_path / "i", tmp_path / "l"
    oi.mkdir()
    ol.mkdir()
    A.write_tiles(img, [(0, 100, 150, 90, 90)], 1.0, str(oi), str(ol), "t",
                  np.random.default_rng(0), 0.0)
    files = os.listdir(oi)
    assert files and all(f.endswith(".png") for f in files)
    t = cv2.imread(str(oi / files[0]))
    assert np.array_equal(t[:900, :1280], img[:900, :1280]), "the tile is not the picture"


def test_a_check_failing_on_the_new_pictures_is_not_taught_from_them(tmp_path, faked):
    """A part deliberately wrong on the board: its check fails on the pictures,
    and they must not become its good examples or its AI labels."""
    p = job(tmp_path)
    labels = []
    real = A.write_tiles

    def spy(img, boxes, *a, **k):
        labels.append(len(boxes))
        return real(img, boxes, *a, **k)
    import unittest.mock as um
    with um.patch.object(A, "write_tiles", spy):
        st = run(A.Session(p, grabber(board(dark_missing=True))))
    assert st["ok"], st["error"]
    assert "u2" in st["result"]["failing"]
    assert not glob.glob(os.path.join(p.folder, "training", "u2", "ok", "*")),         "the wrong part was taught as good"
    assert len(glob.glob(os.path.join(p.folder, "training", "u1", "ok", "*"))) == 3
    # the reference and its moved copies have all 7 regions labelled; the new
    # pictures (and their copies) one fewer
    k = 1 + A.SYNTH_PER
    assert all(n == 7 for n in labels[:k]) and all(n == 6 for n in labels[k:]), labels


def test_a_bigger_model_trains_fewer_tiles_at_once():
    s = A.base_weights("small")[1]
    m = A.base_weights("medium")[1]
    l = A.base_weights("large")[1]
    assert s >= m >= l >= 1
    assert os.path.basename(A.base_weights("medium")[0]) == "yolo11m.pt"
    assert os.path.basename(A.base_weights("large")[0]) == "yolo11l.pt"


def test_the_model_size_setting_reaches_the_training(qb, qapp, monkeypatch):
    from vision_station.ui.settings_dialog import SettingsDialog
    monkeypatch.setattr(A, "can_train", lambda: (True, "fake"))
    dlg = SettingsDialog(qb, qb._settings(), qb=qb)
    combo = dlg.generalPage.modelSize
    combo.setCurrentIndex(combo.findData("large"))
    assert qb.advanced_model_size() == "large"
    dlg.close()
    sessions = _fake_training(monkeypatch)
    qb.set_advanced_training(True)
    qb.teaching_run()
    assert sessions and sessions[0].model_size == "large"


def test_both_output_formats_are_read_alike(monkeypatch):
    """YOLO11 gives (1, 4 + classes, N) before overlap removal; YOLO26 gives
    (1, N, 6) finds, ready. The same part comes out the same either way."""
    det = A.Detector("", {"tools": {}, "zoom": 1.0, "imgsz": 640})
    frame = np.zeros((1000, 1000, 3), np.uint8)
    v11 = np.zeros((1, 6, 3), np.float32)               # 2 classes, 3 candidates
    v11[0, :4, 0] = [120, 110, 50, 40]                  # cx cy w h
    v11[0, 4, 0] = 0.9
    v26 = np.zeros((1, 300, 6), np.float32)
    v26[0, 0] = [95, 90, 145, 130, 0.9, 0]
    got = []
    for out in (v11, v26):
        monkeypatch.setattr(A, "_net", lambda path, o=out: (lambda x: [o]))
        d = det.detect_tile(frame, 200, 300)
        assert len(d) == 1
        got.append(d[0])
    assert np.allclose(got[0], got[1], atol=1e-3), got
    assert np.allclose(got[0][:4], [295, 390, 345, 430])


def test_sixteen_good_pictures_are_still_decoded_once(tmp_path, monkeypatch):
    from vision_station.tools import count
    p = job(tmp_path)
    for tid in ("u0", "u1", "u2"):
        d = os.path.join(p.folder, "training", tid, "ok")
        os.makedirs(d)
    pics = []
    for i in range(16):
        f = os.path.join(p.folder, "good_%02d.png" % i)
        imio.imwrite(f, board())
        pics.append(f)
        for tid in ("u0", "u1", "u2"):
            os.link(f, os.path.join(p.folder, "training", tid, "ok", "advanced_good_%02d.png" % i))
    p = Project(p.folder)
    reads = []
    real = imio.imread
    monkeypatch.setattr(imio, "imread", lambda path, *a, **k: (reads.append(path),
                                                                 real(path, *a, **k))[1])
    count._EXAMPLES.clear()
    p.inspect(board(dark_missing=True))          # u2 fails: its good examples are looked at
    shared = [r for r in reads if "advanced_good" in r]
    assert len(shared) == 16, "decoded %d times for 16 pictures" % len(shared)


def test_restore_defaults_on_general_and_security(qb, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from vision_station.ui.settings_dialog import SettingsDialog
    monkeypatch.setattr(A, "can_train", lambda: (True, "on a fake GPU"))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: 0))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: 0))
    dlg = SettingsDialog(qb, qb._settings(), qb=qb)
    g = dlg.generalPage
    g.parallel.setChecked(True)
    g.useAdvanced.setChecked(True)
    g.modelSize.setCurrentIndex(g.modelSize.findData("large"))
    g.restore_defaults()
    assert not qb.aParallel.isChecked() and not qb.advanced_training_on()
    assert qb.advanced_model_size() == "small" and not qb.advanced
    # security: the default is no password -- removing one needs it
    sec = dlg.securityPage
    qb.set_supervisor_password("1234")
    assert sec.restore_defaults(password="wrong") is False
    assert qb._settings().value("supervisor_password", "")
    assert sec.restore_defaults(password="1234")
    assert not qb._settings().value("supervisor_password", "")
    assert sec.restore_defaults() is True              # already at defaults
    # line connection: broadcast off, listening and port kept
    lp = dlg.linePage
    if lp.available:
        lp.port.setValue(4100)
        lp.broadcast.setChecked(True)
        lp.restore_defaults()
        assert lp.port.value() == 4100 and not lp.broadcast.isChecked()
    dlg.close()
