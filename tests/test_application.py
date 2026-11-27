"""Test suite.

The point of these is not to re-test each feature in isolation -- each was
checked when it was written. The point is the *interactions*: rotation combined
with ellipse regions, undo restoring both, logging surviving a rotated recipe.
Those combinations are where regressions actually live, and none of them had
been exercised before this file existed.

Run:  python -m pytest tests.py -v
"""

import json
import math
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_station.core.engine import Context, ProjectStore                    # noqa: E402
from vision_station.core.history import History                                 # noqa: E402
from vision_station.tools.base import Graphic, Tool, ToolResult, make_tool  # noqa: E402
from vision_station.core import diagnostics                                          # noqa: E402
from vision_station.core.dataset import Dataset                                 # noqa: E402
from vision_station.core.results import ResultStore                             # noqa: E402


# --------------------------------------------------------------- fixtures
@pytest.fixture
def board():
    """A synthetic board: pale substrate, four screws, a row of LEDs, text."""
    img = np.full((420, 640, 3), 40, np.uint8)
    cv2.rectangle(img, (20, 20), (620, 400), (150, 140, 130), -1)
    for cx, cy in ((60, 60), (580, 60), (60, 360), (580, 360)):
        cv2.circle(img, (cx, cy), 18, (200, 200, 205), -1)
        cv2.circle(img, (cx, cy), 18, (90, 90, 95), 2)
    for i, col in enumerate([(60, 60, 220), (60, 200, 60), (220, 160, 60), (220, 90, 60)]):
        cv2.circle(img, (170 + i * 100, 150), 22, col, -1)
    cv2.putText(img, "AB12345X", (170, 300), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (250, 250, 250), 3)
    return img


@pytest.fixture
def tmpproj(board):
    d = tempfile.mkdtemp()
    pdir = os.path.join(d, "projects", "t1")
    os.makedirs(pdir)
    cv2.imwrite(os.path.join(pdir, "reference.png"), board)
    spec = {"id": "t1", "name": "T1", "tools": [
        {"id": "screws", "type": "count_blob",
         "params": {"roi": [30, 30, 580, 360], "enabled": True, "critical": True,
                    "angle": 0.0, "shape": "rect", "space": "gray",
                    "threshold": "otsu", "min_area": 200, "max_area": 8000,
                    "expect_count": 4, "row_tol": 40}}]}
    with open(os.path.join(pdir, "project.json"), "w") as f:
        json.dump(spec, f)
    yield os.path.join(d, "projects"), "t1", board
    shutil.rmtree(d, ignore_errors=True)


class Probe(Tool):
    """Minimal tool used to exercise the base-class geometry machinery."""
    type_name = "probe"

    def run(self, image, ctx):
        return ToolResult(self.id, self.type_name, "pass", 1.0, {}, [])


# --------------------------------------------- geometry: the risky overlaps
def test_rotation_and_ellipse_compose(board):
    """Both features touch crop(). Neither was ever tested with the other on."""
    ctx = Context(reference=board)
    roi = [140, 110, 100, 100]
    for angle in (0.0, 17.0, -33.0, 90.0):
        t = Probe("p", {"roi": roi, "angle": angle, "shape": "ellipse"})
        patch, _off = t.crop(board, ctx)
        assert patch.size > 0, "empty patch at %s deg" % angle
        assert t._crop_mask is not None, "ellipse mask missing at %s deg" % angle
        inside = t._crop_mask > 0
        outside = ~inside
        if outside.any():
            vals = np.unique(patch[outside])
            assert len(vals) <= 3, "outside fill not uniform at %s deg" % angle


def test_rotated_graphics_roundtrip_exactly(board):
    """A tool reporting a box at its patch origin must land on the true rotated
    corner. This is the calculation that silently lies if it is wrong."""
    ctx = Context(reference=board)
    x, y, w, h = 200, 120, 160, 80
    ang = 25.0
    t = Probe("p", {"roi": [x, y, w, h], "angle": ang})
    _patch, (ox, oy) = t.crop(board, ctx)
    res = ToolResult("p", "probe", "pass", 1.0, {},
                     [Graphic("rect", [(ox, oy), (ox + 10, oy + 10)])])
    t.correct_graphics(res, ctx)
    quad = t.mapped_quad(ctx)
    got = np.array(res.graphics[0].points[0])
    assert np.linalg.norm(got - quad[0]) < 1.5, \
        "corner off by %.2f px" % np.linalg.norm(got - quad[0])


def test_rotated_nonrect_graphics_are_mapped(board):
    """Rects take a separate branch in correct_graphics. Lines, polys and points
    go through the general path, and mutation testing showed that path was
    completely uncovered -- breaking it left the whole suite green."""
    ctx = Context(reference=board)
    ang = 40.0
    t = Probe("p", {"roi": [180, 100, 200, 100], "angle": ang})
    _patch, (ox, oy) = t.crop(board, ctx)
    quad = t.mapped_quad(ctx)
    res = ToolResult("p", "probe", "pass", 1.0, {}, [
        Graphic("line", [(ox, oy), (ox + 20, oy)]),
        Graphic("point", [(ox, oy)]),
        Graphic("poly", [(ox, oy), (ox + 10, oy), (ox + 10, oy + 10)]),
    ])
    t.correct_graphics(res, ctx)
    for g in res.graphics:
        first = np.array(g.points[0])
        assert np.linalg.norm(first - quad[0]) < 1.5, \
            "%s graphic not mapped: %s vs corner %s" % (g.kind, first, quad[0])
    # And the mapping must actually rotate, not merely offset.
    line = res.graphics[0].points
    dx, dy = line[1][0] - line[0][0], line[1][1] - line[0][1]
    assert abs(dy) > 1.0, "a rotated line came back perfectly horizontal"


def test_roi_graphic_is_absolute_and_not_double_mapped(board):
    """roi_graphic emits image coordinates already. If correct_graphics ever
    re-maps it, regions drift away from the part -- subtly, and only when
    rotated, which is the worst kind of bug to find in production."""
    ctx = Context(reference=board)
    t = Probe("p", {"roi": [100, 100, 120, 60], "angle": 30.0})
    t.crop(board, ctx)
    g = t.roi_graphic(ctx, "pass")[0]
    before = list(g.points)
    res = ToolResult("p", "probe", "pass", 1.0, {}, [g])
    t.correct_graphics(res, ctx)
    assert res.graphics[0].points == before, "absolute graphic was re-mapped"


def test_ellipse_excludes_corner_content():
    img = np.zeros((300, 300, 3), np.uint8)
    cv2.circle(img, (150, 150), 62, (200, 200, 200), -1)
    for cx, cy in ((88, 88), (212, 88), (88, 212), (212, 212)):
        cv2.circle(img, (cx, cy), 7, (255, 255, 255), -1)
    ctx = Context(reference=img)
    roi = [80, 80, 140, 140]

    def bright_blobs(p):
        g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(g, 230, 255, cv2.THRESH_BINARY)
        n, _, st, _ = cv2.connectedComponentsWithStats(th)
        return sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 20)

    rect, _ = Probe("r", {"roi": roi, "shape": "rect"}).crop(img, ctx)
    ell, _ = Probe("e", {"roi": roi, "shape": "ellipse"}).crop(img, ctx)
    assert bright_blobs(rect) == 4
    assert bright_blobs(ell) == 0


def test_zero_angle_crop_is_a_plain_slice(board):
    """Unrotated regions must not go through warpPerspective -- that would cost
    interpolation on the common path and blur every pixel very slightly."""
    ctx = Context(reference=board)
    t = Probe("p", {"roi": [100, 80, 120, 90], "angle": 0.0})
    patch, (ox, oy) = t.crop(board, ctx)
    assert (ox, oy) == (100, 80)
    assert np.array_equal(patch, board[80:170, 100:220])
    assert t._crop_M is None


# ------------------------------------------------------------- engine flow
def test_engine_runs_and_shapes_result(tmpproj):
    root, pid, board = tmpproj
    proj = ProjectStore(root).get(pid)
    res = proj.inspect(board)
    assert res["verdict"] in ("PASS", "FAIL", "WARN")
    assert res["tools"] and "tool_id" in res["tools"][0]
    for t in res["tools"]:
        assert "absolute" not in json.dumps(t["graphics"]), \
            "internal flag leaked into the result payload"


def test_engine_survives_a_throwing_tool(tmpproj):
    """One broken tool must not take down the inspection."""
    root, pid, board = tmpproj
    proj = ProjectStore(root).get(pid)

    class Boom(Tool):
        type_name = "boom"

        def run(self, image, ctx):
            raise RuntimeError("deliberate")

    proj.tools.append(Boom("boom", {"roi": [0, 0, 10, 10]}))
    res = proj.inspect(board)
    statuses = {t["tool_id"]: t["status"] for t in res["tools"]}
    assert statuses.get("boom") == "error"
    assert "deliberate" in [t["message"] for t in res["tools"]
                            if t["tool_id"] == "boom"][0]


def test_rotated_ellipse_recipe_runs_end_to_end(tmpproj):
    """The combination, through the real engine rather than the base class."""
    root, pid, board = tmpproj
    store = ProjectStore(root)
    proj = store.get(pid)
    proj.spec["tools"][0]["params"].update({"angle": 12.0, "shape": "ellipse"})
    proj.tools = [make_tool(s) for s in proj.spec["tools"]]
    res = proj.inspect(board)
    assert res["tools"][0]["status"] in ("pass", "fail")
    kinds = {g["kind"] for g in res["tools"][0]["graphics"]}
    assert "poly" in kinds, "rotated/elliptical region should draw as a polygon"


# ------------------------------------------------------------------ history
def test_undo_restores_angle_and_shape():
    """Undo stores whole snapshots, so newer params must ride along for free.
    This asserts that, because a hand-written undo would have missed them."""
    h = History()
    spec = {"tools": [{"id": "a", "params": {"roi": [0, 0, 10, 10],
                                             "angle": 0.0, "shape": "rect"}}]}
    h.push("change", spec)
    spec["tools"][0]["params"].update({"angle": 45.0, "shape": "ellipse",
                                       "roi": [5, 5, 20, 20]})
    snap, _ = h.undo(spec)
    p = snap["tools"][0]["params"]
    assert p["angle"] == 0.0 and p["shape"] == "rect" and p["roi"] == [0, 0, 10, 10]


def test_undo_coalescing_and_redo_order():
    h = History()
    s = {"n": 0}
    h.push("move region", s)
    for i in range(1, 30):
        h.push("move region", {"n": i})       # one gesture
    assert len(h._undo) == 1, "a single gesture became %d undo steps" % len(h._undo)
    h.break_coalesce()
    h.push("add tool", {"n": 99})
    assert len(h._undo) == 2
    a, _ = h.undo({"n": 100})
    assert a == {"n": 99}
    b, _ = h.undo(a)
    assert b == {"n": 0}
    r, _ = h.redo(b)
    assert r == {"n": 99}


def test_history_snapshots_are_deep():
    h = History()
    spec = {"tools": [{"id": "a", "params": {"roi": [1, 2, 3, 4]}}]}
    h.push("e", spec)
    spec["tools"][0]["params"]["roi"][0] = 999
    snap, _ = h.undo(spec)
    assert snap["tools"][0]["params"]["roi"][0] == 1, "snapshot aliased live state"


# -------------------------------------------------------------- diagnostics
def test_focus_metric_separates_blur_from_darkness(board):
    sharp = diagnostics.image_quality(board)["metrics"]["focus_normalised"]
    dark = diagnostics.image_quality((board * 0.3).astype(np.uint8))
    blurred = diagnostics.image_quality(cv2.GaussianBlur(board, (21, 21), 0))
    assert dark["metrics"]["focus_normalised"] > sharp * 0.5, \
        "a dark but sharp image was penalised as unfocused"
    assert blurred["metrics"]["focus_normalised"] < sharp * 0.2
    assert any("focus" in m.lower() for k, m in blurred["findings"] if k == "bad")
    assert not any(k == "bad" and "focus" in m.lower()
                   for k, m in dark["findings"]), \
        "dark image wrongly reported as out of focus"


def test_image_quality_never_raises():
    for img in (np.zeros((4, 4, 3), np.uint8),
                np.full((10, 10, 3), 255, np.uint8),
                np.zeros((2, 2), np.uint8)):
        q = diagnostics.image_quality(img)
        assert "findings" in q and q["findings"]
    assert diagnostics.image_quality(None)["ok"] is False


def test_margin_flags_a_thin_pass():
    spec = {"tools": [{"id": "a", "type": "presence",
                       "params": {"min_score": 0.70}},
                      {"id": "b", "type": "presence",
                       "params": {"min_score": 0.70}}]}
    res = {"tools": [{"tool_id": "a", "type": "presence", "score": 0.71,
                      "status": "pass"},
                     {"tool_id": "b", "type": "presence", "score": 0.98,
                      "status": "pass"}]}
    rows = {r["tool_id"]: r for r in diagnostics.margin_report(res, spec)}
    assert rows["a"]["verdict"] == "fragile"
    assert rows["b"]["verdict"] == "healthy"


# ------------------------------------------------------------------ results
def test_result_logging_is_durable_and_fast(tmpproj):
    root, pid, board = tmpproj
    folder = os.path.join(root, pid)
    st = ResultStore(folder, pid, archive="fail", max_archived=4)
    img = np.zeros((40, 40, 3), np.uint8)

    def mk(i, verdict):
        return {"id": "u%d" % i, "verdict": verdict, "total_ms": 100,
                "source": "test",
                "tools": [{"tool_id": "screws", "type": "count_blob",
                           "status": "fail" if verdict == "FAIL" else "pass",
                           "score": 0.5, "ms": 2, "message": "", "values": {}}]}

    # Median, not mean. The mean over a burst is dominated by one-off costs --
    # the first disk write, a garbage collection -- which the queue is there to
    # absorb. What matters is the cost of a typical call on the inspection
    # thread, and a mean made this assertion fail intermittently while the
    # typical call was measuring 0.002 ms.
    per_call = []
    for i in range(120):
        t0 = time.perf_counter()
        st.log(mk(i, "FAIL" if i % 3 == 0 else "PASS"), image=img)
        per_call.append((time.perf_counter() - t0) * 1000)
    per_call.sort()
    median_ms = per_call[len(per_call) // 2]
    assert median_ms < 0.5, \
        "logging cost %.3f ms on the inspection thread" % median_ms

    st.flush(6.0)
    s = st.summary()
    assert s["total"] == 120 and s["failed"] == 40
    imgs = [f for _r, _d, fs in os.walk(os.path.join(folder, "results", "images"))
            for f in fs]
    assert len(imgs) <= 4, "archive cap exceeded: %d files" % len(imgs)
    assert st.summary()["total"] == 120, "pruning images must not delete records"
    p = st.pareto()
    assert p and p[0]["tool_id"] == "screws"
    st.close()


def test_export_csv_roundtrip(tmpproj):
    root, pid, _b = tmpproj
    folder = os.path.join(root, pid)
    st = ResultStore(folder, pid, archive="none")
    st.log({"id": "x", "verdict": "FAIL", "total_ms": 12, "source": "t",
            "tools": [{"tool_id": "a", "type": "presence", "status": "fail",
                       "score": 0.1, "ms": 1, "message": "nope", "values": {}}]})
    st.flush(4.0)
    out = os.path.join(folder, "e.csv")
    n = st.export_csv(out)
    assert n == 1
    body = _slurp(out)
    assert "nope" in body and "FAIL" in body
    st.close()


# ------------------------------------------------------------------ dataset
def test_dataset_export_produces_valid_yolo(tmpproj):
    root, pid, board = tmpproj
    folder = os.path.join(root, pid)
    imgs = os.path.join(folder, "images")
    os.makedirs(imgs, exist_ok=True)
    cv2.imwrite(os.path.join(imgs, "a.png"), board)
    cv2.imwrite(os.path.join(imgs, "b.png"), board)
    ds = Dataset(folder)
    ds.classes = ["screw", "led"]
    ds.boxes = {"a.png": [{"cls": 0, "x": 42, "y": 42, "w": 36, "h": 36}],
                "b.png": [{"cls": 1, "x": 148, "y": 128, "w": 44, "h": 44}]}
    ds.save()
    out = os.path.join(folder, "ds")
    info = ds.export(out)
    assert info["classes"] == 2
    yaml = _slurp(os.path.join(out, "data.yaml"))
    assert "0: screw" in yaml and "1: led" in yaml
    labels = []
    for sub in ("train", "val"):
        d = os.path.join(out, "labels", sub)
        for n in os.listdir(d):
            labels.append(_slurp(os.path.join(d, n)).strip())
    assert labels, "no label files written"
    for line in labels:
        if not line:
            continue
        parts = line.split()
        assert len(parts) == 5
        assert 0 <= int(parts[0]) < 2
        for v in parts[1:]:
            assert 0.0 <= float(v) <= 1.0, "un-normalised coordinate %s" % v


def test_dataset_remove_class_reindexes():
    """Deleting a middle class must renumber the ones above it, or every later
    label silently points at the wrong class."""
    d = tempfile.mkdtemp()
    try:
        ds = Dataset(d)
        ds.classes = ["a", "b", "c"]
        ds.boxes = {"i.png": [{"cls": 0, "x": 0, "y": 0, "w": 5, "h": 5},
                              {"cls": 1, "x": 0, "y": 0, "w": 5, "h": 5},
                              {"cls": 2, "x": 0, "y": 0, "w": 5, "h": 5}]}
        ds.save()
        ds.remove_class("b")
        assert ds.classes == ["a", "c"]
        remaining = sorted(b["cls"] for b in ds.boxes["i.png"])
        assert remaining == [0, 1], "class indices not reindexed: %s" % remaining
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------------------------- align
def _pose(img, dx, dy, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, np.vstack([M, [0, 0, 1]]).astype(np.float32)


def _corner_err(M_true, M_est, w, h):
    pts = np.float32([[80, 80], [w - 80, 80], [w - 80, h - 80],
                      [80, h - 80]]).reshape(-1, 1, 2)
    a = cv2.transform(pts, M_true[:2]).reshape(-1, 2)
    b = cv2.transform(pts, np.asarray(M_est, np.float32)[:2]).reshape(-1, 2)
    return float(np.max(np.linalg.norm(a - b, axis=1)))


@pytest.mark.parametrize("dx,dy,deg", [
    (0, 0, 0.0), (18, -12, 0.0), (0, 0, 3.0), (10, -8, 6.5), (-14, 9, -5.0),
])
def test_align_recovers_pose_accurately(board, dx, dy, deg):
    """Regression guard for a real bug: the template matcher built its rotation
    about the image origin instead of the template centre. Translation-only
    cases were perfect, so it looked fine, but a 9 degree rotation put every
    downstream ROI 147 px out of place while reporting a match score of 0.967.
    A confidently wrong fixture is the worst failure this app can produce."""
    from vision_station.tools.align import Align
    h, w = board.shape[:2]
    cur, M_true = _pose(board, dx, dy, deg)
    ctx = Context(reference=board)
    t = Align("a", {"method": "template", "min_score": 0.3,
                    "angle_range": [-10, 10], "angle_step": 1.0,
                    "roi": [60, 60, w - 120, h - 120]})
    res = t.run(cur, ctx)
    assert res.status == "pass", "align failed: %s" % res.message
    assert ctx.fixture is not None, "fixture not published to downstream tools"
    err = _corner_err(M_true, ctx.fixture, w, h)
    assert err < 2.0, "corner error %.2f px at dx=%s dy=%s deg=%s" % (err, dx, dy, deg)


@pytest.mark.parametrize("deg", [3.0, 7.0, -5.0])
def test_align_template_seed_is_correct_without_refinement(board, deg):
    """Checks the raw template estimate with refinement OFF.

    This test exists because the obvious version of it does not work. With
    refinement on, ECC rescues a badly wrong seed, so re-introducing the
    original origin-rotation bug left every other align test green. The bug was
    only visible once refinement was disabled. Any matcher that relies on a
    polish step to hide a broken estimate will fail the moment that polish does
    not converge, which on a real part is exactly when things are already hard."""
    from vision_station.tools.align import Align
    h, w = board.shape[:2]
    cur, M_true = _pose(board, 5, -5, deg)
    ctx = Context(reference=board)
    t = Align("a", {"method": "template", "min_score": 0.3,
                    "angle_range": [-10, 10], "angle_step": 0.5, "refine": False,
                    "roi": [60, 60, w - 120, h - 120]})
    res = t.run(cur, ctx)
    assert res.status == "pass", res.message
    err = _corner_err(M_true, ctx.fixture, w, h)
    assert err < 8.0, "unrefined seed off by %.1f px at %s deg" % (err, deg)


def test_align_features_method_also_accurate(board):
    from vision_station.tools.align import Align
    h, w = board.shape[:2]
    cur, M_true = _pose(board, 12, -9, 5.0)
    ctx = Context(reference=board)
    t = Align("a", {"method": "features", "min_score": 0.2, "n_features": 3000,
                    "roi": [60, 60, w - 120, h - 120]})
    res = t.run(cur, ctx)
    assert res.status == "pass", res.message
    assert _corner_err(M_true, ctx.fixture, w, h) < 3.0


def test_align_failure_does_not_publish_a_fixture(board):
    """If align cannot find the part it must leave ctx.fixture unset, so later
    tools are skipped rather than run against a fabricated pose."""
    from vision_station.tools.align import Align
    noise = np.random.RandomState(1).randint(0, 255, board.shape).astype(np.uint8)
    ctx = Context(reference=board)
    h, w = board.shape[:2]
    t = Align("a", {"method": "template", "min_score": 0.95,
                    "angle_range": [0, 0], "roi": [60, 60, w - 120, h - 120]})
    res = t.run(noise, ctx)
    assert res.status == "fail"
    assert ctx.fixture is None, "a failed align still set a fixture"


def test_align_refinement_never_makes_it_worse(board):
    """Refinement is accepted only if it measurably improves the match. This
    asserts the comparison exists, because an unchecked ECC step can converge
    into a neighbouring optimum and silently degrade a good answer."""
    from vision_station.tools.align import Align
    h, w = board.shape[:2]
    cur, M_true = _pose(board, 8, 6, 4.0)
    params = {"method": "template", "min_score": 0.3, "angle_range": [-10, 10],
              "angle_step": 1.0, "roi": [60, 60, w - 120, h - 120]}
    errs = {}
    for refine in (False, True):
        ctx = Context(reference=board)
        Align("a", dict(params, refine=refine)).run(cur, ctx)
        errs[refine] = _corner_err(M_true, ctx.fixture, w, h)
    assert errs[True] <= errs[False] + 0.5, \
        "refinement degraded the result: %.2f -> %.2f" % (errs[False], errs[True])


# ------------------------------------------------- measurement & counting
def _bar(w=800, h=300, x0=200, x1=600, bg=40, fg=210):
    img = np.full((h, w), bg, np.uint8)
    img[:, x0:x1] = fg
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _onnx_available():
    try:
        import onnx, onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


onnx_required = pytest.mark.skipif(not _onnx_available(),
                                   reason="onnx / onnxruntime not installed")


def _slurp(path, encoding="utf-8"):
    """Read a text file and close it. open(...).read() left the handle to the
    garbage collector, which on Windows can keep the file locked."""
    with open(path, encoding=encoding) as f:
        return f.read()


def _run(ttype, params, image, ctx=None, reference=None):
    base = {"enabled": True, "critical": True, "angle": 0.0, "shape": "rect"}
    spec = {"id": "t", "type": ttype, "params": dict(base, **params)}
    ctx = ctx or Context(reference=reference if reference is not None else image)
    return make_tool(spec).run(image, ctx), ctx


@pytest.mark.parametrize("x0,x1", [(150, 650), (300, 500), (370, 430)])
def test_measure_length_is_subpixel_accurate(x0, x1):
    """Measurement is the one tool whose output goes on a report and cannot be
    eyeballed, so it gets the tightest tolerance in the suite."""
    img = _bar(x0=x0, x1=x1)
    r, _ = _run("measure_length",
                {"roi": [140, 100, 520, 100], "axis": "x", "polarity": "any",
                 "min_contrast": 2.0, "max_edges": 2, "mm_per_px": 0.0,
                 "nominal_mm": 0.0, "tolerance_mm": 1e9}, img)
    got = r.values.get("length_px")
    assert got is not None, r.message
    assert abs(got - (x1 - x0)) < 0.5, "measured %.3f, true %d" % (got, x1 - x0)


def test_measure_length_survives_blur_and_noise():
    img = _bar(x0=200, x1=600)
    img = cv2.GaussianBlur(img, (7, 7), 3)
    img = np.clip(img.astype(np.float32) +
                  np.random.RandomState(0).normal(0, 8, img.shape),
                  0, 255).astype(np.uint8)
    r, _ = _run("measure_length",
                {"roi": [190, 100, 420, 100], "axis": "x", "polarity": "any",
                 "min_contrast": 2.0, "max_edges": 2, "mm_per_px": 0.0,
                 "nominal_mm": 0.0, "tolerance_mm": 1e9}, img)
    assert abs(r.values["length_px"] - 400) < 2.0


def test_measure_mm_conversion_and_tolerance():
    img = _bar(x0=200, x1=600)                     # 400 px
    base = {"roi": [190, 100, 420, 100], "axis": "x", "polarity": "any",
            "min_contrast": 2.0, "max_edges": 2, "mm_per_px": 0.05}
    inside, _ = _run("measure_length",
                     dict(base, nominal_mm=20.0, tolerance_mm=0.1), img)
    outside, _ = _run("measure_length",
                      dict(base, nominal_mm=20.5, tolerance_mm=0.1), img)
    assert abs(inside.values["length_mm"] - 20.0) < 0.05
    assert inside.status == "pass"
    assert outside.status == "fail", "an out-of-tolerance part was passed"


def test_calibrate_scale_feeds_measure_length():
    """The two tools communicate through the context. If that link breaks,
    every measurement silently reverts to pixels."""
    img = _bar(x0=200, x1=600)
    ctx = Context(reference=img)
    cal, _ = _run("calibrate_scale",
                  {"roi": [190, 100, 420, 100], "known_mm": 20.0, "axis": "x",
                   "polarity": "any", "min_contrast": 2.0}, img, ctx=ctx)
    assert cal.status == "pass"
    assert abs(ctx.mm_per_px - 0.05) < 0.001
    m, _ = _run("measure_length",
                {"roi": [190, 100, 420, 100], "axis": "x", "polarity": "any",
                 "min_contrast": 2.0, "max_edges": 2, "mm_per_px": 0.0,
                 "nominal_mm": 20.0, "tolerance_mm": 0.05}, img, ctx=ctx)
    assert m.values.get("length_mm") is not None, "calibration did not reach the tool"
    assert abs(m.values["length_mm"] - 20.0) < 0.05


@pytest.mark.parametrize("n", [1, 3, 4, 6, 10])
def test_count_blob_counts_correctly(n):
    img = np.full((200, 640, 3), 30, np.uint8)
    gap = 640 // (n + 1)
    for i in range(n):
        cv2.circle(img, (gap * (i + 1), 100), 16, (225, 225, 225), -1)
    r, _ = _run("count_blob",
                {"roi": [0, 0, 640, 200], "space": "gray", "threshold": "otsu",
                 "min_area": 50, "max_area": 100000, "expect_count": n,
                 "row_tol": 40, "invert": False, "open_px": 3}, img)
    assert r.values["count"] == n
    assert r.status == "pass"


def test_defect_diff_does_not_false_reject_on_lighting_drift():
    """The failure that gets a vision system switched off is not a missed
    defect, it is rejecting good parts. Illumination normalisation has to
    absorb a lighting change without crying defect."""
    def panel(gain=1.0, shift=0, noise=0):
        img = np.full((300, 400, 3), 150, np.uint8)
        cv2.rectangle(img, (40, 40), (360, 260), (175, 175, 175), -1)
        for i in range(4):
            cv2.circle(img, (90 + i * 70, 150), 18, (105, 105, 105), -1)
        img = np.clip(img.astype(np.float32) * gain + shift, 0, 255).astype(np.uint8)
        if noise:
            img = np.clip(img.astype(np.float32) +
                          np.random.RandomState(2).normal(0, noise, img.shape),
                          0, 255).astype(np.uint8)
        return img

    ref = panel()
    P = {"roi": [20, 20, 360, 260], "threshold": 22, "illum_blur": 51,
         "open_px": 2, "min_area": 25, "scratch_elongation": 3.5,
         "max_defects": 0, "max_area_px": 0}
    for label, kw in (("identical", {}), ("noise", {"noise": 4}),
                      ("8% brighter", {"gain": 1.08}), ("offset +10", {"shift": 10})):
        r, _ = _run("defect_diff", P, panel(**kw), reference=ref)
        assert r.status == "pass", "false reject on a good part (%s)" % label

    dented = panel()
    cv2.circle(dented, (200, 90), 8, (60, 60, 60), -1)
    r, _ = _run("defect_diff", P, dented, reference=ref)
    assert r.status == "fail", "an 8 px dent was missed"


def test_color_classify_discriminates_not_just_agrees():
    """A classifier that passes whenever told what to expect is worthless. The
    off-diagonal cases are the ones that matter."""
    paints = {"red": (40, 40, 210), "green": (60, 190, 60), "blue": (210, 70, 50)}
    for painted, bgr in paints.items():
        for expect in paints:
            img = np.zeros((120, 200, 3), np.uint8)
            img[:, :] = bgr
            ctx = Context(reference=img)
            ctx.share["src"] = [{"cx": 100, "cy": 60, "x": 60, "y": 30,
                                 "w": 80, "h": 60, "area": 4800}]
            r, _ = _run("color_classify",
                        {"roi": [0, 0, 200, 120], "source": "src",
                         "expect": [expect], "deltaE_max": 40.0}, img, ctx=ctx)
            want = "pass" if painted == expect else "fail"
            assert r.status == want, \
                "painted %s expected %s gave %s" % (painted, expect, r.status)


# -------------------------------------------------------------------- OCR
def _text_image(text="AB12345X", w=520, h=120, fg=245, bg=35, scale=1.4,
                thick=3, noise=0.0):
    img = np.full((h, w), bg, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, int(fg), thick, cv2.LINE_AA)
    if noise:
        img = np.clip(img.astype(np.float32) +
                      np.random.RandomState(0).normal(0, noise, img.shape),
                      0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _ocr_params(**kw):
    return dict({"roi": [0, 0, 520, 120], "upscale": 2.0, "denoise": True,
                 "binarize": True, "adaptive_block": 51, "clahe": False,
                 "invert": False, "close_dots": 0, "min_confidence": 0.0,
                 "lang": "en",
                 "charset": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"}, **kw)


def _has_ocr():
    from vision_station.tools import ocr_engine
    if ocr_engine.onnx_available():
        return True
    for m in ("paddleocr", "easyocr", "pytesseract"):
        try:
            __import__(m)
            return True
        except ImportError:
            continue
    return False


ocr_required = pytest.mark.skipif(not _has_ocr(), reason="no OCR engine installed")


def _cer(truth, got):
    a, b = truth, got or ""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(a))


@ocr_required
@pytest.mark.parametrize("noise", [0.0, 10.0])
def test_ocr_reads_clean_and_noisy_text(noise):
    """Noise used to destroy OCR completely: the old pipeline ran CLAHE on an
    un-denoised, un-binarised image and returned ': : OB ee' for AB12345X, a
    character error rate above 100%. Denoise-then-threshold fixed it."""
    img = _text_image(noise=noise)
    r, _ = _run("ocr", _ocr_params(), img)
    got = (r.values.get("text") or "").strip()
    assert _cer("AB12345X", got) < 0.20, "read %r" % got


@ocr_required
def test_ocr_confidence_is_real_not_hardcoded():
    """Tesseract confidence was previously a constant 0.5, which made
    min_confidence useless and fed a fake number into the margin report.
    Confidence must vary with the image and gate the verdict."""
    blank, _ = _run("ocr", _ocr_params(), _text_image(fg=35))
    assert blank.status == "fail", "blank image was read as text"

    # Two readable images of clearly different quality must not report the same
    # confidence. Comparing one image against a constant is too weak a test:
    # a hardcoded 0.5 survives it.
    scores = set()
    for kw in ({}, {"fg": 120, "bg": 92, "scale": 0.8, "thick": 1},
               {"noise": 12.0}, {"fg": 150, "bg": 110, "noise": 8.0},
               {"fg": 120, "bg": 90, "scale": 0.7, "thick": 1, "noise": 14}):
        r, _ = _run("ocr", _ocr_params(), _text_image(**kw))
        if (r.values.get("text") or "").strip():
            scores.add(round(r.score, 3))
    assert len(scores) > 1, \
        "confidence identical across differing image quality: %s" % scores

    hard = _text_image(fg=120, bg=90, scale=0.7, thick=1, noise=14)
    lenient, _ = _run("ocr", _ocr_params(min_confidence=0.0), hard)
    strict, _ = _run("ocr", _ocr_params(min_confidence=0.99), hard)
    assert strict.status == "fail", "min_confidence=0.99 accepted a poor read"
    assert strict.score == lenient.score, "threshold changed the measured score"


@ocr_required
def test_ocv_verifies_pattern_and_rejects_wrong_text():
    img = _text_image("AB12345X")
    # A loose pattern, because the point here is the verification logic, not
    # whether the engine resolves every digit on a synthetic font.
    ok, _ = _run("ocv", _ocr_params(expect_regex="^[A-Z]{2}[0-9]{5}[A-Z]$",
                                    expect_text="", strip_spaces=True), img)
    assert ok.status == "pass", "correct marking rejected: %s" % ok.message
    bad, _ = _run("ocv", _ocr_params(expect_regex="^ZZ[0-9]{5}Q$",
                                     expect_text="", strip_spaces=True), img)
    assert bad.status == "fail", "wrong pattern accepted"


@ocr_required
def test_ocr_reads_through_uneven_lighting():
    """Glare over half the marking. The reader works on the photograph as
    taken, so the old 'binarize' setting -- still present in earlier recipes
    -- must neither be needed nor do harm."""
    img = _text_image("AB12345X")
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    glare = np.exp(-(((xx - w * 0.65) ** 2) / (2 * (w * 0.16) ** 2) +
                     ((yy - h * 0.4) ** 2) / (2 * (h * 0.5) ** 2)))
    g = np.clip(g + 0.55 * 255 * glare, 0, 255).astype(np.uint8)
    lit = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    on, _ = _run("ocr", _ocr_params(binarize=True), lit)
    off, _ = _run("ocr", _ocr_params(binarize=False), lit)
    err_on = _cer("AB12345X", (on.values.get("text") or "").strip())
    err_off = _cer("AB12345X", (off.values.get("text") or "").strip())
    assert err_on < 0.25, "glare not handled even with binarisation (CER %.2f)" % err_on
    assert err_on <= err_off, \
        "binarisation made glare worse: %.2f with, %.2f without" % (err_on, err_off)


@ocr_required
def test_ocv_blank_expect_text_is_not_a_demand_for_empty_text():
    """Regression guard for a bug that made every OCV tool built in the UI fail
    every part. The code tested `expect_text is not None`, and the property
    sheet supplies an empty string, so the tool demanded that the part be blank.
    Only the demo recipe escaped it, because its JSON omitted the field."""
    img = _text_image("AB12345X")
    r, _ = _run("ocv", _ocr_params(expect_regex="^[A-Z0-9]{6,12}$",
                                   expect_text="", strip_spaces=True), img)
    assert r.status == "pass", \
        "blank expect_text blocked a matching regex: %s" % r.message

    none_set, _ = _run("ocv", _ocr_params(expect_regex="", expect_text="",
                                          strip_spaces=True), img)
    assert none_set.status == "pass", "any text should satisfy an empty rule"

    exact, _ = _run("ocv", _ocr_params(expect_regex="", expect_text="NOPE",
                                       strip_spaces=True), img)
    assert exact.status == "fail", "a wrong exact string was accepted"


@ocr_required
def test_ocr_charset_restricts_output():
    """A whitelist is the cheapest accuracy win on industrial marking. If it is
    not reaching the engine, digits will appear in a letters-only read."""
    img = _text_image("AB12345X")
    r, _ = _run("ocr", _ocr_params(charset="ABCDEFGHIJKLMNOPQRSTUVWXYZ"), img)
    got = (r.values.get("text") or "").strip()
    assert not any(ch.isdigit() for ch in got), \
        "digits present despite a letters-only charset: %r" % got


# ------------------------------------------------------------ persistence
def test_recipe_roundtrips_every_field(tmpproj):
    """Save then reload must preserve angle, shape and per-tool flags. Fields
    added later are exactly the ones a save path quietly forgets."""
    root, pid, board = tmpproj
    store = ProjectStore(root)
    proj = store.get(pid)
    proj.spec["tools"][0]["params"].update({
        "angle": -23.5, "shape": "ellipse", "enabled": False, "critical": False,
        "expect_count": 7})
    proj.save(proj.spec)

    fresh = ProjectStore(root).reload(pid)
    p = fresh.spec["tools"][0]["params"]
    assert p["angle"] == -23.5, "angle lost on save/reload"
    assert p["shape"] == "ellipse", "shape lost on save/reload"
    assert p["enabled"] is False and p["critical"] is False
    assert p["expect_count"] == 7


def test_disabled_tool_is_skipped(tmpproj):
    root, pid, board = tmpproj
    proj = ProjectStore(root).get(pid)
    proj.tools[0].enabled = False
    res = proj.inspect(board)
    assert all(t["tool_id"] != "screws" for t in res["tools"]), \
        "a disabled tool still ran"


# ------------------------------------------------- tools as the UI makes them
def _ui_scene(w=640, h=440):
    img = np.full((h, w, 3), 38, np.uint8)
    cv2.rectangle(img, (30, 30), (w - 30, h - 30), (150, 142, 132), -1)
    for cx, cy in ((70, 70), (w - 70, 70), (70, h - 70), (w - 70, h - 70)):
        cv2.circle(img, (cx, cy), 16, (205, 205, 210), -1)
        cv2.circle(img, (cx, cy), 16, (70, 70, 75), 2)
    for i, col in enumerate([(60, 60, 215), (60, 195, 60),
                             (60, 215, 225), (215, 90, 60)]):
        cv2.circle(img, (170 + i * 90, 160), 22, col, -1)
    cv2.putText(img, "AB12345X", (170, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (248, 248, 248), 3)
    return img


_UI_SETUP = {
    "find_ellipse": {"roi": [140, 120, 60, 60]},
    # across the LED row: four dark-on-light... the LEDs are darker than the board in grey
    "caliper": {"roi": [120, 138, 400, 24], "pair_polarity": "dark", "channel": "blue"},
    # the board's top edge runs along y = 20; an LED of radius 22 sits at (170, 150)
    "find_line": {"roi": [100, 5, 400, 30]},
    "find_circle": {"roi": [140, 120, 60, 60]},
    "align": {"roi": [50, 50, 540, 340]},
    "calibrate_scale": {"roi": [40, 50, 560, 60], "known_mm": 20.0},
    "presence": {"roi": [50, 50, 60, 60]},
    "count_template": {"roi": [30, 30, 580, 380], "template_roi": [52, 52, 40, 40],
                       "expect_count": 4},
    "count_blob": {"roi": [120, 120, 400, 90], "expect_count": 4},
    "color_classify": {"roi": [120, 120, 400, 90], "source": "blobs",
                       "expect": ["red", "green", "yellow", "blue"]},
    "led_glow": {"roi": [120, 120, 400, 90], "source": "blobs"},
    "defect_diff": {"roi": [40, 40, 560, 360]},
    "anomaly": {"roi": [40, 40, 560, 360]},
    "ocr": {"roi": [150, 280, 320, 60]},
    "ocv": {"roi": [150, 280, 320, 60], "expect_regex": "^[A-Z0-9]{6,12}$"},
    "code_read": {"roi": [150, 280, 320, 60]},
    "measure_length": {"roi": [10, 200, 620, 60]},
    "yolo_detect": {"roi": [40, 40, 560, 360]},
    "yolo_defect": {"roi": [40, 40, 560, 360]},
}

# Cannot pass without a trained model file or a printed code on the part.
_NEEDS_EXTERNAL = {"anomaly", "yolo_detect", "yolo_defect", "code_read",
                   "trained_surface",
                   "distance",          # measures between features other tools find
                   "intersect", "fit_shape",   # likewise
                   "results_analysis",  # needs an expression written for the job
                   "classify",          # needs a trained model
                   "fixture_npoint", "construct",      # need features from other tools
                   "calibrate_npoint",  # needs known point pairs
                   "data_analysis",     # needs a value to follow
                   "bead"}              # needs a path drawn along a bead


def _registry_types():
    from vision_station.tools import REGISTRY
    return sorted(REGISTRY)


@pytest.mark.parametrize("ttype", _registry_types())
def test_tool_works_with_schema_defaults(ttype):
    """Build every tool exactly as `add_tool` does and run it on a good part.

    This exists because of a bug that only showed up through the property
    sheet: OCV tested `expect_text is not None`, the sheet supplies an empty
    string, and the tool therefore demanded a blank part and rejected
    everything. Hand-written JSON recipes omitted the field and worked, and
    every test at the time used hand-written JSON. A tool that cannot pass a
    good part on stock settings is one nobody can use out of the box."""
    from vision_station.core.schema import defaults
    if ttype in ("ocr", "ocv") and not _has_ocr():
        pytest.skip("no OCR engine installed")
    img = _ui_scene()
    params = defaults(ttype)
    params.update(_UI_SETUP.get(ttype, {"roi": [40, 40, 560, 360]}))
    ctx = Context(reference=img)
    if ttype in ("color_classify", "led_glow"):
        make_tool({"id": "blobs", "type": "count_blob",
                   "params": dict(defaults("count_blob"),
                                  **_UI_SETUP["count_blob"])}).run(img, ctx)
    r = make_tool({"id": ttype, "type": ttype, "params": params}).run(img, ctx)
    if ttype in _NEEDS_EXTERNAL:
        assert r.status in ("pass", "fail", "error")
        return
    assert r.status == "pass", \
        "%s rejects a good part with stock settings: %s" % (ttype, r.message)


def test_color_swatches_match_realistic_leds():
    """The stock swatches must accept LEDs as a camera actually records them,
    not only pure primaries. Two earlier versions of this table failed in
    opposite directions -- estimated by eye and 66-74 LAB units off, then fully
    saturated and unreachable by any real LED. Both rejected good parts."""
    from vision_station.tools.color import DEFAULT_SWATCHES, _deltaE
    real = {"red": [(40, 40, 210), (55, 60, 190), (30, 25, 235)],
            "green": [(60, 190, 60), (80, 200, 90), (40, 175, 45)],
            "blue": [(210, 70, 50), (200, 90, 70), (230, 60, 35)],
            "yellow": [(40, 210, 215), (70, 205, 220), (30, 225, 235)],
            "white": [(235, 235, 235), (215, 220, 225)],
            "off": [(30, 30, 30), (45, 45, 48)]}
    for name, samples in real.items():
        for bgr in samples:
            lab = cv2.cvtColor(np.uint8([[list(bgr)]]), cv2.COLOR_BGR2LAB)[0, 0]
            best = min(((k, _deltaE(lab, v)) for k, v in DEFAULT_SWATCHES.items()),
                       key=lambda t: t[1])
            assert best[0] == name, "%s BGR%s named %s" % (name, bgr, best[0])
            assert best[1] <= 28.0, \
                "%s BGR%s scored dE %.1f, past the default acceptance" % (
                    name, bgr, best[1])


def test_colour_survives_a_blown_out_led_core():
    """A lit LED photographed at an exposure that shows anything else is
    normally clipped at its centre, reading (253, 249, 255) whatever colour it
    is. Sampling the blob median therefore called every lit LED white -- on a
    real switch faceplate, all thirteen of them, including obvious greens and
    reds. The colour is read from the unclipped halo instead."""
    from vision_station.tools.color import ColorClassify, DEFAULT_SWATCHES, _deltaE
    img = np.zeros((120, 120, 3), np.uint8)
    cv2.circle(img, (60, 60), 26, (60, 200, 60), -1)     # green halo
    cv2.circle(img, (60, 60), 12, (254, 250, 255), -1)   # clipped white core
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    blob = {"x": 48, "y": 48, "w": 24, "h": 24, "cx": 60, "cy": 60, "area": 576}

    # Through the tool, not the helper. Testing _representative on its own
    # passed even with the tool reverted to the plain median, because nothing
    # asserted the tool actually calls it.
    from vision_station.core.schema import defaults
    ctx = Context(reference=img)
    ctx.share["blobs"] = [blob]
    r, _ = _run("color_classify",
                dict(defaults("color_classify"), roi=[0, 0, 120, 120],
                     source="blobs", expect=[], deltaE_max=200), img, ctx=ctx)
    got = r.values["leds"][0]["color"]
    assert got != "white", "the clipped core was sampled instead of the halo"
    assert got == "green", "a green LED with a blown core was called %s" % got

    rep = ColorClassify._representative(img, lab, blob)
    assert rep is not None

    plain = np.median(lab[48:72, 48:72].reshape(-1, 3), axis=0)
    plain_name = min(((k, _deltaE(plain, v)) for k, v in DEFAULT_SWATCHES.items()),
                     key=lambda t: t[1])[0]
    assert plain_name == "white", \
        "the test image does not reproduce the clipping it is meant to"


def test_lightness_is_weighted_below_chroma():
    """Drive current moves L a long way while a and b barely shift, so an LED
    dimmed by half must not read as a different colour."""
    from vision_station.tools.color import _deltaE
    bright = cv2.cvtColor(np.uint8([[[60, 190, 60]]]), cv2.COLOR_BGR2LAB)[0, 0]
    dim = cv2.cvtColor(np.uint8([[[30, 95, 30]]]), cv2.COLOR_BGR2LAB)[0, 0]
    weighted = _deltaE(bright, dim)
    unweighted = _deltaE(bright, dim, l_weight=1.0)
    assert weighted < unweighted, "lightness weighting is not applied"


# ------------------------------------------------------ fixture ambiguity
def test_align_refuses_an_ambiguous_fixture(board):
    """A high score does not mean the right feature was found.

    Measured on a real chassis photograph: a small fixture region returned a
    confidently wrong pose at three of five offsets, scoring 0.64 to 0.68
    against a 0.45 threshold, because the feature it matched appears more than
    once on the panel. Publishing that pose moves every downstream region onto
    the wrong place, so the operator sees plausible results about the wrong
    part of the image -- worse than an honest failure."""
    from vision_station.core.schema import defaults
    # a panel carrying the same small motif four times over
    img = np.full((400, 600, 3), 60, np.uint8)
    for cx, cy in ((90, 90), (300, 90), (90, 300), (300, 300)):
        cv2.rectangle(img, (cx - 24, cy - 24), (cx + 24, cy + 24), (170, 170, 170), -1)
        cv2.circle(img, (cx, cy), 10, (40, 40, 40), -1)

    ctx = Context(reference=img)
    p = dict(defaults("align"), roi=[66, 66, 48, 48], method="template",
             min_score=0.4, angle_range=[0, 0])
    r = make_tool({"id": "a", "type": "align", "params": p}).run(img, ctx)
    assert r.status == "fail", \
        "a motif repeated four times was accepted as a fixture (score %.2f)" % r.score
    assert "unique" in r.message.lower()
    assert ctx.fixture is None, "an ambiguous match still published a fixture"


def test_align_accepts_a_distinctive_fixture(board):
    """The guard must not reject good fixtures."""
    from vision_station.core.schema import defaults
    ctx = Context(reference=board)
    h, w = board.shape[:2]
    p = dict(defaults("align"), roi=[40, 40, w - 80, h - 80], method="template",
             min_score=0.4, angle_range=[-4, 4], angle_step=1.0)
    r = make_tool({"id": "a", "type": "align", "params": p}).run(board, ctx)
    assert r.status == "pass", r.message
    assert ctx.fixture is not None
    assert r.values.get("uniqueness") is not None


# ------------------------------------------------- fixture propagation
def test_fixture_moves_downstream_regions_with_the_part():
    """The whole point of `Locate part` is that every other region follows the
    part. Alignment can be accurate and this can still be broken -- they are
    separate pieces of code -- so it gets its own test.

    A feature is placed at a known spot, the part is then shifted and rotated,
    and a presence check on that feature must still pass. Without fixturing its
    region stays put while the feature moves away, so a passing result here is
    real evidence the transform is being applied downstream."""
    ref = np.full((440, 640, 3), 38, np.uint8)
    cv2.rectangle(ref, (30, 30), (610, 410), (150, 142, 132), -1)
    for cx, cy in ((70, 70), (570, 70), (70, 370), (570, 370)):
        cv2.circle(ref, (cx, cy), 16, (205, 205, 210), -1)
        cv2.circle(ref, (cx, cy), 16, (70, 70, 75), 2)
    cv2.rectangle(ref, (300, 200), (360, 240), (40, 60, 200), -1)   # the feature
    cv2.putText(ref, "AB12345X", (150, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (245, 245, 245), 2)

    from vision_station.core.schema import defaults
    align_p = dict(defaults("align"), roi=[50, 50, 540, 340], method="template",
                   min_score=0.3, angle_range=[-10, 10], angle_step=1.0)
    pres_p = dict(defaults("presence"), roi=[295, 195, 70, 50], min_score=0.55,
                  search_pad=6)

    for dx, dy, deg in ((0, 0, 0.0), (22, -15, 0.0), (10, 8, 5.0), (-18, 12, -4.0)):
        M = cv2.getRotationMatrix2D((320.0, 220.0), deg, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        cur = cv2.warpAffine(ref, M, (640, 440), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

        ctx = Context(reference=ref)
        a = make_tool({"id": "a", "type": "align", "params": align_p}).run(cur, ctx)
        assert a.status == "pass", "align failed at %s,%s,%s" % (dx, dy, deg)
        with_fix = make_tool({"id": "p", "type": "presence",
                              "params": pres_p}).run(cur, ctx)
        assert with_fix.status == "pass", \
            "feature lost at dx=%s dy=%s deg=%s despite alignment: %s" % (
                dx, dy, deg, with_fix.message)

    # Control: with no fixture at all, a large shift must lose the feature.
    # If this passes too, the test above proves nothing.
    M = cv2.getRotationMatrix2D((320.0, 220.0), 0.0, 1.0)
    M[0, 2] += 90
    moved = cv2.warpAffine(ref, M, (640, 440), borderMode=cv2.BORDER_REPLICATE)
    bare = Context(reference=ref)
    unfixed = make_tool({"id": "p", "type": "presence",
                         "params": pres_p}).run(moved, bare)
    assert unfixed.status == "fail", \
        "presence passed a 90 px shift with no fixture; the test above is vacuous"


def test_align_failure_stops_the_rest_of_the_recipe(tmpproj):
    """Documented behaviour, never tested: if the part cannot be located, the
    engine stops instead of running every later tool against regions that
    landed on nothing. The value is diagnostic -- one honest failure tells an
    operator the part was not found, whereas eight bogus ones tell them
    nothing and bury the real cause."""
    root, pid, board = tmpproj
    proj = ProjectStore(root).get(pid)
    from vision_station.core.schema import defaults
    proj.spec["tools"] = [
        {"id": "locate", "type": "align",
         "params": dict(defaults("align"), roi=[40, 40, 200, 200],
                        min_score=0.99, angle_range=[0, 0])},
        {"id": "later", "type": "count_blob",
         "params": dict(defaults("count_blob"), roi=[30, 30, 300, 200],
                        expect_count=4)},
    ]
    proj.tools = [make_tool(t) for t in proj.spec["tools"]]
    noise = np.random.RandomState(3).randint(0, 255, board.shape).astype(np.uint8)
    res = proj.inspect(noise)
    ids = [t["tool_id"] for t in res["tools"]]
    assert "locate" in ids
    assert res["tools"][0]["status"] in ("fail", "error")
    assert "later" not in ids, \
        "downstream tools ran after the fixture failed: %s" % ids
    assert res["verdict"] != "PASS"


def test_fixture_also_moves_rotated_regions():
    """Rotated regions take a different path through the geometry code
    (mapped_quad, not mapped_roi), so fixturing them is separately breakable.
    Mutation testing found this uncovered: disabling the fixture transform for
    rotated regions left the whole suite green."""
    ref = np.full((440, 640, 3), 38, np.uint8)
    cv2.rectangle(ref, (30, 30), (610, 410), (150, 142, 132), -1)
    for cx, cy in ((70, 70), (570, 70), (70, 370), (570, 370)):
        cv2.circle(ref, (cx, cy), 16, (205, 205, 210), -1)
        cv2.circle(ref, (cx, cy), 16, (70, 70, 75), 2)
    cv2.circle(ref, (330, 220), 24, (40, 60, 200), -1)

    from vision_station.core.schema import defaults
    align_p = dict(defaults("align"), roi=[50, 50, 540, 340], method="template",
                   min_score=0.3, angle_range=[-8, 8], angle_step=1.0)
    # rotated AND elliptical, so both newer geometry features are exercised
    pres_p = dict(defaults("presence"), roi=[302, 192, 56, 56], min_score=0.5,
                  search_pad=6, angle=20.0, shape="ellipse")

    for dx, dy in ((0, 0), (24, -16), (-20, 14)):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        cur = cv2.warpAffine(ref, M, (640, 440), borderMode=cv2.BORDER_REPLICATE)
        ctx = Context(reference=ref)
        a = make_tool({"id": "a", "type": "align", "params": align_p}).run(cur, ctx)
        assert a.status == "pass", "align failed at %s,%s" % (dx, dy)
        r = make_tool({"id": "p", "type": "presence", "params": pres_p}).run(cur, ctx)
        assert r.status == "pass", \
            "rotated/elliptical region lost the feature at dx=%s dy=%s: %s" % (
                dx, dy, r.message)


# ------------------------------------------------------- defect classification
def _defect_panel(gain=1.0, shift=0):
    img = np.full((300, 400, 3), 150, np.uint8)
    cv2.rectangle(img, (40, 40), (360, 260), (175, 175, 175), -1)
    for i in range(4):
        cv2.circle(img, (90 + i * 70, 150), 18, (105, 105, 105), -1)
    return np.clip(img.astype(np.float32) * gain + shift, 0, 255).astype(np.uint8)


_DEFECT_P = {"roi": [20, 20, 360, 260], "threshold": 22, "illum_blur": 51,
             "open_px": 2, "min_area": 25, "scratch_elongation": 3.5,
             "max_defects": 0, "max_area_px": 0}


def test_defect_diff_separates_scratches_from_dents():
    """The tool labels each finding as a scratch or a dent from its aspect
    ratio. Operators sort by that label, so mixing them up sends someone to
    look for the wrong process fault."""
    ref = _defect_panel()
    scratched = _defect_panel()
    cv2.line(scratched, (110, 220), (300, 224), (70, 70, 70), 3)
    r, _ = _run("defect_diff", _DEFECT_P, scratched, reference=ref)
    kinds = [d["kind"] for d in r.values.get("defects", [])]
    assert "scratch" in kinds, "a long thin mark was not called a scratch: %s" % kinds

    dented = _defect_panel()
    cv2.circle(dented, (200, 90), 11, (60, 60, 60), -1)
    r2, _ = _run("defect_diff", _DEFECT_P, dented, reference=ref)
    kinds2 = [d["kind"] for d in r2.values.get("defects", [])]
    assert "dent" in kinds2, "a round mark was not called a dent: %s" % kinds2
    assert "scratch" not in kinds2, "a round mark was called a scratch"


def test_defect_diff_reports_size_in_mm_when_calibrated():
    """Defect size in pixels means nothing on a report. If calibration is
    present the tool must convert, otherwise a specification written in
    millimetres cannot be checked."""
    ref = _defect_panel()
    cur = _defect_panel()
    cv2.circle(cur, (200, 90), 12, (60, 60, 60), -1)
    ctx = Context(reference=ref)
    ctx.mm_per_px = 0.05
    r, _ = _run("defect_diff", _DEFECT_P, cur, ctx=ctx)
    defects = r.values.get("defects", [])
    assert defects, "no defect found"
    assert "length_mm" in defects[0], "calibration present but no mm reported"
    assert defects[0]["length_mm"] > 0


def test_defect_diff_uses_the_fixture_to_align_the_reference():
    """When the part has moved, the reference must be warped into its pose
    before differencing. Skip that and the whole part reads as one enormous
    defect -- a guaranteed false reject on any line where parts are not placed
    to the pixel."""
    ref = _defect_panel()
    M = np.float32([[1, 0, 14], [0, 1, 9]])
    moved = cv2.warpAffine(ref, M, (400, 300), borderMode=cv2.BORDER_REPLICATE)

    fixed = Context(reference=ref)
    fixed.fixture = np.float32([[1, 0, 14], [0, 1, 9], [0, 0, 1]])
    with_fix, _ = _run("defect_diff", _DEFECT_P, moved, ctx=fixed)

    without, _ = _run("defect_diff", _DEFECT_P, moved, reference=ref)
    assert with_fix.status == "pass", \
        "a correctly fixtured shift was reported as a defect: %s" % with_fix.message
    assert without.status == "fail", \
        "an unfixtured 14 px shift produced no difference; the test is vacuous"


# ------------------------------------------------ orientation and polygons
def _rotated_text(text="AB12345X", angle=0.0, size=560):
    img = np.full((size, size), 30, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    cv2.putText(img, text, ((size - tw) // 2, (size + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, 245, 3, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle, 1.0)
    return cv2.cvtColor(cv2.warpAffine(img, M, (size, size), borderValue=30),
                        cv2.COLOR_GRAY2BGR)


@ocr_required
@pytest.mark.parametrize("angle", [0, 25, 45, 90, 135, 180, 270, -40])
def test_text_reads_at_any_orientation(angle):
    """Parts do not arrive aligned to the camera. Plain OCR collapsed past
    about five degrees: ten degrees returned nothing at all, forty-five
    degrees nothing again. The tool now estimates the skew, corrects it, and
    tries the four right-angle orientations, because deskew fixes the baseline
    but cannot tell upright text from upside-down."""
    from vision_station.core.schema import defaults
    img = _rotated_text(angle=angle)
    p = dict(defaults("ocv"), roi=[0, 0, 560, 560],
             expect_regex="^[A-Z]{2}[0-9]{5}[A-Z]$", expect_text="",
             min_confidence=0.0)
    r, _ = _run("ocv", p, img)
    assert r.status == "pass", \
        "text at %d deg read as %r" % (angle, r.values.get("text"))


@ocr_required
def test_the_orientation_used_is_recorded():
    """Traceability: a logged result must say how the image had to be turned,
    or an operator cannot tell a misfeed from a misread."""
    from vision_station.core.schema import defaults
    r, _ = _run("ocr", dict(defaults("ocr"), roi=[0, 0, 560, 560],
                            min_confidence=0.0), _rotated_text(angle=180))
    assert r.values.get("quarter_turn") == 180
    assert "skew_deg" in r.values
    side, _ = _run("ocr", dict(defaults("ocr"), roi=[0, 0, 560, 560]), _rotated_text(angle=90))
    assert side.values.get("quarter_turn") in (90, 270) and side.values["text"] == "AB12345X"


def test_polygon_region_excludes_what_a_rectangle_would_include():
    """A part sitting at an angle cannot be bounded by a rectangle without
    dragging in whatever is beside it. The diamond here has junk in each corner
    of its bounding box -- exactly the situation a polygon exists for."""
    img = np.zeros((300, 300, 3), np.uint8)
    cv2.fillPoly(img, [np.int32([[60, 150], [150, 60], [240, 150], [150, 240]])],
                 (180, 180, 180))
    for cx, cy in ((70, 70), (230, 70), (70, 230), (230, 230)):
        cv2.circle(img, (cx, cy), 12, (255, 255, 255), -1)
    ctx = Context(reference=img)
    poly = [[60, 150], [150, 60], [240, 150], [150, 240]]

    def bright(patch):
        g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(g, 240, 255, cv2.THRESH_BINARY)
        n, _l, st, _c = cv2.connectedComponentsWithStats(th)
        return sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 30)

    rect, _ = Probe("r", {"roi": [60, 60, 180, 180], "shape": "rect"}).crop(img, ctx)
    pg, _ = Probe("p", {"roi": [60, 60, 180, 180], "shape": "polygon",
                        "points": poly}).crop(img, ctx)
    assert bright(rect) == 4
    assert bright(pg) == 0, "the polygon let corner content through"


def test_polygon_outline_follows_the_fixture():
    """Like every other region, a polygon has to move with the part."""
    img = np.zeros((300, 300, 3), np.uint8)
    poly = [[60, 150], [150, 60], [240, 150], [150, 240]]
    t = Probe("p", {"roi": [60, 60, 180, 180], "shape": "polygon", "points": poly})
    plain = Context(reference=img)
    shifted = Context(reference=img)
    shifted.fixture = np.float32([[1, 0, 25], [0, 1, -10], [0, 0, 1]])
    a = t.polygon_points(plain)
    b = t.polygon_points(shifted)
    assert np.allclose(b - a, np.float32([25, -10])), \
        "the polygon did not follow the fixture"
    g = t.roi_graphic(shifted, "pass")[0]
    assert g.kind == "poly" and len(g.points) == 4
    assert g.absolute is True


# ---------------------------------------------------------------- polygon
def _qt_available():
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PySide6  # noqa: F401
        return True
    except ImportError:
        return False


qt_required = pytest.mark.skipif(not _qt_available(), reason="PySide6 unavailable")


def test_polygon_region_excludes_what_a_rectangle_would_catch():
    """The reason polygons exist. A component sitting at an angle cannot be
    tightly bounded by a rectangle, so the box drags in whatever is next to it
    -- on a dense board that is another component."""
    img = np.zeros((300, 400, 3), np.uint8)
    box = cv2.boxPoints(((200, 150), (220, 70), 25)).astype(np.int32)
    cv2.fillPoly(img, [box], (190, 190, 190))
    cv2.circle(img, (60, 40), 16, (255, 255, 255), -1)      # neighbours
    cv2.circle(img, (350, 260), 16, (255, 255, 255), -1)
    ctx = Context(reference=img)

    def bright(patch):
        g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        _, t = cv2.threshold(g, 235, 255, cv2.THRESH_BINARY)
        n, _, st, _ = cv2.connectedComponentsWithStats(t)
        return sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 40)

    rect = Probe("r", {"roi": [60, 40, 300, 240], "shape": "rect"})
    poly = Probe("p", {"shape": "polygon",
                       "points": [[float(a), float(b)] for a, b in box]})
    assert bright(rect.crop(img, ctx)[0]) == 2
    assert bright(poly.crop(img, ctx)[0]) == 0, \
        "the polygon let in content a rectangle would have caught"
    assert poly._crop_mask is not None


def test_polygon_follows_the_fixture(board):
    """A polygon is stored in absolute image coordinates, so it must still be
    moved by the fixture when the part shifts."""
    ctx = Context(reference=board)
    pts = [[100.0, 100.0], [200.0, 90.0], [210.0, 160.0], [95.0, 170.0]]
    t = Probe("p", {"shape": "polygon", "points": pts})
    plain = t.polygon_points(ctx)
    ctx.fixture = np.float32([[1, 0, 25], [0, 1, -12], [0, 0, 1]])
    moved = t.polygon_points(ctx)
    assert np.allclose(moved - plain, [25, -12], atol=0.01), \
        "polygon vertices ignored the fixture transform"


def test_polygon_outline_is_reported_as_a_polygon(board):
    ctx = Context(reference=board)
    t = Probe("p", {"shape": "polygon",
                    "points": [[10.0, 10.0], [80.0, 20.0], [60.0, 90.0]]})
    g = t.roi_graphic(ctx, "pass")[0]
    assert g.kind == "poly" and len(g.points) == 3
    assert g.absolute, "the outline would be transformed a second time"


def test_polygon_roi_reports_its_bounding_box(board):
    """Anything that reads mapped_roi -- diagnostics, the property sheet, the
    results overlay -- must get something sensible for a polygon too."""
    ctx = Context(reference=board)
    t = Probe("p", {"shape": "polygon",
                    "points": [[30.0, 40.0], [130.0, 45.0], [120.0, 140.0]]})
    x, y, w, h = t.mapped_roi(ctx)
    assert (x, y) == (30, 40)
    assert 95 <= w <= 101 and 95 <= h <= 101


def _mk_polygon(w, pts=((60, 55), (115, 90), (60, 135), (15, 90)),
                ttype="defect_diff"):
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtTest import QTest
    c = w.canvas
    c.resetTransform()
    i = w.newTypeBox.findData(ttype)
    w.newTypeBox.setCurrentIndex(max(0, i))
    w.set_tool("draw_polygon")
    for pt in pts:
        p = c.mapFromScene(QPointF(*pt))
        QTest.mousePress(c.viewport(), Qt.LeftButton, Qt.NoModifier, p)
        QTest.mouseRelease(c.viewport(), Qt.LeftButton, Qt.NoModifier, p)
    QTest.keyClick(c, Qt.Key_Return)
    tid = w.spec["tools"][-1]["id"]
    c.select_ids([tid])
    return tid, c.regions[tid]


def _drag(c, x0, y0, x1, y1, steps=5):
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtTest import QTest
    QTest.mousePress(c.viewport(), Qt.LeftButton, Qt.NoModifier,
                     c.mapFromScene(QPointF(x0, y0)))
    for i in range(1, steps + 1):
        QTest.mouseMove(c.viewport(), c.mapFromScene(
            QPointF(x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps)))
    QTest.mouseRelease(c.viewport(), Qt.LeftButton, Qt.NoModifier,
                       c.mapFromScene(QPointF(x1, y1)))


@qt_required
def test_dragging_a_rectangle_works_with_a_polygon_present(ui):
    """Snapping asked every region for .rect(), which a polygon does not have.
    Any recipe holding both a traced region and a rectangle therefore crashed
    the moment the rectangle was dragged."""
    w, _tmp, _img = ui
    c = w.canvas
    _mk_polygon(w)
    w.set_tool("draw_rect")
    _drag(c, 300, 250, 420, 320)
    tid = w.spec["tools"][-1]["id"]
    c.select_ids([tid])
    rect = c.regions[tid]
    before = rect.roi()
    _drag(c, before[0] + before[2] / 2, before[1] + before[3] / 2,
          before[0] + before[2] / 2 + 60, before[1] + before[3] / 2 + 40)
    after = rect.roi()
    assert abs((after[0] - before[0]) - 60) < 10, \
        "rectangle did not move with a polygon on the canvas"
    w.dirty = False


@qt_required
def test_polygon_scales_from_its_corner_handles(ui):
    """A traced region must be resizable as a whole, not only vertex by
    vertex. The corner handles sit outside the outline on any non-rectangular
    shape -- on a diamond all four are in empty space -- so the item's hit area
    has to widen when selected or they cannot be grabbed at all."""
    w, _tmp, _img = ui
    c = w.canvas
    _tid, item = _mk_polygon(w)
    before = item.roi()
    _drag(c, before[0] + before[2], before[1] + before[3],
          before[0] + before[2] + 40, before[1] + before[3] + 30)
    after = item.roi()
    assert after[2] > before[2] + 15, \
        "corner handle did not scale the polygon (%s -> %s)" % (before, after)
    assert after[3] > before[3] + 10
    assert len(item.points()) == 4, "scaling changed the vertex count"
    w.dirty = False


@qt_required
def test_polygon_vertices_still_edit_individually(ui):
    w, _tmp, _img = ui
    c = w.canvas
    tid, item = _mk_polygon(w)
    pts = item.points()
    v = pts[0]
    _drag(c, v[0], v[1], v[0] + 22, v[1] - 16)
    assert item.points()[0] != pts[0], "a vertex handle no longer moves a vertex"
    recipe = [t for t in w.spec["tools"] if t["id"] == tid][0]
    assert len(recipe["params"]["points"]) == 4
    assert recipe["params"]["points"][0] != pts[0], \
        "the edited vertex did not reach the recipe"
    w.dirty = False


def _centre(it):
    r = it.roi()
    return (r[0] + r[2] / 2.0, r[1] + r[3] / 2.0)


def _drag_mod(c, x0, y0, x1, y1, mod, steps=5):
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtTest import QTest
    QTest.mousePress(c.viewport(), Qt.LeftButton, mod, c.mapFromScene(QPointF(x0, y0)))
    for i in range(1, steps + 1):
        QTest.mouseMove(c.viewport(), c.mapFromScene(
            QPointF(x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps)))
    QTest.mouseRelease(c.viewport(), Qt.LeftButton, mod, c.mapFromScene(QPointF(x1, y1)))


@qt_required
def test_ellipse_is_edited_by_its_centre_and_radii(ui):
    """VisionPro-style: four handles on the curve, and dragging one changes that
    radius with the centre fixed. The old box-style editing put handles on empty
    corners and moved the centre whenever a side was dragged."""
    from PySide6.QtCore import Qt
    w, _tmp, _img = ui
    c = w.canvas
    c.resetTransform()
    w.set_tool("draw_ellipse")
    _drag(c, 200, 150, 320, 230)
    e = c.regions[w.spec["tools"][-1]["id"]]
    c.select_ids([e.tool_id])
    assert sorted(e._handles()) == ["bm", "ml", "mr", "tm"]
    r, c0 = e.roi(), _centre(e)
    _drag(c, r[0] + r[2], r[1] + r[3] / 2, r[0] + r[2] + 40, r[1] + r[3] / 2)
    r2, c1 = e.roi(), _centre(e)
    assert abs(r2[2] - (r[2] + 80)) <= 2, "radius did not grow by the drag (%s)" % r2
    assert abs(c0[0] - c1[0]) <= 1 and abs(c0[1] - c1[1]) <= 1, \
        "the centre moved from %s to %s" % (c0, c1)
    w.dirty = False


@qt_required
def test_shift_makes_an_ellipse_a_circle(ui):
    """Checked on a case where Shift changes the answer: without it the result
    would be 140 x 80. Modifiers held at the start of a drag were being dropped
    part way through it."""
    from PySide6.QtCore import Qt
    w, _tmp, _img = ui
    c = w.canvas
    c.resetTransform()
    w.set_tool("draw_ellipse")
    _drag(c, 200, 150, 320, 230)
    e = c.regions[w.spec["tools"][-1]["id"]]
    c.select_ids([e.tool_id])
    r = e.roi()
    _drag_mod(c, r[0] + r[2], r[1] + r[3] / 2, r[0] + r[2] + 10, r[1] + r[3] / 2,
              Qt.ShiftModifier)
    r2 = e.roi()
    assert abs(r2[2] - r2[3]) <= 2, "Shift did not produce a circle: %s" % r2
    w.dirty = False


@qt_required
def test_alt_resizes_a_rectangle_from_its_centre(ui):
    from PySide6.QtCore import Qt
    w, _tmp, _img = ui
    c = w.canvas
    c.resetTransform()
    w.set_tool("draw_rect")
    _drag(c, 420, 150, 520, 220)
    rc = c.regions[w.spec["tools"][-1]["id"]]
    c.select_ids([rc.tool_id])
    r, c0 = rc.roi(), _centre(rc)
    _drag_mod(c, r[0] + r[2], r[1] + r[3], r[0] + r[2] + 30, r[1] + r[3] + 20,
              Qt.AltModifier)
    r2, c1 = rc.roi(), _centre(rc)
    assert abs(r2[2] - (r[2] + 60)) <= 2, "Alt did not grow both sides: %s -> %s" % (r, r2)
    assert abs(c0[0] - c1[0]) <= 1 and abs(c0[1] - c1[1]) <= 1
    # and without Alt the opposite corner stays put, as before
    r, c0 = rc.roi(), _centre(rc)
    _drag(c, r[0] + r[2], r[1] + r[3], r[0] + r[2] + 20, r[1] + r[3] + 10)
    r3 = rc.roi()
    assert (r3[0], r3[1]) == (r[0], r[1]), "a plain drag moved the anchored corner"
    w.dirty = False


@qt_required
def test_every_region_type_moves_and_resizes(ui):
    """The three shapes must behave the same way: free to move, free to
    resize."""
    w, _tmp, _img = ui
    c = w.canvas
    c.resetTransform()
    for tool, geom in (("draw_rect", (300, 120, 400, 190)),
                       ("draw_ellipse", (300, 220, 400, 290))):
        w.set_tool(tool)
        _drag(c, *geom)
        tid = w.spec["tools"][-1]["id"]
        c.select_ids([tid])
        it = c.regions[tid]
        b = it.roi()
        _drag(c, b[0] + b[2] / 2, b[1] + b[3] / 2,
              b[0] + b[2] / 2 + 40, b[1] + b[3] / 2 + 25)
        moved = it.roi()
        assert abs((moved[0] - b[0]) - 40) < 10, "%s did not move" % tool
        if tool == "draw_ellipse":
            # an ellipse's handles are on the curve, at the ends of its axes;
            # its bounding-box corners are deliberately not handles
            _drag(c, moved[0] + moved[2], moved[1] + moved[3] / 2,
                  moved[0] + moved[2] + 35, moved[1] + moved[3] / 2)
        else:
            _drag(c, moved[0] + moved[2], moved[1] + moved[3],
                  moved[0] + moved[2] + 35, moved[1] + moved[3] + 25)
        grown = it.roi()
        assert grown[2] > moved[2] + 12, "%s did not resize" % tool
    w.dirty = False


@qt_required
def test_ctrl_z_takes_back_the_last_polygon_point(ui):
    """Undo during a trace removes the last point placed.

    It used to go straight to the recipe-snapshot history, which knows nothing
    about a trace in progress, so the keystroke did nothing visible and the
    only way out of a misplaced corner was to abandon the whole polygon."""
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtTest import QTest
    w, _tmp, _img = ui
    c = w.canvas
    c.resetTransform()
    n0 = len(w.spec["tools"])
    w.set_tool("draw_polygon")
    for pt in ((60, 55), (115, 90), (60, 135), (15, 90)):
        p = c.mapFromScene(QPointF(*pt))
        QTest.mousePress(c.viewport(), Qt.LeftButton, Qt.NoModifier, p)
        QTest.mouseRelease(c.viewport(), Qt.LeftButton, Qt.NoModifier, p)
    assert len(c._poly_pending) == 4

    QTest.keyClick(c, Qt.Key_Z, Qt.ControlModifier)
    assert len(c._poly_pending) == 3, "Ctrl+Z did not remove a point"
    w.undo()                       # the menu action must behave the same way
    assert len(c._poly_pending) == 2

    assert len(w.spec["tools"]) == n0, \
        "undo during a trace altered the recipe instead of the trace"

    for _ in range(2):
        w.undo()
    assert not c._poly_pending
    assert c._poly_preview is None, "the preview outline was left behind"
    w.set_tool("select")


@qt_required
def test_inspection_does_not_block_the_interface(ui):
    """Inspection runs on a worker thread.

    Running it inline froze the application for the length of every cycle: with
    a 280 ms recipe a 50 ms probe timer was serviced only every ~300 ms, so
    clicks, repaints and menu input all stalled. That is the lag -- the
    rendering itself measures well under a millisecond."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    w, _tmp, _img = ui
    ticks = []
    probe = QTimer()
    probe.setInterval(20)
    probe.timeout.connect(lambda: ticks.append(time.perf_counter()))
    probe.start()
    # The first inspection after start-up loads the text reader's networks,
    # once; onnxruntime holds the interpreter while it builds a session. What
    # is measured here is every inspection after that.
    w.run_once()
    warm = time.perf_counter() + 30.0
    while w._worker is not None and time.perf_counter() < warm:
        QApplication.processEvents()
        time.sleep(0.002)
    ticks.clear()
    started = time.perf_counter()
    w.run_once()
    assert w._worker is not None, "run_once did not dispatch to a worker"
    # the interface must be serviced while the inspection is in flight
    deadline = time.perf_counter() + 5.0
    while w._worker is not None and time.perf_counter() < deadline:
        QApplication.processEvents()
        time.sleep(0.002)
    took = time.perf_counter() - started
    probe.stop()
    assert w.last is not None, "no result arrived from the worker"
    # With text read on the GPU the whole recipe can finish inside two probe
    # periods, so the count expected follows how long the inspection took.
    expected = min(3, int(took / 0.05))
    assert len(ticks) >= expected, \
        "the event loop was serviced only %d times during a %.0f ms inspection" % (
            len(ticks), took * 1000)
    w.dirty = False


@pytest.fixture
def job_ui(tmp_path, monkeypatch):
    """The application opened on a two-side job."""
    _qt_available()
    from PySide6.QtWidgets import QApplication, QMessageBox
    from vision_station.ui import classic_window as appmod
    (tmp_path / "projects").mkdir()
    job, ref = _mini_job(tmp_path / "projects", 2)
    os.rename(str(tmp_path / "projects" / "job"), str(tmp_path / "projects" / "chassis"))
    monkeypatch.setattr(appmod, "HERE", str(tmp_path))
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Discard))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    QApplication.instance() or QApplication([])
    w = appmod.MainWindow()
    w.projBox.setCurrentText("chassis/side0")
    yield w, ref
    w.dirty = False
    w.close()


@qt_required
def test_job_tree_lists_every_job_with_image_source_and_tools(job_ui):
    """Application -> Jobs -> Image source and tools, the way VisionPro shows a
    job: the image source first, because everything below depends on which
    image arrives. Every job is listed, so the whole application is visible."""
    w, _ref = job_ui
    root = w.tree.topLevelItem(0)
    assert root.text(0) == "Test job"
    sides = [root.child(i) for i in range(root.childCount())]
    assert [x.data(0, w.SIDE_ROLE) for x in sides] == ["chassis/side0", "chassis/side1"]

    kids = [sides[0].child(i) for i in range(sides[0].childCount())]
    assert kids[0].text(0) == "Image source"
    assert kids[0].data(0, w.SRC_ROLE) == "chassis/side0"
    assert [k.text(0) for k in kids[1:]] == ["fixture  (fixture)", "surface"]
    assert sides[1].childCount() == 1, "an unopened job should still be expandable"

    # canvas selection still finds tools, now two levels down
    w.canvas.select_ids(["surface"])
    w.on_canvas_selection(["surface"])
    assert [it.text(0) for it in w.tree.selectedItems()] == ["surface"]


@qt_required
def test_toolbox_lists_every_tool_and_adds_one(job_ui):
    """The Toolbox shows what the application can do. Tools were only in a menu,
    which hides the catalogue."""
    from vision_station.core.schema import GROUPS
    w, _ref = job_ui
    from PySide6.QtCore import Qt
    groups = [w.toolbox.topLevelItem(i) for i in range(w.toolbox.topLevelItemCount())]
    assert [g.text(0) for g in groups] == [name for name, _ in GROUPS]
    listed = set()
    for g in groups:
        for i in range(g.childCount()):
            listed.add(g.child(i).data(0, Qt.UserRole))
    from vision_station.tools import REGISTRY
    from vision_station.core.schema import LEGACY_TYPES
    assert listed == set(REGISTRY) - set(LEGACY_TYPES), \
        "the Toolbox and the tool registry disagree"

    n = len(w.spec["tools"])
    hist = next(g.child(i) for g in groups for i in range(g.childCount())
                if g.child(i).data(0, Qt.UserRole) == "histogram")
    w._toolbox_add(hist)
    assert len(w.spec["tools"]) == n + 1
    assert w.spec["tools"][-1]["type"] == "histogram"
    w.dirty = False


@qt_required
def test_clicking_a_side_opens_it(job_ui):
    w, _ref = job_ui
    root = w.tree.topLevelItem(0)
    w.on_tree_clicked(root.child(1), 0)
    assert w.current_pid() == "chassis/side1"
    assert w.project.folder.endswith(os.path.join("sides", "side1"))


@qt_required
def test_switching_model_offers_to_keep_unsaved_edits(job_ui, monkeypatch):
    """Choosing another model used to load it straight away and silently
    discard unsaved edits."""
    from PySide6.QtWidgets import QMessageBox
    w, _ref = job_ui
    w.dirty = True
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Cancel))
    w.projBox.setCurrentText("chassis/side1")
    assert w.current_pid() == "chassis/side0", "cancel did not keep the current side"
    assert w.project.folder.endswith(os.path.join("sides", "side0"))


@qt_required
def test_training_panel_adds_images_and_trains(job_ui):
    w, ref = job_ui
    w.canvas.select_ids(["surface"])
    w.refresh_training()
    assert w.trainBtn.isEnabled()
    w.canvas.set_image(ref, keep_view=False)
    w.add_training("ok")
    w.canvas.set_image(_surface_part(dx=8, dy=-5, deg=1), keep_view=False)
    w.add_training("ok")
    w.canvas.set_image(_surface_part(seed=1, scratch=True), keep_view=False)
    w.add_training("ng")
    assert w.trainLists["ok"].count() == 2 and w.trainLists["ng"].count() == 1
    w.canvas.select_ids(["surface"])
    w.train_selected()
    assert "Threshold" in w.trainReport.text(), w.trainReport.text()
    model = [t for t in w.spec["tools"] if t["id"] == "surface"][0]["params"]["model"]
    assert os.path.exists(os.path.join(w.project.folder, model))

    w.canvas.set_image(_surface_part(seed=3, dx=12, dy=8, deg=2, scratch=True),
                       keep_view=False)
    w.run_once()
    w.wait_for_inspection()
    res = {t["tool_id"]: t for t in w.last["tools"]}
    assert res["surface"]["status"] == "fail", "the trained tool passed a scratched part"


@qt_required
def test_training_panel_adapts_to_the_kind_of_tool(job_ui):
    """Every tool can be tested against OK/NG images; a score-threshold tool can
    have its threshold tuned from them; only a trained tool trains."""
    w, _ref = job_ui
    w.canvas.select_ids(["fixture"])
    w.refresh_training()
    assert w.testBtn.isEnabled()
    assert w.trainBtn.isEnabled() and w.trainBtn.text() == "Tune threshold"
    w.canvas.select_ids(["surface"])
    w.refresh_training()
    assert w.trainBtn.text() == "Train"


def _presence_side(tmp_path):
    """A side with a presence check on the panel's white label."""
    from vision_station.core.job import create_job
    from vision_station.core.schema import defaults
    job = create_job(str(tmp_path / "pj"), "Presence job")
    ref = _surface_part()
    tools = [{"id": "label", "type": "presence",
              "params": dict(defaults("presence"), roi=[90, 250, 160, 50],
                             min_score=0.2, search_pad=20)}]
    return job.add_side("s", "S", ref, tools), ref


def _no_label(img):
    out = img.copy()
    out[245:305, 85:255] = (150, 150, 150)
    return out


def test_presence_in_a_featureless_region_detects_absence():
    """A presence region drawn wholly inside a plain sticker -- the natural way
    to draw it -- scored 1.000 against anything, because normalised
    correlation divides by the template's variance and a plain patch has none.
    A missing sticker passed. Such regions are now judged on brightness."""
    from vision_station.core.schema import defaults
    ref = np.full((200, 300, 3), 150, np.uint8)
    cv2.rectangle(ref, (60, 60), (240, 140), (235, 235, 235), -1)
    params = dict(defaults("presence"), roi=[80, 80, 140, 40], min_score=0.6,
                  search_pad=15)
    present, _ = _run("presence", params, ref.copy(), reference=ref)
    missing = np.full_like(ref, 150)
    gone, _ = _run("presence", params, missing, reference=ref)
    clutter = np.random.default_rng(0).integers(0, 255, ref.shape).astype(np.uint8)
    noise, _ = _run("presence", params, clutter, reference=ref)
    assert present.status == "pass"
    assert gone.status == "fail", "a missing sticker was reported present (%.3f)" % gone.score
    assert noise.status == "fail", "random clutter was reported present (%.3f)" % noise.score
    assert present.values["mode"] == "brightness"


def test_presence_on_a_textured_region_still_matches_patterns():
    from vision_station.core.schema import defaults
    ref = _surface_part()
    r, _ = _run("presence", dict(defaults("presence"), roi=[100, 130, 250, 45],
                                 search_pad=15), ref, reference=ref)
    assert r.values["mode"] == "pattern" and r.status == "pass"


def test_evaluation_counts_false_rejects_and_escapes(tmp_path):
    """An OK that fails is a false reject, an NG that passes an escape. With a
    threshold no part can reach, the good parts fail and the report has to
    say so. A Part present check's own NG examples are rejected whatever its
    threshold -- it learns from them -- so with a threshold that passes
    everything they are still caught."""
    side, ref = _presence_side(tmp_path)
    side.add_training_image("label", ref, "ok")
    side.add_training_image("label", _surface_part(dx=6, dy=-4), "ok")
    side.add_training_image("label", _no_label(ref), "ng")
    side.add_training_image("label", _no_label(_surface_part(dx=-5, dy=3)), "ng")
    spec = [t for t in side.spec["tools"] if t["id"] == "label"][0]
    spec["params"]["min_score"] = 2.0          # no part can reach it
    side.save(side.spec)
    ev = side.evaluate_tool("label")
    assert ev["false_rejects"] == 2 and ev["escapes"] == 0
    assert not ev["all_correct"]
    spec["params"]["min_score"] = -1.0         # passes everything ...
    side.save(side.spec)
    ev = side.evaluate_tool("label")
    assert ev["escapes"] == 0, "... but its own NG examples are still rejected"


def test_tuning_sets_the_threshold_between_ok_and_ng(tmp_path):
    """Measured on the real chassis the hand-picked 0.55 sat much nearer the NG
    scores than the OK ones; tuning moved it to the midpoint."""
    side, ref = _presence_side(tmp_path)
    side.add_training_image("label", ref, "ok")
    side.add_training_image("label", _surface_part(dx=6, dy=-4), "ok")
    side.add_training_image("label", _no_label(ref), "ng")
    rep = side.tune_threshold("label")
    assert rep["separates"]
    assert rep["highest_ng"] < rep["new"] < rep["lowest_ok"]
    assert abs(rep["new"] - (rep["lowest_ok"] + rep["highest_ng"]) / 2) < 1e-3
    ev = side.evaluate_tool("label")
    assert ev["all_correct"], ev


def test_tuning_refuses_when_ok_and_ng_overlap(tmp_path):
    """No threshold separates an NG image that looks exactly like an OK one;
    the tool must say so rather than invent a number."""
    side, ref = _presence_side(tmp_path)
    side.add_training_image("label", ref, "ok")
    side.add_training_image("label", _surface_part(dx=3), "ng")     # not really bad
    before = [t for t in side.spec["tools"] if t["id"] == "label"][0]["params"]["min_score"]
    rep = side.tune_threshold("label")
    assert not rep["separates"]
    assert rep["new"] == before, "the threshold was changed despite no separation"


@qt_required
def test_test_button_marks_each_image_right_or_wrong(job_ui):
    w, ref = job_ui
    w.canvas.select_ids(["surface"])
    w.refresh_training()
    w.canvas.set_image(ref, keep_view=False)
    w.add_training("ok")
    w.canvas.set_image(_surface_part(seed=1, scratch=True), keep_view=False)
    w.add_training("ng")
    w.canvas.select_ids(["surface"])
    w.train_selected()
    w.canvas.select_ids(["surface"])
    w.test_selected()
    assert len(w._eval_marks) == 2
    assert all(w._eval_marks.values()), "a correctly judged image was marked wrong"
    assert "OK 1/1 passed" in w.trainReport.text() and "NG 1/1 rejected" in w.trainReport.text()


def test_no_keyboard_shortcut_has_two_owners(ui):
    """Qt fires neither action of an ambiguous shortcut. The toolbar
    reorganisation gave ten keys two owners -- F5, F6, Ctrl+S, Ctrl+O, Ctrl+Y,
    F7-F9, V, H -- and every one of them silently did nothing."""
    from PySide6.QtGui import QAction
    from collections import defaultdict
    w, _tmp, _img = ui
    owners = defaultdict(list)
    for a in w.findChildren(QAction):
        for ks in a.shortcuts():
            if ks.toString():
                owners[ks.toString()].append(a.text() or "(unnamed)")
    clashes = {k: v for k, v in owners.items() if len(v) > 1}
    assert not clashes, "ambiguous shortcuts - Qt will fire none of them: %s" % clashes


@qt_required
def test_the_main_keys_actually_fire(ui):
    """Pressed, not inspected: the collisions above were invisible to every test
    that called the methods directly."""
    from PySide6.QtGui import QAction
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    w, _tmp, _img = ui
    w.show()
    w.activateWindow()
    QApplication.processEvents()
    fired = []
    for a in w.findChildren(QAction):
        if a.text() in ("Run once", "Save job"):
            a.triggered.connect(lambda *_, t=a.text(): fired.append(t))
    QTest.keySequence(w, "F5")
    QApplication.processEvents()
    w.wait_for_inspection()
    QTest.keySequence(w, "Ctrl+S")
    QApplication.processEvents()
    assert "Run once" in fired, "F5 did nothing"
    assert "Save job" in fired, "Ctrl+S did nothing"
    QTest.keySequence(w, "H")
    QApplication.processEvents()
    assert w.canvas.mode == "hand", "H did not pick the hand tool"
    QTest.keySequence(w, "V")
    QApplication.processEvents()
    assert w.canvas.mode == "select"
    assert getattr(w, "mode", "build") == "build", "a tool key changed Build/Run mode"
    w.dirty = False


def _drag_region(c, tid, dx, dy):
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtTest import QTest
    c.resetTransform()
    r = c.regions[tid].roi()
    cx, cy = r[0] + r[2] / 2.0, r[1] + r[3] / 2.0
    QTest.mousePress(c.viewport(), Qt.LeftButton, Qt.NoModifier, c.mapFromScene(QPointF(cx, cy)))
    for i in range(1, 6):
        QTest.mouseMove(c.viewport(), c.mapFromScene(QPointF(cx + dx * i / 5.0, cy + dy * i / 5.0)))
    QTest.mouseRelease(c.viewport(), Qt.LeftButton, Qt.NoModifier,
                       c.mapFromScene(QPointF(cx + dx, cy + dy)))


def _roi(w, tid):
    return [t for t in w.spec["tools"] if t["id"] == tid][0]["params"]["roi"]


@qt_required
def test_double_clicking_a_tool_opens_its_own_editor_window(ui):
    """VisionPro's Tool Edit Control: the tool in its own window, with only its
    region and its results. Asking again brings the same window forward."""
    from PySide6.QtCore import Qt
    w, _tmp, _img = ui
    w.run_once()
    w.wait_for_inspection()
    item = next(it for it in w._tool_items()
                if w.spec["tools"][it.data(0, Qt.UserRole)]["id"] == "lot_code")
    w.on_tree_double_clicked(item, 0)
    ed = w._editors.get("lot_code")
    assert ed is not None and ed.isVisible()
    assert list(ed.canvas.regions) == ["lot_code"], "the editor shows other tools' regions"
    assert ed.verdict.text() in ("PASS", "FAIL")
    assert ed.results.rowCount() > 1, "the editor shows no measurements"
    w.on_tree_double_clicked(item, 0)
    assert len(w._editors) == 1, "a second window opened for the same tool"
    ed.close()
    assert "lot_code" not in w._editors
    w.dirty = False


@qt_required
def test_tool_editor_opens_framed_on_its_tool(ui):
    """An editor for one tool opens on that tool, not on the whole frame --
    in a full chassis photograph the hole it was about was a speck."""
    w, _tmp, img = ui
    w.show()
    ed = w.open_tool_editor("lot_code")               # a small region
    ed.resize(1000, 700)
    ed.frame_tool()
    c = ed.canvas
    it = c.regions["lot_code"]
    region = it.mapRectToScene(it.local_rect())
    view = c.mapToScene(c.viewport().rect()).boundingRect()
    assert view.contains(region.center()), "the tool is not in view"
    assert view.width() < img.shape[1] * 0.9, "the editor opened on the whole image"

    # a region nearly the size of the image: fit the image, never zoom out past it
    big = w.open_tool_editor("corner_screws")
    big.resize(1000, 700)
    big.frame_tool()
    shown = big.canvas.mapToScene(big.canvas.viewport().rect()).boundingRect()
    assert shown.width() < img.shape[1] * 1.6, \
        "a large region zoomed out to %.0f px of empty canvas" % shown.width()
    w.dirty = False


@qt_required
def test_editor_and_main_window_edit_one_recipe(ui):
    """Nothing is copied: a move in either window is one change to one recipe."""
    w, _tmp, _img = ui
    ed = w.open_tool_editor("lot_code")
    before = list(_roi(w, "lot_code"))
    _drag_region(ed.canvas, "lot_code", 30, 20)
    moved = list(_roi(w, "lot_code"))
    assert moved != before, "a drag in the editor did not reach the recipe"
    assert w.canvas.regions["lot_code"].roi() == moved, "the main canvas did not follow"
    w.canvas.select_ids(["lot_code"])
    _drag_region(w.canvas, "lot_code", -25, 0)
    assert ed.canvas.regions["lot_code"].roi() == list(_roi(w, "lot_code")), \
        "the editor did not follow a drag in the main window"
    ed.props.tool["params"]["min_confidence"] = 0.42
    ed._props_changed()
    assert [t for t in w.spec["tools"] if t["id"] == "lot_code"][0]["params"]["min_confidence"] == 0.42
    w.dirty = False


@qt_required
def test_undo_reaches_open_tool_editors(ui):
    """Undo restores the recipe from a snapshot -- new objects -- and nothing
    told the editors: the recipe went back while the editor still showed the
    moved region."""
    w, _tmp, _img = ui
    ed = w.open_tool_editor("lot_code")
    before = list(_roi(w, "lot_code"))
    _drag_region(ed.canvas, "lot_code", 30, 20)
    w.undo()
    assert list(_roi(w, "lot_code")) == before
    assert ed.canvas.regions["lot_code"].roi() == before, "the editor kept the undone move"
    w.redo()
    assert ed.canvas.regions["lot_code"].roi() == list(_roi(w, "lot_code"))
    w.dirty = False


@qt_required
def test_deleting_a_tool_or_opening_another_model_closes_its_editor(ui):
    w, _tmp, _img = ui
    w.open_tool_editor("lot_code")
    w.canvas.select_ids(["lot_code"])
    w.remove_tool()
    assert "lot_code" not in w._editors, "an editor stayed open for a deleted tool"
    w.open_tool_editor(w.spec["tools"][0]["id"])
    w.load_project(w.current_pid())
    assert not w._editors, "editors survived opening a model"
    w.dirty = False


@qt_required
def test_switching_a_tool_off_keeps_it_selected(ui):
    """Unticking "Run this tool" hid the region, Qt deselected it, and the
    property panel went blank -- the tool could not be switched back on from
    the same place. It stays selected, drawn faded."""
    from PySide6.QtWidgets import QApplication, QCheckBox
    w, _tmp, _img = ui
    w.canvas.select_ids(["lot_code"])
    QApplication.processEvents()
    box = next(b for b in w.props.findChildren(QCheckBox) if b.text() == "Run this tool")
    box.setChecked(False)
    QApplication.processEvents()
    assert w.props.tool is not None and w.props.tool["id"] == "lot_code", "the tool was deselected"
    item = w.canvas.regions["lot_code"]
    assert item.isVisible() and item.opacity() < 0.5, "a disabled region should show, faded"
    box = next(b for b in w.props.findChildren(QCheckBox) if b.text() == "Run this tool")
    box.setChecked(True)
    QApplication.processEvents()
    assert w.canvas.regions["lot_code"].opacity() == 1.0
    w.dirty = False


@qt_required
@pytest.mark.parametrize("ttype", _registry_types())
def test_every_tool_type_works_in_the_classic_window(ui, ttype, monkeypatch):
    """The classic window shows the same tools through its own property panel:
    each added, every setting changed, inspected, every tab shown, removed --
    with no exception, including inside Qt's event handlers."""
    from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                                   QSpinBox)
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))
    w, _tmp, _img = ui
    n0 = len(w.spec["tools"])
    w.add_tool(ttype)
    assert len(w.spec["tools"]) == n0 + 1
    tid = w.spec["tools"][-1]["id"]
    QApplication.processEvents()
    import shiboken6
    # a change can rebuild the panel, so widgets are looked up afresh and any
    # that were replaced are skipped
    for cls in (QSpinBox, QDoubleSpinBox, QComboBox):
        for x in w.props.findChildren(cls):
            if not shiboken6.isValid(x) or not x.isEnabled():
                continue
            if cls in (QSpinBox, QDoubleSpinBox) and x.isVisible():
                x.setValue(min(x.maximum(), x.value() + x.singleStep()))
            elif cls is QComboBox and x.count() > 1:
                x.setCurrentIndex((x.currentIndex() + 1) % x.count())
            QApplication.processEvents()
    # checkboxes by label: toggled and toggled back, finding the box again in
    # case the panel was rebuilt in between (a stale one would leave "Run this
    # tool" switched off)
    for label in [x.text() for x in w.props.findChildren(QCheckBox) if x.isEnabled()]:
        for _ in range(2):
            box = next((b for b in w.props.findChildren(QCheckBox)
                        if shiboken6.isValid(b) and b.text() == label), None)
            if box is not None:
                box.setChecked(not box.isChecked())
                QApplication.processEvents()
    w.run_once()
    w.wait_for_inspection()
    QApplication.processEvents()
    assert any(t["tool_id"] == tid for t in w.last["tools"]), "the new tool did not run"
    for i in range(w.props.tabs.count()):
        w.props.tabs.setCurrentIndex(i)
        QApplication.processEvents()
    w.canvas.select_ids([tid])
    w.remove_tool()
    QApplication.processEvents()
    assert tid not in [t["id"] for t in w.spec["tools"]]
    w.dirty = False
    assert not errors, "an exception inside the interface: %s" % [
        "%s: %s" % (e[0].__name__, e[1]) for e in errors]


@qt_required
@pytest.mark.parametrize("corner", ["top-left", "top-right", "bottom-left", "bottom-right"])
def test_a_polygon_scales_from_every_corner_handle(ui, corner):
    """The scale was measured against the box's right and bottom edges
    whatever was grabbed, so dragging a left or top handle did nothing."""
    from PySide6.QtCore import QPointF
    from vision_station.core.schema import defaults
    w, _tmp, _img = ui
    c = w.canvas
    c.resetTransform()
    w.spec["tools"].append({"id": "poly", "type": "histogram",
                            "params": dict(defaults("histogram"), shape="polygon",
                                           points=[[200, 150], [300, 150], [300, 230], [200, 230]],
                                           roi=[200, 150, 100, 80])})
    c.sync_regions(w.spec["tools"], None)
    c.select_ids(["poly"])
    it = c.regions["poly"]
    x = 200 if "left" in corner else 300
    y = 150 if "top" in corner else 230
    dx = -30 if "left" in corner else 30
    dy = -20 if "top" in corner else 20
    _drag(c, x, y, x + dx, y + dy)
    r = it.roi()
    assert abs(r[2] - 130) <= 2 and abs(r[3] - 100) <= 2, \
        "dragging the %s handle gave %s" % (corner, r)
    ox = 170 if "left" in corner else 200          # the opposite corner stays put
    oy = 130 if "top" in corner else 150
    assert abs(r[0] - ox) <= 2 and abs(r[1] - oy) <= 2, r
    w.dirty = False


@qt_required
def test_polygon_and_rectangle_regions_follow_one_label_rule(ui):
    """Each region shape had its own copy of the rule for when a name tag
    shows, and polygons had dropped the labelling layer."""
    from vision_station.ui.canvas import show_label
    w, _tmp, _img = ui
    w.canvas.label_policy = "smart"
    rect = next(it for it in w.canvas.regions.values() if it.__class__.__name__ == "RegionItem")
    rect.setSelected(False)
    rect.status = "pass"
    assert not show_label(rect)
    rect.status = "fail"
    assert show_label(rect)
    for pol, sel, want in (("all", False, True), ("none", False, False), ("none", True, True)):
        w.canvas.label_policy = pol
        rect.setSelected(sel)
        assert show_label(rect) == want, (pol, sel)
    w.canvas.label_policy = "smart"
    w.dirty = False


@qt_required
def test_zoom_is_bounded_and_reports_its_level(ui):
    """Sixty scroll clicks used to shrink a 640 px image to 0.1 px on screen: it
    vanished and nothing said where it had gone."""
    from PySide6.QtWidgets import QApplication
    w, _tmp, img = ui
    c = w.canvas
    for _ in range(80):
        c.zoom(1 / 1.16)
    on_screen = img.shape[1] * c.transform().m11()
    assert on_screen >= c.MIN_ON_SCREEN * 0.9, "the image shrank to %.1f px" % on_screen
    for _ in range(120):
        c.zoom(1.16)
    assert c.transform().m11() <= c.MAX_SCALE + 1e-6
    c.fit()
    # the label must still be alive: it was once added to the default status bar
    # and destroyed when the window installed its own
    assert w.zoomLabel.parent() is not None
    assert w.zoomLabel.text().endswith("%")
    c.zoom_100()
    assert w.zoomLabel.text() == "100%"
    w.dirty = False


@qt_required
def test_zoom_from_a_menu_anchors_at_the_centre(ui):
    """The wheel zooms at the pointer. Menu and keyboard zoom must not: the
    pointer is on the menu at the time, so the image jumped away from whatever
    the operator was looking at."""
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    w, _tmp, _img = ui
    w.show()
    c = w.canvas
    c.fit()
    mid = c.viewport().rect().center()
    corner = mid - QPoint(200, 140)          # where a pointer might happen to be
    # Deliver the move to the view itself: QTest.mouseMove depends on mouse
    # state other tests leave behind in the session, and when no move arrived
    # the view had no pointer position to anchor at.
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    c.viewport().setMouseTracking(True)
    # Qt anchors under the mouse only while the view counts itself as under
    # it (set by Enter events) -- after hundreds of tests, another window may
    # hold that. A real wheel event is always over the view.
    c.setAttribute(Qt.WA_UnderMouse, True)
    c.viewport().setAttribute(Qt.WA_UnderMouse, True)
    ev = QMouseEvent(QMouseEvent.MouseMove, QPointF(corner),
                     QPointF(c.viewport().mapToGlobal(corner)),
                     Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    QApplication.sendEvent(c.viewport(), ev)

    before = c.mapToScene(mid)
    c.zoom(2.0)                              # as the menu and keys call it
    centred = (c.mapToScene(mid) - before).manhattanLength()
    c.fit()
    before = c.mapToScene(mid)
    c.zoom(2.0, anchor="mouse")              # as the wheel calls it
    at_mouse = (c.mapToScene(mid) - before).manhattanLength()
    assert centred < 4.0, "the view centre moved %.1f px" % centred
    assert at_mouse > centred, \
        "the two anchors behave the same, so the fix does nothing (%.1f vs %.1f)" % (
            at_mouse, centred)
    w.dirty = False


@qt_required
def test_run_mode_hides_editing_and_locks_regions(ui):
    """The operator's screen: image, verdict and run controls; every editing
    surface put away and the regions locked, so a stray click cannot move a
    tool on the line."""
    from PySide6.QtWidgets import QApplication, QDockWidget, QGraphicsItem
    w, _tmp, _img = ui
    w.show()                          # visibility means nothing in an unshown window
    QApplication.processEvents()
    before = sum(d.isVisible() for d in w.findChildren(QDockWidget))
    assert before > 0
    w.set_mode("run")
    assert not any(d.isVisible() for d in w.findChildren(QDockWidget))
    assert not w.mainBar.isVisible() and not w.runBar.isVisible(), \
        "builder controls such as Save job and Diagnose are still showing"
    assert w.runModeBar.isVisible()
    assert all(not (it.flags() & QGraphicsItem.ItemIsMovable) for it in w.canvas.regions.values())
    w.run_once()
    w.wait_for_inspection()
    assert w._recent and w._recent[-1] in ("PASS", "FAIL", "WARN")
    w.set_mode("build")
    assert sum(d.isVisible() for d in w.findChildren(QDockWidget)) == before
    assert all(it.flags() & QGraphicsItem.ItemIsMovable for it in w.canvas.regions.values())
    w.dirty = False


@qt_required
def test_run_mode_has_a_way_back(ui):
    """The Build/Run switch lives on the builder's toolbar, which Run mode hides,
    and Qt does not fire the shortcut of an action on a hidden toolbar. Without
    a button on the run bar and window-level shortcuts, the only way out of Run
    mode was to close the application."""
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    w, _tmp, _img = ui
    w.show()
    w.activateWindow()                # shortcuts go to the active window
    QApplication.processEvents()
    w.set_mode("run")
    QApplication.processEvents()
    assert any(a.text() == "Build mode" for a in w.runModeBar.actions())
    QTest.keySequence(w, "Ctrl+B")
    QApplication.processEvents()
    assert w.mode == "build", "Ctrl+B did nothing in Run mode"
    QTest.keySequence(w, "Ctrl+R")
    QApplication.processEvents()
    assert w.mode == "run"
    w.set_mode("build")
    w.dirty = False


@qt_required
def test_lens_calibration_from_the_application(ui, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from vision_station.core import calibration as C
    w, _tmp, _img = ui
    shown = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: shown.append(a[2])))
    views, _maps = _lens_views(8, 6)
    cal = w.calibrate_lens(views, _LPAT, _LSQ)
    assert cal is not None and cal["rms_px"] < 0.3
    assert os.path.exists(os.path.join(w.project.folder, C.FILE))
    assert w.project.calibration is not None, "the model was not reloaded with its calibration"
    assert "Reprojection error" in shown[-1]
    w.remove_lens_calibration()
    assert w.project.calibration is None
    w.dirty = False


@qt_required
def test_selecting_during_an_inspection_does_not_orphan_the_worker(ui):
    """A patch that meant to initialise the worker reference once instead reset
    it after every tree refresh and every selection. Clicking a region while an
    inspection was running then made the application forget it, and the next
    Run started a second worker alongside -- two threads rebuilding the same
    project's tools at once."""
    w, _tmp, _img = ui
    w.run_once()
    running = w._worker
    assert running is not None
    w.canvas.select_ids([w.spec["tools"][0]["id"]])
    w.on_tree_select()
    w.refresh_tree()
    assert w._worker is running, "a selection or refresh dropped the running worker"
    w.run_once()                         # must not start a second one
    assert w._worker is running, "a second inspection started alongside the first"
    w.wait_for_inspection()
    assert w._worker is None
    w.dirty = False


@qt_required
def test_continuous_mode_reschedules_rather_than_free_running(ui):
    """A repeating timer shorter than the cycle queues work faster than it
    completes, and the event loop never drains. The next cycle is started when
    the previous one finishes instead."""
    w, _tmp, _img = ui
    assert w.runTimer.isSingleShot(), \
        "continuous mode uses a repeating timer, which stacks up work"
    w.toggle_continuous()
    assert w.continuous
    w.toggle_continuous()
    assert not w.continuous
    assert not w.runTimer.isActive(), "the timer kept running after stopping"
    w.wait_for_inspection()
    w.dirty = False


@qt_required
def test_display_is_downscaled_but_measurement_is_not(ui):
    """The canvas paints a reduced pixmap; the tools still get every pixel.

    Painting a full 2576x1944 frame cost 37 ms each time, which is where a live
    camera or a large photograph turns treacly. Only what is drawn is reduced:
    if the downscale ever reached the array the tools read, every measurement
    in the application would quietly lose accuracy while still looking
    plausible."""
    w, _tmp, _img = ui
    big = np.zeros((1600, 2200, 3), np.uint8)
    cv2.rectangle(big, (200, 200), (900, 700), (180, 180, 180), -1)
    w.canvas.set_image(big, keep_view=False)

    assert w.canvas._pix_scale < 0.9, \
        "a large frame was not downscaled for display (%.2f)" % w.canvas._pix_scale
    assert w.canvas._img.shape == big.shape, \
        "the array handed to the tools was downscaled"
    # the scene reaches past the image (so zooming at an edge can keep the
    # spot under the pointer), but the image itself lies at 0..w, 0..h
    rect = w.canvas.pix.sceneBoundingRect()
    assert w.canvas.sc.sceneRect().contains(rect)
    # (the reduced display copy rounds its height to whole pixels: < 1 px)
    assert abs(rect.x()) < 1e-6 and abs(rect.y()) < 1e-6 and \
        abs(rect.width() - 2200) < 1.0 and abs(rect.height() - 1600) < 1.0, \
        "scene coordinates no longer match image pixels: %s" % rect
    w.dirty = False


@qt_required
def test_zooming_in_restores_full_detail(ui):
    """The reduced pixmap is a display proxy, so zooming in has to fetch a
    sharper one -- otherwise close inspection shows a blurred image."""
    w, _tmp, _img = ui
    big = np.zeros((1600, 2200, 3), np.uint8)
    cv2.rectangle(big, (200, 200), (900, 700), (180, 180, 180), -1)
    w.canvas.set_image(big, keep_view=False)
    coarse = w.canvas._pix_scale
    for _ in range(6):
        w.canvas.zoom(1.5)
    assert w.canvas._pix_scale > coarse, \
        "zooming in did not re-render at higher detail (%.3f -> %.3f)" % (
            coarse, w.canvas._pix_scale)
    w.dirty = False


@qt_required
def test_toolbar_does_not_duplicate_the_palette(ui):
    """Select and Hand belong to the palette. They were on the toolbar as well,
    which is two controls for one state and two places to look for it."""
    from PySide6.QtWidgets import QToolBar
    w, _tmp, _img = ui
    on_toolbars = []
    for bar in w.findChildren(QToolBar):
        if bar.objectName() == "palette":
            continue
        on_toolbars += [a.text() for a in bar.actions() if a.text()]
    for name in ("Select", "Hand"):
        assert name not in on_toolbars, \
            "%s is on a toolbar as well as the palette" % name
    # and the palette still has them
    assert {"select", "hand"} <= set(w.toolActions)


@qt_required
def test_model_indicator_tracks_the_recipe(ui):
    """How many checks a recipe has, and whether it needs more than one
    exposure, are things an operator otherwise has to count by hand."""
    w, _tmp, _img = ui
    n = len(w.spec["tools"])
    assert "%d check" % n in w.modelLabel.text(), w.modelLabel.text()
    assert "captures" not in w.modelLabel.text()

    w.spec["captures"] = [{"id": "panel"}, {"id": "leds", "exposure_ms": 1.0}]
    w.refresh_model_label()
    assert "2 captures" in w.modelLabel.text(), w.modelLabel.text()

    w.canvas.select_ids([w.spec["tools"][0]["id"]])
    w.remove_tool()
    assert "%d check" % (n - 1) in w.modelLabel.text(), \
        "the indicator did not follow a deletion: %s" % w.modelLabel.text()
    w.dirty = False


@qt_required
def test_result_banner_states_the_verdict_and_the_reason(ui):
    """The verdict used to live in twelve-point text at the bottom of the
    screen. An operator glances at a station monitor for a fraction of a second
    and needs the answer and the reason, not a label to hunt for."""
    w, _tmp, _img = ui
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad = cv2.imread(os.path.join(here, "projects", "demo_led_panel", "bad.png"))
    w.canvas.set_image(bad, keep_view=False)
    w.run_once()
    w.wait_for_inspection()
    assert w.bigVerdict.text() == "FAIL"
    detail = w.bannerDetail.text()
    failed = [t["tool_id"] for t in w.last["tools"] if t["status"] == "fail"]
    assert any(f in detail for f in failed), \
        "the banner does not name a failing check: %r" % detail
    assert w.bannerStats["cycle"].text().endswith("ms")
    assert w.bannerStats["count"].text() == "1"

    good = cv2.imread(os.path.join(here, "projects", "demo_led_panel", "good.png"))
    w.canvas.set_image(good, keep_view=False)
    w.run_once()
    w.wait_for_inspection()
    assert w.bigVerdict.text() == "PASS"
    assert "passed" in w.bannerDetail.text().lower()
    assert w.bannerStats["yield"].text() == "50.0%", \
        "yield after one pass and one fail was %s" % w.bannerStats["yield"].text()
    w.dirty = False


@qt_required
def test_label_density_can_be_controlled(ui):
    """Naming every region at once made a dense recipe unreadable: nine
    overlapping boxes each with a tag, over the part being examined."""
    w, _tmp, _img = ui
    assert w.canvas.label_policy == "smart"
    for mode in ("all", "none", "smart"):
        w.set_label_policy(mode)
        assert w.canvas.label_policy == mode

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    w.canvas.set_image(cv2.imread(os.path.join(here, "projects",
                                               "demo_led_panel", "bad.png")),
                       keep_view=False)
    w.run_once()
    w.wait_for_inspection()
    w.set_label_policy("smart")
    w.canvas.sync_regions(w.spec["tools"], w.last)
    shown = [r for r in w.canvas.regions.values()
             if hasattr(r, "_show_label") and r._show_label()]
    hidden = [r for r in w.canvas.regions.values()
              if hasattr(r, "_show_label") and not r._show_label()]
    assert shown, "no region is labelled at all"
    assert hidden, "every region is still labelled in smart mode"
    for r in shown:
        assert r.isSelected() or r.status in ("fail", "error", "warn"), \
            "%s is labelled but neither selected nor failing" % r.tool_id
    w.dirty = False


@qt_required
def test_palette_offers_every_drawing_tool(ui):
    """Elliptical regions existed for some time and went unfound because the
    only way to reach them was a dropdown on a tab. A capability nobody can see
    may as well not be there."""
    w, _tmp, _img = ui
    assert set(w.toolActions) >= {"select", "draw_rect", "draw_ellipse",
                                  "draw_polygon", "hand"}
    for key in ("draw_rect", "draw_ellipse", "draw_polygon"):
        w.set_tool(key)
        assert w.canvas.mode == key
        assert w.toolActions[key].isChecked()
    w.set_tool("select")


@qt_required
def test_drawing_a_polygon_creates_a_working_region(ui):
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtTest import QTest
    w, _tmp, _img = ui
    c = w.canvas
    c.resetTransform()
    n0 = len(w.spec["tools"])
    i = w.newTypeBox.findData("count_blob")
    w.newTypeBox.setCurrentIndex(max(0, i))

    w.set_tool("draw_polygon")
    for pt in ((300, 200), (420, 190), (450, 260), (330, 280)):
        p = c.mapFromScene(QPointF(*pt))
        QTest.mousePress(c.viewport(), Qt.LeftButton, Qt.NoModifier, p)
        QTest.mouseRelease(c.viewport(), Qt.LeftButton, Qt.NoModifier, p)
    QTest.keyClick(c, Qt.Key_Return)

    assert len(w.spec["tools"]) == n0 + 1, "no region was created"
    t = w.spec["tools"][-1]
    assert t["params"]["shape"] == "polygon"
    assert len(t["params"]["points"]) == 4
    assert t["type"] == "count_blob", "the selected region type was ignored"
    assert c.mode == "select", "the tool did not return to select after drawing"
    w.run_once()
    w.wait_for_inspection()
    r = [x for x in w.last["tools"] if x["tool_id"] == t["id"]][0]
    assert r["status"] in ("pass", "fail")
    assert "poly" in {g["kind"] for g in r["graphics"]}
    w.dirty = False


@qt_required
def test_escape_abandons_a_half_drawn_polygon(ui):
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtTest import QTest
    w, _tmp, _img = ui
    c = w.canvas
    c.resetTransform()
    n0 = len(w.spec["tools"])
    w.set_tool("draw_polygon")
    for pt in ((100, 100), (200, 110)):
        p = c.mapFromScene(QPointF(*pt))
        QTest.mousePress(c.viewport(), Qt.LeftButton, Qt.NoModifier, p)
        QTest.mouseRelease(c.viewport(), Qt.LeftButton, Qt.NoModifier, p)
    QTest.keyClick(c, Qt.Key_Escape)
    assert len(w.spec["tools"]) == n0, "an abandoned trace still created a region"
    assert not c._poly_pending
    w.set_tool("select")


# ------------------------------------------------- fixture derotation
def test_regions_are_derotated_under_a_rotated_fixture(board):
    """A region with no rotation of its own must still be cut in the part's
    coordinate space when the fixture is rotated.

    It used to be cut as the axis-aligned bounding box of where it landed, so
    the tool saw the part still rotated, in an oversized box. That affected
    every tool whenever the part turned; it surfaced when a pixel-by-pixel tool
    rejected a good part rotated 3.7 degrees on a real chassis."""
    h, w = board.shape[:2]
    t = Probe("p", {"roi": [150, 120, 200, 120]})
    ctx = Context(reference=board)
    straight, _ = t.crop(board, ctx)

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 4.0, 1.0)
    turned = cv2.warpAffine(board, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    ctx.fixture = np.vstack([M, [0, 0, 1]]).astype(np.float32)
    cut, _ = t.crop(turned, ctx)

    assert cut.shape[:2] == straight.shape[:2], \
        "region grew to %s under a rotated fixture (should stay %s)" % (
            cut.shape[:2], straight.shape[:2])
    a = cv2.cvtColor(straight, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inner = (slice(6, -6), slice(6, -6))
    assert float(np.abs(a[inner] - b[inner]).mean()) < 6.0, \
        "the cut region is not the same patch of the part"


# ------------------------------------------------------ trained surface
def _surface_part(seed=0, dx=0, dy=0, deg=0.0, gain=1.0, scratch=False):
    """A textured panel with holes, posed and lit like a real part."""
    r = np.random.default_rng(1)                       # the part itself is fixed
    img = np.full((420, 560, 3), 128, np.uint8)
    noise = cv2.GaussianBlur(r.normal(0, 7, (420, 560)).astype(np.float32), (1, 15), 0)
    img = np.clip(img + noise[..., None], 0, 255).astype(np.uint8)
    cv2.rectangle(img, (60, 60), (500, 360), (150, 150, 150), -1)
    for cx in (120, 220, 320, 420):
        cv2.circle(img, (cx, 150), 12, (60, 60, 60), -1)
    cv2.rectangle(img, (90, 250), (250, 300), (235, 235, 235), -1)
    if scratch:
        q = np.random.default_rng(100 + seed)
        x0 = int(q.integers(290, 380))
        y0 = int(q.integers(210, 300))
        cv2.line(img, (x0, y0), (x0 + 70, y0 + int(q.integers(-15, 15))),
                 (215, 215, 215), 3, cv2.LINE_AA)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _surface_fixture(ref):
    from vision_station.core.schema import defaults
    spec = {"id": "fx", "type": "align",
            "params": dict(defaults("align"), roi=[40, 40, 480, 340],
                           method="template", min_score=0.4,
                           angle_range=[-6, 6], angle_step=1.0)}

    def fx(img):
        ctx = Context(reference=ref)
        make_tool(spec).run(img, ctx)
        return ctx
    return fx


def test_trained_surface_learns_ok_and_sets_threshold_from_ng(tmp_path):
    """Trained from OK images, threshold chosen between the worst OK and the
    best NG, and the gap between them reported -- rather than a threshold
    picked by hand, which is where inspections go wrong."""
    from vision_station.tools.learned import train
    ref = _surface_part()
    fx = _surface_fixture(ref)
    tool = make_tool({"id": "s", "type": "trained_surface",
                      "params": {"roi": [80, 110, 400, 220]}})
    rep = train(tool, [(ref, "ok"), (_surface_part(dx=8, dy=-5, deg=1), "ok"),
                       (_surface_part(seed=1, scratch=True), "ng"),
                       (_surface_part(seed=2, dx=-6, scratch=True), "ng")],
                fx, str(tmp_path / "m.npz"))
    assert rep["separates"], "OK and NG overlap in training: %s" % rep
    assert max(rep["ok_scores"]) < rep["threshold"] < min(rep["ng_scores"])
    assert rep["real_ok"] == 2 and rep["augmented"] > 0


def test_trained_surface_generalises_to_parts_it_never_saw(tmp_path):
    """Held-out parts moved, rotated and relit: good ones pass, scratched ones
    fail. On the real chassis this went 5 of 6, then 12 of 12 once the fixture
    derotation bug was fixed -- the tool had been handed rotated crops."""
    from vision_station.tools.learned import train, TrainedSurface
    ref = _surface_part()
    fx = _surface_fixture(ref)
    tool = make_tool({"id": "s", "type": "trained_surface",
                      "params": {"roi": [80, 110, 400, 220]}})
    path = str(tmp_path / "m.npz")
    train(tool, [(ref, "ok"), (_surface_part(dx=8, dy=-5, deg=1), "ok"),
                 (_surface_part(seed=1, scratch=True), "ng"),
                 (_surface_part(seed=2, dx=-6, scratch=True), "ng")], fx, path)
    tool.p["model"] = path
    TrainedSurface._models.clear()
    wrong = []
    for i, (dx, dy, deg, gain) in enumerate([(25, -15, 3.5, 0.9), (-30, 20, -3.0, 1.08),
                                             (15, 25, 2.0, 1.0)]):
        for bad in (False, True):
            img = _surface_part(seed=10 + i, dx=dx, dy=dy, deg=deg, gain=gain,
                                scratch=bad)
            r = tool.run(img, fx(img))
            want = "fail" if bad else "pass"
            if r.status != want:
                wrong.append((i, bad, r.status, r.values.get("defect_area")))
    assert not wrong, "misjudged held-out parts: %s" % wrong


def test_trained_surface_untrained_is_an_honest_error():
    tool = make_tool({"id": "s", "type": "trained_surface",
                      "params": {"roi": [0, 0, 50, 50],
                                 "model": "/nonexistent/m.npz"}})
    img = np.zeros((80, 80, 3), np.uint8)
    r = tool.run(img, Context(reference=img))
    assert r.status == "error" and "not trained" in r.message


# -------------------------------------------------------------- geometry
def _geo(ttype, params, img, ctx=None):
    base = {"enabled": True, "critical": True, "angle": 0.0, "shape": "rect"}
    ctx = ctx or Context(reference=img)
    r = make_tool({"id": params.pop("id", "g"), "type": ttype,
                   "params": dict(base, **params)}).run(img, ctx)
    return r, ctx


def _aa_circle(cx, cy, r, W=320, H=240, seed=0, fg=190, bg=60):
    S = 8
    big = np.full((H * S, W * S), bg, np.uint8)
    cv2.circle(big, (int(round(cx * S + S / 2 - 0.5)), int(round(cy * S + S / 2 - 0.5))),
               int(round(r * S)), fg, -1, cv2.LINE_AA)
    im = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    im = cv2.GaussianBlur(im, (0, 0), 1.2) + np.random.default_rng(seed).normal(0, 4, im.shape)
    return cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def test_find_circle_is_subpixel_accurate():
    """Measured: centre within 0.14 px and radius within 0.08 px over twenty
    random circles with blur and noise."""
    rng = np.random.default_rng(3)
    for i in range(8):
        cx, cy, rad = 160 + rng.uniform(-5, 5), 120 + rng.uniform(-5, 5), rng.uniform(35, 55)
        R = rad * 1.35
        r, _ = _geo("find_circle", {"roi": [int(cx - R), int(cy - R), int(2 * R), int(2 * R)],
                                    "search_inner": 0.5}, _aa_circle(cx, cy, rad, seed=i))
        assert r.status == "pass", r.message
        assert math.hypot(r.values["cx"] - cx, r.values["cy"] - cy) < 0.3
        assert abs(r.values["radius_px"] - rad) < 0.2


def test_find_circle_rejects_an_outlier_on_the_edge():
    """A chip in the rim: trusting every caliper put the centre 0.69 px out;
    discarding the worst three brought it to 0.11 px."""
    cx, cy, rad = 160.3, 119.6, 45.0
    img = _aa_circle(cx, cy, rad, seed=7)
    cv2.circle(img, (int(cx + rad), int(cy)), 9, (60, 60, 60), -1)
    R = rad * 1.35
    roi = [int(cx - R), int(cy - R), int(2 * R), int(2 * R)]
    # The rays that agree on one circle are fitted and the chipped ones set
    # aside without being asked: 'num_to_ignore' in an earlier recipe is
    # accepted and changes nothing.
    for ignore in (0, 3):
        r, _ = _geo("find_circle", {"roi": list(roi), "search_inner": 0.5,
                                    "num_to_ignore": ignore}, img)
        err = math.hypot(r.values["cx"] - cx, r.values["cy"] - cy)
        assert r.status == "pass" and err < 0.25, (ignore, err, r.message)
        assert r.values["used"] < r.values["found"], "the chipped rays were fitted too"


def test_find_circle_fits_one_edge_of_a_plated_hole():
    """A plated hole has two edges a few pixels apart: the hole and its ring.
    Each ray took whichever was stronger there, and the fit ran through a
    mixture of both -- 'fit error 4.86 px over 1.50' on a real board for a
    hole in plain view. The circle most rays agree on is fitted instead."""
    cx, cy = 150.4, 149.7
    img = np.full((300, 300), 120, np.uint8)
    cv2.circle(img, (150, 150), 78, 200, -1, cv2.LINE_AA)        # the ring
    cv2.circle(img, (150, 150), 60, 25, -1, cv2.LINE_AA)         # the hole
    rng = np.random.RandomState(3)
    # the ring is brighter on one side, as under a lamp: which edge is the
    # stronger then changes from ray to ray
    ramp = np.tile(np.linspace(0.55, 1.25, 300, dtype=np.float32), (300, 1))
    img = np.clip(img.astype(np.float32) * ramp + rng.normal(0, 2.0, img.shape), 0, 255)
    img = cv2.cvtColor(cv2.GaussianBlur(img.astype(np.uint8), (0, 0), 1.0), cv2.COLOR_GRAY2BGR)
    r, _ = _geo("find_circle", {"roi": [40, 40, 220, 220], "search_inner": 0.3}, img)
    assert r.status == "pass", r.message
    assert r.values["rms_px"] < 1.0, r.values
    assert min(abs(r.values["radius_px"] - 60), abs(r.values["radius_px"] - 78)) < 1.5, \
        "fitted neither edge: r=%.2f" % r.values["radius_px"]
    assert math.hypot(r.values["cx"] - 150, r.values["cy"] - 150) < 1.0


def test_find_line_recovers_angle_and_position():
    S, W, H = 8, 320, 240
    for ang, yc in ((0.0, 120.0), (5.5, 118.3), (-7.2, 123.6)):
        big = np.full((H * S, W * S), 60, np.uint8)
        a = math.radians(ang)
        d = np.array([math.cos(a), math.sin(a)])
        c = np.array([W / 2.0, yc])
        p0, p1 = c - d * 2000, c + d * 2000
        poly = np.array([p0, p1, p1 + [0, 3000], p0 + [0, 3000]]) * S + S / 2 - 0.5
        cv2.fillPoly(big, [np.round(poly).astype(np.int32)], 190, cv2.LINE_AA)
        img = cv2.cvtColor(cv2.GaussianBlur(cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA),
                                            (0, 0), 1.0), cv2.COLOR_GRAY2BGR)
        r, ctx = _geo("find_line", {"roi": [40, 70, 240, 100]}, img)
        assert r.status == "pass", r.message
        assert abs(r.values["angle_deg"] - ang) < 0.1
        f = ctx.share["g"]
        pt, dd = np.array(f["point"]), np.array(f["direction"])
        assert abs(pt[1] + dd[1] * ((160 - pt[0]) / dd[0]) - yc) < 0.3


def test_geometry_tools_see_colour_contrast_that_grey_hides():
    """A green LED is 142 grey on a 138 board: invisible after greyscale
    conversion, though every colour channel shows 60 to 90 levels of contrast.
    The default channel choice finds it."""
    img = np.full((200, 200, 3), (150, 140, 130), np.uint8)
    cv2.circle(img, (100, 100), 22, (60, 200, 60), -1)
    grey, _ = _geo("find_circle", {"roi": [70, 70, 60, 60], "channel": "grey"}, img)
    auto, _ = _geo("find_circle", {"roi": [70, 70, 60, 60], "channel": "auto"}, img)
    assert grey.status == "fail"
    assert auto.status == "pass"
    assert math.hypot(auto.values["cx"] - 100, auto.values["cy"] - 100) < 0.5


def test_find_circle_contrast_is_measured_per_pixel():
    """Find Circle samples its rays every half pixel, and its gradient was once
    per sample -- half the per-pixel value Find Line uses -- so a moderate edge
    that Find Line would accept fell under the same threshold here. On the demo
    board's red LED (108 grey on 138) that left 4 of 24 calipers finding it."""
    img = np.full((200, 200, 3), (150, 140, 130), np.uint8)
    cv2.circle(img, (100, 100), 22, (60, 60, 220), -1)          # the red LED
    r, _ = _geo("find_circle", {"roi": [70, 70, 60, 60], "channel": "grey"}, img)
    assert r.status == "pass", r.message
    assert r.values["found"] >= 20, "only %d calipers found a 30-level edge" % r.values["found"]


def test_distance_between_holes_is_independent_of_part_pose():
    """Two holes found through the fixture, then measured: 150.00 +- 0.04 px
    whether the part is moved, turned seven degrees, or both."""
    from vision_station.core.schema import defaults
    S, W, H = 8, 480, 320

    def part(dx, dy, deg, seed):
        big = np.full((H * S, W * S), 60, np.uint8)
        for hx in (165, 315):
            cv2.circle(big, (hx * S, 160 * S), 30 * S, 190, -1, cv2.LINE_AA)
        im = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        im = cv2.warpAffine(im, M, (W, H), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        im = cv2.GaussianBlur(im, (0, 0), 1.0) + np.random.default_rng(seed).normal(0, 3, im.shape)
        return cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    ref = part(0, 0, 0, 0)
    fx = {"id": "fx", "type": "align",
          "params": dict(defaults("align"), roi=[100, 100, 280, 120], method="template",
                         min_score=0.4, angle_range=[-8, 8], angle_step=1.0)}
    for i, (dx, dy, deg) in enumerate([(0, 0, 0), (-25, 15, 4), (30, -20, 7)]):
        img = part(dx, dy, deg, i + 1)
        ctx = Context(reference=ref)
        make_tool(fx).run(img, ctx)
        for hid, hx in (("hole_a", 165), ("hole_b", 315)):
            _geo("find_circle", {"id": hid, "roi": [hx - 42, 118, 84, 84], "shape": "ellipse",
                                 "search_inner": 0.5}, img, ctx)
        r, _ = _geo("distance", {"first": "hole_a", "second": "hole_b",
                                 "min_value": 149.5, "max_value": 150.5}, img, ctx)
        assert r.status == "pass", r.message
        assert abs(r.values["value_px"] - 150.0) < 0.15


def test_distance_reports_angle_between_lines_and_diameter():
    ctx = Context(reference=np.zeros((10, 10, 3), np.uint8))
    ctx.share["a"] = {"type": "line", "p0": [0, 0], "p1": [100, 0],
                      "point": [50, 0], "direction": [1, 0]}
    b = math.radians(30)
    ctx.share["b"] = {"type": "line", "p0": [0, 0], "p1": [100 * math.cos(b), 100 * math.sin(b)],
                      "point": [0, 0], "direction": [math.cos(b), math.sin(b)]}
    ctx.share["c"] = {"type": "circle", "center": [10, 10], "radius": 12.5}
    img = np.zeros((10, 10, 3), np.uint8)
    r, _ = _geo("distance", {"first": "a", "second": "b"}, img, ctx)
    assert r.values["unit"] == "deg" and abs(r.values["value"] - 30.0) < 1e-6
    r, _ = _geo("distance", {"first": "c", "second": ""}, img, ctx)
    assert r.values["measure"] == "diameter" and abs(r.values["value"] - 25.0) < 1e-6
    ctx.mm_per_px = 0.1
    r, _ = _geo("distance", {"first": "c", "second": ""}, img, ctx)
    assert r.values["unit"] == "mm" and abs(r.values["value"] - 2.5) < 1e-6


def test_distance_without_its_features_is_an_honest_error():
    img = np.zeros((10, 10, 3), np.uint8)
    r, _ = _geo("distance", {"first": "nowhere", "second": ""}, img)
    assert r.status == "error" and "nowhere" in r.message


def test_histogram_judges_presence_and_ignores_pixels_outside_the_shape():
    """Statistics use only pixels inside an ellipse. The tool base fills the
    outside with the inside mean, which would drag the spread towards zero if
    it were counted."""
    img = np.full((200, 200, 3), 40, np.uint8)
    cv2.circle(img, (100, 100), 50, (200, 200, 200), -1)
    cv2.circle(img, (100, 100), 20, (80, 80, 80), -1)
    r, _ = _geo("histogram", {"roi": [50, 50, 100, 100], "shape": "ellipse",
                              "mean_min": 120}, img)
    assert r.status == "pass", r.message
    inside = np.zeros((100, 100), np.uint8)
    cv2.ellipse(inside, (50, 50), (50, 50), 0, 0, 360, 255, -1)
    true_std = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[50:150, 50:150][inside > 0].std())
    assert abs(r.values["std"] - true_std) < 1.0, (r.values["std"], true_std)
    empty = np.full_like(img, 40)
    r, _ = _geo("histogram", {"roi": [50, 50, 100, 100], "shape": "ellipse",
                              "mean_min": 120}, empty)
    assert r.status == "fail"


def _comb(starts, widths, clutter=True, seed=0, W=400, H=80):
    S = 8
    big = np.full((H * S, W * S), 200, np.uint8)
    for x, w in zip(starts, widths):
        cv2.rectangle(big, (int(round(x * S)), 0), (int(round((x + w) * S)) - 1, H * S), 40, -1)
    if clutter:
        cv2.rectangle(big, (355 * S, 0), (390 * S, H * S), 150, -1)
    im = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    im = cv2.GaussianBlur(im, (0, 0), 1.0) + np.random.default_rng(seed).normal(0, 3, im.shape)
    return cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def test_caliper_counts_pins_measures_widths_and_pitch():
    """Six pins at sub-pixel positions beside a lighter step that is not a dark
    bar. Measured: widths within 0.024 px, pitch 50.251 against 50.25."""
    starts = [40.3, 90.55, 140.8, 191.05, 241.3, 291.55]
    img = _comb(starts, [20.25] * 6)
    r, _ = _geo("caliper", {"roi": [10, 20, 380, 40], "pair_polarity": "dark",
                            "expect_count": 6, "nominal_width": 20.25, "tolerance": 1.0}, img)
    assert r.status == "pass", r.message
    assert r.values["count"] == 6, "the lighter step was counted as a pin"
    assert abs(r.values["pitch"] - 50.25) < 0.1
    assert all(abs(w - 20.25) < 0.15 for w in r.values["widths"]), r.values["widths"]


def test_caliper_fails_a_part_with_one_wrong_width():
    starts = [40.3, 90.55, 140.8, 191.05, 241.3, 291.55]
    widths = [20.25, 20.25, 20.25, 20.25, 26.5, 20.25]
    r, _ = _geo("caliper", {"roi": [10, 20, 380, 40], "pair_polarity": "dark",
                            "expect_count": 6, "nominal_width": 20.25, "tolerance": 1.0},
                _comb(starts, widths))
    assert r.status == "fail" and "1 width" in r.message
    assert abs(r.values["widths"][4] - 26.5) < 0.15


def test_caliper_fails_a_missing_pin():
    starts = [40.3, 90.55, 191.05, 241.3, 291.55]          # the third pin is gone
    r, _ = _geo("caliper", {"roi": [10, 20, 380, 40], "pair_polarity": "dark",
                            "expect_count": 6}, _comb(starts, [20.25] * 5))
    assert r.status == "fail" and "found 5" in r.message


def test_caliper_reports_in_mm_when_scaled():
    ctx = Context(reference=np.zeros((80, 400, 3), np.uint8))
    ctx.mm_per_px = 0.05
    img = _comb([40.3, 90.55], [20.25, 20.25], clutter=False)
    r, _ = _geo("caliper", {"roi": [10, 20, 380, 40]}, img, ctx)
    assert r.values["unit"] == "mm"
    assert abs(r.values["widths"][0] - 20.25 * 0.05) < 0.01


# ------------------------------------------------------ terminals and links
def _linked_project(tmp_path, tools):
    from vision_station.core.engine import Project
    ref = _surface_part()
    cv2.imwrite(str(tmp_path / "reference.png"), ref)
    with open(str(tmp_path / "project.json"), "w") as f:
        json.dump({"id": "p", "tools": tools}, f)
    return Project(str(tmp_path)), ref


def _fx_spec(tid="fx"):
    from vision_station.core.schema import defaults
    return {"id": tid, "type": "align",
            "params": dict(defaults("align"), roi=[40, 40, 480, 340], method="template",
                           min_score=0.4, angle_range=[-6, 6], angle_step=1.0)}


def test_fixture_input_chooses_which_fixture_a_tool_follows(tmp_path):
    """VisionPro links each tool's input to a fixture explicitly. 'auto' keeps
    the behaviour every recipe relies on; 'none' reads image space."""
    from vision_station.core.schema import defaults
    label = lambda fx: {"id": "label_" + fx, "type": "presence",
                        "params": dict(defaults("presence"), roi=[90, 250, 160, 50],
                                       min_score=0.8, search_pad=4, fixture=fx)}
    proj, ref = _linked_project(tmp_path, [_fx_spec(), label("auto"), label("none"),
                                           label("fx")])
    moved = _surface_part(dx=30, dy=22)
    res = {t["tool_id"]: t for t in proj.inspect(moved)["tools"]}
    assert res["label_auto"]["status"] == "pass"
    assert res["label_fx"]["status"] == "pass", "the named fixture was not applied"
    assert res["label_none"]["status"] == "fail", \
        "fixture 'none' still followed the part (%s)" % res["label_none"]["message"]


def test_a_fixture_named_before_it_runs_is_an_honest_error(tmp_path):
    from vision_station.core.schema import defaults
    proj, ref = _linked_project(tmp_path, [
        {"id": "early", "type": "histogram",
         "params": dict(defaults("histogram"), roi=[90, 250, 160, 50], fixture="fx")},
        _fx_spec()])
    res = {t["tool_id"]: t for t in proj.inspect(ref)["tools"]}
    assert res["early"]["status"] == "error"
    assert "has not run before" in res["early"]["message"]


def test_a_setting_can_be_driven_by_an_earlier_result(tmp_path):
    """VisionPro's Add Input: a writable setting fed from another tool's output.
    Here the second histogram's lower limit is the first one's mean."""
    from vision_station.core.schema import defaults
    proj, ref = _linked_project(tmp_path, [
        {"id": "white_label", "type": "histogram",
         "params": dict(defaults("histogram"), roi=[95, 255, 150, 40])},
        {"id": "panel", "type": "histogram",
         "params": dict(defaults("histogram"), roi=[300, 200, 150, 60],
                        links={"mean_min": "white_label.mean"})}])
    res = {t["tool_id"]: t for t in proj.inspect(ref)["tools"]}
    assert res["panel"]["status"] == "fail", "the linked limit was not applied"
    assert "under %.1f" % res["white_label"]["values"]["mean"] in res["panel"]["message"]


def test_a_link_with_no_value_is_an_error_not_a_silent_default(tmp_path):
    from vision_station.core.schema import defaults
    proj, ref = _linked_project(tmp_path, [
        {"id": "h", "type": "histogram",
         "params": dict(defaults("histogram"), roi=[95, 255, 150, 40],
                        links={"mean_min": "nowhere.mean"})}])
    r = proj.inspect(ref)["tools"][0]
    assert r["status"] == "error" and "nowhere.mean" in r["message"]


def _link_recipe():
    from vision_station.core.schema import defaults
    mk = lambda tid, ty, **kw: {"id": tid, "type": ty, "params": dict(defaults(ty), **kw)}
    return [mk("fx", "align", roi=[40, 40, 480, 340]),
            mk("smooth", "image_process"),
            mk("hole_a", "find_circle", roi=[100, 130, 40, 40]),
            mk("hole_b", "find_circle", roi=[200, 130, 40, 40]),
            mk("span", "distance"),
            mk("leds", "count_blob", roi=[10, 10, 50, 50]),
            mk("colour", "color_classify")]


def test_links_of_matching_type_are_made():
    from vision_station.core import terminals as T
    tools = _link_recipe()
    assert T.link(tools, "span", "first", "hole_a", "Feature") is None
    assert T.link(tools, "span", "second", "hole_b", "Feature") is None
    assert T.link(tools, "hole_a", "InputImage", "smooth", "OutputImage") is None
    assert T.link(tools, "colour", "source", "leds", "Blobs") is None
    assert T.link(tools, "hole_b", "Fixture", "fx", "Pose") is None
    p = {t["id"]: t["params"] for t in tools}
    assert (p["span"]["first"], p["span"]["second"]) == ("hole_a", "hole_b")
    assert p["hole_a"]["capture"] == "smooth"
    assert p["colour"]["source"] == "leds" and p["hole_b"]["fixture"] == "fx"


def test_links_of_the_wrong_type_are_refused_and_change_nothing():
    from vision_station.core import terminals as T
    tools = _link_recipe()
    before = json.dumps(tools, sort_keys=True)
    why = T.link(tools, "span", "first", "fx", "Pose")     # earlier, but a fixture
    assert why and "needs a feature" in why, why
    assert T.link(tools, "hole_a", "InputImage", "fx", "Pose")
    assert json.dumps(tools, sort_keys=True) == before, "a refused link changed the recipe"


def test_a_link_to_a_later_tool_is_refused():
    """A later tool's output does not exist yet when this one runs; linking to
    it would read nothing, silently."""
    from vision_station.core import terminals as T
    from vision_station.core.schema import defaults
    tools = _link_recipe()
    tools.append({"id": "fx_late", "type": "align", "params": defaults("align")})
    why = T.link(tools, "hole_a", "Fixture", "fx_late", "Pose")
    assert why and "runs after" in why, why
    assert tools[2]["params"].get("fixture", "auto") == "auto"
    assert T.link(tools, "hole_a", "Fixture", "fx", "Pose") is None


def test_compatible_sources_offer_only_legal_earlier_outputs():
    from vision_station.core import terminals as T
    tools = _link_recipe()
    srcs = T.compatible_sources(tools, "span", "first")
    assert srcs == [("hole_a", "Feature"), ("hole_b", "Feature")]
    imgs = T.compatible_sources(tools, "hole_b", "InputImage")
    assert imgs == [(T.SOURCE, "OutputImage"), ("smooth", "OutputImage")]


def test_member_browser_links_a_setting_to_a_result_value():
    from vision_station.core import terminals as T
    tools = _link_recipe()
    params = dict(T.linkable_params(tools[3]))
    assert "min_contrast" in params and "roi" not in params
    assert T.link(tools, "hole_b", "min_contrast", "hole_a", "radius_px") is None
    assert tools[3]["params"]["links"] == {"min_contrast": "hole_a.radius_px"}
    T.unlink(tools, "hole_b", "min_contrast")
    assert "links" not in tools[3]["params"]


def test_renaming_a_tool_keeps_every_link_to_it():
    from vision_station.core import terminals as T
    tools = _link_recipe()
    T.link(tools, "span", "first", "hole_a", "Feature")
    T.link(tools, "hole_b", "min_contrast", "hole_a", "radius_px")
    T.link(tools, "hole_b", "InputImage", "smooth", "OutputImage")
    T.rename_references(tools, "hole_a", "big_hole")
    T.rename_references(tools, "smooth", "blur")
    p = {t["id"]: t["params"] for t in tools}
    assert p["span"]["first"] == "big_hole"
    assert p["hole_b"]["links"]["min_contrast"] == "big_hole.radius_px"
    assert p["hole_b"]["capture"] == "blur"


# ------------------------------------------------------- more VisionPro tools
def _aa_ellipse(cx, cy, a, b, ang, W=320, H=240, seed=0):
    S = 8
    big = np.full((H * S, W * S), 60, np.uint8)
    cv2.ellipse(big, (int(round(cx * S + S / 2 - .5)), int(round(cy * S + S / 2 - .5))),
                (int(round(a * S)), int(round(b * S))), ang, 0, 360, 190, -1, cv2.LINE_AA)
    im = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    im = cv2.GaussianBlur(im, (0, 0), 1.0) + np.random.default_rng(seed).normal(0, 3, im.shape)
    return cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("ttype", ["find_circle", "find_ellipse"])
def test_radial_tools_report_an_empty_region_as_an_error(ttype):
    r, _ = _geo(ttype, {"roi": [500, 500, 40, 40]}, np.zeros((100, 100, 3), np.uint8))
    assert r.status == "error" and "empty region" in r.message


def test_find_ellipse_recovers_centre_axes_and_angle():
    for cx, cy, a, b, ang in ((160.3, 119.6, 55.0, 32.0, 20.0), (158.7, 121.2, 40.0, 26.0, -35.0)):
        img = _aa_ellipse(cx, cy, a, b, ang)
        r, ctx = _geo("find_ellipse", {"roi": [int(cx - a * 1.35), int(cy - a * 1.35),
                                               int(2.7 * a), int(2.7 * a)],
                                       "search_inner": 0.3}, img)
        assert r.status == "pass", r.message
        v = r.values
        assert math.hypot(v["cx"] - cx, v["cy"] - cy) < 0.5, v
        assert abs(v["major_px"] - 2 * a) < 1.0 and abs(v["minor_px"] - 2 * b) < 1.0, v
        d = (v["angle_deg"] - ang) % 180
        assert min(d, 180 - d) < 2.0 or min(abs(d - 90), abs(90 - d)) < 2.0, v
        assert ctx.share["g"]["type"] == "circle"      # Distance can measure to it


def test_sharpness_fails_an_out_of_focus_image(board):
    """A knocked lens makes every tool fail confusingly; this names the cause."""
    from vision_station.core.schema import defaults
    p = dict(defaults("sharpness"), roi=[120, 110, 420, 90])
    sharp, _ = _run("sharpness", p, board, reference=board)
    blurred, _ = _run("sharpness", p, cv2.GaussianBlur(board, (0, 0), 3.0), reference=board)
    assert sharp.status == "pass" and abs(sharp.values["relative"] - 1.0) < 1e-6
    assert blurred.status == "fail" and blurred.values["relative"] < 0.3, blurred.values


def test_color_segmenter_measures_coverage_and_wraps_past_red():
    img = np.full((100, 200, 3), (128, 128, 128), np.uint8)
    img[:, :50] = (0, 0, 230)                       # red: a quarter of the image
    img[:, 150:] = (0, 200, 0)                      # green: another quarter
    red, _ = _geo("color_segment", {"roi": [0, 0, 200, 100], "hue_min": 170, "hue_max": 10,
                                    "min_pct": 20, "max_pct": 30}, img)
    assert red.status == "pass" and abs(red.values["percent"] - 25.0) < 1.0, red.values
    green, _ = _geo("color_segment", {"roi": [0, 0, 200, 100], "hue_min": 45, "hue_max": 75,
                                      "min_pct": 40}, img)
    assert green.status == "fail" and abs(green.values["percent"] - 25.0) < 1.0


def _tiny_classifier(path, classes):
    from make_test_onnx import build_color_classifier
    return build_color_classifier(path, classes)


@onnx_required
def test_classify_sorts_the_region_and_accepts_the_right_class(tmp_path):
    from vision_station.core.schema import defaults
    model = _tiny_classifier(str(tmp_path / "c.onnx"), ["red", "green", "blue"])
    base = dict(defaults("classify"), roi=[0, 0, 64, 64], model=model, normalize="none",
                accept="green", min_confidence=0.6)
    greenish = np.zeros((64, 64, 3), np.uint8)
    greenish[:] = (30, 220, 40)                     # BGR
    reddish = np.zeros((64, 64, 3), np.uint8)
    reddish[:] = (30, 40, 220)
    g, _ = _run("classify", dict(base), greenish)
    r, _ = _run("classify", dict(base), reddish)
    assert g.status == "pass" and g.values["class"] == "green", g.values
    assert r.status == "fail" and r.values["class"] == "red" and "classified as red" in r.message
    assert abs(sum(g.values["scores"].values()) - 1.0) < 1e-3


def test_classify_without_a_model_is_an_honest_error():
    from vision_station.core.schema import defaults
    r, _ = _run("classify", dict(defaults("classify"), roi=[0, 0, 10, 10]),
                np.zeros((10, 10, 3), np.uint8))
    assert r.status == "error" and "model file not found" in r.message


def test_intersect_line_circle_and_circle_circle():
    ctx = Context(reference=np.zeros((10, 10, 3), np.uint8))
    img = np.zeros((10, 10, 3), np.uint8)
    ctx.share["h"] = {"type": "line", "point": [0.0, 100.0], "direction": [1.0, 0.0]}
    ctx.share["c"] = {"type": "circle", "center": [50.0, 100.0], "radius": 30.0}
    ctx.share["c2"] = {"type": "circle", "center": [90.0, 100.0], "radius": 30.0}
    ctx.share["far"] = {"type": "circle", "center": [500.0, 500.0], "radius": 5.0}
    r, _ = _geo("intersect", {"first": "h", "second": "c", "expect_points": 2}, img, ctx)
    assert r.values["points"] == 2 and r.status == "pass"
    assert sorted((r.values["x1"], r.values["x2"])) == [20.0, 80.0]
    r, _ = _geo("intersect", {"first": "c", "second": "c2"}, img, ctx)
    assert r.values["points"] == 2 and abs(r.values["x1"] - 70.0) < 1e-6
    assert abs(abs(r.values["y1"] - 100.0) - math.sqrt(30 ** 2 - 20 ** 2)) < 1e-3   # 3 decimals
    r, _ = _geo("intersect", {"first": "c", "second": "far"}, img, ctx)
    assert r.status == "fail" and "do not meet" in r.message


# ------------------------------------------------------- image tools (new coordinates)
def _project(tmp_path, ref, tools):
    from vision_station.core.engine import Project
    cv2.imwrite(str(tmp_path / "reference.png"), ref)
    with open(str(tmp_path / "project.json"), "w") as f:
        json.dump({"id": "p", "tools": tools}, f)
    return Project(str(tmp_path))


def test_a_tool_reading_a_transformed_image_uses_its_coordinates(tmp_path):
    """Affine transform resamples the fixtured region upright. A tool reading
    its image draws its region in that image's coordinates: no fixture on top
    (the transform already followed the part), and a reference transformed
    the same way. On a moved part the inspection must be unchanged."""
    from vision_station.core.schema import defaults
    ref = _surface_part()
    tools = [_fx_spec(),
             {"id": "label_img", "type": "affine_transform",
              "params": dict(defaults("affine_transform"), roi=[80, 240, 180, 70], scale=2.0)},
             {"id": "label_present", "type": "presence",
              "params": dict(defaults("presence"), roi=[20, 20, 320, 100], min_score=0.8,
                             search_pad=4, capture="label_img")}]
    p = _project(tmp_path, ref, tools)
    for dx, dy in ((0, 0), (30, 22), (-25, 18)):
        res = {t["tool_id"]: t for t in p.inspect(_surface_part(dx=dx, dy=dy), keep_images=True)["tools"]}
        assert res["label_img"]["values"]["width"] == 360 and res["label_img"]["values"]["height"] == 140
        assert res["label_present"]["status"] == "pass", (dx, dy, res["label_present"]["message"])


def test_a_missing_transformed_image_is_an_error_never_the_camera_image(tmp_path):
    from vision_station.core.schema import defaults
    ref = _surface_part()
    tools = [{"id": "gone", "type": "affine_transform",
              "params": dict(defaults("affine_transform"), roi=[80, 240, 180, 70], enabled=False)},
             {"id": "reads_it", "type": "histogram",
              "params": dict(defaults("histogram"), roi=[0, 0, 50, 50], capture="gone")}]
    p = _project(tmp_path, ref, tools)
    r = p.inspect(ref)["tools"][-1]
    assert r["status"] == "error" and "not produced" in r["message"], r


def _ring_with_mark(angle_deg, rotate=0.0):
    img = np.full((300, 300, 3), 40, np.uint8)
    cv2.circle(img, (150, 150), 100, (200, 200, 200), 30)
    a = math.radians(angle_deg)
    cv2.circle(img, (int(150 + 100 * math.cos(a)), int(150 + 100 * math.sin(a))), 9, (20, 20, 20), -1)
    if rotate:
        M = cv2.getRotationMatrix2D((150, 150), -rotate, 1.0)
        img = cv2.warpAffine(img, M, (300, 300), borderMode=cv2.BORDER_REPLICATE)
    return img


def test_polar_unwrap_places_a_mark_by_its_angle():
    img = _ring_with_mark(90)
    r, ctx = _geo("polar_unwrap", {"roi": [35, 35, 230, 230], "inner_radius": 0.6,
                                   "width": 360}, img)
    strip = ctx.images["g"]
    assert strip.shape[1] == 360
    col = int(np.argmin(cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)[strip.shape[0] // 2 - 5:
                                                                 strip.shape[0] // 2 + 5].mean(axis=0)))
    assert abs(col - 90) <= 4, "the mark at 90 deg landed in column %d" % col


def test_polar_unwrap_turns_with_the_part(tmp_path):
    """The fixture's rotation is added to the start angle: rotate the part and
    the mark stays in the same column of the strip."""
    from vision_station.core.schema import defaults
    ref = _ring_with_mark(90)
    for i in range(0, 300, 40):               # texture so the fixture can lock on
        cv2.line(ref, (i, 0), (0, i), (90, 90, 90), 1)
    tools = [{"id": "fx", "type": "align",
              "params": dict(defaults("align"), roi=[10, 10, 280, 280], method="features",
                             n_features=1500, min_score=0.2)},
             {"id": "strip", "type": "polar_unwrap",
              "params": dict(defaults("polar_unwrap"), roi=[35, 35, 230, 230],
                             inner_radius=0.6, width=360)}]
    p = _project(tmp_path, ref, tools)
    cols = []
    for rot in (0.0, 25.0):
        M = cv2.getRotationMatrix2D((150, 150), -rot, 1.0)
        img = cv2.warpAffine(ref, M, (300, 300), borderMode=cv2.BORDER_REPLICATE)
        out = p.inspect(img, keep_images=True)["images"]["strip"]
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        cols.append(int(np.argmin(g[g.shape[0] // 2 - 5:g.shape[0] // 2 + 5].mean(axis=0))))
    assert abs(cols[0] - cols[1]) <= 4, "the mark moved from column %d to %d" % tuple(cols)


def test_two_image_difference_shows_only_what_changed(tmp_path):
    from vision_station.core.schema import defaults
    ref = _surface_part()
    bad = _surface_part(dx=20, dy=10)
    cv2.line(bad, (320, 200), (420, 240), (40, 40, 40), 3)          # a scratch
    tools = [_fx_spec(), {"id": "diff", "type": "image_combine",
                          "params": dict(defaults("image_combine"))}]
    p = _project(tmp_path, ref, tools)
    out = p.inspect(bad, keep_images=True)["images"]["diff"]
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    assert g[215:225, 360:380].max() > 60, "the scratch did not show in the difference"
    inner = g[100:180, 100:250]
    assert inner.mean() < 6, "the part was not lined up before differencing (%.1f)" % inner.mean()


# ------------------------------------------------------- fixtures, calibration, geometry
def test_fixture_from_two_holes_follows_the_part(tmp_path):
    """Two fiducials define the pose; tools after it follow the part."""
    from vision_station.core.schema import defaults
    ref = _surface_part()
    cv2.circle(ref, (120, 120), 14, (20, 20, 20), -1)
    cv2.circle(ref, (440, 330), 14, (20, 20, 20), -1)
    # search regions big enough for how far each fiducial can move -- 2 degrees
    # about the centre moves the far hole ~14 px, out of a tight region
    tools = [{"id": "h1", "type": "find_circle", "params": dict(defaults("find_circle"),
              roi=[75, 75, 90, 90], fixture="none", search_inner=0.1)},
             {"id": "h2", "type": "find_circle", "params": dict(defaults("find_circle"),
              roi=[395, 285, 90, 90], fixture="none", search_inner=0.1)},
             {"id": "fx2", "type": "fixture_npoint", "params": dict(defaults("fixture_npoint"),
              points="h1,h2")},
             {"id": "label", "type": "presence", "params": dict(defaults("presence"),
              roi=[90, 250, 160, 50], min_score=0.8, search_pad=4)}]
    p = _project(tmp_path, ref, tools)
    h, w = ref.shape[:2]
    # a movement Find circle can follow: each hole's centre stays inside it
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 1.0, 1.0)
    M[0, 2] += 4
    M[1, 2] -= 3
    moved = cv2.warpAffine(ref, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    res = {t["tool_id"]: t for t in p.inspect(moved)["tools"]}
    assert res["h1"]["status"] == "pass" and res["h2"]["status"] == "pass"
    fx = res["fx2"]
    assert fx["status"] == "pass", fx["message"]
    Mf = np.array(fx["values"]["matrix"])
    for x, y in ((120, 120), (300, 200)):
        true = M @ np.array([x, y, 1.0])
        got = Mf @ np.array([x, y, 1.0])
        assert np.hypot(*(true - got)) < 1.0, (true, got)
    assert res["label"]["status"] == "pass", "the tool after the fixture did not follow"

    # A fiducial moved further than Find circle can follow: it fails, and the
    # fixture must refuse rather than build a pose from the failed guess -- it
    # used to, and set a pose 8 px out with a 2.6% scale error.
    M2 = cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1.0)
    M2[0, 2] += 9
    M2[1, 2] -= 6
    far = cv2.warpAffine(ref, M2, (w, h), borderMode=cv2.BORDER_REPLICATE)
    res = {t["tool_id"]: t for t in p.inspect(far)["tools"]}
    assert res["h2"]["status"] == "fail"
    assert res["fx2"]["status"] == "fail" and "h2" in res["fx2"]["message"], res["fx2"]
    assert res["fx2"]["values"].get("matrix") is None


def test_distance_refuses_a_feature_from_a_failed_tool(tmp_path):
    """The same rule for Distance: a failed fit is not measured from."""
    from vision_station.tools.base import ToolResult
    ctx = Context(reference=np.zeros((10, 10, 3), np.uint8))
    ctx.share.update(a={"type": "circle", "center": [0.0, 0.0], "radius": 5.0},
                     b={"type": "circle", "center": [30.0, 40.0], "radius": 5.0})
    ctx.results["a"] = ToolResult("a", "find_circle", "pass", 1.0)
    ctx.results["b"] = ToolResult("b", "find_circle", "fail", 0.0)
    r, _ = _geo("distance", {"first": "a", "second": "b"}, np.zeros((10, 10, 3), np.uint8), ctx)
    assert r.status in ("error", "fail") and "value_px" not in r.values, r.values
    ctx.results["b"] = ToolResult("b", "find_circle", "pass", 1.0)
    r, _ = _geo("distance", {"first": "a", "second": "b"}, np.zeros((10, 10, 3), np.uint8), ctx)
    assert r.values["value_px"] == pytest.approx(50.0)


def test_npoint_calibration_recovers_a_rotated_scale():
    ang = math.radians(5.0)
    k = 0.1
    Mt = k * np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    pix = [(100, 100), (500, 120), (480, 400), (120, 380)]
    pairs = "; ".join("%g,%g = %.6f,%.6f" % ((px, py) + tuple(Mt @ [px, py] + [3.0, -2.0]))
                      for px, py in pix)
    r, ctx = _geo("calibrate_npoint", {"pairs": pairs}, np.zeros((10, 10, 3), np.uint8))
    assert r.status == "pass" and abs(r.values["units_per_px"] - 0.1) < 1e-6
    assert ctx.mm_per_px == pytest.approx(0.1, abs=1e-6)
    r, _ = _geo("calibrate_npoint", {"pairs": "1,2 = 3,4"}, np.zeros((10, 10, 3), np.uint8))
    assert r.status == "error"


def test_create_geometry_lines_midpoints_perpendiculars_and_circles():
    ctx = Context(reference=np.zeros((10, 10, 3), np.uint8))
    img = np.zeros((10, 10, 3), np.uint8)
    ctx.share.update(a={"type": "point", "point": [0.0, 0.0]}, b={"type": "point", "point": [100.0, 0.0]},
                     c={"type": "circle", "center": [50.0, 50.0], "radius": 5.0})
    r, _ = _geo("construct", {"operation": "midpoint", "first": "a", "second": "b"}, img, ctx)
    assert (r.values["x"], r.values["y"]) == (50.0, 0.0)
    r, _ = _geo("construct", {"operation": "line_through_points", "first": "a", "second": "b"}, img, ctx)
    assert r.values["angle_deg"] == 0.0 and r.values["length_px"] == 100.0
    ctx.share["ab"] = ctx.share["g"]
    r, _ = _geo("construct", {"operation": "perpendicular", "first": "ab", "second": "c"}, img, ctx)
    assert abs(abs(r.values["angle_deg"]) - 90.0) < 1e-6
    ctx.share["top"] = {"type": "point", "point": [50.0, -50.0]}
    r, _ = _geo("construct", {"operation": "circle_through_points", "first": "a", "second": "b",
                              "third": "top"}, img, ctx)
    assert (r.values["cx"], r.values["cy"], r.values["radius_px"]) == (50.0, 0.0, 50.0)


def _bead_image(gap=None, narrow=None):
    img = np.full((200, 500, 3), 200, np.uint8)
    for x in range(40, 461):
        if gap and gap[0] <= x <= gap[1]:
            continue
        half = 3 if narrow and narrow[0] <= x <= narrow[1] else 6
        cv2.line(img, (x, 100 - half), (x, 100 + half), (60, 60, 60), 1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_a_bead_path_is_drawn_open():
    """No line from the path's end back to its start."""
    path = [[40, 100], [240, 60], [460, 100]]
    img = np.full((200, 500, 3), 200, np.uint8)
    cv2.polylines(img, [np.int32(path)], False, (60, 60, 60), 12)
    r, _ = _geo("bead", {"shape": "polygon", "points": path, "roi": [40, 60, 420, 40],
                         "width": 12}, img)
    segs = [g for g in r.graphics if g.kind == "line" and g.status != "search" and
            len(g.points) == 2 and g.role == "region"]
    closing = [g for g in segs if {tuple(map(round, g.points[0])), tuple(map(round, g.points[1]))}
               == {(40, 100), (460, 100)}]
    assert len(segs) == 2 and not closing, [g.points for g in segs]
    assert not any(g.kind == "poly" for g in r.graphics), "a closed outline was drawn"


def test_bead_inspect_finds_gaps_and_narrow_sections():
    p = {"roi": [50, 80, 400, 40], "width": 12, "width_tolerance": 3, "stations": 20}
    ok, _ = _geo("bead", dict(p), _bead_image())
    assert ok.status == "pass", ok.message
    assert abs(ok.values["mean_width"] - 12) < 1.5
    gap, _ = _geo("bead", dict(p), _bead_image(gap=(200, 240)))
    assert gap.status == "fail" and gap.values["gaps"] >= 1
    narrow, _ = _geo("bead", dict(p), _bead_image(narrow=(300, 360)))
    assert narrow.status == "fail" and narrow.values["gaps"] == 0 and "width" in narrow.message


def test_data_analysis_statistics_limits_and_drift():
    from vision_station.tools.base import ToolResult
    from vision_station.tools.construct_tools import reset_series
    reset_series()
    ctx = Context(reference=np.zeros((10, 10, 3), np.uint8))
    img = np.zeros((10, 10, 3), np.uint8)
    p = {"source": "gauge.v", "lower_limit": "9.0", "upper_limit": "11.0", "min_cpk": 1.33,
         "min_samples": 10}
    rng = np.random.default_rng(1)
    for v in rng.normal(10.0, 0.1, 30):
        ctx.results["gauge"] = ToolResult("gauge", "x", "pass", 1.0, {"v": float(v)})
        r, _ = _geo("data_analysis", dict(p), img, ctx)
    assert r.status == "pass" and r.values["count"] == 30 and r.values["cpk"] > 2.5
    ctx.results["gauge"] = ToolResult("gauge", "x", "pass", 1.0, {"v": 11.5})
    r, _ = _geo("data_analysis", dict(p), img, ctx)
    assert r.status == "fail" and "over" in r.message
    reset_series()
    for v in rng.normal(10.7, 0.12, 30):           # in limits, but drifting to the top
        ctx.results["gauge"] = ToolResult("gauge", "x", "pass", 1.0, {"v": float(min(v, 10.99))})
        r, _ = _geo("data_analysis", dict(p), img, ctx)
    assert r.status == "fail" and "Cpk" in r.message, r.values


def test_fit_shape_fits_an_ellipse_through_points():
    ctx = Context(reference=np.zeros((10, 10, 3), np.uint8))
    ts = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    ctx.share["pts"] = [{"cx": 200 + 60 * math.cos(t), "cy": 150 + 30 * math.sin(t)} for t in ts]
    r, _ = _geo("fit_shape", {"fit": "ellipse", "source": "pts"}, np.zeros((10, 10, 3), np.uint8), ctx)
    assert abs(r.values["major"] - 120) < 0.5 and abs(r.values["minor"] - 60) < 0.5, r.values


# ------------------------------------------------------- graphics settings
def test_a_tools_graphics_settings_apply_to_its_results(tmpproj):
    """VisionPro's Graphics tab: region, results, labels and search guides can
    each be switched off per tool, applied once in the engine."""
    from vision_station.core.engine import Project
    from vision_station.core.schema import defaults
    root, pid, board = tmpproj
    p = Project(os.path.join(root, pid))
    p.spec["tools"] = [{"id": "hole", "type": "find_circle",
                        "params": dict(defaults("find_circle"), roi=[140, 120, 60, 60],
                                       channel="blue")}]

    def graphics(opt):
        if opt is None:
            p.spec["tools"][0]["params"].pop("graphics", None)
        else:
            p.spec["tools"][0]["params"]["graphics"] = opt
        return p.inspect(board, tools=p.build_tools())["tools"][0]["graphics"]

    full = graphics(None)
    roles = {(g.get("role"), g["status"]) for g in full}
    assert ("region", "pass") in roles and any(s == "search" for _r, s in roles)
    assert any(g.get("label") for g in full)
    no_region = graphics({"region": False})
    assert not any(g.get("role") == "region" for g in no_region)
    no_search = graphics({"search": False})
    assert not any(g["status"] == "search" for g in no_search)
    no_labels = graphics({"labels": False})
    assert not any(g.get("label") for g in no_labels) and len(no_labels) == len(full)
    only_results = graphics({"region": False, "search": False})
    assert only_results and all(g.get("role") != "region" and g["status"] != "search"
                                for g in only_results)


def test_find_tools_draw_where_they_search():
    """Find Circle draws its inner and outer search radii, Find Line each
    caliper strip, the edge-pair caliper its measuring direction."""
    img = _aa_circle(160, 120, 40)
    r, _ = _geo("find_circle", {"roi": [106, 66, 108, 108], "search_inner": 0.5}, img)
    assert sum(1 for g in r.graphics if g.status == "search") == 2
    r, _ = _geo("find_line", {"roi": [40, 70, 240, 100], "num_calipers": 9},
                _comb([100.0], [60.0], clutter=False, W=320, H=240))
    assert r.status == "fail"                    # no edge in that direction
    assert sum(1 for g in r.graphics if g.status == "search") == 9, \
        "a failed tool must still show where it searched"
    blank = np.full((200, 200, 3), 120, np.uint8)
    r, _ = _geo("find_circle", {"roi": [50, 50, 100, 100]}, blank)
    assert r.status == "fail" and sum(1 for g in r.graphics if g.status == "search") == 2


# ------------------------------------------------------- fixture on a static scene
def _static_scene(dx, dy):
    """A textured scene where only the part inside the fixture region moves --
    as on a line, where the robot, conveyor and cell stay put."""
    rng = np.random.default_rng(5)
    # a busy cell: sharp-edged clutter everywhere, the strongest features in view
    scene = np.full((600, 800), 90, np.uint8)
    for _ in range(420):
        x, y = int(rng.integers(0, 780)), int(rng.integers(0, 580))
        cv2.rectangle(scene, (x, y), (x + int(rng.integers(6, 30)), y + int(rng.integers(6, 30))),
                      int(rng.integers(0, 255)), -1)
    scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    # the part has sharp detail of its own -- labels, screws, edges
    prng = np.random.default_rng(9)
    part = np.full((160, 200), 170, np.uint8)
    for _ in range(40):
        x, y = int(prng.integers(0, 185)), int(prng.integers(0, 145))
        cv2.rectangle(part, (x, y), (x + int(prng.integers(5, 16)), y + int(prng.integers(5, 16))),
                      int(prng.integers(0, 120)), -1)
    img = scene.copy()
    img[220 + dy:380 + dy, 300 + dx:500 + dx] = cv2.cvtColor(part, cv2.COLOR_GRAY2BGR)
    return img


def test_features_fixture_follows_a_small_part_not_the_background():
    """Features were matched across the whole image, so the static background
    outvoted a small part: on the real chassis, with the region a third of the
    frame, every movement was reported as 'nothing moved'."""
    from vision_station.core.schema import defaults
    ref = _static_scene(0, 0)
    spec = {"id": "fx", "type": "align",
            "params": dict(defaults("align"), roi=[290, 210, 220, 180], method="features",
                           n_features=1500, min_score=0.2)}
    for dx, dy in ((35, -20), (-40, 30)):
        ctx = Context(reference=ref)
        r = make_tool(spec).run(_static_scene(dx, dy), ctx)
        assert ctx.fixture is not None, r.message
        got = (float(ctx.fixture[0, 2]), float(ctx.fixture[1, 2]))
        assert abs(got[0] - dx) < 2 and abs(got[1] - dy) < 2, \
            "part moved %s, fixture says %s" % ((dx, dy), got)


def test_features_fixture_does_not_reuse_a_stale_reference():
    """Reference features are cached; a changed reference must not reuse them."""
    from vision_station.core.schema import defaults
    spec = {"id": "fx", "type": "align",
            "params": dict(defaults("align"), roi=[290, 210, 220, 180], method="features",
                           n_features=1500, min_score=0.2)}
    ref_a = _static_scene(0, 0)
    ctx = Context(reference=ref_a)
    make_tool(spec).run(_static_scene(20, 10), ctx)
    ref_b = _static_scene(20, 10)               # the moved scene becomes the reference
    ctx2 = Context(reference=ref_b)
    make_tool(spec).run(_static_scene(20, 10), ctx2)
    assert ctx2.fixture is not None
    assert abs(float(ctx2.fixture[0, 2])) < 2 and abs(float(ctx2.fixture[1, 2])) < 2, \
        "the fixture used features cached from the previous reference"


# ------------------------------------------------------- analysis tools
def test_expression_evaluator_allows_only_safe_expressions():
    """A job file is data; an expression in it must not be able to run code."""
    from vision_station.tools.analysis import evaluate, ExpressionError
    vals = {("a", "x"): 3.0, ("b", "y"): 4.0, ("c", "status"): "pass"}
    look = lambda t, k: vals.get((t, k))
    assert evaluate("a.x > 2 and b.y <= 4", look) is True
    assert evaluate("abs(a.x - b.y) < 1.5", look) is True
    assert evaluate("1 < a.x < 3", look) is False
    assert evaluate("c.status == 'pass' or a.x > 100", look) is True
    for attack in ("__import__('os').system('echo x')", "a.x.__class__",
                   "open('f')", "[1, 2]", "lambda: 1", "a.x if 1 else 2",
                   "().__class__.__bases__"):
        with pytest.raises(ExpressionError):
            evaluate(attack, look)
    with pytest.raises(ExpressionError):
        evaluate("missing.v > 1", look)


def test_results_analysis_accepts_and_rejects_from_earlier_results(tmpproj):
    from vision_station.core.engine import Project
    root, pid, board = tmpproj
    p = Project(os.path.join(root, pid))
    first = p.spec["tools"][0]["id"]
    for expr, want in (("%s.count >= 0" % first, "pass"), ("%s.count > 999" % first, "fail"),
                       ("%s.nothing > 1" % first, "error"), ("1 +", "error")):
        p.spec["tools"].append({"id": "logic", "type": "results_analysis",
                                "params": {"expression": expr}})
        r = {t["tool_id"]: t for t in p.inspect(board, tools=p.build_tools())["tools"]}["logic"]
        assert r["status"] == want, (expr, r["message"])
        p.spec["tools"].pop()


def test_intersect_finds_the_corner_of_two_lines():
    import math
    ctx = Context(reference=np.zeros((10, 10, 3), np.uint8))
    a = math.radians(20)
    ctx.share["h"] = {"type": "line", "point": [100.0, 50.0], "direction": [1.0, 0.0]}
    ctx.share["v"] = {"type": "line", "point": [30.0, 0.0],
                      "direction": [math.sin(a), math.cos(a)]}     # 70 deg to horizontal
    img = np.zeros((10, 10, 3), np.uint8)
    r, _ = _geo("intersect", {"first": "h", "second": "v"}, img, ctx)
    x_true = 30.0 + 50.0 * math.tan(a)
    assert abs(r.values["x"] - x_true) < 1e-3 and abs(r.values["y"] - 50.0) < 1e-3
    assert abs(r.values["angle_deg"] - 70.0) < 1e-3          # reported to 3 decimals
    assert ctx.share["g"]["type"] == "point"
    ctx.share["p"] = {"type": "line", "point": [0.0, 90.0], "direction": [1.0, 0.0]}
    r, _ = _geo("intersect", {"first": "h", "second": "p"}, img, ctx)
    assert r.status == "fail" and "parallel" in r.message


def test_fit_line_through_the_demo_leds_and_catch_one_out_of_line(board):
    """The four LEDs on the demo board sit in a row: a line through their
    centres fits closely. Move one LED down and the fit flags it."""
    from vision_station.core.schema import defaults
    blob = lambda img: make_tool({"id": "leds", "type": "count_blob",
                                  "params": dict(defaults("count_blob"), roi=[120, 110, 420, 80],
                                                 channel="saturation", min_area=200)})
    ctx = Context(reference=board)
    blob(board).run(board, ctx)
    assert len(ctx.share["leds"]) == 4
    r, _ = _geo("fit_shape", {"fit": "line", "source": "leds", "max_deviation": 2.0}, board, ctx)
    assert r.status == "pass" and r.values["max_deviation"] < 1.0, r.values
    moved = board.copy()
    cv2.circle(moved, (370, 150), 24, (150, 140, 130), -1)
    cv2.circle(moved, (370, 162), 22, (220, 160, 60), -1)          # 12 px low
    ctx2 = Context(reference=board)
    blob(moved).run(moved, ctx2)
    r2, _ = _geo("fit_shape", {"fit": "line", "source": "leds", "max_deviation": 2.0},
                 moved, ctx2)
    assert r2.status == "fail" and r2.values["max_deviation"] > 4.0, r2.values


def test_fit_circle_through_a_bolt_circle():
    import math
    ctx = Context(reference=np.zeros((10, 10, 3), np.uint8))
    cx, cy, rad = 200.0, 150.0, 80.0
    ctx.share["holes"] = [{"cx": cx + rad * math.cos(a), "cy": cy + rad * math.sin(a)}
                          for a in np.linspace(0, 2 * math.pi, 6, endpoint=False)]
    r, _ = _geo("fit_shape", {"fit": "circle", "source": "holes"}, np.zeros((10, 10, 3), np.uint8), ctx)
    assert abs(r.values["cx"] - cx) < 1e-3 and abs(r.values["radius"] - rad) < 1e-3
    assert r.values["max_deviation"] < 1e-3


def test_fit_shape_uses_shape_count_matches(board):
    """Count by shape now reports where its matches are, so a line can be
    fitted through them."""
    from vision_station.core.schema import defaults
    ctx = Context(reference=board)
    r = make_tool({"id": "screws", "type": "count_template",
                   "params": dict(defaults("count_template"), roi=[20, 20, 600, 380],
                                  template_roi=[52, 52, 40, 40], min_score=0.8)}).run(board, ctx)
    assert r.status != "error", r.message
    ctx.results["screws"] = r
    assert len(r.values["matches"]) == r.values["count"] >= 2
    f, _ = _geo("fit_shape", {"fit": "line", "source": "screws"}, board, ctx)
    assert f.values["points"] == r.values["count"]


def test_color_match_passes_the_same_colour_and_fails_a_different_one():
    ref = np.full((100, 160, 3), (40, 40, 40), np.uint8)
    cv2.rectangle(ref, (40, 30), (120, 70), (30, 200, 230), -1)          # yellow label
    same = ref.copy()
    cv2.circle(same, (80, 50), 3, (255, 255, 255), -1)                  # a glint
    blue = ref.copy()
    cv2.rectangle(blue, (40, 30), (120, 70), (200, 120, 30), -1)
    p = {"roi": [45, 35, 70, 30], "max_delta_e": 10.0}
    ok, _ = _geo("color_match", dict(p), same, Context(reference=ref))
    bad, _ = _geo("color_match", dict(p), blue, Context(reference=ref))
    assert ok.status == "pass" and ok.values["delta_e"] < 2.0, ok.values
    assert bad.status == "fail" and bad.values["delta_e"] > 30, bad.values


# ------------------------------------------------------- diagnostic records
def _records_ctx(img):
    ctx = Context(reference=img)
    ctx.want_records = True
    return ctx


def test_caliper_projection_marks_the_edges_it_found():
    """The plot must show the tool's own data: an edge mark in the plot lies
    at the same column as the edge the caliper measured."""
    from vision_station.tools.plots import profile_plot
    img = _comb([40.3, 90.55, 140.8], [20.25] * 3, clutter=False)
    ctx = _records_ctx(img)
    r, _ = _geo("caliper", {"roi": [10, 20, 380, 40], "pair_polarity": "dark"}, img, ctx)
    plot = ctx.records["g"]["LastRun.Projection"]
    assert plot.shape[2] == 3 and plot.shape[1] >= 600
    # pass-coloured vertical lines should appear where the pair edges are
    green = np.all(plot == (74, 158, 30), axis=2)
    cols = np.where(green.sum(axis=0) > plot.shape[0] * 0.4)[0]
    assert len(cols) >= 4, "the edges were not marked"
    left, right = 46, plot.shape[1] - 12
    span = 380 - 1
    for start in (40.3 - 10, 60.55 - 10):                 # first pair, crop coords
        x = left + (right - left) * start / span
        assert np.min(np.abs(cols - x)) <= 3, "no mark near the edge at %.1f" % start


def test_histogram_record_and_blob_mask_are_published():
    from vision_station.core.schema import defaults
    img = np.full((120, 160, 3), 60, np.uint8)
    cv2.circle(img, (80, 60), 25, (200, 200, 200), -1)
    ctx = _records_ctx(img)
    _geo("histogram", {"roi": [10, 10, 140, 100]}, img, ctx)
    assert "LastRun.Histogram" in ctx.records["g"]
    ctx2 = _records_ctx(img)
    make_tool({"id": "b", "type": "count_blob",
               "params": dict(defaults("count_blob"), roi=[10, 10, 140, 100])}).run(img, ctx2)
    mask = ctx2.records["b"]["LastRun.BlobImage"]
    assert mask.shape[:2] == (100, 140) and mask.max() == 255


def test_records_cost_nothing_in_a_production_run(tmpproj):
    """Records are drawn only when asked for: a line run must not spend time
    on plots nobody sees, nor carry the images."""
    from vision_station.core.engine import Project
    root, pid, board = tmpproj
    p = Project(os.path.join(root, pid))
    img = board
    fast = p.inspect(img)
    assert fast["records"] == {} and fast["images"] == {}
    rich = p.inspect(img, keep_images=True)
    assert rich["records"], "no records even when asked for"
    some = next(iter(rich["records"].values()))
    assert "LastRun.RegionImage" in some


# ------------------------------------------------------- lens calibration
_LW, _LH = 640, 480
_LK = np.array([[560, 0, _LW / 2 + 4], [0, 560, _LH / 2 - 3], [0, 0, 1]], np.float64)
_LD = np.array([-0.30, 0.10, 0, 0, 0], np.float64)
_LPAT, _LSQ = (7, 5), 30.0


def _lens_maps():
    gx, gy = np.meshgrid(np.arange(_LW, dtype=np.float32), np.arange(_LH, dtype=np.float32))
    u = cv2.undistortPoints(np.stack([gx.ravel(), gy.ravel()], 1).reshape(-1, 1, 2),
                            _LK, _LD, P=_LK).reshape(_LH, _LW, 2)
    return u[..., 0].astype(np.float32), u[..., 1].astype(np.float32)


def _distort(ideal, maps):
    return cv2.remap(ideal, maps[0], maps[1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(120, 120, 120))


def _lens_views(n_centre=10, n_edge=8):
    """Checkerboard photographs through one known distorting lens. Views come
    from real 3D poses: a random homography is a plane seen by SOME camera, not
    one consistent camera, and calibrating against those returned a focal
    length three times too long."""
    s = int(_LSQ)
    cols, rows = _LPAT[0] + 1, _LPAT[1] + 1
    board = np.full((rows * s + 2 * s, cols * s + 2 * s), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[s + r * s:s + (r + 1) * s, s + c * s:s + (c + 1) * s] = 0
    bh, bw = board.shape
    maps = _lens_maps()
    T = np.array([[1, 0, -bw / 2.0], [0, 1, -bh / 2.0], [0, 0, 1]])
    out = []
    for i in range(n_centre + n_edge):
        r = np.random.default_rng(i)
        R, _ = cv2.Rodrigues(np.array([r.uniform(-.4, .4), r.uniform(-.4, .4), r.uniform(-.2, .2)]))
        if i < n_centre:
            t = [r.uniform(-50, 50), r.uniform(-40, 40), r.uniform(430, 560)]
        else:                                        # reaching the frame edges
            t = [r.choice([-1, 1]) * r.uniform(100, 150),
                 r.choice([-1, 1]) * r.uniform(70, 110), r.uniform(470, 540)]
        H = _LK @ np.column_stack([R[:, 0], R[:, 1], t]) @ T
        ideal = cv2.cvtColor(cv2.warpPerspective(board, H, (_LW, _LH), borderValue=120),
                             cv2.COLOR_GRAY2BGR)
        im = cv2.GaussianBlur(_distort(ideal, maps).astype(np.float32), (0, 0), 0.8)
        out.append(np.clip(im + r.normal(0, 2, im.shape), 0, 255).astype(np.uint8))
    return out, maps


def test_lens_calibration_recovers_a_known_lens():
    from vision_station.core import calibration as C
    views, _maps = _lens_views()
    cal = C.calibrate(views, _LPAT, _LSQ)
    K = np.array(cal["camera_matrix"])
    assert cal["rms_px"] < 0.3, cal["rms_px"]
    assert abs(K[0, 0] - 560) < 8, K[0, 0]
    assert abs(K[0, 2] - _LK[0, 2]) < 4 and abs(K[1, 2] - _LK[1, 2]) < 4
    assert abs(cal["dist_coeffs"][0] - _LD[0]) < 0.04


def test_lens_correction_straightens_an_edge_near_the_border():
    """Measured: a straight edge near the top of the frame bowed 6 px as the lens
    saw it and 0.07 px after correction -- the same as correcting with the true
    lens -- once the calibration views reached the frame edges."""
    from vision_station.core import calibration as C
    from vision_station.core.schema import defaults
    views, maps = _lens_views()
    cal = C.calibrate(views, _LPAT, _LSQ)
    ideal = np.full((_LH, _LW, 3), 60, np.uint8)
    ideal[70:] = 190
    edge = _distort(ideal, maps)

    def rms(img):
        r = make_tool({"id": "l", "type": "find_line",
                       "params": dict(defaults("find_line"), roi=[60, 15, 520, 110],
                                      num_calipers=30, num_to_ignore=0, max_rms=99)}
                      ).run(img, Context(reference=img))
        return r.values["rms_px"]

    bent, fixed = rms(edge), rms(C.undistort(edge, cal))
    assert bent > 2.0, "the test lens is not bending the edge (%.2f px)" % bent
    assert fixed < 0.4, "the corrected edge still bows %.2f px" % fixed


def test_calibration_corrects_the_reference_as_well_as_live_frames(tmp_path):
    """Tools compare the live frame with the reference. Correcting only the live
    frame would mismatch everywhere, most at the edges."""
    from vision_station.core import calibration as C
    from vision_station.core.engine import Project
    views, maps = _lens_views()
    cal = C.calibrate(views, _LPAT, _LSQ)
    ref = views[0]
    cv2.imwrite(str(tmp_path / "reference.png"), ref)
    with open(str(tmp_path / "project.json"), "w") as f:
        json.dump({"id": "p", "tools": []}, f)
    C.save(str(tmp_path), cal)
    proj = Project(str(tmp_path))
    expected = C.undistort(ref, cal)
    assert float(np.abs(proj.reference.astype(int) - expected.astype(int)).mean()) < 1.0
    assert proj.inspect(ref) is not None


def test_calibration_with_too_few_boards_says_so():
    from vision_station.core import calibration as C
    blank = [np.full((_LH, _LW, 3), 128, np.uint8)] * 4
    with pytest.raises(ValueError) as e:
        C.calibrate(blank, _LPAT, _LSQ)
    assert "found in only 0" in str(e.value)


# ------------------------------------------------------- image processing
def test_image_processing_output_feeds_a_later_tool(tmp_path):
    """A median filter ahead of Find Circle, chained through the recipe. With 12%
    speckle, the circle centre was 0.78 px out read raw and 0.16 px read
    through the filter."""
    from vision_station.core.engine import Project
    from vision_station.core.schema import defaults
    S, W, H = 8, 320, 240
    cx, cy, rad = 160.4, 119.7, 40.0
    big = np.full((H * S, W * S), 60, np.uint8)
    cv2.circle(big, (int(cx * S + 3.5), int(cy * S + 3.5)), int(rad * S), 190, -1, cv2.LINE_AA)
    clean = cv2.GaussianBlur(cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA), (0, 0), 1.0)
    cv2.imwrite(str(tmp_path / "reference.png"), cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR))
    def err(noisy, use_filter):
        tools = []
        if use_filter:
            tools.append({"id": "despeckle", "type": "image_process",
                          "params": dict(defaults("image_process"), operator="median")})
        tools.append({"id": "hole", "type": "find_circle",
                      "params": dict(defaults("find_circle"), roi=[106, 66, 108, 108],
                                     search_inner=0.5,
                                     capture="despeckle" if use_filter else "default")})
        with open(str(tmp_path / "project.json"), "w") as f:
            json.dump({"id": "p", "tools": tools}, f)
        r = Project(str(tmp_path)).inspect(noisy)
        v = [t for t in r["tools"] if t["tool_id"] == "hole"][0]["values"]
        return math.hypot(v["cx"] - cx, v["cy"] - cy)

    # Several noise patterns, not one: with a single seed the raw read happened
    # to be accurate already -- outlier rejection coped -- and the test proved
    # nothing either way. The filter's value is mostly in the worst case.
    raw, filtered = [], []
    for seed in range(6):
        n = clean.copy()
        m = np.random.default_rng(seed).random(n.shape)
        n[m < 0.06] = 0
        n[m > 0.94] = 255
        n = cv2.cvtColor(n, cv2.COLOR_GRAY2BGR)
        raw.append(err(n, False))
        filtered.append(err(n, True))
    assert np.mean(filtered) < np.mean(raw) / 2, (raw, filtered)
    assert max(filtered) < 0.4, "worst filtered error %.3f px" % max(filtered)


@pytest.mark.parametrize("op", ["smooth", "median", "sharpen", "equalize", "threshold",
                                "erode", "dilate", "open", "close", "edges", "invert",
                                "channel", "gain"])
def test_every_operator_preserves_geometry(op, board):
    """Same size, three channels, pixels where they were -- otherwise every
    region drawn after the filter would silently land in the wrong place."""
    from vision_station.tools.imageproc import apply
    from vision_station.core.schema import defaults
    out = apply(board, op, defaults("image_process"))
    assert out.shape == board.shape, "%s changed the image size" % op
    assert out.dtype == np.uint8


# ------------------------------------------------------------------ jobs
def _mini_job(tmp_path, n_sides=2):
    from vision_station.core.job import create_job
    from vision_station.core.schema import defaults
    job = create_job(str(tmp_path / "job"), "Test job")
    ref = _surface_part()
    for i in range(n_sides):
        tools = [{"id": "fixture", "type": "align",
                  "params": dict(defaults("align"), roi=[40, 40, 480, 340],
                                 method="template", min_score=0.4,
                                 angle_range=[-6, 6], angle_step=1.0)},
                 {"id": "surface", "type": "trained_surface",
                  "params": dict(defaults("trained_surface"), roi=[80, 110, 400, 220])}]
        job.add_side("side%d" % i, "Side %d" % i, ref, tools)
    return job, ref


def test_job_sides_train_from_ok_ng_folders_and_inspect_together(tmp_path):
    """One job, several sides, each with its own fixture and a tool trained from
    OK/NG images stored as plain files."""
    job, ref = _mini_job(tmp_path)
    for side in job.sides:
        side.add_training_image("surface", ref, "ok")
        side.add_training_image("surface", _surface_part(dx=8, dy=-5, deg=1), "ok")
        side.add_training_image("surface", _surface_part(seed=1, scratch=True), "ng")
        rep = side.train_tool("surface")
        assert rep["separates"], rep
        assert len(side.training_images("surface")) == 3
    job.load()
    good = {s.side_id: _surface_part(dx=15, dy=10, deg=2) for s in job.sides}
    assert job.inspect(good)["verdict"] == "PASS"

    bad = dict(good)
    bad["side1"] = _surface_part(seed=4, dx=15, dy=10, deg=2, scratch=True)
    r = job.inspect(bad)
    assert r["verdict"] == "FAIL"
    assert r["sides"]["side0"]["verdict"] == "PASS"
    assert r["sides"]["side1"]["verdict"] == "FAIL", "the failing side was not identified"


def test_trained_tool_finds_its_model_when_opened_by_relative_path(tmp_path, monkeypatch):
    """The loader resolves model paths against the project folder and the
    trained tool resolved them again; with a relative folder the path doubled
    and the tool reported 'not trained yet' next to its own model."""
    from vision_station.core.job import Job
    job, ref = _mini_job(tmp_path, 1)
    side = job.sides[0]
    side.add_training_image("surface", ref, "ok")
    side.train_tool("surface")
    monkeypatch.chdir(str(tmp_path))
    rel = Job("job").sides[0]                        # opened by a relative path
    r = rel.inspect(ref)
    status = {t["tool_id"]: t for t in r["tools"]}["surface"]
    assert status["status"] == "pass", status["message"]


def test_job_with_a_side_not_photographed_is_incomplete_not_passed(tmp_path):
    """A job that only looked at some of its sides has not passed."""
    job, ref = _mini_job(tmp_path)
    for side in job.sides:
        side.add_training_image("surface", ref, "ok")
        side.train_tool("surface")
    job.load()
    r = job.inspect({"side0": ref})
    assert r["verdict"] == "INCOMPLETE"
    assert r["missing"] == ["side1"]


def test_relabelling_moves_an_image_between_ok_and_ng(tmp_path):
    job, ref = _mini_job(tmp_path, 1)
    side = job.sides[0]
    p = side.add_training_image("surface", ref, "ng")
    moved = side.relabel(p, "ok")
    labels = dict((os.path.basename(a), b) for a, b in side.training_images("surface"))
    assert labels[os.path.basename(moved)] == "ok"
    assert not os.path.exists(p)


def test_invisible_ng_is_named_not_allowed_to_collapse_the_threshold(tmp_path):
    """An NG image scoring like a good part used to drag the threshold down to
    one pixel above the best OK -- a pale scratch on a pale surface did that on
    a real chassis. It is now set aside and named for the operator to check."""
    job, ref = _mini_job(tmp_path, 1)
    side = job.sides[0]
    side.add_training_image("surface", ref, "ok")
    side.add_training_image("surface", _surface_part(dx=5, deg=0.5), "ng")   # not defective
    side.add_training_image("surface", _surface_part(seed=1, scratch=True), "ng")
    rep = side.train_tool("surface")
    assert len(rep["unseparated_ng_paths"]) == 1, rep
    assert rep["threshold"] > 5, "a non-defective NG collapsed the threshold to %.1f" % rep["threshold"]


# ------------------------------------------------------ inference backends
def _backends():
    try:
        from vision_station.core import inference
        return inference.available_backends()
    except Exception:
        return []


@pytest.mark.parametrize("backend", ["onnxruntime", "openvino"])
def test_yolo_decodes_identically_on_each_runtime(backend, tmp_path):
    """'auto' prefers OpenVINO when it is installed, so the rest of the suite
    runs on whichever runtime this machine happens to have. A line PC without
    OpenVINO would otherwise be exercising a path nothing tests. Both are pinned
    here and must agree."""
    if backend not in _backends():
        pytest.skip("%s not installed" % backend)
    from vision_station.core import inference
    from vision_station.tools import yolo as Y
    from make_test_onnx import build
    path = build(str(tmp_path / "m.onnx"),
                 [(160, 320, 48, 48, 0, .90), (260, 320, 48, 48, 0, .88),
                  (320, 200, 40, 40, 1, .79)], ["screw", "led"])
    Y._SESSIONS.clear()
    inference.clear_cache()
    dets, names = Y.infer(path, np.zeros((420, 640, 3), np.uint8), backend=backend)
    assert names == {0: "screw", 1: "led"}, \
        "class names were not read from the model file on %s" % backend
    assert len(dets) == 3
    assert sorted(round(d["x"]) for d in dets) == [136, 236, 300]


@pytest.mark.parametrize("ttype", ["yolo_detect", "yolo_defect"])
def test_both_yolo_tools_explain_a_missing_runtime(ttype, tmp_path, monkeypatch):
    """The two YOLO tools had separate copies of their set-up code, and only
    one explained a missing inference runtime; the other showed a bare error."""
    from vision_station.tools import yolo as Y
    from vision_station.core.schema import defaults
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")

    def no_runtime(*a, **k):
        raise ImportError("no module named openvino")
    monkeypatch.setattr(Y, "infer", no_runtime)
    img = np.zeros((60, 60, 3), np.uint8)
    r, _ = _run(ttype, dict(defaults(ttype), roi=[0, 0, 60, 60], model=str(model)), img)
    assert r.status == "error" and "pip install" in r.message, r.message


def test_text_lines_are_prepared_dark_on_light_whatever_their_polarity():
    """Both text detectors now share one preparation step. Only one of their
    old copies turned white-on-black print the right way round."""
    from vision_station.tools.text import line_for_recognition
    for bg, fg in ((30, 235), (235, 30)):
        g = np.full((40, 200), bg, np.uint8)
        cv2.putText(g, "SN 12345", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fg, 2)
        b = line_for_recognition(g, (0, 0, 200, 40), scale=2.0)
        assert b.mean() > 128, "print on a %d background was left inverted" % bg
    assert line_for_recognition(np.zeros((10, 10), np.uint8), (20, 20, 5, 5)) is None


def test_accuracy_precision_matches_the_reference_runtime():
    """OpenVINO's default silently drops to reduced precision. Measured on the
    text detector its logits then moved by up to 1.3, and on a real label one
    box grew a pixel. 'accuracy' must be numerically the same as onnxruntime."""
    if not {"onnxruntime", "openvino"} <= set(_backends()):
        pytest.skip("needs both runtimes")
    from vision_station.core import inference
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "models", "textnet.onnx")
    if not os.path.exists(path):
        pytest.skip("textnet.onnx not built")
    x = np.random.RandomState(0).rand(1, 1, 128, 128).astype(np.float32)
    inference.clear_cache()
    ref = inference.load(path, "onnxruntime")(x)[0]
    acc = inference.load(path, "openvino", "accuracy")(x)[0]
    assert float(np.abs(ref - acc).max()) < 1e-3, \
        "accuracy mode differs from the reference by %.4f" % float(np.abs(ref - acc).max())


@pytest.mark.parametrize("backend", ["onnxruntime", "openvino"])
def test_input_shape_is_reported_for_dynamic_models(backend):
    """OpenVINO raises on a concrete shape for a dynamic input, and this used to
    return None -- so the YOLO tool silently fell back to a 640 letterbox
    whatever the model expected."""
    if backend not in _backends():
        pytest.skip("%s not installed" % backend)
    from vision_station.core import inference
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "models", "textnet.onnx")
    if not os.path.exists(path):
        pytest.skip("textnet.onnx not built")
    inference.clear_cache()
    shape = inference.load(path, backend).input_shape()
    assert shape is not None and len(shape) == 4
    assert shape[1] == 1, "channel count lost: %s" % shape


def test_no_runtime_is_an_honest_error(monkeypatch):
    from vision_station.core import inference
    monkeypatch.setattr(inference, "available_backends", lambda: [])
    with pytest.raises(ImportError) as e:
        inference.choose_backend("auto")
    assert "pip install" in str(e.value)


# ------------------------------------------------------- one text reader
def _label(lines, w=620, h=220, scale=0.75):
    img = np.full((h, w, 3), 40, np.uint8)
    cv2.rectangle(img, (40, 40), (w - 40, h - 30), (235, 235, 235), -1)
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (60, 80 + i * 45), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (20, 20, 20), 2, cv2.LINE_AA)
    return img


@ocr_required
def test_read_text_locates_and_reads_every_line():
    """A label with three fields in one loose region: each line comes back
    with its own text, confidence and position. The old fixed-box reader
    returned ':' for a label like this."""
    from vision_station.core.schema import defaults
    lines = ["SN QPX3018045K", "MX40 27Q8L5DB SYS", "CODE QWMGX10CRB"]
    r, _ = _run("ocr", dict(defaults("ocr"), roi=[0, 0, 620, 220]), _label(lines))
    assert r.status == "pass", r.message
    assert r.values["blocks_found"] == 3, r.values["texts"]
    for want, got in zip(lines, r.values["texts"]):
        assert _cer(want.replace(" ", ""), got.replace(" ", "")) < 0.15, (want, got)
    for b in r.values["blocks"]:
        x, y, w, h = b["image_box"]
        assert 40 <= x < 300 and 40 <= y < 200 and w > 100, b


@ocr_required
@pytest.mark.parametrize("turn", [cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE,
                                  cv2.ROTATE_90_COUNTERCLOCKWISE])
def test_read_text_reads_a_label_whichever_way_up(turn):
    """A board loaded upside down, a label stuck on sideways. Nobody has to
    say which: 'MX40 27Q8L5DB' is read, not its mirror-image nonsense."""
    from vision_station.core.schema import defaults
    img = cv2.rotate(_label(["MX40 27Q8L5DB"], 520, 160, 0.9), turn)
    h, w = img.shape[:2]
    r, _ = _run("ocr", dict(defaults("ocr"), roi=[0, 0, w, h]), img)
    assert r.status == "pass", r.message
    assert _cer("MX4027Q8L5DB", r.values["text"].replace(" ", "")) < 0.15, r.values["text"]


@ocr_required
def test_read_text_must_contain_ignores_case_and_spacing():
    from vision_station.core.schema import defaults
    img = _label(["SN QPX3018045K", "CODE QWMGX10CRB"])
    base = dict(defaults("ocr"), roi=[0, 0, 620, 220])
    hit, _ = _run("ocr", dict(base, expect_contains="qpx 3018045k"), img)
    assert hit.status == "pass", hit.message
    miss, _ = _run("ocr", dict(base, expect_contains="ZZZ999"), img)
    assert miss.status == "fail" and "ZZZ999" in miss.message and "QPX" in miss.message


@ocr_required
def test_read_text_finds_small_print_on_a_large_image():
    """A whole 12-megapixel photograph handed to the reader: print 22 px tall
    in one corner. The region is tiled at full resolution, so it is the same
    22 px when the detector sees it -- shrunk to fit one pass it was 7."""
    from vision_station.core.schema import defaults
    img = np.full((3000, 4000, 3), 90, np.uint8)
    rng = np.random.RandomState(1)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    cv2.putText(img, "LOT 7741-A", (3350, 2890), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(img, "REV C", (1262, 1290), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (235, 235, 235), 2, cv2.LINE_AA)      # across a tile seam
    r, _ = _run("ocr", dict(defaults("ocr"), roi=[0, 0, 4000, 3000]), img)
    joined = " ".join(r.values["texts"]).replace(" ", "")
    assert "7741-A" in joined, r.values["texts"]
    assert "REVC" in joined, "print lying across a tile seam was lost: %s" % r.values["texts"]


@ocr_required
def test_verify_text_matches_one_line_among_several():
    """The serial number is one line of a label. Verify text used to join
    every line into one string and test that, so a pattern for the serial
    could never match once the region held anything else."""
    from vision_station.core.schema import defaults
    img = _label(["MODEL MX40", "QPX3018045K", "MADE IN INDIA"])
    base = dict(defaults("ocv"), roi=[0, 0, 620, 220])
    ok, _ = _run("ocv", dict(base, expect_regex="QPX[0-9]{7}K"), img)
    assert ok.status == "pass" and ok.values["text"] == "QPX3018045K", (ok.message, ok.values)
    exact, _ = _run("ocv", dict(base, expect_text="MADE IN INDIA"), img)
    assert exact.status == "pass", exact.message
    bad, _ = _run("ocv", dict(base, expect_regex="QPX[0-9]{9}"), img)
    assert bad.status == "fail"
    broken, _ = _run("ocv", dict(base, expect_regex="QPX[0-9"), img)
    assert broken.status == "error" and "pattern" in broken.message


@ocr_required
def test_a_line_read_without_confidence_is_reported_not_hidden():
    from vision_station.core.schema import defaults
    img = _label(["SN QPX3018045K"])
    r, _ = _run("ocr", dict(defaults("ocr"), roi=[0, 0, 620, 220], min_confidence=1.0), img)
    assert r.status == "fail"
    assert "confidence" in r.message and "QPX" in r.message, r.message
    assert r.score > 0.5, "the measured confidence was thrown away with the line"


@ocr_required
def test_short_lines_can_be_left_out():
    from vision_station.core.schema import defaults
    img = _label(["AB", "LONGENOUGHSTRING"])
    r, _ = _run("ocr", dict(defaults("ocr"), roi=[0, 0, 620, 220], min_chars=6), img)
    assert r.values["texts"] and all(len(t) >= 6 for t in r.values["texts"]), r.values["texts"]


@ocr_required
def test_capitals_only_charset_reads_small_letters_as_capitals():
    """'Allowed characters: A-Z' on a marking printed 'Orion' gave 'O': the
    small letters were deleted. They are that marking's capitals."""
    from vision_station.tools.text import _fit_charset
    assert _fit_charset("Orion 2021", "ABCDEFGHIJKLMNOPQRSTUVWXYZ") == "ORION"
    assert _fit_charset("Orion 2021", "") == "Orion 2021"
    assert _fit_charset("AB-12", "abcdefghijklmnopqrstuvwxyz0123456789") == "ab12"


@ocr_required
@pytest.mark.parametrize("old_type", ["ocr_auto", "text_detect"])
def test_recipes_with_the_removed_text_finders_still_run(old_type):
    """'Find and read all text' and 'AI text detection' were folded into Read
    text. A recipe saved with either -- and with the settings they had, such
    as a six-character minimum that hid 'ORION' -- loads and reads."""
    from vision_station.core import schema
    old = {"roi": [0, 0, 620, 220], "expect_contains": "", "upscale": 2.0,
           "min_text_height": 10, "min_text_width": 24, "try_quarter_turns": True,
           "turn": "180", "orientation": "auto", "min_chars": 6, "accept_score": 5.0,
           "min_confidence": 0.0, "model": "", "threshold": 0.5, "long_side": 512}
    r, _ = _run(old_type, old, _label(["ORION", "SN QPX3018045K"]))
    assert r.status == "pass", r.message
    assert "ORION" in r.values["texts"], r.values["texts"]
    assert r.values["blocks_found"] == 2 and "image_box" in r.values["blocks"][0]
    offered = {t for _g, types in schema.GROUPS for t in types} | \
        {t for _q, _c, types in schema.QUESTIONS for t in types}
    assert old_type not in offered, "%s is still offered for new checks" % old_type
    assert schema.LEGACY_TYPES[old_type] == "ocr"


def test_no_duplicate_checks_are_offered():
    """Each removed check duplicated another. None may be offered again, and
    every check that is offered must exist."""
    from vision_station.core import schema
    from vision_station.tools import REGISTRY
    offered = [t for _g, types in schema.GROUPS for t in types]
    assert len(offered) == len(set(offered)), "a check is listed twice"
    assert not set(offered) & set(schema.LEGACY_TYPES)
    assert set(offered) <= set(REGISTRY)
    asked = {t for _q, _c, types in schema.QUESTIONS for t in types}
    assert asked == set(offered), "the simple picker and the full menu disagree: %s" % (
        sorted(asked ^ set(offered)))
    for old, new in schema.LEGACY_TYPES.items():
        assert old in REGISTRY and new in offered


def test_text_reader_merges_one_line_seen_in_two_tiles():
    from vision_station.tools.ocr_engine import _merge_across_tiles, _tiles
    box = lambda x0, y0, x1, y1: np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    # one line cut by a seam, the same short line seen twice, and a stranger
    items = [(box(900, 100, 1270, 130), 0.9, 0), (box(1100, 101, 1500, 131), 0.8, 1),
             (box(1150, 400, 1250, 425), 0.9, 0), (box(1151, 400, 1251, 426), 0.9, 1),
             (box(200, 800, 400, 830), 0.9, 0)]
    out = _merge_across_tiles(items)
    assert len(out) == 3, [b.tolist() for b, _ in out]
    widths = sorted(float(b[:, 0].max() - b[:, 0].min()) for b, _ in out)
    assert abs(widths[-1] - 600) < 4, "the two halves of the long line were not joined"
    # two lines one above the other in the same tile stay two
    two = _merge_across_tiles([(box(0, 0, 200, 30), 0.9, 0), (box(0, 20, 200, 50), 0.9, 0)])
    assert len(two) == 2
    tiles = _tiles(3000, 4000)
    assert max(x + w for x, y, w, h in tiles) == 4000 and max(y + h for x, y, w, h in tiles) == 3000


def test_read_code_decodes_a_small_low_contrast_qr():
    """A code a third of the usual size, grey on grey. Read as taken it does
    not decode; the reader now tries it enlarged and with the contrast
    stretched before giving up."""
    from vision_station.core.schema import defaults
    try:
        enc = cv2.QRCodeEncoder_create()
        qr = enc.encode("VS-LOT-7741A")
    except Exception:
        pytest.skip("this OpenCV build cannot make a QR code")
    qr = cv2.resize(qr, (58, 58), interpolation=cv2.INTER_NEAREST)
    img = np.full((160, 200), 150, np.uint8)
    img[50:108, 70:128] = np.where(qr > 128, 150, 105).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    r, _ = _run("code_read", dict(defaults("code_read"), roi=[0, 0, 200, 160]),
                cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    assert r.status == "pass" and r.values["data"] == "VS-LOT-7741A", (r.message, r.values)
    wrong, _ = _run("code_read", dict(defaults("code_read"), roi=[0, 0, 200, 160],
                                      expect_regex="XX.*"),
                    cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    assert wrong.status == "fail" and "VS-LOT-7741A" in wrong.message


# --------------------------------------------------------------- captures
def test_single_frame_recipes_are_unaffected_by_captures(tmpproj):
    """Multi-capture must not disturb anything that already works. A model
    defining no captures behaves exactly as before."""
    root, pid, board = tmpproj
    proj = ProjectStore(root).get(pid)
    assert [c["id"] for c in proj.captures()] == ["default"]
    plain = proj.inspect(board)
    wrapped = proj.inspect({"default": board})
    assert plain["verdict"] == wrapped["verdict"]
    assert plain["captures_used"] == ["default"]
    assert plain["image_size"] == wrapped["image_size"]


def test_tools_read_the_capture_they_ask_for(tmpproj):
    """The point of the feature: two exposures, and a tool sees the one it
    named. Verified by making the captures visibly different and checking the
    tool's own numbers change with them."""
    root, pid, board = tmpproj
    proj = ProjectStore(root).get(pid)
    dark = (board * 0.25).astype(np.uint8)

    from vision_station.core.schema import defaults
    proj.spec["captures"] = [{"id": "panel", "exposure_ms": 10.0},
                             {"id": "leds", "exposure_ms": 0.8}]
    proj.spec["tools"] = [
        {"id": "on_panel", "type": "count_blob",
         "params": dict(defaults("count_blob"), roi=[30, 30, 300, 200],
                        capture="panel", threshold="128", expect_count=0)},
        {"id": "on_leds", "type": "count_blob",
         "params": dict(defaults("count_blob"), roi=[30, 30, 300, 200],
                        capture="leds", threshold="128", expect_count=0)},
    ]
    proj.tools = [make_tool(t) for t in proj.spec["tools"]]
    res = proj.inspect({"panel": board, "leds": dark})

    got = {t["tool_id"]: t["values"]["count"] for t in res["tools"]}
    assert res["captures_used"] == ["leds", "panel"]
    assert got["on_panel"] != got["on_leds"], \
        "both tools returned %s - they read the same frame" % got


def test_a_missing_capture_falls_back_and_says_so(tmpproj):
    """A recipe built on a two-camera station must still run when someone opens
    a single image from disk. Falling back silently would be worse: the tool
    would use the wrong exposure and report a plausible, wrong result."""
    root, pid, board = tmpproj
    proj = ProjectStore(root).get(pid)
    from vision_station.core.schema import defaults
    proj.spec["captures"] = [{"id": "panel"}, {"id": "leds", "exposure_ms": 1.0}]
    proj.spec["tools"] = [{"id": "t", "type": "count_blob",
                           "params": dict(defaults("count_blob"),
                                          roi=[30, 30, 200, 150], capture="leds",
                                          expect_count=0)}]
    proj.tools = [make_tool(t) for t in proj.spec["tools"]]
    res = proj.inspect(board)
    assert res["tools"][0]["status"] != "error", "the tool refused to run at all"
    assert res["captures_missing"] == ["leds"], \
        "a missing capture was not reported: %s" % res["captures_missing"]


def test_camera_grabs_one_frame_per_capture_at_its_own_exposure():
    """Each capture must actually get its own exposure. A camera in free-running
    acquisition has frames already in flight from the previous setting, and
    using one of those silently gives a capture the wrong exposure -- which
    surfaces later as an intermittent false reject, not an error."""
    scene = np.full((80, 160), 30, np.float32)
    scene[30:50, 60:100] = 1500.0
    cam_mod = _sim_camera(scene)
    t = cam_mod.CameraThread(0, cam_mod.CameraSettings())
    t.start()
    for _ in range(60):
        if t.get_frame() is not None:
            break
        time.sleep(0.05)
    frames = t.grab_sequence([{"id": "panel", "exposure_ms": 10.0},
                              {"id": "leds", "exposure_ms": 0.5}], settle=3)
    t.stop()
    assert set(frames) == {"panel", "leds"}
    panel = frames["panel"].max(axis=2)
    leds = frames["leds"].max(axis=2)
    assert (panel >= 254).mean() > 0.01, "the long exposure did not clip anything"
    assert (leds >= 254).mean() == 0.0, "the short exposure still clips"
    assert leds.mean() < panel.mean(), "the two captures are indistinguishable"


def test_capture_skips_frames_still_in_flight_from_the_old_exposure():
    """Measured on the real acA5472-5gc (5 fps): after an exposure change the
    camera still delivers two frames taken at the old setting. grab_sequence
    used to count how often it *polled* the latest frame, so at 5 fps both
    captures of a pair came back as the frame from before the change. It must
    count frames that are actually new since the setting was applied."""
    import importlib
    import fake_pylon as F
    scene = np.full((40, 60), 100, np.float32)
    F.install([F.Device("usb3", scene=scene, ref_exposure_us=10000.0,
                        frame_period_s=0.04, pipeline_depth=2)])
    for m in [k for k in sys.modules if k.startswith("camera")]:
        del sys.modules[m]
    from vision_station.hardware import camera as cam_mod
    importlib.reload(cam_mod)
    s = cam_mod.CameraSettings()
    s.exposure_ms = 10.0                      # mean 100 at steady state
    t = cam_mod.CameraThread(0, s)
    t.start()
    try:
        for _ in range(100):
            if t.get_frame() is not None:
                break
            time.sleep(0.02)
        time.sleep(0.3)
        frames = t.grab_sequence([{"id": "short", "exposure_ms": 2.0},
                                  {"id": "long", "exposure_ms": 20.0}], settle=3)
    finally:
        t.stop()
    assert set(frames) == {"short", "long"}
    assert abs(frames["short"].mean() - 20) < 2, frames["short"].mean()
    assert abs(frames["long"].mean() - 200) < 2, frames["long"].mean()


def test_two_captures_are_what_make_a_mixed_scene_readable():
    """End to end on a scene that genuinely cannot be done in one frame: dim
    printed text plus LEDs six times over the sensor range."""
    from vision_station.core import exposure
    # drawn on 8 bits and lifted to float: OpenCV 5 draws text on 8-bit images only
    text = np.full((240, 480, 3), 18, np.uint8)
    cv2.putText(text, "REV C", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (90, 90, 90), 3)
    scene = text.astype(np.float32)
    for i, bgr in enumerate([(40, 40, 255), (60, 220, 60), (255, 80, 50)]):
        x = 90 + i * 110
        scene[140:170, x:x + 50] = np.array(bgr, np.float32) * 6.0
    cam_mod = _sim_camera(scene)
    t = cam_mod.CameraThread(0, cam_mod.CameraSettings())
    t.start()
    for _ in range(60):
        if t.get_frame() is not None:
            break
        time.sleep(0.05)
    frames = t.grab_sequence([{"id": "panel", "exposure_ms": 12.0},
                              {"id": "leds", "exposure_ms": 0.8}], settle=3)
    t.stop()
    led_roi = [60, 130, 340, 50]

    # the long exposure cannot support colour, the short one can
    assert not exposure.advise(frames["panel"], led_roi, "colour")["ok"]
    assert exposure.advise(frames["leds"], led_roi, "colour")["ok"]

    from vision_station.core.schema import defaults
    ctx = Context(reference=frames["leds"])
    cb = dict(defaults("count_blob"), roi=led_roi, space="hsv_v",
              threshold="120", min_area=200, max_area=20000, expect_count=3,
              row_tol=40, open_px=3)
    make_tool({"id": "b", "type": "count_blob", "params": cb}).run(frames["leds"], ctx)
    cc = dict(defaults("color_classify"), roi=led_roi, source="b",
              expect=["red", "green", "blue"], deltaE_max=60)
    r = make_tool({"id": "c", "type": "color_classify",
                   "params": cc}).run(frames["leds"], ctx)
    assert r.status == "pass", r.message
    assert [l["color"] for l in r.values["leds"]] == ["red", "green", "blue"]

    # and the panel text is only legible on the long exposure
    assert frames["panel"][40:70, 40:200].mean() > \
        frames["leds"][40:70, 40:200].mean() * 3, \
        "the long capture is not meaningfully brighter on the printed area"


# --------------------------------------------------------------- exposure
def _sim_camera(scene, ref_us=5000.0):
    """A simulated Basler whose frames respond to the exposure setting."""
    import importlib
    import fake_pylon as F
    F.install([F.Device("usb3", scene=scene, ref_exposure_us=ref_us)])
    for m in [k for k in sys.modules if k.startswith("camera")]:
        del sys.modules[m]
    from vision_station.hardware import camera as cam
    importlib.reload(cam)
    return cam


def test_clipping_is_measured_per_region_not_per_frame():
    """A frame can be almost entirely unclipped while the only region that
    matters is destroyed. The real faceplate photograph was 0.02% clipped
    overall and 10.6% clipped across the LED band, which is why the whole-image
    number was no use."""
    from vision_station.core import exposure
    img = np.full((200, 400, 3), 20, np.uint8)
    img[90:110, 180:220] = 255
    whole = exposure.clipping(img)
    region = exposure.clipping(img, [175, 85, 50, 30])
    assert whole["clipped_pct"] < 2.0
    assert region["clipped_pct"] > 40.0, \
        "region clipping %.1f%% did not stand out" % region["clipped_pct"]


def test_exposure_advice_rejects_a_clipped_region_for_colour():
    from vision_station.core import exposure
    img = np.full((120, 200, 3), 20, np.uint8)
    img[40:80, 60:140] = 255
    bad = exposure.advise(img, [55, 35, 90, 50], purpose="colour")
    assert not bad["ok"]
    assert any("clip" in m.lower() for _k, m in bad["findings"])
    # the same frame is fine for deciding merely whether something is lit
    lit = exposure.advise(img, [55, 35, 90, 50], purpose="lit")
    assert lit["ok"]


def test_exposure_search_finds_the_brightest_unclipped_setting():
    """Bisection on exposure, which is monotonic in clipping. The result must
    be usable and not merely dark -- an over-short exposure throws away signal
    and makes everything noisy."""
    from vision_station.core import exposure
    scene = np.full((120, 300), 25, np.float32)
    scene[50:70, 40:260] = 1400.0            # far beyond 8-bit range
    cam_mod = _sim_camera(scene)
    c = cam_mod.PylonCamera(0)
    c.open()
    c.start()
    s = cam_mod.CameraSettings()
    s.auto_exposure = False
    s.auto_gain = False

    def set_exp(ms):
        s.exposure_ms = ms
        c.apply_settings(s)

    r = exposure.find_exposure(c.grab, set_exp, roi=[30, 40, 240, 40], settle=1)
    assert r["ok"], r.get("reason")
    assert r["clipped_pct"] <= 0.05
    assert r["peak"] >= 200, \
        "settled on an unnecessarily dark exposure (peak %s)" % r["peak"]
    assert len(r["history"]) < 20, "search did not converge promptly"
    c.close()


def test_exposure_search_reports_when_even_the_shortest_clips():
    """A scene too bright for the sensor cannot be fixed with exposure alone.
    Saying so is more use than returning a number that will not work."""
    from vision_station.core import exposure
    scene = np.full((60, 80), 900000.0, np.float32)
    cam_mod = _sim_camera(scene)
    c = cam_mod.PylonCamera(0)
    c.open()
    c.start()
    s = cam_mod.CameraSettings()
    s.auto_exposure = False

    def set_exp(ms):
        s.exposure_ms = ms
        c.apply_settings(s)

    r = exposure.find_exposure(c.grab, set_exp, settle=1)
    assert not r["ok"]
    assert "gain" in r["reason"].lower() or "aperture" in r["reason"].lower()
    c.close()


def test_correct_exposure_is_what_makes_colour_work():
    """The whole point, end to end: the same LEDs are unreadable over-exposed
    and correct once the search has run."""
    from vision_station.core import exposure
    cols = {"red": (40, 40, 255), "green": (60, 220, 60),
            "blue": (255, 80, 50), "amber": (30, 170, 255)}
    scene = np.full((120, 300, 3), 22, np.float32)
    for i, bgr in enumerate(cols.values()):
        x = 30 + i * 65
        scene[50:74, x:x + 34] = np.array(bgr, np.float32) * 5.0
    cam_mod = _sim_camera(scene)
    c = cam_mod.PylonCamera(0)
    c.open()
    c.start()
    s = cam_mod.CameraSettings()
    s.auto_exposure = False
    s.auto_gain = False
    roi = [20, 40, 270, 45]

    def set_exp(ms):
        s.exposure_ms = ms
        c.apply_settings(s)

    def read():
        c.grab()
        frame = c.grab()
        from vision_station.core.schema import defaults
        ctx = Context(reference=frame)
        cb = dict(defaults("count_blob"), roi=roi, space="hsv_v",
                  threshold="120", min_area=100, max_area=20000,
                  expect_count=0, row_tol=60, open_px=3)
        make_tool({"id": "b", "type": "count_blob", "params": cb}).run(frame, ctx)
        cc = dict(defaults("color_classify"), roi=roi, source="b", expect=[],
                  deltaE_max=200)
        r = make_tool({"id": "c", "type": "color_classify",
                       "params": cc}).run(frame, ctx)
        return [l["color"] for l in r.values["leds"]]

    set_exp(10.0)
    over = read()
    assert over != list(cols), "the over-exposed case was supposed to fail"

    found = exposure.find_exposure(c.grab, set_exp, roi=roi, settle=1)
    set_exp(found["exposure_ms"])
    assert read() == list(cols), "colour still wrong at the chosen exposure"
    c.close()


# ------------------------------------------------- real-image regression
REAL_BAND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "tests", "testdata", "real_led_band.png")
real_image = pytest.mark.skipif(not os.path.exists(REAL_BAND),
                                reason="real image fixture not present")


@real_image
def test_real_faceplate_is_diagnosed_as_over_exposed():
    """A crop of an actual switch faceplate. Synthetic images are built to be
    solvable; this one was taken in a lab with no thought for machine vision,
    and it is the only test here that can fail for reasons nobody designed in."""
    from vision_station.core import exposure
    img = cv2.imread(REAL_BAND)
    assert img is not None
    c = exposure.clipping(img)
    assert c["clipped_pct"] > 2.0, \
        "the fixture is supposed to be badly clipped, measured %.2f%%" % c["clipped_pct"]
    adv = exposure.advise(img, purpose="colour")
    assert not adv["ok"], "a clipped faceplate was passed as fit for colour work"
    assert any("exposure" in m.lower() for _k, m in adv["findings"]), \
        "the advice does not mention exposure"


@real_image
def test_real_faceplate_colour_is_not_all_white():
    """Before the halo fix this returned 'white' for every LED, greens and reds
    included. It is still not fully correct on an image this poor -- greens read
    as yellow because the halo blends with a neighbour's bloom -- so the
    assertion is the one that matters: the clipped core is no longer what gets
    sampled."""
    from vision_station.core.schema import defaults
    img = cv2.imread(REAL_BAND)
    h, w = img.shape[:2]
    roi = [0, 0, w, h]
    ctx = Context(reference=img)
    cb = dict(defaults("count_blob"), roi=roi, space="hsv_v", threshold="240",
              min_area=60, max_area=20000, expect_count=0, row_tol=40, open_px=3)
    make_tool({"id": "blobs", "type": "count_blob", "params": cb}).run(img, ctx)
    blobs = ctx.share.get("blobs") or []
    assert len(blobs) >= 5, "found only %d bright regions on the faceplate" % len(blobs)

    cc = dict(defaults("color_classify"), roi=roi, source="blobs", expect=[],
              deltaE_max=200)
    r = make_tool({"id": "col", "type": "color_classify", "params": cc}).run(img, ctx)
    names = [l["color"] for l in r.values["leds"]]
    assert names.count("white") < len(names) * 0.6, \
        "most LEDs still read as white - the clipped core is being sampled: %s" % names


# ---------------------------------------------------------------- anomaly
@onnx_required
def test_anomaly_tool_runs_and_thresholds(tmp_path):
    """The anomaly tool shipped without ever having been executed. A fake
    heatmap model exercises everything that can be wrong -- preprocessing, the
    threshold comparison, blob extraction and the region offset -- without
    needing a trained network."""
    from make_test_onnx import build_anomaly
    from vision_station.core.schema import defaults
    hot = np.zeros((64, 64), np.float32)
    hot[20:30, 35:45] = 0.9
    m_hot = build_anomaly(str(tmp_path / "hot.onnx"), hot)
    m_cool = build_anomaly(str(tmp_path / "cool.onnx"), np.zeros((64, 64), np.float32))
    img = np.full((200, 200, 3), 120, np.uint8)
    base = dict(defaults("anomaly"), roi=[0, 0, 200, 200], threshold=0.5,
                input_size=256, min_area=5)

    bad, _ = _run("anomaly", dict(base, model=m_hot), img)
    assert bad.status == "fail" and bad.score > 0.5
    assert any(g.label == "anomaly" for g in bad.graphics), \
        "no defect region drawn from the heatmap"

    good, _ = _run("anomaly", dict(base, model=m_cool), img)
    assert good.status == "pass", "a clean heatmap was reported as a defect"


@onnx_required
def test_anomaly_sessions_are_keyed_by_model_file(tmp_path):
    """Regression guard for a real defect. The session was cached in a single
    unkeyed class attribute, so the first model loaded was reused for every
    anomaly tool in the recipe -- a second tool pointed at a different model
    silently returned the first model's heatmap. On a two-surface inspection
    that means one surface is never checked and nothing says so."""
    from make_test_onnx import build_anomaly
    from vision_station.core.schema import defaults
    hot = np.zeros((64, 64), np.float32)
    hot[10:20, 10:20] = 0.95
    a = build_anomaly(str(tmp_path / "a.onnx"), hot)
    b = build_anomaly(str(tmp_path / "b.onnx"), np.zeros((64, 64), np.float32))
    img = np.full((160, 160, 3), 100, np.uint8)
    base = dict(defaults("anomaly"), roi=[0, 0, 160, 160], threshold=0.5,
                input_size=256, min_area=5)

    first, _ = _run("anomaly", dict(base, model=a), img)
    second, _ = _run("anomaly", dict(base, model=b), img)
    third, _ = _run("anomaly", dict(base, model=a), img)
    assert first.status == "fail"
    assert second.status == "pass", \
        "the second model returned the first model's result (score %.3f)" % second.score
    assert third.status == "fail", "caching broke after loading a second model"


@onnx_required
def test_anomaly_missing_model_is_an_honest_error():
    from vision_station.core.schema import defaults
    img = np.zeros((80, 80, 3), np.uint8)
    r, _ = _run("anomaly", dict(defaults("anomaly"), roi=[0, 0, 80, 80],
                                model="/nonexistent/model.onnx"), img)
    assert r.status == "error" and "not found" in (r.message or "").lower()


# ------------------------------------------------------------------- YOLO
@pytest.fixture
def fake_model(tmp_path):
    """A real ONNX file in YOLOv8 layout returning fixed detections.

    The YOLO tools sat at 18% coverage because testing them looked like it
    needed a trained network. It does not: the parts that can plausibly be
    wrong are metadata parsing, letterbox un-padding, NMS and the count logic,
    and none of those care whether the weights are any good."""
    from make_test_onnx import build
    # four screws and one led, in 640x640 letterboxed input space
    boxes = [(160, 320, 48, 48, 0, 0.90), (260, 320, 48, 48, 0, 0.88),
             (360, 320, 48, 48, 0, 0.86), (460, 320, 48, 48, 0, 0.84),
             (320, 200, 40, 40, 1, 0.79)]
    return build(str(tmp_path / "m.onnx"), boxes, ["screw", "led"])


@onnx_required
def test_yolo_decodes_through_real_onnxruntime(fake_model):
    from vision_station.tools import yolo as Y
    img = np.zeros((420, 640, 3), np.uint8)
    dets, names = Y.infer(fake_model, img, conf=0.3)
    assert names == {0: "screw", 1: "led"}, "class names not read from metadata"
    assert len(dets) == 5
    # letterbox of a 640x420 image into 640x640 pads 110 px top and bottom;
    # if that is not undone every box is a hundred pixels low.
    screw = [d for d in dets if d["label"] == "screw"][0]
    assert abs(screw["y"] - (320 - 24 - 110)) <= 1, \
        "letterbox padding not removed: y=%s" % screw["y"]
    assert abs(screw["w"] - 48) <= 1


@onnx_required
def test_yolo_detect_counts_per_class(fake_model):
    img = np.zeros((420, 640, 3), np.uint8)
    from vision_station.core.schema import defaults
    p = dict(defaults("yolo_detect"), roi=[0, 0, 640, 420], model=fake_model,
             confidence=0.3, expect_counts={"screw": 4, "led": 1})
    r, _ = _run("yolo_detect", p, img)
    assert r.status == "pass", r.message
    assert r.values["counts"] == {"screw": 4, "led": 1}

    wrong = dict(p, expect_counts={"screw": 5, "led": 1})
    r2, _ = _run("yolo_detect", wrong, img)
    assert r2.status == "fail", "a wrong expected count was accepted"
    assert "screw" in r2.message


@onnx_required
def test_yolo_confidence_threshold_filters(fake_model):
    img = np.zeros((420, 640, 3), np.uint8)
    from vision_station.core.schema import defaults
    base = dict(defaults("yolo_detect"), roi=[0, 0, 640, 420], model=fake_model,
                expect_counts={}, expect_total=0)
    low, _ = _run("yolo_detect", dict(base, confidence=0.3), img)
    high, _ = _run("yolo_detect", dict(base, confidence=0.87), img)
    assert low.values["total"] == 5
    assert high.values["total"] == 2, \
        "confidence threshold not applied: %s" % high.values["total"]


@onnx_required
def test_yolo_only_classes_filter(fake_model):
    img = np.zeros((420, 640, 3), np.uint8)
    from vision_station.core.schema import defaults
    p = dict(defaults("yolo_detect"), roi=[0, 0, 640, 420], model=fake_model,
             confidence=0.3, expect_counts={}, expect_total=0,
             only_classes=["led"])
    r, _ = _run("yolo_detect", p, img)
    assert r.values["total"] == 1
    assert r.values["counts"] == {"led": 1}


@onnx_required
def test_yolo_defect_rejects_on_detection(fake_model):
    img = np.zeros((420, 640, 3), np.uint8)
    from vision_station.core.schema import defaults
    p = dict(defaults("yolo_defect"), roi=[0, 0, 640, 420], model=fake_model,
             confidence=0.3, max_defects=0)
    r, _ = _run("yolo_defect", p, img)
    assert r.status == "fail", "defect model found 5 things and still passed"
    assert r.values["defect_count"] == 5
    tolerant, _ = _run("yolo_defect", dict(p, max_defects=10), img)
    assert tolerant.status == "pass", "max_defects allowance ignored"


@onnx_required
def test_yolo_missing_model_is_an_honest_error(tmp_path):
    from vision_station.core.schema import defaults
    img = np.zeros((100, 100, 3), np.uint8)
    p = dict(defaults("yolo_detect"), roi=[0, 0, 100, 100],
             model=str(tmp_path / "nope.onnx"))
    r, _ = _run("yolo_detect", p, img)
    assert r.status == "error"
    assert "no trained model yet" in (r.message or "").lower(), \
        "a missing model should say so plainly, got: %s" % r.message


@onnx_required
def test_yolo_roi_offsets_are_applied(fake_model):
    """Detections are found inside the cropped region but reported in whole-
    image coordinates. Forgetting the offset puts every overlay in the corner."""
    img = np.zeros((420, 640, 3), np.uint8)
    from vision_station.core.schema import defaults
    full = dict(defaults("yolo_detect"), roi=[0, 0, 640, 420], model=fake_model,
                confidence=0.3, expect_counts={}, expect_total=0)
    a, _ = _run("yolo_detect", full, img)
    sub = dict(full, roi=[100, 60, 400, 300])
    b, _ = _run("yolo_detect", sub, img)
    assert b.values["detections"], "no detections in the sub-region"
    assert min(d["x"] for d in b.values["detections"]) >= 100, \
        "region offset not added to detection coordinates"


# ----------------------------------------------------------------- camera
@pytest.fixture
def pylon_env():
    """Install a fake pypylon and give back a freshly imported camera module.

    The Basler code has never run against a camera and cannot here, yet it is
    the next thing that gets plugged in. What can realistically be wrong is not
    the grab loop but the GenICam node names, which differ between camera
    families -- and getting those wrong fails silently: the camera opens, grabs
    frames, and ignores every setting."""
    import importlib
    import fake_pylon as F
    F.install([F.Device("usb3"), F.Device("gige_classic"),
               F.Device("minimal"), F.Device("readonly")])
    for m in [k for k in sys.modules if k.startswith("camera")]:
        del sys.modules[m]
    from vision_station.hardware import camera as cam
    importlib.reload(cam)
    yield cam, F
    F.uninstall()
    for m in [k for k in sys.modules if k.startswith("camera")]:
        del sys.modules[m]


def test_camera_enumerates_devices(pylon_env):
    cam, _F = pylon_env
    assert cam.PYLON_AVAILABLE
    devs = cam.list_devices()
    assert len(devs) == 4
    assert devs[0]["serial"] == "USB0001"
    assert all({"index", "vendor", "model", "serial"} <= set(d) for d in devs)


@pytest.mark.parametrize("idx,exp_node,gain_node,unit", [
    (0, "ExposureTime", "Gain", "dB"),          # USB3 / ace2 / dart
    (1, "ExposureTimeAbs", "GainRaw", "raw"),   # older GigE "classic" ace
])
def test_exposure_and_gain_reach_both_camera_families(pylon_env, idx, exp_node,
                                                      gain_node, unit):
    """Milliseconds must arrive as microseconds, on whichever node this camera
    family happens to expose."""
    cam, _F = pylon_env
    c = cam.PylonCamera(idx)
    c.open()
    s = cam.CameraSettings()
    s.auto_exposure = False
    s.auto_gain = False
    s.exposure_ms = 12.5
    s.gain_db = 4.0
    c.apply_settings(s)
    nodes = c.cam.nodes
    assert nodes[exp_node].sets == [12500.0], \
        "%s got %s, expected 12500 us" % (exp_node, nodes[exp_node].sets)
    assert nodes[gain_node].sets, "%s never written" % gain_node
    assert c.gain_unit()[0] == unit
    c.close()


def test_gain_raw_receives_an_integer(pylon_env):
    """GainRaw is an integer node in sensor units, not decibels. Writing a
    float can be rejected outright, and treating the number as dB understates
    the real gain by roughly two orders of magnitude."""
    cam, _F = pylon_env
    c = cam.PylonCamera(1)          # classic GigE
    c.open()
    s = cam.CameraSettings()
    s.auto_gain = False
    s.auto_exposure = False
    s.gain_db = 3.4
    c.apply_settings(s)
    written = c.cam.nodes["GainRaw"].sets[-1]
    assert isinstance(written, int), "GainRaw got a %s" % type(written).__name__
    unit, lo, hi = c.gain_unit()
    assert unit == "raw" and hi == 512
    c.close()


@pytest.mark.parametrize("idx", [0, 1])
def test_settings_are_clamped_to_the_camera_range(pylon_env, idx):
    """Both exposure and gain, on both families. A GenICam node rejects an
    out-of-range value outright, so an unclamped write does not merely saturate
    -- it throws, and the setting silently never applies."""
    cam, _F = pylon_env
    c = cam.PylonCamera(idx)
    c.open()
    exp_name = "ExposureTime" if idx == 0 else "ExposureTimeAbs"
    gain_name = "Gain" if idx == 0 else "GainRaw"

    for exposure, gain in ((99999999.0, 1e6), (-500.0, -50.0)):
        s = cam.CameraSettings()
        s.auto_exposure = False
        s.auto_gain = False
        s.exposure_ms = exposure
        s.gain_db = gain
        c.apply_settings(s)
        e_node, g_node = c.cam.nodes[exp_name], c.cam.nodes[gain_name]
        assert e_node.GetMin() <= e_node.sets[-1] <= e_node.GetMax(), \
            "exposure %s not clamped: wrote %s" % (exposure, e_node.sets[-1])
        assert g_node.GetMin() <= g_node.sets[-1] <= g_node.GetMax(), \
            "gain %s not clamped: wrote %s" % (gain, g_node.sets[-1])
    c.close()


def test_auto_modes_switch_the_right_nodes(pylon_env):
    cam, _F = pylon_env
    c = cam.PylonCamera(0)
    c.open()
    s = cam.CameraSettings()
    s.auto_exposure = True
    s.auto_gain = True
    s.auto_white_balance = True
    c.apply_settings(s)
    assert c.cam.nodes["ExposureAuto"].sets[-1] == "Continuous"
    assert c.cam.nodes["GainAuto"].sets[-1] == "Continuous"
    assert c.cam.nodes["BalanceWhiteAuto"].sets[-1] == "Continuous"
    # and with auto on, the manual node must be left alone
    assert not c.cam.nodes["ExposureTime"].sets
    c.close()


def test_camera_without_exposure_nodes_still_works(pylon_env):
    """A camera exposing neither node must keep grabbing. Losing the video
    because a setting could not be applied would be a much worse failure than
    the setting simply not taking effect."""
    cam, _F = pylon_env
    c = cam.PylonCamera(2)          # 'minimal'
    c.open()
    c.apply_settings(cam.CameraSettings())   # must not raise
    c.start()
    frame = c.grab()
    assert frame is not None and frame.ndim == 3
    c.close()


def test_readonly_nodes_do_not_break_acquisition(pylon_env):
    cam, _F = pylon_env
    c = cam.PylonCamera(3)          # 'readonly'
    c.open()
    c.apply_settings(cam.CameraSettings())
    c.start()
    assert c.grab() is not None
    c.close()


def test_mono_frames_are_converted_to_three_channels(pylon_env):
    """Every tool downstream indexes colour channels. A mono camera handing
    back a 2-D array would break all of them at once."""
    cam, _F = pylon_env
    c = cam.PylonCamera(0)
    c.open()
    c.start()
    f = c.grab()
    assert f.ndim == 3 and f.shape[2] == 3
    c.close()


def test_failed_grabs_return_none_rather_than_raising(pylon_env):
    cam, F = pylon_env
    F.install([F.Device("usb3", fail_every=1)])
    import importlib
    importlib.reload(cam)
    c = cam.PylonCamera(0)
    c.open()
    c.start()
    assert c.grab() is None, "a failed grab should be None, not an exception"
    c.close()


def test_disconnect_raises_cameraerror_not_a_bare_exception(pylon_env):
    cam, F = pylon_env
    F.install([F.Device("usb3", raise_on_grab=True)])
    import importlib
    importlib.reload(cam)
    c = cam.PylonCamera(0)
    c.open()
    c.start()
    with pytest.raises(cam.CameraError):
        c.grab()
    c.close()


def test_missing_camera_index_is_a_clear_error(pylon_env):
    cam, _F = pylon_env
    with pytest.raises(cam.CameraError) as e:
        cam.PylonCamera(99).open()
    assert "99" in str(e.value) or "camera" in str(e.value).lower()


def test_a_busy_basler_is_reported_not_swapped_for_a_webcam():
    """Measured on the acA5472-5gc: a second open is refused ("controlled by
    another application"). That exception was not a CameraError, so the thread
    died with no backend and no message. Reported as one, it would have fallen
    back to OpenCV device 0 -- some other camera -- and inspected that."""
    import importlib
    import fake_pylon as F
    F.install([F.Device("gige_classic", fail_open=True)])
    for m in [k for k in sys.modules if k.startswith("camera")]:
        del sys.modules[m]
    from vision_station.hardware import camera as cam_mod
    importlib.reload(cam_mod)
    cam_mod.PylonCamera.OPEN_RETRY_S = 0.6        # a held camera is retried a while
    t = cam_mod.CameraThread(0, cam_mod.CameraSettings())
    t.start()
    t.join(timeout=5)
    assert not t.is_alive()
    assert t.backend is None, "fell back to %s" % t.backend
    assert "held by another connection" in (t.last_error or ""), t.last_error
    assert "another application" in t.last_error


def test_jobs_share_one_camera_and_get_frames_taken_after_the_trigger(tmp_path, monkeypatch):
    """A job per side, one GigE camera: they must share a connection, the
    first part must wait for the camera to start rather than fail, and each
    grab must be a frame exposed after it was asked for."""
    import importlib
    import fake_pylon as F
    from PySide6.QtCore import QSettings
    from vision_station.ui import main_window as vp_app
    # a private settings store: this PC's own holds the real camera's address,
    # which the stand-in camera here does not answer to
    ini = str(tmp_path / "camera.ini")
    monkeypatch.setattr(vp_app, "camera_settings_store",
                        lambda: QSettings(ini, QSettings.IniFormat))
    scene = np.full((30, 40), 100, np.float32)
    F.install([F.Device("usb3", scene=scene, frame_period_s=0.03,
                        pipeline_depth=2)])
    for m in [k for k in sys.modules if k.startswith("camera")]:
        del sys.modules[m]
    from vision_station.hardware import camera as cam_mod
    importlib.reload(cam_mod)
    from vision_station.ui import main_window as vp_app

    class P:
        spec, reference = {}, None

    srcs = []
    for _ in range(3):
        s = vp_app.ImageSource(P())
        s.mode, s.camera_index, s.camera_factory = "camera", 0, vp_app.open_camera
        srcs.append(s)
    try:
        for s in srcs:
            img = s.grab()
            assert img is not None, s.last_error
        assert srcs[0].camera.thread is srcs[2].camera.thread
        seq = srcs[0].camera.thread._seq
        srcs[1].grab()
        assert srcs[0].camera.thread._seq >= seq + vp_app.ImageSource.FRESH_FRAMES
        srcs[0].stop_camera()
        srcs[1].stop_camera()
        assert srcs[2].grab() is not None, "released by another job"
    finally:
        for s in srcs:
            s.stop_camera()
    assert vp_app._shared_cameras == {}


def test_camera_thread_delivers_frames_and_stops(pylon_env):
    cam, _F = pylon_env
    t = cam.CameraThread(0, cam.CameraSettings())
    t.start()
    frame = None
    for _ in range(60):
        frame = t.get_frame()
        if frame is not None:
            break
        time.sleep(0.05)
    assert frame is not None, "camera thread produced no frame"
    assert t.backend == "pylon"
    assert t.info.get("serial") == "USB0001"
    t.stop()
    assert not t.alive


def test_camera_thread_falls_back_when_pylon_finds_nothing(pylon_env):
    """No Basler attached must degrade to OpenCV rather than dying. In this
    container OpenCV has nothing either, so the requirement is a clean stop
    with an explanation, not a hang or a traceback."""
    cam, F = pylon_env
    F.install([])                   # no devices at all
    import importlib
    importlib.reload(cam)
    t = cam.CameraThread(0, cam.CameraSettings())
    t.start()
    t.join(timeout=6.0)
    assert not t.is_alive(), "camera thread hung with no device present"
    assert t.last_error, "no explanation recorded for the failure"


# --------------------------------------------------- canvas interaction
@pytest.fixture
def canvas_env():
    """A real ImageCanvas driven with real Qt mouse events.

    Everything below could have been written months ago and was not, because
    selection was always driven programmatically through select_ids(). That gap
    hid a serious defect: RegionItem stored the region shape in an attribute
    named `shape`, which shadows QGraphicsItem.shape() -- the method Qt calls to
    hit-test a click. Clicking a region, marquee selection and contains() all
    raised TypeError, so the primary interaction of the application was broken,
    and a hundred passing tests said nothing because not one of them generated a
    mouse event."""
    from PySide6.QtWidgets import QApplication
    from vision_station.ui.canvas import ImageCanvas
    app = QApplication.instance() or QApplication([])
    c = ImageCanvas()
    c.resize(800, 600)
    c.set_image(np.full((400, 600, 3), 80, np.uint8), keep_view=False)
    c.sync_regions([
        {"id": "a", "params": {"roi": [100, 100, 120, 80], "enabled": True}},
        {"id": "b", "params": {"roi": [320, 220, 100, 60], "enabled": True}},
    ])
    c.resetTransform()          # 1:1 so scene and view coordinates agree
    c.centerOn(300, 200)
    yield c
    c.deleteLater()


def _vp(canvas, x, y):
    from PySide6.QtCore import QPointF
    return canvas.mapFromScene(QPointF(x, y))


def _click(canvas, x, y, modifier=None):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    mod = modifier or Qt.NoModifier
    pos = _vp(canvas, x, y)
    QTest.mousePress(canvas.viewport(), Qt.LeftButton, mod, pos)
    QTest.mouseRelease(canvas.viewport(), Qt.LeftButton, mod, pos)


def _drag(canvas, x0, y0, x1, y1, modifier=None, steps=4):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    mod = modifier or Qt.NoModifier
    QTest.mousePress(canvas.viewport(), Qt.LeftButton, mod, _vp(canvas, x0, y0))
    for i in range(1, steps + 1):
        QTest.mouseMove(canvas.viewport(),
                        _vp(canvas, x0 + (x1 - x0) * i / steps,
                            y0 + (y1 - y0) * i / steps))
    QTest.mouseRelease(canvas.viewport(), Qt.LeftButton, mod, _vp(canvas, x1, y1))


def test_region_hit_testing_is_not_shadowed(canvas_env):
    """Direct guard on the exact defect: `shape` must remain the Qt method."""
    from PySide6.QtCore import QPointF, QRectF
    c = canvas_env
    item = c.regions["a"]
    assert callable(item.shape), \
        "RegionItem.shape is %s, shadowing QGraphicsItem.shape()" % type(item.shape)
    assert c.sc.items(QPointF(150, 140)), "a point inside a region hits nothing"
    assert c.sc.items(QRectF(90, 90, 200, 120)), "rubber-band area hits nothing"
    assert item.contains(item.rect().center())


def test_click_selects_and_empty_click_clears(canvas_env):
    c = canvas_env
    _click(c, 150, 140)
    assert c.selected_ids() == ["a"], "clicking a region did not select it"
    _click(c, 550, 380)
    assert c.selected_ids() == [], "clicking empty canvas did not deselect"


def test_shift_click_adds_to_the_selection(canvas_env):
    from PySide6.QtCore import Qt
    c = canvas_env
    _click(c, 150, 140)
    _click(c, 360, 245, Qt.ShiftModifier)
    assert set(c.selected_ids()) == {"a", "b"}, \
        "shift-click gave %s" % c.selected_ids()


def test_marquee_selects_several(canvas_env):
    c = canvas_env
    _drag(c, 60, 60, 460, 320)
    assert set(c.selected_ids()) == {"a", "b"}, \
        "marquee selected %s" % c.selected_ids()


def test_drag_moves_the_region(canvas_env):
    c = canvas_env
    _click(c, 150, 140)
    before = c.regions["a"].roi()
    _drag(c, 150, 140, 210, 190)
    after = c.regions["a"].roi()
    assert abs((after[0] - before[0]) - 60) <= 3, "x moved %s" % (after[0] - before[0])
    assert abs((after[1] - before[1]) - 50) <= 3, "y moved %s" % (after[1] - before[1])
    assert after[2:] == before[2:], "dragging changed the size"


def test_multi_selection_drags_together(canvas_env):
    from PySide6.QtCore import Qt
    c = canvas_env
    _click(c, 150, 140)
    _click(c, 360, 245, Qt.ShiftModifier)
    a0, b0 = c.regions["a"].roi(), c.regions["b"].roi()
    _drag(c, 150, 140, 190, 170)
    a1, b1 = c.regions["a"].roi(), c.regions["b"].roi()
    assert abs((a1[0] - a0[0]) - 40) <= 3
    assert abs((b1[0] - b0[0]) - 40) <= 3, \
        "the second selected region did not move with the first"


def test_corner_handle_resizes(canvas_env):
    c = canvas_env
    _click(c, 150, 140)
    x, y, w, h = c.regions["a"].roi()
    _drag(c, x + w, y + h, x + w + 40, y + h + 30)      # bottom-right handle
    nx, ny, nw, nh = c.regions["a"].roi()
    assert nw > w + 20, "width did not grow: %s -> %s" % (w, nw)
    assert nh > h + 15, "height did not grow: %s -> %s" % (h, nh)
    assert (nx, ny) == (x, y), "resizing from the corner moved the origin"


def test_arrow_keys_nudge_the_selection(canvas_env):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    c = canvas_env
    _click(c, 150, 140)
    x0 = c.regions["a"].roi()[0]
    QTest.keyClick(c, Qt.Key_Right)
    assert c.regions["a"].roi()[0] == x0 + 1, "arrow key did not nudge by 1 px"
    QTest.keyClick(c, Qt.Key_Right, Qt.ShiftModifier)
    assert c.regions["a"].roi()[0] == x0 + 11, "shift-arrow did not nudge by 10 px"


def test_escape_clears_the_selection(canvas_env):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    c = canvas_env
    _click(c, 150, 140)
    assert c.selected_ids()
    QTest.keyClick(c, Qt.Key_Escape)
    assert c.selected_ids() == []


def test_alt_drag_asks_for_a_duplicate(canvas_env):
    from PySide6.QtCore import Qt
    c = canvas_env
    got = []
    c.duplicateRequested.connect(lambda ids: got.append(list(ids)))
    _click(c, 150, 140)
    _drag(c, 150, 140, 200, 180, Qt.AltModifier)
    assert got and "a" in got[0], "alt-drag did not request a duplicate: %s" % got


def test_hand_tool_pans_without_moving_regions(canvas_env):
    c = canvas_env
    before = c.regions["a"].roi()
    c.set_mode("hand")
    _drag(c, 150, 140, 220, 200)
    assert c.regions["a"].roi() == before, \
        "the hand tool dragged a region instead of panning"
    c.set_mode("select")


def test_geometry_signal_fires_on_drag(canvas_env):
    """app.py keeps the recipe in step through these signals. Silent geometry
    is the same as no geometry."""
    c = canvas_env
    committed = []
    c.geometryCommitted.connect(lambda tid, roi: committed.append((tid, list(roi))))
    _click(c, 150, 140)
    _drag(c, 150, 140, 200, 175)
    assert committed, "no geometryCommitted signal after a drag"
    assert committed[-1][0] == "a"


def test_region_edges_snap_to_a_neighbour(canvas_env):
    """Dragging one region's left edge close to another's must land exactly on
    it, otherwise the alignment guides are decoration."""
    c = canvas_env
    _click(c, 150, 140)
    target_left = c.regions["b"].roi()[0]
    x, y, w, h = c.regions["a"].roi()
    # aim to land 3 px short of b's left edge; snapping should close the gap
    _drag(c, x + w / 2, y + h / 2,
          target_left - 3 + w / 2, y + h / 2)
    assert c.regions["a"].roi()[0] == target_left, \
        "left edge landed at %s, expected snap to %s" % (
            c.regions["a"].roi()[0], target_left)


def test_label_layer_drag_creates_a_box(canvas_env):
    """In the labelling workspace an empty-canvas drag draws a new box rather
    than a marquee."""
    c = canvas_env
    made = []
    c.labelAdded.connect(lambda cls, roi: made.append((cls, list(roi))))
    c.set_layer("labels")
    c.new_class = 0
    _drag(c, 420, 100, 500, 160)
    assert made, "no label box created by dragging in the label layer"
    cls, roi = made[0]
    assert cls == 0
    assert abs(roi[2] - 80) <= 4 and abs(roi[3] - 60) <= 4, "box size %s" % roi


# ------------------------------------------------------- environment
def test_preflight_reports_the_real_environment():
    from vision_station.core import preflight
    rows = preflight.check()
    names = {r["name"] for r in rows}
    assert {"PySide6", "OpenCV", "NumPy"} <= names
    for r in rows:
        assert isinstance(r["ok"], bool)
        assert r["why"] and r["fix"], "%s has no explanation or fix" % r["name"]
    # the three the app cannot start without are present, or nothing else ran
    assert not preflight.missing_required()


def _fake_paddle(monkeypatch, generation):
    """A stand-in paddleocr module with one generation's interface: version 3
    rejects show_log and answers predict(); version 2 rejects the version-3
    argument names and answers ocr(img, cls=True)."""
    import types
    from vision_station.tools import text as T

    class Result3(dict):
        pass

    class PaddleOCR:
        def __init__(self, **kw):
            if generation == 3 and "show_log" in kw:
                raise ValueError("Unknown argument: show_log")
            if generation == 2 and "use_textline_orientation" in kw:
                raise ValueError("unexpected keyword use_textline_orientation")
            self.kw = kw

        def predict(self, img):
            return [Result3(rec_texts=["LOT 42", "SN 7"], rec_scores=[0.97, 0.91])]

        def ocr(self, img, cls=False):
            return [[[[[0, 0]] * 4, ("LOT 42", 0.97)], [[[0, 0]] * 4, ("SN 7", 0.91)]]]

    if generation == 2:
        del PaddleOCR.predict
    mod = types.ModuleType("paddleocr")
    mod.PaddleOCR = PaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", mod)
    monkeypatch.setitem(T._ENGINE, "name", None)
    monkeypatch.setitem(T._ENGINE, "obj", None)
    monkeypatch.setitem(T._ENGINE, "errors", {})
    return T


@pytest.mark.parametrize("generation", [2, 3])
def test_paddleocr_starts_and_reads_with_either_generation(monkeypatch, generation):
    """With PaddleOCR 3 -- what pip installs today -- the old constructor failed
    on show_log, so PaddleOCR never ran, even with network, and every read fell
    back to Tesseract without a word."""
    T = _fake_paddle(monkeypatch, generation)
    e = T._engine()
    assert e["name"] == "paddle", "PaddleOCR %d did not start: %s" % (generation, e["errors"])
    lines = T._recognise(np.zeros((40, 200, 3), np.uint8), {})
    assert lines == [("LOT 42", 0.97), ("SN 7", 0.91)]


def test_why_an_engine_did_not_start_is_recorded(monkeypatch):
    import types
    from vision_station.tools import text as T

    class Broken:
        def __init__(self, **kw):
            raise RuntimeError("No available model hosting platforms detected.")
    mod = types.ModuleType("paddleocr")
    mod.PaddleOCR = Broken
    monkeypatch.setitem(sys.modules, "paddleocr", mod)
    monkeypatch.setitem(T._ENGINE, "name", None)
    monkeypatch.setitem(T._ENGINE, "obj", None)
    monkeypatch.setitem(T._ENGINE, "errors", {})
    T._engine()
    assert "model hosting" in T._ENGINE["errors"]["paddle"], T._ENGINE["errors"]


def test_preflight_names_the_engine_that_will_actually_run():
    """A recipe tuned against PaddleOCR and deployed where only Tesseract is
    installed reads differently, and nothing in the inspection result explains
    why. Knowing which engine is live is the first thing to check."""
    from vision_station.core import preflight
    from vision_station.tools import ocr_engine
    from vision_station.tools.text import _engine
    reported = preflight.ocr_engine()
    if reported is None:
        pytest.skip("no OCR engine installed")
    if ocr_engine.onnx_available():
        st = ocr_engine.status()
        assert st["engine"] == "onnx" and reported == "PP-OCR on the %s" % st["device"], \
            "preflight says %s, the reader is %s" % (reported, st)
        r, _ = _run("ocr", {"roi": [0, 0, 520, 120]}, _text_image("AB12345X"))
        assert st["device"].lower() in r.values["engine"], \
            "the result does not name the device that read it: %s" % r.values["engine"]
        return
    actual = _engine()["name"]
    mapping = {"PaddleOCR": "paddle", "EasyOCR": "easyocr", "Tesseract": "tesseract"}
    assert mapping[reported] == actual, \
        "preflight says %s, the tool actually loaded %s" % (reported, actual)


def test_missing_code_reader_is_reported_as_such():
    """A missing decoder library must not look like an unreadable code. It used
    to: the ImportError was swallowed and the tool reported 'no code decoded',
    sending an operator to check lighting and focus for a missing pip package."""
    from vision_station.core.schema import defaults
    img = np.full((120, 200, 3), 200, np.uint8)
    r, _ = _run("code_read", dict(defaults("code_read"), roi=[0, 0, 200, 120]), img)
    try:
        import pyzbar  # noqa: F401
        installed = True
    except ImportError:
        installed = False
    if installed:
        assert r.status in ("pass", "fail")
    else:
        assert r.status == "error", \
            "no decoder installed but the tool said %r" % r.message
        assert "install" in (r.message or "").lower()


def test_text_tools_record_which_engine_read_the_marking():
    """Traceability: a logged result has to say what read it, because the
    answer changes with the engine."""
    if not _has_ocr():
        pytest.skip("no OCR engine installed")
    img = _text_image("AB12345X")
    for ttype, extra in (("ocr", {}),
                         ("ocv", {"expect_regex": "^[A-Z0-9]{6,12}$",
                                  "expect_text": ""})):
        r, _ = _run(ttype, _ocr_params(**extra), img)
        assert r.values.get("engine"), "%s did not record the OCR engine" % ttype


# --------------------------------------------------------------- installer
class _Proc:
    def __init__(self, rc, out=""):
        self.returncode, self.stdout, self.stderr = rc, out, ""


def test_installer_continues_when_optional_packages_fail():
    """The installer logic lives in Python precisely so this can be tested. In
    batch it was shipped unexecuted. A machine behind a proxy commonly gets some
    packages and not others, and stopping at the first refusal leaves it worse
    off than carrying on."""
    import install
    logs = []
    fails = install.install_group(install.OPTIONAL,
                                  runner=lambda c: _Proc(1, "proxy refused"),
                                  log=logs.append)
    assert len(fails) == len(install.OPTIONAL), "an optional failure was hidden"
    assert all("skipped" in l for l in logs)
    assert all("unavailable" in l for l in logs), \
        "a skipped package did not say what it costs"


def test_installer_marks_core_failures_as_fatal():
    import install
    logs = []

    def picky(cmd):
        return _Proc(1, "no matching distribution") \
            if any("PySide6" in c for c in cmd) else _Proc(0)

    fails = install.install_group(install.CORE, runner=picky, required=True,
                                  log=logs.append)
    assert [f[0] for f in fails] == ["PySide6"]
    assert any("FAILED" in l for l in logs), "a core failure was reported as a skip"


def test_installer_survives_pip_being_absent():
    import install
    ok, out = install.pip_install(
        ["anything"], runner=lambda c: (_ for _ in ()).throw(FileNotFoundError("no pip")))
    assert ok is False
    assert "no pip" in out


def test_installer_refuses_an_old_python(monkeypatch):
    import install
    monkeypatch.setattr(install, "MIN_PYTHON", (99, 0))
    assert install.main(["--dry-run", "--yes"]) == 2, \
        "an unsupported Python was allowed to proceed"


def test_installer_aborts_on_core_failure(monkeypatch):
    import install
    monkeypatch.setattr(install, "install_group",
                        lambda g, d=False, r=None, required=False, log=print:
                        [("PySide6", "the window", "no distribution")] if required else [])
    assert install.main(["--yes"]) == 1


def test_installer_dry_run_changes_nothing():
    import install
    called = []
    ok, out = install.pip_install(["PySide6"], dry_run=True,
                                  runner=lambda c: called.append(c) or _Proc(0))
    assert ok and not called, "a dry run still invoked pip"
    assert "would run" in out


def test_install_batch_is_only_a_launcher():
    """Everything that can go wrong belongs in install.py where it can be run.
    If logic creeps back into the batch file it becomes untestable again."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "install.bat")
    if not os.path.exists(path):
        pytest.skip("install.bat not present")
    body = _slurp(path)
    assert "install.py" in body, "the batch file does not call the Python installer"
    assert "pip install" not in body, \
        "package installation has crept back into the untestable batch file"
    assert len([l for l in body.splitlines() if l.strip()]) < 20, \
        "install.bat is growing logic again"


# ------------------------------------------------------------------- the UI
@pytest.fixture
def ui(tmp_path):
    """A MainWindow on a scratch copy of the projects folder."""
    import shutil
    from PySide6.QtWidgets import QApplication
    from vision_station.ui import classic_window as appmod
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Never copy an existing results log into a test fixture. A stale database
    # left in the source tree by a manual run makes history assertions count
    # somebody else's inspections, which fails in a way that looks like a bug
    # in the history dialog rather than contamination.
    # Only the demo project. Copying the whole projects folder duplicated every
    # job on disk for every UI test -- once a seven-side job of full-resolution
    # photographs existed, one test run wrote several gigabytes and filled the
    # disk. A stale results log is excluded too: history assertions would
    # otherwise count somebody else's inspections.
    shutil.copytree(os.path.join(here, "projects", "demo_led_panel"),
                    str(tmp_path / "projects" / "demo_led_panel"),
                    ignore=shutil.ignore_patterns("results", "dataset"))
    orig = appmod.HERE
    appmod.HERE = str(tmp_path)
    QApplication.instance() or QApplication([])
    w = appmod.MainWindow()
    # Be explicit about the model. The window opens on whichever project sorts
    # first, so adding a second one silently moved these tests onto it.
    if w.projBox.findText("demo_led_panel") >= 0:
        w.projBox.setCurrentText("demo_led_panel")
    img = cv2.imread(os.path.join(here, "projects", "demo_led_panel", "good.png"))
    w.canvas.set_image(img, keep_view=False)
    w.canvas.sync_regions(w.spec["tools"])
    yield w, tmp_path, img
    w.dirty = False
    # Close the window. Left open, a test that showed and activated it kept an
    # active window alive, and later tests' key events went astray -- two Escape
    # tests failed only when they ran after a Run-mode test.
    if w._worker is not None:
        w.wait_for_inspection()
    w.close()
    w.deleteLater()
    appmod.HERE = orig


@qt_required
def test_ui_copy_paste_duplicates_the_whole_tool(ui):
    """A pasted region must carry every setting, not just its box. Copying a
    tuned tool and getting back a default one is worse than no copy at all,
    because the settings look right in the tree and are not."""
    w, _tmp, _img = ui
    src = [t for t in w.spec["tools"] if t["id"] == "lot_code"][0]
    src["params"]["expect_regex"] = "^UNIQUE[0-9]{3}$"
    src["params"]["upscale"] = 3.5
    n_before = len(w.spec["tools"])

    w.canvas.select_ids(["lot_code"])
    w.copy_sel()
    w.paste()

    assert len(w.spec["tools"]) == n_before + 1, "paste added nothing"
    new = w.spec["tools"][-1]
    assert new["id"] != "lot_code", "the copy reused the original name"
    assert new["params"]["expect_regex"] == "^UNIQUE[0-9]{3}$", \
        "settings were not carried into the copy"
    assert new["params"]["upscale"] == 3.5
    assert new["params"]["roi"] != src["params"]["roi"], \
        "the copy landed exactly on top of the original"


@qt_required
def test_ui_undo_restores_a_deleted_tool(ui):
    w, _tmp, _img = ui
    n = len(w.spec["tools"])
    w.canvas.select_ids(["surface"])
    w.remove_tool()
    assert len(w.spec["tools"]) == n - 1
    assert "surface" not in [t["id"] for t in w.spec["tools"]]
    w.undo()
    assert len(w.spec["tools"]) == n, "undo did not restore the deleted tool"
    assert "surface" in [t["id"] for t in w.spec["tools"]]
    w.redo()
    assert "surface" not in [t["id"] for t in w.spec["tools"]], "redo did not reapply"


@qt_required
def test_ui_reorder_changes_execution_order_and_persists(ui):
    """Order matters -- Locate part has to run before anything that depends on
    it -- so a reorder that does not survive a save is a trap."""
    w, tmp, _img = ui
    ids = [t["id"] for t in w.spec["tools"]]
    w.canvas.select_ids([ids[2]])
    w.move_tool(-1)
    moved = [t["id"] for t in w.spec["tools"]]
    assert moved[1] == ids[2], "move up did not reorder: %s" % moved
    w.save_job()
    pid = w.projBox.currentText()
    saved = json.loads(_slurp(str(tmp / "projects" / pid / "project.json"),
                           encoding="utf-8"))
    assert [t["id"] for t in saved["tools"]] == moved, "reorder did not persist"


@qt_required
def test_ui_disabling_a_tool_skips_it_at_runtime(ui):
    w, _tmp, img = ui
    w.run_once()
    w.wait_for_inspection()
    assert any(t["tool_id"] == "surface" for t in w.last["tools"])
    for t in w.spec["tools"]:
        if t["id"] == "surface":
            t["params"]["enabled"] = False
    w.run_once()
    w.wait_for_inspection()
    assert not any(t["tool_id"] == "surface" for t in w.last["tools"]), \
        "a tool disabled in the UI still ran"


@qt_required
def test_ui_batch_run_over_test_images(ui):
    """Run on every test image is how a recipe change gets checked before it
    reaches a line, so it has to actually inspect each one."""
    w, _tmp, _img = ui
    w.refresh_images()
    assert w.imgList.count() >= 2, "the demo model has no test images"
    # run_batch finishes with a modal summary box, which blocks forever with
    # nobody to dismiss it. Stubbing it is not hiding a defect: showing a
    # summary is the correct behaviour, it just cannot be answered headlessly.
    from PySide6.QtWidgets import QMessageBox
    seen = []
    real = w.inspect
    real_info = QMessageBox.information
    w.inspect = lambda im: (seen.append(im.shape), real(im))[1]
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
    try:
        w.run_batch()
    finally:
        w.inspect = real
        QMessageBox.information = real_info
    assert len(seen) == w.imgList.count(), \
        "batch inspected %d of %d images" % (len(seen), w.imgList.count())


@qt_required
def test_ui_results_are_logged_and_queryable(ui):
    w, _tmp, _img = ui
    for _ in range(3):
        w.run_once()
        w.wait_for_inspection()
    w.results.flush(4.0)
    summary = w.results.summary()
    assert summary["total"] >= 3, "inspections were not logged: %s" % summary
    assert w.results.recent(1), "no recent runs recorded"


@qt_required
def test_ui_diagnostics_reports_on_the_current_image(ui):
    """The dialog is read-only, but it must not raise on a real recipe and its
    margin table has to reflect the run that just happened."""
    w, _tmp, img = ui
    w.run_once()
    w.wait_for_inspection()
    from vision_station.core import diagnostics
    q = diagnostics.image_quality(img)
    assert q["findings"]
    rows = diagnostics.margin_report(w.last, w.spec)
    assert rows, "no tool in the demo recipe produced a margin row"
    assert all(set(r) >= {"tool_id", "score", "threshold", "verdict"} for r in rows)
    timing = diagnostics.timing_report(w.last)
    assert timing["total_ms"] > 0
    assert len(timing["rows"]) == len(w.last["tools"])


@qt_required
def test_ui_labelling_round_trip(ui):
    """Labels drawn in the label workspace must survive a switch away and back,
    and land in the dataset the exporter reads."""
    w, _tmp, _img = ui
    w.refresh_images()
    w.set_workspace("label")
    w.ds.add_class("screw")
    w.refresh_classes()
    w.imgList.setCurrentRow(0)
    w.on_image_picked(w.imgList.item(0))
    w.on_label_added(0, [40, 40, 60, 60])
    assert len(w.canvas.label_boxes()) == 1, "the box did not appear on the canvas"
    name = w.cur_image
    w.set_workspace("inspect")
    w.set_workspace("label")
    w.on_image_picked(w.imgList.item(0))
    assert len(w.ds.get(name)) == 1, "the label was lost switching workspaces"
    stats = w.ds.stats()
    assert stats["total_boxes"] == 1 and stats["per_class"].get("screw") == 1


@qt_required
def test_every_dialog_opens_without_raising(ui, monkeypatch):
    """Open every dialog in the application with the modals auto-dismissed.

    These are the last paths that had never been exercised by anything but a
    hand-written script. A dialog that raises is invisible to every other test
    -- the traceback goes to a console nobody is watching and the window simply
    never appears."""
    from PySide6.QtWidgets import (QDialog, QMessageBox, QInputDialog,
                                   QFileDialog)
    w, tmp, _img = ui
    w.run_once()
    w.wait_for_inspection()
    w.refresh_images()

    opened = []
    monkeypatch.setattr(QDialog, "exec",
                        lambda self: (opened.append(self), QDialog.Accepted)[-1])
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("Probe", True)))
    monkeypatch.setattr(QInputDialog, "getInt",
                        staticmethod(lambda *a, **k: (0, True)))
    monkeypatch.setattr(QInputDialog, "getItem",
                        staticmethod(lambda *a, **k: ("Save every image", True)))
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp / "out.csv"), "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp)))

    for name in ("show_shortcuts", "show_history", "show_diagnostics",
                 "set_archive_mode", "show_camera_settings", "check_dataset",
                 "show_training_help", "run_stability", "export_job",
                 "rename_tool"):
        getattr(w, name)()          # any exception fails the test

    # The two real dialogs must actually build, not slip out through a guard.
    assert any(d.windowTitle().startswith("Production history") for d in opened), \
        "the history dialog never opened"
    assert any(d.windowTitle() == "Diagnostics" for d in opened), \
        "the diagnostics dialog never opened"
    w.dirty = False


@qt_required
def test_history_dialog_matches_the_stored_results(ui, monkeypatch):
    """The tables have to agree with the database. A reporting screen that
    quietly under-counts is worse than none, because it will be believed."""
    from PySide6.QtWidgets import QDialog, QTableWidget
    w, _tmp, _img = ui
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    good = cv2.imread(os.path.join(here, "projects", "demo_led_panel", "good.png"))
    bad = cv2.imread(os.path.join(here, "projects", "demo_led_panel", "bad.png"))
    w.canvas.set_image(good, keep_view=False)
    for _ in range(3):
        w.run_once()
        w.wait_for_inspection()
    w.canvas.set_image(bad, keep_view=False)
    for _ in range(2):
        w.run_once()
        w.wait_for_inspection()
    w.results.flush(5.0)

    opened = []
    monkeypatch.setattr(QDialog, "exec",
                        lambda self: (opened.append(self), QDialog.Accepted)[-1])
    w.show_history()
    d = opened[0]
    tables = {}
    for t in d.findChildren(QTableWidget):
        key = t.horizontalHeaderItem(0).text()
        tables[key] = t

    assert tables["Time"].rowCount() == 5, \
        "runs table shows %d of 5 inspections" % tables["Time"].rowCount()
    assert tables["Check"].rowCount() > 0, "no failing checks listed despite 2 rejects"
    assert tables["Hour"].rowCount() >= 1

    summary = w.results.summary()
    assert summary["total"] == 5 and summary["failed"] == 2, \
        "store disagrees with what was run: %s" % summary
    w.dirty = False


@qt_required
def test_clicking_a_region_syncs_tree_and_property_sheet(tmp_path):
    """The three panels have to agree. Clicking a region on the image must
    highlight it in the job tree and load it into the property sheet, and a
    drag must reach the recipe and mark it unsaved.

    This is the integration the `shape` shadowing bug destroyed: hit-testing
    raised, so none of it happened, and no test noticed because none of them
    clicked anything."""
    import shutil
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtTest import QTest
    from vision_station.ui import classic_window as appmod

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.copytree(os.path.join(here, "projects", "demo_led_panel"),
                    str(tmp_path / "projects" / "demo_led_panel"),
                    ignore=shutil.ignore_patterns("results", "dataset"))
    orig_here = appmod.HERE
    appmod.HERE = str(tmp_path)
    QApplication.instance() or QApplication([])
    try:
        w = appmod.MainWindow()
        if w.projBox.findText("demo_led_panel") >= 0:
            w.projBox.setCurrentText("demo_led_panel")
        img = cv2.imread(os.path.join(here, "projects", "demo_led_panel", "good.png"))
        w.canvas.set_image(img, keep_view=False)
        w.canvas.sync_regions(w.spec["tools"])
        c = w.canvas

        def centre_of(tool_id):
            r = [t for t in w.spec["tools"] if t["id"] == tool_id][0]["params"]["roi"]
            return QPointF(r[0] + r[2] / 2.0, r[1] + r[3] / 2.0)

        def click_at(pt, mod=Qt.NoModifier):
            p = c.mapFromScene(pt)
            QTest.mousePress(c.viewport(), Qt.LeftButton, mod, p)
            QTest.mouseRelease(c.viewport(), Qt.LeftButton, mod, p)

        click_at(centre_of("lot_code"))
        assert c.selected_ids() == ["lot_code"], \
            "clicking a region selected %s" % c.selected_ids()
        assert [i.text(0) for i in w.tree.selectedItems()] == ["lot_code"], \
            "the job tree did not follow the canvas selection"
        assert "lot_code" in w.props.header.text(), \
            "the property sheet did not load the clicked tool"

        click_at(centre_of("surface"), Qt.ShiftModifier)
        assert set(c.selected_ids()) == {"lot_code", "surface"}
        assert "2 regions" in w.props.header.text()

        before = {t["id"]: list(t["params"]["roi"]) for t in w.spec["tools"]
                  if t["id"] in ("lot_code", "surface")}
        start = centre_of("surface")
        QTest.mousePress(c.viewport(), Qt.LeftButton, Qt.NoModifier,
                         c.mapFromScene(start))
        for k in range(1, 5):
            QTest.mouseMove(c.viewport(), c.mapFromScene(
                QPointF(start.x() + 8 * k, start.y() + 5 * k)))
        QTest.mouseRelease(c.viewport(), Qt.LeftButton, Qt.NoModifier,
                           c.mapFromScene(QPointF(start.x() + 32, start.y() + 20)))
        after = {t["id"]: list(t["params"]["roi"]) for t in w.spec["tools"]
                 if t["id"] in ("lot_code", "surface")}
        for tid in before:
            assert after[tid][0] > before[tid][0], \
                "%s did not move with the group drag" % tid
        assert w.dirty, "a geometry change did not mark the recipe unsaved"
        w.dirty = False
    finally:
        appmod.HERE = orig_here


@qt_required
def test_building_a_recipe_through_the_ui(tmp_path):
    """Drives the real window: create a model, add tools, edit regions through
    the property-sheet spinboxes, save, reload, run.

    Worth its weight because the one bug that reached a shipped release lived
    exactly here -- correct through hand-written JSON, broken through the
    property sheet. Every test at the time used hand-written JSON."""
    import shutil
    from PySide6.QtWidgets import QApplication, QInputDialog
    from vision_station.ui import classic_window as appmod

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.copytree(os.path.join(here, "projects", "demo_led_panel"),
                    str(tmp_path / "projects" / "demo_led_panel"),
                    ignore=shutil.ignore_patterns("results", "dataset"))
    orig_here = appmod.HERE
    appmod.HERE = str(tmp_path)
    qapp = QApplication.instance() or QApplication([])
    orig_prompt = QInputDialog.getText
    try:
        w = appmod.MainWindow()
        if w.projBox.findText("demo_led_panel") >= 0:
            w.projBox.setCurrentText("demo_led_panel")
        img = cv2.imread(os.path.join(here, "projects", "demo_led_panel", "good.png"))
        assert img is not None
        w.canvas.set_image(img, keep_view=False)

        QInputDialog.getText = staticmethod(lambda *a, **k: ("Widget Rev A", True))
        w.new_project()
        assert w.projBox.currentText() == "widget_rev_a"
        assert w.spec["tools"] == []

        w.canvas.set_image(img, keep_view=False)
        w.set_reference()
        assert w.project.reference is not None

        for t in ("align", "count_blob", "ocv"):
            w.add_tool(t)
        ids = [x["id"] for x in w.spec["tools"]]
        assert len(ids) == 3
        assert all(x["params"].get("roi") for x in w.spec["tools"]), \
            "a UI-added tool has no region"

        w.canvas.select_ids([ids[1]])
        for widget, val in ((w.props.rx, 120), (w.props.ry, 120),
                            (w.props.rw, 400), (w.props.rh, 90)):
            widget.setValue(val)
        target = [x for x in w.spec["tools"] if x["id"] == ids[1]][0]["params"]
        assert target["roi"] == [120, 120, 400, 90], \
            "spinbox edits did not reach the recipe: %s" % target["roi"]

        w.props.rangle.setValue(15.0)
        w.props.rshape.setCurrentIndex(1)
        assert abs(target.get("angle", 0) - 15.0) < 1e-6
        assert target.get("shape") == "ellipse"

        w.save_job()
        pid = w.projBox.currentText()
        saved = json.loads(_slurp(str(tmp_path / "projects" / pid / "project.json"),
                               encoding="utf-8"))
        assert len(saved["tools"]) == 3, "recipe did not persist"
        w.load_project(pid)
        assert len(w.spec["tools"]) == 3

        w.canvas.set_image(img, keep_view=False)
        w.run_once()
        w.wait_for_inspection()
        assert w.last is not None, "no result from a UI-built recipe"
        errored = [t["tool_id"] for t in w.last["tools"] if t["status"] == "error"]
        assert not errored, "tools errored in a UI-built recipe: %s" % errored
        w.dirty = False
    finally:
        QInputDialog.getText = orig_prompt
        appmod.HERE = orig_here


# -------------------------------------------------------------------- misc
def test_every_registered_tool_has_a_schema():
    """A tool with no schema entry renders an empty property page, which looks
    like a broken app rather than a missing definition."""
    from vision_station.tools import REGISTRY
    from vision_station.core.schema import LABELS, SCHEMA
    missing_schema = sorted(set(REGISTRY) - set(SCHEMA))
    missing_label = sorted(set(REGISTRY) - set(LABELS))
    assert not missing_schema, "no schema for: %s" % missing_schema
    assert not missing_label, "no label for: %s" % missing_label


def test_schema_defaults_are_within_declared_ranges():
    from vision_station.core.schema import SCHEMA
    bad = []
    for ttype, rows in SCHEMA.items():
        for key, _lab, kind, default, extra in rows:
            if kind in ("int", "float") and default is not None and extra:
                lo, hi = extra[0], extra[1]
                if not (lo <= default <= hi):
                    bad.append("%s.%s=%s outside [%s,%s]" % (ttype, key, default, lo, hi))
    assert not bad, "; ".join(bad)


def test_defaults_include_geometry_keys():
    from vision_station.core.schema import defaults
    d = defaults("presence")
    for k in ("enabled", "critical", "angle", "shape"):
        assert k in d, "defaults() missing %s" % k


def test_zooming_and_replacing_regions_does_not_crash(tmp_path):
    """The application aborted with no message ("pure virtual function call")
    or faulted. A region's bounding rectangle includes handles sized in screen
    pixels, so it changes with the zoom; Qt's BSP scene index never heard of
    it, kept stale pointers to removed regions, and the next repaint called
    into freed memory. Zoom, replace regions and repaint, many times over -- in
    a separate process, because the failure is a crash, not an exception."""
    import subprocess
    script = tmp_path / "zoom_and_replace.py"
    script.write_text(r'''
import os, sys, random
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, %r)
import numpy as np
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv)
from vision_station.ui.canvas import ImageCanvas
c = ImageCanvas(); c.resize(640, 480); c.show()
c.set_image(np.zeros((1944, 2592, 3), np.uint8))
random.seed(1)
tools = [{"id": "t%%d" %% k, "params": {"roi": [random.randint(0, 2400), random.randint(0, 1800),
                                              random.randint(20, 300), random.randint(20, 300)]}}
         for k in range(30)]
for i in range(1500):
    r = random.random()
    if r < 0.35:
        c.zoom(random.choice([1.6, 0.6, 2.5, 0.4]))
    elif r < 0.45:
        c.fit()
    elif r < 0.75:
        keep = random.sample(tools, random.randint(0, len(tools)))
        c.sync_regions(keep)
        if keep:
            c.select_ids([random.choice(keep)["id"]])
    else:
        c.centerOn(random.randint(0, 2592), random.randint(0, 1944))
    c.viewport().repaint()
    app.processEvents()
print("SURVIVED")
''' % os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0 and "SURVIVED" in r.stdout, \
        "the canvas crashed (exit %s): %s" % (r.returncode, r.stderr[-800:])


# ------------------------------------------------------ installer and GPU
def test_installer_never_puts_a_second_onnxruntime_beside_a_gpu_build(monkeypatch):
    """Every onnxruntime build provides the same module, so the plain build
    installed after a GPU build replaces it. Running the installer a second
    time did exactly that on a machine with an RTX card: text reading and
    every trained model went back to the CPU without a word."""
    import importlib.metadata as md
    import install

    def have(*names):
        def version(n):
            if n in names:
                return "1.0"
            raise md.PackageNotFoundError(n)
        return version

    monkeypatch.setattr(md, "version", have("onnxruntime-directml"))
    assert install.onnxruntime_package() is None
    monkeypatch.setattr(md, "version", have("onnxruntime-gpu"))
    assert install.onnxruntime_package() is None
    monkeypatch.setattr(md, "version", have("onnxruntime"))
    assert install.onnxruntime_package() is None
    monkeypatch.setattr(md, "version", have())
    monkeypatch.setattr(sys, "platform", "win32")
    assert install.onnxruntime_package() == "onnxruntime-directml"
    monkeypatch.setattr(sys, "platform", "linux")
    assert install.onnxruntime_package() == "onnxruntime"
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wanted = [l.split(";")[0].split(">")[0].strip()
              for l in _slurp(os.path.join(here, "requirements.txt")).splitlines()
              if l.strip().startswith("onnxruntime") and 'sys_platform == "win32"' in l]
    assert wanted == ["onnxruntime-directml"], wanted


def test_text_reader_models_are_fetched_under_the_names_it_looks_for():
    import get_ocr_models as g
    from vision_station.tools import ocr_engine
    for table in (g.FULL, g.SMALL):
        names = set(table)
        assert names & set(ocr_engine.DET_FILES) and names & set(ocr_engine.REC_FILES)
        assert ocr_engine.CLS_FILE in names
        rec = next(n for n in names if n in ocr_engine.REC_FILES)
        assert rec[:-5] + ".yml" in names, "no dictionary source for %s" % rec


def test_recogniser_dictionary_is_read_from_its_yml(tmp_path):
    import get_ocr_models as g
    yml = tmp_path / "rec.yml"
    body = ["Global:", "  model_name: x", "PostProcess:", "  name: CTCLabelDecode",
            "  character_dict:", "  - '!'", "  - ''''", "  - A", "  - '0'",
            "  - 　", "  - '-'", "Other:", "  - no", ""]
    yml.write_text("\n".join(body), encoding="utf-8")
    n = g.dictionary_from_yml(str(yml), str(tmp_path / "rec.txt"))
    got = (tmp_path / "rec.txt").read_text(encoding="utf-8").split("\n")[:-1]
    assert n == 6 and got == ["!", "'", "A", "0", "　", "-"], got


def test_camera_leaves_a_yuv_format_for_the_bayer_one():
    """A colour GigE camera in YUV 4:2:2 sends two bytes a pixel. At 20
    megapixels the frames no longer fit the link and arrived with the bottom
    eighth missing -- a solid green band, inspected as if it were the part."""
    from vision_station.hardware import camera

    class Node:
        def __init__(self, value, symbolics=()):
            self.value, self.Symbolics, self.sets = value, list(symbolics), []

        def IsWritable(self):
            return True

        def GetValue(self):
            return self.value

        def SetValue(self, v):
            if v not in self.Symbolics:
                raise ValueError(v)
            self.value = v
            self.sets.append(v)

    class Cam:
        pass

    pc = camera.PylonCamera(0)
    pc.cam = Cam()
    pc.cam.PixelFormat = Node("YUV422Packed", ["Mono8", "BayerRG8", "BayerRG12", "YUV422Packed"])
    pc._use_lean_pixel_format()
    assert pc.cam.PixelFormat.value == "BayerRG8"
    pc.cam.PixelFormat = Node("BayerRG12", ["Mono8", "BayerRG8", "BayerRG12"])
    pc._use_lean_pixel_format()
    assert pc.cam.PixelFormat.sets == [], "a format that was not YUV was changed"
    pc.cam.PixelFormat = Node("YUV422Packed", ["Mono8", "YUV422Packed"])
    pc._use_lean_pixel_format()                       # nothing better: left alone
    assert pc.cam.PixelFormat.value == "YUV422Packed"


def test_image_source_and_save_ignore_a_buttons_checked_flag():
    """Qt hands a button's or menu action's checked state to its slot. Both of
    these took it for their argument: 'expected str ... not bool' on saving a
    camera image, and 'bool object is not iterable' on opening Image Source."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert "isinstance(path, str)" in _slurp(os.path.join(here, "vision_station", "ui", "live_view.py"))
    assert "isinstance(paths, (list, tuple))" in _slurp(os.path.join(here, "vision_station", "ui", "main_window.py"))


def test_locate_follows_a_part_that_moved_out_of_the_template_margin():
    """The default search was template matching: the part could move a little
    and turn five degrees, no more. 'auto' matches features first, which
    follows any shift or turn."""
    from vision_station.core.schema import defaults
    rng = np.random.RandomState(5)
    ref = np.full((600, 800, 3), 60, np.uint8)
    for _ in range(140):
        x, y = int(rng.randint(60, 740)), int(rng.randint(60, 540))
        c = tuple(int(v) for v in rng.randint(90, 255, 3))
        if rng.rand() < 0.5:
            cv2.rectangle(ref, (x, y), (x + int(rng.randint(8, 40)),
                                        y + int(rng.randint(8, 30))), c, -1)
        else:
            cv2.circle(ref, (x, y), int(rng.randint(4, 16)), c, -1)
    ref = cv2.GaussianBlur(ref, (0, 0), 0.8)
    M = cv2.getRotationMatrix2D((400, 300), 17.0, 1.0)
    M[0, 2] += 55
    M[1, 2] -= 38
    moved = cv2.warpAffine(ref, M, (800, 600), borderMode=cv2.BORDER_REPLICATE)
    assert defaults("align")["method"] == "auto"
    p = dict(defaults("align"), roi=[200, 150, 400, 300])
    r, ctx = _run("align", p, moved, reference=ref)
    assert r.status == "pass", r.message
    assert abs(abs(r.values["angle_deg"]) - 17.0) < 0.4, r.values
    got = cv2.transform(np.float32([[[400, 300]]]), ctx.fixture[:2])[0, 0]
    assert np.hypot(got[0] - 455, got[1] - 262) < 1.5, got
    old, _ = _run("align", dict(p, method="template"), moved, reference=ref)
    assert old.status == "fail", "the test is vacuous: template matching found it too"


def test_yolo_without_a_model_says_what_to_do(tmp_path):
    from vision_station.core.schema import defaults
    img = np.zeros((80, 80, 3), np.uint8)
    ctx = Context(reference=img, folder=str(tmp_path))
    r, _ = _run("yolo_detect", dict(defaults("yolo_detect"), roi=[0, 0, 80, 80]), img, ctx=ctx)
    assert r.status == "error" and "no trained model yet" in r.message
    assert str(tmp_path) not in r.message, "the message is a file path again"


@ocr_required
def test_faint_print_under_heavy_noise_is_still_read():
    """Grey print on grey, thin strokes, and sensor noise half as strong as
    the print itself. The detector alone finds nothing in it; the reader then
    tries the image smoothed and with its contrast raised."""
    from vision_station.tools import ocr_engine
    if not ocr_engine.onnx_available():
        pytest.skip("needs the reader's own models")
    img = _text_image(fg=120, bg=90, scale=0.7, thick=1, noise=14)
    r, _ = _run("ocr", {"roi": [0, 0, 520, 120]}, img)
    assert r.status == "pass", r.message
    assert _cer("AB12345X", r.values["text"]) < 0.15, r.values["text"]
    x, y, w, h = r.values["blocks"][0]["image_box"]
    assert 100 < x < 260 and 20 < y < 70 and 100 < w < 320, "the line is drawn in the wrong place"
    blank, _ = _run("ocr", {"roi": [0, 0, 520, 120]}, _text_image(fg=35))
    assert blank.status == "fail", "the rescue invented text on a blank image"


# ------------------------------------------------- deep trained surface check
def _brushed(seed, w=420, h=300):
    """Brushed metal: horizontal grain that is different on every part."""
    rng = np.random.RandomState(seed)
    base = rng.normal(0, 1, (h, w)).astype(np.float32)
    grain = cv2.GaussianBlur(base, (0, 0), sigmaX=14, sigmaY=0.7) * 60
    img = np.clip(128 + grain + rng.normal(0, 3, (h, w)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _deep_ready():
    from vision_station.tools import deep_surface
    return deep_surface.available() and _onnx_available()


deep_required = pytest.mark.skipif(not _deep_ready(),
                                   reason="models/backbone/resnet18_features.onnx not built")


@deep_required
def test_deep_surface_model_finds_a_scratch_on_grain_that_never_repeats(tmp_path):
    """Brushed metal has different grain on every part, so comparing pixel
    with pixel sees every good part as a defect or, loosened, sees nothing.
    The deep model learns what grain looks like, from good parts alone, and
    a scratch across it stands out."""
    from vision_station.core.schema import defaults
    from vision_station.tools import learned, make_tool
    spec = {"id": "s", "type": "trained_surface",
            "params": dict(defaults("trained_surface"), roi=[10, 10, 400, 280], method="deep")}
    ok = [(_brushed(i), "ok") for i in range(6)]
    model = str(tmp_path / "s.npz")
    rep = learned.train(make_tool(spec), ok, lambda img: Context(reference=ok[0][0]), model)
    assert rep["method"] == "deep" and rep["real_ok"] == 6 and rep["threshold"] > 0
    spec["params"]["model"] = model

    good = _brushed(101)
    r_good, _ = _run("trained_surface", spec["params"], good)
    assert r_good.status == "pass", "an unseen good part was rejected: %s" % r_good.message

    bad = _brushed(102)
    cv2.line(bad, (120, 90), (300, 210), (235, 235, 235), 2, cv2.LINE_AA)
    r_bad, _ = _run("trained_surface", spec["params"], bad)
    assert r_bad.status == "fail", "a scratch across the grain passed (score %s of %s)" % (
        r_bad.values.get("anomaly_score"), r_bad.values.get("threshold"))
    boxes = [g for g in r_bad.graphics if g.label == "defect"]
    assert boxes, "the defect was not outlined"
    (x0, y0), (x1, y1) = boxes[0].points
    assert x0 < 300 and x1 > 120 and y0 < 210 and y1 > 90, "outlined in the wrong place"
    # the threshold scale moves the verdict without retraining
    lax, _ = _run("trained_surface", dict(spec["params"], deep_sensitivity=3.0), bad)
    assert lax.status == "pass"


@deep_required
def test_deep_surface_threshold_comes_from_ng_examples_when_given(tmp_path):
    from vision_station.core.schema import defaults
    from vision_station.tools import learned, make_tool
    spec = {"id": "s", "type": "trained_surface",
            "params": dict(defaults("trained_surface"), roi=[10, 10, 400, 280], method="deep")}
    samples = [(_brushed(i), "ok") for i in range(5)]
    for i in range(3):
        bad = _brushed(50 + i)
        cv2.circle(bad, (150 + 40 * i, 150), 14, (30, 30, 30), -1)
        samples.append((bad, "ng"))
    rep = learned.train(make_tool(spec), samples, lambda img: Context(reference=samples[0][0]),
                        str(tmp_path / "s.npz"))
    assert rep["separates"], rep
    assert rep["worst_ok"] < rep["threshold"] < rep["best_ng"], rep


def test_surface_model_choice_follows_what_is_installed(monkeypatch):
    from vision_station.tools import deep_surface, learned
    monkeypatch.setattr(deep_surface, "available", lambda: False)
    assert learned.uses_deep({"method": "auto"}) is False
    assert learned.uses_deep({"method": "statistical"}) is False
    with pytest.raises(RuntimeError) as e:
        learned.uses_deep({"method": "deep"})
    assert "make_backbone.py" in str(e.value)
    monkeypatch.setattr(deep_surface, "available", lambda: True)
    assert learned.uses_deep({"method": "auto"}) is True
    assert learned.uses_deep({"method": "statistical"}) is False
    assert learned.uses_deep({}) is True
    # one or two good parts say nothing about how good parts vary
    assert learned.uses_deep({"method": "auto"}, ok_images=2) is False
    assert learned.uses_deep({"method": "auto"}, ok_images=4) is True
    assert learned.uses_deep({"method": "deep"}, ok_images=1) is True
