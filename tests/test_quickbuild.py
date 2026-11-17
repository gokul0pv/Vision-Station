"""Tests for the VisionPro-style front end (vision_station/ui/main_window.py) and its display control.

    python -m pytest tests_vp.py -q
"""
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
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QSettings, Qt          # noqa: E402
from PySide6.QtGui import QAction                           # noqa: E402
from PySide6.QtTest import QTest                            # noqa: E402
from PySide6.QtWidgets import QApplication, QDockWidget, QToolBar  # noqa: E402

from vision_station.ui import main_window as V                                          # noqa: E402
from vision_station.ui.display import VisionDisplay                        # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def qb(qapp, tmp_path, monkeypatch):
    """QuickBuild on a private copy of the demo model, with its own settings
    file so no saved layout leaks between tests."""
    shutil.copytree(os.path.join(HERE, "projects", "demo_led_panel"),
                    str(tmp_path / "projects" / "demo_led_panel"),
                    ignore=shutil.ignore_patterns("results", "dataset"))
    ini = str(tmp_path / "layout.ini")
    monkeypatch.setattr(V.QuickBuild, "_settings",
                        lambda self: QSettings(ini, QSettings.IniFormat))
    w = V.QuickBuild(str(tmp_path / "projects"))
    w.show()
    QApplication.processEvents()
    yield w
    w.dirty.clear()
    w.close()
    QApplication.processEvents()


def _pid(qb):
    return qb.station.jobs[0]["pid"]


def _run(qb):
    qb.run_once()
    qb.station.wait(60)


# ------------------------------------------------------------- structure
def test_three_levels_open_from_one_another(qb):
    """QuickBuild -> Job Editor -> Tool Edit Control, as VisionPro has them."""
    assert qb.jobs.topLevelItemCount() == 1
    je = qb.open_job_editor(_pid(qb))
    assert je.tree.topLevelItem(0).text(0) == "Image Source"
    assert je.tree.topLevelItemCount() == 1 + len(qb.station.projects[_pid(qb)].spec["tools"])
    item = je.tree.topLevelItem(2)
    je._double_clicked(item, 0)
    tid = item.data(0, Qt.UserRole)
    assert (_pid(qb), tid) in qb._tool_editors
    assert qb.open_tool_edit(_pid(qb), tid) is qb._tool_editors[(_pid(qb), tid)], \
        "a second window opened for the same tool"


def test_run_once_fills_status_queue_and_records(qb):
    _run(qb)
    res = qb.station.last[_pid(qb)]
    assert res["verdict"] in ("PASS", "FAIL")
    assert qb.queue.rowCount() == 1
    assert qb.jobs.topLevelItem(0).text(1) in ("Accept", "Reject")
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    recs = [te.display.record.itemText(i) for i in range(te.display.record.count())]
    assert recs[:2] == ["Current.InputImage", "LastRun.InputImage"]
    assert te.sTime.text().startswith("Tool ") and "Total" in te.sTime.text()


def test_tool_edit_control_offers_the_tools_diagnostic_records(qb):
    """VisionPro's record selector: a caliper's projection, a histogram, the
    region as the tool saw it -- how you see why a tool missed."""
    je = qb.open_job_editor(_pid(qb))
    cal = je.add_tool("caliper")
    hist = je.add_tool("histogram")
    tools = {t["id"]: t for t in qb.station.projects[_pid(qb)].spec["tools"]}
    tools[cal]["params"].update(roi=[120, 140, 400, 24], channel="blue", fixture="none")
    _run(qb)
    te = qb.open_tool_edit(_pid(qb), cal)
    recs = [te.display.record.itemText(i) for i in range(te.display.record.count())]
    assert "LastRun.Projection" in recs and "LastRun.RegionImage" in recs, recs
    te.display.record.setCurrentText("LastRun.Projection")
    assert te.display.canvas._img.shape[1] >= 600
    assert not te.display.canvas.regions, "the tool's outline was drawn over the plot"
    te.display.record.setCurrentText("LastRun.InputImage")
    assert cal in te.display.canvas.regions
    th = qb.open_tool_edit(_pid(qb), hist)
    assert "LastRun.Histogram" in [th.display.record.itemText(i)
                                   for i in range(th.display.record.count())]


def test_a_transformed_output_is_shown_without_the_input_region(qb):
    """An unwrapped or resampled image has its own coordinates: the tool's
    region, drawn in the camera image's, does not belong on it."""
    je = qb.open_job_editor(_pid(qb))
    tid = je.add_tool("affine_transform")
    tools = {t["id"]: t for t in qb.station.projects[_pid(qb)].spec["tools"]}
    tools[tid]["params"].update(roi=[150, 230, 260, 46], scale=2.0, fixture="none")
    reader = je.add_tool("histogram")
    tools = {t["id"]: t for t in qb.station.projects[_pid(qb)].spec["tools"]}
    tools[reader]["params"].update(capture=tid, roi=[10, 10, 100, 40])
    _run(qb)
    te = qb.open_tool_edit(_pid(qb), tid)
    te.display.record.setCurrentText("LastRun.OutputImage")
    assert not te.display.canvas.regions, "the input region was drawn on the output"
    tr = qb.open_tool_edit(_pid(qb), reader)
    img = tr.display.canvas._img
    assert img.shape[:2] == (92, 520), "the reader did not show the image it reads: %s" % (img.shape,)


def test_image_processing_tool_offers_its_output_record(qb):
    je = qb.open_job_editor(_pid(qb))
    tid = je.add_tool("image_process")
    _run(qb)
    te = qb.open_tool_edit(_pid(qb), tid)
    recs = [te.display.record.itemText(i) for i in range(te.display.record.count())]
    assert "LastRun.OutputImage" in recs


def test_job_editor_shows_last_run_but_keeps_the_operators_choice(qb):
    _run(qb)
    je = qb.open_job_editor(_pid(qb))
    je.tree.setCurrentItem(je.tree.topLevelItem(2))
    assert je.display.current_record() == "LastRun.InputImage"
    je.display.record.setCurrentText("Current.InputImage")
    _run(qb)
    assert je.display.current_record() == "Current.InputImage", \
        "a new run overrode the record the operator picked"
    je.tree.setCurrentItem(je.tree.topLevelItem(3))
    assert je.display.current_record() == "LastRun.InputImage"


def test_job_editor_shows_terminals(qb):
    """Terminals belong to the engineering view: Advanced mode."""
    qb.set_advanced(True, save=False)
    _run(qb)
    je = qb.open_job_editor(_pid(qb))
    tool_items = [je.tree.topLevelItem(i) for i in range(1, je.tree.topLevelItemCount())]
    terms = [it.child(k).data(0, V.TERM_ROLE) for it in tool_items
             for k in range(it.childCount())]
    assert ("image" in [t[3] for t in terms if t[0] == "in" and t[2] == "InputImage"])
    assert any(t[0] == "in" and t[2] == "Fixture" for t in terms), "no fixture input shown"
    assert any(t[0] == "out" and t[3] == "value" for t in terms), "no outputs after a run"


# ------------------------------------------------------------- terminals
def _term_item(je, kind, tool_id, name):
    for i in range(je.tree.topLevelItemCount()):
        top = je.tree.topLevelItem(i)
        for k in range(top.childCount()):
            t = top.child(k).data(0, V.TERM_ROLE)
            if t and t[0] == kind and t[1] == tool_id and t[2] == name:
                return top.child(k)
    return None


def _drop(je, src_item, dst_item):
    """A real drag-and-drop through Qt's event types: the mime data the tree
    produces for the dragged row, delivered as move and drop events over the
    target row."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QDragMoveEvent, QDropEvent
    md = je.tree.mimeData([src_item])
    pos = je.tree.visualItemRect(dst_item).center()
    mv = QDragMoveEvent(pos, Qt.LinkAction, md, Qt.LeftButton, Qt.NoModifier)
    je.tree.dragMoveEvent(mv)
    accepted = mv.isAccepted()
    if accepted:
        de = QDropEvent(QPointF(pos), Qt.LinkAction, md, Qt.LeftButton, Qt.NoModifier)
        je.tree.dropEvent(de)
    return accepted


def _geometry_job(qb):
    qb.set_advanced(True, save=False)        # the terminal rows are an Advanced-mode thing
    je = qb.open_job_editor(_pid(qb))
    je.resize(1200, 800)
    a = je.add_tool("find_circle")
    b = je.add_tool("find_circle")
    d = je.add_tool("distance")
    for tid in (a, b, d):
        top = [je.tree.topLevelItem(i) for i in range(je.tree.topLevelItemCount())
               if je.tree.topLevelItem(i).data(0, Qt.UserRole) == tid][0]
        top.setExpanded(True)
    QApplication.processEvents()
    return je, a, b, d


def test_dragging_an_output_onto_an_input_links_them(qb):
    je, a, b, d = _geometry_job(qb)
    ok = _drop(je, _term_item(je, "out", a, "Feature"), _term_item(je, "in", d, "first"))
    assert ok, "a legal drop was refused"
    dist = [t for t in qb.station.projects[_pid(qb)].spec["tools"] if t["id"] == d][0]
    assert dist["params"]["first"] == a
    assert _pid(qb) in qb.dirty


def test_an_illegal_drop_is_refused_before_it_lands(qb):
    """Type mismatch and a source that runs later are both rejected while the
    drag hovers, so the drop never happens."""
    je, a, b, d = _geometry_job(qb)
    for i in range(je.tree.topLevelItemCount()):
        je.tree.topLevelItem(i).setExpanded(True)
    QApplication.processEvents()
    fx = next(t["id"] for t in qb.station.projects[_pid(qb)].spec["tools"] if t["type"] == "align")
    assert not _drop(je, _term_item(je, "out", fx, "Pose"), _term_item(je, "in", d, "first")), \
        "a fixture was accepted where a feature is needed"
    assert not _drop(je, _term_item(je, "out", b, "Feature"), _term_item(je, "in", a, "Fixture")), \
        "a later tool's output was accepted"
    dist = [t for t in qb.station.projects[_pid(qb)].spec["tools"] if t["id"] == d][0]
    assert not dist["params"].get("first")


def test_rows_are_never_moved_by_a_drop(qb):
    """Tree drag-and-drop moves rows by default; here a drop links terminals
    and must leave the recipe's order alone."""
    je, a, b, d = _geometry_job(qb)
    order = [t["id"] for t in qb.station.projects[_pid(qb)].spec["tools"]]
    rows = je.tree.topLevelItemCount()
    _drop(je, _term_item(je, "out", a, "Feature"), _term_item(je, "in", d, "first"))
    assert [t["id"] for t in qb.station.projects[_pid(qb)].spec["tools"]] == order
    assert je.tree.topLevelItemCount() == rows


def test_linked_distance_measures_after_a_run(qb):
    """Two LEDs on the demo board sit 100 px apart (182,152) and (282,152)."""
    je, a, b, d = _geometry_job(qb)
    tools = {t["id"]: t for t in qb.station.projects[_pid(qb)].spec["tools"]}
    tools[a]["params"].update(roi=[154, 124, 56, 56], shape="ellipse", fixture="none")
    tools[b]["params"].update(roi=[254, 124, 56, 56], shape="ellipse", fixture="none")
    assert je.link_terminal(d, "first", a, "Feature") is None
    assert je.link_terminal(d, "second", b, "Feature") is None
    _run(qb)
    res = {t["tool_id"]: t for t in qb.station.last[_pid(qb)]["tools"]}
    assert res[d]["status"] == "pass", res[d]["message"]
    assert abs(res[d]["values"]["value_px"] - 100.0) < 1.5, res[d]["values"]


def test_member_browser_adds_an_input_driven_by_a_result(qb):
    je, a, b, d = _geometry_job(qb)
    _run(qb)                                     # results give value outputs
    dlg = je.member_browser(b)
    keys = [dlg.params.item(i).data(Qt.UserRole) for i in range(dlg.params.count())]
    assert "min_contrast" in keys and "roi" not in keys
    dlg.params.setCurrentRow(keys.index("min_contrast"))
    srcs = [dlg.sources.itemText(i) for i in range(dlg.sources.count())]
    assert any(x.startswith(a + ".") for x in srcs), srcs
    assert not any(x.startswith(b + ".") or x.startswith(d + ".") for x in srcs), \
        "the browser offered itself or a later tool"
    dlg.sources.setCurrentIndex(next(i for i, x in enumerate(srcs) if x.startswith(a + ".")))
    assert dlg.add()
    spec = [t for t in qb.station.projects[_pid(qb)].spec["tools"] if t["id"] == b][0]
    assert list(spec["params"]["links"]) == ["min_contrast"]
    dlg.close()


# ------------------------------------------------------------- editing
def test_electric_mode_reruns_after_a_change(qb):
    """Electric: edits re-run the job, debounced to one run per burst."""
    _run(qb)
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    te.aElectric.setChecked(True)
    n0 = sum(qb.station.counts.values())
    for v in (0.31, 0.32, 0.33):                     # a burst of edits
        te.props.tool["params"]["min_confidence"] = v
        te._settings_changed()
    t0 = time.time()
    while time.time() - t0 < 10 and sum(qb.station.counts.values()) == n0:
        QApplication.processEvents()
        time.sleep(0.01)
    qb.station.wait(30)
    assert sum(qb.station.counts.values()) == n0 + 1, "the burst did not run exactly once"


def test_edits_take_effect_on_the_next_run_without_saving(qb):
    """Tools copy their region and settings when they are made, and the engine
    built them only when a job loaded -- so a region dragged in a tool editor
    was inspected at its OLD position, and a tool added from the Toolbox never
    ran at all, until the job was saved and reopened."""
    _run(qb)
    spec = [t for t in qb.station.projects[_pid(qb)].spec["tools"] if t["id"] == "lot_code"][0]
    spec["params"]["roi"] = [0, 0, 40, 30]                 # moved onto blank board
    _run(qb)
    res = {t["tool_id"]: t for t in qb.station.last[_pid(qb)]["tools"]}
    g = [x for x in res["lot_code"]["graphics"] if x.get("kind") == "rect"][0]
    xs = [p[0] for p in g["points"]]
    assert max(xs) <= 45, "the run used the old region: %s" % g["points"]
    je = qb.open_job_editor(_pid(qb))
    tid = je.add_tool("histogram")
    _run(qb)
    assert tid in {t["tool_id"] for t in qb.station.last[_pid(qb)]["tools"]}, \
        "a tool added from the Toolbox did not run"


def test_regions_edit_only_on_the_current_image(qb):
    """On LastRun records a region is shown with its results, not dragged:
    moving it there would edit the recipe against an old image."""
    _run(qb)
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    te.display.record.setCurrentText("LastRun.InputImage")
    it = te.display.canvas.regions["lot_code"]
    assert not (it.flags() & it.GraphicsItemFlag.ItemIsMovable)
    te.display.record.setCurrentText("Current.InputImage")
    it = te.display.canvas.regions["lot_code"]
    assert it.flags() & it.GraphicsItemFlag.ItemIsMovable
    c = te.display.canvas
    c.resetTransform()
    r = list(it.roi())
    a = c.mapFromScene(QPointF(r[0] + r[2] / 2, r[1] + r[3] / 2))
    QTest.mousePress(c.viewport(), Qt.LeftButton, Qt.NoModifier, a)
    for i in range(1, 6):
        QTest.mouseMove(c.viewport(), c.mapFromScene(QPointF(r[0] + r[2] / 2 + 5 * i,
                                                             r[1] + r[3] / 2 + 3 * i)))
    QTest.mouseRelease(c.viewport(), Qt.LeftButton, Qt.NoModifier,
                       c.mapFromScene(QPointF(r[0] + r[2] / 2 + 25, r[1] + r[3] / 2 + 15)))
    spec = [t for t in qb.station.projects[_pid(qb)].spec["tools"] if t["id"] == "lot_code"][0]
    assert spec["params"]["roi"] != r and _pid(qb) in qb.dirty


def test_rename_keeps_links_and_save_persists(qb):
    je = qb.open_job_editor(_pid(qb))
    circ = je.add_tool("find_circle")
    je.add_tool("distance")
    tools = qb.station.projects[_pid(qb)].spec["tools"]
    dist = [t for t in tools if t["type"] == "distance"][0]
    dist["params"]["first"] = circ
    je.rename_tool(circ, "hole_x")
    assert dist["params"]["first"] == "hole_x", "a rename broke the link to the tool"
    qb.save_job(_pid(qb))
    assert _pid(qb) not in qb.dirty
    reloaded = qb.station.store.reload(_pid(qb))
    ids = [t["id"] for t in reloaded.spec["tools"]]
    assert "hole_x" in ids and circ not in ids


def test_saving_does_not_strand_open_tool_editors(qb):
    """Saving reloads the job, which replaces its recipe objects; an open tool
    editor must follow the new ones rather than edit a copy nobody reads."""
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    qb.mark_dirty(_pid(qb))
    qb.save_job(_pid(qb))
    te.props.tool["params"]["min_confidence"] = 0.44
    te._settings_changed()
    spec = [t for t in qb.station.projects[_pid(qb)].spec["tools"] if t["id"] == "lot_code"][0]
    assert spec["params"]["min_confidence"] == 0.44, "the editor edited a stale recipe"


def test_reset_restores_defaults_but_keeps_the_region(qb):
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    t = te.spec()
    roi = list(t["params"]["roi"])
    t["params"]["min_confidence"] = 0.99
    te.reset_tool()
    from vision_station.core.schema import defaults
    assert te.spec()["params"]["min_confidence"] == defaults("ocv")["min_confidence"]
    assert te.spec()["params"]["roi"] == roi


def test_tool_open_and_save_round_trip(qb, tmp_path):
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    te.spec()["params"]["min_confidence"] = 0.77
    path = te.save_tool(str(tmp_path / "t.vtool.json"))
    te.spec()["params"]["min_confidence"] = 0.1
    te.open_tool(path)
    assert te.spec()["params"]["min_confidence"] == 0.77


# ------------------------------------------------------------- running
def test_stale_results_are_ignored(qb):
    """A result from an older run arriving after a newer run started must not
    be shown over the newer one."""
    st = qb.station
    st._seq = 5
    st.last.pop(_pid(qb), None)
    st._on_done(4, _pid(qb), {"verdict": "FAIL", "tools": [], "total_ms": 1.0})
    assert _pid(qb) not in st.last, "a stale result was accepted"


def test_continuous_run_starts_and_stops_cleanly(qb):
    qb.aRunCont.setChecked(True)
    qb.toggle_continuous()
    t0 = time.time()
    while time.time() - t0 < 20 and sum(qb.station.counts.values()) < 2:
        QApplication.processEvents()
        time.sleep(0.01)
    qb.stop()
    qb.station.wait(60)
    n = sum(qb.station.counts.values())
    assert n >= 2, "continuous mode did not keep running"
    for _ in range(30):
        QApplication.processEvents()
        time.sleep(0.01)
    assert sum(qb.station.counts.values()) == n, "it kept running after Stop"


def test_failure_queue_holds_only_rejects_with_compressed_images(qb):
    _run(qb)
    bad = cv2.imread(os.path.join(HERE, "projects", "demo_led_panel", "bad.png"))
    qb.station.sources[_pid(qb)].use_files([os.path.join(HERE, "projects",
                                                         "demo_led_panel", "bad.png")])
    _run(qb)
    assert len(qb.station.history) == 2
    assert [e["verdict"] for e in qb.station.failures] == ["FAIL"]
    e = qb.station.failures[0]
    assert isinstance(e["jpg"], bytes) and len(e["jpg"]) < bad.nbytes / 3
    qb._set_queue_mode(True)
    assert qb.queue.rowCount() == 1


# ------------------------------------------------------------- training
def _tab_names(te):
    return [te.props.tabs.tabText(i) for i in range(te.props.tabs.count())]


def test_train_tab_survives_reloading_the_tool(qb):
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    assert "Train" in _tab_names(te)
    te.props.load(te.spec())
    _run(qb)
    qb.mark_dirty(_pid(qb))
    qb.save_job(_pid(qb))
    assert _tab_names(te).count("Train") == 1, _tab_names(te)


def test_tune_threshold_from_ok_and_ng_images(qb):
    """A presence check's threshold set from examples: good boards, and boards
    with the checked feature painted out."""
    pid = _pid(qb)
    je = qb.open_job_editor(pid)
    tid = je.add_tool("presence")
    tools = qb.station.projects[pid].spec["tools"]
    spec = next(t for t in tools if t["id"] == tid)
    spec["params"].update(roi=[160, 130, 44, 44], search_pad=6)   # the first LED
    te = qb.open_tool_edit(pid, tid)
    assert te.bTrain.text() == "Tune threshold" and te.bTrain.isEnabled()
    good = cv2.imread(os.path.join(HERE, "projects", "demo_led_panel", "good.png"))
    roi = next(t for t in tools if t["id"] == tid)["params"]["roi"]
    x, y, w, h = roi
    bad = good.copy()
    bad[y:y + h, x:x + w] = cv2.GaussianBlur(np.full((h, w, 3), good[y + h + 5, x].tolist(),
                                                     np.uint8), (5, 5), 0)
    src = qb.station.sources[pid]
    for img, lab in ((good, "ok"), (good, "ok"), (bad, "ng")):
        src.current = img
        te.add_training(lab)
    assert te.trainLists["ok"].count() == 2 and te.trainLists["ng"].count() == 1
    rep = te.train()
    assert rep["separates"], te.trainReport.text()
    assert rep["highest_ng"] < rep["new"] < rep["lowest_ok"]
    fresh = [t for t in qb.station.projects[pid].spec["tools"] if t["id"] == tid][0]
    assert fresh["params"]["min_score"] == rep["new"], "the tuned threshold did not reach the job"
    ev = te.test()
    assert ev["all_correct"] and len(te._marks) == 3
    assert "OK 2/2 passed" in te.trainReport.text()


def test_training_saves_unsaved_edits_first(qb):
    """Training reads and writes the job on disk; unsaved edits would be
    overwritten if they were not saved first."""
    pid = _pid(qb)
    te = qb.open_tool_edit(pid, "lot_code")
    te.spec()["params"]["min_confidence"] = 0.66
    te._settings_changed()
    assert pid in qb.dirty
    src = qb.station.sources[pid]
    src.current = cv2.imread(os.path.join(HERE, "projects", "demo_led_panel", "good.png"))
    te.add_training("ok")
    te.test()
    on_disk = qb.station.store.reload(pid)
    spec = [t for t in on_disk.spec["tools"] if t["id"] == "lot_code"][0]
    assert spec["params"]["min_confidence"] == 0.66, "an unsaved edit was lost"


def test_testing_an_empty_or_ok_only_set_is_not_shown_as_success(qb):
    """An empty set used to report 'OK 0/0 passed, NG 0/0 rejected' in green.
    And a set with no NG images passes trivially: nothing in it shows the tool
    can reject a bad part."""
    from vision_station.core import job as jobmod
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    te.test()
    assert "No OK or NG images" in te.trainReport.text()
    qb.station.sources[_pid(qb)].current = cv2.imread(
        os.path.join(HERE, "projects", "demo_led_panel", "good.png"))
    te.add_training("ok")
    ev = te.test()
    text, level = jobmod.evaluation_summary(ev)
    assert level == "warn" and "no NG images" in text


def test_trained_surface_trains_from_the_tool_editor(qb):
    pid = _pid(qb)
    je = qb.open_job_editor(pid)
    tid = je.add_tool("trained_surface")
    te = qb.open_tool_edit(pid, tid)
    assert te.bTrain.text() == "Train"
    good = cv2.imread(os.path.join(HERE, "projects", "demo_led_panel", "good.png"))
    src = qb.station.sources[pid]
    src.current = good
    te.add_training("ok")
    rep = te.train()
    assert rep is not None and rep["real_ok"] == 1, te.trainReport.text()
    model = [t for t in qb.station.projects[pid].spec["tools"] if t["id"] == tid][0]["params"]["model"]
    assert os.path.exists(os.path.join(qb.station.projects[pid].folder, model))
    _run(qb)
    res = {t["tool_id"]: t for t in qb.station.last[pid]["tools"]}
    assert res[tid]["status"] == "pass", res[tid]["message"]


def test_relabel_and_remove_training_images(qb):
    pid = _pid(qb)
    te = qb.open_tool_edit(pid, "lot_code")
    qb.station.sources[pid].current = cv2.imread(os.path.join(HERE, "projects",
                                                              "demo_led_panel", "good.png"))
    path = te.add_training("ok")
    te.side().relabel(path, "ng")
    te.refresh_train()
    assert te.trainLists["ok"].count() == 0 and te.trainLists["ng"].count() == 1


# ------------------------------------------------------------- jobs
def _new_app(qb, name="Chassis"):
    return qb.new_application(name, os.path.join(HERE, "projects", "demo_led_panel", "good.png"))


def test_new_application_starts_with_a_job_and_a_fixture(qb):
    app = _new_app(qb)
    assert qb.station.app_id == app and qb.station.is_job_app()
    assert [j["name"] for j in qb.station.jobs] == ["Job 1"]
    tools = qb.station.projects[qb.station.jobs[0]["pid"]].spec["tools"]
    assert [t["type"] for t in tools] == ["align"]
    _run(qb)
    assert qb.station.last[qb.station.jobs[0]["pid"]]["verdict"] == "PASS"


def test_add_rename_and_remove_jobs(qb):
    _new_app(qb)
    pid = qb.add_job("Top face", os.path.join(HERE, "projects", "demo_led_panel", "good.png"))
    assert [j["name"] for j in qb.station.jobs] == ["Job 1", "Top face"]
    assert len(qb._displays) == 2
    qb.rename_job(pid, "Top side")
    assert qb.job_name(pid) == "Top side"
    from vision_station.core import job as jobmod
    assert [m["name"] for m in jobmod.Job(qb.station.app_folder()).spec["sides"]] == \
        ["Job 1", "Top side"], "the rename was not saved"
    assert qb.remove_job(pid, confirm=False)
    assert [j["name"] for j in qb.station.jobs] == ["Job 1"]
    removed = os.path.join(qb.station.app_folder(), "removed")
    assert os.path.isdir(removed) and os.listdir(removed), "the removed job was deleted, not kept"


def test_add_job_is_refused_for_a_single_model(qb, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    assert not qb.station.is_job_app()
    assert qb.add_job("x", os.path.join(HERE, "projects", "demo_led_panel", "good.png")) is None


def test_posted_items_become_queue_columns_and_persist(qb):
    _run(qb)
    dlg = qb.configure_posted()
    leaves = dlg.leaves()
    want = [lf for lf in leaves if lf.data(0, Qt.UserRole).endswith("body_width.length_px")
            or lf.data(0, Qt.UserRole).endswith("led_row.count")]
    assert len(want) == 2, [lf.data(0, Qt.UserRole) for lf in leaves][:10]
    for lf in want:
        lf.setCheckState(0, Qt.Checked)
    items = dlg.apply()
    _run(qb)
    heads = [qb.queue.horizontalHeaderItem(i).text() for i in range(qb.queue.columnCount())]
    tips = [qb.queue.horizontalHeaderItem(i).toolTip() for i in range(qb.queue.columnCount())]
    assert set(items) <= set(tips), "posted items missing from the queue"
    assert "led_row.count" in heads, "the header shows the raw id: %s" % heads
    col = tips.index(next(i for i in items if i.endswith("led_row.count")))
    assert qb.queue.item(0, col).text() == "4"
    qb.open_application(qb.station.app_id)            # reopen: posted items kept
    assert qb.station.posted == items


# ------------------------------------------------------------- every tool, through the UI
def _all_tool_types():
    from vision_station.tools import REGISTRY
    return sorted(REGISTRY)


@pytest.fixture
def slot_errors(monkeypatch):
    """Exceptions raised inside Qt slots do not fail a test by themselves --
    Qt prints them and carries on. Collect them so they do."""
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))
    return errors


@pytest.mark.parametrize("ttype", _all_tool_types())
def test_every_tool_type_works_through_the_interface(qb, ttype, slot_errors):
    """Added from the Toolbox, its editor opened, every setting changed, run,
    every record and tab shown, saved, reopened, deleted -- with no exception
    anywhere, including inside Qt's event handlers."""
    from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox
    pid = _pid(qb)
    qb.open_job_editor(pid)
    tid = qb._toolbox_add(ttype)
    assert tid, "the Toolbox did not add %s" % ttype
    _run(qb)
    te = qb.open_tool_edit(pid, tid)
    assert te is not None
    # every setting widget gets a change, as a person would make
    for w in te.props.findChildren(QSpinBox) + te.props.findChildren(QDoubleSpinBox):
        if w.isEnabled() and w.isVisible():
            w.setValue(min(w.maximum(), w.value() + w.singleStep()))
    for w in te.props.findChildren(QComboBox):
        if w.isEnabled() and w.count() > 1:
            w.setCurrentIndex((w.currentIndex() + 1) % w.count())
    for w in te.props.findChildren(QCheckBox):
        if w.isEnabled():
            w.setChecked(not w.isChecked())
            w.setChecked(not w.isChecked())
    QApplication.processEvents()
    te.run()
    qb.station.wait(60)
    QApplication.processEvents()
    for i in range(te.display.record.count()):
        te.display.record.setCurrentIndex(i)
        QApplication.processEvents()
    for i in range(te.props.tabs.count()):
        te.props.tabs.setCurrentIndex(i)
        QApplication.processEvents()
    te.results_grid()
    qb.save_job(pid)
    reloaded = qb.station.store.reload(pid)
    assert tid in [t["id"] for t in reloaded.spec["tools"]], "the tool did not survive saving"
    je = qb.open_job_editor(pid)
    je.delete_tool(tid)
    QApplication.processEvents()
    assert tid not in [t["id"] for t in qb.station.projects[pid].spec["tools"]]
    assert not slot_errors, "an exception inside the interface: %s" % [
        "%s: %s" % (e[0].__name__, e[1]) for e in slot_errors]


# ------------------------------------------------------------- parallel jobs
def _summary(station):
    return {pid: (r["verdict"], [(t["tool_id"], t["status"],
                                  json.dumps(t["values"], sort_keys=True, default=str))
                                 for t in r["tools"]]) for pid, r in station.last.items()}


def test_parallel_jobs_give_the_same_results_as_one_after_another(qb, monkeypatch):
    """Four workers forced, so this runs even on a single-core machine."""
    _new_app(qb)
    good = os.path.join(HERE, "projects", "demo_led_panel", "good.png")
    for name in ("B", "C", "D"):
        qb.add_job(name, good)
    qb.station.set_parallel(False)
    _run(qb)
    seq = _summary(qb.station)
    finished = []
    qb.station.runFinished.connect(lambda: finished.append(1))
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    qb.station.set_parallel(True)
    assert qb.station._worker.workers == 4
    qb.station.last.clear()
    _run(qb)
    assert _summary(qb.station) == seq, "parallel results differ from one-after-another"
    assert len(qb.station.last) == 4 and finished == [1], "a job was lost or finished twice"


def test_the_parallel_choice_belongs_to_the_pc(qb):
    qb.aParallel.setChecked(True)
    assert qb._settings().value("parallel_jobs") in (True, "true")
    qb.aParallel.setChecked(False)
    assert qb.station._worker.workers == 1


# ------------------------------------------------------------- undo / redo
def _spec_of(qb, tid):
    return next((t for t in qb.station.projects[_pid(qb)].spec["tools"] if t["id"] == tid), None)


def test_undo_and_redo_a_region_move(qb):
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    before = list(te.spec()["params"]["roi"])
    te.spec()["params"]["roi"] = [before[0] + 30, before[1] + 10] + before[2:]
    te._changed()
    assert qb.aUndo.text() == "Undo move lot_code region", qb.aUndo.text()
    assert qb.undo() == "move lot_code region"
    assert _spec_of(qb, "lot_code")["params"]["roi"] == before
    assert te.spec()["params"]["roi"] == before, "the open editor kept the undone region"
    assert qb.redo()
    assert _spec_of(qb, "lot_code")["params"]["roi"][0] == before[0] + 30


def test_undo_an_added_tool_closes_its_editor_and_redo_brings_it_back(qb):
    qb.open_job_editor(_pid(qb))
    tid = qb._toolbox_add("histogram")
    qb.open_tool_edit(_pid(qb), tid)
    assert qb.undo() == "add %s" % tid
    QApplication.processEvents()
    assert _spec_of(qb, tid) is None
    assert (_pid(qb), tid) not in qb._tool_editors, "an editor for a tool that is gone stayed open"
    qb.redo()
    assert _spec_of(qb, tid) is not None


def test_deleting_a_tool_closes_its_editor(qb):
    qb.open_tool_edit(_pid(qb), "lot_code")
    qb.open_job_editor(_pid(qb)).delete_tool("lot_code")
    QApplication.processEvents()
    assert (_pid(qb), "lot_code") not in qb._tool_editors, "an editor outlived its tool"


def test_undo_a_deleted_tool_restores_it_with_its_settings(qb):
    kept = json.loads(json.dumps(_spec_of(qb, "lot_code")))
    je = qb.open_job_editor(_pid(qb))
    je.delete_tool("lot_code")
    assert _spec_of(qb, "lot_code") is None
    assert qb.undo() == "delete lot_code"
    assert _spec_of(qb, "lot_code") == kept


def test_a_settings_sweep_is_one_undo_step(qb):
    """A spinbox sweep fires per tick; one gesture must be one undo step."""
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    start = te.spec()["params"]["min_confidence"]
    for v in (0.51, 0.52, 0.53, 0.54, 0.55):
        te.spec()["params"]["min_confidence"] = v
        te._changed()
    qb.undo()
    assert _spec_of(qb, "lot_code")["params"]["min_confidence"] == start
    assert not qb._hist[_pid(qb)].can_undo(), "the sweep left extra undo steps"


def test_editing_after_undo_changes_the_restored_recipe(qb):
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    te.spec()["params"]["min_confidence"] = 0.9
    te._changed()
    qb.undo()
    qb.refresh_all() if hasattr(qb, "refresh_all") else None
    te.refresh()
    te.spec()["params"]["min_confidence"] = 0.66
    te._changed()
    assert _spec_of(qb, "lot_code")["params"]["min_confidence"] == 0.66, \
        "the editor changed a recipe object that undo had replaced"
    qb.save_job(_pid(qb))
    spec = next(t for t in qb.station.store.reload(_pid(qb)).spec["tools"] if t["id"] == "lot_code")
    assert spec["params"]["min_confidence"] == 0.66


def test_ctrl_z_undoes_the_job_in_front(qb):
    from PySide6.QtTest import QTest
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    before = te.spec()["params"]["min_confidence"]
    te.spec()["params"]["min_confidence"] = 0.95
    te._changed()
    qb.activateWindow()
    QTest.keySequence(qb, "Ctrl+Z")
    QApplication.processEvents()
    assert _spec_of(qb, "lot_code")["params"]["min_confidence"] == before
    QTest.keySequence(qb, "Ctrl+Y")
    QApplication.processEvents()
    assert _spec_of(qb, "lot_code")["params"]["min_confidence"] == 0.95


def test_undo_is_locked_for_operators(qb):
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    te.spec()["params"]["min_confidence"] = 0.95
    te._changed()
    qb.save_job(_pid(qb))
    qb.enter_operator_mode("op")
    assert not qb.aUndo.isEnabled()
    qb.exit_operator_mode()


# ------------------------------------------------------------- results log
def test_every_inspection_is_logged_and_survives_reopening(qb, tmp_path):
    """The on-screen queue holds 32 runs and is gone when the application
    closes; the log keeps every inspection, with images of rejects."""
    pid = _pid(qb)
    good = os.path.join(HERE, "projects", "demo_led_panel", "good.png")
    bad = os.path.join(HERE, "projects", "demo_led_panel", "bad.png")
    qb.station.sources[pid].use_files([good, bad, good])
    for _ in range(3):
        _run(qb)
    folder = qb.station.projects[pid].folder
    qb.open_application(qb.station.app_id)            # closes and reopens the logs
    from vision_station.core.results import ResultStore
    lg = ResultStore(folder, pid)
    sm = lg.summary()
    assert (sm["total"], sm["passed"]) == (3, 2), sm
    runs = lg.recent()
    rejects = [r for r in runs if r["verdict"] != "PASS"]
    assert len(rejects) == 1 and rejects[0]["image_path"], "the reject's image was not archived"
    assert os.path.exists(os.path.join(folder, rejects[0]["image_path"]))
    assert not [r for r in runs if r["verdict"] == "PASS" and r["image_path"]], \
        "passing parts were archived under the default policy"
    lg.close()


def test_production_history_shows_yield_pareto_and_exports(qb, tmp_path):
    pid = _pid(qb)
    good = os.path.join(HERE, "projects", "demo_led_panel", "good.png")
    bad = os.path.join(HERE, "projects", "demo_led_panel", "bad.png")
    qb.station.sources[pid].use_files([good, bad, bad, good])
    for _ in range(4):
        _run(qb)
    h = qb.show_history()
    assert _sub_of(qb, h) is not None, "history opened outside the workspace"
    rows, par = h.refresh()
    assert rows[0][1:4] == (4, 2, "50.0%"), rows
    assert par and par[0][2] == 2, "the most frequent failing check should appear twice"
    paths = h.export_csv(str(tmp_path / "csv"))
    assert len(paths) == 1
    with open(paths[0], encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    assert len(lines) >= 5, "the CSV should hold a header and four runs"


# ------------------------------------------------------------- the line (TCP/IP)
def _talk(qb, commands, replies_per_command, timeout=60):
    """A real TCP client on its own thread, as a PLC would connect, while the
    application keeps processing events. replies_per_command: how many lines
    to read after each command (None = read up to a DONE line)."""
    import socket
    import threading
    got, err = [], []

    def client():
        try:
            s = socket.create_connection(("127.0.0.1", qb.line.port), timeout=timeout)
            f = s.makefile("rwb")
            for cmd, n in zip(commands, replies_per_command):
                f.write((cmd + "\r\n").encode("ascii"))
                f.flush()
                while True:
                    line = f.readline().decode("ascii").strip()
                    got.append(line)
                    if (n is None and line.startswith("DONE")) or \
                            (n is not None and sum(1 for _ in got) >= n):
                        break
            s.close()
        except Exception as e:
            err.append(repr(e))
    th = threading.Thread(target=client)
    th.start()
    t0 = time.time()
    while th.is_alive() and time.time() - t0 < timeout:
        QApplication.processEvents()
        time.sleep(0.005)
    th.join(1)
    assert not err, err
    return got


def test_a_plc_triggers_an_inspection_and_reads_the_result(qb):
    c = qb.communication(enabled=True, port=0)
    assert qb.line is not None and qb.line.server.isListening(), qb.sComms.text()
    got = _talk(qb, ["PING", "STATUS", "TRIGGER"], [1, 2, None])
    assert got[:2] == ["PONG", "READY"], got
    res = [g for g in got if g.startswith("RESULT")]
    assert len(res) == 1 and res[0].split()[1] == "demo_led_panel", got
    assert res[0].split()[2] in ("PASS", "FAIL")
    assert got[-1].startswith("DONE ") and got[-1].split()[2] == "1"
    assert sum(qb.station.counts.values()) == 1, "the inspection was not run and logged"
    assert c["enabled"] and qb.comms_settings()["enabled"], "the settings were not saved"


def test_a_plc_reads_a_measurement_from_the_last_run(qb):
    qb.communication(enabled=True, port=0)
    got = _talk(qb, ["TRIGGER", "GET demo_led_panel.led_row.count", "GET nope.x.y",
                     "GET demo_led_panel.led_row.nothing"], [None, 1, 1, 1])
    vals = got[-3:]
    assert vals[0] == "VALUE 4", got
    assert vals[1].startswith("ERROR") and vals[2].startswith("ERROR"), vals


def test_a_trigger_during_a_run_is_answered_busy_not_queued(qb):
    """A controller must know its trigger was not acted on."""
    qb.communication(enabled=True, port=0)
    qb.station.running = True                 # an inspection in progress
    got = _talk(qb, ["TRIGGER"], [1])
    qb.station.running = False
    assert got == ["BUSY"], got
    assert sum(qb.station.counts.values()) == 0


def test_an_unacquirable_job_is_reported_to_the_line(qb, tmp_path):
    qb.communication(enabled=True, port=0)
    qb.station.sources[_pid(qb)].use_files([str(tmp_path / "missing.png")])
    got = _talk(qb, ["TRIGGER"], [None])
    assert got[0].startswith("RESULT demo_led_panel ERROR"), got
    assert got[-1] == "DONE INCOMPLETE 1", got


def test_unknown_commands_and_jobs_get_errors(qb):
    qb.communication(enabled=True, port=0)
    got = _talk(qb, ["FROB", "trigger nosuchjob", "version"], [1, 1, 1])
    assert got[0].startswith("ERROR unknown command")
    assert got[1] == "ERROR no job nosuchjob"
    assert got[2] == "VISIONSTATION 1"


def test_broadcast_sends_every_run_to_listening_clients(qb):
    """With broadcast on, a controller that only listens receives results of
    runs it did not trigger -- here, a run started from QuickBuild."""
    import socket
    import threading
    qb.communication(enabled=True, port=0, broadcast=True)
    got = []

    def listener():
        s = socket.create_connection(("127.0.0.1", qb.line.port), timeout=30)
        f = s.makefile("rb")
        while True:
            line = f.readline().decode("ascii").strip()
            got.append(line)
            if line.startswith("DONE"):
                break
        s.close()
    th = threading.Thread(target=listener)
    th.start()
    t0 = time.time()
    while len(qb.line.clients) < 1 and time.time() - t0 < 10:
        QApplication.processEvents()
        time.sleep(0.005)
    _run(qb)                                   # started here, not by the listener
    while th.is_alive() and time.time() - t0 < 30:
        QApplication.processEvents()
        time.sleep(0.005)
    assert got and got[0].startswith("RESULT") and got[-1].startswith("DONE"), got


# ------------------------------------------------------------- shutdown
def test_closing_while_engines_warm_up_does_not_crash(tmp_path):
    """Closing within seconds of opening an application aborted the process:
    the engine warm-up ran on the inspection thread, outlasted the shutdown
    wait, and Qt aborts when a running QThread is destroyed. Run as a real
    process, because the symptom is the process dying."""
    import subprocess
    code = (
        "import sys, time, shutil, os\n"
        "sys.path.insert(0, %r)\n"
        "os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
        "from vision_station.tools import text as T\n"
        "def slow(lang='en'):\n"
        "    time.sleep(6); return T._ENGINE\n"
        "T._engine = slow\n"
        "from PySide6.QtWidgets import QApplication\n"
        "from vision_station.ui import main_window as V\n"
        "shutil.copytree(%r, %r)\n"
        "app = QApplication(sys.argv)\n"
        "qb = V.QuickBuild(%r); qb.show(); app.processEvents()\n"
        "qb.close(); app.processEvents(); print('closed')\n") % (
        HERE, os.path.join(HERE, "projects", "demo_led_panel"),
        str(tmp_path / "p" / "demo_led_panel"), str(tmp_path / "p"))
    t0 = time.time()
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=90, env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)})
    assert out.returncode == 0, "the process died: %s" % (out.stdout + out.stderr)[-600:]
    assert "Destroyed while" not in out.stderr
    assert time.time() - t0 < 5.5, "closing waited for the warm-up"


def test_two_threads_asking_for_the_engine_start_it_once(monkeypatch):
    import threading
    from vision_station.tools import text as T
    started = []

    def fake_start(lang):
        started.append(lang)
        time.sleep(0.3)
        T._ENGINE.update(name="stub", obj=object())
        return T._ENGINE
    monkeypatch.setitem(T._ENGINE, "obj", None)
    monkeypatch.setitem(T._ENGINE, "name", None)
    monkeypatch.setattr(T, "_start_engine", fake_start)
    ths = [threading.Thread(target=T._engine) for _ in range(4)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    assert len(started) == 1, "the engine was started %d times" % len(started)


# ------------------------------------------------------------- operator mode
def test_operator_mode_runs_but_does_not_edit(qb):
    """VisionPro separates building from running; without this anyone at the
    line could double-click a tool and move its region."""
    qb.open_tool_edit(_pid(qb), "lot_code")
    assert qb.enter_operator_mode("Asha")
    QApplication.processEvents()
    assert not qb._tool_editors and not qb._job_editors, "editors stayed open"
    assert qb.open_job_editor(_pid(qb)) is None
    assert qb.open_tool_edit(_pid(qb), "lot_code") is None
    assert qb._toolbox_add("histogram") is None
    assert not qb.toolboxDock.isVisible()
    locked = {a.text() for a in qb._locked}
    for name in ("Export Job File (Python)...", "Add Job...", "Save All", "Open Application..."):
        assert name in locked, "%s is still available to operators" % name
    assert qb.aRunOnce.isEnabled() and qb.aRunCont.isEnabled(), "operators cannot run"
    _run(qb)
    assert sum(qb.station.counts.values()) == 1
    assert qb.exit_operator_mode()                  # no password set: free to leave
    assert all(a.isEnabled() for a in qb._locked)
    assert qb.open_job_editor(_pid(qb)) is not None


def test_leaving_operator_mode_needs_the_supervisor_password(qb, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    qb.set_supervisor_password("line-7")
    stored = qb._settings().value("supervisor_password")
    assert "line-7" not in stored, "the password was stored in the clear"
    qb.enter_operator_mode("Ravi")
    assert not qb.exit_operator_mode("wrong")
    assert qb.operator_mode
    assert qb.exit_operator_mode("line-7")
    assert not qb.operator_mode
    qb.set_supervisor_password("")


def test_the_operators_name_is_logged_with_each_inspection(qb):
    qb.enter_operator_mode("Meena")
    _run(qb)
    lg = qb.station.logs[_pid(qb)]
    lg.flush()
    import sqlite3
    c = sqlite3.connect(lg.path)
    try:
        ops = [r[0] for r in c.execute("SELECT operator FROM runs")]
    finally:
        c.close()
    assert ops == ["Meena"], ops
    qb.exit_operator_mode()


# ------------------------------------------------------------- workspace
def _sub_of(qb, widget):
    from PySide6.QtWidgets import QMdiSubWindow
    sub = widget.parent()
    return sub if isinstance(sub, QMdiSubWindow) and sub in qb.mdi.subWindowList() else None


def test_editors_open_as_windows_inside_the_workspace(qb):
    je = qb.open_job_editor(_pid(qb))
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    fd = te.floating_display()
    for w in (je, te, fd):
        assert _sub_of(qb, w) is not None, "%s opened outside the workspace" % w
    assert qb.runSub in qb.mdi.subWindowList()
    assert qb.open_job_editor(_pid(qb)) is je and \
        len([s for s in qb.mdi.subWindowList() if s.widget() is je]) == 1


def test_closing_a_window_frame_or_the_window_leaves_nothing_behind(qb):
    """Closing the sub-window's frame must close the editor (and forget it);
    an editor closing itself must take its frame with it."""
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    _sub_of(qb, te).close()
    QApplication.processEvents()
    assert (_pid(qb), "lot_code") not in qb._tool_editors, "the editor outlived its frame"
    je = qb.open_job_editor(_pid(qb))
    frames = len(qb.mdi.subWindowList())
    je.close()
    for _ in range(5):
        QApplication.processEvents()
    assert len(qb.mdi.subWindowList()) == frames - 1, "an empty frame was left behind"


def test_f5_follows_the_active_window(qb):
    """Inside one main window an editor's F5 and QuickBuild's F5 would be one
    window's shortcut -- ambiguous, so neither would fire. QuickBuild owns the
    key and runs the active job, or the whole application from the run view."""
    from PySide6.QtTest import QTest
    _new_app(qb)
    pid2 = qb.add_job("Second", os.path.join(HERE, "projects", "demo_led_panel", "good.png"))
    qb.activateWindow()
    je = qb.open_job_editor(pid2)
    QApplication.processEvents()
    n0 = sum(qb.station.counts.values())
    QTest.keySequence(qb, "F5")
    QApplication.processEvents()
    qb.station.wait(60)
    assert sum(qb.station.counts.values()) == n0 + 1, "F5 in a job editor did not run just that job"
    assert list(qb.station.last) == [pid2]
    qb.mdi.setActiveSubWindow(qb.runSub)
    QTest.keySequence(qb, "F5")
    QApplication.processEvents()
    qb.station.wait(60)
    assert sum(qb.station.counts.values()) == n0 + 3, "F5 in the run view did not run every job"


def test_the_workspace_has_no_ambiguous_shortcut(qb):
    from PySide6.QtGui import QAction
    qb.open_job_editor(_pid(qb))
    qb.open_tool_edit(_pid(qb), "lot_code")
    seen = {}
    for a in qb.findChildren(QAction):
        host = a.parent().window() if a.parent() is not None else None
        if host is not qb or a.shortcutContext() == Qt.WidgetShortcut:
            continue
        for ks in a.shortcuts():
            k = ks.toString()
            if k:
                assert k not in seen, "%s owned by both %r and %r" % (k, seen[k], a.text())
                seen[k] = a.text()
    assert {"F5", "F6", "Ctrl+S"} <= set(seen)


def test_fixture_space_reads_the_same_point_on_the_part_wherever_it_sits(qb):
    """VisionPro's coordinate spaces: in a fixture's space the pointer reads
    where it is on the part, so a feature reads the same however the part moved."""
    pid = _pid(qb)
    # the model's reference itself -- good.png is a different photograph,
    # 9 px right and 5 px up of it, which a test once mistook for an error
    good = qb.station.projects[pid].reference
    h, w = good.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.0)
    M[0, 2] += 18
    M[1, 2] -= 11
    moved = cv2.warpAffine(good, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    from vision_station.ui.main_window import spaces_for
    res = qb.station.projects[pid].inspect(moved, keep_images=True)
    spaces = spaces_for(res)
    assert "locate_part" in spaces, spaces
    d = VisionDisplay()
    d.set_spaces(spaces)
    assert [d.sSpace.itemText(i) for i in range(d.sSpace.count())] == ["#", "locate_part"]
    led = np.array([170.0, 150.0, 1.0])            # the red LED's centre on the reference
    x, y = M @ led                                  # where it is in the moved image
    fx, fy = d.to_space(x, y, "locate_part")
    assert abs(fx - 170.0) < 1.5 and abs(fy - 150.0) < 1.5, (fx, fy)
    assert d.to_space(x, y, "#") == (float(x), float(y))
    d.close()


def test_graphics_tab_switches_what_a_tool_draws(qb):
    _run(qb)
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    assert "Graphics" in _tab_names(te)
    assert all(cb.isChecked() for cb in te.gfxBoxes.values())
    te.gfxBoxes["region"].setChecked(False)
    assert te.spec()["params"]["graphics"]["region"] is False
    _run(qb)
    res = {t["tool_id"]: t for t in qb.station.last[_pid(qb)]["tools"]}
    assert not any(g.get("role") == "region" for g in res["lot_code"]["graphics"])
    te.gfxBoxes["region"].setChecked(True)
    assert "graphics" not in te.spec()["params"], "the default should store nothing"


def test_editors_remember_how_their_panes_were_divided(qb):
    je = qb.open_job_editor(_pid(qb))
    je.resize(1200, 700)
    QApplication.processEvents()
    je.split.setSizes([620, 560])
    want = je.split.sizes()
    je.close()
    for _ in range(5):
        QApplication.processEvents()
    je2 = qb.open_job_editor(_pid(qb))
    je2.resize(1200, 700)
    QApplication.processEvents()
    got = je2.split.sizes()
    assert abs(got[0] - want[0]) <= 40, (want, got)


def test_the_one_toolbox_adds_to_the_active_job(qb):
    """One Toolbox for the application: it adds to whichever job is in front,
    and with several jobs and none open it asks rather than guessing."""
    _new_app(qb)
    pid2 = qb.add_job("Second", os.path.join(HERE, "projects", "demo_led_panel", "good.png"))
    pid1 = qb.station.jobs[0]["pid"]
    assert qb._toolbox_add("histogram") is None, "added to a job nobody chose"
    qb.open_job_editor(pid2)
    tid = qb._toolbox_add("histogram")
    assert tid in [t["id"] for t in qb.station.projects[pid2].spec["tools"]]
    assert tid not in [t["id"] for t in qb.station.projects[pid1].spec["tools"]]
    qb.open_job_editor(pid1)
    tid1 = qb._toolbox_add("find_line")
    assert tid1 in [t["id"] for t in qb.station.projects[pid1].spec["tools"]]


def test_queue_display_shows_the_latest_run_until_a_row_is_chosen(qb):
    _run(qb)
    assert qb.queueDisplay.canvas._img is not None, "the queue display stayed blank"


def test_ctrl_s_saves_the_active_job(qb):
    from PySide6.QtTest import QTest
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    te.spec()["params"]["min_confidence"] = 0.57
    te._settings_changed()
    qb.activateWindow()
    QTest.keySequence(qb, "Ctrl+S")
    QApplication.processEvents()
    assert not qb.dirty, "Ctrl+S did not save"


# ------------------------------------------------------------- packages
def test_export_and_import_a_package_from_quickbuild(qb, tmp_path, monkeypatch):
    """Export saves unsaved edits first -- the package is built from disk --
    and importing it back opens it as a new application."""
    monkeypatch.setenv("VISION_STATION_CACHE", str(tmp_path / "cache"))
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    te.spec()["params"]["min_confidence"] = 0.61
    te._settings_changed()
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Save))
    path = str(tmp_path / "out.vsp")
    m = qb.export_package(path)
    assert m and os.path.exists(path) and not qb.dirty
    from vision_station.core import vsruntime
    app = vsruntime.load(path)
    spec = next(t for t in app._projects[app.jobs[0][0]][1].spec["tools"] if t["id"] == "lot_code")
    assert spec["params"]["min_confidence"] == 0.61, "the unsaved edit did not reach the package"
    new_id = qb.import_package(path)
    assert qb.station.app_id == new_id != "demo_led_panel"
    _run(qb)
    assert qb.station.last[qb.station.jobs[0]["pid"]]["verdict"] in ("PASS", "FAIL")


# ------------------------------------------------------------- image source
class _StubCamera:
    def __init__(self, frames):
        self.frames = list(frames)
        self.stopped = False

    def get_frame(self):
        return self.frames.pop(0) if self.frames else None

    def stop(self):
        self.stopped = True


def test_a_dropped_camera_is_an_error_not_a_pass(qb):
    """With no frame the source used to fall back to the reference image -- a
    dropped camera would inspect the stored golden image and pass."""
    errors = []
    qb.station.runError.connect(lambda pid, m: errors.append(m))
    qb.station.sources[_pid(qb)].use_camera(_StubCamera([]))
    _run(qb)
    assert _pid(qb) not in qb.station.last, "an image-less job was inspected"
    assert sum(qb.station.counts.values()) == 0, "an image-less job was counted"
    assert errors and "no image" in errors[0]
    title, _d = qb._displays[_pid(qb)]
    assert "no image" in title.text()


def test_an_unreadable_image_file_is_an_error(qb, tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    errors = []
    qb.station.runError.connect(lambda pid, m: errors.append(m))
    qb.station.sources[_pid(qb)].use_files([str(bad)])
    _run(qb)
    assert _pid(qb) not in qb.station.last and errors and "broken.png" in errors[0]


def test_camera_frames_are_inspected(qb):
    good = cv2.imread(os.path.join(HERE, "projects", "demo_led_panel", "good.png"))
    qb.station.sources[_pid(qb)].use_camera(_StubCamera([good]))
    _run(qb)
    assert qb.station.last[_pid(qb)]["verdict"] == "PASS"


def test_image_source_dialog_sets_files_folder_and_camera(qb, tmp_path, monkeypatch):
    je = qb.open_job_editor(_pid(qb))
    folder = tmp_path / "imgs"
    folder.mkdir()
    for n in ("a.png", "b.png"):
        shutil.copy(os.path.join(HERE, "projects", "demo_led_panel", "good.png"), str(folder / n))
    dlg = je.image_source()
    dlg._pick_folder(str(folder))
    assert dlg.apply()
    src = qb.station.sources[_pid(qb)]
    assert src.mode == "folder" and src.folder == str(folder)
    assert qb.station.projects[_pid(qb)].spec["image_source"]["folder"] == str(folder)
    assert _pid(qb) in qb.dirty, "the taught source was not marked as a change to save"

    cam = _StubCamera([])
    dlg = je.image_source()
    monkeypatch.setattr(dlg, "open_camera", lambda index: cam)
    dlg.rCam.setChecked(True)
    assert dlg.apply()
    assert src.mode == "camera" and src.camera is cam

    dlg = je.image_source()
    dlg.rRef.setChecked(True)
    dlg.apply()
    assert src.mode == "reference" and cam.stopped, "switching away left the camera running"


def test_taught_image_source_survives_closing_and_reopening(qb, tmp_path):
    """Taught once, with the job: reopening the application uses the same
    images without setting them up again."""
    folder = tmp_path / "parts"
    folder.mkdir()
    shutil.copy(os.path.join(HERE, "projects", "demo_led_panel", "bad.png"), str(folder / "p1.png"))
    je = qb.open_job_editor(_pid(qb))
    dlg = je.image_source()
    dlg._pick_folder(str(folder))
    dlg.apply()
    qb.save_job(_pid(qb))
    qb.open_application(qb.station.app_id)
    src = qb.station.sources[_pid(qb)]
    assert src.mode == "folder" and src.folder == str(folder)
    _run(qb)
    assert qb.station.last[_pid(qb)]["verdict"] == "FAIL", "the taught image was not used"


def test_relative_taught_paths_are_relative_to_the_application(qb, tmp_path, monkeypatch):
    """The same rule as an exported job file: a relative path is relative to
    the application's folder, not to wherever QuickBuild was started."""
    pid = _pid(qb)
    app_folder = qb.station.app_folder()
    os.makedirs(os.path.join(app_folder, "parts"), exist_ok=True)
    shutil.copy(os.path.join(HERE, "projects", "demo_led_panel", "bad.png"),
                os.path.join(app_folder, "parts", "p.png"))
    qb.station.projects[pid].spec["image_source"] = {"mode": "files", "paths": ["parts/p.png"]}
    qb.mark_dirty(pid)
    qb.save_job(pid)
    qb.open_application(qb.station.app_id)
    monkeypatch.chdir(str(tmp_path))                 # started from somewhere else
    _run(qb)
    assert qb.station.last.get(pid, {}).get("verdict") == "FAIL", \
        "the relative path was not found from the application's folder"


def test_a_folder_source_picks_up_images_added_later(qb, tmp_path):
    folder = tmp_path / "live"
    folder.mkdir()
    good = os.path.join(HERE, "projects", "demo_led_panel", "good.png")
    shutil.copy(good, str(folder / "001.png"))
    src = qb.station.sources[_pid(qb)]
    src.use_folder(str(folder))
    assert src.grab() is not None and src.current_path.endswith("001.png")
    shutil.copy(good, str(folder / "002.png"))
    src.grab()
    assert src.current_path.endswith("002.png"), "a newly written image was not picked up"


def test_opening_another_application_stops_cameras(qb):
    cam = _StubCamera([])
    qb.station.sources[_pid(qb)].use_camera(cam)
    qb.open_application(qb.station.app_id)
    assert cam.stopped


# ------------------------------------------------------------- display control
def test_fit_is_sticky_across_resizes_until_a_manual_zoom(qapp):
    """Fit used to be applied once; resizing the window left the image at its
    old size. It is now a mode that manual zoom turns off."""
    d = VisionDisplay()
    d.resize(800, 600)
    d.show()
    img = np.full((400, 600, 3), 120, np.uint8)
    d.set_records({"img": (img, [])})
    QApplication.processEvents()
    d.canvas.fit()
    s1 = d.canvas.transform().m11()
    d.resize(1200, 900)
    QApplication.processEvents()
    s2 = d.canvas.transform().m11()
    assert s2 > s1 * 1.2, "the image did not refit when the window grew"
    d.canvas.zoom(1.5)
    s3 = d.canvas.transform().m11()
    d.resize(900, 700)
    QApplication.processEvents()
    assert abs(d.canvas.transform().m11() - s3) < 1e-6, "a manual zoom was undone by a resize"
    d.close()


def test_pixel_readout_is_exact_and_floors_outside_the_image(qapp):
    """int() truncates towards zero, so a pointer just left of the image
    (x = -0.3) reported pixel 0 instead of nothing."""
    d = VisionDisplay()
    img = np.zeros((10, 10, 3), np.uint8)
    img[3, 7] = (10, 20, 30)
    d.set_records({"img": (img, [])})
    assert d.canvas.pixel_at(QPointF(7.9, 3.2)) == (7, 3, "R 30  G 20  B 10")
    assert d.canvas.pixel_at(QPointF(-0.3, 2.0)) is None
    assert d.canvas.pixel_at(QPointF(2.0, -0.4)) is None
    grey = np.zeros((6, 6), np.uint8)
    grey[1, 2] = 77
    d.canvas.set_image(grey, keep_view=False)
    assert d.canvas.pixel_at(QPointF(2.5, 1.5)) == (2, 1, "77"), "a mono image broke the readout"
    d.close()


def test_record_selection_survives_a_new_run(qapp):
    d = VisionDisplay()
    a = np.zeros((20, 20, 3), np.uint8)
    d.set_records({"Current.InputImage": (a, []), "LastRun.InputImage": (a, [])})
    d.record.setCurrentText("LastRun.InputImage")
    d.set_records({"Current.InputImage": (a, []), "LastRun.InputImage": (a, [])})
    assert d.current_record() == "LastRun.InputImage"
    d.close()


# ------------------------------------------------------------- engineering
def _window_actions(w):
    return [a for a in w.findChildren(QAction) if a.window() is w or a.parent() is w
            or (a.parent() is not None and a.parent().window() is w)]


def test_no_window_has_an_ambiguous_shortcut(qb):
    """Qt fires neither action of an ambiguous shortcut in the same window.
    Checked per window: QuickBuild, a Job Editor and a Tool Edit Control."""
    _run(qb)
    je = qb.open_job_editor(_pid(qb))
    te = qb.open_tool_edit(_pid(qb), "lot_code")
    for w in (qb, je, te):
        seen = {}
        for a in w.findChildren(QAction):
            host = a.parent().window() if a.parent() is not None else None
            if host is not w:
                continue
            for ks in a.shortcuts():
                k = ks.toString()
                if k:
                    assert k not in seen, "%s: %s owned by both %r and %r" % (
                        w.objectName(), k, seen[k], a.text())
                    seen[k] = a.text()


def test_f5_in_a_job_editor_runs_that_job(qb):
    je = qb.open_job_editor(_pid(qb))
    je.activateWindow()
    QApplication.processEvents()
    n0 = sum(qb.station.counts.values())
    QTest.keySequence(je, "F5")
    QApplication.processEvents()
    qb.station.wait(60)
    assert sum(qb.station.counts.values()) == n0 + 1, "F5 did nothing in the Job Editor"


def test_every_dock_and_toolbar_has_a_unique_name(qb):
    """saveState identifies docks and toolbars by objectName; an unnamed or
    duplicated one silently fails to restore."""
    je = qb.open_job_editor(_pid(qb))
    for w in (qb, je):
        # the window's own docks and toolbars -- the ones saveState tracks; a
        # display control's toolbar belongs to the display, not the layout
        names = [x.objectName() for x in w.findChildren(QDockWidget) + w.findChildren(QToolBar)
                 if x.parent() is w]
        assert all(names), "an unnamed dock or toolbar in %s" % w.objectName()
        assert len(names) == len(set(names)), "duplicate names in %s: %s" % (w.objectName(), names)


def test_layout_round_trips_through_saved_state(qb):
    qb.queueDock.hide()
    state = qb.saveState(V.LAYOUT_VERSION)
    qb.reset_layout()
    assert qb.queueDock.isVisible()
    assert qb.restoreState(state, V.LAYOUT_VERSION)
    assert not qb.queueDock.isVisible()
    assert not qb.restoreState(state, V.LAYOUT_VERSION + 1), \
        "a layout from another version was accepted"


def test_closing_with_unsaved_edits_asks_first(qb, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: (asked.append(1), QMessageBox.Cancel)[1]))
    qb.mark_dirty(_pid(qb))
    assert not qb.open_application(qb.station.app_id), "unsaved edits were discarded"
    assert asked


def test_live_camera_shows_the_picture_sets_exposure_and_lets_go(qb):
    """Live Camera, for aiming and focusing: frames from the shared camera,
    exposure that reaches the camera and is remembered for the jobs, guides
    over the image, and the camera released when the window closes."""
    import importlib
    import time as _t
    import fake_pylon as F
    from vision_station.ui import main_window as vp_app
    scene = np.full((60, 80), 50, np.float32)
    F.install([F.Device("usb3", scene=scene)])
    for m in [k for k in sys.modules if k.startswith("camera")]:
        del sys.modules[m]
    from vision_station.hardware import camera as cam_mod
    importlib.reload(cam_mod)
    w = qb.show_live_camera()
    try:
        t0 = _t.time()
        while w._frame is None and _t.time() - t0 < 5:
            QApplication.processEvents()
            _t.sleep(0.02)
        assert w._frame is not None, w.status.text()
        assert "frames/s" in w.status.text()
        assert w.gainLabel.text() == "Gain (dB)"
        assert len(w._overlay_items) == 3           # centre cross: 2 lines + ring
        w.overlay.setCurrentIndex(3)
        assert len(w._overlay_items) == 18
        assert qb.show_live_camera() is w, "a second window was opened"

        w.exposure.setValue(20.0)
        node = w.cam.thread._pylon_cam.cam.ExposureTime
        t0 = _t.time()
        while node.GetValue() != 20000.0 and _t.time() - t0 < 3:
            QApplication.processEvents()
            _t.sleep(0.02)
        assert node.GetValue() == 20000.0
        from vision_station.ui.live_view import saved_settings
        assert saved_settings(qb._settings(), 0).exposure_ms == 20.0
        assert saved_settings(vp_app.camera_settings_store(), 0).exposure_ms == 20.0
    finally:
        w.close()
        QApplication.processEvents()
    assert vp_app._shared_cameras == {}
