"""Teaching on the teach picture, the checks offered on a drawn box, Verify
text taught from its box, and barcode-gated inspection.

    python -m pytest tests/test_teach_and_barcode.py -q
"""
import glob
import os
import shutil
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt                    # noqa: E402
from PySide6.QtTest import QTest                            # noqa: E402
from PySide6.QtWidgets import QApplication                  # noqa: E402

from vision_station.core import barcode as rules            # noqa: E402
from vision_station.ui import main_window as V              # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def qb(qapp, tmp_path, monkeypatch):
    shutil.copytree(os.path.join(HERE, "projects", "demo_led_panel"),
                    str(tmp_path / "projects" / "demo_led_panel"),
                    ignore=shutil.ignore_patterns("results", "dataset"))
    ini = str(tmp_path / "settings.ini")
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


def _pump(n=10):
    for _ in range(n):
        QApplication.processEvents()


def _drop_closed_windows():
    """Windows earlier tests closed but never deleted. A change of look
    restyles every widget alive -- with hundreds left over from a whole test
    run, one switch took 90 s; in the application there is one window."""
    from PySide6.QtCore import QCoreApplication, QEvent
    for w in QApplication.topLevelWidgets():
        if not w.isVisible():
            w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


# ---------------------------------------------------------------- the rule
def test_the_barcode_rule_checks_start_and_length():
    cfg = {"prefix": "FIC", "length": 11}
    assert rules.check("FIC12345678", cfg) == (True, "")
    assert rules.check("fic12345678", cfg)[0], "capitals must not matter unless asked"
    assert not rules.check("fic12345678", dict(cfg, match_case=True))[0]
    ok, why = rules.check("ABC12345678", cfg)
    assert not ok and "start with FIC" in why
    ok, why = rules.check("FIC1234", cfg)
    assert not ok and "11 characters" in why and "has 7" in why
    assert rules.check("FIC12345678\r\n", cfg)[0], "the scanner's line end is not the code"
    assert rules.check("anything", {})[0], "an empty rule accepts any code"
    assert not rules.check("", cfg)[0]
    assert rules.describe(cfg) == "starts with FIC, 11 characters"


def test_a_code_becomes_a_safe_folder_name():
    assert rules.folder_name("FIC123/45:6") == "FIC123_45_6"
    assert rules.folder_name(" \r\n") == "no_code"


def test_the_gate_holds_one_code_for_the_next_inspection():
    g = rules.Gate({"enabled": True, "prefix": "FIC", "length": 11})
    assert g.scan("FIC1")[0] is False and g.pending is None
    assert g.scan("FIC12345678")[0] and g.pending == "FIC12345678"
    assert g.take() == "FIC12345678" and g.take() is None
    g.configure({"enabled": False})
    assert not g.enabled


# ---------------------------------------------------------------- gated runs
def test_run_view_inspects_only_after_a_valid_barcode(qb, tmp_path):
    root = str(tmp_path / "pictures")
    # the one folder: Settings > Results (a barcode folder of its own is ignored)
    qb.station.result_images.configure({"folder": root, "which": "all", "annotated": True})
    qb.apply_barcode({"enabled": True, "prefix": "FIC", "length": 11,
                      "folder": str(tmp_path / "ignored")})
    _pump()
    assert qb.barcodeBar.isVisible()
    done = sum(qb.station.counts.values())
    qb.run_once()
    assert not qb.station.running and sum(qb.station.counts.values()) == done
    assert "Scan the barcode first" in qb.barcodeBar.state.text()
    assert not qb.barcode_scanned("XYZ12345678")
    assert not qb.station.running, "a wrong code started an inspection"
    assert qb.barcode_scanned("FIC12345678")
    qb.station.wait(60)
    res = qb.station.last[_pid(qb)]
    assert res["barcode"] == "FIC12345678"
    qb.station.result_images.flush()
    from vision_station.core.result_images import day_folder
    pics = glob.glob(os.path.join(root, day_folder(), "FIC12345678", "*.jpg"))
    assert pics, "the part's picture is not in today's folder, in its barcode's folder"
    assert not os.path.exists(str(tmp_path / "ignored")), "saved in a second folder"
    proj = qb.station.projects[_pid(qb)].folder
    assert not glob.glob(os.path.join(proj, "results", "images", "**", "*.jpg"),
                         recursive=True), "a second copy kept in the job's folder"
    qb.station.logs[_pid(qb)].flush()
    rows = qb.station.logs[_pid(qb)].recent(5)
    assert rows and rows[0]["image_path"] in pics, "the run does not record where its picture went"
    assert qb.station.barcode.last[2] == os.path.join(root, day_folder(), "FIC12345678")
    heads = [qb.queue.horizontalHeaderItem(i).text() for i in range(qb.queue.columnCount())]
    assert qb.queue.item(0, heads.index("Barcode")).text() == "FIC12345678"

    # Find picture: the picture is found by its code as it is typed
    opened = []
    import vision_station.ui.finder_dialog as F
    F.open_in_explorer = lambda path: opened.append(path)
    dlg = qb.find_picture()
    for _ in range(200):                     # the folders are walked in the background
        _pump()
        if not dlg.index.scanning and len(dlg.index):
            break
    QTest.keyClicks(dlg.query, "345678")
    _pump()
    assert dlg.model.rows and dlg.model.rows[0].code == "FIC12345678"
    assert dlg.model.rows[0].path in pics
    QTest.keyClicks(dlg.query, "X")
    _pump()
    assert not dlg.model.rows, "a code that was never scanned found pictures"
    dlg.query.setText("fic12345678")
    _pump()
    dlg.table.selectRow(0)
    dlg.table.doubleClicked.emit(dlg.model.index(0, 0))
    assert opened == [dlg.model.rows[0].path], "double-click did not open its folder"
    dlg.close()


def test_a_scanner_typing_onto_the_picture_reaches_the_barcode_bar(qb, tmp_path):
    qb.apply_barcode({"enabled": True, "prefix": "FIC", "length": 11,
                      "folder": str(tmp_path / "p")})
    _pump()
    canvas = qb._displays[_pid(qb)][1].canvas
    QTest.keyClicks(canvas, "FIC00000002")
    QTest.keyClick(qb.barcodeBar.field, Qt.Key_Return)
    qb.station.wait(60)
    assert qb.station.last[_pid(qb)].get("barcode") == "FIC00000002"


def test_a_job_editors_run_is_never_held_for_a_barcode(qb):
    qb.apply_barcode({"enabled": True, "prefix": "FIC", "length": 11})
    qb.open_job_editor(_pid(qb))
    done = sum(qb.station.counts.values())
    qb.run_job(_pid(qb))
    qb.station.wait(60)
    assert sum(qb.station.counts.values()) == done + 1


def test_barcode_checking_off_runs_as_before(qb):
    qb.apply_barcode({"enabled": False})
    done = sum(qb.station.counts.values())
    qb.run_once()
    qb.station.wait(60)
    assert sum(qb.station.counts.values()) == done + 1
    assert not qb.barcodeBar.isVisible()


def test_barcode_settings_are_kept(qb):
    qb.apply_barcode({"enabled": True, "prefix": "FIC", "length": 11})
    cfg = rules.load(qb._settings())
    assert cfg["enabled"] and cfg["prefix"] == "FIC" and cfg["length"] == 11


# ---------------------------------------------------------------- the teach picture
def test_regions_are_edited_on_the_teach_picture_only(qb):
    """A board lying elsewhere on the last inspection's picture: the regions,
    in teach-picture coordinates, are not drawn over it at the wrong place."""
    pid = _pid(qb)
    p = qb.station.projects[pid]
    moved = cv2.warpAffine(p.reference, np.float32([[1, 0, 12], [0, 1, 7]]),
                           (p.reference.shape[1], p.reference.shape[0]),
                           borderMode=cv2.BORDER_REPLICATE)
    path = os.path.join(p.folder, "moved.png")
    cv2.imwrite(path, moved)
    qb.station.sources[pid].use_files([path])
    je = qb.open_job_editor(pid)
    qb.run_once()
    qb.station.wait(60)
    _pump()
    je.display.record.setCurrentText("LastRun.InputImage")
    _pump()
    assert not je.display.canvas.regions, "teach-picture regions drawn over a moved part"
    assert not je.display.noteBar.isHidden()
    je.display._set_mode("draw_rect")
    assert je.display.current_record() == "Reference"
    assert je.display.canvas.regions
    je.display._set_mode("select")


def test_drawing_a_box_offers_checks_instead_of_moving_the_selected_one(qb):
    je = qb.open_job_editor(_pid(qb))
    je.select_tool("led_row")
    before = list(je._tool("led_row")["params"]["roi"])
    je._canvas_drawn("rect", {"roi": [300, 300, 40, 30]})
    _pump()
    assert je._tool("led_row")["params"]["roi"] == before, "the selected check was moved"
    assert je._drawn_popup is not None and je._drawn_popup.isVisible()
    texts = [a.text() for a in je._drawn_popup.actions()]
    assert "Part present" in texts and any(t.startswith("Move ") for t in texts)
    je._drawn_popup.close()
    tid = je.add_tool("presence")
    assert je._tool(tid)["params"]["roi"] == [300, 300, 40, 30]
    # and "Move ... here" moves it
    je._canvas_drawn("rect", {"roi": [50, 60, 30, 20]})
    je._drawn_popup.close()
    je.move_tool_region("led_row")
    assert je._tool("led_row")["params"]["roi"] == [50, 60, 30, 20]


def test_teaching_on_a_moved_picture_moves_every_region_with_the_part():
    tools = [{"id": "a", "type": "presence", "params": {"roi": [100, 50, 40, 20]}},
             {"id": "b", "type": "presence",
              "params": {"shape": "polygon", "points": [[0, 0], [10, 0], [10, 10]],
                         "roi": [0, 0, 10, 10]}},
             {"id": "c", "type": "count_template",
              "params": {"roi": [0, 0, 50, 50], "template_roi": [5, 5, 10, 10]}}]
    V.move_regions(tools, [[1, 0, 60], [0, 1, 35]])
    assert tools[0]["params"]["roi"] == [160, 85, 40, 20]
    assert tools[1]["params"]["points"][1] == [70, 35]
    assert tools[2]["params"]["template_roi"] == [65, 40, 10, 10]


def test_same_picture_and_fixture_moved():
    a = np.zeros((40, 60, 3), np.uint8)
    assert V.same_picture(a, a.copy()) and not V.same_picture(a, a[:20])
    assert not V.fixture_moved({"tools": [{"type": "align", "tool_id": "f",
                                            "values": {"matrix": [[1, 0, 0], [0, 1, 0]]}}]})
    assert V.fixture_moved({"tools": [{"type": "align", "tool_id": "f",
                                        "values": {"matrix": [[1, 0, 12], [0, 1, 7]]}}]})


# ---------------------------------------------------------------- Verify text
def test_verify_text_treats_look_alike_characters_as_the_same():
    from vision_station.tools.text import look_alike
    assert look_alike("CO1_94_OO8") == look_alike("C01_94_008")
    assert look_alike("R15") != look_alike("R16")


def test_a_verify_text_check_expects_the_clearest_whole_line(monkeypatch):
    """Fragments cut by the box's edge and single characters are not what
    the check expects."""
    from vision_station.tools import text as T
    lines = [{"text": "O", "confidence": 0.9, "box": [2, 2, 8, 8]},
             {"text": "059400", "confidence": 0.95, "box": [0, 10, 40, 10]},     # cut by the edge
             {"text": "C01_94_008", "confidence": 0.93, "box": [10, 25, 60, 12]}]
    monkeypatch.setattr(T, "read_region", lambda tool, img, ctx: (lines, [], (0, 0), "x"))
    img = np.zeros((60, 100, 3), np.uint8)
    t = {"id": "o", "type": "ocv", "params": {"roi": [0, 0, 100, 60]}}
    text, others = V.taught_text(img, t)
    assert text == "C01_94_008" and others


def test_text_typed_by_hand_is_never_replaced_by_teaching(monkeypatch):
    class P:
        reference = np.zeros((60, 100, 3), np.uint8)
    monkeypatch.setattr(V, "taught_text", lambda img, t: ("R15", False))
    t = {"id": "o", "type": "ocv", "params": {"roi": [0, 0, 100, 60]}}
    assert V.teach_check(P(), t)[0] == "R15" and t["params"]["expect_text"] == "R15"
    monkeypatch.setattr(V, "taught_text", lambda img, t: ("R16", False))
    assert V.teach_check(P(), t)[0] == "R16", "taught text is taught again when the box moves"
    t["params"]["expect_text"] = "LOT 7"                 # typed by hand
    assert V.teach_check(P(), t)[0] is None and t["params"]["expect_text"] == "LOT 7"


# ---------------------------------------------------------------- the run view
def test_every_inspection_is_shown_whole_in_the_run_view(qb):
    d = qb._displays[_pid(qb)][1]
    d.canvas.zoom(3.0)                       # someone zoomed into the tile once
    assert not d.canvas.fit_mode
    qb.run_once()
    qb.station.wait(60)
    _pump()
    assert d.canvas.fit_mode, "the next part was shown at the old zoom, cut off"


def test_the_results_queue_keeps_a_small_preview(qb):
    qb.run_once()
    qb.station.wait(60)
    e = qb.station.history[-1]
    small = cv2.imdecode(np.frombuffer(e["jpg"], np.uint8), cv2.IMREAD_COLOR)
    assert max(small.shape[:2]) <= V.QUEUE_PREVIEW_PX
    assert tuple(e["size"]) == (qb.station.last[_pid(qb)]["image_size"][0],
                                qb.station.last[_pid(qb)]["image_size"][1])


# ---------------------------------------------------------------- round two
def test_a_deleted_check_leaves_nothing_on_the_picture(qb):
    pid = _pid(qb)
    je = qb.open_job_editor(pid)
    qb.run_once()
    qb.station.wait(60)
    _pump()
    c = je.display.canvas
    je.display.record.setCurrentText("LastRun.InputImage")
    je.delete_tools(["lot_code"])
    _pump()
    left = [it for it in c._outlines.get("lot_code", []) if it.isVisible()]
    assert not left and "lot_code" not in c.regions, "its old outline stayed until the next run"
    assert not (c._tags.get("lot_code") and c._tags["lot_code"].isVisible())


def test_adding_a_check_draws_its_area_where_it_belongs(qb):
    je = qb.open_job_editor(_pid(qb))
    n = len(je._tools())
    assert je.begin_add("presence") is None and len(je._tools()) == n
    assert je._armed == "presence" and je.display.canvas.mode == "draw_rect"
    je.display.canvas.shapeDrawn.emit("rect", {"roi": [33, 44, 55, 66]})
    _pump()
    t = je._tools()[-1]
    assert len(je._tools()) == n + 1 and t["type"] == "presence"
    assert t["params"]["roi"] == [33, 44, 55, 66], "the new check was not where it was drawn"
    assert je._armed is None


def test_escape_or_the_pointer_cancels_adding(qb):
    je = qb.open_job_editor(_pid(qb))
    n = len(je._tools())
    je.begin_add("presence")
    QTest.keyClick(je.display.canvas.viewport(), Qt.Key_Escape)
    assert je._armed is None and len(je._tools()) == n
    je.begin_add("presence")
    je.display._set_mode("select")
    assert je._armed is None and len(je._tools()) == n


def test_checks_with_no_area_are_added_at_once(qb):
    je = qb.open_job_editor(_pid(qb))
    n = len(je._tools())
    tid = je.begin_add("image_process")
    assert tid and len(je._tools()) == n + 1


def test_every_check_has_a_live_result_on_the_teach_picture(qb):
    """Without pressing Run: as the Find parts panel shows what it reads."""
    pid = _pid(qb)
    je = qb.open_job_editor(pid)
    res = qb.station.wait_preview(60)
    assert res is not None and res["source"] == "teach"
    _pump()
    ids = {t["id"] for t in je._tools()}
    assert ids <= {t["tool_id"] for t in je._preview["tools"]}
    assert not sum(qb.station.counts.values()), "a teach preview was counted as an inspection"
    # an edit is reflected without a run
    je._tool("lot_code")["params"]["expect_text"] = "NOT THIS"
    je._edited()
    deadline = __import__("time").time() + 60
    st = None
    while __import__("time").time() < deadline:
        _pump(5)
        st = {t["tool_id"]: t["status"] for t in je._preview["tools"]}.get("lot_code")
        if st == "fail":
            break
    assert st == "fail"
    je.select_tool("lot_code")
    _pump()
    assert "lot_code" in je.display.canvas._captions


def test_rows_of_the_find_panel_toggle_with_a_click_anywhere(qb):
    from vision_station.ui.part_finder_panel import PartFinderPanel
    je = qb.open_job_editor(_pid(qb))
    p = je.find_parts()
    p.mode, p.parts = "ocr", [{"box": [10, 10, 40, 20], "text": "AB12", "score": 0.9},
                              {"box": [60, 10, 40, 20], "text": "CD34", "score": 0.9}]
    p.dropped = set()
    p._fill()
    _pump()
    pos = p.table.visualItemRect(p.table.item(1, 1)).center()
    QTest.mouseClick(p.table.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
    assert 1 in p.dropped
    QTest.mouseClick(p.table.viewport(), Qt.LeftButton, Qt.NoModifier,
                     p.table.visualItemRect(p.table.item(1, 2)).center())
    assert 1 not in p.dropped
    p.close()


def test_the_find_panel_rows_are_readable_on_the_dark_theme(qb):
    from vision_station.ui import theme
    _drop_closed_windows()
    V.apply_theme(QApplication.instance(), "dark")
    try:
        je = qb.open_job_editor(_pid(qb))
        p = je.find_parts()
        p.mode, p.parts = "ocr", [{"box": [10, 10, 40, 20], "text": "AB12", "score": 0.9}]
        p.dropped = set()
        p._fill()
        col = p.table.item(0, 1).foreground().color().name().upper()
        assert col == theme.color("text").upper(), "kept text drawn in %s on the dark theme" % col
        p.close()
    finally:
        V.apply_theme(QApplication.instance(), "light")


def test_settings_offer_no_style_switch(qb):
    from vision_station.ui.settings_dialog import SettingsDialog
    d = SettingsDialog(qb, qb._settings(), qb=qb)
    assert not hasattr(d.generalPage, "look")
    d.close()


# ---------------------------------------------------------------- groups of one kind
def test_find_parts_gives_each_kind_its_threshold_and_group():
    from vision_station.core import part_finder as pf
    parts = [{"box": [10, 10, 20, 20], "type": 0}, {"box": [50, 10, 20, 20], "type": 0},
             {"box": [90, 10, 20, 20], "type": 1}]
    out = pf.presence_tools(parts, {0: "inductor"}, [], thresholds={0: 0.55})
    assert [t["params"]["min_score"] for t in out] == [0.55, 0.55, 0.65]
    assert [t["params"]["group"] for t in out][:2] == ["inductor", "inductor"]
    txt = pf.text_tools([{"box": [0, 0, 30, 10], "text": "R15", "name": "lot"}], "ocv", [],
                        min_confidence=0.7)
    assert txt[0]["params"]["group"] == "lot" and txt[0]["params"]["min_confidence"] == 0.7


def test_checks_named_alike_are_one_group():
    tools = [{"id": "inductor_01", "type": "presence", "params": {}},
             {"id": "inductor_02", "type": "presence", "params": {}},
             {"id": "inductor_03", "type": "ocv", "params": {}},
             {"id": "cap_01", "type": "presence", "params": {"group": "cap"}}]
    assert V.group_name(tools[0]) == "inductor"
    assert [t["id"] for t in V.group_members(tools[0], tools)] == ["inductor_01", "inductor_02"]


def test_the_group_bar_sets_a_whole_kind_at_once(qb):
    pid = _pid(qb)
    je = qb.open_job_editor(pid)
    from vision_station.core import part_finder as pf
    tools = je._tools()
    tools.extend(pf.presence_tools([{"box": [100 + 40 * i, 100, 30, 30], "type": 0}
                                    for i in range(4)], {0: "screw"}, [t["id"] for t in tools]))
    je._edited()
    je.select_tool("screw_02")
    _pump()
    gb = je.groupBar
    assert gb.isVisible() and "4" in gb.title.text()
    gb.thr.setValue(0.4)
    _pump()
    spec = qb.station.projects[pid].spec["tools"]
    assert [t["params"]["min_score"] for t in spec if t["id"].startswith("screw")] == [0.4] * 4
    assert abs(je.props.tool["params"]["min_score"] - 0.4) < 1e-9
    assert qb.undo()
    spec = qb.station.projects[pid].spec["tools"]
    assert [t["params"]["min_score"] for t in spec if t["id"].startswith("screw")] == [0.65] * 4, \
        "one Ctrl+Z did not take the whole group back"
    je = qb.open_job_editor(pid)
    je.select_tool("led_row")                # a check of its own: no group bar
    _pump()
    assert not je.groupBar.isVisible()


def test_the_run_view_labels_only_failures_when_there_are_many_checks(qb):
    pid = _pid(qb)
    je = qb.open_job_editor(pid)
    from vision_station.core import part_finder as pf
    tools = je._tools()
    tools.extend(pf.presence_tools([{"box": [20 + 22 * i, 20, 18, 18], "type": 0}
                                    for i in range(14)], {0: "dot"}, [t["id"] for t in tools]))
    je._edited()
    qb.run_once()
    qb.station.wait(60)
    _pump()
    c = qb._displays[pid][1].canvas
    res = qb.station.last[pid]
    passed = [t["tool_id"] for t in res["tools"] if t["status"] == "pass"]
    assert passed and not any(tid in c._captions for tid in passed)


def test_after_a_run_the_list_shows_the_run(qb):
    pid = _pid(qb)
    je = qb.open_job_editor(pid)
    qb.station.wait_preview(60)
    p = qb.station.projects[pid]
    bad = os.path.join(p.folder, "bad.png")
    qb.station.sources[pid].use_files([bad])
    qb.run_job(pid)
    qb.station.wait(60)
    je._preview = dict(je._preview, timestamp=__import__("time").time() + 5)   # came back later
    je.refresh()
    res = qb.station.last[pid]
    failed = [t["tool_id"] for t in res["tools"] if t["status"] != "pass"]
    assert failed
    row = next(je.tree.topLevelItem(i) for i in range(je.tree.topLevelItemCount())
               if je.tree.topLevelItem(i).data(0, Qt.UserRole) == failed[0])
    assert row.text(1) != "PASS", "the list showed the teach picture's OK beside the run's NG"


def test_the_checks_list_can_be_filtered(qb):
    je = qb.open_job_editor(_pid(qb))
    je.checkFilter.setText("lot")
    shown = [je.tree.topLevelItem(i).data(0, Qt.UserRole) for i in range(1, je.tree.topLevelItemCount())
             if not je.tree.topLevelItem(i).isHidden()]
    assert shown == ["lot_code"]
    je.checkFilter.clear()


# ---------------------------------------------------------------- Read text that expects nothing
def test_read_text_from_find_text_expects_what_it_read():
    from vision_station.core import part_finder as pf
    out = pf.text_tools([{"box": [0, 0, 30, 10], "text": "R15"}], "ocr", [])
    assert out[0]["params"]["expect_contains"] == "R15", \
        "a Read text check passed a scribbled marking read as 'D'"


def test_read_text_fails_a_wrong_marking_and_says_what_it_expected(monkeypatch):
    from vision_station.core import annotate
    from vision_station.core.engine import Context
    from vision_station.tools import make_tool
    from vision_station.tools import text as T
    img = np.zeros((40, 80, 3), np.uint8)
    for read, ok in (("D", False), ("R1", False), ("RI5", True), ("R15", True)):
        monkeypatch.setattr(T, "read_region", lambda tool, im, ctx, r=read: (
            [{"text": r, "confidence": 0.9, "box": [5, 5, 30, 12], "poly": [[5, 5], [35, 5],
                                                                             [35, 17], [5, 17]]}],
            [], (0, 0), "x"))
        t = make_tool({"id": "m", "type": "ocr", "params": {"roi": [0, 0, 80, 40],
                                                              "expect_contains": "R15"}})
        r = t.run(img, Context(reference=img))
        assert (r.status == "pass") == ok, (read, r.status)
        if not ok:
            assert all(g.status == "fail" for g in r.graphics if g.kind == "poly"), \
                "the line read was drawn blue on a failed check"
            assert annotate.caption(r, t.p) == ["NG  read: %s" % read, "expected: R15"]


def test_the_group_bar_makes_text_checks_expect_what_they_read(qb):
    pid = _pid(qb)
    je = qb.open_job_editor(pid)
    tools = je._tools()
    for i in (1, 2):
        tools.append({"id": "mark_%02d" % i, "type": "ocr",
                      "params": {"roi": [150, 230, 260, 46], "group": "mark"}})
    je._edited()
    qb.station.wait_preview(60)
    _pump()
    je.select_tool("mark_01")
    _pump()
    assert je.groupBar.teachBtn.isVisible()
    je.groupBar.teachBtn.click()
    exp = [t["params"].get("expect_contains") for t in je._tools() if t["id"].startswith("mark_")]
    assert exp[0] and exp[0] == exp[1], exp


# ---------------------------------------------------------------- Part present learns from its examples
def _presence_case(tmp_path):
    from vision_station.core import imio
    ref = cv2.imread(os.path.join(HERE, "projects", "demo_led_panel", "reference.png"))
    roi = [120, 120, 400, 70]                      # the LED row
    folder = str(tmp_path / "proj")
    os.makedirs(folder, exist_ok=True)

    def add(label, img, n):
        d = os.path.join(folder, "training", "p", label)
        os.makedirs(d, exist_ok=True)
        imio.imwrite(os.path.join(d, "%s_%d.png" % (label, n)), img)

    def run(img, thr=0.65):
        from vision_station.core.engine import Context
        from vision_station.tools import make_tool
        t = make_tool({"id": "p", "type": "presence", "params": {"roi": roi, "min_score": thr}})
        return t.run(img, Context(reference=ref, folder=folder))
    return ref, roi, add, run


def test_the_reference_labelled_ng_is_rejected_and_other_good_parts_are_not(tmp_path):
    ref, roi, add, run = _presence_case(tmp_path)
    assert run(ref).status == "pass"
    add("ng", ref, 1)
    r = run(ref)
    assert r.status == "fail" and "picture of NG example" in r.message
    brighter = cv2.convertScaleAbs(ref, alpha=1.06, beta=4)
    assert run(brighter).status == "pass", "a good part failed for an NG example of a good part"


def test_a_part_like_a_real_ng_example_is_rejected(tmp_path):
    ref, roi, add, run = _presence_case(tmp_path)
    x, y, w, h = roi
    missing = ref.copy()
    missing[y + 10:y + h - 10, x + 60:x + 160] = missing[y + 10:y + h - 10, x - 110:x - 10] \
        if x >= 110 else 40
    add("ng", missing, 1)
    other = cv2.convertScaleAbs(missing, alpha=1.05, beta=3)   # another board, same fault
    r = run(other, thr=0.30)                    # a threshold too low to catch it alone
    assert r.status == "fail" and "looks like NG example" in r.message, r.message
    assert run(ref, thr=0.30).status == "pass"


def test_an_ok_example_accepts_its_variant(tmp_path):
    ref, roi, add, run = _presence_case(tmp_path)
    x, y, w, h = roi
    variant = ref.copy()
    variant[y:y + h, x:x + w] = cv2.bitwise_not(ref[y:y + h, x:x + w])   # looks quite other
    assert run(variant, thr=0.9).status == "fail"
    add("ok", variant, 1)
    assert run(variant, thr=0.9).status == "pass", "an accepted variant still failed"
