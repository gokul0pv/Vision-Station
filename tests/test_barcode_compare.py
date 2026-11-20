"""Compare barcode with barcode scanned: a Read text or Read code check
whose reading must be the scanned code, else the part is NG - Barcode not
good - and its other checks are skipped. Only with the barcode check on in
Settings; checking it moves the check to just after the fixture.

    python -m pytest tests/test_barcode_compare.py -q
"""
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

from vision_station.core import barcode as B                 # noqa: E402


# ---------------------------------------------------------------- matching
def test_a_reading_carries_the_scanned_code():
    assert B.code_matches("FIC12345678", ["S/N FIC12345678"])
    assert B.code_matches("fic12345678", ["FIC 1234 5678"]), "spaces or capitals counted"
    assert B.code_matches("FIC1234567O", ["FIC12345670"]), "O read for 0"
    assert not B.code_matches("FIC12345678", ["FIC12345679"])
    assert not B.code_matches("FIC12345678", [])
    assert not B.code_matches("", ["anything"])
    # a decoded QR must be the code itself, letter for letter
    assert B.code_matches("FIC1", ["fic1"], look_alike=False, exact=True)
    assert not B.code_matches("FIC1", ["FIC12"], look_alike=False, exact=True)
    assert not B.code_matches("FICO", ["FIC0"], look_alike=False, exact=True)


def _tool(tid, typ, compare=False):
    return {"id": tid, "type": typ, "params": {"compare_barcode": compare}}


def test_checks_comparing_go_just_after_the_fixture():
    tools = [_tool("a", "presence"), _tool("fix", "align"), _tool("b", "presence"),
             _tool("serial", "ocr", True), _tool("c", "ocv"), _tool("qr", "code_read", True)]
    moved = B.place_compare_checks(tools)
    assert [t["id"] for t in tools] == ["a", "fix", "serial", "qr", "b", "c"]
    assert moved == ["serial", "qr"]
    assert B.place_compare_checks(tools) == [], "moved when already in place"
    tools = [_tool("a", "presence"), _tool("serial", "ocr", True)]
    B.place_compare_checks(tools)
    assert [t["id"] for t in tools] == ["serial", "a"], "no fixture: first"
    plain = [_tool("a", "presence"), _tool("fix", "align"), _tool("t", "ocr")]
    assert B.place_compare_checks(plain) == [] and [t["id"] for t in plain] == ["a", "fix", "t"]


# ---------------------------------------------------------------- the checks
def _ctx(code=None):
    from vision_station.core.engine import Context
    ctx = Context()
    if code:
        ctx.share["barcode"] = code
    return ctx


def _ocr(monkeypatch, read, compare=True):
    from vision_station.tools import text as T
    tool = T.Ocr("serial", {"roi": [0, 0, 10, 10], "compare_barcode": compare})
    lines = [{"text": t, "confidence": 0.99, "poly": None, "box": [0, 0, 1, 1]} for t in read]
    monkeypatch.setattr(T, "read_region", lambda *_a: (lines, [], (0, 0), "test"))
    return tool


def test_read_text_compares_with_the_scanned_code(monkeypatch):
    img = np.zeros((10, 10, 3), np.uint8)
    tool = _ocr(monkeypatch, ["S/N FIC12345678"])
    r = tool.run(img, _ctx("FIC12345678"))
    assert r.status == "pass" and r.values["barcode_match"] is True
    r = tool.run(img, _ctx("FIC99999999"))
    assert r.status == "fail" and r.values["barcode_match"] is False
    assert r.message.startswith("Barcode not good") and "FIC99999999" in r.message
    # no code scanned (the barcode check off in Settings): reads as before
    r = tool.run(img, _ctx(None))
    assert r.status == "pass" and "barcode_match" not in r.values
    # the box not checked: never compared
    r = _ocr(monkeypatch, ["S/N FIC12345678"], compare=False).run(img, _ctx("OTHER"))
    assert r.status == "pass" and "barcode_match" not in r.values


def _qr(text, size=300):
    q = cv2.QRCodeEncoder.create().encode(text)
    q = cv2.resize(q, (size, size), interpolation=cv2.INTER_NEAREST)
    q = cv2.copyMakeBorder(q, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(q, cv2.COLOR_GRAY2BGR)


def test_read_code_compares_exactly():
    from vision_station.tools import text as T
    img = _qr("FIC12345678")
    tool = T.CodeRead("qr", {"compare_barcode": True})
    r = tool.run(img, _ctx("FIC12345678"))
    assert r.status == "pass" and r.values["barcode_match"], r.message
    r = tool.run(img, _ctx("FIC1234567"))
    assert r.status == "fail" and "Barcode not good" in r.message, "a shorter code matched"


# ---------------------------------------------------------------- the inspection
def _project(tmp_path, compare_critical=True):
    import json
    from vision_station.core.engine import Project
    d = tmp_path / "job"
    d.mkdir()
    img = _qr("FIC12345678")
    cv2.imwrite(str(d / "reference.png"), img)
    spec = {"name": "j", "reference_image": "reference.png", "tools": [
        {"id": "qr", "type": "code_read",
         "params": {"compare_barcode": True, "critical": compare_critical}},
        {"id": "after", "type": "presence",
         "params": {"roi": [60, 60, 100, 100], "min_score": 0.5}}]}
    (d / "project.json").write_text(json.dumps(spec), encoding="utf-8")
    return Project(str(d)), img


def test_a_wrong_part_is_ng_and_nothing_else_is_inspected(tmp_path):
    p, img = _project(tmp_path)
    good = p.inspect(img, barcode="FIC12345678")
    assert good["verdict"] == "PASS" and not good["barcode_ng"]
    assert [t["tool_id"] for t in good["tools"]] == ["qr", "after"]
    bad = p.inspect(img, barcode="FIC00000000")
    assert bad["verdict"] == "FAIL" and bad["barcode_ng"].startswith("Barcode not good")
    assert [t["tool_id"] for t in bad["tools"]] == ["qr"], "checks ran after Barcode not good"
    assert "Barcode not good" in bad["aborted"]
    plain = p.inspect(img)                                  # no code: as before
    assert plain["verdict"] == "PASS" and not plain["barcode_ng"]


def test_not_critical_still_rejects_a_wrong_part(tmp_path):
    p, img = _project(tmp_path, compare_critical=False)
    assert p.inspect(img, barcode="WRONG")["verdict"] == "FAIL"


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
    w.close()


def _pump(qapp, s=0.05):
    t0 = time.time()
    while time.time() - t0 < s:
        qapp.processEvents()
        time.sleep(0.005)


def test_checking_the_box_moves_the_check_after_the_fixture(qb, qapp):
    pid = qb.station.jobs[0]["pid"]
    p = qb.station.projects[pid]
    p.spec["tools"].append({"id": "serial", "type": "ocr",
                            "params": {"roi": [150, 230, 260, 46]}})
    ed = qb.open_job_editor(pid)
    t = next(t for t in p.spec["tools"] if t["id"] == "serial")
    t["params"]["compare_barcode"] = True                 # as the checkbox does
    ed._props_changed()
    ids = [t["id"] for t in p.spec["tools"]]
    assert ids[:2] == ["locate_part", "serial"], ids
    assert "after the fixture" in ed.statusBar().currentMessage()


def test_a_scan_that_does_not_match_the_part_is_ng(qb, qapp):
    pid = qb.station.jobs[0]["pid"]
    p = qb.station.projects[pid]
    p.spec["tools"].insert(1, {"id": "serial", "type": "ocr",
                               "params": {"roi": [150, 230, 260, 46],
                                          "compare_barcode": True}})
    qb.apply_barcode({"enabled": True, "prefix": "", "length": 0, "auto_run": True},
                     save=False)
    assert qb.barcode_scanned("AB123456")                 # what is printed on the part
    qb.station.wait(120)
    res = qb.station.last[pid]
    tools = {t["tool_id"]: t for t in res["tools"]}
    assert tools["serial"]["values"]["barcode_match"] is True, tools["serial"]
    assert not res["barcode_ng"] and len(res["tools"]) > 2
    assert qb.barcode_scanned("AB999999")
    qb.station.wait(120)
    res = qb.station.last[pid]
    assert res["verdict"] == "FAIL" and res["barcode_ng"].startswith("Barcode not good")
    assert [t["tool_id"] for t in res["tools"]] == ["locate_part", "serial"]
    _pump(qapp, 0.1)
    assert "Barcode not good" in qb.barcodeBar.state.text()

    # the box checked, the barcode check off in Settings: read as before
    qb.apply_barcode({"enabled": False}, save=False)
    qb.station.run_once()
    qb.station.wait(120)
    res = qb.station.last[pid]
    assert not res["barcode_ng"] and "barcode_match" not in \
        {t["tool_id"]: t for t in res["tools"]}["serial"]["values"]
