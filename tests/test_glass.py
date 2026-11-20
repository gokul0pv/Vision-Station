"""Vision Glass: the web front end's server over the Vision Station engine.

    python -m pytest tests/test_glass.py -q
"""
import json
import os
import shutil
import sys
import urllib.error
import urllib.request

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                      # noqa: E402
from PySide6.QtWidgets import QApplication                # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def glass(qapp, tmp_path):
    from vision_station.glass.server import GlassStation, serve
    shutil.copytree(os.path.join(HERE, "projects", "demo_led_panel"),
                    str(tmp_path / "projects" / "demo_led_panel"),
                    ignore=shutil.ignore_patterns("results", "dataset"))
    qs = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)   # never the user's
    st = GlassStation(str(tmp_path / "projects"), qs)
    httpd = serve(st, 0)
    base = "http://127.0.0.1:%d" % httpd.server_address[1]

    def get(p, raw=False):
        with urllib.request.urlopen(base + p) as r:
            d = r.read()
            return d if raw else json.loads(d)

    def post(p, b):
        req = urllib.request.Request(base + p, json.dumps(b).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return dict(json.loads(e.read()), HTTP=e.code)
    yield st, get, post
    st.continuous = False
    httpd.shutdown()


def test_the_page_and_its_files_are_served(glass):
    _st, get, _post = glass
    assert b"Vision Glass" in get("/", raw=True)
    for f in ("/web/glass.css", "/web/app.js", "/web/glass.js", "/web/icons.js"):
        assert len(get(f, raw=True)) > 1000, f


def test_files_outside_the_web_folder_are_not_served(glass):
    _st, get, _post = glass
    with pytest.raises(urllib.error.HTTPError):
        get("/web/..%2F..%2Fserver.py", raw=True)


def test_state_run_and_pictures(glass):
    _st, get, post = glass
    s = get("/api/state")
    pid = s["jobs"][0]["pid"]
    assert s["jobs"][0]["tools"] and s["jobs"][0]["size"]
    assert get("/img/reference/%s.jpg?w=400" % pid, raw=True)[:2] == b"\xff\xd8"
    r = post("/api/run", {})
    assert r["results"][pid]["verdict"] in ("PASS", "FAIL")
    assert get("/img/last/%s.jpg?w=400" % pid, raw=True)[:2] == b"\xff\xd8"
    assert get("/api/history")[0]["pid"] == pid
    assert get("/api/state")["counts"]["accept"] + get("/api/state")["counts"]["reject"] == 1


def test_checks_are_added_edited_and_undone(glass):
    _st, get, post = glass
    pid = get("/api/state")["jobs"][0]["pid"]
    a = post("/api/tool/add", {"pid": pid, "type": "presence", "roi": [20, 20, 60, 40]})
    tools = {t["id"]: t for t in get("/api/state")["jobs"][0]["tools"]}
    assert tools[a["id"]]["params"]["roi"] == [20, 20, 60, 40]
    post("/api/tool/update", {"pid": pid, "id": a["id"], "params": {"min_score": 0.4}})
    post("/api/tool/update-many", {"pid": pid, "ids": [a["id"]], "params": {"min_score": 0.3}})
    tools = {t["id"]: t for t in get("/api/state")["jobs"][0]["tools"]}
    assert tools[a["id"]]["params"]["min_score"] == 0.3
    post("/api/tool/delete", {"pid": pid, "ids": [a["id"]]})
    assert a["id"] not in {t["id"] for t in get("/api/state")["jobs"][0]["tools"]}
    assert post("/api/undo", {"pid": pid})["undone"]
    assert a["id"] in {t["id"] for t in get("/api/state")["jobs"][0]["tools"]}
    pv = post("/api/preview", {"pid": pid})
    assert a["id"] in {t["tool_id"] for t in pv["tools"]}


def test_a_verify_text_check_learns_its_text(glass):
    _st, get, post = glass
    pid = get("/api/state")["jobs"][0]["pid"]
    a = post("/api/tool/add", {"pid": pid, "type": "ocv", "roi": [150, 230, 260, 46]})
    tools = {t["id"]: t for t in get("/api/state")["jobs"][0]["tools"]}
    assert tools[a["id"]]["params"]["expect_text"] == "AB123456"


def test_the_barcode_rule_holds_runs(glass):
    _st, get, post = glass
    pid = get("/api/state")["jobs"][0]["pid"]
    post("/api/barcode", {"cfg": {"enabled": True, "prefix": "FIC", "length": 11}})
    assert post("/api/run", {}).get("need_barcode")
    assert not post("/api/barcode/scan", {"code": "XYZ"})["ok"]
    r = post("/api/barcode/scan", {"code": "FIC12345678"})
    assert r["ok"] and r["run"]["results"][pid]["barcode"] == "FIC12345678"
    assert post("/api/run", {"pid": pid}).get("results"), "a job's own run is held"


def test_errors_come_back_as_messages(glass):
    _st, get, post = glass
    pid = get("/api/state")["jobs"][0]["pid"]
    r = post("/api/source", {"pid": pid, "mode": "folder", "folder": "Z:/no/such/folder"})
    assert r["HTTP"] == 400 and "does not exist" in r["error"]
