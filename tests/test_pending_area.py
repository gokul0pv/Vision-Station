"""A box drawn before its check is chosen waits on the picture -- and can be
moved and resized while it waits; the check then takes it as adjusted.
(It was a plain outline: a box drawn a little off could only be drawn again.)"""
import os
import shutil

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def editor(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    from vision_station.ui import main_window as V
    shutil.copytree(os.path.join(HERE, "projects", "demo_led_panel"),
                    str(tmp_path / "projects" / "demo_led_panel"),
                    ignore=shutil.ignore_patterns("results", "dataset"))
    ini = str(tmp_path / "s.ini")
    monkeypatch.setattr(V.QuickBuild, "_settings", lambda self: QSettings(ini, QSettings.IniFormat))
    w = V.QuickBuild(str(tmp_path / "projects"))
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    ed = w.open_job_editor(w.station.jobs[0]["pid"])
    for _ in range(20):
        app.processEvents()
    yield app, ed
    w.dirty.clear()
    w.close()
    app.processEvents()


def _drag(app, c, a, b, n=8):
    vp = c.viewport()
    pa, pb = c.mapFromScene(QPointF(*a)), c.mapFromScene(QPointF(*b))
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, pa)
    for k in range(1, n + 1):
        QTest.mouseMove(vp, QPoint(int(pa.x() + (pb.x() - pa.x()) * k / n),
                                   int(pa.y() + (pb.y() - pa.y()) * k / n)))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, pb)
    for _ in range(5):
        app.processEvents()


def test_the_waiting_box_is_adjusted_and_the_check_takes_it(editor):
    app, ed = editor
    c = ed.display.canvas
    ed.display._set_mode("draw_rect")
    _drag(app, c, (300, 300), (420, 380))
    m = getattr(ed, "_drawn_popup", None)
    if m is not None:
        m.close()                                    # the menu dismissed: choose later
    kind, geom, box = ed._pending_area
    near = lambda a, b: all(abs(x - y) <= 2 for x, y in zip(a, b))   # screen-pixel rounding
    x, y, w, h = geom["roi"]
    assert kind == "rect" and near([x, y, w, h], [300, 300, 120, 80])
    corner = box.mapToScene(box.rect().bottomRight())
    _drag(app, c, (corner.x(), corner.y()), (corner.x() + 50, corner.y() + 30))
    assert near(ed._pending_area[1]["roi"], [x, y, w + 50, h + 30])
    mid = box.mapToScene(box.rect().center())
    _drag(app, c, (mid.x(), mid.y()), (mid.x() + 20, mid.y() + 10))
    adjusted = list(ed._pending_area[1]["roi"])
    assert near(adjusted, [x + 20, y + 10, w + 50, h + 30])
    before = {t["id"] for t in ed._tools()}
    ed.add_tool("presence")
    new = [t for t in ed._tools() if t["id"] not in before]
    assert [t["params"]["roi"] for t in new] == [adjusted]
    assert ed._pending_area is None and box.scene() is None
    # and the check's own region resizes as any other
    it = c.regions[new[0]["id"]]
    corner = it.mapToScene(it.rect().bottomRight())
    _drag(app, c, (corner.x(), corner.y()), (corner.x() + 30, corner.y() + 20))
    assert ed._tool(new[0]["id"])["params"]["roi"][2:] != adjusted[2:]
