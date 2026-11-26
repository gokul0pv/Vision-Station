"""Turning boxes: the Find parts panel's boxes (search area and found parts)
turn by a drag just outside a corner, resize along their own sides, and a
turned part becomes a turned check whose reference patch is cut upright."""
import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _canvas(app):
    from vision_station.ui.canvas import ImageCanvas
    c = ImageCanvas()
    c.resize(900, 700)
    c.show()
    c.set_image(np.full((600, 800, 3), 90, np.uint8))
    c.zoom_100()
    app.processEvents()
    return c


def _drag(c, a, b, mods=Qt.NoModifier):
    vp = c.viewport()
    pa, pb = c.mapFromScene(a), c.mapFromScene(b)
    QTest.mousePress(vp, Qt.LeftButton, mods, pa)
    for k in range(1, 9):
        p = QPoint(int(pa.x() + (pb.x() - pa.x()) * k / 8), int(pa.y() + (pb.y() - pa.y()) * k / 8))
        QTest.mouseMove(vp, p)
    QTest.mouseRelease(vp, Qt.LeftButton, mods, pb)


def test_box_turns_from_outside_a_corner(app):
    from vision_station.ui.part_finder_panel import EditableBox
    c = _canvas(app)
    got = []
    box = EditableBox([300, 250, 200, 100], on_change=lambda r, a: got.append((r, a)))
    c.sc.addItem(box)
    app.processEvents()
    # from just beyond the top-right corner, a quarter turn about the centre
    centre = QPointF(400, 300)
    start = QPointF(508, 242)
    v = start - centre
    end = centre + QPointF(-v.y(), v.x())
    _drag(c, start, end)
    assert got, "turning did not report a change"
    r, a = got[-1]
    assert r == [300, 250, 200, 100]                 # turned in place
    assert abs(abs(a) - 90) < 3, a


def test_turned_box_resizes_along_its_own_side(app):
    from vision_station.ui.part_finder_panel import EditableBox
    c = _canvas(app)
    got = []
    box = EditableBox([300, 250, 200, 100], on_change=lambda r, a: got.append((r, a)),
                      angle=90)
    c.sc.addItem(box)
    app.processEvents()
    # turned 90 deg, the box's right side lies at the bottom of the picture
    right = box.mapToScene(QPointF(500, 300))
    left = box.mapToScene(QPointF(300, 300))
    _drag(c, right, right + QPointF(0, 40))
    r, a = got[-1]
    assert abs(r[2] - 240) <= 2 and r[3] == 100, r        # 40 px longer, same height
    assert abs(a - 90) < 1e-6
    left_after = box.mapToScene(QPointF(r[0], r[1] + r[3] / 2.0))
    assert abs(left_after.x() - left.x()) < 1.5 and abs(left_after.y() - left.y()) < 1.5


def test_turned_area_and_checks(app):
    """A part found in a turned area comes back turned with it, and its
    check is turned: Part present then matches it upright (it compared an
    upright window with an unturned reference patch before)."""
    import cv2
    from vision_station.core import part_finder as pf
    from vision_station.core.engine import Context
    from vision_station.tools.base import make_tool
    img = np.full((400, 400, 3), 60, np.uint8)
    # a textured part turned 30 degrees round (200, 200)
    part = np.zeros((40, 80, 3), np.uint8)
    part[:, :] = (200, 200, 200)
    part[8:32, 10:30] = (30, 30, 30)
    part[12:28, 50:70] = (120, 120, 120)
    M = cv2.getRotationMatrix2D((40, 20), -30, 1.0)
    M[:, 2] += (160, 180)
    cv2.warpAffine(part, M, (400, 400), dst=img, borderMode=cv2.BORDER_TRANSPARENT)
    crop, back = pf.turned_area(img, [100, 150, 200, 100], 30)
    assert crop.shape[:2] == (100, 200)
    q = back({"box": [60, 30, 80, 40], "score": 1.0, "type": 0})
    assert q["angle"] == 30.0
    cx, cy = q["box"][0] + 40, q["box"][1] + 20
    assert abs(cx - 200) <= 1 and abs(cy - 200) <= 1
    spec = pf.presence_tools([q], {}, [], image_size=(400, 400), seated=True)[0]
    assert spec["params"]["angle"] == 30.0 and spec["params"]["check_seated"]
    t = make_tool(spec)
    ctx = Context(reference=img)
    r = t.run(img.copy(), ctx)
    assert r.status == "pass" and r.score > 0.95, (r.status, r.score, r.message)


def test_seated_check_fails_a_lifted_end(app):
    from vision_station.core.engine import Context
    from vision_station.tools.base import make_tool
    rng = np.random.default_rng(1)
    ref = np.full((300, 300, 3), 70, np.uint8)
    ref[120:160, 100:200] = 190
    ref[125:155, 105:120] = 240                       # end caps
    ref[125:155, 180:195] = 240
    ref = np.clip(ref.astype(np.int16) + rng.integers(-3, 4, ref.shape), 0, 255).astype(np.uint8)
    spec = {"id": "p", "type": "presence",
            "params": {"roi": [92, 112, 116, 56], "min_score": 0.6, "search_pad": 10,
                       "check_seated": True}}
    t = make_tool(spec)
    good = t.run(ref.copy(), Context(reference=ref))
    assert good.status == "pass", good.message
    lifted = ref.copy()
    lifted[125:155, 178:197] = 60                     # one end's cap and fillet gone dark
    r = make_tool(spec).run(lifted, Context(reference=ref))
    assert r.status == "fail" and "looks different" in r.message, (r.status, r.message)
    pushed = np.roll(ref, 6, axis=1)
    r = make_tool(spec).run(pushed, Context(reference=ref))
    assert r.status == "fail" and "from its place" in r.message, (r.status, r.message)
    # without the option the same lifted part passes, as before
    spec["params"]["check_seated"] = False
    assert make_tool(spec).run(lifted, Context(reference=ref)).status == "pass"
