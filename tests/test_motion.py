"""Motion (ui/motion.py) with animation switched on: every animation ends
where it should and leaves nothing behind -- no opacity effect on a page, no
half-transparent window, the switch's knob home, the view pill under the
chosen view."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QStackedWidget


@pytest.fixture
def moving():
    app = QApplication.instance() or QApplication([])
    from vision_station.ui import motion
    was = motion._state["enabled"]
    motion._state["enabled"] = True
    yield app
    motion._state["enabled"] = was


def _settle(app, ms=700):
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def test_page_fade_ends_without_an_effect(moving):
    from vision_station.ui import motion
    st = QStackedWidget()
    a, b = QLabel("a"), QLabel("b")
    st.addWidget(a)
    st.addWidget(b)
    st.show()
    motion.fade_in(b)
    assert b.graphicsEffect() is None          # not visible yet: nothing done
    st.setCurrentWidget(b)
    motion.fade_in(b)
    assert b.graphicsEffect() is not None
    _settle(moving)
    assert b.graphicsEffect() is None


def test_dialog_fades_in_to_fully_opaque(moving):
    from vision_station.ui import motion
    d = QDialog()
    d.show()
    motion.window_in(d)
    _settle(moving)
    assert d.windowOpacity() == pytest.approx(1.0)
    d.close()


def test_switch_knob_springs_home(moving):
    from vision_station.ui import design
    sw = design.make_switch(False)
    sw.show()
    _settle(moving, 100)
    sw.setChecked(True)
    assert sw.isChecked()
    _settle(moving)
    assert sw._pos == pytest.approx(1.0, abs=1e-3)
    sw.setChecked(False)
    _settle(moving)
    assert sw._pos == pytest.approx(0.0, abs=1e-3)


def test_view_pill_glides_under_the_choice(moving):
    from vision_station.ui import simple_view
    ms = simple_view.ModeSwitch()
    ms.show()
    _settle(moving, 100)
    assert ms.thumb.geometry() == ms.buttons["run"].geometry()
    seen = []
    ms.changed.connect(seen.append)
    ms.buttons["history"].click()
    _settle(moving)
    assert seen == ["history"] and ms.mode() == "history"
    assert ms.thumb.geometry() == ms.buttons["history"].geometry()
    ms.set_mode("build")
    _settle(moving)
    assert ms.thumb.geometry() == ms.buttons["build"].geometry()
