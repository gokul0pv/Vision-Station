"""The application's icon: one drawing (assets/icon), every size Windows asks
for, the same picture in the windows, the .exe's .ico and Vision Glass."""
import json
import os
import shutil
import urllib.error
import urllib.request

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICON_DIR = os.path.join(HERE, "assets", "icon")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_the_drawings_and_the_ico_are_there(app):
    from PIL import Image
    for f in ("vision_station.svg", "vision_station_small.svg", "vision_station.ico"):
        assert os.path.isfile(os.path.join(ICON_DIR, f)), f
    sizes = set(Image.open(os.path.join(ICON_DIR, "vision_station.ico")).info["sizes"])
    assert {(16, 16), (24, 24), (32, 32), (48, 48), (256, 256)} <= sizes


def test_app_icon_has_every_size_and_is_not_blank(app):
    from vision_station.ui import icons
    ic = icons.app_icon()
    have = {(s.width(), s.height()) for s in ic.availableSizes()}
    assert {(16, 16), (32, 32), (256, 256)} <= have
    img = ic.pixmap(64, 64).toImage()
    c = img.pixelColor(32, 18)                        # inside the tile: the blue
    assert c.alpha() == 255 and c.blue() > c.red() + 60
    assert img.pixelColor(0, 0).alpha() == 0          # the margin round the tile is clear


def test_small_sizes_use_the_simplified_drawing(app):
    """At 16 px the corners of the viewfinder would be noise: the lens alone."""
    from vision_station.ui import icons
    img = icons.app_icon().pixmap(16, 16).toImage()
    # the small drawing's tile reaches the edge (1 px margin of 32 -> 0.5 px)
    assert img.pixelColor(1, 8).alpha() > 200


def test_glass_serves_the_icon_and_nothing_beside_it(app, tmp_path):
    from vision_station.glass.server import GlassStation, serve
    shutil.copytree(os.path.join(HERE, "projects", "demo_led_panel"),
                    str(tmp_path / "projects" / "demo_led_panel"),
                    ignore=shutil.ignore_patterns("results", "dataset"))
    st = GlassStation(str(tmp_path / "projects"), QSettings(str(tmp_path / "s.ini"),
                                                             QSettings.IniFormat))
    httpd = serve(st, 0)
    base = "http://127.0.0.1:%d" % httpd.server_address[1]
    try:
        with urllib.request.urlopen(base + "/icon/vision_station.svg") as r:
            assert r.headers["Content-Type"] == "image/svg+xml" and b"<svg" in r.read()
        with urllib.request.urlopen(base + "/favicon.ico") as r:
            assert r.headers["Content-Type"] == "image/x-icon"
        with urllib.request.urlopen(base + "/") as r:
            page = r.read().decode("utf-8")
        assert 'rel="icon"' in page and "/icon/vision_station.svg" in page
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/icon/..%2F..%2Fvision_station%2Fglass%2Fserver.py")
    finally:
        st.continuous = False
        httpd.shutdown()
