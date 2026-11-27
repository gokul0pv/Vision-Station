"""Make the application's icon files from its two drawings.

    assets/icon/vision_station.svg         the icon, 32 px and up
    assets/icon/vision_station_small.svg   the icon at 16-24 px (the lens alone)

Writes beside them:

    vision_station.ico        every Windows size (16-256), each drawn at its own
                              size from the right drawing -- not one picture
                              shrunk -- for the .exe, shortcuts and the taskbar
    vision_station_256.png    and _512.png, for documents and installers

    python scripts/make_icon.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ICON_DIR = os.path.join(ROOT, "assets", "icon")
SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
SMALL_UP_TO = 24


def render(size):
    """The icon at size x size as a PIL image, from the drawing for that size."""
    from PIL import Image
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    name = "vision_station_small.svg" if size <= SMALL_UP_TO else "vision_station.svg"
    r = QSvgRenderer(os.path.join(ICON_DIR, name))
    img = QImage(size, size, QImage.Format_RGBA8888)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    r.render(p, QRectF(0, 0, size, size))
    p.end()
    return Image.frombytes("RGBA", (size, size), bytes(img.constBits()))


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841 -- QtSvg needs one
    images = {s: render(s) for s in SIZES}
    ico = os.path.join(ICON_DIR, "vision_station.ico")
    big = images[256]
    big.save(ico, format="ICO", sizes=[(s, s) for s in SIZES],
             append_images=[images[s] for s in SIZES if s != 256])
    big.save(os.path.join(ICON_DIR, "vision_station_256.png"))
    render(512).save(os.path.join(ICON_DIR, "vision_station_512.png"))
    print("  icon written: %s" % ico)
    return ico


if __name__ == "__main__":
    main()
