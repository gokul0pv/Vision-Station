"""python -m vision_station.glass  (or run_glass.bat) -- Vision Glass.

A separate front end: HTML, CSS and JavaScript in web/, over the Vision
Station engine served from server.py on 127.0.0.1. It opens in its own
window (Chromium, through Qt WebEngine); without Qt WebEngine it opens in the
default browser. Vision Station (run.bat) is not touched by it.

    python -m vision_station.glass            its own window
    python -m vision_station.glass --browser  in the default browser
    python -m vision_station.glass --port 8765
"""
import os
import sys

if not __package__:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

os.environ.setdefault("VS_DEVICE", "GPU")


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    port = 0
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    try:                                   # WebEngine must be imported before the app exists
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except Exception:
        QWebEngineView = None
    if sys.platform == "win32":
        # its own taskbar button and icon, not Python's
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("VisionStation.Glass")
        except Exception:
            pass
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(argv[:1])
    app.setApplicationName("Vision Glass")
    from vision_station.glass.server import GlassStation, serve
    from vision_station.ui import icons
    app.setWindowIcon(icons.app_icon())
    station = GlassStation()
    httpd = serve(station, port)
    url = "http://127.0.0.1:%d/" % httpd.server_address[1]
    print("Vision Glass: %s" % url)
    if QWebEngineView is None or "--browser" in argv:
        import webbrowser
        webbrowser.open(url)
        try:
            import time
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return 0
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QColor
    view = QWebEngineView()
    view.setWindowTitle("Vision Glass")
    view.setWindowIcon(icons.app_icon())
    view.page().setBackgroundColor(QColor("#0b0d12"))
    view.resize(1520, 960)
    view.setUrl(QUrl(url))
    view.show()
    rc = app.exec()
    station.continuous = False
    for src in station.sources.values():
        src.stop_camera()
    httpd.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
