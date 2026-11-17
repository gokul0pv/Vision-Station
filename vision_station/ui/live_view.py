"""Live Camera: the camera's picture at full rate, for setting up -- aiming the
camera, positioning the part, focusing and choosing an exposure. Nothing is
inspected.

It shares the camera with the jobs (a GigE Basler admits one connection, see
vp_app.SharedCamera), so it can stay open while a job runs. Exposure and gain
set here are remembered per camera and used whenever the camera is opened.

Positioning aids are drawn as scene items over the image, in full image
coordinates, rather than painted into each frame: a 20-megapixel frame costs
tens of milliseconds to copy and draw on, a line item nothing.
"""
import time

import cv2
from PySide6.QtCore import QLineF, Qt, QTimer
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                               QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from vision_station.ui.display import VisionDisplay

OVERLAYS = ("None", "Centre cross", "Thirds grid", "Fine grid (10 x 10)")


def focus_score(img):
    """Sharpness of the centre of the frame: variance of the Laplacian over a
    full-resolution crop. Only comparable from moment to moment on the same
    scene -- turn the focus ring and watch for the peak."""
    h, w = img.shape[:2]
    s = min(h, w, 800)
    y, x = (h - s) // 2, (w - s) // 2
    g = img[y:y + s, x:x + s]
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def brightness(img):
    """(mean 0-255, percent of pixels at 255) on a cheap subsample."""
    sub = img[::8, ::8]
    return float(sub.mean()), float((sub.max(axis=2) if sub.ndim == 3 else sub)
                                    .__ge__(255).mean() * 100.0)


class LiveCameraWindow(QWidget):
    """settings: a QSettings to remember exposure and gain in (may be None).
    open_camera: index -> started (shared) camera handle."""

    POLL_MS = 60

    def __init__(self, open_camera, settings=None, index=0, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Live Camera")
        self.setObjectName("liveCamera")
        self.resize(1100, 760)
        self._open_camera = open_camera
        self._qs = settings
        self.cam = None
        self._last_seq = -1
        self._frame = None
        self._fps_t, self._fps_n, self._fps = time.time(), 0, 0.0
        self._overlay_items = []
        self._fitted_size = None

        # ---- controls ------------------------------------------------------
        self.index = QSpinBox()
        self.index.setRange(0, 16)
        self.index.setValue(index)
        self.index.setToolTip("Which camera: its address and settings are under "
                              "Settings > Camera (0 = the first, or only, camera)")
        self.camName = QLabel("")
        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(0.05, 5000.0)
        self.exposure.setDecimals(2)
        self.exposure.setSuffix(" ms")
        self.exposure.setSingleStep(5.0)
        self.exposure.setKeyboardTracking(False)
        self.exposure.setToolTip("Longer is brighter; a moving part blurs when it is too long")
        self.gain = QDoubleSpinBox()
        self.gain.setRange(0.0, 5000.0)
        self.gain.setDecimals(1)
        self.gain.setKeyboardTracking(False)
        self.gain.setToolTip("Electronic brightening: adds noise. Prefer exposure or light.")
        self.autoExp = QCheckBox("Auto")
        self.autoExp.setToolTip("Let the camera choose the exposure -- for looking around "
                                "only; turn it off before inspecting")
        self.overlay = QComboBox()
        self.overlay.addItems(OVERLAYS)
        self.overlay.setCurrentIndex(1)
        self.overlay.setToolTip("Lines over the picture for lining the part up")
        self.fitBtn = QPushButton("Fit")
        self.pauseBtn = QPushButton("Pause")
        self.pauseBtn.setCheckable(True)
        self.saveBtn = QPushButton("Save image...")
        self.saveBtn.setToolTip("Save the frame on screen at full resolution -- "
                                "for example as a job's reference image")
        self.settingsBtn = QPushButton("Camera settings...")
        self.settingsBtn.setToolTip("The camera's IP address, and every setting: gain, white "
                                    "balance, trigger, sensor region, network")
        self.settingsBtn.clicked.connect(self._open_settings)

        top = QHBoxLayout()
        top.setContentsMargins(10, 8, 10, 4)
        for w in (QLabel("Camera"), self.index, self.camName):
            top.addWidget(w)
        top.addSpacing(16)
        for w in (QLabel("Exposure"), self.exposure, self.autoExp):
            top.addWidget(w)
        top.addSpacing(8)
        self.gainLabel = QLabel("Gain")
        top.addWidget(self.gainLabel)
        top.addWidget(self.gain)
        top.addSpacing(16)
        top.addWidget(QLabel("Guide"))
        top.addWidget(self.overlay)
        top.addStretch(1)
        for w in (self.fitBtn, self.pauseBtn, self.saveBtn, self.settingsBtn):
            top.addWidget(w)

        self.display = VisionDisplay(records=False)
        self.display.canvas.placeholder = "Opening the camera..."
        self.status = QLabel("")
        self.status.setObjectName("liveStatus")
        self.status.setContentsMargins(10, 4, 10, 6)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(top)
        lay.addWidget(self.display, 1)
        lay.addWidget(self.status)

        self.index.valueChanged.connect(self._reopen)
        self.exposure.valueChanged.connect(self._apply)
        self.gain.valueChanged.connect(self._apply)
        self.autoExp.toggled.connect(self._apply)
        self.overlay.currentIndexChanged.connect(self._draw_overlay)
        self.fitBtn.clicked.connect(self.display.canvas.fit)
        self.saveBtn.clicked.connect(self.save_image)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self._reopen()

    # ---- camera ------------------------------------------------------------
    def _open_settings(self):
        qb = self.parent()
        if qb is not None and hasattr(qb, "show_settings"):
            dlg = qb.show_settings("camera")
            dlg.cameraPage.slot.setValue(self.index.value())

    def _key(self, name):
        return "camera%d/%s" % (self.index.value(), name)

    def _reopen(self):
        self.timer.stop()
        self._release()
        self._last_seq, self._frame, self._fitted_size = -1, None, None
        self.display.canvas.set_image(None)
        self.camName.setText("")
        self.status.setText("Opening camera %d..." % self.index.value())
        try:
            self.cam = self._open_camera(self.index.value())
        except Exception as e:
            self.cam = None
            self.status.setText("The camera would not open: %s" % e)
            return
        self._settings_loaded = False
        self.timer.start(self.POLL_MS)

    def _release(self):
        if self.cam is not None:
            try:
                self.cam.stop()
            except Exception:
                pass
        self.cam = None

    def _load_settings(self):
        """Once the camera is up: show the settings it is running with, its
        limits, and the gain unit this model actually uses."""
        cam = self.cam
        s = cam.settings
        pc = getattr(cam, "_pylon_cam", None)
        widgets = (self.exposure, self.gain, self.autoExp)
        for w in widgets:
            w.blockSignals(True)
        try:
            if pc is not None:
                lo, hi = pc.exposure_range_us()
                if lo is not None:
                    self.exposure.setRange(max(0.01, lo / 1000.0), min(hi / 1000.0, 10000.0))
                unit, glo, ghi = pc.gain_unit()
                if unit is not None:
                    self.gain.setRange(float(glo), float(ghi))
                    self.gain.setDecimals(0 if unit == "raw" else 1)
                    self.gainLabel.setText("Gain (%s)" % unit)
            self.exposure.setValue(s.exposure_ms)
            self.gain.setValue(s.gain_db)
            self.autoExp.setChecked(bool(s.auto_exposure))
        finally:
            for w in widgets:
                w.blockSignals(False)
        settable = cam.backend == "pylon"
        for w in widgets:
            w.setEnabled(settable)
        if not settable:
            self.exposure.setToolTip("An OpenCV device: exposure cannot be set from here")
        self.exposure.setEnabled(settable and not self.autoExp.isChecked())
        info = cam.info or {}
        self.camName.setText("%s  %s" % (info.get("model", ""),
                                         ("serial %s" % info["serial"]) if info.get("serial")
                                         else ""))
        self._settings_loaded = True

    def _apply(self):
        if self.cam is None or getattr(self.cam, "backend", None) != "pylon":
            return
        s = self.cam.settings.copy()
        s.auto_exposure = self.autoExp.isChecked()
        s.exposure_ms = float(self.exposure.value())
        s.gain_db = float(self.gain.value())
        s.auto_gain = False
        self.exposure.setEnabled(not s.auto_exposure)
        self.cam.update_settings(s)
        if self._qs is not None:
            self._qs.setValue(self._key("exposure_ms"), s.exposure_ms)
            self._qs.setValue(self._key("gain"), s.gain_db)
            self._qs.setValue(self._key("auto_exposure"), s.auto_exposure)

    # ---- frames ------------------------------------------------------------
    def _poll(self):
        cam = self.cam
        if cam is None:
            return
        if cam.backend is None:
            if not cam.is_alive():
                self.timer.stop()
                self.display.canvas.placeholder = "No picture"
                self.display.canvas.viewport().update()
                self.status.setText(cam.last_error or "The camera did not start.")
            return
        if not self._settings_loaded:
            self._load_settings()
        seq = getattr(cam.thread, "_seq", None) if hasattr(cam, "thread") else None
        if seq is not None and seq == self._last_seq:
            return
        self._last_seq = seq
        self._fps_n += 1
        now = time.time()
        if now - self._fps_t >= 1.0:
            self._fps = self._fps_n / (now - self._fps_t)
            self._fps_t, self._fps_n = now, 0
        if self.pauseBtn.isChecked():
            return
        f = cam.get_frame()
        if f is None:
            return
        self._frame = f
        canvas = self.display.canvas
        canvas.set_image(f, keep_view=True)
        h, w = f.shape[:2]
        if self._fitted_size != (w, h):
            self._fitted_size = (w, h)
            canvas.fit()
            self._draw_overlay()
        mean, sat = brightness(f)
        msg = ("%d x %d   %.1f frames/s   brightness %.0f / 255   saturated %.2f%%   "
               "focus %.0f" % (w, h, self._fps, mean, sat, focus_score(f)))
        if mean < 40:
            msg += "   - dark: lengthen the exposure or add light"
        elif sat > 2.0:
            msg += "   - parts are burnt out: shorten the exposure"
        if cam.last_error:
            msg += "   (last camera message: %s)" % cam.last_error
        self.status.setText(msg)

    # ---- positioning guides ------------------------------------------------
    def _draw_overlay(self):
        sc = self.display.canvas.sc
        for it in self._overlay_items:
            sc.removeItem(it)
        self._overlay_items = []
        if self._fitted_size is None:
            return
        w, h = self._fitted_size
        kind = self.overlay.currentIndex()
        lines = []
        if kind == 1:
            lines = [(w / 2, 0, w / 2, h), (0, h / 2, w, h / 2)]
        elif kind == 2:
            lines = [(w * i / 3, 0, w * i / 3, h) for i in (1, 2)] + \
                    [(0, h * i / 3, w, h * i / 3) for i in (1, 2)]
        elif kind == 3:
            lines = [(w * i / 10, 0, w * i / 10, h) for i in range(1, 10)] + \
                    [(0, h * i / 10, w, h * i / 10) for i in range(1, 10)]
        pen = QPen(QColor(0, 230, 255, 200))
        pen.setCosmetic(True)                 # 1.5 screen pixels at any zoom
        pen.setWidthF(1.5)
        for x1, y1, x2, y2 in lines:
            it = sc.addLine(QLineF(x1, y1, x2, y2), pen)
            it.setZValue(50)
            self._overlay_items.append(it)
        if kind == 1:                         # a small ring marks the exact centre
            r = min(w, h) * 0.03
            it = sc.addEllipse(w / 2 - r, h / 2 - r, 2 * r, 2 * r, pen)
            it.setZValue(50)
            self._overlay_items.append(it)

    # ---- saving ------------------------------------------------------------
    def save_image(self, path=None):
        if self._frame is None:
            return None
        # The button's clicked signal passes its checked state: False arrived
        # here as the path, and saving raised TypeError inside imwrite.
        if not isinstance(path, str) or not path:
            name = time.strftime("camera_%Y%m%d_%H%M%S.png")
            path, _ = QFileDialog.getSaveFileName(self, "Save camera image", name,
                                                  "PNG (*.png);;JPEG (*.jpg)")
            if not path:
                return None
        from vision_station.core import imio
        ok = imio.imwrite(path, self._frame) if hasattr(imio, "imwrite") else \
            cv2.imencode(os.path.splitext(path)[1] or ".png", self._frame)[1].tofile(path)
        self.status.setText("Saved %s" % path)
        return path

    def closeEvent(self, e):
        self.timer.stop()
        self._release()
        super().closeEvent(e)


def saved_settings(qs, index):
    """The settings saved for this camera -- its address and every parameter
    from Settings > Camera, with the exposure and gain last chosen here -- as
    a CameraSettings, or None if nothing was ever saved."""
    from vision_station.ui.settings_dialog import load_camera_settings
    return load_camera_settings(qs, index)
