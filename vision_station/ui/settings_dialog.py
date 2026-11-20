"""Settings: every setting of the application in one window, laid out as
macOS System Settings -- a sidebar of panes, each pane grouped sections of
rows with the control on the right.

    General          appearance, Advanced mode, running jobs in parallel
    Camera           the camera's address and every parameter -- exposure,
                     gain, white balance, rotation, mirroring, pixel format,
                     gamma, black level, trigger, frame rate, sensor region,
                     GigE network -- with ranges read from the camera
    Results          where each part's picture is saved, and what an exported
                     job file does with its results
    Line connection  the TCP/IP interface a PLC or robot uses
    Security         Operator Mode and the supervisor password

Camera and result-picture settings belong to the PC (QSettings); the line
connection and exported-job output belong to the application and are saved
with it. How it all looks is in ui_design.py.
"""
import json
import os

from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QSlider, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QGridLayout,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMenu, QMessageBox, QPushButton, QScrollArea, QSpinBox, QStackedWidget,
                               QVBoxLayout, QWidget)

from vision_station.hardware import camera as camera_mod
from vision_station.ui import icons
from vision_station.ui import theme
from vision_station.ui import design as ui_design
from vision_station.hardware.camera import CameraSettings
from vision_station.ui.design import FormGroup, make_switch, page_header

from vision_station.core import annotate as _look

RESULT_DEFAULTS = {"folder": "", "which": "all", "annotated": True, "original": False,
                   "format": "jpg", "quality": 97, "per_day": True, "labels": True,
                   "banner": True, "ok_colour": _look.DEFAULT_OK,
                   "ng_colour": _look.DEFAULT_NG, "line_scale": 1.0}
LINE_SCALE_NAMES = {0.5: "Thin", 1.0: "Standard", 1.5: "Thick", 2.0: "Extra thick", 3.0: "Heavy"}


# ---------------------------------------------------------------- storage
def load_camera_settings(qs, index=0):
    """The camera settings saved for camera `index`, or None when nothing has
    been saved. Exposure, gain and auto exposure may also have been changed
    in Live Camera since, which writes only those three keys -- they win."""
    if qs is None:
        return None
    raw = qs.value("camera%d/settings" % index)
    has_legacy = qs.value("camera%d/exposure_ms" % index) is not None
    if raw is None and not has_legacy:
        return None
    s = CameraSettings()
    if raw:
        try:
            s = CameraSettings.from_dict(json.loads(str(raw)))
        except (TypeError, ValueError):
            pass
    if has_legacy:
        try:
            s.exposure_ms = float(qs.value("camera%d/exposure_ms" % index))
            s.gain_db = float(qs.value("camera%d/gain" % index, s.gain_db))
            s.auto_exposure = str(qs.value("camera%d/auto_exposure" % index,
                                           "false")).lower() == "true"
        except (TypeError, ValueError):
            pass
    return s


def save_camera_settings(qs, index, s):
    qs.setValue("camera%d/settings" % index, json.dumps(s.to_dict()))
    qs.setValue("camera%d/exposure_ms" % index, float(s.exposure_ms))
    qs.setValue("camera%d/gain" % index, float(s.gain_db))
    qs.setValue("camera%d/auto_exposure" % index, bool(s.auto_exposure))
    qs.sync()


def load_result_settings(qs):
    out = dict(RESULT_DEFAULTS)
    if qs is None:
        return out
    raw = qs.value("results/saving")
    if raw:
        try:
            out.update(json.loads(str(raw)))
        except (TypeError, ValueError):
            pass
    return out


def save_result_settings(qs, d):
    qs.setValue("results/saving", json.dumps(d))
    qs.sync()


# ---------------------------------------------------------------- small helpers
def _dspin(lo, hi, dec=2, step=1.0, suffix="", width=130):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setKeyboardTracking(False)
    if suffix:
        w.setSuffix(suffix)
    w.setFixedWidth(width)
    return w


def _ispin(lo, hi, step=1, suffix="", width=130):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setKeyboardTracking(False)
    if suffix:
        w.setSuffix(suffix)
    w.setFixedWidth(width)
    return w


def _combo(items, width=240):
    c = QComboBox()
    for text, data in items:
        c.addItem(text, data)
    c.setFixedWidth(width)
    return c


def _row(*widgets):
    h = QHBoxLayout()
    h.setSpacing(8)
    for w in widgets:
        h.addWidget(w)
    return h


class Page(QScrollArea):
    """One pane: a title with Restore defaults beside it, then grouped
    sections, scrolling when tall.

    A page restores its defaults in restore_defaults(); KEEPS names what it
    leaves alone (where the camera, light and folders are), APPLIES_NOW says
    whether the change takes effect at once rather than on Apply / Done."""
    KEEPS = ""
    APPLIES_NOW = False

    def __init__(self, title, subtitle=""):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.title = title
        body = QWidget()
        body.setObjectName("prefContent")
        self.lay = QVBoxLayout(body)
        self.lay.setContentsMargins(28, 22, 28, 24)
        self.lay.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(12)
        head.addWidget(page_header(title, subtitle), 1)
        self.restoreBtn = QPushButton("Restore defaults")
        self.restoreBtn.setObjectName("restoreDefaults")
        self.restoreBtn.setAutoDefault(False)
        self.restoreBtn.setToolTip("Put this page's settings back as they were when installed")
        self.restoreBtn.clicked.connect(self.ask_restore)
        head.addWidget(self.restoreBtn, 0, Qt.AlignTop)
        self.lay.addLayout(head)
        self._end = None
        self.setWidget(body)

    def ask_restore(self):
        """Restore defaults, after asking. False when the user said no."""
        text = "Restore the %s settings to their defaults?" % self.title
        if self.KEEPS:
            text += "\n\nKept as they are: %s." % self.KEEPS
        text += ("\n\nThis takes effect at once." if self.APPLIES_NOW else
                 "\n\nNothing is saved until Apply or Done - Cancel keeps the "
                 "current settings.")
        if QMessageBox.question(self, "Restore defaults", text,
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return False
        return self.restore_defaults() is not False

    def restore_defaults(self):
        raise NotImplementedError

    def add(self, group):
        self.lay.addWidget(group.widget if isinstance(group, FormGroup) else group)
        return group

    def finish(self):
        self.lay.addStretch(1)


# ---------------------------------------------------------------- General
class GeneralPage(Page):
    def __init__(self, dialog):
        super().__init__("General", "How the application looks and works on this PC.")
        self.dlg = dialog
        qb = dialog.qb
        g = self.add(FormGroup("Appearance"))
        self.appearance = _combo([("Automatic (follow Windows)", "system"), ("Light", "light"),
                                  ("Dark", "dark")], 220)
        cur = str(dialog.qs.value("theme", "system") or "system") if dialog.qs is not None else "system"
        self.appearance.setCurrentIndex(max(0, self.appearance.findData(cur)))
        g.add_row("Appearance", self.appearance)
        self.appearance.currentIndexChanged.connect(self._theme)

        g = self.add(FormGroup("Working", footer="Advanced mode shows the engineering view: "
                                                 "menu bar, Toolbox, every tool and setting, "
                                                 "terminals and document tabs."))
        self.advanced = make_switch(bool(getattr(qb, "advanced", False)))
        g.add_row("Advanced mode", self.advanced, help="For people who build inspections")
        self.advanced.toggled.connect(lambda on: qb is not None and qb.set_advanced(on))
        self.parallel = make_switch(bool(qb is not None and qb.aParallel.isChecked()))
        g.add_row("Run inspections in parallel", self.parallel,
                  help="Overlaps a cycle's inspections on a PC with many cores")
        self.parallel.toggled.connect(lambda on: qb is not None and qb.aParallel.setChecked(on))

        # advanced training (core/advanced.py): new jobs train their own AI
        # model on their first Run
        from vision_station.core import advanced as adv
        possible, why = adv.can_train()
        g = self.add(FormGroup("AI training", footer=(
            "Adds Model teaching run to Run's dropdown (Normal run stays the default): more "
            "pictures of the good part are taken, added to every check's good pictures, and "
            "a YOLO model learns the parts and text the job checks. It then confirms those "
            "checks on every run. Training: %s." % (why if possible else "not possible on "
                                                     "this PC - " + why))))
        on = bool(qb is not None and hasattr(qb, "advanced_training_on")
                  and qb.advanced_training_on())
        self.useAdvanced = make_switch(on)
        self.useAdvanced.setObjectName("useAdvancedTraining")
        g.add_row("Use advanced model for training", self.useAdvanced,
                  help="Run gets a Model teaching run, besides Normal run")
        self.useAdvanced.toggled.connect(
            lambda v: qb is not None and hasattr(qb, "set_advanced_training")
            and qb.set_advanced_training(v))
        # not possible here: it can be switched off, never on
        self.useAdvanced.setEnabled(possible or on)
        self.modelSize = _combo([("YOLO11 small - fastest", "small"),
                                 ("YOLO11 medium", "medium"),
                                 ("YOLO11 large", "large"),
                                 ("YOLO11 extra large", "xlarge"),
                                 ("YOLO26 small (newest)", "26-small"),
                                 ("YOLO26 medium (newest)", "26-medium"),
                                 ("YOLO26 large (newest)", "26-large"),
                                 ("YOLO26 extra large (newest) - slowest", "26-xlarge")], 280)
        self.modelSize.setObjectName("advancedModelSize")
        cur = qb.advanced_model_size() if qb is not None and hasattr(qb, "advanced_model_size")             else "small"
        self.modelSize.setCurrentIndex(max(0, self.modelSize.findData(cur)))
        g.add_row("AI model size", self.modelSize,
                  help="Bigger learns more but trains 2-4x longer and runs slower")
        self.modelSize.currentIndexChanged.connect(
            lambda _i: qb is not None and hasattr(qb, "set_advanced_model_size")
            and qb.set_advanced_model_size(self.modelSize.currentData()))
        self.modelSize.setEnabled(possible)

        g = self.add(FormGroup("Installation"))
        check = QPushButton("Check installation...")
        check.clicked.connect(lambda: qb is not None and qb.check_installation())
        g.add_row("Everything the application needs", check,
                  help="Camera driver, text reader, AI runtime and models, the graphics card")
        # what this PC is, and what was chosen for it (core/system_profile.py)
        try:
            from vision_station.core import system_profile
            prof, plan = system_profile.current()
            pc = QLabel(system_profile.describe(prof, plan))
        except Exception as e:
            pc = QLabel("This PC could not be looked at: %s" % e)
        pc.setObjectName("rowHelp")
        pc.setWordWrap(True)
        pc.setTextInteractionFlags(Qt.TextSelectableByMouse)
        g.add_widget(pc)
        if qb is None:
            for w in (self.advanced, self.parallel, check, self.useAdvanced, self.modelSize):
                w.setEnabled(False)
        self.finish()

    APPLIES_NOW = True

    def restore_defaults(self):
        self.appearance.setCurrentIndex(self.appearance.findData("system"))
        self.advanced.setChecked(False)
        self.parallel.setChecked(False)
        self.useAdvanced.setChecked(False)
        self.modelSize.setCurrentIndex(self.modelSize.findData("small"))

    def _theme(self):
        key = self.appearance.currentData()
        qb = self.dlg.qb
        if qb is not None and hasattr(qb, "set_theme"):
            qb.set_theme(key)
        elif self.dlg.qs is not None:
            self.dlg.qs.setValue("theme", key)


# ---------------------------------------------------------------- Camera
ROTATIONS = [("Upright", 0), ("Upside down (180°)", 180), ("90° clockwise", 90),
             ("90° counter-clockwise", 270)]


class CameraPage(Page):
    """Every camera setting, for one camera number."""

    def __init__(self, dialog):
        super().__init__("Camera", "The application opens and sets up the camera itself - "
                                   "Pylon Viewer is not needed, and must be closed.")
        self.dlg = dialog
        self.handle = None              # a shared-camera handle while connected
        self.desc = None

        # ---- connection -----------------------------------------------
        g = self.add(FormGroup("Connection", footer="An IP address reaches the camera directly, "
                                                    "even where automatic discovery does not."))
        self.slot = _ispin(0, 7, width=80)
        self.slot.setToolTip("The camera number inspections use. One camera: leave it at 0.")
        g.add_row("Camera number", self.slot, help="Used by every inspection set to 'camera'")
        self.address = QLineEdit()
        self.address.setPlaceholderText("e.g. 192.168.3.6")
        self.address.setFixedWidth(200)
        self.findBtn = QPushButton("Find...")
        self.findBtn.setToolTip("List the Basler cameras this PC can discover")
        g.add_row("IP address or serial number", _row(self.address, self.findBtn),
                  help="Empty: the first camera found")
        self.connectBtn = QPushButton("Connect")
        self.connectBtn.setToolTip("Open the camera with these settings and read its ranges "
                                   "and current values")
        self.liveBtn = QPushButton("Live View...")
        self.state = QLabel("Not connected")
        self.state.setObjectName("statusLine")
        self.state.setWordWrap(True)
        self.state.setTextInteractionFlags(Qt.TextSelectableByMouse)
        st = QHBoxLayout()
        st.addWidget(self.state, 1)
        st.addWidget(self.connectBtn)
        st.addWidget(self.liveBtn)
        g.add_widget(st)
        # The packaged program without its Windows Firewall rule opens a GigE
        # camera but never receives a picture (hardware/firewall.py)
        self.fwNote = QLabel()
        self.fwNote.setObjectName("rowHelp")
        self.fwNote.setWordWrap(True)
        self.fwBtn = QPushButton("Allow through Windows Firewall...")
        self.fwBtn.setToolTip("Adds a firewall rule for this program, once. Windows asks "
                              "for an administrator.")
        self.fwBtn.clicked.connect(self.allow_firewall)
        fw = QHBoxLayout()
        fw.addWidget(self.fwNote, 1)
        fw.addWidget(self.fwBtn)
        self._fwRow = QWidget()
        self._fwRow.setLayout(fw)
        self._fwRow.setVisible(False)
        g.add_widget(self._fwRow)
        # What the camera comes back with after a power cycle. A stored mask
        # or address that does not fit the network (the acA5472-5gc had
        # 255.255.0.0 against the PC's /24) leaves it unreachable after every
        # restart, until a temporary IP is set by hand in IP Configurator.
        self.ipInfo = QLabel("Connect to see the address the camera keeps after a restart")
        self.ipInfo.setObjectName("rowHelp")
        self.ipInfo.setWordWrap(True)
        self.keepIpBtn = QPushButton("Keep this address after restart")
        self.keepIpBtn.setToolTip("Store the camera's current IP address and mask in the "
                                  "camera, so it is reachable after a power cycle without "
                                  "setting a temporary IP")
        self.keepIpBtn.setEnabled(False)
        self.keepIpBtn.clicked.connect(self.keep_ip)
        g.add_row("After a restart", self.keepIpBtn)
        g.add_widget(self.ipInfo)

        # ---- picture --------------------------------------------------
        g = self.add(FormGroup("Picture"))
        self.exposure = _dspin(0.01, 10000.0, 3, 1.0, " ms")
        self.autoExp = make_switch()
        g.add_row("Exposure", _row(self.exposure, QLabel("Auto"), self.autoExp),
                  help="Longer is brighter; a moving part blurs when it is too long")
        self.gain = _dspin(0.0, 5000.0, 1, 1.0)
        self.gainUnit = QLabel("")
        self.gainUnit.setObjectName("rowHelp")
        self.autoGain = make_switch()
        g.add_row("Gain", _row(self.gainUnit, self.gain, QLabel("Auto"), self.autoGain),
                  help="Electronic brightening - adds noise; prefer exposure or light")
        self.autoWB = make_switch()
        self.wbOnce = QPushButton("Balance Once")
        self.wbOnce.setToolTip("Point the camera at something white or grey and press: the "
                               "balance is held from then on")
        g.add_row("White balance", _row(self.wbOnce, QLabel("Auto"), self.autoWB),
                  help="Turn Auto off once the lighting is fixed")
        self.rotation = _combo([(t, d) for t, d in ROTATIONS], 220)
        self.rotation.setToolTip("For a camera mounted upside down or on its side. Inspections "
                                 "set up before a change see the picture turned - set up their "
                                 "reference picture again.")
        g.add_row("Rotation", self.rotation,
                  help="Upside down is done on the sensor itself, at no cost")
        self.revX = make_switch()
        g.add_row("Mirror left-right", self.revX)
        self.revY = make_switch()
        g.add_row("Mirror top-bottom", self.revY)

        g = self.add(FormGroup("Image quality"))
        self.pixfmt = _combo([("Automatic", "auto")], 240)
        g.add_row("Pixel format", self.pixfmt, help="Automatic: Bayer 8-bit on a colour camera - "
                                                    "full detail at full speed")
        self.gamma = _dspin(0.0, 4.0, 2, 0.05)
        self.gamma.setSpecialValueText("Camera's own")
        self.gamma.setToolTip("1.00 = off (linear, best for measuring)")
        g.add_row("Gamma", self.gamma)
        self.black = _dspin(-1.0, 4095.0, 0, 1.0)
        self.black.setSpecialValueText("Camera's own")
        g.add_row("Black level", self.black)

        # ---- acquisition ---------------------------------------------
        g = self.add(FormGroup("Acquisition"))
        self.trigger = _combo([("Free run (continuous)", "free"),
                               ("External trigger (input line)", "hardware")], 240)
        g.add_row("Trigger", self.trigger, help="With a trigger, each part's picture is the "
                                                "next frame - a shorter cycle")
        self.trigSource = QComboBox()
        self.trigSource.setEditable(True)
        self.trigSource.setFixedWidth(160)
        g.add_row("Trigger input", self.trigSource)
        self.trigEdge = QComboBox()
        self.trigEdge.setFixedWidth(160)
        g.add_row("Trigger edge", self.trigEdge)
        self.trigDelay = _dspin(0.0, 1000000.0, 0, 100.0, " µs")
        g.add_row("Trigger delay", self.trigDelay)
        self.rate = _dspin(0.0, 1000.0, 2, 1.0, " fps")
        self.rate.setSpecialValueText("Maximum")
        g.add_row("Frame rate limit", self.rate)

        # ---- sensor region -------------------------------------------
        g = self.add(FormGroup("Sensor region", footer="A smaller region transfers faster. "
                                                       "The picture is never scaled down."))
        self.roiMode = _combo([("As the camera has it", "keep"), ("Whole sensor", "full"),
                               ("Custom region", "custom")], 220)
        self.roiMode.setToolTip("Inspections' reference pictures and boxes are at the size the "
                                "camera delivers - change this only when setting up a new one")
        g.add_row("Region", self.roiMode)
        self.roiX, self.roiY = _ispin(0, 20000, width=96), _ispin(0, 20000, width=96)
        self.roiW, self.roiH = _ispin(0, 20000, width=96), _ispin(0, 20000, width=96)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        for i, (lab, w) in enumerate((("Left", self.roiX), ("Top", self.roiY),
                                      ("Width", self.roiW), ("Height", self.roiH))):
            l = QLabel(lab)
            l.setObjectName("rowHelp")
            grid.addWidget(l, 0, i)
            grid.addWidget(w, 1, i)
        grid.setColumnStretch(4, 1)
        g.add_widget(grid)

        # ---- network ---------------------------------------------------
        g = self.add(FormGroup("Network (GigE cameras)"))
        self.packet = _ispin(0, 16404, 4)
        self.packet.setSpecialValueText("Camera's own")
        g.add_row("Packet size", self.packet,
                  help="9000 with jumbo frames on the network card - fewer lost frames")
        self.ipd = _ispin(0, 1000000, 100)
        g.add_row("Inter-packet delay", self.ipd,
                  help="Raise it when frames arrive incomplete, especially with several cameras")
        self.heartbeat = _ispin(0, 60000, 500, " ms")
        self.heartbeat.setSpecialValueText("Camera's own")
        g.add_row("Heartbeat timeout", self.heartbeat,
                  help="How soon the camera frees itself after a program ends abruptly")
        self.finish()

        self.slot.valueChanged.connect(lambda _v: self.load(self.slot.value()))
        self.findBtn.clicked.connect(self.find_cameras)
        self.connectBtn.clicked.connect(lambda: self.connect_camera())
        self.liveBtn.clicked.connect(self._live)
        self.wbOnce.clicked.connect(self._balance_once)
        self.roiMode.currentIndexChanged.connect(lambda _i: self._sync_enabled())
        self.trigger.currentIndexChanged.connect(lambda _i: self._sync_enabled())
        self.autoExp.toggled.connect(lambda _b: self._sync_enabled())
        self.autoGain.toggled.connect(lambda _b: self._sync_enabled())
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._check_connected)

    # ---- values <-> widgets ----------------------------------------
    def load(self, index):
        s = load_camera_settings(self.dlg.qs, index) or CameraSettings()
        self.show_settings(s)
        self.desc = None
        self._release()
        self._say("Not connected - press Connect to read the camera's ranges and current values")
        try:
            self.check_firewall()
        except Exception as e:                   # never stop the settings opening
            print("[settings] firewall check: %s" % e)

    def _say(self, text, tone=None):
        self.state.setText(text)
        theme.set_tone(self.state, tone or "idle")

    KEEPS = "the camera number and its IP address or serial number"

    def restore_defaults(self):
        s = CameraSettings()
        s.address = self.address.text().strip()
        self.show_settings(s)
        self._say("Defaults shown - Apply or Done sends them to the camera")

    def check_firewall(self):
        """Show the firewall row when this program needs its rule."""
        from vision_station.hardware import firewall
        need = firewall.should_offer()
        if need:
            self.fwNote.setText(firewall.EXPLANATION)
        self._fwRow.setVisible(bool(need))
        return need

    def allow_firewall(self):
        from vision_station.hardware import firewall
        self._say("Waiting for Windows to allow this program...")
        if firewall.allow():
            self._fwRow.setVisible(False)
            self._say("Allowed. Pictures from the camera now reach this program - "
                      "press Connect.", "pass")
            return True
        self._say("The firewall rule was not added - the administrator prompt was "
                  "declined, or this account may not change the firewall. Ask IT to allow "
                  "%s for inbound connections." % firewall.program(), "warn")
        return False

    def show_settings(self, s):
        self.address.setText(s.address or "")
        self.exposure.setValue(float(s.exposure_ms))
        self.autoExp.setChecked(bool(s.auto_exposure))
        self.gain.setValue(float(s.gain_db))
        self.autoGain.setChecked(bool(s.auto_gain))
        self.autoWB.setChecked(bool(s.auto_white_balance))
        # not on the page: set by the AI reference image for a coloured light,
        # and kept unless automatic white balance is switched on here
        self._neutral = bool(getattr(s, "neutral_balance", False))
        self.rotation.setCurrentIndex(max(0, self.rotation.findData(
            int(getattr(s, "rotation", 0) or 0) % 360)))
        self._set_combo(self.pixfmt, [], s.pixel_format or "auto")
        self.gamma.setValue(0.0 if s.gamma in (None, "") else float(s.gamma))
        self.black.setValue(-1.0 if s.black_level in (None, "") else float(s.black_level))
        self.revX.setChecked(bool(s.reverse_x))
        self.revY.setChecked(bool(s.reverse_y))
        self.trigger.setCurrentIndex(max(0, self.trigger.findData(s.trigger or "free")))
        self._set_combo(self.trigSource, ["Line1", "Line2", "Line3"], s.trigger_source or "Line1")
        self._set_combo(self.trigEdge, ["RisingEdge", "FallingEdge"], s.trigger_edge or "RisingEdge")
        self.trigDelay.setValue(float(s.trigger_delay_us or 0))
        self.rate.setValue(float(s.frame_rate or 0))
        mode = "keep" if not s.roi else ("full" if s.roi == "full" else "custom")
        self.roiMode.setCurrentIndex(self.roiMode.findData(mode))
        if mode == "custom":
            for w, v in zip((self.roiX, self.roiY, self.roiW, self.roiH), s.roi):
                w.setValue(int(v))
        self.packet.setValue(int(s.packet_size or 0))
        self.ipd.setValue(int(s.inter_packet_delay or 0))
        self.heartbeat.setValue(int(s.heartbeat_ms or 0))
        self._sync_enabled()

    @staticmethod
    def _set_combo(combo, options, value):
        if combo.count() == 0:
            for o in options:
                combo.addItem(o, o)
        i = combo.findData(value)
        if i < 0:
            combo.addItem(value, value)
            i = combo.count() - 1
        combo.setCurrentIndex(i)

    def settings(self):
        s = CameraSettings()
        s.address = self.address.text().strip()
        s.exposure_ms = float(self.exposure.value())
        s.auto_exposure = self.autoExp.isChecked()
        s.gain_db = float(self.gain.value())
        s.auto_gain = self.autoGain.isChecked()
        s.auto_white_balance = self.autoWB.isChecked()
        s.neutral_balance = getattr(self, "_neutral", False) and not s.auto_white_balance
        s.rotation = int(self.rotation.currentData() or 0)
        s.pixel_format = self.pixfmt.currentData() or "auto"
        s.gamma = float(self.gamma.value()) or None
        s.black_level = None if self.black.value() < 0 else float(self.black.value())
        s.reverse_x, s.reverse_y = self.revX.isChecked(), self.revY.isChecked()
        s.trigger = self.trigger.currentData() or "free"
        s.trigger_source = self.trigSource.currentText().strip() or "Line1"
        s.trigger_edge = self.trigEdge.currentData() or self.trigEdge.currentText() or "RisingEdge"
        s.trigger_delay_us = float(self.trigDelay.value())
        s.frame_rate = float(self.rate.value())
        mode = self.roiMode.currentData()
        s.roi = None if mode == "keep" else "full" if mode == "full" else [
            int(self.roiX.value()), int(self.roiY.value()),
            int(self.roiW.value()), int(self.roiH.value())]
        if isinstance(s.roi, list) and (s.roi[2] <= 0 or s.roi[3] <= 0):
            s.roi = None
        s.packet_size = int(self.packet.value())
        s.inter_packet_delay = int(self.ipd.value())
        s.heartbeat_ms = int(self.heartbeat.value())
        return s

    def _sync_enabled(self):
        hw = self.trigger.currentData() == "hardware"
        for w in (self.trigSource, self.trigEdge, self.trigDelay):
            w.setEnabled(hw)
        for w in (self.roiX, self.roiY, self.roiW, self.roiH):
            w.setEnabled(self.roiMode.currentData() == "custom")
        self.exposure.setEnabled(not self.autoExp.isChecked())
        self.gain.setEnabled(not self.autoGain.isChecked())

    # ---- the camera ----------------------------------------------------
    def find_cameras(self):
        try:
            found = camera_mod.list_devices()
        except Exception as e:
            found, err = [], str(e)
        else:
            err = ""
        if not found:
            self._say("Discovery found no camera%s. Type the camera's IP address - it works even "
                      "where discovery does not (a firewall or a second network card often "
                      "blocks discovery)." % ((" (%s)" % err) if err else ""), "warn")
            return []
        m = QMenu(self)
        for d in found:
            label = "%s   serial %s%s" % (d["model"], d["serial"],
                                          ("   %s" % d["ip"]) if d.get("ip") else "")
            m.addAction(label, lambda d=d: self.address.setText(d.get("ip") or d["serial"]))
        self._say("Found %d camera(s) - choose one." % len(found))
        if self.isVisible():
            m.popup(self.findBtn.mapToGlobal(self.findBtn.rect().bottomLeft()))
        self._menu = m
        return found

    def connect_camera(self, wait=False):
        """Save, then open the camera through the application's shared
        camera (so inspections and Live View keep using one connection) and
        read its ranges and current values."""
        import time
        idx = self.slot.value()
        self.dlg.apply_camera()
        self._release()
        self._say("Connecting to %s..." % (self.address.text().strip() or "the first camera found"))
        if self.dlg.open_camera is None:
            self._say("No camera can be opened from here.", "warn")
            return False
        try:
            self.handle = self.dlg.open_camera(idx)
        except Exception as e:
            self._say("The camera would not open: %s" % e, "fail")
            return False
        self._poll_started = time.time()
        self._poll.start(150)
        if wait:
            from PySide6.QtWidgets import QApplication
            t0 = time.time()
            while self._poll.isActive() and time.time() - t0 < 15:
                QApplication.processEvents()
                time.sleep(0.02)
        return True

    def _check_connected(self):
        import time
        h = self.handle
        if h is None:
            self._poll.stop()
            return
        if h.backend is None:
            if not h.is_alive() or time.time() - self._poll_started > 20:
                self._poll.stop()
                self._say(h.last_error or "The camera did not start.", "fail")
            return
        self._poll.stop()
        try:
            d = h.describe() if hasattr(h, "describe") else None
        except Exception:
            d = None
        self.desc = d
        if not d:
            self._say("Connected (%s) - its settings cannot be read on this kind of camera."
                      % h.backend, "pass")
            return
        self._show_ranges(d)
        self._show_ip(d.get("ip"))
        info = d.get("info") or {}
        fps = (d.get("resulting_frame_rate") or {}).get("value")
        w, h_ = (d.get("width") or {}).get("value"), (d.get("height") or {}).get("value")
        self._say("Connected - Basler %s, serial %s, %s x %s px%s" % (
            info.get("model", "?"), info.get("serial", "?"), w, h_,
            (", %.1f frames/s" % fps) if fps else ""), "pass")

    def _show_ranges(self, d):
        e = d.get("exposure_us")
        if e and e.get("min") is not None:
            self.exposure.setRange(max(0.001, e["min"] / 1000.0), min(e["max"] / 1000.0, 100000.0))
        gn = d.get("gain")
        if gn and gn.get("min") is not None:
            self.gain.setRange(float(gn["min"]), float(gn["max"]))
            raw = d.get("gain_unit") == "raw"
            self.gain.setDecimals(0 if raw else 1)
            self.gainUnit.setText("camera units" if raw else "dB")
        pf = d.get("pixel_format")
        if pf and pf.get("options"):
            cur = self.pixfmt.currentData() or "auto"
            self.pixfmt.blockSignals(True)
            self.pixfmt.clear()
            self.pixfmt.addItem("Automatic", "auto")
            for o in pf["options"]:
                self.pixfmt.addItem(o, o)
            self.pixfmt.setCurrentIndex(max(0, self.pixfmt.findData(cur)))
            self.pixfmt.blockSignals(False)
        for combo, key in ((self.trigSource, "trigger_source"), (self.trigEdge, "trigger_activation")):
            en = d.get(key)
            if en and en.get("options"):
                cur = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                for o in en["options"]:
                    combo.addItem(o, o)
                i = combo.findData(cur)
                combo.setCurrentIndex(i if i >= 0 else 0)
                combo.blockSignals(False)
        wm, hm = d.get("width_max"), d.get("height_max")
        if wm and hm:
            for w, m in ((self.roiX, wm), (self.roiW, wm), (self.roiY, hm), (self.roiH, hm)):
                w.setMaximum(int(m))
            if self.roiMode.currentData() != "custom":
                for w, key in ((self.roiX, "offset_x"), (self.roiY, "offset_y"),
                               (self.roiW, "width"), (self.roiH, "height")):
                    v = (d.get(key) or {}).get("value")
                    if v is not None:
                        w.setValue(int(v))
        for widget, key, cast in ((self.black, "black_level", float), (self.gamma, "gamma", float),
                                  (self.rate, "frame_rate", float), (self.packet, "packet_size", int)):
            rng = d.get(key)
            if rng and rng.get("max") is not None:
                widget.setMaximum(cast(rng["max"]))

    def _show_ip(self, ip):
        if not ip:
            self.ipInfo.setText("This camera has no IP settings (not a GigE camera).")
            self.keepIpBtn.setEnabled(False)
            return
        now = "%s / %s" % (ip["current_ip"], ip["current_mask"])
        kept = "%s / %s" % (ip["stored_ip"], ip["stored_mask"]) if ip["stored_on"] else             "none (it falls back to an automatic address)"
        if ip["keeps_current"]:
            self.ipInfo.setText("Now %s - kept after a restart." % now)
            theme.set_tone(self.ipInfo, "idle")
        else:
            self.ipInfo.setText("Now %s, but after a restart: %s. The camera will not be "
                                "reachable after a power cycle - press Keep this address."
                                % (now, kept))
            theme.set_tone(self.ipInfo, "warn")
        self.keepIpBtn.setEnabled(not ip["keeps_current"])

    def keep_ip(self):
        h = self.handle
        if h is None or getattr(h, "backend", None) != "pylon":
            self._say("Connect to the camera first.", "warn")
            return None
        try:
            ip = h.call(lambda pc: pc.store_ip() if pc else None)
        except Exception as e:
            self._say("The address could not be stored: %s" % e, "fail")
            return None
        self._show_ip(ip)
        if ip and ip["keeps_current"]:
            self._say("Stored in the camera: %s / %s - it keeps this address after a "
                      "restart." % (ip["stored_ip"], ip["stored_mask"]), "pass")
        return ip

    def _balance_once(self):
        h = self.handle
        if h is None or getattr(h, "backend", None) != "pylon":
            self._say("Connect to the camera first.", "warn")
            return
        try:
            ok = h.call(lambda pc: pc.white_balance_once() if pc else False)
            msg = ""
        except Exception as e:
            ok, msg = False, str(e)
        self.autoWB.setChecked(False)
        self._say("White balanced on the current picture." if ok else
                  "This camera cannot balance white%s." % ((": " + msg) if msg else ""),
                  "pass" if ok else "warn")

    def _live(self):
        self.dlg.apply_camera()
        qb = self.dlg.qb
        if qb is not None and hasattr(qb, "show_live_camera"):
            qb.show_live_camera()

    def _release(self):
        self._poll.stop()
        if self.handle is not None:
            try:
                self.handle.stop()
            except Exception:
                pass
        self.handle = None


# ---------------------------------------------------------------- Results
class ResultsPage(Page):
    def __init__(self, dialog):
        super().__init__("Results", "Where each inspected part's picture is kept.")
        self.dlg = dialog
        g = self.add(FormGroup("Result pictures", footer=(
            "Pictures are written in the background, so saving never slows an inspection. "
            "Each day has its own folder, and a scanned part its own folder inside it: "
            "2026-11-17\\143502_118_Top_REJECT.jpg, or "
            "2026-11-17\\FIC12345678\\143502_118_Top_REJECT.jpg. "
            "Find one with Find picture (Ctrl+F).")))
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("Not saved")
        self.folder.setFixedWidth(300)
        browse = QPushButton("Choose...")
        browse.clicked.connect(self._browse)
        self.openBtn = QPushButton("Show")
        self.openBtn.setToolTip("Open the folder in Explorer")
        self.openBtn.clicked.connect(self._open)
        g.add_row("Folder", _row(self.folder, browse, self.openBtn))
        self.which = _combo([("Every part", "all"), ("Rejected parts only", "fail"),
                             ("Accepted parts only", "pass")], 220)
        g.add_row("Save", self.which)
        self.annotated = make_switch(True)
        g.add_row("Annotated picture", self.annotated,
                  help="The result page's boxes and verdict on the full-resolution image")
        self.labels = _combo([("Every check", True), ("Failed checks only", "failed"),
                              ("None - outlines only", False)], 220)
        g.add_row("Results in words", self.labels, help="OK / NG, the text read, sizes in mm - beside each check")
        self.banner = make_switch(True)
        g.add_row("Verdict banner above the picture", self.banner)
        self.original = make_switch(False)
        g.add_row("Original picture", self.original,
                  help="Exactly what the camera delivered - full resolution, nothing drawn")
        self.format = _combo([("JPEG - best quality", "jpg"),
                              ("PNG - lossless", "png")], 220)
        g.add_row("File format", self.format)
        self.quality = _ispin(70, 100, 1, width=90)
        g.add_row("JPEG quality", self.quality)
        self.format.currentIndexChanged.connect(
            lambda _i: self.quality.setEnabled(self.format.currentData() == "jpg"))
        self.annotated.toggled.connect(lambda on: (self.labels.setEnabled(on),
                                                   self.banner.setEnabled(on)))

        # OK / NG colours and line thickness: on screen, in saved pictures
        from vision_station.ui.accent_picker import AccentPicker, interlock
        g = self.add(FormGroup("Result colours and lines", footer=(
            "Used on screen and in saved result pictures. Solid colours only. "
            "OK and NG can never share a colour: the one chosen for OK is greyed out "
            "for NG, and the other way round. Pictures already saved keep their colours.")))
        both = (_look.DEFAULT_OK, _look.DEFAULT_NG)
        self.okColour = AccentPicker(_look.DEFAULT_OK, both, "OK")
        self.ngColour = AccentPicker(_look.DEFAULT_NG, both, "NG")
        interlock(self.okColour, self.ngColour)
        g.add_row("OK colour", self.okColour)
        g.add_row("NG colour", self.ngColour)
        self.lineScale = _combo([("%s  (x%g)" % (LINE_SCALE_NAMES[v], v), v)
                                 for v in _look.LINE_SCALES], 220)
        self.lineScale.setCurrentIndex(self.lineScale.findData(1.0))
        g.add_row("Line thickness", self.lineScale,
                  help="Outlines of every check; the words keep their size")
        resetBtn = QPushButton("Restore default colours and lines")
        resetBtn.setAutoDefault(False)
        resetBtn.clicked.connect(self._reset_look)
        g.add_row("", resetBtn)

        # what an exported job file (File > Export Job File) does with results
        self.exported = None
        qb = dialog.qb
        o = None
        if qb is not None and hasattr(qb, "_output_spec_holder"):
            try:
                o = dict(qb._output_spec_holder()[0].get("output") or
                         {"folder": "results", "show": True, "only_failures": False})
            except Exception:
                o = None
        if o is not None:
            g = self.add(FormGroup("Exported job files", footer=(
                "What a job file exported from this application does with each result - "
                "saved with the application.")))
            self.expFolder = QLineEdit(o.get("folder", ""))
            self.expFolder.setPlaceholderText("Not saved")
            self.expFolder.setFixedWidth(300)
            b2 = QPushButton("Choose...")
            b2.clicked.connect(lambda: self._browse(self.expFolder))
            g.add_row("Folder for annotated pictures", _row(self.expFolder, b2))
            self.expFail = make_switch(bool(o.get("only_failures", False)))
            g.add_row("Rejected parts only", self.expFail)
            self.expShow = make_switch(bool(o.get("show", True)))
            g.add_row("Show each result in a window", self.expShow)
            self.exported = o
        self.finish()

    def show_values(self, d):
        self.folder.setText(d.get("folder", ""))
        self.which.setCurrentIndex(max(0, self.which.findData(d.get("which", "all"))))
        self.annotated.setChecked(bool(d.get("annotated", True)))
        self.original.setChecked(bool(d.get("original", False)))
        lab = d.get("labels", True)
        self.labels.setCurrentIndex(1 if lab == "failed" else 0 if lab is True else 2)
        self.banner.setChecked(bool(d.get("banner", True)))
        self.format.setCurrentIndex(max(0, self.format.findData(d.get("format", "jpg"))))
        self.quality.setValue(int(d.get("quality", 97)))
        self.quality.setEnabled(self.format.currentData() == "jpg")
        self.labels.setEnabled(self.annotated.isChecked())
        self.banner.setEnabled(self.annotated.isChecked())
        self._show_look(d.get("ok_colour"), d.get("ng_colour"), d.get("line_scale", 1.0))

    def _show_look(self, ok, ng, scale):
        from vision_station.ui.accent_picker import set_pair
        set_pair(self.okColour, self.ngColour, ok or _look.DEFAULT_OK,
                 ng or _look.DEFAULT_NG, _look.DEFAULT_NG)
        s = _look.line_scale(scale)
        i = self.lineScale.findData(s)
        if i < 0:                                   # a value between the steps: the nearest
            i = min(range(self.lineScale.count()),
                    key=lambda k: abs(self.lineScale.itemData(k) - s))
        self.lineScale.setCurrentIndex(i)

    def _reset_look(self):
        self._show_look(_look.DEFAULT_OK, _look.DEFAULT_NG, 1.0)

    KEEPS = "the folders pictures are saved in"

    def restore_defaults(self):
        self.show_values(dict(RESULT_DEFAULTS, folder=self.folder.text().strip()))
        if self.exported is not None:
            self.expFail.setChecked(False)
            self.expShow.setChecked(True)

    def values(self):
        return {"folder": self.folder.text().strip(), "which": self.which.currentData(),
                "annotated": self.annotated.isChecked(), "original": self.original.isChecked(),
                "labels": self.labels.currentData(), "banner": self.banner.isChecked(),
                "format": self.format.currentData(), "quality": int(self.quality.value()),
                "ok_colour": self.okColour.colour(), "ng_colour": self.ngColour.colour(),
                "line_scale": float(self.lineScale.currentData())}

    def exported_values(self):
        if self.exported is None:
            return None
        return {"folder": self.expFolder.text().strip(), "show": self.expShow.isChecked(),
                "only_failures": self.expFail.isChecked()}

    def _browse(self, field=None):
        field = field if isinstance(field, QLineEdit) else self.folder
        d = QFileDialog.getExistingDirectory(self, "Choose a folder",
                                             field.text() or os.path.expanduser("~"))
        if d:
            field.setText(os.path.normpath(d))

    def _open(self):
        d = self.folder.text().strip()
        if d:
            os.makedirs(d, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))


# ---------------------------------------------------------------- Line connection
class LinePage(Page):
    def __init__(self, dialog):
        super().__init__("Line connection", "A PLC or robot triggers inspections and reads "
                                            "results over TCP/IP. Saved with the application.")
        self.dlg = dialog
        qb = dialog.qb
        c = {}
        if qb is not None and hasattr(qb, "comms_settings"):
            try:
                c = qb.comms_settings()
            except Exception:
                c = {}
        self.available = bool(c)
        g = self.add(FormGroup("TCP/IP"))
        self.enabled = make_switch(bool(c.get("enabled")))
        g.add_row("Listen for the line", self.enabled,
                  help="The server starts whenever this application is open")
        self.port = _ispin(1, 65535, width=100)
        self.port.setValue(int(c.get("port", 3000)))
        g.add_row("Port", self.port)
        self.broadcast = make_switch(bool(c.get("broadcast")))
        g.add_row("Send every result to all clients", self.broadcast)
        g = self.add(FormGroup("Commands", footer=(
            "One command per line. GET reads a value from the last run; a trigger that "
            "arrives during an inspection is answered BUSY.")))
        cmds = QLabel("TRIGGER  or  TRIGGER &lt;job&gt;  &rarr;  RESULT &lt;job&gt; PASS|FAIL|ERROR "
                      "&lt;ms&gt; &lt;failed&gt;, then DONE &lt;verdict&gt;<br>"
                      "GET &lt;job&gt;.&lt;check&gt;.&lt;value&gt;<br>STATUS &nbsp; PING &nbsp; VERSION")
        cmds.setObjectName("mono")
        cmds.setTextInteractionFlags(Qt.TextSelectableByMouse)
        g.add_widget(cmds)
        if not self.available:
            for w in (self.enabled, self.port, self.broadcast):
                w.setEnabled(False)
        self.finish()

    def values(self):
        return {"enabled": self.enabled.isChecked(), "port": int(self.port.value()),
                "broadcast": self.broadcast.isChecked()}

    KEEPS = "whether the line is listened for, and its port (the PLC is set up for it)"

    def restore_defaults(self):
        if not self.available:
            return False
        self.broadcast.setChecked(False)


# ---------------------------------------------------------------- Security
class SecurityPage(Page):
    def __init__(self, dialog):
        super().__init__("Security", "Lock the application to running parts, so settings and "
                                     "inspections cannot be changed at the line.")
        qb = dialog.qb
        g = self.add(FormGroup("Operator Mode", footer=(
            "In Operator Mode only Run, Continuous and the results are available. "
            "Leaving it asks for the supervisor password, when one is set.")))
        self.operatorBtn = QPushButton("Turn On Operator Mode")
        self.operatorBtn.clicked.connect(lambda: qb is not None and (
            qb.toggle_operator_mode(), dialog.close()))
        g.add_row("Operator Mode", self.operatorBtn)
        self.passwordBtn = QPushButton("Set Password...")
        self.passwordBtn.clicked.connect(lambda: qb is not None and qb.set_supervisor_password())
        g.add_row("Supervisor password", self.passwordBtn)
        if qb is None:
            self.operatorBtn.setEnabled(False)
            self.passwordBtn.setEnabled(False)
            self.restoreBtn.setEnabled(False)
        self.qb = qb
        self.finish()

    APPLIES_NOW = True

    def ask_restore(self):
        return self.restore_defaults()

    def restore_defaults(self, password=None):
        qb = self.qb
        if qb is None:
            return False
        stored = qb._settings().value("supervisor_password", "")
        if not stored:
            QMessageBox.information(self, "Restore defaults",
                                    "Security is at its defaults: no supervisor password.")
            return True
        if password is None:
            password, ok = QInputDialog.getText(
                self, "Restore defaults", "Restoring the defaults removes the supervisor "
                "password.\n\nSupervisor password:", QLineEdit.Password)
            if not ok:
                return False
        from vision_station.ui.main_window import _check_password
        if not _check_password(password, stored):
            QMessageBox.warning(self, "Restore defaults", "That password is not right.")
            return False
        qb.set_supervisor_password("")
        return True


# ---------------------------------------------------------------- Barcode
class BarcodePage(Page):
    def __init__(self, dialog):
        super().__init__("Barcode", "Inspect a part only after its barcode is scanned, and "
                                    "keep its pictures in a folder named after it.")
        from vision_station.core import barcode as rules
        from vision_station.ui.barcode_panel import BarcodeForm
        self.form = BarcodeForm(self, load_result_settings(dialog.qs).get("folder", ""))
        self.form.show_values(rules.load(dialog.qs))
        self.lay.addWidget(self.form)
        self.finish()

    def values(self):
        return self.form.values()

    KEEPS = "how the scanner is connected (keyboard or COM port, baud) and the folder"

    def restore_defaults(self):
        from vision_station.core import barcode as rules
        v = self.form.values()
        self.form.show_values(dict(rules.DEFAULTS, **{k: v[k] for k in
                                                      ("source", "port", "baud", "folder")
                                                      if k in v}))


# ---------------------------------------------------------------- Light
class LightPage(Page):
    """The external light's controller on a serial port: when it is on, how
    bright -- by hand or adjusted from the camera's pictures -- and buttons
    to try it."""

    def __init__(self, dialog):
        super().__init__("Light", "The external light, switched and dimmed through its "
                                  "controller's RS232 port.")
        from vision_station.hardware import light as L
        self.dlg = dialog
        self._loading = False
        g = self.add(FormGroup("Light controller", footer=(
            "HIFLY DCS and other controllers speaking the same protocol (OPT's), at 9600 "
            "baud. When Vision Station closes, the controller is handed back as it was "
            "found - its own brightness, light on - and works on its own again.")))
        self.enabled = make_switch(False)
        g.add_row("Control the light from Vision Station", self.enabled)
        self.port = QComboBox()
        self.port.setEditable(True)
        self.port.setFixedWidth(320)
        for dev, desc in L.list_ports():
            self.port.addItem("%s   %s" % (dev, desc.replace("(%s)" % dev, "").strip()), dev)
        self.findBtn = QPushButton("Find")
        self.findBtn.setToolTip("Ask every serial port for a light controller (a brightness "
                                "read - nothing is switched)")
        self.findBtn.clicked.connect(self.find)
        g.add_row("Serial port", _row(self.port, self.findBtn))
        self.channels = _ispin(1, L.CHANNELS_MAX, 1, width=90)
        self.channels.valueChanged.connect(self._show_channels)
        g.add_row("Channels", self.channels, help="How many lights the controller drives "
                                                  "(1C in its model name: one)")
        self.state = QLabel("")
        self.state.setObjectName("statusLine")
        self.state.setWordWrap(True)
        self.onBtn = QPushButton("Light on")
        self.offBtn = QPushButton("Light off")
        self.onBtn.clicked.connect(lambda: self.test(True))
        self.offBtn.clicked.connect(lambda: self.test(False))
        st = QHBoxLayout()
        st.addWidget(self.state, 1)
        st.addWidget(self.onBtn)
        st.addWidget(self.offBtn)
        g.add_widget(st)

        g = self.add(FormGroup("When it is on", footer=(
            "Always on off: the light comes on for each Run that takes its picture from the "
            "camera, stays on through a continuous run and while Live Camera is open, and "
            "is off the rest of the time.")))
        self.alwaysOn = make_switch(True)
        g.add_row("Always on", self.alwaysOn, help="On for as long as Vision Station runs")
        self.onBefore = _ispin(0, 3000, 50, " ms", width=120)
        self._onBeforeRow = g.add_row(
            "Light on before the picture", self.onBefore,
            help="Time for the light to settle before the camera takes the picture")
        self.offAfter = _ispin(0, 10000, 100, " ms", width=120)
        self._offAfterRow = g.add_row(
            "Keep on after the picture", self.offAfter,
            help="How long the light stays on once the picture is taken")
        self.alwaysOn.toggled.connect(self._enable_timing)

        g = self.add(FormGroup("Brightness", footer=(
            "0 to 255 - drag or type; the light follows as you change it. Off: the "
            "brightness set on the controller's own panel is left alone.")))
        self.setBright = make_switch(False)
        g.add_row("Set the brightness from Vision Station", self.setBright)
        self.levels, self.sliders, self._level_rows = [], [], []
        for ch in range(1, L.CHANNELS_MAX + 1):
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 255)
            sl.setPageStep(16)
            sl.setMinimumWidth(240)
            sp = _ispin(0, 255, 1, width=80)
            sl.valueChanged.connect(sp.setValue)
            sp.valueChanged.connect(sl.setValue)
            sp.valueChanged.connect(lambda v, ch=ch: self._level_changed(ch, v))
            self.levels.append(sp)
            self.sliders.append(sl)
            self._level_rows.append(g.add_row("Channel %d" % ch, _row(sl, sp)))
        self.readBtn = QPushButton("Read from the controller")
        self.readBtn.clicked.connect(self.read)
        g.add_row("", self.readBtn)
        self.setBright.toggled.connect(self._enable_levels)
        # the light follows the slider, a moment after it stops moving
        self._live = QTimer(self)
        self._live.setSingleShot(True)
        self._live.timeout.connect(self._send_levels)
        self._pending = {}

        g = self.add(FormGroup("Automatic brightness", footer=(
            "Sets the light from the camera's picture, so every picture is as bright as you "
            "aim for - whatever the light's age or the daylight. Adjust now does it once, "
            "in a few seconds.")))
        self.auto = make_switch(False)
        g.add_row("Adjust automatically on every run", self.auto,
                  help="Each part's picture sets the light for the next part")
        self.target = _combo([("The teach picture", "reference"),
                              ("A brightness I choose", "fixed")], 220)
        g.add_row("Make pictures as bright as", self.target)
        self.targetPct = _ispin(2, 95, 1, " %", width=80)
        self.targetSlider = QSlider(Qt.Horizontal)
        self.targetSlider.setRange(2, 95)
        self.targetSlider.setMinimumWidth(200)
        self.targetSlider.valueChanged.connect(self.targetPct.setValue)
        self.targetPct.valueChanged.connect(self.targetSlider.setValue)
        self._targetRow = g.add_row("Brightness", _row(self.targetSlider, self.targetPct),
                                    help="0 % black, 100 % white")
        self.adjustBtn = QPushButton("Adjust now")
        self.adjustBtn.setToolTip("Find the light's level from the camera now")
        self.adjustBtn.clicked.connect(self.adjust_now)
        self.adjustState = QLabel("")
        self.adjustState.setObjectName("rowHelp")
        self.adjustState.setWordWrap(True)
        ar = QHBoxLayout()
        ar.addWidget(self.adjustState, 1)
        ar.addWidget(self.adjustBtn)
        g.add_widget(ar)
        self.target.currentIndexChanged.connect(lambda _i: self._show_target())
        self._adjust = None
        self._adjust_poll = QTimer(self)
        self._adjust_poll.timeout.connect(self._adjust_tick)
        self.finish()

    # -- values ---------------------------------------------------------------
    def show_values(self, d):
        self._loading = True
        try:
            self.enabled.setChecked(bool(d.get("enabled")))
            port = d.get("port") or ""
            i = self.port.findData(port)
            if i < 0 and port:
                self.port.addItem(port, port)
                i = self.port.findData(port)
            if i >= 0:
                self.port.setCurrentIndex(i)
            self.channels.setValue(int(d.get("channels") or 1))
            self.alwaysOn.setChecked(bool(d.get("always_on", True)))
            self.setBright.setChecked(bool(d.get("set_brightness")))
            for sp, v in zip(self.levels, list(d.get("brightness") or []) + [255] * 4):
                sp.setValue(int(v))
            self.target.setCurrentIndex(max(0, self.target.findData(d.get("target",
                                                                          "reference"))))
            self.targetPct.setValue(int(d.get("target_percent", 45)))
            self.auto.setChecked(bool(d.get("auto")))
            self.onBefore.setValue(int(d.get("on_before_ms", 150)))
            self.offAfter.setValue(int(d.get("off_after_ms", 300)))
        finally:
            self._loading = False
        self._show_channels()
        self._enable_levels()
        self._enable_timing()
        self._show_target()
        self._show_state()

    KEEPS = "whether Vision Station controls the light, its COM port and number of channels"

    def restore_defaults(self):
        from vision_station.hardware import light as L
        v = self.values()
        self.show_values(dict(L.DEFAULTS, enabled=v["enabled"], port=v["port"],
                              channels=v["channels"]))

    def values(self):
        port = self.port.currentData()
        typed = self.port.currentText().split("   ")[0].strip()
        if port is None or typed != port:
            port = typed
        return {"enabled": self.enabled.isChecked(), "port": port,
                "channels": int(self.channels.value()),
                "always_on": self.alwaysOn.isChecked(),
                "set_brightness": self.setBright.isChecked() or self.auto.isChecked(),
                "brightness": [int(sp.value()) for sp in self.levels],
                "auto": self.auto.isChecked(),
                "target": self.target.currentData(),
                "target_percent": int(self.targetPct.value()),
                "on_before_ms": int(self.onBefore.value()),
                "off_after_ms": int(self.offAfter.value())}

    def _show_channels(self):
        n = int(self.channels.value())
        for i, row in enumerate(self._level_rows):
            row.setVisible(i < n)
            # and the line drawn above it, or hidden rows leave their lines
            lay = row.parentWidget().layout() if row.parentWidget() else None
            k = lay.indexOf(row) if lay is not None else -1
            sep = lay.itemAt(k - 1).widget() if k > 0 else None
            if sep is not None and sep.objectName() == "formRowSep":
                sep.setVisible(i < n)

    def _enable_timing(self):
        switched = not self.alwaysOn.isChecked()
        self._onBeforeRow.setEnabled(switched)
        self._offAfterRow.setEnabled(switched)

    def _show_target(self):
        self._targetRow.setVisible(self.target.currentData() == "fixed")

    def _enable_levels(self):
        on = self.setBright.isChecked()
        for sp in self.levels:
            sp.parentWidget().setEnabled(on)

    def _say(self, text, tone=None):
        self.state.setText(text)
        theme.set_tone(self.state, tone or "idle")

    # -- the controller ---------------------------------------------------------
    def _light(self):
        """The application's own light connection -- a serial port has one
        user at a time."""
        qb = self.dlg.qb
        if qb is None or not hasattr(qb, "station"):
            return None
        return qb.station.light

    def _show_state(self):
        lt = self._light()
        if lt is None or not lt.enabled:
            self._say("Not controlled by Vision Station")
        elif lt.error:
            self._say(lt.error, "fail")
        elif lt.ctl is not None:
            self._say("Connected on %s - light %s, brightness %d" % (
                lt.cfg["port"], {True: "on", False: "off", None: "as it was"}[lt.is_on],
                lt.level()), "pass")
        else:
            self._say("Not connected")

    def _level_changed(self, ch, v):
        if self._loading or not self.setBright.isChecked():
            return
        self._pending[ch] = v
        self._live.start(120)

    def _send_levels(self):
        lt = self._light()
        if lt is None or not lt.enabled or lt.cfg.get("port") != self.values()["port"]:
            return                        # not connected with these settings yet
        pending, self._pending = self._pending, {}
        for ch, v in pending.items():
            if ch <= int(self.channels.value()) and not lt.set_channel(ch, v):
                self._say(lt.error or "the brightness was not sent", "fail")
                return

    def find(self):
        from vision_station.hardware import light as L
        lt = self._light()
        if lt is not None and lt.ctl is not None:
            port = lt.cfg["port"]            # connected: it is answering there
        else:
            self._say("Asking every serial port...")
            QApplication.processEvents()
            port = L.find_port()
        if not port:
            self._say("No light controller answered on any serial port. Is it switched on, "
                      "and is its cable in?", "fail")
            return None
        i = self.port.findData(port)
        if i < 0:
            self.port.addItem(port, port)
            i = self.port.findData(port)
        self.port.setCurrentIndex(i)
        self._say("A light controller answers on %s." % port, "pass")
        return port

    def _apply_here(self):
        lt = self._light()
        if lt is None:
            self._say("Only from the application's Settings.", "warn")
            return None
        lt.configure(dict(self.values(), enabled=True))
        return lt

    def test(self, on):
        from vision_station.hardware.light import LightError
        lt = self._apply_here()
        if lt is None:
            return False
        try:
            lt.test(on)
        except LightError as e:
            self._say(str(e), "fail")
            return False
        self._say("Light %s (%s). Done or Apply puts the rules above back in charge."
                  % ("on" if on else "off", lt.cfg["port"]), "pass")
        return True

    def read(self):
        from vision_station.hardware.light import LightError
        lt = self._apply_here()
        if lt is None:
            return None
        try:
            got = lt.read()
        except LightError as e:
            self._say(str(e), "fail")
            return None
        self._loading = True
        try:
            for ch, v in got.items():
                self.levels[ch - 1].setValue(v)
        finally:
            self._loading = False
        self._say("The controller has: %s" % ", ".join("channel %d at %d" % kv
                                                       for kv in sorted(got.items())), "pass")
        return got

    # -- automatic --------------------------------------------------------------
    def _reference(self):
        """The teach picture of the first job that takes its picture from the
        camera -- what Match the teach picture aims for."""
        qb = self.dlg.qb
        st = getattr(qb, "station", None)
        if st is None:
            return None
        for j in st.jobs:
            src = st.sources.get(j["pid"])
            if src is not None and src.mode == "camera":
                p = st.projects.get(j["pid"])
                return getattr(p, "reference", None)
        return None

    def adjust_now(self):
        import threading
        import time
        if self._adjust is not None:
            self._adjust["stop"] = True          # pressed again: stop
            return False
        lt = self._apply_here()
        if lt is None:
            return False
        if lt.ctl is None:
            self._say(lt.error or "the light controller is not connected", "fail")
            return False
        if self.target.currentData() == "reference" and self._reference() is None:
            self.adjustState.setText("No job takes its picture from the camera, so there is "
                                     "no teach picture to match - choose a fixed brightness.")
            return False
        if self.dlg.open_camera is None:
            self.adjustState.setText("No camera can be opened from here.")
            return False
        try:
            cam = self.dlg.open_camera(0)
        except Exception as e:
            self.adjustState.setText("The camera would not open: %s" % e)
            return False
        ref = self._reference()
        job = {"stop": False, "progress": None, "result": None, "error": None,
               "cam": cam, "done": False}

        def grab():
            t0 = time.time()
            while getattr(cam, "backend", None) is None and time.time() - t0 < 8:
                time.sleep(0.05)              # the camera may still be opening
            return cam.frame_after(None, settle=3, timeout_s=5.0)

        def work():
            try:
                job["result"] = lt.adjust_now(
                    grab, ref, stop=lambda: job["stop"],
                    progress=lambda i, n, lv, m, c: job.__setitem__("progress", (i, n, lv, m, c)))
            except Exception as e:
                job["error"] = str(e)
            finally:
                job["done"] = True

        self._adjust = job
        self.adjustBtn.setText("Stop")
        self.adjustState.setText("Adjusting...")
        threading.Thread(target=work, name="light-adjust", daemon=True).start()
        self._adjust_poll.start(100)
        return True

    def _adjust_tick(self):
        job = self._adjust
        if job is None:
            self._adjust_poll.stop()
            return
        if job["progress"] and not job["done"]:
            i, n, lv, m, c = job["progress"]
            self.adjustState.setText("Try %d: level %d gives %.0f %% grey, %.1f %% white"
                                     % (i, lv, 100 * m, 100 * c))
        if not job["done"]:
            return
        self._adjust_poll.stop()
        self._adjust = None
        self.adjustBtn.setText("Adjust now")
        try:
            job["cam"].stop()
        except Exception:
            pass
        if job["error"]:
            self.adjustState.setText("Not adjusted: %s" % job["error"])
            return
        level, m, c, tries = job["result"]
        self._loading = True
        try:
            self.setBright.setChecked(True)
            for sp in self.levels:
                sp.setValue(level)
        finally:
            self._loading = False
        note = " - the light is at its brightest and the picture is still darker than " \
               "the target" if level >= 254 and m < 0.95 * self._target_value() else ""
        self.adjustState.setText("Level %d: %.0f %% grey, %.1f %% white, after %d pictures%s. "
                                 "Done or Apply keeps it." % (level, 100 * m, 100 * c,
                                                              len(tries), note))

    def _target_value(self):
        from vision_station.hardware import light as L
        return L.target_of(self.values(), self._reference())


# ---------------------------------------------------------------- the window
PAGES = [("general", "General", "sliders"), ("camera", "Camera", "image_source"),
         ("light", "Light", "light"), ("results", "Results", "photo"), ("barcode", "Barcode", "barcode"),
         ("line", "Line connection", "network"), ("security", "Security", "lock")]


class SettingsDialog(QDialog):
    """qb: the main window, or None. qs: a QSettings. open_camera: index ->
    shared camera handle. on_camera(index, settings) and on_results(dict) are
    told about changes."""

    def __init__(self, parent=None, qs=None, open_camera=None, on_camera=None,
                 on_results=None, page="general", qb=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setObjectName("settingsDialog")
        self.resize(900, 700)
        self.setMinimumSize(760, 520)
        self.qb, self.qs = qb, qs
        self.open_camera = open_camera
        self._on_camera, self._on_results = on_camera, on_results

        self.sidebar = QListWidget()
        self.sidebar.setObjectName("prefSidebar")
        self.sidebar.setIconSize(QSize(18, 18))
        self.stack = QStackedWidget()
        self.generalPage = GeneralPage(self)
        self.cameraPage = CameraPage(self)
        self.resultsPage = ResultsPage(self)
        self.linePage = LinePage(self)
        self.securityPage = SecurityPage(self)
        self.barcodePage = BarcodePage(self)
        self.lightPage = LightPage(self)
        self._pages = {"general": self.generalPage, "camera": self.cameraPage,
                       "light": self.lightPage,
                       "results": self.resultsPage, "barcode": self.barcodePage,
                       "line": self.linePage, "security": self.securityPage}
        for key, text, ic in PAGES:
            it = QListWidgetItem(icons.icon(ic), text)
            it.setData(Qt.UserRole, key)
            it.setSizeHint(QSize(0, 34))          # the stylesheet's row height is ignored here
            self.sidebar.addItem(it)
            self.stack.addWidget(self._pages[key])
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)

        side = QWidget()
        side.setObjectName("prefSidebarPane")
        side.setFixedWidth(210)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(6, 12, 6, 12)
        sl.addWidget(self.sidebar)

        foot = QWidget()
        foot.setObjectName("prefFooter")
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(16, 10, 16, 10)
        fl.addStretch(1)
        self.cancelBtn = QPushButton("Cancel")
        self.applyBtn = QPushButton("Apply")
        self.doneBtn = QPushButton("Done")
        self.doneBtn.setDefault(True)
        self.doneBtn.setObjectName("primaryButton")
        for b in (self.cancelBtn, self.applyBtn, self.doneBtn):
            b.setMinimumWidth(84)
            fl.addWidget(b)
        self.cancelBtn.clicked.connect(self.reject)
        self.applyBtn.clicked.connect(self.apply)
        self.doneBtn.clicked.connect(self._ok)

        right = QWidget()
        right.setObjectName("prefContent")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self.stack, 1)
        rl.addWidget(foot)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(side)
        lay.addWidget(right, 1)

        # Only Done is the default button. Qt makes any focused push button in
        # a dialog the default -- the first button on a page showed in accent
        # blue and Enter pressed it.
        for b in self.findChildren(QPushButton):
            b.setAutoDefault(b is self.doneBtn)
        self.cameraPage.load(0)
        self.resultsPage.show_values(load_result_settings(qs))
        from vision_station.hardware import light as _light
        self.lightPage.show_values(_light.load(qs))
        self.show_page(page)

    # ---- pages -------------------------------------------------------
    def show_page(self, key):
        keys = [k for k, _t, _i in PAGES]
        self.sidebar.setCurrentRow(keys.index(key) if key in keys else 0)
        return self._pages.get(key)

    def current_page(self):
        return PAGES[self.sidebar.currentRow()][0]

    # ---- applying ----------------------------------------------------
    def apply_camera(self):
        idx = self.cameraPage.slot.value()
        s = self.cameraPage.settings()
        if self.qs is not None:
            save_camera_settings(self.qs, idx, s)
        if self._on_camera:
            self._on_camera(idx, s)
        return s

    def apply_results(self):
        d = self.resultsPage.values()
        if d["folder"]:
            try:
                os.makedirs(d["folder"], exist_ok=True)
            except OSError as e:
                self.show_page("results")
                raise ValueError("The folder cannot be created: %s" % e)
        if self.qs is not None:
            save_result_settings(self.qs, d)
        if self._on_results:
            self._on_results(d)
        exp = self.resultsPage.exported_values()
        if exp is not None and self.qb is not None and exp != self.resultsPage.exported:
            self.qb.result_output(**exp)
            self.resultsPage.exported = exp
        return d

    def apply_barcode(self):
        from vision_station.core import barcode as rules
        v = self.barcodePage.values()
        if self.qs is not None:
            rules.save(self.qs, v)
        if self.qb is not None and hasattr(self.qb, "apply_barcode"):
            self.qb.apply_barcode(v, save=False)
        return v

    def apply_light(self):
        from vision_station.hardware import light
        v = self.lightPage.values()
        # what the application keeps -- each job's level, the controller's own
        # setting and whether it is owed back -- is not on the page: carried
        lt = self.lightPage._light()
        have = lt.cfg if lt is not None else light.load(self.qs)
        for k in light.KEPT:
            val = have.get(k, light.DEFAULTS[k])
            v[k] = dict(val) if isinstance(val, dict) else val
        if self.qs is not None:
            light.save(self.qs, v)
        if self.qb is not None and hasattr(self.qb, "apply_light"):
            self.qb.apply_light(v, save=False)
        self.lightPage._show_state()
        return v

    def apply_line(self):
        if not self.linePage.available or self.qb is None:
            return None
        v = self.linePage.values()
        if v != {k: self.qb.comms_settings().get(k) for k in v}:
            self.qb.communication(**v)
        return v

    def apply(self):
        self.apply_camera()
        try:
            self.apply_results()
        except ValueError as e:
            self.resultsPage.folder.setFocus()
            self.cameraPage._say(str(e), "fail")
            return False
        self.apply_barcode()
        self.apply_light()
        try:
            self.apply_line()
        except Exception as e:
            self.show_page("line")
            return False
        return True

    def _ok(self):
        if self.apply():
            self.accept()

    def done(self, r):
        self.cameraPage._release()
        if self.lightPage._adjust is not None:
            self.lightPage._adjust["stop"] = True
        super().done(r)
