"""Barcode-gated inspection, on screen.

    BarcodeForm      the rule -- starts with, length -- the scanner, the folder,
                     and a box to try a code against the rule. Used by
                     Settings > Barcode and by the Run view's dialog.
    BarcodeDialog    the form in a window of its own, from the Run view
    BarcodeBar       the strip on the Run view: the code to scan, whether it
                     was accepted, and where the part's pictures went
    SerialScanner    a scanner on a serial (COM) port; USB scanners in
                     keyboard mode need nothing -- they type into the bar
"""
import os

from PySide6.QtCore import QObject, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QSpinBox, QToolButton,
                               QVBoxLayout, QWidget)

from vision_station.core import barcode as rules
from vision_station.ui import icons, theme
from vision_station.ui.design import FormGroup, make_switch


def serial_ports():
    try:
        from PySide6.QtSerialPort import QSerialPortInfo
        return [p.portName() for p in QSerialPortInfo.availablePorts()]
    except Exception:                           # no serial support in this Qt
        return []


class BarcodeForm(QWidget):
    """The barcode settings as FormGroups. values() and show_values() move a
    dict in the shape of barcode.DEFAULTS in and out."""
    changed = Signal()

    def __init__(self, parent=None, results_folder=""):
        super().__init__(parent)
        self.results_folder = results_folder or ""
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        g = FormGroup("Barcode check", footer=(
            "When this is on, the Run view inspects a part only after its barcode has been "
            "scanned and passes the rule below, and keeps that part's result pictures in a "
            "folder named after the barcode. When it is off, inspection works as before. "
            "Runs from a job editor, while teaching, are never held for a barcode."))
        self.enabled = make_switch(False)
        g.add_row("Check the barcode before inspecting", self.enabled)
        lay.addWidget(g.widget)

        g = FormGroup("Rule", footer="Example: starts with FIC, 11 characters - FIC12345678 "
                                     "passes; FIC1234 and ABC12345678 do not.")
        self.prefix = QLineEdit()
        self.prefix.setPlaceholderText("e.g. FIC")
        self.prefix.setFixedWidth(200)
        self.prefix.setObjectName("monoField")
        g.add_row("Starts with", self.prefix, help="The first letters every valid code has. "
                                                   "Empty: any start.")
        self.length = QSpinBox()
        self.length.setRange(0, 200)
        self.length.setSpecialValueText("Any length")
        self.length.setSuffix(" characters")
        self.length.setFixedWidth(200)
        g.add_row("Length", self.length, help="The whole code, the starting letters included")
        self.match_case = make_switch(False)
        g.add_row("Capital letters must match", self.match_case,
                  help="Off: fic12345678 passes a rule written FIC")
        lay.addWidget(g.widget)

        g = FormGroup("Try a code")
        self.trial = QLineEdit()
        self.trial.setPlaceholderText("Scan or type a code to test the rule")
        self.trial.setObjectName("monoField")
        self.trialResult = QLabel("")
        self.trialResult.setObjectName("statusLine")
        self.trialResult.setWordWrap(True)
        box = QVBoxLayout()
        box.addWidget(self.trial)
        box.addWidget(self.trialResult)
        g.add_widget(box)
        lay.addWidget(g.widget)

        g = FormGroup("Scanner and pictures")
        self.auto_run = make_switch(True)
        g.add_row("Inspect as soon as a valid code is scanned", self.auto_run,
                  help="Off: a valid scan waits for Run (F5) or the line's trigger")
        self.source = QComboBox()
        self.source.addItem("USB scanner (keyboard)", "keyboard")
        self.source.addItem("Serial port (COM)", "serial")
        self.source.setFixedWidth(220)
        g.add_row("Scanner", self.source)
        self.port = QComboBox()
        self.port.setEditable(True)
        self.port.addItems(serial_ports())
        self.port.setFixedWidth(110)
        self.baud = QComboBox()
        for b in (9600, 19200, 38400, 57600, 115200):
            self.baud.addItem(str(b), b)
        self.baud.setFixedWidth(100)
        ph = QHBoxLayout()
        ph.setSpacing(8)
        ph.addWidget(self.port)
        ph.addWidget(self.baud)
        self.portRow = g.add_row("Port and speed", ph)
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("the result pictures folder")
        self.folder.setMinimumWidth(240)
        browse = QPushButton("Choose...")
        browse.clicked.connect(self._browse)
        # not shown: a part's pictures go in the folder of Settings > Results
        self.folder.setVisible(False)
        browse.setVisible(False)
        self.whereLabel = QLabel("")
        self.whereLabel.setObjectName("rowHelp")
        self.whereLabel.setWordWrap(True)
        g.add_widget(self.whereLabel)
        lay.addWidget(g.widget)

        for w in (self.prefix, self.trial, self.folder):
            w.textChanged.connect(self._update)
        self.length.valueChanged.connect(self._update)
        for w in (self.enabled, self.match_case, self.auto_run):
            w.toggled.connect(self._update)
        self.source.currentIndexChanged.connect(self._update)
        self.show_values(rules.DEFAULTS)

    # ---- values --------------------------------------------------------
    def show_values(self, cfg):
        cfg = dict(rules.DEFAULTS, **(cfg or {}))
        self.enabled.setChecked(bool(cfg["enabled"]))
        self.prefix.setText(str(cfg["prefix"] or ""))
        self.length.setValue(int(cfg["length"] or 0))
        self.match_case.setChecked(bool(cfg["match_case"]))
        self.auto_run.setChecked(bool(cfg["auto_run"]))
        i = self.source.findData(cfg["source"])
        self.source.setCurrentIndex(max(0, i))
        self.port.setCurrentText(str(cfg["port"] or ""))
        i = self.baud.findData(int(cfg["baud"] or 9600))
        self.baud.setCurrentIndex(max(0, i))
        self.folder.setText(str(cfg["folder"] or ""))
        self._update()

    def values(self):
        return {"enabled": self.enabled.isChecked(),
                "prefix": self.prefix.text().strip(),
                "length": int(self.length.value()),
                "match_case": self.match_case.isChecked(),
                "auto_run": self.auto_run.isChecked(),
                "source": self.source.currentData(),
                "port": self.port.currentText().strip(),
                "baud": int(self.baud.currentData() or 9600),
                "folder": ""}

    # ---- behaviour -----------------------------------------------------
    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Folder for the barcode folders",
                                             self.folder.text() or self.results_folder)
        if d:
            self.folder.setText(d)

    def _update(self, *_a):
        v = self.values()
        on = v["enabled"]
        for w in (self.prefix, self.length, self.match_case, self.auto_run, self.source,
                  self.port, self.baud, self.folder, self.trial):
            w.setEnabled(on)
        self.portRow.setVisible(v["source"] == "serial")
        code = self.trial.text()
        if code.strip():
            ok, why = rules.check(code, v)
            self.trialResult.setText(("Passes - %s" % rules.describe(v)) if ok else
                                     ("Refused: %s" % why))
            theme.set_tone(self.trialResult, "pass" if ok else "fail")
        else:
            self.trialResult.setText("Rule: %s" % rules.describe(v))
            theme.set_tone(self.trialResult, "idle")
        root = v["folder"] or self.results_folder
        self.whereLabel.setText(
            "Each part's pictures go in a folder named after its barcode, e.g. %s" %
            os.path.join(root or "<the results folder>",
                         (v["prefix"] or "CODE") + "...") if on else "")
        self.changed.emit()


class BarcodeDialog(QDialog):
    """The barcode settings from the Run view. on_apply(dict) is told."""

    def __init__(self, parent, cfg, on_apply, results_folder=""):
        super().__init__(parent)
        self.setWindowTitle("Barcode check")
        self.setObjectName("barcodeDialog")
        self.resize(640, 780)
        self.on_apply = on_apply
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 16)
        self.form = BarcodeForm(None, results_folder)
        self.form.show_values(cfg)
        from PySide6.QtWidgets import QScrollArea
        body = QWidget()
        body.setObjectName("prefContent")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(22, 10, 22, 10)
        bl.addWidget(self.form)
        bl.addStretch(1)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(body)
        lay.addWidget(area, 1)
        row = QHBoxLayout()
        row.setContentsMargins(22, 0, 22, 0)
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        ok = QPushButton("Done")
        ok.setObjectName("primaryButton")
        ok.setDefault(True)
        cancel.clicked.connect(self.reject)
        ok.clicked.connect(self._ok)
        row.addWidget(cancel)
        row.addWidget(ok)
        lay.addLayout(row)

    def _ok(self):
        self.on_apply(self.form.values())
        self.accept()


class BarcodeBar(QFrame):
    """The Run view's barcode strip. scanned(code) is emitted for whatever
    is typed or scanned into it and ended with Enter."""
    scanned = Signal(str)
    settingsRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("barcodeBar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        h = QHBoxLayout(self)
        h.setContentsMargins(14, 8, 10, 8)
        h.setSpacing(10)
        ic = QLabel()
        ic.setPixmap(icons.icon("barcode", ink=theme.color("accent")).pixmap(QSize(22, 22)))
        self.icon = ic
        cap = QLabel("Barcode")
        cap.setObjectName("barcodeCaption")
        self.field = QLineEdit()
        self.field.setObjectName("barcodeField")
        self.field.setPlaceholderText("Scan the part's barcode")
        self.field.setClearButtonEnabled(True)
        self.field.setMinimumWidth(260)
        self.field.returnPressed.connect(self._enter)
        self.state = QLabel("")
        self.state.setObjectName("barcodeState")
        self.state.setProperty("badge", True)
        self.rule = QLabel("")
        self.rule.setObjectName("secondary")
        self.where = QToolButton()
        self.where.setObjectName("linkButton")
        self.where.setAutoRaise(True)
        self.where.setToolTip("Open this part's folder")
        self.where.clicked.connect(self._open_folder)
        self.where.hide()
        self._folder = ""
        gear = QToolButton()
        gear.setIcon(icons.icon("gear"))
        gear.setAutoRaise(True)
        gear.setToolTip("Barcode rule and scanner...")
        gear.clicked.connect(self.settingsRequested)
        self.gear = gear
        h.addWidget(ic)
        h.addWidget(cap)
        h.addWidget(self.field, 2)
        h.addWidget(self.state)
        h.addWidget(self.rule, 1)
        h.addWidget(self.where)
        h.addWidget(gear)
        self.waiting("")

    def _enter(self):
        code = self.field.text()
        if code.strip():
            self.scanned.emit(code)

    def take_keys(self, text):
        """Keys typed while something else had the focus (a scanner's burst)."""
        self.field.setFocus()
        self.field.insert(text)

    def set_rule(self, cfg):
        self.rule.setText("Rule: %s" % rules.describe(cfg))

    def _say(self, text, tone):
        self.state.setText(text)
        theme.set_tone(self.state, tone)

    def waiting(self, note=""):
        self._say(note or "Waiting for a scan", "idle")
        self.field.selectAll()

    def accepted(self, code, running=False):
        self._say("%s  -  %s" % (code, "inspecting" if running else "ready, press Run"),
                  "pass")
        self.field.clear()

    def refused(self, code, why):
        self._say("Wrong barcode - %s" % why, "fail")
        self.field.selectAll()

    def needed(self):
        self._say("Scan the barcode first", "warn")
        self.field.setFocus()
        self.field.selectAll()

    def done(self, code, verdict, folder, note=None):
        """note: said instead of the verdict -- "Barcode not good" when the
        part does not carry the scanned code."""
        self._say("%s  -  %s" % (code, note or ("ACCEPT" if verdict == "PASS" else "REJECT")),
                  "pass" if verdict == "PASS" and not note else "fail")
        self._folder = folder or ""
        self.where.setText("Pictures: %s" % os.path.basename(folder) if folder else "")
        self.where.setToolTip(folder or "")
        self.where.setVisible(bool(folder))
        self.field.clear()
        self.field.setFocus()

    def _open_folder(self):
        if self._folder:
            os.makedirs(self._folder, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._folder))


class SerialScanner(QObject):
    """A barcode scanner on a serial port: each line it sends is a code."""
    scanned = Signal(str)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.port = None
        self._buf = b""
        self._idle = QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.timeout.connect(self._flush)     # scanners that end with nothing

    def open(self, name, baud=9600):
        self.close()
        try:
            from PySide6.QtSerialPort import QSerialPort
        except Exception as e:
            self.error.emit("serial ports are not available: %s" % e)
            return False
        p = QSerialPort(self)
        p.setPortName(name)
        p.setBaudRate(int(baud))
        if not p.open(QSerialPort.ReadOnly):
            self.error.emit("%s would not open: %s" % (name, p.errorString()))
            p.deleteLater()
            return False
        p.readyRead.connect(self._read)
        self.port = p
        return True

    def close(self):
        if self.port is not None:
            try:
                self.port.close()
            except Exception:
                pass
            self.port.deleteLater()
            self.port = None
        self._buf = b""

    def _read(self):
        self._buf += bytes(self.port.readAll().data())
        while True:
            cut = min([i for i in (self._buf.find(b"\r"), self._buf.find(b"\n")) if i >= 0],
                      default=-1)
            if cut < 0:
                break
            line, self._buf = self._buf[:cut], self._buf[cut + 1:]
            self._emit(line)
        self._idle.start(150)

    def _flush(self):
        if self._buf:
            line, self._buf = self._buf, b""
            self._emit(line)

    def _emit(self, raw):
        code = rules.clean(raw.decode("latin-1", "replace"))
        if code:
            self.scanned.emit(code)
