"""Vision Station — desktop inspection builder.

Run:  python -m vision_station --classic
"""

import json
import os
import sys
import time

import cv2
from PySide6.QtCore import QPointF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (QAction, QActionGroup, QBrush, QColor, QCursor, QFont,
                           QIcon, QKeySequence, QPainter, QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import (QGraphicsItem, QSizePolicy, QScrollArea, QFrame, QAbstractItemView, QApplication, QCheckBox, QComboBox,
                               QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu,
                               QMessageBox, QPushButton, QSpinBox, QStatusBar, QTabWidget,
                               QTableWidget, QTableWidgetItem, QToolBar, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget,
                               QDialog, QDialogButtonBox, QRadioButton, QButtonGroup)

from vision_station.ui.canvas import FAIL, PASS, WARN, ImageCanvas, ndarray_to_qpixmap
from vision_station.ui.widgets import (EmptyStateTable, confirm_unsaved, dot_icon, fill_tool_menu,
                   guard_wheel, read_only_table, readable_images, toolbox_tree,
                   training_columns, training_menu)
from vision_station.core.dataset import Dataset
from vision_station.core.history import History
from vision_station.core import diagnostics
from vision_station.core import preflight
from vision_station.core import exposure as exposure_mod
from vision_station.core.results import ResultStore
from vision_station.hardware.camera import CameraSettings, CameraThread, PYLON_AVAILABLE, list_devices
from vision_station.core.engine import ProjectStore
from vision_station.core import imio
from vision_station.core.schema import GROUPS, LABELS, SCHEMA, defaults

from vision_station import paths

HERE = paths.ROOT

TRAINING_HELP = """Building your own detector

1  Add test images        Left dock. Get the real variation in there: every
                          lighting condition, every part orientation, good
                          parts and bad. Aim for 100+ images per class.

2  Switch to Label        Toolbar, top left.

3  Make your classes      "New class" for each thing you want found:
                          screw, led, scratch, chip_body. Keep it minimal.

4  Draw boxes             Click a test image, then drag on the picture.
                          Box the object tightly, and label EVERY instance
                          in every image you touch. One missed screw teaches
                          the model that screws are sometimes background, and
                          that single mistake costs more accuracy than fifty
                          extra images will win back.

5  Check dataset          Tells you what is thin before you spend GPU hours.

6  Export for training    Writes images, labels, data.yaml and train.py.

7  On a GPU machine       pip install ultralytics
                          python train.py

                          Read the mAP50 it prints at the end:
                            above 0.95   ready for the line
                            0.85 - 0.95  usable, expect occasional misses
                            below 0.85   label more images, not more epochs

8  Bring back best.onnx   Put it in the model's models folder, add a
                          "Detect with YOLO" tool and set its Model file.
                          No GPU is needed to run it."""
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

STYLE = """
/* Instrument panel, not a web page. Low-chroma greys so the only saturated
   colour on screen is a result: green, amber or red. Anything else competing
   for attention is noise on a machine an operator watches all shift. */
QMainWindow, QWidget { background:#20242A; color:#E4E9EF;
    font-family:'Segoe UI','Inter',sans-serif; font-size:12px; }

QDockWidget::title { background:#282D35; padding:7px 10px; font-weight:600;
    color:#9AA9B8; letter-spacing:0.4px; text-transform:uppercase;
    font-size:10px; border-bottom:1px solid #14171B; }
QDockWidget { titlebar-close-icon:none; titlebar-normal-icon:none; }

QToolBar { background:#282D35; border:0; border-bottom:1px solid #14171B;
    padding:5px 6px; spacing:2px; }
QToolBar#palette { border-right:1px solid #14171B; border-bottom:0;
    padding:6px 4px; spacing:4px; }
QToolBar QToolButton { padding:6px 12px; border-radius:4px; color:#E4E9EF;
    border:1px solid transparent; }
QToolBar QToolButton:hover { background:#333A44; border-color:#3E4652; }
QToolBar QToolButton:checked { background:#0E639C; border-color:#1177B8; }
QToolBar QToolButton:disabled { color:#5C6874; }
QToolBar::separator { background:#343B45; width:1px; height:1px; margin:5px 7px; }

QMenuBar { background:#282D35; border-bottom:1px solid #14171B; padding:2px; }
QMenuBar::item { padding:6px 12px; border-radius:4px; }
QMenuBar::item:selected { background:#333A44; }
QMenu { background:#282D35; border:1px solid #14171B; padding:5px; }
QMenu::item { padding:6px 26px 6px 20px; border-radius:3px; }
QMenu::item:selected { background:#0E639C; }
QMenu::separator { height:1px; background:#343B45; margin:5px 8px; }

QTreeWidget, QTableWidget, QListWidget { background:#191D22; border:1px solid #14171B;
    alternate-background-color:#1D2228; selection-background-color:#0E639C;
    outline:0; border-radius:3px; }
QTreeWidget::item, QTableWidget::item, QListWidget::item { padding:5px 3px; }
QTreeWidget::item:hover, QTableWidget::item:hover { background:#232931; }
QHeaderView::section { background:#282D35; color:#8E9DAC; padding:6px 5px;
    border:0; border-right:1px solid #14171B; border-bottom:1px solid #14171B;
    font-weight:600; font-size:10px; letter-spacing:0.4px;
    text-transform:uppercase; }

QTabWidget::pane { border:1px solid #14171B; background:#20242A; border-radius:3px; }
QTabBar::tab { background:#282D35; padding:7px 15px; border:1px solid #14171B;
    border-bottom:0; color:#8E9DAC; margin-right:1px; }
QTabBar::tab:first { border-top-left-radius:3px; }
QTabBar::tab:selected { background:#20242A; color:#E4E9EF;
    border-top:2px solid #0E90D2; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background:#191D22;
    border:1px solid #343B45; border-radius:4px; padding:5px 7px;
    selection-background-color:#0E639C; }
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
    border-color:#434C59; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color:#0E90D2; }
QComboBox::drop-down { border:0; width:18px; }
QComboBox QAbstractItemView { background:#20242A; border:1px solid #14171B;
    selection-background-color:#0E639C; }

QPushButton { background:#333A44; border:1px solid #414A57; border-radius:4px;
    padding:6px 13px; font-weight:500; }
QPushButton:hover { background:#3C4450; border-color:#4C5766; }
QPushButton:pressed { background:#0E639C; border-color:#1177B8; }
QPushButton:disabled { color:#5C6874; background:#252A31; border-color:#2E353E; }

QGroupBox { border:1px solid #343B45; border-radius:5px; margin-top:15px;
    padding-top:8px; }
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 6px;
    color:#8E9DAC; font-weight:600; font-size:10px; letter-spacing:0.4px;
    text-transform:uppercase; }

QStatusBar { background:#282D35; border-top:1px solid #14171B; }
QStatusBar::item { border:0; }
QSplitter::handle { background:#14171B; }
QLabel { background:transparent; }

QScrollBar:vertical { background:transparent; width:12px; margin:2px; }
QScrollBar::handle:vertical { background:#3E4652; border-radius:5px; min-height:26px; }
QScrollBar::handle:vertical:hover { background:#4C5766; }
QScrollBar:horizontal { background:transparent; height:12px; margin:2px; }
QScrollBar::handle:horizontal { background:#3E4652; border-radius:5px; min-width:26px; }
QScrollBar::handle:horizontal:hover { background:#4C5766; }
QScrollBar::add-line, QScrollBar::sub-line { height:0; width:0; }
QScrollBar::add-page, QScrollBar::sub-page { background:transparent; }

QCheckBox { spacing:7px; }
QCheckBox::indicator { width:15px; height:15px; border:1px solid #4C5766;
    border-radius:3px; background:#191D22; }
QCheckBox::indicator:hover { border-color:#5E6B7C; }
QCheckBox::indicator:checked { background:#0E90D2; border-color:#0E90D2; }
QRadioButton::indicator { width:14px; height:14px; border:1px solid #4C5766;
    border-radius:7px; background:#191D22; }
QRadioButton::indicator:checked { background:#0E90D2; border-color:#0E90D2; }
QToolTip { background:#14171B; color:#E4E9EF; border:1px solid #3E4652;
    padding:5px 7px; }
"""



def shape_icon(kind, size=20):
    """Palette icons drawn in code, so there are no image files to lose."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor("#DDE2E8"), 1.6)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    m = 3
    if kind == "select":
        path = QPolygonF([QPointF(5, 3), QPointF(5, 16), QPointF(8.5, 12.5),
                          QPointF(11, 17), QPointF(13, 16), QPointF(10.5, 11.5),
                          QPointF(15, 11)])
        p.setBrush(QBrush(QColor("#DDE2E8")))
        p.drawPolygon(path)
    elif kind == "draw_rect":
        p.drawRect(m, m + 2, size - 2 * m, size - 2 * m - 4)
    elif kind == "draw_ellipse":
        p.drawEllipse(m, m + 2, size - 2 * m, size - 2 * m - 4)
    elif kind == "draw_polygon":
        p.drawPolygon(QPolygonF([QPointF(10, 3), QPointF(17, 8),
                                 QPointF(14, 17), QPointF(6, 16),
                                 QPointF(3, 8)]))
    elif kind == "hand":
        p.drawRoundedRect(6, 7, 8, 10, 3, 3)
        for x in (7.5, 10, 12.5):
            p.drawLine(QPointF(x, 7), QPointF(x, 3.5))
    p.end()
    return QIcon(pm)

dot = dot_icon            # shared with the VisionPro front end


class InspectionWorker(QThread):
    """Runs one inspection off the interface thread.

    Inspection used to run inline on the UI thread. A 280 ms recipe therefore
    froze the whole application for 280 ms of every cycle: measured with a
    50 ms probe timer during continuous running, the event loop was serviced
    only every ~300 ms, so clicks, repaints and menu input all stalled. That is
    the lag -- not the rendering, which measures well under a millisecond.

    The recipe is rebuilt from a deep copy inside the worker, so the operator
    can keep editing regions while an inspection is in flight without mutating
    what is being inspected underneath it.
    """
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, project, spec, payload, parent=None):
        super().__init__(parent)
        self.project = project
        self.spec = json.loads(json.dumps(spec))
        self.payload = payload

    def run(self):
        from vision_station.tools import make_tool
        try:
            tools = []
            for src in self.spec.get("tools", []):
                sp = json.loads(json.dumps(src))
                prm = sp.setdefault("params", {})
                model = prm.get("model")
                if model and not os.path.isabs(model):
                    prm["model"] = os.path.join(self.project.folder, model)
                tools.append(make_tool(sp))
            self.project.spec = self.spec
            self.project.tools = tools
            self.finished_ok.emit(
                self.project.inspect(self.payload, source="desktop"))
        except Exception as e:
            self.failed.emit("%s: %s" % (type(e).__name__, e))


class ScrollTabs(QTabWidget):
    """Tabs whose pages scroll. A tool with many settings, or the Train
    page's thumbnails, made the sheet taller than the editor on a laptop
    screen, and the editor's own status bar was pushed out of sight."""

    def addTab(self, page, *label):
        sa = QScrollArea()
        sa.setObjectName("tabScroll")
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sa.setWidget(page)
        return super().addTab(sa, *label)


class PropertySheet(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.tool = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.header = QLabel("Nothing selected")
        self.header.setObjectName("sheetHeader")       # styled by theme.py
        v.addWidget(self.header)
        self.desc = QLabel("")
        self.desc.setWordWrap(True)
        self.desc.setObjectName("sheetDesc")
        v.addWidget(self.desc)

        self.tabs = ScrollTabs()
        v.addWidget(self.tabs, 1)

        self.settings = QWidget()
        self.form = QFormLayout(self.settings)
        self.form.setLabelAlignment(Qt.AlignRight)
        self.form.setContentsMargins(11, 11, 11, 11)
        self.form.setVerticalSpacing(7)
        self.tabs.addTab(self.settings, "Settings")

        reg = QWidget()
        rv = QVBoxLayout(reg)
        rv.setContentsMargins(11, 11, 11, 11)
        gb = QGroupBox("Region")
        gf = QFormLayout(gb)
        self.rcapture = QComboBox()
        self.rcapture.setToolTip("The image this tool reads: a camera capture, or "
                                 "the output of an image-processing tool before it.")
        self.rcapture.currentIndexChanged.connect(self._typed)
        gf.addRow("Reads image", self.rcapture)
        self.rshape = QComboBox()
        self.rshape.addItem("Rectangle", "rect")
        self.rshape.addItem("Ellipse / circle", "ellipse")
        self.rshape.addItem("Polygon", "polygon")
        self.rshape.currentIndexChanged.connect(self._typed)
        gf.addRow("Shape", self.rshape)
        self.rx, self.ry, self.rw, self.rh = (QSpinBox() for _ in range(4))
        for sp, lab in ((self.rx, "X"), (self.ry, "Y"), (self.rw, "W"), (self.rh, "H")):
            sp.setRange(0, 100000)
            sp.setFont(QFont("Consolas", 9))
            sp.valueChanged.connect(self._typed)
            gf.addRow(lab, sp)
        self.rangle = QDoubleSpinBox()
        self.rangle.setRange(-180.0, 180.0)
        self.rangle.setDecimals(2)
        self.rangle.setSingleStep(1.0)
        self.rangle.setSuffix(" deg")
        self.rangle.setFont(QFont("Consolas", 9))
        self.rangle.valueChanged.connect(self._typed)
        gf.addRow("Angle", self.rangle)
        rv.addWidget(gb)
        # paragraphs that wrap to the pane: hard line breaks clipped the text
        # in a narrow editor and left a ragged column in a wide one
        tips = QLabel(
            "Drag the region on the image, or type exact values here. Drag the "
            "round grip above a selected region to rotate it; hold Shift while "
            "rotating to snap to 15 degrees.<br><br>"
            "An ellipse uses the same X/Y/W/H box, inscribed. Pixels outside it "
            "are neutralised so round pads, screw heads and LEDs do not drag in "
            "their square corners.<br><br>"
            "Click a region to select it. Shift-click to add another. Drag on empty "
            "canvas to marquee-select several. Alt-drag or Ctrl+D duplicates. "
            "Arrow keys nudge.")
        tips.setWordWrap(True)
        tips.setObjectName("hint")
        rv.addWidget(tips)
        rv.addStretch(1)
        self.tabs.addTab(reg, "Region")

        self.resultTab = EmptyStateTable(0, 2, "Run the job to see what this tool measured.")
        self.resultTab.setHorizontalHeaderLabels(["Value", "Reading"])
        read_only_table(self.resultTab)
        self.tabs.addTab(self.resultTab, "Last result")
        guard_wheel(reg)
        self._loading = False
        # The simple view: the few settings a check is usually tuned with,
        # each with a sentence under it, and "Show all settings" for the rest.
        self.simple = False
        self.show_all = False

    def set_simple(self, on):
        on = bool(on)
        if on != self.simple:
            self.simple = on
            if self.tool is not None:
                self.load(self.tool)

    def _toggle_all(self):
        self.show_all = not self.show_all
        if self.tool is not None:
            self.load(self.tool)

    def load(self, tool, multi=0):
        from vision_station.core.schema import ESSENTIAL, help_for
        self.tool = tool
        self._loading = True
        while self.form.rowCount():
            self.form.removeRow(0)
        if tool is None:
            self.header.setText(f"{multi} regions selected" if multi > 1 else "Nothing selected")
            self.desc.setText("Move them together with the mouse or the arrow keys."
                              if multi > 1 else "")
            self._loading = False
            return

        label, help_ = LABELS.get(tool["type"], (tool["type"], ""))
        self.header.setText(f"{tool['id']}    \u2014  {label}")
        self.desc.setText(help_)
        p = tool.setdefault("params", {})

        noun = "check" if self.simple else "tool"
        for text, key in (("Run this %s" % noun, "enabled"),
                          ("A failure rejects the part", "critical")):
            w = QCheckBox(text)
            w.setChecked(p.get(key, True))
            w.toggled.connect(lambda v, k=key: self._set(k, v))
            self.form.addRow("", w)

        rows = SCHEMA.get(tool["type"], [])
        essential = ESSENTIAL.get(tool["type"])
        trimmed = self.simple and not self.show_all and essential is not None
        if trimmed:
            byname = {r[0]: r for r in rows}
            rows = [byname[k] for k in essential if k in byname]
        for key, lab, kind, dv, extra in rows:
            val = p.get(key, dv)
            if kind == "int":
                w = QSpinBox()
                w.setRange(int(extra[0]), int(extra[1]))
                w.setSingleStep(int(extra[2]))
                w.setValue(int(val if val is not None else 0))
                w.valueChanged.connect(lambda v, k=key: self._set(k, v))
            elif kind == "float":
                w = QDoubleSpinBox()
                w.setRange(float(extra[0]), float(extra[1]))
                w.setSingleStep(float(extra[2]))
                w.setDecimals(max(2, len(str(extra[2]).split(".")[-1])))
                w.setValue(float(val if val is not None else 0))
                w.valueChanged.connect(lambda v, k=key: self._set(k, v))
            elif kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(val))
                w.toggled.connect(lambda v, k=key: self._set(k, v))
            elif kind == "choice":
                w = QComboBox()
                w.addItems(extra)
                w.setCurrentText(str(val))
                w.currentTextChanged.connect(lambda v, k=key: self._set(k, v))
            elif kind == "toolref":
                # A dropdown of the geometry tools that run earlier in the
                # recipe -- only earlier ones, since a later tool's result does
                # not exist yet when this one runs. Typing ids by hand invited a
                # silent typo and an "no result from" error at run time.
                w = QComboBox()
                recipe = getattr(self, "recipe", []) or []
                ids = []
                for t in recipe:
                    if t["id"] == tool["id"]:
                        break
                    if t.get("type") in ("find_line", "find_circle"):
                        ids.append(t["id"])
                w.addItem("")
                w.addItems(ids)
                cur = str(val or "")
                if cur and cur not in ids:
                    w.addItem(cur + "  (not found earlier)", cur)
                    w.setCurrentIndex(w.count() - 1)
                else:
                    w.setCurrentText(cur)
                w.setToolTip("Features found by Find line / Find circle tools "
                             "that run before this one")
                w.currentIndexChanged.connect(
                    lambda _i, k=key, ww=w: self._set(k, ww.currentData() or ww.currentText()))
            else:
                w = QLineEdit("" if val is None else
                              (json.dumps(val) if isinstance(val, (list, dict)) else str(val)))
                w.setPlaceholderText({"strlist": '["red","green","blue"]',
                                      "boollist": "[true,true,false]",
                                      "intlist": "[-5,5]"}.get(kind, ""))
                w.editingFinished.connect(
                    lambda k=key, kk=kind, ww=w: self._set_text(k, kk, ww.text(), ww))
            self.form.addRow(lab, w)
            if self.simple:
                why = help_for(tool["type"], key)
                if why:
                    h = QLabel(why)
                    h.setObjectName("fieldHelp")
                    h.setWordWrap(True)
                    self.form.addRow("", h)
        if self.simple and essential is not None and len(essential) < len(SCHEMA.get(tool["type"], [])):
            more = QPushButton("Show fewer settings" if self.show_all else
                               "Show all settings  (%d)" % len(SCHEMA.get(tool["type"], [])))
            more.setObjectName("linkButton")
            more.setCursor(Qt.PointingHandCursor)
            more.clicked.connect(self._toggle_all)
            self.form.addRow("", more)

        for sp, v in zip((self.rx, self.ry, self.rw, self.rh), p.get("roi") or [0, 0, 0, 0]):
            sp.setValue(int(v))
        self.rangle.setValue(float(p.get("angle", 0.0) or 0.0))
        i = self.rshape.findData(p.get("shape", "rect") or "rect")
        self.rshape.setCurrentIndex(max(0, i))
        want = p.get("capture", "default") or "default"
        j = self.rcapture.findText(want)
        if j < 0 and self.rcapture.count():
            self.rcapture.addItem(want)
            j = self.rcapture.findText(want)
        self.rcapture.setCurrentIndex(max(0, j))
        guard_wheel(self.settings)
        self._loading = False

    def _set(self, key, val):
        if self._loading or not self.tool:
            return
        self.tool["params"][key] = val
        self.changed.emit()

    def _set_text(self, key, kind, text, field=None):
        """A typed entry. One the tool cannot use is marked on the field -- red,
        with the reason in its tooltip -- and not applied. It was a message box,
        which took focus from the field, so the field finished editing a second
        time and the same warning came up twice."""
        if self._loading or not self.tool:
            return
        text = text.strip()
        if kind in ("intlist", "strlist", "boollist"):
            try:
                val = json.loads(text) if text else []
                if not isinstance(val, list):
                    raise ValueError("not a list")
            except ValueError:
                self._mark_invalid(field, "'%s' is not a list - not applied. Use square "
                                          "brackets and commas, like %s"
                                   % (text, field.placeholderText() if field else '["red","green"]'))
                return
        else:
            val = text
        self._mark_invalid(field, None)
        if self.tool["params"].get(key) == val:
            return                            # finishing an edit that changed nothing
        self.tool["params"][key] = val
        self.changed.emit()

    @staticmethod
    def _mark_invalid(field, why):
        if field is None or bool(field.property("invalid")) == bool(why):
            return
        field.setProperty("invalid", bool(why))
        field.setToolTip(why or "")
        field.style().unpolish(field)
        field.style().polish(field)

    def _typed(self):
        if self._loading or not self.tool:
            return
        self.tool["params"]["roi"] = [self.rx.value(), self.ry.value(),
                                      self.rw.value(), self.rh.value()]
        self.tool["params"]["angle"] = self.rangle.value()
        self.tool["params"]["shape"] = self.rshape.currentData()
        if self.rcapture.count():
            self.tool["params"]["capture"] = self.rcapture.currentText()
        self.changed.emit()

    def set_roi(self, roi, emit=True):
        self._loading = True
        for sp, v in zip((self.rx, self.ry, self.rw, self.rh), roi):
            sp.setValue(int(v))
        self._loading = False
        if self.tool:
            self.tool["params"]["roi"] = list(roi)
            if emit:
                self.changed.emit()

    def set_captures(self, names):
        cur = self.rcapture.currentText()
        self._loading = True
        self.rcapture.clear()
        self.rcapture.addItems(names or ["default"])
        k = self.rcapture.findText(cur)
        self.rcapture.setCurrentIndex(max(0, k))
        self._loading = False

    def set_angle(self, deg):
        self._loading = True
        self.rangle.setValue(float(deg))
        self._loading = False
        if self.tool:
            self.tool["params"]["angle"] = float(deg)

    def show_result(self, res):
        """Every reading the tool made. A list of plain values -- the lines a
        text tool read, the codes a code reader decoded -- is shown one row
        per item: lists were skipped, so the text tools' Last result showed
        how many blocks they found and never what the blocks said."""
        rows = []
        for k, v in (res or {}).get("values", {}).items():
            if isinstance(v, dict):
                continue
            if isinstance(v, list):
                if not v or not all(isinstance(x, (str, int, float)) for x in v):
                    continue
                name = k[:-1] if k.endswith("s") and len(v) > 1 else k
                rows += [("%s %d" % (name, i + 1) if len(v) > 1 else name, x)
                         for i, x in enumerate(v)]
                continue
            rows.append((k, v))
        self.resultTab.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.resultTab.setItem(i, 0, QTableWidgetItem(k.replace("_", " ")))
            it = QTableWidgetItem(str(v))
            it.setFont(QFont("Consolas", 9))
            self.resultTab.setItem(i, 1, it)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Station")
        self.resize(1600, 960)
        self.store = ProjectStore(os.path.join(HERE, "projects"))
        self.project = None
        self.spec = None
        self.dirty = False
        self.camera = None
        self.cam_settings = CameraSettings()
        self.live = None
        self.last = None
        self.clipboard = []
        self.history = History()
        self.results = None
        self.archive_mode = "fail"
        self.ds = None
        self.cur_image = None      # filename of the test image on screen
        self.counts = {"PASS": 0, "FAIL": 0, "WARN": 0}
        self.continuous = False
        self.workspace = "inspect"
        self._syncing = False
        self._worker = None
        self._editors = {}                 # tool id -> open tool editor window

        self._build_canvas()
        self._build_banner()
        self._build_job_dock()
        self._build_images_dock()
        self._build_label_dock()
        self._build_props()
        self._build_grid()
        self._build_toolbar()
        self._build_palette()
        self._build_training_dock()
        self._build_toolbox()
        self._build_run_bar()
        self._apply_icons()
        self._build_menus()
        self._build_status()

        # Single-shot and rescheduled after each cycle finishes, rather than a
        # repeating timer. A repeating 120 ms timer against a cycle that takes
        # longer than 120 ms queues work faster than it completes: Qt piles up
        # timeout events, the event loop never drains, and the whole interface
        # goes treacly -- buttons lag, the canvas stops redrawing. Rescheduling
        # on completion means the application always runs as fast as the recipe
        # allows and no faster.
        self.runTimer = QTimer(self)
        self.runTimer.setSingleShot(True)
        self.runTimer.timeout.connect(self._continuous_tick)
        # Report a thin environment once, shortly after the window appears.
        # The failure this prevents is quiet and expensive: a recipe tuned on a
        # bench machine that has PaddleOCR, moved to a line PC that does not,
        # where text reading silently drops to a weaker engine.
        QTimer.singleShot(1200, self.report_environment)
        self.pollTimer = QTimer(self)
        self.pollTimer.timeout.connect(self.on_poll_camera)
        self.reload_projects()

    # ------------------------------------------------------------ build
    def _build_canvas(self):
        self.canvas = ImageCanvas()
        self.canvas.geometryCommitted.connect(self.on_geometry)
        self.canvas.liveGeometry.connect(self.on_live_geometry)
        self.canvas.selectionChanged.connect(self.on_canvas_selection)
        self.canvas.duplicateRequested.connect(self.duplicate_ids)
        self.canvas.deleteRequested.connect(self.delete_ids)
        self.canvas.statusText.connect(self.pixLabelSet)
        self.canvas.angleChanged.connect(self.on_angle_changed)
        self.canvas.editBegan.connect(self.on_edit_began)
        self.canvas.shapeDrawn.connect(self.on_shape_drawn)
        self.canvas.zoomChanged.connect(self.on_zoom_changed)
        self.canvas.polygonCommitted.connect(self.on_polygon_committed)
        self.canvas.labelAdded.connect(self.on_label_added)
        self.canvas.labelsEdited.connect(self.save_labels)
        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.canvas_menu)

    def pixLabelSet(self, s):
        if hasattr(self, "pixLabel"):
            self.pixLabel.setText(s)

    def _build_job_dock(self):
        self.tree = QTreeWidget()
        self.tree.itemClicked.connect(self.on_tree_clicked)
        self.tree.itemDoubleClicked.connect(self.on_tree_double_clicked)
        self.tree.setHeaderLabels(["Inspection recipe", "Result"])
        self.tree.setMinimumHeight(240)
        self.tree.setColumnWidth(0, 200)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self.on_tree_select)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.tree_menu)

        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        top = QWidget()
        h = QHBoxLayout(top)
        h.setContentsMargins(7, 7, 7, 7)
        h.addWidget(QLabel("Model"))
        self.projBox = QComboBox()
        self.projBox.currentTextChanged.connect(self.on_project_change)
        h.addWidget(self.projBox, 1)
        v.addWidget(top)
        v.addWidget(self.tree, 1)
        btns = QWidget()
        hb = QHBoxLayout(btns)
        hb.setContentsMargins(7, 7, 7, 7)
        for text, fn in (("Add", self.add_tool_menu), ("Copy", self.copy_sel),
                         ("Paste", self.paste), ("Up", lambda: self.move_tool(-1)),
                         ("Down", lambda: self.move_tool(1))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            hb.addWidget(b)
        v.addWidget(btns)

        d = QDockWidget("Job", self)
        self.jobDock = d
        d.setWidget(w)
        d.setMinimumWidth(300)
        self.addDockWidget(Qt.LeftDockWidgetArea, d)

    def _build_images_dock(self):
        self.imgList = QListWidget()
        self.imgList.setIconSize(QSize(96, 62))
        self.imgList.itemClicked.connect(self.on_image_picked)
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self.imgList, 1)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(7, 7, 7, 7)
        for t, fn in (("Add images", self.add_images), ("Remove", self.remove_image)):
            b = QPushButton(t)
            b.clicked.connect(fn)
            h.addWidget(b)
        v.addWidget(row)
        d = QDockWidget("Test images", self)
        self.imagesDock = d
        d.setWidget(w)
        self.addDockWidget(Qt.LeftDockWidgetArea, d)
        d.setMinimumHeight(200)

    def _build_label_dock(self):
        self.classList = QListWidget()
        self.classList.setMinimumHeight(150)
        self.classList.currentRowChanged.connect(self.on_class_changed)
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        hint = QLabel("Drag on the image to draw a box of the highlighted class.")
        hint.setWordWrap(True)
        hint.setStyleSheet("padding:7px 9px; color:#8C99A6; background:#262B31;")
        v.addWidget(hint)
        v.addWidget(self.classList, 1)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(7, 7, 7, 7)
        for t, fn in (("New class", self.add_class), ("Delete class", self.del_class)):
            b = QPushButton(t)
            b.clicked.connect(fn)
            h.addWidget(b)
        v.addWidget(row)
        self.dsStats = QLabel("")
        self.dsStats.setWordWrap(True)
        self.dsStats.setStyleSheet("padding:7px 9px; color:#9FB0C0; font-family:Consolas;")
        v.addWidget(self.dsStats)
        row2 = QWidget()
        h2 = QHBoxLayout(row2)
        h2.setContentsMargins(7, 0, 7, 7)
        b = QPushButton("Check dataset")
        b.clicked.connect(self.check_dataset)
        h2.addWidget(b)
        b = QPushButton("Export for training")
        b.clicked.connect(self.export_dataset)
        h2.addWidget(b)
        v.addWidget(row2)

        self.labelDock = QDockWidget("Labelling", self)
        self.labelDock.setWidget(w)
        self.labelDock.setMinimumWidth(300)
        self.addDockWidget(Qt.RightDockWidgetArea, self.labelDock)
        self.labelDock.hide()

    def _build_props(self):
        self.props = PropertySheet()
        self.props.changed.connect(self.on_props_changed)
        d = QDockWidget("Properties", self)
        # Scrollable, so the sheet can give up height. Unwrapped it demanded a
        # fixed 522 px, which starved the training panel below it and clipped
        # long settings lists on a smaller screen.
        sc = QScrollArea()
        sc.setWidget(self.props)
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        d.setWidget(sc)
        d.setMinimumWidth(340)
        self.addDockWidget(Qt.RightDockWidgetArea, d)
        self.propsDock = d

    def _build_grid(self):
        self.grid = QTableWidget(0, 6)
        self.grid.setHorizontalHeaderLabels(["Tool", "Type", "Result", "Score", "Time", "Detail"])
        self.grid.verticalHeader().hide()
        self.grid.setAlternatingRowColors(True)
        self.grid.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.grid.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for i, wdt in enumerate([150, 130, 80, 70, 70]):
            self.grid.setColumnWidth(i, wdt)
        self.grid.horizontalHeader().setStretchLastSection(True)
        self.grid.itemSelectionChanged.connect(self.on_grid_select)
        d = QDockWidget("Results", self)
        d.setWidget(self.grid)
        d.setMinimumHeight(170)
        self.addDockWidget(Qt.BottomDockWidgetArea, d)

    # Shortcuts no menu defines, so the toolbar action is their only owner.
    TOOLBAR_OWNS = {"Ctrl+B", "Ctrl+R"}

    def _build_toolbar(self):
        """Two rows, grouped by what they are for.

        The single row this replaces put the workspace switch, the tool
        selection, the camera, the run controls, the region type and four zoom
        buttons in one uninterrupted line of eleven items. Select and Hand also
        appeared twice over once the palette existed. Nothing indicated which
        controls belonged together, so finding anything meant reading the whole
        row every time.

        Row one is the workspace and what to inspect. Row two is running it and
        looking at the result. Zooming moved to the View menu and the wheel,
        where it costs nothing to reach and stops competing with the controls
        that matter."""
        def make_bar(name):
            tb = QToolBar(name)
            tb.setIconSize(QSize(13, 13))
            tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            tb.setMovable(True)
            return tb

        top = make_bar("Workspace")
        self.addToolBar(top)
        self.addToolBarBreak()
        run = make_bar("Inspection")
        self.addToolBar(run)

        def act(bar, text, fn, icon=None, key=None, tip="", check=False):
            a = QAction(text, self)
            if icon:
                a.setIcon(icon)
            a.setCheckable(check)
            a.triggered.connect(fn)
            # The key is shown, not registered: the menus own every shortcut.
            # Registering it here as well gave ten keys two owners -- F5, F6,
            # Ctrl+S, Ctrl+O among them -- and Qt fires neither action of an
            # ambiguous shortcut, so those keys silently did nothing at all.
            if key and key in self.TOOLBAR_OWNS:
                a.setShortcut(QKeySequence(key))
            a.setToolTip("%s   (%s)" % (tip or text, key) if key else (tip or text))
            bar.addAction(a)
            return a

        # ---- Build / Run ------------------------------------------------
        mgrp = QActionGroup(self)
        self.aModeBuild = act(top, "Build", lambda: self.set_mode("build"), None, "Ctrl+B",
                              "Make and tune the model", True)
        self.aModeRun = act(top, "Run", lambda: self.set_mode("run"), None, "Ctrl+R",
                            "The operator screen: inspect, with editing locked", True)
        mgrp.addAction(self.aModeBuild)
        mgrp.addAction(self.aModeRun)
        self.aModeBuild.setChecked(True)
        self.addAction(self.aModeBuild)      # shortcuts keep working in Run mode,
        self.addAction(self.aModeRun)        # where this toolbar is hidden
        top.addSeparator()

        # ---- row one: which workspace, and what we are looking at ----------
        wgrp = QActionGroup(self)
        self.aInspect = act(top, "Inspect", lambda: self.set_workspace("inspect"),
                            None, None, "Build and run the inspection recipe", True)
        self.aLabel = act(top, "Label", lambda: self.set_workspace("label"),
                          None, None, "Draw training boxes for a YOLO model", True)
        wgrp.addAction(self.aInspect)
        wgrp.addAction(self.aLabel)
        self.aInspect.setChecked(True)
        top.addSeparator()

        self.aCam = act(top, "Start camera", self.toggle_camera, dot("#8899AA"))
        act(top, "Load image", self.open_image, None, "Ctrl+O")
        act(top, "Set reference", self.set_reference, None, None,
            "Freeze the current image as this model's golden part")
        top.addSeparator()

        lbl = QLabel(" Draw as: ")
        lbl.setStyleSheet("color:#8E9DAC;")
        top.addWidget(lbl)
        self.newTypeBox = QComboBox()
        self.newTypeBox.setMinimumWidth(210)
        self.newTypeBox.setToolTip("What a region drawn with the rectangle, "
                                   "ellipse or polygon tool will check")
        for group, types in GROUPS:
            for tname in types:
                self.newTypeBox.addItem("%s \u00b7 %s" % (group, LABELS[tname][0]),
                                        tname)
        i = self.newTypeBox.findData("ocv")
        self.newTypeBox.setCurrentIndex(max(0, i))
        top.addWidget(self.newTypeBox)

        # Select and Hand live in the palette. They were here as well, which is
        # two controls for one state and two places to look for it.
        self.aSelect = QAction("Select", self)
        self.aSelect.setCheckable(True)
        self.aSelect.setChecked(True)
        self.aHand = QAction("Hand", self)
        self.aHand.setCheckable(True)

        # ---- row two: run it, then read the result -------------------------
        self.aRun = act(run, "Run once", self.run_once, dot("#18C25A"), "F5")
        self.aCont = act(run, "Run continuous", self.toggle_continuous,
                         dot("#0E90D2"), "F6")
        act(run, "Save job", self.save_job, None, "Ctrl+S")
        run.addSeparator()
        act(run, "Diagnose", self.show_diagnostics, None, "F8",
            "Image quality, recipe margins and cycle time")
        act(run, "Exposure", self.auto_expose, None, "F7",
            "Find the longest exposure that does not clip the selected region")
        act(run, "History", self.show_history, None, "F9",
            "Production history, yield and reject Pareto")
        run.addSeparator()
        self.modelLabel = QLabel("")
        self.modelLabel.setStyleSheet("color:#8E9DAC; padding-left:4px;")
        run.addWidget(self.modelLabel)
        self.mainBar, self.runBar = top, run

    def _build_menus(self):
        m = self.menuBar()
        f = m.addMenu("&File")
        f.addAction("New model...", self.new_project)
        f.addAction("Open image...", self.open_image, QKeySequence("Ctrl+O"))
        f.addAction("Add test images...", self.add_images)
        f.addAction("Captures...", self.edit_captures)
        f.addSeparator()
        f.addAction("Save job", self.save_job, QKeySequence("Ctrl+S"))
        f.addAction("Export job as...", self.export_job)
        f.addSeparator()
        f.addAction("Exit", self.close)

        e = m.addMenu("&Edit")
        self.aUndo = e.addAction("Undo", self.undo, QKeySequence.Undo)
        self.aRedo = e.addAction("Redo", self.redo, QKeySequence.Redo)
        self.aUndo.setEnabled(False)
        self.aRedo.setEnabled(False)
        e.addSeparator()
        e.addAction("Copy", self.copy_sel, QKeySequence.Copy)
        e.addAction("Paste", self.paste, QKeySequence.Paste)
        e.addAction("Duplicate", self.duplicate_sel, QKeySequence("Ctrl+D"))
        e.addAction("Open tool editor", lambda: self.open_tool_editor(),
                    QKeySequence("Ctrl+E"))
        e.addSeparator()
        e.addAction("Select all regions", self.select_all, QKeySequence("Ctrl+A"))
        e.addAction("Deselect", lambda: self.canvas.sc.clearSelection(), QKeySequence("Esc"))
        e.addSeparator()
        e.addAction("Delete", self.remove_tool, QKeySequence("Del"))

        t = m.addMenu("&Tools")
        fill_tool_menu(t, self.add_tool)
        t.addSeparator()
        t.addAction("Calibrate lens...", self.calibrate_lens)
        t.addAction("Remove lens calibration", self.remove_lens_calibration)

        d = m.addMenu("&Dataset")
        d.addAction("Inspect workspace", lambda: self.set_workspace("inspect"))
        d.addAction("Label workspace", lambda: self.set_workspace("label"))
        d.addSeparator()
        d.addAction("New class...", self.add_class)
        d.addAction("Check dataset", self.check_dataset)
        d.addAction("Export for training...", self.export_dataset)
        d.addAction("How to train", self.show_training_help)

        r = m.addMenu("&Run")
        r.addAction("Run once", self.run_once, QKeySequence("F5"))
        r.addAction("Run continuous", self.toggle_continuous, QKeySequence("F6"))
        r.addAction("Run on every test image", self.run_batch)
        r.addSeparator()
        r.addAction("Find exposure for selected region", self.auto_expose,
                    QKeySequence("F7"))
        r.addAction("Diagnostics...", self.show_diagnostics, QKeySequence("F8"))
        r.addAction("Production history...", self.show_history, QKeySequence("F9"))
        r.addAction("Image archiving...", self.set_archive_mode)
        r.addAction("Stability check", self.run_stability)
        r.addSeparator()
        r.addAction("Reset counters", self.reset_counts)

        # Redo answers both Ctrl+Shift+Z and Ctrl+Y, on one action. A second
        # action for Ctrl+Y collided with the standard Redo key, which IS Ctrl+Y
        # on Windows, so redo by keyboard did nothing there.
        self.aRedo.setShortcuts([QKeySequence("Ctrl+Shift+Z"), QKeySequence("Ctrl+Y")])

        v = m.addMenu("&View")
        # V and H belong to the palette. These entries called set_mode(), a name
        # that now means Build/Run -- so choosing them switched modes with a
        # nonsense value instead of picking the tool.
        v.addAction("Select tool   (V)", lambda: self.set_tool("select"))
        v.addAction("Hand tool   (H)", lambda: self.set_tool("hand"))
        v.addSeparator()
        v.addSeparator()
        lab = v.addMenu("Region names")
        for cap, mode, tip in (
                ("Only what matters", "smart",
                 "Names on the selected region and on anything that failed"),
                ("Every region", "all", "Name every region all the time"),
                ("Only the selection", "none", "Names only on the selection")):
            a = lab.addAction(cap)
            a.setToolTip(tip)
            a.triggered.connect(lambda _c=False, m=mode: self.set_label_policy(m))
        v.addSeparator()
        v.addAction("Fit", self.canvas.fit, QKeySequence("Ctrl+0"))
        v.addAction("Actual size", self.canvas.zoom_100, QKeySequence("Ctrl+1"))
        v.addAction("Zoom to selection", self.canvas.zoom_to_selection, QKeySequence("Ctrl+2"))

        h = m.addMenu("&Help")
        h.addAction("Shortcuts", self.show_shortcuts)
        h.addAction("Check installation", self.show_environment)

    CLASSIC_ICONS = {"Build": "build", "Run": "run", "Inspect": "job", "Label": "tag",
                     "Start camera": "image_source", "Load image": "open",
                     "Set reference": "fit", "Run once": "run",
                     "Run continuous": "run_continuous", "Save job": "save",
                     "Diagnose": "test", "Exposure": "exposure", "History": "history",
                     "Select": "pointer", "Rectangle": "rect", "Ellipse": "ellipse",
                     "Polygon": "polygon", "Pan": "pan"}

    def _apply_icons(self):
        """The shared icon set on the classic window's toolbars too."""
        from vision_station.ui import icons
        for bar in (self.mainBar, self.runBar, getattr(self, "paletteBar", None)):
            if bar is None:
                continue
            for act in bar.actions():
                name = self.CLASSIC_ICONS.get(act.text())
                if name:
                    act.setIcon(icons.icon(name, ink="#DCE3EA"))   # light ink on the dark theme

    def _build_status(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        # Belongs here, not earlier. Added to the default status bar before this
        # line replaced it, the label was destroyed with the old bar while
        # Python still held the dead reference -- every zoom then raised.
        self.zoomLabel = QLabel("100%")
        self.zoomLabel.setStyleSheet("color:#9AA9B8;padding:0 10px;")
        self.zoomLabel.setToolTip("Zoom - wheel zooms at the pointer, "
                                  "Ctrl+0 fits, Ctrl+1 actual size")
        sb.addPermanentWidget(self.zoomLabel)
        self.verdict = QLabel("READY")
        self.verdict.setMinimumWidth(94)
        self.verdict.setAlignment(Qt.AlignCenter)
        self.verdict.setStyleSheet("background:#3A424C; color:#DDE2E8; font-weight:700;"
                                   "font-size:13px; padding:3px 12px; border-radius:3px;")
        sb.addWidget(self.verdict)
        self.timeLabel = QLabel("—")
        self.timeLabel.setFont(QFont("Consolas", 9))
        sb.addWidget(self.timeLabel)
        self.yieldLabel = QLabel("")
        self.yieldLabel.setFont(QFont("Consolas", 9))
        sb.addWidget(self.yieldLabel)
        self.pixLabel = QLabel("")
        self.pixLabel.setFont(QFont("Consolas", 9))
        sb.addPermanentWidget(self.pixLabel)

    def report_environment(self):
        lines = preflight.summary_lines()
        bad = [m for k, m in lines if k == "bad"]
        warn = [m for k, m in lines if k == "warn"]
        if bad:
            QMessageBox.critical(self, "Cannot run", "\n\n".join(bad))
            return
        if warn:
            self.statusBar().showMessage(
                "%d optional component%s missing - Help, Check installation"
                % (len(warn), "s" if len(warn) > 1 else ""), 12000)

    def show_environment(self):
        lines = preflight.summary_lines()
        engine = preflight.ocr_engine()
        body = ["Text reading will use: %s" % (engine or "nothing installed"), ""]
        for kind, msg in lines:
            body.append(("OK   " if kind == "good" else
                         "  -  ") + msg)
        rows = preflight.check()
        body += ["", "Installed:"]
        for r in rows:
            body.append("   %-13s %s" % (r["name"], "yes" if r["ok"] else "no"))
        QMessageBox.information(self, "Installation", "\n".join(body))

    def show_shortcuts(self):
        QMessageBox.information(self, "Shortcuts",
            "V / H            select tool / hand tool\n"
            "Space + drag     pan (any tool)\n"
            "Middle drag      pan\n"
            "Wheel            zoom\n\n"
            "Click            select a region\n"
            "Shift + click    add to selection\n"
            "Drag empty area  marquee select\n"
            "Alt + drag       duplicate\n"
            "Arrows           nudge 1 px  (Shift = 10 px)\n"
            "Shift while resizing keeps the aspect ratio\n\n"
            "Ctrl+Z / Ctrl+Shift+Z   undo / redo  (Ctrl+Y also redoes)\n"
            "Ctrl+C / Ctrl+V  copy / paste regions\n"
            "Ctrl+D           duplicate\n"
            "Ctrl+A           select all regions\n"
            "Del              delete\n\n"
            "F5 / F6          run once / run continuous\n"
            "F8               diagnostics\n"
            "F9               production history\n"
            "Ctrl+0 / 1 / 2   fit / actual size / zoom to selection")

    # -------------------------------------------------------- projects
    def reload_projects(self):
        self.projBox.blockSignals(True)
        self.projBox.clear()
        items = self.store.list()
        for p in items:
            self.projBox.addItem(p["id"])
        self.projBox.blockSignals(False)
        if items:
            self.load_project(items[0]["id"])

    def confirm_discard(self):
        """True to go ahead. Offers to save unsaved edits first."""
        if not self.dirty:
            return True
        choice = confirm_unsaved(self)
        if choice == "save":
            self.save_job()
        return choice != "cancel"

    def on_project_change(self, pid):
        """Switching model or side. This used to load the new one straight
        away, so choosing another model in the dropdown silently discarded any
        unsaved edits. Opening a side from the job tree goes through here too."""
        if not pid:
            return
        prev = getattr(self, "_loaded_pid", None)
        if prev and pid != prev and not self.confirm_discard():
            self.projBox.blockSignals(True)
            self.projBox.setCurrentText(prev)
            self.projBox.blockSignals(False)
            return
        self.load_project(pid)

    def load_project(self, pid):
        try:
            self.project = self.store.reload(pid)
        except Exception as e:
            QMessageBox.critical(self, "Could not open that model", str(e))
            return
        self._loaded_pid = pid
        if getattr(self, "_editors", None):
            self.close_editors()
        self.spec = json.loads(json.dumps(self.project.spec))
        if self.results:
            self.results.close()
        self.results = ResultStore(self.project.folder, pid, archive=self.archive_mode)
        self.ds = Dataset(self.project.folder)
        self.history.reset()
        self.cur_image = None
        self.dirty = False
        self.last = None
        self.refresh_tree()
        self.refresh_images()
        if self.project.reference is not None and self.live is None:
            self.canvas.set_image(self.project.reference, keep_view=False)
        self.canvas.sync_regions(self.spec.get("tools", []))
        self.refresh_model_label()
        self.refresh_classes()
        self.refresh_training()
        self.props.set_captures(self.capture_names())
        self.update_edit_actions()
        self.setWindowTitle(f"Vision Station — {self.spec.get('name', pid)}")

    def new_project(self):
        name, ok = QInputDialog.getText(self, "New model", "Name this model or variant")
        if not ok or not name.strip():
            return
        pid = name.strip().lower().replace(" ", "_")
        self.store.create(pid, name.strip())
        self.reload_projects()
        self.projBox.setCurrentText(pid)

    def save_job(self):
        if self.project:
            self.project.save(self.spec)
            self.dirty = False
            self.statusBar().showMessage("Job saved", 2500)

    def export_job(self):
        if not self.spec:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export job", "job.json", "Job (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.spec, fh, indent=2)

    # ---------------------------------------------------- test images
    def images_dir(self):
        if not self.project:
            return None
        d = os.path.join(self.project.folder, "images")
        os.makedirs(d, exist_ok=True)
        return d

    def refresh_images(self):
        self.imgList.clear()
        d = self.images_dir()
        if not d:
            return
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(IMG_EXT):
                continue
            path = os.path.join(d, name)
            it = QListWidgetItem(QIcon(QPixmap(path).scaled(
                96, 62, Qt.KeepAspectRatio, Qt.SmoothTransformation)), name)
            it.setData(Qt.UserRole, path)
            self.imgList.addItem(it)

    def add_images(self):
        d = self.images_dir()
        if not d:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Add test images", "",
                                                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        import shutil
        for p in paths:
            try:
                shutil.copy(p, os.path.join(d, os.path.basename(p)))
            except shutil.SameFileError:
                pass
        self.refresh_images()

    def remove_image(self):
        it = self.imgList.currentItem()
        if not it:
            return
        try:
            os.remove(it.data(Qt.UserRole))
        except OSError as e:
            QMessageBox.warning(self, "Remove image", str(e))
        self.refresh_images()

    def on_image_picked(self, it):
        img = imio.imread(it.data(Qt.UserRole))
        if img is None:
            return
        self.live = None
        self.cur_image = it.text()
        self.canvas.set_image(img, keep_view=True)
        if self.workspace == "label":
            self.load_labels()
        else:
            self.run_once()

    def run_batch(self):
        if self.imgList.count() == 0:
            QMessageBox.information(self, "No test images",
                                    "Add some images to this model first.")
            return
        rows = []
        for i in range(self.imgList.count()):
            it = self.imgList.item(i)
            img = imio.imread(it.data(Qt.UserRole))
            if img is None:
                continue
            res = self.inspect(img)
            if res:
                rows.append((it.text(), res["verdict"], res["total_ms"]))
        if rows:
            body = "\n".join(f"{n:<34} {v:<5} {ms:7.1f} ms" for n, v, ms in rows)
            npass = sum(1 for _, v, _ in rows if v == "PASS")
            QMessageBox.information(self, "Batch result",
                                    f"{npass} of {len(rows)} passed\n\n{body}")

    # ------------------------------------------------------------ tree
    SIDE_ROLE = Qt.UserRole + 1

    def current_pid(self):
        return self.projBox.currentText() if hasattr(self, "projBox") else ""

    SRC_ROLE = Qt.UserRole + 2

    def _add_tool_items(self, parent, side_pid=None):
        """A job's children: its image source, then its tools in the order they
        run. VisionPro shows a job the same way -- the image source first,
        because everything below it depends on which image arrives."""
        src = QTreeWidgetItem(["Image source", ""])
        src.setData(0, self.SRC_ROLE, side_pid or self.current_pid())
        src.setIcon(0, dot("#8E9DAC"))
        src.setToolTip(0, "Camera or image file for this job - double-click to set up")
        parent.addChild(src)
        for i, t in enumerate(self.spec.get("tools", [])):
            res = self._result_for(t["id"])
            it = QTreeWidgetItem([t["id"], ""])
            it.setData(0, Qt.UserRole, i)
            from vision_station.ui import icons
            it.setIcon(0, icons.tool_icon(t["type"]))      # what kind of tool
            kind = LABELS.get(t["type"], (t["type"], ""))[0]
            it.setToolTip(0, "%s\n%s" % (kind, LABELS.get(t["type"], ("", ""))[1]))
            it.setText(1, res["status"].upper() if res else "")
            if t.get("type") == "align":
                it.setText(0, "%s  (fixture)" % t["id"])
            if t.get("params", {}).get("enabled", True) is False:
                it.setForeground(0, QColor("#63707D"))
                it.setText(1, "off")
            if res:
                col = {"pass": PASS, "fail": FAIL}.get(res["status"], WARN)
                it.setIcon(1, dot(col.name()))
                it.setForeground(1, col)
            else:
                it.setIcon(1, dot("#5A6673"))
            parent.addChild(it)

    def refresh_tree(self):
        """Application -> Jobs -> Image source and tools.

        Every job is listed and can be expanded, so the whole application is
        visible at once; the open one is marked and carries the live results.
        A plain model has no job layer and lists its tools directly."""
        self._syncing = True
        self.tree.clear()
        if self.spec:
            pid = self.current_pid()

            def bold(it):
                f = it.font(0)
                f.setBold(True)
                it.setFont(0, f)

            if "/" in pid:
                job_id, side_id = pid.split("/", 1)
                sides = [x for x in self.store.list() if x.get("job") == job_id]
                root = QTreeWidgetItem([sides[0]["job_name"] if sides else job_id, ""])
                root.setIcon(0, dot("#0E90D2"))
                bold(root)
                self.tree.addTopLevelItem(root)
                for x in sides:
                    si = QTreeWidgetItem([x["name"], ""])
                    si.setData(0, self.SIDE_ROLE, x["id"])
                    si.setToolTip(0, "Job %s - click to open" % x["side"])
                    root.addChild(si)
                    if x["side"] == side_id:
                        si.setIcon(0, dot("#18C25A"))
                        bold(si)
                        self._add_tool_items(si, x["id"])
                        si.setExpanded(True)
                    else:
                        si.setIcon(0, dot("#3E4652"))
                        # a placeholder so unopened jobs can still be expanded
                        ph = QTreeWidgetItem(["open this job to see its tools", ""])
                        ph.setForeground(0, QColor("#63707D"))
                        si.addChild(ph)
                root.setExpanded(True)
            else:
                root = QTreeWidgetItem([self.spec.get("name", "Job"), ""])
                root.setIcon(0, dot("#0E90D2"))
                bold(root)
                self.tree.addTopLevelItem(root)
                self._add_tool_items(root)
                root.setExpanded(True)
        self._syncing = False
        # Every path that changes the recipe -- undo, redo, paste, delete, a
        # setting, a reload -- ends by refreshing this tree, so open tool editors
        # are refreshed here too. Refreshing them path by path missed undo: the
        # recipe went back and the editor kept showing the moved region.
        if getattr(self, "_editors", None):
            self.refresh_editors()

    def on_tree_clicked(self, item, _col):
        """Clicking a job opens it; the image source opens its setup."""
        pid = item.data(0, self.SIDE_ROLE)
        if pid and pid != self.current_pid():
            self.projBox.setCurrentText(pid)      # the unsaved-edits guard is there
            return
        if item.data(0, self.SRC_ROLE):
            self.image_source_setup()

    def image_source_setup(self):
        """What feeds this job: a camera, or images from disk."""
        m = QMenu(self)
        m.addAction("Start camera" if not self.camera else "Camera settings...",
                    self.show_camera_settings if self.camera else self.toggle_camera)
        m.addAction("Load image from disk...", self.open_image)
        m.addAction("Add test images...", self.add_images)
        m.addSeparator()
        m.addAction("Set the current image as the reference", self.set_reference)
        m.exec(QCursor.pos())

    def _result_for(self, tid):
        if not self.last:
            return None
        return next((t for t in self.last["tools"] if t["tool_id"] == tid), None)

    def selected_indices(self):
        return [it.data(0, Qt.UserRole) for it in self.tree.selectedItems()
                if isinstance(it.data(0, Qt.UserRole), int)]

    def selected_ids(self):
        return [self.spec["tools"][i]["id"] for i in self.selected_indices()]

    def on_tree_select(self):
        if self._syncing:
            return
        ids = self.selected_ids()
        self._syncing = True
        self.canvas.select_ids(ids, silent=True)
        self._syncing = False
        self._load_props(ids)

    def _tool_items(self):
        """Every tree item that stands for a tool, at whatever depth. Under a
        job the tools sit beneath their side rather than beneath the root, and
        code that assumed the root's children were tools would read a side's
        empty index and crash on the first click."""
        out, stack = [], [self.tree.topLevelItem(i)
                          for i in range(self.tree.topLevelItemCount())]
        while stack:
            it = stack.pop()
            if it is None:
                continue
            if isinstance(it.data(0, Qt.UserRole), int):
                out.append(it)
            stack.extend(it.child(i) for i in range(it.childCount()))
        return out

    def on_canvas_selection(self, ids):
        if self._syncing:
            return
        self._syncing = True
        self.tree.clearSelection()
        for ch in self._tool_items():
            i = ch.data(0, Qt.UserRole)
            if 0 <= i < len(self.spec["tools"]) and self.spec["tools"][i]["id"] in ids:
                ch.setSelected(True)
        self._syncing = False
        self._load_props(ids)
        self.refresh_training()

    def _load_props(self, ids):
        self.props.recipe = self.spec.get("tools", []) if self.spec else []
        if len(ids) == 1:
            t = next((x for x in self.spec["tools"] if x["id"] == ids[0]), None)
            # The images a tool can read: the camera captures, plus the output of
            # any image-processing tool that runs before it -- not after, since
            # that image does not exist yet when this tool runs.
            self.props.set_captures(readable_images(
                self.spec.get("tools", []), t["id"] if t else None, self.capture_names()))
            self.props.load(t)
            if t:
                self.props.show_result(self._result_for(t["id"]))
        else:
            self.props.load(None, multi=len(ids))

    def select_all(self):
        self.canvas.select_ids(list(self.canvas.regions))

    # ----------------------------------------------------- undo / redo
    def mark(self, label):
        """Record the state as it is right now, before a change is made."""
        if self.spec is not None:
            self.history.push(label, self.spec)

    def apply_snapshot(self, snap):
        self.spec = snap
        self.history.mute(True)
        try:
            self.refresh_tree()
            self.canvas.sync_regions(self.spec.get("tools", []), self.last)
            sel = self.canvas.selected_ids()
            existing = [i for i in sel if i in self.canvas.regions]
            self.canvas.select_ids(existing, silent=True)
            self._load_props(existing)
        finally:
            self.history.mute(False)
        self.dirty = True
        self.update_edit_actions()

    def undo(self):
        # A trace in progress owns the keystroke. Restoring a recipe snapshot
        # underneath a half-drawn polygon would discard the points silently and
        # undo something the operator was not looking at.
        if self.canvas.tracing():
            self.canvas.undo_last_point()
            n = len(self.canvas._poly_pending)
            self.statusBar().showMessage(
                "Removed the last point - %d left" % n if n
                else "Trace cancelled", 3000)
            return
        if not self.history.can_undo():
            self.statusBar().showMessage("Nothing to undo", 1500)
            return
        snap, label = self.history.undo(self.spec)
        self.apply_snapshot(snap)
        self.statusBar().showMessage("Undid %s" % label, 2000)

    def redo(self):
        if not self.history.can_redo():
            self.statusBar().showMessage("Nothing to redo", 1500)
            return
        snap, label = self.history.redo(self.spec)
        self.apply_snapshot(snap)
        self.statusBar().showMessage("Redid %s" % label, 2000)

    def update_edit_actions(self):
        u, r = self.history.can_undo(), self.history.can_redo()
        self.aUndo.setEnabled(u)
        self.aRedo.setEnabled(r)
        self.aUndo.setText("Undo %s" % self.history.undo_label() if u else "Undo")
        self.aRedo.setText("Redo %s" % self.history.redo_label() if r else "Redo")

    def on_edit_began(self, label):
        self.mark(label)

    # ------------------------------------------------------- tool edit
    def add_tool_menu(self):
        m = QMenu(self)
        fill_tool_menu(m, self.add_tool)
        m.exec(self.cursor().pos())

    def _insert_tool(self, ty, params, verb):
        """Append a new tool, record it for undo, and select it. Adding from a
        menu and drawing on the canvas each carried a copy of this."""
        tid = self._unique_id(ty.split("_")[0])
        self.history.break_coalesce()
        self.mark("%s %s" % (verb, LABELS.get(ty, (ty,))[0].lower()))
        self.spec.setdefault("tools", []).append({"id": tid, "type": ty, "params": params})
        self.dirty = True
        self.refresh_tree()
        self.canvas.sync_regions(self.spec["tools"], self.last)
        self.canvas.select_ids([tid])
        self.props.tabs.setCurrentIndex(0)
        self.update_edit_actions()
        self.refresh_model_label()
        return tid

    def _unique_id(self, base):
        existing = {t["id"] for t in self.spec.get("tools", [])}
        n = 1
        while f"{base}_{n}" in existing:
            n += 1
        return f"{base}_{n}"

    def add_tool(self, ty):
        if not self.spec:
            QMessageBox.information(self, "No model open", "Create a model first.")
            return
        img = self.canvas._img
        roi = None
        if img is not None:
            h, w = img.shape[:2]
            roi = [int(w * .35), int(h * .35), int(w * .22), int(h * .18)]
        p = defaults(ty)
        p["roi"] = roi
        self._insert_tool(ty, p, "add")

    def copy_sel(self):
        ids = self.canvas.selected_ids() or self.selected_ids()
        self.clipboard = [json.loads(json.dumps(t)) for t in self.spec["tools"]
                          if t["id"] in ids]
        if self.clipboard:
            self.statusBar().showMessage(
                f"Copied {len(self.clipboard)} region"
                f"{'s' if len(self.clipboard) > 1 else ''}", 2000)

    def paste(self):
        if not self.clipboard or not self.spec:
            return
        self._insert(self.clipboard, offset=14)

    def duplicate_sel(self):
        self.duplicate_ids(self.canvas.selected_ids() or self.selected_ids())

    def duplicate_ids(self, ids):
        src = [t for t in self.spec["tools"] if t["id"] in ids]
        if src:
            self._insert([json.loads(json.dumps(t)) for t in src], offset=14)

    def _insert(self, tools, offset=0):
        self.history.break_coalesce()
        self.mark("paste" if offset else "duplicate")
        new_ids = []
        for t in tools:
            c = json.loads(json.dumps(t))
            base = c["id"].rstrip("0123456789_") or c["type"].split("_")[0]
            c["id"] = self._unique_id(base)
            roi = (c.get("params") or {}).get("roi")
            if roi and offset:
                c["params"]["roi"] = [roi[0] + offset, roi[1] + offset, roi[2], roi[3]]
            self.spec["tools"].append(c)
            new_ids.append(c["id"])
        self.dirty = True
        self.refresh_tree()
        self.canvas.sync_regions(self.spec["tools"], self.last)
        self.canvas.select_ids(new_ids)
        self.update_edit_actions()
        self.statusBar().showMessage(f"Added {', '.join(new_ids)}", 2500)

    def remove_tool(self):
        self.delete_ids(self.canvas.selected_ids() or self.selected_ids())

    def delete_ids(self, ids):
        if not ids:
            return
        self.history.break_coalesce()
        self.mark("delete %d region%s" % (len(ids), "s" if len(ids) > 1 else ""))
        self.spec["tools"] = [t for t in self.spec["tools"] if t["id"] not in ids]
        self.dirty = True
        self.refresh_tree()
        self.canvas.sync_regions(self.spec["tools"], self.last)
        self.props.load(None)
        self.update_edit_actions()
        self.refresh_model_label()

    def move_tool(self, d):
        idx = self.selected_indices()
        if len(idx) != 1:
            return
        i = idx[0]
        j = i + d
        if not 0 <= j < len(self.spec["tools"]):
            return
        ts = self.spec["tools"]
        self.history.break_coalesce()
        self.mark("reorder")
        ts[i], ts[j] = ts[j], ts[i]
        self.dirty = True
        tid = ts[j]["id"]
        self.refresh_tree()
        self.canvas.select_ids([tid])

    def rename_tool(self):
        idx = self.selected_indices()
        if len(idx) != 1:
            return
        t = self.spec["tools"][idx[0]]
        new, ok = QInputDialog.getText(self, "Rename", "Name", text=t["id"])
        if ok and new.strip():
            self.history.break_coalesce()
            self.mark("rename")
            t["id"] = new.strip().replace(" ", "_")
            self.dirty = True
            self.refresh_tree()
            self.canvas.sync_regions(self.spec["tools"], self.last)

    def tree_menu(self, pos):
        m = QMenu(self)
        add = m.addMenu("Add tool")
        fill_tool_menu(add, self.add_tool)
        if self.selected_indices():
            m.addSeparator()
            m.addAction("Rename...", self.rename_tool)
            m.addAction("Copy", self.copy_sel)
            m.addAction("Duplicate", self.duplicate_sel)
            m.addAction("Move up", lambda: self.move_tool(-1))
            m.addAction("Move down", lambda: self.move_tool(1))
            m.addSeparator()
            m.addAction("Delete", self.remove_tool)
        m.exec(self.tree.viewport().mapToGlobal(pos))

    def canvas_menu(self, pos):
        m = QMenu(self)
        ids = self.canvas.selected_ids()
        if ids:
            m.addAction("Copy", self.copy_sel)
            m.addAction("Duplicate", self.duplicate_sel)
            m.addAction("Delete", self.remove_tool)
            m.addSeparator()
        if self.clipboard:
            m.addAction(f"Paste {len(self.clipboard)} region(s)", self.paste)
        add = m.addMenu("Add tool here")
        fill_tool_menu(add, self.add_tool)
        m.addSeparator()
        m.addAction("Select all regions", self.select_all)
        m.exec(self.canvas.mapToGlobal(pos))

    # -------------------------------------------------------- geometry
    def on_live_geometry(self, tid, roi):
        t = next((x for x in self.spec["tools"] if x["id"] == tid), None)
        if t:
            t["params"]["roi"] = list(roi)
            if self.props.tool is t:
                self.props.set_roi(roi, emit=False)

    def on_angle_changed(self, tid, deg):
        self.update_edit_actions()
        t = next((x for x in self.spec["tools"] if x["id"] == tid), None)
        if t:
            t["params"]["angle"] = round(float(deg), 3)
            self.dirty = True
            if self.props.tool is t:
                self.props.set_angle(t["params"]["angle"])

    def on_geometry(self, tid, roi):
        self.on_live_geometry(tid, roi)
        self.dirty = True
        self.update_edit_actions()
        if tid in self._editors:
            self._editors[tid].refresh()

    def on_props_changed(self):
        self.mark("change setting")
        self.dirty = True
        self.refresh_tree()
        self.canvas.sync_regions(self.spec["tools"], self.last)
        self.canvas.select_ids([self.props.tool["id"]] if self.props.tool else [], silent=True)
        self.refresh_editors()

    # ------------------------------------------------------ tool editors
    def open_tool_editor(self, tid=None):
        """Open a tool in its own editor window, as VisionPro does. One window
        per tool; asking again brings the existing one forward."""
        from vision_station.ui.tool_editor import ToolEditor
        if tid is None:
            ids = self.canvas.selected_ids()
            if len(ids) != 1:
                self.statusBar().showMessage("Select one tool to open its editor", 3000)
                return None
            tid = ids[0]
        if not any(t["id"] == tid for t in self.spec.get("tools", [])):
            return None
        ed = self._editors.get(tid)
        if ed is None:
            ed = ToolEditor(self, tid)
            self._editors[tid] = ed
        ed.show()
        ed.raise_()
        ed.activateWindow()
        return ed

    def refresh_editors(self):
        for ed in list(self._editors.values()):
            ed.refresh()

    def close_editors(self):
        for ed in list(self._editors.values()):
            ed.close()
        self._editors.clear()

    def editor_changed(self, tid):
        """A tool was edited in its own window: redraw the main window, then
        every editor, from the one recipe they share."""
        self.dirty = True
        self.refresh_tree()
        self.canvas.sync_regions(self.spec["tools"], self.last)
        if self.props.tool is not None and self.props.tool.get("id") == tid:
            self.props.load(self.props.tool)
        self.update_edit_actions()
        self.refresh_editors()

    def on_tree_double_clicked(self, item, _col):
        i = item.data(0, Qt.UserRole)
        if isinstance(i, int) and 0 <= i < len(self.spec.get("tools", [])):
            self.open_tool_editor(self.spec["tools"][i]["id"])

    def set_canvas_mode(self, mode):
        self.canvas.set_mode(mode)
        self.aSelect.setChecked(mode == "select")
        self.aHand.setChecked(mode == "hand")

    def _build_banner(self):
        """A verdict that reads across a workshop.

        The status bar was carrying it before, in twelve-point text at the very
        bottom of the screen. An operator standing at a station glances at the
        monitor for a fraction of a second and needs the answer, not a label to
        go hunting for."""
        w = QWidget()
        w.setObjectName("banner")
        h = QHBoxLayout(w)
        h.setContentsMargins(14, 8, 14, 8)
        h.setSpacing(18)

        self.bigVerdict = QLabel("READY")
        f = QFont("Segoe UI", 22)
        f.setBold(True)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        self.bigVerdict.setFont(f)
        self.bigVerdict.setMinimumWidth(190)
        self.bigVerdict.setAlignment(Qt.AlignCenter)
        h.addWidget(self.bigVerdict)

        self.bannerDetail = QLabel("No inspection yet")
        self.bannerDetail.setStyleSheet("color:#9AA9B8; font-size:12px;")
        self.bannerDetail.setWordWrap(True)
        h.addWidget(self.bannerDetail, 1)

        stats = QWidget()
        sg = QHBoxLayout(stats)
        sg.setContentsMargins(0, 0, 0, 0)
        sg.setSpacing(22)
        self.bannerStats = {}
        for key, cap in (("yield", "YIELD"), ("cycle", "CYCLE"),
                         ("count", "INSPECTED")):
            col = QVBoxLayout()
            col.setSpacing(1)
            val = QLabel("\u2014")
            val.setFont(QFont("Consolas", 15))
            val.setAlignment(Qt.AlignRight)
            cl = QLabel(cap)
            cl.setStyleSheet("color:#77869A; font-size:9px; letter-spacing:0.6px;")
            cl.setAlignment(Qt.AlignRight)
            col.addWidget(val)
            col.addWidget(cl)
            sg.addLayout(col)
            self.bannerStats[key] = val
        h.addWidget(stats)

        w.setStyleSheet("#banner { background:#282D35; "
                        "border-bottom:1px solid #14171B; }")
        self._banner = w
        self._paint_verdict("READY", "#3E4652", "#9AA9B8")

        holder = QWidget()
        v = QVBoxLayout(holder)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(w)
        v.addWidget(self.canvas, 1)
        self.setCentralWidget(holder)

    def _paint_verdict(self, text, bg, fg):
        self.bigVerdict.setText(text)
        self.bigVerdict.setStyleSheet(
            "background:%s; color:%s; border-radius:5px; padding:6px 18px;" % (bg, fg))

    def update_banner(self, res=None):
        counts = self.counts
        total = sum(counts.values())
        npass = counts.get("PASS", 0)
        self.bannerStats["yield"].setText(
            "%.1f%%" % (100.0 * npass / total) if total else "\u2014")
        self.bannerStats["count"].setText(str(total) if total else "\u2014")
        if res is None:
            return
        self.bannerStats["cycle"].setText("%.0f ms" % res["total_ms"])
        v = res["verdict"]
        bg, fg = {"PASS": ("#12A150", "#04140A"),
                  "FAIL": ("#E5484D", "#1B0405"),
                  "WARN": ("#F5A524", "#1A1102")}[v]
        self._paint_verdict(v, bg, fg)
        bad = [t for t in res["tools"] if t["status"] in ("fail", "error")]
        if not bad:
            self.bannerDetail.setText("All %d checks passed" % len(res["tools"]))
        else:
            first = bad[0]
            more = ("  (+%d more)" % (len(bad) - 1)) if len(bad) > 1 else ""
            self.bannerDetail.setText(
                "%s: %s%s" % (first["tool_id"],
                              first["message"] or first["status"], more))

    # ----------------------------------------------------------- palette
    TOOLS = [
        ("select",       "V", "Select",         "Click a region to select it. "
         "Shift-click to add, drag empty space to marquee."),
        ("draw_rect",    "R", "Rectangle",      "Drag on the image to draw a "
         "rectangular region."),
        ("draw_ellipse", "E", "Ellipse",        "Drag to draw a round region. "
         "Use for pads, screw heads and LEDs so the square corners are excluded."),
        ("draw_polygon", "P", "Polygon",        "Click each corner, then press "
         "Enter or right-click to close. Use for angled or irregular features."),
        ("hand",         "H", "Pan",            "Drag to move the image. Space "
         "does this temporarily whatever tool is active."),
    ]

    def _build_palette(self):
        """A visible palette rather than a shape dropdown buried in a tab.

        Elliptical regions existed for some time and went unfound, which is a
        discoverability failure and not a user error: a capability nobody can
        see may as well not be there."""
        bar = QToolBar("Tools")
        bar.setOrientation(Qt.Vertical)
        bar.setMovable(True)
        bar.setIconSize(QSize(20, 20))
        bar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        bar.setObjectName("palette")
        self.addToolBar(Qt.LeftToolBarArea, bar)

        group = QActionGroup(self)
        group.setExclusive(True)
        self.toolActions = {}
        for key, shortcut, label, tip in self.TOOLS:
            a = QAction(shape_icon(key), label, self)
            a.setCheckable(True)
            a.setShortcut(QKeySequence(shortcut))
            a.setToolTip("%s  (%s)\n%s" % (label, shortcut, tip))
            a.triggered.connect(lambda _c=False, k=key: self.set_tool(k))
            group.addAction(a)
            bar.addAction(a)
            self.toolActions[key] = a
            if key in ("select", "draw_polygon"):
                bar.addSeparator()
        self.toolActions["select"].setChecked(True)
        self.paletteBar = bar

    def refresh_model_label(self):
        """State an operator otherwise has to go looking for: how many checks
        the recipe has, and whether it takes more than one exposure."""
        if not hasattr(self, "modelLabel") or not self.spec:
            return
        n = len(self.spec.get("tools", []))
        caps = self.capture_names()
        bits = ["%d check%s" % (n, "" if n == 1 else "s")]
        if len(caps) > 1:
            bits.append("%d captures" % len(caps))
        self.modelLabel.setText("   ".join(bits))

    def on_zoom_changed(self, scale):
        if hasattr(self, "zoomLabel"):
            self.zoomLabel.setText("%d%%" % round(scale * 100))

    def set_label_policy(self, mode):
        self.canvas.label_policy = mode
        self.canvas.sync_regions(self.spec.get("tools", []) if self.spec else [],
                                 self.last)
        if self.last:
            self.canvas.draw_results(self.last["tools"])
        self.statusBar().showMessage("Region names: %s" % mode, 3000)

    def set_tool(self, key):
        self.canvas.set_mode(key)
        for k, a in self.toolActions.items():
            a.setChecked(k == key)
        if key.startswith("draw_"):
            what = {"draw_rect": "rectangle", "draw_ellipse": "ellipse",
                    "draw_polygon": "polygon"}[key]
            extra = (" Click each corner, then Enter or right-click to close."
                     if key == "draw_polygon" else " Drag on the image.")
            self.statusBar().showMessage(
                "Drawing a %s region.%s  Esc cancels." % (what, extra), 8000)
        # keep the legacy Select/Hand toolbar buttons in step
        if hasattr(self, "aSelect"):
            self.aSelect.setChecked(key == "select")
        if hasattr(self, "aHand"):
            self.aHand.setChecked(key == "hand")

    def on_shape_drawn(self, shape, geom):
        """A region was drawn; create it as whatever check the type selector
        names.

        The type is chosen before drawing rather than in a popup afterwards.
        A modal menu interrupting every stroke is the wrong shape of
        interaction, and picking the tool first is how a drawing application
        works: you select the brush, then paint."""
        if not self.spec:
            return
        ty = self.newTypeBox.currentData() if hasattr(self, "newTypeBox") else "ocv"
        params = defaults(ty)
        params["shape"] = shape
        if shape == "polygon":
            params["points"] = geom["points"]
            xs = [p[0] for p in geom["points"]]
            ys = [p[1] for p in geom["points"]]
            params["roi"] = [int(min(xs)), int(min(ys)),
                             int(max(xs) - min(xs)), int(max(ys) - min(ys))]
        else:
            params["roi"] = geom["roi"]
        tid = self._insert_tool(ty, params, "draw")
        self.set_tool("select")
        self.statusBar().showMessage("Added %s" % tid, 4000)

    def on_polygon_committed(self, tid, points):
        t = next((x for x in self.spec["tools"] if x["id"] == tid), None)
        if not t:
            return
        t["params"]["points"] = [[float(a), float(b)] for a, b in points]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        t["params"]["roi"] = [int(min(xs)), int(min(ys)),
                              int(max(xs) - min(xs)), int(max(ys) - min(ys))]
        self.dirty = True
        self.update_edit_actions()

    def _build_toolbox(self):
        """The tools that can be added, by category, as VisionPro's Toolbox does.
        Double-click one to add it to the open job. They were only in a menu
        before, which hides what the application can actually do."""
        self.toolbox = toolbox_tree(self.add_tool, group_colour="#8E9DAC")
        d = QDockWidget("Toolbox", self)
        d.setObjectName("toolboxDock")
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)
        hint = QLabel("Double-click a tool to add it to this job")
        hint.setStyleSheet("color:#8E9DAC;")
        hint.setWordWrap(True)
        v.addWidget(hint)
        v.addWidget(self.toolbox, 1)
        d.setWidget(w)
        self.addDockWidget(Qt.LeftDockWidgetArea, d)
        self.toolboxDock = d
        # The job tree is the panel in constant use, so it gets the room. Left
        # to Qt the three left docks shared the column evenly and the tree
        # showed two rows of a ten-tool job.
        # Toolbox and Test images share one tabbed panel under the job tree:
        # three stacked docks left the tree showing two rows of a ten-tool job,
        # and Test images is usually empty while you are building.
        self.tabifyDockWidget(self.toolboxDock, self.imagesDock)
        self.toolboxDock.raise_()
        QTimer.singleShot(0, lambda: self.resizeDocks(
            [self.jobDock, self.toolboxDock], [520, 340], Qt.Vertical))

    def _toolbox_add(self, item, _col=0):
        self.toolbox.add_item(item)

    # ------------------------------------------------------ lens calibration
    def calibrate_lens(self, images=None, pattern=None, square_mm=None):
        """Measure the lens from checkerboard photographs and correct every frame
        from then on. Arguments are for scripts and tests; interactively they
        are asked for."""
        from vision_station.core import calibration
        if not self.project:
            return None
        if images is None:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Checkerboard photographs - 8 to 15, reaching the frame corners",
                self.project.folder, "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
            if not paths:
                return None
            images = [imio.imread(pth) for pth in paths]
            images = [im for im in images if im is not None]
        if pattern is None:
            cols, ok = QInputDialog.getInt(self, "Calibrate lens",
                                           "Inner corners across the board:", 9, 3, 50)
            if not ok:
                return None
            rows, ok = QInputDialog.getInt(self, "Calibrate lens",
                                           "Inner corners down the board:", 6, 3, 50)
            if not ok:
                return None
            pattern = (cols, rows)
        if square_mm is None:
            square_mm, ok = QInputDialog.getDouble(self, "Calibrate lens",
                                                   "Square size (mm):", 10.0, 0.1, 500.0, 3)
            if not ok:
                return None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            cal = calibration.calibrate(images, pattern, square_mm)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Calibrate lens", str(e))
            return None
        QApplication.restoreOverrideCursor()
        calibration.save(self.project.folder, cal)
        self.load_project(self.current_pid())       # reference re-corrected on load
        quality = ("good" if cal["rms_px"] < 0.5 else
                   "usable" if cal["rms_px"] < 1.0 else
                   "poor - blurred or mis-detected images")
        msg = ("Lens calibrated from %d image%s. Reprojection error %.3f px (%s)." % (
            len(cal["images_used"]), "" if len(cal["images_used"]) == 1 else "s",
            cal["rms_px"], quality))
        if cal["images_rejected"]:
            msg += ("\n\n%d image%s rejected - the whole board must be visible." % (
                len(cal["images_rejected"]), "" if len(cal["images_rejected"]) == 1 else "s"))
        msg += ("\n\nEvery frame and the reference are now corrected. For measurements in "
                "millimetres, calibrate the scale with the board lying where the part sits.")
        QMessageBox.information(self, "Calibrate lens", msg)
        return cal

    def remove_lens_calibration(self):
        from vision_station.core import calibration
        if not self.project:
            return
        pth = os.path.join(self.project.folder, calibration.FILE)
        if os.path.exists(pth):
            os.remove(pth)
            self.load_project(self.current_pid())
            self.statusBar().showMessage("Lens calibration removed", 4000)

    # ------------------------------------------------------------ run mode
    def _build_run_bar(self):
        """The operator's controls, shown only in Run mode: large buttons that
        work with gloves on, and nothing that edits the model."""
        bar = QToolBar("Run")
        bar.setObjectName("runModeBar")
        bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        bar.setStyleSheet(
            "QToolBar#runModeBar{background:#1B1F24;border-bottom:1px solid #14171B;"
            "padding:8px 10px;spacing:10px;}"
            "QToolBar#runModeBar QToolButton{font-size:15px;font-weight:600;"
            "padding:12px 26px;border-radius:6px;background:#2B323B;border:1px solid #3A434E;}"
            "QToolBar#runModeBar QToolButton:hover{background:#353E49;}")
        self.addToolBar(Qt.TopToolBarArea, bar)
        self.insertToolBarBreak(bar)
        self.rmInspect = bar.addAction("Inspect once", self.run_once)
        self.rmStart = bar.addAction("Start", self.toggle_continuous)
        bar.addAction("Reset counts", self.reset_counts)
        bar.addSeparator()
        self.rmStrip = QLabel("")
        self.rmStrip.setToolTip("The most recent inspections, newest on the right")
        bar.addWidget(self.rmStrip)
        bar.addAction("History", self.show_history)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        self.rmModel = QLabel("")
        self.rmModel.setStyleSheet("color:#9AA9B8;font-size:13px;padding-right:12px;")
        bar.addWidget(self.rmModel)
        # The way back to Build. The Build/Run switch lives on the builder's
        # toolbar, which Run mode hides -- and Qt does not fire the shortcut of
        # an action whose only widget is hidden -- so without this the only exit
        # from Run mode was closing the application.
        back = bar.addAction("Build mode", lambda: self.set_mode("build"))
        back.setToolTip("Return to building the model  (Ctrl+B)")
        self.runModeBar = bar
        bar.hide()
        self._recent = []

    def set_mode(self, mode):
        """Build: everything for making and tuning a model. Run: the operator's
        screen -- the image, the verdict, the counters and the run controls,
        with every editing surface put away and the regions locked, so a
        stray click on the line cannot move a tool.

        VisionPro keeps these apart too: QuickBuild is where a job is made, and
        the application is where it runs."""
        if mode == getattr(self, "mode", "build"):
            return
        if mode == "run" and self.dirty:
            if not self.confirm_discard():
                self.aModeBuild.setChecked(True)
                return
        self.mode = mode
        run = mode == "run"
        editing = [d for d in self.findChildren(QDockWidget)]
        bars = [self.mainBar, self.runBar, getattr(self, "paletteBar", None)]
        if run:
            self._build_state = {w: w.isVisible() for w in editing + bars if w is not None}
            for w in self._build_state:
                w.hide()
            self.runModeBar.show()
            self.set_tool("select")
            for it in self.canvas.regions.values():
                it.setFlag(QGraphicsItem.ItemIsMovable, False)
                it.setFlag(QGraphicsItem.ItemIsSelectable, False)
            self.canvas.sc.clearSelection()
            self.canvas.fit()
            self.rmModel.setText(self._run_title())
        else:
            if self.continuous:
                self.toggle_continuous()
            self.runModeBar.hide()
            for w, vis in getattr(self, "_build_state", {}).items():
                w.setVisible(vis)
            for it in self.canvas.regions.values():
                it.setFlag(QGraphicsItem.ItemIsMovable, True)
                it.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.aModeBuild.setChecked(not run)
        self.aModeRun.setChecked(run)
        self.menuBar().setVisible(not run)
        self.statusBar().showMessage("Run mode - regions locked" if run else "Build mode", 3000)

    def _run_title(self):
        pid = self.current_pid()
        if "/" in pid:
            job_id, side_id = pid.split("/", 1)
            meta = next((x for x in self.store.list() if x["id"] == pid), {})
            return "%s  -  %s" % (meta.get("job_name", job_id), meta.get("name", side_id))
        return self.spec.get("name", pid) if self.spec else pid

    def _update_recent(self, verdict):
        """A strip of the last thirty verdicts. A run of reds tells the operator
        more than any single result -- it is usually the lamp, the fixture or a
        batch of bad parts, and it shows at a glance."""
        if not hasattr(self, "rmStrip"):
            return
        self._recent = (self._recent + [verdict])[-30:]
        col = {"PASS": "#12A150", "FAIL": "#E5484D", "WARN": "#F5A524"}
        cells = "".join('<span style="color:%s;font-size:20px;">&#9632;</span>'
                        % col.get(v, "#555") for v in self._recent)
        self.rmStrip.setText(cells)
        if self.continuous:
            self.rmStart.setText("Stop")
        else:
            self.rmStart.setText("Start")

    # ---------------------------------------------------------- training
    def _build_training_dock(self):
        """OK and NG columns of training images for the selected tool.

        This is how a trainable tool is taught: real photographs, sorted into
        good and bad, with the tool's region applied to every one of them
        through the side's fixture. Add the image on the canvas as OK or NG,
        press Train, and read the margin -- how far apart the worst good part
        and the best bad part scored."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)
        self.trainTitle = QLabel("Select a trainable tool")
        self.trainTitle.setStyleSheet("font-weight:600; color:#E4E9EF;")
        v.addWidget(self.trainTitle)

        cols, self.trainLists = training_columns(
            self._training_menu, self.add_training, ("#12A150", "#E5484D"),
            "background:%s; color:#0B0D10; font-weight:700; border-radius:3px; padding:3px;")
        for lst in self.trainLists.values():
            lst.itemDoubleClicked.connect(self._training_show)
        v.addLayout(cols, 1)

        row = QHBoxLayout()
        self.trainBtn = QPushButton("Train")
        self.trainBtn.setStyleSheet("QPushButton{background:#0E639C;border-color:#1177B8;"
                                    "font-weight:600;padding:7px 18px;}")
        self.trainBtn.clicked.connect(self.train_selected)
        row.addWidget(self.trainBtn)
        self.testBtn = QPushButton("Test")
        self.testBtn.setToolTip("Run the recipe on every image here and check the "
                                "tool passes the OKs and rejects the NGs")
        self.testBtn.clicked.connect(self.test_selected)
        row.addWidget(self.testBtn)
        self.trainReport = QLabel("")
        self.trainReport.setWordWrap(True)
        self.trainReport.setStyleSheet("color:#9AA9B8;")
        row.addWidget(self.trainReport, 1)
        v.addLayout(row)

        self.trainDock = QDockWidget("Training", self)
        self.trainDock.setObjectName("trainDock")
        self.trainDock.setWidget(w)
        self.trainDock.setMinimumHeight(250)
        self.addDockWidget(Qt.RightDockWidgetArea, self.trainDock)
        # Split the right column roughly evenly. Left to Qt, the property sheet
        # took almost all of it and the training images were a sliver too small
        # to tell one part from another -- which is the whole job of the panel.
        if hasattr(self, "propsDock"):
            self.splitDockWidget(self.propsDock, self.trainDock, Qt.Vertical)
            # Sizing has to wait until the window is laid out; asked for during
            # construction it is silently overridden.
            QTimer.singleShot(0, lambda: self.resizeDocks(
                [self.propsDock, self.trainDock], [360, 460], Qt.Vertical))
        self._thumb_cache = {}
        self.refresh_training()

    def _side(self):
        """The open project as a Side, so it can hold training sets. A plain
        project works too: its training images live in its own folder."""
        from vision_station.core import job as jobmod
        meta = {"id": self.current_pid().split("/")[-1] or "project"}
        return jobmod.Side(self.project.folder, None, meta)

    def _training_tool(self):
        ids = self.canvas.selected_ids() if hasattr(self, "canvas") else []
        if len(ids) != 1 or not self.spec:
            return None
        return next((x for x in self.spec["tools"] if x["id"] == ids[0]), None)

    def refresh_training(self):
        if not hasattr(self, "trainLists"):
            return
        cur = self._training_tool()
        cur_id = cur["id"] if cur else None
        if cur_id != getattr(self, "_marks_for", None):
            self._eval_marks = {}
            self._marks_for = cur_id
        from vision_station.core import job as jobmod
        t = self._training_tool()
        for lst in self.trainLists.values():
            lst.clear()
        enabled = t is not None and self.project is not None
        for lst in self.trainLists.values():
            lst.setEnabled(enabled)
        self.testBtn.setEnabled(enabled)
        if not enabled:
            self.trainBtn.setEnabled(False)
            self.trainTitle.setText("Select a tool to see its OK / NG images")
            self.trainReport.setText("")
            return
        # One panel, three jobs: a trained tool learns from these images, a
        # score-threshold tool has its threshold set from them, and every tool
        # can be tested against them.
        if t["type"] == "trained_surface":
            self.trainBtn.setText("Train")
            self.trainBtn.setEnabled(True)
        elif t["type"] in jobmod.TUNABLE:
            self.trainBtn.setText("Tune threshold")
            self.trainBtn.setEnabled(True)
        else:
            self.trainBtn.setText("Train")
            self.trainBtn.setEnabled(False)
        self.trainTitle.setText("OK / NG images - %s" % t["id"])
        marks = getattr(self, "_eval_marks", {})
        side = self._side()
        fixture = None
        for path, label in side.training_images(t["id"]):
            thumb = self._region_thumb(side, t, path, fixture)
            if thumb is None:
                continue
            if path in marks:
                # green frame: judged correctly; red: a false reject or an escape
                thumb = cv2.copyMakeBorder(thumb[3:-3, 3:-3], 3, 3, 3, 3, cv2.BORDER_CONSTANT,
                                           value=(80, 190, 40) if marks[path] else (60, 60, 230))
            item = QListWidgetItem(QIcon(ndarray_to_qpixmap(thumb)),
                                   os.path.basename(path).split(".")[0][-10:])
            item.setToolTip(path)
            item.setData(Qt.UserRole, path)
            self.trainLists[label].addItem(item)
        n_ok = self.trainLists["ok"].count()
        n_ng = self.trainLists["ng"].count()
        trained = os.path.exists(os.path.join(self.project.folder,
                                              t.get("params", {}).get("model") or
                                              "models/%s.npz" % t["id"]))
        self.trainReport.setText("%d OK, %d NG - %s" % (
            n_ok, n_ng, "trained" if trained else "not trained yet"))

    def _region_thumb(self, side, tool_spec, path, _unused=None):
        # sides are made fresh per refresh; keep their thumbnail cache here
        side.__dict__["_thumbs"] = self._thumb_cache
        return side.region_thumb(tool_spec, path)

    def add_training(self, label):
        t = self._training_tool()
        img = self.canvas._img
        if t is None or img is None:
            return
        self._side().add_training_image(t["id"], img, label)
        self.refresh_training()
        self.statusBar().showMessage("Added the current image to %s as %s"
                                     % (t["id"], label.upper()), 4000)

    def _training_menu(self, lst, label, pos):
        if training_menu(self, lst, label, pos, self._side(),
                         extra=[("Show on canvas", self._training_show)]):
            self.refresh_training()

    def tune_selected(self):
        t = self._training_tool()
        if t is None:
            return
        if self.dirty:
            self.save_job()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rep = self._side().tune_threshold(t["id"])
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Tune threshold", str(e))
            return
        QApplication.restoreOverrideCursor()
        self.project = self.store.reload(self.current_pid())
        self.spec = json.loads(json.dumps(self.project.spec))
        self.dirty = False
        self._load_props(self.canvas.selected_ids())
        if rep["separates"]:
            self.trainReport.setText(
                "%s set to %.3f - midway between the lowest OK (%.3f) and the highest "
                "NG (%.3f). Margin %.3f." % (rep["param"], rep["new"], rep["lowest_ok"],
                                             rep["highest_ng"], rep["margin"]))
            self.trainReport.setStyleSheet("color:#18C25A;")
        else:
            self.trainReport.setText(
                "Not changed: an NG scored %.3f, at or above the lowest OK at %.3f. These "
                "images cannot be separated by this tool's threshold - check the labels, "
                "or the region." % (rep["highest_ng"], rep["lowest_ok"]))
            self.trainReport.setStyleSheet("color:#E5484D;")

    def _training_show(self, item):
        img = imio.imread(item.data(Qt.UserRole))
        if img is not None:
            self.canvas.set_image(img, keep_view=True)

    def test_selected(self):
        t = self._training_tool()
        if t is None:
            return
        if self.dirty:
            self.save_job()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ev = self._side().evaluate_tool(t["id"])
        finally:
            QApplication.restoreOverrideCursor()
        self._eval_marks = {r["path"]: r["correct"] for r in ev["rows"]}
        self.refresh_training()
        from vision_station.core import job as jobmod
        text, level = jobmod.evaluation_summary(ev)
        self.trainReport.setText(text)
        self.trainReport.setStyleSheet("color:%s;" % {
            "good": "#18C25A", "warn": "#F5A524", "bad": "#E5484D"}.get(level, "#9AA9B8"))
        self._last_eval = ev

    def train_selected(self):
        from vision_station.core import job as jobmod
        t = self._training_tool()
        if t is None:
            return
        if t["type"] in jobmod.TUNABLE:
            return self.tune_selected()
        if self.dirty:
            self.save_job()
        side = self._side()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rep = side.train_tool(t["id"])
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Training", str(e))
            return
        QApplication.restoreOverrideCursor()
        # the side saved its spec with the model path; pick it up
        self.project = self.store.reload(self.current_pid())
        self.spec = json.loads(json.dumps(self.project.spec))
        self.dirty = False
        msg = "Trained on %d OK (+%d augmented) and %d NG. Threshold %.0f." % (
            rep["real_ok"], rep["augmented"], rep["ng"], rep["threshold"])
        if rep["margin"] is not None:
            msg += "  Margin %s." % rep["margin"]
        bad = rep.get("unseparated_ng_paths") or []
        if bad:
            msg += ("  %d NG image%s scored like a good part and %s set aside - "
                    "check %s really defective." % (
                        len(bad), "s" if len(bad) > 1 else "",
                        "were" if len(bad) > 1 else "was",
                        "they are" if len(bad) > 1 else "it is"))
        self.trainReport.setText(msg)
        self.trainReport.setStyleSheet("color:%s;" % ("#F5A524" if bad else "#18C25A"))
        self.refresh_tree()
        self.statusBar().showMessage("Training finished for %s" % t["id"], 5000)

    # ---------------------------------------------------------- captures
    def capture_names(self):
        caps = (self.spec or {}).get("captures") or []
        return [c.get("id", "default") for c in caps] or ["default"]

    def edit_captures(self):
        """Define the exposures this model needs.

        A switch faceplate needs two. Any exposure long enough to show the panel
        clips the LEDs to white, and any exposure short enough to keep the LED
        colours leaves the panel invisible -- on a real photograph the LED band
        wanted roughly a twentieth of the faceplate's exposure. Rather than
        compromise on one frame and do both jobs badly, the camera takes both
        and each tool says which it reads."""
        if not self.spec:
            return
        d = QDialog(self)
        d.setWindowTitle("Captures")
        d.resize(560, 380)
        lay = QVBoxLayout(d)
        intro = QLabel(
            "One frame per row, taken in order every inspection. Leave the "
            "exposure blank to use whatever the camera is currently set to.\n"
            "Tools choose their capture on the Region tab.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#8C99A6; padding-bottom:6px;")
        lay.addWidget(intro)

        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Name", "Exposure (ms)", "Note"])
        table.setColumnWidth(0, 150)
        table.setColumnWidth(1, 120)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().hide()
        lay.addWidget(table, 1)

        def load(rows):
            table.setRowCount(len(rows))
            for i, c in enumerate(rows):
                table.setItem(i, 0, QTableWidgetItem(str(c.get("id", ""))))
                ms = c.get("exposure_ms")
                table.setItem(i, 1, QTableWidgetItem("" if ms is None else str(ms)))
                table.setItem(i, 2, QTableWidgetItem(str(c.get("note", ""))))

        load(self.spec.get("captures") or [{"id": "default", "exposure_ms": None,
                                            "note": "as the camera is set"}])

        row = QHBoxLayout()
        b_add = QPushButton("Add")
        b_del = QPushButton("Remove")
        b_led = QPushButton("Add an LED capture")
        b_led.setToolTip("Adds a short exposure for LED colour. Use F7 with the "
                         "LED region selected to find the right value.")
        row.addWidget(b_add)
        row.addWidget(b_del)
        row.addWidget(b_led)
        row.addStretch(1)
        lay.addLayout(row)

        def add(name="capture", ms="", note=""):
            i = table.rowCount()
            table.insertRow(i)
            table.setItem(i, 0, QTableWidgetItem(name))
            table.setItem(i, 1, QTableWidgetItem(str(ms)))
            table.setItem(i, 2, QTableWidgetItem(note))

        b_add.clicked.connect(lambda: add("capture%d" % (table.rowCount() + 1)))
        b_led.clicked.connect(lambda: add("leds", "1.0",
                                          "short, so LED cores are not clipped"))
        b_del.clicked.connect(lambda: table.removeRow(table.currentRow())
                              if table.currentRow() >= 0 else None)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        lay.addWidget(bb)
        if d.exec() != QDialog.Accepted:
            return

        caps, seen = [], set()
        for i in range(table.rowCount()):
            name = (table.item(i, 0).text() if table.item(i, 0) else "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            raw = (table.item(i, 1).text() if table.item(i, 1) else "").strip()
            try:
                ms = float(raw) if raw else None
            except ValueError:
                QMessageBox.warning(self, "Captures",
                                    "'%s' is not a number of milliseconds." % raw)
                return
            caps.append({"id": name, "exposure_ms": ms,
                         "note": (table.item(i, 2).text() if table.item(i, 2) else "")})
        if not caps:
            caps = [{"id": "default", "exposure_ms": None, "note": ""}]

        self.history.break_coalesce()
        self.mark("change captures")
        self.spec["captures"] = caps
        # A tool pointing at a capture that no longer exists would quietly fall
        # back to the primary frame, which is the wrong exposure and produces a
        # plausible but wrong result. Repoint them instead and say so.
        names = {c["id"] for c in caps}
        moved = []
        for t in self.spec.get("tools", []):
            want = (t.get("params") or {}).get("capture", "default")
            if want not in names:
                t.setdefault("params", {})["capture"] = caps[0]["id"]
                moved.append(t["id"])
        self.dirty = True
        self.props.set_captures(self.capture_names())
        self._load_props(self.canvas.selected_ids())
        self.update_edit_actions()
        self.refresh_model_label()
        msg = "%d capture%s defined." % (len(caps), "s" if len(caps) > 1 else "")
        if moved:
            msg += ("\n\nThese tools referred to a capture that no longer "
                    "exists and now read '%s':\n   %s" % (caps[0]["id"],
                                                          ", ".join(moved)))
        QMessageBox.information(self, "Captures", msg)

    # ---------------------------------------------------------- camera
    def toggle_camera(self):
        if self.camera:
            self.stop_camera()
            return
        idx = self.pick_camera()
        if idx is None:
            return
        self.camera = CameraThread(idx, self.cam_settings.copy())
        self.camera.start()
        self.pollTimer.start(40)
        self.aCam.setText("Stop camera")
        self.aCam.setIcon(dot("#18C25A"))
        # Give the thread a moment to open the device, then report what we got.
        QTimer.singleShot(1200, self.report_camera)

    def pick_camera(self):
        """Show the cameras Pylon can actually see, rather than making the
        operator guess an index."""
        devices = list_devices()
        if not devices:
            if PYLON_AVAILABLE:
                msg = ("Pylon is installed but no Basler camera was found.\n\n"
                       "Check power and cabling, and confirm Pylon Viewer sees it.\n\n"
                       "Continue with an OpenCV device index instead?")
            else:
                msg = ("pypylon is not installed, so Basler cameras are unavailable.\n\n"
                       "Install the Pylon SDK from baslerweb.com then run\n"
                       "    pip install pypylon\n\n"
                       "Continue with an OpenCV device index instead?")
            if QMessageBox.question(self, "Camera", msg) != QMessageBox.Yes:
                return None
            idx, ok = QInputDialog.getInt(self, "Camera", "OpenCV device index", 0, 0, 16)
            return idx if ok else None

        d = QDialog(self)
        d.setWindowTitle("Choose camera")
        d.setMinimumWidth(420)
        lay = QVBoxLayout(d)
        lay.addWidget(QLabel("Basler cameras found:"))
        grp = QButtonGroup(d)
        for dev in devices:
            rb = QRadioButton("%s  %s\n    serial %s"
                              % (dev["vendor"], dev["model"], dev["serial"]))
            grp.addButton(rb, dev["index"])
            lay.addWidget(rb)
            if dev["index"] == 0:
                rb.setChecked(True)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        lay.addWidget(bb)
        if d.exec() != QDialog.Accepted:
            return None
        return grp.checkedId()

    def report_camera(self):
        if not self.camera:
            return
        if self.camera.backend is None:
            err = self.camera.last_error or "The camera did not start."
            self.stop_camera()
            QMessageBox.warning(self, "Camera", err)
            return
        i = self.camera.info
        name = (i.get("model") or "camera").strip()
        backend = "Pylon" if self.camera.backend == "pylon" else "OpenCV"
        extra = ""
        if self.camera.backend == "opencv" and self.camera.last_error:
            extra = "   (Pylon unavailable: %s)" % self.camera.last_error
        self.statusBar().showMessage("Connected: %s via %s%s" % (name, backend, extra), 6000)

    def stop_camera(self):
        self.continuous = False
        self.runTimer.stop()
        self.pollTimer.stop()
        if self.camera:
            self.camera.stop()
        self.camera = None
        self.live = None
        self.aCam.setText("Start camera")
        self.aCam.setIcon(dot("#8899AA"))
        self.aCont.setText("Run continuous")
        self.aCont.setIcon(dot("#0E90D2"))

    def on_poll_camera(self):
        if not self.camera:
            return
        f = self.camera.get_frame()
        if f is not None:
            self.live = f
            if not self.continuous:
                self.canvas.set_image(f)

    def auto_expose(self):
        """Find the longest exposure that does not clip the selected region.

        Born from a real photograph: a switch faceplate shot bright enough to
        see the panel had every LED core clipped to white, and LED colour
        reported all thirteen as white. There is no way to recover that from the
        image, and no way to compute the right exposure from it either -- once a
        region clips, the file says nothing about how far past 255 it went. The
        only honest method is to change the exposure and look again."""
        if not self.camera or self.camera.backend != "pylon":
            QMessageBox.information(
                self, "Automatic exposure",
                "This needs a Basler camera. Exposure cannot be set on an "
                "ordinary webcam through OpenCV reliably enough to search.")
            return

        sel = self.canvas.selected_ids()
        roi = None
        if len(sel) == 1:
            t = next((x for x in self.spec["tools"] if x["id"] == sel[0]), None)
            roi = (t.get("params") or {}).get("roi") if t else None
        if roi is None:
            QMessageBox.information(
                self, "Automatic exposure",
                "Select the region you care about first -- usually the one over "
                "the LEDs. The search protects that region from clipping, and "
                "the right exposure for LEDs is far shorter than for a faceplate.")
            return

        cam = self.camera
        settings = self.cam_settings
        pending = {"gen": None}

        def set_exposure(ms):
            settings.auto_exposure = False
            settings.exposure_ms = ms
            pending["gen"] = cam.update_settings(settings.copy())

        def grab():
            # A fixed sleep here measured stale frames on a 5 fps camera: the
            # frames in flight still carry the previous exposure. Wait for
            # ones genuinely taken at the new setting.
            QApplication.processEvents()
            gen, pending["gen"] = pending["gen"], None
            budget = 3.0 + 4 * settings.exposure_ms / 1000.0
            f = cam.frame_after(gen, settle=3, timeout_s=budget)
            QApplication.processEvents()
            return f

        self.statusBar().showMessage("Searching for an exposure that does not "
                                     "clip the selected region...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = exposure_mod.find_exposure(grab, set_exposure, roi=roi)
        finally:
            QApplication.restoreOverrideCursor()

        if not result.get("ok"):
            QMessageBox.warning(self, "Automatic exposure", result.get("reason", ""))
            return
        self.cam_settings.exposure_ms = result["exposure_ms"]
        set_exposure(result["exposure_ms"])
        QMessageBox.information(
            self, "Automatic exposure",
            "Exposure set to %.3f ms.\n\n"
            "Clipping in the selected region: %.2f%%\nBrightest pixel: %s\n\n"
            "%s\n\nTook %d measurements. If you also need to inspect the "
            "faceplate, that wants a separate, longer exposure -- one image "
            "rarely serves both."
            % (result["exposure_ms"], result.get("clipped_pct", 0.0),
               result.get("peak"), result.get("reason", ""),
               len(result.get("history", []))))

    def show_camera_settings(self):
        """Reads the camera's real limits where it can, so the operator cannot
        dial in an exposure the sensor will silently clamp."""
        if not self.camera or self.camera.backend is None:
            QMessageBox.information(self, "Camera settings", "Start the camera first.")
            return
        if self.camera.backend != "pylon":
            QMessageBox.information(
                self, "Camera settings",
                "This is an OpenCV device. Exposure and gain control need a "
                "Basler camera with pypylon installed.")
            return

        s = self.cam_settings
        cam = self.camera._pylon_cam
        exp_lo_us, exp_hi_us = cam.exposure_range_us()
        gain_unit, gain_lo, gain_hi = cam.gain_unit()

        d = QDialog(self)
        d.setWindowTitle("Camera settings")
        d.setMinimumWidth(380)
        lay = QVBoxLayout(d)

        info = self.camera.info
        head = QLabel("%s %s" % (info.get("vendor", ""), info.get("model", "")))
        head.setStyleSheet("font-weight:600; padding-bottom:6px;")
        lay.addWidget(head)

        form = QFormLayout()
        auto_exp = QCheckBox("Let the camera choose")
        auto_exp.setChecked(s.auto_exposure)
        form.addRow("Auto exposure", auto_exp)

        exp = QDoubleSpinBox()
        exp.setDecimals(3)
        exp.setSingleStep(0.5)
        if exp_lo_us:
            exp.setRange(exp_lo_us / 1000.0, exp_hi_us / 1000.0)
            form.addRow("", QLabel("range %.3f - %.1f ms"
                                   % (exp_lo_us / 1000.0, exp_hi_us / 1000.0)))
        else:
            exp.setRange(0.01, 1000.0)
        exp.setValue(s.exposure_ms)
        form.addRow("Exposure (ms)", exp)

        auto_gain = QCheckBox("Let the camera choose")
        auto_gain.setChecked(s.auto_gain)
        form.addRow("Auto gain", auto_gain)

        gain = QDoubleSpinBox()
        # Label the control with the unit this camera actually uses. USB3 and
        # ace2 cameras take decibels; older GigE ace cameras take raw integer
        # sensor units, and calling those dB would be a lie of about two orders
        # of magnitude.
        gain.setDecimals(0 if gain_unit == "raw" else 2)
        gain.setSingleStep(1 if gain_unit == "raw" else 0.5)
        if gain_lo is not None:
            gain.setRange(float(gain_lo), float(gain_hi))
            form.addRow("", QLabel("range %g - %g" % (gain_lo, gain_hi)))
        else:
            gain.setRange(0.0, 48.0)
        gain.setValue(s.gain_db)
        form.addRow("Gain (%s)" % (gain_unit or "dB"), gain)
        if gain_unit == "raw":
            note_raw = QLabel("This camera reports gain in raw sensor units, "
                              "not decibels.")
            note_raw.setWordWrap(True)
            note_raw.setStyleSheet("color:#8C99A6;")
            form.addRow("", note_raw)

        auto_wb = QCheckBox("Continuous")
        auto_wb.setChecked(s.auto_white_balance)
        form.addRow("Auto white balance", auto_wb)
        lay.addLayout(form)

        note = QLabel("Changes apply live. For repeatable inspection, turn every "
                      "auto setting off once the lighting is fixed - auto exposure "
                      "drifting between parts is a common cause of intermittent "
                      "false rejects.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8C99A6; padding-top:8px;")
        lay.addWidget(note)

        def apply_live():
            s.auto_exposure = auto_exp.isChecked()
            s.exposure_ms = exp.value()
            s.auto_gain = auto_gain.isChecked()
            s.gain_db = gain.value()
            s.auto_white_balance = auto_wb.isChecked()
            exp.setEnabled(not s.auto_exposure)
            gain.setEnabled(not s.auto_gain)
            if self.camera:
                self.camera.update_settings(s.copy())

        for w in (auto_exp, auto_gain, auto_wb):
            w.toggled.connect(apply_live)
        exp.valueChanged.connect(apply_live)
        gain.valueChanged.connect(apply_live)
        exp.setEnabled(not s.auto_exposure)
        gain.setEnabled(not s.auto_gain)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(d.accept)
        bb.accepted.connect(d.accept)
        lay.addWidget(bb)
        d.exec()

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "",
                                              "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        img = imio.imread(path)
        if img is None:
            QMessageBox.warning(self, "Open image", "That file could not be read as an image.")
            return
        self.live = None
        self.last = None
        self.canvas.set_image(img, keep_view=False)
        self.canvas.clear_overlay()
        self.canvas.sync_regions(self.spec.get("tools", []) if self.spec else [])

    def set_reference(self):
        if self.canvas._img is None or not self.project:
            QMessageBox.information(self, "Set reference", "Show an image first.")
            return
        self.project.set_reference(self.canvas._img)
        self.statusBar().showMessage("Reference image saved for this model", 3000)

    # -------------------------------------------------------- labelling
    def set_workspace(self, mode):
        self.workspace = mode
        self.aInspect.setChecked(mode == "inspect")
        self.aLabel.setChecked(mode == "label")
        self.labelDock.setVisible(mode == "label")
        self.canvas.set_layer("labels" if mode == "label" else "tools")
        if mode == "label":
            self.refresh_classes()
            self.load_labels()
        else:
            self.canvas.sync_regions(self.spec.get("tools", []) if self.spec else [],
                                     self.last)

    def refresh_classes(self):
        if not self.ds:
            return
        row = self.classList.currentRow()
        self.classList.clear()
        for i, c in enumerate(self.ds.classes):
            it = QListWidgetItem("  " + c)
            pm = QPixmap(12, 12)
            pm.fill(QColor(self.ds.color(i)))
            it.setIcon(QIcon(pm))
            self.classList.addItem(it)
        if self.classList.count():
            self.classList.setCurrentRow(min(max(0, row), self.classList.count() - 1))
        self.canvas.class_colour = lambda i: QColor(self.ds.color(i))
        self.canvas.class_name = lambda i: self.ds.name(i)
        self.refresh_stats()

    def refresh_stats(self):
        if not self.ds:
            return
        st = self.ds.stats()
        per = "  ".join("%s:%d" % (k, v) for k, v in st["per_class"].items()) or "none yet"
        self.dsStats.setText("%d images labelled, %d boxes\n%s"
                             % (st["images_labelled"], st["total_boxes"], per))

    def on_class_changed(self, row):
        self.canvas.new_class = max(0, row)

    def add_class(self):
        if not self.ds:
            return
        name, ok = QInputDialog.getText(
            self, "New class", "What is this object called?   e.g. screw, led, scratch")
        if ok and name.strip():
            self.ds.add_class(name)
            self.refresh_classes()
            self.classList.setCurrentRow(len(self.ds.classes) - 1)

    def del_class(self):
        row = self.classList.currentRow()
        if row < 0 or not self.ds:
            return
        name = self.ds.classes[row]
        if QMessageBox.question(self, "Delete class",
                                "Delete '%s' and every box labelled with it?" % name
                                ) == QMessageBox.Yes:
            self.ds.remove_class(name)
            self.refresh_classes()
            self.load_labels()

    def load_labels(self):
        if not self.ds or not self.cur_image:
            self.canvas.sync_labels([])
            return
        self.canvas.sync_labels(self.ds.get(self.cur_image))

    def save_labels(self):
        if self.ds and self.cur_image:
            self.ds.set(self.cur_image, self.canvas.label_boxes())
            self.refresh_stats()

    def on_label_added(self, cls, roi):
        if not self.ds or not self.ds.classes:
            QMessageBox.information(self, "No class yet",
                                    "Create a class first, then draw boxes for it.")
            return
        if not self.cur_image:
            QMessageBox.information(self, "Pick a test image",
                                    "Labels are stored per image. Add test images and "
                                    "click one before labelling.")
            return
        boxes = self.canvas.label_boxes()
        boxes.append({"cls": cls, "x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]})
        self.ds.set(self.cur_image, boxes)
        self.canvas.sync_labels(boxes)
        self.refresh_stats()

    def check_dataset(self):
        if not self.ds:
            return
        st = self.ds.stats()
        warn = self.ds.warnings()
        per = "\n".join("    %-20s %d boxes" % (k, v) for k, v in st["per_class"].items())
        body = ("%d images labelled\n%d boxes total\n\n%s"
                % (st["images_labelled"], st["total_boxes"], per))
        if warn:
            body += "\n\nBefore you train:\n\n" + "\n\n".join("  - " + w for w in warn)
        else:
            body += "\n\nThis dataset looks ready to train."
        QMessageBox.information(self, "Dataset", body)

    def export_dataset(self):
        if not self.ds:
            return
        warn = self.ds.warnings()
        if warn:
            msg = "\n\n".join("  - " + w for w in warn)
            if QMessageBox.question(self, "Export anyway?",
                                    "This dataset has issues:\n\n%s\n\nExport it anyway?"
                                    % msg) != QMessageBox.Yes:
                return
        out = QFileDialog.getExistingDirectory(self, "Where should the training set go?")
        if not out:
            return
        out = os.path.join(out, self.project.id + "_dataset")
        try:
            info = self.ds.export(out)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QMessageBox.information(
            self, "Exported",
            "%d training and %d validation images, %d classes.\n\nWritten to:\n%s\n\n"
            "Copy that folder to a machine with an NVIDIA GPU and run:\n\n"
            "    pip install ultralytics\n    python train.py\n\n"
            "It prints the path to best.onnx when it finishes. Bring that file back "
            "here, put it in the model's models folder, and point a 'Detect with "
            "YOLO' tool at it."
            % (info["train"], info["val"], info["classes"], info["path"]))

    def show_training_help(self):
        QMessageBox.information(self, "Training a model", TRAINING_HELP)

    # --------------------------------------------------------- history
    def show_history(self):
        if not self.results:
            return
        self.results.flush(1.5)

        d = QDialog(self)
        d.setWindowTitle("Production history - %s" % self.projBox.currentText())
        d.resize(880, 620)
        lay = QVBoxLayout(d)

        top = QHBoxLayout()
        top.addWidget(QLabel("Period"))
        period = QComboBox()
        for lab, hrs in (("Last hour", 1), ("Last 8 hours", 8), ("Last 24 hours", 24),
                         ("Last 7 days", 168), ("Everything", None)):
            period.addItem(lab, hrs)
        period.setCurrentIndex(2)
        top.addWidget(period)
        top.addStretch(1)
        headline = QLabel("")
        headline.setStyleSheet("font-weight:600; font-size:14px;")
        top.addWidget(headline)
        lay.addLayout(top)

        tabs = QTabWidget()
        lay.addWidget(tabs, 1)

        runs = QTableWidget(0, 5)
        runs.setHorizontalHeaderLabels(["Time", "Verdict", "Cycle", "Failed checks",
                                        "Image"])
        runs.verticalHeader().hide()
        runs.setEditTriggers(QAbstractItemView.NoEditTriggers)
        runs.setSelectionBehavior(QAbstractItemView.SelectRows)
        for c, wd in enumerate((160, 80, 80, 300)):
            runs.setColumnWidth(c, wd)
        runs.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(runs, "Runs")

        pareto = QTableWidget(0, 3)
        pareto.setHorizontalHeaderLabels(["Check", "Times failed",
                                          "Appeared in this % of rejects"])
        pareto.verticalHeader().hide()
        pareto.setEditTriggers(QAbstractItemView.NoEditTriggers)
        pareto.setColumnWidth(0, 240)
        pareto.setColumnWidth(1, 130)
        pareto.horizontalHeader().setStretchLastSection(True)
        pw = QWidget(); pv = QVBoxLayout(pw)
        pv.addWidget(pareto, 1)
        pnote = QLabel("")
        pnote.setWordWrap(True)
        pnote.setStyleSheet("color:#8C99A6; padding-top:6px;")
        pv.addWidget(pnote)
        tabs.addTab(pw, "What is rejecting parts")

        hourly = QTableWidget(0, 4)
        hourly.setHorizontalHeaderLabels(["Hour", "Inspected", "Passed", "Yield"])
        hourly.verticalHeader().hide()
        hourly.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hourly.setColumnWidth(0, 180)
        hourly.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(hourly, "Yield by hour")

        def refresh():
            hrs = period.currentData()
            sm = self.results.summary(hrs)
            headline.setText("%d inspected    %.1f%% yield    %.0f ms average"
                             % (sm["total"], sm["yield_pct"], sm["avg_ms"]))
            rows = self.results.recent(400)
            if hrs:
                import time as _t
                cut = _t.time() - hrs * 3600
                keep = []
                for r in rows:
                    try:
                        from datetime import datetime as _d
                        if _d.fromisoformat(r["iso"]).timestamp() >= cut:
                            keep.append(r)
                    except ValueError:
                        keep.append(r)
                rows = keep
            runs.setRowCount(len(rows))
            for i, r in enumerate(rows):
                failed = ""
                if r["verdict"] != "PASS":
                    ft = self.results.failed_tools_for(r["id"])
                    failed = ", ".join(x["tool_id"] for x in ft)
                cells = [r["iso"].replace("T", "  "), r["verdict"],
                         "%.0f ms" % (r["total_ms"] or 0), failed,
                         os.path.basename(r["image_path"]) if r["image_path"] else ""]
                for c, txt in enumerate(cells):
                    it = QTableWidgetItem(txt)
                    if c in (0, 2):
                        it.setFont(QFont("Consolas", 9))
                    if c == 1:
                        it.setForeground(PASS if txt == "PASS" else FAIL)
                    runs.setItem(i, c, it)

            pr = self.results.pareto(hrs)
            pareto.setRowCount(len(pr))
            for i, r in enumerate(pr):
                for c, txt in enumerate([r["tool_id"], str(r["count"]),
                                         "%.1f%%" % r["pct_of_rejects"]]):
                    it = QTableWidgetItem(txt)
                    if c:
                        it.setFont(QFont("Consolas", 9))
                    pareto.setItem(i, c, it)
            if pr:
                pnote.setText("%s appeared in %.0f%% of rejected parts, more than any "
                              "other check. Percentages can exceed 100%% in total "
                              "because one part can fail several checks at once. "
                              "Either the parts really are failing there, or that "
                              "check is too tight."
                              % (pr[0]["tool_id"], pr[0]["pct_of_rejects"]))
            else:
                pnote.setText("No rejects recorded in this period.")

            hy = self.results.hourly_yield(hrs or 24 * 30)
            hourly.setRowCount(len(hy))
            for i, r in enumerate(hy):
                for c, txt in enumerate([r["hour"], str(r["total"]), str(r["passed"]),
                                         "%.1f%%" % r["yield_pct"]]):
                    it = QTableWidgetItem(txt)
                    if c:
                        it.setFont(QFont("Consolas", 9))
                    if c == 3:
                        it.setForeground(PASS if r["yield_pct"] >= 95 else
                                         WARN if r["yield_pct"] >= 80 else FAIL)
                    hourly.setItem(i, c, it)

        period.currentIndexChanged.connect(refresh)
        refresh()

        btns = QHBoxLayout()
        b_csv = QPushButton("Export CSV")
        def do_export():
            path, _ = QFileDialog.getSaveFileName(d, "Export results",
                                                  "results.csv", "CSV (*.csv)")
            if path:
                n = self.results.export_csv(path, period.currentData())
                QMessageBox.information(d, "Exported", "%d rows written." % n)
        b_csv.clicked.connect(do_export)
        btns.addWidget(b_csv)

        b_purge = QPushButton("Delete older than 30 days")
        def do_purge():
            if QMessageBox.question(d, "Purge", "Permanently delete records older "
                                    "than 30 days?") == QMessageBox.Yes:
                n = self.results.purge_older_than(30)
                QMessageBox.information(d, "Purged", "%d runs removed." % n)
                refresh()
        b_purge.clicked.connect(do_purge)
        btns.addWidget(b_purge)
        btns.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(d.accept); bb.accepted.connect(d.accept)
        btns.addWidget(bb)
        lay.addLayout(btns)
        d.exec()

    def set_archive_mode(self):
        opts = {"Never save images": "none",
                "Save images of failures only": "fail",
                "Save every image": "all"}
        cur = [k for k, v in opts.items() if v == self.archive_mode][0]
        choice, ok = QInputDialog.getItem(self, "Image archiving",
                                          "Archiving fills disks quickly. Failures "
                                          "only is the usual choice.",
                                          list(opts), list(opts).index(cur), False)
        if ok:
            self.archive_mode = opts[choice]
            if self.results:
                self.results.archive = self.archive_mode

    # ------------------------------------------------------ diagnostics
    def show_diagnostics(self):
        """Answers the two questions that decide whether a recipe survives a
        real shift: is the image good enough, and how close is each check to
        its own threshold."""
        img = self.live if self.live is not None else self.canvas._img
        if img is None:
            QMessageBox.information(self, "Diagnostics",
                                    "Show an image or start the camera first.")
            return

        d = QDialog(self)
        d.setWindowTitle("Diagnostics")
        d.resize(720, 560)
        lay = QVBoxLayout(d)
        tabs = QTabWidget()
        lay.addWidget(tabs, 1)

        # ---- image quality ------------------------------------------------
        q = diagnostics.image_quality(img)
        w1 = QWidget(); v1 = QVBoxLayout(w1)
        colour = {"good": "#18C25A", "warn": "#FFA000",
                  "bad": "#FF3B30", "error": "#FF3B30"}
        for kind, msg in q["findings"]:
            lb = QLabel(("\u25cf  " if kind != "good" else "\u2713  ") + msg)
            lb.setWordWrap(True)
            lb.setStyleSheet("color:%s; padding:5px 2px;" % colour.get(kind, "#DDE2E8"))
            v1.addWidget(lb)
        tbl = QTableWidget(len(q["metrics"]), 2)
        tbl.setHorizontalHeaderLabels(["Measurement", "Value"])
        tbl.verticalHeader().hide()
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.setColumnWidth(0, 240)
        for i, (k, val) in enumerate(q["metrics"].items()):
            tbl.setItem(i, 0, QTableWidgetItem(k.replace("_", " ")))
            it = QTableWidgetItem(str(val)); it.setFont(QFont("Consolas", 9))
            tbl.setItem(i, 1, it)
        v1.addWidget(tbl, 1)
        # Clipping over the selected region specifically. Whole-image clipping
        # can look negligible while the only part that matters is destroyed:
        # on a real faceplate photo the frame was 0.02% clipped overall and
        # 10.6% clipped across the LED band.
        sel = self.canvas.selected_ids()
        roi = None
        if len(sel) == 1:
            t = next((x for x in self.spec["tools"] if x["id"] == sel[0]), None)
            roi = (t.get("params") or {}).get("roi") if t else None
        if roi:
            adv = exposure_mod.advise(img, roi, purpose="colour")
            head = QLabel("Selected region '%s':" % sel[0])
            head.setStyleSheet("font-weight:600; padding-top:8px;")
            v1.addWidget(head)
            for kind, msg in adv["findings"]:
                lb2 = QLabel(("\u25cf  " if kind != "good" else "\u2713  ") + msg)
                lb2.setWordWrap(True)
                lb2.setStyleSheet("color:%s; padding:3px 2px;"
                                  % colour.get(kind, "#DDE2E8"))
                v1.addWidget(lb2)
            if not adv["ok"]:
                hint = QLabel("Run -> Find exposure for selected region (F7) will "
                              "search for an exposure that does not clip it.")
                hint.setWordWrap(True)
                hint.setStyleSheet("color:#8C99A6; padding:2px 2px 6px 2px;")
                v1.addWidget(hint)

        note = QLabel("Focus is contrast-normalised, so a merely dark image is "
                      "not misreported as out of focus. Below about 0.02 is "
                      "genuinely blurred.")
        note.setWordWrap(True); note.setStyleSheet("color:#8C99A6; padding-top:6px;")
        v1.addWidget(note)
        tabs.addTab(w1, "Image quality")

        # ---- margins ------------------------------------------------------
        w2 = QWidget(); v2 = QVBoxLayout(w2)
        if not self.last:
            v2.addWidget(QLabel("Run an inspection first (F5), then reopen this."))
        else:
            rows = diagnostics.margin_report(self.last, self.spec)
            if not rows:
                v2.addWidget(QLabel("No tools in this recipe have a comparable "
                                    "accept threshold."))
            else:
                t2 = QTableWidget(len(rows), 5)
                t2.setHorizontalHeaderLabels(["Tool", "Score", "Threshold",
                                              "Margin", "Verdict"])
                t2.verticalHeader().hide()
                t2.horizontalHeader().setStretchLastSection(True)
                for i, r in enumerate(rows):
                    cells = [r["tool_id"], "%.3f" % r["score"], "%.3f" % r["threshold"],
                             "%+.3f  (%+.0f%%)" % (r["margin"], 100 * r["relative"]),
                             r["verdict"].upper()]
                    for c, txt in enumerate(cells):
                        it = QTableWidgetItem(txt)
                        if c:
                            it.setFont(QFont("Consolas", 9))
                        if c == 4:
                            it.setForeground(QColor({"healthy": "#18C25A",
                                                     "fragile": "#FFA000",
                                                     "failing": "#FF3B30"}[r["verdict"]]))
                        t2.setItem(i, c, it)
                for c, wd in enumerate((160, 80, 90, 150)):
                    t2.setColumnWidth(c, wd)
                v2.addWidget(t2, 1)
                thin = [r["tool_id"] for r in rows if r["verdict"] == "fragile"]
                msg = ("Every check has comfortable margin."
                       if not thin else
                       "Fragile: %s.\n\nThese pass now but sit close to their "
                       "threshold, so normal part-to-part variation will push "
                       "them over eventually - usually at the worst time. Widen "
                       "the threshold, or fix what is making the score low."
                       % ", ".join(thin))
                lb = QLabel(msg); lb.setWordWrap(True)
                lb.setStyleSheet("color:%s; padding-top:8px;"
                                 % ("#8C99A6" if not thin else "#FFA000"))
                v2.addWidget(lb)
        tabs.addTab(w2, "Recipe margins")

        # ---- timing -------------------------------------------------------
        w3 = QWidget(); v3 = QVBoxLayout(w3)
        if not self.last:
            v3.addWidget(QLabel("Run an inspection first (F5), then reopen this."))
        else:
            tr = diagnostics.timing_report(self.last)
            head = QLabel("Cycle time %.1f ms" % tr["total_ms"])
            head.setStyleSheet("font-weight:600; font-size:14px; padding-bottom:6px;")
            v3.addWidget(head)
            t3 = QTableWidget(len(tr["rows"]), 3)
            t3.setHorizontalHeaderLabels(["Tool", "Time", "Share"])
            t3.verticalHeader().hide()
            t3.horizontalHeader().setStretchLastSection(True)
            t3.setColumnWidth(0, 200)
            for i, r in enumerate(tr["rows"]):
                for c, txt in enumerate([r["tool_id"], "%.2f ms" % r["ms"],
                                         "%.1f%%" % r["pct"]]):
                    it = QTableWidgetItem(txt)
                    if c:
                        it.setFont(QFont("Consolas", 9))
                    if c == 2 and r["pct"] > 40:
                        it.setForeground(QColor("#FFA000"))
                    t3.setItem(i, c, it)
            v3.addWidget(t3, 1)
            top = tr["rows"][0] if tr["rows"] else None
            if top and top["pct"] > 40:
                lb = QLabel("%s alone is %.0f%% of the cycle. If you need to go "
                            "faster, that is the only thing worth optimising - "
                            "shrink its region, or lower an upscale factor."
                            % (top["tool_id"], top["pct"]))
                lb.setWordWrap(True)
                lb.setStyleSheet("color:#FFA000; padding-top:8px;")
                v3.addWidget(lb)
        tabs.addTab(w3, "Cycle time")

        auto_btn = QPushButton("Find exposure for the selected region")
        auto_btn.setToolTip("Search for the longest exposure that does not clip "
                            "the selected region")
        auto_btn.clicked.connect(lambda: (d.accept(), self.auto_expose()))
        lay.addWidget(auto_btn)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(d.accept); bb.accepted.connect(d.accept)
        lay.addWidget(bb)
        d.exec()

    def run_stability(self):
        """Run the recipe over every test image and report which checks wander.
        Point it at images of parts you consider GOOD."""
        if not self.project or self.imgList.count() == 0:
            QMessageBox.information(self, "Stability check",
                                    "Add some test images to this model first.")
            return
        results = []
        for i in range(self.imgList.count()):
            im = imio.imread(self.imgList.item(i).data(Qt.UserRole))
            if im is None:
                continue
            r = self.inspect(im)
            if r:
                results.append(r)
        rows = diagnostics.stability_report(results)
        if not rows:
            QMessageBox.information(self, "Stability check",
                                    "Need at least two readable images.")
            return
        body = "\n".join("  %-18s mean %.3f   spread %.3f" %
                          (r["tool_id"], r["mean"], r["spread"]) for r in rows)
        worst = rows[0]
        tail = ("\n\nWidest spread is %s at %.3f. If those images are all good "
                "parts, that check is the least stable in the recipe and the "
                "first place a false reject will come from."
                % (worst["tool_id"], worst["spread"]))
        QMessageBox.information(self, "Stability across %d images" % len(results),
                                "Score spread per tool, widest first:\n\n"
                                + body + tail)

    # ------------------------------------------------------------- run
    def toggle_continuous(self):
        self.continuous = not self.continuous
        from vision_station.ui import icons
        if self.continuous:
            self.aCont.setText("Stop")
            self.aCont.setIcon(icons.icon("stop"))
            self.runTimer.start(0)
        else:
            self.aCont.setText("Run continuous")
            self.aCont.setIcon(icons.icon("run_continuous", ink="#DCE3EA"))
            self.runTimer.stop()
        self._sync_run_buttons()

    def inspect(self, payload):
        """Synchronous inspection. Kept for batch runs, the stability check and
        the tests, where blocking is fine and simpler."""
        from vision_station.tools import make_tool
        try:
            self.project.spec = self.spec
            self.project.tools = []
            for s in self.spec.get("tools", []):
                sp = json.loads(json.dumps(s))
                p = sp.setdefault("params", {})
                if p.get("model") and not os.path.isabs(p["model"]):
                    p["model"] = os.path.join(self.project.folder, p["model"])
                self.project.tools.append(make_tool(sp))
            return self.project.inspect(payload, source="desktop")
        except Exception as e:
            self.continuous = False
            self.runTimer.stop()
            QMessageBox.critical(self, "Inspection failed", f"{type(e).__name__}: {e}")
            return None

    def _start_worker(self, payload):
        """Inspect without blocking the interface. Returns False if one is
        already running, so a slow recipe cannot stack up workers."""
        if self._worker is not None and self._worker.isRunning():
            return False
        self._worker = InspectionWorker(self.project, self.spec, payload, self)
        self._worker.finished_ok.connect(self._on_worker_result)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()
        return True

    def _on_worker_result(self, res):
        self._worker = None
        if res is None:
            return
        self.last = res
        if self.results:
            self.results.log(res, getattr(self, "_pending_image", None))
        if self.live is not None:
            self.canvas.set_image(self.live)
        self.show_result(res)
        if self.continuous:
            self.runTimer.start(0)

    def wait_for_inspection(self, timeout_ms=15000):
        """Block until the running inspection finishes, pumping the event loop.

        Inspection is asynchronous so the interface stays responsive, which
        means anything that needs the result immediately -- a test, a batch
        run, a script -- has to wait for it explicitly rather than assuming
        run_once() has already produced one."""
        import time as _t
        deadline = _t.time() + timeout_ms / 1000.0
        while self._worker is not None and _t.time() < deadline:
            QApplication.processEvents()
            _t.sleep(0.005)
        QApplication.processEvents()
        return self.last

    def _on_worker_failed(self, message):
        self._worker = None
        self.continuous = False
        self.runTimer.stop()
        self.aCont.setText("Run continuous")
        self.aCont.setIcon(dot("#0E90D2"))
        QMessageBox.critical(self, "Inspection failed", message)

    def _sync_run_buttons(self):
        if hasattr(self, "rmStart"):
            self.rmStart.setText("Stop" if self.continuous else "Start")

    def _continuous_tick(self):
        """Start the next cycle. The one after it is queued when this finishes,
        never on a fixed interval -- a repeating timer against a cycle longer
        than its period queues work faster than it completes."""
        if not self.continuous:
            return
        self.run_once()

    def run_once(self):
        if not self.project:
            return
        caps = self.project.captures()
        multi = len(caps) > 1

        if multi and self.camera and self.camera.backend == "pylon":
            frames = self.camera.grab_sequence(caps)
            if not frames:
                QMessageBox.warning(self, "Capture", "No frames came back from "
                                                     "the camera.")
                return
            # `frames.get(...) or ...` raises: the truthiness of a numpy array
            # is ambiguous. This is the third time that pattern has bitten in
            # this codebase, so it is spelled out rather than shortened.
            img = frames.get(caps[0]["id"])
            if img is None:
                img = next(iter(frames.values()))
            payload = frames
        else:
            img = self.live if self.live is not None else self.canvas._img
            if img is None:
                QMessageBox.information(self, "Nothing to inspect",
                                        "Start the camera or load an image first.")
                return
            payload = img

        self._pending_image = img
        if not self._start_worker(payload):
            return      # one already in flight; its completion queues the next

    def show_result(self, res):
        v = res["verdict"]
        self.counts[v] = self.counts.get(v, 0) + 1
        tot = sum(self.counts.values())
        col = {"PASS": "#18C25A", "FAIL": "#FF3B30", "WARN": "#FFA000"}[v]
        self.verdict.setText(v)
        self.verdict.setStyleSheet(f"background:{col}; color:#101418; font-weight:700;"
                                   "font-size:13px; padding:3px 12px; border-radius:3px;")
        self.timeLabel.setText(f"  {res['total_ms']:.1f} ms  ")
        npass = self.counts.get("PASS", 0)
        self.yieldLabel.setText(f"  Pass {npass}/{tot} ({100*npass/tot:.1f}%)  ")

        self.grid.setRowCount(len(res["tools"]))
        for r, t in enumerate(res["tools"]):
            c2 = {"pass": PASS, "fail": FAIL}.get(t["status"], WARN)
            vals = ", ".join(f"{k}={v}" for k, v in t["values"].items()
                             if not isinstance(v, (list, dict)))
            cells = [t["tool_id"], LABELS.get(t["type"], (t["type"],))[0], t["status"].upper(),
                     f"{t['score']:.3f}", f"{t['ms']:.1f}", t["message"] or vals]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(str(text))
                if c in (2, 3, 4):
                    it.setFont(QFont("Consolas", 9))
                if c == 2:
                    it.setForeground(c2)
                self.grid.setItem(r, c, it)

        miss = res.get("captures_missing") or []
        if miss and not getattr(self, "_warned_captures", False):
            self._warned_captures = True
            self.statusBar().showMessage(
                "Tools asked for capture(s) %s which were not taken - they used "
                "the main frame instead, at the wrong exposure."
                % ", ".join(miss), 12000)
        self.update_banner(res)
        self._update_recent(res["verdict"])
        self.refresh_editors()
        self.canvas.draw_results(res["tools"])
        self.canvas.sync_regions(self.spec["tools"], res)
        self.refresh_tree()
        self._load_props(self.canvas.selected_ids())

    def on_grid_select(self):
        rows = self.grid.selectionModel().selectedRows()
        if not rows or not self.last:
            return
        tid = self.last["tools"][rows[0].row()]["tool_id"]
        self.canvas.select_ids([tid])

    def reset_counts(self):
        self.counts = {"PASS": 0, "FAIL": 0, "WARN": 0}
        self.yieldLabel.setText("")
        self.update_banner()
        self._paint_verdict("READY", "#3E4652", "#9AA9B8")
        self.bannerDetail.setText("Counters reset")

    def closeEvent(self, e):
        if self.dirty:
            r = QMessageBox.question(self, "Unsaved changes", "Save the job before closing?",
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if r == QMessageBox.Cancel:
                e.ignore()
                return
            if r == QMessageBox.Save:
                self.save_job()
        if self.camera:
            self.pollTimer.stop()
            self.camera.stop()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)
        if self.results:
            self.results.flush(2.0)
            self.results.close()
        e.accept()


def main():
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    app = QApplication(sys.argv)
    # The same Fluent theme as QuickBuild, in its dark mode: this window's
    # panels carry dark colours of their own. STYLE is kept for reference.
    from vision_station.ui import theme
    theme.apply(app, "dark")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
