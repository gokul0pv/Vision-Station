"""Find parts and text: the panel beside a job editor that finds what is in an
area of the reference picture, shows it as boxes on the picture, and turns
what the user keeps into checks -- each one a separate check.

    Find          Parts             -> a "Part present" check per kept part
                  Text  (verify)    -> a "Verify text" (OCV) check per kept
                                       line of text, expecting exactly that text
                  Text  (read)      -> a "Read text" (OCR) check per kept line

    1. Area       the whole picture, or a box dragged round part of it; the
                  area's box can be moved and resized on the picture
    2. Find       parts: a neural network on the graphics card; text: the
                  application's own text reader (also on the graphics card)
    3. Review     every box can be moved and resized on the picture; a click
                  keeps or drops it. Parts are coloured by type (untick a type
                  to drop all of its kind, name it "inductor" and its checks
                  are called that; Sensitivity re-filters at once; "Find all
                  like this" finds every copy of one part). Text: each line is
                  a row -- keep it or not, correct the text it must read, name
                  the check; resizing a text box reads it again.
    4. Add        one check per kept box, in one step (Ctrl+Z takes them all
                  back). The panel stays open, to find parts and then text,
                  or another area, without starting over.
"""
import math
import threading
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QGraphicsItem,
                               QGraphicsRectItem, QGraphicsSimpleTextItem, QHBoxLayout,
                               QHeaderView, QLabel, QProgressBar, QPushButton, QSlider,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from vision_station.core import part_finder as pf

PALETTE = ["#00B3FF", "#FF9F1C", "#2EC4B6", "#E71D73", "#9B5DE5", "#F15BB5", "#00F5D4",
           "#FEE440", "#8AC926", "#FF595E", "#6A4C93", "#1982C4", "#FFCA3A", "#C77DFF",
           "#06D6A0", "#EF476F", "#118AB2", "#FFD166", "#73D2DE", "#D81159"]
TEXT_COLOUR = "#1F6FFF"
DROPPED = QColor(150, 150, 150)

MODES = [("parts", "Parts  →  Part present checks"),
         ("ocv", "Text  →  Verify text checks (OCV)"),
         ("ocr", "Text  →  Read text checks (OCR)")]


def type_colour(t):
    return QColor(PALETTE[t % len(PALETTE)])


class EditableBox(QGraphicsRectItem):
    """A box on the picture that can be moved (drag inside), resized (drag an
    edge or a corner), turned (drag just outside a corner; Shift: 15 degree
    steps) and clicked. on_click() for a click without a drag,
    on_change(rect, angle) once a move, resize or turn is finished. rect is
    the box before turning, [x, y, w, h]; angle in degrees about its centre,
    as every check's region is stored.

    The search area and the boxes found in it could only be moved and
    resized: a part sitting at an angle got a loose upright box, with board
    and neighbours in it."""

    EDGE = 7.0           # screen pixels round the edge that resize
    ROT = 16.0           # screen pixels beyond a corner that turn the box

    def __init__(self, rect, on_click=None, on_change=None, z=45, angle=0.0, ellipse=False,
                 handles=False):
        super().__init__(QRectF(*rect))
        self.on_click, self.on_change = on_click, on_change
        self.angle = float(angle or 0.0) % 360.0
        self.ellipse = bool(ellipse)          # drawn as the ellipse in its box
        self.show_handles = bool(handles)     # round grips at corners and sides
        self.setZValue(z)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self._drag = None
        self._label = None
        self._sync()

    def _sync(self):
        self.setTransformOriginPoint(self.rect().center())
        self.setRotation(self.angle)

    def _k(self):
        views = self.scene().views() if self.scene() is not None else []
        return max(1e-6, views[0].transform().m11()) if views else 1.0

    def _corners(self, r):
        return {"tl": r.topLeft(), "tr": r.topRight(), "br": r.bottomRight(),
                "bl": r.bottomLeft()}

    def _zone(self, p):
        r, e, rz = self.rect(), self.EDGE / self._k(), self.ROT / self._k()
        inside = r.adjusted(-e, -e, e, e).contains(p)
        if not inside:
            for c in self._corners(r).values():
                if abs(p.x() - c.x()) <= rz and abs(p.y() - c.y()) <= rz:
                    return "rot"
            return None
        z = ""
        if abs(p.y() - r.top()) <= e:
            z += "t"
        elif abs(p.y() - r.bottom()) <= e:
            z += "b"
        if abs(p.x() - r.left()) <= e:
            z += "l"
        elif abs(p.x() - r.right()) <= e:
            z += "r"
        return z or "move"

    def boundingRect(self):
        rz = self.ROT / self._k() + 2.0
        return self.rect().adjusted(-rz, -rz, rz, rz)

    def shape(self):
        """The box, its edges, and the turning zones beyond its corners: what
        a click lands on. Not the whole bounding square, or a click beside a
        box's side would be taken from the box underneath."""
        from PySide6.QtGui import QPainterPath
        r, e, rz = self.rect(), self.EDGE / self._k(), self.ROT / self._k()
        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        path.addRect(r.adjusted(-e, -e, e, e))
        for c in self._corners(r).values():
            path.addRect(QRectF(c.x() - rz, c.y() - rz, 2 * rz, 2 * rz))
        return path

    def paint(self, p, opt, widget=None):
        from PySide6.QtGui import QPainter
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(self.pen())
        p.setBrush(self.brush())
        r = self.rect()
        if self.ellipse:
            p.drawEllipse(r)
        else:
            p.drawRect(r)
        if self.show_handles:
            k = self._k()
            rad = 4.5 / k
            col = self.pen().color()
            p.setPen(QPen(col, 1.6 / k))
            p.setBrush(QBrush(QColor("#FFFFFF")))
            for pt in (r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft(),
                       QPointF(r.center().x(), r.top()), QPointF(r.center().x(), r.bottom()),
                       QPointF(r.left(), r.center().y()), QPointF(r.right(), r.center().y())):
                p.drawEllipse(pt, rad, rad)

    _DIR = {"r": 0, "br": 45, "b": 90, "bl": 135, "l": 180, "tl": 225, "t": 270, "tr": 315}

    def _cursor(self, zone):
        from vision_station.ui.canvas import _resize_cursor, rotate_cursor
        if zone == "rot":
            return rotate_cursor()
        if zone == "move" or zone is None:
            return Qt.SizeAllCursor
        return _resize_cursor(self._DIR[zone] + self.angle)

    def hoverMoveEvent(self, e):
        z = self._zone(e.pos())
        if z is None:
            self.unsetCursor()
        else:
            self.setCursor(self._cursor(z))

    def mousePressEvent(self, e):
        zone = self._zone(e.pos())
        if zone is None:
            e.ignore()
            return
        self._drag = {"start": e.scenePos(), "rect": QRectF(self.rect()), "zone": zone,
                      "angle": self.angle, "moved": False,
                      "centre": self.mapToScene(self.rect().center())}
        e.accept()

    @staticmethod
    def _turn(v, deg):
        a = math.radians(deg)
        ca, sa = math.cos(a), math.sin(a)
        return QPointF(v.x() * ca - v.y() * sa, v.x() * sa + v.y() * ca)

    def mouseMoveEvent(self, e):
        d0 = self._drag
        if d0 is None:
            return
        d = e.scenePos() - d0["start"]
        if not d0["moved"] and (abs(d.x()) + abs(d.y())) * self._k() < 3:
            return                                    # still a click
        d0["moved"] = True
        r0, zone = d0["rect"], d0["zone"]
        self.prepareGeometryChange()
        if zone == "rot":
            c = d0["centre"]
            a0 = math.degrees(math.atan2(d0["start"].y() - c.y(), d0["start"].x() - c.x()))
            a1 = math.degrees(math.atan2(e.scenePos().y() - c.y(), e.scenePos().x() - c.x()))
            deg = d0["angle"] + (a1 - a0)
            from vision_station.ui.canvas import QApplication_mods
            if (e.modifiers() | QApplication_mods()) & Qt.ShiftModifier:
                deg = round(deg / 15.0) * 15.0
            self.angle = deg % 360.0
            self._sync()
            self._place_label()
            return
        if zone == "move":
            r = QRectF(r0)
            r.translate(d)
        else:
            # the pointer's movement along the box's own sides
            u = self._turn(d, -d0["angle"])
            r = QRectF(r0)
            if "l" in zone:
                r.setLeft(min(r0.left() + u.x(), r0.right() - 4))
            if "r" in zone:
                r.setRight(max(r0.right() + u.x(), r0.left() + 4))
            if "t" in zone:
                r.setTop(min(r0.top() + u.y(), r0.bottom() - 4))
            if "b" in zone:
                r.setBottom(max(r0.bottom() + u.y(), r0.top() + 4))
            # the opposite side stays where it was on the picture: the new
            # box's centre, seen through the box's turn as it was
            c0 = r0.center()
            off = self._turn(r.center() - c0, d0["angle"])
            c = d0["centre"] + off
            r = QRectF(c.x() - r.width() / 2.0, c.y() - r.height() / 2.0, r.width(), r.height())
        self.setRect(r)
        self._sync()
        self._place_label()

    def mouseReleaseEvent(self, e):
        drag, self._drag = self._drag, None
        if drag is None:
            return
        if not drag["moved"]:
            if self.on_click:
                self.on_click()
        elif self.on_change:
            r = self.rect()
            self.on_change([int(round(r.x())), int(round(r.y())),
                            max(1, int(round(r.width()))), max(1, int(round(r.height())))],
                           round(self.signed_angle(), 2))

    def signed_angle(self):
        a = self.angle % 360.0
        return a - 360.0 if a > 180.0 else a

    # a small caption above the box (the text read), the same size at any zoom
    def set_label(self, text, colour):
        if not text:
            if self._label is not None:
                self._label.setVisible(False)
            return
        if self._label is None:
            self._label = QGraphicsSimpleTextItem(self)
            f = QFont()
            f.setPointSizeF(8.5)
            f.setWeight(QFont.DemiBold)
            self._label.setFont(f)
            self._label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            self._label.setAcceptedMouseButtons(Qt.NoButton)
        self._label.setVisible(True)
        self._label.setText(text)
        self._label.setBrush(QBrush(colour))
        self._place_label()

    def _place_label(self):
        if self._label is not None:
            r = self.rect()
            self._label.setPos(QPointF(r.left(), r.top() - 18 / self._k()))


class PartFinderPanel(QDialog):
    """editor: the JobEditor (its display's canvas, add/undo plumbing)."""

    def __init__(self, editor):
        super().__init__(editor, Qt.Tool)
        self.ed = editor
        self.setWindowTitle("Find parts and text")
        self.setObjectName("partFinder")
        self.resize(460, 680)
        self.image = editor.project().reference
        self.canvas = editor.display.canvas
        self.mode = "parts"
        self.area = None                    # [x, y, w, h] or None = whole picture
        self.area_angle = 0.0               # the area's turn, degrees about its centre
        self.raw = None                     # network candidates for the area (parts)
        self.parts = []                     # [{"box", "score", "type", "text"?}]
        self.dropped = set()                # indices the user dropped
        self.off_types = set()
        self.names = {}                     # parts: type -> name; text: index -> name
        self.items = []
        self._area_item = None
        self._job = None

        intro = QLabel("Find what is in an area of the picture, adjust the boxes, and add "
                       "a separate check for each one you keep.")
        intro.setWordWrap(True)
        intro.setObjectName("hint")

        self.modeBox = QComboBox()
        for key, text in MODES:
            self.modeBox.addItem(text, key)
        self.modeBox.setToolTip("What to find in the area, and which check each kept box "
                                "becomes")

        self.areaLabel = QLabel("Area: the whole picture")
        self.drawArea = QPushButton("Draw area...")
        self.drawArea.setToolTip("Drag a box round the part of the picture to search - it "
                                 "can then be moved and resized on the picture")
        self.wholeBtn = QPushButton("Whole picture")
        self.findBtn = QPushButton("Find parts")
        self.findBtn.setObjectName("primaryButton")
        self.findBtn.setDefault(True)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        self.sens = QSlider(Qt.Horizontal)
        self.sens.setRange(0, 100)
        self.sens.setValue(50)
        self.sens.setToolTip("Higher finds fainter parts - and more things that are not parts")
        self.sensValue = QLabel("50")

        self.table = QTableWidget(0, 4)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        # one accept threshold per kind of part, for all its checks at once
        self.thresholds = {}
        # text checks: how sure the reader must be, for all of them
        from PySide6.QtWidgets import QDoubleSpinBox
        self.textConf = QDoubleSpinBox()
        self.textConf.setRange(0.05, 0.99)
        self.textConf.setSingleStep(0.05)
        self.textConf.setDecimals(2)
        self.textConf.setValue(0.5)
        self.textConf.setToolTip("Lines read with less confidence than this are not trusted - "
                                 "for every text check added")
        self.table.itemChanged.connect(self._table_changed)
        self.table.itemSelectionChanged.connect(self._highlight)
        # A click anywhere on a row keeps or drops it -- the tick box alone
        # was too small to hit. Double-click edits a text or a name.
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked |
                                   QAbstractItemView.EditKeyPressed)
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.setIconSize(QSize(20, 20))
        self.table.cellClicked.connect(self._row_clicked)
        self.table.cellDoubleClicked.connect(self._row_double_clicked)
        self._row_toggles = []                   # (time, row) of clicks that toggled

        self.likeBtn = QPushButton("Find all like this...")
        self.likeBtn.setToolTip("Drag a box round one part: every copy of it in the area is "
                                "found, turned or not - for rows of identical parts")
        self.match = QSlider(Qt.Horizontal)
        self.match.setRange(40, 95)
        self.match.setValue(60)
        self.match.setToolTip("How alike a copy must be (correlation)")
        self.matchValue = QLabel("0.60")

        from PySide6.QtWidgets import QCheckBox
        self.seated = QCheckBox("Also check each part sits right (lifted, tilted parts fail)")
        self.seated.setChecked(True)
        self.seated.setToolTip(
            "Part present alone only asks whether a part is there - a lifted or tilted part "
            "is. Ticked, each check also fails a part found away from its place or looking "
            "different from the reference part in a patch (raised end, missing solder).")
        self.allBtn = QPushButton("Keep all")
        self.noneBtn = QPushButton("Drop all")
        self.addBtn = QPushButton("Add checks")
        self.addBtn.setObjectName("primaryButton")
        self.addBtn.setEnabled(False)
        self.closeBtn = QPushButton("Close")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        lay.addWidget(intro)
        r = QHBoxLayout()
        r.addWidget(QLabel("Find"))
        r.addWidget(self.modeBox, 1)
        lay.addLayout(r)
        r = QHBoxLayout()
        r.addWidget(self.areaLabel, 1)
        r.addWidget(self.drawArea)
        r.addWidget(self.wholeBtn)
        lay.addLayout(r)
        lay.addWidget(self.findBtn)
        lay.addWidget(self.progress)
        lay.addWidget(self.status)
        self.sensRow = QWidget()
        r = QHBoxLayout(self.sensRow)
        r.setContentsMargins(0, 0, 0, 0)
        r.addWidget(QLabel("Sensitivity"))
        r.addWidget(self.sens, 1)
        r.addWidget(self.sensValue)
        lay.addWidget(self.sensRow)
        lay.addWidget(self.table, 1)
        self.confRow = QWidget()
        r = QHBoxLayout(self.confRow)
        r.setContentsMargins(0, 0, 0, 0)
        r.addWidget(QLabel("Reading confidence for these checks"))
        r.addStretch(1)
        r.addWidget(self.textConf)
        lay.addWidget(self.confRow)
        self.likeRow = QWidget()
        r = QHBoxLayout(self.likeRow)
        r.setContentsMargins(0, 0, 0, 0)
        r.addWidget(self.likeBtn)
        r.addWidget(QLabel("alike"))
        r.addWidget(self.match, 1)
        r.addWidget(self.matchValue)
        lay.addWidget(self.likeRow)
        lay.addWidget(self.seated)
        r = QHBoxLayout()
        r.addWidget(self.allBtn)
        r.addWidget(self.noneBtn)
        r.addStretch(1)
        r.addWidget(self.closeBtn)
        r.addWidget(self.addBtn)
        lay.addLayout(r)

        self.modeBox.currentIndexChanged.connect(lambda _i: self.set_mode(self.modeBox.currentData()))
        self.drawArea.clicked.connect(lambda: self._draw(self.set_area,
                                                         "Drag a box round the area to search"))
        self.wholeBtn.clicked.connect(lambda: self.set_area(None))
        self.findBtn.clicked.connect(lambda: self.find())
        self.sens.valueChanged.connect(self._sens_changed)
        self.likeBtn.clicked.connect(lambda: self._draw(self.find_like,
                                                        "Drag a box round one part"))
        self.match.valueChanged.connect(lambda v: self.matchValue.setText("%.2f" % (v / 100.0)))
        self.allBtn.clicked.connect(lambda: self._keep_all(True))
        self.noneBtn.clicked.connect(lambda: self._keep_all(False))
        self.addBtn.clicked.connect(lambda: self.add_checks())
        self.closeBtn.clicked.connect(self.close)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._refilter = QTimer(self)
        self._refilter.setSingleShot(True)
        self._refilter.timeout.connect(self._apply_filter)
        # the boxes' turning zones are screen-sized: their outline changes
        # with every zoom
        self.canvas.zoomChanged.connect(self._zoomed)
        self.set_mode("parts")

        if self.image is None:
            self.findBtn.setEnabled(False)
            self.status.setText("This inspection has no reference picture yet. Set one first "
                                "(Good reference), then find its parts.")

    # ---- mode ------------------------------------------------------------
    @property
    def text_mode(self):
        return self.mode in ("ocv", "ocr")

    def set_mode(self, mode):
        """Parts, or text read into Verify text / Read text checks. What was
        found in the other mode is cleared -- it was for other checks."""
        mode = mode if mode in dict(MODES) else "parts"
        changed = mode != self.mode
        was_text = self.text_mode
        self.mode = mode
        i = self.modeBox.findData(mode)
        if self.modeBox.currentIndex() != i:
            self.modeBox.blockSignals(True)
            self.modeBox.setCurrentIndex(i)
            self.modeBox.blockSignals(False)
        self.findBtn.setText("Find text" if self.text_mode else "Find parts")
        self.sensRow.setVisible(not self.text_mode)
        self.likeRow.setVisible(not self.text_mode)
        self.seated.setVisible(not self.text_mode)
        self.confRow.setVisible(self.text_mode)
        if self.text_mode:
            self.table.setHorizontalHeaderLabels(
                ["Keep", "Text it must read" if mode == "ocv" else "Must contain (clear: any)",
                 "Name the check", ""])
        else:
            self.table.setHorizontalHeaderLabels(["Type", "Found", "Name the checks",
                                                  "Threshold"])
        self.table.setColumnHidden(3, self.text_mode)
        if changed and (was_text != self.text_mode):
            self.parts, self.raw, self.names = [], None, {}
            self.dropped, self.off_types = set(), set()
            self._fill()
        elif changed:
            self._fill()                       # OCV <-> OCR: same boxes, other column title
        if self.image is not None and not self.text_mode and not pf.available():
            self.findBtn.setEnabled(False)
            self.status.setText("The part-finding model is missing (models/parts/%s)."
                                % pf.os.path.basename(pf.MODEL))
        elif self.image is not None:
            self.findBtn.setEnabled(True)
            self.status.setText("Press %s.%s" % (self.findBtn.text(), "" if self.text_mode else
                                                 "  Model: %s." % pf.model_name()))

    # ---- area ------------------------------------------------------------
    def _draw(self, then, message):
        self.ed._draw_hook = lambda kind, geom: then(geom.get("roi"))
        self.ed.display._set_mode("draw_rect")
        self.status.setText(message + " on the picture.")

    def set_area(self, roi, angle=0.0):
        self.area = [int(v) for v in roi] if roi else None
        self.area_angle = float(angle or 0.0) if roi else 0.0
        if self._area_item is not None:
            if self._area_item.scene() is not None:
                self.canvas.sc.removeItem(self._area_item)
            self._area_item = None
        if self.area:
            it = EditableBox(self.area, on_change=self._area_moved, z=44, angle=self.area_angle)
            pen = QPen(QColor("#FFFFFF"))
            pen.setCosmetic(True)
            pen.setWidthF(1.5)
            pen.setStyle(Qt.DashLine)
            it.setPen(pen)
            it.setToolTip("The area searched - drag to move it, drag an edge to resize it, "
                          "drag just outside a corner to turn it (Shift: 15 degree steps)")
            self.canvas.sc.addItem(it)
            self._area_item = it
            self.areaLabel.setText("Area: %d x %d px" % (self.area[2], self.area[3]))
        else:
            self.areaLabel.setText("Area: the whole picture")
        self.raw = None
        self.status.setText("Press %s." % self.findBtn.text())

    def _area_moved(self, roi, angle=0.0):
        self.area = list(roi)
        self.area_angle = float(angle or 0.0)
        self.raw = None
        self.areaLabel.setText("Area: %d x %d px%s" % (
            roi[2], roi[3], ("  turned %.1f deg" % self.area_angle) if self.area_angle else ""))
        self.status.setText("Area changed - press %s again." % self.findBtn.text())

    # ---- finding -----------------------------------------------------------
    def _search_view(self):
        """(picture, area, back): what the finder searches. A turned area is
        searched in an upright copy of it, and back() puts what is found in
        it onto the picture, turned with the area."""
        if self.area and abs(self.area_angle) > 1e-6:
            crop, back = pf.turned_area(self.image, self.area, self.area_angle)
            return crop, None, back
        return self.image, self.area, (lambda p: p)

    def _area_bounds(self):
        """The upright box round the area, turned or not (None: whole picture)."""
        if not self.area:
            return None
        if not self.area_angle:
            return list(self.area)
        x, y, w, h = self.area
        a = math.radians(self.area_angle)
        ex = abs(w * math.cos(a)) + abs(h * math.sin(a))
        ey = abs(w * math.sin(a)) + abs(h * math.cos(a))
        cx, cy = x + w / 2.0, y + h / 2.0
        return [int(cx - ex / 2), int(cy - ey / 2), int(ex), int(ey)]

    def find(self, wait=False):
        if self.image is None or self._job is not None:
            return
        self.findBtn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        text = self.text_mode
        self.status.setText("Reading text..." if text else "Finding parts...")
        img, area, back = self._search_view()
        sens = self.sens.value() / 100.0
        job = {"done": False, "result": None, "error": None, "progress": (0, 0), "text": text}

        def progress(i, n):
            job["progress"] = (i, n)

        def work():
            try:
                r = (pf.find_text(img, area, progress=progress) if text else
                     pf.find_parts(img, area, sens, progress=progress))
                r["parts"] = [back(p) for p in r["parts"]]
                job["result"] = r
            except Exception as e:
                job["error"] = e
            job["done"] = True

        self._job = job
        threading.Thread(target=work, daemon=True).start()
        self._timer.start(50)
        if wait:
            from PySide6.QtWidgets import QApplication
            t0 = time.time()
            while self._job is not None and time.time() - t0 < 300:
                QApplication.processEvents()
                time.sleep(0.02)

    def _poll(self):
        job = self._job
        if job is None:
            self._timer.stop()
            return
        i, n = job["progress"]
        if n:
            self.progress.setRange(0, n)
            self.progress.setValue(i)
        if not job["done"]:
            return
        self._timer.stop()
        self._job = None
        self.progress.setVisible(False)
        self.findBtn.setEnabled(True)
        if job["error"] is not None:
            self.status.setText("%s failed: %s" % ("Text reading" if job["text"] else
                                                   "The part finder", job["error"]))
            return
        if job["text"] != self.text_mode:
            return                              # the mode changed while it ran
        r = job["result"]
        self.raw = r.get("candidates")
        self.thresholds = {}                    # new types: thresholds start afresh
        self._show(r)

    def _sens_changed(self, v):
        self.sensValue.setText(str(v))
        if self.raw is not None and not self.text_mode:
            self._refilter.start(150)

    def _apply_filter(self):
        if self.raw is None or self.image is None or self.text_mode:
            return
        img, area, back = self._search_view()
        r = pf.find_parts(img, area, self.sens.value() / 100.0, raw=self.raw)
        r["parts"] = [back(p) for p in r["parts"]]
        self._show(r)

    def _show(self, r):
        self.parts = list(r["parts"])
        self.dropped = set()
        self.off_types = set()
        if self.text_mode:
            self.names = {}
            # silkscreen noise -- a lone character, a doubtful read -- starts
            # unticked: shown, one click to keep, never added unasked
            self.dropped = {i for i, p in enumerate(self.parts)
                            if len(p.get("text", "").strip()) < 2 or p.get("score", 0) < 0.6}
        self._fill()
        if self.text_mode:
            self.status.setText(
                "%d line%s of text in %.1f s (%s). Click a row (or its box) to keep or drop "
                "it; double-click the text to correct it; drag a box to move or resize it." % (
                    len(self.parts), "" if len(self.parts) == 1 else "s", r["ms"] / 1000.0,
                    r["device"]) if self.parts else "No text found in the area.")
        else:
            self.status.setText("%d parts of %d types in %.1f s (%s). Click a box to drop it; "
                                "drag it to move or resize; click a type's row to drop all of them."
                                % (len(self.parts), r["types"], r["ms"] / 1000.0, r["device"]))

    def find_like(self, roi):
        if self.image is None or not roi or self.text_mode:
            return
        found = pf.find_similar(self.image, roi, self._area_bounds(),
                                min_score=self.match.value() / 100.0)
        if not found:
            self.status.setText("No copies found - lower 'alike', or draw the box tighter "
                                "round the part.")
            return
        # a new type; parts already found where a copy is are replaced by it
        new_t = (max([p["type"] for p in self.parts]) + 1) if self.parts else 0
        boxes = np.array([[*f["box"][:2], f["box"][0] + f["box"][2], f["box"][1] + f["box"][3]]
                          for f in found], np.float32)
        keep_old = []
        for i, p in enumerate(self.parts):
            x, y, w, h = p["box"]
            b = np.array([[x, y, x + w, y + h]], np.float32)
            inter = pf._inter(b, boxes)[0]
            if (inter / max(1.0, w * h) > 0.5).any():
                continue
            keep_old.append(i)
        self.parts = [self.parts[i] for i in keep_old] + \
            [{"box": f["box"], "score": f["score"], "type": new_t} for f in found]
        self.dropped = set()
        self._fill()
        self.status.setText("Found %d copies (type %s). Name them in the list."
                            % (len(found), pf.type_name(new_t)))

    # ---- review --------------------------------------------------------------
    def type_label(self, t):
        name = self.names.get(t, "")
        return "Type %s%s" % (pf.type_name(t), (" - " + name) if name else "")

    def is_kept(self, i):
        if self.text_mode:
            return i not in self.dropped
        return i not in self.dropped and self.parts[i]["type"] not in self.off_types

    def kept(self):
        return [p for i, p in enumerate(self.parts) if self.is_kept(i)]

    def _colour(self, i):
        if not self.is_kept(i):
            return DROPPED
        return QColor(TEXT_COLOUR) if self.text_mode else type_colour(self.parts[i]["type"])

    def _style(self, i, bold=False):
        it = self.items[i]
        kept = self.is_kept(i)
        c = self._colour(i)
        pen = QPen(c)
        pen.setCosmetic(True)
        pen.setWidthF(3.5 if bold else (2.5 if kept else 1.2))
        if not kept:
            pen.setStyle(Qt.DashLine)
        it.setPen(pen)
        fill = QColor(c)
        fill.setAlpha(36 if kept else 0)
        it.setBrush(QBrush(fill))
        if self.text_mode:
            it.set_label(self.parts[i].get("text", ""), c)

    def toggle_part(self, i):
        if not self.text_mode and self.parts[i]["type"] in self.off_types:
            self.off_types.discard(self.parts[i]["type"])
            self._sync_table_checks()
            for k in range(len(self.items)):
                self._style(k)
        elif i in self.dropped:
            self.dropped.discard(i)
        else:
            self.dropped.add(i)
        self._style(i)
        if self.text_mode:
            self._sync_table_checks()
        self._update_add()

    def _box_changed(self, i, roi, angle=0.0):
        """A box moved, resized or turned on the picture: the check will use
        the box as it is now; a text box is read again, upright to it."""
        self.parts[i]["box"] = list(roi)
        self.parts[i]["angle"] = float(angle or 0.0)
        if self.text_mode and self.image is not None:
            text, conf = pf.read_box(self.image, roi, angle=self.parts[i]["angle"])
            self.parts[i]["text"], self.parts[i]["score"] = text, conf
            row = self._row_of(i)
            if row is not None:
                self.table.blockSignals(True)
                self.table.item(row, 1).setText(text)
                self.table.blockSignals(False)
            self.status.setText("Read again: %s" % (text or "no text in this box"))
        self._style(i)

    def _keep_all(self, on):
        self.dropped = set() if on else set(range(len(self.parts)))
        self.off_types = set()
        self._sync_table_checks()
        for k in range(len(self.items)):
            self._style(k)
        self._update_add()

    def _clear_items(self):
        for it in self.items:
            if it.scene() is not None:
                self.canvas.sc.removeItem(it)
        self.items = []

    def _fill(self):
        self._clear_items()
        for i, p in enumerate(self.parts):
            it = EditableBox(p["box"], on_click=lambda i=i: self.toggle_part(i),
                             on_change=lambda roi, ang, i=i: self._box_changed(i, roi, ang),
                             angle=p.get("angle", 0.0))
            what = ("%s  -  %s" % (p.get("text", ""), "confidence %.2f" % p.get("score", 0))
                    if self.text_mode else
                    "%s  -  confidence %.2f" % (self.type_label(p["type"]), p["score"]))
            it.setToolTip(what + "\nclick to keep or drop it; drag to move; drag an edge "
                                 "to resize; drag just outside a corner to turn it")
            self.canvas.sc.addItem(it)
            self.items.append(it)
        for k in range(len(self.items)):
            self._style(k)
        self.table.blockSignals(True)
        if self.text_mode:
            self.table.setRowCount(len(self.parts))
            for row, p in enumerate(self.parts):
                a = QTableWidgetItem(" %d" % (row + 1))
                a.setData(Qt.UserRole, row)
                a.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                b = QTableWidgetItem(p.get("text", ""))
                b.setToolTip("The text this check must read - correct it if it was misread"
                             if self.mode == "ocv" else
                             "The check fails unless this is read here - clear it for text "
                             "that changes from part to part, such as a serial number")
                c = QTableWidgetItem(self.names.get(row, ""))
                c.setToolTip("e.g. lot_code, serial - the check is named lot_code_01")
                for col, x in enumerate((a, b, c)):
                    self.table.setItem(row, col, x)
        else:
            types = sorted(set(p["type"] for p in self.parts))
            counts = {t: sum(1 for p in self.parts if p["type"] == t) for t in types}
            self.table.setRowCount(len(types))
            for row, t in enumerate(types):
                a = QTableWidgetItem(" %s" % pf.type_name(t))
                a.setData(Qt.UserRole, t)
                a.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                a.setForeground(QBrush(type_colour(t)))
                f = a.font()
                f.setBold(True)
                a.setFont(f)
                b = QTableWidgetItem(str(counts[t]))
                b.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                c = QTableWidgetItem(self.names.get(t, ""))
                c.setToolTip("e.g. inductor, sfp_cage, capacitor - the checks are named "
                             "inductor_01, inductor_02...")
                for col, x in enumerate((a, b, c)):
                    self.table.setItem(row, col, x)
                self.table.setCellWidget(row, 3, self._threshold_box(t))
        self.table.blockSignals(False)
        self._sync_table_checks()
        self._update_add()

    # ---- whole-row keep / drop -------------------------------------------
    def _kept_icon(self, kept):
        from vision_station.ui import icons
        if not hasattr(self, "_icons"):
            ok = icons._ph_icon("check-circle-fill", "#1F6FFF") or icons.icon("add")
            off = icons._ph_icon("circle", "#9AA0A6") or icons.QIcon()
            self._icons = (ok, off)
        return self._icons[0 if kept else 1]

    def _row_key_kept(self, row):
        a = self.table.item(row, 0)
        if a is None:
            return None, False
        key = a.data(Qt.UserRole)
        off = key in self.dropped if self.text_mode else key in self.off_types
        return key, not off

    def _row_clicked(self, row, _col):
        import time as _t
        self._row_toggles.append((_t.monotonic(), row))
        self._toggle_row(row)

    def _row_double_clicked(self, row, col):
        """A double-click is for editing: the clicks that made it are not
        meant to keep or drop the row."""
        import time as _t
        from PySide6.QtWidgets import QApplication
        now = _t.monotonic()
        window = QApplication.doubleClickInterval() / 1000.0 + 0.05
        undo = [r for (t, r) in self._row_toggles if r == row and now - t <= window]
        for _ in undo:
            self._toggle_row(row)
        self._row_toggles = []
        item = self.table.item(row, col)
        if item is not None and item.flags() & Qt.ItemIsEditable:
            self.table.editItem(item)

    def _toggle_row(self, row):
        key, kept = self._row_key_kept(row)
        if key is None:
            return
        if self.text_mode:
            (self.dropped.add if kept else self.dropped.discard)(key)
            self._style(key)
        else:
            (self.off_types.add if kept else self.off_types.discard)(key)
            for k in range(len(self.items)):
                if self.parts[k]["type"] == key:
                    self._style(k)
        self._sync_table_checks()
        self._update_add()

    def _paint_row(self, row, kept):
        """A kept row reads normally with a blue tick; a dropped one is
        greyed and struck through, with an empty circle."""
        for col in range(self.table.columnCount()):
            it = self.table.item(row, col)
            if it is None:
                continue
            f = it.font()
            f.setStrikeOut(not kept and col > 0)
            it.setFont(f)
            if col > 0:
                # the theme's own text colours: a fixed near-black was
                # invisible on the dark theme
                from vision_station.ui import theme
                it.setForeground(QBrush(QColor(theme.color("text" if kept else "text3"))))
        a = self.table.item(row, 0)
        if a is not None:
            a.setIcon(self._kept_icon(kept))

    def _threshold_box(self, t):
        """The accept threshold every check of this kind of part gets."""
        from PySide6.QtWidgets import QDoubleSpinBox
        sb = QDoubleSpinBox()
        sb.setRange(0.05, 0.99)
        sb.setSingleStep(0.05)
        sb.setDecimals(2)
        sb.setValue(float(self.thresholds.get(t, 0.65)))
        sb.setToolTip("How closely each %s must match the reference, 0 to 1, for every check "
                      "of this kind - lower it if good parts are missed, raise it if missing "
                      "parts get through. Can be changed for the whole group later, in the "
                      "editor." % pf.type_name(t))
        sb.valueChanged.connect(lambda v, t=t: self.thresholds.__setitem__(t, float(v)))
        self.thresholds.setdefault(t, float(sb.value()))
        return sb

    def _row_of(self, i):
        for row in range(self.table.rowCount()):
            a = self.table.item(row, 0)
            if a is not None and a.data(Qt.UserRole) == i:
                return row
        return None

    def _sync_table_checks(self):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            _key, kept = self._row_key_kept(row)
            self._paint_row(row, kept)
        self.table.blockSignals(False)

    def _table_changed(self, item):
        row = item.row()
        key = self.table.item(row, 0).data(Qt.UserRole)
        if self.text_mode:
            if item.column() == 1:
                self.parts[key]["text"] = item.text().strip()
                self._style(key)
            elif item.column() == 2:
                self.names[key] = item.text().strip()
        else:
            if item.column() == 2:
                self.names[key] = item.text().strip()
        self._update_add()

    def _highlight(self):
        rows = self.table.selectionModel().selectedRows()
        key = self.table.item(rows[0].row(), 0).data(Qt.UserRole) if rows else None
        if self.text_mode:
            sel = [key] if key is not None else []
        else:
            sel = [k for k, p in enumerate(self.parts) if p["type"] == key]
        for k in range(len(self.items)):
            self._style(k, bold=k in sel)
        if sel:
            r = self.items[sel[0]].sceneBoundingRect()
            for k in sel[1:]:
                r = r.united(self.items[k].sceneBoundingRect())
            self.canvas.ensureVisible(r, 40, 40)

    def _update_add(self):
        n = len(self.kept())
        what = {"parts": "Part present", "ocv": "Verify text", "ocr": "Read text"}[self.mode]
        self.addBtn.setText("Add %d %s check%s" % (n, what, "" if n == 1 else "s"))
        self.addBtn.setEnabled(n > 0)

    # ---- adding --------------------------------------------------------------
    def add_checks(self):
        """One check per kept box, in one undo step. The added boxes leave
        the picture; the panel stays open for the next area or kind."""
        kept_idx = [i for i in range(len(self.parts)) if self.is_kept(i)]
        if not kept_idx:
            return []
        tools = self.ed._tools()
        h, w = self.image.shape[:2]
        ids = [t["id"] for t in tools]
        if self.text_mode:
            items = [dict(self.parts[i], name=self.names.get(i, "")) for i in kept_idx]
            new = pf.text_tools(items, self.mode, ids, image_size=(w, h),
                                min_confidence=self.textConf.value())
            what = "Verify text" if self.mode == "ocv" else "Read text"
        else:
            names = {t: n for t, n in self.names.items() if n}
            new = pf.presence_tools([self.parts[i] for i in kept_idx], names, ids,
                                    image_size=(w, h), thresholds=self.thresholds,
                                    seated=self.seated.isChecked())
            what = "Part present"
        tools.extend(new)
        self.ed._edited()                    # one step: Ctrl+Z takes them all back
        msg = ("Added %d %s check%s - each box can be moved, resized or deleted; Ctrl+Z "
               "takes them all back" % (len(new), what, "" if len(new) == 1 else "s"))
        self.ed.statusBar().showMessage(msg, 10000)
        # what was added leaves the list; the rest stays to keep or drop
        left = [p for i, p in enumerate(self.parts) if i not in set(kept_idx)]
        self.parts, self.dropped, self.off_types = left, set(range(len(left))), set()
        if self.text_mode:
            self.names = {}
        self._fill()
        self.status.setText("Added %d %s check%s. Find more - another area or kind - or "
                            "Close." % (len(new), what, "" if len(new) == 1 else "s"))
        return [t["id"] for t in new]

    def _zoomed(self, _k=None):
        for it in self.items + ([self._area_item] if self._area_item is not None else []):
            if it.scene() is not None:
                it.prepareGeometryChange()

    def closeEvent(self, e):
        try:
            self.canvas.zoomChanged.disconnect(self._zoomed)
        except (RuntimeError, TypeError):
            pass
        self._timer.stop()
        self._clear_items()
        if self._area_item is not None and self._area_item.scene() is not None:
            self.canvas.sc.removeItem(self._area_item)
        self._area_item = None
        if getattr(self.ed, "_draw_hook", None) is not None:
            self.ed._draw_hook = None
        if getattr(self.ed, "_part_finder", None) is self:
            self.ed._part_finder = None
        super().closeEvent(e)
