"""Find parts: the panel beside a job editor that runs the part finder on the
job's reference picture, shows what it found as boxes on the picture, and
turns what the user keeps into "Part present" checks.

    1. Area       the whole picture, or a box dragged round part of it
    2. Find       the network runs on the graphics card (a few seconds for a
                  whole 20-megapixel board); a progress bar shows the passes
    3. Review     a box per part, coloured by type. Click a box to drop it or
                  bring it back; untick a type to drop all of its kind;
                  Sensitivity re-filters at once without running again;
                  name a type ("inductor") and its checks are called that.
                  "Find all like this" takes one box drawn round a part and
                  finds every copy of it -- for the rows of identical parts
                  the network can miss.
    4. Add        one check per kept part, in one step (Ctrl+Z takes them
                  all back), each an ordinary box to move, resize or delete.
"""
import threading
import time

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QGraphicsRectItem,
                               QGraphicsSimpleTextItem, QHBoxLayout, QHeaderView, QLabel,
                               QProgressBar, QPushButton, QSlider, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from vision_station.core import part_finder as pf

PALETTE = ["#00B3FF", "#FF9F1C", "#2EC4B6", "#E71D73", "#9B5DE5", "#F15BB5", "#00F5D4",
           "#FEE440", "#8AC926", "#FF595E", "#6A4C93", "#1982C4", "#FFCA3A", "#C77DFF",
           "#06D6A0", "#EF476F", "#118AB2", "#FFD166", "#73D2DE", "#D81159"]


def type_colour(t):
    return QColor(PALETTE[t % len(PALETTE)])


class PreviewBox(QGraphicsRectItem):
    """One found part on the picture. A click drops it or brings it back."""

    def __init__(self, panel, index, part):
        x, y, w, h = part["box"]
        super().__init__(QRectF(x, y, w, h))
        self.panel, self.index = panel, index
        self.setZValue(45)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("%s  -  confidence %.2f\nclick to drop it or bring it back"
                        % (panel.type_label(part["type"]), part["score"]))
        self.restyle()

    def restyle(self):
        part = self.panel.parts[self.index]
        kept = self.panel.is_kept(self.index)
        c = type_colour(part["type"]) if kept else QColor(150, 150, 150)
        pen = QPen(c)
        pen.setCosmetic(True)
        pen.setWidthF(2.0 if kept else 1.2)
        if not kept:
            pen.setStyle(Qt.DashLine)
        self.setPen(pen)
        fill = QColor(c)
        fill.setAlpha(38 if kept else 0)
        self.setBrush(QBrush(fill))

    def mousePressEvent(self, e):
        self.panel.toggle_part(self.index)
        e.accept()


class PartFinderPanel(QDialog):
    """editor: the JobEditor (its display's canvas, add/undo plumbing)."""

    def __init__(self, editor):
        super().__init__(editor, Qt.Tool)
        self.ed = editor
        self.setWindowTitle("Find parts")
        self.setObjectName("partFinder")
        self.resize(430, 640)
        self.image = editor.project().reference
        self.canvas = editor.display.canvas
        self.area = None                    # [x, y, w, h] or None = whole picture
        self.raw = None                     # network candidates for the area
        self.parts = []                     # [{"box", "score", "type"}]
        self.dropped = set()                # part indices the user dropped
        self.off_types = set()
        self.names = {}                     # type -> name typed by the user
        self.items = []
        self._area_item = None
        self._job = None

        intro = QLabel("Finds every component in an area with a neural network on the "
                       "graphics card. Review what it found, then add a \"Part present\" "
                       "check for each part you keep.")
        intro.setWordWrap(True)
        intro.setObjectName("hint")

        self.areaLabel = QLabel("Area: the whole picture")
        self.drawArea = QPushButton("Draw area...")
        self.drawArea.setToolTip("Drag a box round the part of the picture to search")
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

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Type", "Found", "Name the checks"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.itemChanged.connect(self._table_changed)
        self.table.itemSelectionChanged.connect(self._highlight_type)

        self.likeBtn = QPushButton("Find all like this...")
        self.likeBtn.setToolTip("Drag a box round one part: every copy of it in the area is "
                                "found, turned or not - for rows of identical parts")
        self.match = QSlider(Qt.Horizontal)
        self.match.setRange(40, 95)
        self.match.setValue(60)
        self.match.setToolTip("How alike a copy must be (correlation)")
        self.matchValue = QLabel("0.60")

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
        r.addWidget(self.areaLabel, 1)
        r.addWidget(self.drawArea)
        r.addWidget(self.wholeBtn)
        lay.addLayout(r)
        lay.addWidget(self.findBtn)
        lay.addWidget(self.progress)
        lay.addWidget(self.status)
        r = QHBoxLayout()
        r.addWidget(QLabel("Sensitivity"))
        r.addWidget(self.sens, 1)
        r.addWidget(self.sensValue)
        lay.addLayout(r)
        lay.addWidget(self.table, 1)
        r = QHBoxLayout()
        r.addWidget(self.likeBtn)
        r.addWidget(QLabel("alike"))
        r.addWidget(self.match, 1)
        r.addWidget(self.matchValue)
        lay.addLayout(r)
        r = QHBoxLayout()
        r.addWidget(self.allBtn)
        r.addWidget(self.noneBtn)
        r.addStretch(1)
        r.addWidget(self.closeBtn)
        r.addWidget(self.addBtn)
        lay.addLayout(r)

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

        if self.image is None:
            self.findBtn.setEnabled(False)
            self.status.setText("This inspection has no reference picture yet. Set one first "
                                "(Image Source), then find its parts.")
        elif not pf.available():
            self.findBtn.setEnabled(False)
            self.status.setText("The part-finding model is missing (models/parts/%s)."
                                % pf.os.path.basename(pf.MODEL))

    # ---- area ------------------------------------------------------------
    def _draw(self, then, message):
        self.ed._draw_hook = lambda kind, geom: then(geom.get("roi"))
        self.ed.display._set_mode("draw_rect")
        self.status.setText(message + " on the picture.")

    def set_area(self, roi):
        self.area = list(roi) if roi else None
        if self._area_item is not None:
            self.canvas.sc.removeItem(self._area_item)
            self._area_item = None
        if self.area:
            x, y, w, h = self.area
            it = QGraphicsRectItem(QRectF(x, y, w, h))
            pen = QPen(QColor("#FFFFFF"))
            pen.setCosmetic(True)
            pen.setWidthF(1.5)
            pen.setStyle(Qt.DashLine)
            it.setPen(pen)
            it.setZValue(44)
            it.setAcceptedMouseButtons(Qt.NoButton)
            self.canvas.sc.addItem(it)
            self._area_item = it
            self.areaLabel.setText("Area: %d x %d px" % (w, h))
        else:
            self.areaLabel.setText("Area: the whole picture")
        self.raw = None
        self.status.setText("Press Find parts.")

    # ---- finding -----------------------------------------------------------
    def find(self, wait=False):
        if self.image is None or self._job is not None:
            return
        self.findBtn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText("Finding parts...")
        img, area, sens = self.image, self.area, self.sens.value() / 100.0
        job = {"done": False, "result": None, "error": None, "progress": (0, 0)}

        def progress(i, n):
            job["progress"] = (i, n)

        def work():
            try:
                job["result"] = pf.find_parts(img, area, sens, progress=progress)
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
            self.status.setText("The part finder failed: %s" % job["error"])
            return
        r = job["result"]
        self.raw = r["candidates"]
        self._show(r)

    def _sens_changed(self, v):
        self.sensValue.setText(str(v))
        if self.raw is not None:
            self._refilter.start(150)

    def _apply_filter(self):
        if self.raw is None or self.image is None:
            return
        r = pf.find_parts(self.image, self.area, self.sens.value() / 100.0, raw=self.raw)
        self._show(r)

    def _show(self, r):
        self.parts = list(r["parts"])
        self.dropped = set()
        self.off_types = set()
        self._fill()
        self.status.setText("%d parts of %d types in %.1f s (%s). Click a box to drop it; "
                            "untick a type to drop all of them."
                            % (len(self.parts), r["types"], r["ms"] / 1000.0, r["device"]))

    def find_like(self, roi):
        if self.image is None or not roi:
            return
        found = pf.find_similar(self.image, roi, self.area, min_score=self.match.value() / 100.0)
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
        return i not in self.dropped and self.parts[i]["type"] not in self.off_types

    def kept(self):
        return [p for i, p in enumerate(self.parts) if self.is_kept(i)]

    def toggle_part(self, i):
        if self.parts[i]["type"] in self.off_types:
            self.off_types.discard(self.parts[i]["type"])
            self._sync_table_checks()
            for it in self.items:
                it.restyle()
        elif i in self.dropped:
            self.dropped.discard(i)
        else:
            self.dropped.add(i)
        self.items[i].restyle()
        self._update_add()

    def _keep_all(self, on):
        self.dropped = set() if on else set(range(len(self.parts)))
        self.off_types = set()
        self._sync_table_checks()
        for it in self.items:
            it.restyle()
        self._update_add()

    def _clear_items(self):
        for it in self.items:
            if it.scene() is not None:
                self.canvas.sc.removeItem(it)
        self.items = []

    def _fill(self):
        self._clear_items()
        for i, p in enumerate(self.parts):
            it = PreviewBox(self, i, p)
            self.canvas.sc.addItem(it)
            self.items.append(it)
        types = sorted(set(p["type"] for p in self.parts))
        counts = {t: sum(1 for p in self.parts if p["type"] == t) for t in types}
        self.table.blockSignals(True)
        self.table.setRowCount(len(types))
        for row, t in enumerate(types):
            a = QTableWidgetItem("  %s" % pf.type_name(t))
            a.setData(Qt.UserRole, t)
            a.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            a.setCheckState(Qt.Unchecked if t in self.off_types else Qt.Checked)
            a.setForeground(QBrush(type_colour(t)))
            f = a.font()
            f.setBold(True)
            a.setFont(f)
            b = QTableWidgetItem(str(counts[t]))
            b.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            c = QTableWidgetItem(self.names.get(t, ""))
            c.setToolTip("e.g. inductor, sfp_cage, capacitor - the checks are named "
                         "inductor_01, inductor_02...")
            self.table.setItem(row, 0, a)
            self.table.setItem(row, 1, b)
            self.table.setItem(row, 2, c)
        self.table.blockSignals(False)
        self._update_add()

    def _sync_table_checks(self):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            a = self.table.item(row, 0)
            a.setCheckState(Qt.Unchecked if a.data(Qt.UserRole) in self.off_types
                            else Qt.Checked)
        self.table.blockSignals(False)

    def _table_changed(self, item):
        row = item.row()
        t = self.table.item(row, 0).data(Qt.UserRole)
        if item.column() == 0:
            if item.checkState() == Qt.Checked:
                self.off_types.discard(t)
            else:
                self.off_types.add(t)
            for it in self.items:
                if self.parts[it.index]["type"] == t:
                    it.restyle()
        elif item.column() == 2:
            self.names[t] = item.text().strip()
        self._update_add()

    def _highlight_type(self):
        rows = self.table.selectionModel().selectedRows()
        t = self.table.item(rows[0].row(), 0).data(Qt.UserRole) if rows else None
        sel = [it for it in self.items if self.parts[it.index]["type"] == t]
        for it in self.items:
            pen = it.pen()
            pen.setWidthF(3.5 if it in sel else (2.0 if self.is_kept(it.index) else 1.2))
            it.setPen(pen)
        if sel:
            r = sel[0].sceneBoundingRect()
            for it in sel[1:]:
                r = r.united(it.sceneBoundingRect())
            self.canvas.ensureVisible(r, 40, 40)

    def _update_add(self):
        n = len(self.kept())
        self.addBtn.setText("Add %d check%s" % (n, "" if n == 1 else "s"))
        self.addBtn.setEnabled(n > 0)

    # ---- adding --------------------------------------------------------------
    def add_checks(self):
        kept = self.kept()
        if not kept:
            return []
        tools = self.ed._tools()
        h, w = self.image.shape[:2]
        names = {t: n for t, n in self.names.items() if n}
        new = pf.presence_tools(kept, names, [t["id"] for t in tools], image_size=(w, h))
        tools.extend(new)
        self.ed._edited()                    # one step: Ctrl+Z takes them all back
        self.ed.statusBar().showMessage(
            "Added %d Part present checks - each box can be moved, resized or deleted; "
            "Ctrl+Z takes them all back" % len(new), 10000)
        self.close()
        return [t["id"] for t in new]

    def closeEvent(self, e):
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
