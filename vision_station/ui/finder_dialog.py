"""Find picture: type or scan a barcode and its pictures are there.

Searches the saved result pictures as you type (core/picture_index.py holds
them in memory, so no key press waits for the disk). Click a picture to see
it; double-click it -- or press Show in folder -- and Explorer opens on its
folder with the picture selected.
"""
import os
import threading

import cv2
import numpy as np
from PySide6.QtCore import (QAbstractTableModel, QModelIndex, QObject, Qt, QTimer,
                            Signal)
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton,
                               QSizePolicy, QSplitter, QTableView, QVBoxLayout, QWidget)

from vision_station.core.picture_index import open_in_explorer
from vision_station.ui import theme

COLUMNS = ("Barcode", "Date", "Time", "Job", "Result")
SHOW_AT_MOST = 1000


class _Model(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.rows = []

    def set_rows(self, rows):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return COLUMNS[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        p = self.rows[index.row()]
        c = index.column()
        if role == Qt.DisplayRole:
            if c == 0:
                return p.code or "-"
            if c == 1:
                return p.day
            if c == 2:
                return p.time
            if c == 3:
                return p.job + ("  (original)" if p.original else "")
            return p.verdict.title()
        if role == Qt.ForegroundRole and c == 4 and p.verdict:
            from PySide6.QtGui import QColor
            return QColor(theme.tone_text("pass" if p.verdict == "ACCEPT" else "fail"))
        if role == Qt.ToolTipRole:
            return p.path
        return None


class _Loader(QObject):
    """Reads the picture for the preview off the window thread: a
    20-megapixel JPEG takes tens of milliseconds even read at a quarter size."""
    loaded = Signal(str, QImage)

    def load(self, path):
        def work():
            img = None
            try:
                data = np.fromfile(path, dtype=np.uint8)
                # a camera picture decoded straight at a quarter of the size:
                # four to eight times faster than decoding it whole and
                # shrinking it; a small one whole, or it would be blurred
                flag = cv2.IMREAD_REDUCED_COLOR_4 if data.size > 1_500_000 else cv2.IMREAD_COLOR
                img = cv2.imdecode(data, flag)
            except Exception:
                img = None
            if img is None:
                self.loaded.emit(path, QImage())
                return
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            q = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            try:
                self.loaded.emit(path, q)
            except RuntimeError:
                pass                                # the window closed first
        threading.Thread(target=work, daemon=True).start()


class FinderDialog(QDialog):
    """index: a PictureIndex. folders(): the folders to search, asked again
    each time the window opens (they follow Settings)."""

    def __init__(self, parent, index, folders):
        super().__init__(parent)
        self.setWindowTitle("Find picture")
        self.setObjectName("finderDialog")
        self.resize(1100, 680)
        self.index = index
        self.folders = folders
        self._version = -1
        self._shown_path = None
        self._pixmap = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(10)

        top = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setObjectName("monoField")
        self.query.setPlaceholderText("Type or scan a barcode - or a date, a job name...")
        self.query.setClearButtonEnabled(True)
        self.query.setMinimumHeight(34)
        f = self.query.font()
        f.setPointSizeF(f.pointSizeF() * 1.25)
        self.query.setFont(f)
        self.query.textChanged.connect(self._search)
        self.query.returnPressed.connect(self._enter)
        top.addWidget(self.query, 1)
        self.which = QComboBox()
        for text, v in (("All results", None), ("Rejected", "REJECT"), ("Accepted", "ACCEPT")):
            self.which.addItem(text, v)
        self.which.currentIndexChanged.connect(self._search)
        top.addWidget(self.which)
        self.originals = QCheckBox("Originals")
        self.originals.setToolTip("Also list the untouched camera pictures")
        self.originals.setChecked(False)
        self.originals.toggled.connect(self._search)
        top.addWidget(self.originals)
        lay.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        self.model = _Model()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        hh = self.table.horizontalHeader()
        # the barcode is what is looked for: it keeps its whole width, the
        # job name takes what is left
        for c in (0, 1, 2, 4):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.selectionModel().currentRowChanged.connect(self._selected)
        self.table.doubleClicked.connect(lambda _i: self._show_in_folder())
        split.addWidget(self.table)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 0, 0)
        self.preview = QLabel("Choose a picture to see it here")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(320, 240)
        # the picture follows the label's size, never the other way round:
        # sized by its pixmap, the preview pushed the list to a sliver
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview.setObjectName("finderPreview")
        self.preview.setCursor(Qt.PointingHandCursor)
        self.preview.mouseDoubleClickEvent = lambda _e: self._open_picture()
        rl.addWidget(self.preview, 1)
        self.where = QLabel("")
        self.where.setWordWrap(True)
        self.where.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.where.setObjectName("rowHelp")
        self.where.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        rl.addWidget(self.where)
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.openPic = QPushButton("Open picture")
        self.openPic.clicked.connect(self._open_picture)
        self.showFolder = QPushButton("Show in folder")
        self.showFolder.setObjectName("primaryButton")
        self.showFolder.setToolTip("Open the folder in Explorer with the picture selected "
                                   "(or double-click the picture in the list)")
        self.showFolder.clicked.connect(self._show_in_folder)
        btns.addWidget(self.openPic)
        btns.addWidget(self.showFolder)
        rl.addLayout(btns)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setChildrenCollapsible(False)
        self.table.setMinimumWidth(420)
        split.setSizes([600, 470])
        lay.addWidget(split, 1)

        self.status = QLabel("")
        self.status.setObjectName("rowHelp")
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(self.status)

        self.loader = _Loader()
        self.loader.loaded.connect(self._loaded)
        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.timeout.connect(
            lambda: self._shown_path and self.loader.load(self._shown_path))
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self._focus_query)
        QShortcut(QKeySequence(Qt.Key_Down), self.query, activated=self._down)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(500)
        self._enable_buttons(False)
        self.refresh()

    # -- index ----------------------------------------------------------------
    def refresh(self):
        """Pick up the folders as Settings has them now and walk them again in
        the background; what was indexed before is searched meanwhile."""
        self.index.set_roots(self.folders())
        self.index.scan_in_background()
        self._search()

    def _poll(self):
        if self.index.version != self._version:
            self._search()
        elif self.index.scanning:
            self._status()

    # -- searching ------------------------------------------------------------
    def _search(self, *_):
        self._version = self.index.version
        keep = self._current()
        found, total = self.index.search(self.query.text(), SHOW_AT_MOST,
                                         verdict=self.which.currentData(),
                                         originals=self.originals.isChecked())
        self.model.set_rows(found)
        self._total = total
        row = next((i for i, p in enumerate(found) if keep and p.path == keep.path),
                   0 if found and self.query.text().strip() else -1)
        if row >= 0:
            self.table.selectRow(row)
        else:
            self._show(None)
        self._status()

    def _status(self):
        n = len(self.index)
        roots = self.index.roots
        if not roots:
            self.status.setText("No result folder yet - set one in Settings > Results "
                                "to keep each part's pictures.")
            return
        where = "; ".join(roots)
        shown = len(self.model.rows)
        txt = "%d of %s pictures" % (self._total, format(n, ","))
        if shown < self._total:
            txt += " (the newest %d listed - type more to narrow it)" % shown
        if self.index.scanning:
            txt += "   -   looking through the folders..."
        self.status.setText("%s   -   %s" % (txt, where))

    def _enter(self):
        """Enter in the box -- a scanner ends every code with it: show the
        newest picture of that code; Enter again opens its folder."""
        cur = self._current()
        if cur is not None and getattr(self, "_entered", None) == (self.query.text(), cur.path):
            self._show_in_folder()
            return
        if self.model.rows:
            self.table.selectRow(0)
            self._entered = (self.query.text(), self.model.rows[0].path)

    def _down(self):
        if self.model.rows:
            self.table.setFocus()
            self.table.selectRow(0)

    def _focus_query(self):
        self.query.setFocus()
        self.query.selectAll()

    # -- one picture ----------------------------------------------------------
    def _current(self):
        idx = self.table.currentIndex()
        if not idx.isValid() or idx.row() >= len(self.model.rows):
            return None
        return self.model.rows[idx.row()]

    def _selected(self, cur, _prev):
        self._show(self.model.rows[cur.row()] if cur.isValid() and
                   cur.row() < len(self.model.rows) else None)

    def _show(self, p):
        self._enable_buttons(p is not None)
        if p is None:
            self._shown_path = None
            self._pixmap = None
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Choose a picture to see it here")
            self.where.setText("")
            return
        if p.path == self._shown_path:
            return
        self._shown_path = p.path
        # below the results folder -- the day, the code, the file -- which is
        # what tells pictures apart; the whole path is in the tooltip
        self.where.setText(os.path.relpath(p.path, p.root))
        self.where.setToolTip(p.path)
        # read once typing pauses: a picture per key press would queue
        # decodes of pictures nobody looks at
        self._load_timer.start(90)

    def _loaded(self, path, qimg):
        if path != self._shown_path:
            return                                   # a newer choice is on its way
        if qimg.isNull():
            self._pixmap = None
            self.preview.setPixmap(QPixmap())
            self.preview.setText("This picture cannot be read - it may have been moved")
            return
        self._pixmap = QPixmap.fromImage(qimg)
        self._fit()

    def _fit(self):
        if self._pixmap is not None:
            self.preview.setPixmap(self._pixmap.scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def _enable_buttons(self, on):
        self.openPic.setEnabled(on)
        self.showFolder.setEnabled(on)

    def _show_in_folder(self):
        p = self._current()
        if p is not None:
            open_in_explorer(p.path if os.path.exists(p.path) else p.folder())

    def _open_picture(self):
        p = self._current()
        if p is not None and os.path.exists(p.path):
            os.startfile(p.path) if hasattr(os, "startfile") else open_in_explorer(p.path)

    def closeEvent(self, e):
        self._timer.stop()
        super().closeEvent(e)
