"""A separate editor window for one tool, like VisionPro's Tool Edit Control.

Double-click a tool in the job tree and it opens in its own window: its own
image display showing only that tool's region and its results, the tool's
Settings / Region / Last result tabs, a table of everything it measured, and a
Run button. Several can be open at once, one per tool, and each can be put on a
second monitor beside the main window.

Nothing is copied. The editor edits the same tool record the main window holds,
and its canvas sends every move and resize to the same handlers the main canvas
uses -- undo included -- so a change made in either window is one change to one
recipe, and both windows redraw from it.
"""
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QHeaderView, QLabel, QMainWindow, QSplitter,
                               QTableWidget, QTableWidgetItem, QToolBar, QVBoxLayout,
                               QWidget)

from vision_station.ui.canvas import ImageCanvas
from vision_station.ui.widgets import readable_images


class ToolEditor(QMainWindow):
    def __init__(self, main, tool_id):
        super().__init__(main)                 # stays with the main window
        self.main = main
        self.tool_id = tool_id
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1180, 760)

        bar = QToolBar("Tool")
        bar.setMovable(False)
        bar.setIconSize(QSize(13, 13))
        self.addToolBar(bar)
        run = QAction("Run job", self)
        run.setToolTip("Run the whole job and show this tool's result")
        run.triggered.connect(self.run)
        bar.addAction(run)
        bar.addSeparator()
        for text, fn in (("Frame tool", lambda: self.frame_tool()),
                         ("Fit", lambda: self.canvas.fit()),
                         ("Actual size", lambda: self.canvas.zoom_100()),
                         ("Zoom in", lambda: self.canvas.zoom(1.25)),
                         ("Zoom out", lambda: self.canvas.zoom(0.8))):
            a = QAction(text, self)
            a.triggered.connect(fn)
            bar.addAction(a)
        bar.addSeparator()
        self.verdict = QLabel("not run")
        self.verdict.setMinimumWidth(80)
        self.verdict.setAlignment(Qt.AlignCenter)
        bar.addWidget(self.verdict)
        self.detail = QLabel("")
        self.detail.setStyleSheet("color:#9AA9B8; padding-left:10px;")
        bar.addWidget(self.detail)

        # --- the tool's own image display
        self.canvas = ImageCanvas()
        # Names only where they add something. The window title names the tool,
        # so tagging its region as well -- beside its result label and the size
        # readout -- stacked three labels on one small feature.
        self.canvas.label_policy = "smart"
        for sig, slot in ((self.canvas.editBegan, main.on_edit_began),
                          (self.canvas.liveGeometry, main.on_live_geometry),
                          (self.canvas.angleChanged, main.on_angle_changed),
                          (self.canvas.polygonCommitted, main.on_polygon_committed)):
            sig.connect(slot)
        self.canvas.geometryCommitted.connect(self._geometry_done)
        self.canvas.polygonCommitted.connect(lambda *_: self._after_edit())
        self.canvas.angleChanged.connect(lambda *_: self._after_edit())

        # --- the tool's settings: the same property sheet, its own instance
        from vision_station.ui.classic_window import PropertySheet
        self.props = PropertySheet()
        self.props.changed.connect(self._props_changed)

        self.results = QTableWidget(0, 2)
        self.results.setHorizontalHeaderLabels(["Measured", "Value"])
        self.results.verticalHeader().hide()
        self.results.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.results.horizontalHeader().setStretchLastSection(True)

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.props)
        right.addWidget(self.results)
        right.setSizes([470, 230])
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.canvas)
        split.addWidget(right)
        split.setSizes([760, 420])
        holder = QWidget()
        v = QVBoxLayout(holder)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(split)
        self.setCentralWidget(holder)

        close = QAction(self)
        close.setShortcut(QKeySequence("Escape"))
        close.triggered.connect(self.close)
        self.addAction(close)
        self.refresh(fit=True)

    # ------------------------------------------------------------ helpers
    def spec(self):
        return next((t for t in self.main.spec.get("tools", [])
                     if t["id"] == self.tool_id), None)

    def refresh(self, fit=False):
        """Redraw from the main window's recipe, image and last result. Called
        by the main window after every inspection and every edit."""
        t = self.spec()
        if t is None:                 # the tool was deleted or renamed
            self.close()
            return
        from vision_station.core.schema import LABELS
        kind = LABELS.get(t["type"], (t["type"], ""))[0]
        self.setWindowTitle("%s  -  %s   (%s)" % (t["id"], kind,
                                                self.main.current_pid() or "model"))
        img = self.main.canvas._img
        if img is not None:
            self.canvas.set_image(img, keep_view=not fit)
        res = self.main._result_for(t["id"])
        # only this tool's region, and only its graphics
        last = {"tools": [res]} if res else None
        self.canvas.sync_regions([t], last)
        self.canvas.draw_results([res] if res else [])
        self.canvas.select_ids([t["id"]], silent=True)
        if fit and img is not None:
            self.frame_tool()

        self.props.recipe = self.main.spec.get("tools", [])
        self.props.set_captures(readable_images(self.main.spec.get("tools", []), t["id"],
                                                self.main.capture_names()))
        if self.props.tool is not t:
            self.props.load(t)
        self.props.show_result(res)
        self._show_result(res)

    def frame_tool(self):
        """Open on the tool's region with some surroundings, not the whole
        frame: in a full photograph of a chassis the one hole this editor is
        about was a speck. A tool with no region of its own -- a distance, an
        image filter -- shows the whole image."""
        from PySide6.QtCore import QRectF
        it = self.canvas.regions.get(self.tool_id)
        if it is None:
            self.canvas.fit()
            return
        r = it.mapRectToScene(it.local_rect())
        cx, cy = r.center().x(), r.center().y()
        # A margin around the region, not a multiple of it: three times a
        # 260 px region is wider than a 640 px image, so a multiple framed wide
        # regions as the whole image. The view is clamped to the image, so a
        # large region shows the image rather than empty canvas beyond it.
        w = r.width() + 2 * max(60.0, 0.4 * r.width())
        h = r.height() + 2 * max(60.0, 0.4 * r.height())
        img = self.canvas._img
        if img is not None:
            iw, ih = float(img.shape[1]), float(img.shape[0])
            if w >= iw and h >= ih:
                self.canvas.fit()
                return
            w, h = min(w, iw), min(h, ih)
            cx = min(max(cx, w / 2), iw - w / 2)
            cy = min(max(cy, h / 2), ih - h / 2)
        self.canvas.fitInView(QRectF(cx - w / 2, cy - h / 2, w, h), Qt.KeepAspectRatio)
        self.canvas._pad_scene()
        self.canvas.centerOn(cx, cy)
        lo, hi = self.canvas.scale_limits()
        k = self.canvas.transform().m11()
        if k > hi or k < lo:
            self.canvas.zoom(max(lo, min(hi, k)) / k)
        self.canvas.fitted = False
        self.canvas.fit_mode = False
        self.canvas._refresh_items()
        self.canvas.refresh_detail()

    def _show_result(self, res):
        colours = {"pass": ("#12A150", "#04140A"), "fail": ("#E5484D", "#1B0405"),
                   "error": ("#F5A524", "#1A1102")}
        if not res:
            self.verdict.setText("not run")
            self.verdict.setStyleSheet("background:#3A424C; color:#DDE2E8; font-weight:700;"
                                       "padding:4px 10px; border-radius:3px;")
            self.detail.setText("")
            self.results.setRowCount(0)
            return
        bg, fg = colours.get(res["status"], ("#3A424C", "#DDE2E8"))
        self.verdict.setText(res["status"].upper())
        self.verdict.setStyleSheet("background:%s; color:%s; font-weight:700;"
                                   "padding:4px 10px; border-radius:3px;" % (bg, fg))
        self.detail.setText("%.1f ms   %s" % (res.get("ms", 0.0), res.get("message") or ""))
        rows = [("score", "%.3f" % float(res.get("score", 0.0)))]
        for k, val in (res.get("values") or {}).items():
            if isinstance(val, dict):
                continue
            if isinstance(val, list):
                if val and isinstance(val[0], dict):
                    continue
                val = ", ".join(str(x) for x in val[:12]) + (" ..." if len(val) > 12 else "")
            rows.append((k, str(val)))
        self.results.setRowCount(len(rows))
        for i, (k, val) in enumerate(rows):
            self.results.setItem(i, 0, QTableWidgetItem(k))
            self.results.setItem(i, 1, QTableWidgetItem(val))

    # ------------------------------------------------------------ edits
    def _geometry_done(self, tid, roi):
        self.main.on_geometry(tid, roi)
        self._after_edit()

    def _after_edit(self):
        """An edit here redraws the main window; the main window then redraws
        every open editor, this one included."""
        self.main.editor_changed(self.tool_id)

    def _props_changed(self):
        self.main.mark("change setting")
        self.main.dirty = True
        self._after_edit()

    def run(self):
        self.main.run_once()

    def closeEvent(self, e):
        self.main._editors.pop(self.tool_id, None)
        super().closeEvent(e)
