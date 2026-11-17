"""VisionDisplay: the one image display used everywhere in the VisionPro-style
front end -- job editors, tool edit controls, floating displays, the run view.

Laid out the way current design tools are (Figma, Illustrator, ArtCraft): the
image fills the panel edge to edge, and its controls float over it on rounded
bars rather than taking strips of the window --

    left        the tool rail: select, pan, the drawing tools where a window
                can use them, zoom in and out, fit, actual size, pixel grid,
                result graphics, search guides
    top centre  the transform bar, when regions are selected: X, Y, width,
                height and angle to type exactly; rotate by 90, straighten,
                lock; with several selected, align, distribute and match size
    top right   the record selector -- which image to show (Current.InputImage,
                LastRun.InputImage, a tool's output image...)
    bottom left the coordinate space, the pointer position and the pixel value
    bottom right zoom out, the zoom level, zoom in, fit

Open-source VisionPro and VisionMaster imitations converge on the same design:
one display control, built and tested once, reused by every window. Most viewer
bugs in this application came from behaviour implemented slightly differently
in several places, so there is now exactly one place.
"""
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QGraphicsDropShadowEffect,
                               QHBoxLayout, QLabel, QSpinBox, QToolBar, QToolButton,
                               QVBoxLayout, QWidget)

from vision_station.ui.canvas import ImageCanvas, RegionItem

MARGIN = 10                                   # from the display's edge to a bar


def _float_bar(parent, vertical=False):
    """A rounded bar that floats over the image."""
    f = QFrame(parent)
    f.setObjectName("floatBar")
    f.setAttribute(Qt.WA_StyledBackground, True)
    lay = QVBoxLayout(f) if vertical else QHBoxLayout(f)
    lay.setContentsMargins(5, 5, 5, 5)
    lay.setSpacing(2)
    shadow = QGraphicsDropShadowEffect(f)
    shadow.setBlurRadius(18)
    shadow.setOffset(0, 3)
    shadow.setColor(QColor(0, 0, 0, 70))
    f.setGraphicsEffect(shadow)
    return f, lay


def _sep(vertical=False):
    s = QFrame()
    s.setObjectName("floatSep")
    if vertical:
        s.setFixedHeight(1)
        s.setStyleSheet("min-height:1px;max-height:1px;min-width:22px;")
    else:
        s.setFixedWidth(1)
    return s


class VisionDisplay(QWidget):
    recordChanged = Signal(str)

    ICONS = {"Pointer": "pointer", "Pan": "pan", "Zoom in": "zoom_in", "Zoom out": "zoom_out",
             "Fit": "fit", "1:1": "actual", "Grid": "grid", "Graphics": "graphics",
             "Search guides": "guides", "Rectangle": "rect", "Ellipse": "ellipse",
             "Polygon": "polygon"}

    def __init__(self, parent=None, records=True, toolbar=True):
        super().__init__(parent)
        self.canvas = ImageCanvas()
        self._records = {}                  # name -> (image, graphics)
        self._toolbar = toolbar
        self._records_shown = records

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.canvas, 1)

        # ---- the tool rail ------------------------------------------------
        self.rail, rl = _float_bar(self, vertical=True)
        bar = QToolBar()
        bar.setObjectName("displayToolbar")
        bar.setOrientation(Qt.Vertical)
        bar.setIconSize(QSize(18, 18))
        bar.setToolButtonStyle(Qt.ToolButtonIconOnly)      # names in tooltips
        rl.addWidget(bar)
        self._bar = bar
        grp = QActionGroup(self)
        self._mode_group = grp
        self.aPointer = self._act(bar, "Pointer", lambda: self._set_mode("select"),
                                  "Select, move, resize and rotate regions   (V)", True, grp)
        self.aPan = self._act(bar, "Pan", lambda: self._set_mode("hand"),
                              "Drag to move the image   (H, or hold Space; the middle "
                              "button always pans)", True, grp)
        self.aPointer.setChecked(True)
        # drawing tools: shown only where the window can use a drawn region
        self.aRect = self._act(bar, "Rectangle", lambda: self._set_mode("draw_rect"),
                               "Draw the region as a rectangle   (R)", True, grp)
        self.aEllipse = self._act(bar, "Ellipse", lambda: self._set_mode("draw_ellipse"),
                                  "Draw the region as an ellipse; Shift for a circle   (E)",
                                  True, grp)
        self.aPolygon = self._act(bar, "Polygon", lambda: self._set_mode("draw_polygon"),
                                  "Trace the region point by point; Enter or right-click "
                                  "closes it   (P)", True, grp)
        self._draw_actions = (self.aRect, self.aEllipse, self.aPolygon)
        self._draw_sep = bar.addSeparator()
        for a in self._draw_actions:
            a.setVisible(False)
        self._draw_sep.setVisible(False)
        # (zooming lives with the zoom level, bottom right: a short rail fits
        # beside the image in a small editor)
        self.aGrid = self._act(bar, "Grid", self._toggle_grid,
                               "Pixel grid, shown at 8x zoom and above", True)
        self.aGraphics = self._act(bar, "Graphics", self._toggle_graphics,
                                   "Show what the tools found", True)
        self.aGraphics.setChecked(True)
        self.aSearch = self._act(bar, "Search guides", self._toggle_search,
                                 "Show where tools search - rings, calipers, directions", True)
        self.aSearch.setChecked(True)
        for b in bar.findChildren(QToolButton):
            b.setFocusPolicy(Qt.NoFocus)      # the keyboard stays with the image
        # on/off switches are marked quietly; only the tool in use is solid,
        # or three lit buttons read as three tools chosen at once
        for a in (self.aGrid, self.aGraphics, self.aSearch):
            bar.widgetForAction(a).setProperty("toggle", True)

        # ---- the record selector -----------------------------------------
        self.recordBar, rb = _float_bar(self)
        self.record = QComboBox()
        self.record.setMinimumWidth(165)
        self.record.setToolTip("Which image to show - the current input, or a "
                               "record from the last run")
        self.record.currentTextChanged.connect(self._record_picked)
        rb.addWidget(self.record)

        # ---- position and pixel value ------------------------------------
        # The coordinate space the pointer is read in: # is image pixels; a
        # fixture's space is the part's own coordinates -- the same point on the
        # part reads the same wherever the part sits in the image.
        self.statusRow, sl = _float_bar(self)
        sl.setContentsMargins(4, 2, 10, 2)
        sl.setSpacing(10)
        self.sSpace = QComboBox()
        self.sSpace.addItem("#")
        self.sSpace.setToolTip("Coordinate space: # is image pixels; a fixture's space "
                               "is the part's own coordinates")
        self.sSpace.setMinimumWidth(70)
        self._spaces = {"#": None}
        self._last_pixel = None
        self.sSpace.currentTextChanged.connect(lambda _t: self._pixel(self._last_pixel))
        self.sPos = QLabel("")
        self.sValue = QLabel("")
        for lab in (self.sPos, self.sValue):
            lab.setObjectName("floatValue")
        sl.addWidget(self.sSpace)
        sl.addWidget(self.sPos)
        sl.addWidget(self.sValue)

        # ---- zoom --------------------------------------------------------
        self.zoomBar, zl = _float_bar(self)
        zl.setContentsMargins(4, 3, 4, 3)
        self.sZoom = QLabel("100%")
        self.sZoom.setObjectName("floatValue")
        self.sZoom.setMinimumWidth(46)
        self.sZoom.setAlignment(Qt.AlignCenter)
        self.sZoom.setToolTip("Zoom - click to fit")
        self.sZoom.mousePressEvent = lambda _e: self.canvas.fit()
        for name, fn, tip in (("zoom_out", lambda: self.canvas.zoom(0.8), "Zoom out   (Ctrl -)"),
                              (None, None, None),
                              ("zoom_in", lambda: self.canvas.zoom(1.25),
                               "Zoom in   (Ctrl +, or the wheel over the image)"),
                              ("|", None, None),
                              ("fit", lambda: self.canvas.fit(),
                               "Fit the image, and keep fitting on resize   (Ctrl+0)"),
                              ("actual", lambda: self.canvas.zoom_100(),
                               "One image pixel per screen pixel   (Ctrl+1)")):
            if name is None:
                zl.addWidget(self.sZoom)
            elif name == "|":
                zl.addWidget(_sep())
            else:
                zl.addWidget(self._button(name, tip, fn))

        # ---- transform -----------------------------------------------------
        self._build_transform_bar()

        self.canvas.pixelInfo.connect(self._pixel)
        self.canvas.zoomChanged.connect(lambda k: self.sZoom.setText("%d%%" % round(k * 100)))
        self.canvas.modeChanged.connect(self._mode_from_canvas)
        self.canvas.selectionChanged.connect(lambda _ids: self._sync_transform())
        self.canvas.liveGeometry.connect(lambda *_a: self._sync_transform())
        self.canvas.angleChanged.connect(lambda *_a: self._sync_transform())
        self.canvas.lockChanged.connect(lambda *_a: self._sync_transform())
        self.canvas.shapeDrawn.connect(lambda *_a: self._after_draw())
        self._graphics_on = True
        self._last_graphics = []

        self.rail.setVisible(toolbar)
        self.zoomBar.setVisible(toolbar)
        self.recordBar.setVisible(records)
        for w in (self.rail, self.recordBar, self.statusRow, self.zoomBar, self.transformBar):
            w.raise_()

    # ------------------------------------------------------------ building
    @staticmethod
    def _act(bar, text, fn, tip, check=False, grp=None):
        from vision_station.ui import icons
        a = QAction(text, bar)
        on = "#FFFFFF" if grp is not None else None      # white on the solid accent
        a.setIcon(icons.icon(VisionDisplay.ICONS.get(text, ""), on=on))
        a.setProperty("vsIcon", VisionDisplay.ICONS.get(text, ""))
        a.setProperty("vsIconOn", on)
        a.setToolTip("%s - %s" % (text, tip))
        a.setCheckable(check)
        a.triggered.connect(fn)
        if grp is not None:
            grp.addAction(a)
        bar.addAction(a)
        return a

    def _button(self, icon, tip, fn, check=False):
        from vision_station.ui import icons
        b = QToolButton()
        b.setIcon(icons.icon(icon))
        b.setIconSize(QSize(16, 16))
        b.setToolTip(tip)
        b.setAutoRaise(True)
        b.setCheckable(check)
        b.setFocusPolicy(Qt.NoFocus)
        b.setProperty("vsIcon", icon)
        b.clicked.connect(lambda _c=False: fn())
        return b

    def _build_transform_bar(self):
        """Exact position, size and angle for one selected region, as in a
        drawing application's control bar; with several selected, the
        arrange commands."""
        self.transformBar, tl = _float_bar(self)
        tl.setContentsMargins(8, 4, 6, 4)
        tl.setSpacing(4)
        self._fields = {}
        self._single = QWidget()
        sl = QHBoxLayout(self._single)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(4)
        for key, tip in (("X", "Left edge of the region, in image pixels"),
                         ("Y", "Top edge of the region, in image pixels"),
                         ("W", "Width"), ("H", "Height"),
                         ("∠", "Angle in degrees, clockwise")):
            cap = QLabel(key)
            cap.setObjectName("floatCaption")
            if key == "∠":
                f = QDoubleSpinBox()
                f.setRange(-180.0, 180.0)
                f.setDecimals(1)
                f.setSuffix("°")
                f.setWrapping(True)
            else:
                f = QSpinBox()
                f.setRange(-100000 if key in ("X", "Y") else 2, 100000)
            f.setButtonSymbols(QSpinBox.NoButtons)
            f.setKeyboardTracking(False)       # applied on Enter or leaving the field
            f.setAlignment(Qt.AlignRight)
            f.setFixedWidth(64 if key != "∠" else 60)
            f.setToolTip(tip)
            f.editingFinished.connect(self._fields_edited)
            sl.addWidget(cap)
            sl.addWidget(f)
            self._fields[key] = f
        tl.addWidget(self._single)
        self._rot_buttons = []
        self._rot_sep = _sep()
        tl.addWidget(self._rot_sep)
        for icon, tip, fn in (("rotate_ccw", "Rotate 90° anticlockwise   ([ turns 1°, "
                                             "Shift+[ 15°)", lambda: self.canvas.rotate_selection(-90)),
                              ("rotate_cw", "Rotate 90° clockwise   (] turns 1°, "
                                            "Shift+] 15°)", lambda: self.canvas.rotate_selection(90)),
                              ("rotate_zero", "Straighten to 0°",
                               lambda: self.canvas.rotate_selection(0, absolute=True))):
            b = self._button(icon, tip, fn)
            tl.addWidget(b)
            self._rot_buttons.append(b)
        self._multi = QWidget()
        ml = QHBoxLayout(self._multi)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(2)
        ml.addWidget(_sep())
        self._arrange_buttons = {}
        for op, tip in (("align_left", "Align left edges"),
                        ("align_hcenter", "Align centres horizontally"),
                        ("align_right", "Align right edges"),
                        ("align_top", "Align top edges"),
                        ("align_vcenter", "Align centres vertically"),
                        ("align_bottom", "Align bottom edges"),
                        ("distribute_h", "Space evenly, left to right (three or more)"),
                        ("distribute_v", "Space evenly, top to bottom (three or more)"),
                        ("match_width", "Make them all as wide as the widest"),
                        ("match_height", "Make them all as tall as the tallest")):
            b = self._button(op, tip, lambda op=op: self.canvas.arrange(op))
            ml.addWidget(b)
            self._arrange_buttons[op] = b
        tl.addWidget(self._multi)
        tl.addWidget(_sep())
        self._lock = self._button("lock", "Lock the position, so a stray drag cannot move "
                                          "it", self._toggle_lock, check=True)
        tl.addWidget(self._lock)
        self._single_wanted = False
        self.transformBar.hide()

    # ------------------------------------------------------------ layout
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place()

    def showEvent(self, e):
        super().showEvent(e)
        self._place()

    def _place(self):
        """The bars sit in a band round the image, never over it: on a small
        editor they covered its left edge and top, and the transform bar ran
        off the right-hand side with half its buttons out of reach."""
        W, H = self.width(), self.height()
        for bar in (self.rail, self.recordBar, self.statusRow, self.zoomBar):
            bar.adjustSize()
        r, rb, s, z, t = self.rail, self.recordBar, self.statusRow, self.zoomBar, self.transformBar
        rail_on = not r.isHidden()
        top_on = not rb.isHidden() or (self._toolbar and self._has_selection())
        bottom_on = not s.isHidden() or not z.isHidden()
        gap = 6
        left = r.sizeHint().width() + 2 * gap if rail_on else 0
        top_h = max(rb.sizeHint().height() if not rb.isHidden() else 0, 38)
        top = top_h + 2 * gap if top_on else 0
        bottom = max(s.sizeHint().height() if not s.isHidden() else 0,
                     z.sizeHint().height() if not z.isHidden() else 0) + 2 * gap if bottom_on else 0
        self.canvas.set_chrome_margins(left, top, 0, bottom)
        r.move(gap, top + gap if top else gap)
        rb.move(W - rb.width() - gap, gap)
        s.move(left + gap if rail_on else gap, H - s.height() - gap)
        z.move(W - z.width() - gap, H - z.height() - gap)
        # the transform bar in the top band, between the rail and the record
        # selector -- whole if it fits, its arrange buttons only if not (the
        # exact numbers are in the tool's Region settings too), else not at all
        if t.isHidden():
            return
        x0 = gap + (left if rail_on else 0)
        x1 = (rb.x() - gap) if not rb.isHidden() else W - gap
        self._single.setVisible(self._single_wanted)
        for w in self._rot_buttons + [self._rot_sep]:
            w.setVisible(True)
        t.adjustSize()
        if t.width() > x1 - x0 and self._single_wanted:
            self._single.setVisible(False)
            t.adjustSize()
        if t.width() > x1 - x0 and self._multi.isVisible():
            # aligning several matters more here than turning them, which
            # [ and ] and the right-click menu still do
            for w in self._rot_buttons + [self._rot_sep]:
                w.setVisible(False)
            t.adjustSize()
        if t.width() > x1 - x0:
            t.hide()
            return
        t.move(x0 + max(0, (x1 - x0 - t.width()) // 2), gap)

    def _has_selection(self):
        return bool(self.canvas.selected_ids())

    # ------------------------------------------------------------ modes
    def _set_mode(self, mode):
        self.canvas.set_mode(mode)
        self.canvas.setFocus()

    def _mode_from_canvas(self, mode):
        a = {"select": self.aPointer, "hand": self.aPan, "draw_rect": self.aRect,
             "draw_ellipse": self.aEllipse, "draw_polygon": self.aPolygon}.get(mode)
        if a is not None and a.isVisible() or mode in ("select", "hand"):
            (a or self.aPointer).setChecked(True)

    def enable_drawing(self, on=True):
        """Show the drawing tools: for a window whose drawn shape becomes a
        tool's region. R, E and P choose them from the keyboard."""
        for a in self._draw_actions:
            a.setVisible(on)
        self._draw_sep.setVisible(on)
        self.canvas.draw_keys = bool(on)
        self._place()

    def _after_draw(self):
        """One region drawn: back to the pointer to adjust it."""
        self.canvas.set_mode("select")
        self.aPointer.setChecked(True)

    # ------------------------------------------------------------ transform bar
    def _selected_regions(self):
        return [self.canvas.regions[i] for i in self.canvas.selected_ids()
                if isinstance(self.canvas.regions.get(i), RegionItem)]

    def _sync_transform(self):
        if not self._toolbar:
            return
        items = self._selected_regions()
        editable = [it for it in items if it.flags() & it.GraphicsItemFlag.ItemIsMovable]
        if not editable:
            self.transformBar.hide()
            self._place()
            return
        one = len(items) == 1
        self._single_wanted = one
        self._multi.setVisible(len(editable) > 1)
        locked = all(it.locked for it in items)
        self._lock.setChecked(locked)
        self._lock.setToolTip("Unlock - let it be moved again" if locked else
                              "Lock the position, so a stray drag cannot move it")
        from vision_station.ui import icons
        self._lock.setIcon(icons.icon("lock" if locked else "unlock"))
        self._lock.setProperty("vsIcon", "lock" if locked else "unlock")
        for b in self._rot_buttons:
            b.setEnabled(not locked)
        n = len([it for it in editable if not it.locked])
        for op, b in self._arrange_buttons.items():
            b.setEnabled(n >= (3 if op.startswith("distribute") else 2))
        if one:
            it = items[0]
            r = it.roi()
            for key, v in zip(("X", "Y", "W", "H"), r):
                f = self._fields[key]
                if not f.hasFocus():
                    f.blockSignals(True)
                    f.setValue(int(v))
                    f.blockSignals(False)
                f.setEnabled(not it.locked)
            a = self._fields["∠"]
            if not a.hasFocus():
                a.blockSignals(True)
                a.setValue(it.signed_angle())
                a.blockSignals(False)
            a.setEnabled(not it.locked and it.region_shape != "polygon")
        self.transformBar.show()
        self.transformBar.adjustSize()
        self._place()

    def _fields_edited(self):
        items = self._selected_regions()
        if len(items) != 1 or items[0].locked:
            return
        it = items[0]
        roi = [self._fields[k].value() for k in ("X", "Y", "W", "H")]
        ang = float(self._fields["∠"].value())
        if roi == it.roi() and abs(ang - it.signed_angle()) < 1e-6:
            return
        self.canvas.editBegan.emit("resize region")
        if roi != it.roi():
            it.set_roi(roi)
            self.canvas.liveGeometry.emit(it.tool_id, it.roi())
            self.canvas.geometryCommitted.emit(it.tool_id, it.roi())
        if abs(ang - it.signed_angle()) > 1e-6:
            it.set_angle(ang)
            self.canvas.angleChanged.emit(it.tool_id, it.angle)

    def _toggle_lock(self):
        ids = [it.tool_id for it in self._selected_regions()]
        self.canvas.set_locked(ids, self._lock.isChecked())

    # ------------------------------------------------------------ records
    def set_records(self, records, prefer=None):
        """records: ordered {name: (image, graphics list)}. Keeps the chosen
        record when it still exists, as VisionPro does between runs."""
        self._records = dict(records)
        cur = self.record.currentText()
        self.record.blockSignals(True)
        self.record.clear()
        self.record.addItems(list(self._records))
        want = prefer if prefer in self._records else (
            cur if cur in self._records else (next(iter(self._records), "")))
        self.record.setCurrentText(want)
        self.record.blockSignals(False)
        self._show(want)
        self.recordBar.adjustSize()
        self._place()

    def current_record(self):
        return self.record.currentText()

    def set_placeholder(self, text):
        """What an empty display says, in its middle."""
        self.canvas.set_placeholder(text)

    def _record_picked(self, name):
        self._show(name)
        self.recordChanged.emit(name)

    def _show(self, name):
        img, graphics = self._records.get(name, (None, []))
        # With no image the display is cleared, not left showing the previous
        # one: after Reset Counts the queue's preview still showed a run that
        # was no longer in the queue.
        self.canvas.set_image(img, keep_view=True)
        self._last_graphics = graphics or []
        self.canvas.draw_results(self._last_graphics if self._graphics_on else [])

    # ------------------------------------------------------------ toggles
    def _toggle_grid(self):
        self.canvas.show_grid = self.aGrid.isChecked()
        self.canvas.viewport().update()

    def _toggle_graphics(self):
        self._graphics_on = self.aGraphics.isChecked()
        self.canvas.draw_results(self._last_graphics if self._graphics_on else [])

    def _toggle_search(self):
        self.canvas.show_search = self.aSearch.isChecked()
        self.canvas.draw_results(self._last_graphics if self._graphics_on else [])

    def set_spaces(self, spaces):
        """spaces: {name: 2x3 matrix reference->image}. '#' is always there."""
        import numpy as np
        cur = self.sSpace.currentText()
        self._spaces = {"#": None}
        for name, M in (spaces or {}).items():
            if M is not None:
                self._spaces[name] = np.vstack([np.array(M, float), [0, 0, 1]])
        self.sSpace.blockSignals(True)
        self.sSpace.clear()
        self.sSpace.addItems(list(self._spaces))
        self.sSpace.setCurrentText(cur if cur in self._spaces else "#")
        self.sSpace.blockSignals(False)

    def to_space(self, x, y, space=None):
        """Image pixel (x, y) in the given coordinate space (default: the one
        shown). In a fixture's space it is where that point lies on the part."""
        import numpy as np
        M = self._spaces.get(space or self.sSpace.currentText())
        if M is None:
            return float(x), float(y)
        p = np.linalg.inv(M) @ np.array([x, y, 1.0])
        return float(p[0]), float(p[1])

    def _pixel(self, info):
        self._last_pixel = info
        if info is None:
            self.sPos.setText("x -   y -")
            self.sValue.setText("")
        else:
            x, y, val = info
            if self.sSpace.currentText() == "#":
                self.sPos.setText("x %d   y %d" % (x, y))
            else:
                fx, fy = self.to_space(x + 0.5, y + 0.5)
                self.sPos.setText("x %.1f   y %.1f" % (fx, fy))
            self.sValue.setText(val)
        if self.statusRow.isVisible():
            self.statusRow.adjustSize()
