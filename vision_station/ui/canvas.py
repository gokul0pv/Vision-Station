"""Image canvas with direct manipulation.

Every region is a live object on the canvas, not just the selected one. Click any
region to select it, shift-click to add, drag a marquee on empty space to select
several, then drag them together. Alt-drag duplicates. Arrow keys nudge, shift for
ten pixels. Edges snap to other regions and to the image centre with pink guides.

Panning: hold space and drag, press H for the hand tool, or middle-drag. Left-drag
on empty canvas draws a marquee, matching how XD and Figma behave.
"""

import math

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QCursor, QFont, QFontMetricsF, QImage, QPainter,
                           QPainterPath, QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import (QGraphicsItem, QGraphicsPixmapItem, QGraphicsPolygonItem,
                               QGraphicsRectItem, QGraphicsScene, QGraphicsView, QWidget)

# Result colours come from core/annotate.py, so the screen, saved result
# pictures and a line PC's pictures agree: blue OK, red NG, orange warnings.
from vision_station.core.annotate import STYLE as _ANN

PASS = QColor(_ANN["ok"])
FAIL = QColor(_ANN["ng"])
WARN = QColor(_ANN["warn"])
# Selection is violet, as current design tools mark what is being edited:
# blue means OK, red NG and orange a warning, so the selection takes the one
# colour none of the results use. (It was amber, too close to the warnings.)
SEL = QColor("#7C5CFF")
RESULT_W = float(_ANN["line_screen_px"])
IDLE = QColor("#9AA7B4")
GUIDE = QColor("#FF3B9A")
SEARCH = QColor("#38BDF8")
# "search" marks where a tool looks -- a Find Circle's search ring, Find Line's
# calipers -- drawn thin and dashed so it never reads as something found.
STATUS_COLOR = {"pass": PASS, "fail": FAIL, "warn": WARN, "error": WARN, "search": SEARCH}


def _follow_look():
    """Settings > Results changed the OK / NG colours or line thickness. The
    colours are changed in place, so every module holding PASS / FAIL (and
    STATUS_COLOR) sees them; RESULT_W is read at each paint."""
    global RESULT_W
    PASS.setRgb(*QColor(_ANN["ok"]).getRgb()[:3])
    FAIL.setRgb(*QColor(_ANN["ng"]).getRgb()[:3])
    RESULT_W = float(_ANN["line_screen_px"])
    for c in list(_SHOWING):                  # results on screen now: redrawn at once
        try:
            c.draw_results(c._drawn_tools)
            c.viewport().update()
        except RuntimeError:                  # its window has been closed
            _SHOWING.discard(c)


import weakref  # noqa: E402
_SHOWING = weakref.WeakSet()                  # canvases showing results


from vision_station.core import annotate as _annotate_look  # noqa: E402
_annotate_look.on_look_changed(_follow_look)

SNAP = 7.0           # snapping distance, in screen pixels
HANDLE = 4.5         # half a resize handle, in screen pixels
ROT_ZONE = 18.0      # how far outside a corner the pointer rotates, in screen pixels

# Handle directions in the region's own frame, for cursors that follow a
# rotated region the way Illustrator's do.
_HANDLE_DIR = {"mr": 0, "br": 45, "bm": 90, "bl": 135, "ml": 180, "tl": 225,
               "tm": 270, "tr": 315}
_OPPOSITE = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl",
             "tm": "bm", "bm": "tm", "ml": "mr", "mr": "ml"}


def _resize_cursor(deg):
    """The double arrow nearest a direction, in degrees on screen."""
    d = round((deg % 180) / 45.0) % 4
    return (Qt.SizeHorCursor, Qt.SizeFDiagCursor, Qt.SizeVerCursor, Qt.SizeBDiagCursor)[d]


_rot_cursor = None


def rotate_cursor():
    """A curved double arrow: the rotate cursor of every drawing program. Qt
    has none of its own, and the cross that stood in for it said 'precise
    point', not 'turn'."""
    global _rot_cursor
    if _rot_cursor is None:
        from PySide6.QtGui import QCursor
        pm = QPixmap(26, 26)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.arcMoveTo(QRectF(5, 5, 16, 16), 200)
        path.arcTo(QRectF(5, 5, 16, 16), 200, -220)
        for width, colour in ((4.2, QColor("#FFFFFF")), (1.8, QColor("#111111"))):
            pen = QPen(colour, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            a, b = path.pointAtPercent(0.0), path.pointAtPercent(1.0)
            p.setBrush(colour)
            for tip, back in ((a, path.pointAtPercent(0.12)), (b, path.pointAtPercent(0.88))):
                v = QPointF(tip.x() - back.x(), tip.y() - back.y())
                n = math.hypot(v.x(), v.y()) or 1.0
                v = QPointF(v.x() / n, v.y() / n)
                q = QPointF(-v.y(), v.x())
                s = 3.6 if width > 2 else 2.6
                p.drawPolygon(QPolygonF([QPointF(tip.x() + v.x() * s, tip.y() + v.y() * s),
                                         QPointF(tip.x() + q.x() * s, tip.y() + q.y() * s),
                                         QPointF(tip.x() - q.x() * s, tip.y() - q.y() * s)]))
        p.end()
        _rot_cursor = QCursor(pm, 13, 13)
    return _rot_cursor


def QApplication_mods():
    """The modifier keys held right now. Mouse-move events do not always carry
    a key pressed after the button went down -- Shift pressed mid-drag to snap
    an angle was missed."""
    from PySide6.QtGui import QGuiApplication
    # Qt's own record of the keys, from the key events it has seen, and the
    # keyboard's state asked of the system: either may lag the other
    return QGuiApplication.keyboardModifiers() | QGuiApplication.queryKeyboardModifiers()


def _rot(v, deg):
    """A vector turned by deg degrees (screen convention: y down, clockwise)."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return QPointF(v.x() * c - v.y() * s, v.x() * s + v.y() * c)


def ndarray_to_qpixmap(img):
    if img is None:
        return QPixmap()
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    h, w = img.shape[:2]
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())


def _show_enabled(item, enabled):
    """A disabled tool's region is drawn faded, not hidden. Hidden, Qt
    deselected it, so unticking "Run this tool" emptied the property panel and
    the tool could not be switched back on without finding it again -- and
    nobody could see where a disabled tool sat."""
    item.setVisible(True)
    item.setOpacity(1.0 if enabled else 0.35)
    item.setToolTip("" if enabled else "%s - not run (switched off)" % item.tool_id)


MANY_SELECTED = 3      # more boxes selected than this: no name and size on each


def show_label(item):
    """Whether a region's name tag is drawn, for every region shape.

    Rectangles and polygons each had their own copy of this rule, and they had
    drifted: polygons ignored the labelling layer that rectangles respected."""
    pol = getattr(item.canvas, "label_policy", "smart")
    if pol == "all":
        return True
    if item.isSelected() and getattr(item.canvas, "_n_selected", 1) > MANY_SELECTED:
        # thirteen boxes selected at once each with its name and size was a
        # pile of labels over the parts; the list and the group bar name them
        return item.status in ("fail", "error", "warn")
    if pol == "none":
        return item.isSelected()
    return (item.isSelected() or getattr(item, "layer", None) == "labels"
            or item.status in ("fail", "error", "warn"))


def ink_on(col):
    """White or near-black, whichever has the higher WCAG contrast on a fill
    of this colour. A plain brightness cut-off put white on the selection
    blue, at 2.5:1; the contrast ratio picks black there, at 8.4:1."""
    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    lum = (0.2126 * lin(col.redF()) + 0.7152 * lin(col.greenF())
           + 0.0722 * lin(col.blueF()))
    on_white = 1.05 / (lum + 0.05)
    on_black = (lum + 0.05) / 0.05
    return QColor("#111316") if on_black > on_white else QColor("#FFFFFF")


TAG_MIN_SCREEN_PX = 9.0      # a result tag never shrinks below this on screen
TAG_MAX_SCREEN_PX = 24.0     # ... nor grows beyond this, however far zoomed in


def result_tag_height(box_h, picture_long, floor=0.0):
    """One line of a result tag, in image pixels: a third of its box, kept
    between 1/220 and 1/60 of the picture's long side. In image pixels, so
    it grows and shrinks with the zoom as its box does -- down to `floor`
    (the image pixels that make TAG_MIN_SCREEN_PX at the current zoom),
    below which it would no longer be readable."""
    ceil = None
    if isinstance(floor, tuple):             # (smallest, largest) at this zoom
        floor, ceil = floor
    lo, hi = picture_long / 220.0, picture_long / 60.0
    h = max(floor, lo, min(hi, box_h * 0.32))
    if ceil:
        # Never taller than this on screen: zoomed in on a few parts a tag a
        # third of its box was 80 px of "OK" covering the board.
        h = max(floor, min(h, ceil))
    return h


def result_tag_rect(rect, lines, picture_long, floor=0.0):
    """Where the tag for a box goes: on its top-left corner, sitting on the
    top edge, as wide as its text."""
    th = result_tag_height(rect.height(), picture_long, floor)
    f = QFont()
    f.setPixelSize(max(1, int(round(th * 0.70))))
    f.setWeight(QFont.DemiBold)
    fm = QFontMetricsF(f)
    pad = th * 0.25
    w = max(fm.horizontalAdvance(t) for t in lines) + 2 * pad
    h = th * len(lines)
    return QRectF(rect.left(), rect.top() - h, w, h), f, th, pad


def paint_result_tag(p, rect, lines, col, picture_long, lift=0.0, floor=0.0):
    """A check's result in words, attached to its box: a tag in the result's
    colour on the box's top-left corner (core/annotate.py has the words).
    lift raises it above the tags of other checks on the same corner.
    Returns the tag's rectangle."""
    box, f, th, pad = result_tag_rect(rect, lines, picture_long, floor)
    box.translate(0, -lift)
    p.save()
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(Qt.NoPen)
    fill = QColor(col)
    fill.setAlpha(240)
    p.setBrush(QBrush(fill))
    p.drawRoundedRect(box, th * 0.18, th * 0.18)
    p.setFont(f)
    p.setPen(QPen(ink_on(col), 0))
    for i, t in enumerate(lines):
        p.drawText(QRectF(box.left() + pad, box.top() + i * th, box.width() - pad, th),
                   Qt.AlignVCenter | Qt.AlignLeft, t)
    p.restore()
    return box


class ResultTag(QGraphicsItem):
    """A result tag for a check whose region is not on the canvas as an
    editable item -- the run view: drawn on the corner of its outline, in
    image pixels so it scales with the zoom."""

    def __init__(self, rect, lines, col, picture_long):
        super().__init__()
        self._rect, self._lines, self._col = QRectF(rect), list(lines), col
        self._long = picture_long
        self._lift = 0.0
        self._floor = 0.0
        self.setAcceptedMouseButtons(Qt.NoButton)

    def set_caption_floor(self, floor):
        if floor != self._floor:
            self.prepareGeometryChange()
            self._floor = floor

    def box_rect(self):
        return self._rect

    def set_caption_lift(self, lift):
        if lift != self._lift:
            self.prepareGeometryChange()
            self._lift = lift

    def boundingRect(self):
        r = result_tag_rect(self._rect, self._lines, self._long, self._floor)[0]
        return r.translated(0, -self._lift).adjusted(-2, -2, 2, 2)

    def paint(self, p, opt, widget=None):
        paint_result_tag(p, self._rect, self._lines, self._col, self._long, self._lift,
                         self._floor)


def draw_name_tag(p, item, rect, col, s):
    """The tag above a region with its caption: a rounded pill in the
    region's own colour, as design tools label a selection -- it was dark
    monospace-looking text on a black box, which read as noise over a busy
    part. Returns the font metrics and tag height, which the rectangle reuses
    for its size readout. Polygons used to show the bare tool id, never a
    custom caption."""
    f = QFont("Segoe UI")
    f.setPointSizeF(max(1.0, 8.0 * s))
    f.setWeight(QFont.DemiBold)
    p.setFont(f)
    fm = p.fontMetrics()
    cap = getattr(item, "text", None) or item.tool_id
    tw = fm.horizontalAdvance(cap) + 12 * s
    th = fm.height() + 3 * s
    box = QRectF(rect.left(), rect.top() - th - 4 * s, tw, th)
    p.save()
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(Qt.NoPen)
    fill = QColor(col)
    fill.setAlpha(235)
    p.setBrush(QBrush(fill))
    p.drawRoundedRect(box, th / 2.0, th / 2.0)
    p.setPen(QPen(ink_on(col), 0))
    p.drawText(box, Qt.AlignCenter, cap)
    p.restore()
    return fm, th


class RegionItem(QGraphicsRectItem):
    """One tool's region. The rect is anchored at 0,0 and the item position
    carries the location, which keeps multi-select dragging simple."""

    def __init__(self, tool_id, roi, canvas, colour=None, text=None, layer="tools",
                 angle=0.0, shape="rect"):
        # NB: the region shape is stored as `region_shape`, never `shape`.
        # QGraphicsItem.shape() is the method Qt calls to hit-test a click, and
        # an attribute of that name shadows it. Doing so broke clicking on a
        # region, marquee selection and contains() -- the whole of canvas
        # interaction -- while every unit test kept passing, because none of
        # them generated a real mouse event.
        x, y, w, h = roi
        super().__init__(QRectF(0, 0, max(2, w), max(2, h)))
        self.angle = float(angle or 0.0)
        self.region_shape = shape or "rect"
        self._rotating = False
        self.tool_id = tool_id
        self.canvas = canvas
        self.status = None
        self.colour = colour
        self.text = text
        self.layer = layer
        self.setPos(x, y)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable |
                      QGraphicsItem.ItemIsFocusable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self._grab = None
        self._origin = None
        self._origin_pos = None
        self._dup_armed = False
        self._shift_select = False
        self._hover = False
        self.locked = False
        self._sync_rotation()

    def _sync_rotation(self):
        self.setTransformOriginPoint(self.rect().center())
        self.setRotation(self.angle)

    def set_angle(self, deg):
        self.angle = float(deg) % 360.0
        self._sync_rotation()
        self.update()

    def local_rect(self):
        """The item's own outline in item coordinates. Shared by both region
        types so geometry code never has to ask which one it is holding."""
        return self.rect()

    def roi(self):
        r = self.rect()
        return [int(round(self.x())), int(round(self.y())),
                int(round(r.width())), int(round(r.height()))]

    def set_shape(self, shape):
        self.region_shape = shape or "rect"
        self.update()

    def set_roi(self, roi, angle=None):
        x, y, w, h = roi
        self.prepareGeometryChange()
        self.setPos(x, y)
        self.setRect(QRectF(0, 0, max(2, w), max(2, h)))
        if angle is not None:
            self.angle = float(angle or 0.0)
        self._sync_rotation()

    def _s(self):
        v = self.scene().views() if self.scene() else None
        return 1.0 / max(0.02, v[0].transform().m11()) if v else 1.0

    def _handles(self):
        """Resize handles.

        An ellipse is edited the way VisionPro edits one: by its centre and its
        two radii, with a handle at each end of each axis -- points that lie on
        the ellipse itself. Corner handles on an ellipse's bounding box sit in
        empty space away from the shape and imply a box that is not there."""
        r = self.rect()
        cx, cy = r.center().x(), r.center().y()
        if self.region_shape == "ellipse":
            return {"tm": QPointF(cx, r.top()), "bm": QPointF(cx, r.bottom()),
                    "ml": QPointF(r.left(), cy), "mr": QPointF(r.right(), cy)}
        return {"tl": QPointF(r.left(), r.top()), "tm": QPointF(cx, r.top()),
                "tr": QPointF(r.right(), r.top()), "ml": QPointF(r.left(), cy),
                "mr": QPointF(r.right(), cy), "bl": QPointF(r.left(), r.bottom()),
                "bm": QPointF(cx, r.bottom()), "br": QPointF(r.right(), r.bottom())}

    def _grip(self):
        r = self.rect()
        return QPointF(r.center().x(), r.top() - 24 * self._s())

    def set_caption(self, lines, picture_long):
        """The check's result in words, painted on this box's corner -- it
        moves and scales with the box. None clears it."""
        lines = list(lines) if lines else None
        if lines == getattr(self, "caption", None) and \
                picture_long == getattr(self, "_cap_long", 0):
            return
        self.prepareGeometryChange()
        self.caption, self._cap_long = lines, picture_long

    def set_caption_lift(self, lift):
        if lift != getattr(self, "_cap_lift", 0.0):
            self.prepareGeometryChange()
            self._cap_lift = lift

    def set_caption_floor(self, floor):
        if floor != getattr(self, "_cap_floor", 0.0):
            self.prepareGeometryChange()
            self._cap_floor = floor

    def _caption_extent(self, rect):
        if not getattr(self, "caption", None):
            return 0.0, 0.0
        box = result_tag_rect(rect, self.caption, self._cap_long,
                              getattr(self, "_cap_floor", 0.0))[0]
        return box.height() + getattr(self, "_cap_lift", 0.0), max(0.0, box.right() - rect.right())

    def boundingRect(self):
        m = (ROT_ZONE + 6) * self._s()
        ch, cw = self._caption_extent(self.rect())
        return self.rect().adjusted(-m, -m - 34 * self._s() - ch, m + cw, m + 12 * self._s())

    def shape(self):
        """What a click hits. A selected region also owns its handles, its
        rotate grip and the rotate zones just outside its corners -- they lie
        outside the rectangle, and with the rectangle alone as the hit area a
        click on the grip went through to the canvas: rotating by the grip had
        never worked."""
        path = QPainterPath()
        r = self.rect()
        if self.isSelected() and not self.locked:
            s = self._s()
            m = (ROT_ZONE if self.layer == "tools" else HANDLE * 2.6) * s
            path.addRect(r.adjusted(-m, -m, m, m))
            if self.layer == "tools":
                g = self._grip()
                t = HANDLE * 3.0 * s
                path.addRect(QRectF(g.x() - t, g.y() - t, 2 * t, 2 * t))
                path.addRect(QRectF(r.center().x() - t / 2, g.y(), t, r.top() - g.y()))
            path.setFillRule(Qt.WindingFill)
        elif self.region_shape == "ellipse":
            path.addEllipse(r)
        else:
            path.addRect(r)
        return path

    # -- painting --------------------------------------------------------
    def paint(self, p, opt, widget=None):
        s = self._s()
        sel = self.isSelected()
        base = self.colour or STATUS_COLOR.get(self.status, IDLE)
        col = SEL if sel else base
        p.setRenderHint(QPainter.Antialiasing, True)

        hover = getattr(self, "_hover", False) and not sel
        # With a result the outline is the result -- blue OK, red NG -- drawn
        # firmly enough to read at a glance; without one, thin and dashed.
        width = RESULT_W if (self.status or sel or hover) else 1.25
        pen = QPen(col, width * s)
        pen.setStyle(Qt.SolidLine if (sel or hover or self.status) else Qt.DashLine)
        p.setPen(pen)
        alpha = 30 if sel else 22 if hover else 10
        if sel:
            # a soft glow round the selected region, so it stands out of a
            # board full of outlines
            p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 60), (width + 5.0) * s))
            p.setBrush(Qt.NoBrush)
            if self.region_shape == "ellipse":
                p.drawEllipse(self.rect())
            else:
                p.drawRect(self.rect())
            p.setPen(pen)
        p.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), alpha)))
        if self.region_shape == "ellipse":
            p.drawEllipse(self.rect())
            if sel:
                # faint axes, so it is obvious what each handle changes
                gp = QPen(QColor(col.red(), col.green(), col.blue(), 110), 1.0 * s, Qt.DotLine)
                p.setPen(gp)
                p.setBrush(Qt.NoBrush)
                rr = self.rect()
                p.drawLine(QPointF(rr.left(), rr.center().y()),
                           QPointF(rr.right(), rr.center().y()))
                p.drawLine(QPointF(rr.center().x(), rr.top()),
                           QPointF(rr.center().x(), rr.bottom()))
                p.setPen(pen)
        else:
            p.drawRect(self.rect())

        if sel and self.locked:
            self._paint_lock(p, s)
        elif sel:
            hs = HANDLE * s
            p.setBrush(QBrush(QColor("#FFFFFF")))
            p.setPen(QPen(SEL, 1.5 * s))
            if self.region_shape != "ellipse":
                # round corner handles and pill-shaped edge handles, each with
                # a soft shadow so it stands off any picture
                for key, pt in self._handles().items():
                    if key in ("tm", "bm", "ml", "mr"):
                        long_, thin = hs * 2.9, hs * 1.45
                        horiz = key in ("tm", "bm")
                        w, h = (long_, thin) if horiz else (thin, long_)
                        box = QRectF(pt.x() - w / 2, pt.y() - h / 2, w, h)
                        rad = thin / 2
                    else:
                        rr = hs * 1.25
                        box = QRectF(pt.x() - rr, pt.y() - rr, rr * 2, rr * 2)
                        rad = rr
                    p.setPen(Qt.NoPen)
                    p.setBrush(QBrush(QColor(0, 0, 0, 55)))
                    p.drawRoundedRect(box.translated(0, 0.9 * s).adjusted(-0.6 * s, -0.6 * s,
                                                                         0.6 * s, 0.6 * s),
                                      rad + 0.6 * s, rad + 0.6 * s)
                    p.setBrush(QBrush(QColor("#FFFFFF")))
                    p.setPen(QPen(SEL, 1.6 * s))
                    p.drawRoundedRect(box, rad, rad)
            elif self.region_shape == "ellipse":
                # round handles on the curve, and a centre mark -- the ellipse
                # is defined by that centre and stays put while it is resized
                for pt in self._handles().values():
                    p.drawEllipse(pt, hs * 1.15, hs * 1.15)
                c = self.rect().center()
                p.drawLine(QPointF(c.x() - hs * 1.6, c.y()), QPointF(c.x() + hs * 1.6, c.y()))
                p.drawLine(QPointF(c.x(), c.y() - hs * 1.6), QPointF(c.x(), c.y() + hs * 1.6))
            if self.layer == "tools":
                grip = self._grip()
                p.setPen(QPen(SEL, 1.5 * s))
                p.drawLine(QPointF(self.rect().center().x(), self.rect().top()), grip)
                p.setBrush(QBrush(QColor("#FFFFFF")))
                p.drawEllipse(grip, hs * 1.3, hs * 1.3)

        r = self.rect()
        top = r.top()
        if getattr(self, "caption", None):
            # the result, attached to the box -- in the result's colour even
            # while the box is selected
            top = paint_result_tag(p, r, self.caption, base, self._cap_long,
                                   getattr(self, "_cap_lift", 0.0),
                                   getattr(self, "_cap_floor", 0.0)).top()
        if not self._show_label() or (getattr(self, "caption", None) and not sel):
            return
        p.setRenderHint(QPainter.Antialiasing, False)
        fm, th = draw_name_tag(p, self, QRectF(r.left(), top, r.width(), r.height()), col, s)

        if sel and getattr(self.canvas, "_n_selected", 1) <= MANY_SELECTED:
            if self.region_shape == "ellipse":
                if abs(r.width() - r.height()) < 1.0:
                    dims = "r %d" % int(round(r.width() / 2))
                else:
                    dims = "rx %d · ry %d" % (int(round(r.width() / 2)),
                                                 int(round(r.height() / 2)))
            else:
                dims = f"{int(round(r.width()))} × {int(round(r.height()))}"
            if abs(self.angle) > 1e-6:
                dims += "  ∠ %.1f°" % self.signed_angle()
            dw = fm.horizontalAdvance(dims) + 16 * s
            box = QRectF(r.center().x() - dw / 2, r.bottom() + 8 * s, dw, th)
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(SEL))
            p.drawRoundedRect(box, th / 2.0, th / 2.0)
            p.setPen(QPen(ink_on(SEL), 0))
            p.drawText(box, Qt.AlignCenter, dims)

    def _paint_lock(self, p, s):
        """A locked region shows a padlock instead of handles: selectable, so
        its settings can be seen, but not dragged by accident."""
        r = self.rect()
        c = QPointF(r.right() - 9 * s, r.top() + 9 * s)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(12, 14, 17, 200)))
        p.drawRoundedRect(QRectF(c.x() - 8 * s, c.y() - 8 * s, 16 * s, 16 * s), 4 * s, 4 * s)
        p.setPen(QPen(QColor("#FFFFFF"), 1.3 * s))
        p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(c.x() - 3 * s, c.y() - 6 * s, 6 * s, 6 * s), 0, 180 * 16)
        p.setBrush(QBrush(QColor("#FFFFFF")))
        p.drawRoundedRect(QRectF(c.x() - 4.5 * s, c.y() - 2 * s, 9 * s, 7 * s), 1.5 * s, 1.5 * s)

    def signed_angle(self):
        """The angle as people read it: -180 to 180, not 0 to 360."""
        a = self.angle % 360.0
        return a - 360.0 if a > 180.0 else a

    def _show_label(self):
        return show_label(self)

    # -- hit testing -----------------------------------------------------
    def _hit(self, pos):
        """What the pointer is over: a resize handle, 'rot' (the grip, or just
        outside a corner, as in Illustrator), or None for the body."""
        if not self.isSelected() or self.locked:
            return None
        s = self._s()
        tol = HANDLE * s * 2.2
        for name, pt in self._handles().items():
            if abs(pos.x() - pt.x()) <= tol and abs(pos.y() - pt.y()) <= tol:
                return name
        if self.layer != "tools":
            return None
        g = self._grip()
        if abs(pos.x() - g.x()) <= tol * 1.6 and abs(pos.y() - g.y()) <= tol * 1.6:
            return "rot"
        r = self.rect()
        if not r.contains(pos):
            zone = ROT_ZONE * s
            for corner in (r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()):
                if math.hypot(pos.x() - corner.x(), pos.y() - corner.y()) <= zone + tol:
                    return "rot"
        return None

    def hoverEnterEvent(self, e):
        self._hover = True
        self.update()
        super().hoverEnterEvent(e)

    def hoverLeaveEvent(self, e):
        self._hover = False
        self.update()
        super().hoverLeaveEvent(e)

    def hoverMoveEvent(self, e):
        h = self._hit(e.pos())
        if h == "rot":
            self.setCursor(rotate_cursor())
        elif h:
            # the arrow follows the region's turn, so it points the way the
            # edge will actually move
            self.setCursor(_resize_cursor(_HANDLE_DIR[h] + self.angle))
        elif self.locked or not (self.flags() & QGraphicsItem.ItemIsMovable):
            self.setCursor(Qt.ArrowCursor)
        else:
            self.setCursor(Qt.SizeAllCursor)
        super().hoverMoveEvent(e)

    # -- dragging --------------------------------------------------------
    def _local_to_scene(self, pt, pos=None, rect=None, angle=None):
        """Where a point of the unrotated box lies in the image, for a given
        position, size and turn (the current ones by default)."""
        pos = self.pos() if pos is None else pos
        rect = self.rect() if rect is None else rect
        angle = self.angle if angle is None else angle
        c = rect.center()
        v = _rot(QPointF(pt.x() - c.x(), pt.y() - c.y()), angle)
        return QPointF(pos.x() + c.x() + v.x(), pos.y() + c.y() + v.y())

    def _apply_box(self, centre, w, h):
        """Set the region from its centre in the image and its size, keeping
        its turn. The stored ROI is the unturned box, turned about its centre."""
        w, h = max(2.0, w), max(2.0, h)
        self.prepareGeometryChange()
        self.setRect(QRectF(0, 0, w, h))
        self.setPos(centre.x() - w / 2.0, centre.y() - h / 2.0)
        self._sync_rotation()

    def mousePressEvent(self, e):
        if self.canvas.mode == "hand":
            e.ignore()
            return

        # Shift adds to the selection. Qt's own multi-select modifier is Ctrl,
        # but every design tool this canvas is modelled on uses Shift, and that
        # is what the shortcuts documentation promises. Ctrl keeps working too,
        # since Qt handles it in the base class.
        if (e.modifiers() & Qt.ShiftModifier) and not self._hit(e.pos()):
            self._grab = None
            if not self.isSelected():
                self.setSelected(True)
                # Qt finishes a plain click's selection on *release*, clearing
                # everything else. Skipping the press alone is not enough; the
                # release has to be skipped too or the extra selection is
                # undone a moment later.
                self._shift_select = True
                e.accept()
                return
            # Already selected: as in Illustrator, Shift-drag moves the
            # selection along one axis, and a Shift-click that does not drag
            # takes it out of the selection -- decided on release.
            self._shift_toggle = True
            self._shift_moved = False
            self._press_mods = e.modifiers()
            if self.locked or not (self.flags() & QGraphicsItem.ItemIsMovable):
                e.accept()
                return
            if self.layer == "tools":
                self.canvas.editBegan.emit("move region")
            QGraphicsRectItem.mousePressEvent(self, e)
            self.canvas.begin_move(e.scenePos(), e.modifiers())
            return

        self._grab = self._hit(e.pos())
        self._origin = QRectF(self.rect())
        self._origin_pos = QPointF(self.pos())
        self._origin_angle = self.angle
        # Modifiers held when the drag starts apply for the whole drag. Move
        # events do not reliably report keys pressed before the button went
        # down, so Alt-to-resize-from-centre and Shift-to-constrain were being
        # dropped part way through a drag.
        self._press_mods = e.modifiers()
        if self.layer == "tools" and not self.locked:
            self.canvas.editBegan.emit("rotate region" if self._grab == "rot"
                                       else "resize region" if self._grab
                                       else "move region")
        if self._grab == "rot":
            self._rotating = True
            c = self._local_to_scene(self._origin.center())
            self._rot_centre = c
            v = e.scenePos() - c
            self._rot_start = math.degrees(math.atan2(v.y(), v.x()))
            e.accept()
            return
        if self._grab:
            # the corner or edge opposite the one grabbed stays where it is in
            # the image -- for a turned region too, which used to jump sideways
            # because the box was resized in its own frame and then re-turned
            # about a centre that had moved
            self._anchor = self._local_to_scene(self._handles_of(self._origin)[_OPPOSITE[self._grab]])
            self._centre0 = self._local_to_scene(self._origin.center())
            e.accept()
            return
        if e.modifiers() & Qt.AltModifier:
            self._dup_armed = True
        super().mousePressEvent(e)
        self.canvas.begin_move(e.scenePos())

    def _handles_of(self, r):
        cx, cy = r.center().x(), r.center().y()
        return {"tl": QPointF(r.left(), r.top()), "tm": QPointF(cx, r.top()),
                "tr": QPointF(r.right(), r.top()), "ml": QPointF(r.left(), cy),
                "mr": QPointF(r.right(), cy), "bl": QPointF(r.left(), r.bottom()),
                "bm": QPointF(cx, r.bottom()), "br": QPointF(r.right(), r.bottom())}

    def mouseMoveEvent(self, e):
        if self._dup_armed:
            self._dup_armed = False
            if self.layer == "tools":
                self.canvas.duplicateRequested.emit(
                    self.canvas.selected_ids() or [self.tool_id])
                return
            self._dup_armed = False
        if self._rotating:
            v = e.scenePos() - self._rot_centre
            now = math.degrees(math.atan2(v.y(), v.x()))
            deg = self._origin_angle + (now - self._rot_start)
            mods = e.modifiers() | QApplication_mods() | getattr(self, "_press_mods", Qt.NoModifier)
            if mods & Qt.ShiftModifier:
                deg = round(deg / 15.0) * 15.0        # the angle itself, in 15 degree steps
            self.set_angle(deg)
            self.canvas.angleChanged.emit(self.tool_id, self.angle)
            self.canvas.show_hud("%.1f°" % self.signed_angle(), e.scenePos())
            e.accept()
            return
        if not self._grab:
            if getattr(self, "_shift_toggle", False):
                self._shift_moved = True
            super().mouseMoveEvent(e)
            if self.layer == "tools":
                self.canvas.liveGeometry.emit(self.tool_id, self.roi())
            r = self.roi()
            self.canvas.show_hud("X %d  Y %d" % (r[0], r[1]), e.scenePos())
            return
        self._resize_to(e.scenePos(), e.modifiers() | getattr(self, "_press_mods", Qt.NoModifier)
                        | QApplication_mods())
        if self.layer == "tools":
            self.canvas.liveGeometry.emit(self.tool_id, self.roi())
        r = self.rect()
        self.canvas.show_hud("W %d  H %d" % (round(r.width()), round(r.height())), e.scenePos())
        e.accept()

    def _resize_to(self, P, mods):
        g = self._grab
        w0, h0 = self._origin.width(), self._origin.height()
        horiz = g in ("ml", "mr", "tl", "tr", "bl", "br")
        vert = g in ("tm", "bm", "tl", "tr", "bl", "br")
        corner = horiz and vert
        # Resize about the centre for an ellipse, always -- it is defined by its
        # centre, and dragging one side must not move it. For a rectangle, Alt
        # does the same, as in every drawing application.
        central = self.region_shape == "ellipse" or bool(mods & Qt.AltModifier)
        if central:
            u = _rot(P - self._centre0, -self._origin_angle)
            hw = max(1.0, abs(u.x())) if horiz else w0 / 2.0
            hh = max(1.0, abs(u.y())) if vert else h0 / 2.0
            if mods & Qt.ShiftModifier:
                if self.region_shape == "ellipse":
                    hw = hh = max(hw if g in ("ml", "mr") else hh, 1.0)   # a circle
                elif corner:
                    k = max(hw / max(1.0, w0 / 2.0), hh / max(1.0, h0 / 2.0))
                    hw, hh = k * w0 / 2.0, k * h0 / 2.0
            self._apply_box(self._centre0, 2 * hw, 2 * hh)
            return
        # in the box's own frame, measured from the anchored corner or edge
        u = _rot(P - self._anchor, -self._origin_angle)
        ux, uy = u.x(), u.y()
        if mods & Qt.ShiftModifier and corner:
            # proportional: the larger of the two stretches wins, signs kept,
            # so the anchored corner stays put whichever corner is dragged
            k = max(abs(ux) / max(1.0, w0), abs(uy) / max(1.0, h0))
            ux = math.copysign(k * w0, ux if ux else (1 if g in ("tr", "br") else -1))
            uy = math.copysign(k * h0, uy if uy else (1 if g in ("bl", "br") else -1))
        x0, x1 = (min(0.0, ux), max(0.0, ux)) if horiz else (-w0 / 2.0, w0 / 2.0)
        y0, y1 = (min(0.0, uy), max(0.0, uy)) if vert else (-h0 / 2.0, h0 / 2.0)
        # an unturned region's moving edges snap to the image and to the
        # other regions, as they do when it is moved; Ctrl turns snapping off
        if abs(self._origin_angle % 360.0) < 1e-6 and not (mods & Qt.ControlModifier):
            ax, ay = self._anchor.x(), self._anchor.y()
            sx = self.canvas.snap_value(ax + (x1 if (horiz and ux >= 0) else x0), "x", self) \
                if horiz else None
            sy = self.canvas.snap_value(ay + (y1 if (vert and uy >= 0) else y0), "y", self) \
                if vert else None
            if sx is not None and not (mods & Qt.ShiftModifier and corner):
                if ux >= 0:
                    x1 = sx - ax
                else:
                    x0 = sx - ax
            if sy is not None and not (mods & Qt.ShiftModifier and corner):
                if uy >= 0:
                    y1 = sy - ay
                else:
                    y0 = sy - ay
            self.canvas.draw_guides([sx] if sx is not None else [], [sy] if sy is not None else [])
        c = _rot(QPointF((x0 + x1) / 2.0, (y0 + y1) / 2.0), self._origin_angle)
        self._apply_box(QPointF(self._anchor.x() + c.x(), self._anchor.y() + c.y()),
                        x1 - x0, y1 - y0)

    def mouseReleaseEvent(self, e):
        self._dup_armed = False
        self.canvas.hide_hud()
        if getattr(self, "_shift_select", False):
            self._shift_select = False
            e.accept()
            return
        if getattr(self, "_shift_toggle", False):
            self._shift_toggle = False
            self.canvas.end_move()
            self.canvas.clear_guides()
            if not self._shift_moved:
                self.setSelected(False)          # a Shift-click: out of the selection
            else:
                self.canvas.commit_selection()
            e.accept()
            return
        if self._rotating:
            self._rotating = False
            self._grab = None
            self.canvas.angleChanged.emit(self.tool_id, self.angle)
            self.canvas.commit_selection()
            e.accept()
            return
        self._grab = None
        super().mouseReleaseEvent(e)
        self.canvas.end_move()
        self.canvas.clear_guides()
        self.canvas.commit_selection()

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene() and not self._grab:
            value = self.canvas.constrain_move(self, value)
        return super().itemChange(change, value)


class PolygonItem(QGraphicsPolygonItem):
    """A region traced round an arbitrary shape.

    Vertices are held in image coordinates rather than relative to a bounding
    box. A polygon drawn round an angled connector has no meaningful box to be
    relative to, and re-deriving one whenever the part moves would accumulate
    rounding error."""

    def __init__(self, tool_id, points, canvas, colour=None):
        super().__init__(QPolygonF([QPointF(*p) for p in points]))
        self.tool_id = tool_id
        self.canvas = canvas
        self.colour = colour
        self.status = None
        self.layer = "tools"
        self.region_shape = "polygon"
        self.angle = 0.0
        self._drag_vertex = None
        self._scale_handle = None
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable |
                      QGraphicsItem.ItemIsFocusable |
                      QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

    # -- geometry --------------------------------------------------------
    def points(self):
        off = self.pos()
        return [[float(p.x() + off.x()), float(p.y() + off.y())]
                for p in self.polygon()]

    def set_points(self, pts):
        self.prepareGeometryChange()
        self.setPos(0, 0)
        self.setPolygon(QPolygonF([QPointF(*p) for p in pts]))

    def shape(self):
        """Hit area. Qt tests clicks against this, and for a polygon it is the
        outline -- so the bounding-box scale handles, which sit outside the
        shape on any non-rectangular region, could never be grabbed. A diamond
        is the obvious case: all four corner handles are in empty space. When
        selected, the hit area widens to the handle box."""
        path = QPainterPath()
        if self.isSelected():
            m = HANDLE * self._s() * 2.6
            path.addRect(self.polygon().boundingRect().adjusted(-m, -m, m, m))
        else:
            path.addPolygon(self.polygon())
            path.closeSubpath()
        return path

    def local_rect(self):
        return self.polygon().boundingRect()

    def roi(self):
        r = self.mapRectToScene(self.polygon().boundingRect())
        return [int(r.x()), int(r.y()), int(r.width()), int(r.height())]

    def _s(self):
        v = self.scene().views() if self.scene() else None
        return 1.0 / max(0.02, v[0].transform().m11()) if v else 1.0

    set_caption = RegionItem.set_caption
    set_caption_lift = RegionItem.set_caption_lift
    set_caption_floor = RegionItem.set_caption_floor
    _caption_extent = RegionItem._caption_extent

    def boundingRect(self):
        m = 16 * self._s()
        bb = self.polygon().boundingRect()
        ch, cw = self._caption_extent(bb)
        return bb.adjusted(-m, -m - 14 * self._s() - ch, m + cw, m)

    # -- painting --------------------------------------------------------
    def paint(self, p, opt, widget=None):
        s = self._s()
        sel = self.isSelected()
        base = self.colour or STATUS_COLOR.get(self.status, IDLE)
        col = SEL if sel else base
        pen = QPen(col, (RESULT_W if (self.status or sel) else 1.25) * s)
        pen.setStyle(Qt.SolidLine if (sel or self.status) else Qt.DashLine)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(pen)
        p.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(),
                                 26 if sel else 12)))
        p.drawPolygon(self.polygon())
        if sel:
            hs = HANDLE * s
            # bounding box, faint, with square corner handles that scale the
            # whole shape
            bb = self.polygon().boundingRect()
            gp = QPen(QColor(SEL.red(), SEL.green(), SEL.blue(), 90), 0, Qt.DotLine)
            gp.setCosmetic(True)
            p.setPen(gp)
            p.setBrush(Qt.NoBrush)
            p.drawRect(bb)
            p.setBrush(QBrush(QColor("#FFFFFF")))
            p.setPen(QPen(SEL, 0))
            for pt in self._box_handles().values():
                p.drawRect(QRectF(pt.x() - hs, pt.y() - hs, hs * 2, hs * 2))
            # round handles on the vertices themselves
            p.setRenderHint(QPainter.Antialiasing, True)
            for pt in self.polygon():
                p.drawEllipse(pt, hs, hs)
            p.setRenderHint(QPainter.Antialiasing, False)
        p.setRenderHint(QPainter.Antialiasing, False)

        bb = self.polygon().boundingRect()
        top = bb.top()
        if getattr(self, "caption", None):
            top = paint_result_tag(p, bb, self.caption, base, self._cap_long,
                                   getattr(self, "_cap_lift", 0.0),
                                   getattr(self, "_cap_floor", 0.0)).top()
        if not show_label(self) or (getattr(self, "caption", None) and not sel):
            return
        draw_name_tag(p, self, QRectF(bb.left(), top, bb.width(), bb.height()), col, s)

    # -- editing ---------------------------------------------------------
    def _vertex_at(self, pos):
        if not self.isSelected():
            return None
        tol = HANDLE * self._s() * 2.6
        for i, pt in enumerate(self.polygon()):
            if abs(pos.x() - pt.x()) <= tol and abs(pos.y() - pt.y()) <= tol:
                return i
        return None

    def _box_handles(self):
        """Corner handles on the bounding box, for scaling the whole shape.

        Editing a traced region one vertex at a time is fine for adjusting a
        corner and useless for making the whole thing slightly bigger, which is
        the common case when a region was drawn a little tight."""
        r = self.polygon().boundingRect()
        return {"tl": r.topLeft(), "tr": r.topRight(),
                "br": r.bottomRight(), "bl": r.bottomLeft()}

    def _box_handle_at(self, pos):
        if not self.isSelected():
            return None
        tol = HANDLE * self._s() * 2.6
        for name, pt in self._box_handles().items():
            if abs(pos.x() - pt.x()) <= tol and abs(pos.y() - pt.y()) <= tol:
                return name
        return None

    def scale_from(self, anchor, cur, start_rect, start_pts):
        """Scale every vertex about the opposite corner of the bounding box."""
        ax, ay = anchor
        # Measured against the corner actually grabbed -- the one opposite the
        # anchor. This used the box's right and bottom edges whatever was
        # grabbed, so for a left or top handle the anchor sat on those edges,
        # the divisor was zero, and the fallback scale of 1.0 meant dragging
        # the top-left handle did nothing at all.
        gx = start_rect.left() if abs(ax - start_rect.right()) < 1e-6 else start_rect.right()
        gy = start_rect.top() if abs(ay - start_rect.bottom()) < 1e-6 else start_rect.bottom()
        sx = (cur.x() - ax) / (gx - ax) if abs(gx - ax) > 1e-6 else 1.0
        sy = (cur.y() - ay) / (gy - ay) if abs(gy - ay) > 1e-6 else 1.0
        sx = max(0.05, abs(sx))
        sy = max(0.05, abs(sy))
        pts = [QPointF(ax + (p.x() - ax) * sx, ay + (p.y() - ay) * sy)
               for p in start_pts]
        self.prepareGeometryChange()
        self.setPolygon(QPolygonF(pts))

    def hoverMoveEvent(self, e):
        h = self._box_handle_at(e.pos())
        if h:
            self.setCursor({"tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                            "tr": Qt.SizeBDiagCursor,
                            "bl": Qt.SizeBDiagCursor}[h])
        elif self._vertex_at(e.pos()) is not None:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.SizeAllCursor)
        super().hoverMoveEvent(e)

    def mousePressEvent(self, e):
        if self.canvas.mode == "hand":
            e.ignore()
            return
        if (e.modifiers() & Qt.ShiftModifier) and self._vertex_at(e.pos()) is None:
            self.setSelected(not self.isSelected())
            self._shift_select = True
            e.accept()
            return
        self._scale_handle = self._box_handle_at(e.pos())
        if self._scale_handle is not None:
            self._start_rect = QRectF(self.polygon().boundingRect())
            self._start_pts = [QPointF(p) for p in self.polygon()]
            self.canvas.editBegan.emit("resize region")
            e.accept()
            return
        self._drag_vertex = self._vertex_at(e.pos())
        if self._drag_vertex is not None:
            self.canvas.editBegan.emit("move vertex")
            e.accept()
            return
        self.canvas.editBegan.emit("move region")
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if getattr(self, "_scale_handle", None) is not None:
            r = self._start_rect
            anchor = {"tl": (r.right(), r.bottom()), "tr": (r.left(), r.bottom()),
                      "br": (r.left(), r.top()), "bl": (r.right(), r.top())
                      }[self._scale_handle]
            self.scale_from(anchor, e.pos(), r, self._start_pts)
            e.accept()
            return
        if self._drag_vertex is not None:
            poly = self.polygon()
            poly[self._drag_vertex] = e.pos()
            self.prepareGeometryChange()
            self.setPolygon(poly)
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if getattr(self, "_shift_select", False):
            self._shift_select = False
            e.accept()
            return
        self._drag_vertex = None
        self._scale_handle = None
        super().mouseReleaseEvent(e)
        self.canvas.polygonCommitted.emit(self.tool_id, self.points())

    def mouseDoubleClickEvent(self, e):
        """Double-click an edge to add a vertex, or a vertex to remove it."""
        poly = self.polygon()
        i = self._vertex_at(e.pos())
        if i is not None and poly.count() > 3:
            poly.remove(i)
        else:
            best, bi = None, 0
            for k in range(poly.count()):
                a, b = poly[k], poly[(k + 1) % poly.count()]
                mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
                d = (mid.x() - e.pos().x()) ** 2 + (mid.y() - e.pos().y()) ** 2
                if best is None or d < best:
                    best, bi = d, k + 1
            poly.insert(bi, e.pos())
        self.prepareGeometryChange()
        self.setPolygon(poly)
        self.canvas.polygonCommitted.emit(self.tool_id, self.points())
        e.accept()


class _RoundCorners(QWidget):
    """The picture's corners, rounded: four small wedges in the page's
    colour painted over the view's corners (a view cannot clip itself to a
    rounded shape, and a mask would leave them jagged). Takes no clicks."""
    RADIUS = 18

    def __init__(self, view):
        super().__init__(view)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def paintEvent(self, _e):
        # the colour of what the picture sits on: a card, or the page
        try:
            from vision_station.ui import theme
            key, w = "bg", self.parentWidget()
            while w is not None:
                if w.objectName() in ("tile", "dockCard", "card", "panel", "formGroup"):
                    key = "layer"
                    break
                w = w.parentWidget()
            col = QColor(theme.color(key))
        except Exception:
            return
        r = QRectF(self.rect())
        outer = QPainterPath()
        outer.addRect(r)
        inner = QPainterPath()
        inner.addRoundedRect(r, self.RADIUS, self.RADIUS)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillPath(outer.subtracted(inner), col)
        p.end()


class ImageCanvas(QGraphicsView):
    geometryCommitted = Signal(str, list)
    liveGeometry = Signal(str, list)
    selectionChanged = Signal(list)
    duplicateRequested = Signal(list)
    deleteRequested = Signal(list)
    angleChanged = Signal(str, float)
    polygonCommitted = Signal(str, list)
    shapeDrawn = Signal(str, dict)
    zoomChanged = Signal(float)
    pixelInfo = Signal(object)          # (x, y, value text) or None outside
    editBegan = Signal(str)      # a drag/resize/rotate is about to change geometry
    labelAdded = Signal(int, list)
    labelsEdited = Signal()
    statusText = Signal(str)
    lockChanged = Signal(str, bool)     # a region was locked or unlocked
    modeChanged = Signal(str)           # select | hand | draw_* chosen from the keyboard
    escaped = Signal()                  # Esc with nothing being drawn
    arranged = Signal(str)              # align / distribute / match size / rotate, by name

    def __init__(self):
        super().__init__()
        self.sc = QGraphicsScene(self)
        # No spatial index. A region's bounding rectangle includes its handles,
        # rotate grip and name tag, which are sized in screen pixels -- so it
        # changes with every zoom, and Qt's default BSP index never hears of it
        # (prepareGeometryChange after the zoom is too late). The index then
        # kept pointers to a region in cells its new rectangle no longer
        # covered; when the region was removed -- the Job Editor removes and
        # recreates them on every selection -- those pointers were left
        # dangling, and the next repaint called into freed memory: the
        # application aborted ("pure virtual function call") or faulted with no
        # message. Reproduced every time with the index, never without it. A
        # recipe has tens of regions and results, which a linear search
        # handles without being noticed.
        self.sc.setItemIndexMethod(QGraphicsScene.NoIndex)
        self.setScene(self.sc)
        self.pix = QGraphicsPixmapItem()
        self.pix.setTransformationMode(Qt.SmoothTransformation)
        self.pix.setZValue(-100)
        self.sc.addItem(self.pix)

        # the colour behind the image comes from the design (ui_design.py,
        # "canvas"); the band kept round the image for the toolbars matches it
        try:
            from vision_station.ui import theme
            behind = theme.color("canvas")
        except Exception:
            behind = "#1B1C1F"
        self.setBackgroundBrush(QBrush(QColor(behind)))
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        # The scene reaches most of a window beyond the image (_pad_scene), so
        # the point under the pointer can stay under it while zooming at the
        # image's edge -- and the scroll bars that range would show are off:
        # the image is moved with the middle button, Space or the hand.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setDragMode(QGraphicsView.RubberBandDrag)

        self.regions = {}
        self.labels = []          # label-mode boxes, one item per annotation
        self.layer = "tools"      # "tools" or "labels"
        self.new_class = 0
        self.class_colour = lambda i: QColor("#00A8FF")
        self.class_name = lambda i: str(i)
        self._draft = None
        self._draft_origin = None
        self.overlay = []
        self.guides = []
        self._slots = {}
        # "select" | "hand" | "draw_rect" | "draw_ellipse" | "draw_polygon"
        self.mode = "select"
        self._poly_pending = []      # vertices of a polygon being traced
        self._poly_preview = None
        # Labelling every region at once turns a dense recipe into unreadable
        # soup: nine overlapping boxes each with a name tag, over a part the
        # operator is trying to look at. Names are shown for what is selected
        # and what failed, which is what anyone is actually looking for.
        self.label_policy = "smart"     # "smart" | "all" | "none"
        self._pan = None
        self._space = False
        self._suppress_band = False
        self._img = None
        self._pix_scale = 1.0
        self._scene_size = None
        self.fitted = False
        # Fit is a mode, not a one-off: while it is on the view refits when the
        # window is resized and when a new image of a different size arrives,
        # and any manual zoom turns it off. It used to be applied once, so
        # resizing the window left the image at its old size.
        self.fit_mode = True
        # the smallest a result tag is drawn on screen; the run view, read
        # from a step away, sets it larger
        self.tag_min_px = TAG_MIN_SCREEN_PX
        self.tag_max_px = TAG_MAX_SCREEN_PX
        self.show_grid = False
        # Said in the middle of an empty display, so a blank panel reads as
        # "nothing yet" rather than as a broken camera.
        self.placeholder = ""
        # What the window holding this canvas can do with regions: the
        # context menu and the keys offer only these. A tool's own editor has
        # one region, which it can neither duplicate nor delete.
        self.allow_duplicate = True
        self.allow_delete = True
        self.draw_keys = False          # R / E / P switch to the drawing tools
        self._moving = False
        self._move_start = {}
        self._hud = None
        self.sc.selectionChanged.connect(self._emit_selection)
        # result tags keep a readable size when zoomed far out: re-placed
        # whenever the zoom changes
        self.zoomChanged.connect(lambda _k: self._apply_captions())
        self._corners = _RoundCorners(self)

    def set_chrome_margins(self, left, top, right, bottom):
        """Keep a band round the image for the display's toolbars, so they sit
        beside the image instead of over it. The band is the canvas colour, so
        it reads as one surface. Fit then fits the image inside the band."""
        m = (int(left), int(top), int(right), int(bottom))
        if m == getattr(self, "_chrome", None):
            return
        self._chrome = m
        self.setViewportMargins(*m)
        if self.fit_mode and self._img is not None:
            self.fit()

    def set_placeholder(self, text):
        self.placeholder = text or ""
        self.viewport().update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._img is None and self.placeholder:
            p = QPainter(self.viewport())
            p.setPen(QColor("#9A9A9A"))
            p.drawText(self.viewport().rect().adjusted(16, 16, -16, -16),
                       Qt.AlignCenter | Qt.TextWordWrap, self.placeholder)
            p.end()

    # -- image ----------------------------------------------------------
    def _dpr(self):
        try:
            return max(1.0, float(self.viewport().devicePixelRatioF()))
        except Exception:
            return 1.0

    def _fit_zoom(self, w, h):
        """The zoom fit() will give an image of w x h (screen px per image px)."""
        vw, vh = max(50, self.viewport().width() - 4), max(50, self.viewport().height() - 4)
        return min(vw / float(w), vh / float(h))

    def _needed_scale(self, zoom):
        """Pixmap pixels per image pixel for a zoom: one per physical screen
        pixel (Windows display scaling included), a little over, and the full
        image as soon as that is nearly all of it."""
        need = zoom * self._dpr() * 1.15
        return 1.0 if need >= 0.7 else max(need, 1e-3)

    def set_image(self, img, keep_view=True):
        """Show a frame at full quality. The scene stays in full image
        coordinates.

        What is drawn has one pixmap pixel for every physical screen pixel
        the image covers -- the image's pixels times the zoom times Windows'
        display scaling -- and every image pixel once zoomed to about 60% or
        more. It was sized from the viewport instead, times the zoom only
        above 100%: at 100% a 5472-pixel frame was drawn from an 1800-pixel
        copy blown up three times, so everything between "fit" and 300% was
        blurred, and on a 150%-scaled screen even "fit" was soft. Reducing
        uses area averaging (5-8 ms for 20 megapixels): plain linear
        sampling, used before for speed, aliased fine traces and print into
        jagged noise. The full-resolution array is what is inspected and
        what the pixel readout reads; only the drawing is reduced.
        """
        self._img = img
        if img is None:
            self._shown_src = None
            self.pix.setPixmap(QPixmap())
            self.viewport().update()
            return
        h, w = img.shape[:2]
        resized = (self._scene_size != (w, h))
        refit = resized or not keep_view or not self.fitted or self.fit_mode
        zoom = self._fit_zoom(w, h) if refit else self.transform().m11()
        scale = self._needed_scale(zoom)
        # The same picture at the same detail is already drawn. Every edit
        # redraws the editor, and scaling a 20-megapixel frame again each
        # time (15-40 ms) was the lag felt while resizing or reshaping a box.
        # A sparse sample of the pixels goes with the array's identity, so a
        # frame refilled in place is still drawn again.
        print_ = img[::max(1, h // 9), ::max(1, w // 9)].tobytes()
        want = 1.0 if scale >= 1.0 else scale
        same = (getattr(self, "_shown_src", None) is img and not resized and
                getattr(self, "_shown_print", None) == print_ and
                abs(self._pix_scale - want) <= 0.02 * want)
        self._shown_print = print_
        if same:
            if refit and not getattr(self, "_fitting", False):
                self._fitting = True
                try:
                    self.fit()
                finally:
                    self._fitting = False
            return
        self._shown_src = img
        if scale < 1.0:
            sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
            scale = sw / float(w)
        else:
            scale, small = 1.0, img

        pm = ndarray_to_qpixmap(small)
        self.pix.setPixmap(pm)
        self.pix.setScale(1.0 / scale if scale else 1.0)
        self._pix_scale = scale
        self._scene_size = (w, h)
        self._pad_scene()
        if refit:
            self._fitting = True
            try:
                self.fit()
            finally:
                self._fitting = False

    def refresh_detail(self):
        """Redraw the backing pixmap at the detail the current zoom needs:
        sharper once zoomed in, and smaller again once zoomed out -- a
        full-size pixmap shrunk on screen aliases as badly as a small one
        enlarged blurs."""
        if self._img is None or getattr(self, "_fitting", False):
            return
        need = self._needed_scale(self.transform().m11())
        if need > self._pix_scale * 1.2 or need < self._pix_scale / 2.0:
            self.set_image(self._img, keep_view=True)

    def _pad_scene(self):
        """The scene: the image and, round it, 90 % of the window at the
        current zoom. With the scene exactly the image, a view can never
        scroll past the image's edge, so zooming with the pointer near an
        edge -- or anywhere while the whole image fitted in the window --
        could not keep that spot under the pointer: the picture slid away,
        300-600 image pixels per wheel step."""
        if self._scene_size is None:
            return
        w, h = self._scene_size
        k = max(1e-6, self.transform().m11())
        mx = 0.9 * max(1, self.viewport().width()) / k
        my = 0.9 * max(1, self.viewport().height()) / k
        r = QRectF(-mx, -my, w + 2 * mx, h + 2 * my)
        if r != self.sc.sceneRect():
            self.sc.setSceneRect(r)

    def fit(self):
        if not self.pix.pixmap().isNull():
            self.fitInView(self.pix, Qt.KeepAspectRatio)
            self._pad_scene()
            self.centerOn(self.pix)
            self.fitted = True
            self.fit_mode = True
            self._refresh_items()
            self.zoomChanged.emit(self.transform().m11())
            self.refresh_detail()

    # The image may not be shrunk below this many pixels on its long side, nor
    # magnified beyond this many screen pixels per image pixel. Without limits
    # sixty scroll clicks took a 640 px image down to 0.1 px on screen -- it
    # simply vanished, and nothing told you where it had gone.
    MIN_ON_SCREEN = 80
    MAX_SCALE = 32.0

    def scale_limits(self):
        if self._img is None:
            return 0.02, self.MAX_SCALE
        long_side = max(self._img.shape[1], self._img.shape[0])
        return self.MIN_ON_SCREEN / float(long_side), self.MAX_SCALE

    def zoom(self, f, anchor="centre", pos=None):
        """anchor 'mouse' keeps the image point under the pointer where it is,
        for the wheel; pos is the pointer in viewport pixels (the wheel
        event's own position -- Qt's AnchorUnderMouse used the last mouse
        MOVE it had seen, which the canvas's own move handling often skipped,
        so the zoom went somewhere else). Everything else -- menu, keys,
        buttons -- anchors at the view centre: anchoring those under the
        pointer moved the image away from wherever the operator was looking,
        because the pointer was on a menu at the time."""
        lo, hi = self.scale_limits()
        cur = self.transform().m11()
        f = max(lo / cur, min(hi / cur, f))
        if abs(f - 1.0) < 1e-6:
            return
        vp = self.viewport().rect()
        if anchor == "mouse":
            if pos is None:
                pos = getattr(self, "_last_mouse", None) or                     self.viewport().mapFromGlobal(QCursor.pos())
            pos = QPointF(pos)
            if not QRectF(vp).contains(pos):
                pos = QPointF(vp.center())
        else:
            pos = QPointF(vp.center())
        at = self.mapToScene(pos.toPoint())
        at = QPointF(at.x() + (pos.x() - pos.toPoint().x()) / cur,
                     at.y() + (pos.y() - pos.toPoint().y()) / cur)
        old = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.scale(f, f)
        self._pad_scene()
        # put that image point back under the pointer: scroll by how far, in
        # screen pixels, it now lies from the pointer
        t = self.transform()
        now = QPointF(t.m11() * at.x() + t.dx() - self.horizontalScrollBar().value(),
                      t.m22() * at.y() + t.dy() - self.verticalScrollBar().value())
        for bar, d in ((self.horizontalScrollBar(), now.x() - pos.x()),
                       (self.verticalScrollBar(), now.y() - pos.y())):
            bar.setValue(bar.value() + int(round(d)))
        self.setTransformationAnchor(old)
        self.fit_mode = False
        self._refresh_items()
        self.zoomChanged.emit(self.transform().m11())
        self.refresh_detail()

    def zoom_100(self):
        self.fit_mode = False
        centre = self.mapToScene(self.viewport().rect().center())
        self.resetTransform()
        self._pad_scene()
        self.centerOn(centre)
        self._refresh_items()
        self.zoomChanged.emit(self.transform().m11())
        self.refresh_detail()

    def zoom_to_selection(self):
        items = [i for i in self.sc.selectedItems() if isinstance(i, RegionItem)]
        if not items:
            return self.fit()
        r = items[0].sceneBoundingRect()
        for it in items[1:]:
            r = r.united(it.sceneBoundingRect())
        self.fitInView(r.adjusted(-40, -40, 40, 40), Qt.KeepAspectRatio)
        self._pad_scene()
        self.centerOn(r.center())
        self.fit_mode = False
        self._refresh_items()
        self.zoomChanged.emit(self.transform().m11())
        # the display copy of a large image is reduced for the fitted view;
        # zoomed in this far it must be redrawn in detail, as wheel zoom does,
        # or the selection is shown blurred
        self.refresh_detail()

    def _refresh_items(self):
        for it in self.regions.values():
            it.prepareGeometryChange()
            it.update()

    # -- regions --------------------------------------------------------
    def sync_regions(self, tools, results=None):
        keep = {t["id"] for t in tools
                if (t.get("params") or {}).get("roi")
                or ((t.get("params") or {}).get("shape") == "polygon"
                    and len((t.get("params") or {}).get("points") or []) >= 3)}
        selected = set(self.selected_ids())
        for tid in list(self.regions):
            if tid not in keep:
                self.sc.removeItem(self.regions.pop(tid))
        rmap = {}
        if results:
            rmap = {r["tool_id"]: r["status"] for r in results.get("tools", [])}
        for t in tools:
            roi = (t.get("params") or {}).get("roi")
            if not roi:
                continue
            tid = t["id"]
            it = self.regions.get(tid)
            params = t.get("params") or {}
            ang = float(params.get("angle", 0.0) or 0.0)
            shp = params.get("shape", "rect") or "rect"
            if shp == "polygon":
                pts = params.get("points") or []
                if len(pts) < 3:
                    continue
                old = self.regions.get(tid)
                if isinstance(old, PolygonItem):
                    if old.points() != [[float(a), float(b)] for a, b in pts]:
                        old.set_points(pts)
                    it = old
                else:
                    if old is not None:
                        self.sc.removeItem(old)
                    it = PolygonItem(tid, pts, self)
                    self.regions[tid] = it
                    self.sc.addItem(it)
                it.status = rmap.get(tid)
                _show_enabled(it, params.get("enabled", True))
                it.update()
                continue
            if isinstance(self.regions.get(tid), PolygonItem):
                self.sc.removeItem(self.regions.pop(tid))
                it = None
            if it is None:
                it = RegionItem(tid, roi, self, angle=ang, shape=shp)
                self.regions[tid] = it
                self.sc.addItem(it)
            else:
                if it.roi() != list(roi) or abs(it.angle - ang) > 1e-6:
                    it.set_roi(roi, ang)
                if it.region_shape != shp:
                    it.set_shape(shp)
            it.locked = bool(params.get("locked", False))
            it.status = rmap.get(tid)
            _show_enabled(it, (t.get("params") or {}).get("enabled", True))
            it.update()
        self.select_ids([t for t in selected if t in self.regions], silent=True)
        for tid in list(getattr(self, "_outlines", {})):
            self._hide_outline_under_region(tid)
        self._apply_captions()

    # -- label layer ----------------------------------------------------
    def set_layer(self, layer):
        self.layer = layer
        for it in self.regions.values():
            it.setFlag(QGraphicsItem.ItemIsSelectable, layer == "tools")
            it.setFlag(QGraphicsItem.ItemIsMovable, layer == "tools")
            it.setOpacity(1.0 if layer == "tools" else 0.25)
        for it in self.labels:
            it.setFlag(QGraphicsItem.ItemIsSelectable, layer == "labels")
            it.setFlag(QGraphicsItem.ItemIsMovable, layer == "labels")
            it.setVisible(layer == "labels")
        self.sc.clearSelection()

    def sync_labels(self, boxes):
        """boxes: list of {cls, x, y, w, h}. Rebuilt wholesale — cheap at these
        counts and avoids stale-index bugs after a delete."""
        for it in self.labels:
            self.sc.removeItem(it)
        self.labels = []
        for i, b in enumerate(boxes):
            col = self.class_colour(b["cls"])
            it = RegionItem(f"lbl{i}", [b["x"], b["y"], b["w"], b["h"]], self,
                            colour=col, text=self.class_name(b["cls"]), layer="labels")
            it.cls = b["cls"]
            it.index = i
            it.setZValue(5)
            self.sc.addItem(it)
            self.labels.append(it)
        self.set_layer(self.layer)

    def label_boxes(self):
        out = []
        for it in self.labels:
            r = it.roi()
            out.append({"cls": it.cls, "x": r[0], "y": r[1], "w": r[2], "h": r[3]})
        return out

    def selected_label_indices(self):
        return [i for i, it in enumerate(self.labels) if it.isSelected()]

    def selected_ids(self):
        return [i.tool_id for i in self.sc.selectedItems()
                if isinstance(i, (RegionItem, PolygonItem))
                and getattr(i, "layer", "tools") == "tools"]

    def select_ids(self, ids, silent=False):
        if silent:
            self.sc.blockSignals(True)
        for tid, it in self.regions.items():
            it.setSelected(tid in ids)
        if ids and ids[0] in self.regions:
            self.regions[ids[0]].setFocus()
        if silent:
            self.sc.blockSignals(False)

    def _emit_selection(self):
        ids = self.selected_ids()
        many = len(ids) > MANY_SELECTED
        if many != (getattr(self, "_n_selected", 0) > MANY_SELECTED):
            self._n_selected = len(ids)
            for it in self.regions.values():
                it.update()
        self._n_selected = len(ids)
        self.selectionChanged.emit(ids)

    def commit_selection(self):
        if self.layer == "labels":
            self.labelsEdited.emit()
            return
        for it in self.sc.selectedItems():
            if isinstance(it, RegionItem):
                self.geometryCommitted.emit(it.tool_id, it.roi())

    # -- moving ---------------------------------------------------------
    def begin_move(self, _scene_pos=None, mods=None):
        """A drag of the selection starts: where everything started, so Shift
        can hold the drag to one axis and snapping knows what is moving."""
        self._moving = True
        self._drag_mods = mods if mods is not None else Qt.NoModifier
        self._move_start = {it: QPointF(it.pos()) for it in self.sc.selectedItems()
                            if isinstance(it, (RegionItem, PolygonItem))}

    def end_move(self):
        self._moving = False
        self._move_start = {}
        self.hide_hud()

    def constrain_move(self, item, value):
        """Where a dragged region may go. Only a drag by the mouse is
        constrained: snapping used to apply to every position change, so an
        arrow-key nudge next to another region's edge snapped straight back,
        and a stored ROI near an edge was drawn up to 7 px from where it is."""
        if not self._moving:
            return value
        start = self._move_start.get(item)
        if getattr(item, "locked", False):
            return QPointF(item.pos())          # a locked region stays put
        mods = QApplication_mods() | getattr(self, "_drag_mods", Qt.NoModifier)
        if start is not None and mods & Qt.ShiftModifier:
            d = value - start                   # Shift: along one axis only
            value = QPointF(value.x(), start.y()) if abs(d.x()) >= abs(d.y()) \
                else QPointF(start.x(), value.y())
        if mods & Qt.ControlModifier or len(self._move_start) > 1:
            # Ctrl turns snapping off; several regions move as one, unsnapped,
            # since each snapping on its own pulled them apart
            self.clear_guides()
            return QPointF(round(value.x()), round(value.y()))
        return self.snap_position(item, value)

    # -- live readout ----------------------------------------------------
    def show_hud(self, text, scene_pos):
        """A small label by the pointer while dragging: the position, size or
        angle being set, as drawing applications show it."""
        from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsSimpleTextItem
        if self._hud is None:
            bg = QGraphicsPathItem()
            bg.setBrush(QBrush(QColor(20, 22, 26, 225)))
            bg.setPen(QPen(Qt.NoPen))
            bg.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            bg.setZValue(200)
            tx = QGraphicsSimpleTextItem(bg)
            f = QFont("Segoe UI")
            f.setPointSizeF(8.5)
            f.setBold(True)
            tx.setFont(f)
            tx.setBrush(QBrush(QColor("#FFFFFF")))
            self.sc.addItem(bg)
            self._hud = (bg, tx)
        bg, tx = self._hud
        tx.setText(text)
        br = tx.boundingRect()
        tx.setPos(22, 22)
        path = QPainterPath()
        path.addRoundedRect(QRectF(15, 19, br.width() + 14, br.height() + 6), 6, 6)
        bg.setPath(path)
        bg.setPos(scene_pos)
        bg.setVisible(True)

    def hide_hud(self):
        if self._hud is not None:
            self._hud[0].setVisible(False)

    # -- snapping -------------------------------------------------------
    def _snap_tolerance(self):
        """SNAP screen pixels in image pixels. It was 7 image pixels at every
        zoom -- under two screen pixels on a 5 MP image fitted to the window,
        so nothing ever snapped, and a jump of 56 at 800%."""
        return SNAP / max(0.02, self.transform().m11())

    def _snap_lines(self, exclude=()):
        xs = [0.0, self.sc.width(), self.sc.width() / 2]
        ys = [0.0, self.sc.height(), self.sc.height() / 2]
        for o in self.regions.values():
            if o in exclude or not o.isVisible():
                continue
            b = o.mapRectToScene(o.local_rect())
            xs += [b.left(), b.right(), b.center().x()]
            ys += [b.top(), b.bottom(), b.center().y()]
        return xs, ys

    def snap_value(self, v, axis, item):
        """The nearest snapping line to v within reach, or None."""
        xs, ys = self._snap_lines(exclude=(item,))
        tol = self._snap_tolerance()
        best = min(((abs(c - v), c) for c in (xs if axis == "x" else ys)), default=None)
        return best[1] if best is not None and best[0] < tol else None

    # -- arranging, as a drawing application does ------------------------
    def editable_selection(self):
        return [it for it in self.sc.selectedItems()
                if isinstance(it, RegionItem) and it.layer == "tools" and not it.locked
                and it.flags() & QGraphicsItem.ItemIsMovable]

    def arrange(self, op):
        """Align, distribute or match the size of the selected regions.
        One region is aligned to the image, as Illustrator aligns a single
        object to its artboard. op: align_left | align_hcenter | align_right |
        align_top | align_vcenter | align_bottom | distribute_h | distribute_v |
        match_width | match_height. Returns the ids changed."""
        items = self.editable_selection()
        if not items:
            return []
        bounds = {it: it.mapRectToScene(it.local_rect()) for it in items}
        if len(items) == 1:
            ref = QRectF(0, 0, self.sc.width(), self.sc.height())
        else:
            ref = None
            for b in bounds.values():
                ref = b if ref is None else ref.united(b)
        self.editBegan.emit(op.replace("_", " "))
        moves = {}
        if op.startswith("align_"):
            for it, b in bounds.items():
                dx = {"align_left": ref.left() - b.left(),
                      "align_right": ref.right() - b.right(),
                      "align_hcenter": ref.center().x() - b.center().x()}.get(op, 0.0)
                dy = {"align_top": ref.top() - b.top(),
                      "align_bottom": ref.bottom() - b.bottom(),
                      "align_vcenter": ref.center().y() - b.center().y()}.get(op, 0.0)
                moves[it] = (dx, dy)
        elif op in ("distribute_h", "distribute_v") and len(items) >= 3:
            horiz = op == "distribute_h"
            key = (lambda it: bounds[it].center().x()) if horiz else (lambda it: bounds[it].center().y())
            order = sorted(items, key=key)
            a, z = key(order[0]), key(order[-1])
            step = (z - a) / (len(order) - 1)
            for i, it in enumerate(order):
                d = a + i * step - key(it)
                moves[it] = (d, 0.0) if horiz else (0.0, d)
        elif op in ("match_width", "match_height") and len(items) >= 2:
            # to the largest, about each region's own centre
            if op == "match_width":
                size = max(it.rect().width() for it in items)
            else:
                size = max(it.rect().height() for it in items)
            for it in items:
                c = it._local_to_scene(it.rect().center())
                r = it.rect()
                it._apply_box(c, size if op == "match_width" else r.width(),
                              size if op == "match_height" else r.height())
        for it, (dx, dy) in moves.items():
            it.setPos(round(it.x() + dx), round(it.y() + dy))
        changed = [it.tool_id for it in items]
        for it in items:
            self.geometryCommitted.emit(it.tool_id, it.roi())
            self.liveGeometry.emit(it.tool_id, it.roi())
        self.arranged.emit(op)
        return changed

    def rotate_selection(self, deg, absolute=False):
        """Turn the selected regions by deg, or to deg with absolute=True."""
        items = self.editable_selection()
        if not items:
            return []
        self.editBegan.emit("rotate region")
        for it in items:
            it.set_angle(deg if absolute else it.angle + deg)
            self.angleChanged.emit(it.tool_id, it.angle)
        self.arranged.emit("rotate")
        return [it.tool_id for it in items]

    def set_locked(self, ids, locked):
        for tid in ids:
            it = self.regions.get(tid)
            if isinstance(it, RegionItem) and it.locked != bool(locked):
                it.locked = bool(locked)
                it.prepareGeometryChange()
                it.update()
                self.lockChanged.emit(tid, bool(locked))

    def select_all(self):
        for it in self.regions.values():
            if it.isVisible() and it.flags() & QGraphicsItem.ItemIsSelectable:
                it.setSelected(True)

    # -- right-click ----------------------------------------------------
    def contextMenuEvent(self, e):
        """The commands for what was right-clicked, as in any drawing
        application: on a region, its arrangement; on empty canvas, the view."""
        from PySide6.QtWidgets import QMenu
        from vision_station.ui import icons
        if self.layer != "tools" or self.mode.startswith("draw_"):
            return super().contextMenuEvent(e)
        hit = next((it for it in self.items(e.pos()) if isinstance(it, (RegionItem, PolygonItem))
                    and it.flags() & QGraphicsItem.ItemIsSelectable), None)
        if hit is not None and not hit.isSelected():
            self.sc.clearSelection()
            hit.setSelected(True)
        m = QMenu(self)
        ids = self.selected_ids()
        editable = self.editable_selection()
        if ids:
            if self.allow_duplicate:
                m.addAction(icons.icon("duplicate"), "Duplicate\tCtrl+D",
                            lambda: self.duplicateRequested.emit(ids))
            if self.allow_delete:
                m.addAction(icons.icon("delete"), "Delete\tDel",
                            lambda: self.deleteRequested.emit(ids))
            if self.allow_duplicate or self.allow_delete:
                m.addSeparator()
            for text, ic, fn in (("Rotate 90° clockwise\t]", "rotate_cw",
                                  lambda: self.rotate_selection(90)),
                                 ("Rotate 90° anticlockwise\t[", "rotate_ccw",
                                  lambda: self.rotate_selection(-90)),
                                 ("Straighten (0°)", "rotate_zero",
                                  lambda: self.rotate_selection(0, absolute=True))):
                a = m.addAction(icons.icon(ic), text, fn)
                a.setEnabled(bool(editable))
            al = m.addMenu(icons.icon("align_left"), "Align and distribute")
            many = len(editable) > 1
            for text, op in (("Align left", "align_left"), ("Align centres horizontally", "align_hcenter"),
                             ("Align right", "align_right"), (None, None),
                             ("Align top", "align_top"), ("Align centres vertically", "align_vcenter"),
                             ("Align bottom", "align_bottom"), (None, None),
                             ("Distribute horizontally", "distribute_h"),
                             ("Distribute vertically", "distribute_v"), (None, None),
                             ("Make the same width", "match_width"),
                             ("Make the same height", "match_height")):
                if text is None:
                    al.addSeparator()
                    continue
                a = al.addAction(icons.icon(op), text, lambda op=op: self.arrange(op))
                a.setEnabled(bool(editable) and (op.startswith("align_") or
                                                 (len(editable) >= 3 if op.startswith("distribute")
                                                  else many)))
            al.setEnabled(bool(editable))
            regions = [self.regions[i] for i in ids if isinstance(self.regions.get(i), RegionItem)]
            if regions:
                locked = all(r.locked for r in regions)
                m.addAction(icons.icon("unlock" if locked else "lock"),
                            "Unlock" if locked else "Lock position",
                            lambda: self.set_locked(ids, not locked))
            m.addSeparator()
            m.addAction(icons.icon("zoom_selection"), "Zoom to selection\tCtrl+2",
                        self.zoom_to_selection)
        else:
            m.addAction("Select all regions\tCtrl+A", self.select_all)
        m.addAction(icons.icon("fit"), "Fit to window\tCtrl+0", self.fit)
        m.addAction(icons.icon("actual"), "Actual size\tCtrl+1", self.zoom_100)
        m.exec(e.globalPos())
        e.accept()

    # -- snapping -------------------------------------------------------
    def snap_position(self, item, newpos):
        # the region as it would be at newpos -- its turned outline, so a
        # turned region snaps by what is seen, not by its unturned box
        off = newpos - item.pos()
        b = item.mapRectToScene(item.local_rect()).translated(off)
        left, top = b.left(), b.top()
        right, bottom = b.right(), b.bottom()
        cx, cy = b.center().x(), b.center().y()

        xs = [0.0, self.sc.width(), self.sc.width() / 2]
        ys = [0.0, self.sc.height(), self.sc.height() / 2]
        for o in self.regions.values():
            if o is item or not o.isVisible():
                continue
            # The region's own rectangle, not sceneBoundingRect(). boundingRect
            # is deliberately inflated to leave room for the grab handles and
            # the label above the box, so sceneBoundingRect sits roughly 15 px
            # outside the region on every side. Snapping to that lined edges up
            # with an invisible margin instead of the region, putting them about
            # 15 px out and leaving the guide lines pointing at nothing.
            # Not o.rect(): a PolygonItem has no rect, and calling it threw
            # the moment a recipe held both a polygon and a rectangle -- so
            # dragging any rectangle on a board that also had a traced region
            # crashed. Ask the item for its own shape instead.
            b = o.mapRectToScene(o.local_rect())
            xs += [b.left(), b.right(), b.center().x()]
            ys += [b.top(), b.bottom(), b.center().y()]

        # the nearest line within reach on each axis, not the first found
        tol = self._snap_tolerance()
        gx, gy, dx, dy = [], [], 0.0, 0.0
        bx = min(((abs(m - c), c - m, c) for m in (left, right, cx) for c in xs), default=None)
        if bx is not None and bx[0] < tol:
            dx, gx = bx[1], [bx[2]]
        by = min(((abs(m - c), c - m, c) for m in (top, bottom, cy) for c in ys), default=None)
        if by is not None and by[0] < tol:
            dy, gy = by[1], [by[2]]

        self.draw_guides(gx, gy)
        return QPointF(round(newpos.x() + dx), round(newpos.y() + dy))

    def draw_guides(self, xs, ys):
        self.clear_guides()
        pen = QPen(GUIDE, 0, Qt.DashLine)
        pen.setCosmetic(True)
        for x in xs:
            self.guides.append(self.sc.addLine(x, 0, x, self.sc.height(), pen))
        for y in ys:
            self.guides.append(self.sc.addLine(0, y, self.sc.width(), y, pen))
        for g in self.guides:
            g.setZValue(50)

    def clear_guides(self):
        for g in self.guides:
            self.sc.removeItem(g)
        self.guides = []

    # -- result graphics ------------------------------------------------
    def clear_overlay(self):
        for it in self.overlay:
            self.sc.removeItem(it)
        self.overlay = []

    def draw_results(self, tools):
        """Each check's result: its outline in the shape it was drawn (the
        region item itself, when the region is on the canvas for editing),
        what it found, and its result in words just below it -- OK or NG,
        the text read, the size in mm. Blue OK, red NG (core/annotate.py)."""
        from vision_station.core import annotate
        self._drawn_tools = list(tools or [])
        _SHOWING.add(self)
        self.clear_overlay()
        self._slots = {}
        self._captions = {}                        # tool id -> (caption lines, colour)
        self._tags = {}                            # tool id -> its own ResultTag
        self._outlines = {}                        # tool id -> its result outline items
        for t in tools:
            tid = t.get("tool_id")
            item = self.regions.get(tid)
            for g in t.get("graphics", []):
                pts = g["points"]
                if not pts:
                    continue
                c = STATUS_COLOR.get(g.get("status"), IDLE)
                pen = QPen(c, 0)
                pen.setCosmetic(True)
                pen.setWidthF(RESULT_W if g.get("role") == "region" else max(1.5, RESULT_W - 1))
                if g.get("status") == "search":
                    if not getattr(self, "show_search", True):
                        continue
                    pen.setWidthF(1.0)
                    pen.setStyle(Qt.DashLine)
                k = g["kind"]
                if k == "rect" and len(pts) >= 2:
                    it = self.sc.addRect(QRectF(QPointF(*pts[0]), QPointF(*pts[1])), pen)
                elif k == "line" and len(pts) >= 2:
                    it = self.sc.addLine(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1], pen)
                elif k == "poly":
                    it = self.sc.addPolygon(QPolygonF([QPointF(*p) for p in pts]), pen)
                elif k == "circle" and len(pts) >= 2:
                    r = ((pts[1][0] - pts[0][0]) ** 2 + (pts[1][1] - pts[0][1]) ** 2) ** 0.5
                    it = self.sc.addEllipse(pts[0][0] - r, pts[0][1] - r, 2 * r, 2 * r, pen)
                else:
                    it = self.sc.addEllipse(pts[0][0] - 3, pts[0][1] - 3, 6, 6, pen)
                it.setZValue(20)
                self.overlay.append(it)
                if g.get("role") == "region":
                    self._outlines.setdefault(tid, []).append(it)
            self._hide_outline_under_region(tid)
            lines = t.get("caption") or []
            if getattr(self, "tags_only_failures", False) and t.get("status") == "pass":
                lines = []                    # the run view, many checks: failures only
            if lines and self.label_policy != "none":
                self._captions[tid] = (lines, STATUS_COLOR.get(t.get("status"), IDLE))
                # A tag of its own on the result outline. While the check's
                # region is on the canvas for editing, that region carries
                # the tag instead and this one is hidden with the outline.
                g = annotate.outline(t)
                pts = g["points"] if g else [q for x in t.get("graphics", [])
                                             if x.get("status") != "search"
                                             for q in x.get("points", [])]
                if pts:
                    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
                    tag = ResultTag(QRectF(min(xs), min(ys), max(xs) - min(xs),
                                           max(ys) - min(ys)), lines,
                                    self._captions[tid][1], self._picture_long())
                    tag.setZValue(29)
                    self.sc.addItem(tag)
                    self.overlay.append(tag)
                    self._tags[tid] = tag
                    self._outlines.setdefault(tid, []).append(tag)
                    self._hide_outline_under_region(tid)
        self._apply_captions()

    def _picture_long(self):
        size = getattr(self, "_scene_size", None)
        return float(max(size)) if size else 1000.0

    def _apply_captions(self):
        """Every check's result tag on its box: carried by the region item
        while the region is on the canvas for editing, else by its own tag
        on the result outline. Tags of boxes sharing a corner stack upward
        instead of covering one another."""
        caps = getattr(self, "_captions", {})
        tags = getattr(self, "_tags", {})
        L = self._picture_long()
        k = max(1e-6, self.transform().m11())
        lo_px = float(getattr(self, "tag_min_px", TAG_MIN_SCREEN_PX))
        hi_px = max(lo_px, float(getattr(self, "tag_max_px", TAG_MAX_SCREEN_PX)))
        floor = (lo_px / k, hi_px / k)
        for tid, item in self.regions.items():
            if hasattr(item, "set_caption"):
                c = caps.get(tid)
                item.set_caption(c[0] if c else None, L)
        placed = []                                   # (left, top, height) in the image
        for tid, (lines, _col) in caps.items():
            item = self.regions.get(tid)
            if item is not None and item.isVisible() and hasattr(item, "set_caption"):
                holder = item
                local = item.polygon().boundingRect() if hasattr(item, "polygon") else item.rect()
                br = item.mapRectToScene(local)
            elif tid in tags:
                holder = tags[tid]
                br = holder.box_rect()
            else:
                continue
            holder.set_caption_floor(floor)
            h = result_tag_rect(br, lines, L, floor)[0].height()
            th = result_tag_height(br.height(), L, floor)
            top = br.top() - h
            for _ in range(8):
                hit = [q for q in placed if abs(q[0] - br.left()) < th and
                       top < q[1] + q[2] and q[1] < top + h]
                if not hit:
                    break
                top = min(q[1] for q in hit) - h
            placed.append((br.left(), top, h))
            holder.set_caption_lift((br.top() - h) - top)
            holder.update()

    def _hide_outline_under_region(self, tid):
        """A check whose region is on the canvas for editing is outlined by
        that region item; its result outline as well showed every box
        twice, a few pixels apart where the part had moved."""
        item = self.regions.get(tid)
        show = item is None or not item.isVisible()
        for it in getattr(self, "_outlines", {}).get(tid, []):
            it.setVisible(show)

    # -- navigation -----------------------------------------------------
    def cancel_draw(self):
        for it in ([self._poly_preview] if self._poly_preview else []):
            self.sc.removeItem(it)
        self._poly_preview = None
        self._poly_pending = []
        if self._draft is not None:
            self.sc.removeItem(self._draft)
            self._draft = None

    def set_mode(self, mode):
        if mode != self.mode:
            self.cancel_draw()
        self.mode = mode
        if mode.startswith("draw_"):
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)
            for it in self.regions.values():
                it.setFlag(QGraphicsItem.ItemIsMovable, False)
            return
        for it in self.regions.values():
            it.setFlag(QGraphicsItem.ItemIsMovable, self.layer == "tools")
        if mode == "hand":
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.OpenHandCursor)
        else:
            self.setDragMode(QGraphicsView.RubberBandDrag)
            self.viewport().setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Space and not self._space:
            self._space = True
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.OpenHandCursor)
            e.accept()
            return
        if k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            if self.layer == "tools" and self.selected_ids():
                self.editBegan.emit("nudge region")
            step = 10 if e.modifiers() & Qt.ShiftModifier else 1
            dx = -step if k == Qt.Key_Left else step if k == Qt.Key_Right else 0
            dy = -step if k == Qt.Key_Up else step if k == Qt.Key_Down else 0
            moved = False
            for it in self.sc.selectedItems():
                if isinstance(it, RegionItem) and not it.locked \
                        and it.flags() & QGraphicsItem.ItemIsMovable:
                    it.setPos(it.x() + dx, it.y() + dy)
                    if it.layer == "tools":
                        self.liveGeometry.emit(it.tool_id, it.roi())
                    moved = True
            if moved:
                self.commit_selection()
                e.accept()
                return
        # Ctrl+Z while tracing takes back the last point placed. Undo used to
        # go straight to the recipe-snapshot history, which knows nothing about
        # a trace in progress, so the keystroke appeared to do nothing at all
        # and the only way out of a misplaced corner was to abandon the whole
        # polygon with Esc.
        if (k == Qt.Key_Z and (e.modifiers() & Qt.ControlModifier)
                and self._poly_pending):
            self.undo_last_point()
            e.accept()
            return
        if k in (Qt.Key_Return, Qt.Key_Enter) and self._poly_pending:
            self._finish_polygon()
            e.accept()
            return
        if k == Qt.Key_Escape:
            if self._poly_pending or self._draft is not None:
                self.cancel_draw()      # abandon the trace, keep the selection
            else:
                self.sc.clearSelection()
                self.escaped.emit()
            e.accept()
            return
        if k in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.layer == "labels":
                for it in [x for x in self.labels if x.isSelected()]:
                    self.sc.removeItem(it)
                    self.labels.remove(it)
                self.labelsEdited.emit()
                e.accept()
                return
            ids = self.selected_ids()
            if ids and self.allow_delete:
                self.deleteRequested.emit(ids)
                e.accept()
                return
        ctrl = bool(e.modifiers() & Qt.ControlModifier)
        if ctrl and k in (Qt.Key_A, Qt.Key_D, Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_Equal,
                          Qt.Key_Plus, Qt.Key_Minus) and self.layer == "tools":
            # the view and selection keys of every drawing application
            if k == Qt.Key_A:
                self.select_all()
            elif k == Qt.Key_D:
                ids = self.selected_ids()
                if ids and self.allow_duplicate:
                    self.duplicateRequested.emit(ids)
            elif k == Qt.Key_0:
                self.fit()
            elif k == Qt.Key_1:
                self.zoom_100()
            elif k == Qt.Key_2:
                self.zoom_to_selection()
            elif k in (Qt.Key_Equal, Qt.Key_Plus):
                self.zoom(1.25)
            else:
                self.zoom(0.8)
            e.accept()
            return
        if k in (Qt.Key_BracketLeft, Qt.Key_BracketRight) and self.layer == "tools":
            # [ and ] turn the selection a degree at a time, Shift for 15
            step = 15.0 if e.modifiers() & Qt.ShiftModifier else 1.0
            self.rotate_selection(-step if k == Qt.Key_BracketLeft else step)
            e.accept()
            return
        if k in (Qt.Key_V, Qt.Key_H) and not ctrl:
            self.set_mode("select" if k == Qt.Key_V else "hand")
            self.modeChanged.emit(self.mode)
        elif self.draw_keys and k in (Qt.Key_R, Qt.Key_E, Qt.Key_P) and not ctrl:
            self.set_mode({Qt.Key_R: "draw_rect", Qt.Key_E: "draw_ellipse",
                           Qt.Key_P: "draw_polygon"}[k])
            self.modeChanged.emit(self.mode)
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space:
            self._space = False
            self.set_mode(self.mode)
            e.accept()
            return
        super().keyReleaseEvent(e)

    def _draw_press(self, e):
        p = self.mapToScene(e.position().toPoint())
        if self.mode == "draw_polygon":
            if e.button() == Qt.RightButton:
                self._finish_polygon()
                return True
            self._poly_pending.append([p.x(), p.y()])
            if self._poly_preview is None:
                pen = QPen(SEL, 0, Qt.DashLine)
                pen.setCosmetic(True)
                self._poly_preview = self.sc.addPolygon(QPolygonF(), pen)
                self._poly_preview.setZValue(70)
                self._start_ants()
            self._poly_preview.setPolygon(
                QPolygonF([QPointF(*v) for v in self._poly_pending]))
            return True
        if e.button() == Qt.LeftButton:
            self._draft_origin = p
            pen = QPen(SEL, 1.6, Qt.DashLine)
            pen.setCosmetic(True)
            self._draft = (self.sc.addEllipse(QRectF(p, p), pen, QBrush(QColor(124, 92, 255, 30)))
                           if self.mode == "draw_ellipse"
                           else self.sc.addRect(QRectF(p, p), pen, QBrush(QColor(124, 92, 255, 30))))
            self._draft.setZValue(70)
            self._start_ants()
            return True
        return False

    def _start_ants(self):
        """Marching ants: the dashes of a box being drawn run round it, as
        in every drawing program -- a still dashed line read as a region
        already made. Stops by itself once nothing is being drawn."""
        from PySide6.QtCore import QTimer
        t = getattr(self, "_ants", None)
        if t is None:
            t = self._ants = QTimer(self)
            t.setInterval(60)
            t.timeout.connect(self._march)
            self._ants_off = 0.0
        if not t.isActive():
            t.start()

    def _march(self):
        items = [i for i in (self._draft, self._poly_preview) if i is not None]
        if not items:
            self._ants.stop()
            return
        self._ants_off = (self._ants_off - 1.0) % 64.0
        for it in items:
            pen = it.pen()
            pen.setDashOffset(self._ants_off)
            it.setPen(pen)

    def undo_last_point(self):
        """Remove the most recently placed vertex of a trace in progress."""
        if not self._poly_pending:
            return False
        self._poly_pending.pop()
        if not self._poly_pending:
            self.cancel_draw()
            return True
        if self._poly_preview is not None:
            self._poly_preview.setPolygon(
                QPolygonF([QPointF(*v) for v in self._poly_pending]))
            self._poly_preview.update()
        return True

    def tracing(self):
        return bool(self._poly_pending)

    def _finish_polygon(self):
        pts = list(self._poly_pending)
        self.cancel_draw()
        if len(pts) >= 3:
            self.shapeDrawn.emit("polygon", {"points": pts})

    def mousePressEvent(self, e):
        if self.mode.startswith("draw_") and self._draw_press(e):
            e.accept()
            return
        if e.button() == Qt.MiddleButton:
            self._pan = e.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            e.accept()
            return
        # Shift-clicking an existing region extends the selection. The item
        # handles the toggle itself, but the view's rubber band would then
        # replace the whole selection on release -- Qt's band honours Ctrl, not
        # Shift -- so the band is suppressed for this one click.
        if (e.button() == Qt.LeftButton and self.mode == "select"
                and self.layer == "tools" and (e.modifiers() & Qt.ShiftModifier)):
            p = self.mapToScene(e.position().toPoint())
            if any(it.sceneBoundingRect().contains(p) for it in self.regions.values()):
                self._suppress_band = True
                self.setDragMode(QGraphicsView.NoDrag)

        if (self.layer == "labels" and self.mode == "select"
                and e.button() == Qt.LeftButton and not self._space):
            p = self.mapToScene(e.position().toPoint())
            if not any(it.sceneBoundingRect().contains(p) and it.isSelected()
                       for it in self.labels):
                self._draft_origin = p
                pen = QPen(self.class_colour(self.new_class), 0, Qt.DashLine)
                pen.setCosmetic(True)
                self._draft = self.sc.addRect(QRectF(p, p), pen)
                self._draft.setZValue(60)
                self._start_ants()
                e.accept()
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        self._last_mouse = QPointF(e.position())
        if self._draft is not None:
            p = self.mapToScene(e.position().toPoint())
            rect = QRectF(self._draft_origin, p).normalized()
            if self.mode == "draw_ellipse":
                self._draft.setRect(rect)
            else:
                self._draft.setRect(rect)
            e.accept()
            return
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            e.accept()
            return
        p = self.mapToScene(e.position().toPoint())
        info = self.pixel_at(p)
        self.pixelInfo.emit(info)
        if info is not None:
            x, y, val = info
            self.statusText.emit(f"X {x}  Y {y}     {val}"
                                 f"     Zoom {self.transform().m11()*100:.0f}%")
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self.mode in ("draw_rect", "draw_ellipse") and self._draft is not None:
            r = (self._draft.rect() if hasattr(self._draft, "rect")
                 else self._draft.boundingRect()).normalized()
            self.sc.removeItem(self._draft)
            self._draft = None
            if r.width() > 5 and r.height() > 5:
                self.shapeDrawn.emit(
                    "ellipse" if self.mode == "draw_ellipse" else "rect",
                    {"roi": [int(r.x()), int(r.y()),
                             int(r.width()), int(r.height())]})
            e.accept()
            return
        if getattr(self, "_suppress_band", False):
            self._suppress_band = False
            super().mouseReleaseEvent(e)
            self.setDragMode(QGraphicsView.RubberBandDrag)
            self.clear_guides()
            return
        if self._draft is not None:
            r = self._draft.rect().normalized()
            self.sc.removeItem(self._draft)
            self._draft = None
            if r.width() > 4 and r.height() > 4:
                self.labelAdded.emit(self.new_class, [int(r.x()), int(r.y()),
                                                      int(r.width()), int(r.height())])
            else:
                self.sc.clearSelection()
            e.accept()
            return
        if e.button() == Qt.MiddleButton:
            self._pan = None
            self.viewport().setCursor(Qt.OpenHandCursor if self.mode == "hand" else Qt.ArrowCursor)
            e.accept()
            return
        super().mouseReleaseEvent(e)
        self.clear_guides()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._corners.setGeometry(self.rect())
        self._corners.raise_()
        self._pad_scene()
        if self.fit_mode and self._img is not None:
            self.fit()

    def drawForeground(self, painter, rect):
        """Pixel grid at 8x and above, where individual pixels are visible:
        what anyone checking an edge wants to see."""
        super().drawForeground(painter, rect)
        if not self.show_grid or self._img is None or self.transform().m11() < 8.0:
            return
        h, w = self._img.shape[:2]
        x0, x1 = max(0, int(rect.left())), min(w, int(rect.right()) + 1)
        y0, y1 = max(0, int(rect.top())), min(h, int(rect.bottom()) + 1)
        pen = QPen(QColor(255, 255, 255, 60), 0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        for x in range(x0, x1 + 1):
            painter.drawLine(QPointF(x, y0), QPointF(x, y1))
        for y in range(y0, y1 + 1):
            painter.drawLine(QPointF(x0, y), QPointF(x1, y))

    def pixel_at(self, scene_pt):
        """(x, y, value text) for the image pixel under a scene point, read from
        the source array so the value is exact, or None outside the image.

        floor, not int(): int() truncates towards zero, so a pointer just left
        of or above the image (x = -0.3) reported pixel 0 instead of nothing."""
        import math as _m
        if self._img is None:
            return None
        x, y = int(_m.floor(scene_pt.x())), int(_m.floor(scene_pt.y()))
        h, w = self._img.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        v = self._img[y, x]
        if getattr(v, "ndim", 0) == 0 or self._img.ndim == 2:
            return x, y, "%d" % int(v)
        if len(v) >= 3:
            b, g, r = int(v[0]), int(v[1]), int(v[2])
            return x, y, "R %d  G %d  B %d" % (r, g, b)
        return x, y, "%d" % int(v[0])

    def wheelEvent(self, e):
        dy = e.angleDelta().y() or e.angleDelta().x()
        if not dy:
            e.accept()
            return
        # a precise touchpad sends small steps: zoom by as much as it scrolled
        steps = max(-4.0, min(4.0, dy / 120.0))
        self.zoom(1.16 ** steps, anchor="mouse", pos=e.position())
        e.accept()
