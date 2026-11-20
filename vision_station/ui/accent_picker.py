"""A colour picker laid out as Windows Settings > Personalization > Colors >
Accent color: a row of recent colours, the 48 Windows colours in a grid of
square swatches (the chosen one ringed and ticked), and Custom colour.

Solid colours only -- the custom colour dialog has no transparency, and
every value is kept as '#RRGGBB'.

Two pickers can be interlocked (OK and NG result colours): the colour one
has taken is greyed out and cannot be chosen in the other.

    ok, ng = AccentPicker("#1F6FFF"), AccentPicker("#FF2A2A")
    interlock(ok, ng)
"""
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractButton, QColorDialog, QGridLayout, QHBoxLayout,
                               QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from vision_station.core.annotate import solid_hex

# Windows' own accent colours, 8 per row, as Settings shows them
WINDOWS_COLOURS = (
    "#FFB900", "#FF8C00", "#F7630C", "#CA5010", "#DA3B01", "#EF6950", "#D13438", "#FF4343",
    "#E74856", "#E81123", "#EA005E", "#C30052", "#E3008C", "#BF0077", "#C239B3", "#9A0089",
    "#0078D7", "#0063B1", "#8E8CD8", "#6B69D6", "#8764B8", "#744DA9", "#B146C2", "#881798",
    "#0099BC", "#2D7D9A", "#00B7C3", "#038387", "#00B294", "#018574", "#00CC6A", "#10893E",
    "#7A7574", "#5D5A58", "#68768A", "#515C6B", "#567C73", "#486860", "#498205", "#107C10",
    "#767676", "#4C4A48", "#69797E", "#4A5459", "#647C64", "#525E54", "#847545", "#7E735F",
)
COLUMNS = 8
SWATCH = 30
RECENT_MAX = 6


def _ink(c):
    q = QColor(c)
    return QColor("#000000") if (0.299 * q.red() + 0.587 * q.green() + 0.114 * q.blue()) > 160 \
        else QColor("#FFFFFF")


class Swatch(QAbstractButton):
    """One square of solid colour. Checked: ringed and ticked. Disabled
    (taken by the interlocked picker): faded and struck through."""

    def __init__(self, colour, parent=None):
        super().__init__(parent)
        self.colour = solid_hex(colour)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(SWATCH, SWATCH)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(self.colour)
        self.blocked_by = ""
        self._tip()

    def _tip(self):
        self.setToolTip(self.colour + ("  -  used for %s" % self.blocked_by
                                       if self.blocked_by else ""))

    def set_blocked(self, who):
        self.blocked_by = who or ""
        self.setEnabled(not self.blocked_by)
        self._tip()

    def sizeHint(self):
        return QSize(SWATCH, SWATCH)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        fg = self.palette().windowText().color()
        fill = QColor(self.colour)
        if not self.isEnabled():
            fill.setAlpha(60)
        p.setPen(Qt.NoPen)
        p.setBrush(fill)
        inner = r.adjusted(3, 3, -3, -3) if self.isChecked() else r
        p.drawRoundedRect(inner, 4, 4)
        if self.isChecked():                          # Windows: a ring and a tick
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(fg, 2))
            p.drawRoundedRect(r, 5, 5)
            c = inner.center()
            tick = QPainterPath(QPointF(c.x() - 5, c.y()))
            tick.lineTo(c.x() - 1.5, c.y() + 3.5)
            tick.lineTo(c.x() + 5, c.y() - 3.5)
            p.setPen(QPen(_ink(self.colour), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPath(tick)
        elif not self.isEnabled():                    # taken by the other picker
            grey = QColor(fg)
            grey.setAlpha(140)
            p.setPen(QPen(grey, 1.5))
            p.drawLine(QPointF(r.left() + 5, r.bottom() - 5), QPointF(r.right() - 5, r.top() + 5))
        elif self.underMouse() or self.hasFocus():
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(fg, 1.5))
            p.drawRoundedRect(r, 4, 4)
        p.end()

    def enterEvent(self, e):
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.update()
        super().leaveEvent(e)


class AccentPicker(QWidget):
    """Recent colours, the Windows colours, and Custom colour. changed(hex)
    is emitted when the user picks a colour; moved(hex) whenever the colour
    changes, by the user or set_colour (the interlock follows moved)."""
    changed = Signal(str)
    moved = Signal(str)

    def __init__(self, colour, defaults=(), name="", parent=None):
        super().__init__(parent)
        self.name = name                 # "OK" / "NG": named in the other picker's hints
        self._colour = solid_hex(colour) or WINDOWS_COLOURS[16]
        self._blocked = None             # the interlocked picker's colour
        self._blocked_by = ""
        self._recent = []
        for c in list(defaults) + [self._colour]:
            self._remember(c)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.preview = QLabel()
        self.preview.setFixedSize(54, 22)
        self.preview.setAlignment(Qt.AlignCenter)
        top.addWidget(self.preview)
        self.hexLabel = QLabel()
        top.addWidget(self.hexLabel)
        top.addStretch(1)
        v.addLayout(top)

        v.addWidget(self._caption("Recent colours"))
        self.recentRow = QHBoxLayout()
        self.recentRow.setSpacing(4)
        self.recentRow.setContentsMargins(0, 0, 0, 0)
        holder = QWidget()
        holder.setLayout(self.recentRow)
        v.addWidget(holder)

        v.addWidget(self._caption("Windows colours"))
        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 0)
        self.swatches = []
        for i, c in enumerate(WINDOWS_COLOURS):
            s = self._swatch(c)
            grid.addWidget(s, i // COLUMNS, i % COLUMNS)
            self.swatches.append(s)
        g = QWidget()
        g.setLayout(grid)
        v.addWidget(g)

        row = QHBoxLayout()
        row.addWidget(QLabel("Custom colour"))
        row.addStretch(1)
        self.customBtn = QPushButton("View colours")
        self.customBtn.setAutoDefault(False)
        self.customBtn.clicked.connect(self._custom)
        row.addWidget(self.customBtn)
        v.addLayout(row)

        self.recentSwatches = []
        self._rebuild_recent()
        self._refresh()

    # ---- building ------------------------------------------------------
    def _caption(self, text):
        l = QLabel(text)
        f = l.font()
        f.setPointSizeF(max(7.0, f.pointSizeF() - 1))
        l.setFont(f)
        l.setEnabled(False)                          # the theme's secondary text
        return l

    def _swatch(self, c):
        s = Swatch(c, self)
        s.clicked.connect(lambda _on=False, col=s.colour: self._pick(col))
        return s

    def _remember(self, c):
        c = solid_hex(c)
        if not c:
            return
        if c in self._recent:
            self._recent.remove(c)
        self._recent.insert(0, c)
        del self._recent[RECENT_MAX:]

    def _rebuild_recent(self):
        while self.recentRow.count():
            item = self.recentRow.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.recentSwatches = [self._swatch(c) for c in self._recent]
        for s in self.recentSwatches:
            self.recentRow.addWidget(s)
        self.recentRow.addStretch(1)

    def _all(self):
        return self.swatches + self.recentSwatches

    def _refresh(self):
        c = self._colour
        for s in self._all():
            s.setChecked(s.colour == c)
            s.set_blocked(self._blocked_by if s.colour == self._blocked else "")
        self.preview.setText(self.name or "")
        self.preview.setStyleSheet(
            "background:%s;color:%s;border-radius:4px;font-weight:600;" % (c, _ink(c).name()))
        self.hexLabel.setText(c)

    # ---- choosing --------------------------------------------------------
    def _pick(self, c):
        c = solid_hex(c)
        if not c:
            return
        if c == self._blocked:                       # an interlocked colour cannot be taken
            self._refresh()
            return
        changed = c != self._colour
        self._colour = c
        if c not in self._recent:
            self._remember(c)
            self._rebuild_recent()
        self._refresh()
        if changed:
            self.moved.emit(c)
            self.changed.emit(c)

    def _custom(self):
        # no ShowAlphaChannel: solid colours only
        q = QColorDialog.getColor(QColor(self._colour), self,
                                  "%s colour" % self.name if self.name else "Custom colour")
        if not q.isValid():
            return
        c = solid_hex(q.name())
        if c == self._blocked:
            QMessageBox.information(self, "Colour in use",
                                    "%s is already used for %s. Choose another colour."
                                    % (c, self._blocked_by or "the other result"))
            return
        self._pick(c)

    # ---- the interface ---------------------------------------------------
    def colour(self):
        return self._colour

    def set_colour(self, c):
        """Show colour c as chosen. Refused (False) when it is not a solid
        colour or the interlocked picker has it."""
        c = solid_hex(c)
        if not c or c == self._blocked:
            self._refresh()
            return False
        moved = c != self._colour
        self._colour = c
        if c not in self._recent:
            self._remember(c)
            self._rebuild_recent()
        self._refresh()
        if moved:
            self.moved.emit(c)
        return True

    def block(self, c, by=""):
        """Grey out colour c here: the interlocked picker has taken it."""
        self._blocked = solid_hex(c)
        self._blocked_by = by
        self._refresh()


def interlock(a, b):
    """A colour taken by one picker is greyed out in the other, both ways."""
    a.moved.connect(lambda c: b.block(c, a.name))
    b.moved.connect(lambda c: a.block(c, b.name))
    a.block(b.colour(), b.name)
    b.block(a.colour(), a.name)


def set_pair(a, b, ca, cb, fallback_b):
    """Set two interlocked pickers at once (saved settings): unlocked first, so
    swapped colours load, and b takes fallback_b when both are the same."""
    ca, cb = solid_hex(ca) or a.colour(), solid_hex(cb) or b.colour()
    if cb == ca:
        cb = next(c for c in [solid_hex(fallback_b), b.colour()] + list(WINDOWS_COLOURS)
                  if c and c != ca)
    a.block(None)
    b.block(None)
    a.set_colour(ca)
    b.set_colour(cb)
    a.block(b.colour(), b.name)
    b.block(a.colour(), a.name)
