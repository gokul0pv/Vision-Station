"""Modern list rendering shared by the editors.

    CheckRowDelegate   a job's checks as current design tools list layers:
                       a tinted tile with the check's icon, its name, and its
                       last result as a coloured pill (OK blue, NG red) --
                       instead of plain "PASS" text in a column
"""
from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem

from vision_station.core.annotate import STYLE as _ANN
from vision_station.ui import icons, theme

TYPE_ROLE = Qt.UserRole + 7           # the tool type of a check's row

_PILL = {"PASS": ("OK", _ANN["ok"]), "FAIL": ("NG", _ANN["ng"]),
         "ERROR": ("ERROR", _ANN["warn"]), "WARN": ("WARN", _ANN["warn"])}


def _follow_look():
    _PILL["PASS"] = ("OK", _ANN["ok"])
    _PILL["FAIL"] = ("NG", _ANN["ng"])


from vision_station.core import annotate as _annotate_look  # noqa: E402
_annotate_look.on_look_changed(_follow_look)


def _tone(hexcol, alpha):
    c = QColor(hexcol)
    c.setAlpha(alpha)
    return c


class CheckRowDelegate(QStyledItemDelegate):
    ROW_H = 36

    def sizeHint(self, opt, idx):
        sz = super().sizeHint(opt, idx)
        if not idx.parent().isValid() and idx.sibling(idx.row(), 0).data(TYPE_ROLE):
            sz.setHeight(max(sz.height(), self.ROW_H))
            if idx.column() == 1:
                sz.setWidth(max(sz.width(), 64))
        return sz

    def paint(self, p, opt, idx):
        ty = idx.sibling(idx.row(), 0).data(TYPE_ROLE) if not idx.parent().isValid() else None
        if not ty or idx.column() not in (0, 1):
            return super().paint(p, opt, idx)
        o = QStyleOptionViewItem(opt)
        self.initStyleOption(o, idx)
        style = o.widget.style() if o.widget else None
        # the row's hover / selection background, as the style draws it
        text, o.text = o.text, ""
        o.icon = icons.QIcon()
        if style is not None:
            style.drawControl(QStyle.CE_ItemViewItem, o, p, o.widget)
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(o.rect)
        off = idx.data(Qt.ForegroundRole) is not None and idx.column() == 0 and \
            idx.sibling(idx.row(), 1).data(Qt.DisplayRole) == "off"
        if idx.column() == 0:
            fam = icons.TOOLS.get(ty, ("logic", ""))[0]
            acc = icons.ACCENTS.get(fam, "#475569")
            tile = QRectF(r.left() + 4, r.center().y() - 12, 24, 24)
            p.setPen(Qt.NoPen)
            p.setBrush(_tone(acc, 40 if not off else 18))
            p.drawRoundedRect(tile, 7, 7)
            icons.tool_icon(ty).paint(p, tile.adjusted(4, 4, -4, -4).toRect(),
                                      Qt.AlignCenter,
                                      icons.QIcon.Disabled if off else icons.QIcon.Normal)
            f = QFont(o.font)
            f.setWeight(QFont.DemiBold if (o.state & QStyle.State_Selected) else QFont.Medium)
            p.setFont(f)
            p.setPen(QColor(theme.color("text_disabled") if off else theme.color("text")))
            tr = r.adjusted(36, 0, -4, 0)
            p.drawText(tr, Qt.AlignVCenter | Qt.AlignLeft,
                       p.fontMetrics().elidedText(text, Qt.ElideRight, int(tr.width())))
        else:
            label, col = _PILL.get(text, (text, None))
            if label:
                f = QFont(o.font)
                f.setPointSizeF(max(7.0, f.pointSizeF() - 1.0))
                f.setWeight(QFont.DemiBold)
                p.setFont(f)
                w = p.fontMetrics().horizontalAdvance(label) + 16
                h = 20
                pill = QRectF(r.right() - w - 6, r.center().y() - h / 2, w, h)
                if col is None:                     # "off": a quiet outline
                    p.setPen(QColor(theme.color("stroke_strong")))
                    p.setBrush(Qt.NoBrush)
                    ink = QColor(theme.color("text3"))
                else:
                    p.setPen(Qt.NoPen)
                    p.setBrush(_tone(col, 34))
                    ink = QColor(col)
                p.drawRoundedRect(pill, h / 2, h / 2)
                p.setPen(ink)
                p.drawText(pill, Qt.AlignCenter, label)
        p.restore()


def style_check_list(tree):
    """A checks list (a QTreeWidget) drawn the modern way."""
    tree.setItemDelegate(CheckRowDelegate(tree))
    tree.setProperty("checkList", True)
    tree.setIconSize(QSize(16, 16))
    tree.setIndentation(14)
    tree.setUniformRowHeights(False)
    return tree
