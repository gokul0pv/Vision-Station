"""Small UI pieces shared by both front ends -- the classic window, its tool
editor, and the VisionPro-style application.

Each of these existed in two to four copies, and copies drift: the YOLO tools'
duplicated set-up code had already diverged in how it reported a missing
runtime, and the text detectors' in how they handled light-on-dark print.
One definition each, here.
"""
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (QAbstractItemView, QAbstractSpinBox, QComboBox, QHBoxLayout,
                               QHeaderView, QLabel, QListWidget, QMenu, QMessageBox,
                               QPushButton, QTableWidget, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout)

from vision_station.core.schema import GROUPS, LABELS


class _WheelGuard(QObject):
    """Scrolling a page of settings must not change them. A spin box or list
    under the pointer took the wheel, so scrolling down a tool's settings could
    move its accept threshold without anyone noticing. Now a field takes the
    wheel only once it has been clicked into; otherwise the page scrolls."""

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Wheel and not obj.hasFocus():
            ev.ignore()
            return True
        return False


_wheel_guard = None


def guard_wheel(widget):
    """Apply the wheel guard to every spin box and drop-down list in widget."""
    global _wheel_guard
    if _wheel_guard is None:
        _wheel_guard = _WheelGuard()
    for w in widget.findChildren(QAbstractSpinBox) + widget.findChildren(QComboBox):
        if not w.property("wheelGuarded"):
            w.setFocusPolicy(Qt.StrongFocus)
            w.installEventFilter(_wheel_guard)
            w.setProperty("wheelGuarded", True)
    return widget


class EmptyStateTable(QTableWidget):
    """A table that says why it is empty, instead of showing a blank box."""

    def __init__(self, rows=0, cols=0, empty_text="", parent=None):
        super().__init__(rows, cols, parent)
        self.empty_text = empty_text

    def set_empty_text(self, text):
        self.empty_text = text
        self.viewport().update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.rowCount() == 0 and self.empty_text:
            p = QPainter(self.viewport())
            p.setPen(self.palette().color(self.palette().ColorRole.PlaceholderText))
            p.drawText(self.viewport().rect().adjusted(12, 8, -12, -8),
                       Qt.AlignCenter | Qt.TextWordWrap, self.empty_text)
            p.end()


def read_only_table(table, stretch_last=True):
    """The look every results table shares: no row numbers, no editing, whole
    rows selected, quiet alternate shading."""
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setShowGrid(False)
    table.setAlternatingRowColors(True)
    table.setWordWrap(False)
    hh = table.horizontalHeader()
    hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    hh.setStretchLastSection(stretch_last)
    hh.setHighlightSections(False)
    if table.columnCount():
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
    return table


def dot_icon(color, size=11):
    """A filled circle: the pass/fail/idle marker in trees and lists."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return QIcon(pm)


def dot_style(color):
    """Style for a status-bar label shown as a coloured dot."""
    return ("background:%s;border-radius:6px;min-width:12px;max-width:12px;"
            "min-height:12px;max-height:12px;padding:0;margin:0 4px;" % color)


def make_action(owner, text, fn=None, key=None, tip="", check=False, register_key=True,
                icon=None):
    """A toolbar action. `owner` is a QToolBar (the action is added to it) or a
    window. With register_key=False the key is shown in the tooltip but not
    registered -- for a toolbar button whose shortcut a menu already owns: two
    owners of one key make it ambiguous, and Qt then fires neither."""
    a = QAction(text, owner)
    a.setCheckable(check)
    if icon:
        from vision_station.ui import icons
        a.setIcon(icons.icon(icon))
        a.setProperty("vsIcon", icon)        # redrawn in the new ink when the theme changes
    if fn is not None:
        a.triggered.connect(fn)
    if key and register_key:
        a.setShortcut(QKeySequence(key))
        a.setShortcutContext(Qt.WindowShortcut)
    a.setToolTip("%s   (%s)" % (tip or text, key) if key else (tip or text))
    if hasattr(owner, "addAction") and owner.__class__.__name__ == "QToolBar":
        owner.addAction(a)
    return a


def fill_tool_menu(menu, on_add):
    """The tool catalogue as submenus by category: 'Add tool' everywhere.
    The submenus are kept referenced on the menu: on PySide6 6.11 a submenu
    wrapper left to the garbage collector takes its actions' wrappers with it."""
    from vision_station.ui import icons
    menu._subs = []
    for group, types in GROUPS:
        sub = menu.addMenu(group)
        menu._subs.append(sub)
        for ty in types:
            name, why = LABELS[ty]
            a = sub.addAction(icons.tool_icon(ty), name, lambda ty=ty: on_add(ty))
            a.setToolTip(why)
        sub.setToolTipsVisible(True)
    return menu


def toolbox_tree(on_add, group_colour=None):
    """The Toolbox: every tool by category; double-click adds one."""
    tree = QTreeWidget()
    tree.setHeaderHidden(True)
    tree.setIndentation(12)
    for group, types in GROUPS:
        gi = QTreeWidgetItem([group])
        f = gi.font(0)
        f.setBold(True)
        gi.setFont(0, f)
        if group_colour:
            gi.setForeground(0, QColor(group_colour))
        tree.addTopLevelItem(gi)
        for ty in types:
            name, why = LABELS.get(ty, (ty, ""))
            it = QTreeWidgetItem([name])
            it.setData(0, Qt.UserRole, ty)
            it.setToolTip(0, why)
            from vision_station.ui import icons
            it.setIcon(0, icons.tool_icon(ty))
            gi.addChild(it)
        gi.setExpanded(True)

    def added(item, _col=0):
        ty = item.data(0, Qt.UserRole)
        if ty:
            on_add(ty)
    tree.itemDoubleClicked.connect(added)
    tree.add_item = added                   # for tests and keyboard entry
    return tree


# tools whose output image later tools can read
IMAGE_PRODUCERS = {"image_process", "affine_transform", "polar_unwrap", "image_combine"}


def readable_images(tools, tool_id, captures):
    """The images a tool can read: the camera captures, plus the output of any
    image-processing tool that runs BEFORE it -- a later one's image does not
    exist yet when this tool runs."""
    names = list(captures)
    for t in tools:
        if t["id"] == tool_id:
            break
        if t.get("type") in IMAGE_PRODUCERS:
            names.append(t["id"])
    return names


def find_tool(tools, tool_id):
    return next((t for t in tools if t["id"] == tool_id), None)


def confirm_unsaved(parent, what="this model"):
    """'save' | 'discard' | 'cancel' for unsaved edits. Save is the default
    button: Enter keeps the work, it never throws it away."""
    r = QMessageBox.question(parent, "Unsaved changes",
                             "Save the changes to %s first?" % what,
                             QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                             QMessageBox.Save)
    return {QMessageBox.Save: "save", QMessageBox.Discard: "discard"}.get(r, "cancel")


def training_list(on_menu, icon=(120, 90)):
    """One column of training thumbnails, OK or NG. on_menu(list, pos) handles
    the right-click menu."""
    lst = QListWidget()
    lst.setViewMode(QListWidget.IconMode)
    lst.setIconSize(QSize(*icon))
    lst.setGridSize(QSize(icon[0] + 12, icon[1] + 26))
    lst.setResizeMode(QListWidget.Adjust)
    lst.setMovement(QListWidget.Static)
    lst.setWordWrap(True)
    lst.setUniformItemSizes(True)
    lst.setMinimumHeight(icon[1] + 40)
    lst.setContextMenuPolicy(Qt.CustomContextMenu)
    lst.customContextMenuRequested.connect(lambda pos: on_menu(lst, pos))
    return lst


def training_menu(parent, lst, label, pos, side, extra=None):
    """Right-click on a training image: move it to the other column or remove
    it. extra: [(text, callable(item))] for front-end specific entries.
    Returns True when the set changed or an extra action ran."""
    item = lst.itemAt(pos)
    if item is None:
        return False
    path = item.data(Qt.UserRole)
    other = "ng" if label == "ok" else "ok"
    m = QMenu(parent)
    a_move = m.addAction("Move to %s" % other.upper())
    extras = [(m.addAction(text), fn) for text, fn in (extra or [])]
    m.addSeparator()
    a_del = m.addAction("Remove from training set")
    chosen = m.exec(lst.mapToGlobal(pos))
    if chosen is a_move:
        side.relabel(path, other)
    elif chosen is a_del:
        side.remove_training_image(path)
    else:
        for act, fn in extras:
            if chosen is act:
                fn(item)
                break
        else:
            return False
    return True


def training_columns(on_menu, on_add, colours, head_style, icon=(120, 90)):
    """The OK and NG columns side by side: a coloured header, the thumbnail
    list and an 'add current image' button each. Returns (layout, {label:
    list}). on_menu(list, label, pos) and on_add(label) are the callbacks;
    head_style is a format string taking the colour -- or None, with colours
    given as tones ("pass", "fail") for the theme's stylesheet to colour."""
    cols = QHBoxLayout()
    lists = {}
    for label, colour in zip(("ok", "ng"), colours):
        box = QVBoxLayout()
        head = QLabel(label.upper())
        head.setAlignment(Qt.AlignCenter)
        if head_style is None:
            head.setObjectName("trainHead")
            head.setProperty("tone", colour)
        else:
            head.setStyleSheet(head_style % colour)
        box.addWidget(head)
        lst = training_list(lambda l, pos, lab=label: on_menu(l, lab, pos), icon=icon)
        box.addWidget(lst, 1)
        add = QPushButton("+ Add current image")
        add.setToolTip("Add the image on the display to the %s set" % label.upper())
        add.clicked.connect(lambda _c=False, lab=label: on_add(lab))
        box.addWidget(add)
        cols.addLayout(box)
        lists[label] = lst
    return cols, lists


def ok_cancel(dialog, layout, on_ok=None):
    """The OK / Cancel row every dialog ends with."""
    from PySide6.QtWidgets import QDialogButtonBox
    bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    bb.accepted.connect(on_ok or dialog.accept)
    bb.rejected.connect(dialog.reject)
    layout.addWidget(bb)
    return bb
