"""Applies the design to the application: light or dark, following Windows
unless the user chose one.

THE LOOK ITSELF IS IN design.py (and how it moves, in motion.py) -- colours, fonts, sizes and the whole
stylesheet. This module only applies it, and gives code the few things it
needs at run time:

    apply(app, "system" | "light" | "dark")
    color("accent")             a colour of the current mode
    is_dark()
    tone_text("fail")           a result colour readable as text
    set_tone(widget, "pass")    sets the property the stylesheet reads
"""
import sys

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QMenu, QStyleFactory, QWidget

from vision_station.ui import design as ui_design

# Results keep one colour in both modes (markers: dots, outlines). As text,
# use tone_text().
PASS, FAIL, WARN, IDLE = (ui_design.STATUS[k] for k in ("pass", "fail", "warn", "idle"))
DARK = ui_design.DARK
LIGHT = ui_design.LIGHT

_state = {"mode": "light", "tokens": LIGHT, "filter": None}


def color(name):
    return _state["tokens"][name]


def is_dark():
    return _state["mode"] == "dark"


def mode():
    return _state["mode"]


def system_prefers_dark():
    """Windows' own app-mode setting; Qt's colour scheme first, the registry
    as a fallback for Qt builds that do not report it."""
    try:
        scheme = QApplication.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return True
        if scheme == Qt.ColorScheme.Light:
            return False
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                               r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            return winreg.QueryValueEx(k, "AppsUseLightTheme")[0] == 0
        except OSError:
            pass
    return False


def stylesheet(t, glyphs):
    return ui_design.build_stylesheet(t, glyphs)


def _palette(t):
    pal = QPalette()

    def solid(c):
        return QColor(c) if not c.startswith("rgba") else QColor(t["layer"])
    for role, key in ((QPalette.Window, "bg"), (QPalette.WindowText, "text"),
                      (QPalette.Base, "layer"), (QPalette.AlternateBase, "layer_alt"),
                      (QPalette.Text, "text"), (QPalette.Button, "control"),
                      (QPalette.ButtonText, "text"), (QPalette.Highlight, "accent"),
                      (QPalette.HighlightedText, "on_accent"), (QPalette.ToolTipBase, "tooltip"),
                      (QPalette.ToolTipText, "text"), (QPalette.PlaceholderText, "text3"),
                      (QPalette.Link, "accent"), (QPalette.BrightText, "text"),
                      (QPalette.Light, "stroke_strong"), (QPalette.Midlight, "stroke"),
                      (QPalette.Mid, "stroke_strong"), (QPalette.Dark, "bg_alt"),
                      (QPalette.Shadow, "bg_alt")):
        pal.setColor(role, solid(t[key]))
    for role, key in ((QPalette.WindowText, "text_disabled"), (QPalette.Text, "text_disabled"),
                      (QPalette.ButtonText, "text_disabled")):
        pal.setColor(QPalette.Disabled, role, QColor(t[key]))
    return pal


class _WindowPolish(QObject):
    """Rounded, borderless menus, and every window's title bar in the
    design's colours."""

    def eventFilter(self, obj, ev):
        et = ev.type()
        if et == QEvent.Polish and isinstance(obj, QMenu):
            if not obj.testAttribute(Qt.WA_TranslucentBackground):
                obj.setAttribute(Qt.WA_TranslucentBackground, True)
                obj.setWindowFlags(obj.windowFlags() | Qt.FramelessWindowHint
                                   | Qt.NoDropShadowWindowHint)
        elif et == QEvent.Show and isinstance(obj, QWidget) and obj.isWindow() \
                and not isinstance(obj, QMenu) and not (obj.windowFlags() & Qt.Popup) == Qt.Popup:
            ui_design.native_window(obj, _state["tokens"], is_dark())
        return False


def dark_title_bar(widget, dark=True):
    ui_design.native_window(widget, DARK if dark else LIGHT, dark)


def apply(app=None, mode_name="system"):
    """Apply a mode to the whole application. 'system' follows Windows."""
    app = app or QApplication.instance()
    if mode_name not in ("light", "dark"):
        mode_name = "dark" if system_prefers_dark() else "light"
    t = DARK if mode_name == "dark" else LIGHT
    _state.update(mode=mode_name, tokens=t)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setFont(ui_design.app_font())
    app.setPalette(_palette(t))
    app.setStyleSheet(stylesheet(t, ui_design.write_glyphs(t, mode_name)))
    try:
        from vision_station.ui import icons
        icons.set_ink(t["ink"], t["accent"] if mode_name == "light" else t["accent_text"])
    except Exception:
        pass
    if _state["filter"] is None:
        _state["filter"] = _WindowPolish(app)
        app.installEventFilter(_state["filter"])
        # the design's motion: fades on open and on switch (ui/motion.py)
        try:
            from vision_station.ui import motion
            motion.install(app)
        except Exception as e:
            print("[theme] motion not installed: %s" % e)
    for w in app.topLevelWidgets():
        if w.isWindow() and w.isVisible() and not isinstance(w, QMenu):
            ui_design.native_window(w, t, mode_name == "dark")
    return mode_name


def restyle(widget):
    """Re-evaluate the stylesheet after a dynamic property changed."""
    st = widget.style()
    st.unpolish(widget)
    st.polish(widget)
    widget.update()


def tone_text(tone):
    """The colour for a result written as text -- "Accept" in a table, PASS in a
    tree. PASS, FAIL and WARN are marker colours for dots and borders; as text
    they would not reach WCAG's 4.5:1 on every surface."""
    return color({"pass": "success_text", "fail": "critical_text",
                  "warn": "caution_text"}.get(tone, "text2"))


def set_tone(widget, tone):
    """tone: pass | fail | warn | idle -- read by the stylesheet."""
    if widget.property("tone") != tone:
        widget.setProperty("tone", tone)
        restyle(widget)
