"""Motion: how the application moves. The look is in design.py; this is the
other half of the "Fluid" design language -- things ease in instead of
appearing, and controls glide to where they are going.

    install(app)        once, from theme.apply(): watches the application
                        and animates what opens and what switches
    enabled()           False when Windows' "Animation effects" is off, or
                        VS_MOTION=0 -- then everything appears at once
    fade_in(widget)     a widget's contents ease in (opacity), once
    glide(widget, rect) a child widget slides to a rectangle on a spring

What moves, and nothing else changes -- no widget is created, resized or
re-ordered by this module except through those calls:
    dialogs and windows   fade in as they open
    pages and views       fade in when a stack, a tab or a document switches
    menus, drop-down lists, tooltips   Qt's own fades, switched on
    lists and tables      scroll by pixels, not by whole rows
Each animation is short (under a quarter of a second): an operator waiting
on a verdict never waits for an animation.
"""
import os
import sys

from PySide6.QtCore import (QAbstractAnimation, QEasingCurve, QEvent, QObject, QPropertyAnimation,
                            QRect, Qt, QTimer, QVariantAnimation)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDialog, QGraphicsOpacityEffect,
                               QMdiArea, QMenu, QStackedWidget, QTabWidget, QWidget)

FADE_MS = 200
WINDOW_MS = 180
GLIDE_MS = 380

_state = {"filter": None, "enabled": None}


def _windows_animations_on():
    """Windows' own setting (Settings > Accessibility > Visual effects >
    Animation effects). Off there, off here."""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        val = ctypes.c_int(1)
        SPI_GETCLIENTAREAANIMATION = 0x1042
        if ctypes.windll.user32.SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION, 0,
                                                      ctypes.byref(val), 0):
            return bool(val.value)
    except Exception:
        pass
    return True


def enabled():
    if _state["enabled"] is None:
        _state["enabled"] = (os.environ.get("VS_MOTION", "1") != "0" and
                             os.environ.get("QT_QPA_PLATFORM", "") != "offscreen" and
                             _windows_animations_on())
    return _state["enabled"]


def spring():
    """The easing every glide uses: overshoots a little, then settles."""
    c = QEasingCurve(QEasingCurve.OutBack)
    c.setOvershoot(1.15)
    return c


def fade_in(widget, ms=FADE_MS):
    """The widget's contents ease in from transparent. Skipped for a widget
    that already carries an effect of its own (a drop shadow), and the
    effect is removed when done: an opacity effect renders the widget
    off-screen, which nothing should pay for once it is still."""
    if not enabled() or widget is None or not widget.isVisible() or widget.graphicsEffect():
        return
    if widget.property("_vsFading"):
        return
    eff = QGraphicsOpacityEffect(widget)
    eff.setOpacity(0.0)
    widget.setGraphicsEffect(eff)
    widget.setProperty("_vsFading", True)
    anim = QPropertyAnimation(eff, b"opacity", widget)
    anim.setDuration(ms)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.OutCubic)

    def done():
        widget.setProperty("_vsFading", False)
        if widget.graphicsEffect() is eff:
            widget.setGraphicsEffect(None)
    anim.finished.connect(done)
    anim.start(QAbstractAnimation.DeleteWhenStopped)


def window_in(win, ms=WINDOW_MS):
    """A window or dialog fades in as it opens."""
    if not enabled() or win.property("_vsShown"):
        return
    win.setProperty("_vsShown", True)
    win.setWindowOpacity(0.0)
    anim = QPropertyAnimation(win, b"windowOpacity", win)
    anim.setDuration(ms)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.OutCubic)
    anim.finished.connect(lambda: win.setWindowOpacity(1.0))
    anim.start(QAbstractAnimation.DeleteWhenStopped)


def glide(widget, rect, ms=GLIDE_MS):
    """A child widget slides (and resizes) to rect on a spring; at once when
    motion is off or the widget is not on screen yet."""
    old = getattr(widget, "_vsGlide", None)
    if old is not None:
        old.stop()
    if not enabled() or not widget.isVisible() or widget.geometry().isEmpty():
        widget.setGeometry(rect)
        return
    anim = QPropertyAnimation(widget, b"geometry", widget)
    anim.setDuration(ms)
    anim.setStartValue(widget.geometry())
    anim.setEndValue(QRect(rect))
    anim.setEasingCurve(spring())
    widget._vsGlide = anim
    anim.start()


class _Motion(QObject):
    """Watches the application: stacks, tabs and document areas get a fade
    on every switch; windows fade in; lists scroll smoothly."""

    def eventFilter(self, obj, ev):
        et = ev.type()
        if et == QEvent.Polish and isinstance(obj, QWidget) and not obj.property("_vsMotion"):
            obj.setProperty("_vsMotion", True)
            if isinstance(obj, QStackedWidget):
                obj.currentChanged.connect(lambda _i, s=obj: fade_in(s.currentWidget()))
            elif isinstance(obj, QTabWidget):
                obj.currentChanged.connect(lambda _i, s=obj: fade_in(s.currentWidget()))
            elif isinstance(obj, QMdiArea):
                obj.subWindowActivated.connect(
                    lambda sub: sub is not None and fade_in(sub.widget()))
            elif isinstance(obj, QAbstractItemView):
                # scrolled by pixels: a long list glides instead of jumping a row
                obj.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
                obj.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
                obj.verticalScrollBar().setSingleStep(18)
        elif et == QEvent.Show and isinstance(obj, QDialog) and obj.isWindow():
            window_in(obj)
        return False


def install(app=None):
    """Once per application: Qt's own menu, list and tooltip fades, and the
    watcher above."""
    app = app or QApplication.instance()
    if app is None or _state["filter"] is not None:
        return
    on = enabled()
    for effect in (Qt.UI_AnimateMenu, Qt.UI_FadeMenu, Qt.UI_AnimateCombo,
                   Qt.UI_AnimateTooltip, Qt.UI_FadeTooltip, Qt.UI_AnimateToolBox):
        QApplication.setEffectEnabled(effect, on)
    if not on:
        return
    _state["filter"] = _Motion(app)
    app.installEventFilter(_state["filter"])


def switch_animation(owner, update):
    """A 0..1 value animator for a painted control (the switch): returns
    start(target) -- call it with 0 or 1; update(value) is called each frame."""
    anim = QVariantAnimation(owner)
    anim.setDuration(GLIDE_MS - 80)
    anim.setEasingCurve(spring())
    anim.valueChanged.connect(lambda v: update(float(v)))

    def start(target, current):
        anim.stop()
        if not enabled() or not owner.isVisible():
            update(float(target))
            return
        anim.setStartValue(float(current))
        anim.setEndValue(float(target))
        anim.start()
    return start


def later(ms, fn):
    QTimer.singleShot(ms, fn)
