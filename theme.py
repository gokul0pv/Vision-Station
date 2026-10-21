"""Vision Station's look: a modern design-tool style, light and dark.

One set of design tokens per mode, one stylesheet built from them, applied to
the whole application -- QuickBuild, its editors and dialogs, and the classic
window. Widgets never carry colours of their own; they set an object name or a
dynamic property and the stylesheet styles it. Colours that code needs (status
dots, the canvas background) come from color().

    apply(app, "system" | "light" | "dark")
    color("accent")             a token of the current mode
    is_dark()

The shapes are those of current design tools -- Figma, Illustrator, ArtCraft:
8 px rounded controls, 12 px cards and floating toolbars, pill-shaped badges
and tabs, a focus ring in the accent colour, and selection shown as a quiet
tinted fill. The image is the content, so the chrome around it is calm; the
only saturated colours on screen are the accent and the results.

Every text colour is checked against WCAG AA (4.5:1) on the surfaces it sits
on, and white on the accent is 4.9:1.
"""
import os
import sys
import tempfile

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QMenu, QStyleFactory, QWidget

# Results keep one colour in both modes: an operator learns them once. These
# are marker colours -- dots, borders, region outlines. As text, use
# tone_text(), which picks the mode's readable shade.
PASS, FAIL, WARN, IDLE = "#16A34A", "#DC2626", "#F59E0B", "#8D9199"

DARK = {
    "bg": "#17181B", "bg_alt": "#111214", "layer": "#1F2024", "layer_alt": "#232428",
    "flyout": "#26272C", "control": "#2A2B31", "control_hover": "#33353C",
    "control_pressed": "#2D2E34", "control_disabled": "#222327",
    "subtle_hover": "rgba(255,255,255,0.06)", "subtle_pressed": "rgba(255,255,255,0.04)",
    "selected": "rgba(255,255,255,0.09)", "stroke": "#2E3036", "stroke_strong": "#3B3D44",
    "divider": "#2A2B30", "input_bottom": "#4A4D55",
    "text": "#F2F3F5", "text2": "#B4B7BE", "text3": "#8D9199", "text_disabled": "#5B5E66",
    "accent": "#3366F0", "accent_hover": "#4474F2", "accent_pressed": "#2B58D6",
    "on_accent": "#FFFFFF", "accent_subtle": "rgba(51,102,240,0.22)",
    "accent_text": "#8AB4FF", "canvas": "#0E0F11", "ink": "#E4E6EA",
    "tooltip": "#2C2D33", "scroll": "#4A4D55", "scroll_hover": "#62656E",
    "success_bg": "rgba(74,222,128,0.14)", "success_text": "#4ADE80",
    "critical_bg": "rgba(255,138,138,0.14)", "critical_text": "#FF8A8A",
    "caution_bg": "rgba(250,204,21,0.13)", "caution_text": "#FACC15",
    "float": "rgba(38,39,44,0.94)",
}

LIGHT = {
    "bg": "#F5F6F8", "bg_alt": "#ECEEF2", "layer": "#FFFFFF", "layer_alt": "#F8F9FB",
    "flyout": "#FFFFFF", "control": "#FFFFFF", "control_hover": "#F3F4F7",
    "control_pressed": "#ECEEF2", "control_disabled": "#F3F4F6",
    "subtle_hover": "rgba(20,24,35,0.05)", "subtle_pressed": "rgba(20,24,35,0.03)",
    "selected": "rgba(20,24,35,0.07)", "stroke": "#E3E5EA", "stroke_strong": "#D3D6DD",
    "divider": "#E8EAEE", "input_bottom": "#B9BDC6",
    "text": "#16181D", "text2": "#4B5059", "text3": "#646A74", "text_disabled": "#A3A8B1",
    "accent": "#3366F0", "accent_hover": "#4474F2", "accent_pressed": "#2B58D6",
    "on_accent": "#FFFFFF", "accent_subtle": "rgba(51,102,240,0.12)",
    "accent_text": "#2550C8", "canvas": "#1A1C20", "ink": "#2B2F36",
    "tooltip": "#FFFFFF", "scroll": "#C2C6CE", "scroll_hover": "#9EA3AD",
    "success_bg": "#DCFCE7", "success_text": "#15803D",
    "critical_bg": "#FEE2E2", "critical_text": "#C62828",
    "caution_bg": "#FEF3C7", "caution_text": "#A15C00",
    "float": "rgba(255,255,255,0.96)",
}

_state = {"mode": "dark", "tokens": DARK, "filter": None}


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


# ------------------------------------------------------------ assets
# Stylesheets take images only from files, so the few glyphs controls need --
# a check mark, chevrons -- are written once per mode as SVG.
def _assets(t, mode_name):
    folder = os.path.join(tempfile.gettempdir(), "vision_station_theme", mode_name)
    os.makedirs(folder, exist_ok=True)
    stroke = 'fill="none" stroke="%s" stroke-width="%s" stroke-linecap="round" stroke-linejoin="round"'
    files = {
        "check": '<path d="M3.8 8.4l2.7 2.7 5.7-6.2" %s/>' % (stroke % (t["on_accent"], 1.8)),
        "check_ink": '<path d="M3.5 8.5l3 3 6-7" %s/>' % (stroke % (t["text"], 1.6)),
        "dash": '<path d="M4.5 8h7" %s/>' % (stroke % (t["on_accent"], 1.8)),
        "down": '<path d="M4.5 6.2l3.5 3.6 3.5-3.6" %s/>' % (stroke % (t["text2"], 1.4)),
        "up": '<path d="M4.5 9.8l3.5-3.6 3.5 3.6" %s/>' % (stroke % (t["text2"], 1.4)),
        "right": '<path d="M6.2 4.5l3.6 3.5-3.6 3.5" %s/>' % (stroke % (t["text2"], 1.4)),
        "left": '<path d="M9.8 4.5L6.2 8l3.6 3.5" %s/>' % (stroke % (t["text2"], 1.4)),
        "branch_open": '<path d="M4.5 6.2l3.5 3.6 3.5-3.6" %s/>' % (stroke % (t["text3"], 1.4)),
        "branch_closed": '<path d="M6.2 4.5l3.6 3.5-3.6 3.5" %s/>' % (stroke % (t["text3"], 1.4)),
        "close": '<path d="M4.8 4.8l6.4 6.4M11.2 4.8l-6.4 6.4" %s/>' % (stroke % (t["text2"], 1.3)),
        "float": '<rect x="3.5" y="3.5" width="9" height="9" rx="2" %s/>' % (stroke % (t["text2"], 1.1)),
        "radio": '<circle cx="8" cy="8" r="3" fill="%s"/>' % t["on_accent"],
    }
    out = {}
    for name, body in files.items():
        path = os.path.join(folder, name + ".svg")
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" '
               'viewBox="0 0 16 16">%s</svg>' % body)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(svg)
        except OSError:
            pass
        out[name] = path.replace("\\", "/")
    return out


# ------------------------------------------------------------ stylesheet
def stylesheet(t, a):
    return """
/* Fusion's dotted focus rectangle is replaced, not dropped: every control
   that takes keyboard focus shows a ring in the accent colour below. */
* { outline: none; }
QWidget { color: %(text)s; }
QMainWindow, QDialog { background: %(bg)s; }
QMainWindow::separator { background: transparent; width: 6px; height: 6px; }
QWidget:disabled { color: %(text_disabled)s; }

/* ---- menus ---------------------------------------------------------- */
QMenuBar { background: %(bg)s; padding: 4px 8px 0 8px; border: 0; }
QMenuBar::item { background: transparent; padding: 6px 11px; border-radius: 8px; color: %(text2)s; }
QMenuBar::item:selected { background: %(subtle_hover)s; color: %(text)s; }
QMenuBar::item:pressed { background: %(selected)s; color: %(text)s; }
QMenu { background: %(flyout)s; border: 1px solid %(stroke_strong)s; border-radius: 12px;
        padding: 6px; }
QMenu::item { padding: 7px 30px 7px 12px; border-radius: 7px; margin: 1px 0; color: %(text)s;
              background: transparent; }
QMenu::item:selected { background: %(accent_subtle)s; }
QMenu::item:disabled { color: %(text_disabled)s; }
QMenu::icon { padding-left: 8px; }
QMenu::separator { height: 1px; background: %(divider)s; margin: 5px 8px; }
QMenu::indicator { width: 14px; height: 14px; margin-left: 6px; }
QMenu::indicator:checked { image: url(%(a_check_ink)s); }
QMenu::right-arrow { image: url(%(a_right)s); width: 12px; height: 12px; margin-right: 8px; }

/* ---- command bars --------------------------------------------------- */
QToolBar { background: %(bg)s; border: 0; padding: 6px 10px; spacing: 4px; }
QToolBar::separator { background: %(divider)s; width: 1px; margin: 7px 6px; }
QToolButton { background: transparent; border: 1px solid transparent; border-radius: 8px;
              padding: 6px 10px; color: %(text)s; }
QToolButton:hover { background: %(subtle_hover)s; }
QToolButton:pressed { background: %(subtle_pressed)s; color: %(text2)s; }
QToolButton:checked { background: %(accent_subtle)s; color: %(accent_text)s; }
QToolButton:disabled { color: %(text_disabled)s; }
QToolButton:focus { border-color: %(accent)s; }
QToolButton#accentButton { background: %(accent)s; color: %(on_accent)s; border-color: %(accent)s;
                           font-weight: 600; padding: 6px 16px; border-radius: 10px; }
QToolButton#accentButton:hover { background: %(accent_hover)s; border-color: %(accent_hover)s; }
QToolButton#accentButton:pressed { background: %(accent_pressed)s; }
QToolButton#accentButton:focus { border: 1px solid %(text)s; }
QToolButton#accentButton:disabled { background: %(control_disabled)s; color: %(text_disabled)s;
                                    border-color: %(stroke)s; }
QToolButton#runningButton { border-radius: 10px; }
QToolButton#runningButton:checked { background: %(critical_bg)s; color: %(critical_text)s;
                                    border-color: transparent; font-weight: 600; }
QToolButton::menu-indicator { image: url(%(a_down)s); width: 10px; height: 10px;
                              subcontrol-position: right center; right: 4px; }

/* ---- floating toolbars over the image, as design tools have them ---- */
QFrame#floatBar { background: %(float)s; border: 1px solid %(stroke_strong)s; border-radius: 12px; }
QFrame#floatBar QToolButton { padding: 4px; border-radius: 7px; border: 1px solid transparent; }
QFrame#floatBar QToolButton:hover { background: %(subtle_hover)s; }
QFrame#floatBar QToolButton:checked { background: %(accent)s; color: %(on_accent)s; }
QFrame#floatBar QToolButton[toggle="true"]:checked { background: %(accent_subtle)s; color: %(accent_text)s; }
QFrame#floatBar QToolButton:focus { border-color: %(accent)s; }
QFrame#floatBar QToolBar { background: transparent; border: 0; padding: 0; spacing: 2px; }
QFrame#floatBar QToolBar::separator { background: %(divider)s; height: 1px; width: 1px; margin: 4px 4px; }
QFrame#floatBar QLabel { color: %(text2)s; background: transparent; }
QFrame#floatBar QLabel#floatValue { color: %(text)s; font-family: 'Cascadia Mono', Consolas, monospace; }
QFrame#floatBar QLabel#floatCaption { color: %(text3)s; font-size: 8pt; font-weight: 600; }
QFrame#floatBar QComboBox { border-radius: 8px; padding: 3px 24px 3px 8px; min-height: 16px;
                            background: transparent; border: 1px solid transparent; }
QFrame#floatBar QComboBox:hover { background: %(subtle_hover)s; }
QFrame#floatBar QAbstractSpinBox { border-radius: 6px; padding: 2px 4px; min-height: 16px;
                                   background: %(control)s; border: 1px solid %(stroke)s; }
QFrame#floatBar QAbstractSpinBox:focus { border: 1px solid %(accent)s; padding: 2px 4px; }
QFrame#floatBar QAbstractSpinBox::up-button, QFrame#floatBar QAbstractSpinBox::down-button { width: 0; }
QFrame#floatSep { background: %(divider)s; min-width: 1px; max-width: 1px; margin: 4px 2px; }

/* ---- docks: titled cards -------------------------------------------- */
QDockWidget { color: %(text2)s; titlebar-close-icon: url(%(a_close)s);
              titlebar-normal-icon: url(%(a_float)s); }
QDockWidget::title { background: %(bg)s; padding: 8px 10px 6px 12px; text-align: left; }
QDockWidget::close-button, QDockWidget::float-button { border: 0; border-radius: 6px;
              background: transparent; padding: 2px; icon-size: 12px; }
QDockWidget::close-button:hover, QDockWidget::float-button:hover { background: %(subtle_hover)s; }
/* Panels are told apart by their surface, not by outlines: a border round
   every panel, inside another bordered panel, was most of the clutter. */
QDockWidget > QWidget { background: %(bg)s; border: 0; }
QWidget#panel { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 12px; }
QWidget#panelBody { background: transparent; border: 0; }

/* ---- lists, trees, tables ------------------------------------------- */
QAbstractItemView { background: %(layer)s; alternate-background-color: %(layer_alt)s;
                    border: 1px solid %(stroke)s; border-radius: 10px; color: %(text)s;
                    selection-background-color: %(accent_subtle)s; selection-color: %(text)s; }
QDockWidget QAbstractItemView, QWidget#panel QAbstractItemView { border: 0; border-radius: 0;
                    background: transparent; }
QAbstractItemView:focus { border-color: %(accent)s; }
QAbstractItemView::item { padding: 4px 6px; border-radius: 6px; }
QTreeView::item, QListView::item { min-height: 26px; }
QAbstractItemView::item:hover { background: %(subtle_hover)s; }
QAbstractItemView::item:selected { background: %(accent_subtle)s; color: %(text)s; }
QTreeView { show-decoration-selected: 0; }
QTreeView::branch { background: transparent; }
QTreeView::branch:has-children:!has-siblings:closed, QTreeView::branch:closed:has-children:has-siblings
    { image: url(%(a_branch_closed)s); }
QTreeView::branch:open:has-children:!has-siblings, QTreeView::branch:open:has-children:has-siblings
    { image: url(%(a_branch_open)s); }
QTableView { gridline-color: %(divider)s; }
QTableView::item { border-radius: 0; padding: 3px 8px; }
QHeaderView { background: transparent; border: 0; }
QHeaderView::section { background: transparent; color: %(text3)s; border: 0;
                       border-bottom: 1px solid %(divider)s; padding: 7px 8px; font-weight: 600; }
QHeaderView::section:hover { background: %(subtle_hover)s; }
QTableCornerButton::section { background: transparent; border: 0; }

/* ---- tabs: pills ----------------------------------------------------- */
QTabWidget::pane { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 12px; top: -1px; }
QTabBar { qproperty-drawBase: 0; background: transparent; }
QTabBar::tab { background: transparent; color: %(text2)s; padding: 6px 14px; margin: 3px 2px;
               border: 1px solid transparent; border-radius: 8px; }
QTabBar::tab:hover { background: %(subtle_hover)s; color: %(text)s; }
QTabBar::tab:selected { background: %(selected)s; color: %(text)s; font-weight: 600; }
QTabBar:focus { border-bottom: 1px solid %(accent)s; }
QTabBar::close-button { image: url(%(a_close)s); subcontrol-position: right; border-radius: 5px;
                        margin: 2px; }
QTabBar::close-button:hover { background: %(subtle_hover)s; }
QTabBar::scroller { width: 28px; }
QTabBar QToolButton { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 8px; padding: 0; }
QTabBar QToolButton:hover { background: %(control_hover)s; }
QTabBar QToolButton::right-arrow { image: url(%(a_right)s); width: 12px; height: 12px; }
QTabBar QToolButton::left-arrow { image: url(%(a_left)s); width: 12px; height: 12px; }
QTabBar#mdiTabs::tab { padding: 6px 14px 6px 12px; min-width: 90px; max-width: 280px; margin: 5px 4px 5px 0;
                       border-radius: 9px; color: %(text2)s; }
QTabBar#mdiTabs::tab:selected { background: %(layer)s; border: 1px solid %(stroke)s; color: %(text)s; }

/* ---- inputs --------------------------------------------------------- */
QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit, QTextEdit {
    background: %(control)s; border: 1px solid %(stroke_strong)s;
    border-radius: 8px; padding: 5px 9px; min-height: 18px; color: %(text)s;
    selection-background-color: %(accent)s; selection-color: %(on_accent)s; }
QPlainTextEdit, QTextEdit { padding: 6px; }
QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover { background: %(control_hover)s;
    border-color: %(input_bottom)s; }
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {
    background: %(control)s; border: 1px solid %(accent)s; }
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {
    background: %(control_disabled)s; color: %(text_disabled)s; border-color: %(stroke)s; }
QLineEdit#searchBox { padding-left: 10px; border-radius: 10px; }
QLineEdit[invalid="true"] { border: 1px solid %(critical_text)s; }
QComboBox { padding-right: 28px; }
QComboBox::drop-down { border: 0; width: 28px; subcontrol-origin: padding; subcontrol-position: right center; }
QComboBox::down-arrow { image: url(%(a_down)s); width: 12px; height: 12px; }
QComboBox QAbstractItemView { background: %(flyout)s; border: 1px solid %(stroke_strong)s;
                              border-radius: 10px; padding: 4px; outline: 0; }
QComboBox QAbstractItemView::item { min-height: 28px; padding: 0 8px; border-radius: 6px; }
QAbstractSpinBox { padding-right: 24px; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { border: 0; width: 20px; background: transparent;
    border-radius: 5px; margin: 2px 3px; subcontrol-origin: border; }
QAbstractSpinBox::up-button { subcontrol-position: top right; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: %(subtle_hover)s; }
QAbstractSpinBox::up-arrow { image: url(%(a_up)s); width: 10px; height: 10px; }
QAbstractSpinBox::down-arrow { image: url(%(a_down)s); width: 10px; height: 10px; }

/* ---- buttons -------------------------------------------------------- */
QPushButton { background: %(control)s; border: 1px solid %(stroke_strong)s;
              border-radius: 8px; padding: 6px 16px; min-height: 18px; color: %(text)s; font-weight: 500; }
QPushButton:hover { background: %(control_hover)s; }
QPushButton:pressed { background: %(control_pressed)s; color: %(text2)s; }
QPushButton:disabled { background: %(control_disabled)s; color: %(text_disabled)s; border-color: %(stroke)s; }
QPushButton:default, QPushButton[accent="true"] { background: %(accent)s; color: %(on_accent)s;
              border: 1px solid %(accent)s; font-weight: 600; }
QPushButton:default:hover, QPushButton[accent="true"]:hover { background: %(accent_hover)s;
              border-color: %(accent_hover)s; }
QPushButton:default:pressed, QPushButton[accent="true"]:pressed { background: %(accent_pressed)s; }
QPushButton:checked { background: %(accent_subtle)s; border-color: %(accent)s; color: %(accent_text)s; }
QPushButton:focus { border: 1px solid %(accent)s; }
QPushButton:default:focus, QPushButton[accent="true"]:focus { border: 1px solid %(text)s; }
QPushButton[flat="true"] { background: transparent; border-color: transparent; }
QPushButton[flat="true"]:hover { background: %(subtle_hover)s; }

QCheckBox, QRadioButton { spacing: 9px; color: %(text)s; background: transparent; }
QCheckBox::indicator, QRadioButton::indicator { width: 16px; height: 16px; background: %(control)s;
              border: 1px solid %(input_bottom)s; }
QCheckBox::indicator { border-radius: 5px; }
QRadioButton::indicator { border-radius: 9px; }
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: %(accent)s; }
QCheckBox::indicator:checked { background: %(accent)s; border-color: %(accent)s; image: url(%(a_check)s); }
QCheckBox::indicator:indeterminate { background: %(accent)s; border-color: %(accent)s; image: url(%(a_dash)s); }
QRadioButton::indicator:checked { background: %(accent)s; border-color: %(accent)s; image: url(%(a_radio)s); }
QCheckBox::indicator:focus, QRadioButton::indicator:focus { border: 1px solid %(accent)s; }
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled { border-color: %(text_disabled)s;
              background: %(control_disabled)s; }

QGroupBox { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 12px;
            margin-top: 24px; padding: 12px 10px 10px 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 6px; top: 3px; color: %(text2)s; font-weight: 600; }

QSlider::groove:horizontal { height: 4px; background: %(input_bottom)s; border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(accent)s; border-radius: 2px; }
QSlider::handle:horizontal { background: %(accent)s; border: 4px solid %(control)s; width: 12px;
                             height: 12px; margin: -8px 0; border-radius: 10px; }
QProgressBar { background: %(stroke)s; border: 0; border-radius: 3px; height: 6px; text-align: center;
               color: transparent; }
QProgressBar::chunk { background: %(accent)s; border-radius: 3px; }

/* ---- scroll bars: thin, rounded ------------------------------------- */
QScrollBar { background: transparent; border: 0; }
QScrollBar:vertical { width: 10px; margin: 2px; }
QScrollBar:horizontal { height: 10px; margin: 2px; }
QScrollBar::handle { background: %(scroll)s; border-radius: 3px; }
QScrollBar::handle:hover { background: %(scroll_hover)s; }
QScrollBar::handle:vertical { min-height: 32px; margin: 0 2px; }
QScrollBar::handle:horizontal { min-width: 32px; margin: 2px 0; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; border: 0; background: transparent; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ---- splitters, status, tooltips ----------------------------------- */
QSplitter::handle { background: transparent; }
QSplitter::handle:horizontal { width: 6px; }
QSplitter::handle:vertical { height: 6px; }
QSplitter::handle:hover { background: %(accent_subtle)s; border-radius: 3px; }
QStatusBar { background: %(bg)s; color: %(text2)s; border: 0; min-height: 28px; }
QStatusBar::item { border: 0; }
QStatusBar QLabel { color: %(text2)s; padding: 0 6px; }
QStatusBar QLabel[tone="pass"] { color: %(success_text)s; }
QStatusBar QLabel[tone="fail"] { color: %(critical_text)s; }
QStatusBar QLabel[tone="warn"] { color: %(caution_text)s; }
QToolTip { background: %(tooltip)s; color: %(text)s; border: 1px solid %(stroke_strong)s;
           border-radius: 8px; padding: 6px 9px; }
QMdiArea { background: %(bg_alt)s; border: 0; }
QMdiSubWindow { background: %(bg)s; border: 1px solid %(stroke_strong)s; }
QMdiSubWindow > QWidget { background: %(bg)s; }
QDialogButtonBox QPushButton { min-width: 84px; }
QMessageBox QLabel { color: %(text)s; }

/* ---- Vision Station pieces ----------------------------------------- */
QLabel#caption { color: %(text3)s; font-size: 8pt; font-weight: 600; letter-spacing: 0.6px; }
QLabel#secondary, QLabel[secondary="true"] { color: %(text2)s; }
QLabel#hint { color: %(text3)s; }
QLabel#sectionTitle { color: %(text)s; font-weight: 600; font-size: 10.5pt; padding: 2px 0 4px 0; }
QLabel#pageTitle { color: %(text)s; font-weight: 700; font-size: 17pt; }
QLabel#mono { font-family: 'Cascadia Mono', Consolas, monospace; color: %(text2)s; }

/* production numbers as plain figures, not four boxed cards */
QFrame#kpi { background: transparent; border: 0; }
QFrame#kpi QLabel#caption { font-size: 8pt; font-weight: 600; letter-spacing: 0; }
QFrame#kpi QLabel#kpiValue { font-size: 13.5pt; font-weight: 600; color: %(text)s; }
QFrame#kpi[tone="pass"] QLabel#kpiValue { color: %(success_text)s; }
QFrame#kpi[tone="fail"] QLabel#kpiValue { color: %(critical_text)s; }

QLabel[badge="true"] { border-radius: 10px; padding: 3px 11px; font-weight: 700; font-size: 8.5pt;
                       letter-spacing: 0.5px; }
QLabel[badge="true"][tone="idle"] { background: %(selected)s; color: %(text2)s; }
QLabel[badge="true"][tone="pass"] { background: %(success_bg)s; color: %(success_text)s; }
QLabel[badge="true"][tone="fail"] { background: %(critical_bg)s; color: %(critical_text)s; }
QLabel[badge="true"][tone="warn"] { background: %(caution_bg)s; color: %(caution_text)s; }

QFrame#tile { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 14px; }
QFrame#tile[tone="pass"] { border: 2px solid %(pass)s; }
QFrame#tile[tone="fail"] { border: 2px solid %(fail)s; }
QFrame#tile[tone="warn"] { border: 2px solid %(warn)s; }
QWidget#tileHead { background: transparent; }
QLabel#tileName { font-weight: 700; font-size: 10.5pt; }

QWidget#displayBar { background: %(layer)s; border-bottom: 1px solid %(divider)s; }
QWidget#displayBar QToolButton { padding: 4px 6px; }
QWidget#displayStatus { background: transparent; }
QWidget#displayStatus QLabel { color: %(text2)s; font-family: 'Cascadia Mono', Consolas, monospace; }
QWidget#displayStatus QComboBox { padding: 1px 22px 1px 6px; min-height: 16px; }

QLabel#sheetHeader { background: transparent; color: %(text)s; font-weight: 700; font-size: 11.5pt;
                     padding: 12px 14px 2px 14px; border: 0; }
QLabel#sheetDesc { background: transparent; color: %(text2)s; padding: 0 14px 10px 14px; border: 0; }

QWidget#welcome, QWidget#runGrid { background: %(bg_alt)s; }
QWidget#dockTitle { background: %(bg)s; }
QLabel#dockCaption { color: %(text3)s; font-weight: 600; font-size: 9pt; }
/* the sidebar: a source list under quiet section headings */
QLabel#sidebarSection { color: %(text3)s; font-weight: 600; font-size: 8.5pt; padding: 10px 8px 4px 8px; }
QAbstractItemView#sidebarList { background: transparent; border: 0; }
QAbstractItemView#sidebarList::item { min-height: 30px; padding: 2px 8px; border-radius: 8px; margin: 1px 0; }
QAbstractItemView#sidebarList::item:selected { background: %(accent_subtle)s; color: %(text)s; }
QAbstractItemView#sidebarList:focus { border: 0; }
QPushButton#linkButton { background: transparent; border: 0; color: %(accent_text)s; font-weight: 600;
                         padding: 4px 6px; text-align: left; }
QPushButton#linkButton:hover { background: %(subtle_hover)s; }
QPushButton#linkButton:focus { border: 1px solid %(accent)s; }
QToolButton#dockClose { padding: 4px; border-radius: 6px; }
QScrollArea#tabScroll { background: transparent; border: 0; }
QScrollArea#tabScroll > QWidget > QWidget { background: transparent; }
QFrame#welcomeCard { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 16px; }
QLabel#trainHead { border-radius: 8px; padding: 4px; font-weight: 700; color: white; }
QLabel#trainHead[tone="pass"] { background: %(pass)s; }
QLabel#trainHead[tone="fail"] { background: %(fail)s; }
QLabel[tone="pass"]#report { color: %(success_text)s; }
QLabel[tone="fail"]#report { color: %(critical_text)s; }
QLabel[tone="warn"]#report { color: %(caution_text)s; }
""" % dict(t, **{"pass": PASS, "fail": FAIL, "warn": WARN},
           **{"a_" + k: v for k, v in a.items()})


def _palette(t):
    pal = QPalette()
    solid = lambda c: QColor(c) if not c.startswith("rgba") else QColor(t["layer"])
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


# ------------------------------------------------------------ native touches
class _WindowPolish(QObject):
    """Rounded, borderless flyout menus, and a title bar that matches the
    mode: Windows draws a light caption on a dark window unless asked not to."""

    def eventFilter(self, obj, ev):
        et = ev.type()
        if et == QEvent.Polish and isinstance(obj, QMenu):
            if not obj.testAttribute(Qt.WA_TranslucentBackground):
                obj.setAttribute(Qt.WA_TranslucentBackground, True)
                obj.setWindowFlags(obj.windowFlags() | Qt.FramelessWindowHint
                                   | Qt.NoDropShadowWindowHint)
        elif et == QEvent.Show and isinstance(obj, QWidget) and obj.isWindow() \
                and not isinstance(obj, QMenu) and not (obj.windowFlags() & Qt.Popup) == Qt.Popup:
            dark_title_bar(obj, is_dark())
        return False


def dark_title_bar(widget, dark=True):
    """Windows 10 1809+ and 11: DWMWA_USE_IMMERSIVE_DARK_MODE (20; 19 on the
    earliest builds). Silently nothing elsewhere."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = int(widget.winId())
        val = ctypes.c_int(1 if dark else 0)
        for attr in (20, 19):
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(val),
                                                          ctypes.sizeof(val)) == 0:
                break
    except Exception:
        pass


def _font():
    f = QFont()
    from PySide6.QtGui import QFontDatabase
    fams = set(QFontDatabase.families())
    for fam in ("Segoe UI Variable Text", "Inter", "Segoe UI", "Noto Sans"):
        if fam in fams:
            f.setFamily(fam)
            break
    f.setPointSizeF(9.75)
    return f


def apply(app=None, mode_name="system"):
    """Apply a mode to the whole application. 'system' follows Windows."""
    app = app or QApplication.instance()
    if mode_name not in ("light", "dark"):
        mode_name = "dark" if system_prefers_dark() else "light"
    t = DARK if mode_name == "dark" else LIGHT
    _state.update(mode=mode_name, tokens=t)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setFont(_font())
    app.setPalette(_palette(t))
    a = _assets(t, mode_name)
    app.setStyleSheet(stylesheet(t, a))
    try:
        import icons
        icons.set_ink(t["ink"], t["accent"] if mode_name == "light" else t["accent_text"])
    except Exception:
        pass
    if _state["filter"] is None:
        _state["filter"] = _WindowPolish(app)
        app.installEventFilter(_state["filter"])
    for w in app.topLevelWidgets():
        if w.isWindow() and w.isVisible() and not isinstance(w, QMenu):
            dark_title_bar(w, mode_name == "dark")
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
    they missed WCAG's 4.5:1 (green on white 3.4:1, red on the dark theme 2.7:1)."""
    return color({"pass": "success_text", "fail": "critical_text",
                  "warn": "caution_text"}.get(tone, "text2"))


def set_tone(widget, tone):
    """tone: pass | fail | warn | idle -- read by the stylesheet."""
    if widget.property("tone") != tone:
        widget.setProperty("tone", tone)
        restyle(widget)
