"""Vision Station's design -- the ONE file that decides how the application looks.

Edit this file to change the look. Nothing here changes what the application
does: windows and dialogs only name their parts (an object name such as
"verdict", or a property such as tone="fail"), and this file says how each
named part is drawn. So colours, fonts, sizes, corner radii and spacing can
all be changed here, and every window follows.

    1. FONTS        which typeface, at what size
    2. COLOURS      the light and the dark palette (macOS system colours)
    3. METRICS      corner radii, control heights, padding
    4. STYLESHEET   how every control and every named part is drawn
    5. CONTROLS     the few macOS controls Qt does not have (a switch, grouped
                    form sections), drawn from the same colours
    6. WINDOW       the title bar coloured to match the toolbar (Windows 11)

For a quick experiment without touching Python, put Qt stylesheet rules in a
file called ui_custom.qss beside this one: it is applied after everything here,
so its rules win. Delete the file to go back.

The look is "Fluid" -- the same design language as Vision Glass: a soft
grey page with large, well-rounded white cards on it, one strong blue,
filled controls instead of outlined ones, selections tinted with the blue,
and motion that eases in rather than snapping (ui/motion.py: windows and
pages fade in, the view switch slides, switches spring). Results keep their
green / red / orange.

The font: Apple's SF Pro is licensed for Apple platforms only and may not be
shipped inside a Windows program, so the application ships Inter (SIL Open
Font License, assets/fonts), the typeface closest to SF Pro in design. If SF
Pro is installed on the PC it is used instead -- it comes first in FONTS.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
from vision_station import paths  # noqa: E402

# =====================================================================
# 1. FONTS
# =====================================================================
FONTS = ["Inter", "SF Pro Text", "SF Pro", "Segoe UI Variable Text", "Segoe UI"]
MONO_FONTS = ["SF Mono", "Cascadia Mono", "Consolas"]
FONT_DIR = os.path.join(paths.ASSETS, "fonts")   # bundled .ttf/.otf files
BASE_PT = 9.75          # body text, points: 13 px at 100 % -- macOS's body size

# Text styles, relative to the body size (points).
TYPE = {
    "caption": 8.0,       # small headings: section captions, figure labels
    "small": 8.75,
    "body": BASE_PT,
    "callout": 10.5,      # card and section titles
    "title3": 12.0,       # settings page titles
    "title2": 14.0,
    "title1": 17.0,       # page titles
    "figure": 14.0,       # production numbers in the toolbar
    "verdict": 30.0,      # ACCEPT / REJECT, readable across the room
}

# =====================================================================
# 2. COLOURS
# =====================================================================
# Results keep one colour in both modes -- an operator learns them once. These
# are marker colours (dots, outlines on the picture); as text the palettes'
# success_text / critical_text / caution_text are used, which read on their
# background.
STATUS = {"pass": "#28C840", "fail": "#FF3B30", "warn": "#FF9F0A", "idle": "#8E8E93"}

LIGHT = {
    # surfaces, back to front
    "bg": "#F1F3F5",              # the page: window chrome, toolbar, status bar, dialogs
    "bg_alt": "#ECEEF1",          # content wells: behind the run page's cards
    "sidebar": "#F1F3F5",         # source lists on the left
    "layer": "#FFFFFF",           # cards, lists, tables, grouped sections
    "layer_alt": "#F7F8FA",       # alternate table rows
    "flyout": "#FFFFFF",          # menus, popups
    "float": "rgba(255,255,255,0.93)",   # floating toolbars over the picture
    "canvas": "#101216",          # behind the image
    # controls: filled, not outlined -- a tint that reads on grey and on white
    "control": "rgba(0,0,0,0.05)", "control_hover": "rgba(0,0,0,0.08)",
    "control_pressed": "rgba(0,0,0,0.12)", "control_disabled": "rgba(0,0,0,0.03)",
    "subtle_hover": "rgba(0,0,0,0.05)", "subtle_pressed": "rgba(0,0,0,0.09)",
    "selected": "rgba(10,89,247,0.10)", "segment_bg": "rgba(0,0,0,0.06)",
    # lines
    "stroke": "#E6E8EB", "stroke_strong": "#D3D7DC", "divider": "#E6E8EB",
    "input_bottom": "#BFC4CA",
    # text
    "text": "#1A1A1A", "text2": "#5C5C5C", "text3": "#737373", "text_disabled": "#B0B0B0",
    # accent
    "accent": "#0A59F7", "accent_hover": "#2B6EF8", "accent_pressed": "#0847C6",
    "on_accent": "#FFFFFF", "accent_subtle": "rgba(10,89,247,0.10)", "accent_text": "#0A59F7",
    "ink": "#333333",             # icon strokes
    "tooltip": "#FFFFFF", "scroll": "rgba(0,0,0,0.22)", "scroll_hover": "rgba(0,0,0,0.40)",
    # status surfaces
    "success_bg": "rgba(40,200,64,0.12)", "success_text": "#1E7F34",
    "critical_bg": "rgba(232,64,38,0.10)", "critical_text": "#C4161C",
    "caution_bg": "rgba(237,111,33,0.13)", "caution_text": "#A15200",
}

DARK = {
    "bg": "#121316", "bg_alt": "#0C0D0F", "sidebar": "#121316",
    "layer": "#1C1D20", "layer_alt": "#212226", "flyout": "#232428",
    "float": "rgba(30,31,35,0.94)", "canvas": "#07080A",
    "control": "rgba(255,255,255,0.08)", "control_hover": "rgba(255,255,255,0.12)",
    "control_pressed": "rgba(255,255,255,0.16)", "control_disabled": "rgba(255,255,255,0.04)",
    "subtle_hover": "rgba(255,255,255,0.07)", "subtle_pressed": "rgba(255,255,255,0.11)",
    "selected": "rgba(49,122,247,0.22)", "segment_bg": "rgba(255,255,255,0.08)",
    "stroke": "#2A2B2F", "stroke_strong": "#3B3D42", "divider": "#26272B",
    "input_bottom": "#56585E",
    "text": "#EDEDEE", "text2": "#B4B4B8", "text3": "#8E8E93", "text_disabled": "#55565A",
    "accent": "#317AF7", "accent_hover": "#4A8BFF", "accent_pressed": "#2563D6",
    "on_accent": "#FFFFFF", "accent_subtle": "rgba(49,122,247,0.22)", "accent_text": "#7AA7FF",
    "ink": "#DADADC",
    "tooltip": "#2A2B2F", "scroll": "rgba(255,255,255,0.26)", "scroll_hover": "rgba(255,255,255,0.45)",
    "success_bg": "rgba(48,209,88,0.15)", "success_text": "#4CD964",
    "critical_bg": "rgba(255,69,58,0.16)", "critical_text": "#FF7A66",
    "caution_bg": "rgba(255,159,10,0.15)", "caution_text": "#FFB340",
}


# =====================================================================
# 3. METRICS (pixels)
# =====================================================================
METRICS = {
    "r_control": 10,      # buttons, fields, menus' rows
    "r_card": 16,         # cards, grouped sections, lists
    "r_large": 22,        # the verdict, popups, floating toolbars
    "pad_control": "4px 11px",
    "pad_button": "4px 16px",
    "control_min_h": 20,
    "row_h": 30,          # list and tree rows
    "sidebar_row_h": 32,
}


# =====================================================================
# 4. STYLESHEET
# =====================================================================
# %(name)s is a colour from the palette, a font size from TYPE (t_name), a
# metric (m_name) or a drawn glyph (a_name: check marks and chevrons, written
# as small SVG files at start-up because stylesheets read images from files).
STYLESHEET = r"""
* { outline: none; }
QWidget { color: %(text)s; }
QWidget:disabled { color: %(text_disabled)s; }
QMainWindow, QDialog { background: %(bg)s; }
/* panels float as cards on one page: the gaps between them are the page */
QMainWindow::separator { background: transparent; width: 8px; height: 8px; }
QMainWindow::separator:hover { background: %(accent_subtle)s; }
QWidget#dockWell { background: transparent; }
QFrame#dockCard { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 22px; }
QFrame#dockCard QAbstractItemView { background: transparent; border: 0; }
QFrame#dockCard QSplitter::handle { background: transparent; }

/* ---------------------------------------------------------------- menus */
QMenuBar { background: %(bg)s; padding: 2px 6px; border: 0; border-bottom: 1px solid %(divider)s; }
QMenuBar::item { background: transparent; padding: 4px 10px; border-radius: %(m_r_control)spx; }
QMenuBar::item:selected { background: %(subtle_hover)s; }
QMenuBar::item:pressed { background: %(selected)s; }
QMenu { background: %(flyout)s; border: 1px solid %(stroke)s; border-radius: 16px;
        padding: 6px; }
QMenu::item { padding: 7px 30px 7px 12px; border-radius: 10px; margin: 0; color: %(text)s;
              background: transparent; }
QMenu::item:selected { background: %(accent_subtle)s; color: %(accent_text)s; }
QMenu::item:disabled { color: %(text_disabled)s; background: transparent; }
QMenu::icon { padding-left: 6px; }
QMenu::separator { height: 1px; background: %(divider)s; margin: 6px 12px; }
QMenu::indicator { width: 14px; height: 14px; margin-left: 4px; }
QMenu::indicator:checked { image: url(%(a_check_ink)s); }
QMenu::right-arrow { image: url(%(a_right)s); width: 12px; height: 12px; margin-right: 8px; }

/* ---------------------------------------------------------------- toolbar */
QToolBar { background: %(bg)s; border: 0; padding: 8px 14px; spacing: 8px; }
QToolBar::separator { background: %(divider)s; width: 1px; margin: 6px 6px; }
QToolButton { background: transparent; border: 1px solid transparent; border-radius: %(m_r_control)spx;
              padding: 4px 8px; color: %(text)s; }
QToolButton:hover { background: %(subtle_hover)s; }
QToolButton:pressed { background: %(subtle_pressed)s; }
QToolButton:checked { background: %(selected)s; color: %(text)s; }
QToolButton:disabled { color: %(text_disabled)s; }
QToolButton:focus { border-color: %(accent)s; }
QToolButton::menu-indicator { image: none; width: 0; }
/* the one primary action: Run */
QToolButton#accentButton { background: %(accent)s; color: %(on_accent)s; border: 1px solid %(accent)s;
                           font-weight: 600; padding: 6px 20px; border-radius: 16px; }
QToolButton#accentButton:hover { background: %(accent_hover)s; border-color: %(accent_hover)s; }
QToolButton#accentButton:pressed { background: %(accent_pressed)s; }
QToolButton#accentButton:disabled { background: %(control_disabled)s; color: %(text_disabled)s;
                                    border-color: %(stroke)s; }
/* Run's dropdown (Normal run / Model teaching run): the end of the same pill */
QToolButton#accentButton[popupMode="1"] { padding-right: 30px; }
QToolButton#accentButton::menu-button { border: 0; width: 24px; background: transparent;
        border-left: 1px solid rgba(255,255,255,0.35);
        border-top-right-radius: 16px; border-bottom-right-radius: 16px; }
QToolButton#accentButton::menu-button:hover { background: rgba(255,255,255,0.14); }
QToolButton#accentButton::menu-arrow { image: url(%(a_down_on)s); width: 12px; height: 12px; }
QToolButton#runningButton { border-radius: 16px; padding: 6px 14px; }
QToolButton#runningButton:checked { background: %(critical_bg)s; color: %(critical_text)s; font-weight: 600; }
QToolButton#toolbarIcon { padding: 6px; border-radius: 15px; background: %(layer)s; }
QToolButton#toolbarIcon:hover { background: %(layer_alt)s; }
QToolButton#moreButton { padding: 6px; border-radius: 15px; background: %(layer)s; }
QToolButton#moreButton:hover { background: %(layer_alt)s; }
QToolButton[big="true"] { font-size: %(t_callout)spt; }

/* Run | Build | History: a macOS segmented control */
QWidget#modeSwitch { background: %(segment_bg)s; border-radius: 14px; }
QWidget#modeSwitch QToolButton#modeButton { padding: 4px 18px; border-radius: 11px; margin: 1px 0;
        color: %(text2)s; font-weight: 500; border: 1px solid transparent; background: transparent; }
QWidget#modeSwitch QToolButton#modeButton:hover { color: %(text)s; }
QWidget#modeSwitch QToolButton#modeButton:checked { background: transparent; color: %(text)s;
        font-weight: 600; }
QWidget#modeSwitch[animated="false"] QToolButton#modeButton:checked { background: %(layer)s; }
QFrame#modeThumb { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 11px; }
QWidget#modeSwitch QToolButton#modeButton:focus { border-color: %(accent)s; }

/* production figures in the toolbar */
QFrame#kpi { background: %(layer)s; border: 0; border-radius: 12px; padding: 1px 10px; }
QFrame#kpi QLabel#caption { color: %(text3)s; font-size: %(t_caption)spt; font-weight: 500; letter-spacing: 0; }
QFrame#kpi QLabel#kpiValue { font-size: %(t_figure)spt; font-weight: 600; color: %(text)s; }
QFrame#kpi[tone="pass"] QLabel#kpiValue { color: %(success_text)s; }
QFrame#kpi[tone="fail"] QLabel#kpiValue { color: %(critical_text)s; }

/* ---------------------------------------------------------------- floating bars over the picture */
QFrame#floatBar { background: %(float)s; border: 1px solid %(stroke)s; border-radius: 18px; }
QFrame#floatBar QToolButton { padding: 5px; border-radius: 12px; border: 1px solid transparent; }
QFrame#floatBar QToolButton:hover { background: %(subtle_hover)s; }
QFrame#floatBar QToolButton:checked { background: %(accent)s; color: %(on_accent)s; }
QFrame#floatBar QToolButton[toggle="true"]:checked { background: %(accent_subtle)s; color: %(accent_text)s; }
QFrame#floatBar QToolBar { background: transparent; border: 0; padding: 0; spacing: 2px; }
QFrame#floatBar QToolBar::separator { background: %(divider)s; height: 1px; width: 1px; margin: 4px; }
QFrame#floatBar QLabel { color: %(text2)s; background: transparent; }
QFrame#floatBar QLabel#floatValue { color: %(text)s; font-family: %(mono)s; }
QFrame#floatBar QLabel#floatCaption { color: %(text3)s; font-size: %(t_caption)spt; font-weight: 600; }
QFrame#floatBar QComboBox { border-radius: 10px; padding: 2px 22px 2px 8px; min-height: 16px;
                            background: transparent; border: 1px solid transparent; }
QFrame#floatBar QComboBox:hover { background: %(subtle_hover)s; }
QFrame#floatBar QAbstractSpinBox { border-radius: 8px; padding: 2px 4px; min-height: 16px; }
QFrame#floatBar QAbstractSpinBox::up-button, QFrame#floatBar QAbstractSpinBox::down-button { width: 0; }
QFrame#floatSep { background: %(divider)s; min-width: 1px; max-width: 1px; margin: 4px 2px; }

/* ---------------------------------------------------------------- docks and panels */
QDockWidget { color: %(text2)s; titlebar-close-icon: url(%(a_close)s); titlebar-normal-icon: url(%(a_float)s); }
QDockWidget::title { background: %(bg)s; padding: 8px 10px 4px 14px; text-align: left; }
QDockWidget::close-button, QDockWidget::float-button { border: 0; border-radius: 8px;
        background: transparent; padding: 2px; icon-size: 12px; }
QDockWidget::close-button:hover, QDockWidget::float-button:hover { background: %(subtle_hover)s; }
QDockWidget > QWidget { background: %(bg)s; border: 0; }
QWidget#dockTitle { background: %(bg)s; }
QLabel#dockCaption { color: %(text3)s; font-weight: 600; font-size: %(t_caption)spt; }
QToolButton#dockClose { padding: 3px; border-radius: 8px; }
QWidget#panel { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 18px; }
QWidget#panelBody { background: transparent; border: 0; }

/* the sidebar: a macOS source list */
QLabel#sidebarSection { color: %(text3)s; font-weight: 600; font-size: %(t_caption)spt;
                        padding: 12px 10px 4px 10px; }
QAbstractItemView#sidebarList { background: transparent; border: 0; }
QAbstractItemView#sidebarList::item { min-height: %(m_sidebar_row_h)spx; padding: 0 10px; border-radius: 12px;
                                      margin: 1px 4px; }
QAbstractItemView#sidebarList::item:hover { background: %(subtle_hover)s; }
QAbstractItemView#sidebarList::item:selected { background: %(selected)s; color: %(accent_text)s; }
QAbstractItemView#sidebarList:focus { border: 0; }

/* ---------------------------------------------------------------- lists, trees, tables */
QAbstractItemView { background: %(layer)s; alternate-background-color: %(layer_alt)s;
        border: 1px solid %(stroke)s; border-radius: %(m_r_card)spx; color: %(text)s;
        selection-background-color: %(accent)s; selection-color: %(on_accent)s; }
QDockWidget QAbstractItemView, QWidget#panel QAbstractItemView { border: 0; border-radius: 0;
        background: transparent; }
QAbstractItemView:focus { border-color: %(stroke_strong)s; }
QAbstractItemView::item { padding: 3px 6px; border-radius: 0; }
QListView::item { border-radius: 10px; }
QTreeView::item, QListView::item { min-height: %(m_row_h)spx; }
QAbstractItemView::item:hover { background: %(subtle_hover)s; }
QAbstractItemView::item:selected { background: %(accent_subtle)s; color: %(text)s; }
QAbstractItemView::item:selected:active { background: %(accent_subtle)s; }
QTreeView { show-decoration-selected: 0; }
QTreeView::branch { background: transparent; }
QTreeView::branch:has-children:!has-siblings:closed, QTreeView::branch:closed:has-children:has-siblings
    { image: url(%(a_branch_closed)s); }
QTreeView::branch:open:has-children:!has-siblings, QTreeView::branch:open:has-children:has-siblings
    { image: url(%(a_branch_open)s); }
QTableView { gridline-color: transparent; }
QTableView::item { border-radius: 0; padding: 3px 8px; }
QHeaderView { background: transparent; border: 0; }
QHeaderView::section { background: transparent; color: %(text3)s; border: 0;
        border-bottom: 1px solid %(divider)s; border-right: 1px solid %(divider)s;
        padding: 5px 8px; font-weight: 500; font-size: %(t_small)spt; }
QHeaderView::section:last { border-right: 0; }
QHeaderView::section:hover { background: %(subtle_hover)s; }
QTableCornerButton::section { background: transparent; border: 0; }

/* ---------------------------------------------------------------- tabs */
QTabWidget::pane { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: %(m_r_card)spx; top: -1px; }
QTabBar { qproperty-drawBase: 0; background: transparent; }
QTabBar::tab { background: transparent; color: %(text2)s; padding: 5px 16px; margin: 3px 2px;
               border: 1px solid transparent; border-radius: 12px; }
QTabBar::tab:hover { background: %(subtle_hover)s; color: %(text)s; }
QTabBar::tab:selected { background: %(layer)s; border: 1px solid %(stroke)s; color: %(text)s; font-weight: 600; }
QTabBar::close-button { image: url(%(a_close)s); subcontrol-position: right; border-radius: 7px; margin: 2px; }
QTabBar::close-button:hover { background: %(subtle_hover)s; }
QTabBar::scroller { width: 28px; }
QTabBar QToolButton { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 10px; padding: 0; }
QTabBar QToolButton::right-arrow { image: url(%(a_right)s); width: 12px; height: 12px; }
QTabBar QToolButton::left-arrow { image: url(%(a_left)s); width: 12px; height: 12px; }
QTabBar#mdiTabs::tab { padding: 6px 16px 6px 14px; min-width: 90px; max-width: 280px; margin: 5px 4px 5px 0;
                       border-radius: 13px; color: %(text2)s; }
QTabBar#mdiTabs::tab:selected { background: %(layer)s; border: 1px solid %(stroke)s; color: %(text)s; }

/* ---------------------------------------------------------------- fields */
QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit, QTextEdit {
    background: %(control)s; border: 1px solid transparent; border-radius: %(m_r_control)spx;
    padding: %(m_pad_control)s; min-height: %(m_control_min_h)spx; color: %(text)s;
    selection-background-color: %(accent)s; selection-color: %(on_accent)s; }
QPlainTextEdit, QTextEdit { padding: 6px; }
QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover { background: %(control_hover)s; }
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {
    border: 1px solid %(accent)s; background: %(layer)s; }
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {
    background: %(control_disabled)s; color: %(text_disabled)s; border-color: %(stroke)s; }
QLineEdit#searchBox { border-radius: 14px; padding-left: 12px; background: %(segment_bg)s; border-color: transparent; }
QLineEdit#searchBox:focus { background: %(control)s; border-color: %(accent)s; }
QLineEdit[invalid="true"] { border: 1px solid %(critical_text)s; }
QComboBox { padding-right: 26px; }
QComboBox::drop-down { border: 0; width: 24px; subcontrol-origin: padding; subcontrol-position: right center; }
QComboBox::down-arrow { image: url(%(a_updown)s); width: 12px; height: 12px; }
QComboBox QAbstractItemView { background: %(flyout)s; border: 1px solid %(stroke_strong)s;
        border-radius: %(m_r_card)spx; padding: 4px; outline: 0; }
QComboBox QAbstractItemView::item { min-height: 28px; padding: 0 10px; border-radius: 9px; }
QComboBox QAbstractItemView::item:selected { background: %(accent_subtle)s; color: %(accent_text)s; }
QAbstractSpinBox { padding-right: 22px; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { border: 0; width: 18px; background: transparent;
        border-radius: 7px; margin: 2px 2px; subcontrol-origin: border; }
QAbstractSpinBox::up-button { subcontrol-position: top right; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: %(subtle_hover)s; }
QAbstractSpinBox::up-arrow { image: url(%(a_up)s); width: 9px; height: 9px; }
QAbstractSpinBox::down-arrow { image: url(%(a_down)s); width: 9px; height: 9px; }

/* ---------------------------------------------------------------- buttons */
QPushButton { background: %(control)s; border: 1px solid transparent; border-radius: 13px;
              padding: %(m_pad_button)s; min-height: %(m_control_min_h)spx; color: %(text)s; font-weight: 500; }
QPushButton:hover { background: %(control_hover)s; }
QPushButton:pressed { background: %(control_pressed)s; }
QPushButton:disabled { background: %(control_disabled)s; color: %(text_disabled)s; border-color: transparent; }
QPushButton:default, QPushButton[accent="true"], QPushButton#primaryButton {
        background: %(accent)s; color: %(on_accent)s; border: 1px solid %(accent)s; font-weight: 600; }
QPushButton:default:hover, QPushButton[accent="true"]:hover, QPushButton#primaryButton:hover {
        background: %(accent_hover)s; border-color: %(accent_hover)s; }
QPushButton:default:pressed, QPushButton[accent="true"]:pressed, QPushButton#primaryButton:pressed {
        background: %(accent_pressed)s; }
QPushButton:default:disabled, QPushButton#primaryButton:disabled {
        background: %(control_disabled)s; color: %(text_disabled)s; border-color: transparent; }
QPushButton:checked { background: %(accent_subtle)s; border-color: %(accent)s; color: %(accent_text)s; }
QPushButton:focus { border: 1px solid %(accent)s; }
QPushButton[flat="true"] { background: transparent; border-color: transparent; }
QPushButton[flat="true"]:hover { background: %(subtle_hover)s; }
QPushButton#linkButton { background: transparent; border: 0; color: %(accent_text)s; font-weight: 500;
                         padding: 2px 4px; text-align: left; }
QPushButton#linkButton:hover { text-decoration: underline; }
QPushButton#bigChoice { text-align: left; padding: 12px 18px; min-height: 44px; border-radius: 18px;
                         background: %(layer)s; border: 1px solid %(stroke)s; }
QPushButton#bigChoice:hover { border-color: %(accent)s; }
QDialogButtonBox QPushButton { min-width: 80px; }

QCheckBox, QRadioButton { spacing: 8px; color: %(text)s; background: transparent; }
QCheckBox::indicator, QRadioButton::indicator { width: 16px; height: 16px; background: %(layer)s;
        border: 1.5px solid %(input_bottom)s; }
QCheckBox::indicator { border-radius: 6px; }
QRadioButton::indicator { border-radius: 8px; }
QCheckBox::indicator:checked { background: %(accent)s; border-color: %(accent)s; image: url(%(a_check)s); }
QCheckBox::indicator:indeterminate { background: %(accent)s; border-color: %(accent)s; image: url(%(a_dash)s); }
QRadioButton::indicator:checked { background: %(accent)s; border-color: %(accent)s; image: url(%(a_radio)s); }
QCheckBox::indicator:focus, QRadioButton::indicator:focus { border: 1px solid %(accent)s; }
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled { border-color: %(stroke)s;
        background: %(control_disabled)s; }

QGroupBox { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 16px;
            margin-top: 22px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 4px; top: 2px; color: %(text3)s; font-weight: 600;
                   font-size: %(t_small)spt; }

QSlider::groove:horizontal { height: 4px; background: %(stroke_strong)s; border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(accent)s; border-radius: 2px; }
QSlider::groove:horizontal:disabled { background: %(stroke)s; }
QSlider::handle:horizontal { background: #FFFFFF; border: 4px solid %(accent)s; width: 10px;
                             height: 10px; margin: -7px 0; border-radius: 9px; }
QSlider::handle:horizontal:hover { border-color: %(accent_hover)s; }
QProgressBar { background: %(segment_bg)s; border: 0; border-radius: 3px; height: 6px; text-align: center;
               color: transparent; }
QProgressBar::chunk { background: %(accent)s; border-radius: 3px; }

/* ---------------------------------------------------------------- scroll bars: thin overlays */
QScrollBar { background: transparent; border: 0; }
QScrollBar:vertical { width: 10px; margin: 2px; }
QScrollBar:horizontal { height: 10px; margin: 2px; }
QScrollBar::handle { background: %(scroll)s; border-radius: 3px; }
QScrollBar:hover { background: transparent; }
QScrollBar::handle:hover { background: %(scroll_hover)s; }
QScrollBar::handle:vertical { min-height: 32px; margin: 0 2px; }
QScrollBar::handle:horizontal { min-width: 32px; margin: 2px 0; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; border: 0; background: transparent; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QScrollArea { background: transparent; border: 0; }
QScrollArea#tabScroll > QWidget > QWidget { background: transparent; }

/* ---------------------------------------------------------------- splitters, status bar, tooltips */
QSplitter::handle { background: %(divider)s; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }
QSplitter::handle:hover { background: %(accent)s; }
QStatusBar { background: %(bg)s; color: %(text3)s; border: 0;
             min-height: 24px; font-size: %(t_small)spt; }
QStatusBar::item { border: 0; }
QStatusBar QLabel { color: %(text3)s; padding: 0 6px; }
QStatusBar QLabel[tone="pass"] { color: %(success_text)s; }
QStatusBar QLabel[tone="fail"] { color: %(critical_text)s; }
QStatusBar QLabel[tone="warn"] { color: %(caution_text)s; }
QToolTip { background: %(tooltip)s; color: %(text)s; border: 1px solid %(stroke)s;
           border-radius: 10px; padding: 6px 10px; }
/* the box dragged out to select several regions */
QRubberBand { background: %(accent_subtle)s; border: 1px solid %(accent)s; border-radius: 6px; }
QMdiArea { background: %(bg)s; border: 0; }
QMdiSubWindow { background: %(bg)s; border: 1px solid %(stroke)s; border-radius: 16px; }
QMdiSubWindow > QWidget { background: %(bg)s; }
QMessageBox QLabel { color: %(text)s; }

/* ---------------------------------------------------------------- text styles */
QLabel#caption { color: %(text3)s; font-size: %(t_caption)spt; font-weight: 600; }
QLabel#secondary, QLabel[secondary="true"] { color: %(text2)s; }
QLabel#hint { color: %(text3)s; }
QLabel#fieldHelp { color: %(text3)s; font-size: %(t_small)spt; padding: 0 0 6px 2px; }
QLabel#sectionTitle { color: %(text)s; font-weight: 600; font-size: %(t_callout)spt; padding: 2px 0 4px 0; }
QLabel#pageTitle { color: %(text)s; font-weight: 700; font-size: %(t_title1)spt; }
QLabel#mono { font-family: %(mono)s; color: %(text2)s; }
QLabel#report[tone="pass"] { color: %(success_text)s; }
QLabel#report[tone="fail"] { color: %(critical_text)s; }
QLabel#report[tone="warn"] { color: %(caution_text)s; }

/* result badges: ACCEPT / REJECT pills */
QLabel[badge="true"] { border-radius: 11px; padding: 3px 12px; font-weight: 600; font-size: %(t_small)spt; }
QLabel[badge="true"][tone="idle"] { background: %(selected)s; color: %(text2)s; }
QLabel[badge="true"][tone="pass"] { background: %(success_bg)s; color: %(success_text)s; }
QLabel[badge="true"][tone="fail"] { background: %(critical_bg)s; color: %(critical_text)s; }
QLabel[badge="true"][tone="warn"] { background: %(caution_bg)s; color: %(caution_text)s; }

/* ---------------------------------------------------------------- the run page */
QWidget#runPage, QWidget#welcome, QWidget#runGrid { background: %(bg)s; }
QFrame#verdict { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 24px; }
QFrame#verdict[tone="pass"] { background: %(success_bg)s; border: 1px solid %(success_bg)s; }
QFrame#verdict[tone="fail"] { background: %(critical_bg)s; border: 1px solid %(critical_bg)s; }
QFrame#verdict[tone="warn"] { background: %(caution_bg)s; border: 1px solid %(caution_bg)s; }
QFrame#verdict QLabel { background: transparent; border: 0; }
QLabel#verdictWord { font-size: %(t_verdict)spt; font-weight: 700; color: %(text2)s; }
QFrame#verdict[tone="pass"] QLabel#verdictWord { color: %(success_text)s; }
QFrame#verdict[tone="fail"] QLabel#verdictWord { color: %(critical_text)s; }
QFrame#verdict[tone="warn"] QLabel#verdictWord { color: %(caution_text)s; }
QLabel#verdictReason { font-size: %(t_callout)spt; color: %(text)s; }
QLabel#verdictMeta { color: %(text3)s; font-size: %(t_small)spt; }

QFrame#tile { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 22px; }
QFrame#tile[tone="pass"] { border: 2px solid %(pass)s; }
QFrame#tile[tone="fail"] { border: 2px solid %(fail)s; }
QFrame#tile[tone="warn"] { border: 2px solid %(warn)s; }
QWidget#tileHead { background: transparent; }
QLabel#tileName { font-weight: 600; font-size: %(t_callout)spt; }

QWidget#displayBar { background: %(layer)s; border: 0; }
QWidget#displayBar QToolButton { padding: 3px 6px; }
QWidget#displayStatus { background: transparent; }
QWidget#displayStatus QLabel { color: %(text2)s; font-family: %(mono)s; }
QWidget#displayStatus QComboBox { padding: 1px 22px 1px 6px; min-height: 16px; }
ImageCanvas { background: %(canvas)s; border: 0; }

/* properties panel */
QLabel#sheetHeader { background: transparent; color: %(text)s; font-weight: 600; font-size: %(t_callout)spt;
                     padding: 12px 14px 2px 14px; border: 0; }
QLabel#sheetDesc { background: transparent; color: %(text2)s; padding: 0 14px 10px 14px; border: 0; }

/* ---------------------------------------------------------------- Build: guide, picker, welcome */
QFrame#guide { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 18px; }
QFrame#guide QLabel { background: transparent; border: 0; }
QLabel#guideTitle { font-weight: 600; font-size: %(t_callout)spt; color: %(text)s; }
QLabel#guideStep { color: %(text2)s; padding: 1px 0; }
QLabel#guideStep[state="done"] { color: %(success_text)s; }
QLabel#guideStep[state="now"] { color: %(text)s; font-weight: 600; }
QLabel#guideHint { color: %(text)s; }
QFrame#card { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 18px; }
QFrame#card:hover { border: 1px solid %(accent)s; }
QFrame#card:focus { border: 2px solid %(accent)s; }
QFrame#card QLabel { background: transparent; border: 0; }
QLabel#cardTitle { font-weight: 600; font-size: %(t_callout)spt; color: %(text)s; }
QLabel#cardDesc { color: %(text2)s; }
QLabel#question { font-weight: 600; font-size: %(t_title3)spt; color: %(text)s; padding-top: 10px; }
QLabel#questionSub { color: %(text2)s; }
QFrame#welcomeCard { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 26px; }
QLabel#trainHead { border-radius: 10px; padding: 4px; font-weight: 600; color: white; }
QLabel#trainHead[tone="pass"] { background: %(pass)s; }
QLabel#trainHead[tone="fail"] { background: %(fail)s; }

/* ---------------------------------------------------------------- Settings window */
QWidget#prefWindow { background: %(bg)s; }
QWidget#prefSidebarPane { background: %(sidebar)s; border-right: 1px solid %(divider)s; }
QListWidget#prefSidebar { background: transparent; border: 0; padding: 8px; }
QListWidget#prefSidebar::item { min-height: 34px; padding: 0 10px; border-radius: 12px; margin: 1px 0;
                                color: %(text)s; }
QListWidget#prefSidebar::item:hover { background: %(subtle_hover)s; }
QListWidget#prefSidebar::item:selected { background: %(selected)s; color: %(accent_text)s; }
QWidget#prefContent { background: %(bg)s; }
QLabel#prefTitle { font-size: %(t_title3)spt; font-weight: 700; color: %(text)s; }
QLabel#prefSubtitle { color: %(text3)s; }
QLabel#groupTitle { color: %(text2)s; font-weight: 600; font-size: %(t_small)spt; padding: 14px 4px 5px 4px; }
QLabel#groupFooter { color: %(text3)s; font-size: %(t_small)spt; padding: 5px 4px 0 4px; }
QFrame#formGroup { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 18px; }
QFrame#formGroup QLabel { background: transparent; }
QFrame#formRowSep { background: %(divider)s; min-height: 1px; max-height: 1px; border: 0; margin: 0 16px; }
QLabel#rowLabel { color: %(text)s; }
QLabel#rowHelp { color: %(text3)s; font-size: %(t_small)spt; }
QLabel#statusLine { color: %(text2)s; }
QLabel#statusLine[tone="pass"] { color: %(success_text)s; }
QLabel#statusLine[tone="fail"] { color: %(critical_text)s; }
QLabel#statusLine[tone="warn"] { color: %(caution_text)s; }
QLabel#rowHelp[tone="warn"] { color: %(caution_text)s; font-weight: 600; }
QWidget#prefFooter { background: %(bg)s; border-top: 1px solid %(divider)s; }

/* Find parts panel */
QWidget#partFinder { background: %(bg)s; }

/* ---------------------------------------------------------------- barcode */
QFrame#barcodeBar { background: %(layer)s; border: 1px solid %(stroke)s; border-radius: 22px; }
QFrame#barcodeBar QLabel { background: transparent; }
QLabel#barcodeCaption { font-weight: 600; font-size: %(t_callout)spt; color: %(text)s; }
QLineEdit#barcodeField { font-family: %(mono)s; font-size: %(t_callout)spt; padding: 5px 10px;
                         border: 1.5px solid %(stroke_strong)s; border-radius: 14px; background: %(bg)s; }
QLineEdit#barcodeField:focus { border: 2px solid %(accent)s; }
QLabel#barcodeState { padding: 3px 12px; font-size: %(t_body)spt; }
QLineEdit#monoField { font-family: %(mono)s; }
QFrame#barcodeBar QToolButton#linkButton { color: %(accent_text)s; border: 0; background: transparent; }
QFrame#barcodeBar QToolButton#linkButton:hover { text-decoration: underline; }

/* a note floating over the picture, with its one action */
QToolButton#noteAction { background: %(accent)s; color: %(on_accent)s; border: 0; border-radius: 12px;
                         padding: 4px 14px; font-weight: 600; }
QToolButton#noteAction:hover { background: %(accent_hover)s; }

/* a group of checks of one kind, adjusted together */
QFrame#groupBar { background: %(accent_subtle)s; border: 0;
                  border-radius: 16px; margin: 2px 10px 6px 10px; }
QFrame#groupBar QLabel { background: transparent; }
QLabel#groupBarTitle { font-weight: 600; color: %(text)s; }

/* the checks offered on a box just drawn */
QMenu#drawnMenu { padding: 6px; }
QMenu#drawnMenu::item { padding: 8px 30px 8px 12px; }
"""


# =====================================================================
# glyphs the stylesheet uses (written as small SVG files)
# =====================================================================
def _glyphs(t):
    s = 'fill="none" stroke="%s" stroke-width="%s" stroke-linecap="round" stroke-linejoin="round"'
    return {
        "check": '<path d="M4 8.3l2.6 2.6 5.4-5.8" %s/>' % (s % (t["on_accent"], 1.9)),
        "check_ink": '<path d="M3.5 8.5l3 3 6-7" %s/>' % (s % (t["text"], 1.6)),
        "dash": '<path d="M4.5 8h7" %s/>' % (s % (t["on_accent"], 1.9)),
        "down": '<path d="M4.5 6.2l3.5 3.6 3.5-3.6" %s/>' % (s % (t["text2"], 1.5)),
        "down_on": '<path d="M4.5 6.2l3.5 3.6 3.5-3.6" %s/>' % (s % (t["on_accent"], 1.8)),
        "up": '<path d="M4.5 9.8l3.5-3.6 3.5 3.6" %s/>' % (s % (t["text2"], 1.5)),
        "updown": '<path d="M5 6.3l3-3 3 3M5 9.7l3 3 3-3" %s/>' % (s % (t["text2"], 1.4)),
        "right": '<path d="M6.2 4.5l3.6 3.5-3.6 3.5" %s/>' % (s % (t["text3"], 1.5)),
        "left": '<path d="M9.8 4.5L6.2 8l3.6 3.5" %s/>' % (s % (t["text3"], 1.5)),
        "branch_open": '<path d="M4.5 6.2l3.5 3.6 3.5-3.6" %s/>' % (s % (t["text3"], 1.4)),
        "branch_closed": '<path d="M6.2 4.5l3.6 3.5-3.6 3.5" %s/>' % (s % (t["text3"], 1.4)),
        "close": '<path d="M5 5l6 6M11 5l-6 6" %s/>' % (s % (t["text2"], 1.3)),
        "float": '<rect x="3.5" y="3.5" width="9" height="9" rx="2" %s/>' % (s % (t["text2"], 1.1)),
        "radio": '<circle cx="8" cy="8" r="3" fill="%s"/>' % t["on_accent"],
    }


def write_glyphs(t, mode_name):
    import tempfile
    folder = os.path.join(tempfile.gettempdir(), "vision_station_design", mode_name)
    os.makedirs(folder, exist_ok=True)
    out = {}
    for name, body in _glyphs(t).items():
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


def build_stylesheet(t, glyphs):
    """The whole stylesheet for palette t, with ui_custom.qss appended."""
    values = dict(t)
    values.update({k: v for k, v in STATUS.items()})
    values.update({"t_" + k: v for k, v in TYPE.items()})
    values.update({"m_" + k: v for k, v in METRICS.items()})
    values.update({"a_" + k: v for k, v in glyphs.items()})
    values["mono"] = ", ".join("'%s'" % f for f in MONO_FONTS) + ", monospace"
    qss = STYLESHEET % values
    custom = os.path.join(HERE, "ui_custom.qss")
    if os.path.exists(custom):
        try:
            with open(custom, encoding="utf-8") as f:
                qss += "\n/* ui_custom.qss */\n" + f.read()
        except OSError:
            pass
    return qss


# =====================================================================
# fonts
# =====================================================================
_loaded = {"done": False}


def load_fonts():
    """Register the bundled fonts with Qt, once."""
    if _loaded["done"]:
        return
    _loaded["done"] = True
    from PySide6.QtGui import QFontDatabase
    if os.path.isdir(FONT_DIR):
        for name in sorted(os.listdir(FONT_DIR)):
            if name.lower().endswith((".ttf", ".otf")):
                QFontDatabase.addApplicationFont(os.path.join(FONT_DIR, name))


def family():
    """The first of FONTS this PC has (bundled ones included)."""
    from PySide6.QtGui import QFontDatabase
    load_fonts()
    have = set(QFontDatabase.families())
    return next((f for f in FONTS if f in have), "")


def app_font():
    from PySide6.QtGui import QFont
    f = QFont()
    fam = family()
    if fam:
        f.setFamily(fam)
    f.setPointSizeF(BASE_PT)
    # smooth outlines with crisp baselines -- closest to macOS text on Windows
    f.setHintingPreference(QFont.PreferVerticalHinting)
    f.setStyleStrategy(QFont.PreferAntialias)
    return f


def font(style="body", weight=None):
    """A QFont in one of the TYPE styles, e.g. font("title3", 600)."""
    from PySide6.QtGui import QFont
    f = app_font()
    f.setPointSizeF(TYPE.get(style, BASE_PT))
    if weight is not None:
        f.setWeight(QFont.Weight(weight))
    return f


# =====================================================================
# 5. CONTROLS that Qt does not have
# =====================================================================
def _palette_now():
    try:
        from vision_station.ui import theme
        return theme._state["tokens"]
    except Exception:
        return LIGHT


def _qt():
    from PySide6 import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


def make_switch(checked=False, parent=None):
    """A switch: a QCheckBox drawn as a pill with a knob that springs across
    (and stretches while pressed), so it is checked, connected and tested
    like any check box."""
    QtCore, QtGui, QtWidgets = _qt()

    class Switch(QtWidgets.QCheckBox):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setCursor(QtCore.Qt.PointingHandCursor)
            self.setFixedSize(42, 24)
            self.setFocusPolicy(QtCore.Qt.TabFocus)
            self._pos = 0.0               # 0 off .. 1 on, animated
            self._down = False
            from vision_station.ui import motion

            def frame(v):
                self._pos = v
                self.update()
            self._animate = motion.switch_animation(self, frame)
            self.toggled.connect(lambda on: self._animate(1.0 if on else 0.0, self._pos))

        def sizeHint(self):
            return QtCore.QSize(42, 24)

        def hitButton(self, pos):
            return self.rect().contains(pos)

        def setChecked(self, on):
            super().setChecked(on)
            if not self.isVisible():
                self._pos = 1.0 if on else 0.0

        def mousePressEvent(self, e):
            self._down = True
            self.update()
            super().mousePressEvent(e)

        def mouseReleaseEvent(self, e):
            self._down = False
            self.update()
            super().mouseReleaseEvent(e)

        def showEvent(self, e):
            self._pos = 1.0 if self.isChecked() else 0.0
            super().showEvent(e)

        def paintEvent(self, _e):
            t = _palette_now()
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            r = QtCore.QRectF(1, 1, 40, 22)
            k = max(0.0, min(1.0, self._pos))
            off = QtGui.QColor(t["stroke_strong"])
            on = QtGui.QColor(t["accent"])
            track = QtGui.QColor(int(off.red() + (on.red() - off.red()) * k),
                                 int(off.green() + (on.green() - off.green()) * k),
                                 int(off.blue() + (on.blue() - off.blue()) * k))
            if not self.isEnabled():
                track.setAlphaF(0.4)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(track)
            p.drawRoundedRect(r, 11, 11)
            # the knob: 18 px, 24 while pressed, its centre sliding with _pos
            # (beyond 0..1 on the spring's overshoot)
            kw = 24.0 if self._down else 18.0
            left = r.left() + 2 + (r.width() - 4 - kw) * self._pos
            knob = QtCore.QRectF(left, r.top() + 2, kw, 18)
            p.setBrush(QtGui.QColor(0, 0, 0, 35))
            p.drawRoundedRect(knob.translated(0, 0.8), 9, 9)
            p.setBrush(QtGui.QColor("#FFFFFF"))
            p.drawRoundedRect(knob, 9, 9)
            if self.hasFocus():
                p.setBrush(QtCore.Qt.NoBrush)
                p.setPen(QtGui.QPen(QtGui.QColor(t["accent"]), 1.5))
                p.drawRoundedRect(r.adjusted(-0.5, -0.5, 0.5, 0.5), 11.5, 11.5)
            p.end()

    w = Switch(parent)
    w.setChecked(bool(checked))
    return w


class FormGroup:
    """A macOS grouped section: a caption above, a rounded white box of rows,
    each row a label on the left and its control on the right, rows divided
    by hairlines, and an optional note below.

        g = FormGroup("Connection", footer="An IP address always works.")
        g.add_row("Address", line_edit, help="The camera's IP address")
        layout.addWidget(g.widget)
    """

    def __init__(self, title="", footer=""):
        QtCore, QtGui, QtWidgets = _qt()
        self.widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(self.widget)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        if title:
            cap = QtWidgets.QLabel(title)
            cap.setObjectName("groupTitle")
            outer.addWidget(cap)
        self.box = QtWidgets.QFrame()
        self.box.setObjectName("formGroup")
        self.rows = QtWidgets.QVBoxLayout(self.box)
        self.rows.setContentsMargins(0, 2, 0, 2)
        self.rows.setSpacing(0)
        outer.addWidget(self.box)
        self.footer = None
        if footer:
            self.footer = QtWidgets.QLabel(footer)
            self.footer.setObjectName("groupFooter")
            self.footer.setWordWrap(True)
            outer.addWidget(self.footer)
        self._count = 0

    def _sep(self):
        QtCore, QtGui, QtWidgets = _qt()
        if self._count:
            s = QtWidgets.QFrame()
            s.setObjectName("formRowSep")
            s.setFixedHeight(1)
            self.rows.addWidget(s)
        self._count += 1

    def add_row(self, label, control=None, help=""):
        """label on the left, control (a widget or a layout) on the right."""
        QtCore, QtGui, QtWidgets = _qt()
        self._sep()
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(12, 7, 12, 7)
        h.setSpacing(12)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(1)
        lab = QtWidgets.QLabel(label)
        lab.setObjectName("rowLabel")
        text.addWidget(lab)
        if help:
            hl = QtWidgets.QLabel(help)
            hl.setObjectName("rowHelp")
            hl.setWordWrap(True)
            text.addWidget(hl)
        h.addLayout(text, 1)
        if isinstance(control, QtWidgets.QLayout):
            h.addLayout(control)
        elif control is not None:
            h.addWidget(control, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.rows.addWidget(row)
        row.label = lab
        return row

    def add_widget(self, widget):
        """A full-width row: a status line, a button bar, a grid."""
        QtCore, QtGui, QtWidgets = _qt()
        self._sep()
        wrap = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(wrap)
        h.setContentsMargins(12, 8, 12, 8)
        if isinstance(widget, QtWidgets.QLayout):
            h.addLayout(widget)
        else:
            h.addWidget(widget)
        self.rows.addWidget(wrap)
        return wrap


def page_header(title, subtitle=""):
    """The title (and a line under it) at the top of a settings page."""
    QtCore, QtGui, QtWidgets = _qt()
    w = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(w)
    v.setContentsMargins(4, 0, 4, 2)
    v.setSpacing(2)
    t = QtWidgets.QLabel(title)
    t.setObjectName("prefTitle")
    v.addWidget(t)
    if subtitle:
        s = QtWidgets.QLabel(subtitle)
        s.setObjectName("prefSubtitle")
        s.setWordWrap(True)
        v.addWidget(s)
    return w


# =====================================================================
# 6. WINDOW: title bar in the toolbar's colour (Windows 11)
# =====================================================================
def _colorref(hexcol):
    c = hexcol.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return b << 16 | g << 8 | r


def native_window(widget, t, dark):
    """Dark or light caption buttons, the caption in the toolbar colour so
    title bar and toolbar read as one (as a Mac window's unified toolbar
    does), rounded corners and a hairline border. Silently nothing where
    Windows does not offer it."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = int(widget.winId())
        dwm = ctypes.windll.dwmapi

        def setattr_(attr, value):
            v = ctypes.c_int(value)
            return dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))
        for attr in (20, 19):                      # immersive dark mode
            if setattr_(attr, 1 if dark else 0) == 0:
                break
        solid = lambda c: c if c.startswith("#") else ("#2A2A2D" if dark else "#F5F5F7")
        setattr_(35, _colorref(solid(t["bg"])))    # caption colour
        setattr_(36, _colorref(solid(t["text"])))  # caption text
        setattr_(34, _colorref(solid(t["stroke"])))  # border colour
        setattr_(33, 2)                            # rounded corners
    except Exception:
        pass
