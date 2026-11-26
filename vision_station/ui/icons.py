"""Vision Station's icons.

The Phosphor icon set (MIT licence, phosphoricons.com) through QtAwesome
(MIT licence) when it is installed -- one consistent, current family of line
icons for every action and tool. The original drawn set below is the
fallback, so the application never depends on it.

The drawn set: an original set of modern line icons.

Drawn as SVG on a 24-unit grid -- 1.7-unit rounded strokes, a neutral ink,
and a category accent -- so they stay crisp at any size and on high-DPI
screens. Every tool type has its own icon, coloured by its family, the way
VisionPro distinguishes tool families; the drawings themselves are original,
since Cognex's icons are their copyright.

    icon("run")                    an action icon
    tool_icon("find_circle")       a tool type's icon, in its family's colour
"""
import os

from vision_station import paths

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap

INK = "#2E3A46"
ACCENTS = {"image": "#8B5CF6", "locate": "#2563EB", "presence": "#0D9488",
           "colour": "#EA580C", "geometry": "#4F46E5", "surface": "#B45309",
           "trained": "#DB2777", "text": "#16A34A", "logic": "#475569",
           "measure": "#0891B2", "run": "#16A34A", "stop": "#DC2626",
           "warn": "#D97706"}

# ------------------------------------------------------------ drawings
# {c} is the ink, {a} the accent. Paths are on a 24 x 24 grid.
_S = 'fill="none" stroke="{c}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"'
_A = 'fill="none" stroke="{a}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"'
_F = 'fill="{a}" stroke="none"'

ACTIONS = {
    "run": '<path d="M8 5.5v13l10.5-6.5z" fill="{a}" stroke="{a}" stroke-width="1.4" stroke-linejoin="round"/>',
    "run_continuous": '<path d="M9.5 8v8l6-4z" ' + _F + '/>'
                      '<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3" ' + _A + '/>'
                      '<path d="M19.8 3.8v3.5h-3.5" ' + _A + '/>',
    "stop": '<rect x="6.5" y="6.5" width="11" height="11" rx="2" ' + _F + '/>',
    "open": '<path d="M3.5 7.5a2 2 0 0 1 2-2h4l2 2h7a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z" ' + _S + '/>',
    "save": '<path d="M5 4h11l3 3v12a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1z" ' + _S + '/>'
            '<path d="M8 4v5h7V4M8 20v-6h8v6" ' + _S + '/>',
    "export": '<path d="M12 14V3.5M8 7.5l4-4 4 4" ' + _A + '/>'
              '<path d="M5 12v6.5a1.5 1.5 0 0 0 1.5 1.5h11a1.5 1.5 0 0 0 1.5-1.5V12" ' + _S + '/>',
    "import": '<path d="M12 3.5V14M8 10l4 4 4-4" ' + _A + '/>'
              '<path d="M5 12v6.5a1.5 1.5 0 0 0 1.5 1.5h11a1.5 1.5 0 0 0 1.5-1.5V12" ' + _S + '/>',
    "python": '<path d="M12 3.5c-3.5 0-4 1.3-4 3v2h4.5v1H6.3c-1.9 0-3 1.3-3 3.9 0 2.7 1.1 3.9 3 3.9H8v-2.2c0-1.6 1.3-2.8 3-2.8h3.5c1.4 0 2.5-1.1 2.5-2.5V6.5c0-1.7-1.5-3-5-3z" ' + _S + '/>'
              '<path d="M12 20.5c3.5 0 4-1.3 4-3v-2h-4.5v-1h6.2c1.9 0 3-1.3 3-3.9" ' + _A + '/>',
    "reset": '<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3" ' + _S + '/><path d="M4.2 3.8v3.5h3.5" ' + _S + '/>',
    "toolbox": '<rect x="3.5" y="8" width="17" height="11.5" rx="2" ' + _S + '/>'
               '<path d="M9 8V6a1.5 1.5 0 0 1 1.5-1.5h3A1.5 1.5 0 0 1 15 6v2M3.5 13h17" ' + _S + '/>'
               '<path d="M10.5 13v2h3v-2" ' + _A + '/>',
    "image_source": '<path d="M4 8.5a2 2 0 0 1 2-2h2l1.5-2h5l1.5 2h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z" ' + _S + '/>'
                    '<circle cx="12" cy="13" r="3.5" ' + _A + '/>',
    "electric": '<path d="M13.5 3 5.5 13.5h6l-1 7.5 8-10.5h-6z" fill="{a}" stroke="{a}" stroke-width="1.2" stroke-linejoin="round"/>',
    "floating": '<rect x="3.5" y="7.5" width="12" height="10" rx="1.5" ' + _S + '/>'
                '<path d="M8.5 7.5V5A1.5 1.5 0 0 1 10 3.5h9A1.5 1.5 0 0 1 20.5 5v7.5A1.5 1.5 0 0 1 19 14h-3.5" ' + _A + '/>',
    "grid_results": '<rect x="3.5" y="4.5" width="17" height="15" rx="1.5" ' + _S + '/>'
                    '<path d="M3.5 9.5h17M3.5 14.5h17M10 4.5v15" ' + _S + '/>',
    "help": '<circle cx="12" cy="12" r="8.5" ' + _S + '/>'
            '<path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 .9-1 1.7v.5" ' + _S + '/>'
            '<circle cx="12" cy="17" r=".6" fill="{c}" stroke="{c}" stroke-width=".8"/>',
    "pointer": '<path d="M6 3.5 18.5 12l-5.5 1.3 3 6.2-2.3 1-3-6.2L6 18z" ' + _S + '/>',
    "pan": '<path d="M12 3.5v17M3.5 12h17M9.5 6 12 3.5 14.5 6M9.5 18 12 20.5 14.5 18M6 9.5 3.5 12 6 14.5M18 9.5l2.5 2.5-2.5 2.5" ' + _S + '/>',
    "zoom_in": '<circle cx="10.5" cy="10.5" r="6" ' + _S + '/><path d="m15 15 5 5M8 10.5h5M10.5 8v5" ' + _S + '/>',
    "zoom_out": '<circle cx="10.5" cy="10.5" r="6" ' + _S + '/><path d="m15 15 5 5M8 10.5h5" ' + _S + '/>',
    "fit": '<path d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v4M20 15v4a1 1 0 0 1-1 1h-4M9 20H5a1 1 0 0 1-1-1v-4" ' + _S + '/>'
           '<rect x="8" y="8" width="8" height="8" rx="1" ' + _A + '/>',
    "actual": '<path d="M6 8v8M4.8 9.2 6 8M18 8v8M16.8 9.2 18 8" ' + _S + '/>'
              '<circle cx="12" cy="10" r=".7" fill="{c}"/><circle cx="12" cy="14" r=".7" fill="{c}"/>',
    "grid": '<path d="M4 9h16M4 15h16M9 4v16M15 4v16" ' + _S + '/>',
    "graphics": '<circle cx="9" cy="10" r="4.5" ' + _A + '/><rect x="11" y="12" width="8.5" height="7.5" rx="1" ' + _S + '/>',
    "guides": '<circle cx="12" cy="12" r="7.5" stroke-dasharray="2.4 2.2" ' + _A + '/>'
              '<path d="M12 3v4M12 17v4M3 12h4M17 12h4" ' + _S + '/>',
    "exposure": '<circle cx="12" cy="12" r="4" ' + _A + '/><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6 7 7M17 17l1.4 1.4M5.6 18.4 7 17M17 7l1.4-1.4" ' + _S + '/>',
    "history": '<circle cx="12" cy="12" r="8.5" ' + _S + '/><path d="M12 7v5l3.5 2" ' + _A + '/>',
    "tag": '<path d="M3.5 12.5V4.5a1 1 0 0 1 1-1h8l8 8-9 9z" ' + _S + '/><circle cx="8" cy="8" r="1.6" ' + _F + '/>',
    "rect": '<rect x="4" y="6" width="16" height="12" rx="1" ' + _S + '/>',
    "ellipse": '<ellipse cx="12" cy="12" rx="8.5" ry="6" ' + _S + '/>',
    "polygon": '<path d="M12 3.5 20 9l-3 10.5H7L4 9z" ' + _S + '/>',
    "build": '<path d="M14.5 5.5a4 4 0 0 0-5 5L4 16l4 4 5.5-5.5a4 4 0 0 0 5-5l-2.5 2.5-2.5-.5-.5-2.5z" ' + _S + '/>',
    "layout": '<rect x="3.5" y="4.5" width="17" height="15" rx="1.5" ' + _S + '/><path d="M9 4.5v15M9 12h11.5" ' + _S + '/>',
    "output": '<path d="M4 13.5h4l1.5 2.5h5l1.5-2.5h4" ' + _S + '/>'
              '<path d="M4 13.5 6.5 6h11l2.5 7.5v5a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 18.5z" ' + _S + '/>',
    "train": '<rect x="4" y="7" width="12" height="9" rx="1.5" ' + _S + '/>'
             '<path d="M7 19.5h12a1.5 1.5 0 0 0 1.5-1.5V10" ' + _S + '/>'
             '<path d="m7.5 12 2 2 3.5-4" ' + _A + '/>',
    "test": '<path d="M9 3.5h6M10 3.5v5.5l-5 9a1.5 1.5 0 0 0 1.3 2.2h11.4a1.5 1.5 0 0 0 1.3-2.2l-5-9V3.5" ' + _S + '/>'
            '<path d="M7.5 15h9" ' + _A + '/>',
    "add": '<path d="M12 5v14M5 12h14" ' + _S + '/>',
    "delete": '<path d="M5 7h14M10 7V5h4v2M7 7l1 12.5h8L17 7" ' + _S + '/>',
    "close": '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11" ' + _S + '/>',
    "rename": '<path d="M4 20h4L19 9l-4-4L4 16z" ' + _S + '/><path d="m13.5 6.5 4 4" ' + _S + '/>',
    "up": '<path d="M12 19V5M6 11l6-6 6 6" ' + _S + '/>',
    "down": '<path d="M12 5v14M6 13l6 6 6-6" ' + _S + '/>',
    "cascade": '<rect x="3.5" y="3.5" width="11" height="9" rx="1" ' + _S + '/><rect x="9.5" y="11.5" width="11" height="9" rx="1" ' + _A + '/>',
    "tile": '<rect x="3.5" y="4.5" width="7.5" height="15" rx="1" ' + _S + '/><rect x="13" y="4.5" width="7.5" height="15" rx="1" ' + _S + '/>',
    "counts": '<path d="M5 19V11M10 19V6M15 19v-5M20 19V9" ' + _S + '/>',
    "job": '<rect x="4" y="4" width="16" height="16" rx="2.5" ' + _S + '/><circle cx="12" cy="12" r="3.5" ' + _A + '/>',
    # ---- arranging regions, as a drawing application does
    "align_left": '<path d="M4 3.5v17" ' + _S + '/><rect x="7" y="6" width="12" height="4.5" rx="1.2" ' + _A + '/><rect x="7" y="13.5" width="7.5" height="4.5" rx="1.2" ' + _A + '/>',
    "align_hcenter": '<path d="M12 3.5v17" ' + _S + '/><rect x="5" y="6" width="14" height="4.5" rx="1.2" ' + _A + '/><rect x="7.5" y="13.5" width="9" height="4.5" rx="1.2" ' + _A + '/>',
    "align_right": '<path d="M20 3.5v17" ' + _S + '/><rect x="5" y="6" width="12" height="4.5" rx="1.2" ' + _A + '/><rect x="9.5" y="13.5" width="7.5" height="4.5" rx="1.2" ' + _A + '/>',
    "align_top": '<path d="M3.5 4h17" ' + _S + '/><rect x="6" y="7" width="4.5" height="12" rx="1.2" ' + _A + '/><rect x="13.5" y="7" width="4.5" height="7.5" rx="1.2" ' + _A + '/>',
    "align_vcenter": '<path d="M3.5 12h17" ' + _S + '/><rect x="6" y="5" width="4.5" height="14" rx="1.2" ' + _A + '/><rect x="13.5" y="7.5" width="4.5" height="9" rx="1.2" ' + _A + '/>',
    "align_bottom": '<path d="M3.5 20h17" ' + _S + '/><rect x="6" y="5" width="4.5" height="12" rx="1.2" ' + _A + '/><rect x="13.5" y="9.5" width="4.5" height="7.5" rx="1.2" ' + _A + '/>',
    "distribute_h": '<path d="M3.5 4v16M20.5 4v16" ' + _S + '/><rect x="9.5" y="7" width="5" height="10" rx="1.2" ' + _A + '/>',
    "distribute_v": '<path d="M4 3.5h16M4 20.5h16" ' + _S + '/><rect x="7" y="9.5" width="10" height="5" rx="1.2" ' + _A + '/>',
    "match_width": '<path d="M4 7.5h16M4 16.5h16" ' + _A + '/><path d="M7 4.5 4 7.5l3 3M17 4.5l3 3-3 3M7 13.5l-3 3 3 3M17 13.5l3 3-3 3" ' + _S + '/>',
    "match_height": '<path d="M7.5 4v16M16.5 4v16" ' + _A + '/><path d="M4.5 7l3-3 3 3M4.5 17l3 3 3-3M13.5 7l3-3 3 3M13.5 17l3 3 3-3" ' + _S + '/>',
    "rotate_cw": '<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3" ' + _S + '/><path d="M19.8 3.8v3.5h-3.5" ' + _A + '/>',
    "rotate_ccw": '<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3" ' + _S + '/><path d="M4.2 3.8v3.5h3.5" ' + _A + '/>',
    "rotate_zero": '<path d="M4 19.5h16M4 19.5 15 6" ' + _S + '/><path d="M10 19.5a6 6 0 0 0-1.6-4.1" ' + _A + '/>',
    "lock": '<rect x="5" y="10.5" width="14" height="10" rx="2" ' + _S + '/><path d="M8 10.5V8a4 4 0 0 1 8 0v2.5" ' + _S + '/><circle cx="12" cy="15.5" r="1.4" ' + _F + '/>',
    "unlock": '<rect x="5" y="10.5" width="14" height="10" rx="2" ' + _S + '/><path d="M8 10.5V8a4 4 0 0 1 7.6-1.7" ' + _S + '/><circle cx="12" cy="15.5" r="1.4" ' + _F + '/>',
    "duplicate": '<rect x="8" y="8" width="12" height="12" rx="2" ' + _S + '/><path d="M16 8V5.5A1.5 1.5 0 0 0 14.5 4h-9A1.5 1.5 0 0 0 4 5.5v9A1.5 1.5 0 0 0 5.5 16H8" ' + _S + '/><path d="M14 11v6M11 14h6" ' + _A + '/>',
    "zoom_selection": '<circle cx="10.5" cy="10.5" r="6" ' + _S + '/><path d="m15 15 5 5" ' + _S + '/><rect x="8" y="8" width="5" height="5" rx="1" ' + _A + '/>',
    # ---- the toolbar and the Settings window
    "gear": '<circle cx="12" cy="12" r="3" ' + _A + '/>'
            '<path d="M12 2.8l1.6 2.3 2.7-.7.5 2.8 2.6 1-1 2.6 1.9 2.1-2.1 1.9 1 2.6-2.6 1-.5 2.8-2.7-.7L12 21.2l-1.6-2.3-2.7.7-.5-2.8-2.6-1 1-2.6L3.7 12l1.9-2.1-1-2.6 2.6-1 .5-2.8 2.7.7z" ' + _S + '/>',
    "more": '<circle cx="6" cy="12" r="1.5" fill="{c}"/><circle cx="12" cy="12" r="1.5" fill="{c}"/>'
            '<circle cx="18" cy="12" r="1.5" fill="{c}"/>',
    "sliders": '<path d="M5 7h9M18 7h1M5 17h3M12 17h7" ' + _S + '/>'
               '<circle cx="16" cy="7" r="2" ' + _A + '/><circle cx="10" cy="17" r="2" ' + _A + '/>',
    "network": '<rect x="9" y="3.5" width="6" height="5" rx="1" ' + _S + '/><rect x="3.5" y="15.5" width="6" height="5" rx="1" ' + _S + '/>'
               '<rect x="14.5" y="15.5" width="6" height="5" rx="1" ' + _S + '/><path d="M12 8.5v3.5M6.5 15.5V12h11v3.5" ' + _A + '/>',
    "barcode": '<path d="M3.5 7V5.5a1 1 0 0 1 1-1H6M18 4.5h1.5a1 1 0 0 1 1 1V7M20.5 17v1.5a1 1 0 0 1-1 1H18M6 19.5H4.5a1 1 0 0 1-1-1V17" ' + _S + '/>'
               '<path d="M7 8v8M10 8v8M13.5 8v8M17 8v8" fill="none" stroke="{a}" stroke-width="1.7" stroke-linecap="round"/>'
               '<path d="M11.7 8v8M15.3 8v8" fill="none" stroke="{a}" stroke-width=".9" stroke-linecap="round"/>',
    "photo": '<rect x="3.5" y="5" width="17" height="14" rx="2" ' + _S + '/>'
             '<path d="m3.5 16 4.5-4.5 4 4 3-3 5.5 5.5" ' + _S + '/><circle cx="15.5" cy="9" r="1.5" ' + _F + '/>',
    "sparkle": '<path d="M11 3.5c.6 4 2.5 5.9 6.5 6.5-4 .6-5.9 2.5-6.5 6.5-.6-4-2.5-5.9-6.5-6.5 4-.6 5.9-2.5 6.5-6.5z" ' + _A + '/>'
               '<path d="M18 15c.3 1.7 1 2.4 2.5 2.7-1.5.3-2.2 1-2.5 2.8-.3-1.8-1-2.5-2.5-2.8 1.5-.3 2.2-1 2.5-2.7z" ' + _S + '/>',
    "rotate_180": '<path d="M5 12a7 7 0 0 1 12.2-4.7" ' + _S + '/><path d="M17.8 3.8v3.7h-3.7" ' + _A + '/>'
                  '<path d="M19 12a7 7 0 0 1-12.2 4.7" ' + _S + '/><path d="M6.2 20.2v-3.7h3.7" ' + _A + '/>',
}

# One per tool type, drawn in the family's accent.
TOOLS = {
    "image_process": ('image', '<path d="M4 20 15 9" ' + _S + '/><path d="m14 5 1-2 1 2 2 1-2 1-1 2-1-2-2-1zM19 11l.6-1.2L21 9.2l-1.4-.6L19 7.4l-.6 1.2-1.4.6 1.4.6z" ' + _F + '/>'),
    "align": ('locate', '<circle cx="12" cy="12" r="7" ' + _A + '/><circle cx="12" cy="12" r="2.5" ' + _F + '/><path d="M12 2.5v4M12 17.5v4M2.5 12h4M17.5 12h4" ' + _S + '/>'),
    "calibrate_scale": ('locate', '<rect x="3" y="8" width="18" height="8" rx="1.5" ' + _A + '/><path d="M7 8v3M11 8v4.5M15 8v3M19 8v4.5" ' + _S + '/>'),
    "presence": ('presence', '<rect x="4" y="4" width="16" height="16" rx="3" ' + _A + '/><path d="m8 12.5 3 3 5.5-6" ' + _S + '/>'),
    "count_template": ('presence', '<rect x="3.5" y="4" width="7" height="7" rx="1.5" ' + _A + '/><rect x="13.5" y="4" width="7" height="7" rx="1.5" ' + _A + '/><rect x="8.5" y="13.5" width="7" height="7" rx="1.5" ' + _A + '/>'),
    "count_blob": ('presence', '<path d="M6.5 5c2.5-1 5 1 4 3.5S7 12 5 10s-.5-4 1.5-5z" ' + _F + '/><path d="M16 7.5c2.3-.3 4 1.8 3 4s-4.5 1.5-4.8-.7S14 7.8 16 7.5z" ' + _F + '/><path d="M9.5 14.5c2.8-.6 5 2 3.7 4.3s-6 1.4-6.1-1.2c0-1.4 1-2.8 2.4-3.1z" ' + _F + '/>'),
    "color_match": ('colour', '<path d="M12 3.5s6 6.2 6 10.3a6 6 0 0 1-12 0C6 9.7 12 3.5 12 3.5z" ' + _F + '/><path d="M9.5 14.5a2.5 2.5 0 0 0 2.5 2.5" fill="none" stroke="#FFFFFF" stroke-width="1.5" stroke-linecap="round"/>'),
    "color_classify": ('colour', '<circle cx="5.5" cy="12" r="2.4" fill="#DC2626"/><circle cx="12" cy="12" r="2.4" fill="#16A34A"/><circle cx="18.5" cy="12" r="2.4" fill="#2563EB"/>'),
    "led_glow": ('colour', '<path d="M9 17.5h6M10 20.5h4M12 3.5a5.5 5.5 0 0 0-3.5 9.8c.6.5 1 1.3 1 2.2h5c0-.9.4-1.7 1-2.2A5.5 5.5 0 0 0 12 3.5z" ' + _A + '/>'),
    "find_line": ('geometry', '<path d="M3.5 18.5 20.5 5.5" ' + _A + '/><circle cx="6" cy="15" r="1.4" ' + _F + '/><circle cx="12" cy="11.5" r="1.4" ' + _F + '/><circle cx="17.5" cy="7" r="1.4" ' + _F + '/>'),
    "find_circle": ('geometry', '<circle cx="12" cy="12" r="7.5" ' + _A + '/><circle cx="12" cy="4.5" r="1.4" ' + _F + '/><circle cx="19.5" cy="12" r="1.4" ' + _F + '/><circle cx="12" cy="19.5" r="1.4" ' + _F + '/><circle cx="4.5" cy="12" r="1.4" ' + _F + '/>'),
    "caliper": ('geometry', '<path d="M5 3.5v17M19 3.5v17" ' + _A + '/><path d="M8 12h8M10 9.5 7.5 12l2.5 2.5M14 9.5l2.5 2.5-2.5 2.5" ' + _S + '/>'),
    "fit_shape": ('geometry', '<path d="M3.5 17c5-9 12-9 17-2" stroke-dasharray="2 2.2" ' + _A + '/><circle cx="6" cy="14" r="1.3" ' + _F + '/><circle cx="12" cy="9.2" r="1.3" ' + _F + '/><circle cx="18" cy="12.5" r="1.3" ' + _F + '/>'),
    "intersect": ('geometry', '<path d="M4 20 20 4M4 8l16 8" ' + _A + '/><circle cx="12" cy="12" r="2" ' + _F + '/>'),
    "distance": ('geometry', '<path d="M4 12h16M7 9l-3 3 3 3M17 9l3 3-3 3" ' + _A + '/><path d="M4 5v3M20 5v3" ' + _S + '/>'),
    "trained_surface": ('surface', '<rect x="4" y="4" width="16" height="16" rx="2" ' + _A + '/><path d="M4 10h16M4 15h16M10 4v16M15 4v16" stroke="{a}" stroke-width="1" opacity=".5" fill="none"/><path d="m16.5 3 .8 1.7 1.7.8-1.7.8-.8 1.7-.8-1.7-1.7-.8 1.7-.8z" ' + _F + '/>'),
    "defect_diff": ('surface', '<rect x="4" y="4" width="16" height="16" rx="2" ' + _S + '/><path d="M7 16c2-1 3-4 5-4s2.5 2 5-4" ' + _A + '/>'),
    "anomaly": ('surface', '<path d="M3 13h4l2-3 2.5 8 3-14 2 9h4.5" ' + _A + '/>'),
    "histogram": ('surface', '<path d="M4 20h16" ' + _S + '/><rect x="5" y="12" width="3" height="8" ' + _F + '/><rect x="10.5" y="6" width="3" height="14" ' + _F + '/><rect x="16" y="10" width="3" height="10" ' + _F + '/>'),
    "yolo_detect": ('trained', '<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" ' + _A + '/><rect x="8.5" y="8.5" width="7" height="7" rx="1.5" ' + _S + '/><path d="M10.5 6.5v2M13.5 6.5v2M10.5 15.5v2M13.5 15.5v2M6.5 10.5h2M6.5 13.5h2M15.5 10.5h2M15.5 13.5h2" stroke="{c}" stroke-width="1.2" stroke-linecap="round" fill="none"/>'),
    "yolo_defect": ('trained', '<rect x="4" y="4" width="16" height="16" rx="3" ' + _A + '/><path d="M12 8v5" ' + _S + '/><circle cx="12" cy="16" r=".9" fill="{c}"/>'),
    "text_detect": ('text', '<rect x="3.5" y="5" width="17" height="14" rx="2" stroke-dasharray="2.5 2" ' + _A + '/><path d="M8 9.5h8M12 9.5v6" ' + _S + '/>'),
    "ocr_auto": ('text', '<path d="M4 17 7.5 7l3.5 10M5.3 13.5h4.4" ' + _S + '/><circle cx="16" cy="13" r="3.5" ' + _A + '/><path d="m18.5 15.5 2 2" ' + _A + '/>'),
    "ocr": ('text', '<path d="M3.5 17 7.5 6.5l4 10.5M5 13h5" ' + _S + '/><path d="M18.5 11v6M18.5 13.5a2.5 2.5 0 1 0 0 1" ' + _A + '/>'),
    "ocv": ('text', '<path d="M3.5 17 7.5 6.5l4 10.5M5 13h5" ' + _S + '/><path d="m13.5 13.5 2.5 2.5 4.5-5.5" ' + _A + '/>'),
    "code_read": ('text', '<path d="M4 5v14M7 5v14M9.5 5v14M13 5v14M15 5v14M18 5v14M20 5v14" stroke="{a}" stroke-width="1.4" fill="none"/>'),
    "find_ellipse": ('geometry', '<ellipse cx="12" cy="12" rx="8.5" ry="5.5" ' + _A + '/><circle cx="12" cy="6.5" r="1.3" ' + _F + '/><circle cx="20.5" cy="12" r="1.3" ' + _F + '/><circle cx="12" cy="17.5" r="1.3" ' + _F + '/><circle cx="3.5" cy="12" r="1.3" ' + _F + '/>'),
    "sharpness": ('image', '<circle cx="12" cy="12" r="8" ' + _S + '/><path d="M12 4a8 8 0 0 1 0 16z" ' + _F + '/><path d="M12 8v8" stroke="#FFFFFF" stroke-width="1.2"/>'),
    "color_segment": ('colour', '<rect x="4" y="4" width="16" height="16" rx="2.5" ' + _S + '/><path d="M8 17c0-3 2-4 4-4s4-3 4-6" stroke="{a}" stroke-width="3.2" stroke-linecap="round" fill="none"/>'),
    "classify": ('trained', '<rect x="3.5" y="4.5" width="7" height="7" rx="1.5" ' + _A + '/><rect x="3.5" y="13" width="7" height="7" rx="1.5" ' + _S + '/><path d="M13.5 8h7M13.5 16.5h7" ' + _S + '/><path d="m5.3 8 1.3 1.3 2.3-2.6" stroke="{a}" stroke-width="1.4" fill="none" stroke-linecap="round"/>'),
    "affine_transform": ('image', '<path d="M4 8 12 4l8 4-8 4z" ' + _S + '/><rect x="6" y="14" width="12" height="6" rx="1" ' + _A + '/><path d="M12 12v2" ' + _S + '/>'),
    "polar_unwrap": ('image', '<circle cx="8" cy="9" r="4.5" ' + _S + '/><circle cx="8" cy="9" r="1.6" ' + _S + '/><rect x="4" y="16" width="16" height="4" rx="1" ' + _A + '/><path d="M13 9h6M17 7l2 2-2 2" ' + _S + '/>'),
    "image_combine": ('image', '<rect x="3.5" y="4.5" width="10" height="10" rx="1.5" ' + _S + '/><rect x="10.5" y="9.5" width="10" height="10" rx="1.5" ' + _A + '/><path d="M10.5 9.5h3v5h-3z" ' + _F + '/>'),
    "fixture_npoint": ('locate', '<circle cx="6" cy="7" r="2.3" ' + _F + '/><circle cx="18" cy="17" r="2.3" ' + _F + '/><path d="M7.8 8.5 16.2 15.5" stroke-dasharray="2 2" ' + _S + '/><path d="M4 20h6M4 20v-6" ' + _S + '/>'),
    "calibrate_npoint": ('locate', '<path d="M4 4h16v16H4z" ' + _S + '/><path d="M4 12h16M12 4v16" stroke="{c}" stroke-width="1" opacity=".5" fill="none"/><circle cx="8" cy="8" r="1.5" ' + _F + '/><circle cx="16" cy="8" r="1.5" ' + _F + '/><circle cx="8" cy="16" r="1.5" ' + _F + '/><circle cx="16" cy="16" r="1.5" ' + _F + '/>'),
    "construct": ('geometry', '<circle cx="6" cy="17" r="1.8" ' + _F + '/><circle cx="18" cy="7" r="1.8" ' + _F + '/><path d="M3 19.5 21 4.5" ' + _A + '/><path d="M12 12l4 4M14.5 12H12v2.5" ' + _S + '/>'),
    "bead": ('surface', '<path d="M3 15c3-6 6 2 9-3s6 1 9-4" stroke="{a}" stroke-width="3.4" stroke-linecap="round" fill="none"/><path d="M6 19v-3M12 19v-3M18 19v-3" ' + _S + '/>'),
    "data_analysis": ('logic', '<path d="M4 20V4M4 20h16" ' + _S + '/><path d="M6 15c3-8 5-8 7-3s4 4 6-2" ' + _A + '/><path d="M4 8h16" stroke="{c}" stroke-width="1" stroke-dasharray="2 2" fill="none"/>'),
    "results_analysis": ('logic', '<rect x="3.5" y="4" width="17" height="16" rx="2.5" ' + _A + '/><path d="M8 16l1.5-8.5a1.8 1.8 0 0 1 3-1M7.5 11h4.5M13.5 11l4 5M17.5 11l-4 5" ' + _S + '/>'),
    "measure_length": ('measure', '<path d="M3.5 16.5 16.5 3.5l4 4-13 13z" ' + _A + '/><path d="m7 13 1.5 1.5M10 10l2 2M13 7l1.5 1.5" ' + _S + '/>'),
}

APP = ('<rect x="1.5" y="1.5" width="21" height="21" rx="5.5" fill="#1F6FB2"/>'
       '<circle cx="12" cy="12" r="6.2" fill="none" stroke="#FFFFFF" stroke-width="1.8"/>'
       '<circle cx="12" cy="12" r="2.4" fill="#7FD1FF"/>'
       '<path d="M16.4 16.4 20 20" stroke="#FFFFFF" stroke-width="1.8" stroke-linecap="round"/>')

_cache = {}

# ------------------------------------------------------------ Phosphor
# action -> (Phosphor name, coloured with the accent rather than the ink)
PHOSPHOR = {
    "run": ("play-fill", True), "run_continuous": ("repeat", True), "stop": ("stop-fill", True),
    "open": ("folder-open", False), "save": ("floppy-disk", False), "export": ("export", False),
    "import": ("download-simple", False), "python": ("code", False),
    "reset": ("arrow-counter-clockwise", False), "toolbox": ("briefcase", False),
    "image_source": ("camera", False), "electric": ("lightning", True),
    "floating": ("app-window", False), "grid_results": ("table", False),
    "help": ("question", False), "pointer": ("cursor", False), "pan": ("hand", False),
    "zoom_in": ("magnifying-glass-plus", False), "zoom_out": ("magnifying-glass-minus", False),
    "fit": ("corners-out", False), "actual": ("frame-corners", False),
    "grid": ("grid-four", False), "graphics": ("selection", False),
    "guides": ("crosshair", False), "exposure": ("aperture", False),
    "history": ("clock-counter-clockwise", False), "tag": ("tag", False),
    "rect": ("rectangle", False), "ellipse": ("circle", False), "polygon": ("polygon", False),
    "build": ("wrench", False), "layout": ("layout", False), "output": ("image-square", False),
    "train": ("graduation-cap", False), "test": ("flask", False), "add": ("plus", False),
    "delete": ("trash", False), "close": ("x", False), "rename": ("pencil-simple", False),
    "up": ("caret-up", False), "down": ("caret-down", False), "cascade": ("stack", False),
    "tile": ("squares-four", False), "counts": ("hash", False), "job": ("clipboard-text", False),
    "align_left": ("align-left", False), "align_hcenter": ("align-center-horizontal", False),
    "align_right": ("align-right", False), "align_top": ("align-top", False),
    "align_vcenter": ("align-center-vertical", False), "align_bottom": ("align-bottom", False),
    "distribute_h": ("columns", False), "distribute_v": ("rows", False),
    "match_width": ("arrows-out-line-horizontal", False),
    "match_height": ("arrows-out-line-vertical", False),
    "rotate_cw": ("arrow-clockwise", False), "rotate_ccw": ("arrow-counter-clockwise", False),
    "rotate_zero": ("arrow-u-up-left", False), "lock": ("lock-simple", False),
    "unlock": ("lock-simple-open", False), "duplicate": ("copy", False),
    "zoom_selection": ("magnifying-glass", False), "gear": ("gear-six", False),
    "more": ("dots-three", False), "sliders": ("sliders-horizontal", False),
    "network": ("wifi-high", False), "barcode": ("barcode", False), "photo": ("image", False),
    "sparkle": ("sparkle", True), "image_process": ("magic-wand", False),
    "find_picture": ("magnifying-glass", False), "light": ("lightbulb", False),
}
# tool type -> Phosphor name; drawn in its family's colour
PHOSPHOR_TOOLS = {
    "image_process": "magic-wand", "align": "crosshair", "calibrate_scale": "ruler",
    "presence": "check-square-offset", "count_template": "circles-four",
    "count_blob": "circles-three", "color_match": "palette", "color_classify": "lightbulb",
    "led_glow": "lightbulb-filament", "find_line": "line-segment", "find_circle": "circle-dashed",
    "caliper": "arrows-left-right", "fit_shape": "pen-nib", "intersect": "intersect",
    "distance": "ruler", "trained_surface": "brain", "defect_diff": "magnifying-glass",
    "anomaly": "warning-octagon", "histogram": "chart-bar", "yolo_detect": "robot",
    "yolo_defect": "robot", "text_detect": "text-t", "ocr_auto": "text-t", "ocr": "text-t",
    "ocv": "text-aa", "code_read": "barcode", "find_ellipse": "circle-half",
    "sharpness": "aperture", "color_segment": "paint-bucket", "classify": "tag",
    "affine_transform": "arrows-out", "polar_unwrap": "arrows-clockwise",
    "image_combine": "intersect", "fixture_npoint": "map-pin", "calibrate_npoint": "grid-four",
    "construct": "compass", "bead": "path", "data_analysis": "calculator",
    "results_analysis": "list-numbers", "measure_length": "ruler",
}
_qta = None


def _phosphor():
    """QtAwesome, or None when it is not installed (or VS_ICONS=drawn)."""
    global _qta
    if _qta is None:
        _qta = False
        if os.environ.get("VS_ICONS", "").lower() != "drawn":
            try:
                os.environ.setdefault("QT_API", "pyside6")
                import qtawesome
                qtawesome.icon("ph.cursor")          # its font loads, or this fails
                _qta = qtawesome
            except Exception:
                _qta = False
    return _qta or None


def _ph_icon(name, colour, on=None):
    qta = _phosphor()
    if qta is None:
        return None
    try:
        opts = {"color": colour, "color_active": colour, "color_disabled": "#A0A4AA"}
        if on is not None:
            opts.update(color_on=on, color_on_active=on)
        return qta.icon("ph." + name, **opts)
    except Exception:
        return None


def _svg(body, ink, accent):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">%s</svg>'
            % body.replace("{c}", ink).replace("{a}", accent))


def _render(svg, sizes=(16, 20, 24, 32, 48), ic=None, state=None):
    from PySide6.QtSvg import QSvgRenderer
    r = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    ic = ic if ic is not None else QIcon()
    for n in sizes:
        for scale in (1, 2):                 # crisp on high-DPI screens too
            pm = QPixmap(n * scale, n * scale)
            pm.setDevicePixelRatio(scale)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.Antialiasing)
            r.render(p, QRectF(0, 0, n, n))
            p.end()
            if state is None:
                ic.addPixmap(pm)
            else:
                ic.addPixmap(pm, QIcon.Normal, state)
    return ic


ACTION_ACCENT = "#2563EB"


def set_ink(ink, accent=None):
    """The theme's stroke colour: dark ink on the light theme, light on the
    dark one. Icons made after this use it; cached ones keep their own key."""
    global INK, ACTION_ACCENT
    INK = ink
    if accent:
        ACTION_ACCENT = accent


def icon(name, ink=None, accent=None, on=None):
    """An action icon. Unknown names give an empty icon, never an error.
    on: the ink for the checked state -- white for a button that turns solid
    accent when checked, where the normal ink would be lost on the blue."""
    if name not in ACTIONS and name not in PHOSPHOR:
        return QIcon()
    ink = ink or INK
    acc = accent or {"run": ACCENTS["run"], "run_continuous": ACCENTS["run"],
                     "stop": ACCENTS["stop"], "electric": ACCENTS["warn"]}.get(name, ACTION_ACCENT)
    key = ("a", name, ink, acc, on)
    if key not in _cache and name in PHOSPHOR:
        ph, accented = PHOSPHOR[name]
        ic = _ph_icon(ph, acc if accented else ink, on)
        if ic is not None:
            _cache[key] = ic
    if key not in _cache:
        if on is None:
            _cache[key] = _render(_svg(ACTIONS[name], ink, acc))
        else:
            ic = _render(_svg(ACTIONS[name], ink, acc), state=QIcon.Off)
            _cache[key] = _render(_svg(ACTIONS[name], on, on), ic=ic, state=QIcon.On)
    return _cache[key]


def tool_icon(tool_type, ink=None):
    ink = ink or INK
    """A tool type's icon in its family's accent colour."""
    family, body = TOOLS.get(tool_type, ("logic", '<rect x="5" y="5" width="14" height="14" rx="3" ' + _A + '/>'))
    key = ("t", tool_type, ink)
    if key not in _cache and tool_type in PHOSPHOR_TOOLS:
        ic = _ph_icon(PHOSPHOR_TOOLS[tool_type], ACCENTS[family])
        if ic is not None:
            _cache[key] = ic
    if key not in _cache:
        _cache[key] = _render(_svg(body, ink, ACCENTS[family]))
    return _cache[key]


APP_ICON_DIR = os.path.join(paths.ASSETS, "icon")


def app_icon():
    """The application's icon (assets/icon): the full drawing from 32 px up,
    the lens alone below that. Windows, the taskbar, the .exe and the
    shortcuts all show this one picture."""
    if "app" not in _cache:
        big = os.path.join(APP_ICON_DIR, "vision_station.svg")
        small = os.path.join(APP_ICON_DIR, "vision_station_small.svg")
        if os.path.isfile(big) and os.path.isfile(small):
            ic = QIcon()
            for path, sizes in ((small, (16, 20, 24)), (big, (32, 40, 48, 64, 96, 128, 256))):
                with open(path, encoding="utf-8") as f:
                    _render(f.read(), sizes=sizes, ic=ic)
            _cache["app"] = ic
        else:
            _cache["app"] = _render(_svg(APP, INK, INK), sizes=(16, 24, 32, 48, 64, 128, 256))
    return _cache["app"]
