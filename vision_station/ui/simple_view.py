"""The simple view: what Vision Station looks like to someone who did not
build it.

The VisionPro-style windows in main_window.py are the engineering view, kept whole
behind "Advanced mode". In front of them, this module puts:

    ModeSwitch      Run | Build | History - the three things anyone does
    VerdictBanner   ACCEPT / REJECT in large type, with the reasons in words
    CheckPicker     "What do you want to check?" - the tools as questions
    GuidePanel      the five steps of setting up an inspection, ticked off
    WelcomePage     a first start with three clear choices

and the plain-language helpers the windows use: a check's name as a person
reads it, and a tool's message as a sentence rather than an engineer's note.
"""
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QLayout, QLineEdit, QPushButton,
                               QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from vision_station.ui import icons
from vision_station.ui import theme
from vision_station.core.schema import ADVANCED_TYPES, LABELS, QUESTIONS


# ---------------------------------------------------------------- words
def kind_of(tool):
    return LABELS.get(tool.get("type"), (tool.get("type", "?"), ""))[0]


def display_name(tool):
    """'Verify text 1' for a check still carrying its automatic name,
    'lot code  -  Verify text' for one that has been named."""
    tid = tool.get("id", "")
    base, _, n = tid.rpartition("_")
    if n.isdigit() and base == tool.get("type"):
        return "%s %s" % (kind_of(tool), n)
    return "%s   ·   %s" % (tid.replace("_", " "), kind_of(tool))


def short_name(tool):
    tid = tool.get("id", "")
    base, _, n = tid.rpartition("_")
    if n.isdigit() and base == tool.get("type"):
        return "%s %s" % (kind_of(tool), n)
    return tid.replace("_", " ")


# A tool's message, as it would be said to someone at the line. Matched on a
# fragment of the engineer's message; {detail} is the original.
_FRIENDLY = [
    ("no OCR engine installed",
     "Text reading is not installed on this PC. It needs PaddleOCR - whoever set the PC "
     "up can add it with install.bat."),
    ("no inference runtime installed",
     "The AI runtime (OpenVINO or onnxruntime) is not installed on this PC."),
    ("inference runtime:", "The AI runtime could not run this model: {detail}"),
    ("model file not found", "The trained model file is missing. {detail}"),
    ("model not found", "The trained model file is missing. {detail}"),
    ("no blobs from source", "Nothing to read: the counting check it depends on found nothing, "
                             "or is not chosen under Settings."),
    ("reference image", "This check needs the reference picture of a good part."),
    ("empty region", "The box is empty. Drag it over the feature on the picture."),
    ("no region", "No box yet. Draw one on the picture with the rectangle tool."),
    ("no pixels in region", "The box is empty. Drag it over the feature on the picture."),
    ("set Sample region", "Choose one example to count: under Settings, set Sample region to "
                          "its x, y, width and height."),
    ("ROI smaller than template", "The box is smaller than the example to count. Make it bigger."),
    ("search window too small", "The box is too small for the search. Make it bigger."),
    ("no expression set", "No condition set yet. Type one under Settings."),
    ("no path", "Draw the bead's path as a polygon region."),
    ("points do not agree", "The features were found, but not where they should be relative "
                            "to each other - the part may be damaged or a feature mis-found."),
    ("has no value", "The value to follow does not exist. Check the name under Settings."),
    ("set source as tool.value", "Choose which value to follow under Settings, like "
                                 "measure_length_1.value."),
    ("needs two Find line / Find circle", "Choose the two features to measure between under "
                                          "Settings - they must be found by earlier checks."),
    ("needs a positive known_mm", "Set the real length in millimetres under Settings."),
    ("need two edges", "Two edges were not found in the box. Move it over the feature, or "
                       "lower Minimum contrast."),
]
_NOMATCH = re.compile(r"read '(.*)' does not match /(.*)/")


def friendly_message(tres):
    """One sentence for a tool's result."""
    if not tres:
        return ""
    msg = (tres.get("message") or "").strip()
    status = tres.get("status")
    low = msg.lower()
    for key, text in _FRIENDLY:
        if key.lower() in low:
            return text.format(detail=msg)
    m = _NOMATCH.search(msg)
    if m:
        return "Read '%s', which does not match the expected pattern %s." % (m.group(1), m.group(2))
    if status == "fail":
        if msg:
            return msg[0].upper() + msg[1:]
        if "score" in tres:
            return "Did not pass (score %.2f)." % float(tres.get("score") or 0.0)
        return "Did not pass."
    if status == "error":
        return "Could not run%s" % ((": " + msg) if msg else ".")
    return msg


def reasons(result, tools=None, prefix=""):
    """['Lot code: Read ..., which does not match ...'] for every check that
    did not pass in a job's result."""
    byid = {t["id"]: t for t in (tools or [])}
    out = []
    for tres in (result or {}).get("tools", []):
        if tres.get("status") == "pass":
            continue
        t = byid.get(tres.get("tool_id"))
        name = short_name(t) if t else tres.get("tool_id", "?")
        out.append("%s%s: %s" % (prefix, name[0].upper() + name[1:], friendly_message(tres)))
    return out


def failed_names(result, tools=None):
    byid = {t["id"]: t for t in (tools or [])}
    out = []
    for tres in (result or {}).get("tools", []):
        if tres.get("status") != "pass":
            t = byid.get(tres.get("tool_id"))
            out.append(short_name(t) if t else tres.get("tool_id", "?"))
    return out


# ---------------------------------------------------------------- mode switch
class ModeSwitch(QWidget):
    """Run | Build | History, as a segmented control."""
    changed = Signal(str)
    MODES = (("run", "Run", "job", "Watch parts being inspected  (Ctrl+1)"),
             ("build", "Build", "build", "Set up what to check  (Ctrl+2)"),
             ("history", "History", "history", "Yield, and the checks that reject most  (Ctrl+3)"))

    def __init__(self):
        super().__init__()
        self.setObjectName("modeSwitch")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        for key, text, ic, tip in self.MODES:
            b = QToolButton()
            b.setObjectName("modeButton")
            b.setText(text)
            b.setIcon(icons.icon(ic))
            b.setProperty("vsIcon", ic)
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setAccessibleName("%s view" % text)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _c=False, k=key: self._clicked(k))
            self.group.addButton(b)
            lay.addWidget(b)
            self.buttons[key] = b
        self.buttons["run"].setChecked(True)
        self._mode = "run"
        self._lock_tips = {}
        # the white pill under the chosen view, gliding to the next choice
        # (ui/motion.py); the buttons draw no background of their own then
        from PySide6.QtWidgets import QFrame
        self.thumb = QFrame(self)
        self.thumb.setObjectName("modeThumb")
        self.thumb.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.thumb.lower()
        self.setProperty("animated", "true")
        self.group.buttonToggled.connect(lambda b, on: on and self._place_thumb())

    def _place_thumb(self, now=False):
        from vision_station.ui import motion
        b = self.buttons.get(self._mode) or next(
            (x for x in self.buttons.values() if x.isChecked()), None)
        if b is None or b.geometry().isEmpty():
            return
        rect = b.geometry()
        if now:
            self.thumb.setGeometry(rect)
        else:
            motion.glide(self.thumb, rect)
        self.thumb.lower()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_thumb(now=True)

    def showEvent(self, e):
        super().showEvent(e)
        self._place_thumb(now=True)

    def _clicked(self, key):
        if self.buttons[key].isEnabled():
            self._mode = key
            self._place_thumb()
            self.changed.emit(key)
        else:
            self.buttons[self._mode].setChecked(True)

    def mode(self):
        return self._mode

    def set_mode(self, key):
        """Reflect a change made elsewhere; emits nothing."""
        if key in self.buttons:
            self._mode = key
            self.buttons[key].setChecked(True)
            self._place_thumb()

    def set_locked(self, key, locked, why=""):
        b = self.buttons[key]
        b.setEnabled(not locked)
        if locked:
            self._lock_tips[key] = b.toolTip()
            b.setToolTip(why or "Locked")
        elif key in self._lock_tips:
            b.setToolTip(self._lock_tips.pop(key))


# ---------------------------------------------------------------- verdict banner
class VerdictBanner(QFrame):
    """The last inspection's verdict, readable across the room, with the
    reasons written as sentences underneath."""

    def __init__(self):
        super().__init__()
        self.setObjectName("verdict")
        self.setProperty("tone", "idle")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(22, 14, 22, 14)
        lay.setSpacing(18)
        left = QVBoxLayout()
        left.setSpacing(2)
        self.word = QLabel("READY")
        self.word.setObjectName("verdictWord")
        self.reason = QLabel("")
        self.reason.setObjectName("verdictReason")
        self.reason.setWordWrap(True)
        self.reason.setTextInteractionFlags(Qt.TextSelectableByMouse)
        left.addWidget(self.word)
        left.addWidget(self.reason)
        lay.addLayout(left, 1)
        self.meta = QLabel("")
        self.meta.setObjectName("verdictMeta")
        self.meta.setAlignment(Qt.AlignRight | Qt.AlignTop)
        lay.addWidget(self.meta)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

    def set(self, tone, word, reason, meta=""):
        self.word.setText(word)
        self.reason.setText(reason)
        self.reason.setVisible(bool(reason))
        self.meta.setText(meta)
        theme.set_tone(self, tone)
        self.setAccessibleName("%s. %s" % (word, reason))


# ---------------------------------------------------------------- check picker
class _Card(QFrame):
    def __init__(self, ty, picker):
        super().__init__()
        self.ty, self.picker = ty, picker
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFixedHeight(92)
        self.setMinimumWidth(220)
        name, why = LABELS.get(ty, (ty, ""))
        self.name, self.why = name, why
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(12)
        ic = QLabel()
        ic.setPixmap(icons.tool_icon(ty).pixmap(30, 30))
        ic.setFixedSize(34, 34)
        ic.setAlignment(Qt.AlignTop)
        lay.addWidget(ic, 0, Qt.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(2)
        t = QLabel(name)
        t.setObjectName("cardTitle")
        d = QLabel(why)
        d.setObjectName("cardDesc")
        d.setWordWrap(True)
        col.addWidget(t)
        col.addWidget(d)
        col.addStretch(1)
        lay.addLayout(col, 1)
        self.setToolTip(why)
        self.setAccessibleName(name)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.picker.choose(self.ty)
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.picker.choose(self.ty)
        else:
            super().keyPressEvent(e)


class CheckPicker(QDialog):
    """'What do you want to check?' - every tool, grouped by the question it
    answers, with a search box. Returns the chosen tool type."""
    COLS = 3

    def __init__(self, parent=None, advanced=False):
        super().__init__(parent)
        self.setWindowTitle("Add a check")
        self.resize(940, 680)
        self.choice = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 16)
        lay.setSpacing(8)
        title = QLabel("What do you want to check?")
        title.setObjectName("pageTitle")
        sub = QLabel("Pick a question. The check is added to the picture as a box you drag over "
                     "the feature; then you say what to expect.")
        sub.setObjectName("secondary")
        sub.setWordWrap(True)
        lay.addWidget(title)
        lay.addWidget(sub)
        row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setObjectName("searchBox")
        self.search.setPlaceholderText("Search - screws, text, colour, scratch, length...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        self.search.returnPressed.connect(self._choose_first)
        self.adv = QCheckBox("Show advanced checks")
        self.adv.setChecked(advanced)
        self.adv.toggled.connect(self._filter)
        row.addWidget(self.search, 1)
        row.addWidget(self.adv)
        lay.addLayout(row)

        self.area = QScrollArea()
        self.area.setObjectName("tabScroll")
        self.area.setWidgetResizable(True)
        self.area.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        self.body = QVBoxLayout(body)
        self.body.setContentsMargins(0, 6, 8, 6)
        self.body.setSpacing(6)
        self.groups = []                      # (header widgets, grid, cards)
        for question, caption, types in QUESTIONS:
            h = QLabel(question)
            h.setObjectName("question")
            c = QLabel(caption)
            c.setObjectName("questionSub")
            c.setWordWrap(True)
            holder = QWidget()
            grid = QGridLayout(holder)
            grid.setContentsMargins(0, 2, 0, 10)
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(10)
            cards = [_Card(ty, self) for ty in types if ty in LABELS]
            self.body.addWidget(h)
            self.body.addWidget(c)
            self.body.addWidget(holder)
            self.groups.append(((h, c, holder), grid, cards))
        self.body.addStretch(1)
        self.area.setWidget(body)
        lay.addWidget(self.area, 1)
        foot = QHBoxLayout()
        self.hint = QLabel("")
        self.hint.setObjectName("hint")
        foot.addWidget(self.hint, 1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        foot.addWidget(cancel)
        lay.addLayout(foot)
        self._filter()
        self.search.setFocus()

    def _visible(self, card, text):
        if card.ty in ADVANCED_TYPES and not self.adv.isChecked():
            return bool(text) and text in card.name.lower()   # a search still finds it
        if not text:
            return True
        return text in card.name.lower() or text in card.why.lower() or text in card.ty

    def _filter(self, *_):
        text = self.search.text().strip().lower()
        shown = 0
        for (h, c, holder), grid, cards in self.groups:
            while grid.count():
                grid.takeAt(0)
            vis = [k for k in cards if self._visible(k, text)]
            for k in cards:
                k.setVisible(k in vis)
            for i, k in enumerate(vis):
                grid.addWidget(k, i // self.COLS, i % self.COLS)
            for col in range(self.COLS):
                grid.setColumnStretch(col, 1)
            for w in (h, c, holder):
                w.setVisible(bool(vis))
            shown += len(vis)
        self.hint.setText("" if shown else "Nothing matches - try another word, or tick "
                                           "Show advanced checks.")

    def visible_cards(self):
        return [k for _g, _grid, cards in self.groups for k in cards if k.isVisible()]

    def _choose_first(self):
        text = self.search.text().strip().lower()
        vis = self.visible_cards()
        best = next((k for k in vis if text and text in k.name.lower()), vis[0] if vis else None)
        if best is not None:
            self.choose(best.ty)

    def choose(self, ty):
        self.choice = ty
        self.accept()

    @classmethod
    def pick(cls, parent=None, advanced=False):
        dlg = cls(parent, advanced)
        return dlg.choice if dlg.exec() == QDialog.Accepted else None


# ---------------------------------------------------------------- guide
class GuidePanel(QFrame):
    """The five steps of setting up an inspection, with the next one spelt
    out. It reads the job, so a step is ticked when the job shows it done,
    not when a button was pressed."""
    STEPS = (
        ("Reference picture", "This is a picture of a good part. Every check is drawn on it."),
        ("Locate the part", "Add the 'Locate part' check and drag its box around something that "
                            "never changes - a corner, a logo, a hole. The other checks follow it."),
        ("Add checks", "Press Add a check. Drag the blue box over the feature, then say what to "
                       "expect on the right."),
        ("Test it", "Press Run (F5). Each check shows PASS or FAIL in the list; click one to "
                    "see why."),
        ("Save, then switch to Run", "Press Ctrl+S to save. The Run view inspects parts from the "
                                     "camera or a folder of pictures."),
    )

    def __init__(self, on_add=None, on_run=None, on_save=None):
        super().__init__()
        self.setObjectName("guide")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._on = {"add": on_add, "run": on_run, "save": on_save}
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 10, 12)
        lay.setSpacing(4)
        lay.setSizeConstraint(QLayout.SetMinimumSize)     # grows to fit the wrapped hint
        head = QHBoxLayout()
        self.title = QLabel("Setting up this inspection")
        self.title.setObjectName("guideTitle")
        head.addWidget(self.title, 1)
        hide = QToolButton()
        hide.setObjectName("dockClose")
        hide.setAutoRaise(True)
        hide.setIcon(icons.icon("close"))
        hide.setProperty("vsIcon", "close")
        hide.setToolTip("Hide the guide")
        hide.setAccessibleName("Hide the guide")
        hide.clicked.connect(self.hide)
        head.addWidget(hide)
        lay.addLayout(head)
        self.rows = []
        for i, (name, _hint) in enumerate(self.STEPS):
            r = QLabel("%d.  %s" % (i + 1, name))
            r.setObjectName("guideStep")
            r.setProperty("state", "todo")
            lay.addWidget(r)
            self.rows.append(r)
        self.hint = QLabel("")
        self.hint.setObjectName("guideHint")
        self.hint.setWordWrap(True)
        lay.addSpacing(4)
        lay.addWidget(self.hint)
        self.action = QPushButton("")
        self.action.setProperty("accent", True)
        self.action.setCursor(Qt.PointingHandCursor)
        self.action.clicked.connect(self._act)
        lay.addSpacing(4)
        lay.addWidget(self.action, 0, Qt.AlignLeft)
        self._act_key = None

    def _act(self):
        fn = self._on.get(self._act_key)
        if fn:
            fn()

    def update_state(self, tools, result, dirty):
        tools = tools or []
        has_locate = any(t.get("type") == "align" for t in tools)
        n_checks = sum(1 for t in tools if t.get("type") != "align")
        done = [True, has_locate, n_checks > 0, result is not None,
                result is not None and not dirty]
        current = next((i for i, d in enumerate(done) if not d), None)
        # Once there are checks the five steps shrink to the one in hand:
        # the list of checks below needs the room -- forty from Find parts
        # showed three rows at a time.
        compact = n_checks > 0
        self.title.setText("Setting up: step %d of 5" % (current + 1)
                           if compact and current is not None else
                           "Inspection set up" if compact else "Setting up this inspection")
        for i, r in enumerate(self.rows):
            r.setVisible(not compact or i == current)
            name = self.STEPS[i][0]
            if done[i]:
                r.setText("✓   %s" % name)
                state = "done"
            elif i == current:
                r.setText("→   %d.  %s" % (i + 1, name))
                state = "now"
            else:
                r.setText("%d.  %s" % (i + 1, name))
                state = "todo"
            if r.property("state") != state:
                r.setProperty("state", state)
                theme.restyle(r)
        if current is None:
            self.hint.setText("All set. Switch to Run to inspect parts - or add more checks.")
            self.action.setVisible(False)
            self._act_key = None
            return
        self.hint.setText(self.STEPS[current][1])
        key, text = {1: ("add", "Add a check"), 2: ("add", "Add a check"),
                     3: ("run", "Run now  (F5)"), 4: ("save", "Save  (Ctrl+S)")}.get(
                         current, (None, ""))
        self._act_key = key
        self.action.setText(text)
        self.action.setVisible(bool(key and self._on.get(key)))


# ---------------------------------------------------------------- welcome
class WelcomePage(QWidget):
    """A first start: three things to do, each explained in a line."""

    def __init__(self, qb, demo_id=None):
        super().__init__()
        self.setObjectName("welcome")
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        card = QFrame()
        card.setObjectName("welcomeCard")
        card.setMaximumWidth(640)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(30, 26, 30, 26)
        lay.setSpacing(10)
        mark = QLabel()
        mark.setPixmap(icons.app_icon().pixmap(64, 64))
        mark.setFixedSize(64, 64)
        mark.setAccessibleName("Vision Station")
        lay.addWidget(mark)
        t = QLabel("Welcome to Vision Station")
        t.setObjectName("pageTitle")
        s = QLabel("Teach it what a good part looks like, draw the checks, and it tells you "
                   "Accept or Reject for every part that follows.")
        s.setWordWrap(True)
        s.setObjectName("secondary")
        lay.addWidget(t)
        lay.addWidget(s)
        lay.addSpacing(8)
        choices = []
        if demo_id:
            choices.append(("Try the demo", "A small LED panel with a good and a faulty part - "
                            "see every kind of check at work.", "job",
                            lambda: qb.open_application(demo_id)))
        choices += [
            ("Start from a picture of a good part", "Choose a photo. You then draw the checks "
             "on it.", "add", lambda: qb.new_application()),
            ("Open a project", "One already set up on this PC.", "open", qb.choose_application),
            ("Import a package", "A .vsp file exported from another Vision Station.", "import",
             lambda: qb.import_package()),
        ]
        for text, why, ic, fn in choices:
            b = QPushButton("%s\n%s" % (text, why))
            b.setObjectName("bigChoice")
            b.setIcon(icons.icon(ic))
            b.setProperty("vsIcon", ic)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(fn)
            lay.addWidget(b)
        h = QHBoxLayout()
        h.addStretch(1)
        h.addWidget(card, 4)
        h.addStretch(1)
        outer.addLayout(h)
        outer.addStretch(2)
