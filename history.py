"""Undo/redo for the recipe.

Snapshot-based rather than command-based. A recipe is a few kilobytes of JSON, so
copying the whole thing per edit costs nothing measurable, and it cannot drift out
of sync with the real state the way hand-written undo/redo command pairs do once
there are a dozen edit paths.

The part that needs thought is coalescing. Dragging a region fires geometry
updates continuously and holding an arrow key repeats; a spinbox sweep fires per
tick. Without coalescing, one gesture becomes fifty undo steps and Ctrl+Z stops
being useful. So edits with the same label arriving within COALESCE_SEC of each
other collapse into the one entry that holds the state from before the gesture
started.
"""

import copy
import time

COALESCE_SEC = 0.9
LIMIT = 100


class History:
    def __init__(self, limit=LIMIT):
        self.limit = limit
        self._undo = []          # list of (label, snapshot)
        self._redo = []
        self._last_label = None
        self._last_time = 0.0
        self._muted = False

    def reset(self, snapshot=None):
        self._undo = []
        self._redo = []
        self._last_label = None
        self._last_time = 0.0

    def mute(self, on=True):
        """Used while applying an undo, so restoring state does not itself get
        recorded as a new edit."""
        self._muted = on

    def push(self, label, snapshot):
        """Call BEFORE mutating. snapshot is the state as it is right now."""
        if self._muted:
            return
        now = time.monotonic()
        if (self._undo and label == self._last_label
                and now - self._last_time < COALESCE_SEC):
            # Same gesture continuing: keep the older snapshot, refresh the clock.
            self._last_time = now
            return
        self._undo.append((label, copy.deepcopy(snapshot)))
        if len(self._undo) > self.limit:
            self._undo.pop(0)
        self._redo.clear()
        self._last_label = label
        self._last_time = now

    def break_coalesce(self):
        """Force the next push to start a new entry even if the label matches."""
        self._last_label = None

    def can_undo(self):
        return bool(self._undo)

    def can_redo(self):
        return bool(self._redo)

    def undo_label(self):
        return self._undo[-1][0] if self._undo else ""

    def redo_label(self):
        return self._redo[-1][0] if self._redo else ""

    def undo(self, current):
        if not self._undo:
            return None, ""
        label, snap = self._undo.pop()
        self._redo.append((label, copy.deepcopy(current)))
        self._last_label = None
        return snap, label

    def redo(self, current):
        if not self._redo:
            return None, ""
        label, snap = self._redo.pop()
        self._undo.append((label, copy.deepcopy(current)))
        self._last_label = None
        return snap, label
