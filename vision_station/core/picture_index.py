"""Every saved result picture, searchable as you type.

The folders the application saves pictures in are walked once, on a
background thread, into a list held in memory; each later picture is added
as it is written. A search is then a scan of that list -- about 20 ms for a
hundred thousand pictures -- so the Find picture window answers every key
press without touching the disk.

What a picture is found by is its path below the results folder:

    2026-11-19/FIC12345678/143502_118_Top_REJECT.jpg
    ^ day      ^ barcode    ^ time     ^ job ^ verdict

so a barcode, part of one, a date or a job name all find it.
"""
import os
import re
import threading
import time

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME = re.compile(r"^(\d{2})(\d{2})(\d{2})_(\d{3})_")
MAX_DEPTH = 4             # root/day/code/file is 3; one more for safety


class Picture:
    __slots__ = ("path", "root", "day", "code", "job", "verdict", "time", "original",
                 "mtime", "key")

    def __init__(self, path, root, mtime):
        self.path = path
        self.root = root
        self.mtime = mtime
        rel = os.path.relpath(path, root)
        parts = rel.replace("\\", "/").split("/")
        folders, name = parts[:-1], parts[-1]
        self.day = next((p for p in folders if _DAY.match(p)), "")
        # the barcode is the folder the picture is in, unless that is the
        # day's folder (a part without a code) or the root itself
        self.code = folders[-1] if folders and not _DAY.match(folders[-1]) else ""
        stem = os.path.splitext(name)[0]
        self.original = stem.endswith("_original")
        if self.original:
            stem = stem[:-len("_original")]
        self.verdict = "ACCEPT" if stem.endswith("_ACCEPT") else \
            "REJECT" if stem.endswith("_REJECT") else ""
        m = _TIME.match(stem)
        self.time = "%s:%s:%s" % m.groups()[:3] if m else \
            time.strftime("%H:%M:%S", time.localtime(mtime))
        if not self.day:
            self.day = time.strftime("%Y-%m-%d", time.localtime(mtime))
        job = stem[m.end():] if m else stem
        for tail in ("_ACCEPT", "_REJECT"):
            if job.endswith(tail):
                job = job[:-len(tail)]
        self.job = job.replace("_", " ")
        self.key = rel.lower()

    def folder(self):
        return os.path.dirname(self.path)


def _walk(root, depth=0):
    """(path, mtime) of every picture below root. os.scandir: on Windows the
    directory listing already carries each file's times, so this costs one
    call per folder, not one per picture."""
    try:
        it = os.scandir(root)
    except OSError:
        return
    with it:
        for e in it:
            try:
                if e.is_dir(follow_symlinks=False):
                    if depth < MAX_DEPTH:
                        yield from _walk(e.path, depth + 1)
                elif os.path.splitext(e.name)[1].lower() in IMAGE_EXT:
                    yield e.path, e.stat().st_mtime
            except OSError:
                continue


class PictureIndex:
    """Thread-safe: the index is built and added to from background threads
    while the window searches it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._items = {}             # normalised path -> Picture
        self._sorted = []            # newest first
        self._dirty = False
        self.roots = []
        self.scanning = False
        self.version = 0             # bumped on every change, for windows to poll
        self.scanned_at = 0.0

    # -- building -----------------------------------------------------------
    def set_roots(self, roots):
        seen, out = set(), []
        for r in roots:
            if not r:
                continue
            n = os.path.normcase(os.path.abspath(r))
            if n not in seen:
                seen.add(n)
                out.append(os.path.abspath(r))
        self.roots = out

    def scan(self):
        """Walk every root again, now. Pictures deleted since are dropped."""
        self.scanning = True
        try:
            fresh = {}
            for root in list(self.roots):
                for path, mtime in _walk(root):
                    fresh[os.path.normcase(path)] = Picture(path, root, mtime)
            with self._lock:
                self._items = fresh
                self._dirty = True
                self.version += 1
            self.scanned_at = time.time()
        finally:
            self.scanning = False

    def scan_in_background(self):
        if self.scanning:
            return None
        t = threading.Thread(target=self.scan, name="picture-index", daemon=True)
        self.scanning = True
        t.start()
        return t

    def add(self, paths):
        """Pictures just written. Ones outside every root are ignored."""
        with self._lock:
            for p in paths or []:
                ap = os.path.abspath(p)
                root = next((r for r in self.roots
                             if os.path.normcase(ap).startswith(os.path.normcase(r) + os.sep)),
                            None)
                if root is None:
                    continue
                try:
                    mtime = os.path.getmtime(ap)
                except OSError:
                    mtime = time.time()
                self._items[os.path.normcase(ap)] = Picture(ap, root, mtime)
            self._dirty = True
            self.version += 1

    def __len__(self):
        return len(self._items)

    def _newest_first(self):
        with self._lock:
            if self._dirty:
                self._sorted = sorted(self._items.values(), key=lambda p: -p.mtime)
                self._dirty = False
            return self._sorted

    # -- searching ----------------------------------------------------------
    def search(self, query, limit=500, verdict=None, originals=True):
        """(pictures, total matching). Every word typed must appear in the
        picture's path; a picture whose barcode is exactly what was typed
        comes first, then barcodes starting with it, then the rest -- each
        group newest first."""
        words = [w for w in str(query or "").lower().split() if w]
        items = self._newest_first()
        exact, starts, rest = [], [], []
        whole = " ".join(words)
        for p in items:
            if verdict and p.verdict != verdict:
                continue
            if not originals and p.original:
                continue
            k = p.key
            if words and not all(w in k for w in words):
                continue
            c = p.code.lower()
            if whole and c == whole:
                exact.append(p)
            elif whole and c.startswith(whole):
                starts.append(p)
            else:
                rest.append(p)
        found = exact + starts + rest
        return found[:limit], len(found)


def open_in_explorer(path):
    """Show the picture's folder in Explorer with the picture selected; on
    other systems, open the folder."""
    import subprocess
    import sys
    path = os.path.normpath(path)
    if sys.platform.startswith("win"):
        if os.path.isfile(path):
            subprocess.Popen('explorer /select,"%s"' % path)
        else:
            os.startfile(path if os.path.isdir(path) else os.path.dirname(path))
    else:
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", folder])
