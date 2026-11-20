"""Each inspected part's picture, saved where Settings > Results says: the
full-resolution image with the result page's graphics drawn on it (boxes,
names, scores, the verdict), and/or the original image untouched.

Written on a background thread from a short queue, so saving a 20-megapixel
picture -- 150 ms as JPEG, about a second as PNG -- never lengthens an
inspection. When the disk cannot keep up the oldest waiting picture is
dropped, and counted, rather than the line being held up.
"""
import os
import queue
import threading
import time
from datetime import datetime

import cv2

from vision_station.core import imio


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name)) or "job"


def wanted(cfg, verdict):
    if not cfg or not cfg.get("folder"):
        return False
    if not (cfg.get("annotated", True) or cfg.get("original", False)):
        return False
    which = cfg.get("which", "all")
    if which == "fail":
        return verdict != "PASS"
    if which == "pass":
        return verdict == "PASS"
    return which == "all"


def annotate(image, result, title="", labels=True, banner=True):
    """The result page's picture: every check's graphics on the image."""
    from vision_station.core.vsruntime import draw
    return draw(image, result, banner=banner, title=title, labels=labels)


DAY_FORMAT = "%Y-%m-%d"


def day_folder(when=None):
    """The name of the day's folder: 2026-11-19."""
    return datetime.fromtimestamp(when or time.time()).strftime(DAY_FORMAT)


def folder_for(cfg, when=None):
    """Where a result's pictures go: always in the day's folder, and inside
    it the part's own folder when it has one (its barcode):

        <folder>/2026-11-19/                 parts without a barcode
        <folder>/2026-11-19/FIC12345678/     each scanned part

    One day's work is then one folder, whatever way the line runs."""
    folder = os.path.join(cfg["folder"], day_folder(when))
    if cfg.get("subfolder"):
        folder = os.path.join(folder, cfg["subfolder"])
    return folder


def paths_for(cfg, job_name, verdict, when=None):
    """(annotated path, original path) for a result at time `when`."""
    when = when or time.time()
    dt = datetime.fromtimestamp(when)
    folder = folder_for(cfg, when)
    stem = "%s_%s_%s" % (dt.strftime("%H%M%S_%f")[:-3], _safe(job_name),
                         "ACCEPT" if verdict == "PASS" else "REJECT")
    ext = ".png" if cfg.get("format") == "png" else ".jpg"
    return os.path.join(folder, stem + ext), os.path.join(folder, stem + "_original" + ext)


def write(cfg, image, result, job_name, when=None):
    """Write the picture(s) for one result now. Returns the paths written."""
    ann_path, orig_path = paths_for(cfg, job_name, result.get("verdict"), when)
    os.makedirs(os.path.dirname(ann_path), exist_ok=True)
    if cfg.get("format") == "png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    else:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.get("quality", 97))]
    out = []
    if cfg.get("annotated", True):
        pic = annotate(image, result, job_name, cfg.get("labels", True),
                       cfg.get("banner", True))
        if imio.imwrite(ann_path, pic, params):
            out.append(ann_path)
    if cfg.get("original", False):
        if imio.imwrite(orig_path, image, params):
            out.append(orig_path)
    return out


class ResultImageWriter:
    def __init__(self, depth=6):
        self.cfg = {}
        self._q = queue.Queue(maxsize=depth)
        self.dropped = 0
        self.written = 0
        self.last_paths = []
        self.last_error = ""
        # told the paths of every picture written (the Find picture index)
        self.on_written = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def configure(self, cfg):
        self.cfg = dict(cfg or {})
        # the OK / NG colours and line thickness chosen in Settings > Results
        # apply to every result drawn -- on screen and in saved pictures
        from vision_station.core import annotate as _look
        _look.set_look(self.cfg.get("ok_colour"), self.cfg.get("ng_colour"),
                       self.cfg.get("line_scale"))

    def submit(self, image, result, job_name, cfg=None):
        """From the window thread, once per result. Never blocks. cfg: this
        result's own settings (a barcode's folder), else the configured ones.
        The path the picture will be written to (the annotated one when it
        is kept, else the original), or None when none is kept."""
        cfg = self.cfg if cfg is None else cfg
        if image is None or not wanted(cfg, result.get("verdict")):
            return None
        when = time.time()
        ann, orig = paths_for(cfg, job_name, result.get("verdict"), when)
        item = (dict(cfg), image, result, job_name, when)
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self.dropped += 1
            try:
                self._q.put_nowait(item)
            except queue.Full:
                pass
        return ann if cfg.get("annotated", True) else orig

    def _run(self):
        while True:
            cfg, image, result, job_name, when = self._q.get()
            try:
                self.last_paths = write(cfg, image, result, job_name, when)
                self.written += 1
                self.last_error = ""
                for fn in list(self.on_written):
                    try:
                        fn(self.last_paths)
                    except Exception as e:
                        print("[results] %s" % e)
            except Exception as e:                # saving must never stop the line
                self.last_error = str(e)
                print("[results] picture not saved: %s" % e)
            finally:
                self._q.task_done()

    def flush(self, timeout_s=10.0):
        t0 = time.time()
        while self._q.unfinished_tasks and time.time() - t0 < timeout_s:
            time.sleep(0.01)
