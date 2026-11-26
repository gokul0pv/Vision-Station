"""Vision Station, laid out like Cognex VisionPro QuickBuild.

Three levels of window, as VisionPro has them:

    QuickBuild      the application: its jobs, a run display per job, run
                    controls, accept/reject counts and the results queues.
    Job Editor      one job: its Image Source and tools as a tree with
                    pass/fail status and input/output terminals, a Toolbox to
                    add tools, a display of the selected tool's last run, and
                    Run Job Once / Run Job Continuously.
    Tool Edit       one tool: Run, Electric, floating display, Open, Save,
    Control         Reset, results grid; its settings tabs; a display with a
                    record selector (Current.InputImage, LastRun.InputImage,
                    LastRun.OutputImage); a status bar with pass/fail and times.

Everything below the windows -- the engine, the tools, the job files -- is the
tested code the classic window uses; this is a different front end on the same
application, not a second implementation. Run it with

    python -m vision_station [projects folder]
"""

# Started as a file -- VS Code's Run button, or "python vision_station/ui/main_window.py" -- rather than
# as part of the package: start the application through the package's entry
# point instead. Importing on from here would load this module twice (once as
# __main__), with two copies of every module-level registry.
if __name__ == "__main__" and not __package__:
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__)))))
    from vision_station.__main__ import main as _main
    _sys.exit(_main([_sys.argv[0]] + _sys.argv[1:]))

import collections
import json
import math
import os
import re
import sys
import time

# PaddleOCR checks it can reach its model hosts every time it starts -- over a
# minute on a line PC with no internet, while the models it wants are already
# on disk. It still downloads a model it does not have.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, QSettings, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QActionGroup, QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDockWidget,
                               QListWidgetItem,
                               QDialog, QComboBox, QFrame, QListWidget,
                               QPushButton, QHBoxLayout, QFormLayout,
                               QFileDialog, QGridLayout, QHeaderView, QInputDialog,
                               QLabel, QLineEdit, QMainWindow, QMdiArea, QMdiSubWindow, QMenu,
                               QMessageBox, QSizePolicy, QSplitter,
                               QTableWidget, QTableWidgetItem, QToolBar, QToolButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from vision_station import paths

HERE = paths.APP_DIR        # projects/ and logs/ are here (beside the .exe when packaged)

from vision_station.core.engine import ProjectStore            # noqa: E402
from vision_station.core import imio                                 # noqa: E402
from vision_station.core.schema import LABELS, defaults  # noqa: E402
from vision_station.ui.display import VisionDisplay       # noqa: E402
from vision_station.core import terminals as T                     # noqa: E402
from vision_station.ui import icons                               # noqa: E402
from vision_station.ui import simple_view as simple                              # noqa: E402
from vision_station.ui import theme                               # noqa: E402
from vision_station.ui.widgets import (EmptyStateTable, confirm_unsaved, dot_icon, dot_style,  # noqa: E402
                   find_tool, guard_wheel, make_action, ok_cancel, read_only_table,
                   readable_images, toolbox_tree, training_columns, training_menu)

LAYOUT_VERSION = 7        # bumped when docks change, so an old saved layout is discarded
QUEUE_LEN = 32                              # VisionPro's default for both queues
PASS_C, FAIL_C, WARN_C, IDLE_C = theme.PASS, theme.FAIL, theme.WARN, theme.IDLE


# ================================================================ theme
def _app_settings():
    return QSettings("VisionStation", "QuickBuild")


def apply_theme(app, mode=None):
    """Windows 11 Fluent, light or dark -- following Windows unless the
    operator chose one under View > Theme. See theme.py."""
    if mode is None:
        try:
            mode = str(_app_settings().value("theme", "system") or "system")
        except Exception:
            mode = "system"
    return theme.apply(app, mode)


def tone_of(status):
    """pass | fail | warn | idle, for anything coloured by a result."""
    return {"pass": "pass", "PASS": "pass", "fail": "fail", "FAIL": "fail",
            "error": "warn", "ERROR": "warn", "WARN": "warn"}.get(status, "idle")


def status_dot(status):
    return {"pass": PASS_C, "PASS": PASS_C, "fail": FAIL_C, "FAIL": FAIL_C,
            "error": WARN_C, "WARN": WARN_C}.get(status, IDLE_C)


def status_text(status):
    """A result written as text: the theme's readable tone, not the dot colour."""
    return QColor(theme.tone_text(tone_of(status)))


def fmt_ms(ms):
    """A duration as people read it: '85 ms', but '35.9 s' rather than
    '35866 ms', which had to be counted digit by digit."""
    ms = float(ms or 0.0)
    return "%.0f ms" % ms if ms < 1000 else "%.1f s" % (ms / 1000.0)


EDITOR_ICON = QSize(18, 18)                 # every command bar, so they line up


def _unique_name(obj, name):
    obj.setObjectName(name)
    return obj


# ================================================================ image source
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def images_in(folder):
    try:
        return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(IMAGE_EXTS))
    except OSError:
        return []


class ImageSource:
    """What feeds a job: its reference image, image files, a folder, or a
    camera. Taught with the job and saved in it (project.json, "image_source"),
    so reopening the application -- or running an exported job file -- uses the
    same images without setting them up again.

    A folder is re-scanned whenever it has been worked through, so images a
    line camera keeps writing into it are picked up.

    grab() returns None when the source cannot deliver -- a camera with no
    frame, an image file that will not load. It used to fall back to the
    reference image, which on a line means a dropped camera inspects the stored
    golden image and passes: the worst failure a vision system can have. The
    station reports an acquisition error for that job instead."""

    def __init__(self, project):
        self.project = project
        self.mode = "reference"
        self.files = []
        self.folder = ""
        self.index = -1
        self.camera = None
        self.camera_index = 0
        self.camera_factory = None           # index -> started camera, set by the station
        self.current = None
        self.current_path = None
        self.last_error = ""
        # Black and white from one colour channel ('G'...): a job whose light
        # is set so takes its camera pictures so (core/light_tuning.to_mono)
        self.mono = None
        # Relative taught paths are relative to the application's folder --
        # the same rule an exported job file applies to its own folder. They
        # were resolved against the current directory, so an application with
        # relative paths found no images when QuickBuild ran from elsewhere.
        self.base = ""

    def _abs(self, path):
        if not path or os.path.isabs(path) or not self.base:
            return path
        return os.path.join(self.base, path)

    # ---- persistence -------------------------------------------------
    def to_spec(self):
        out = {"mode": self.mode, "paths": list(self.files) if self.mode == "files" else [],
               "folder": self.folder if self.mode == "folder" else "",
               "camera": int(self.camera_index)}
        if self.mono:
            out["mono"] = self.mono
        return out

    @classmethod
    def from_spec(cls, project, spec=None):
        src = cls(project)
        spec = spec if spec is not None else (project.spec.get("image_source") or {})
        mode = spec.get("mode", "reference")
        if mode == "files":
            src.files = list(spec.get("paths") or [])
            src.mode = "files" if src.files else "reference"
        elif mode == "folder" and spec.get("folder"):
            src.mode, src.folder = "folder", spec["folder"]
        elif mode == "camera":
            src.mode, src.camera_index = "camera", int(spec.get("camera", 0))
        src.mono = spec.get("mono") or None
        return src

    # ---- choosing ----------------------------------------------------
    def use_reference(self):
        self.stop_camera()
        self.mode = "reference"

    def use_files(self, paths):
        """Paths are kept as given. Dropping ones that did not exist -- and
        falling back to the reference image when none were left -- meant a
        missing part image inspected the stored golden image instead, and the
        line was told PASS. A missing file is an acquisition error when it is
        grabbed. Only an empty list means the reference image."""
        self.stop_camera()
        self.files = list(paths)
        self.index = -1
        self.mode = "files" if self.files else "reference"

    def use_folder(self, folder):
        self.stop_camera()
        self.folder, self.files, self.index = folder, [], -1
        self.mode = "folder"

    def use_camera(self, camera, index=0):
        """camera: anything with get_frame() -- a started CameraThread."""
        self.stop_camera()
        self.camera, self.camera_index = camera, index
        self.mode = "camera"

    def stop_camera(self):
        if self.camera is not None and hasattr(self.camera, "stop"):
            try:
                self.camera.stop()
            except Exception:
                pass
        self.camera = None

    # ---- acquiring ---------------------------------------------------
    def grab(self):
        self.last_error = ""
        img, path = None, None
        if self.mode in ("files", "folder"):
            if self.mode == "folder" and (not self.files or self.index + 1 >= len(self.files)):
                # Re-scan to pick up new images, then carry on AFTER the last
                # one inspected. Restarting from the top re-inspected old parts
                # and reached a newly written image only after all the others.
                self.files = images_in(self._abs(self.folder))
                last = self.current_path
                self.index = self.files.index(last) if last in self.files else -1
            if not self.files:
                self.last_error = ("no images in %s" % self.folder if self.mode == "folder"
                                   else "no image files")
            else:
                self.index = (self.index + 1) % len(self.files)
                path = self._abs(self.files[self.index])
                img = imio.imread(path)
                if img is None:
                    self.last_error = "could not read %s" % os.path.basename(path)
        elif self.mode == "camera":
            if self.camera is None and self.camera_factory is not None:
                try:
                    self.camera = self.camera_factory(self.camera_index)
                except Exception as e:
                    self.last_error = "the camera would not open: %s" % e
            if self.camera is not None:
                img = self._camera_frame()
                if img is not None and self.mono:
                    from vision_station.core.light_tuning import to_mono
                    img = to_mono(img, self.mono)
                if img is None:
                    err = getattr(self.camera, "last_error", None)
                    self.last_error = ("the camera delivered no image"
                                       + (": %s" % err if err else ""))
            elif not self.last_error:
                self.last_error = "no camera"
        else:
            img = self.project.reference
            if img is None:
                self.last_error = "this job has no reference image"
        if img is not None:
            self.current, self.current_path = img, path
        return img

    # Frames to let go by after the trigger. A free-running camera has frames
    # in flight that were exposed before it -- two on the acA5472-5gc -- and at
    # 5 fps the newest frame can predate the part being in position.
    FRESH_FRAMES = 3

    def _camera_frame(self, timeout_s=6.0):
        """A frame exposed after this call, waiting for a camera that is still
        starting. The first part after opening used to fail outright: the
        camera takes about a second to deliver anything."""
        cam = self.camera
        if not hasattr(cam, "frame_after"):    # a plain stand-in
            return cam.get_frame()
        deadline = time.time() + timeout_s
        while cam.backend is None and cam.is_alive() and time.time() < deadline:
            time.sleep(0.02)                   # still opening
        if cam.backend is None:
            return None                        # failed: last_error says why
        # a triggered camera exposes only on the next pulse: one new frame is
        # a frame of the part in position; free-running, frames are in flight
        fresh = min(self.FRESH_FRAMES, int(getattr(cam, "fresh_frames", self.FRESH_FRAMES)
                                           or self.FRESH_FRAMES))
        return cam.frame_after(None, settle=fresh,
                               timeout_s=max(0.5, deadline - time.time()))

    def describe(self):
        if self.mode == "files":
            return "image files (%d), #%d" % (len(self.files), self.index + 1)
        if self.mode == "folder":
            return "folder %s" % self.folder
        if self.mode == "camera":
            return "camera %d" % self.camera_index
        return "reference image"


_shared_cameras = {}          # index -> [CameraThread, users]
# Where Live Camera's exposure and gain are remembered; the main window points
# this at its own settings (a test's private file, in the tests).
camera_settings_store = _app_settings


class SharedCamera:
    """One job's handle on a camera other jobs may be using too.

    An application with a job per side of a product grabs every side from the
    same camera, but a GigE Basler admits one connection: the second job to
    open it was refused ("controlled by another application"). All jobs now
    share one CameraThread per index, and stopping a handle only closes the
    camera when the last job lets go."""

    def __init__(self, index):
        entry = _shared_cameras.get(index)
        if entry is None or not entry[0].is_alive():
            entry = _shared_cameras[index] = [_new_camera_thread(index), 0]
        entry[1] += 1
        self.index, self._entry, self._released = index, entry, False

    @property
    def thread(self):
        """The camera thread -- looked up each time, so every handle follows
        the camera when Settings reopens it at a new address."""
        return self.__dict__["_entry"][0]

    def __getattr__(self, name):              # backend, info, last_error, ...
        return getattr(self.__dict__["_entry"][0], name)

    def get_frame(self):
        return self.thread.get_frame()

    def frame_after(self, *a, **kw):
        return self.thread.frame_after(*a, **kw)

    def grab_sequence(self, *a, **kw):
        return self.thread.grab_sequence(*a, **kw)

    def is_alive(self):
        return self.thread.is_alive()

    def stop(self):
        if self._released:
            return
        self._released = True
        entry = _shared_cameras.get(self.index)
        if entry is None or entry is not self._entry:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            del _shared_cameras[self.index]
            self.thread.stop()


def _new_camera_thread(index, settings=None):
    """A started camera thread with the settings saved for this camera
    (Settings > Camera, and exposure from Live Camera)."""
    from vision_station.hardware.camera import CameraThread
    if settings is None:
        try:
            from vision_station.ui.settings_dialog import load_camera_settings
            settings = load_camera_settings(camera_settings_store(), index)
        except Exception:
            settings = None
    cam = CameraThread(index=index, settings=settings)
    cam.start()
    return cam


def apply_camera_settings(index, settings):
    """New settings for a camera that may be open. Exposure, gain, region,
    trigger... reach the running camera; a new address closes it and opens
    the camera there, and every job's handle follows."""
    entry = _shared_cameras.get(index)
    if entry is None:
        return False
    cam = entry[0]
    old = (getattr(cam.settings, "address", "") or "").strip()
    if (settings.address or "").strip() != old or not cam.is_alive():
        cam.stop()
        entry[0] = _new_camera_thread(index, settings)
        return True
    cam.update_settings(settings)
    return True


def open_camera(index):
    return SharedCamera(index)


# ================================================================ station
class _Worker(QObject):
    done = Signal(int, str, object)          # sequence, job id, result
    failed = Signal(int, str, str)
    finished = Signal(int)
    warmed = Signal(float)

    def warm_up(self):
        """Initialise the slow engines before the first part arrives. The text
        engine took six seconds to start on the first inspection, and over a
        minute when an OCR package tried to fetch models with no network: the
        first part on the line was the one that timed out.

        On its own daemon thread, not the inspection thread: run there, closing
        the application during warm-up left the inspection thread busy past the
        shutdown wait, and Qt aborted the process when it was destroyed still
        running."""
        import threading

        def work():
            t0 = time.perf_counter()
            try:
                # loads the reader's networks and runs one line through
                # them, so the first inspection does not pay for it
                from vision_station.tools import ocr_engine
                ocr_engine.status()
            except Exception:
                pass
            try:
                self.warmed.emit(time.perf_counter() - t0)
            except RuntimeError:
                pass                             # the application closed first
        threading.Thread(target=work, name="engine-warm-up", daemon=True).start()

    workers = 1                                # set by the station

    def _one(self, seq, job):
        pid, project, image, tools = job[:4]
        code = job[4] if len(job) > 4 else None    # the barcode scanned for this part
        try:
            res = project.inspect(image, source="quickbuild", keep_images=True, tools=tools,
                                  barcode=code)
            self.done.emit(seq, pid, res)
        except Exception as e:
            self.failed.emit(seq, pid, "%s: %s" % (type(e).__name__, e))

    def run(self, seq, jobs):
        """jobs: list of (pid, project, image, tools). Runs in the worker thread;
        with more than one worker, jobs run side by side on a thread pool --
        OpenCV and the OCR and inference engines release Python's lock while
        they work, so on a multi-core PC the jobs of one cycle overlap."""
        n = min(self.workers, len(jobs))
        if n > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n) as ex:
                list(ex.map(lambda j: self._one(seq, j), jobs))
        else:
            for job in jobs:
                self._one(seq, job)
        self.finished.emit(seq)


class _PreviewWorker(QObject):
    """Runs a job on its own teach picture, off the window thread: what each
    check gives there, while it is being set up."""
    done = Signal(str, int, object)

    def __init__(self):
        super().__init__()
        self.latest = {}                     # pid -> newest request number

    def run(self, pid, seq, project, snapshot):
        if self.latest.get(pid, seq) != seq:
            return                           # a newer edit is waiting: skip this one
        res = None
        try:
            if project.reference is not None:
                res = project.inspect(project.reference, source="teach", keep_images=False,
                                      tools=project.build_tools(snapshot))
        except Exception as e:               # a half-set-up check must not stop the editor
            print("[teach preview] %s" % e)
        self.done.emit(pid, seq, res)


class Station(QObject):
    """The application being run: its jobs, their image sources, the worker
    thread and the results queues.

    Inspection runs on a worker QObject moved to a QThread, never on the window
    thread; windows only ever receive results through queued signals. Each run
    carries a sequence number so a result that arrives after a newer run was
    started is ignored rather than displayed over it."""

    jobResult = Signal(str, object)          # pid, result
    runFinished = Signal()
    runError = Signal(str, str)
    _start = Signal(int, object)
    _warm = Signal()
    warmedUp = Signal(float)
    # barcode-gated runs: an inspection asked for with no valid code waiting
    barcodeNeeded = Signal()
    # every check run on its job's teach picture, while the job is edited
    previewResult = Signal(str, object)      # pid, result
    _preview = Signal(str, int, object, object)

    def __init__(self, root):
        super().__init__()
        self.store = ProjectStore(root)
        self.root = root
        self.app_id = None
        self.jobs = []                       # [{"pid", "name"}]
        self.projects = {}
        self.sources = {}
        self.last = {}                       # pid -> last result
        self.history = collections.deque(maxlen=QUEUE_LEN)
        self.failures = collections.deque(maxlen=QUEUE_LEN)
        self.counts = {"accept": 0, "reject": 0}
        self.acquisition_errors = 0
        # The results log keeps no pictures of its own: it records where
        # Settings > Results saved each one -- one folder, not two
        self.archive_mode = "none"
        # Settings > Results: each part's picture, annotated and/or original,
        # saved to a folder of the user's choosing on a background thread
        from vision_station.core.result_images import ResultImageWriter
        self.result_images = ResultImageWriter()
        # Settings > Barcode: when on, a run from the Run view or the line
        # needs a valid scanned code, and its pictures go in that code's folder
        from vision_station.core import barcode as barcodes
        self.barcode = barcodes.Gate()
        self.barcode_root = os.path.join(paths.APP_DIR, "results")
        self._codes = {}                     # run sequence -> its barcode
        # every saved picture, searchable as you type (Find picture, Ctrl+F);
        # each picture written is added the moment it is on disk
        from vision_station.core.picture_index import PictureIndex
        self.pictures = PictureIndex()
        self.result_images.on_written.append(self.pictures.add)
        # the external light (Settings > Light): on while a run takes its
        # camera picture, or always; handed back as found when this closes
        from vision_station.hardware.light import Light
        self.light = Light()
        self.parallel = False
        self.operator = ""
        self.logs = {}
        self.running = False
        self.continuous = False
        self._seq = 0
        self._thread = QThread()
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._start.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._warm.connect(self._worker.warm_up)
        self._worker.warmed.connect(self.warmedUp)
        self._thread.start()
        self._pthread = QThread()
        self._pworker = _PreviewWorker()
        self._pworker.moveToThread(self._pthread)
        self._preview.connect(self._pworker.run)
        self._pworker.done.connect(self._on_preview)
        self._pseq = 0
        self._pthread.start()

    def preview(self, pid):
        """Run job pid on its teach picture, in the background; the result
        comes as previewResult. Never counted, logged or saved."""
        p = self.projects.get(pid)
        if p is None or p.reference is None:
            return False
        self._pseq += 1
        self._pworker.latest[pid] = self._pseq
        self._preview.emit(pid, self._pseq, p, json.loads(json.dumps(p.spec)))
        return True

    def _on_preview(self, pid, seq, res):
        if res is not None and self._pworker.latest.get(pid) == seq:
            self.previewResult.emit(pid, res)

    def wait_preview(self, timeout_s=30.0):
        """For tests: until the newest preview has come back."""
        got = {}
        self.previewResult.connect(lambda pid, r: got.setdefault("r", r))
        t0 = time.time()
        while "r" not in got and time.time() - t0 < timeout_s:
            QApplication.processEvents()
            time.sleep(0.005)
        return got.get("r")

    # -------------------------------------------------------- application
    def applications(self):
        """Openable applications: every job folder, and every plain model."""
        seen, out = set(), []
        for x in self.store.list():
            key = x.get("job") or x["id"]
            if key in seen:
                continue
            seen.add(key)
            out.append({"id": key, "name": x.get("job_name") or x.get("name", key)})
        return out

    def open(self, app_id):
        self.stop()
        for src in self.sources.values():
            src.stop_camera()                # a camera must not outlive its job
        self.app_id = app_id
        entries = [x for x in self.store.list() if x.get("job") == app_id]
        if not entries:
            entries = [x for x in self.store.list() if x["id"] == app_id]
        self.jobs = [{"pid": x["id"], "name": x.get("name", x["id"])} for x in entries]
        self.projects = {j["pid"]: self.store.reload(j["pid"]) for j in self.jobs}
        self._open_logs()
        # each job's taught image source, as saved with it
        self.sources = {pid: ImageSource.from_spec(p) for pid, p in self.projects.items()}
        for src in self.sources.values():
            src.camera_factory = open_camera
            src.base = self.app_folder()
        self.posted = self._load_posted()
        self.last = {}
        self.history.clear()
        self.failures.clear()
        self.counts = {"accept": 0, "reject": 0}
        self._warm.emit()                    # in the worker thread, not this one
        from vision_station.core import advanced
        advanced.warm(list(self.projects.values()))   # the jobs' own AI models

    # -- results log -----------------------------------------------------
    def _open_logs(self):
        """Every inspection logged to its job's results database, with images of
        rejected parts archived -- the queue on screen holds only the last 32
        runs and is gone when the application closes. The writer runs on its
        own thread and never blocks an inspection."""
        from vision_station.core.results import ResultStore
        self._close_logs()
        self.logs = {}
        for pid, p in self.projects.items():
            try:
                self.logs[pid] = ResultStore(p.folder, pid, archive=self.archive_mode)
            except Exception as e:
                self.log_error = str(e)          # a read-only folder must not stop inspection

    def _close_logs(self):
        for lg in getattr(self, "logs", {}).values():
            try:
                lg.flush()
                lg.close()
            except Exception:
                pass
        self.logs = {}

    def picture_folders(self):
        """The folders result pictures are saved in: Settings > Results, and
        the barcode folders' root when that is somewhere else."""
        from vision_station.core import barcode as barcodes
        base = dict(self.result_images.cfg)
        root = barcodes.results_root(self.barcode.cfg, base)
        return [root] if root and os.path.isdir(root) else []

    def set_parallel(self, on):
        """Run a cycle's jobs side by side, one thread per job, never more than
        the PC has cores. Off by default: it only helps on a multi-core PC, and
        bench_parallel.py measures whether it does on this one."""
        import os
        self._worker.workers = max(1, os.cpu_count() or 1) if on else 1
        self.parallel = bool(on)

    def app_folder(self):
        return os.path.join(self.root, self.app_id) if self.app_id else None

    def is_job_app(self):
        from vision_station.core import job as jobmod
        return bool(self.app_id) and jobmod.is_job(self.app_folder())

    def _load_posted(self):
        """Posted items live with the application: in job.json for a job, in
        project.json for a single model."""
        from vision_station.core import job as jobmod
        if self.is_job_app():
            return jobmod.Job(self.app_folder()).posted
        p = self.projects.get(self.app_id)
        return list((p.spec if p else {}).get("posted", []))

    def set_posted(self, items):
        from vision_station.core import job as jobmod
        self.posted = list(items)
        if self.is_job_app():
            jobmod.Job(self.app_folder()).set_posted(self.posted)
        else:
            p = self.projects[self.app_id]
            p.spec["posted"] = self.posted
            p.save(p.spec)

    def reload_job(self, pid):
        p = self.store.reload(pid)
        self.projects[pid] = p
        if pid in self.sources:
            self.sources[pid].project = p
        if getattr(p, "advanced", None) is not None:
            from vision_station.core import advanced
            advanced.warm([p])
        return p

    # -------------------------------------------------------- running
    def run_once(self, pids=None, gate=True):
        """gate: the barcode rule applies (the Run view, the line). A job
        editor's run, while teaching, is never held for a barcode."""
        if self.running:
            return False
        # a job waiting for its advanced training trains first (the window
        # sets before_run; core/advanced.py)
        before = getattr(self, "before_run", None)
        if before is not None and not before(pids or [j["pid"] for j in self.jobs]):
            return False
        code = None
        if gate and self.barcode.enabled:
            code = self.barcode.take()
            if code is None:
                self.barcodeNeeded.emit()
                return False
        pids = pids or [j["pid"] for j in self.jobs]
        work = []
        # The light on before the camera's pictures are taken and settled
        # for Settings > Light's "on before" time, then kept on for its "on
        # after" time. The camera's sensor exposes its rows one after
        # another: the frame being read out when the light switches is lit
        # in one part and dark in the rest. The grab waits for frames
        # exposed after it starts; the settle time is on top of that.
        cam = any(self.sources[pid].mode == "camera" for pid in pids if pid in self.sources)
        token = None
        if cam:
            self._light_runs = getattr(self, "_light_runs", 0) + 1
            token = "run %d" % self._light_runs       # its own: a late release
            self.light.hold(token)                    # never ends the next run's
            self.light.settle()
        try:
            work = self._grab_all(pids, code)
        finally:
            if token is not None:
                after = int(self.light.cfg.get("off_after_ms", 0) or 0)
                if after > 0:
                    QTimer.singleShot(after, lambda t=token: self.light.release(t))
                else:
                    self.light.release(token)
        if work is None:
            return False
        self._seq += 1
        self._codes = {self._seq: code}
        self.running = True
        self._start.emit(self._seq, work)
        return True

    def _grab_all(self, pids, code):
        """(pid, project, image, tools) for every job that delivered a
        picture; None when none did."""
        work = []
        for pid in pids:
            p = self.projects.get(pid)
            if p is None:
                continue
            if self.sources[pid].mode == "camera":
                # the job's own light level -- the one its teach picture was
                # taken at -- before its picture; the grab waits for frames
                # exposed after the change
                self.light.use_job_level(pid, p.spec.get("light_level"))
            img = self.sources[pid].grab()
            if img is None:
                self.acquisition_errors += 1
                self.runError.emit(pid, "no image: " + self.sources[pid].last_error)
                continue
            # Fresh tools from a snapshot of the recipe as it is now: edits made
            # in the editors take effect on this run, and the worker never
            # reads a dictionary the window is in the middle of changing.
            snapshot = json.loads(json.dumps(p.spec))
            work.append((pid, p, img, p.build_tools(snapshot), code))
        if not work:
            if code is not None:
                self.barcode.pending = code      # not used: still this part's code
            if self.continuous:              # keep trying: the camera may recover
                QTimer.singleShot(200, lambda: self.continuous and
                                  self.run_once(getattr(self, "_cont_pids", None),
                                                getattr(self, "_cont_gate", True)))
            return None
        return work

    def start_continuous(self, pids=None, gate=True):
        self.continuous = True
        self._cont_pids = pids
        self._cont_gate = gate
        # one part after another: the light stays on until Stop rather than
        # flashing for every part
        if any(src.mode == "camera" for pid, src in self.sources.items()
               if pids is None or pid in pids):
            self.light.hold("continuous")
        if not self.running:
            self.run_once(pids, gate)

    def stop(self):
        self.continuous = False
        self.light.release("continuous")

    def wait(self, timeout_s=30.0):
        """Block, pumping events, until the current run finishes. For scripts
        and tests; the windows never call it."""
        t0 = time.time()
        while self.running and time.time() - t0 < timeout_s:
            QApplication.processEvents()
            time.sleep(0.005)
        QApplication.processEvents()

    def _on_done(self, seq, pid, res):
        if seq != self._seq:
            return                           # a stale result from an older run
        code = self._codes.get(seq)
        if code:
            res["barcode"] = code
        self.last[pid] = res
        self._keep_light_adjusted(pid, res)
        saved = None
        try:
            name = next((j["name"] for j in self.jobs if j["pid"] == pid), pid)
            cfg = None
            if code:
                # every picture of this part, in today's folder, in the
                # folder named after its code -- inside the one folder of
                # Settings > Results; none chosen, none kept
                from vision_station.core import barcode as barcodes
                from vision_station.core.result_images import folder_for
                base = dict(self.result_images.cfg)
                root = barcodes.results_root(self.barcode.cfg, base)
                cfg = barcodes.result_cfg(base, root, code) if root else {}
                self.barcode.last = (code, res["verdict"], folder_for(cfg) if root else "")
            saved = self.result_images.submit(res.get("input_image"), res, name, cfg=cfg)
        except Exception as e:                    # saving must never stop the line
            print("[results] %s" % e)
        lg = self.logs.get(pid)
        if lg is not None:
            # the run recorded with where its picture went: no second copy
            lg.log(res, None, self.operator, image_path=saved or "")
        accept = res["verdict"] == "PASS"
        self.counts["accept" if accept else "reject"] += 1
        entry = self._queue_entry(pid, res)
        self.history.append(entry)
        if not accept:
            self.failures.append(entry)
        self.jobResult.emit(pid, res)

    def _keep_light_adjusted(self, pid, res):
        """Settings > Light > Keep adjusted: this part's picture sets the
        light for the next, aiming at the job's teach picture. A job with its
        own light level (its teach picture taken with the best light) keeps
        its own correction; jobs without one share the Brightness setting,
        and only the first of them corrects it -- one light, one measure per
        cycle."""
        lt = self.light
        if not (lt.enabled and lt.cfg.get("auto")) or res.get("input_image") is None:
            return
        p = self.projects.get(pid)
        if p is None or self.sources.get(pid) is None or self.sources[pid].mode != "camera":
            return
        taught = p.spec.get("light_level")
        own = taught is not None or pid in (lt.cfg.get("job_levels") or {})
        if not own:
            first = next((j["pid"] for j in self.jobs
                          if self.sources.get(j["pid"]) is not None
                          and self.sources[j["pid"]].mode == "camera"
                          and self.projects.get(j["pid"]) is not None
                          and self.projects[j["pid"]].spec.get("light_level") is None), None)
            if pid != first:
                return
        try:
            lt.auto_step(res["input_image"], getattr(p, "reference", None),
                         pid=pid if own else None, taught=taught)
        except Exception as e:                    # never stops the line
            print("[light] not adjusted: %s" % e)

    def _on_failed(self, seq, pid, msg):
        if seq == self._seq:
            self.runError.emit(pid, msg)

    def _on_finished(self, seq):
        if seq != self._seq:
            return
        self.running = False
        self.runFinished.emit()
        if self.continuous:
            # Checked again when the timer fires: Stop pressed in between must
            # cancel a next run that was already scheduled. Without the check
            # one extra inspection slipped through after every Stop.
            QTimer.singleShot(0, lambda: self.continuous and
                              self.run_once(getattr(self, "_cont_pids", None),
                                            getattr(self, "_cont_gate", True)))

    def _queue_entry(self, pid, res):
        """Results queue entries keep their image JPEG-encoded: thirty-two
        full-resolution frames of a 5-megapixel camera would hold half a
        gigabyte; encoded they hold about fifteen megabytes."""
        img = res.get("input_image")
        jpg, size = None, None
        if img is not None:
            # A preview, not an archive: reduced to the queue display's size
            # first. Encoding the whole 20-megapixel frame took 220 ms of the
            # window's thread every cycle, and decoding it again 100 ms more.
            # (Full pictures are saved by the result-image writer.)
            size = (int(img.shape[1]), int(img.shape[0]))
            k = min(1.0, QUEUE_PREVIEW_PX / float(max(size)))
            step = max(1, int(0.5 / k)) if k < 1.0 else 1     # thinned first: area
            pre = img[::step, ::step] if step > 1 else img     # averaging of 20 MP is slow
            small = img if k >= 1.0 else cv2.resize(
                pre, (max(1, int(size[0] * k)), max(1, int(size[1] * k))),
                interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])
            jpg = buf.tobytes() if ok else None
        posted = {}
        byid = {t["tool_id"]: t for t in res["tools"]}
        for item in getattr(self, "posted", []):
            owner, _, ref = item.rpartition("/")     # "<job id>/<tool>.<value>"
            if owner != pid:
                continue
            tid, _, key = ref.partition(".")
            r = byid.get(tid)
            if r is not None:
                v = r["score"] if key == "score" else (r.get("values") or {}).get(key)
                posted[item] = v
        return {"posted": posted,
                "time": time.strftime("%H:%M:%S"), "job": pid, "verdict": res["verdict"],
                "failed": [t["tool_id"] for t in res["tools"] if t["status"] != "pass"],
                "ms": res.get("total_ms", 0.0), "jpg": jpg, "size": size,
                "tools": res["tools"], "barcode": res.get("barcode", "")}

    def shutdown(self):
        self.stop()
        self.wait(30)
        self.light.shutdown()                # the controller back on its own
        self._close_logs()
        for src in self.sources.values():
            src.stop_camera()
        self._pthread.quit()
        self._pthread.wait()
        self._thread.quit()
        # Wait for the inspection thread to finish: destroying a QThread that is
        # still running aborts the process. It only runs inspections now, which
        # end; the old 3 s limit was shorter than the engine warm-up it used to run.
        self._thread.wait()


# ================================================================ records
def _hash_password(password, salt=None):
    import hashlib
    import secrets
    salt = salt or secrets.token_hex(8)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("ascii"), 120000)
    return "%s$%s" % (salt, digest.hex())


def _check_password(password, stored):
    import hmac
    try:
        salt, _digest = stored.split("$", 1)
    except ValueError:
        return False
    return hmac.compare_digest(_hash_password(password or "", salt), stored)


def spaces_for(res):
    """The fixture coordinate spaces a result offers: each Locate tool's
    transform, by tool id."""
    out = {}
    for t in (res or {}).get("tools", []):
        M = (t.get("values") or {}).get("matrix")
        if t.get("type") == "align" and M:
            out[t["tool_id"]] = M
    return out


GEOMETRY_CHANGING = {"affine_transform", "polar_unwrap"}


def apply_drawn_region(t, kind, geom):
    """A shape drawn on the image becomes a tool's region: a rectangle or an
    ellipse by its box and its turn (the waiting area can be turned); a
    polygon by its points."""
    if t is None:
        return
    p = t.setdefault("params", {})
    if kind == "polygon":
        pts = [[float(a), float(b)] for a, b in geom["points"]]
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        p.update(shape="polygon", points=pts,
                 roi=[int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))])
    else:
        p.update(shape="ellipse" if kind == "ellipse" else "rect",
                 roi=[int(v) for v in geom["roi"]], angle=float(geom.get("angle", 0.0) or 0.0))
        p.pop("points", None)
FULL_FRAME_RECORDS = ("Reference", "Current.InputImage", "LastRun.InputImage",
                      "LastRun.OutputImage")
TEACH = "Reference"          # the teach picture: the job's reference image
QUEUE_PREVIEW_PX = 1600      # long side of a results-queue preview
RUN_VIEW_ALL_TAGS = 12       # more checks than this: the Run view labels only failures


def same_picture(a, b):
    """Whether two frames are the same picture -- the same array, or the
    same pixels on a sparse grid (a copy of the reference read back)."""
    if a is None or b is None:
        return False
    if a is b:
        return True
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a[::29, ::31], b[::29, ::31]))


def fixture_moved(res):
    """Whether a run's fixture moved the regions: then they are not where
    they were drawn on that run's picture."""
    for M in spaces_for(res).values():
        m = np.asarray(M, np.float64)
        if m.shape[0] >= 2 and (abs(m[0, 2]) > 0.5 or abs(m[1, 2]) > 0.5
                                or abs(m[0, 1]) > 1e-3 or abs(m[0, 0] - 1.0) > 1e-3):
            return True
    return False


def regions_fit(record, image, reference, res):
    """Whether the checks' regions, which are in teach-picture coordinates,
    sit on the right place of this record's picture. Shown on any other
    picture they marked the wrong place -- a board lying 60 px further
    along had every box 60 px off its part -- while the true result
    outlines were hidden beneath them."""
    if record == TEACH:
        return True
    if record not in ("Current.InputImage", "LastRun.InputImage"):
        return record in FULL_FRAME_RECORDS
    if reference is None:
        return True                      # nothing taught yet: the picture is the frame
    if not same_picture(image, reference):
        return False
    return record == "Current.InputImage" or not fixture_moved(res)


def move_regions(tools, M):
    """Every region moved by a 2x3 (or 3x3) transform: a box's centre moved
    and its angle turned, polygon points and a counting sample moved."""
    m = np.asarray(M, np.float64)[:2]
    rot = math.degrees(math.atan2(m[1, 0], m[0, 0]))

    def pt(x, y):
        return (m[0, 0] * x + m[0, 1] * y + m[0, 2], m[1, 0] * x + m[1, 1] * y + m[1, 2])

    def box(r, turn=True):
        x, y, w, h = r
        cx, cy = pt(x + w / 2.0, y + h / 2.0)
        return [int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(w), int(h)]

    for t in tools:
        prm = t.get("params") or {}
        if prm.get("roi"):
            prm["roi"] = box(prm["roi"])
            if prm.get("shape") != "polygon" and abs(rot) > 0.05:
                prm["angle"] = round(float(prm.get("angle", 0.0) or 0.0) + rot, 3)
        if prm.get("points"):
            prm["points"] = [[round(a, 2) for a in pt(x, y)] for x, y in prm["points"]]
        if prm.get("template_roi"):
            prm["template_roi"] = box(prm["template_roi"])


# The one setting that decides how strict a check is, by check type: what a
# group of checks of one kind shares and is adjusted by together.
THRESHOLD_KEYS = {
    "presence": ("min_score", "Accept threshold"),
    "count_template": ("min_score", "Accept threshold"),
    "align": ("min_score", "Accept threshold"),
    "ocv": ("min_confidence", "Reading confidence"),
    "ocr": ("min_confidence", "Reading confidence"),
}


def group_name(t):
    """A check's group: the one Find parts gave it, else its name without
    the number ('inductor' for inductor_07) -- so checks added before groups
    existed are grouped too."""
    import re as _re
    prm = t.get("params") or {}
    return prm.get("group") or _re.sub(r"_?\d+$", "", t.get("id", "")) or t.get("id", "")


def group_members(t, tools):
    """Every check of the same kind and group as t (t included)."""
    if t is None:
        return []
    g, ty = group_name(t), t.get("type")
    return [x for x in tools if x.get("type") == ty and group_name(x) == g]


class GroupBar(QFrame):
    """Above a check's properties when it belongs to a group of the same
    kind -- the twelve inductors Find parts added -- or when several boxes
    are selected: one threshold for all of them, select them all on the
    picture, turn them all on or off. A kind of part that fails on every
    board is put right once, not twelve times."""

    def __init__(self, editor):
        super().__init__()
        from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox
        self.ed = editor
        self.members = []
        self.setObjectName("groupBar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 8, 12, 8)
        v.setSpacing(6)
        self.title = QLabel("")
        self.title.setObjectName("groupBarTitle")
        self.title.setWordWrap(True)
        v.addWidget(self.title)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.thrLabel = QLabel("Threshold for all")
        self.thr = QDoubleSpinBox()
        self.thr.setRange(0.0, 1.0)
        self.thr.setSingleStep(0.05)
        self.thr.setDecimals(2)
        self.thr.setKeyboardTracking(False)
        self.thr.valueChanged.connect(self._apply_threshold)
        row.addWidget(self.thrLabel)
        row.addStretch(1)
        row.addWidget(self.thr)
        v.addLayout(row)
        # what the group scored on the last part: how far the threshold may go
        self.scores = QLabel("")
        self.scores.setObjectName("rowHelp")
        self.scores.setWordWrap(True)
        v.addWidget(self.scores)
        row = QHBoxLayout()
        self.selectBtn = QPushButton("Select all on the picture")
        self.selectBtn.setObjectName("linkButton")
        self.selectBtn.clicked.connect(self._select_all)
        self.teachBtn = QPushButton("Expect what each reads now")
        self.teachBtn.setToolTip("Each text check expects the text it reads on the teach "
                                 "picture - for Read text checks made before they expected "
                                 "anything, which passed a scribbled or wrong marking")
        self.teachBtn.clicked.connect(self._teach_texts)
        v.addWidget(self.teachBtn)
        self.onBox = QCheckBox("Run them")
        self.onBox.setToolTip("Turn every check of the group on or off")
        self.onBox.clicked.connect(self._set_enabled)
        row.addWidget(self.selectBtn)
        row.addStretch(1)
        row.addWidget(self.onBox)
        v.addLayout(row)
        self.hide()

    def show_for(self, members, title):
        self.members = list(members)
        if len(self.members) < 2:
            self.hide()
            return
        keyed = [m for m in self.members if m.get("type") in THRESHOLD_KEYS]
        self.title.setText(title)
        self.thr.blockSignals(True)
        if keyed:
            key, label = THRESHOLD_KEYS[keyed[0]["type"]]
            vals = sorted({round(float(m["params"].get(key, 0.65 if key == "min_score" else 0.5)),
                                 3) for m in keyed})
            self.thr.setValue(vals[0])
            self.thrLabel.setText("%s for all %d%s" % (label, len(keyed),
                                                     "" if len(vals) == 1 else
                                                     "  (now %s - %s)" % (vals[0], vals[-1])))
        self.thr.blockSignals(False)
        for w in (self.thr, self.thrLabel):
            w.setVisible(bool(keyed))
        self.teachBtn.setVisible(any(m.get("type") in ("ocv", "ocr") for m in self.members))
        res = self.ed._run_res()
        got = {t["tool_id"]: t for t in (res or {}).get("tools", [])}
        sc = [(got[m["id"]]["score"], m["id"], got[m["id"]]["status"]) for m in keyed
              if m["id"] in got]
        if sc:
            lo = min(sc)
            nfail = sum(1 for x in sc if x[2] != "pass")
            self.scores.setText("Last run: lowest score %.2f (%s)%s" % (
                lo[0], lo[1], "  -  %d failed" % nfail if nfail else ""))
        self.scores.setVisible(bool(sc))
        on = [m["params"].get("enabled", True) is not False for m in self.members]
        self.onBox.setTristate(False)
        self.onBox.setCheckState(Qt.Checked if all(on) else
                                 Qt.Unchecked if not any(on) else Qt.PartiallyChecked)
        self.show()

    def _apply_threshold(self, v):
        n = 0
        for m in self.members:
            k = THRESHOLD_KEYS.get(m.get("type"))
            if k:
                m.setdefault("params", {})[k[0]] = round(float(v), 3)
                n += 1
        if n:
            self.ed._group_edited("%s %.2f on %d checks" % (
                THRESHOLD_KEYS[next(m for m in self.members
                                    if m.get("type") in THRESHOLD_KEYS)["type"]][1], v, n))

    def _teach_texts(self):
        """Every text check of the group expects what it reads on the teach
        picture, as the live results read it."""
        pv = self.ed._preview_res() or {}
        read = {t["tool_id"]: (t.get("values") or {}) for t in pv.get("tools", [])}
        n, missing = 0, []
        for m in self.members:
            if m.get("type") not in ("ocv", "ocr"):
                continue
            vals = read.get(m["id"])
            # Verify text: the text it read (the matching line when it passed);
            # Read text: every line it read
            text = (vals.get("text") if m["type"] == "ocv" else
                    " ".join(vals.get("lines") or [])) if vals else ""
            text = (text or "").strip()
            if not text:
                missing.append(m["id"])
                continue
            prm = m.setdefault("params", {})
            if m["type"] == "ocv":
                prm["expect_text"] = re.sub(r"\s+", "", text) if prm.get("strip_spaces", True) \
                    else text
                prm["expect_auto"] = prm["expect_text"]
            else:
                prm["expect_contains"] = text
            n += 1
        self.ed._group_edited("%d text checks now expect what they read on the teach picture%s"
                              % (n, ("; nothing read in %s" % ", ".join(missing[:4]))
                                 if missing else ""))

    def _set_enabled(self):
        on = self.onBox.checkState() != Qt.Unchecked
        self.onBox.setCheckState(Qt.Checked if on else Qt.Unchecked)
        for m in self.members:
            if on:
                m.setdefault("params", {}).pop("enabled", None)
            else:
                m.setdefault("params", {})["enabled"] = False
        self.ed._group_edited("%d checks turned %s" % (len(self.members), "on" if on else "off"))

    def _select_all(self):
        ids = [m["id"] for m in self.members]
        self.ed.display.record.setCurrentText(TEACH)
        self.ed.display.canvas.select_ids([i for i in ids if i in self.ed.display.canvas.regions])
        self.ed.display.canvas.zoom_to_selection()


def _examples_signature(folder):
    """What the OK / NG examples in a project are now: their files and times."""
    import glob as _glob
    out = []
    for f in _glob.glob(os.path.join(folder, "training", "*", "*", "*")):
        try:
            out.append((f, os.path.getmtime(f)))
        except OSError:
            pass
    return tuple(sorted(out))


def teach_check(project, t, name=None):
    """What a check learns from the teach picture when its region is set:
    (text, message). A Verify text check reads its box there and expects
    that text -- it used to expect nothing and so passed whatever was
    printed. Text typed in by hand is never replaced; text taught is
    remembered as taught (expect_auto) so a moved box is read again."""
    if t is None or t.get("type") != "ocv":
        return None, ""
    name = name or t.get("id", "")
    prm = t.setdefault("params", {})
    if prm.get("expect_regex") or (prm.get("expect_text") and
                                   prm.get("expect_text") != prm.get("expect_auto")):
        return None, ""
    img = project.reference if project is not None else None
    if img is None or not prm.get("roi"):
        return None, ""
    text, others = taught_text(img, t)
    if text is None:
        return None, others
    if not text:
        prm["expect_text"] = ""
        prm.pop("expect_auto", None)
        return "", ("No text read in the box of %s - fit the box round the marking, or "
                    "type the text under Settings" % name)
    prm["expect_text"] = text
    prm["expect_auto"] = text
    return text, "%s taught: it expects '%s'%s (change it under Settings)" % (
        name, text, "; other text in the box is ignored" if others else "")


def taught_text(img, t):
    """(text, others) a Verify text check should expect from its box on
    the teach picture: the clearest whole line in it. A box usually holds
    more than its marking -- the edge of a neighbouring word, a via read as
    "O" -- and expecting everything it read failed the check on its own
    teach picture as soon as the part moved a fraction of a pixel and a
    fragment read differently. Lines cut by the box's edge, single
    characters and doubtful reads are passed over while a clear line
    exists. text is None when the reader is not available (others is then
    the reason)."""
    import re as _re
    from vision_station.core.engine import Context
    from vision_station.tools import make_tool
    from vision_station.tools.text import read_region
    prm = dict(t.get("params") or {}, expect_text="", expect_regex="")
    tool = make_tool({"id": t["id"], "type": "ocv", "params": prm})
    # the wait cursor only from the window's thread (the glass front end
    # teaches from its server's threads)
    app = QApplication.instance()
    gui = app is not None and QThread.currentThread() is app.thread()
    if gui:
        QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        lines, _unsure, _o, _e = read_region(tool, img, Context(reference=img))
        crop, _o2 = tool.crop(img, Context(reference=img))
    except Exception as e:                   # no reader on this PC: say so, change nothing
        return None, "Could not read the text to teach it: %s" % e
    finally:
        if gui:
            QApplication.restoreOverrideCursor()
    strip = prm.get("strip_spaces", True)
    H, W = crop.shape[:2] if crop is not None else (0, 0)

    def norm(x):
        return _re.sub(r"\s+", "", x) if strip else x.strip()

    def cut(ln):
        b = ln.get("box") or [0, 0, 0, 0]
        return b[0] <= 1 or b[1] <= 1 or b[0] + b[2] >= W - 1 or b[1] + b[3] >= H - 1

    cands = [ln for ln in lines if len(norm(ln["text"])) >= 2] or \
        [ln for ln in lines if norm(ln["text"])]
    if not cands:
        return "", False
    best = max(cands, key=lambda ln: (not cut(ln), ln["confidence"] >= 0.8,
                                      len(norm(ln["text"])), ln["confidence"]))
    return norm(best["text"]), len(lines) > 1


def records_for(res, tool_id=None, current=None, reads=None, reference=None,
                teach=None):
    """The record list a display offers, VisionPro-style. Current.InputImage is
    the image that will be inspected next; the LastRun records are what the
    last run saw and produced.

    reads: the image the tool reads when it is not the camera's -- an image
    tool's output. Its records are then that image: a region is drawn and
    edited in the coordinates of the image the tool inspects, which for an
    unwrapped ring or a resampled region are not the camera's."""
    out = collections.OrderedDict()
    own = (res or {}).get("images", {}).get(reads) if reads and reads != "default" else None
    if own is not None:
        current = own
    elif reference is not None:
        # the picture the checks are drawn and taught on, always first --
        # with what every check gives on it, when that has been worked out
        out[TEACH] = (reference, list((teach or {}).get("tools", [])))
    if current is not None:
        out["Current.InputImage"] = (current, [])
    if res:
        tools = res["tools"] if tool_id is None else \
            [t for t in res["tools"] if t["tool_id"] == tool_id]
        if own is not None:
            out["LastRun.InputImage"] = (own, tools)
        elif res.get("input_image") is not None:
            out["LastRun.InputImage"] = (res["input_image"], tools)
        if tool_id is not None and tool_id in (res.get("images") or {}):
            out["LastRun.OutputImage"] = (res["images"][tool_id], [])
        if tool_id is not None:
            # the tool's diagnostics: its region, projection, histogram, mask
            for name, img in sorted(((res.get("records") or {}).get(tool_id) or {}).items()):
                out[name] = (img, [])
    return out


# ================================================================ tool edit control
# The light's level while the camera's white is balanced: lit, not burnt out.
BALANCE_LEVEL = 96
from vision_station.core.advanced import N_PICTURES as ADVANCED_PICTURES  # noqa: E402


class ReferenceSourceDialog(QDialog):
    """New application or job: where the picture of a good part comes from.

    Its own dialog, not a QMessageBox: the message box's buttons were held to
    a fixed width by the button box, and in the dark theme they showed
    "e for a picture" and "nce image from the" -- neither choice readable.
    Here each choice is a full-width button with a line under it saying
    what it does. choice: 'file', 'camera' or None."""

    def __init__(self, parent, title, advanced=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setObjectName("referenceSourceDialog")
        self.choice = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 18)
        lay.setSpacing(10)
        head = QLabel("Where does the picture of a GOOD part come from?")
        f = head.font()
        f.setPointSizeF(f.pointSizeF() * 1.2)
        f.setBold(True)
        head.setFont(f)
        head.setWordWrap(True)
        lay.addWidget(head)
        sub = QLabel("Every part is compared with it.")
        sub.setObjectName("rowHelp")
        lay.addWidget(sub)
        self.advancedNote = None
        if advanced:
            # Settings > General > Use advanced model for training
            note = QLabel(
                "<b>Advanced model training will be used for this job.</b><br>"
                "Take one reference picture as usual, then mark everything to inspect: "
                "parts that must be present (Find parts) and text to verify or read "
                "(Find text). Then choose <b>Run &#9662; Model teaching run</b>: Vision Station "
                "takes %d more pictures of the good part, adds them to every check's good "
                "pictures and trains the job's own AI model on the graphics card - a few "
                "minutes, with a window showing how far it is. A normal Run only "
                "inspects." % ADVANCED_PICTURES)
            note.setObjectName("advancedNote")
            note.setWordWrap(True)
            note.setTextFormat(Qt.RichText)
            # the frame's margins, not a style sheet's padding: a wrapped
            # label's height is worked out without the padding, and its last
            # line was cut off
            box = QFrame()
            box.setObjectName("advancedNoteBox")
            box.setStyleSheet("QFrame#advancedNoteBox { border: 1px solid palette(highlight); "
                              "border-radius: 6px; }")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(12, 10, 12, 10)
            bl.addWidget(note)
            lay.addSpacing(4)
            lay.addWidget(box)
            self.advancedNote = note
        lay.addSpacing(6)
        self.aiBtn = self._option(
            lay, "AI reference image from the camera", "sparkle",
            "The camera takes it. When Vision Station controls the light, the AI first "
            "tries it at several levels and keeps the clearest picture, with the camera's "
            "white balanced under the light. The job then takes its pictures from the "
            "camera, at that light.", "camera", primary=True)
        self.fileBtn = self._option(
            lay, "Browse for a picture file...", "open",
            "A picture of a good part you already have.", "file")
        lay.addSpacing(4)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setMinimumWidth(100)
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        lay.addLayout(row)
        self.setMinimumWidth(520)
        self.aiBtn.setFocus()

    def _option(self, lay, text, icon, help_text, choice, primary=False):
        b = QPushButton(icons.icon(icon), "  " + text)
        b.setMinimumHeight(44)
        b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        b.setAutoDefault(primary)
        b.setDefault(primary)
        if primary:
            b.setObjectName("primaryButton")
        f = b.font()
        f.setPointSizeF(f.pointSizeF() * 1.08)
        b.setFont(f)
        b.clicked.connect(lambda: self._chosen(choice))
        lay.addWidget(b)
        note = QLabel(help_text)
        note.setObjectName("rowHelp")
        note.setWordWrap(True)
        note.setContentsMargins(6, 0, 6, 8)
        lay.addWidget(note)
        return b

    def _chosen(self, choice):
        self.choice = choice
        self.accept()


def write_recipe_keys(path, keys):
    """These keys written into a recipe file on disk, the rest as it is."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        spec.update(keys)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
        return True
    except Exception as e:
        print("[recipe] %s not written: %s" % (", ".join(keys), e))
        return False


def place_barcode_checks(editor):
    """A check just set to compare with the scanned barcode goes to just
    after the fixture, so the barcode is the first thing checked and is
    found there in the list (core/barcode.py). The ids moved."""
    from vision_station.core import barcode as barcodes
    try:
        p = editor.project()
    except Exception:
        return []
    if p is None:
        return []
    moved = barcodes.place_compare_checks(p.spec.get("tools", []))
    if moved:
        editor.statusBar().showMessage(
            "%s moved to just after the fixture: the barcode is checked first"
            % ", ".join(moved), 8000)
    return moved


class ToolEditControl(QMainWindow):
    """One tool, in its own window, as VisionPro's tool edit controls are."""

    def __init__(self, qb, pid, tool_id):
        super().__init__(qb)
        self.qb, self.pid, self.tool_id = qb, pid, tool_id
        _unique_name(self, "toolEdit:%s:%s" % (pid, tool_id))
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1100, 720)
        self._floating = []
        self._electric = QTimer(self)
        self._electric.setSingleShot(True)
        self._electric.setInterval(200)
        self._electric.timeout.connect(self.run)

        bar = _unique_name(QToolBar("Tool"), "toolEditBar")
        bar.setMovable(False)
        bar.setIconSize(EDITOR_ICON)
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(bar)
        # Run is the main window's Run (it runs the job in front). Here: the
        # one setting people toggle while tuning, named for what it does, and
        # the rarer commands as icons on the right.
        self.aElectric = make_action(bar, "Live Update", None, None,
                                     "Re-run the job whenever a setting changes "
                                     "(VisionPro's Electric mode)", check=True, icon='electric')
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        for text, fn, tip, ic in (
                ("Results grid", self.results_grid, "Everything this tool measured, in a table",
                 'grid_results'),
                ("Floating display", self.floating_display,
                 "Open another, movable display of this tool's records", 'floating'),
                (None, None, None, None),
                ("Open...", self.open_tool, "Load this tool's settings from a file", 'open'),
                ("Save...", self.save_tool, "Save this tool's settings to a file", 'save'),
                ("Reset", self.reset_tool, "Restore default settings, keeping the region", 'reset'),
                (None, None, None, None),
                ("Help", self.help, "What this tool does", 'help')):
            if text is None:
                bar.addSeparator()
                continue
            a = make_action(bar, text, fn, None, tip, icon=ic)
            b = bar.widgetForAction(a)
            b.setToolButtonStyle(Qt.ToolButtonIconOnly)
            b.setAccessibleName(text.rstrip("."))

        from vision_station.ui.classic_window import PropertySheet
        self.props = PropertySheet()
        self.props.changed.connect(self._settings_changed)
        self._build_train_tab()
        self._build_graphics_tab()
        self.display = VisionDisplay()
        self.display.recordChanged.connect(lambda _n: self._sync_regions())
        c = self.display.canvas
        c.geometryCommitted.connect(self._geometry)
        c.liveGeometry.connect(self._geometry_live)
        c.angleChanged.connect(self._angle)
        c.polygonCommitted.connect(self._polygon)
        # draw the region afresh with R / E / P or the rail; it has one region,
        # which it can neither duplicate nor delete
        self.display.enable_drawing(True)
        c.allow_duplicate = c.allow_delete = False
        c.shapeDrawn.connect(self._drawn)
        c.lockChanged.connect(self._lock)
        # a region is drawn on the teach picture, wherever the drawing starts
        self._editable = True
        self.display.before_draw = lambda: (
            not self._editable and self.display.record.setCurrentText(TEACH))

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.props)
        split.addWidget(self.display)
        split.setSizes([440, 660])
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)          # the image takes the extra room
        self.setCentralWidget(split)
        self.split = split
        _restore_split(qb, split, "toolEdit/split")

        self.sDot = QLabel("")
        self.sDot.setFixedSize(12, 12)
        self.sText = QLabel("not run")
        self.sTime = QLabel("")
        sb = self.statusBar()
        sb.addWidget(self.sDot)
        sb.addWidget(self.sText, 1)
        sb.addPermanentWidget(self.sTime)
        self.refresh(first=True)


    # ------------------------------------------------------------ model
    def project(self):
        return self.qb.station.projects.get(self.pid)

    def spec(self):
        p = self.project()
        if p is None:
            return None
        return find_tool(p.spec.get("tools", []), self.tool_id)

    def refresh(self, first=False):
        t = self.spec()
        if t is None:
            self.close()
            return
        kind = LABELS.get(t["type"], (t["type"], ""))[0]
        self.setWindowTitle("%s  ·  %s" % (t["id"], kind))
        self.setToolTip("%s (%s) in %s" % (t["id"], kind, self.qb.job_name(self.pid)))
        p = self.project()
        self.props.recipe = p.spec.get("tools", [])
        self.props.set_captures(readable_images(p.spec.get("tools", []), t["id"], ["default"]))
        if self.props.tool is not t:
            self.props.load(t)
        res = self.qb.station.last.get(self.pid)
        tres = next((x for x in (res or {}).get("tools", []) if x["tool_id"] == t["id"]), None)
        self.props.show_result(tres)
        src = self.qb.station.sources.get(self.pid)
        current = src.current if src and src.current is not None else p.reference
        recs = records_for(res, t["id"], current, reads=t["params"].get("capture"),
                           reference=p.reference)
        self.display.set_records(recs, prefer="LastRun.InputImage" if first and res
                                 else None)
        self.display.set_spaces(spaces_for(res))
        self._sync_regions()
        self._status(tres, res)
        self.refresh_train()
        self.refresh_graphics()
        for fd in self._floating:
            fd.set_records(recs)

    def _sync_regions(self):
        """The region is editable on Current.InputImage; on LastRun records it
        is shown with the results, as VisionPro does."""
        t = self.spec()
        if t is None:
            return
        c = self.display.canvas
        # Regions only on full-frame records. A projection plot, a histogram or
        # the cropped region has its own coordinates, and the tool's outline
        # drawn over it was meaningless.
        rec = self.display.current_record()
        img = self.display._records.get(rec, (None, None))[0]
        p = self.project()
        res = self.qb.station.last.get(self.pid)
        full_frame = rec in FULL_FRAME_RECORDS and (
            t["params"].get("capture") not in (None, "", "default")
            or regions_fit(rec, img, p.reference, res))
        if rec == "LastRun.OutputImage" and t["type"] in GEOMETRY_CHANGING:
            full_frame = False       # a resampled or unwrapped image has its own coordinates
        c.sync_regions([t] if full_frame else [], None)
        editable = full_frame and rec in (TEACH, "Current.InputImage")
        self._editable = editable
        if rec in ("Current.InputImage", "LastRun.InputImage") and not full_frame \
                and img is not None:
            self.display.set_note(
                "The box shows where this check found its part on this picture. "
                "It is drawn and moved on the Teach picture.",
                ("Show Teach picture", lambda: self.display.record.setCurrentText(TEACH)))
        else:
            self.display.set_note(None)
        for it in c.regions.values():
            it.setFlag(it.GraphicsItemFlag.ItemIsMovable, editable)
            it.setFlag(it.GraphicsItemFlag.ItemIsSelectable, editable)
        if editable:
            c.select_ids([t["id"]], silent=True)

    def _status(self, tres, res):
        if tres is None:
            self.sDot.setStyleSheet(dot_style(IDLE_C))
            self.sText.setText("not run")
            self.sTime.setText("")
            return
        self.sDot.setStyleSheet(dot_style(status_dot(tres["status"])))
        self.sText.setText(("Accept" if tres["status"] == "pass" else
                            "Reject" if tres["status"] == "fail" else "Error")
                           + ("   " + tres["message"] if tres.get("message") else ""))
        ms = tres.get("ms", 0.0)
        self.sTime.setText("Tool %s   (Total %s)" % ("%.1f ms" % ms if ms < 1000 else fmt_ms(ms),
                                                     fmt_ms(res.get("total_ms", 0.0))))


    # ------------------------------------------------------------ graphics tab
    GRAPHICS = (("region", "Region outline"), ("results", "Results - what the tool found"),
                ("labels", "Labels"), ("search", "Search guides - where it looks"))

    def _build_graphics_tab(self):
        """What this tool draws, as VisionPro's Graphics tab sets it -- applied
        everywhere the tool's results appear, run view and job files included.
        Switching the region and labels off for a large fixture is how a busy
        job's run view stays readable."""
        from PySide6.QtWidgets import QCheckBox
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(10, 10, 10, 10)
        self.gfxBoxes = {}
        for key, text in self.GRAPHICS:
            cb = QCheckBox(text)
            cb.toggled.connect(lambda _on, k=key: self._graphics_changed())
            v.addWidget(cb)
            self.gfxBoxes[key] = cb
        v.addStretch(1)
        self.props.tabs.addTab(w, "Graphics")

    def refresh_graphics(self):
        t = self.spec()
        if t is None or not hasattr(self, "gfxBoxes"):
            return
        opt = t["params"].get("graphics") or {}
        for key, cb in self.gfxBoxes.items():
            cb.blockSignals(True)
            cb.setChecked(bool(opt.get(key, True)))
            cb.blockSignals(False)

    def _graphics_changed(self):
        t = self.spec()
        if t is None:
            return
        opt = {k: cb.isChecked() for k, cb in self.gfxBoxes.items()}
        if all(opt.values()):
            t["params"].pop("graphics", None)        # the default: nothing stored
        else:
            t["params"]["graphics"] = opt
        self._changed()

    # ------------------------------------------------------------ train tab
    def _build_train_tab(self):
        """OK and NG images for this tool. What they are for depends on the
        tool: a trained tool learns from them, a score-threshold tool has its
        threshold set from them, and every tool can be tested against them."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        cols, self.trainLists = training_columns(
            self._train_menu, self.add_training, ("pass", "fail"), None, icon=(110, 82))
        v.addLayout(cols, 1)
        row = QHBoxLayout()
        self.bTrain = QPushButton("Train")
        self.bTrain.clicked.connect(self.train)
        self.bTest = QPushButton("Test")
        self.bTest.setToolTip("Run the job on every image here: the OKs must pass, "
                              "the NGs must be rejected")
        self.bTest.clicked.connect(self.test)
        row.addWidget(self.bTrain)
        row.addWidget(self.bTest)
        self.trainReport = QLabel("")
        self.trainReport.setObjectName("report")
        self.trainReport.setWordWrap(True)
        row.addWidget(self.trainReport, 1)
        v.addLayout(row)
        self.props.tabs.addTab(w, "Train")
        self._marks = {}

    def side(self):
        from vision_station.core import job as jobmod
        return jobmod.Side(self.project().folder, None, {"id": self.pid.split("/")[-1]})

    def refresh_train(self):
        from vision_station.core import job as jobmod
        from vision_station.ui.classic_window import ndarray_to_qpixmap
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QListWidgetItem
        t = self.spec()
        if t is None or not hasattr(self, "trainLists"):
            return
        if t["type"] == "trained_surface":
            self.bTrain.setText("Train")
            self.bTrain.setEnabled(True)
        elif t["type"] in jobmod.TUNABLE:
            self.bTrain.setText("Tune threshold")
            self.bTrain.setEnabled(True)
        else:
            self.bTrain.setText("Train")
            self.bTrain.setEnabled(False)
        side = self.side()
        side.__dict__["_thumbs"] = self.__dict__.setdefault("_thumbs", {})
        for lst in self.trainLists.values():
            lst.clear()
        for path, label in side.training_images(t["id"]):
            th = side.region_thumb(t, path, 110, 82)
            if th is None:
                continue
            if path in self._marks:           # green: judged right; red: wrong
                th = cv2.copyMakeBorder(th[3:-3, 3:-3], 3, 3, 3, 3, cv2.BORDER_CONSTANT,
                                        value=(74, 158, 30) if self._marks[path] else (58, 52, 214))
            it = QListWidgetItem(QIcon(ndarray_to_qpixmap(th)),
                                 os.path.basename(path).split(".")[0][-10:])
            it.setData(Qt.UserRole, path)
            it.setToolTip(path)
            self.trainLists[label].addItem(it)
        self._warn_examples(t)

    def _warn_examples(self, t):
        """An NG example of a Part present check that looks just like the good
        reference here rejects only that very picture: every other good
        board is as like it as the reference is. Said, so nobody expects it
        to reject more."""
        if t.get("type") != "presence" or not t["params"].get("roi"):
            return
        p = self.project()
        if p.reference is None:
            return
        from vision_station.tools import count as C
        x, y, w, h = t["params"]["roi"]
        ang = float(t["params"].get("angle", 0.0) or 0.0)
        tpl = C.upright(p.reference, [x, y, w, h], ang)
        try:
            ex = C.training_examples(p.folder, t["id"], [x, y, w, h], tpl, ang)
        except Exception:
            return
        alike = [e for e in ex if e[0] == "ng" and C._patch_sim(e[2], tpl) >= 0.97]
        if alike and not self.trainReport.text():
            self.trainReport.setText(
                "%d NG example%s look%s like the good reference here: %s only that very "
                "picture. Add pictures of the real fault (the part missing, wrong or "
                "damaged) to reject it on every board." % (
                    len(alike), "" if len(alike) == 1 else "s", "s" if len(alike) == 1 else "",
                    "it rejects" if len(alike) == 1 else "they reject"))
            theme.set_tone(self.trainReport, "warn")

    def _current_image(self):
        src = self.qb.station.sources.get(self.pid)
        if src is not None and src.current is not None:
            return src.current
        return self.project().reference

    def add_training(self, label):
        img = self._current_image()
        if img is None:
            return None
        path = self.side().add_training_image(self.tool_id, img, label)
        self.refresh_train()
        self._examples_changed()
        return path

    def _examples_changed(self):
        """An OK or NG example decides a Part present check now: every window
        showing the job is brought up to date, and with Live Update on the
        check is run again."""
        self.qb.job_changed(self.pid, source=self)
        if self.aElectric.isChecked():
            self._electric.start()

    def _train_menu(self, lst, label, pos):
        item = lst.itemAt(pos)
        if training_menu(self, lst, label, pos, self.side()):
            if item is not None:
                self._marks.pop(item.data(Qt.UserRole), None)
            self.refresh_train()
            self._examples_changed()

    def _saved_first(self):
        """Training reads and writes the job on disk, so unsaved edits are
        saved first -- otherwise training would overwrite them."""
        if self.pid in self.qb.dirty:
            self.qb.save_job(self.pid)

    def train(self):
        from vision_station.core import job as jobmod
        t = self.spec()
        if t is None:
            return None
        self._saved_first()
        side = self.side()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if t["type"] in jobmod.TUNABLE:
                rep = side.tune_threshold(self.tool_id)
                msg = (("%s set to %.3f - between the lowest OK (%.3f) and the highest "
                        "NG (%.3f)." % (rep["param"], rep["new"], rep["lowest_ok"],
                                        rep["highest_ng"])) if rep["separates"] else
                       "Not changed: an NG scored %.3f, level with or above the lowest OK "
                       "(%.3f). Check the labels or the region." % (rep["highest_ng"],
                                                                    rep["lowest_ok"]))
                good = rep["separates"]
            else:
                rep = side.train_tool(self.tool_id)
                msg = "Trained on %d OK (+%d augmented), %d NG. Threshold %.0f." % (
                    rep["real_ok"], rep["augmented"], rep["ng"], rep["threshold"])
                bad = rep.get("unseparated_ng_paths") or []
                if bad:
                    msg += " %d NG scored like a good part and was set aside." % len(bad)
                good = not bad
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.trainReport.setText(str(e))
            theme.set_tone(self.trainReport, "fail")
            return None
        QApplication.restoreOverrideCursor()
        self.qb.reload_job(self.pid)          # the job now holds the model / threshold
        self.trainReport.setText(msg)
        theme.set_tone(self.trainReport, "pass" if good else "warn")
        return rep

    def test(self):
        self._saved_first()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ev = self.side().evaluate_tool(self.tool_id)
        finally:
            QApplication.restoreOverrideCursor()
        self._marks = {r["path"]: r["correct"] for r in ev["rows"]}
        self.refresh_train()
        from vision_station.core import job as jobmod
        text, level = jobmod.evaluation_summary(ev)
        self.trainReport.setText(text)
        theme.set_tone(self.trainReport, {"good": "pass", "warn": "warn",
                                          "bad": "fail"}.get(level, "idle"))
        return ev

    # ------------------------------------------------------------ edits
    def _changed(self):
        self.qb.mark_dirty(self.pid)
        self.qb.job_changed(self.pid, source=self)
        if self.aElectric.isChecked():
            self._electric.start()           # debounced: one run per burst of edits

    def _settings_changed(self):
        place_barcode_checks(self)
        self._changed()

    def _geometry_live(self, tid, roi):
        t = self.spec()
        if t is not None and tid == t["id"]:
            t["params"]["roi"] = list(roi)
            self.props.set_roi(roi, emit=False)

    def _geometry(self, tid, roi):
        self._geometry_live(tid, roi)
        self._teach()
        self._changed()

    def _teach(self):
        t = self.spec()
        text, msg = teach_check(self.project(), t)
        if msg:
            self.statusBar().showMessage(msg, 10000)
        if text is not None and t is not None:
            self.props.load(t)

    def _angle(self, tid, deg):
        t = self.spec()
        if t is not None:
            t["params"]["angle"] = round(float(deg), 3)
            self.props.set_angle(t["params"]["angle"])
            self._changed()

    def _drawn(self, kind, geom):
        t = self.spec()
        if t is None:
            return
        apply_drawn_region(t, kind, geom)
        self._teach()
        self.props.load(t)
        self._sync_regions()
        self._changed()
        self.statusBar().showMessage("New %s region" % kind, 3000)

    def _lock(self, tid, locked):
        t = self.spec()
        if t is not None and tid == t["id"]:
            if locked:
                t["params"]["locked"] = True
            else:
                t["params"].pop("locked", None)
            self._changed()

    def _polygon(self, tid, pts):
        t = self.spec()
        if t is not None:
            t["params"]["points"] = [[float(a), float(b)] for a, b in pts]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            t["params"]["roi"] = [int(min(xs)), int(min(ys)),
                                  int(max(xs) - min(xs)), int(max(ys) - min(ys))]
            self._changed()

    # ------------------------------------------------------------ buttons
    def run(self):
        self.qb.run_job(self.pid)

    def floating_display(self):
        d = VisionDisplay()
        d.setWindowTitle("Display - %s" % self.tool_id)
        d.set_records(self.display._records, prefer=self.display.current_record())
        self.qb.present(d, (700, 520))
        self._floating.append(d)
        return d

    def save_tool(self, path=None):
        t = self.spec()
        if t is None:
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Save tool", "%s.vtool.json" % t["id"],
                                                  "Tool (*.vtool.json)")
            if not path:
                return None
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(t, f, indent=2)
        except OSError as e:
            QMessageBox.warning(self, "Save tool", "Could not save the tool:\n%s" % e)
            return None
        self.statusBar().showMessage("Saved to %s" % os.path.basename(path), 4000)
        return path

    def open_tool(self, path=None):
        t = self.spec()
        if t is None:
            return
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Open tool", "", "Tool (*.vtool.json)")
            if not path:
                return
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict) or not isinstance(loaded.get("params", {}), dict):
                raise ValueError("it is not a saved tool")
        except (OSError, ValueError) as e:      # JSONDecodeError is a ValueError
            QMessageBox.warning(self, "Open tool", "Could not open %s:\n%s" % (
                os.path.basename(path), e))
            return
        if loaded.get("type") != t["type"]:
            QMessageBox.warning(self, "Open tool", "That file is a %s, not a %s." % (
                loaded.get("type"), t["type"]))
            return
        t["params"] = dict(loaded.get("params", {}))
        self.props.load(t)
        self._changed()

    def reset_tool(self):
        t = self.spec()
        if t is None:
            return
        keep = {k: t["params"].get(k) for k in ("roi", "angle", "shape", "points", "capture")
                if k in t["params"]}
        t["params"] = dict(defaults(t["type"]), **keep)
        self.props.load(t)
        self._changed()

    def results_grid(self):
        res = self.qb.station.last.get(self.pid)
        tres = next((x for x in (res or {}).get("tools", []) if x["tool_id"] == self.tool_id), None)
        w = EmptyStateTable(0, 2, "This tool has not run yet - press Run (F5).")
        w.setWindowTitle("Results - %s" % self.tool_id)
        w.setHorizontalHeaderLabels(["Result", "Value"])
        read_only_table(w)
        rows = []
        if tres:
            rows = [("status", tres["status"]), ("score", "%.4f" % tres["score"]),
                    ("time", "%.2f ms" % tres.get("ms", 0.0))]
            if tres.get("message"):
                rows.append(("message", tres["message"]))
            for k, v in (tres.get("values") or {}).items():
                if not isinstance(v, (dict,)) and not (isinstance(v, list) and v and isinstance(v[0], dict)):
                    rows.append((k, str(v)))
        w.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            w.setItem(i, 0, QTableWidgetItem(k))
            w.setItem(i, 1, QTableWidgetItem(v))
        self.qb.present(w, (460, 420))
        self._grid = w
        return w

    def help(self):
        t = self.spec()
        if t is None:
            return
        name, why = LABELS.get(t["type"], (t["type"], ""))
        QMessageBox.information(self, name, why or name)

    def closeEvent(self, e):
        _save_split(self.qb, self.split, "toolEdit/split")
        self.qb._tool_editors.pop((self.pid, self.tool_id), None)
        for fd in self._floating:
            fd.close()
        super().closeEvent(e)
        _close_frame(self)



def _save_split(qb, split, key):
    """Editors remember how their panes were divided: widen one Job Editor's
    tool tree and the next opens the same way."""
    try:
        qb._settings().setValue(key, [int(v) for v in split.sizes()])
    except Exception:
        pass


def _restore_split(qb, split, key):
    try:
        sizes = qb._settings().value(key)
        if sizes:
            sizes = [int(v) for v in sizes]
            if len(sizes) == split.count() and all(v >= 0 for v in sizes) and sum(sizes) > 0:
                split.setSizes(sizes)
    except Exception:
        pass


def _close_frame(widget):
    """Close the workspace sub-window around a window that closed itself, so
    no empty frame is left behind."""
    from PySide6.QtWidgets import QMdiSubWindow
    sub = widget.parent()
    if isinstance(sub, QMdiSubWindow):
        import shiboken6

        def close_if_alive():
            # the frame may already be gone -- closing it via its title bar
            # deletes it before this deferred call runs
            if shiboken6.isValid(sub):
                sub.close()
        QTimer.singleShot(0, close_if_alive)


TERM_ROLE = Qt.UserRole + 5                  # ("in"|"out", tool id, name, type)
TERM_MIME = "application/x-visionstation-terminal"


class TerminalTree(QTreeWidget):
    """The Job Editor's tool tree, with VisionPro's drag-to-link: drag an
    output terminal onto an input terminal. Only legal drops are accepted --
    matching type, source running earlier -- and the rules live in terminals.py,
    shared with the menus and the member browser."""

    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.LinkAction)

    def mimeTypes(self):
        return [TERM_MIME]

    def mimeData(self, items):
        from PySide6.QtCore import QMimeData
        md = QMimeData()
        t = items[0].data(0, TERM_ROLE) if items else None
        if t and t[0] == "out":
            md.setData(TERM_MIME, json.dumps(t).encode("utf-8"))
        return md

    def supportedDropActions(self):
        return Qt.LinkAction

    def _target(self, e):
        it = self.itemAt(e.position().toPoint())
        t = it.data(0, TERM_ROLE) if it else None
        if not t or t[0] != "in" or not e.mimeData().hasFormat(TERM_MIME):
            return None, None
        src = json.loads(bytes(e.mimeData().data(TERM_MIME)).decode("utf-8"))
        return t, src

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(TERM_MIME):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        dst, src = self._target(e)
        ok = dst is not None and self.editor.can_link(dst, src)
        e.setDropAction(Qt.LinkAction)
        e.accept() if ok else e.ignore()

    def dropEvent(self, e):
        dst, src = self._target(e)
        if dst is not None:
            self.editor.link_terminal(dst[1], dst[2], src[1], src[2])
        e.setDropAction(Qt.LinkAction)
        e.accept()                           # never let Qt move the rows


class MemberBrowser(QDialog):
    """VisionPro's Add Terminals: turn one of a tool's settings into an input
    driven by an earlier tool's result."""

    def __init__(self, editor, tid):
        super().__init__(editor)
        self.editor, self.tid = editor, tid
        self.setWindowTitle("Add Terminals - %s" % tid)
        self.resize(520, 360)
        tools = editor._tools()
        tool = find_tool(tools, tid)
        self.params = QListWidget()
        for key, label in T.linkable_params(tool):
            self.params.addItem("%s   (%s)" % (label, key))
            self.params.item(self.params.count() - 1).setData(Qt.UserRole, key)
        self.params.currentItemChanged.connect(lambda *_: self._fill_sources())
        self.sources = QComboBox()
        self.why = QLabel("")
        self.why.setWordWrap(True)
        self.why.setObjectName("report")
        self.why.setProperty("tone", "warn")
        add = QPushButton("Add Input")
        add.clicked.connect(self.add)
        form = QFormLayout()
        form.addRow("Setting to drive:", self.params)
        form.addRow("From earlier result:", self.sources)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.why)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(add)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)
        if self.params.count():
            self.params.setCurrentRow(0)

    def _fill_sources(self):
        self.sources.clear()
        it = self.params.currentItem()
        if it is None:
            return
        res = self.editor.results_by_id()
        for src, out in T.compatible_sources(self.editor._tools(), self.tid,
                                             it.data(Qt.UserRole), res):
            self.sources.addItem("%s.%s" % (src, out), (src, out))
        self.why.setText("" if self.sources.count() else
                         "No earlier tool has produced a number yet - run the job once, "
                         "or move this tool after the one it should read.")

    def add(self):
        it = self.params.currentItem()
        data = self.sources.currentData()
        if it is None or data is None:
            return False
        why = self.editor.link_terminal(self.tid, it.data(Qt.UserRole), data[0], data[1])
        self.why.setText(why or "Linked %s <- %s.%s" % (it.data(Qt.UserRole), *data))
        return why is None



class ImageSourceDialog(QDialog):
    """Where a job's images come from -- taught here and saved with the job:
    its reference image, a list of image files, a folder (re-scanned, so new
    images are picked up), or a camera."""

    def __init__(self, editor):
        super().__init__(editor)
        from PySide6.QtWidgets import QLineEdit, QRadioButton, QSpinBox
        self.editor = editor
        self.src = editor.qb.station.sources[editor.pid]
        self.setWindowTitle("Image Source - %s" % editor.qb.job_name(editor.pid))
        self.resize(600, 460)
        self.rRef = QRadioButton("Reference image")
        self.rFiles = QRadioButton("Image files, one per run")
        self.rFolder = QRadioButton("Folder - every image in it, one per run")
        self.rCam = QRadioButton("Camera")
        self.files = QListWidget()
        for f in (self.src.files if self.src.mode == "files" else []):
            self.files.addItem(f)
        addf = QPushButton("Add files...")
        addf.clicked.connect(self._add_files)
        clr = QPushButton("Clear")
        clr.clicked.connect(self.files.clear)
        self.folderEdit = QLineEdit(self.src.folder)
        self.folderEdit.setPlaceholderText("folder of part images")
        self.folderEdit.textEdited.connect(lambda _t: self.rFolder.setChecked(True))
        browse = QPushButton("Browse...")
        browse.clicked.connect(lambda: self._pick_folder())
        self.camIndex = QSpinBox()
        self.camIndex.setRange(0, 16)
        self.camIndex.setValue(self.src.camera_index)
        self.camStatus = QLabel("")
        self.camStatus.setWordWrap(True)
        {"reference": self.rRef, "files": self.rFiles, "folder": self.rFolder,
         "camera": self.rCam}[self.src.mode].setChecked(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 14)
        lay.setSpacing(6)
        INDENT = 26                          # under the option's label, not its circle

        def indented(inner):
            box = QWidget()
            h = QHBoxLayout(box)
            h.setContentsMargins(INDENT, 0, 0, 8)
            h.addLayout(inner, 1)
            return box
        lay.addWidget(self.rRef)
        lay.addSpacing(4)
        lay.addWidget(self.rFiles)
        row = QHBoxLayout()
        row.addWidget(self.files, 1)
        col = QVBoxLayout()
        for b in (addf, clr):
            col.addWidget(b)
        col.addStretch(1)
        row.addLayout(col)
        self.filesBox = indented(row)
        lay.addWidget(self.filesBox, 1)
        lay.addWidget(self.rFolder)
        frow = QHBoxLayout()
        frow.addWidget(self.folderEdit, 1)
        frow.addWidget(browse)
        self.folderBox = indented(frow)
        lay.addWidget(self.folderBox)
        lay.addWidget(self.rCam)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Camera number:"))
        self.camIndex.setFixedWidth(90)
        crow.addWidget(self.camIndex)
        live = QPushButton("Live view...")
        live.setToolTip("See the camera's picture to aim it and set the exposure")
        live.clicked.connect(lambda: self.editor.qb.show_live_camera(self.camIndex.value()))
        crow.addWidget(live)
        crow.addWidget(self.camStatus, 1)
        self.camBox = indented(crow)
        lay.addWidget(self.camBox)
        note = QLabel("Saved with the job, and used by an exported job file.")
        note.setObjectName("hint")
        lay.addWidget(note)
        ok_cancel(self, lay, self.apply)
        # Only the chosen source's settings are live: with every field enabled
        # it was unclear which of them the job would actually use.
        for rb in (self.rRef, self.rFiles, self.rFolder, self.rCam):
            rb.toggled.connect(self._mode_changed)
        self._mode_changed()
        guard_wheel(self)

    def _mode_changed(self, *_):
        self.filesBox.setEnabled(self.rFiles.isChecked())
        self.folderBox.setEnabled(self.rFolder.isChecked())
        self.camBox.setEnabled(self.rCam.isChecked())

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Images", "",
                                                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        for p in paths:
            self.files.addItem(p)
        if paths:
            self.rFiles.setChecked(True)

    def _pick_folder(self, folder=None):
        folder = folder or QFileDialog.getExistingDirectory(self, "Folder of part images")
        if folder:
            self.folderEdit.setText(folder)
            self.rFolder.setChecked(True)

    def open_camera(self, index):
        """A started camera. Separate so tests can supply a stand-in."""
        return open_camera(index)

    def apply(self):
        if self.rFiles.isChecked():
            paths = [self.files.item(i).text() for i in range(self.files.count())]
            if not paths:
                QMessageBox.warning(self, "Image Source", "Add at least one image.")
                return False
            self.src.use_files(paths)
        elif self.rFolder.isChecked():
            folder = self.folderEdit.text().strip()
            if not os.path.isdir(folder):
                QMessageBox.warning(self, "Image Source", "That folder does not exist.")
                return False
            self.src.use_folder(folder)
        elif self.rCam.isChecked():
            try:
                cam = self.open_camera(self.camIndex.value())
            except Exception as e:
                self.camStatus.setText("Could not open the camera: %s" % e)
                return False
            self.src.use_camera(cam, self.camIndex.value())
        else:
            self.src.use_reference()
        # part of the job: saved with it, used by an exported job file
        self.editor.project().spec["image_source"] = self.src.to_spec()
        self.editor._edited()
        self.accept()
        return True

# ================================================================ job editor
class JobEditor(QMainWindow):
    """One job: Image Source and tools, a Toolbox, and the selected tool's last
    run on the display."""

    def __init__(self, qb, pid):
        super().__init__(qb)
        self.qb, self.pid = qb, pid
        _unique_name(self, "jobEditor:%s" % pid)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1250, 780)

        bar = _unique_name(QToolBar("Job"), "jobEditorBar")
        bar.setMovable(False)
        bar.setIconSize(EDITOR_ICON)
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(bar)
        # The keys are shown but owned by QuickBuild: inside one main window
        # an editor's F5 and QuickBuild's F5 would be the same window's
        # shortcut -- ambiguous, so Qt would fire neither. QuickBuild's act on
        # whichever window is active.
        # Running and saving are the main window's Run and Save, which act on
        # the job in front; repeated here they were a second set of the same
        # buttons a hundred pixels away. This bar holds only what edits a job.
        # (The continuous toggle stays an action: F6 in this editor uses it.)
        self.aCont = make_action(self, "Run Job Continuously", self.toggle_continuous, "F6",
                                 register_key=False, check=True, icon='run_continuous')
        make_action(bar, "Image Source...", self.image_source, None,
                    "Where this job's images come from - reference, files, a folder "
                    "or a camera", icon='image_source')
        # The good-part picture every check compares with. A project started
        # from a picture of a faulty part accepted that fault on every part --
        # the checks compared it with itself -- with no way here to replace it.
        refb = QToolButton()
        refb.setText("Good reference")
        refb.setIcon(icons.icon("photo"))
        refb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        refb.setPopupMode(QToolButton.InstantPopup)
        refb.setToolTip("The picture of a GOOD part that every check compares with")
        rm = QMenu(refb)
        rm.addAction("Use the picture on screen as the good reference",
                     lambda: self.set_reference())
        rm.addAction("Choose a picture of a good part...", lambda: self.set_reference(ask=True))
        refb.setMenu(rm)
        self._ref_menu = rm
        bar.addWidget(refb)
        # Add a tool from here, by category -- the Toolbox panel needs a column
        # of the window to itself, and a small screen cannot spare one
        from vision_station.ui.widgets import fill_tool_menu
        addb = QToolButton()
        addb.setText("Add Tool")
        addb.setIcon(icons.icon("add"))
        addb.setProperty("vsIcon", "add")
        addb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        addb.setPopupMode(QToolButton.InstantPopup)
        addb.setToolTip("Add a tool to this job - it is selected, with its region in "
                        "the middle of the image, ready to place")
        self._add_menu = fill_tool_menu(QMenu(addb), self.begin_add)
        addb.setMenu(self._add_menu)
        addb.clicked.connect(self._add_clicked)     # fires only without the menu: simple view
        self.addb = addb
        bar.addWidget(addb)
        # every component in an area found by the network on the graphics
        # card, reviewed, and added as Part present checks in one step
        self.aFindParts = make_action(bar, "Find parts / text (AI)", lambda: self.find_parts(),
                                      None, "Find every component - or every line of text - "
                                      "in an area, adjust the boxes, and add a Part present, "
                                      "Verify text or Read text check for each one you keep",
                                      register_key=False, icon='sparkle')
        self._draw_hook = None
        self._part_finder = None
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        clear = make_action(bar, "Clear Results", self.reset_job, None,
                            "Clear this job's last results from the editor", icon='reset')
        bar.widgetForAction(clear).setToolButtonStyle(Qt.ToolButtonIconOnly)
        bar.widgetForAction(clear).setAccessibleName("Clear Results")

        self.tree = TerminalTree(self)
        from vision_station.ui import modern
        modern.style_check_list(self.tree)   # icon tiles and OK / NG pills
        self.tree.setMinimumWidth(240)       # names stay readable in a tiled window
        self.tree.setHeaderLabels(["Tool", "Status", "Time"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.setColumnWidth(1, 70)
        self.tree.setColumnWidth(2, 70)
        self.tree.itemDoubleClicked.connect(self._double_clicked)
        self.tree.currentItemChanged.connect(lambda *_: self._show_selected())
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context)
        # The tree's commands from the keyboard too, as any list of things has
        # them: Enter opens, F2 renames, Delete deletes (Ctrl+Z brings it back).
        from PySide6.QtGui import QShortcut
        for key, fn in ((Qt.Key_Delete, lambda: self._key_tool(self.delete_tool)),
                        (Qt.Key_F2, lambda: self._key_tool(self.rename_tool)),
                        (Qt.Key_Return, lambda: self._key_tool(
                            lambda tid: self.qb.open_tool_edit(self.pid, tid))),
                        (Qt.Key_Enter, lambda: self._key_tool(
                            lambda tid: self.qb.open_tool_edit(self.pid, tid)))):
            sc = QShortcut(key, self.tree)
            sc.setContext(Qt.WidgetShortcut)
            sc.activated.connect(fn)

        self.display = VisionDisplay()
        self.display.recordChanged.connect(lambda _n: self._sync_job_regions())
        # The image is a drawing canvas, as in Illustrator: every tool's
        # region is on it, and any of them can be selected, moved, resized,
        # rotated, aligned, locked, duplicated or deleted right there. The
        # tool list is its layers panel -- selecting in either selects in both.
        self.display.enable_drawing(True)
        c = self.display.canvas
        c.selectionChanged.connect(self._canvas_selected)
        c.liveGeometry.connect(self._canvas_live)
        c.geometryCommitted.connect(self._canvas_geometry)
        c.angleChanged.connect(self._canvas_angle)
        c.polygonCommitted.connect(self._canvas_polygon)
        c.lockChanged.connect(self._canvas_lock)
        c.duplicateRequested.connect(self.duplicate_tools)
        c.deleteRequested.connect(self.delete_tools)
        c.shapeDrawn.connect(self._canvas_drawn)
        self._from_canvas = False
        self._regions_editable = True
        self._drawn_popup = None
        # adding a check: its kind, while its area is being drawn
        self._armed = None
        c.escaped.connect(self.cancel_add)
        self.display.modeSet.connect(self._mode_set)
        # the job on its own teach picture: every check's result while it is
        # set up, as the Find parts panel shows what it reads as boxes move
        self._preview = None
        self._preview_key = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(lambda: self.qb.station.preview(self.pid))
        self.qb.station.previewResult.connect(self._previewed)
        # a drawing tool taken up on a picture the checks are not drawn on
        # brings the teach picture first
        self.display.before_draw = self._before_draw

        # Laid out as Illustrator and Figma are: the image takes the window,
        # and one panel on the right holds the layers -- the tools, in the
        # order they run -- above the properties of the one selected, edited
        # right here. Settings used to need a second tab opened for each tool,
        # and the tool list took a third of the width while mostly empty.
        from vision_station.ui.classic_window import PropertySheet
        self.props = PropertySheet()
        self.props.changed.connect(self._props_changed)
        self.props.header.setText("Properties")
        self.props.desc.setText("Select a tool in Layers, or click its region on the "
                                "image, to edit it here.")
        full = QPushButton("Open in Tool Editor  ›")
        full.setObjectName("linkButton")         # a secondary path, not a second primary button
        full.setToolTip("Training images, graphics and the results grid, in the tool's own tab")
        full.clicked.connect(lambda: self.selected_tool() and
                             self.qb.open_tool_edit(self.pid, self.selected_tool()))
        self.fullEditor = full
        self.props.layout().insertWidget(2, full)
        self.groupBar = GroupBar(self)
        self.props.layout().insertWidget(3, self.groupBar)
        self._multi_ids = None
        layers_title = QLabel("Layers")
        layers_title.setObjectName("sectionTitle")
        self.layersTitle = layers_title
        layers = QWidget()
        ll = QVBoxLayout(layers)
        ll.setContentsMargins(10, 8, 6, 0)
        ll.setSpacing(4)
        # The guide: the five steps of setting up an inspection, for the
        # simple view. It reads the job, so it ticks itself off.
        self.guide = simple.GuidePanel(on_add=self.pick_check,
                                       on_run=lambda: self.qb.run_job(self.pid),
                                       on_save=lambda: self.qb.save_job(self.pid))
        ll.addWidget(self.guide)
        ll.addWidget(layers_title)
        # forty checks from Find parts: one is found by typing part of its name
        self.checkFilter = QLineEdit()
        self.checkFilter.setObjectName("searchBox")
        self.checkFilter.setPlaceholderText("Filter checks - name or kind")
        self.checkFilter.setClearButtonEnabled(True)
        self.checkFilter.textChanged.connect(self._filter_tree)
        ll.addWidget(self.checkFilter)
        ll.addWidget(self.tree, 1)
        side = QSplitter(Qt.Vertical)
        side.addWidget(layers)
        side.addWidget(self.props)
        side.setSizes([260, 420])
        side.setChildrenCollapsible(False)
        self.side = side
        self.tree.setMinimumWidth(200)
        # the names get the width; status and time only what they need
        hdr = self.tree.header()
        self.tree.setHeaderHidden(True)          # names, status and time read without labels
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.display)
        split.addWidget(side)
        split.setSizes([820, 340])
        split.setStretchFactor(0, 1)          # the image takes the extra room
        split.setStretchFactor(1, 0)
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)
        self.split = split
        _restore_split(qb, split, "jobEditor/split2")
        _restore_split(qb, side, "jobEditor/side")

        self.sDot = QLabel("")
        self.sDot.setFixedSize(12, 12)
        self.sText = QLabel("not run")
        self.statusBar().addWidget(self.sDot)
        self.statusBar().addWidget(self.sText, 1)
        self.simple = not getattr(qb, "advanced", False)
        self._apply_simple()
        self.refresh()


    def project(self):
        return self.qb.station.projects.get(self.pid)

    # ------------------------------------------------------------ simple view
    def set_simple(self, on):
        on = bool(on)
        if on == self.simple:
            return
        self.simple = on
        self._apply_simple()
        self.refresh()

    def _apply_simple(self):
        """The same editor, worded for someone who did not build it: checks
        not tools and layers, a picker of questions instead of a menu of
        algorithms, the guide, and no terminals in the list."""
        on = self.simple
        if on:
            self.addb.setMenu(None)
            self.addb.setText("Add a check")
            self.addb.setToolTip("Pick what you want to check; its box appears on the picture, "
                                 "ready to drag over the feature")
        else:
            self.addb.setMenu(self._add_menu)
            self.addb.setText("Add Tool")
            self.addb.setToolTip("Add a tool to this job - it is selected, with its region in "
                                 "the middle of the image, ready to place")
        self.addb.setPopupMode(QToolButton.InstantPopup)
        self.layersTitle.setText("Checks" if on else "Layers")
        self.fullEditor.setText("All settings, training and graphics  ›" if on else
                                "Open in Tool Editor  ›")
        self.guide.setVisible(on)
        self.tree.setColumnHidden(2, on)         # the time is in the tooltip; names get the room
        self.props.set_simple(on)
        if on:
            self.side.setSizes([max(self.side.sizes()[0], 440), 380])

    def _add_clicked(self):
        if self.simple:
            self.pick_check()

    def pick_check(self):
        """The check picker, then the tool it chose."""
        if self.qb._locked_out():
            return None
        ty = simple.CheckPicker.pick(self, advanced=not self.simple)
        if not ty:
            return None
        tid = self.begin_add(ty)
        self.display.canvas.setFocus()
        return tid

    # ------------------------------------------------------------ tree
    def refresh(self):
        p = self.project()
        if p is None:
            self.close()
            return
        name = self.qb.job_name(self.pid)
        dirty = " *" if self.pid in self.qb.dirty else ""
        self.setWindowTitle("Job: %s%s" % (name, dirty))
        self._schedule_preview()
        res = self._run_res()
        shown = res
        pv = self._preview_res()
        # The list shows the last run -- unless the job was changed since: then
        # what each check gives now, on the teach picture. (By the time a
        # result came back it showed the teach picture's OK beside a run's NG.)
        if pv is not None and (res is None or
                               getattr(self, "_edited_at", 0) > res.get("timestamp", 0)):
            shown = pv
        byid = {t["tool_id"]: t for t in (shown or {}).get("tools", [])}
        cur = self.tree.currentItem()
        cur_id = cur.data(0, Qt.UserRole) if cur else None
        # Rebuilding the rows fires selection changes; they are not the
        # operator choosing a tool, and treating them as one reset the display
        # to LastRun after every run.
        self.tree.blockSignals(True)
        self.tree.clear()
        src = QTreeWidgetItem(["Image Source", "", ""])
        src.setIcon(0, icons.icon("image_source"))
        src.setData(0, Qt.UserRole, "__source__")
        src.setToolTip(0, self.qb.station.sources[self.pid].describe())
        so = QTreeWidgetItem(["[out] OutputImage", "", ""])
        so.setData(0, TERM_ROLE, ("out", T.SOURCE, "OutputImage", "image"))
        so.setFlags((so.flags() | Qt.ItemIsDragEnabled) & ~Qt.ItemIsDropEnabled)
        src.addChild(so)
        src.setFlags(src.flags() & ~Qt.ItemIsDragEnabled & ~Qt.ItemIsDropEnabled)
        self.tree.addTopLevelItem(src)
        fixture = None
        plain = self.simple
        for t in p.spec.get("tools", []):
            r = byid.get(t["id"])
            name = simple.display_name(t) if plain else \
                t["id"] + ("  (fixture)" if t["type"] == "align" else "")
            it = QTreeWidgetItem([name, (r["status"].upper() if r else ""),
                                  ("%.1f ms" % r["ms"] if r["ms"] < 1000 else fmt_ms(r["ms"]))
                                  if r else ""])
            it.setData(0, Qt.UserRole, t["id"])
            it.setData(0, Qt.UserRole + 7, t["type"])      # modern.TYPE_ROLE
            it.setFlags(it.flags() & ~Qt.ItemIsDragEnabled & ~Qt.ItemIsDropEnabled)
            if plain:
                it.setToolTip(0, "%s  (%s)\n%s\nF2 renames it, Del removes it" % (
                    LABELS.get(t["type"], (t["type"], ""))[0], t["id"],
                    simple.friendly_message(r) if r and r["status"] != "pass" else
                    LABELS.get(t["type"], ("", ""))[1]))
            else:
                it.setToolTip(0, "%s\nDouble-click or Enter to edit, F2 to rename, Del to delete"
                              % LABELS.get(t["type"], (t["type"], ""))[0])
            col = QColor(status_dot(r["status"]) if r else IDLE_C)
            it.setForeground(1, QBrush(status_text(r["status"] if r else None)))
            it.setIcon(0, icons.tool_icon(t["type"]))     # what kind of tool
            it.setIcon(1, dot_icon(col))                  # how it did
            if t.get("params", {}).get("enabled", True) is False:
                it.setForeground(0, QBrush(QColor(theme.color("text_disabled"))))
                it.setText(1, "off")
            # terminals, from terminals.py: inputs with their source, then outputs
            for name, typ, source in ([] if plain else T.inputs(t)):
                c = QTreeWidgetItem(["[in]  %s  <-  %s" % (name, source), "", ""])
                c.setData(0, TERM_ROLE, ("in", t["id"], name, typ))
                c.setToolTip(0, "%s input. Drop a matching output here, or right-click "
                                "to link." % typ)
                c.setFlags((c.flags() | Qt.ItemIsDropEnabled) & ~Qt.ItemIsDragEnabled)
                it.addChild(c)
            for name, typ in ([] if plain else T.outputs(t, r)):
                val = ""
                if typ == "value" and r:
                    v = r["score"] if name == "score" else (r.get("values") or {}).get(name)
                    val = "  = %s" % (("%.4g" % v) if isinstance(v, float) else v)
                c = QTreeWidgetItem(["[out] %s%s" % (name, val), "", ""])
                c.setData(0, TERM_ROLE, ("out", t["id"], name, typ))
                c.setToolTip(0, "%s output. Drag onto a matching input to link it." % typ)
                c.setFlags((c.flags() | Qt.ItemIsDragEnabled) & ~Qt.ItemIsDropEnabled)
                it.addChild(c)
            if t["type"] == "align" and fixture is None:
                fixture = t["id"]
            self.tree.addTopLevelItem(it)
            for k in range(it.childCount()):
                it.child(k).setFirstColumnSpanned(True)   # terminal text uses the row
            if t["id"] == cur_id:
                self.tree.setCurrentItem(it)
        if self.tree.currentItem() is None:
            self.tree.setCurrentItem(src)
        self.tree.blockSignals(False)
        self.checkFilter.setVisible(len(p.spec.get("tools", [])) >= 6)
        self._filter_tree(self.checkFilter.text())
        if plain:
            self.guide.update_state(p.spec.get("tools", []), res, self.pid in self.qb.dirty)
        self._status(res)
        self._show_selected()

    def _filter_tree(self, text):
        words = (text or "").lower().split()
        for i in range(1, self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            tid = it.data(0, Qt.UserRole) or ""
            hay = "%s %s %s" % (it.text(0), tid, it.data(0, Qt.UserRole + 7) or "")
            it.setHidden(bool(words) and not all(w in hay.lower() for w in words))

    def _status(self, res):
        if not res:
            self.sDot.setStyleSheet(dot_style(IDLE_C))
            self.sText.setText("Not run yet - press Run (F5) to test" if self.simple else "not run")
            return
        self.sDot.setStyleSheet(dot_style(status_dot(res["verdict"])))
        if self.simple:
            why = simple.reasons(res, self._tools())
            self.sText.setText("%s   %s%s" % (
                "Accept" if res["verdict"] == "PASS" else "Reject", fmt_ms(res["total_ms"]),
                ("   ·   " + "   ·   ".join(why)) if why else ""))
            self.sText.setToolTip("\n".join(why))
            return
        failed = [t["tool_id"] for t in res["tools"] if t["status"] != "pass"]
        self.sText.setText("%s   %s%s" % (
            "Accept" if res["verdict"] == "PASS" else "Reject", fmt_ms(res["total_ms"]),
            ("   failed: " + ", ".join(failed)) if failed else ""))

    def selected_tool(self):
        it = self.tree.currentItem()
        while it is not None and it.parent() is not None:
            it = it.parent()
        tid = it.data(0, Qt.UserRole) if it else None
        return None if tid in (None, "__source__") else tid

    # ------------------------------------------------------------ results shown
    def _run_res(self):
        """The last run, without checks deleted since: their old outlines and
        tags stayed on the picture until the next run."""
        res = self.qb.station.last.get(self.pid)
        if not res:
            return res
        ids = {t["id"] for t in self._tools()}
        tools = [t for t in res.get("tools", []) if t["tool_id"] in ids]
        return res if len(tools) == len(res.get("tools", [])) else dict(res, tools=tools)

    def _preview_res(self):
        pv = self._preview
        if not pv:
            return None
        ids = {t["id"] for t in self._tools()}
        return dict(pv, tools=[t for t in pv.get("tools", []) if t["tool_id"] in ids])

    def _schedule_preview(self):
        p = self.project()
        if p is None or p.reference is None:
            return
        key = (json.dumps(p.spec, sort_keys=True, default=str), id(p.reference),
               _examples_signature(p.folder))
        if key != self._preview_key:
            if self._preview_key is not None:
                self._edited_at = time.time()    # a real edit, not the editor opening
            self._preview_key = key
            self._preview_timer.start()

    def _previewed(self, pid, res):
        if pid != self.pid:
            return
        self._preview = res
        self.refresh()

    def _results_here(self):
        """The results that belong on the picture showing: the teach
        picture's own on the teach picture, the last run's elsewhere."""
        if self.display.current_record() == TEACH:
            return self._preview_res()
        return self._run_res()

    def _show_selected(self):
        res = self._run_res()
        src = self.qb.station.sources[self.pid]
        p = self.project()
        current = src.current if src.current is not None else p.reference
        tid = self.selected_tool()
        # A check chosen in the list is shown on the teach picture, where it
        # is drawn and taught; otherwise the record the operator chose stays,
        # across runs. (It showed the last run, where the box could be moved
        # but was taught from a different picture: the second part drawn on
        # a new board failed the moment it was drawn.)
        switched = tid != getattr(self, "_shown_tid", object())
        self._shown_tid = tid
        prefer = None
        if switched and not self._from_canvas:
            prefer = TEACH if tid is not None else ("LastRun.InputImage" if res else None)
        # a region clicked on the image keeps the image that is showing
        self.display.set_records(records_for(res, tid, current, reference=p.reference,
                                             teach=self._preview_res()),
                                 prefer=prefer)
        self.display.set_spaces(spaces_for(res))
        self._sync_job_regions()
        self._show_properties(tid, res)

    def _show_properties(self, tid, res):
        """The selected tool's settings, editable in place. Reloaded only for a
        different tool (or a recipe restored by undo), so typing in a field
        is never interrupted by the refresh its own edit causes."""
        t = self._tool(tid) if tid else None
        self.fullEditor.setEnabled(t is not None)
        if t is None:
            if self.props.tool is not None:
                self.props.load(None)
            self.props.header.setText("Settings" if self.simple else "Properties")
            self.props.desc.setText("Click a check in the list, or its box on the picture, "
                                    "to say what it should expect." if self.simple else
                                    "Select a tool in Layers, or click its region on "
                                    "the image, to edit it here.")
            self.props.show_result(None)
            return
        self.props.recipe = self._tools()
        if self.props.tool is not t:
            self.props.set_captures(readable_images(self._tools(), t["id"], ["default"]))
            self.props.load(t)
        here = self._results_here() or res
        tres = next((x for x in (here or {}).get("tools", []) if x["tool_id"] == tid), None)
        self.props.show_result(tres)
        self._update_group_bar()

    def _update_group_bar(self):
        tools = self._tools()
        sel = set(self.display.canvas.selected_ids())
        multi = [t for t in tools if t["id"] in sel] if len(sel) > 1 else []
        if len(multi) > 1:
            self.groupBar.show_for(multi, "%d checks selected" % len(multi))
            return
        tid = self.selected_tool()
        t = self._tool(tid) if tid else None
        members = group_members(t, tools)
        name = group_name(t) if t else ""
        kind = LABELS.get(t["type"], (t["type"], ""))[0] if t else ""
        self.groupBar.show_for(members, "Group %s  -  %d %s checks" % (name, len(members), kind))

    def _group_edited(self, what):
        """A change made to a whole group: one undo step, and the selected
        check's own settings show it."""
        self._from_canvas = True
        try:
            self._edited()
        finally:
            self._from_canvas = False
        t = self.props.tool
        if t is not None and any(t is m for m in self.groupBar.members):
            self.props.load(t)
        self._update_group_bar()
        self.statusBar().showMessage(what, 6000)

    def _props_changed(self):
        """A setting changed in the properties panel."""
        place_barcode_checks(self)
        self._from_canvas = True              # keep the selection on the image as it is
        try:
            self._edited()
        finally:
            self._from_canvas = False

    def _sync_job_regions(self):
        """Every tool's region on the full-frame records, editable in place.
        A tool that reads another tool's output image is left off: its region
        is in that image's coordinates, not the camera's."""
        p = self.project()
        tid = self.selected_tool()
        rec = self.display.current_record()
        res = self._run_res()
        img = self.display._records.get(rec, (None, None))[0]
        full = rec in FULL_FRAME_RECORDS and regions_fit(rec, img, p.reference, res)
        self._regions_editable = full
        tools = p.spec.get("tools", [])
        producers = {t["id"] for t in tools}
        regions = [t for t in tools if full and
                   (t.get("params", {}).get("capture") or "default") not in producers]
        c = self.display.canvas
        c.sync_regions(regions, self._results_here() if rec == TEACH else res)
        if self._armed:
            pass                              # the note says what to draw
        elif rec in ("Current.InputImage", "LastRun.InputImage") and not full \
                and img is not None:
            self.display.set_note(
                "Boxes show where each check found its part on this picture. "
                "Checks are drawn and moved on the Teach picture.",
                ("Show Teach picture", lambda: self.display.record.setCurrentText(TEACH)))
        else:
            self.display.set_note(None)
        for item in c.regions.values():
            item.setFlag(item.GraphicsItemFlag.ItemIsMovable, True)
            item.setFlag(item.GraphicsItemFlag.ItemIsSelectable, True)
        if not self._from_canvas:
            c.select_ids([tid] if tid in c.regions else [], silent=True)
        self.display._sync_transform()

    # ------------------------------------------------------------ the canvas
    def _canvas_selected(self, ids):
        """Clicking a region selects its tool in the list. Several regions
        selected leave the list alone, and the group bar takes them all."""
        if len(ids) > 1:
            self._multi_ids = list(ids)
            self._update_group_bar()
            return
        self._multi_ids = None
        if len(ids) != 1 or ids[0] == self.selected_tool():
            self._update_group_bar()
            return
        self._from_canvas = True
        try:
            self.select_tool(ids[0])
        finally:
            self._from_canvas = False

    def _tool(self, tid):
        return find_tool(self._tools(), tid)

    def _canvas_live(self, tid, roi):
        t = self._tool(tid)
        if t is not None:
            t["params"]["roi"] = list(roi)          # in step while dragging; saved on release
            if self.props.tool is t:
                self.props.set_roi(roi, emit=False)

    def _changed_since_recorded(self):
        return self.qb._snap.get(self.pid) != self.project().spec

    def _canvas_geometry(self, tid, roi):
        """On release, after arranging, or typed in the transform bar. A click
        that moved nothing changes nothing -- the job is not marked unsaved."""
        t = self._tool(tid)
        if t is None:
            return
        t["params"]["roi"] = list(roi)
        if self._changed_since_recorded():
            self._teach(tid)
            self._edited_from_canvas()

    def _canvas_angle(self, tid, deg):
        t = self._tool(tid)
        if t is None:
            return
        t["params"]["angle"] = round(float(deg), 3)
        if self.props.tool is t:
            self.props.set_angle(t["params"]["angle"])
        # while the grip is being dragged the recipe just follows; the windows
        # are redrawn once, when the button is let go
        if QApplication.mouseButtons() == Qt.NoButton and self._changed_since_recorded():
            self._edited_from_canvas()

    def _canvas_polygon(self, tid, pts):
        t = self._tool(tid)
        if t is not None:
            t["params"]["points"] = [[float(a), float(b)] for a, b in pts]
            xs, ys = [q[0] for q in pts], [q[1] for q in pts]
            t["params"]["roi"] = [int(min(xs)), int(min(ys)),
                                  int(max(xs) - min(xs)), int(max(ys) - min(ys))]
            self._edited_from_canvas()

    def _canvas_lock(self, tid, locked):
        t = self._tool(tid)
        if t is not None:
            if locked:
                t["params"]["locked"] = True
            else:
                t["params"].pop("locked", None)
            self._edited_from_canvas()

    # ------------------------------------------------------------ adding by drawing
    NO_REGION = ("distance", "image_process", "data_analysis", "results_analysis",
                 "intersect", "construct", "fixture_npoint", "calibrate_npoint")

    def begin_add(self, ty):
        """A check chosen from the menu or the picker: the pointer becomes a
        cross and the check's area is drawn where it belongs -- anywhere,
        any size. It used to appear as a box in the middle of the picture,
        to be dragged into place every time. A box, ellipse (E) or polygon
        (P); Esc cancels. Checks with no area of their own are added at once,
        and so is one whose area was drawn first."""
        if self.qb._locked_out():
            return None
        if ty in self.NO_REGION or getattr(self, "_pending_area", None) is not None \
                or self.project().reference is None:
            return self.add_tool(ty)
        if self._drawn_popup is not None:
            self._drawn_popup.close()
        self.display.record.setCurrentText(TEACH)
        self._armed = ty
        shape = {"ocv": "draw_rect", "ocr": "draw_rect"}.get(ty, "draw_rect")
        self.display._set_mode(shape)
        self.display.aRect.setChecked(True)
        name = LABELS.get(ty, (ty, ""))[0]
        self.display.set_note("Drag on the picture where the %s check goes - E for an "
                              "ellipse or circle, P for a polygon. Esc cancels." % name,
                              ("Cancel", self.cancel_add))
        self.statusBar().showMessage("Draw the area of the new %s check" % name, 8000)
        self.display.canvas.setFocus()
        return None

    def cancel_add(self):
        if not self._armed:
            return
        self._armed = None
        self.display.set_note(None)
        self.display._set_mode("select")
        self.display.aPointer.setChecked(True)
        self.statusBar().showMessage("Not added", 3000)

    def _mode_set(self, mode):
        # the pointer or the hand taken up while adding: the add is called off
        if self._armed and not mode.startswith("draw"):
            self.cancel_add()

    # ------------------------------------------------------------ teaching
    def _teach(self, tid, quiet=False):
        text, msg = teach_check(self.project(), self._tool(tid), self._name_of(tid))
        if msg and not (quiet and text):           # adding says it once, itself
            self.statusBar().showMessage(msg, 10000)
        t = self._tool(tid)
        if text is not None and t is not None and self.props.tool is t:
            self.props.load(t)
        return text

    # ------------------------------------------------------------ the good reference
    def teach_with_best_light(self, wait=True):
        """A new teach picture for this job from its camera, at the light
        level that gives the clearest picture (QuickBuild.ai_reference). The
        level becomes this job's own. Offered when an application or a job
        is created; kept here for a job that already exists."""
        st = self.qb.station
        src = st.sources.get(self.pid)
        if src is None or src.mode != "camera":
            QMessageBox.information(self, "AI reference image", "This job does not take its "
                                    "picture from the camera - set its image source to the "
                                    "camera first.")
            return None

        def grab():
            img = src.grab()
            if img is None:
                raise RuntimeError(src.last_error or "the camera gave no picture")
            return img
        # the raw camera picture: whether to take it in black and white is
        # decided afresh from it, not from a picture already made so
        keep, src.mono = src.mono, None
        try:
            got = self.qb.ai_reference("AI reference image", grab=grab)
        finally:
            src.mono = keep
        if got is None:
            return None
        if not self.set_reference(image=got["image"]):
            return None
        p = self.project()
        if got["level"] is not None:
            p.spec["light_level"] = int(got["level"])
            write_recipe_keys(p.path, {"light_level": int(got["level"])})
            st.light.forget_job(self.pid)
        if (got.get("mono") or None) != src.mono:
            src.mono = got.get("mono") or None
            p.spec["image_source"] = src.to_spec()
            write_recipe_keys(p.path, {"image_source": p.spec["image_source"]})
        self.statusBar().showMessage(got["message"], 15000)
        self._last_best_light = got["result"]
        return got

    def set_reference(self, image=None, ask=False):
        """Make a picture of a good part this inspection's reference. Checks
        that compare (Part present, surface, colour...) compare with it, so
        it must show a GOOD part; regions stay where they are."""
        if ask:
            path = self.qb._ask_image("Picture of a GOOD part")
            if not path:
                return None
            image = self.qb._read_reference(path, "Good reference")
        M = None
        if image is None:
            # the picture on screen: the last inspection, or the next picture
            rec = self.display.current_record()
            if rec in ("LastRun.InputImage", "Current.InputImage"):
                image = self.display._records.get(rec, (None, None))[0]
                res = self.qb.station.last.get(self.pid)
                if rec == "LastRun.InputImage" and fixture_moved(res):
                    M = next(iter(spaces_for(res).values()))
        if image is None:
            src = self.qb.station.sources.get(self.pid)
            image = src.current if src is not None else None
        if image is None:
            QMessageBox.information(self, "Good reference", "No picture yet - run the "
                                    "inspection once on a good part, or choose a file.")
            return None
        p = self.project()
        old = p.reference
        if old is not None and old.shape[:2] != image.shape[:2]:
            QMessageBox.warning(self, "Good reference", "That picture is %d x %d; this "
                                "inspection's regions were drawn on %d x %d. Use a picture "
                                "from the same camera and position." % (
                                    image.shape[1], image.shape[0], old.shape[1], old.shape[0]))
            return None
        p.set_reference(image.copy())
        if M is not None:
            # The part lies elsewhere on this picture: every region goes with
            # it, by the transform the fixture found, so each stays on its
            # feature. Left where they were, they sat beside their parts.
            move_regions(self._tools(), M)
            self.qb.mark_dirty(self.pid)
        self.qb.job_changed(self.pid)
        self.display.record.setCurrentText(TEACH)
        self.statusBar().showMessage(
            "Teach picture replaced%s - every check now compares with this picture" % (
                ", and every box moved with the part" if M is not None else ""), 8000)
        return True

    # ------------------------------------------------------------ an area drawn first
    def _keep_pending_area(self, kind, geom):
        """A shape drawn with no check selected, shown dashed until a check
        is added (which takes it) or another shape replaces it."""
        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtGui import QColor, QPen, QPolygonF
        self._drop_pending_area()
        sc = self.display.canvas.sc
        from vision_station.ui.canvas import SEL
        pen = QPen(SEL)
        pen.setCosmetic(True)
        pen.setWidthF(2.0)
        pen.setStyle(Qt.DashLine)
        geom = dict(geom)
        if kind == "polygon":
            item = sc.addPolygon(QPolygonF([QPointF(*p) for p in geom["points"]]), pen)
            item.setZValue(40)
        else:
            # Adjustable while it waits for its check: drag to move, drag a
            # grip to resize, drag just outside a corner to turn. It was a
            # plain dashed outline -- a box drawn a little off could not be
            # fixed before choosing the check, only drawn again.
            from PySide6.QtGui import QBrush
            from vision_station.ui.part_finder_panel import EditableBox

            def changed(roi, angle):
                geom["roi"] = [int(v) for v in roi]
                geom["angle"] = float(angle or 0.0)
            item = EditableBox(geom["roi"], on_change=changed, z=40,
                               angle=float(geom.get("angle", 0.0) or 0.0),
                               ellipse=(kind == "ellipse"), handles=True)
            item.setPen(pen)
            item.setBrush(QBrush(QColor(SEL.red(), SEL.green(), SEL.blue(), 26)))
            sc.addItem(item)
        item.setToolTip("This area becomes the region of the next check you add - drag it to "
                        "move it, drag a grip to resize it, drag just outside a corner to turn it")
        self._pending_area = (kind, geom, item)

    def _drop_pending_area(self):
        p = getattr(self, "_pending_area", None)
        self._pending_area = None
        if p is not None and p[2].scene() is not None:
            p[2].scene().removeItem(p[2])

    def _take_pending_area(self):
        p = getattr(self, "_pending_area", None)
        if p is None:
            return None
        self._drop_pending_area()
        return p[0], p[1]

    def find_parts(self):
        """The Find parts panel beside this editor: one, brought forward."""
        from vision_station.ui.part_finder_panel import PartFinderPanel
        # it finds on the teach picture, so that is what is shown under it
        self.display.record.setCurrentText(TEACH)
        p = self._part_finder
        try:
            alive = p is not None and p.isVisible()
        except RuntimeError:
            alive = False
        if not alive:
            p = PartFinderPanel(self)
            p.setAttribute(Qt.WA_DeleteOnClose, True)
            self._part_finder = p
            g = self.geometry()
            p.move(self.mapToGlobal(g.topLeft()).x() + max(0, self.width() - p.width() - 30),
                   self.mapToGlobal(g.topLeft()).y() + 80)
        p.show()
        p.raise_()
        return p

    # the checks offered on a box just drawn, most used first
    QUICK_CHECKS = ("presence", "ocv", "ocr", "code_read", "count_template",
                    "measure_length", "color_match", "defect_diff")

    def _before_draw(self):
        if not getattr(self, "_regions_editable", True):
            self.display.record.setCurrentText(TEACH)
            self.statusBar().showMessage("Drawing on the Teach picture - checks are taught "
                                         "from the picture they are drawn on", 6000)

    def _canvas_drawn(self, kind, geom):
        """A shape drawn on the image: a new check goes there -- the kind is
        chosen from a menu beside it. Unless a panel (Find parts) asked for
        the next box drawn.

        It used to become the selected check's region. Drawing a second part
        with the first still selected moved the first onto it, and the
        first check then failed on its own reference: "the second part I
        draw shows NG at once". Moving a check is done by dragging its box,
        or by "Move ... here" in the same menu."""
        hook = getattr(self, "_draw_hook", None)
        if hook is not None:
            self._draw_hook = None
            hook(kind, geom)
            return
        if self._armed:
            # the check chosen first: its area is what was just drawn
            ty, self._armed = self._armed, None
            self.display.set_note(None)
            self._keep_pending_area(kind, geom)
            return self.add_tool(ty)
        # Kept, not thrown away: drawn first, it becomes the region of the
        # check added next -- from this menu, the toolbar or the picker.
        self._keep_pending_area(kind, geom)
        self.statusBar().showMessage(
            "Area kept - choose the check for it, or add one and it goes exactly here", 8000)
        self._offer_checks()

    def _offer_checks(self):
        from PySide6.QtGui import QCursor
        m = QMenu(self)
        m.setObjectName("drawnMenu")
        m.addSection("Add a check on this area")
        for ty in self.QUICK_CHECKS:
            if ty in LABELS:
                a = m.addAction(icons.tool_icon(ty), LABELS[ty][0])
                a.setToolTip(LABELS[ty][1])
                a.triggered.connect(lambda _c=False, ty=ty: self.add_tool(ty))
        m.addAction(icons.icon("add"), "More checks...", self.pick_check)
        tid = self.selected_tool()
        t = self._tool(tid) if tid is not None else None
        if t is not None and t.get("type") not in ("distance", "image_process"):
            m.addSeparator()
            m.addAction("Move %s here" % self._name_of(tid),
                        lambda: self.move_tool_region(tid))
        m.setToolTipsVisible(True)
        self._drawn_popup = m
        m.popup(QCursor.pos())
        return m

    def _name_of(self, tid):
        t = self._tool(tid)
        return simple.display_name(t) if (t is not None and self.simple) else tid

    def move_tool_region(self, tid):
        """The area drawn last becomes this check's region, and the check is
        taught again from the picture it is drawn on."""
        pending = self._take_pending_area()
        t = self._tool(tid)
        if pending is None or t is None:
            return
        apply_drawn_region(t, *pending)
        self._teach(tid)
        self._edited()
        self.select_tool(tid)
        self.statusBar().showMessage("New %s region for %s" % (pending[0], tid), 4000)

    def _edited_from_canvas(self):
        """A region changed on the image: the recipe follows, and every other
        window showing the job is redrawn -- this display included, but
        without taking the selection away from what is being edited."""
        self._from_canvas = True
        try:
            self._edited()
        finally:
            self._from_canvas = False

    def duplicate_tools(self, ids):
        """Copies placed just beside their originals, each with a new name,
        and selected -- Ctrl+D or Alt-drag on the image."""
        import copy
        tools = self._tools()
        new = []
        for tid in ids:
            t = self._tool(tid)
            if t is None:
                continue
            d = copy.deepcopy(t)
            base, n = "%s_copy" % tid, 1
            nid = base
            while any(x["id"] == nid for x in tools):
                n += 1
                nid = "%s%d" % (base, n)
            d["id"] = nid
            r = d["params"].get("roi")
            if r:
                d["params"]["roi"] = [r[0] + 20, r[1] + 20, r[2], r[3]]
            if d["params"].get("points"):
                d["params"]["points"] = [[x + 20, y + 20] for x, y in d["params"]["points"]]
            d["params"].pop("locked", None)
            tools.insert(tools.index(t) + 1, d)
            new.append(nid)
        if new:
            self._edited()
            self.display.canvas.select_ids(new)
            if len(new) == 1:
                self.select_tool(new[0])
            self.statusBar().showMessage("Duplicated %s" % ", ".join(ids), 4000)
        return new

    def delete_tools(self, ids):
        if self.qb._locked_out():
            return
        tools = self._tools()
        tools[:] = [t for t in tools if t["id"] not in ids]
        self._edited()
        self.statusBar().showMessage("Deleted %s - Ctrl+Z brings %s back" % (
            ", ".join(ids), "it" if len(ids) == 1 else "them"), 6000)

    def _key_tool(self, fn):
        tid = self.selected_tool()
        if tid and not self.qb._locked_out():
            fn(tid)

    def select_tool(self, tid):
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.data(0, Qt.UserRole) == tid:
                self.tree.setCurrentItem(it)
                self.tree.scrollToItem(it)
                return True
        return False

    def _double_clicked(self, item, _col):
        while item.parent() is not None:
            item = item.parent()
        tid = item.data(0, Qt.UserRole)
        if tid == "__source__":
            self.image_source()
        elif tid:
            self.qb.open_tool_edit(self.pid, tid)

    def _context(self, pos):
        clicked = self.tree.itemAt(pos)
        term = clicked.data(0, TERM_ROLE) if clicked else None
        m = QMenu(self)
        if term and term[0] == "in":
            _k, dst, name, _typ = term
            sub = m.addMenu("Link to")
            srcs = T.compatible_sources(self._tools(), dst, name, self.results_by_id())
            for sid, out in srcs:
                sub.addAction("%s.%s" % (sid, out),
                              lambda sid=sid, out=out: self.link_terminal(dst, name, sid, out))
            sub.setEnabled(bool(srcs))
            m.addAction("Unlink", lambda: self.unlink_terminal(dst, name))
            m.exec(self.tree.viewport().mapToGlobal(pos))
            return
        tid = self.selected_tool()
        if tid:
            if not self.simple:
                m.addAction("Add Terminals...", lambda: self.member_browser(tid))
            m.addAction("All settings, training and graphics..." if self.simple else "Open",
                        lambda: self.qb.open_tool_edit(self.pid, tid))
            m.addAction("Rename...", lambda: self.rename_tool(tid))
            m.addAction("Enable / disable", lambda: self.toggle_tool(tid))
            m.addSeparator()
            m.addAction("Move up", lambda: self.move_tool(tid, -1))
            m.addAction("Move down", lambda: self.move_tool(tid, 1))
            m.addSeparator()
            m.addAction("Delete", lambda: self.delete_tool(tid))
        else:
            m.addAction("Image Source...", self.image_source)
        m.exec(self.tree.viewport().mapToGlobal(pos))

    def _edited(self):
        """The job's recipe changed here: mark it unsaved and redraw every
        window showing it."""
        self.qb.mark_dirty(self.pid)
        self.qb.job_changed(self.pid)

    # ------------------------------------------------------------ terminals
    def results_by_id(self):
        res = self.qb.station.last.get(self.pid)
        return {t["tool_id"]: t for t in (res or {}).get("tools", [])}

    def can_link(self, dst, src):
        """dst: ("in", tool, name, type); src: ["out", tool, name, type]."""
        return (src[1], src[2]) in T.compatible_sources(self._tools(), dst[1], dst[2],
                                                        self.results_by_id())

    def link_terminal(self, dst_id, input_name, src_id, output_name):
        why = T.link(self._tools(), dst_id, input_name, src_id, output_name)
        if why:
            self.statusBar().showMessage("Not linked: " + why, 8000)
            return why
        self._edited()
        self.statusBar().showMessage("Linked %s.%s -> %s.%s" % (src_id, output_name,
                                                                dst_id, input_name), 4000)
        return None

    def unlink_terminal(self, dst_id, input_name):
        T.unlink(self._tools(), dst_id, input_name)
        self._edited()

    def member_browser(self, tid):
        dlg = MemberBrowser(self, tid)
        dlg.show()
        self._browser = dlg
        return dlg

    # ------------------------------------------------------------ editing
    def _tools(self):
        return self.project().spec.setdefault("tools", [])

    def add_tool(self, ty):
        p = self.project()
        tools = self._tools()
        n = 1
        while any(t["id"] == "%s_%d" % (ty, n) for t in tools):
            n += 1
        params = defaults(ty)
        img = p.reference
        if img is not None and ty not in ("distance", "image_process"):
            h, w = img.shape[:2]
            if ty == "align":
                params["roi"] = [int(w * .1), int(h * .1), int(w * .8), int(h * .8)]
            else:
                params["roi"] = [int(w * .375), int(h * .375), int(w * .25), int(h * .25)]
        tid = "%s_%d" % (ty, n)
        spec = {"id": tid, "type": ty, "params": params}
        pending = self._take_pending_area()
        if pending is not None and ty not in ("distance", "image_process"):
            apply_drawn_region(spec, *pending)       # the area drawn before adding
        tools.append(spec)
        taught = self._teach(tid, quiet=True)
        self._edited()
        # The new tool is the one being worked on: selected, with its region
        # on the display. It was added at the bottom of the tree unselected,
        # and in a long job off screen, so nothing visibly happened.
        self.select_tool(tid)
        if taught:
            self.statusBar().showMessage("Added %s - it expects '%s', read from its box "
                                         "(change it under Settings)" % (
                                             self._name_of(tid), taught), 10000)
        elif self.simple:
            self.statusBar().showMessage(
                "Added %s - drag the blue box over the feature, then say what to expect "
                "under Settings" % simple.display_name(tools[-1]), 8000)
        else:
            self.statusBar().showMessage("Added %s - double-click it to set it up" % tid, 5000)
        return tid

    def delete_tool(self, tid):
        tools = self._tools()
        tools[:] = [t for t in tools if t["id"] != tid]
        self._edited()
        self.statusBar().showMessage("Deleted %s - Ctrl+Z brings it back" % tid, 6000)

    def move_tool(self, tid, step):
        tools = self._tools()
        i = next((k for k, t in enumerate(tools) if t["id"] == tid), None)
        j = None if i is None else i + step
        if i is None or not (0 <= j < len(tools)):
            return
        tools[i], tools[j] = tools[j], tools[i]
        self._edited()

    def toggle_tool(self, tid):
        t = find_tool(self._tools(), tid)
        if t:
            t["params"]["enabled"] = not t["params"].get("enabled", True)
            self._edited()

    def rename_tool(self, tid, new=None):
        if new is None:
            new, ok = QInputDialog.getText(self, "Rename tool", "New name:", text=tid)
            if not ok:
                return
        new = new.strip().replace(" ", "_")
        tools = self._tools()
        if new == tid:
            return
        # Refused with the reason: it used to do nothing at all, which looked
        # like the rename had been ignored.
        why = ("A tool needs a name." if not new else
               "Use letters, digits, '_' and '-' only - a '.' or '/' would break "
               "the links that read this tool's results." if any(c in new for c in "./\\")
               else "There is already a tool called %s." % new
               if any(t["id"] == new for t in tools) else None)
        if why:
            self.statusBar().showMessage("Not renamed: " + why, 8000)
            return
        for t in tools:
            if t["id"] == tid:
                t["id"] = new
        T.rename_references(tools, tid, new)       # keep every link pointing at it
        ed = self.qb._tool_editors.pop((self.pid, tid), None)
        if ed is not None:
            ed.close()
        self._edited()

    def image_source(self, paths=None):
        """The Image Source dialog; with paths, set image files directly."""
        # A menu action or button passes its checked state (False) as the
        # first argument; that is not a list of files.
        if isinstance(paths, (list, tuple)):
            self.qb.station.sources[self.pid].use_files(paths)
            self.refresh()
            return None
        dlg = ImageSourceDialog(self)
        dlg.show()
        self._source_dialog = dlg
        return dlg

    def toggle_continuous(self):
        if self.aCont.isChecked():
            self.qb.station.start_continuous([self.pid], gate=False)
        else:
            self.qb.station.stop()

    def reset_job(self):
        self.qb.station.last.pop(self.pid, None)
        self.qb.job_changed(self.pid)

    def closeEvent(self, e):
        _save_split(self.qb, self.split, "jobEditor/split2")
        _save_split(self.qb, self.side, "jobEditor/side")
        if self.aCont.isChecked():
            self.qb.station.stop()
        self.qb._job_editors.pop(self.pid, None)
        super().closeEvent(e)
        _close_frame(self)



class PostedItemsDialog(QDialog):
    """VisionPro's Configure Posted Items: choose result values to show as
    columns in the results queue, for every run."""

    def __init__(self, qb):
        super().__init__(qb)
        self.qb = qb
        self.setWindowTitle("Configure Posted Items")
        self.resize(520, 480)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Result", "Last value"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        chosen = set(qb.station.posted)
        for j in qb.station.jobs:
            ji = QTreeWidgetItem([j["name"], ""])
            self.tree.addTopLevelItem(ji)
            res = qb.station.last.get(j["pid"])
            tools = qb.station.projects[j["pid"]].spec.get("tools", [])
            last = {t["tool_id"]: t for t in (res or {}).get("tools", [])}
            for t in tools:
                r = last.get(t["id"])
                keys = ["score"] + [k for k, v in ((r or {}).get("values") or {}).items()
                                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
                if not r:
                    continue
                ti = QTreeWidgetItem([t["id"], ""])
                ji.addChild(ti)
                for k in keys:
                    item = "%s/%s.%s" % (j["pid"], t["id"], k)
                    v = r["score"] if k == "score" else r["values"][k]
                    leaf = QTreeWidgetItem([k, "%.4g" % v if isinstance(v, float) else str(v)])
                    leaf.setData(0, Qt.UserRole, item)
                    leaf.setFlags(leaf.flags() | Qt.ItemIsUserCheckable)
                    leaf.setCheckState(0, Qt.Checked if item in chosen else Qt.Unchecked)
                    ti.addChild(leaf)
                # open where something is already chosen, so it can be seen
                ti.setExpanded(any(ti.child(k).checkState(0) == Qt.Checked
                                   for k in range(ti.childCount())))
            ji.setExpanded(True)
        hint = QLabel("Tick the values to show as columns in the results queue. They "
                      "come from the last run - run the application once first if a "
                      "tool is missing.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.tree, 1)
        lay.addWidget(hint)
        ok_cancel(self, lay, self.apply)

    def leaves(self):
        out, stack = [], [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            it = stack.pop()
            if it.data(0, Qt.UserRole):
                out.append(it)
            stack.extend(it.child(k) for k in range(it.childCount()))
        return out

    def apply(self):
        items = sorted(it.data(0, Qt.UserRole) for it in self.leaves()
                       if it.checkState(0) == Qt.Checked)
        self.qb.station.set_posted(items)
        self.qb._refresh_queue()
        self.accept()
        return items



class HistoryView(QWidget):
    """Production history from the results logs: yield per job, the checks
    rejecting parts most often, and the latest runs -- with CSV export."""

    def __init__(self, qb):
        super().__init__()
        self.qb = qb
        self.setWindowTitle("Production History")
        from PySide6.QtWidgets import QComboBox
        top = QHBoxLayout()
        top.addWidget(QLabel("Period:"))
        self.period = QComboBox()
        for label, hours in (("Last hour", 1), ("Last 8 hours", 8), ("Last 24 hours", 24),
                             ("Last 7 days", 168), ("Everything", 0)):
            self.period.addItem(label, hours)
        self.period.setCurrentIndex(2)
        self.period.currentIndexChanged.connect(lambda _i: self.refresh())
        top.addWidget(self.period)
        top.addStretch(1)
        ref = QPushButton("Refresh")
        ref.setIcon(icons.icon("reset"))
        ref.clicked.connect(self.refresh)
        exp = QPushButton("Export CSV...")
        exp.setIcon(icons.icon("export"))
        exp.clicked.connect(lambda: self._export_clicked())
        top.addWidget(ref)
        top.addWidget(exp)
        noun = "Inspection" if getattr(qb, "simple", False) else "Job"
        self.jobs = EmptyStateTable(0, 5, "No parts inspected in this period.")
        self.jobs.setHorizontalHeaderLabels([noun, "Inspected", "Accepted", "Yield", "Avg cycle"])
        self.pareto = EmptyStateTable(0, 3, "No rejected parts in this period.")
        self.pareto.setHorizontalHeaderLabels([noun, "Check", "Rejected parts"])
        for t in (self.jobs, self.pareto):
            t.horizontalHeader().setStretchLastSection(True)
            t.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            t.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            t.verticalHeader().setVisible(False)
            t.setShowGrid(False)
            t.setAlternatingRowColors(True)
            t.setSelectionBehavior(QAbstractItemView.SelectRows)
            t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # a check's name in full: it was cut to "corner_scre..."
        self.pareto.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(8)
        head = QLabel("Production history")
        head.setObjectName("pageTitle")
        lay.addWidget(head)
        lay.addLayout(top)
        t1 = QLabel("Yield by %s" % noun.lower())
        t1.setObjectName("sectionTitle")
        lay.addWidget(t1)
        lay.addWidget(self.jobs, 1)
        t2 = QLabel("Checks rejecting parts, most first")
        t2.setObjectName("sectionTitle")
        lay.addWidget(t2)
        lay.addWidget(self.pareto, 1)
        self.refresh()

    def hours(self):
        return self.period.currentData() or None

    def refresh(self):
        st = self.qb.station
        for lg in st.logs.values():
            lg.flush()
        rows, par = [], []
        for j in st.jobs:
            lg = st.logs.get(j["pid"])
            if lg is None:
                continue
            sm = lg.summary(self.hours())
            rows.append((j["name"], sm["total"], sm["passed"],
                         ("%.1f%%" % sm["yield_pct"]) if sm["total"] else "-",
                         fmt_ms(sm["avg_ms"]) if sm["total"] else "-"))
            for item in lg.pareto(self.hours()):
                par.append((j["name"], self.qb.check_name(j["pid"], item["tool_id"])
                            if getattr(self.qb, "simple", False) else item["tool_id"],
                            item["count"]))
        par.sort(key=lambda r: -r[2])
        for table, data in ((self.jobs, rows), (self.pareto, par)):
            table.setRowCount(len(data))
            for i, row in enumerate(data):
                for k, v in enumerate(row):
                    table.setItem(i, k, QTableWidgetItem(str(v)))
        return rows, par

    def _export_clicked(self):
        out = self.export_csv()
        if out:
            self.qb.statusBar().showMessage("Exported %d file(s) to %s" % (
                len(out), os.path.dirname(out[0])), 8000)

    def export_csv(self, folder=None):
        """One CSV per job, for the period shown."""
        folder = folder or QFileDialog.getExistingDirectory(self, "Export history to folder")
        if not folder:
            return []
        os.makedirs(folder, exist_ok=True)       # a new folder is ordinary, not an error
        out = []
        for j in self.qb.station.jobs:
            lg = self.qb.station.logs.get(j["pid"])
            if lg is None:
                continue
            lg.flush()
            name = "".join(c if c.isalnum() or c in "-_" else "_" for c in j["name"])
            path = os.path.join(folder, "%s_history.csv" % name)
            lg.export_csv(path, self.hours())
            out.append(path)
        return out



class CommunicationDialog(QDialog):
    """The TCP/IP line interface, as VisionPro's Communication Explorer sets up
    a job's connection to the line."""

    def __init__(self, parent, c):
        super().__init__(parent)
        from PySide6.QtWidgets import QCheckBox, QSpinBox
        self.setWindowTitle("Communication")
        self.resize(560, 360)
        self.enabled = QCheckBox("Listen for the line on TCP/IP")
        self.enabled.setChecked(bool(c.get("enabled")))
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(int(c.get("port", 3000)))
        self.broadcast = QCheckBox("Send every run's results to all connected clients")
        self.broadcast.setChecked(bool(c.get("broadcast")))
        self.port.setFixedWidth(110)         # five digits, not the dialog's width
        cmds = QLabel(
            "TRIGGER  or  TRIGGER &lt;job&gt;<br>"
            "&nbsp;&nbsp;&rarr; RESULT &lt;job&gt; &lt;PASS|FAIL|ERROR&gt; &lt;ms&gt; &lt;failed tools&gt;"
            "<br>&nbsp;&nbsp;&rarr; DONE &lt;verdict&gt;<br>"
            "GET &lt;job&gt;.&lt;tool&gt;.&lt;value&gt;<br>"
            "STATUS &nbsp; PING &nbsp; VERSION")
        cmds.setObjectName("mono")
        cmds.setTextInteractionFlags(Qt.TextSelectableByMouse)
        note = QLabel("One command per line. GET reads a value from the last run; a "
                      "trigger that arrives during an inspection is answered BUSY.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        from PySide6.QtWidgets import QGroupBox
        proto = QGroupBox("Commands the line can send")
        pv = QVBoxLayout(proto)
        pv.addWidget(cmds)
        pv.addWidget(note)
        self.settingsBox = QWidget()
        form = QFormLayout(self.settingsBox)
        form.setContentsMargins(26, 0, 0, 0)
        form.addRow("Port:", self.port)
        form.addRow(self.broadcast)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 14)
        lay.addWidget(self.enabled)
        lay.addWidget(self.settingsBox)
        lay.addSpacing(6)
        lay.addWidget(proto)
        lay.addStretch(1)
        ok_cancel(self, lay)
        self.enabled.toggled.connect(self.settingsBox.setEnabled)
        self.settingsBox.setEnabled(self.enabled.isChecked())
        guard_wheel(self)

    def values(self):
        return {"enabled": self.enabled.isChecked(), "port": self.port.value(),
                "broadcast": self.broadcast.isChecked()}


# ================================================================ QuickBuild pieces
class KpiCard(QFrame):
    """One production number on the command bar: a caption and a value."""

    def __init__(self, caption, tip=""):
        super().__init__()
        self.setObjectName("kpi")
        self.setToolTip(tip)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 3, 14, 4)
        lay.setSpacing(0)
        cap = QLabel(caption)
        cap.setObjectName("caption")
        self.value = QLabel("-")
        self.value.setObjectName("kpiValue")
        lay.addWidget(cap)
        lay.addWidget(self.value)
        self.setMinimumWidth(92)

    def set(self, text, tone=None):
        self.value.setText(text)
        theme.set_tone(self, tone or "idle")


class JobTile(QFrame):
    """A job in the Run View: its name, its verdict and the last run's image.
    The border takes the verdict's colour, so a reject is seen across the
    room, not read."""

    def __init__(self, name, on_open):
        super().__init__()
        self.setObjectName("tile")
        self.setProperty("tone", "idle")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(1, 1, 1, 1)
        lay.setSpacing(0)
        head = QWidget()
        head.setObjectName("tileHead")
        h = QHBoxLayout(head)
        h.setContentsMargins(12, 7, 8, 7)
        h.setSpacing(8)
        self.name = QLabel(name)
        self.name.setObjectName("tileName")
        self.status = QLabel("Not run yet")
        self.status.setObjectName("secondary")
        self.status.setMinimumWidth(10)
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.badge = QLabel("IDLE")
        self.badge.setProperty("badge", True)
        self.badge.setProperty("tone", "idle")
        edit = QToolButton()
        self.edit = edit
        edit.setIcon(icons.icon("build"))
        edit.setProperty("vsIcon", "build")
        # said in words: a bare spanner in the corner was easy to miss, and it
        # is the way from watching a job to changing it
        edit.setText("Edit")
        edit.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        edit.setToolTip("Open this job in the Job Editor (or double-click the image)")
        edit.setAutoRaise(True)
        edit.clicked.connect(on_open)
        h.addWidget(self.name)
        h.addWidget(self.status, 1)
        h.addWidget(self.badge)
        h.addWidget(edit)
        self.display = VisionDisplay(records=False, toolbar=False)
        self.display.statusRow.hide()        # pixel read-outs belong in the editors
        lay.addWidget(head)
        lay.addWidget(self.display, 1)

    def show_verdict(self, tone, badge, status):
        self.badge.setText(badge)
        self.status.setText(status)
        self.status.setToolTip(status)
        theme.set_tone(self.badge, tone)
        theme.set_tone(self, tone)


class _RunViewFrame(QMdiSubWindow):
    """The Run View's frame. It is the application's home, so it cannot be
    closed -- only brought to the front."""

    def closeEvent(self, e):
        e.ignore()


def _set_frame(sub, framed):
    """A document's frame on or off. The simple view shows one document at
    a time, filling the workspace with no title bar and no tab bar: Qt's
    tabbed mode keeps the tab bar's height even when the bar is hidden, so
    the simple view uses sub-window mode with frameless, maximized windows."""
    if sub.property("vsFramed") == framed:
        return
    if sub.property("vsFlags") is None:
        sub.setProperty("vsFlags", int(sub.windowFlags().value))
    flags = Qt.WindowType(sub.property("vsFlags")) if framed else \
        (Qt.SubWindow | Qt.FramelessWindowHint)
    was = sub.isVisible()
    sub.setWindowFlags(flags)
    sub.setProperty("vsFramed", framed)
    if was:
        sub.show()


def _tool_search(tree):
    """A search box over the Toolbox: forty-odd tools in nine groups do not
    fit on a screen, and finding one by scrolling was the slowest part of
    building a job."""
    box = QLineEdit()
    box.setObjectName("searchBox")
    box.setPlaceholderText("Search tools")
    box.setClearButtonEnabled(True)

    def items():
        for i in range(tree.topLevelItemCount()):
            g = tree.topLevelItem(i)
            for k in range(g.childCount()):
                yield g, g.child(k)

    def filt(text):
        """A tool's name or description matching wins; a group's name only
        counts when no tool matches -- 'count' shows the counting tools, not
        everything in 'Presence and count'."""
        text = text.strip().lower()
        named = any(text in it.text(0).lower() or text in (it.toolTip(0) or "").lower()
                    for _g, it in items()) if text else False
        shown = {}
        for g, it in items():
            hit = (not text or text in it.text(0).lower()
                   or text in (it.toolTip(0) or "").lower()
                   or (not named and text in g.text(0).lower()))
            it.setHidden(not hit)
            shown[id(g)] = shown.get(id(g), 0) + hit
        for i in range(tree.topLevelItemCount()):
            g = tree.topLevelItem(i)
            g.setHidden(shown.get(id(g), 0) == 0)
            if text:
                g.setExpanded(True)
    box.textChanged.connect(filt)

    def add_first():
        """Enter adds the best visible match: a name match before a
        description match."""
        text = box.text().strip().lower()
        visible = [it for _g, it in items() if not it.isHidden()]
        best = next((it for it in visible if text and text in it.text(0).lower()),
                    visible[0] if visible else None)
        if best is not None:
            tree.add_item(best)
    box.returnPressed.connect(add_first)
    return box


class DockTitle(QWidget):
    """A panel's title bar: its name and a close button. Qt's own dock
    buttons ignored the stylesheet's icon size and drew as specks; and a
    panel torn off by accident to float over the image was half of what made
    the old layout a mess, so panels dock but do not float. View brings a
    closed panel back."""

    def __init__(self, dock):
        super().__init__(dock)
        self.setObjectName("dockTitle")
        h = QHBoxLayout(self)
        h.setContentsMargins(12, 6, 4, 2)
        lab = QLabel(dock.windowTitle())
        lab.setObjectName("dockCaption")
        dock.windowTitleChanged.connect(lab.setText)
        btn = QToolButton()
        btn.setObjectName("dockClose")
        btn.setAutoRaise(True)
        btn.setIcon(icons.icon("close"))
        btn.setProperty("vsIcon", "close")
        btn.setIconSize(QSize(14, 14))
        btn.setToolTip("Hide this panel - View brings it back")
        btn.setAccessibleName("Hide %s" % dock.windowTitle())
        btn.clicked.connect(dock.close)
        h.addWidget(lab, 1)
        h.addWidget(btn)
        self.close_button = btn
        # shown when the pointer is over the panel's title, as a Mac panel's
        # controls are: a row of permanent x's was clutter at rest
        btn.setVisible(False)

    def enterEvent(self, e):
        self.close_button.setVisible(True)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.close_button.setVisible(False)
        super().leaveEvent(e)


def _card(widget):
    """A dock's contents as a floating rounded card on the page, with a
    little of the page round it -- the panels of Vision Glass. Only how it
    is framed changes; the widget inside is the same one, doing the same."""
    well = QWidget()
    well.setObjectName("dockWell")
    outer = QVBoxLayout(well)
    outer.setContentsMargins(4, 4, 4, 4)
    card = QFrame()
    card.setObjectName("dockCard")
    inner = QVBoxLayout(card)
    inner.setContentsMargins(6, 6, 6, 6)
    inner.addWidget(widget)
    outer.addWidget(card)
    well.content = widget
    return well


def _dock(title, name, owner, widget):
    d = _unique_name(QDockWidget(title, owner), name)
    d.setWidget(_card(widget))
    d.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)
    d.setTitleBarWidget(DockTitle(d))
    return d


def _panel(*widgets, margins=(0, 0, 0, 0), spacing=0):
    w = QWidget()
    w.setObjectName("panelBody")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    for x in widgets:
        if isinstance(x, tuple):
            lay.addWidget(*x)
        else:
            lay.addWidget(x)
    return w


# ================================================================ QuickBuild
class _ScanKeys(QObject):
    """A USB scanner types its code as keys into whatever has the focus.
    While the Run view is in front and no text box has the focus, those keys
    go to the barcode bar -- a scan is never lost because someone clicked
    the picture or the results queue last."""

    def __init__(self, qb):
        super().__init__(qb)
        self.qb = qb

    def eventFilter(self, obj, e):
        if e.type() != QEvent.KeyPress:
            return False
        qb = self.qb
        bar = getattr(qb, "barcodeBar", None)
        if bar is None or not bar.isVisible():
            return False
        # keys meant for this window -- not a dialog or another window
        win = obj.window() if isinstance(obj, QWidget) else None
        if win is not qb:
            return False
        from PySide6.QtWidgets import QAbstractSpinBox, QLineEdit, QPlainTextEdit, QTextEdit
        # the widget the key went to: a text box keeps its own typing
        if obj is bar.field or isinstance(obj, (QLineEdit, QTextEdit, QPlainTextEdit,
                                                QAbstractSpinBox)):
            return False
        if isinstance(obj, QWidget) and isinstance(obj.parentWidget(), QAbstractSpinBox):
            return False
        if qb.mdi.activeSubWindow() not in (None, qb.runSub):
            return False
        text = e.text()
        if e.modifiers() & (Qt.ControlModifier | Qt.AltModifier):
            return False
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            if bar.field.text().strip():
                bar.field.setFocus()
                bar._enter()
                return True
            return False
        if text and text.isprintable():
            bar.take_keys(text)
            return True
        return False


class QuickBuild(QMainWindow):
    """The application: its jobs, a run display per job, and the queues."""
    # where the window's size and panel layout are kept
    LAYOUT_KEYS = ("geometry", "state")

    def __init__(self, root=None):
        global camera_settings_store
        camera_settings_store = self._settings
        super().__init__()
        _unique_name(self, "quickBuild")
        self.root = root or os.environ.get("VS_PROJECTS") or os.path.join(HERE, "projects")
        self.station = Station(self.root)
        try:
            from vision_station.ui.settings_dialog import load_result_settings
            self.station.result_images.configure(load_result_settings(self._settings()))
        except Exception as e:
            print("[settings] result saving not loaded: %s" % e)
        self.station.jobResult.connect(self._job_result)
        self.station.barcodeNeeded.connect(self._barcode_needed)
        self.station.runFinished.connect(self._run_finished)
        self.station.runError.connect(self._job_error)
        self.station.warmedUp.connect(self._warmed_up)
        # while a model is being taught, nothing else runs (the training
        # window is modal; this stops a continuous run's timer too)
        self.station.before_run = lambda pids: not self._adv_gating
        self._adv_postponed = set()          # cancelled or impossible: not again this session
        self._adv_gating = False
        self.dirty = set()
        self._job_editors = {}
        self._tool_editors = {}
        self._displays = {}
        self._tiles = {}
        self._job_errors = {}
        self._closing = False
        # The simple view is the default; Advanced mode brings the menu bar,
        # the Toolbox, the Layers terminals and the document tabs back.
        self.advanced, self.simple = False, True
        self.resize(1400, 880)
        self.setMinimumSize(980, 620)
        self.setWindowIcon(icons.app_icon())
        self.setWindowTitle("Vision Station")
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowTabbedDocks)

        self._build_command_bar()
        self._build_docks()
        self._build_workspace()
        self._build_status_bar()
        self._build_menus()
        self._build_simple()
        self._load_barcode()
        self._load_light()

        self._default_state = self.saveState(LAYOUT_VERSION)
        self._restore_layout()
        self.set_advanced(self._pref_advanced(), save=False)
        apps = self.station.applications()
        if apps:
            sizes = {a["id"]: sum(1 for x in self.station.store.list()
                                  if (x.get("job") or x["id"]) == a["id"]) for a in apps}
            self.open_application(max(apps, key=lambda a: sizes[a["id"]])["id"])
        else:
            self._build_displays()

    # ------------------------------------------------------------ construction
    def _build_command_bar(self):
        bar = _unique_name(QToolBar("Application"), "qbToolbar")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        bar.setIconSize(QSize(18, 18))
        bar.toggleViewAction().setVisible(False)
        self.addToolBar(bar)
        # One Run button for the whole application, as a Mac app has one
        # primary action: it runs the job whose editor is in front, or every
        # job from the Run View. The editors had their own Run buttons a
        # hundred pixels below this one, and nobody could tell which to press.
        self.aRunOnce = make_action(bar, "Run", self.key_run_once, "F5",
                                    "Inspect once - the job in front, or every job",
                                    register_key=False, icon='run')
        self.aRunCont = make_action(bar, "Continuous", self.toggle_continuous, "F6",
                                    "Inspect every part as it comes, until pressed again",
                                    register_key=False, check=True, icon='run_continuous')
        bar.widgetForAction(self.aRunOnce).setObjectName("accentButton")
        bar.widgetForAction(self.aRunCont).setObjectName("runningButton")
        # Run's dropdown: Normal run (the default) or Model teaching run --
        # the job's own AI model trained (core/advanced.py). Teaching is
        # offered only with Settings > Use advanced model for training on,
        # and Run goes back to normal after it: an operator never trains by
        # pressing Run.
        self.run_mode = "normal"
        runMenu = QMenu(self)
        runMenu.setObjectName("runModeMenu")
        grp = QActionGroup(self)
        grp.setExclusive(True)
        self.aRunNormal = runMenu.addAction("Normal run")
        self.aRunTeach = runMenu.addAction("Model teaching run")
        for a in (self.aRunNormal, self.aRunTeach):
            a.setCheckable(True)
            grp.addAction(a)
        self.aRunNormal.setChecked(True)
        self.aRunNormal.triggered.connect(lambda: self.set_run_mode("normal"))
        self.aRunTeach.triggered.connect(lambda: self.set_run_mode("teach"))
        runMenu.aboutToShow.connect(self._refresh_run_menu)
        rb = bar.widgetForAction(self.aRunOnce)
        rb.setMenu(runMenu)
        rb.setPopupMode(QToolButton.MenuButtonPopup)
        self._refresh_run_menu()
        self._accent_run_icon()
        bar.addSeparator()
        # secondary commands as icons, named in their tooltips and to screen
        # readers: ten labelled buttons left nothing standing out
        self._secondary = [
            make_action(bar, "Open Application...", self.choose_application, "Ctrl+O",
                        "Open another application", icon='open'),
            make_action(bar, "Save All", self.save_all, "Ctrl+S", "Save every changed job",
                        register_key=False, icon='save'),
            make_action(bar, "Reset Counts", self.reset_counts, None,
                        "Zero the counts and clear the results queue", icon='counts')]
        # Live Camera stays in both views: aiming and focusing is everyone's job
        self.aLive = make_action(bar, "Live Camera", lambda: self.show_live_camera(), None,
                                 "Live Camera - the camera's picture for aiming, positioning "
                                 "the part, focus and exposure. Nothing is inspected.",
                                 register_key=False, icon='image_source')
        self.aBarcode = make_action(bar, "Barcode", lambda: self.barcode_settings(), None,
                                    "Barcode check - inspect a part only after its barcode "
                                    "is scanned, and keep its pictures in a folder named "
                                    "after it", register_key=False, check=True,
                                    icon='barcode')
        self.aFind = make_action(bar, "Find picture", lambda: self.find_picture(), "Ctrl+F",
                                 "Find picture - type or scan a barcode and see its "
                                 "pictures; double-click one to open its folder",
                                 icon='find_picture')
        for a in self._secondary + [self.aLive, self.aBarcode, self.aFind]:
            b = bar.widgetForAction(a)
            b.setToolButtonStyle(Qt.ToolButtonIconOnly)
            b.setObjectName("toolbarIcon")
            b.setAccessibleName(a.text().rstrip("."))
        for key, fn in (("F5", self.key_run_once), ("F6", self.key_run_continuous),
                        ("Ctrl+S", self.key_save), ("Ctrl+Z", lambda: self.undo()),
                        ("Ctrl+Y", lambda: self.redo()), ("Ctrl+Shift+Z", lambda: self.redo())):
            a = make_action(self, key, fn, key)
            self.addAction(a)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        self.kpi = {}
        for key, caption, tip in (("accept", "ACCEPTED", "Parts accepted since the counts were reset"),
                                  ("reject", "REJECTED", "Parts rejected since the counts were reset"),
                                  ("yield", "YIELD", "Accepted as a share of all inspected"),
                                  ("cycle", "LAST CYCLE", "Time the last inspection took")):
            card = KpiCard(caption.title(), tip)
            self.kpi[key] = card
            bar.addWidget(card)
            if key != "cycle":
                gap = QWidget()
                gap.setFixedWidth(14)
                bar.addWidget(gap)
        bar.addSeparator()
        # every setting, in one window (settings_dialog.py) -- as a Mac app's
        # Settings: the camera, result pictures, the line, appearance, security
        self.aSettings = make_action(bar, "Settings", lambda: self.show_settings(), "Ctrl+,",
                                     "Settings - camera, result pictures, line connection, "
                                     "appearance and security", icon='gear')
        sb = bar.widgetForAction(self.aSettings)
        sb.setToolButtonStyle(Qt.ToolButtonIconOnly)
        sb.setObjectName("toolbarIcon")
        sb.setAccessibleName("Settings")
        self.commandBar = bar
        self._compact = None

    def _accent_run_icon(self):
        """Run Once's icon in the same ink as its label. It was always white,
        so on the dark theme's light-blue button the label was black and the
        icon beside it white."""
        ink = theme.color("on_accent")
        self.aRunOnce.setIcon(icons.icon("run", ink=ink, accent=ink))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit_command_bar()

    def _fit_command_bar(self):
        """On a narrow window the secondary commands show their icons only,
        so the production numbers stay on screen instead of disappearing
        behind the toolbar's overflow arrow."""
        compact = self.width() < 1000
        if compact == self._compact:
            return
        self._compact = compact
        b = self.commandBar.widgetForAction(self.aRunCont)
        b.setToolButtonStyle(Qt.ToolButtonIconOnly if compact else Qt.ToolButtonTextBesideIcon)

    def _build_docks(self):
        self.jobs = QTreeWidget()
        self.jobs.setHeaderLabels(["Job", "Result", "Cycle"])
        self.jobs.setRootIsDecorated(False)
        self.jobs.setUniformRowHeights(True)
        hdr = self.jobs.header()
        # The job's name takes the room; it was cut to "Network chas..." because
        # the last column stretched instead.
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.jobs.setMinimumWidth(220)
        self.jobs.setMinimumHeight(118)          # three jobs visible however small the window
        self.jobs.itemDoubleClicked.connect(
            lambda it, _c: self.open_job_editor(it.data(0, Qt.UserRole)))
        self.jobs.setContextMenuPolicy(Qt.CustomContextMenu)
        self.jobs.customContextMenuRequested.connect(self._jobs_menu)
        self.jobs.setToolTip("Double-click a job to edit it; right-click for more")
        # A sidebar, as a Mac app's source list: plain rows under quiet section
        # headings, no table header -- one job does not need a three-column
        # grid with "Job / Result / Cycle" above it.
        self.jobs.setHeaderHidden(True)
        self.jobs.setObjectName("sidebarList")
        self.views = QListWidget()
        self.views.setObjectName("sidebarList")
        for text, ic, fn in (("Run View", "job", self.show_run_view),
                             ("Production History", "history", self.show_history)):
            it = QListWidgetItem(icons.icon(ic), text)
            it.setData(Qt.UserRole, fn)
            it.setData(Qt.UserRole + 1, ic)
            self.views.addItem(it)
        self.views.itemClicked.connect(lambda it: it.data(Qt.UserRole)())
        self.views.setFixedHeight(2 * 34 + 6)
        self.views.setFrameShape(QFrame.NoFrame)
        self.jobs.setFrameShape(QFrame.NoFrame)

        def section(text):
            lab = QLabel(text)
            lab.setObjectName("sidebarSection")
            return lab
        # navigation first, then the jobs, stacked from the top
        self.jobs.setColumnHidden(2, True)        # the cycle time is in the row's tooltip
        self._secViews, self._secJobs = section("Views"), section("Jobs")
        self.jobsDock = _dock("Sidebar", "jobsDock", self,
                              _panel(self._secViews, self.views, self._secJobs,
                                     (self.jobs, 1), margins=(8, 6, 8, 8), spacing=2))
        self.jobsDock.setTitleBarWidget(QWidget())      # the sections are the titles
        self.addDockWidget(Qt.LeftDockWidgetArea, self.jobsDock)
        # One Toolbox for the application, as VisionPro has: it adds to the
        # active job. A Toolbox inside every Job Editor squeezed the tool tree
        # to a sliver once the editors shared the workspace.
        self.toolbox = toolbox_tree(lambda ty: self._toolbox_add(ty, draw=True))
        self.toolbox.setToolTip("")
        self.toolSearch = _tool_search(self.toolbox)
        hint = QLabel("Double-click a tool to add it to the job in front")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        self.toolboxDock = _dock("Toolbox", "toolboxDock", self,
                                 _panel(self.toolSearch, hint, (self.toolbox, 1),
                                        margins=(8, 8, 4, 4), spacing=6))
        self.addDockWidget(Qt.LeftDockWidgetArea, self.toolboxDock)
        # Hidden until asked for (View > Toolbox): the Job Editor's Add Tool
        # menu adds tools where they are wanted, and a second panel in the
        # sidebar took a small screen's width away from the image.
        self.splitDockWidget(self.jobsDock, self.toolboxDock, Qt.Vertical)
        self.toolboxDock.hide()
        self.resizeDocks([self.jobsDock], [240], Qt.Horizontal)

        self.queue = EmptyStateTable(0, 5, "No inspections yet - press Run Once (F5).")
        self.queue.setHorizontalHeaderLabels(["Time", "Job", "Result", "Cycle", "Failed tools"])
        qh = self.queue.horizontalHeader()
        qh.setStretchLastSection(True)
        qh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.queue.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.queue.setSelectionMode(QAbstractItemView.SingleSelection)
        self.queue.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.queue.setShowGrid(False)
        self.queue.setAlternatingRowColors(True)
        self.queue.setWordWrap(False)
        self.queue.itemSelectionChanged.connect(self._queue_selected)
        self.queue.verticalHeader().setVisible(False)
        self.queue.verticalHeader().setDefaultSectionSize(26)
        self.queue.setColumnWidth(0, 80)
        self.queue.setColumnWidth(1, 180)
        self.queue.setColumnWidth(2, 96)          # the dot and "Accept" / "Reject" in full
        self.queue.setColumnWidth(3, 80)
        # The record preview shows the image, not a toolbar over it: with one,
        # a 4000-pixel frame was shown at 4%, a smudge.
        self.queueDisplay = VisionDisplay(records=False, toolbar=False)
        self.queueDisplay.statusRow.hide()
        self.queueDisplay.set_placeholder("The selected run's image appears here")
        qw = QSplitter(Qt.Horizontal)
        qw.addWidget(_panel(self.queue, margins=(4, 4, 0, 4)))
        qw.addWidget(self.queueDisplay)
        qw.setStretchFactor(0, 3)
        qw.setStretchFactor(1, 2)
        qw.setSizes([720, 420])
        self.queueDock = _dock("Results queue", "queueDock", self, qw)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.queueDock)
        # the queue is for glancing at, not working in: a short strip by
        # default, so the editors above it get the height
        self.resizeDocks([self.queueDock], [150], Qt.Vertical)
        self.queueFailuresOnly = False

    def _build_workspace(self):
        self.grid = QWidget()
        self.grid.setObjectName("runGrid")
        self.grid.setAttribute(Qt.WA_StyledBackground, True)
        self.gridLayout = QGridLayout(self.grid)
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.gridLayout.setSpacing(10)
        # The run view, job editors, tool edit controls, floating displays and
        # results grids are documents of one workspace, as VisionPro's
        # windows are -- shown as tabs. As floating windows they opened larger
        # than the workspace, covered each other and squashed the Run View to
        # its title bar; Window > Tabbed documents switches back for those who
        # want them side by side.
        self.mdi = QMdiArea()
        self.mdi.setObjectName("workspace")
        self.mdi.setOption(QMdiArea.DontMaximizeSubWindowOnActivation, True)
        self.mdi.setActivationOrder(QMdiArea.ActivationHistoryOrder)
        self.mdi.setBackground(QBrush(QColor(theme.color("bg"))))
        self.setCentralWidget(self.mdi)
        # The Run View: the verdict in words above the jobs' images.
        self.banner = simple.VerdictBanner()
        self.runPage = QWidget()
        self.runPage.setObjectName("runPage")
        self.runPage.setAttribute(Qt.WA_StyledBackground, True)
        rp = QVBoxLayout(self.runPage)
        rp.setContentsMargins(10, 10, 10, 0)
        rp.setSpacing(0)
        rp.addWidget(self.banner)
        # barcode-gated inspection: what to scan, and what the scan did
        from vision_station.ui.barcode_panel import BarcodeBar, SerialScanner
        self.barcodeBar = BarcodeBar()
        self.barcodeBar.scanned.connect(self.barcode_scanned)
        self.barcodeBar.settingsRequested.connect(self.barcode_settings)
        self.barcodeBar.hide()
        rp.addSpacing(8)
        rp.addWidget(self.barcodeBar)
        self.scanner = SerialScanner(self)
        self.scanner.scanned.connect(self.barcode_scanned)
        self.scanner.error.connect(lambda m: self.statusBar().showMessage("Scanner: " + m, 10000))
        rp.addWidget(self.grid, 1)
        self.runSub = _RunViewFrame()
        self.runSub.setWidget(self.runPage)
        self.mdi.addSubWindow(self.runSub)
        self.runSub.setWindowTitle("Run View")
        self.runSub.setWindowIcon(icons.icon("job"))
        self.runSub.setWindowFlags(self.runSub.windowFlags() & ~Qt.WindowCloseButtonHint)
        tabbed = str(self._settings().value("tabbed", "true")).lower() not in ("false", "0")
        self._set_tabbed(tabbed, save=False)
        self.runSub.showMaximized()
        self.mdi.subWindowActivated.connect(self._sub_activated)

    def _build_status_bar(self):
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        self.sCounts = QLabel("")            # the counts in words, for screen readers
        self.sCounts.setVisible(False)
        self.sCycle = QLabel("")
        self.sCycle.setVisible(False)
        self.sEngine = QLabel("")
        self.sEngine.setToolTip("Text and model engines warm up in the background "
                                "so the first part is not the slow one")
        self.sComms = QLabel("")
        self.sComms.setToolTip("TCP/IP line interface - Run > Communication")
        sb.addWidget(self.sCounts)
        sb.addPermanentWidget(self.sComms)
        sb.addPermanentWidget(self.sEngine)
        self.line = None

    def _build_menus(self):
        m = self.menuBar()
        # Every menu is kept referenced. Their Python wrappers were left to
        # the garbage collector, which on PySide6 6.11 also invalidated the
        # wrappers of their actions: Operator Mode then crashed collecting the
        # actions to lock, and Undo's menu entry raised on every window switch.
        self._menus = []

        def menu(title):
            mm = m.addMenu(title)
            self._menus.append(mm)
            return mm
        f = menu("&File")
        self._build_menu_start = len(f.actions())
        f.addAction(icons.icon("add"), "New Application...", self.new_application)
        f.addAction(icons.icon("open"), "Open Application...", self.choose_application)
        f.addAction(icons.icon("save"), "Save All", self.save_all)
        f.addSeparator()
        f.addAction(icons.icon("python"), "Export Job File (Python)...", self.export_job_file)
        f.addAction(icons.icon("export"), "Export Package...", self.export_package)
        f.addAction(icons.icon("import"), "Import Package...", self.import_package)
        f.addAction(icons.icon("output"), "Result Output...", lambda: self.show_settings("results"))
        f.addAction(icons.icon("gear"), "Settings...", lambda: self.show_settings())
        f.addSeparator()
        f.addAction(icons.icon("job"), "Add Job...", self.add_job)
        f.addAction(icons.icon("grid_results"), "Configure Posted Items...", self.configure_posted)
        f.addSeparator()
        f.addAction("Classic editor", self.classic_editor)
        f.addSeparator()
        f.addAction("Exit", self.close)
        r = menu("&Run")
        r.addAction(icons.icon("run"), "Run Once   (F5)", self.run_once)
        r.addAction(icons.icon("run_continuous"), "Run Continuously   (F6)", lambda: (
            self.aRunCont.setChecked(not self.aRunCont.isChecked()), self.toggle_continuous()))
        r.addAction(icons.icon("stop"), "Stop", self.stop)
        self.aParallel = r.addAction("Run Jobs in Parallel")
        self.aParallel.setCheckable(True)
        self.aParallel.setToolTip("Overlap a cycle's jobs on a multi-core PC")
        self.aParallel.toggled.connect(self._set_parallel)
        r.addSeparator()
        r.addAction(icons.icon("history"), "Production History", self.show_history)
        r.addAction(icons.icon("find_picture"), "Find Picture...   (Ctrl+F)",
                    lambda: self.find_picture())
        r.addSeparator()
        self.aOperator = r.addAction(icons.icon("job"), "Operator Mode", self.toggle_operator_mode)
        self.aSupervisor = r.addAction("Set Supervisor Password...", self.set_supervisor_password)
        r.addSeparator()
        r.addAction(icons.icon("output"), "Communication...", lambda: self.show_settings("line"))
        self._file_menu = f
        em = menu("&Edit")
        self.aUndo = em.addAction(icons.icon("reset"), "Undo", lambda: self.undo())
        self.aRedo = em.addAction("Redo", lambda: self.redo())
        self.aUndo.setToolTip("Ctrl+Z - undo the last change to the job in front")
        self.aRedo.setToolTip("Ctrl+Y - redo")
        wm = menu("&Window")
        wm.addAction(icons.icon("job"), "Run View", self.show_run_view)
        wm.addAction("Next window", lambda: self.mdi.activateNextSubWindow())
        wm.addAction("Previous window", lambda: self.mdi.activatePreviousSubWindow())
        wm.addAction("Close all editors", self.close_editors)
        wm.addSeparator()
        self.aTabbed = wm.addAction("Tabbed documents")
        self.aTabbed.setCheckable(True)
        self.aTabbed.setChecked(self.mdi.viewMode() == self.mdi.ViewMode.TabbedView)
        self.aTabbed.toggled.connect(self._set_tabbed)
        self.aCascade = wm.addAction(icons.icon("cascade"), "Cascade",
                                     lambda: self.mdi.cascadeSubWindows())
        self.aTile = wm.addAction(icons.icon("tile"), "Tile", lambda: self.mdi.tileSubWindows())
        self._sync_window_actions()
        v = menu("&View")
        v.addAction("Results queue: all runs", lambda: self._set_queue_mode(False))
        v.addAction("Results queue: failures only", lambda: self._set_queue_mode(True))
        v.addSeparator()
        for d in (self.jobsDock, self.toolboxDock, self.queueDock):
            v.addAction(d.toggleViewAction())
        v.addSeparator()
        tm = v.addMenu("Theme")
        self._menus.append(tm)
        grp = QActionGroup(self)
        current = str(self._settings().value("theme", "system") or "system")
        self._theme_actions = {}
        for key, text in (("system", "Use Windows setting"), ("light", "Light"), ("dark", "Dark")):
            a = tm.addAction(text, lambda k=key: self.set_theme(k))
            a.setCheckable(True)
            a.setChecked(key == current)
            grp.addAction(a)
            self._theme_actions[key] = a
        self._theme_group = grp
        v.addSeparator()
        self.aAdvanced = v.addAction("Advanced mode")
        self.aAdvanced.setCheckable(True)
        self.aAdvanced.setToolTip("The engineering view: menu bar, Toolbox, every tool and "
                                  "setting, terminals and document tabs")
        self.aAdvanced.toggled.connect(lambda on: self.set_advanced(on))
        v.addAction(icons.icon("layout"), "Reset Layout", self.reset_layout)
        hm = menu("&Help")
        hm.addAction(icons.icon("help"), "Getting started", self.getting_started)
        hm.addAction("Keyboard shortcuts", self.shortcuts_help)
        hm.addAction(icons.icon("test"), "Check installation...", self.check_installation)

    # ------------------------------------------------------------ the simple view
    def _build_simple(self):
        """Run | Build | History on the command bar, and a More menu holding
        what the menu bar holds -- the menu bar itself is shown only in
        Advanced mode."""
        bar = self.commandBar
        self.modeSwitch = simple.ModeSwitch()
        self.modeSwitch.changed.connect(self._mode_chosen)
        self._modeAction = bar.insertWidget(self.aRunOnce, self.modeSwitch)
        self._modeSep = bar.insertSeparator(self.aRunOnce)
        for key, mode in (("Ctrl+1", "run"), ("Ctrl+2", "build"), ("Ctrl+3", "history")):
            self.addAction(make_action(self, key, lambda m=mode: self._mode_chosen(m), key))

        more = QToolButton()
        more.setObjectName("moreButton")
        more.setText("More")
        more.setIcon(icons.icon("more"))
        more.setProperty("vsIcon", "more")
        more.setToolButtonStyle(Qt.ToolButtonIconOnly)
        more.setPopupMode(QToolButton.InstantPopup)
        more.setToolTip("Projects, packages, the line connection, Operator Mode, theme and help")
        more.setAccessibleName("More")
        m = QMenu(more)
        self._menus.append(m)                   # its actions lock in Operator Mode too
        m.addAction(icons.icon("add"), "New project...", self.new_application)
        m.addAction(icons.icon("open"), "Open project...", self.choose_application)
        m.addAction(icons.icon("save"), "Save all", self.save_all)
        m.addSeparator()
        m.addAction(icons.icon("job"), "Add an inspection to this project...", self.add_job)
        m.addAction(icons.icon("counts"), "Reset counts", self.reset_counts)
        m.addSeparator()
        m.addAction(icons.icon("export"), "Export package...", self.export_package)
        m.addAction(icons.icon("import"), "Import package...", self.import_package)
        m.addAction(icons.icon("python"), "Export job file (Python)...", self.export_job_file)
        m.addSeparator()
        m.addAction(self.aOperator)
        # every setting -- appearance, Advanced mode, the camera, result
        # pictures, the line connection, the supervisor password -- is in
        # the one Settings window, not spread over this menu
        m.addAction(icons.icon("gear"), "Settings...", lambda: self.show_settings())
        m.addSeparator()
        m.addAction(icons.icon("help"), "Getting started", self.getting_started)
        m.addAction("Keyboard shortcuts", self.shortcuts_help)
        more.setMenu(m)
        self.moreButton = more
        self._moreAction = bar.addWidget(more)                  # last, after Settings

    def _pref_advanced(self):
        return self._settings().value("advanced", False) in (True, "true", "True", 1, "1")

    def set_advanced(self, on, save=True):
        """The engineering view on or off. Off, the menu bar, the Views list,
        the Toolbox, the document tabs and the Layers terminals are put away;
        the three views are reached from the switch on the command bar."""
        on = bool(on)
        self.advanced, self.simple = on, not on
        if self.aAdvanced.isChecked() != on:
            self.aAdvanced.blockSignals(True)
            self.aAdvanced.setChecked(on)
            self.aAdvanced.blockSignals(False)
        self.menuBar().setVisible(on)
        self.views.setVisible(on)
        self._secViews.setVisible(on)
        self._secJobs.setText("Jobs" if on else "Inspections")
        self.jobs.setToolTip("Double-click a job to edit it; right-click for more" if on else
                             "Double-click an inspection to set it up in Build")
        for a in self._secondary:
            a.setVisible(on)
        self._moreAction.setVisible(not on)
        for a in (self.aRunOnce, self.aRunCont):
            b = self.commandBar.widgetForAction(a)
            b.setProperty("big", not on)
            theme.restyle(b)
        self.commandBar.setIconSize(QSize(18, 18) if on else QSize(20, 20))
        if not on:
            self.toolboxDock.hide()
        self._sync_tabs()
        for ed in list(self._job_editors.values()):
            ed.set_simple(not on)
        if save:
            self._settings().setValue("advanced", on)
            self.statusBar().showMessage(
                "Advanced mode - the menu bar, Toolbox and every setting are back" if on else
                "Simple view - More holds the menus; View > Advanced mode brings them back", 6000)

    def _sync_tabs(self):
        """How the workspace shows its documents. Advanced mode: as the
        person set it under Window (tabs by default). The simple view: one
        frameless document filling the workspace, switched from Run | Build |
        History -- except while a Tool Edit Control is open, when the tabs
        come back so it can be closed from its tab."""
        if getattr(self, "_syncing_tabs", False):
            return                            # showing a window activates it, which calls back here
        self._syncing_tabs = True
        try:
            subs = self.mdi.subWindowList()
            active = self.mdi.activeSubWindow()
            tool_open = any(isinstance(s.widget(), ToolEditControl) for s in subs)
            was_open = getattr(self, "_tool_was_open", False)
            self._tool_was_open = tool_open
            if self.advanced:
                want_tabs = str(self._settings().value("tabbed", "true")).lower() not in ("false", "0")
                framed = True
            else:
                want_tabs, framed = tool_open, tool_open
            for sub in subs:
                _set_frame(sub, framed)
            if (self.mdi.viewMode() == self.mdi.ViewMode.TabbedView) != want_tabs:
                self._set_tabbed(want_tabs, save=False)
            if not framed:
                self.mdi.setOption(QMdiArea.DontMaximizeSubWindowOnActivation, False)
                for sub in subs:
                    if not sub.isMaximized():
                        sub.showMaximized()
            # Showing windows activates them; the one in front stays in front.
            # A Tool Edit Control that just closed hands back to its job's
            # Build view, not to whichever window Qt picked.
            if was_open and not tool_open and not self.advanced:
                ed = self._job_editors.get(getattr(self, "_tool_return", None))
                if ed is not None and isinstance(ed.parent(), QMdiSubWindow):
                    active = ed.parent()
            if active is not None and active in subs and self.mdi.activeSubWindow() is not active:
                self.mdi.setActiveSubWindow(active)
        finally:
            self._syncing_tabs = False

    def _fills_workspace(self):
        """Whether a shown document should fill the workspace."""
        return self.mdi.viewMode() == self.mdi.ViewMode.TabbedView or not self.advanced

    def _current_mode(self):
        w = self.active_window()
        if isinstance(w, (JobEditor, ToolEditControl)):
            return "build"
        if isinstance(w, HistoryView):
            return "history"
        return "run"

    def _mode_chosen(self, name):
        if name == "run":
            self.show_run_view()
        elif name == "history":
            self.show_history()
        elif name == "build" and not self.build_current():
            self.modeSwitch.set_mode(self._current_mode())

    def build_current(self):
        """Build: the job in front, the one chosen in the sidebar, or the
        only one there is."""
        if self._locked_out():
            return False
        pid = self._active_job()
        if pid is None:
            it = self.jobs.currentItem()
            pid = it.data(0, Qt.UserRole) if it else None
        if pid is None and self.station.jobs:
            pid = self.station.jobs[0]["pid"]
        if pid is None:
            self.statusBar().showMessage("Nothing to build yet - start a project from "
                                         "More > New project", 6000)
            return False
        return self.open_job_editor(pid) is not None

    def _refresh_banner(self):
        """The verdict in words: every job's last result, and any job that
        could not be inspected."""
        jobs = self.station.jobs
        if not jobs:
            self.banner.set("idle", "READY", "Open or start a project to begin.")
            return
        results = [(j, self.station.last.get(j["pid"])) for j in jobs]
        errors = self._job_errors
        if all(r is None for _j, r in results) and not errors:
            self.banner.set("idle", "READY", "Press Run (F5) to inspect one part, or Continuous "
                            "(F6) to inspect every part as it comes.")
            return
        tone, lines, total = "pass", [], 0.0
        many = len(jobs) > 1
        for j, r in results:
            prefix = (j["name"] + " - ") if many else ""
            if j["pid"] in errors:
                tone = "fail" if tone == "fail" else "warn"
                lines.append("%s%s" % (prefix, simple.friendly_message(
                    {"status": "error", "message": errors[j["pid"]]})))
                continue
            if r is None:
                continue
            total += float(r.get("total_ms") or 0.0)
            if r["verdict"] != "PASS":
                tone = "fail"
                tools = self.station.projects[j["pid"]].spec.get("tools", [])
                lines += simple.reasons(r, tools, prefix)
        word = {"pass": "ACCEPT", "fail": "REJECT", "warn": "NOT INSPECTED"}[tone]
        if tone == "pass":
            reason = "Every check passed."
        else:
            reason = "\n".join(lines[:4]) + ("\n+ %d more" % (len(lines) - 4) if len(lines) > 4 else "")
        meta = "%s\nlast cycle %s" % (time.strftime("%H:%M:%S"), fmt_ms(total))
        self.banner.set(tone, word, reason, meta)

    # ------------------------------------------------------------ workspace modes
    def _set_tabbed(self, on, save=True):
        from PySide6.QtWidgets import QMdiArea, QTabBar
        # A tab shows its document filling the workspace; floating windows
        # keep their own size when activated.
        self.mdi.setOption(QMdiArea.DontMaximizeSubWindowOnActivation, not on)
        if on:
            self.mdi.setViewMode(QMdiArea.TabbedView)
            self.mdi.setDocumentMode(True)
            self.mdi.setTabsClosable(True)
            self.mdi.setTabsMovable(True)
            bar = self._mdi_tabs()
            if bar is not None and bar.objectName() != "mdiTabs":
                # named, so the stylesheet styles this bar and not the tab
                # bars inside the editors it holds
                bar.setObjectName("mdiTabs")
                bar.setElideMode(Qt.ElideRight)
                bar.setExpanding(False)
                bar.setUsesScrollButtons(True)
                bar.tabMoved.connect(self._pin_run_tab)
                theme.restyle(bar)
            for sub in self.mdi.subWindowList():
                sub.showMaximized()
            self._pin_run_tab()
        else:
            self.mdi.setViewMode(QMdiArea.SubWindowView)
            self.runSub.showMaximized()
        if save:
            self._settings().setValue("tabbed", bool(on))
        self._sync_window_actions()

    def _sync_window_actions(self):
        if hasattr(self, "aTile"):
            tabbed = self.mdi.viewMode() == self.mdi.ViewMode.TabbedView
            self.aTile.setEnabled(not tabbed)
            self.aCascade.setEnabled(not tabbed)

    def _mdi_tabs(self):
        """The workspace's own tab bar -- not one inside an editor it holds."""
        from PySide6.QtWidgets import QTabBar
        return next((b for b in self.mdi.findChildren(QTabBar) if b.parent() is self.mdi), None)

    def _pin_run_tab(self, *_a):
        """No close button on the Run View's tab: it is the home tab."""
        from PySide6.QtWidgets import QTabBar
        bar = self._mdi_tabs()
        if bar is None:
            return
        for i in range(bar.count()):
            if bar.tabText(i) == "Run View":
                for side in (QTabBar.RightSide, QTabBar.LeftSide):
                    btn = bar.tabButton(i, side)
                    if btn is not None:
                        btn.hide()
                        btn.setFixedSize(0, 0)

    def show_run_view(self):
        if self.mdi.viewMode() != self.mdi.ViewMode.TabbedView:
            self.runSub.showMaximized()
        self.mdi.setActiveSubWindow(self.runSub)
        if not self.advanced:
            self._sync_tabs()

    def _load_light(self):
        try:
            from vision_station.hardware import light
            self.apply_light(light.load(self._settings()), save=False)
        except Exception as e:
            print("[settings] light settings not loaded: %s" % e)

    def apply_light(self, cfg, save=True):
        """Settings > Light."""
        from vision_station.hardware import light
        cfg = dict(light.DEFAULTS, **(cfg or {}))
        if save:
            light.save(self._settings(), cfg)
        lt = self.station.light
        if not getattr(self, "_light_saver", None):
            # a level found by Keep adjusted or Adjust now is the starting
            # level next time
            def keep(_level):
                try:
                    light.save(self._settings(), lt.cfg)
                except Exception:
                    pass
            self._light_saver = keep
            lt.on_level.append(keep)
        lt.configure(cfg)
        try:
            self.hold_exposure()
        except Exception as e:
            print("[light] camera exposure not held: %s" % e)
        try:
            self._keep_camera_balance()
        except Exception as e:
            print("[light] camera white balance not set: %s" % e)
        err = self.station.light.error
        if cfg.get("enabled") and err:
            self.statusBar().showMessage("Light: %s" % err, 10000)
        return cfg

    def hold_exposure(self, index=0):
        """With Vision Station setting the light, the camera's automatic
        exposure and gain are switched off, at its manual values. Left on,
        they undid every change of the light: the camera answered a brighter
        light with a shorter exposure and lower gain, a dimmer one with more
        -- one light level gave pictures 0.29, 0.44 and 0.88 grey, the AI
        chose a level from pictures the camera had already corrected, and
        Keep adjusted chased the camera. True when they were on."""
        lt = self.station.light
        if not lt.enabled:
            return False
        from vision_station.ui.settings_dialog import (load_camera_settings,
                                                       save_camera_settings)
        qs = self._settings()
        s = load_camera_settings(qs, index)
        if s is None or not (s.auto_exposure or s.auto_gain):
            return False
        s = s.copy()
        s.auto_exposure = False
        s.auto_gain = False
        save_camera_settings(qs, index, s)
        apply_camera_settings(index, s)
        self._camera_note("Camera: automatic exposure and gain switched off (exposure %g ms, "
                          "gain %g) - with Vision Station setting the light, they undid every "
                          "change of it" % (s.exposure_ms, s.gain_db))
        return True

    def _camera_note(self, text):
        """What was done to the camera for the light, in the status bar --
        added to a note shown a moment ago rather than replacing it."""
        now = time.time()
        prev = getattr(self, "_camera_noted", (0.0, ""))
        if now - prev[0] < 2.0 and prev[1] and text not in prev[1]:
            text = prev[1] + ".   " + text.replace("Camera: ", "")
        self._camera_noted = (now, text)
        self.statusBar().showMessage(text, 15000)

    def _keep_camera_balance(self, index=0):
        """A light switched off between runs and a camera balancing white
        continuously do not mix: in the dark the camera keeps re-balancing
        on a black picture, and every run's picture starts from whatever
        colour gains it drifted to. Measured on the acA5472-5gc: steady at
        0.43 grey with the light left on; 0.30-0.41 after an off and on with
        continuous balance, 0.43 every time without -- and Keep adjusted
        then chased the camera's drift.

        So, when the light is switched, white is balanced once with the
        light on and then held, and the camera's setting says so. Done in
        the background; True when it was needed."""
        lt = self.station.light
        if not lt.enabled or lt.cfg.get("always_on"):
            return False
        from vision_station.ui.settings_dialog import (load_camera_settings,
                                                       save_camera_settings)
        qs = self._settings()
        s = load_camera_settings(qs, index)
        if s is None or not s.auto_white_balance:
            return False
        s = s.copy()
        s.auto_white_balance = False
        save_camera_settings(qs, index, s)
        self._camera_note(
            "Camera: white balanced once with the light on, then held - balancing "
            "continuously, it drifted in the dark between runs")

        def work():
            import time as _t
            handle = None
            try:
                handle = open_camera(index)
                t0 = _t.time()
                while getattr(handle, "backend", None) is None and _t.time() - t0 < 10:
                    _t.sleep(0.05)
                if getattr(handle, "backend", None) != "pylon":
                    return
                apply_camera_settings(index, s)          # continuous balance off
                lt.hold("balance")
                lt.send_level(BALANCE_LEVEL)             # lit, not burnt out
                lt.settle()
                handle.frame_after(None, settle=3, timeout_s=5)   # lit frames only
                # balanced on lit frames only: balanced in the dark, every
                # picture came out pink; with the gains at 1, green
                handle.call(lambda pc: pc.white_balance_once() if pc else False)
                t0 = _t.time()                           # Once ends by itself
                while _t.time() - t0 < 4:
                    v = handle.call(lambda pc: str(pc.cam.BalanceWhiteAuto.GetValue())
                                    if pc else "Off")
                    if v != "Once":
                        break
                    _t.sleep(0.2)
            except Exception as e:
                print("[light] white balance once: %s" % e)
            finally:
                lt.release("balance")
                if handle is not None:
                    handle.stop()
        import threading
        threading.Thread(target=work, name="balance-once", daemon=True).start()
        return True

    def _load_barcode(self):
        try:
            from vision_station.core import barcode as rules
            self.apply_barcode(rules.load(self._settings()), save=False)
        except Exception as e:
            print("[settings] barcode settings not loaded: %s" % e)

    def _sub_activated(self, _sub):
        if self._closing:
            return
        try:
            self._update_edit_actions()
            self._queue_follows(_sub)
            if hasattr(self, "modeSwitch"):
                self.modeSwitch.set_mode(self._current_mode())
                self._sync_tabs()
        except RuntimeError:
            pass                                 # the window is being torn down

    def _queue_follows(self, sub):
        """The results queue is for watching parts go by. It stands aside
        while an editor is in front -- which has its own result line -- and
        comes back with the Run View. Hidden by hand, it stays hidden."""
        w = sub.widget() if sub is not None else None
        editing = isinstance(w, (JobEditor, ToolEditControl))
        if editing and self.queueDock.isVisible():
            self._queue_stood_aside = True
            self.queueDock.hide()
        elif not editing and getattr(self, "_queue_stood_aside", False):
            self._queue_stood_aside = False
            self.queueDock.show()

    def _warmed_up(self, s):
        self.sEngine.setText("Engines ready")
        self.statusBar().showMessage("Ready - engines started in %.1f s" % s, 5000)

    # ------------------------------------------------------------ theme
    def set_theme(self, mode):
        """Light, dark, or Windows' own setting -- applied at once, kept for
        next time."""
        self._settings().setValue("theme", mode)
        apply_theme(QApplication.instance(), mode)
        self._reicon()
        self.mdi.setBackground(QBrush(QColor(theme.color("bg_alt"))))
        # rows coloured in the old theme's tones are redrawn in the new one's
        self._refresh_jobs()
        self._fill_queue()
        a = self._theme_actions.get(mode)
        if a is not None and not a.isChecked():
            a.setChecked(True)

    def _reicon(self):
        """Icons are drawn in the theme's ink; redraw them in the new one."""
        for w in QApplication.allWidgets():
            for a in w.actions():
                name = a.property("vsIcon")
                if name:
                    a.setIcon(icons.icon(name, on=a.property("vsIconOn") or None))
            # buttons that are not actions: the display's floating bars
            name = w.property("vsIcon") if isinstance(w, QToolButton) else None
            if name and isinstance(name, str):
                w.setIcon(icons.icon(name))
        self._accent_run_icon()
        for i in range(self.toolbox.topLevelItemCount()):
            g = self.toolbox.topLevelItem(i)
            for k in range(g.childCount()):
                it = g.child(k)
                ty = it.data(0, Qt.UserRole)
                if ty:
                    it.setIcon(0, icons.tool_icon(ty))
        for i in range(self.views.count()):
            it = self.views.item(i)
            it.setIcon(icons.icon(it.data(Qt.UserRole + 1)))
        for ed in self._job_editors.values():
            ed.refresh()

    # ------------------------------------------------------------ help
    def getting_started(self):
        QMessageBox.information(self, "Getting started", (
            "Three views, on the switch at the top left:\n\n"
            "RUN  -  watch parts being inspected. Run (F5) inspects one part; Continuous\n"
            "(F6) inspects every part as it comes. The verdict and its reasons are at\n"
            "the top; recent parts are in the list below.\n\n"
            "BUILD  -  set up what to check. The guide on the right walks through it:\n"
            "a picture of a good part, a Locate check so the others follow the part,\n"
            "then Add a check - pick the question, drag the blue box over the feature,\n"
            "and say what to expect. Press Run to test, Ctrl+S to save.\n\n"
            "HISTORY  -  yield over time and the checks that reject most parts.\n\n"
            "More (top right) holds projects, packages, the line connection, Operator\n"
            "Mode (locks Build for the line) and Advanced mode - the full engineering\n"
            "view with every tool, setting and terminal."))

    def shortcuts_help(self):
        QMessageBox.information(self, "Keyboard shortcuts", (
            "F5\tRun once - the job in front, or every job\n"
            "F6\tRun continuously / stop\n"
            "Ctrl+S\tSave the job in front, or all\n"
            "Ctrl+O\tOpen application\n"
            "Ctrl+Z / Ctrl+Y\tUndo / redo\n\n"
            "In a Job Editor's tool list:\n"
            "Enter\tOpen the tool\n"
            "F2\tRename it\n"
            "Del\tDelete it (Ctrl+Z brings it back)\n\n"
            "On an image - as in a drawing application:\n"
            "V / H\tSelect / pan (or hold Space, or the middle button)\n"
            "R / E / P\tDraw a rectangle / ellipse / polygon region\n"
            "Drag a handle\tResize; Shift keeps proportions, Alt from the centre\n"
            "Grip or corner\tDrag the round grip, or just outside a corner, to rotate;\n"
            "\tShift snaps to 15°\n"
            "[  ]\tRotate 1° (Shift: 15°)\n"
            "Shift-drag\tMove along one axis; Ctrl turns snapping off\n"
            "Shift-click\tAdd to or remove from the selection\n"
            "Alt-drag\tDuplicate (or Ctrl+D)\n"
            "Arrows\tNudge 1 px (Shift: 10 px)\n"
            "Ctrl+A\tSelect every region\n"
            "Ctrl+0 / 1 / 2\tFit / actual size / zoom to the selection\n"
            "Right-click\tRotate, align, distribute, lock\n"
            "Wheel\tZoom about the pointer"))

    def check_installation(self):
        import contextlib
        import io
        buf = io.StringIO()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            from vision_station.core import preflight
            with contextlib.redirect_stdout(buf):
                preflight.main()
            text = buf.getvalue()
        except Exception as e:
            text = buf.getvalue() + "\nThe installation check could not finish: %s" % e
        finally:
            QApplication.restoreOverrideCursor()
        dlg = QDialog(self)
        dlg.setWindowTitle("Check installation")
        dlg.resize(720, 480)
        from PySide6.QtWidgets import QPlainTextEdit
        box = QPlainTextEdit(text)
        box.setReadOnly(True)
        box.setObjectName("mono")
        lay = QVBoxLayout(dlg)
        lay.addWidget(box)
        ok_cancel(dlg, lay)
        dlg.show()
        self._preflight = dlg
        return dlg

    # ------------------------------------------------------------ application
    def choose_application(self):
        apps = self.station.applications()
        if not apps:
            if QMessageBox.question(self, "Open application",
                                    "There are no applications in %s yet.\n\n"
                                    "Create one now?" % self.root) == QMessageBox.Yes:
                self.new_application()
            return
        names = ["%s  (%s)" % (a["name"], a["id"]) for a in apps]
        cur = next((i for i, a in enumerate(apps) if a["id"] == self.station.app_id), 0)
        pick, ok = QInputDialog.getItem(self, "Open application", "Application:", names, cur, False)
        if ok:
            self.open_application(apps[names.index(pick)]["id"])

    def open_application(self, app_id):
        if self.dirty and not self._confirm_discard(revert=False):
            return False
        for ed in list(self._tool_editors.values()) + list(self._job_editors.values()):
            ed.close()
        h = getattr(self, "_history", None)
        if h is not None:                     # it showed the previous application's
            sub = h.parent()
            h.close()
            if isinstance(sub, QMdiSubWindow):
                sub.close()
            self._history = None
        self.aRunCont.setChecked(False)
        self.station.open(app_id)
        self.dirty.clear()
        name = next((a["name"] for a in self.station.applications() if a["id"] == app_id), app_id)
        self.setWindowTitle("%s - Vision Station" % name)
        self._build_displays()
        self._refresh_jobs()
        self._refresh_queue()
        self._refresh_counts()
        self._start_line()
        self._reset_history()
        self.sEngine.setText("Starting engines...")
        return True

    def _clear_grid(self):
        while self.gridLayout.count():
            w = self.gridLayout.takeAt(0).widget()
            if w is not None:
                w.hide()                     # gone now; deleted when the loop next runs
                w.deleteLater()
        for r in range(self.gridLayout.rowCount()):
            self.gridLayout.setRowStretch(r, 0)
        for c in range(self.gridLayout.columnCount()):
            self.gridLayout.setColumnStretch(c, 0)

    def _build_displays(self):
        self._clear_grid()
        self._displays = {}
        self._tiles = {}
        n = len(self.station.jobs)
        if n == 0:
            self.gridLayout.addWidget(self._welcome(), 0, 0)
            return
        cols = 1 if n <= 1 else 2 if n <= 4 else 3
        for i, j in enumerate(self.station.jobs):
            pid = j["pid"]
            tile = JobTile(j["name"], lambda _c=False, pid=pid: self.open_job_editor(pid))
            d = tile.display
            d.canvas.mouseDoubleClickEvent = (lambda e, pid=pid: self.open_job_editor(pid))
            d.canvas.setToolTip("Double-click to open the Job Editor")
            p = self.station.projects[pid]
            d.canvas.tag_min_px = 13.0           # read from a step away
            d.canvas.tag_max_px = 30.0
            d.set_records({"Reference": (p.reference, [])})
            tile.edit.setVisible(not getattr(self, "operator_mode", False))
            self.gridLayout.addWidget(tile, i // cols, i % cols)
            self._displays[pid] = (tile.status, d)
            self._tiles[pid] = tile
        for r in range((n + cols - 1) // cols):
            self.gridLayout.setRowStretch(r, 1)
        for c in range(cols):
            self.gridLayout.setColumnStretch(c, 1)
        self._job_errors.clear()
        self._refresh_banner()

    def _welcome(self):
        """What a first start shows instead of an empty grey workspace."""
        self.banner.set("idle", "WELCOME", "")
        ids = [a["id"] for a in self.station.applications()]
        demo = "demo_led_panel" if "demo_led_panel" in ids and self.station.app_id != "demo_led_panel" \
            else None
        return simple.WelcomePage(self, demo)

    def _refresh_jobs(self):
        self.jobs.clear()
        for j in self.station.jobs:
            res = self.station.last.get(j["pid"])
            it = QTreeWidgetItem([j["name"] + (" *" if j["pid"] in self.dirty else ""),
                                  ("Accept" if res and res["verdict"] == "PASS" else
                                   "Reject" if res else ""),
                                  fmt_ms(res["total_ms"]) if res else ""])
            it.setData(0, Qt.UserRole, j["pid"])
            it.setToolTip(0, "%s%s%s\nDouble-click to open the Job Editor" % (
                j["name"], "  (unsaved changes)" if j["pid"] in self.dirty else "",
                ("\nLast run: %s, %s" % ("Accept" if res["verdict"] == "PASS" else "Reject",
                                        fmt_ms(res["total_ms"]))) if res else ""))
            col = QColor(status_dot(res["verdict"]) if res else IDLE_C)
            it.setIcon(0, dot_icon(col))
            it.setForeground(1, QBrush(status_text(res["verdict"] if res else None)))
            self.jobs.addTopLevelItem(it)

    def _refresh_counts(self):
        c = self.station.counts
        tot = c["accept"] + c["reject"]
        y = (100.0 * c["accept"] / tot) if tot else None
        self.sCounts.setText("Accept %d    Reject %d    Total %d    Yield %s" % (
            c["accept"], c["reject"], tot, ("%.1f%%" % y) if tot else "-"))
        self.kpi["accept"].set(str(c["accept"]), "pass" if c["accept"] else None)
        self.kpi["reject"].set(str(c["reject"]), "fail" if c["reject"] else None)
        self.kpi["yield"].set(("%.1f%%" % y) if tot else "-",
                              None if y is None else "pass" if y >= 99.0 else
                              "fail" if y < 90.0 else None)
        if not tot:
            self.kpi["cycle"].set("-")

    # ------------------------------------------------------------ running
    def _nothing_to_run(self):
        if self.station.jobs:
            return False
        self.statusBar().showMessage("Nothing to run - open or create an application first "
                                     "(File > Open Application)", 6000)
        return True

    def run_once(self):
        if self._nothing_to_run():
            return
        if self.station.running:
            self.statusBar().showMessage("Still inspecting - the next run starts when "
                                         "this one finishes", 3000)
            return
        self.station.run_once()

    def run_job(self, pid):
        if self.station.running:
            self.statusBar().showMessage("Still inspecting - the next run starts when "
                                         "this one finishes", 3000)
            return
        # a job editor's run is for teaching: never held for a barcode
        self.station.run_once([pid], gate=False)

    def toggle_continuous(self):
        if self.aRunCont.isChecked():
            if self._nothing_to_run():
                self.aRunCont.setChecked(False)
                return
            self.station.start_continuous()
        else:
            self.station.stop()

    def stop(self):
        self.aRunCont.setChecked(False)
        self.station.stop()

    def _job_result(self, pid, res):
        _title, d = self._displays.get(pid, (None, None))
        tile = self._tiles.get(pid)
        if d is not None:
            # every inspection is shown whole: a tile zoomed into once kept
            # that zoom, and every later part was shown cut off
            d.canvas.fit_mode = True
            # With many checks the OK tags buried the NG one the operator is
            # looking for (41 "OK" round two "NG"): then only failures are
            # labelled. Every outline stays, blue or red.
            d.canvas.tags_only_failures = len(res.get("tools", [])) > RUN_VIEW_ALL_TAGS
            d.set_records(records_for(res))
            d.set_spaces(spaces_for(res))
        ok = res["verdict"] == "PASS"
        if tile is not None:
            if self.simple:
                p = self.station.projects.get(pid)
                failed = simple.failed_names(res, p.spec.get("tools", []) if p else [])
            else:
                failed = [t["tool_id"] for t in res["tools"] if t["status"] != "pass"]
            if res.get("barcode_ng"):
                # the part does not carry the scanned code: said as such
                tile.show_verdict("fail", "NG", "%s   %s" % (fmt_ms(res["total_ms"]),
                                                            res["barcode_ng"]))
            else:
                tile.show_verdict("pass" if ok else "fail", "ACCEPT" if ok else "REJECT",
                                  "%s%s" % (fmt_ms(res["total_ms"]),
                                            ("   failed: " + ", ".join(failed)) if failed else ""))
        if res.get("barcode_ng"):
            self.statusBar().showMessage(res["barcode_ng"], 15000)
        if res.get("barcode") and self.station.barcode.last:
            code, verdict, folder = self.station.barcode.last
            self.barcodeBar.done(code, verdict, folder,
                                 "NG - Barcode not good" if res.get("barcode_ng") else None)
        self.sCycle.setText("Last cycle %s" % fmt_ms(res["total_ms"]))
        self.kpi["cycle"].set(fmt_ms(res["total_ms"]))
        self._job_errors.pop(pid, None)
        self._refresh_jobs()
        self._refresh_counts()
        self._refresh_queue()
        self._refresh_banner()
        ed = self._job_editors.get(pid)
        if ed is not None:
            ed.refresh()
        for te in self.tool_editors_for(pid):
            te.refresh()

    def _job_error(self, pid, msg):
        """Shown on the job's own display, in amber: the job was not inspected,
        which is neither a pass nor a fail."""
        self.statusBar().showMessage("%s: %s" % (self.job_name(pid), msg), 8000)
        tile = self._tiles.get(pid)
        if tile is not None:
            tile.show_verdict("warn", "ERROR", msg)
        self._job_errors[pid] = msg
        self._refresh_banner()
        if "camera" in msg.lower() and not getattr(self, "_firewall_asked", False):
            self._firewall_asked = True          # asked once a session, never again
            QTimer.singleShot(0, self._offer_firewall)

    def _offer_firewall(self):
        """The packaged program without its firewall rule opens a GigE camera
        but no picture ever arrives (hardware/firewall.py): say so, and offer
        the one-time fix."""
        from vision_station.hardware import firewall
        try:
            if not firewall.should_offer():
                return
        except Exception:
            return
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox(QMessageBox.Warning, "Camera blocked by Windows Firewall",
                          firewall.EXPLANATION, parent=self)
        allow = box.addButton("Allow...", QMessageBox.AcceptRole)
        box.addButton("Not now", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is allow:
            if firewall.allow():
                self.statusBar().showMessage("Allowed - the camera's pictures now reach "
                                             "Vision Station. Run again.", 10000)
            else:
                QMessageBox.information(self, "Not allowed",
                                        "The rule was not added. Ask IT to allow %s for "
                                        "inbound connections." % firewall.program())

    # ------------------------------------------------------------ barcode
    def apply_barcode(self, cfg, save=True):
        """Settings > Barcode, or the Run view's dialog."""
        from vision_station.core import barcode as rules
        cfg = dict(rules.DEFAULTS, **(cfg or {}))
        if save:
            rules.save(self._settings(), cfg)
        self.station.barcode.configure(cfg)
        on = bool(cfg["enabled"])
        self.barcodeBar.setVisible(on)
        self.barcodeBar.set_rule(cfg)
        if hasattr(self, "aBarcode"):
            self.aBarcode.setChecked(on)
        self.scanner.close()
        if on and cfg["source"] == "serial" and cfg["port"]:
            if self.scanner.open(cfg["port"], cfg["baud"]):
                self.statusBar().showMessage("Scanner on %s" % cfg["port"], 5000)
        app = QApplication.instance()
        if on and cfg["source"] == "keyboard":
            if getattr(self, "_scan_keys", None) is None:
                self._scan_keys = _ScanKeys(self)
                app.installEventFilter(self._scan_keys)
        elif getattr(self, "_scan_keys", None) is not None:
            app.removeEventFilter(self._scan_keys)
            self._scan_keys = None
        if on:
            self.barcodeBar.waiting()
            QTimer.singleShot(0, self.barcodeBar.field.setFocus)
        return cfg

    def barcode_settings(self):
        from vision_station.core import barcode as rules
        from vision_station.ui.barcode_panel import BarcodeDialog
        from vision_station.ui.settings_dialog import load_result_settings
        qs = self._settings()
        dlg = BarcodeDialog(self, rules.load(qs), self.apply_barcode,
                            load_result_settings(qs).get("folder", ""))
        dlg.show()
        self._barcode_dialog = dlg
        if hasattr(self, "aBarcode"):
            self.aBarcode.setChecked(self.station.barcode.enabled)   # the dialog decides
        return dlg

    def barcode_scanned(self, raw):
        """A code from the bar, the keyboard scanner or the serial one."""
        gate = self.station.barcode
        if not gate.enabled:
            return False
        ok, code, why = gate.scan(raw)
        if not ok:
            self.barcodeBar.refused(code, why)
            QApplication.beep()
            self.statusBar().showMessage("Wrong barcode %s: %s" % (code, why), 8000)
            return False
        busy = self.station.running
        auto = gate.cfg.get("auto_run", True)
        self.barcodeBar.accepted(code, running=auto)
        self.statusBar().showMessage("Barcode %s accepted%s" % (
            code, " - inspecting" if auto and not busy else
            " - inspected when this part is done" if auto else " - press Run"), 6000)
        if auto and not busy:
            self.run_once()
        return True

    def _barcode_needed(self):
        if self.station.continuous:
            self.barcodeBar.waiting("Waiting for the next part's barcode")
        else:
            self.barcodeBar.needed()
            self.statusBar().showMessage("Scan the part's barcode first - barcode checking is "
                                         "on (Settings > Barcode)", 6000)

    def _run_finished(self):
        gate = self.station.barcode
        if gate.enabled and gate.pending and gate.cfg.get("auto_run", True):
            # a code scanned during the last inspection: its part is next
            QTimer.singleShot(0, self.run_once)
        if not self.station.continuous:
            self.aRunCont.setChecked(False)
            for ed in self._job_editors.values():
                ed.aCont.setChecked(False)

    # ------------------------------------------------------------ queue
    def _set_queue_mode(self, failures):
        self.queueFailuresOnly = failures
        self.queueDock.setWindowTitle("Results queue - failures" if failures else "Results queue")
        self._refresh_queue()

    def _refresh_queue(self):
        self._fill_queue()
        self._show_newest_record()

    def _fill_queue(self):
        src = list(self.station.failures if self.queueFailuresOnly else self.station.history)
        src.reverse()
        # The newest run goes on top and pushes every row down. The selection
        # used to stay on the same row number -- so it marked a different run
        # from the one on the display. It follows its run now.
        rows = self.queue.selectionModel().selectedRows()
        old = getattr(self, "_queue_rows", None) or []
        chosen = old[rows[0].row()] if rows and rows[0].row() < len(old) else None
        self._queue_rows = src
        posted = list(self.station.posted)
        plain = self.simple
        # the barcode column while barcodes are checked, or any run in the
        # queue has one
        codes = self.station.barcode.enabled or any(e.get("barcode") for e in src)
        heads = ["Time", "Inspection" if plain else "Job", "Result", "Cycle",
                 "Failed checks" if plain else "Failed tools"] + \
            (["Barcode"] if codes else []) + \
            [self.posted_label(item) for item in posted]
        first = 6 if codes else 5                 # the first posted value's column
        if self.queue.columnCount() != len(heads):
            self.queue.setColumnCount(len(heads))
        self.queue.setHorizontalHeaderLabels(heads)
        for k, item in enumerate(posted):
            self.queue.horizontalHeaderItem(first + k).setToolTip(item)
        self.queue.setRowCount(len(src))
        for i, e in enumerate(src):
            name = self.job_name(e["job"])
            failed = e["failed"]
            if plain:
                failed = [self.check_name(e["job"], tid) for tid in failed]
            vals = [e["time"], name, "Accept" if e["verdict"] == "PASS" else "Reject",
                    fmt_ms(e["ms"]), ", ".join(failed)] + \
                ([e.get("barcode") or ""] if codes else [])
            for k, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if k == 2:
                    it.setForeground(QBrush(status_text(e["verdict"])))
                    it.setIcon(dot_icon(status_dot(e["verdict"])))
                if k in (1, 4):
                    it.setToolTip(v)              # the whole text when the column cuts it
                self.queue.setItem(i, k, it)
            for k, item in enumerate(posted):
                v = e.get("posted", {}).get(item)
                txt = "" if v is None else ("%.4g" % v if isinstance(v, float) else str(v))
                self.queue.setItem(i, first + k, QTableWidgetItem(txt))
        if chosen is not None:
            self.queue.blockSignals(True)
            if any(e is chosen for e in src):
                self.queue.selectRow(next(i for i, e in enumerate(src) if e is chosen))
            else:
                self.queue.clearSelection()        # its run has left the queue
            self.queue.blockSignals(False)

    def _show_newest_record(self):
        """Until a row is chosen, the queue's display shows the latest run -- it
        used to stay blank. An emptied queue empties it too."""
        if not getattr(self, "_queue_rows", None):
            self.queueDisplay.set_records({})
            return
        if self.queue.selectionModel().selectedRows():
            return
        self._show_queue_entry(self._queue_rows[0])

    def _show_queue_entry(self, e):
        """A queue entry's preview, with its results drawn where they were
        found: the preview is enlarged back to the frame's size (nearest
        pixel, a few ms) so the results' full-frame coordinates fit it."""
        if not e.get("jpg"):
            return
        img = cv2.imdecode(np.frombuffer(e["jpg"], np.uint8), cv2.IMREAD_COLOR)
        size = e.get("size")
        if img is not None and size and (img.shape[1], img.shape[0]) != tuple(size):
            img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_NEAREST)
        self.queueDisplay.canvas.fit_mode = True
        self.queueDisplay.canvas.tags_only_failures = len(e["tools"]) > RUN_VIEW_ALL_TAGS
        self.queueDisplay.set_records({"Record": (img, e["tools"])})

    def _queue_selected(self):
        rows = self.queue.selectionModel().selectedRows()
        if not rows or not hasattr(self, "_queue_rows"):
            return
        self._show_queue_entry(self._queue_rows[rows[0].row()])

    def reset_counts(self):
        self.station.counts = {"accept": 0, "reject": 0}
        self.station.history.clear()
        self.station.failures.clear()
        self._refresh_counts()
        self._refresh_queue()
        self._refresh_banner()


    # ------------------------------------------------------------ jobs
    def _jobs_menu(self, pos):
        if self._locked_out():
            return
        it = self.jobs.itemAt(pos)
        pid = it.data(0, Qt.UserRole) if it else None
        m = QMenu(self)
        if pid:
            m.addAction("Open Job Editor", lambda: self.open_job_editor(pid))
            if self.station.is_job_app():
                m.addAction("Rename...", lambda: self.rename_job(pid))
                m.addAction("Remove...", lambda: self.remove_job(pid))
            m.addSeparator()
            from vision_station.core import advanced
            p = self.station.projects.get(pid)
            trained = p is not None and advanced.is_trained(p.spec, p.folder)
            if self.advanced_training_on():
                a = m.addAction("Model teaching run (retrain)..." if trained else
                                "Model teaching run...", lambda: self.teaching_run([pid]))
                a.setEnabled(not self.station.running)
            if trained:
                m.addAction("Remove advanced model...", lambda: self.remove_advanced(pid))
            m.addSeparator()
        a = m.addAction("Add Job...", self.add_job)
        a.setEnabled(self.station.is_job_app())
        m.exec(self.jobs.viewport().mapToGlobal(pos))

    @staticmethod
    def _starter_tools(img):
        """A new job starts with a fixture over the middle of its image, the
        way every job here begins."""
        h, w = img.shape[:2]
        return [{"id": "fixture", "type": "align",
                 "params": dict(defaults("align"),
                                roi=[int(w * .1), int(h * .1), int(w * .8), int(h * .8)])}]

    def _ask_image(self, title):
        path, _ = QFileDialog.getOpenFileName(self, title, "",
                                              "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        return path or None

    def _read_reference(self, image_path, title):
        """The reference image, or None -- having said why when a file was
        chosen but cannot be read. It used to stop without a word."""
        img = imio.imread(image_path) if image_path else None
        if img is None and image_path:
            QMessageBox.warning(self, title, "Could not read %s as an image." %
                                os.path.basename(image_path))
        return img

    # ------------------------------------------------------------ advanced training
    ADVANCED_KEY = "advanced_training/use"

    def advanced_training_on(self):
        """Settings > General > Use advanced model for training."""
        v = self._settings().value(self.ADVANCED_KEY, False)
        return v is True or str(v).lower() in ("true", "1")

    def advanced_model_size(self):
        """Settings > General > AI model size: small (default), medium, large."""
        from vision_station.core.advanced import SIZES
        v = str(self._settings().value("advanced_training/model", "small") or "small")
        return v if v in SIZES else "small"

    def set_advanced_model_size(self, size):
        self._settings().setValue("advanced_training/model", size)

    def set_advanced_training(self, on):
        self._settings().setValue(self.ADVANCED_KEY, bool(on))
        if hasattr(self, "aRunTeach"):
            self._refresh_run_menu()

    def _advanced_request(self, extra):
        """The recipe keys of a new job, with the advanced request when the
        setting is on."""
        extra = dict(extra or {})
        if self.advanced_training_on():
            extra["advanced"] = {"requested": True, "trained": False}
        return extra

    def _refresh_run_menu(self):
        on = self.advanced_training_on()
        self.aRunTeach.setEnabled(on)
        self.aRunTeach.setToolTip(
            "Takes pictures of the good part and trains the job's own AI model"
            if on else "Switch on Settings > General > Use advanced model for training first")
        if not on and self.run_mode == "teach":
            self.set_run_mode("normal")

    def set_run_mode(self, mode):
        """'normal' (inspect) or 'teach' (Model teaching run). Teaching only
        with the advanced training setting on."""
        if mode == "teach" and not self.advanced_training_on():
            mode = "normal"
        self.run_mode = mode
        teach = mode == "teach"
        (self.aRunTeach if teach else self.aRunNormal).setChecked(True)
        self.aRunOnce.setText("Teach model" if teach else "Run")
        self.aRunOnce.setToolTip("Model teaching run: take pictures of the good part and train "
                                 "the job's own AI model" if teach else
                                 "Inspect once - the job in front, or every job")
        self.statusBar().showMessage(
            "Run is set to Model teaching run: press Run to teach the AI model" if teach
            else "Run is set to Normal run", 5000)
        return mode

    def teaching_run(self, pids=None):
        """Model teaching run: each job (the one in front, or every job) has
        its AI model taught in the training window. Run is set back to
        Normal run afterwards. {pid: outcome}."""
        if not self.advanced_training_on():
            self.set_run_mode("normal")
            return {}
        if self._nothing_to_run() or self._adv_gating:
            return {}
        if self.station.running or self.station.continuous:
            self.statusBar().showMessage("Stop the inspection before a model teaching run",
                                         5000)
            return {}
        from vision_station.core import advanced
        ok, why = advanced.can_train()
        if not ok:
            QMessageBox.information(self, "Model teaching run",
                                    "The AI model cannot be taught on this PC: %s." % why)
            self.set_run_mode("normal")
            return {}
        pids = [p for p in (pids or [j["pid"] for j in self.station.jobs]) if p]
        out = {}
        self._adv_gating = True
        try:
            for pid in pids:
                out[pid] = self.train_advanced(pid)
                if out[pid] == "cancelled":
                    break                       # cancelled: not the next job either
        finally:
            self._adv_gating = False
            self.set_run_mode("normal")
        return out

    def train_advanced(self, pid, grab=None, session_kw=None):
        """Train job pid's advanced model now, in the training window.
        "ok" | "cancelled" | "failed" | None (could not start)."""
        from vision_station.core import advanced
        from vision_station.ui.advanced_training import run_training
        if pid in self.dirty and self.save_job(pid) is False:
            self._adv_postponed.add(pid)
            return None
        p = self.station.projects.get(pid)
        if p is None:
            return None
        if not advanced.state_of(p.spec).get("requested"):
            p.spec["advanced"] = {"requested": True, "trained": False}
            self.save_job(pid)
            p = self.station.projects[pid]
        src = self.station.sources.get(pid)
        if grab is None:
            if src is None:
                return None
            # the camera opened here, in the window's thread, before the
            # training's own thread takes pictures from it
            first = src.grab()
            if first is None:
                QMessageBox.warning(self, "Advanced training",
                                    "No picture for the training: %s" % (src.last_error or
                                                                         "the source gave none"))
                self._adv_postponed.add(pid)
                return "failed"

            def grab():
                img = src.grab()
                if img is None:
                    raise RuntimeError(src.last_error or "the camera gave no picture")
                return img
        cam = src is not None and src.mode == "camera"
        kw = dict(session_kw or {})
        kw.setdefault("model_size", self.advanced_model_size())
        sess = advanced.Session(p, grab, light=self.station.light if cam else None,
                                light_level=p.spec.get("light_level"), pid=pid, **kw)
        self._adv_session = sess
        outcome = run_training(self, sess, self.job_name(pid))
        self._adv_session = None
        if outcome == "ok":
            self.reload_job(pid)
            self.statusBar().showMessage("%s: advanced model trained and in use" %
                                         self.job_name(pid), 10000)
        else:
            self._adv_postponed.add(pid)
            self.statusBar().showMessage(
                "%s: advanced training %s - the job runs with its classic checks" % (
                    self.job_name(pid), "cancelled" if outcome == "cancelled" else "stopped"),
                10000)
        return outcome

    def remove_advanced(self, pid, confirm=True):
        """The job's advanced model and the good pictures it added, gone."""
        from vision_station.core import advanced
        p = self.station.projects.get(pid)
        if p is None:
            return False
        if confirm and QMessageBox.question(
                self, "Remove advanced model",
                "Remove %s's advanced AI model and the good pictures it added to its checks?"
                % self.job_name(pid)) != QMessageBox.Yes:
            return False
        if pid in self.dirty:
            self.save_job(pid)
            p = self.station.projects[pid]
        advanced.remove(p.folder)
        p.spec.pop("advanced", None)
        self.save_job(pid)
        return True

    def choose_reference_source(self, title):
        """Where the first picture of a GOOD part comes from: 'file' (browse
        for one), 'camera' (the AI reference image) or None (cancelled)."""
        dlg = ReferenceSourceDialog(self, title, advanced=self.advanced_training_on())
        dlg.exec()
        return dlg.choice

    def ai_reference(self, title, grab=None, camera=0):
        """A picture of a good part from the camera, at the best light: the
        light is tried at several levels and each picture judged by
        measurement and the AI (core/light_tuning.py), then the picture is
        taken at the level chosen. With the light not controlled by Vision
        Station, the picture as the camera sees it. grab: a function giving
        a fresh camera picture (default: camera `camera`).
        {"image", "level" (None: light not controlled), "result", "message"},
        or None when stopped or failed -- having said why."""
        import threading
        from PySide6.QtWidgets import QProgressDialog
        from vision_station.core import light_tuning
        lt = self.station.light
        handle = None
        if grab is None:
            handle = open_camera(camera)
            t0 = time.time()
            while getattr(handle, "backend", None) is None and handle.is_alive() \
                    and time.time() - t0 < 12:
                QApplication.processEvents()
                time.sleep(0.05)
            if getattr(handle, "backend", None) is None:
                why = getattr(handle, "last_error", "") or "the camera did not start"
                handle.stop()
                QMessageBox.warning(self, title, "No picture from the camera: %s\n\nCheck "
                                    "Settings > Camera." % why)
                return None

            def grab():
                img = handle.frame_after(None, settle=3, timeout_s=6.0)
                if img is None:
                    raise RuntimeError(getattr(handle, "last_error", "") or
                                       "the camera gave no picture")
                return img
        # Colour pictures, the camera's white balanced once with the light on.
        # Balanced in the dark -- or with its colour gains at 1, where the
        # sensor shows a white light as strongly green -- every picture came
        # out green or pink; balanced under the lit light: blue 171, green
        # 175, red 176 on a grey part, a true colour picture.
        job = {"stop": False, "done": False, "result": None, "image": None,
               "error": None, "progress": None, "balance": None}
        tuned = lt.enabled and lt.ctl is not None or (lt.enabled and lt._connect())
        if tuned and self.hold_exposure(camera):
            t0 = time.time()                        # the camera takes the change
            while time.time() - t0 < 0.8:
                QApplication.processEvents()
                time.sleep(0.05)
        bal_handle = None
        if handle is not None:
            cam_call = handle.call
        else:                                      # the job's camera: one more handle
            bal_handle = open_camera(camera)
            cam_call = getattr(bal_handle, "call", None)

        def balance_first():
            """White balanced once under the lit light, then held."""
            if cam_call is None:
                return
            lt.hold("teach")
            try:
                # a moderate level: a burnt-out picture cannot be balanced
                lt.send_level(BALANCE_LEVEL)
                lt.settle()
                grab()                                 # frames from before the change
                if not cam_call(lambda pc: pc.white_balance_once() if pc else False):
                    return                             # no such camera: as it is
                t0 = time.time()                       # Once ends by itself
                while time.time() - t0 < 6:
                    v = cam_call(lambda pc: str(pc.cam.BalanceWhiteAuto.GetValue())
                                 if pc else "Off")
                    if v != "Once":
                        break
                    time.sleep(0.2)
                job["balance"] = "once"
            finally:
                lt.release("teach")

        def work():
            try:
                if tuned:
                    balance_first()
                if tuned:
                    job["result"] = lt.tune(
                        grab, stop=lambda: job["stop"],
                        progress=lambda i, n, row: job.__setitem__("progress", (i, n, row)))
                if not job["stop"]:
                    lt.hold("teach")              # the picture, at the level chosen
                    try:
                        lt.settle()
                        job["image"] = grab()
                    finally:
                        lt.release("teach")
            except Exception as e:
                job["error"] = str(e)
            finally:
                job["done"] = True

        n = len(light_tuning.LEVELS) if tuned else 1
        dlg = QProgressDialog(("Trying the light at %d levels; the AI looks at each picture..."
                               % n) if tuned else "Taking the picture...", "Stop", 0, n + 1, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        # Stop counts only while it runs: closing the dialog at the end
        # emits canceled too, and the finished picture was thrown away
        dlg.canceled.connect(lambda: job["done"] or job.__setitem__("stop", True))
        threading.Thread(target=work, name="ai-reference", daemon=True).start()
        try:
            while not job["done"]:
                if job["progress"]:
                    i, _n, row = job["progress"]
                    dlg.setValue(i)
                    ai = ""
                    if row.get("parts") is not None:
                        ai = ", the AI finds %d components" % row.get("parts_found", 0)
                    if row.get("text") is not None:
                        ai += " and reads %d characters" % int(row["text"])
                    dlg.setLabelText("Light level %d: %.0f %% grey%s." % (
                        row["level"], 100 * row["mean"], ai))
                QApplication.processEvents()
                time.sleep(0.03)
        finally:
            dlg.setValue(n + 1)
            dlg.close()
            if handle is not None:
                handle.stop()
            if bal_handle is not None:
                bal_handle.stop()
        if job["error"]:
            QMessageBox.warning(self, title, "The picture was not taken: %s" % job["error"])
            return None
        if job["stop"] or job["image"] is None:
            self.statusBar().showMessage("Stopped - no picture taken", 8000)
            return None
        result = job["result"]
        if result is not None:
            msg = light_tuning.describe(result) + " Every run of this job uses it."
            level = int(result["level"])
        else:
            msg = ("Picture taken as the camera sees it - Vision Station does not control "
                   "the light. Settings > Light lets the AI choose the light.")
            level = None
        if job["balance"]:
            self._keep_balance(camera)
        return {"image": job["image"], "level": level, "result": result, "message": msg,
                "mono": None}

    def _keep_balance(self, camera):
        """The white balance the AI reference image set, held: automatic
        balance off in the camera's settings, so it does not drift in the
        dark between runs."""
        from vision_station.ui.settings_dialog import (load_camera_settings,
                                                       save_camera_settings)
        from vision_station.hardware.camera import CameraSettings
        qs = self._settings()
        s = (load_camera_settings(qs, camera) or CameraSettings()).copy()
        s.auto_white_balance = False
        s.neutral_balance = False
        save_camera_settings(qs, camera, s)
        apply_camera_settings(camera, s)

    def _first_picture(self, title, image_path=None, source=None):
        """The first picture of a good part for a new application or job:
        (image, extra recipe keys) or (None, None)."""
        if image_path:
            source = "file"
        if source is None:
            source = self.choose_reference_source(title)
        if source is None:
            return None, None
        if source == "camera":
            got = self.ai_reference(title)
            if got is None:
                return None, None
            extra = {"image_source": {"mode": "camera", "paths": [], "folder": "",
                                      "camera": 0}}
            if got.get("mono"):
                extra["image_source"]["mono"] = got["mono"]
            if got["level"] is not None:
                extra["light_level"] = got["level"]
            self.statusBar().showMessage(got["message"], 15000)
            return got["image"], extra
        image_path = image_path or self._ask_image(
            "Picture of a GOOD part - every part is compared with it")
        img = self._read_reference(image_path, title)
        return (img, {}) if img is not None else (None, None)

    def new_application(self, name=None, image_path=None, source=None):
        """A new application: a job folder with a first job made from a
        picture of a good part -- a file, or the AI reference image from the
        camera (source 'file' | 'camera'; asked when None)."""
        from vision_station.core import job as jobmod
        if name is None:
            name, ok = QInputDialog.getText(self, "New application", "Name:")
            if not ok or not name.strip():
                return None
        img, extra = self._first_picture("New application", image_path, source)
        if img is None:
            return None
        base = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_") or "app"
        app_id, n = base, 2
        while os.path.exists(os.path.join(self.root, app_id)):
            app_id, n = "%s_%d" % (base, n), n + 1
        j = jobmod.create_job(os.path.join(self.root, app_id), name.strip())
        j.add_side("job_1", "Job 1", img, self._starter_tools(img))
        extra = self._advanced_request(extra)
        if extra:
            write_recipe_keys(os.path.join(self.root, app_id, "sides", "job_1",
                                           "project.json"), extra)
        self.station.store.cache.clear()
        self.open_application(app_id)
        return app_id

    def add_job(self, name=None, image_path=None, source=None):
        from vision_station.core import job as jobmod
        if not self.station.is_job_app():
            QMessageBox.information(self, "Add job", "This application is a single model. "
                                    "Use File > New Application for a multi-job application.")
            return None
        if self.dirty and not self._confirm_discard(revert=False):
            return None
        if name is None:
            name, ok = QInputDialog.getText(self, "Add job", "Job name:")
            if not ok or not name.strip():
                return None
        img, extra = self._first_picture("Add job", image_path, source)
        if img is None:
            return None
        j = jobmod.Job(self.station.app_folder())
        sid = j.unique_side_id(name)
        j.add_side(sid, name.strip(), img, self._starter_tools(img))
        extra = self._advanced_request(extra)
        if extra:
            write_recipe_keys(os.path.join(self.station.app_folder(), "sides", sid,
                                           "project.json"), extra)
        self.dirty.clear()
        self.open_application(self.station.app_id)
        return "%s/%s" % (self.station.app_id, sid)

    def rename_job(self, pid, name=None):
        from vision_station.core import job as jobmod
        if name is None:
            name, ok = QInputDialog.getText(self, "Rename job", "Name:", text=self.job_name(pid))
            if not ok or not name.strip():
                return
        jobmod.Job(self.station.app_folder()).rename_side(pid.split("/", 1)[1], name.strip())
        for j in self.station.jobs:
            if j["pid"] == pid:
                j["name"] = name.strip()
        self._refresh_jobs()
        self._build_displays()

    def remove_job(self, pid, confirm=True):
        from vision_station.core import job as jobmod
        if confirm and QMessageBox.question(
                self, "Remove job", "Remove %s from the application?\n\nIts folder is kept "
                "under removed/ and can be moved back." % self.job_name(pid)) != QMessageBox.Yes:
            return False
        self.dirty.discard(pid)
        # Its results database is open, and Windows will not move a folder
        # holding an open file: removing a job failed with "Access is denied".
        self.station.stop()
        self.station.wait(30)
        lg = self.station.logs.pop(pid, None)
        if lg is not None:
            try:
                lg.flush()
                lg.close()
            except Exception:
                pass
        src = self.station.sources.get(pid)
        if src is not None:
            src.stop_camera()
        for te in self.tool_editors_for(pid):
            te.close()
        ed = self._job_editors.get(pid)
        if ed is not None:
            ed.close()
        jobmod.Job(self.station.app_folder()).remove_side(pid.split("/", 1)[1])
        self.station.store.cache.pop(pid, None)
        self.open_application(self.station.app_id)
        return True

    # ------------------------------------------------------------ the line
    def comms_settings(self):
        spec, _save = self._output_spec_holder()
        return dict({"enabled": False, "port": 3000, "broadcast": False},
                    **(spec.get("comms") or {}))

    def communication(self, enabled=None, port=None, broadcast=None):
        """TCP/IP settings for the line -- a PLC or robot triggers inspections
        and reads results. Saved with the application; the server starts
        whenever it is opened."""
        spec, save = self._output_spec_holder()
        c = self.comms_settings()
        if enabled is None and port is None and broadcast is None:
            dlg = CommunicationDialog(self, c)
            if not dlg.exec():
                return None
            c = dlg.values()
        else:
            if enabled is not None:
                c["enabled"] = bool(enabled)
            if port is not None:
                c["port"] = int(port)
            if broadcast is not None:
                c["broadcast"] = bool(broadcast)
        spec["comms"] = c
        save()
        self._start_line()
        return c

    def _start_line(self):
        from vision_station.core.comms import LineServer
        if self.line is not None:
            self.line.stop()
            self.line.deleteLater()
            self.line = None
        c = self.comms_settings() if self.station.app_id else {"enabled": False}
        if not c.get("enabled"):
            self.sComms.setText("")
            return None
        self.line = LineServer(self.station, c.get("port", 3000), c.get("broadcast", False))
        self.line.clientsChanged.connect(self._line_status)
        self.line.codeScanned.connect(lambda c: self.barcodeBar.accepted(c, running=False))
        if not self.line.start():
            self.sComms.setText("TCP %s: %s" % (c.get("port"), self.line.error))
            theme.set_tone(self.sComms, "fail")
            return self.line
        self._line_status(0)
        return self.line

    def _line_status(self, n):
        if self.line is None:
            return
        self.sComms.setText("TCP %d - %s" % (self.line.port, "%d connected" % n if n
                                                 else "waiting for the line"))
        # tones from the theme: a fixed grey was unreadable on the dark theme
        theme.set_tone(self.sComms, "pass" if n else "idle")

    # ------------------------------------------------------------ packages
    def _export_path(self, path, title, ext, filt):
        """Where to export to -- asked for unless given -- after saving unsaved
        edits: exports are built from the files on disk."""
        if not self.station.app_id:
            return None
        if self.dirty and not self._confirm_discard():
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, title, self.station.app_id + ext, filt)
        return path or None

    def export_package(self, path=None, include_training=False):
        """The whole application as one .vsp file your own Python code can run
        with vsruntime. Unsaved edits are saved first: the package is built
        from the files on disk."""
        from vision_station.core import package
        path = self._export_path(path, "Export package", ".vsp",
                                 "Vision Station package (*.vsp)")
        if not path:
            return None
        try:
            m = package.export_package(self.station.app_folder(), path, include_training)
        except Exception as e:
            QMessageBox.warning(self, "Export package", str(e))
            return None
        self.statusBar().showMessage("Exported %s - %d job(s), %d kB" % (
            os.path.basename(path), len(m["jobs"]), os.path.getsize(path) // 1024), 6000)
        return m

    def export_job_file(self, path=None):
        """A Python file that runs every job on its taught images and shows or
        saves the results -- run it directly, or import it from other code.
        The package it runs is written beside it."""
        from vision_station.core import package
        path = self._export_path(path, "Export job file", ".py", "Python job file (*.py)")
        if not path:
            return None
        try:
            py, vsp, m = package.export_job_file(self.station.app_folder(), path)
        except Exception as e:
            QMessageBox.warning(self, "Export job file", str(e))
            return None
        self.statusBar().showMessage("Exported %s and %s - run: python %s" % (
            os.path.basename(py), os.path.basename(vsp), os.path.basename(py)), 8000)
        return py

    # ---- where results go ---------------------------------------------
    def _output_spec_holder(self):
        """(spec dict, save function) for the application's settings: job.json
        for a multi-job application, project.json for a single model."""
        from vision_station.core import job as jobmod
        if self.station.is_job_app():
            j = jobmod.Job(self.station.app_folder())
            return j.spec, j.save
        p = self.station.projects[self.station.app_id]
        return p.spec, lambda: p.save(p.spec)

    def result_output(self, folder=None, show=None, only_failures=None):
        """Where an exported job file puts its results: a folder of annotated
        images, a window, or both -- saved with the application. Set in
        Settings > Results (File > Result Output opens it); called with no
        values, it opens that page. It had a dialog of its own as well, the
        same three settings twice."""
        spec, save = self._output_spec_holder()
        o = dict(spec.get("output") or {"folder": "results", "show": True,
                                        "only_failures": False})
        if folder is None and show is None and only_failures is None:
            self.show_settings("results")
            return None
        if folder is not None:
            o["folder"] = folder
        if show is not None:
            o["show"] = bool(show)
        if only_failures is not None:
            o["only_failures"] = bool(only_failures)
        spec["output"] = o
        save()
        return o

    def import_package(self, path=None):
        from vision_station.core import package
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Import package", "",
                                                  "Vision Station package (*.vsp)")
            if not path:
                return None
        try:
            app_id = package.import_package(path, self.root)
        except Exception as e:
            QMessageBox.warning(self, "Import package", str(e))
            return None
        self.station.store.cache.clear()
        self.open_application(app_id)
        return app_id

    # ------------------------------------------------------------ posted items
    def configure_posted(self):
        dlg = PostedItemsDialog(self)
        dlg.show()
        self._posted_dialog = dlg
        return dlg

    # ------------------------------------------------------------ editors
    def _locked_out(self):
        if getattr(self, "operator_mode", False):
            self.statusBar().showMessage("Operator Mode - editing is locked", 4000)
            return True
        return False

    def open_job_editor(self, pid):
        if pid not in self.station.projects or self._locked_out():
            return None
        ed = self._job_editors.get(pid)
        if ed is None:
            ed = JobEditor(self, pid)
            self._job_editors[pid] = ed
        self.present(ed, (1100, 700))
        return ed

    def open_tool_edit(self, pid, tid):
        if self._locked_out():
            return None
        key = (pid, tid)
        ed = self._tool_editors.get(key)
        if ed is None:
            ed = ToolEditControl(self, pid, tid)
            self._tool_editors[key] = ed
        self._tool_return = pid
        self.present(ed, (1000, 660))
        return ed

    def posted_label(self, item):
        """'app/side/tool.key' as 'Side name - tool.key' for a column header."""
        owner, _, ref = item.rpartition("/")
        return "%s - %s" % (self.job_name(owner), ref) if len(self.station.jobs) > 1 else ref

    def job_name(self, pid):
        return next((j["name"] for j in self.station.jobs if j["pid"] == pid), pid)

    def check_name(self, pid, tid):
        """A check's name as the simple view shows it; the id when the tool
        is gone from the recipe."""
        p = self.station.projects.get(pid)
        t = find_tool(p.spec.get("tools", []), tid) if p else None
        return simple.short_name(t) if t else tid

    def tool_editors_for(self, pid, exclude=None):
        return [te for (jp, _t), te in list(self._tool_editors.items())
                if jp == pid and te is not exclude]

    # ------------------------------------------------------------ workspace
    def show_settings(self, page="general"):
        """Settings: the camera (address and every parameter) and where each
        part's picture is saved. Non-blocking, so Live Camera can be open
        beside it while adjusting."""
        from vision_station.ui.settings_dialog import SettingsDialog
        dlg = getattr(self, "_settings_dialog", None)
        try:
            alive = dlg is not None and dlg.isVisible()
        except RuntimeError:
            alive = False
        if not alive:
            dlg = SettingsDialog(self, self._settings(), open_camera=open_camera,
                                 on_camera=apply_camera_settings,
                                 on_results=self.station.result_images.configure,
                                 page=page, qb=self)
            dlg.setAttribute(Qt.WA_DeleteOnClose, True)
            self._settings_dialog = dlg
        else:
            dlg.show_page(page)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        return dlg

    def show_live_camera(self, index=None):
        """The Live Camera window: one, brought forward if already open."""
        from vision_station.ui.live_view import LiveCameraWindow
        w = getattr(self, "_live_camera", None)
        try:
            alive = w is not None and w.isVisible()
        except RuntimeError:                  # deleted with its window
            alive = False
        if not alive:
            w = LiveCameraWindow(open_camera, self._settings(),
                                 index=0 if index is None else index, parent=self)
            w.setAttribute(Qt.WA_DeleteOnClose, True)
            self._live_camera = w
            # aiming and focusing need the part lit: the light is on while
            # Live Camera is open
            light = self.station.light
            light.hold("live")
            w.destroyed.connect(lambda *_: light.release("live"))
        elif index is not None and w.index.value() != index:
            w.index.setValue(index)
        w.show()
        w.raise_()
        w.activateWindow()
        return w

    def present(self, widget, size=None):
        """Show a window inside the workspace, bringing it forward if it is
        already there."""
        sub = widget.parent() if isinstance(widget.parent(), QMdiSubWindow) else None
        tabbed = self._fills_workspace()
        if sub is None:
            sub = self.mdi.addSubWindow(widget)
            sub.setAttribute(Qt.WA_DeleteOnClose, True)
            if isinstance(widget, ToolEditControl) and not self.advanced:
                self._sync_tabs()                 # the tabs come back for it
            elif not self.advanced:
                _set_frame(sub, False)
            kind = ("build" if isinstance(widget, JobEditor) else
                    "history" if isinstance(widget, HistoryView) else None)
            if isinstance(widget, ToolEditControl):
                t = widget.spec()
                sub.setWindowIcon(icons.tool_icon(t["type"]) if t else self.windowIcon())
            else:
                sub.setWindowIcon(icons.icon(kind) if kind else self.windowIcon())
            if size:
                avail = self.mdi.viewport().size()
                sub.resize(min(size[0], max(400, avail.width() - 20)),
                           min(size[1], max(300, avail.height() - 20)))
            if not tabbed and self.runSub.isMaximized():
                self.runSub.showNormal()         # so new windows are not hidden behind it
                self.mdi.tileSubWindows()
        if tabbed:
            sub.showMaximized()
        else:
            sub.show()
        widget.show()
        self.mdi.setActiveSubWindow(sub)
        if tabbed:
            self._pin_run_tab()
        self._sync_tabs()
        return sub

    def show_toolbox(self):
        self.toolboxDock.show()
        self.toolboxDock.raise_()

    def _toolbox_add(self, ty, draw=False):
        """Add to the active job -- the job editor or tool edit control in
        front -- or, with none open, the only job there is. draw: its area is
        drawn on the picture first (the Toolbox used by hand); otherwise it
        is added at once with a default area."""
        if self._locked_out():
            return None
        pid = self._active_job()
        if pid is None and len(self.station.jobs) == 1:
            pid = self.station.jobs[0]["pid"]
        if pid is None:
            self.statusBar().showMessage("Open a job (double-click it in Jobs) to add tools to it",
                                         5000)
            return None
        ed = self.open_job_editor(pid)
        return ed.begin_add(ty) if draw else ed.add_tool(ty)

    def active_window(self):
        sub = self.mdi.activeSubWindow()
        return sub.widget() if sub is not None else None

    def _active_job(self):
        w = self.active_window()
        return getattr(w, "pid", None) if isinstance(w, (JobEditor, ToolEditControl)) else None

    def key_run_once(self):
        """F5 / Run: the active job when an editor is active, the application
        otherwise -- inspected (Normal run), or its AI model taught (Model
        teaching run)."""
        pid = self._active_job()
        if self.run_mode == "teach":
            return self.teaching_run([pid] if pid else None)
        self.run_job(pid) if pid else self.run_once()

    def key_run_continuous(self):
        pid = self._active_job()
        ed = self._job_editors.get(pid) if pid else None
        if ed is not None:
            ed.aCont.setChecked(not ed.aCont.isChecked())
            ed.toggle_continuous()
        else:
            self.aRunCont.setChecked(not self.aRunCont.isChecked())
            self.toggle_continuous()

    def key_save(self):
        pid = self._active_job()
        self.save_job(pid) if pid else self.save_all()

    # ------------------------------------------------------------ undo / redo
    def _reset_history(self):
        from vision_station.core.history import History
        import copy
        self._hist = {pid: History() for pid in self.station.projects}
        self._snap = {pid: copy.deepcopy(p.spec) for pid, p in self.station.projects.items()}
        self._update_edit_actions()

    @staticmethod
    def _describe_change(old, new):
        """A short name for what changed, for the Undo / Redo menu entries."""
        a = [t["id"] for t in old.get("tools", [])]
        b = [t["id"] for t in new.get("tools", [])]
        if len(b) > len(a):
            return "add %s" % next((i for i in b if i not in a), b[-1])
        if len(b) < len(a):
            return "delete %s" % next((i for i in a if i not in b), a[-1])
        if a != b:
            if sorted(a) == sorted(b):
                return "reorder tools"
            return "rename %s" % next((i for i in a if i not in b), "tool")
        olds = {t["id"]: t for t in old.get("tools", [])}
        for t in new.get("tools", []):
            o = olds.get(t["id"])
            if o != t:
                geo = ("roi", "angle", "points", "shape")
                po, pn = o.get("params", {}), t.get("params", {})
                if any(po.get(k) != pn.get(k) for k in geo) and \
                        all(po.get(k) == pn.get(k) for k in set(po) | set(pn) if k not in geo):
                    return "move %s region" % t["id"]
                return "change %s" % t["id"]
        return "change job settings"

    def _record(self, pid, label=None):
        """After an edit: if the recipe differs from the last recorded state,
        that state goes on the undo stack. Recorded here, where every edit path
        already ends, rather than at each of the dozens of places that change a
        recipe -- one of which would sooner or later be missed."""
        import copy
        p = self.station.projects.get(pid)
        hist = getattr(self, "_hist", {}).get(pid)
        if p is None or hist is None:
            return
        old = self._snap.get(pid)
        if old == p.spec:
            return
        hist.push(label or self._describe_change(old or {}, p.spec), old)
        self._snap[pid] = copy.deepcopy(p.spec)
        self._update_edit_actions()

    def _undo_target(self):
        pid = self._active_job()
        if pid is None and len(self.station.jobs) == 1:
            pid = self.station.jobs[0]["pid"]
        return pid

    def undo(self, pid=None):
        return self._step(pid or self._undo_target(), "undo")

    def redo(self, pid=None):
        return self._step(pid or self._undo_target(), "redo")

    def _step(self, pid, which):
        import copy
        hist = getattr(self, "_hist", {}).get(pid)
        if hist is None or not (hist.can_undo() if which == "undo" else hist.can_redo()):
            self.statusBar().showMessage("Nothing to %s" % which, 2000)
            return None
        p = self.station.projects[pid]
        snap, label = (hist.undo if which == "undo" else hist.redo)(p.spec)
        p.spec = snap
        self._snap[pid] = copy.deepcopy(snap)
        ids = {t["id"] for t in snap.get("tools", [])}
        for te in self.tool_editors_for(pid):
            if te.tool_id not in ids:
                te.close()                     # its tool no longer exists
            else:
                te.props.tool = None           # editors follow the restored recipe objects
        self.dirty.add(pid)
        self.job_changed(pid)
        self._update_edit_actions()
        self.statusBar().showMessage("%s %s" % ("Undid" if which == "undo" else "Redid", label),
                                     3000)
        return label

    def _update_edit_actions(self):
        import shiboken6
        if not hasattr(self, "aUndo") or not shiboken6.isValid(self.aUndo):
            return
        hist = getattr(self, "_hist", {}).get(self._undo_target())
        u = bool(hist and hist.can_undo())
        r = bool(hist and hist.can_redo())
        if not getattr(self, "operator_mode", False):
            self.aUndo.setEnabled(u)
            self.aRedo.setEnabled(r)
        self.aUndo.setText("Undo %s" % hist.undo_label() if u else "Undo")
        self.aRedo.setText("Redo %s" % hist.redo_label() if r else "Redo")

    def mark_dirty(self, pid):
        self._record(pid)
        self.dirty.add(pid)

    def job_changed(self, pid, source=None):
        """A job's recipe changed: redraw every window that shows it."""
        ed = self._job_editors.get(pid)
        if ed is not None:
            ed.refresh()
        for te in self.tool_editors_for(pid, exclude=source):
            te.refresh()
        self._refresh_jobs()

    def save_job(self, pid):
        p = self.station.projects.get(pid)
        if p is None:
            return
        try:
            p.save(p.spec)
        except OSError as e:
            # a read-only or full disk: say so, and keep the edits marked unsaved
            QMessageBox.warning(self, "Save", "Could not save %s:\n%s" % (self.job_name(pid), e))
            return False
        self.dirty.discard(pid)
        self.reload_job(pid)
        self.statusBar().showMessage("Saved %s" % self.job_name(pid), 3000)
        return True

    def reload_job(self, pid):
        """Re-read a job from disk -- after saving, or after training wrote a
        model and a new threshold into it. Editors hold the recipe objects of
        the project they were opened on, so they are pointed at the new ones."""
        self.station.reload_job(pid)
        self._record(pid)                  # training may have changed it (a tuned threshold)
        for te in self.tool_editors_for(pid):
            te.props.tool = None
        self.job_changed(pid)

    def save_all(self):
        """True when everything is saved."""
        ok = True
        for pid in list(self.dirty):
            ok = (self.save_job(pid) is not False) and ok
        if ok and not self.dirty:
            self.statusBar().showMessage("All changes saved", 3000)
        return ok

    def _confirm_discard(self, revert=True):
        """Save, discard or cancel unsaved edits. Discard really discards: the
        jobs are read back from disk. It used to leave the edits in place and
        the jobs still marked unsaved -- so after 'Discard' on the way into
        Operator Mode the line ran the recipe that had just been thrown away.
        revert=False when the application is about to be reloaded or closed
        anyway."""
        names = sorted(self.job_name(pid) for pid in self.dirty)
        choice = confirm_unsaved(self, ", ".join(names) if names else "this application")
        if choice == "save":
            return self.save_all()               # a save that failed goes no further
        elif choice == "discard" and revert:
            self.discard_changes()
        return choice != "cancel"

    def discard_changes(self):
        """Throw away unsaved edits: every changed job is read back from disk.
        Undo can still bring the edits back."""
        for pid in list(self.dirty):
            self.dirty.discard(pid)
            if pid in self.station.projects:
                self.reload_job(pid)
        self._refresh_jobs()

    def _set_parallel(self, on):
        self.station.set_parallel(on)
        self._settings().setValue("parallel_jobs", bool(on))     # this PC's choice

    def find_picture(self):
        """The Find picture window: saved result pictures, searched as the
        barcode is typed or scanned. One window; opening it again brings it
        to the front with the search box ready."""
        from vision_station.ui.finder_dialog import FinderDialog
        dlg = getattr(self, "_finder", None)
        if dlg is None:
            dlg = FinderDialog(self, self.station.pictures, self.station.picture_folders)
            dlg.setAttribute(Qt.WA_DeleteOnClose)
            dlg.destroyed.connect(lambda *_: setattr(self, "_finder", None))
            self._finder = dlg
        else:
            dlg.refresh()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        dlg._focus_query()
        return dlg

    def show_history(self):
        h = getattr(self, "_history", None)
        if h is None:
            h = HistoryView(self)
            self._history = h
            h.destroyed.connect(lambda *_: setattr(self, "_history", None))
        else:
            h.refresh()
        self.present(h, (760, 560))
        return h

    # ------------------------------------------------------------ operator mode
    OPERATOR_ALLOWED = {"Exit", "Run Once   (F5)", "Run Continuously   (F6)", "Stop",
                        "Run", "Continuous", "Sidebar",
                        "Production History  ", "Run View  ",
                        "Production History", "Operator Mode", "Leave Operator Mode",
                        "Results queue: all runs", "Results queue: failures only",
                        "Run View", "Cascade", "Tile", "Next window", "Run Once",
                        "Run Continuously", "F5", "F6", "Ctrl+1", "Ctrl+3",
                        "Getting started", "Keyboard shortcuts", "Check installation...",
                        "Find picture", "Find Picture...   (Ctrl+F)"}

    def _build_actions(self):
        """Every action that changes the application -- disabled for operators.
        Taken explicitly from the menu bar, QuickBuild's toolbar and its window
        shortcuts: picking actions by the window they belong to missed every
        menu item (a menu is a window of its own to Qt) and locked the display
        controls -- zoom, fit -- that operators need."""  
        allowed = self.OPERATOR_ALLOWED
        # The menus are walked from self._menus, never through QAction.menu():
        # on PySide6 6.11 that returns a second Python wrapper for the menu,
        # and when it is garbage-collected every action wrapper of the menu
        # -- self.aUndo included -- is invalidated. Operator Mode then failed
        # with "Internal C++ object already deleted" on the first action it
        # tried to lock.
        submenu_actions = {m.menuAction() for m in self._menus}
        cands = []
        for menu in self._menus:
            cands.extend(a for a in menu.actions()
                         if a not in submenu_actions and a.text() and not a.isSeparator())
        bar = self.findChild(QToolBar, "qbToolbar")
        if bar is not None:
            cands.extend(a for a in bar.actions() if a.text())
        cands.extend(a for a in self.actions() if a.text())
        # Only what can be seen: Qt reports a hidden action as disabled, so a
        # toolbar button the simple view has put away would never read as
        # unlocked again. (Its twin in the More menu is locked and unlocked.)
        return [a for a in cands if a.text() not in allowed and a.isVisible()]

    def toggle_operator_mode(self):
        if getattr(self, "operator_mode", False):
            self.exit_operator_mode()
        else:
            self.enter_operator_mode()

    def enter_operator_mode(self, operator=None):
        """Run and watch, but not change. VisionPro separates building a job
        from running it; without this anyone at the line could double-click a
        tool and move its region. The operator's name goes into the log."""
        if self.dirty and not self._confirm_discard():
            return False
        if operator is None:
            operator, ok = QInputDialog.getText(self, "Operator Mode", "Operator name:")
            if not ok:
                return False
        self.close_editors()
        self.station.operator = (operator or "").strip()
        self.operator_mode = True
        self._locked = self._build_actions()
        for a in self._locked:
            a.setEnabled(False)
        self._toolbox_was = self.toolboxDock.isVisible()
        self.toolboxDock.hide()
        for tile in self._tiles.values():
            tile.edit.hide()
        self.modeSwitch.set_locked("build", True, "Operator Mode - building is locked. "
                                   "More > Leave Operator Mode unlocks it.")
        self.show_run_view()
        self.aOperator.setText("Leave Operator Mode")
        self.aOperator.setEnabled(True)
        self.setWindowTitle(self.windowTitle().split("   [")[0] + "   [Operator: %s]" % (
            self.station.operator or "unnamed"))
        self.statusBar().showMessage("Operator Mode - running only; editing is locked", 6000)
        return True

    def exit_operator_mode(self, password=None):
        pw_hash = self._settings().value("supervisor_password", "")
        if pw_hash:
            if password is None:
                from PySide6.QtWidgets import QLineEdit
                password, ok = QInputDialog.getText(self, "Leave Operator Mode",
                                                    "Supervisor password:", QLineEdit.Password)
                if not ok:
                    return False
            if not _check_password(password, pw_hash):
                QMessageBox.warning(self, "Leave Operator Mode", "That password is not right.")
                return False
        self.operator_mode = False
        for a in getattr(self, "_locked", []):
            a.setEnabled(True)
        if getattr(self, "_toolbox_was", True) and self.advanced:
            self.toolboxDock.show()
        for tile in self._tiles.values():
            tile.edit.show()
        self.modeSwitch.set_locked("build", False)
        self.aOperator.setText("Operator Mode")
        self.station.operator = ""
        self.setWindowTitle(self.windowTitle().split("   [")[0])
        return True

    def set_supervisor_password(self, password=None):
        """Stored as a salted hash, never as the password."""
        if password is None:
            from PySide6.QtWidgets import QLineEdit
            password, ok = QInputDialog.getText(self, "Supervisor password",
                                                "New password (empty to remove):",
                                                QLineEdit.Password)
            if not ok:
                return False
        self._settings().setValue("supervisor_password",
                                  _hash_password(password) if password else "")
        return True

    def close_editors(self):
        for ed in list(self._tool_editors.values()) + list(self._job_editors.values()):
            ed.close()

    def classic_editor(self):
        from vision_station.ui import classic_window as classic
        w = classic.MainWindow()
        w.show()
        self._classic = w

    # ------------------------------------------------------------ layout
    def _settings(self):
        return QSettings("VisionStation", "QuickBuild")

    def _restore_layout(self):
        s = self._settings()
        geo, state = s.value(self.LAYOUT_KEYS[0]), s.value(self.LAYOUT_KEYS[1])
        if geo is not None:
            self.restoreGeometry(geo)
        if state is None or not self.restoreState(state, LAYOUT_VERSION):
            self.restoreState(self._default_state, LAYOUT_VERSION)
        par = s.value("parallel_jobs", False)
        self.aParallel.setChecked(par in (True, "true", "True", 1, "1"))

    def reset_layout(self):
        self.restoreState(self._default_state, LAYOUT_VERSION)
        for d in (self.jobsDock, self.toolboxDock, self.queueDock):
            d.setFloating(False)
        for d in (self.jobsDock, self.queueDock):
            d.show()
        self.toolboxDock.hide()
        self.resizeDocks([self.jobsDock], [240], Qt.Horizontal)
        self.resizeDocks([self.queueDock], [150], Qt.Vertical)

    def closeEvent(self, e):
        if self.dirty and not self._confirm_discard(revert=False):
            e.ignore()
            return
        self._closing = True
        if getattr(self, "_queue_stood_aside", False):
            self.queueDock.show()          # saved as the person left it, not as an editor hid it
        s = self._settings()
        s.setValue(self.LAYOUT_KEYS[0], self.saveGeometry())
        s.setValue(self.LAYOUT_KEYS[1], self.saveState(LAYOUT_VERSION))
        for ed in list(self._tool_editors.values()) + list(self._job_editors.values()):
            ed.close()
        if self.line is not None:
            self.line.stop()
        self.station.shutdown()
        super().closeEvent(e)


def _crash_log():
    """Keep a record of why the application stopped. A native library that
    aborts -- Qt, OpenCV, an inference or OCR engine -- ends the process with no
    Python traceback and no message; the console closes with it. Now:

      logs/crash.log   Python's stack in every thread at a fatal error or abort
                       (faulthandler), Qt's own fatal and critical messages,
                       and any Python exception that reached the event loop.

    A Python exception in a window's code is logged and shown in the status
    bar; it does not take the application down."""
    import datetime
    import faulthandler
    import traceback
    folder = paths.LOGS
    try:
        os.makedirs(folder, exist_ok=True)
        f = open(os.path.join(folder, "crash.log"), "a", encoding="utf-8", buffering=1)
    except OSError:
        return None
    f.write("\n==== started %s  (Python %s, PID %d)\n" % (
        datetime.datetime.now().isoformat(timespec="seconds"), sys.version.split()[0], os.getpid()))
    faulthandler.enable(file=f, all_threads=True)

    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    def qt_messages(kind, ctx, msg):
        if kind in (QtMsgType.QtFatalMsg, QtMsgType.QtCriticalMsg):
            f.write("[Qt %s] %s\n" % ("FATAL" if kind == QtMsgType.QtFatalMsg else "critical", msg))
            f.flush()
        if kind != QtMsgType.QtDebugMsg:
            sys.__stderr__ and sys.__stderr__.write(msg + "\n")
    qInstallMessageHandler(qt_messages)

    def excepthook(et, ev, tb):
        text = "".join(traceback.format_exception(et, ev, tb))
        f.write("[exception %s]\n%s" % (datetime.datetime.now().strftime("%H:%M:%S"), text))
        f.flush()
        sys.__stderr__ and sys.__stderr__.write(text)
        try:
            w = next((w for w in QApplication.topLevelWidgets() if isinstance(w, QuickBuild)), None)
            if w is not None:
                w.statusBar().showMessage("Something went wrong: %s: %s  (details in logs/crash.log)"
                                          % (et.__name__, ev), 10000)
        except Exception:
            pass
    sys.excepthook = excepthook
    import threading
    threading.excepthook = lambda a: excepthook(a.exc_type, a.exc_value, a.exc_traceback)
    return f


def main(argv=None):
    argv = argv if argv is not None else sys.argv
    paths.first_run()
    _crash_log()
    if sys.platform == "win32":
        try:                                 # the taskbar shows this app's icon, not Python's
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("VisionStation.QuickBuild")
        except Exception:
            pass
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("Vision Station")
    app.setWindowIcon(icons.app_icon())
    apply_theme(app)
    root = argv[1] if len(argv) > 1 else None
    w = QuickBuild(root)
    _hand_back_on_console_close(w.station.light)
    w.show()
    return app.exec()


def _hand_back_on_console_close(light):
    """The terminal the application was started from closed (its X, Ctrl+C,
    Ctrl+Break, log-off, shutdown): the light is handed back first -- Windows
    gives a console program a few seconds for this. Closing the terminal
    used to end the program at once and leave the light as it had set it.
    VS Code's Stop button ends it with no warning at all; the controller's
    own setting is kept in the settings for that (hardware/light.py
    _take_home). Windows only; a no-op elsewhere."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    def handler(_event):
        try:
            light.shutdown()
        except Exception:
            pass
        return False                         # and then end as Windows would
    try:
        ctypes.windll.kernel32.SetConsoleCtrlHandler(handler, True)
    except Exception:
        return None
    _hand_back_on_console_close.keep = handler     # must outlive this call
    return handler


if __name__ == "__main__":
    sys.exit(main())
