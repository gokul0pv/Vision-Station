"""Vision Station runtime: run an exported application from your own Python code.

No GUI, no Qt. Load a package exported from QuickBuild (File > Export Package),
give it images, get back the verdict, every measurement, and the image with the
results drawn on it.

    import sys
    sys.path.insert(0, "panel.vsp")            # the package carries this runtime
    import vsruntime

    app = vsruntime.load("panel.vsp")
    print(app.jobs)                            # [('frt_top', 'Front top'), ...]

    result = app.run(cv2.imread("part.png"), job="Front top")
    print(result.verdict)                      # 'PASS' or 'FAIL'
    print(result.value("hole_spacing", "value_px"))
    print(result.failed_tools)

    cv2.imshow("result", result.annotated())   # the image with results drawn
    cv2.waitKey(0)

Every job at once, one image each:

    results = app.run_all({"Front top": img1, "Top right": img2})
    print(app.verdict(results))                # PASS only if every job passed

From the command line:

    python -m vision_station.core.vsruntime panel.vsp part.png --job "Front top" --show
    python -m vision_station.core.vsruntime panel.vsp part.png --save annotated.png

`load` also accepts an application folder (a job folder or a single model) in
place of a package. OpenCV and NumPy must be installed; a package whose tools
use trained models also needs openvino or onnxruntime, and text reading needs
an OCR engine -- the same requirements as the application.
"""
import hashlib
import json
import os
import sys
import zipfile

import cv2
import numpy as np

PACKAGE_FORMAT = "vision-station-package"
PACKAGE_VERSION = 1


# This loader stands alone at the top of a package, so it carries its own copy
# of imio's two functions: cv2.imread and imwrite fail silently on a Windows
# path holding a character outside the system code page.
def _imread(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError, TypeError):
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _imwrite(path, image):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", image)
    if ok:
        buf.tofile(path)
    return bool(ok)



# ================================================================ loading
def _cache_root():
    root = os.environ.get("VISION_STATION_CACHE") or os.path.join(
        os.path.expanduser("~"), ".vision_station", "packages")
    os.makedirs(root, exist_ok=True)
    return root


def _extract(path):
    """Unpack a package once, into a folder named by its content hash: loading
    the same package again reuses it, and a changed package never mixes with
    an old copy."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    dest = os.path.join(_cache_root(), h.hexdigest()[:16])
    done = os.path.join(dest, ".complete")
    if not os.path.exists(done):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            for n in names:              # refuse paths that would escape the folder
                target = os.path.realpath(os.path.join(dest, n))
                if not target.startswith(os.path.realpath(dest) + os.sep) and \
                        target != os.path.realpath(dest):
                    raise ValueError("unsafe path in package: %s" % n)
            z.extractall(dest)
        with open(done, "w") as f:
            f.write("ok")
    return dest


def read_manifest(path):
    with zipfile.ZipFile(path) as z:
        m = json.loads(z.read("manifest.json").decode("utf-8"))
    if m.get("format") != PACKAGE_FORMAT:
        raise ValueError("%s is not a Vision Station package" % path)
    if int(m.get("version", 0)) > PACKAGE_VERSION:
        raise ValueError("%s was made by a newer Vision Station (package version %s); "
                         "update the runtime" % (path, m.get("version")))
    return m


def _install_root():
    """The folder holding the vision_station package this file is part of,
    or None for the copy at the top of a .vsp, which belongs to none."""
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(here) == "core" and \
            os.path.basename(os.path.dirname(here)) == "vision_station":
        return os.path.dirname(os.path.dirname(here))
    return None


def _use_runtime(folder):
    """Make the vision_station package importable from `folder` (a package's
    runtime/ or an installed application), refusing to mix with a different,
    incomplete copy already imported under the same name."""
    folder = os.path.abspath(folder)
    loaded = sys.modules.get("vision_station")
    if loaded is not None:
        have = os.path.dirname(os.path.dirname(os.path.abspath(getattr(loaded, "__file__", ""))))
        if have != folder and not os.path.exists(
                os.path.join(have, "vision_station", "core", "vsruntime.py")):
            raise ImportError("a different package named 'vision_station' is already "
                              "imported (%s); load Vision Station before it" % have)
        return
    if folder not in sys.path:
        sys.path.insert(0, folder)


def load(path):
    """An Application from a .vsp package, a job folder or a model folder."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        manifest = read_manifest(path)
        root = _extract(path)
        runtime = os.path.join(root, "runtime")
        _use_runtime(runtime if os.path.isdir(runtime) else
                     (_install_root() or os.path.dirname(os.path.abspath(__file__))))
        return Application(os.path.join(root, "app"), manifest)
    _use_runtime(_install_root() or os.path.dirname(os.path.abspath(__file__)))
    return Application(path, None)


# ================================================================ results
class Result:
    """One job's inspection: verdict, per-tool results, and the image."""

    def __init__(self, job_id, job_name, raw, image):
        self.job_id, self.job_name = job_id, job_name
        self.raw = raw
        self.image = image
        self.tools = {t["tool_id"]: t for t in raw["tools"]}

    verdict = property(lambda self: self.raw["verdict"])
    passed = property(lambda self: self.raw["verdict"] == "PASS")
    total_ms = property(lambda self: float(self.raw.get("total_ms", 0.0)))
    failed_tools = property(lambda self: [t["tool_id"] for t in self.raw["tools"]
                                          if t["status"] != "pass"])
    error = property(lambda self: self.raw.get("error", ""))

    def value(self, tool_id, key, default=None):
        """A measurement, e.g. value("hole_spacing", "value_px"). 'score',
        'status' and 'message' work for every tool."""
        t = self.tools.get(tool_id)
        if t is None:
            return default
        if key in ("score", "status", "message"):
            return t.get(key, default)
        return (t.get("values") or {}).get(key, default)

    def values(self, tool_id):
        """Everything one tool measured, as a dict -- {} if it did not run."""
        t = self.tools.get(tool_id)
        return dict((t or {}).get("values") or {})

    def pose(self, fixture_id=None):
        """Where the part is: the fixture's shift, angle and full transform
        (reference -> this image), from the named Locate tool or the first one.
        None when the job has no fixture or it did not find the part.

            p = result.pose()
            robot.move(p["x"], p["y"], p["angle"])
        """
        for tid, t in self.tools.items():
            if t.get("type") not in ("align", "fixture_npoint") or (fixture_id and tid != fixture_id):
                continue
            v = t.get("values") or {}
            if t["status"] != "pass" or not v.get("matrix"):
                return None
            return {"tool": tid, "x": v.get("shift_x_px"), "y": v.get("shift_y_px"),
                    "angle": v.get("angle_deg"), "matrix": v["matrix"],
                    "score": t.get("score")}
        return None

    def to_reference(self, x, y, fixture_id=None):
        """An image point in the part's own coordinates (where it lies on the
        reference part), through the fixture. (x, y) unchanged with no fixture."""
        p = self.pose(fixture_id)
        if p is None:
            return float(x), float(y)
        M = np.vstack([np.array(p["matrix"], float), [0, 0, 1]])
        q = np.linalg.inv(M) @ np.array([x, y, 1.0])
        return float(q[0]), float(q[1])

    def shapes(self, tool_id=None, include_regions=False, include_search=False):
        """What the tools found, as geometry in image coordinates:
        [{"tool", "kind": rect|poly|line|point|text, "points": [(x, y), ...],
          "status", "label"}]. Region outlines and search guides are left out
        unless asked for -- they show where a tool looked, not what it found."""
        out = []
        for tid, t in self.tools.items():
            if tool_id and tid != tool_id:
                continue
            for g in t.get("graphics", []):
                if g.get("role") == "region" and not include_regions:
                    continue
                if g.get("status") == "search" and not include_search:
                    continue
                out.append({"tool": tid, "kind": g.get("kind"),
                            "points": [tuple(map(float, q)) for q in g.get("points", [])],
                            "status": g.get("status"), "label": g.get("label", "")})
        return out

    def to_dict(self):
        """The whole result as plain data -- JSON-ready, no images -- for
        logging, a database, or sending to another system."""
        def clean(v):
            if isinstance(v, dict):
                return {str(k): clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [clean(x) for x in v]
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v
        return clean({"job": self.job_id, "job_name": self.job_name, "verdict": self.verdict,
                      "total_ms": self.total_ms, "failed_tools": self.failed_tools,
                      "error": self.error, "source": getattr(self, "source", None),
                      "pose": self.pose(),
                      "tools": {tid: {"type": t.get("type"), "status": t["status"],
                                      "score": t.get("score"), "message": t.get("message", ""),
                                      "ms": t.get("ms"), "values": t.get("values") or {}}
                                for tid, t in self.tools.items()}})

    def annotated(self, image=None, banner=True, labels=True, only_failures=False, search=False):
        """The image with the tools' graphics drawn and a verdict banner.
        labels=False draws the shapes without their text -- cleaner when many
        tools sit close together; only_failures=True draws just what failed."""
        return draw(image if image is not None else self.image, self.raw, banner,
                    title=self.job_name, labels=labels, only_failures=only_failures,
                    search=search)

    def show(self, window=None, wait=True):
        cv2.imshow(window or "Vision Station - %s" % self.job_name, self.annotated())
        if wait:
            cv2.waitKey(0)

    def __repr__(self):
        if self.error:
            return "<Result %s ERROR %s>" % (self.job_name, self.error)
        return "<Result %s %s %.0f ms%s>" % (self.job_name, self.verdict, self.total_ms,
                                            (" failed: " + ", ".join(self.failed_tools))
                                            if self.failed_tools else "")


class Application:
    def __init__(self, folder, manifest):
        from vision_station.core.engine import Project
        from vision_station.core import job as jobmod
        self.folder = folder
        self.manifest = manifest
        self._projects = {}
        if jobmod.is_job(folder):
            j = jobmod.Job(folder)
            self.name = j.name
            for s in j.sides:
                self._projects[s.side_id] = (s.name, Project(s.folder))
        else:
            p = Project(folder)
            self.name = p.spec.get("name", os.path.basename(folder))
            # the model's own id -- not the folder it happened to be unpacked into
            jid = (manifest or {}).get("jobs", [{}])[0].get("id") or p.spec.get("id") or p.id
            self._projects[jid] = (self.name, p)

    @property
    def jobs(self):
        return [(jid, name) for jid, (name, _p) in self._projects.items()]

    def _resolve(self, job):
        if job is None:
            if len(self._projects) != 1:
                raise ValueError("this application has %d jobs - say which: %s" % (
                    len(self._projects), ", ".join(n for _i, n in self.jobs)))
            return next(iter(self._projects))
        if job in self._projects:
            return job
        for jid, (name, _p) in self._projects.items():
            if name == job:
                return jid
        raise KeyError("no job %r - the jobs are: %s" % (job, ", ".join(
            "%s (%s)" % (n, i) for i, n in self.jobs)))

    def run(self, image, job=None):
        """Inspect one image with one job. image: a BGR array or a file path."""
        if isinstance(image, str):
            path, image = image, _imread(image)
            if image is None:
                raise IOError("could not read %s" % path)
        if image is None:
            raise ValueError("no image")
        jid = self._resolve(job)
        name, project = self._projects[jid]
        raw = project.inspect(image, source="runtime")
        return Result(jid, name, raw, image)

    def run_all(self, images):
        """{job: image} -> {job id: Result}."""
        return {r.job_id: r for r in (self.run(img, job) for job, img in images.items())}

    # ---- taught image sources -----------------------------------------
    def source(self, job):
        """Where a job's images come from, as taught in QuickBuild: an image
        file, a folder, a list of files, 'camera:N', or the reference image."""
        jid = self._resolve(job)
        _name, p = self._projects[jid]
        spec = p.spec.get("image_source") or {}
        mode = spec.get("mode", "reference")
        if mode == "files" and spec.get("paths"):
            return list(spec["paths"])
        if mode == "folder" and spec.get("folder"):
            return spec["folder"]
        if mode == "camera":
            return "camera:%d" % int(spec.get("camera", 0))
        return os.path.join(p.folder, p.spec.get("reference_image", "reference.png"))

    @staticmethod
    def _expand(source):
        """A source -> the images to inspect: every image of a folder, each
        file of a list, one frame from a camera."""
        if isinstance(source, (list, tuple)):
            return [s for x in source for s in Application._expand(x)]
        if isinstance(source, str) and source.startswith("camera:"):
            return [source]
        if isinstance(source, str) and os.path.isdir(source):
            exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            return sorted(os.path.join(source, f) for f in os.listdir(source)
                          if f.lower().endswith(exts))
        return [source]

    @staticmethod
    def _grab(item):
        if isinstance(item, str) and item.startswith("camera:"):
            import time
            from vision_station.hardware.camera import CameraSettings, CameraThread
            # "camera:0" is the first camera discovery finds; "camera:192.168.3.6"
            # (or a serial number) is that camera, found or not
            which = item.split(":", 1)[1].strip()
            settings = CameraSettings()
            if which.isdigit() and len(which) < 4:
                cam = CameraThread(index=int(which), settings=settings)
            else:
                settings.address = which
                cam = CameraThread(index=0, settings=settings)
            cam.start()
            try:
                t0 = time.time()
                while time.time() - t0 < 10.0:
                    frame = cam.get_frame()
                    if frame is not None:
                        return frame
                    time.sleep(0.02)
                raise IOError("the camera delivered no image")
            finally:
                cam.stop()
        img = _imread(item)
        if img is None:
            raise IOError("could not read %s" % item)
        return img

    def run_taught(self, sources=None, show=False, save_to=None, only_failures=False,
                   on_result=None, labels=True):
        """Run every job on its taught images -- or on `sources`, a
        {job: file | folder | 'camera:N'} override -- and deliver each result:
        show=True in a window, save_to=<folder> as annotated images,
        on_result=<function> to your own code. Returns every Result, in job
        order. A job whose image cannot be read gets a Result with verdict
        ERROR rather than stopping the others."""
        sources = dict(sources or {})
        out = []
        if save_to:
            os.makedirs(save_to, exist_ok=True)
        for jid, (name, _p) in self._projects.items():
            src = sources.get(name, sources.get(jid)) or self.source(jid)
            for item in self._expand(src):
                try:
                    res = self.run(self._grab(item), jid)
                except Exception as e:
                    res = Result(jid, name, {"verdict": "ERROR", "tools": [], "total_ms": 0.0,
                                             "error": str(e)}, None)
                res.source = item
                out.append(res)
                if only_failures and res.passed:
                    continue
                if on_result is not None:
                    on_result(res)
                if res.image is None:
                    continue
                if save_to:
                    stem = os.path.splitext(os.path.basename(str(item)))[0].replace(":", "_")
                    fname = "%s__%s__%s.jpg" % (_safe(name), stem, res.verdict)
                    _imwrite(os.path.join(save_to, fname), res.annotated(labels=labels))
                if show:
                    shown = res.annotated(labels=labels)
                    h, w = shown.shape[:2]
                    k = min(1.0, 1400.0 / w, 900.0 / h)
                    cv2.imshow("Vision Station", cv2.resize(shown, None, fx=k, fy=k)
                               if k < 1.0 else shown)
                    cv2.waitKey(0)
        return out

    def verdict(self, results):
        """PASS only if every job ran and passed; INCOMPLETE if one had no
        image. results: the dict run_all returns, or the list run_taught does."""
        results = results if isinstance(results, dict) else {i: r for i, r in enumerate(results)}
        if any(r.verdict == "ERROR" for r in results.values()):
            return "INCOMPLETE"
        done = {r.job_id for r in results.values()}
        if any(jid not in done for jid in self._projects):
            return "INCOMPLETE"
        return "PASS" if all(r.passed for r in results.values()) else "FAIL"

    def __repr__(self):
        return "<Application %s: %s>" % (self.name, ", ".join(n for _i, n in self.jobs))


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


# ================================================================ drawing
def draw(image, raw, banner=True, title="", labels=True, only_failures=False, search=False):
    """The result picture: every check outlined in the shape it was drawn,
    blue for OK and red for NG, with its result in words beside it, and the
    verdict in a banner above the image. The drawing itself is
    vision_station/core/annotate.py, shared with the application."""
    from vision_station.core import annotate
    return annotate.draw(image, raw, banner=banner, title=title, labels=labels,
                         only_failures=only_failures, search=search)


# ================================================================ command line
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Run a Vision Station package on an image.")
    ap.add_argument("package", help=".vsp package, or an application folder")
    ap.add_argument("image", help="image file to inspect")
    ap.add_argument("--job", help="job name or id (needed when there are several)")
    ap.add_argument("--save", help="write the annotated image here")
    ap.add_argument("--show", action="store_true", help="show the annotated image")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    a = ap.parse_args(argv)
    app = load(a.package)
    res = app.run(a.image, a.job)
    if a.json:
        print(json.dumps(res.raw, indent=2, default=lambda o: None))
    else:
        print("%s: %s  (%.0f ms)" % (res.job_name, res.verdict, res.total_ms))
        for t in res.raw["tools"]:
            print("  %-24s %-5s %s" % (t["tool_id"], t["status"], t.get("message") or ""))
    if a.save:
        _imwrite(a.save, res.annotated())
    if a.show:
        res.show()
    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main())
