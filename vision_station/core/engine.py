"""Project = one model/variant you inspect. It owns a reference image, a scale,
and an ordered list of tools. Load it once at startup and run it thousands of
times — loading per inspection is the classic mistake that kills cycle time."""

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from vision_station.core import imio
from vision_station.tools import make_tool, ToolResult

from vision_station import paths

PROJECT_ROOT = paths.PROJECTS


@dataclass
class Context:
    reference: Optional[np.ndarray] = None
    fixture: Optional[np.ndarray] = None          # 3x3 reference -> current
    mm_per_px: float = 0.0
    folder: str = ""
    share: Dict[str, Any] = field(default_factory=dict)
    # Images produced by image-processing tools, by tool id. Any later tool can
    # read one by naming that id as the image it reads.
    images: Dict[str, Any] = field(default_factory=dict)
    # Diagnostic records, by tool id then record name -- a caliper's projection
    # profile, a histogram, a blob mask. Only collected when asked for: a
    # production run should not spend time drawing plots nobody looks at.
    records: Dict[str, Any] = field(default_factory=dict)
    want_records: bool = False
    # results of the tools that have run so far this inspection, by id
    results: Dict[str, Any] = field(default_factory=dict)
    # Images whose coordinates differ from the camera's -- an affine-resampled
    # region, an unwrapped ring -- and the reference transformed the same way,
    # by the id of the tool that made them.
    ref_images: Dict[str, Any] = field(default_factory=dict)
    # the job's tool objects by id, for tools that must re-run others (a
    # fixture from points finds its features on the reference once)
    tools: Dict[str, Any] = field(default_factory=dict)


def filter_graphics(graphics, opt):
    """A tool's Graphics settings -- VisionPro's Graphics tab -- applied once,
    here, so the run view, the editors, the results queue and exported job
    files all show the same thing. opt: {"region", "results", "labels",
    "search"}, each True unless switched off."""
    out = []
    for g in graphics:
        if g.role == "region":
            show = opt.get("region", True)
        elif g.status == "search":
            show = opt.get("search", True)
        else:
            show = opt.get("results", True)
        if not show:
            continue
        if not opt.get("labels", True):
            g.label = ""
        out.append(g)
    return out


def _resolve_link(ref, by_id):
    """'tool.key' -> that earlier tool's result value; 'tool.score' and
    'tool.status' work too. None when the tool has not run or has no such
    value -- the caller turns that into an error, never a silent default."""
    if not ref or "." not in ref:
        return None
    src, key = ref.split(".", 1)
    r = by_id.get(src)
    if r is None:
        return None
    if key == "score":
        return float(r.score)
    if key == "status":
        return r.status
    v = (r.values or {}).get(key)
    if isinstance(v, (int, float, str, bool)):
        return v
    return None


class Project:
    def __init__(self, folder: str):
        # Absolute, always. Model paths are resolved against the folder at load
        # and some tools resolve them again at run time; with a relative folder
        # the second join doubled the path -- a trained tool opened from a
        # relative location reported "not trained yet" beside its own model.
        self.folder = os.path.abspath(folder)
        self.id = os.path.basename(self.folder.rstrip("/\\"))
        self.path = os.path.join(folder, "project.json")
        self.spec: Dict[str, Any] = {}
        self.reference: Optional[np.ndarray] = None
        self.tools: List = []
        self.load()

    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            self.spec = json.load(f)
        ref = self.spec.get("reference_image", "reference.png")
        rp = ref if os.path.isabs(ref) else os.path.join(self.folder, ref)
        self.reference = imio.imread(rp) if os.path.exists(rp) else None
        # Lens calibration, if this model has one. The reference is corrected
        # here, once, because every tool compares the live frame against it:
        # correcting live frames alone would mismatch everywhere, most at the
        # edges where the correction is largest.
        from vision_station.core import calibration
        self.calibration = calibration.load(self.folder)
        if self.calibration is not None and self.reference is not None:
            self.reference = calibration.undistort(self.reference, self.calibration)
        self.tools = self.build_tools()
        return self

    def build_tools(self, spec=None):
        """Tool objects for a recipe -- this project's, or a snapshot of it.

        Tools copy their region and settings when they are made, so an edited
        recipe only takes effect once its tools are rebuilt. A front end that
        edits the recipe live must rebuild before each run; one that did not
        kept inspecting with the old region after a drag."""
        out = []
        for t in (spec if spec is not None else self.spec).get("tools", []):
            t = dict(t)
            p = dict(t.get("params", {}))
            if p.get("model") and not os.path.isabs(p["model"]):
                p["model"] = os.path.join(self.folder, p["model"])
            t["params"] = p
            out.append(make_tool(t))
        return out

    def save(self, spec: Dict[str, Any]):
        self.spec = spec
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
        self.load()

    def set_reference(self, image: np.ndarray):
        if not imio.imwrite(os.path.join(self.folder, "reference.png"), image):
            raise IOError("could not write the reference image in %s" % self.folder)
        self.reference = image

    # -- captures --------------------------------------------------------
    def captures(self) -> List[Dict[str, Any]]:
        """The exposures this model wants per inspection.

        A real switch faceplate needs two. The LEDs clip at any exposure long
        enough to show the panel, and the panel is invisible at any exposure
        short enough to keep the LED colours: measured on one photograph, the
        LED band wanted roughly a twentieth of the exposure the faceplate did.
        One frame cannot serve both, so a model may define several and each
        tool says which it reads.

        A model that defines none behaves exactly as before: a single frame
        called "default".
        """
        caps = self.spec.get("captures") or []
        if not caps:
            return [{"id": "default", "exposure_ms": None, "gain": None}]
        out = []
        for c in caps:
            out.append({"id": c.get("id") or "default",
                        "exposure_ms": c.get("exposure_ms"),
                        "gain": c.get("gain"),
                        "note": c.get("note", "")})
        return out

    # -- the hot path ----------------------------------------------------
    def inspect(self, image, source: str = "api", keep_images: bool = False,
                tools=None) -> Dict[str, Any]:
        """`image` is either a single frame, or a dict of capture id to frame.

        Passing a single frame keeps every existing recipe working unchanged:
        it is treated as the only capture, and any tool asking for a capture
        that was not supplied falls back to it rather than failing.
        """
        t0 = time.perf_counter()
        if isinstance(image, dict):
            frames = dict(image)
        else:
            frames = {"default": image}
        cal = getattr(self, "calibration", None)
        if cal is not None:
            from vision_station.core import calibration
            frames = {k: calibration.undistort(v, cal) for k, v in frames.items()}
        # `frames.get(...) or ...` raises on a numpy array: truthiness of an
        # array is ambiguous. Same trap that bit the alignment refinement.
        primary = frames.get("default")
        if primary is None:
            primary = next(iter(frames.values()))

        mm = float(self.spec.get("mm_per_px", 0.0) or 0.0)
        if not mm and cal is not None:
            mm = float(cal.get("mm_per_px", 0.0))      # scale from the calibration
        ctx = Context(reference=self.reference, folder=self.folder, mm_per_px=mm,
                      want_records=keep_images)
        results: List[ToolResult] = []
        aborted = None
        missing = set()

        fixtures, by_id = {}, {}
        tool_list = tools if tools is not None else self.tools
        geo_types = {t.id for t in tool_list if getattr(t, "changes_geometry", False)}
        ctx.tools = {t.id: t for t in tool_list}
        for tool in tool_list:
            if not tool.enabled:
                continue
            want = (tool.p.get("capture") or "default")
            frame = frames.get(want)
            reads_transformed = want in geo_types
            if frame is None and reads_transformed:
                # Never fall back to the camera image here: this tool's region
                # is in the transformed image's coordinates, so the camera image
                # would be inspected in the wrong place without a word.
                r = ToolResult(tool.id, tool.type_name, "error",
                               message="reads the image from %s, which was not produced" % want)
                by_id[tool.id] = r
                ctx.results[tool.id] = r
                results.append(r)
                continue
            if frame is None:
                # Fall back rather than fail: a recipe built on a two-capture
                # station must still run when someone loads a single image from
                # disk to try something out.
                if want != "default":
                    missing.add(want)
                frame = primary
            # Inputs, as VisionPro's terminals. The fixture: 'auto' is whatever
            # the most recent Locate tool set -- the long-standing behaviour --
            # 'none' is image space, or name a Locate tool. And linked settings:
            # a setting driven by an earlier tool's result, read now.
            fx_choice = tool.p.get("fixture") or "auto"
            fx_saved = ctx.fixture
            ref_saved = ctx.reference
            problem = None
            if reads_transformed:
                # the transform already followed the part: no fixture on top,
                # and the reference transformed the same way to compare with
                fx_choice = "none"
                ctx.reference = ctx.ref_images.get(want)
            if fx_choice == "none":
                ctx.fixture = None
            elif fx_choice != "auto":
                if fx_choice not in fixtures:
                    problem = "fixture %s has not run before this tool" % fx_choice
                elif fixtures[fx_choice] is None:
                    problem = "fixture %s did not find the part" % fx_choice
                else:
                    ctx.fixture = fixtures[fx_choice]
            for param, ref in (tool.p.get("links") or {}).items():
                val = _resolve_link(ref, by_id)
                if val is None:
                    problem = "input %s <- %s has no value" % (param, ref)
                    break
                tool.p[param] = val
            try:
                if problem:
                    r = ToolResult(tool.id, tool.type_name, "error", message=problem)
                else:
                    r = tool.run(frame, ctx)
                    r = tool.correct_graphics(r, ctx)
                if ctx.images:
                    # a processed image becomes readable by every later tool
                    for k, v in ctx.images.items():
                        frames.setdefault(k, v)
            except Exception as e:
                r = ToolResult(tool.id, tool.type_name, "error", message=f"{type(e).__name__}: {e}")
            gopt = tool.p.get("graphics")
            if gopt:
                r.graphics = filter_graphics(r.graphics, gopt)
            if ctx.want_records and tool.roi is not None and tool.type_name != "align":
                try:
                    patch, _o = tool.crop(frame, ctx)
                    if patch is not None and patch.size:
                        ctx.records.setdefault(tool.id, {}).setdefault(
                            "LastRun.RegionImage", patch.copy())
                except Exception:
                    pass
            if tool.type_name == "align" or getattr(tool, "is_fixture", False):
                fixtures[tool.id] = ctx.fixture if r.status == "pass" else None
            elif fx_choice != "auto":
                ctx.fixture = fx_saved          # a chosen fixture applies to this tool only
            ctx.reference = ref_saved
            by_id[tool.id] = r
            ctx.results[tool.id] = r
            results.append(r)
            # A failed fixture makes every later ROI meaningless. Stop rather
            # than emit a page of bogus failures the operator has to read past.
            if tool.type_name == "align" and r.status != "pass":
                aborted = "alignment failed — remaining tools skipped"
                break

        crit_fail = [r for r in results
                     if r.status in ("fail", "error")
                     and next((t.critical for t in self.tools if t.id == r.tool_id), True)]
        noncrit = [r for r in results if r.status in ("fail", "error") and r not in crit_fail]
        verdict = "FAIL" if crit_fail else ("WARN" if noncrit else "PASS")

        return {
            "id": uuid.uuid4().hex[:12],
            "project": self.id,
            "verdict": verdict,
            "source": source,
            "captures_used": sorted(frames),
            "captures_missing": sorted(missing),
            # Processed images, for record displays. Opt-in: a production run
            # does not need to carry full-resolution copies of every filter.
            "images": dict(ctx.images) if keep_images else {},
            "records": dict(ctx.records) if keep_images else {},
            "input_image": primary if keep_images else None,
            "timestamp": time.time(),
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
            "image_size": [int(primary.shape[1]), int(primary.shape[0])],
            "aborted": aborted,
            "failed_tools": [r.tool_id for r in crit_fail],
            "tools": [r.to_dict() for r in results],
        }


class ProjectStore:
    def __init__(self, root: str = PROJECT_ROOT):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.cache: Dict[str, Project] = {}

    def _folder(self, pid: str) -> str:
        """A plain project is 'name'; a side of a job is 'job/side', which lives
        at <job>/sides/<side>. Each side is a complete project, so everything
        that opens a project opens a side."""
        if "/" in pid:
            job, side = pid.split("/", 1)
            return os.path.join(self.root, job, "sides", side)
        return os.path.join(self.root, pid)

    def list(self) -> List[Dict[str, Any]]:
        out = []
        for name in sorted(os.listdir(self.root)):
            d = os.path.join(self.root, name)
            jp = os.path.join(d, "job.json")
            if os.path.exists(jp):
                try:
                    with open(jp, "r", encoding="utf-8") as f:
                        job = json.load(f)
                except Exception as e:
                    out.append({"id": name, "name": name, "error": str(e)})
                    continue
                for meta in job.get("sides", []):
                    sid = "%s/%s" % (name, meta["id"])
                    if not os.path.exists(os.path.join(self._folder(sid), "project.json")):
                        continue
                    out.append({"id": sid, "name": meta.get("name", meta["id"]),
                                "job": name, "job_name": job.get("name", name),
                                "side": meta["id"]})
                continue
            if os.path.exists(os.path.join(d, "project.json")):
                try:
                    p = self.get(name)
                    out.append({"id": name, "name": p.spec.get("name", name),
                                "tools": len(p.tools),
                                "has_reference": p.reference is not None})
                except Exception as e:
                    out.append({"id": name, "name": name, "error": str(e)})
        return out

    def get(self, pid: str) -> Project:
        if pid not in self.cache:
            d = self._folder(pid)
            if not os.path.exists(os.path.join(d, "project.json")):
                raise FileNotFoundError(f"no project '{pid}'")
            self.cache[pid] = Project(d)
        return self.cache[pid]

    def create(self, pid: str, name: str) -> Project:
        d = os.path.join(self.root, pid)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "project.json"), "w", encoding="utf-8") as f:
            json.dump({"name": name, "reference_image": "reference.png",
                       "mm_per_px": 0.0, "tools": []}, f, indent=2)
        self.cache.pop(pid, None)
        return self.get(pid)

    def reload(self, pid: str) -> Project:
        self.cache.pop(pid, None)
        return self.get(pid)
