"""A job: several sides of one part, each with its own fixture and tools.

This is the structure VisionPro users work in. One job inspects one product;
the product is photographed from several sides; each side has a fixture that
locates the part in that view, and a set of tools whose regions live in the
fixture's coordinate space. Some tools are trained from example images, sorted
into OK and NG.

Each side is a complete project folder, so everything that works on a project
-- the engine, the canvas, the property sheet, captures, history -- works on a
side without change. The job only adds the list of sides, a combined verdict,
and the training sets.

    projects/<job>/job.json
    projects/<job>/sides/<side>/project.json   reference.png   models/
    projects/<job>/sides/<side>/training/<tool>/ok/*.png
    projects/<job>/sides/<side>/training/<tool>/ng/*.png

Training images are plain files and the folder they sit in is their label.
Nothing to keep in sync, and anyone can inspect or fix a training set in a file
browser -- move a mislabelled image from ok to ng and retrain.
"""
import glob
import json
import os
import time

import cv2

from vision_station.core import imio
from vision_station.core.engine import Context, Project
from vision_station.tools import make_tool

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def is_job(folder):
    return os.path.exists(os.path.join(folder, "job.json"))


class Side(Project):
    """A project that knows it belongs to a job."""

    def __init__(self, folder, job, meta):
        self.job = job
        self.meta = meta
        super().__init__(folder)

    @property
    def side_id(self):
        return self.meta["id"]

    @property
    def name(self):
        return self.meta.get("name") or self.meta["id"]

    # -- the fixture ----------------------------------------------------
    def fixture_tool(self):
        """The side's locator: the first enabled align tool, if any."""
        for spec in self.spec.get("tools", []):
            if spec.get("type") == "align" and spec.get("params", {}).get("enabled", True):
                return spec
        return None

    def fixture_fn(self):
        """image -> Context with this side's fixture applied."""
        fx = self.fixture_tool()
        ref = self.reference

        def run(img):
            ctx = Context(reference=ref, folder=self.folder)
            if fx is not None:
                make_tool(fx).run(img, ctx)
            return ctx
        return run

    # -- training sets --------------------------------------------------
    def training_dir(self, tool_id, label):
        return os.path.join(self.folder, "training", tool_id, label)

    def training_images(self, tool_id):
        out = []
        for label in ("ok", "ng"):
            for p in sorted(glob.glob(os.path.join(self.training_dir(tool_id, label), "*"))):
                if p.lower().endswith(IMG_EXT):
                    out.append((p, label))
        return out

    def add_training_image(self, tool_id, image, label):
        if label not in ("ok", "ng"):
            raise ValueError("label must be 'ok' or 'ng'")
        d = self.training_dir(tool_id, label)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "%s_%d.png" % (label, int(time.time() * 1000)))
        if not imio.imwrite(path, image):
            raise IOError("could not write %s" % path)
        return path

    def relabel(self, path, label):
        """Move a training image between OK and NG."""
        tool_dir = os.path.dirname(os.path.dirname(path))
        dest_dir = os.path.join(tool_dir, label)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(path))
        os.replace(path, dest)
        return dest

    def remove_training_image(self, path):
        if os.path.exists(path):
            os.remove(path)

    def evaluate_tool(self, tool_id):
        """Run the side's recipe on every training image and compare the tool's
        verdict with the image's label.

        The whole recipe runs, not the tool alone: some tools read what earlier
        ones found (LED colour reads the blobs a counting tool located), and if
        the fixture fails on an image the tool genuinely would fail on the line.
        Evaluating the tool in isolation would report a recipe healthier than
        it is.

        An OK image that fails is a false reject; an NG image that passes is an
        escape -- a bad part shipped. Escapes are the ones that matter most."""
        rows = []
        for path, label in self.training_images(tool_id):
            img = imio.imread(path)
            if img is None:
                continue
            res = self.inspect(img, source="validation")
            r = next((t for t in res["tools"] if t["tool_id"] == tool_id), None)
            if r is None:
                status, score, msg = "fail", 0.0, "did not run - an earlier step stopped the inspection"
            else:
                status, score, msg = r["status"], float(r["score"]), r["message"] or ""
            passed = status == "pass"
            correct = passed if label == "ok" else not passed
            rows.append({"path": path, "label": label, "status": status,
                         "score": score, "message": msg, "correct": correct})
        false_rejects = [r for r in rows if r["label"] == "ok" and not r["correct"]]
        escapes = [r for r in rows if r["label"] == "ng" and not r["correct"]]
        return {"rows": rows,
                "ok_total": sum(r["label"] == "ok" for r in rows),
                "ng_total": sum(r["label"] == "ng" for r in rows),
                "false_rejects": len(false_rejects), "escapes": len(escapes),
                "all_correct": not false_rejects and not escapes}

    def tune_threshold(self, tool_id):
        """Set a score-threshold tool's threshold from its OK and NG examples:
        midway between the lowest-scoring OK and the highest-scoring NG. Needs
        both kinds, and reports the gap between them rather than inventing a
        number when there is none."""
        spec = next((t for t in self.spec.get("tools", []) if t["id"] == tool_id), None)
        if spec is None:
            raise KeyError(tool_id)
        param = TUNABLE.get(spec["type"])
        if param is None:
            raise ValueError("%s has no single score threshold to tune" % spec["type"])
        ev = self.evaluate_tool(tool_id)
        ok = [r["score"] for r in ev["rows"] if r["label"] == "ok"]
        ng = [r["score"] for r in ev["rows"] if r["label"] == "ng"]
        if not ok or not ng:
            raise ValueError("tuning needs at least one OK and one NG image")
        lo_ok, hi_ng = min(ok), max(ng)
        margin = lo_ok - hi_ng
        report = {"param": param, "old": spec["params"].get(param),
                  "lowest_ok": lo_ok, "highest_ng": hi_ng, "margin": margin,
                  "separates": margin > 0}
        if margin > 0:
            spec["params"][param] = round((lo_ok + hi_ng) / 2.0, 3)
            self.save(self.spec)
        report["new"] = spec["params"].get(param)
        return report

    def region_thumb(self, tool_spec, path, width=120, max_h=90):
        """A training image's thumbnail showing the tool's region as the tool
        sees it -- fixtured and deskewed -- not the whole photograph: at
        thumbnail size a scratch on a chassis is invisible in the full frame.
        Cached by file, modification time and region, because finding the
        fixture costs a few hundred milliseconds. Shared by both front ends."""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        p = tool_spec.get("params", {})
        key = (path, mtime, json.dumps(p.get("roi")), json.dumps(p.get("points")),
               p.get("angle"), p.get("fixture"))
        cache = self.__dict__.setdefault("_thumbs", {})
        if key in cache:
            return cache[key]
        img = imio.imread(path)
        if img is None:
            return None
        try:
            patch, _off = make_tool(tool_spec).crop(img, self.fixture_fn()(img))
        except Exception:
            patch = None
        if patch is None or patch.size == 0:
            patch = img
        h = max(1, min(max_h, int(width * patch.shape[0] / float(patch.shape[1]))))
        cache[key] = cv2.resize(patch, (width, h), interpolation=cv2.INTER_AREA)
        return cache[key]

    def train_tool(self, tool_id):
        """Train a trainable tool from its OK/NG images, save the model and
        point the tool at it. Returns the training report."""
        from vision_station.tools.learned import train
        spec = next((t for t in self.spec.get("tools", []) if t["id"] == tool_id), None)
        if spec is None:
            raise KeyError(tool_id)
        samples = []
        for path, label in self.training_images(tool_id):
            img = imio.imread(path)
            if img is not None:
                samples.append((img, label))
        if not any(l == "ok" for _i, l in samples):
            raise ValueError("add at least one OK image before training")
        os.makedirs(os.path.join(self.folder, "models"), exist_ok=True)
        rel = "models/%s.npz" % tool_id
        report = train(make_tool(spec), samples, self.fixture_fn(),
                       os.path.join(self.folder, rel))
        # name the NG files that could not be told apart from good parts
        ng_paths = [p for p, l in self.training_images(tool_id) if l == "ng"]
        report["unseparated_ng_paths"] = [ng_paths[i] for i in report["unseparated_ng"]
                                          if i < len(ng_paths)]
        spec.setdefault("params", {})["model"] = rel
        self.save(self.spec)
        return report


def evaluation_summary(ev):
    """(text, level) for an evaluate_tool() report; level is 'good', 'warn',
    'bad' or 'none'. Shared by both front ends.

    An empty training set used to report 'OK 0/0 passed, NG 0/0 rejected' in
    green -- a test of nothing, shown as a success."""
    if not ev["rows"]:
        return ("No OK or NG images yet - add some, then test.", "none")
    text = "OK %d/%d passed   NG %d/%d rejected" % (
        ev["ok_total"] - ev["false_rejects"], ev["ok_total"],
        ev["ng_total"] - ev["escapes"], ev["ng_total"])
    if ev["escapes"]:
        return (text + "   -   %d ESCAPE%s: a bad part would ship" % (
            ev["escapes"], "S" if ev["escapes"] > 1 else ""), "bad")
    if ev["false_rejects"]:
        return (text + "   -   %d false reject%s" % (
            ev["false_rejects"], "s" if ev["false_rejects"] > 1 else ""), "warn")
    if not ev["ng_total"]:
        return (text + "   -   no NG images, so nothing shows it can reject a bad part",
                "warn")
    return (text, "good")


# Tools whose verdict is "score at least this parameter". For these the OK and
# NG examples say where the threshold belongs, so it can be set from evidence
# instead of by feel. The score has to be computed independently of the
# threshold for this to be sound, which is true of each tool listed.
TUNABLE = {"presence": "min_score", "align": "min_score"}


class Job:
    def __init__(self, folder):
        self.folder = folder
        self.id = os.path.basename(folder.rstrip("/\\"))
        self.path = os.path.join(folder, "job.json")
        self.load()

    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            self.spec = json.load(f)
        self.sides = []
        for meta in self.spec.get("sides", []):
            folder = os.path.join(self.folder, "sides", meta["id"])
            if os.path.exists(os.path.join(folder, "project.json")):
                self.sides.append(Side(folder, self, meta))
        return self

    @property
    def name(self):
        return self.spec.get("name") or self.id

    def side(self, side_id):
        return next((s for s in self.sides if s.side_id == side_id), None)

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.spec, f, indent=2)

    def add_side(self, side_id, name, reference, tools=None):
        folder = os.path.join(self.folder, "sides", side_id)
        os.makedirs(os.path.join(folder, "models"), exist_ok=True)
        if not imio.imwrite(os.path.join(folder, "reference.png"), reference):
            raise IOError("could not write the reference image in %s" % folder)
        with open(os.path.join(folder, "project.json"), "w", encoding="utf-8") as f:
            json.dump({"id": side_id, "name": name, "tools": tools or []}, f, indent=2)
        self.spec.setdefault("sides", []).append({"id": side_id, "name": name})
        self.save()
        self.load()
        return self.side(side_id)

    def rename_side(self, side_id, name):
        for meta in self.spec.get("sides", []):
            if meta["id"] == side_id:
                meta["name"] = name
        self.save()
        self.load()

    def remove_side(self, side_id):
        """Take a side out of the job. Its folder is moved to removed/, not
        deleted: a misclick on a line PC must not destroy a trained job, and
        moving it back restores it."""
        import shutil
        self.spec["sides"] = [m for m in self.spec.get("sides", []) if m["id"] != side_id]
        self.save()
        src = os.path.join(self.folder, "sides", side_id)
        if os.path.isdir(src):
            dst_root = os.path.join(self.folder, "removed")
            os.makedirs(dst_root, exist_ok=True)
            dst = os.path.join(dst_root, "%s_%d" % (side_id, int(time.time())))
            shutil.move(src, dst)
        self.load()

    def unique_side_id(self, name):
        base = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_") or "job"
        ids = {m["id"] for m in self.spec.get("sides", [])}
        sid, n = base, 2
        while sid in ids or os.path.exists(os.path.join(self.folder, "sides", sid)):
            sid, n = "%s_%d" % (base, n), n + 1
        return sid

    @property
    def posted(self):
        """Posted items: 'side/tool.key' results shown in the results queue."""
        return list(self.spec.get("posted", []))

    def set_posted(self, items):
        self.spec["posted"] = list(items)
        self.save()

    def inspect(self, images):
        """images: {side_id: image}. A side with no image is reported as not
        inspected rather than silently passed -- a job that only looked at four
        of its seven sides has not passed."""
        t0 = time.perf_counter()
        results, missing = {}, []
        for s in self.sides:
            img = images.get(s.side_id)
            if img is None:
                missing.append(s.side_id)
                continue
            results[s.side_id] = s.inspect(img, source="job")
        if missing:
            verdict = "INCOMPLETE"
        elif any(r["verdict"] == "FAIL" for r in results.values()):
            verdict = "FAIL"
        elif any(r["verdict"] == "WARN" for r in results.values()):
            verdict = "WARN"
        else:
            verdict = "PASS"
        return {"job": self.id, "verdict": verdict, "sides": results,
                "missing": missing, "total_ms": (time.perf_counter() - t0) * 1000}


def create_job(folder, name):
    os.makedirs(os.path.join(folder, "sides"), exist_ok=True)
    with open(os.path.join(folder, "job.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "sides": []}, f, indent=2)
    return Job(folder)
