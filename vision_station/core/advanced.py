"""Advanced model: a YOLO detector trained for one job, from that job's own
checks, that then confirms its part-present and text checks on every run.

Settings > General > "Use advanced model for training" asks for it when a
new application or job is made. The person takes the reference picture and
marks what is to be inspected, as always -- parts present or absent, text to
verify or read. On the job's first Run:

    1. pictures   N_PICTURES fresh camera pictures of the good part
    2. check      the job is run on each: they must be pictures of a GOOD
                  part, found by the job's Align step
    3. prepare    every check's region, followed into each picture by Align,
                  is a labelled box ("part" or "text"); synthetic copies --
                  shifted, turned, lighter, darker -- teach the network what a
                  line's variation looks like; tiles at the network's size
    4. train      the training environment's Python (PyTorch, Ultralytics;
                  core/advanced_worker.py) on the graphics card
    5. validate   on the new pictures: is every check's region found, and is a
                  region painted out -- a missing part -- NOT found? A check
                  the network did not learn is left to its classic test.
    6. install    the model in <job>/advanced/, and the pictures added to every
                  check's good pictures (its Train tab), as hard links: one
                  copy on disk however many checks share it

Cancelled or failed, everything it made is deleted and the job is as it was.

On a run (core/engine.py) a learned check passes only when its classic test
passes AND the network finds its part or text where Align puts it: a missing
or wrong part the classic test let through is caught. The pictures added as
good examples make the classic test kinder to a good part that looks a little
different -- fewer false rejects. The network looks only at the tiles that
hold learned checks' regions, each tile once per run.

Nothing here imports Qt: the window (ui/advanced_training.py) polls a
Session's state.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from vision_station import paths
from vision_station.core import imio

ADV = "advanced"
KINDS = {"presence": "part", "ocv": "text", "ocr": "text"}
CLASSES = ["part", "text"]
N_PICTURES = 16             # camera pictures; each is also a good example of every check
SYNTH_PER = 4               # positions each picture is also shown at, for training only
SHIFT = 0.06                # up to this share of the picture, either way, the board is put
TURN = 4.0                  # up to this many degrees turned
LIGHT_STEPS = (0.0, -0.08, 0.08, -0.04, 0.04)   # the light, around the job's level
IMGSZ = 1280                # the network's input, as the part finder's
MIN_SIDE = 6                # px at the network's scale: smaller cannot be learned
FIT = 0.7                   # a region at most this share of a tile
BIG = 0.4                   # regions over this share get a tile centred on them
AI_MIN = 0.35               # the network's confidence that counts as "found"
MATCH_IOU = 0.3             # a detection belongs to a check's region from this overlap
SURE_PART = 0.85            # a part check's match this high is not overruled by the AI
SURE_TEXT = 0.80            # nor a text check read with this confidence
FAIL_SHARE = 0.25           # more checks than this failing: not a good part
MISSING_TESTS = 30          # regions painted out to see the network notice
STAGES = [("capture", 0.00, 0.12, "Taking pictures of the good part"),
          ("check", 0.12, 0.18, "Checking the pictures"),
          ("prepare", 0.18, 0.25, "Preparing the training pictures"),
          ("train", 0.25, 0.92, "Training the model on the graphics card"),
          ("validate", 0.92, 0.98, "Testing the model"),
          ("install", 0.98, 1.00, "Saving the model")]


# ---------------------------------------------------------------- where
def folder_of(project_folder):
    return os.path.join(project_folder, ADV)


def state_of(spec):
    return dict((spec or {}).get("advanced") or {})


def model_path(project_folder):
    return os.path.join(folder_of(project_folder), "model.onnx")


def is_trained(spec, project_folder):
    st = state_of(spec)
    return bool(st.get("trained")) and os.path.isfile(model_path(project_folder)) and \
        os.path.isfile(os.path.join(folder_of(project_folder), "model.json"))


def is_pending(spec, project_folder):
    """Asked for when the job was made, and not trained yet."""
    return bool(state_of(spec).get("requested")) and not is_trained(spec, project_folder)


def train_python():
    """The training environment's Python: VS_TRAIN_PYTHON, or the
    vision-station-train folder in the user's home or local app data."""
    cands = [os.environ.get("VS_TRAIN_PYTHON", "")]
    for base in (os.path.expanduser("~"), os.environ.get("LOCALAPPDATA", "")):
        if base:
            cands.append(os.path.join(base, "vision-station-train", "Scripts", "python.exe"))
            cands.append(os.path.join(base, "vision-station-train", "bin", "python"))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "advanced_worker.py")


def can_train():
    """(True, how) or (False, why not) -- for this PC."""
    try:
        from vision_station.core import system_profile
        _prof, p = system_profile.current()
        t = p["train"]
    except Exception as e:
        return False, "this PC could not be looked at: %s" % e
    if not t["possible"]:
        return False, t["why"]
    py = train_python()
    if py is None:
        return False, ("the training environment (vision-station-train, with PyTorch and "
                       "Ultralytics) is not installed")
    if not os.path.isfile(WORKER):
        return False, "the training worker is missing (%s)" % WORKER
    return True, t["why"]


# YOLO11 and YOLO26 (the newest, end-to-end: no overlap removal afterwards),
# each in four sizes. Bigger learns more, trains longer and runs slower.
SIZES = {"small": "yolo11s.pt", "medium": "yolo11m.pt", "large": "yolo11l.pt",
         "xlarge": "yolo11x.pt",
         "26-small": "yolo26s.pt", "26-medium": "yolo26m.pt", "26-large": "yolo26l.pt",
         "26-xlarge": "yolo26x.pt"}
# tiles of 1280 px trained together per step, by graphics memory: the bigger
# the network, the fewer fit (8 GB: small 4, medium 2, large and extra 1)
BATCH = {k: ((4, 7.5) if k.endswith("small") else (2, 7.5) if k.endswith("medium")
             else (1, 7.5)) for k in SIZES}


def base_weights(size=None):
    """(starting weights, batch) for a model size -- 'small' (default, this
    PC's plan), 'medium' or 'large'. models/base/<model>.pt when it is there
    (no download), else the name: Ultralytics then downloads it."""
    try:
        from vision_station.core import system_profile
        t = system_profile.current()[1]["train"]
        name, batch, vram = t["model"] or "yolo11n.pt", t["batch"] or 2, 8.0
        gpus = [g for g in system_profile.current()[0].get("gpus") or []
                if g.get("vendor") == "NVIDIA"]
        if gpus:
            vram = max(g.get("vram_gb", 0) for g in gpus)
    except Exception:
        name, batch, vram = "yolo11n.pt", 2, 8.0
    if size in SIZES:
        name = SIZES[size]
        per, at = BATCH[size]
        batch = max(1, int(per * min(2.0, vram / at)))    # more memory, more at once
    for n in (name,) + (() if size in SIZES else ("yolo11s.pt", "yolo11n.pt")):
        p = paths.model("base", n)
        if os.path.isfile(p):
            return p, batch
    return name, batch


# ---------------------------------------------------------------- regions
def _tool(spec):
    from vision_station.tools import make_tool
    return make_tool(spec)


def region_in(tool, ctx):
    """(x, y, w, h) round the check's region in the picture -- turned and
    polygon regions by their bounding box -- or None."""
    try:
        if tool.shape == "polygon" and tool.points:
            r = tool.mapped_roi(ctx)
            return None if r is None else tuple(float(v) for v in r)
        if tool.roi is None:
            return None
        q = tool.mapped_quad(ctx)
        x0, y0 = float(q[:, 0].min()), float(q[:, 1].min())
        return (x0, y0, float(q[:, 0].max()) - x0, float(q[:, 1].max()) - y0)
    except Exception:
        return None


class _Ctx:
    def __init__(self, fixture=None):
        self.fixture = fixture


def learnable(spec):
    """[{"id", "type", "cls", "spec"}] -- the checks the network can learn --
    and {id: why not} for the others it could have (a part or text check)."""
    out, skip = [], {}
    for t in (spec or {}).get("tools", []):
        kind = KINDS.get(t.get("type"))
        if kind is None:
            continue
        p = t.get("params", {})
        if not p.get("enabled", True):
            skip[t["id"]] = "switched off"
            continue
        if p.get("capture") and p.get("capture") != "default":
            skip[t["id"]] = "reads another picture"
            continue
        tool = _tool(t)
        r = region_in(tool, _Ctx())
        if r is None or r[2] < 1 or r[3] < 1:
            skip[t["id"]] = "has no region"
            continue
        out.append({"id": t["id"], "type": t["type"], "cls": CLASSES.index(kind), "spec": t,
                    "ref": r})
    return out, skip


def choose_zoom(items):
    """One scale for the whole job: 1 (the camera's pixels) unless most regions
    are too big for a tile; regions that still are -- or end up too small --
    are left out, with why."""
    if not items:
        return 1.0, [], {}
    sides = np.array([max(i["ref"][2], i["ref"][3]) for i in items], np.float64)
    z = min(1.0, FIT * IMGSZ / max(1.0, float(np.percentile(sides, 90))))
    keep, why = [], {}
    for it, s in zip(items, sides):
        small = min(it["ref"][2], it["ref"][3]) * z
        if s * z > FIT * IMGSZ:
            why[it["id"]] = "too large for the AI (%.0f px)" % s
        elif small < MIN_SIDE:
            why[it["id"]] = "too small for the AI (%.0f px)" % min(it["ref"][2], it["ref"][3])
        else:
            keep.append(it)
    return z, keep, why


def fixture_for(spec, reference, img, folder=""):
    """The job's Align result on img: (True, 3x3 or None) when found (None:
    the job has no Align step), (False, None) when Align did not find it."""
    from vision_station.core.engine import Context
    al = next((t for t in spec.get("tools", []) if t.get("type") == "align"
               and t.get("params", {}).get("enabled", True)), None)
    if al is None:
        return True, None
    ctx = Context(reference=reference, folder=folder)
    try:
        r = _tool(al).run(img, ctx)
    except Exception:
        return False, None
    if r.status != "pass":
        return False, None
    return True, None if ctx.fixture is None else np.asarray(ctx.fixture, np.float64)


def boxes_in(items, fixture):
    """[(cls, x, y, w, h)] of the learned checks in a picture under fixture."""
    out = []
    ctx = _Ctx(fixture)
    for it in items:
        r = region_in(_tool(it["spec"]), ctx)
        if r is not None:
            out.append((it["cls"],) + tuple(r))
    return out


# ---------------------------------------------------------------- synthetic
def poses(n, rng):
    """n positions spread over the whole range a person puts the board in:
    shifts on a grid both ways, turns and scales spread across theirs
    (stratified, so no corner of the range is left out by chance). Taught on
    one board lying one way, the network lost small parts when the board was
    put back 12 px along."""
    k = max(1, int(np.ceil(np.sqrt(n))))
    grid = [((i + 0.5) / k * 2 - 1, (j + 0.5) / k * 2 - 1) for i in range(k) for j in range(k)]
    rng.shuffle(grid)
    turns = np.linspace(-1, 1, n)
    rng.shuffle(turns)
    out = []
    for i in range(n):
        gx, gy = grid[i % len(grid)]
        jit = rng.uniform(-0.5, 0.5, 2) / k
        out.append({"dx": float((gx + jit[0]) * SHIFT), "dy": float((gy + jit[1]) * SHIFT),
                    "turn": float(turns[i] * TURN), "scale": float(rng.uniform(0.97, 1.03))})
    return out


def variant(img, rng, pose=None):
    """A copy as the line could show it: the board put at pose (shifted,
    turned, scaled; a random one when None), lighter or darker, sometimes
    softer, noisier or JPEG-compressed. (picture, 2x3 matrix)."""
    h, w = img.shape[:2]
    if pose is None:
        pose = poses(1, rng)[0]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), pose["turn"], pose["scale"])
    M[0, 2] += pose["dx"] * w
    M[1, 2] += pose["dy"] * h
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a, b = float(rng.uniform(0.75, 1.25)), float(rng.uniform(-20, 20))
    out = cv2.convertScaleAbs(out, alpha=a, beta=b)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.6, 1.4)))
    if rng.random() < 0.3:
        n = rng.normal(0, float(rng.uniform(2, 6)), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + n, 0, 255).astype(np.uint8)
    if rng.random() < 0.15:
        out = cv2.imdecode(cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY),
                                                      int(rng.integers(70, 95))])[1], 1)
    return out, M


def move_boxes(boxes, M, w, h):
    out = []
    for c, x, y, bw, bh in boxes:
        pts = np.float32([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]).reshape(-1, 1, 2)
        q = cv2.transform(pts, M).reshape(-1, 2)
        x0, y0 = q[:, 0].min(), q[:, 1].min()
        x1, y1 = q[:, 0].max(), q[:, 1].max()
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if (cx1 - cx0) * (cy1 - cy0) < 0.6 * (x1 - x0) * (y1 - y0):
            continue                                   # mostly off the picture
        out.append((c, float(cx0), float(cy0), float(cx1 - cx0), float(cy1 - cy0)))
    return out


# ---------------------------------------------------------------- tiles
def tile_starts(n, src, offset=0):
    """Tile starts half a tile apart, the grid moved along by offset: cut
    always at the same places, a network learns where in its tile each part
    sits and loses it in a tile cut elsewhere."""
    if n <= src:
        return [0]
    step = max(1, src // 2)
    st = [0] + list(range(int(offset) % step, n - src, step)) + [n - src]
    return sorted(set(st))


def write_tiles(img, boxes, zoom, out_img, out_lab, stem, rng, keep_empty=0.08):
    """Tiles of img at the network's size, with YOLO labels. Returns how
    many were written."""
    if zoom != 1.0:
        img = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA)
        boxes = [(c, x * zoom, y * zoom, w * zoom, h * zoom) for c, x, y, w, h in boxes]
    H, W = img.shape[:2]
    n = 0
    ox, oy = (int(v) for v in rng.integers(0, IMGSZ // 2, 2))
    origins = [(x0, y0) for y0 in tile_starts(H, IMGSZ, oy) for x0 in tile_starts(W, IMGSZ, ox)]
    # the grid's tiles overlap by half: a region bigger than that can be cut
    # by every one of them -- such a region gets a tile of its own round it
    for c, x, y, w, h in boxes:
        if max(w, h) > BIG * IMGSZ:
            origins.append((int(max(0, min(max(0, W - IMGSZ), x + w / 2.0 - IMGSZ / 2.0))),
                            int(max(0, min(max(0, H - IMGSZ), y + h / 2.0 - IMGSZ / 2.0)))))
    for x0, y0 in origins:
        if True:
            tile = img[y0:y0 + IMGSZ, x0:x0 + IMGSZ]
            th, tw = tile.shape[:2]
            lines, whole = [], 0
            for c, x, y, w, h in boxes:
                ix0, iy0 = max(x, x0), max(y, y0)
                ix1, iy1 = min(x + w, x0 + tw), min(y + h, y0 + th)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                share = (ix1 - ix0) * (iy1 - iy0) / max(1e-6, w * h)
                if share < 0.6:
                    continue                             # its neighbour tile has it
                whole += share >= 0.9
                lines.append("%d %.6f %.6f %.6f %.6f" % (
                    c, ((ix0 + ix1) / 2.0 - x0) / IMGSZ, ((iy0 + iy1) / 2.0 - y0) / IMGSZ,
                    (ix1 - ix0) / IMGSZ, (iy1 - iy0) / IMGSZ))
            if not whole and rng.random() > keep_empty:
                continue
            if th < IMGSZ or tw < IMGSZ:
                tile = cv2.copyMakeBorder(tile, 0, IMGSZ - th, 0, IMGSZ - tw,
                                          cv2.BORDER_CONSTANT, value=(114, 114, 114))
            name = "%s_%d_%d" % (stem, x0, y0)
            # lossless: taught on JPEG tiles, the network saw the camera's raw
            # picture differently -- a part 0.93 in a JPEG copy, 0.00 raw
            cv2.imwrite(os.path.join(out_img, name + ".png"), tile,
                        [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
            with open(os.path.join(out_lab, name + ".txt"), "w") as f:
                f.write("\n".join(lines))
            n += 1
    return n


# ---------------------------------------------------------------- the network at run time
_NETS = {}


def _net(path):
    from vision_station.core import inference
    dev = "CPU" if os.environ.get("VS_DEVICE", "").strip().upper() == "CPU" else \
        ("GPU" if inference.ort_gpu_providers() else "CPU")
    key = (os.path.abspath(path), os.path.getmtime(path), dev)
    if key not in _NETS:
        for k in [k for k in _NETS if k[0] == key[0]]:
            _NETS.pop(k, None)                      # an older model of the same job
        try:
            _NETS[key] = inference.load(path, "onnxruntime", "accuracy", dev)
        except Exception:
            if dev != "GPU":
                raise
            _NETS[key] = inference.load(path, "onnxruntime", "accuracy", "CPU")
    return _NETS[key]


def clear_cache():
    _NETS.clear()


class Detector:
    """A job's trained network: finds its checks' parts and text."""

    def __init__(self, folder, meta):
        self.folder = folder
        self.meta = meta
        self.path = model_path(folder)
        self.tools = {k: int(v) for k, v in (meta.get("tools") or {}).items()}
        self.zoom = float(meta.get("zoom", 1.0))
        self.imgsz = int(meta.get("imgsz", IMGSZ))
        self.src = int(round(self.imgsz / self.zoom))
        self.min_conf = float(meta.get("min_conf", AI_MIN))
        # each check's own pass mark: half the least the network gave it on
        # the teaching pictures (a small part's confidence wandered 0.33 to
        # 0.62 on one unmoved board, and 0.35 for all failed it once in 12)
        self.thresholds = {k: float(v) for k, v in (meta.get("thresholds") or {}).items()}

    def threshold(self, tool_id):
        return self.thresholds.get(tool_id, self.min_conf)

    def tile_for(self, region, W, H):
        x, y, w, h = region
        src = self.src
        step = max(1, src // 4)
        cx, cy = x + w / 2.0, y + h / 2.0

        def place(c, size, n):
            if n <= src:
                return 0
            t = int(round((c - src / 2.0) / step)) * step
            t = max(0, min(n - src, t))
            if c - size / 2.0 < t or c + size / 2.0 > t + src:  # snapped off it: centre
                t = int(max(0, min(n - src, round(c - src / 2.0))))
            return t
        return place(cx, w, W), place(cy, h, H)

    def detect_tile(self, frame, tx, ty):
        """[[x1, y1, x2, y2, score, cls], ...] in the frame's pixels."""
        src = self.src
        tile = frame[ty:ty + src, tx:tx + src]
        if tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        th, tw = tile.shape[:2]
        if th < src or tw < src:
            tile = cv2.copyMakeBorder(tile, 0, src - th, 0, src - tw, cv2.BORDER_CONSTANT,
                                      value=(114, 114, 114))
        if src != self.imgsz:
            tile = cv2.resize(tile, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA
                              if src > self.imgsz else cv2.INTER_LINEAR)
        x = tile[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        raw = np.asarray(_net(self.path)(x)[0])
        s = self.imgsz / float(src)
        if raw.ndim == 3 and raw.shape[2] == 6:
            # YOLO26, end to end: (1, N, 6) finds as x1 y1 x2 y2 score class,
            # overlaps already removed
            d = raw[0]
            d = d[d[:, 4] >= 0.05]
            b = d[:, :4] / s
            b[:, [0, 2]] += tx
            b[:, [1, 3]] += ty
            return np.concatenate([b, d[:, 4:6]], 1).astype(np.float32).reshape(-1, 6)
        out = raw[0]                                        # (4 + nc) x N
        sc = out[4:]
        cls = sc.argmax(0)
        conf = sc.max(0)
        keep = conf >= 0.05
        if not keep.any():
            return np.zeros((0, 6), np.float32)
        cx, cy, w, h = out[0, keep], out[1, keep], out[2, keep], out[3, keep]
        b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) / s
        b[:, [0, 2]] += tx
        b[:, [1, 3]] += ty
        conf, cls = conf[keep], cls[keep]
        res = []
        for c in np.unique(cls):
            m = cls == c
            bb, ss = b[m], conf[m]
            idx = cv2.dnn.NMSBoxes([[float(v[0]), float(v[1]), float(v[2] - v[0]),
                                     float(v[3] - v[1])] for v in bb], ss.tolist(), 0.05, 0.5)
            for i in np.array(idx).reshape(-1):
                res.append([*bb[i], ss[i], c])
        return np.asarray(res, np.float32).reshape(-1, 6)

    def contains(self, tile, region, W, H, margin=16):
        tx, ty = tile
        x, y, w, h = region
        sw, sh = min(self.src, W), min(self.src, H)
        return x - margin >= tx and y - margin >= ty and             x + w + margin <= tx + sw and y + h + margin <= ty + sh

    def plan(self, regions, W, H):
        """{key: tile origin}: the fewest tiles that hold every region whole --
        chosen greedily from origins a quarter-tile apart. 217 checks of a
        board needed 61 tiles placed one per check (3 s a run), about 15 so."""
        src = self.src
        step = max(1, src // 4)
        xs = sorted(set([min(max(0, W - src), v) for v in range(0, max(1, W - src) + step, step)]))
        ys = sorted(set([min(max(0, H - src), v) for v in range(0, max(1, H - src) + step, step)]))
        keys = list(regions)
        if not keys:
            return {}
        R = np.array([regions[k] for k in keys], np.float64)
        m = 16.0
        x0, y0 = R[:, 0] - m, R[:, 1] - m
        x1, y1 = R[:, 0] + R[:, 2] + m, R[:, 1] + R[:, 3] + m
        sw, sh = min(src, W), min(src, H)
        cand = [(tx, ty) for ty in ys for tx in xs]
        T = np.array(cand, np.float64)
        inside = ((x0[None] >= T[:, :1]) & (y0[None] >= T[:, 1:2]) &
                  (x1[None] <= T[:, :1] + sw) & (y1[None] <= T[:, 1:2] + sh))   # tiles x regions
        left = np.ones(len(keys), bool)
        out = {}
        # From the top-left uncovered region, in bands half a tile high: of the
        # tiles holding it, the one holding the most others. Taking the tile
        # with the most regions anywhere first left gaps that each needed one
        # more tile (8 for a row 5 cover).
        band = (R[:, 1] // max(1, src // 2)) * 1e6 + R[:, 0]
        holdable = inside.any(0)
        while (left & holdable).any():
            anchor = int(np.argmin(np.where(left & holdable, band, np.inf)))
            gain = np.where(inside[:, anchor], (inside & left[None]).sum(1), -1)
            best = int(gain.argmax())
            if gain[best] <= 0:
                break
            for i in np.nonzero(inside[best] & left)[0]:
                out[keys[i]] = cand[best]
            left &= ~inside[best]
        for i in np.nonzero(left)[0]:                  # too big for any grid tile: its own
            out[keys[i]] = self.tile_for(regions[keys[i]], W, H)
        return out

    def find(self, frame, region, cls, cache=None, key=None, tile=None, pad=0):
        """(confidence, box) of the network's best find of class cls for the
        region: a find of about its size (half to twice its area) centred
        within pad of the region's -- the same leeway the check's own search
        has. (0, None) when nothing there."""
        H, W = frame.shape[:2]
        if tile is None or not self.contains(tile, region, W, H, margin=0):
            tile = self.tile_for(region, W, H)
        tx, ty = tile
        k = (key if key is not None else id(frame), tx, ty)
        dets = None if cache is None else cache.get(k)
        if dets is None:
            dets = self.detect_tile(frame, tx, ty)
            if cache is not None:
                cache[k] = dets
        x, y, w, h = region
        cx, cy, area = x + w / 2.0, y + h / 2.0, max(1.0, w * h)
        tol_x, tol_y = pad + 0.25 * w, pad + 0.25 * h
        best, box = 0.0, None
        for x1, y1, x2, y2, s, c in dets:
            if int(c) != int(cls) or s <= best:
                continue
            dcx, dcy = (x1 + x2) / 2.0 - cx, (y1 + y2) / 2.0 - cy
            if abs(dcx) > tol_x or abs(dcy) > tol_y:
                continue
            a = (x2 - x1) * (y2 - y1)
            if not 0.5 <= a / area <= 2.0:
                continue
            best, box = float(s), (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
        return best, box


def warm(projects):
    """Each job's network loaded onto the graphics card and run once, in the
    background: the first run after opening paid 3 s for it."""
    def work():
        from vision_station.tools import count
        for p in projects:
            det = getattr(p, "advanced", None)
            if det is None:
                continue
            try:
                det.detect_tile(np.zeros((det.src, det.src, 3), np.uint8), 0, 0)
            except Exception:
                pass
            # the part checks' good examples read now, not by the first part:
            # 8 twenty-megapixel pictures took the first run from 0.8 to 4.3 s
            ref = p.reference
            for t in p.spec.get("tools", []):
                roi = (t.get("params") or {}).get("roi")
                if t.get("type") != "presence" or not roi or ref is None:
                    continue
                try:
                    x, y, w, h = [int(v) for v in roi]
                    tpl = count._gray(ref[max(0, y):y + h, max(0, x):x + w])
                    count.training_examples(p.folder, t["id"], [x, y, w, h], tpl)
                except Exception:
                    pass
            count.release_pictures()
    threading.Thread(target=work, name="advanced-warm-up", daemon=True).start()


def load(folder, spec):
    """The job's Detector, or None (none trained, or its files gone)."""
    if not is_trained(spec, folder):
        return None
    try:
        with open(os.path.join(folder_of(folder), "model.json"), encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    return Detector(folder, meta)


MIN_THRESHOLD = 0.15


def own_threshold(least):
    """A check's AI pass mark from the least confidence it had on the
    teaching pictures: half of it, between MIN_THRESHOLD and AI_MIN."""
    return round(max(MIN_THRESHOLD, min(AI_MIN, 0.5 * float(least))), 3)


def leeway(type_name, params, region):
    """How far from its region a check's part may be found: a part-present
    check's own search, a text check's sixth of its size."""
    if type_name == "presence":
        return float(params.get("search_pad", 15))
    return max(8.0, 0.15 * max(region[2], region[3]))


def apply(project, tool, r, frame, ctx):
    """A learned check's result, confirmed by the network: it passes only when
    the network also finds its part or text where Align puts it. Returns r,
    changed in place."""
    det = getattr(project, "advanced", None)
    if det is None or tool.id not in det.tools or tool.type_name not in KINDS:
        return r
    if not tool.p.get("ai", True) or r.status == "error" or frame is None:
        return r
    region = region_in(tool, ctx)
    if region is None:
        return r
    share = ctx.share
    cache = share.setdefault("_advanced_tiles", {})
    plan = share.get("_advanced_plan")
    if plan is None:
        # every learned check's region, where the current fixture puts it:
        # their tiles planned once for the whole run
        regs = {}
        for t in (getattr(ctx, "tools", None) or {}).values():
            if t.id in det.tools and t.type_name in KINDS and t.enabled and t.p.get("ai", True):
                rg = region_in(t, ctx)
                if rg is not None:
                    regs[t.id] = rg
        plan = det.plan(regs, frame.shape[1], frame.shape[0])
        share["_advanced_plan"] = plan
    pad = leeway(tool.type_name, tool.p, region)
    try:
        conf, box = det.find(frame, region, det.tools[tool.id], cache, tile=plan.get(tool.id),
                             pad=pad)
    except Exception as e:
        r.values = dict(r.values or {}, ai="error: %s" % e)
        return r
    found = conf >= det.threshold(tool.id)
    r.values = dict(r.values or {}, ai_confidence=round(conf, 3), ai_found=bool(found))
    # The network overrules only a check that is itself unsure. Taught on one
    # board lying one way, it lost two small parts (0.00-0.24) when the board
    # was put back 12 px further along -- parts the check matched at 0.98. A
    # missing part never matches that well; one a lenient check lets through
    # (pass mark 0.25) does not either, and that is the one it is there for.
    sure = SURE_TEXT if KINDS[tool.type_name] == "text" else SURE_PART
    if r.status == "pass" and not found and float(r.score) >= sure:
        r.values["ai_overruled"] = False
        r.values["ai_note"] = "AI unsure (%.2f); the check itself is sure (%.2f)" % (
            conf, float(r.score))
        return r
    if r.status == "pass" and not found:
        what = "text" if KINDS[tool.type_name] == "text" else "part"
        r.status = "fail"
        r.message = "AI did not find the %s here (%.2f)" % (what, conf)
        for g in r.graphics:
            if g.status == "pass":
                g.status = "fail"
    return r


# ---------------------------------------------------------------- training session
class Cancelled(Exception):
    pass


class Session:
    """One advanced training, in a thread of its own. The window reads
    .state; .cancel() stops it and deletes what it made.

    project   the job (core.engine.Project or core.job.Side), saved
    grab      () -> a fresh picture (BGR) or None; raises on a camera fault
    light     the light controller, held on while the pictures are taken
    """

    def __init__(self, project, grab, light=None, light_level=None, pid=None,
                 n_pictures=None, epochs=None, python=None, model_size=None):
        self.project = project
        self.model_size = model_size
        self.grab = grab
        self.light = light
        self.light_level = light_level
        self.pid = pid
        self.n_pictures = int(n_pictures or N_PICTURES)
        self.epochs = epochs
        self.python = python or train_python()
        self.folder = project.folder
        self.adv = folder_of(self.folder)
        self.new = os.path.join(self.adv, "new")
        self._cancel = threading.Event()
        self.fail_on = {}                 # picture name -> checks that failed on it
        self._proc = None
        self._thread = None
        self._lock = threading.Lock()
        self.state = {"stage": "capture", "progress": 0.0, "message": STAGES[0][3],
                      "detail": "", "started": time.time(), "eta_s": None,
                      "done": False, "ok": False, "cancelled": False, "error": None,
                      "result": None, "can_cancel": True, "train_started": None,
                      "epoch": 0, "epochs": 0}

    # -- the window's side -------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, name="advanced-training", daemon=True)
        self._thread.start()
        return self

    def cancel(self):
        if not self.state["can_cancel"]:
            return False
        self._cancel.set()
        self._kill()
        return True

    def wait(self, timeout_s=None):
        if self._thread is not None:
            self._thread.join(timeout_s)
        return self.state["done"]

    def snapshot(self):
        with self._lock:
            return dict(self.state)

    # -- inside ------------------------------------------------------------
    def _set(self, **kw):
        with self._lock:
            self.state.update(kw)

    def _stage(self, name, frac=0.0, detail=""):
        st = next(s for s in STAGES if s[0] == name)
        p = st[1] + (st[2] - st[1]) * max(0.0, min(1.0, frac))
        with self._lock:
            self.state.update(stage=name, message=st[3], detail=detail,
                              progress=max(self.state["progress"], p))
            if name == "train" and self.state["train_started"] is not None and 0.02 < frac:
                el = time.time() - self.state["train_started"]
                self.state["eta_s"] = el / frac * (1.0 - frac) + 20.0
            elif name in ("validate", "install"):
                self.state["eta_s"] = 10.0 if name == "validate" else 2.0

    def _check(self):
        if self._cancel.is_set():
            raise Cancelled()

    def _kill(self):
        p = self._proc
        if p is None or p.poll() is not None:
            return
        try:
            if os.name == "nt":                    # the data loader's workers too
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                               capture_output=True, timeout=15,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                p.kill()
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    def _run(self):
        try:
            if os.path.isdir(self.new):
                shutil.rmtree(self.new, ignore_errors=True)   # a crash's leftovers
            os.makedirs(self.new, exist_ok=True)
            self._preflight()
            pics = self._capture()
            fixtures, report = self._check_pictures(pics)
            data, zoom, items, skipped = self._prepare(pics, fixtures)
            onnx, metrics = self._train(data)
            val = self._validate(onnx, pics, fixtures, zoom, items)
            self._set(can_cancel=False)
            result = self._install(onnx, metrics, val, zoom, items, skipped, report, len(pics))
            self._set(stage="install", progress=1.0, eta_s=0.0, ok=True, result=result,
                      message="Training completed")
        except Cancelled:
            self._cleanup()
            self._set(cancelled=True, message="Cancelled - everything it made was deleted")
        except Exception as e:
            self._cleanup()
            self._set(error="%s" % e if isinstance(e, (RuntimeError, ValueError, OSError))
                      else "%s: %s" % (type(e).__name__, e))
        finally:
            self._kill()
            self._set(done=True)

    def _cleanup(self):
        self._kill()
        for _ in range(20):                          # the killed worker lets go of its files
            shutil.rmtree(self.new, ignore_errors=True)
            if not os.path.exists(self.new):
                break
            time.sleep(0.25)
        try:                                        # the advanced folder, if only ours
            if os.path.isdir(self.adv) and not os.listdir(self.adv):
                os.rmdir(self.adv)
        except OSError:
            pass

    def _preflight(self):
        if self.python is None:
            raise RuntimeError("The training environment is not installed on this PC.")
        items, skip = learnable(self.project.spec)
        if not items:
            raise ValueError("This job has no part or text checks to learn. Mark the parts "
                             "and text to inspect first (Find parts / text), then Run.")
        free = shutil.disk_usage(self.folder).free
        if free < 2 * 1024 ** 3:
            raise OSError("Only %.1f GB free on the disk; training needs about 2 GB."
                          % (free / 1024.0 ** 3))
        if self.project.reference is None:
            raise ValueError("The job has no reference picture.")

    def _capture(self):
        self._stage("capture", 0.0, "0 of %d" % self.n_pictures)
        good = os.path.join(self.new, "good")
        os.makedirs(good, exist_ok=True)
        held = False
        if self.light is not None:
            try:
                if self.pid is not None:
                    self.light.use_job_level(self.pid, self.light_level)
                self.light.hold("advanced")
                held = True
                self.light.settle()
            except Exception:
                pass
        pics = []
        base = None
        if held:
            try:
                base = self.light.job_level(self.pid, self.light_level)                     if self.pid is not None else self.light_level
            except Exception:
                base = self.light_level
        try:
            fails = 0
            while len(pics) < self.n_pictures:
                self._check()
                if base:
                    # the light a little brighter or darker from picture to
                    # picture, as a lamp ages and the room changes
                    step = LIGHT_STEPS[len(pics) % len(LIGHT_STEPS)]
                    self.light.send_level(max(1, min(255, int(round(base * (1 + step))))))
                img = self.grab()
                if img is None:
                    fails += 1
                    if fails >= 3:
                        raise RuntimeError("The camera gave no picture.")
                    continue
                if img.shape[:2] != self.project.reference.shape[:2]:
                    raise ValueError("The pictures are %dx%d, the reference %dx%d: the "
                                     "camera's settings changed since it was taken." % (
                                         img.shape[1], img.shape[0],
                                         self.project.reference.shape[1],
                                         self.project.reference.shape[0]))
                p = os.path.join(good, "good_%02d.png" % (len(pics) + 1))
                if not imio.imwrite(p, img):
                    raise OSError("could not write %s" % p)
                pics.append(p)
                self._stage("capture", len(pics) / float(self.n_pictures),
                            "%d of %d" % (len(pics), self.n_pictures))
        finally:
            if held:
                try:
                    if base:
                        self.light.send_level(int(base))
                    self.light.release("advanced")
                except Exception:
                    pass
        return pics

    def _check_pictures(self, pics):
        """Each picture found by Align and passing the job: a good part.
        Returns ({path: fixture}, report)."""
        spec = self.project.spec
        fixtures, lost, failing = {}, [], {}
        checks = [t["id"] for t in spec.get("tools", [])
                  if t.get("type") != "align" and t.get("params", {}).get("enabled", True)]
        bad_pics = 0
        for i, p in enumerate(pics):
            self._check()
            self._stage("check", i / float(len(pics)), "picture %d of %d" % (i + 1, len(pics)))
            img = imio.imread(p)
            ok, M = fixture_for(spec, self.project.reference, img, self.folder)
            if not ok:
                lost.append(os.path.basename(p))
                continue
            fixtures[p] = M
            res = self.project.inspect(img, source="advanced")
            fails = [t["tool_id"] for t in res["tools"]
                     if t["tool_id"] in checks and t["status"] != "pass"]
            # a check failing on a picture is not taught from it: neither as a
            # good example nor as an AI label -- a part deliberately wrong on
            # the board would otherwise be taught as good
            self.fail_on[os.path.basename(p)] = set(fails)
            for f in fails:
                failing[f] = failing.get(f, 0) + 1
            if checks and len(fails) > FAIL_SHARE * len(checks):
                bad_pics += 1
        if len(fixtures) < max(2, len(pics) // 2):
            raise ValueError("The job's Align step did not find the part in %d of %d new "
                             "pictures. Is the part under the camera, where it was taught?"
                             % (len(lost), len(pics)))
        if bad_pics > len(pics) // 2:
            raise ValueError("Most checks fail on the new pictures: put a GOOD part under the "
                             "camera, as in the reference picture, and Run again.")
        return fixtures, {"align_lost": len(lost), "failing": failing}

    def _prepare(self, pics, fixtures):
        self._stage("prepare", 0.0)
        items, skipped = learnable(self.project.spec)
        zoom, items, why = choose_zoom(items)
        skipped.update(why)
        if not items:
            raise ValueError("None of the job's checks can be learned: %s" %
                             "; ".join("%s %s" % kv for kv in list(skipped.items())[:5]))
        ds = os.path.join(self.new, "data")
        for sub in ("images", "labels"):
            for sp in ("train", "val"):
                os.makedirs(os.path.join(ds, sub, sp), exist_ok=True)
        rng = np.random.default_rng(0)
        usable = [p for p in pics if p in fixtures]
        pose_list = poses((len(usable) + 1) * SYNTH_PER, rng)
        n = {"train": 0, "val": 0}
        sources = [(self.project.reference, None, "ref")] + [(None, p, "pic%02d" % i)
                                                            for i, p in enumerate(usable)]
        total = len(sources) * (1 + SYNTH_PER)
        done = 0
        for k, (img, path, stem) in enumerate(sources):
            self._check()
            if img is None:
                img = imio.imread(path)
            M = None if path is None else fixtures[path]
            bad = self.fail_on.get(os.path.basename(path), set()) if path else set()
            here = [it for it in items if it["id"] not in bad]
            boxes = boxes_in(here, M)
            # the last real picture is held back to measure the training by
            split = "val" if k == len(sources) - 1 else "train"
            n[split] += write_tiles(img, boxes, zoom, os.path.join(ds, "images", split),
                                    os.path.join(ds, "labels", split), stem, rng)
            done += 1
            for j in range(SYNTH_PER if split == "train" else 0):
                self._check()
                v, A = variant(img, rng, pose_list[(k * SYNTH_PER + j) % len(pose_list)])
                vb = move_boxes(boxes, A, img.shape[1], img.shape[0])
                n["train"] += write_tiles(v, vb, zoom, os.path.join(ds, "images", "train"),
                                          os.path.join(ds, "labels", "train"),
                                          "%s_v%d" % (stem, j), rng)
                done += 1
                self._stage("prepare", done / float(total))
        if n["train"] == 0 or n["val"] == 0:
            raise ValueError("No training pictures could be made from the checks' regions.")
        data = os.path.join(ds, "data.yaml")
        with open(data, "w", encoding="utf-8") as f:
            f.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n%s" % (
                ds.replace("\\", "/"), "".join("  %d: %s\n" % (i, c)
                                               for i, c in enumerate(CLASSES))))
        self._stage("prepare", 1.0, "%d training tiles, %d test tiles" % (n["train"], n["val"]))
        self.tiles = n
        return data, zoom, items, skipped

    def _epochs_for(self, batches):
        if self.epochs:
            return int(self.epochs)
        # about 2400 steps in all: enough for one board, a few minutes on a GPU
        return int(max(15, min(60, round(2400.0 / max(1, batches)))))

    def _train(self, data):
        base, batch = base_weights(self.model_size)
        batches = max(1, int(np.ceil(self.tiles["train"] / float(batch))))
        epochs = self._epochs_for(batches)
        for attempt in range(3):
            self._check()
            out = os.path.join(self.new, "runs")
            shutil.rmtree(out, ignore_errors=True)
            got = self._train_once(data, out, base, epochs, batch)
            if got.get("oom") and batch > 1:
                batch = max(1, batch // 2)           # too big for this card: smaller steps
                self._set(detail="graphics memory full - trying again with smaller batches")
                continue
            if got.get("error"):
                raise RuntimeError("Training stopped: %s" % got["error"])
            return got["onnx"], got.get("metrics") or {}
        raise RuntimeError("Training stopped: the graphics card has too little memory.")

    def _train_once(self, data, out, base, epochs, batch):
        os.makedirs(out, exist_ok=True)
        err_path = os.path.join(self.new, "worker_stderr.txt")
        cmd = [self.python, "-u", WORKER, "--data", data, "--out", out, "--model", base,
               "--epochs", str(epochs), "--imgsz", str(IMGSZ), "--batch", str(batch)]
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        for k in ("PYTHONPATH", "PYTHONHOME", "VS_DEVICE"):
            env.pop(k, None)
        self._set(train_started=time.time(), epochs=epochs, epoch=0)
        self._stage("train", 0.0, "starting the training environment...")
        got = {}
        with open(err_path, "w", encoding="utf-8", errors="replace") as err:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=err, cwd=out, env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            last = {"precision": None, "recall": None}
            for raw in self._proc.stdout:
                if self._cancel.is_set():
                    break
                try:
                    m = json.loads(raw.decode("utf-8", "replace"))
                except ValueError:
                    continue
                t = m.get("t")
                if t == "start":
                    self._set(epochs=int(m.get("epochs") or epochs))
                    self._stage("train", 0.0, "on %s" % m.get("device", "the GPU"))
                elif t == "batch":
                    e, b, nb = int(m["epoch"]), int(m["batch"]), max(1, int(m["batches"]))
                    E = max(1, self.state["epochs"])
                    frac = ((e - 1) + b / float(nb)) / E
                    self._set(epoch=e)
                    self._stage("train", frac, self._train_detail(e, E, last))
                elif t == "epoch":
                    last = m.get("metrics") or last
                    e, E = int(m["epoch"]), max(1, int(m.get("epochs") or self.state["epochs"]))
                    self._set(epoch=e, metrics=last)
                    self._stage("train", e / float(E), self._train_detail(e, E, last))
                elif t == "export":
                    self._stage("train", 1.0, "exporting the model")
                elif t == "done":
                    got = {"onnx": m.get("onnx"), "metrics": m.get("metrics") or last}
                elif t == "error":
                    got = {"error": m.get("message"), "oom": bool(m.get("oom"))}
            self._proc.wait()
        self._check()
        if not got:
            tail = ""
            try:
                with open(err_path, encoding="utf-8", errors="replace") as f:
                    tail = f.read()[-600:]
            except OSError:
                pass
            oom = "out of memory" in tail.lower()
            return {"error": "the training environment stopped (exit %s). %s" % (
                self._proc.returncode, tail.strip().splitlines()[-1] if tail.strip() else ""),
                "oom": oom}
        if got.get("onnx") and not os.path.isfile(got["onnx"]):
            return {"error": "the exported model is missing"}
        return got

    @staticmethod
    def _train_detail(e, E, last):
        s = "pass %d of %d" % (e, E)
        if last and last.get("precision") is not None and last.get("recall") is not None:
            s += " - precision %.0f %%, recall %.0f %%" % (100 * last["precision"],
                                                            100 * last["recall"])
        return s

    def _validate(self, onnx, pics, fixtures, zoom, items):
        """Every learned check found on every new picture, and regions painted
        out not found. A check not found everywhere is left to its classic
        test."""
        self._stage("validate", 0.0)
        meta = {"zoom": zoom, "imgsz": IMGSZ, "min_conf": AI_MIN,
                "tools": {it["id"]: it["cls"] for it in items}}
        det = Detector(self.folder, meta)
        det.path = onnx
        usable = [p for p in pics if p in fixtures]
        found = {it["id"]: 0 for it in items}
        looked = {it["id"]: 0 for it in items}
        conf_min = {it["id"]: 1.0 for it in items}
        last_img = None
        # the new pictures, and every fourth one again with the board put
        # somewhere it was never taught at: a check is trusted only when the
        # network finds it after the board has been moved too
        vrng = np.random.default_rng(7)
        cases = [(p, None) for p in usable] +             [(p, pose) for p, pose in zip(usable[::4], poses(len(usable[::4]), vrng))]
        for i, (p, pose) in enumerate(cases):
            self._check()
            img = imio.imread(p)
            boxes = boxes_in(items, fixtures[p])
            if pose is not None:
                img, A = variant(img, vrng, dict(pose, dx=pose["dx"] * 0.7, dy=pose["dy"] * 0.7))
                boxes = [(mb[0] if mb else None) for mb in
                         (move_boxes([b], A, img.shape[1], img.shape[0]) for b in boxes)]
            cache = {}
            for it, b in zip(items, boxes):
                if b is None:
                    continue                         # moved off the picture
                c, _box = det.find(img, b[1:], it["cls"], cache,
                                   pad=leeway(it["type"], it["spec"].get("params", {}), b[1:]))
                conf_min[it["id"]] = min(conf_min[it["id"]], c)
                found[it["id"]] += c >= AI_MIN
                looked[it["id"]] += 1
            if pose is None:
                last_img, last_M = img, fixtures[p]
            self._stage("validate", 0.8 * (i + 1) / float(len(cases)))
        learned = [it for it in items if looked[it["id"]] and found[it["id"]] == looked[it["id"]]]
        thresholds = {it["id"]: own_threshold(conf_min[it["id"]]) for it in learned}
        # a missing part: a learned region painted with the board round it
        caught, tried = 0, 0
        rng = np.random.default_rng(1)
        sample = list(learned)
        rng.shuffle(sample)
        for it in sample[:MISSING_TESTS]:
            self._check()
            b = boxes_in([it], last_M)
            if not b:
                continue
            x, y, w, h = [int(round(v)) for v in b[0][1:]]
            img = last_img.copy()
            H, W = img.shape[:2]
            m = max(6, int(0.25 * max(w, h)))
            x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
            ring = np.concatenate([img[y0:max(y0 + 1, y), x0:x1].reshape(-1, 3),
                                   img[min(y1 - 1, y + h):y1, x0:x1].reshape(-1, 3)])
            if not len(ring):
                continue
            img[max(0, y - 2):min(H, y + h + 2), max(0, x - 2):min(W, x + w + 2)] = \
                np.median(ring, 0).astype(np.uint8)
            c, _box = det.find(img, (x, y, w, h), it["cls"], None)
            tried += 1
            caught += c < thresholds[it["id"]]
        self._stage("validate", 1.0)
        return {"learned": [it["id"] for it in learned],
                "not_learned": [it["id"] for it in items if it not in learned],
                "found_share": (sum(found.values()) / float(max(1, sum(looked.values())))),
                "missing_caught": caught, "missing_tried": tried,
                "pictures": len(usable), "conf_min": conf_min, "thresholds": thresholds}

    def _install(self, onnx, metrics, val, zoom, items, skipped, report, n_pics):
        self._stage("install", 0.0)
        adv = self.adv
        os.makedirs(adv, exist_ok=True)
        # the model, replacing an older one only now
        tmp = os.path.join(adv, "model.onnx.tmp")
        shutil.copyfile(onnx, tmp)
        os.replace(tmp, model_path(self.folder))
        learned = {it["id"]: it["cls"] for it in items if it["id"] in val["learned"]}
        meta = {"classes": CLASSES, "zoom": zoom, "imgsz": IMGSZ, "min_conf": AI_MIN,
                "tools": learned, "trained_at": time.time(), "metrics": metrics,
                "validation": {k: v for k, v in val.items()
                               if k not in ("conf_min", "thresholds")},
                "thresholds": val.get("thresholds", {}),
                "skipped": skipped, "pictures": n_pics}
        with open(os.path.join(adv, "model.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        # the pictures: the new set replaces an older one, as good examples
        good_old = os.path.join(adv, "good")
        links = self._unlink_old()
        if os.path.isdir(good_old):
            shutil.rmtree(good_old, ignore_errors=True)
        os.replace(os.path.join(self.new, "good"), good_old)
        pics = sorted(glob.glob(os.path.join(good_old, "good_*.png")))
        added = 0
        for t in self.project.spec.get("tools", []):
            if t.get("type") == "align" or not t.get("params", {}).get("roi") and \
                    not t.get("params", {}).get("points"):
                continue
            d = os.path.join(self.folder, "training", t["id"], "ok")
            os.makedirs(d, exist_ok=True)
            for p in pics:
                if t["id"] in self.fail_on.get(os.path.basename(p), set()):
                    continue
                dst = os.path.join(d, "advanced_" + os.path.basename(p))
                try:
                    os.link(p, dst)
                except OSError:
                    shutil.copyfile(p, dst)
                added += 1
        shutil.rmtree(self.new, ignore_errors=True)
        # the job: trained, read back from disk (the window may hold edits)
        pj = os.path.join(self.folder, "project.json")
        with open(pj, encoding="utf-8") as f:
            spec = json.load(f)
        spec["advanced"] = {"requested": True, "trained": True, "model": ADV + "/model.onnx",
                            "trained_at": meta["trained_at"]}
        with open(pj + ".tmp", "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
        os.replace(pj + ".tmp", pj)
        clear_cache()
        self._stage("install", 1.0)
        return {"metrics": metrics, "validation": meta["validation"], "learned": len(learned),
                "checks": len(items) + len(skipped), "skipped": skipped,
                "not_learned": val["not_learned"], "pictures": n_pics,
                "examples_added": added, "failing": report.get("failing", {}),
                "tiles": getattr(self, "tiles", {}),
                "seconds": time.time() - self.state["started"]}

    def _unlink_old(self):
        n = 0
        for p in glob.glob(os.path.join(self.folder, "training", "*", "ok", "advanced_good_*")):
            try:
                os.remove(p)
                n += 1
            except OSError:
                pass
        return n


def remove(project_folder):
    """The advanced model and its good examples gone; the job keeps its
    request (it trains again on the next Run) only if asked."""
    for p in glob.glob(os.path.join(project_folder, "training", "*", "ok", "advanced_good_*")):
        try:
            os.remove(p)
        except OSError:
            pass
    shutil.rmtree(folder_of(project_folder), ignore_errors=True)
    clear_cache()
