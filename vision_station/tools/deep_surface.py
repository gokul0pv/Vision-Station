"""The trained surface check's deep model: what a good part looks like, in
the features of a pretrained network (the PatchCore method).

The statistical model in learned.py compares each pixel with the same pixel
on the good parts. That is exact where a part repeats to the pixel, and blind
where it does not: a brushed panel whose grain differs on every unit, a cast
surface, a label printed a millimetre to the left. It also has to be told to
ignore lighting.

Here a network trained on a million photographs (ResNet-18, ImageNet) turns
the region into a grid of feature vectors, one per 8 x 8 pixel cell, each
describing the texture and shape around it. Training stores the vectors seen
on good parts -- a memory bank, thinned to its distinct members. A new part is
scored cell by cell by how far its vector is from the nearest one in the bank:
grain that looks like grain is near; a scratch across it, a missing label, a
wrong component is far, wherever it is. No defect has to be shown to it.

The network runs through onnxruntime, on the GPU when a GPU build is
installed. The search through the bank is plain numpy.

Files: models/backbone/resnet18_features.onnx beside the application. Without
it the check falls back to the statistical model.
"""
import os

import cv2
import numpy as np

from .base import runtime

from vision_station import paths

BACKBONE = os.environ.get("VS_BACKBONE",
                          paths.model("backbone", "resnet18_features.onnx"))

BANK_MAX = 12000          # vectors kept in the memory bank
CELL = 8                  # pixels per feature cell
_MEAN = np.float32([0.485, 0.456, 0.406]) * 255.0
_STD = np.float32([0.229, 0.224, 0.225]) * 255.0


def available():
    return os.path.exists(BACKBONE)


def _device():
    if os.environ.get("VS_DEVICE", "").strip().upper() == "CPU":
        return "CPU"
    return "GPU" if runtime().ort_gpu_providers() else "CPU"


_NET = {}


def _net():
    dev = _device()
    if dev not in _NET:
        try:
            _NET[dev] = runtime().load(BACKBONE, "onnxruntime", "accuracy", dev)
            _NET[dev](np.zeros((1, 3, 64, 64), np.float32))
        except Exception:
            if dev != "GPU":
                raise
            _NET[dev] = runtime().load(BACKBONE, "onnxruntime", "accuracy", "CPU")
    return _NET[dev]


def model_size(roi_w, roi_h, resolution=640):
    """(width, height) the region is resampled to: its long side at
    `resolution` or its own size if smaller, both sides a multiple of 32."""
    long_side = max(roi_w, roi_h)
    s = min(1.0, float(resolution) / max(1, long_side))
    w = max(64, int(round(roi_w * s / 32.0)) * 32)
    h = max(64, int(round(roi_h * s / 32.0)) * 32)
    return w, h


def features(patch, size):
    """(vectors N x D float32, (rows, cols)) for a region resampled to size.

    Two depths of the network are combined, as PatchCore does: the shallower
    sees fine texture, the deeper sees what the texture is part of. Each is
    averaged over its 3 x 3 neighbourhood so a vector describes a patch, not
    a point, and a one-cell shift of the part does not change it much."""
    img = patch if patch.ndim == 3 else cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA
                     if img.shape[1] >= size[0] else cv2.INTER_LINEAR)
    x = (img[:, :, ::-1].astype(np.float32) - _MEAN) / _STD
    f2, f3 = _net()(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))[:2]
    f2, f3 = f2[0], f3[0]                                   # C x h x w
    rows, cols = f2.shape[1:]

    def pooled(f):
        return np.stack([cv2.blur(c, (3, 3), borderType=cv2.BORDER_REFLECT) for c in f])

    f2 = pooled(f2)
    f3 = pooled(f3)
    f3 = np.stack([cv2.resize(c, (cols, rows), interpolation=cv2.INTER_LINEAR) for c in f3])
    v = np.concatenate([f2, f3]).reshape(f2.shape[0] + f3.shape[0], -1).T
    return np.ascontiguousarray(v, dtype=np.float32), (rows, cols)


def nearest(v, bank, chunk=4096):
    """Distance from each vector to its nearest neighbour in the bank."""
    b2 = (bank * bank).sum(axis=1)
    out = np.empty(len(v), np.float32)
    for s in range(0, len(v), chunk):
        part = v[s:s + chunk]
        d = (part * part).sum(axis=1)[:, None] + b2[None, :] - 2.0 * (part @ bank.T)
        out[s:s + chunk] = np.sqrt(np.maximum(d.min(axis=1), 0.0))
    return out


def anomaly_map(v, grid, bank):
    """Distance to the bank per cell, lightly smoothed: one odd cell is
    noise, a few together are a mark."""
    d = nearest(v, bank).reshape(grid)
    return cv2.GaussianBlur(d, (0, 0), 1.0)


def score_of(amap):
    """One number for a region: the mean of its worst few cells. The single
    worst cell alone jumps about with noise."""
    flat = np.sort(amap.ravel())
    k = max(4, min(16, flat.size // 400))
    return float(flat[-k:].mean())


def thin(vectors, keep, seed=0):
    """At most `keep` of the vectors, chosen at random. PatchCore's careful
    farthest-point choice was tried: a minute per training against a tenth of
    a second, for a bank this size no better on real parts."""
    n = len(vectors)
    if n <= keep:
        return vectors
    rng = np.random.default_rng(seed)
    return vectors[np.sort(rng.choice(n, keep, replace=False))]


def _jitter(patch, rng):
    """A good part as the next one will arrive: a pixel or two off, a little
    turned, lit a little differently."""
    h, w = patch.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rng.uniform(-1.0, 1.0), 1.0)
    M[0, 2] += rng.uniform(-3, 3)
    M[1, 2] += rng.uniform(-3, 3)
    out = cv2.warpAffine(patch, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return np.clip(out.astype(np.float32) * rng.uniform(0.88, 1.12) + rng.uniform(-8, 8),
                   0, 255).astype(np.uint8)


def train(tool, samples, fixture_fn, out_path, seed=0):
    """Train from (image, "ok" | "ng") samples; write the model to out_path.
    Returns the same report as the statistical trainer.

    The threshold comes from how good parts score against a bank that does
    not contain them -- each OK image against all the others. Scored against
    a bank holding itself every good part scores zero and any threshold
    looks safe."""
    rng = np.random.default_rng(seed)
    res = int(tool.p.get("resolution") or 640)
    size, ok, ng = None, [], []
    for img, label in samples:
        ctx = fixture_fn(img)
        patch, _off = tool.crop(img, ctx)
        if patch is None or patch.size == 0:
            continue
        if size is None:
            size = model_size(patch.shape[1], patch.shape[0], res)
        (ok if label == "ok" else ng).append(patch.copy())
    if not ok:
        raise ValueError("training needs at least one OK image")

    real_ok = len(ok)
    sets = [features(p, size)[0] for p in ok]             # one per real OK image
    grid = features(ok[0], size)[1]
    extra = []
    if real_ok < 8:
        per = int(np.ceil((8 - real_ok) / float(real_ok)))
        for p in ok:
            for _ in range(per):
                extra.append(features(_jitter(p, rng), size)[0])

    def held_out(i):
        others = [s for k, s in enumerate(sets) if k != i] + \
                 [e for k, e in enumerate(extra) if k // max(1, len(extra) // real_ok) != i
                  or real_ok == 1]
        if not others:
            return 0.0
        bank = thin(np.concatenate(others), BANK_MAX // 2, seed)
        return score_of(anomaly_map(sets[i], grid, bank))

    ok_scores = [held_out(i) for i in range(min(real_ok, 12))]
    bank = thin(np.concatenate(sets + extra), BANK_MAX, seed)
    ng_scores = [score_of(anomaly_map(features(p, size)[0], grid, bank)) for p in ng]

    worst_ok = max(ok_scores) if ok_scores else 0.0
    spread = (max(ok_scores) - min(ok_scores)) if len(ok_scores) > 1 else 0.0
    odd_ok = []
    if ng_scores and ok_scores:
        # With both kinds of example the threshold is the split that puts the
        # fewest of them on the wrong side, and of those the widest gap. Not
        # "above the worst OK": one OK photograph taken under another lamp
        # scored 2.9 where the rest scored 1.0 and the bad parts 2.4, and a
        # threshold above it passed every bad part. That photograph is named
        # in the report instead.
        pts = sorted(set(ok_scores) | set(ng_scores))
        best = None
        for lo, hi in zip(pts[:-1], pts[1:]):
            t = (lo + hi) / 2.0
            wrong = sum(v > t for v in ok_scores) + sum(v <= t for v in ng_scores)
            key = (wrong, -(hi - lo))
            if best is None or key < best[0]:
                best = (key, t, lo, hi)
        if best is None:                                   # every score equal
            best = ((len(ng_scores), 0.0), worst_ok * 1.2, worst_ok, worst_ok)
        threshold = best[1]
        odd_ok = [i for i, v in enumerate(ok_scores) if v > threshold]
        unseparated = [i for i, v in enumerate(ng_scores) if v <= threshold]
        kept_ok = [v for v in ok_scores if v <= threshold] or ok_scores
        above = [v for v in ng_scores if v > threshold]
        worst_ok = max(kept_ok)
        best_ng = min(above) if above else min(ng_scores)
        margin = best_ng - worst_ok
        separable = above
    else:
        separable, unseparated = [], []
        best_ng = margin = None
        # no NG examples: a little beyond anything seen on a good part, more
        # where good parts themselves vary a lot
        threshold = worst_ok * 1.2 + 0.25 * spread

    np.savez_compressed(out_path, kind=np.array("deep"), bank=bank.astype(np.float16),
                        size=np.array(size), grid=np.array(grid),
                        threshold=np.array(threshold, np.float32),
                        worst_ok=np.array(worst_ok, np.float32))
    r3 = lambda v: None if v is None else round(float(v), 3)
    return {"real_ok": real_ok, "ng": len(ng), "augmented": len(extra),
            "ok_scores": [r3(v) for v in ok_scores], "ng_scores": [r3(v) for v in ng_scores],
            "worst_ok": r3(worst_ok), "best_ng": r3(best_ng), "margin": r3(margin),
            "threshold": float(threshold), "unseparated_ng": unseparated,
            "separates": (bool(separable) and not unseparated) if ng_scores else True,
            "odd_ok": odd_ok, "method": "deep"}


def load(z):
    return {"kind": "deep", "bank": z["bank"].astype(np.float32),
            "size": tuple(int(v) for v in z["size"]),
            "threshold": float(z["threshold"]),
            "worst_ok": float(z["worst_ok"]) if "worst_ok" in z.files else 0.0}


def inspect(patch, m, sensitivity=1.0):
    """(score, threshold, [defect boxes x, y, w, h in patch pixels])."""
    v, grid = features(patch, m["size"])
    amap = anomaly_map(v, grid, m["bank"])
    thr = m["threshold"] * (float(sensitivity) if sensitivity and sensitivity > 0 else 1.0)
    score = score_of(amap)
    boxes = []
    if score > thr:
        mask = (amap > thr * 0.85).astype(np.uint8)
        sx = patch.shape[1] / float(grid[1])
        sy = patch.shape[0] / float(grid[0])
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ranked = sorted(cnts, key=lambda c: -float(amap[cv2.boundingRect(c)[1]:
                                                        cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3],
                                                        cv2.boundingRect(c)[0]:
                                                        cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]].max()))
        for c in ranked[:6]:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x * sx, y * sy, w * sx, h * sy))
    return score, thr, boxes
