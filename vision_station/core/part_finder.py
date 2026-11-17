"""Find parts: every component in an area, found by a neural network on the
graphics card, so they do not have to be boxed one at a time.

The network is Ultralytics YOLOE-26s in its prompt-free form (released
September 2026): an instance-segmentation model trained to find "anything" --
it was never told what a PCB component is, and does not need to be. Its own
names for what it finds (from a vocabulary of 4585 everyday things) are
meaningless on a circuit board, so the model shipped in models/parts was cut
down to box + objectness + mask coefficients (make_part_model.py), which also
makes it 4x faster: about 40 ms per 1280 x 1280 tile on an RTX 5060 through
DirectML.

What turns its raw boxes into a parts list:

  tiles      The area is searched at the camera's full resolution in
             overlapping 1280-pixel tiles, so a 0402 resistor is still a dozen
             pixels and not one; and once more as a whole, scaled down, for
             parts bigger than a tile. A box cut by a tile's inner edge is
             left to the neighbouring tile or the whole-area pass.
  merging    Duplicates from overlapping passes are merged; a box that is
             really a group -- one drawn round a row of capacitors -- is
             dropped in favour of its members.
  board      A box that is mostly bare board (the same colour as the board
             round it: silkscreen, an empty pad field) is not a part.
  types      Parts are grouped by appearance (ResNet-18 features, the same
             network the Trained surface check uses) and size, so twelve
             identical inductors become "type A x 12" to name or drop at once.

Everything found is only a proposal: the user sees it, drops types or single
boxes, and only what is kept becomes "Part present" checks -- each an ordinary
check with an ordinary box to move, resize or delete.
"""
import os
import time

import cv2
import numpy as np

from vision_station import paths

MODEL = os.environ.get("VS_PART_MODEL",
                       paths.model("parts", "parts_yoloe26s.onnx"))
SIZE = 1280              # the model's input
OVERLAP = 256            # tile overlap at full resolution


def available():
    return os.path.exists(MODEL)


def _runtime():
    from vision_station.core import inference
    return inference


def _device():
    if os.environ.get("VS_DEVICE", "").strip().upper() == "CPU":
        return "CPU"
    return "GPU" if _runtime().ort_gpu_providers() else "CPU"


_NET = {}


def _net():
    dev = _device()
    if dev not in _NET:
        try:
            _NET[dev] = _runtime().load(MODEL, "onnxruntime", "accuracy", dev)
        except Exception:
            if dev != "GPU":
                raise
            _NET[dev] = _runtime().load(MODEL, "onnxruntime", "accuracy", "CPU")
    return _NET[dev]


def device_name():
    try:
        return "GPU" if any(p != "CPUExecutionProvider" for p in _net().providers) else "CPU"
    except Exception:
        return "unavailable"


# ---------------------------------------------------------------- the network
def _letterbox(img, size=SIZE):
    h, w = img.shape[:2]
    s = min(size / float(w), size / float(h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:nh, :nw] = img
    return canvas, s


# A pass keeps boxes up to this share of its input's side. Bigger ones are
# mostly the network calling the whole tile "an object" -- at full
# resolution, boxes over half a tile scored a median 0.66, above most real
# parts. A part that big is seen properly by the next, coarser pass.
MAX_SIDE = 0.45


def _infer(img, max_side=MAX_SIDE):
    """Raw candidates in img's pixels: (boxes x1y1x2y2 N x 4, scores N)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lb, s = _letterbox(img)
    x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = _net()(x)[0][0]                      # 37 x N
    score = out[4]
    keep = (score >= 0.02) & (np.maximum(out[2], out[3]) <= max_side * SIZE)
    cx, cy, w, h = out[0, keep], out[1, keep], out[2, keep], out[3, keep]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) / s
    return boxes.astype(np.float32), score[keep].astype(np.float32)


def _nms(boxes, scores, iou=0.5):
    if len(boxes) == 0:
        return np.zeros(0, int)
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    idx = cv2.dnn.NMSBoxes(xywh, [float(s) for s in scores], 0.0, iou)
    return np.array(idx, int).reshape(-1)


def _no_overlap(boxes, scores, share=0.2):
    """Greedy, best first: a box overlapping a kept one by more than `share`
    of the smaller of the two is dropped. Parts on a board do not overlap,
    so a match shifted half a part along a uniform cage is not a second
    part -- plain IoU let those through."""
    order = np.argsort(-scores)
    kept = []
    area = _area(boxes)
    for i in order:
        if kept:
            k = np.array(kept)
            inter = _inter(boxes[i:i + 1], boxes[k])[0]
            if (inter / np.maximum(np.minimum(area[i], area[k]), 1e-6) > share).any():
                continue
        kept.append(i)
    return np.array(kept, int)


def _starts(n, tile, overlap):
    if n <= tile:
        return [0]
    step = tile - overlap
    out = list(range(0, n - tile, step))
    out.append(n - tile)
    return sorted(set(out))


ZOOMS = (2.0, 1.0, 0.6)   # tile passes: magnified for small parts, native, reduced for big ones


def candidates(img, conf=0.10, progress=None, zooms=ZOOMS):
    """Every box the network proposes in img, from tiles at each zoom (2 =
    magnified twice, for parts of a few dozen pixels) and a whole-image
    pass, before any of the clean-up. (boxes, scores)."""
    h, w = img.shape[:2]
    all_b, all_s = [], []
    tiles = []
    for z in zooms:
        src = int(round(SIZE / z))
        ov = int(round(OVERLAP / z))
        if z <= 1.0 or max(h, w) > src * 0.75:
            tiles += [(x, y, src) for y in _starts(h, src, ov) for x in _starts(w, src, ov)]
    passes = len(tiles) + (1 if max(h, w) > SIZE else 0)
    done = 0
    for x0, y0, src in tiles:
        tile = img[y0:y0 + src, x0:x0 + src]
        th, tw = tile.shape[:2]
        z = SIZE / float(max(th, tw)) if max(th, tw) < SIZE else 1.0
        if z > 1.0:
            tile = cv2.resize(tile, (int(round(tw * z)), int(round(th * z))),
                              interpolation=cv2.INTER_CUBIC)
        b, s = _infer(tile)
        b = b / z
        k = s >= conf
        b, s = b[k], s[k]
        # cut by an inner tile edge: another pass sees it whole
        m = 6 / z
        inner = ((b[:, 0] < m) & (x0 > 0)) | ((b[:, 1] < m) & (y0 > 0)) | \
                ((b[:, 2] > tw - m) & (x0 + tw < w)) | ((b[:, 3] > th - m) & (y0 + th < h))
        b, s = b[~inner], s[~inner]
        all_b.append(b + np.float32([x0, y0, x0, y0]))
        all_s.append(s)
        done += 1
        if progress:
            progress(done, passes)
    if max(h, w) > SIZE:
        b, s = _infer(img, max_side=0.6)     # the coarsest pass: big parts
        k = s >= conf
        all_b.append(b[k])
        all_s.append(s[k])
        done += 1
        if progress:
            progress(done, passes)
    b = np.concatenate(all_b) if all_b else np.zeros((0, 4), np.float32)
    s = np.concatenate(all_s) if all_s else np.zeros(0, np.float32)
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, h)
    return b, s


# ---------------------------------------------------------------- clean-up
def _area(b):
    return np.maximum(0, b[..., 2] - b[..., 0]) * np.maximum(0, b[..., 3] - b[..., 1])


def _inter(a, b):
    """Intersection areas, a: N x 4, b: M x 4 -> N x M."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)


def drop_groups_and_duplicates(boxes, scores, others=None, other_scores=None):
    """Indices kept. Biggest first, a box with two or more smaller boxes
    inside it is either ONE part with details -- an SFP cage, a socket, a
    connector block -- or a GROUP drawn round several parts. The network
    says which: it is surer of a real part than of any detail on it, and
    surer of the parts in a group than of the group. So the more confident
    side wins: the container's details are dropped, or the container is.
    (Telling them apart by colour -- does board show inside? -- failed on a
    real board: its green is lighter in the middle than at the edges.)
    A box mostly inside another of about its size is a duplicate.
    `others` are weaker candidates that count as a group's members when the
    container is weaker than they are, but are never kept themselves."""
    n = len(boxes)
    if n == 0:
        return np.zeros(0, int)
    area = _area(boxes)
    inter = _inter(boxes, boxes)
    np.fill_diagonal(inter, 0)
    inside = inter / np.maximum(area[:, None], 1e-6)          # [i, j]: share of i inside j
    keep = np.ones(n, bool)
    have_others = others is not None and len(others)
    if have_others:
        oarea = _area(others)
        o_in = _inter(others, boxes) / np.maximum(oarea[:, None], 1e-6)   # [k, i]
    order = np.argsort(-area)                                 # biggest first
    for i in order:
        if not keep[i]:
            continue
        inner = (inside[:, i] > 0.85) & (area < area[i] * 0.5) & keep
        members = np.where(inner)[0]
        cover = area[members].sum()
        count = len(members)
        best = float(scores[members].max()) if count else 0.0
        if have_others:
            om = (o_in[:, i] > 0.85) & (oarea < area[i] * 0.5)
            if om.any():
                # union-ish coverage: the members' boxes painted on a grid
                count = max(count, int(om.sum()))
                cover = max(cover, _coverage(boxes[i], others[om]))
                if other_scores is not None:
                    best = max(best, float(other_scores[om].max()))
        if count < 2 or cover <= 0.30 * area[i]:
            continue
        if scores[i] > best:
            keep[inner] = False                               # details of one part
        else:
            keep[i] = False                                   # a group: keep its parts
    for i in np.argsort(scores):                              # weakest first
        if not keep[i]:
            continue
        twin = (inside[i] > 0.75) & (area >= area[i] * 0.6) & (area <= area[i] / 0.6) & keep
        twin[i] = False
        if twin.any() and scores[twin].max() >= scores[i]:
            keep[i] = False
    return np.where(keep)[0]


def _coverage(box, members, grid=64):
    """Area of `box` covered by the union of member boxes (approximate)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
    m = np.zeros((grid, grid), bool)
    for b in members:
        a = int(np.clip((b[0] - x1) / w * grid, 0, grid))
        c = int(np.clip(np.ceil((b[2] - x1) / w * grid), 0, grid))
        d = int(np.clip((b[1] - y1) / h * grid, 0, grid))
        e = int(np.clip(np.ceil((b[3] - y1) / h * grid), 0, grid))
        m[d:e, a:c] = True
    return float(m.mean()) * w * h


def board_colour(img, boxes):
    """The board's colour in Lab: the most common colour among the pixels no
    box covers. The most common, not the median: a selection that takes in
    some of the black fixture round the board must not drag the 'board'
    colour towards black."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    step = max(1, int(np.sqrt(img.shape[0] * img.shape[1] / 250000.0)))
    mask = np.ones(img.shape[:2], bool)
    for b in boxes.astype(int):
        mask[b[1]:b[3], b[0]:b[2]] = False
    pts = lab[::step, ::step][mask[::step, ::step]]
    if len(pts) < 100:
        pts = lab[::step, ::step].reshape(-1, 3)
    pts = pts.reshape(-1, 3)
    q = (pts // 12).astype(np.int32)
    key = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    top = vals[np.argmax(counts)]
    return np.median(pts[key == top], axis=0).astype(np.float32), lab


def _differs(lab, board, x1, y1, x2, y2, delta, budget=4000):
    patch = lab[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if patch.size == 0:
        return None
    step = max(1, int(np.sqrt(patch.shape[0] * patch.shape[1] / float(budget))))
    p = patch[::step, ::step].reshape(-1, 3).astype(np.float32)
    return np.sqrt(((p - board) ** 2).sum(axis=1)) > delta


def foreground_share(lab, box, board, delta=22.0):
    """Share of a box's pixels that are not board."""
    x1, y1, x2, y2 = [int(v) for v in box]
    d = _differs(lab, board, x1, y1, x2, y2, delta)
    return 0.0 if d is None else float(d.mean())


def board_around(lab, box, board, delta=22.0):
    """Share of a ring round the box that is board: about none for a hole
    in the fixture beside the board, plenty for anything mounted on it."""
    x1, y1, x2, y2 = [int(v) for v in box]
    m = max(4, int(0.25 * min(x2 - x1, y2 - y1)))
    H, W = lab.shape[:2]
    parts = []
    for a, b, c, d in ((x1 - m, y1 - m, x2 + m, y1), (x1 - m, y2, x2 + m, y2 + m),
                       (x1 - m, y1, x1, y2), (x2, y1, x2 + m, y2)):
        r = _differs(lab, board, max(0, a), max(0, b), min(W, c), min(H, d), delta, 1000)
        if r is not None and r.size:
            parts.append(r)
    if not parts:
        return 1.0
    return float(1.0 - np.concatenate(parts).mean())


# ---------------------------------------------------------------- types
def _embed(img, boxes):
    """One appearance vector per box (unit length): ResNet-18 features pooled
    over the patch, plus its colour. None when the backbone is missing."""
    try:
        from vision_station.tools import deep_surface as ds
        if not ds.available():
            return None
        net = ds._net()
    except Exception:
        return None
    mean = np.float32([0.485, 0.456, 0.406]) * 255.0
    std = np.float32([0.229, 0.224, 0.225]) * 255.0
    out = []
    batch = []

    def flush():
        if not batch:
            return
        x = np.stack(batch)
        f2, f3 = net(x)[:2]
        v = np.concatenate([f2.mean(axis=(2, 3)), f3.mean(axis=(2, 3))], 1)
        out.extend(list(v))
        batch.clear()

    for b in boxes.astype(int):
        p = img[b[1]:b[3], b[0]:b[2]]
        if p.size == 0:
            p = np.zeros((8, 8, 3), np.uint8)
        # orientation-free: a part turned 90 degrees is the same part
        if p.shape[0] > p.shape[1] * 1.15:
            p = cv2.rotate(p, cv2.ROTATE_90_CLOCKWISE)
        p = cv2.resize(p, (96, 64), interpolation=cv2.INTER_AREA)
        batch.append(((p[:, :, ::-1].astype(np.float32) - mean) / std).transpose(2, 0, 1))
        if len(batch) == 32:
            flush()
    flush()
    v = np.array(out, np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    return v


def group_types(img, boxes, similarity=0.86, size_tol=0.30):
    """A type number per box, 0 for the most common type. Parts are one type
    when they look alike and are about the same size (either way round)."""
    n = len(boxes)
    if n == 0:
        return np.zeros(0, int)
    vec = _embed(img, boxes)
    wh = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
    long_ = wh.max(1)
    short = wh.min(1)
    if vec is None:                            # no backbone: mean colour only
        vec = np.array([cv2.mean(img[int(b[1]):int(b[3]), int(b[0]):int(b[2])])[:3]
                        for b in boxes], np.float32)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-6
        similarity = max(similarity, 0.995)
    sim = vec @ vec.T
    same_size = (np.abs(np.log(long_[:, None] / long_[None, :])) < size_tol) & \
                (np.abs(np.log(short[:, None] / short[None, :])) < size_tol)
    link = (sim >= similarity) & same_size
    # connected groups, but each joined through its most typical member: a
    # chain of slightly-different parts must not end up as one type
    label = -np.ones(n, int)
    order = np.argsort(-link.sum(1))
    t = 0
    for i in order:
        if label[i] >= 0:
            continue
        members = np.where(link[i] & (label < 0))[0]
        label[members] = t
        label[i] = t
        t += 1
    counts = np.bincount(label)
    rank = np.argsort(-counts, kind="stable")
    remap = np.empty_like(rank)
    remap[rank] = np.arange(len(rank))
    return remap[label]


def type_name(k):
    """0 -> 'A', 25 -> 'Z', 26 -> 'AA'."""
    s = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        s = chr(65 + r) + s
    return s


# ---------------------------------------------------------------- the whole job
def find_parts(image, rect=None, sensitivity=0.5, progress=None, raw=None):
    """Parts inside rect ([x, y, w, h], None = whole image), in image
    coordinates. sensitivity 0..1: higher finds fainter parts and more
    clutter. Returns {"parts": [{"box": [x, y, w, h], "score", "type"}],
    "types": n, "ms", "device", "candidates": (boxes, scores)} -- pass the
    candidates back as `raw` to refilter at another sensitivity without
    running the network again."""
    t0 = time.perf_counter()
    H, W = image.shape[:2]
    if rect is None:
        rx, ry, rw, rh = 0, 0, W, H
    else:
        rx, ry, rw, rh = [int(round(v)) for v in rect]
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(W - rx, rw), min(H - ry, rh)
    crop = image[ry:ry + rh, rx:rx + rw]
    if crop.size == 0:
        return {"parts": [], "types": 0, "ms": 0.0, "device": "", "candidates": None}
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    if raw is None:
        raw = candidates(crop, conf=0.05, progress=progress)
    boxes, scores = raw
    sens = float(min(1.0, max(0.0, sensitivity)))
    conf = 0.40 - 0.34 * sens                # 0 -> 0.40, 0.5 -> 0.23, 1 -> 0.06
    k = scores >= conf
    b, s = boxes[k], scores[k]

    # size limits: nothing smaller than a few pixels, nothing that is most
    # of the area (that is the board, or the selection itself)
    wh = np.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1) if len(b) else np.zeros((0, 2))
    ok = (wh.min(1) >= 6) & (_area(b) <= 0.25 * rw * rh) if len(b) else np.zeros(0, bool)
    b, s = b[ok], s[ok]
    keep = _nms(b, s, 0.55)
    b, s = b[keep], s[keep]

    if len(b):
        # bare board (silkscreen, empty pads) is not a part; nor is anything
        # with no board round it at all (the fixture beside the board)
        board, lab = board_colour(crop, b)
        fg = np.array([foreground_share(lab, x, board) for x in b])
        ring = np.array([board_around(lab, x, board, delta=40.0) for x in b])
        ok = (fg >= 0.18) & (ring >= 0.08)
        b, s = b[ok], s[ok]
        wk = (scores >= 0.06) & (_area(boxes) <= 0.25 * rw * rh)
        weak, wsc = boxes[wk], scores[wk]
        if len(weak):
            k = _nms(weak, wsc, 0.6)
            weak, wsc = weak[k], wsc[k]
        keep = drop_groups_and_duplicates(b, s, others=weak, other_scores=wsc)
        b, s = b[keep], s[keep]

    types = group_types(crop, b) if len(b) else np.zeros(0, int)
    parts = []
    for box, sc, ty in zip(b, s, types):
        x1, y1, x2, y2 = [float(v) for v in box]
        parts.append({"box": [int(round(x1)) + rx, int(round(y1)) + ry,
                              max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1)))],
                      "score": round(float(sc), 3), "type": int(ty)})
    parts.sort(key=lambda p: (p["type"], p["box"][1] // 40, p["box"][0]))
    return {"parts": parts, "types": int(types.max()) + 1 if len(types) else 0,
            "ms": (time.perf_counter() - t0) * 1000.0, "device": device_name(),
            "candidates": raw}


def find_similar(image, example, rect=None, min_score=0.70, turns=True, max_found=500):
    """Every copy of the part in box `example` ([x, y, w, h]) inside rect,
    turned by 0/90/180/270 degrees when `turns`: normalised correlation,
    peaks at least half a part apart. [{"box", "score"}], the example first.

    The network is weakest exactly where a board is most regular -- sixteen
    identical cages under the same glare -- and matching one good example is
    strongest there, so the two together find what either alone misses."""
    H, W = image.shape[:2]
    ex, ey, ew, eh = [int(round(v)) for v in example]
    if rect is None:
        rx, ry, rw, rh = 0, 0, W, H
    else:
        rx, ry, rw, rh = [int(round(v)) for v in rect]
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(W - rx, rw), min(H - ry, rh)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tpl = gray[max(0, ey):ey + eh, max(0, ex):ex + ew]
    area = gray[ry:ry + rh, rx:rx + rw]
    if tpl.size == 0 or area.size == 0 or tpl.shape[0] < 4 or tpl.shape[1] < 4:
        return []
    # work at a scale where the example's short side is about 32 px: plenty
    # to tell parts apart, and a 20-megapixel board matches in well under a second
    s = min(1.0, 32.0 / max(1, min(tpl.shape[:2])))
    small = cv2.resize(area, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else area
    t0 = cv2.resize(tpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else tpl
    found = []
    rots = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE] \
        if turns else [None]
    for rot in rots:
        t = t0 if rot is None else cv2.rotate(t0, rot)
        th, tw = t.shape[:2]
        if th > small.shape[0] or tw > small.shape[1] or t.std() < 1.0:
            continue
        res = cv2.matchTemplate(small, t, cv2.TM_CCOEFF_NORMED)
        # local maxima above the threshold, at least half a part apart
        k = max(3, (min(th, tw) // 2) | 1)
        peaks = (res >= min_score) & (res >= cv2.dilate(res, np.ones((k, k), np.uint8)))
        for y, x in zip(*np.nonzero(peaks)):
            fw, fh = tw / s, th / s
            found.append([rx + x / s, ry + y / s, rx + x / s + fw, ry + y / s + fh,
                          float(res[y, x])])
    if not found:
        return []
    f = np.array(found, np.float32)
    keep = _no_overlap(f[:, :4], f[:, 4])[:max_found]
    out = [{"box": [int(round(b[0])), int(round(b[1])), int(round(b[2] - b[0])),
                    int(round(b[3] - b[1]))], "score": round(float(b[4]), 3)} for b in f[keep]]
    out.sort(key=lambda p: -p["score"])
    return out


def presence_tools(parts, names, existing_ids, pad=0.08, image_size=None):
    """'Part present' check specs for the kept parts. names: type -> base
    name ('inductor'); ids are numbered per type and never collide with
    existing ones. The box gets a small margin of board round the part --
    what the check compares includes the part's outline -- and stays inside
    the image when image_size (w, h) is given."""
    used = set(existing_ids)
    out = []
    count = {}
    for p in parts:
        base = (names.get(p["type"]) or "part_%s" % type_name(p["type"]).lower())
        base = "".join(c if c.isalnum() or c in "-_" else "_" for c in base.strip()) or "part"
        n = count.get(base, 0)
        while True:
            n += 1
            tid = "%s_%02d" % (base, n)
            if tid not in used:
                break
        count[base] = n
        used.add(tid)
        x, y, w, h = p["box"]
        mx, my = int(round(w * pad)) + 2, int(round(h * pad)) + 2
        x0, y0, x1, y1 = x - mx, y - my, x + w + mx, y + h + my
        if image_size:
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(int(image_size[0]), x1), min(int(image_size[1]), y1)
        out.append({"id": tid, "type": "presence",
                    "params": {"roi": [x0, y0, x1 - x0, y1 - y0], "min_score": 0.65,
                               "search_pad": int(min(60, max(8, 0.15 * max(w, h))))}})
    return out
