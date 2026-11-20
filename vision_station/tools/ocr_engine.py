"""Find and read every line of text in an image, whatever its quality.

One reader for every text tool. It replaces three that disagreed: a fixed-box
reader that thresholded the picture to black and white before recognising it,
a finder built on edge density, and a finder built on a 41 kB network trained
on synthetic labels. On a real circuit board photographed upside down the
three returned '17 1111111.1', 'MMYTTYYY' and nothing, and took up to 95
seconds doing it.

What it does instead
--------------------
1. A full-size text detector (PP-OCR, differentiable binarisation) marks where
   print is. It works on the photograph as taken: no thresholding, which is
   what destroyed faint silk-screen and low-contrast laser marking.
2. A large photograph is cut into overlapping tiles at full resolution, so 15
   pixel print on a 20 megapixel image is still 15 pixels when the detector
   sees it. A second, coarse pass over the whole image catches lettering too
   big for one tile.
3. Each line is cut out along its own angle and straightened, so print at any
   rotation reaches the recogniser level. Upright against upside down is
   decided per line by an orientation network, and a line that still reads
   poorly is read the other way up and the better reading kept.
4. The recogniser reads the lines in batches.

Runtime
-------
ONNX models in <application>/models/ocr run through onnxruntime: on the GPU
when a GPU build is installed (DirectML or CUDA), on the CPU otherwise. With
no ONNX models there, PaddleOCR is used directly on the CPU -- the same
networks, slower. `status()` says which, for the installation check.
"""
import math
import os
import threading

import cv2
import numpy as np

from .base import runtime

from vision_station import paths

MODEL_DIR = os.environ.get("VS_OCR_MODELS", paths.model("ocr"))

# best first; the first file present is used
DET_FILES = ("det_medium.onnx", "det_server.onnx", "det_small.onnx", "det_mobile.onnx")
REC_FILES = ("rec_medium.onnx", "rec_server.onnx", "rec_small.onnx", "rec_mobile.onnx")
CLS_FILE = "textline_ori.onnx"          # PP-LCNet text line orientation
CLS_FILE_V2 = "textline_cls_v2.onnx"     # the older, smaller classifier

TILE = 1280            # detector tile, pixels
OVERLAP = 192          # shared between neighbouring tiles
REC_H = 48             # recogniser input height
REC_MAX_W = 3200
REC_BATCH = 16

_STATE = {"models": None, "device": None, "error": None, "dict": None}
_LOCK = threading.Lock()


# ------------------------------------------------------------------ models
def _first(names):
    if os.environ.get("VS_DEVICE", "").strip().upper() == "CPU":
        # on the processor the smallest models first: the larger ones are
        # worth their time only on a graphics card (core/system_profile.py)
        names = tuple(reversed(names))
    for n in names:
        p = os.path.join(MODEL_DIR, n)
        if os.path.exists(p):
            return p
    return None


def onnx_available():
    return bool(_first(DET_FILES) and _first(REC_FILES))


def _device():
    """GPU whenever a GPU runtime is installed, unless VS_DEVICE=CPU says no.
    Text reading is the heaviest thing the application does; leaving it on
    the CPU beside an idle GPU is what made it take a minute."""
    if os.environ.get("VS_DEVICE", "").strip().upper() == "CPU":
        return "CPU"
    return "GPU" if runtime().ort_gpu_providers() else "CPU"


def _read_dict(rec_path):
    """The recogniser's characters: blank, the dictionary, then space."""
    stem = os.path.splitext(rec_path)[0]
    for p in (stem + ".txt", os.path.join(MODEL_DIR, "rec_dict.txt")):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                chars = [ln.rstrip("\n").rstrip("\r") for ln in f]
            while chars and chars[-1] == "":
                chars.pop()
            return ["blank"] + chars + [" "]
    raise FileNotFoundError("no character dictionary beside %s" % os.path.basename(rec_path))


def _load(device):
    inf = runtime()
    det, rec = _first(DET_FILES), _first(REC_FILES)
    cls = os.path.join(MODEL_DIR, CLS_FILE)
    cls_v2 = False
    if not os.path.exists(cls):
        cls, cls_v2 = os.path.join(MODEL_DIR, CLS_FILE_V2), True
    m = {"det": inf.load(det, "onnxruntime", "accuracy", device),
         "rec": inf.load(rec, "onnxruntime", "accuracy", device),
         "cls": inf.load(cls, "onnxruntime", "accuracy", device) if os.path.exists(cls) else None,
         "cls_v2": cls_v2,
         "names": (os.path.basename(det), os.path.basename(rec))}
    return m, _read_dict(rec)


def _models():
    """Loaded once. A GPU that will not start the models -- an old driver, no
    memory left -- is not a reason to stop reading text: it is tried once and
    the CPU used from then on, with the reason kept for status()."""
    if _STATE["models"] is not None:
        return _STATE["models"]
    with _LOCK:
        if _STATE["models"] is not None:
            return _STATE["models"]
        device = _device()
        try:
            m, chars = _load(device)
            _self_test(m, chars)
        except Exception as e:
            if device != "GPU":
                raise
            _STATE["error"] = "GPU: %s: %s" % (type(e).__name__, str(e).splitlines()[0][:160])
            device = "CPU"
            m, chars = _load(device)
        _STATE.update(models=m, device=device, dict=chars)
        return m


def _self_test(m, chars):
    img = np.full((64, 256, 3), 255, np.uint8)
    cv2.putText(img, "AB12", (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    _recognise(m, chars, [img])


def status():
    """{'engine': 'onnx' | 'paddle' | None, 'device', 'models', 'error'}."""
    if onnx_available():
        try:
            m = _models()
            return {"engine": "onnx", "device": _STATE["device"], "models": m["names"],
                    "error": _STATE["error"]}
        except Exception as e:
            _STATE["error"] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160])
    from . import text
    e = text._engine()
    return {"engine": e.get("name"), "device": "CPU", "models": (),
            "error": _STATE["error"] or (e.get("errors") or {}).get("paddle")}


def engine_name():
    """What read the text, for the result: 'pp-ocr (gpu)', 'paddle', ..."""
    if _STATE["models"] is not None:
        return "pp-ocr (%s)" % str(_STATE["device"]).lower()
    from . import text
    return text._ENGINE.get("name")


# --------------------------------------------------------------- detection
_MEAN = np.float32([0.485, 0.456, 0.406]) * 255.0
_STD = np.float32([0.229, 0.224, 0.225]) * 255.0


def _det_blob(img):
    """Padded to a multiple of 32 with the image's own edge, never black: a
    black margin is a hard edge the detector reports as print."""
    h, w = img.shape[:2]
    H, W = int(math.ceil(h / 32.0) * 32), int(math.ceil(w / 32.0) * 32)
    if (H, W) != (h, w):
        img = cv2.copyMakeBorder(img, 0, H - h, 0, W - w, cv2.BORDER_REPLICATE)
    x = (img.astype(np.float32) - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def _order(pts):
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    pts = sorted(pts.tolist(), key=lambda p: p[0])
    left = sorted(pts[:2], key=lambda p: p[1])
    right = sorted(pts[2:], key=lambda p: p[1])
    return np.float32([left[0], right[0], right[1], left[1]])


def _boxes_from_map(prob, thresh=0.2, box_thresh=0.45, unclip=1.5, min_side=4):
    """Rotated boxes from the detector's probability map. The map marks the
    shrunken core of each line, so each box is grown back out by `unclip`."""
    mask = (prob > thresh).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts[:3000]:
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        if min(rw, rh) < 2:
            continue
        x, y, w, h = cv2.boundingRect(c)
        sub = np.zeros((h, w), np.uint8)
        cv2.fillPoly(sub, [c - [x, y]], 1)
        score = float(cv2.mean(prob[y:y + h, x:x + w], sub)[0])
        if score < box_thresh:
            continue
        grow = (rw * rh) * unclip / max(1e-6, 2.0 * (rw + rh))
        rw2, rh2 = rw + 2.0 * grow, rh + 2.0 * grow
        if min(rw2, rh2) < min_side:
            continue
        out.append((_order(cv2.boxPoints(((cx, cy), (rw2, rh2), ang))), score))
    return out


def _detect_one(m, img, scale=1.0):
    """Boxes in the coordinates of `img`, detecting at `scale`."""
    work = img if scale == 1.0 else cv2.resize(
        img, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    prob = m["det"](_det_blob(work))[0]
    prob = np.squeeze(prob)[:work.shape[0], :work.shape[1]]
    return [(b / scale, s) for b, s in _boxes_from_map(prob)]


def _tiles(h, w, tile=TILE, overlap=OVERLAP):
    def starts(n):
        if n <= tile:
            return [0]
        step = tile - overlap
        s = list(range(0, n - tile, step)) + [n - tile]
        return sorted(set(s))
    return [(x, y, min(tile, w), min(tile, h)) for y in starts(h) for x in starts(w)]


def _rect(b):
    return (float(b[:, 0].min()), float(b[:, 1].min()),
            float(b[:, 0].max()), float(b[:, 1].max()))


def _merge_across_tiles(items):
    """Boxes found in neighbouring tiles that are one line of print: the same
    line seen twice in the overlap, or a long line cut by the seam and seen as
    two halves. items: [(box, score, tile_index)].

    Only boxes from different tiles are ever joined, and only when one covers
    most of the other or the two lie along one line -- same thickness,
    overlapping across it. Two lines that merely touch stay two: an earlier
    version joined anything that overlapped and, link by link, turned the
    1,540 lines of a circuit board into one box."""
    n = len(items)
    if n < 2:
        return [(b, s) for b, s, _t in items]
    rects = np.float32([_rect(b) for b, _s, _t in items])
    tiles = [t for _b, _s, t in items]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = np.argsort(rects[:, 0])
    for pos, i in enumerate(order):
        ax0, ay0, ax1, ay1 = rects[i]
        aw, ah = ax1 - ax0, ay1 - ay0
        for k in order[pos + 1:]:
            bx0, by0, bx1, by1 = rects[k]
            if bx0 > ax1:
                break
            if tiles[i] == tiles[k]:
                continue
            iw = min(ax1, bx1) - max(ax0, bx0)
            ih = min(ay1, by1) - max(ay0, by0)
            if iw <= 0 or ih <= 0:
                continue
            bw, bh = bx1 - bx0, by1 - by0
            same = iw * ih >= 0.5 * min(aw * ah, bw * bh)
            if not same:
                # along one line: both lie the same way, are as thick as each
                # other, and overlap across their thickness
                if (aw >= ah) != (bw >= bh):
                    continue
                ta, tb, across = (ah, bh, ih) if aw >= ah else (aw, bw, iw)
                if max(ta, tb) > 1.4 * min(ta, tb) or across < 0.7 * min(ta, tb):
                    continue
            ri, rk = find(int(i)), find(int(k))
            if ri != rk:
                parent[rk] = ri
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        if len(members) == 1:
            b, sc, _t = items[members[0]]
            out.append((b, sc))
            continue
        pts = np.concatenate([items[i][0] for i in members]).astype(np.float32)
        out.append((_order(cv2.boxPoints(cv2.minAreaRect(pts))),
                    max(items[i][1] for i in members)))
    return out


def detect(img, detail="auto"):
    """[(box 4x2 float32 in image coordinates, score)] for every line of print.

    detail: 'auto' -- small regions are enlarged, large ones tiled;
            'fine' -- tiled at twice the resolution, for print under ~10 px;
            'fast' -- one pass over the whole region, whatever its size."""
    m = _models()
    h, w = img.shape[:2]
    long_side = max(h, w)
    if detail == "fast" or (detail != "fine" and long_side <= TILE):
        scale = 1.0
        if long_side < 960 and detail != "fast":
            scale = min(3.0, 960.0 / long_side)        # small crop: enlarge the print
        elif long_side > 1600:
            scale = 1600.0 / long_side
        return _detect_one(m, img, scale)

    up = 2.0 if detail == "fine" else 1.0
    work = img if up == 1.0 else cv2.resize(img, None, fx=up, fy=up,
                                            interpolation=cv2.INTER_CUBIC)
    H, W = work.shape[:2]
    found = []
    for ti, (x, y, tw, th) in enumerate(_tiles(H, W)):
        for b, s in _detect_one(m, work[y:y + th, x:x + tw]):
            found.append(((b + [x, y]) / up, s, ti))
    fine = _merge_across_tiles(found)

    # Lettering taller than a tile's overlap is cut into pieces by the tiling;
    # the coarse pass sees it whole. Its boxes replace the pieces inside them.
    coarse_scale = min(1.0, float(TILE) / long_side)
    big = [(b, s) for b, s in _detect_one(m, img, coarse_scale)
           if min(np.linalg.norm(b[0] - b[3]), np.linalg.norm(b[0] - b[1])) > 0.5 * OVERLAP]
    def short(b):
        return float(min(np.linalg.norm(b[0] - b[3]), np.linalg.norm(b[0] - b[1])))

    for bb, bs in big:
        inside = [i for i, (b, _s) in enumerate(fine)
                  if cv2.pointPolygonTest(bb, (float(b[:, 0].mean()), float(b[:, 1].mean())),
                                          False) >= 0]
        # A coarse box holding print much smaller than itself is not big
        # lettering: it is a block of ordinary lines -- or a connector's pin
        # field -- seen from too far away. It swallowed every real line on a
        # whole-board photograph. Only pieces of its own size are replaced.
        if any(short(fine[i][0]) < 0.4 * short(bb) for i in inside):
            continue
        fine = [f for i, f in enumerate(fine) if i not in set(inside)] + [(bb, bs)]
    return fine


# ------------------------------------------------------------- recognition
def _crop_line(img, box):
    """The line cut out along its own angle, not yet turned any way up."""
    w = int(round(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[3] - box[2]))))
    h = int(round(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2]))))
    w, h = max(2, w), max(2, h)
    M = cv2.getPerspectiveTransform(box.astype(np.float32),
                                    np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def _rec_blob(crops, width):
    out = np.zeros((len(crops), 3, REC_H, width), np.float32)
    for i, c in enumerate(crops):
        h, w = c.shape[:2]
        nw = max(8, min(width, int(math.ceil(REC_H * w / float(h)))))
        r = cv2.resize(c, (nw, REC_H), interpolation=cv2.INTER_LINEAR)
        out[i, :, :, :nw] = ((r.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
    return out


def _decode(probs, chars):
    """(text, mean confidence, confidence of each character). Each
    character's own confidence is kept so that junk at the ends of a line --
    the edge of a neighbouring part read as 'P' in 'PR15' -- can be told from
    the print and trimmed (see _trim_junk)."""
    idx = probs.argmax(axis=1)
    mx = probs.max(axis=1)
    text, confs, prev = [], [], 0
    for i, v in zip(idx, mx):
        if i != 0 and i != prev and i < len(chars):
            text.append(chars[i])
            confs.append(float(v))
        prev = i
    return "".join(text), (float(np.mean(confs)) if confs else 0.0), confs


def _recognise(m, chars, crops):
    """[(text, confidence, per-character confidences)] in the order given.
    Lines are batched by shape so a long line does not make forty short ones
    pay for its width, and widths come in steps so the GPU sees a few input
    shapes, not hundreds."""
    out = [("", 0.0, [])] * len(crops)
    order = sorted(range(len(crops)),
                   key=lambda i: crops[i].shape[1] / float(crops[i].shape[0]))
    for s in range(0, len(order), REC_BATCH):
        idx = order[s:s + REC_BATCH]
        ratio = max(crops[i].shape[1] / float(crops[i].shape[0]) for i in idx)
        width = int(min(REC_MAX_W, max(160, math.ceil(REC_H * ratio / 160.0) * 160)))
        probs = m["rec"](_rec_blob([crops[i] for i in idx], width))[0]
        for j, i in enumerate(idx):
            out[i] = _decode(probs[j], chars)
    return out


def _upside_down(m, crops):
    """True for each line the orientation network says is upside down."""
    if m["cls"] is None or not crops:
        return [False] * len(crops)
    flags = []
    for s in range(0, len(crops), 32):
        part = crops[s:s + 32]
        if m.get("cls_v2"):
            # 48 x 192, aspect kept and padded, scaled to -1..1
            x = np.zeros((len(part), 3, 48, 192), np.float32)
            for i, c in enumerate(part):
                nw = max(8, min(192, int(math.ceil(48 * c.shape[1] / float(c.shape[0])))))
                r = cv2.resize(c, (nw, 48), interpolation=cv2.INTER_LINEAR)
                x[i, :, :, :nw] = ((r.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
        else:
            x = np.zeros((len(part), 3, 80, 160), np.float32)
            for i, c in enumerate(part):
                r = cv2.resize(c, (160, 80), interpolation=cv2.INTER_LINEAR)
                x[i] = ((r.astype(np.float32) - _MEAN) / _STD).transpose(2, 0, 1)
        p = m["cls"](x)[0]
        flags.extend(bool(v[1] > v[0]) for v in p)
    return flags


def _strip_confs(text, confs):
    """The per-character confidences of text.strip(): the characters stripped
    from the ends take their confidences with them."""
    lead = len(text) - len(text.lstrip())
    n = len(text.strip())
    return list(confs[lead:lead + n])


def _score(text, conf):
    """How good a reading is, for choosing between the two ways up: its
    confidence, and that it is made of letters and digits. Upside-down print
    often reads as confident punctuation."""
    if not text:
        return 0.0
    alnum = sum(ch.isalnum() for ch in text) / float(len(text))
    return conf * (0.5 + 0.5 * alnum)


def _turns_for(crop):
    """The ways up a cut-out line could be, in quarter turns anticlockwise:
    a wide box is level or upside down, a tall one runs up or down the image,
    and a nearly square one -- 'R15' on a component -- could be any of four."""
    h, w = crop.shape[:2]
    if w >= 1.5 * h:
        return (0, 2)
    if h >= 1.5 * w:
        return (1, 3)
    return (0, 2, 1, 3)


def _read_lines(m, chars, img, boxes, both_ways=True):
    """Read each box every way up it could be, and keep the best.

    An orientation network alone is right about nine lines in ten, and a
    short marking read upside down is often confident nonsense -- 'R15' reads
    as 'SIY'. So every candidate is read, and where the two readings of a
    line are close, the way up that the rest of the region's print agrees on
    decides: the lines of one label, one board, face the same way."""
    crops = [_crop_line(img, b) for b, _s in boxes]
    if not crops:
        return []
    flip = _upside_down(m, crops) if not both_ways else [False] * len(crops)
    jobs, owner = [], []
    for i, c in enumerate(crops):
        turns = _turns_for(c) if both_ways else ((2,) if flip[i] else (0,))
        for k in turns:
            jobs.append(np.ascontiguousarray(np.rot90(c, k)) if k else c)
            owner.append((i, k))
    read = _recognise(m, chars, jobs)
    cands = [[] for _ in crops]
    for (i, k), (t, c, cc) in zip(owner, read):
        cands[i].append((k, t.strip(), c, _score(t.strip(), c), _strip_confs(t, cc)))

    # the way up most of the print agrees on, weighted by how decisive and
    # how long each line is
    votes = {}
    for cs in cands:
        ranked = sorted(cs, key=lambda v: -v[3])
        if len(ranked) < 2 or len(ranked[0][1]) < 3:
            continue
        margin = ranked[0][3] - ranked[1][3]
        if margin >= 0.15:
            votes[ranked[0][0]] = votes.get(ranked[0][0], 0.0) + margin * min(len(ranked[0][1]), 12)
    common = max(votes, key=votes.get) if votes else None

    out = []
    for (box, det_score), cs in zip(boxes, cands):
        best = max(cs, key=lambda v: v[3])
        if common is not None and best[0] != common:
            same = next((v for v in cs if v[0] == common), None)
            if same is not None and same[1] and same[3] >= best[3] - 0.15:
                best = same
        k, t, c, _sc, cc = best
        if not t or looks_like_a_pattern(t):
            continue
        x0, y0, x1, y1 = _rect(box)
        out.append({"text": t, "confidence": round(float(c), 4),
                    "char_conf": [round(float(v), 4) for v in cc],
                    "poly": [[float(px), float(py)] for px, py in box],
                    "box": [int(x0), int(y0), int(round(x1 - x0)), int(round(y1 - y0))],
                    "turn": int(k) * 90, "upside_down": k == 2,
                    "detector_score": round(float(det_score), 3)})
    return out


def looks_like_a_pattern(text):
    """'YYYYYYYYY', 'NYYYNYYYRN': what a row of connector pins, a heat-sink's
    fins or a grille reads as. One letter making up most of a long line is a
    texture, not a word."""
    t = [ch for ch in text if not ch.isspace()]
    if len(t) < 7:
        return False
    top = max(set(t), key=t.count)
    return top.isalpha() and t.count(top) >= 0.6 * len(t)


# ------------------------------------------------------- repeatable results
# A region drawn round a small marking also holds the edge of a pad, the
# corner of the next part or a solder fillet. The reader sees those as print,
# and whether it does changes from one camera frame to the next with sensor
# noise and a fraction of a pixel of movement. Measured on a real board
# (41 text checks, 12 frames of one unmoved part with camera-level noise):
#
#   genuine lines   confidence >= 0.96, the tallest line in the region
#   stray lines     median confidence 0.31, median 0.27 of the main line's height
#   stray letters   joined onto real print ('PR15', 'R15 D'): 0.4 - 0.8 against
#                   >= 0.97 for the print beside them
#
# So one part gave 'R15' on one run and 'PR15' or '451 A' on the next. Stray
# lines are dropped (clean_lines); stray letters are offered as a second
# reading for verification, never removed from what was read (core_text).
FRAGMENT_MAX_CHARS = 2       # a stray mark reads as one or two characters
FRAGMENT_REL_HEIGHT = 0.5    # ... and is under half the height of the print
TRIM_BELOW = 0.85            # an end character read less surely than this ...
TRIM_CORE_ABOVE = 0.95       # ... beside print read at least this surely
TRIM_MAX = 2                 # at most this many from each end


def _thickness(line):
    b = line.get("box") or [0, 0, 0, 0]
    return float(min(b[2], b[3]))


def core_text(line):
    """The line without the characters that are probably stray marks, or
    None when there are none: unsure punctuation anywhere (a glint read as
    an apostrophe in "Ri'5"), and up to TRIM_MAX unsure characters at either
    end, provided what is left was read at TRIM_CORE_ABOVE or better.

    A *second* reading, never a replacement. Confidence cannot tell a
    stray 'P' beside R15 from a faintly printed first letter -- trimming
    in place turned 451 into 51 on one frame in twenty. Verifying against an
    expected text tries both, so a good part matches either way and a wrong
    marking (R10 for R15) matches neither."""
    text, cc = line.get("text", ""), list(line.get("char_conf") or [])
    if len(cc) != len(text) or len(text) < 3:
        return None
    keep = [(ch, c) for ch, c in zip(text, cc)
            if ch.isalnum() or ch == " " or c >= TRIM_BELOW]
    lo, hi, n = 0, len(keep), 0
    while n < TRIM_MAX and hi - lo > 2 and (keep[lo][1] < TRIM_BELOW or keep[lo][0] == " "):
        n += keep[lo][0] != " "
        lo += 1
    n = 0
    while n < TRIM_MAX and hi - lo > 2 and (keep[hi - 1][1] < TRIM_BELOW or keep[hi - 1][0] == " "):
        n += keep[hi - 1][0] != " "
        hi -= 1
    core = keep[lo:hi]
    sure = [c for ch, c in core if ch != " "]
    out = "".join(ch for ch, _c in core).strip()
    if not sure or min(sure) < TRIM_CORE_ABOVE or out == text.strip():
        return None
    return out


def clean_lines(lines):
    """Lines with stray marks removed, for results that repeat run after run:
    fragments -- one or two characters, under half the height of the
    region's main print -- dropped, and each line given its `core` reading
    (core_text) for verification. The main print is the most confidently
    read line of three or more characters; with none, nothing is dropped."""
    lines = [dict(l, core=core_text(l)) if core_text(l) else l for l in lines]
    main = [l for l in lines if len(l["text"].replace(" ", "")) >= 3]
    if not main:
        return lines
    ref = max(main, key=lambda l: l["confidence"])
    tall = _thickness(ref)
    return [l for l in lines
            if l is ref or len(l["text"].replace(" ", "")) > FRAGMENT_MAX_CHARS
            or _thickness(l) >= FRAGMENT_REL_HEIGHT * tall]


def _reading_order(lines):
    """Top to bottom, then left to right, with lines on one row kept together."""
    if not lines:
        return lines
    hs = sorted(min(l["box"][2], l["box"][3]) for l in lines)
    row = max(6.0, 0.7 * hs[len(hs) // 2])
    return sorted(lines, key=lambda l: (int((l["box"][1] + l["box"][3] / 2.0) // row), l["box"][0]))


# ------------------------------------------------------------------ public
def read(image, detail="auto"):
    """Every line of text in `image` (BGR or grey):
    [{'text', 'confidence', 'poly' (4 corners), 'box' [x, y, w, h], ...}],
    in reading order. Coordinates are those of `image`."""
    if image is None or image.size == 0:
        return []
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if onnx_available():
        try:
            m = _models()
        except Exception as e:
            _STATE["error"] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160])
            m = None
        if m is not None:
            boxes = detect(img, detail)
            if detail == "auto" and boxes and img.shape[0] * img.shape[1] <= 8_000_000:
                # Print under about 16 px tall is at the edge of what the
                # detector separates: lines come back in pieces or not at
                # all. Look again at twice the size and keep the richer find.
                short = sorted(min(np.linalg.norm(b[0] - b[3]), np.linalg.norm(b[0] - b[1]))
                               for b, _ in boxes)
                if short[len(short) // 2] < 26 and max(img.shape[:2]) > 480:
                    closer = detect(img, "fine")
                    if len(closer) >= len(boxes):
                        boxes = closer
            lines = _read_lines(m, _STATE["dict"], img, boxes)
            if not any(l["confidence"] >= 0.5 for l in lines) and \
                    img.shape[0] * img.shape[1] <= 4_000_000:
                rescued = _rescue(m, img)
                if rescued:
                    lines = rescued
            return _reading_order(lines)
    return _reading_order(_paddle_read(img, detail))


def _stretch(img):
    g = img.astype(np.float32)
    lo, hi = np.percentile(g, 1), np.percentile(g, 99)
    return np.clip((g - lo) * 255.0 / max(1.0, hi - lo), 0, 255).astype(np.uint8)


def _clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# (name, scale, what to do to the image). In the order tried.
_RESCUES = (
    ("smoothed", 1.0, lambda i: cv2.GaussianBlur(i, (0, 0), 1.2)),
    ("smoothed, contrast raised", 1.0, lambda i: _clahe(cv2.GaussianBlur(i, (0, 0), 1.5))),
    ("enlarged, smoothed, stretched", 2.0,
     lambda i: _stretch(cv2.GaussianBlur(cv2.resize(i, None, fx=2, fy=2,
                                                    interpolation=cv2.INTER_CUBIC), (0, 0), 2.5))),
)


def _rescue(m, img):
    """Nothing was read from the image as taken. Faint print under sensor
    noise is the usual reason: the detector sees grain, not strokes. Smoothing
    the grain away, raising the contrast, or both, brings such print back --
    measured on grey-on-grey marking with noise as strong as the print, where
    the image as taken gave nothing and the smoothed one read every
    character. The best of the attempts is kept, and each line says how it
    was read."""
    best, best_score = [], 0.0
    for name, scale, make in _RESCUES:
        try:
            v = make(img)
            lines = _read_lines(m, _STATE["dict"], v, detect(v, "auto"))
        except cv2.error:
            continue
        lines = [l for l in lines if l["confidence"] >= 0.5]
        score = sum(l["confidence"] * min(len(l["text"]), 20) for l in lines)
        if score > best_score:
            for l in lines:
                l["poly"] = [[x / scale, y / scale] for x, y in l["poly"]]
                l["box"] = [int(v_ / scale) for v_ in l["box"]]
                l["read_from"] = name
            best, best_score = lines, score
    return best


def read_single_line(image):
    """The whole image read as one line, both ways up -- for a box drawn
    tightly round a marking the detector made nothing of."""
    if not onnx_available() or image is None or image.size == 0:
        return None
    m = _models()
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    h, w = img.shape[:2]
    box = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    got = _read_lines(m, _STATE["dict"], img, [(box, 0.0)])
    return got[0] if got else None


# ----------------------------------------------------- PaddleOCR, directly
def _paddle_read(img, detail):
    """The same reading through the PaddleOCR package, for an installation
    without the ONNX models. Its own detector and recogniser run on each tile
    of the photograph as taken."""
    from . import text
    e = text._engine()
    if e["obj"] is None:
        raise RuntimeError("no text reader installed - put the OCR models in models/ocr, "
                           "or pip install paddleocr paddlepaddle")
    if e["name"] != "paddle":
        return [{"text": t, "confidence": float(c), "poly": [], "box": [0, 0, 0, 0],
                 "upside_down": False, "detector_score": 0.0}
                for t, c in text._recognise(img, {"charset": ""}) if t.strip()]
    h, w = img.shape[:2]
    long_side = max(h, w)
    scale = min(3.0, 960.0 / long_side) if long_side < 960 else 1.0
    work = img if scale == 1.0 else cv2.resize(img, None, fx=scale, fy=scale,
                                               interpolation=cv2.INTER_CUBIC)
    tiles = [(0, 0, work.shape[1], work.shape[0])] if (detail == "fast" or long_side <= 2 * TILE) \
        else _tiles(work.shape[0], work.shape[1], 2 * TILE, 2 * OVERLAP)
    found = []
    with text._ENGINE_CALL_LOCK:
        for ti, (x, y, tw, th) in enumerate(tiles):
            for poly, t, c in text._paddle_boxes(e["obj"], work[y:y + th, x:x + tw]):
                box = _order((np.float32(poly).reshape(-1, 2)[:4] + [x, y]) / scale)
                found.append((box, t, c, ti))
    # the same line read in two overlapping tiles: keep the longer reading
    found.sort(key=lambda f: -len(f[1]))
    out, kept = [], []
    for box, t, c, ti in found:
        r = _rect(box)
        dup = False
        for kr, kti in kept:
            if kti == ti:
                continue
            iw = min(r[2], kr[2]) - max(r[0], kr[0])
            ih = min(r[3], kr[3]) - max(r[1], kr[1])
            if iw > 0 and ih > 0 and iw * ih > 0.5 * (r[2] - r[0]) * (r[3] - r[1]):
                dup = True
                break
        if dup or not t.strip():
            continue
        kept.append((r, ti))
        out.append({"text": t.strip(), "confidence": round(float(c), 4),
                    "poly": [[float(px), float(py)] for px, py in box],
                    "box": [int(r[0]), int(r[1]), int(round(r[2] - r[0])), int(round(r[3] - r[1]))],
                    "upside_down": False, "detector_score": 0.0})
    return out
