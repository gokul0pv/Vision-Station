r"""Train a part detector for one board, on this PC's GPU, and measure it.

The general part finder (YOLOE, models/parts/parts_yoloe26s.onnx) was never
taught what a component is; on one unmoved board it reported 142 to 203
parts from one picture to the next. This trains a YOLO11 detector on the
board itself, so the same part is found the same way every time.

    python scripts/train_parts.py capture  D:\parts_ds          pictures of the board
    python scripts/train_parts.py label    D:\parts_ds          labels, by consensus
    C:\Users\ADMIN\vision-station-train\Scripts\python scripts/train_parts.py train D:\parts_ds
    ...\vision-station-train\Scripts\python scripts/train_parts.py export D:\parts_ds   (again)
    ... train D:\parts_ds2 --also D:\parts_ds --start D:\parts_ds\runs\parts\weights\best.pt
              the board moved: a second dataset, both trained, from the earlier model
    python scripts/train_parts.py compare  D:\parts_ds          repeatability, both models
    python scripts/train_parts.py sweep    D:\parts_ds          the light choice, both models

capture   Pictures from the camera at several light levels (the camera's
          automatic exposure and gain held off: they cancel the light).
label     The general part finder on every picture; a part counts when it is
          found in at least AGREE of the pictures, at its median box. What
          flickers in and out is left out -- that flicker is what is being
          trained away. Then tiles at the part finder's three zooms, in YOLO
          format, train/val by picture.
train     Ultralytics YOLO11 (training environment vision-station-train, kept
          OUTSIDE the project -- inside it, VS Code took it for the project's
          Python and the application would not start: PyTorch with
          CUDA), one class "part", input 1280 -- the part finder's own input,
          so the exported model drops into it (core/part_finder.py reads box +
          score). Writes models/parts/parts_trained.onnx.
compare   Fresh pictures; each model's parts per picture, and how often each
          part is found (repeatability), side by side.
"""
import glob
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

LABEL_LEVELS = (160, 192, 224)  # good light: the labels come from these pictures
# Every level the AI reference image tries (core/light_tuning.LEVELS) is
# pictured too and given the same labels -- the board does not move, so a
# part found in good light is there in dim and bright light as well. A model
# taught only on good light judges the light by how familiar a picture
# looks; taught on all of it, the parts it finds say how clear it is.
LEVELS = (32, 64, 96, 128, 160, 176, 192, 224, 255)
PER_LEVEL = 4
TOO_DARK = 0.03                 # mean grey below which a picture teaches nothing
AGREE = 0.6                     # share of pictures a part must be found in
ZOOMS = (2.0, 1.0, 0.6)         # as core/part_finder.ZOOMS
SIZE = 1280
TRAINED = os.path.join(HERE, "models", "parts", "parts_trained.onnx")


# ---------------------------------------------------------------- the line
def _camera_and_light():
    from PySide6.QtCore import QSettings
    from vision_station.hardware.camera import CameraSettings, CameraThread
    from vision_station.hardware import light as L
    qs = QSettings("VisionStation", "QuickBuild")
    s = CameraSettings.from_dict(json.loads(qs.value("camera0/settings")))
    s.auto_exposure = s.auto_gain = False          # they cancel the light
    s.auto_white_balance = False
    cam = CameraThread(0, s)
    cam.start()
    t0 = time.time()
    while cam.backend is None and time.time() - t0 < 15:
        time.sleep(0.05)
    if cam.backend is None:
        raise SystemExit("camera: %s" % cam.last_error)
    lcfg = L.load(qs)
    ctl = L.LightController(lcfg.get("port") or "COM7") if lcfg.get("enabled") else None
    return cam, ctl


def _shots(folder, levels, per_level, tag):
    import cv2
    from vision_station.core import imio
    cam, ctl = _camera_and_light()
    found = ctl.brightness(1) if ctl else None
    out = []
    try:
        for lv in levels:
            if ctl:
                ctl.set_brightness(1, lv)
                ctl.on(1)
            for i in range(per_level):
                img = cam.frame_after(None, settle=3, timeout_s=6)
                if img is None:
                    raise SystemExit("no picture from the camera")
                p = os.path.join(folder, "%s_L%03d_%02d.png" % (tag, lv, i))
                imio.imwrite(p, img)
                out.append(p)
                print("  %s  grey %.2f" % (os.path.basename(p), img[::8, ::8].mean() / 255),
                      flush=True)
    finally:
        if ctl:
            ctl.set_brightness(1, found)
            ctl.on(1)
            ctl.close()
        cam.stop()
    return out


def _level(path):
    return int(os.path.basename(path).split("_L")[1][:3])


def tray(img):
    """(x, y, w, h) of the fixture tray the board lies in: the largest dark,
    unsaturated region (the desk round it is saturated blue). None when
    nothing tray-like covers a fifth of the picture."""
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] < 90) & (hsv[..., 1] < 120)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((41, 41), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
    return (x, y, w, h) if w * h >= 0.2 * img.shape[0] * img.shape[1] else None


def on_tray(boxes, t):
    """The boxes at least 80 % on the tray: a mouse, its cable and whatever
    else lies on the desk are not parts -- labelled once, they were learned
    as parts and became checks that failed when the cable moved. Most of the
    box, not its centre: a cable along the tray's edge had its centre 2 px
    inside."""
    if t is None:
        return list(boxes)
    x, y, w, h = t

    def share(b):
        iw = max(0.0, min(b[0] + b[2], x + w) - max(b[0], x))
        ih = max(0.0, min(b[1] + b[3], y + h) - max(b[1], y))
        return iw * ih / max(1e-6, b[2] * b[3])
    return [b for b in boxes if share(b) >= 0.8]


def img_dark(path):
    from vision_station.core import imio
    return imio.imread(path)[::8, ::8].mean() / 255.0 < TOO_DARK


def capture(ds):
    """Pictures at every level in LEVELS not pictured yet (the board must
    not have moved since the earlier ones: they share the labels)."""
    folder = os.path.join(ds, "shots")
    os.makedirs(folder, exist_ok=True)
    have = {}
    for p in glob.glob(os.path.join(folder, "train_L*.png")):
        have[_level(p)] = have.get(_level(p), 0) + 1
    todo = [lv for lv in LEVELS if have.get(lv, 0) < PER_LEVEL]
    print("pictures at light levels %s (already have %s)" % (todo, sorted(have)))
    if todo:
        _shots(folder, todo, PER_LEVEL, "train")


# ---------------------------------------------------------------- labels
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter)


def consensus(per_picture, agree=AGREE):
    """Parts found in at least `agree` of the pictures, at their median box.
    per_picture: [[x, y, w, h], ...] per picture. Returns (boxes, counts)."""
    import numpy as np
    clusters = []                                  # [boxes]
    for boxes in per_picture:
        used = set()
        for b in boxes:
            best, bi = 0.0, None
            for ci, c in enumerate(clusters):
                if ci in used:
                    continue
                v = _iou(b, np.median(c, 0))
                if v > best:
                    best, bi = v, ci
            if bi is not None and best >= 0.5:
                clusters[bi].append(b)
                used.add(bi)
            else:
                clusters.append([b])
                used.add(len(clusters) - 1)
    need = max(1, int(np.ceil(agree * len(per_picture))))
    keep = [c for c in clusters if len(c) >= need]
    return [list(map(float, np.median(c, 0))) for c in keep], [len(c) for c in keep]


def find(img, model=None):
    """[x, y, w, h] of every part the part finder reports, with `model`
    (default: the general one -- labels never come from a trained model)."""
    from vision_station.core import part_finder as pf
    model = model or pf.GENERAL
    if os.path.abspath(pf.MODEL) != os.path.abspath(model):
        pf.MODEL = model
        pf._NET.clear()
    return [p["box"] for p in pf.find_parts(img, None, 0.5)["parts"]]


def _tiles(h, w, zoom):
    src = int(round(SIZE / zoom))
    ov = int(round(256 / zoom))

    def starts(n):
        if n <= src:
            return [0]
        st = list(range(0, n - src, src - ov)) + [n - src]
        return sorted(set(st))
    return [(x, y, src) for y in starts(h) for x in starts(w)]


def label(ds):
    import cv2
    from vision_station.core import imio
    import shutil
    shots = sorted(glob.glob(os.path.join(ds, "shots", "train_*.png")))
    good = [p for p in shots if _level(p) in LABEL_LEVELS]
    if not good:
        raise SystemExit("no pictures in good light: run capture first")
    lab_file = os.path.join(ds, "labels.json")
    if os.path.isfile(lab_file):                   # the board's parts, found before
        with open(lab_file) as f:
            boxes = json.load(f)["boxes"]
        print("labels: %d parts from %s" % (len(boxes), lab_file))
    else:
        per = []
        for p in good:
            t = time.time()
            per.append(find(imio.imread(p)))
            print("  %s: %d parts (%.1f s)" % (os.path.basename(p), len(per[-1]),
                                               time.time() - t), flush=True)
        boxes, counts = consensus(per)
        print("consensus: %d parts found in at least %d%% of %d pictures (the general finder "
              "found %d-%d per picture)" % (len(boxes), AGREE * 100, len(good),
                                            min(map(len, per)), max(map(len, per))))
        with open(lab_file, "w") as f:
            json.dump({"boxes": boxes, "counts": counts, "pictures": len(good)}, f)
    t = tray(imio.imread(good[0]))
    kept = on_tray(boxes, t)
    if len(kept) != len(boxes):
        print("labels: %d off the tray %s left out (desk, mouse, cable)"
              % (len(boxes) - len(kept), t))
        with open(lab_file) as f:
            saved = json.load(f)
        saved["boxes"], saved["tray"] = kept, list(map(int, t))
        saved.pop("counts", None)                  # counts no longer line up with boxes
        with open(lab_file, "w") as f:
            json.dump(saved, f)
    boxes = kept

    # The labels hold only while the board has not moved: every picture is
    # lined up against the first good-light one.
    def small(p):
        g = cv2.cvtColor(imio.imread(p), cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).astype(np.float32)
        return (g - g.mean()) / (g.std() + 1e-6)
    import numpy as np
    ref = small(good[0])
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    for p in shots:
        if img_dark(p):
            continue
        (dx, dy), _r = cv2.phaseCorrelate(ref, small(p), win)
        if max(abs(dx), abs(dy)) * 4 > 4:
            raise SystemExit("%s: the board moved %.0f, %.0f px since %s -- the labels no "
                             "longer fit. Start a new dataset folder for the board where it "
                             "lies now (capture, label), and train both: train NEW --also OLD."
                             % (os.path.basename(p), dx * 4, dy * 4, os.path.basename(good[0])))

    shutil.rmtree(os.path.join(ds, "yolo"), ignore_errors=True)   # tiles made afresh
    random.seed(0)
    val_pics = set(p for p in shots if int(os.path.basename(p)[-6:-4]) == PER_LEVEL - 1)
    n = {"train": 0, "val": 0}
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(ds, "yolo", sub, split), exist_ok=True)
    for p in shots:
        split = "val" if p in val_pics else "train"
        img = imio.imread(p)
        if img[::8, ::8].mean() / 255.0 < TOO_DARK:
            print("  %s: too dark, left out" % os.path.basename(p))
            continue
        H, W = img.shape[:2]
        for zoom in ZOOMS:
            for (x0, y0, src) in _tiles(H, W, zoom):
                tile = img[y0:y0 + src, x0:x0 + src]
                th, tw = tile.shape[:2]
                lines = []
                for (bx, by, bw, bh) in boxes:
                    ix1, iy1 = max(bx, x0), max(by, y0)
                    ix2, iy2 = min(bx + bw, x0 + tw), min(by + bh, y0 + th)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    if (ix2 - ix1) * (iy2 - iy1) < 0.6 * bw * bh:
                        continue                  # cut by the tile: its neighbour sees it
                    cx, cy = (ix1 + ix2) / 2.0 - x0, (iy1 + iy2) / 2.0 - y0
                    s = float(max(th, tw))
                    lines.append("0 %.6f %.6f %.6f %.6f" % (cx / s, cy / s, (ix2 - ix1) / s,
                                                            (iy2 - iy1) / s))
                if not lines and random.random() > 0.15:
                    continue                      # a few empty tiles teach "not a part"
                if split == "train" and zoom == 2.0 and random.random() > 0.5:
                    continue                      # many near-identical close tiles: half
                canvas = cv2.copyMakeBorder(tile, 0, src - th, 0, src - tw,
                                            cv2.BORDER_CONSTANT, value=(114, 114, 114))
                canvas = cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA
                                    if zoom < 1 else cv2.INTER_LINEAR)
                name = "%s_z%.1f_%d_%d" % (os.path.splitext(os.path.basename(p))[0], zoom, x0, y0)
                cv2.imwrite(os.path.join(ds, "yolo", "images", split, name + ".jpg"), canvas,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                with open(os.path.join(ds, "yolo", "labels", split, name + ".txt"), "w") as f:
                    f.write("\n".join(lines))
                n[split] += 1
    with open(os.path.join(ds, "yolo", "data.yaml"), "w") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: part\n"
                % os.path.join(ds, "yolo").replace("\\", "/"))
    print("tiles: %d train, %d val" % (n["train"], n["val"]))


# ---------------------------------------------------------------- training
def train(ds, epochs=60, model=None, also=()):
    """Run with the training environment's Python (vision-station-train). The model
    size and batch follow this PC's graphics card (core/system_profile.py):
    small at batch 4 with 8 GB, nano with less; no NVIDIA card, no training."""
    from ultralytics import YOLO
    from vision_station.core import system_profile
    ds = os.path.abspath(ds)
    _prof, p = system_profile.current()
    t = p["train"]
    if not t["possible"]:
        raise SystemExit("Training is not possible on this PC: %s" % t["why"])
    print("training %s, batch %d, %s" % (model or t["model"], t["batch"], t["why"]))
    model = os.path.abspath(model) if model and os.path.isfile(model) else model
    # Several datasets -- the board pictured where it lay on different days,
    # each with its own labels -- trained together: one data file listing all.
    sets = [ds] + [os.path.abspath(a) for a in also]
    data = os.path.join(ds, "yolo", "data.yaml")
    if len(sets) > 1:
        data = os.path.join(ds, "yolo", "data_all.yaml")
        with open(data, "w") as f:
            f.write("train:\n%sval:\n%snames:\n  0: part\n" % tuple(
                "".join("  - %s\n" % os.path.join(d, "yolo", "images", sp).replace("\\", "/")
                        for d in sets) for sp in ("train", "val")))
    print("datasets: %s" % ", ".join(sets))
    os.chdir(ds)                       # starting weights download here, not into the project
    y = YOLO(model or t["model"])
    # An absolute project folder: a relative one is put under runs/detect/ by Ultralytics.
    y.train(data=data, imgsz=SIZE, epochs=epochs,
            batch=t["batch"], device=0, workers=2, project=os.path.join(ds, "runs"), name="parts",
            exist_ok=True, patience=20, fliplr=0.0, mosaic=0.5, degrees=2.0,
            translate=0.05, scale=0.25, hsv_v=0.3, hsv_s=0.4, close_mosaic=10,
            seed=0, deterministic=True, verbose=False, plots=False)
    save_dir = getattr(getattr(y, "trainer", None), "save_dir", None)
    export(ds, os.path.join(str(save_dir), "weights", "best.pt") if save_dir else None)


def export(ds, best=None):
    """The trained weights -> models/parts/parts_trained.onnx (the part finder's input)."""
    import shutil
    from ultralytics import YOLO
    best = best or os.path.join(os.path.abspath(ds), "runs", "parts", "weights", "best.pt")
    if not os.path.isfile(best):
        raise SystemExit("no trained weights at %s: run train first" % best)
    onnx = YOLO(best).export(format="onnx", imgsz=SIZE, opset=17, dynamic=False,
                             simplify=True, nms=False)
    os.makedirs(os.path.dirname(TRAINED), exist_ok=True)
    shutil.copyfile(onnx, TRAINED)
    print("model: %s" % TRAINED)


# ---------------------------------------------------------------- comparing
def compare(ds, rounds=8):
    import numpy as np
    from vision_station.core import imio, part_finder as pf
    folder = os.path.join(ds, "check")
    os.makedirs(folder, exist_ok=True)
    pics = sorted(glob.glob(os.path.join(folder, "check_*.png")))
    if not pics:
        pics = _shots(folder, (224,), rounds, "check")
    general, before = pf.GENERAL, pf.MODEL
    for name, model in (("general (YOLOE)", general), ("trained here", TRAINED)):
        per = [find(imio.imread(p), model) for p in pics]
        boxes, counts = consensus(per, agree=0.01)
        steady = [c for c in counts if c == len(pics)]
        flicker = [c for c in counts if c < len(pics)]
        print("%-16s parts per picture %s | found in every picture %d, flickering %d "
              "(repeatability %.1f %%)" % (
                  name, [len(b) for b in per], len(steady), len(flicker),
                  100.0 * sum(counts) / max(1, len(counts) * len(pics))), flush=True)
    pf.MODEL = before
    pf._NET.clear()


def sweep(ds):
    """The AI reference image's light choice (core/light_tuning.py) with each
    model, on one fresh picture per level: what each model finds of the
    board's labelled parts at every level, and the level each would choose."""
    import copy
    from vision_station.core import imio, light_tuning as lt, part_finder as pf
    folder = os.path.join(ds, "sweep")
    os.makedirs(folder, exist_ok=True)
    pics = sorted(glob.glob(os.path.join(folder, "sweep_*.png")))
    if not pics:
        pics = _shots(folder, lt.LEVELS, 1, "sweep")
    with open(os.path.join(ds, "labels.json")) as f:
        lab = json.load(f)["boxes"]
    imgs = [(_level(p), imio.imread(p)) for p in pics]
    # A nudge of the desk moves the board a few pixels: the labels follow it.
    import cv2
    import numpy as np

    def small(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).astype(np.float32)
        return (g - g.mean()) / (g.std() + 1e-6)
    good = sorted(p for p in glob.glob(os.path.join(ds, "shots", "train_*.png"))
                  if _level(p) in LABEL_LEVELS)
    ref = small(imio.imread(good[0]))
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    labs = []
    for lv, im in imgs:
        # shift first (phase correlation), then turn + shift (ECC): a bump of
        # the desk turned the board 0.17 deg -- 12 px across the picture,
        # half a small part
        cur = small(im)
        (dx, dy), _r = cv2.phaseCorrelate(ref, cur, win)
        warp = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
        try:
            _cc, warp = cv2.findTransformECC(ref, cur, warp, cv2.MOTION_EUCLIDEAN,
                                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                              200, 1e-6), None, 5)
        except cv2.error:
            pass
        warp[:, 2] *= 4                            # quarter size -> full size
        moved = []
        for x, y, w, h in lab:
            cx, cy = warp @ np.array([x + w / 2.0, y + h / 2.0, 1.0])
            moved.append([cx - w / 2.0, cy - h / 2.0, w, h])
        labs.append(moved)
        print("  L%d: board moved %.0f, %.0f px, turned %.2f deg since the labels" % (
            lv, warp[0, 2], warp[1, 2], np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))))
    base = [dict(lt.picture_stats(im), level=lv) for lv, im in imgs]
    text = []
    for (lv, im) in imgs:
        a = lt.ai_stats(im)                       # text reader once; parts redone below
        text.append(a["text"])
    before = pf.MODEL
    for name, model in (("general (YOLOE)", pf.GENERAL), ("trained here", TRAINED)):
        rows = copy.deepcopy(base)
        line = []
        for r, (lv, im), tx, lab_here in zip(rows, imgs, text, labs):
            found = find(im, model)
            a = lt.ai_stats(im)                   # with pf.MODEL = model (set by find)
            hit = sum(1 for l in lab_here if any(_iou(l, b) >= 0.5 for b in found))
            r.update(parts=a["parts"], parts_found=a["parts_found"], text=tx)
            line.append("L%d %d/%d" % (lv, hit, len(lab)))
        lt.score(rows)
        best = lt.choose(rows)
        print("%-16s labelled parts found: %s" % (name, "  ".join(line)))
        print("%-16s light step would choose level %d  (scores %s)" % (
            "", best["level"], " ".join("%d:%.2f" % (r["level"], r["score"]) for r in rows)),
            flush=True)
    pf.MODEL = before
    pf._NET.clear()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("capture", "label", "train", "export", "compare", "sweep"))
    ap.add_argument("ds")
    ap.add_argument("--also", nargs="*", default=[], help="train: more dataset folders")
    ap.add_argument("--start", help="train: weights to start from (e.g. an earlier best.pt)")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.ds, a.epochs, a.start, a.also)
    else:
        {"capture": capture, "label": label, "export": export,
         "compare": compare, "sweep": sweep}[a.cmd](a.ds)
