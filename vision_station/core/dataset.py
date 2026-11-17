"""Labelled dataset for a model.

Labels live in `projects/<model>/dataset/labels.json` alongside the test images,
so the labelling you do in the app is the same data you train from — nothing gets
retyped or re-drawn.

Export writes a standard Ultralytics folder, a data.yaml, and a train.py, so the
GPU machine needs no knowledge of this application at all.
"""

import json
import os
import random
import shutil

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

CLASS_COLORS = ["#00A8FF", "#FF3B9A", "#18C25A", "#FFA000", "#B36BFF",
                "#FF6B4A", "#00D4C8", "#FFE14A", "#7A8CFF", "#FF5C8A"]


class Dataset:
    def __init__(self, project_folder):
        self.folder = project_folder
        self.dir = os.path.join(project_folder, "dataset")
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "labels.json")
        self.classes = []
        self.boxes = {}          # image filename -> [ {cls, x, y, w, h} ]
        self.load()

    # -- persistence -----------------------------------------------------
    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.classes = d.get("classes", [])
            self.boxes = d.get("boxes", {})
        return self

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"classes": self.classes, "boxes": self.boxes}, f, indent=1)

    # -- classes ---------------------------------------------------------
    def add_class(self, name):
        name = name.strip()
        if name and name not in self.classes:
            self.classes.append(name)
            self.save()
        return self.classes.index(name) if name in self.classes else -1

    def remove_class(self, name):
        if name not in self.classes:
            return
        idx = self.classes.index(name)
        self.classes.remove(name)
        for img, bs in self.boxes.items():
            kept = []
            for b in bs:
                if b["cls"] == idx:
                    continue
                if b["cls"] > idx:
                    b["cls"] -= 1
                kept.append(b)
            self.boxes[img] = kept
        self.save()

    def color(self, cls_idx):
        return CLASS_COLORS[cls_idx % len(CLASS_COLORS)]

    def name(self, cls_idx):
        return self.classes[cls_idx] if 0 <= cls_idx < len(self.classes) else "?"

    # -- boxes -----------------------------------------------------------
    def get(self, image_name):
        return self.boxes.get(image_name, [])

    def set(self, image_name, boxes):
        self.boxes[image_name] = boxes
        self.save()

    def stats(self):
        per_class = {c: 0 for c in self.classes}
        labelled = 0
        for bs in self.boxes.values():
            if bs:
                labelled += 1
            for b in bs:
                n = self.name(b["cls"])
                per_class[n] = per_class.get(n, 0) + 1
        return {"images_labelled": labelled,
                "total_boxes": sum(len(b) for b in self.boxes.values()),
                "per_class": per_class}

    def warnings(self):
        """The things that actually ruin a model. Surfacing them before training
        is far cheaper than discovering them after a four-hour run."""
        w = []
        s = self.stats()
        if not self.classes:
            w.append("No classes defined yet.")
        if s["images_labelled"] < 50:
            w.append(f"Only {s['images_labelled']} labelled images. Aim for 100 or more "
                     "per class, shot across the lighting and part variation you "
                     "actually see on the line.")
        for name, n in s["per_class"].items():
            if n == 0:
                w.append(f"Class '{name}' has no examples — it cannot be learned.")
            elif n < 40:
                w.append(f"Class '{name}' has only {n} examples. Under about 40 the "
                         "model will miss it often.")
        counts = [n for n in s["per_class"].values() if n]
        if len(counts) > 1 and max(counts) > 8 * min(counts):
            w.append("Classes are badly imbalanced. The rare class will be "
                     "under-detected; add more examples of it.")
        return w

    # -- export ----------------------------------------------------------
    def export(self, out_dir, val_split=0.2, seed=0):
        images_dir = os.path.join(self.folder, "images")
        names = [n for n in sorted(self.boxes)
                 if self.boxes[n] and os.path.exists(os.path.join(images_dir, n))]
        if not names or not self.classes:
            raise ValueError("Nothing to export — label some images first.")

        random.Random(seed).shuffle(names)
        cut = max(1, int(len(names) * (1 - val_split))) if len(names) > 1 else 1
        split = {"train": names[:cut], "val": names[cut:] or names[:1]}

        for sub in ("train", "val"):
            os.makedirs(os.path.join(out_dir, "images", sub), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "labels", sub), exist_ok=True)

        import cv2
        from vision_station.core import imio
        for sub, items in split.items():
            for n in items:
                src = os.path.join(images_dir, n)
                shutil.copy(src, os.path.join(out_dir, "images", sub, n))
                img = imio.imread(src)
                if img is None:
                    continue
                H, W = img.shape[:2]
                lines = []
                for b in self.boxes[n]:
                    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, W - x), min(h, H - y)
                    if w <= 1 or h <= 1:
                        continue
                    lines.append(f"{b['cls']} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                                 f"{w / W:.6f} {h / H:.6f}")
                stem = os.path.splitext(n)[0]
                with open(os.path.join(out_dir, "labels", sub, stem + ".txt"),
                          "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

        yaml = ["path: " + os.path.abspath(out_dir).replace("\\", "/"),
                "train: images/train", "val: images/val", "", "names:"]
        yaml += [f"  {i}: {c}" for i, c in enumerate(self.classes)]
        with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
            f.write("\n".join(yaml) + "\n")

        with open(os.path.join(out_dir, "train.py"), "w", encoding="utf-8") as f:
            f.write(TRAIN_SCRIPT)
        with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
            f.write(TRAIN_README.format(train=len(split["train"]), val=len(split["val"]),
                                        classes=", ".join(self.classes)))
        return {"train": len(split["train"]), "val": len(split["val"]),
                "classes": len(self.classes), "path": os.path.abspath(out_dir)}


TRAIN_SCRIPT = '''"""Train a YOLO detector, then export it to ONNX for the line PC.

Run this on a machine with an NVIDIA GPU:
    pip install ultralytics
    python train.py

Copy the printed .onnx file back to your project's models folder.
"""
from ultralytics import YOLO

# yolov8n is the right default for inspection: fast on CPU at inference time.
# Move up to yolov8s only if nano is not accurate enough after more data.
MODEL   = "yolov8n.pt"
EPOCHS  = 150
IMGSZ   = 640
BATCH   = 16
PATIENCE = 40          # stop early if validation stops improving

if __name__ == "__main__":
    m = YOLO(MODEL)
    m.train(
        data="data.yaml",
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        # Augmentation. Match these to the variation your line really has —
        # augmenting for rotation you will never see just wastes model capacity.
        degrees=8.0,        # part rotation on the fixture
        translate=0.08,     # placement slop
        scale=0.25,         # working distance / lens variation
        shear=2.0,
        fliplr=0.0,         # OFF: a mirrored PCB is a different part
        flipud=0.0,
        hsv_h=0.010,        # small: colour matters for LED work
        hsv_s=0.45,
        hsv_v=0.35,         # lighting drift is the big one
        mosaic=0.7,
        close_mosaic=15,
    )
    metrics = m.val()
    print("mAP50    ", round(metrics.box.map50, 4))
    print("mAP50-95 ", round(metrics.box.map, 4))
    path = m.export(format="onnx", opset=12, simplify=True, dynamic=False)
    print("\\nCopy this file to projects/<model>/models/ :\\n  ", path)
'''

TRAIN_README = """YOLO dataset exported from Vision Station

  train images : {train}
  val images   : {val}
  classes      : {classes}

On a machine with an NVIDIA GPU:

    pip install ultralytics
    python train.py

Training prints mAP50 when it finishes. Read it like this:

    above 0.95   ready for the line
    0.85 - 0.95  usable, expect occasional misses
    below 0.85   more labelled images, not more epochs

The last line printed is the path to best.onnx. Copy that file into
    projects\\<your model>\\models\\
then in Vision Station add a "Detect with YOLO" tool and point its
Model file setting at it. No GPU is needed to run it.
"""
