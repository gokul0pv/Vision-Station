"""YOLO detection tool.

Runs a YOLOv8/v11 model exported to ONNX. Train on a machine with a GPU, export
once, then run here on plain CPU — inference is a single forward pass and needs
no CUDA, no PyTorch, just onnxruntime.

    pip install onnxruntime          (NOT onnxruntime-gpu, on the line PC)

Export from the training machine:
    yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12

Class names travel inside the ONNX metadata, so the tool discovers them
automatically — you do not have to retype them here.
"""

import ast
import os
import time

import cv2
import numpy as np

from .base import Graphic, Tool, ToolResult, register, nms, runtime

_SESSIONS = {}


def _load(path, force_cpu=True, backend="auto", precision="accuracy"):
    """One loaded model per file, runtime and precision, cached. Loading is slow;
    doing it per inspection would dominate the cycle time.

    Runs through the shared backend: OpenVINO when installed, onnxruntime
    otherwise. Accuracy precision by default -- see inference.py for why the
    fast mode is opt-in."""
    key = (os.path.abspath(path), backend, precision)
    if key in _SESSIONS:
        return _SESSIONS[key]
    model = runtime().load(path, backend, precision)

    names = {}
    meta = model.metadata()
    if "names" in meta:
        try:
            names = ast.literal_eval(meta["names"])
        except (ValueError, SyntaxError):
            names = {}
    shape = model.input_shape() or [1, 3, 640, 640]
    size = int(shape[2]) if isinstance(shape[2], int) else 640
    _SESSIONS[key] = (model, size, names)
    return _SESSIONS[key]


def _letterbox(img, size):
    """Resize keeping aspect ratio and pad. Squashing the image instead changes
    object proportions and quietly costs you accuracy."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def infer(path, image, conf=0.35, iou=0.45, force_cpu=True,
          backend="auto", precision="accuracy"):
    """Returns (detections, class_names). Each detection is a dict with
    x, y, w, h in the coordinates of the image you passed in."""
    model, size, names = _load(path, force_cpu, backend, precision)
    padded, ratio, dx, dy = _letterbox(image, size)
    blob = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = model(blob)[0]

    pred = np.squeeze(out)
    if pred.ndim != 2:
        return [], names
    # YOLOv8 exports (4+nc, 8400); some exporters transpose it.
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T                       # -> (8400, 4+nc)

    xywh, cls_scores = pred[:, :4], pred[:, 4:]
    if cls_scores.shape[1] == 0:
        return [], names
    cls_ids = cls_scores.argmax(1)
    confs = cls_scores.max(1)
    keep = confs >= conf
    xywh, cls_ids, confs = xywh[keep], cls_ids[keep], confs[keep]

    boxes = []
    for (cx, cy, w, h) in xywh:
        boxes.append([(cx - w / 2 - dx) / ratio, (cy - h / 2 - dy) / ratio,
                      w / ratio, h / ratio])

    dets = []
    for i in nms(boxes, confs, iou):
        x, y, w, h = boxes[i]
        cid = int(cls_ids[i])
        dets.append({"class_id": cid,
                     "label": names.get(cid, names.get(str(cid), f"class_{cid}")),
                     "confidence": round(float(confs[i]), 4),
                     "x": int(round(x)), "y": int(round(y)),
                     "w": int(round(w)), "h": int(round(h))})
    dets.sort(key=lambda d: (d["y"] // 24, d["x"]))
    return dets, names


def _detections(tool, image, ctx, t0):
    """(detections in image coordinates, None) or (None, error result).

    Shared by both YOLO tools. The two had carried their own copies, and the
    copies had drifted: only one explained a missing inference runtime, so the
    defect tool showed a bare ImportError on a machine without OpenVINO."""
    path = tool.p.get("model")
    if path and not os.path.isabs(path):
        path = os.path.join(getattr(ctx, "folder", "") or "", path)
    if not path or not os.path.exists(path):
        return None, ToolResult(tool.id, tool.type_name, "error",
                                message=f"model file not found: {tool.p.get('model')}",
                                ms=(time.perf_counter() - t0) * 1000)
    crop, (ox, oy) = tool.crop(image, ctx)
    try:
        dets, _ = infer(path, crop,
                        float(tool.p.get("confidence", 0.35)),
                        float(tool.p.get("iou", 0.45)),
                        not tool.p.get("use_gpu", False),
                        tool.p.get("backend", "auto"),
                        tool.p.get("precision", "accuracy"))
    except ImportError:
        return None, ToolResult(tool.id, tool.type_name, "error",
                                message="no inference runtime installed - pip install openvino")
    except Exception as e:
        return None, ToolResult(tool.id, tool.type_name, "error",
                                message=f"{type(e).__name__}: {e}")
    only = tool.p.get("only_classes") or []
    if only:
        dets = [d for d in dets if d["label"] in only]
    min_area = int(tool.p.get("min_area", 0))
    dets = [d for d in dets if d["w"] * d["h"] >= min_area]
    for d in dets:
        d["x"] += ox
        d["y"] += oy
    return dets, None


@register("yolo_detect")
class YoloDetect(Tool):
    """Counts and verifies detected objects inside the region.

    expect_counts is a dict of class name to count, e.g. {"screw": 4, "led": 4}.
    Leave it empty to only require that something was found.
    """

    def run(self, image, ctx):
        t0 = time.perf_counter()
        dets, err = _detections(self, image, ctx, t0)
        if err is not None:
            return err

        counts = {}
        for d in dets:
            counts[d["label"]] = counts.get(d["label"], 0) + 1

        expect = self.p.get("expect_counts") or {}
        total_expect = self.p.get("expect_total")
        problems = []
        for name, want in expect.items():
            got = counts.get(name, 0)
            if got != int(want):
                problems.append(f"{name}: found {got}, expected {want}")
        if total_expect not in (None, "", 0):
            if len(dets) != int(total_expect):
                problems.append(f"total {len(dets)}, expected {total_expect}")
        if not expect and total_expect in (None, "", 0) and not dets:
            problems.append("nothing detected")

        ok = not problems
        g = self.roi_graphic(ctx, "pass" if ok else "fail")
        bad = {n for n in expect if counts.get(n, 0) != int(expect[n])}
        for d in dets:
            st = "fail" if d["label"] in bad else "pass"
            g.append(Graphic("rect", [(d["x"], d["y"]), (d["x"] + d["w"], d["y"] + d["h"])],
                             st, f"{d['label']} {d['confidence']:.2f}"))

        score = float(np.mean([d["confidence"] for d in dets])) if dets else 0.0
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail", score,
                          {"total": len(dets), "counts": counts,
                           "mean_confidence": round(score, 3), "detections": dets},
                          g, "; ".join(problems), (time.perf_counter() - t0) * 1000)


@register("yolo_defect")
class YoloDefect(Tool):
    """Same engine, reject-on-detection. Use for a model trained on defect
    classes: any detection above the threshold fails the part."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        dets, err = _detections(self, image, ctx, t0)
        if err is not None:
            return err

        allowed = int(self.p.get("max_defects", 0))
        ok = len(dets) <= allowed
        counts = {}
        for d in dets:
            counts[d["label"]] = counts.get(d["label"], 0) + 1

        g = self.roi_graphic(ctx, "pass" if ok else "fail")
        for d in dets:
            g.append(Graphic("rect", [(d["x"], d["y"]), (d["x"] + d["w"], d["y"] + d["h"])],
                             "fail", f"{d['label']} {d['confidence']:.2f}"))
        worst = max((d["confidence"] for d in dets), default=0.0)
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail", worst,
                          {"defect_count": len(dets), "counts": counts,
                           "worst_confidence": round(worst, 3), "defects": dets}, g,
                          "" if ok else f"{len(dets)} defect(s): " +
                          ", ".join(f"{k}\u00d7{v}" for k, v in counts.items()),
                          (time.perf_counter() - t0) * 1000)
