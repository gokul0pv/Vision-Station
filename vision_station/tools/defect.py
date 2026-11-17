"""Dent and scratch detection.

Two tools, use both:

  defect_diff   Golden-image differencing after alignment. Deterministic,
                explainable, no training. Classifies each blob by elongation:
                long and thin -> scratch, compact -> dent.

  anomaly       PatchCore / PaDiM exported to ONNX. Trains on GOOD parts only,
                which matters because in semi assembly you rarely have enough
                defect samples to train a supervised detector.

Physics note that outranks both: dents are a GEOMETRY defect, not a texture one.
A dent invisible under flat light is obvious in a surface-normal map. Use a
four-quadrant directional light and feed a photometric-stereo normal image into
defect_diff instead of the raw frame — that single hardware change will beat any
algorithm tuning you do here.
"""
import os
import time
import cv2
import numpy as np
from .base import Tool, ToolResult, Graphic, register, runtime


@register("defect_diff")
class DefectDiff(Tool):
    def run(self, image, ctx):
        t0 = time.perf_counter()
        if ctx.reference is None:
            return ToolResult(self.id, self.type_name, "error", message="reference image required")

        # Warp the reference into the current frame's pose, then compare.
        h, w = image.shape[:2]
        ref = ctx.reference
        if ctx.fixture is not None:
            ref = cv2.warpAffine(ref, ctx.fixture[:2], (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        elif ref.shape[:2] != (h, w):
            ref = cv2.resize(ref, (w, h))

        x, y, cw, ch = self.mapped_roi(ctx) or (0, 0, w, h)
        x, y = max(0, x), max(0, y)
        cw, ch = min(cw, w - x), min(ch, h - y)
        a = cv2.cvtColor(ref[y:y + ch, x:x + cw], cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(image[y:y + ch, x:x + cw], cv2.COLOR_BGR2GRAY)

        # Illumination normalisation — without this, a brighter bulb reads as a
        # defect across the whole part.
        blur = int(self.p.get("illum_blur", 51)) | 1
        a = cv2.divide(a, cv2.GaussianBlur(a, (blur, blur), 0), scale=128)
        b = cv2.divide(b, cv2.GaussianBlur(b, (blur, blur), 0), scale=128)
        a = cv2.GaussianBlur(a, (3, 3), 0)
        b = cv2.GaussianBlur(b, (3, 3), 0)

        diff = cv2.absdiff(a, b)
        thr = int(self.p.get("threshold", 22))
        _, m = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        k = int(self.p.get("open_px", 2))
        if k > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        min_area = float(self.p.get("min_area", 25))
        elong = float(self.p.get("scratch_elongation", 3.5))
        mm_px = float(self.p.get("mm_per_px", 0)) or ctx.mm_per_px

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defects, g = [], self.roi_graphic(ctx)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            long_s, short_s = max(rw, rh), max(1e-6, min(rw, rh))
            kind = "scratch" if long_s / short_s >= elong else "dent"
            bx, by, bw, bh = cv2.boundingRect(c)
            d = {"kind": kind, "area_px": int(area), "length_px": round(long_s, 1),
                 "x": int(bx + x), "y": int(by + y), "w": int(bw), "h": int(bh)}
            if mm_px:
                d["length_mm"] = round(long_s * mm_px, 3)
                d["area_mm2"] = round(area * mm_px * mm_px, 4)
            defects.append(d)
            g.append(Graphic("rect", [(d["x"], d["y"]), (d["x"] + bw, d["y"] + bh)], "fail", kind))

        max_n = int(self.p.get("max_defects", 0))
        max_a = float(self.p.get("max_area_px", 0))
        worst = max((d["area_px"] for d in defects), default=0)
        ok = len(defects) <= max_n and (max_a == 0 or worst <= max_a)

        return ToolResult(self.id, self.type_name, "pass" if ok else "fail",
                          1.0 if ok else 0.0,
                          {"defect_count": len(defects),
                           "scratches": sum(1 for d in defects if d["kind"] == "scratch"),
                           "dents": sum(1 for d in defects if d["kind"] == "dent"),
                           "largest_area_px": worst, "defects": defects},
                          g, "" if ok else f"{len(defects)} defect(s), largest {worst}px",
                          (time.perf_counter() - t0) * 1000)


@register("anomaly")
class Anomaly(Tool):
    """ONNX model exported from anomalib. Train with:
         anomalib train --model Patchcore --data Folder \\
             --data.normal_dir projects/<id>/train_good
         anomalib export --export_type onnx
    Returns a heatmap; blobs above threshold become defect regions."""

    # Sessions are cached per model file. This used to be a single unkeyed
    # class attribute, so the first model loaded was reused for every anomaly
    # tool in the recipe: a second tool pointed at a different model silently
    # returned the first model's heatmap. On a two-surface inspection that means
    # one surface is never actually checked, and nothing in the result says so.
    _sessions = {}

    def _session(self, path):
        """Keyed by model file, runtime and precision -- never shared across
        models (see above). Runs through the shared backend: OpenVINO when
        installed, onnxruntime otherwise, CPU in both cases."""
        inference = runtime()
        backend = self.p.get("backend", "auto")
        precision = self.p.get("precision", "accuracy")
        device = "GPU" if self.p.get("use_gpu", False) else None
        key = (os.path.abspath(path), backend, precision, device)
        if key not in Anomaly._sessions:
            Anomaly._sessions[key] = inference.load(path, backend, precision, device)
        return Anomaly._sessions[key]

    def run(self, image, ctx):
        t0 = time.perf_counter()
        path = self.p.get("model")
        if not path or not os.path.exists(path):
            return ToolResult(self.id, self.type_name, "error",
                              message=f"model not found: {path}")
        try:
            sess = self._session(path)
        except Exception as e:
            return ToolResult(self.id, self.type_name, "error",
                              message=f"inference runtime: {e}")

        crop, (ox, oy) = self.crop(image, ctx)
        size = int(self.p.get("input_size", 256))
        inp = cv2.resize(crop, (size, size)).astype(np.float32) / 255.0
        inp = (inp[:, :, ::-1] - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        inp = np.transpose(inp, (2, 0, 1))[None].astype(np.float32)

        out = sess(inp)
        heat = np.squeeze(out[0]).astype(np.float32)
        thr = float(self.p.get("threshold", 0.5))
        score = float(heat.max())

        g = self.roi_graphic(ctx, "pass" if score < thr else "fail")
        if heat.ndim == 2:
            hm = cv2.resize(heat, (crop.shape[1], crop.shape[0]))
            mask = (hm >= thr).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < float(self.p.get("min_area", 20)):
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                g.append(Graphic("rect", [(bx + ox, by + oy), (bx + ox + bw, by + oy + bh)],
                                 "fail", "anomaly"))

        ok = score < thr
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail", score,
                          {"anomaly_score": round(score, 4), "threshold": thr}, g,
                          "" if ok else f"anomaly score {score:.3f} over {thr}",
                          (time.perf_counter() - t0) * 1000)
