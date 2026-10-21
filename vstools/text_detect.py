"""Find text with a trained network, then read each line that it finds.

The classical detector this replaces works on edge density, which cannot tell a
line of print from a barcode -- both are dense clusters of short vertical edges.
Measured on a real chassis label, the hand-written discriminator I tried had
overlapping distributions for the two classes and could not be tuned reliably.

So the network is trained on synthetic labels in which barcodes appear and are
labelled as not-text. It predicts, per pixel, whether that pixel belongs to a
line of print; boxes come from connected components of that map. Nothing about
it is pretrained or downloaded: `textnet_train.py` builds it from generated data
in minutes on a CPU, and the result is a 41 kB ONNX file that runs in about
16 ms.
"""
import os
import time

import cv2
import numpy as np

from .base import Graphic, Tool, ToolResult, register, runtime
from .text import line_for_recognition, _engine, _read

def _model(path, backend="auto", precision="accuracy"):
    return runtime().load(path, backend, precision)


def default_model():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "models", "textnet.onnx")


def detect_text_map(gray, model_path=None, long_side=512,
                    backend="auto", precision="accuracy"):
    """Per-pixel probability that a pixel belongs to a line of print."""
    path = model_path or default_model()
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    net = _model(path, backend, precision)
    h, w = gray.shape[:2]
    scale = min(1.0, float(long_side) / max(h, w))
    if scale < 1.0:
        work = cv2.resize(gray, (max(8, int(w * scale)), max(8, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
    else:
        work = gray
    # the network is fully convolutional but wants sizes that survive three
    # halvings without rounding drift
    H = int(np.ceil(work.shape[0] / 8.0) * 8)
    W = int(np.ceil(work.shape[1] / 8.0) * 8)
    padded = cv2.copyMakeBorder(work, 0, H - work.shape[0], 0, W - work.shape[1],
                                cv2.BORDER_REPLICATE)
    x = padded.astype(np.float32)[None, None] / 255.0
    logits = net(x)[0]
    prob = 1.0 / (1.0 + np.exp(-logits[0, 0]))
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return prob


def boxes_from_map(prob, thresh=0.5, min_h=8, min_w=20):
    m = (prob > thresh).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    n, _l, stats, _c = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                            stats[i, cv2.CC_STAT_AREA])
        if h < min_h or w < min_w or area < 0.25 * w * h:
            continue
        out.append((int(x), int(y), int(w), int(h)))
    out.sort(key=lambda b: (b[1], b[0]))
    return out


@register("text_detect")
class TextDetect(Tool):
    """Locate every line of print in the region with the trained detector and
    read each one."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        crop, (ox, oy) = self.crop(image, ctx)
        if crop is None or crop.size == 0:
            return ToolResult(self.id, self.type_name, "error", 0.0, {},
                              self.roi_graphic(ctx, "error"), "empty region",
                              (time.perf_counter() - t0) * 1000)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

        model = self.p.get("model") or ""
        if model and not os.path.isabs(model):
            model = os.path.join(getattr(ctx, "folder", "") or "", model)
        try:
            prob = detect_text_map(gray, model or None,
                                   int(self.p.get("long_side", 512)),
                                   self.p.get("backend", "auto"),
                                   self.p.get("precision", "accuracy"))
        except FileNotFoundError as e:
            return ToolResult(
                self.id, self.type_name, "error", 0.0, {},
                self.roi_graphic(ctx, "error"),
                "text model not found (%s) - run textnet_train.py to build it"
                % os.path.basename(str(e)),
                (time.perf_counter() - t0) * 1000)
        except Exception as e:
            return ToolResult(self.id, self.type_name, "error", 0.0, {},
                              self.roi_graphic(ctx, "error"),
                              "%s: %s" % (type(e).__name__, e),
                              (time.perf_counter() - t0) * 1000)

        # Orientation is decided once for the whole region, not per line.
        #
        # Per-line choice was tried and is worse: on a real chassis label the
        # upside-down reading of a line is often longer and no less confident
        # than the correct one, so nothing local separates them. Summed across
        # every line in the region the difference is clear, because a whole
        # label read the right way up produces far more recognisable characters
        # than the same label read upside down.
        read_text = bool(self.p.get("read_text", True))
        orient = str(self.p.get("orientation", "auto")).lower()
        turns = [0] if orient == "0" else ([180] if orient == "180" else [0, 180])
        if not read_text:
            turns = [0]

        best = None
        for turn in turns:
            work = gray if turn == 0 else cv2.rotate(gray, cv2.ROTATE_180)
            pmap = prob if turn == 0 else cv2.rotate(prob, cv2.ROTATE_180)
            bxs = boxes_from_map(pmap, float(self.p.get("threshold", 0.5)),
                                 int(self.p.get("min_text_height", 8)),
                                 int(self.p.get("min_text_width", 20)))
            texts = [self._read_one(work, *b) for b in bxs] if read_text else \
                    ["" for _ in bxs]
            score = sum(sum(ch.isalnum() for ch in t) for t in texts)
            if best is None or score > best[0]:
                best = (score, turn, work, pmap, bxs, texts)
        _score, turn, gwork, prob, boxes, texts = best
        self._turn = turn

        results = []
        for (x, y, w, h), _t in zip(boxes, texts):
            H, W = gwork.shape[:2]
            ix, iy = (x, y) if turn == 0 else (W - x - w, H - y - h)
            results.append({
                "box": [x, y, w, h],
                "image_box": [int(ox + ix), int(oy + iy), int(w), int(h)],
                "confidence": float(prob[y:y + h, x:x + w].mean()),
                "text": _t})

        if read_text:
            results = [r for r in results
                       if len(r["text"]) >= int(self.p.get("min_chars", 3))]

        want = (self.p.get("expect_contains") or "").strip()
        ok = bool(results)
        why = "" if results else "no text found in this region"
        if want:
            joined = " ".join(r["text"] for r in results).upper()
            ok = want.upper() in joined
            why = "" if ok else "%r not found; read: %s" % (
                want, " | ".join(r["text"] for r in results))

        g = self.roi_graphic(ctx, "pass" if ok else "fail")
        for r in results:
            bx, by, bw, bh = r["image_box"]
            g.append(Graphic("rect", [(bx, by), (bx + bw, by + bh)], "pass",
                             (r["text"] or "text")[:22]))

        conf = float(np.mean([r["confidence"] for r in results])) if results else 0.0
        return ToolResult(
            self.id, self.type_name, "pass" if ok else "fail", conf,
            {"blocks_found": len(results),
             "quarter_turn": turn,
             "texts": [r["text"] for r in results],
             "blocks": results, "engine": _engine()["name"]},
            g, why, (time.perf_counter() - t0) * 1000)

    def _read_one(self, gray, x, y, w, h, pad=4):
        """Read one detected line, trying it both ways up.

        The detector locates print regardless of orientation -- it was trained
        on rotated labels -- but the recogniser is not so forgiving. On a real
        chassis label photographed upside down the boxes were perfect and the
        text came back as '-tOASAS-OTWONZ-CrSH'. Each line is cheap to try
        twice, and doing it per line rather than per region also copes with a
        label that has text running in more than one direction."""
        txt, _conf = self._read_oriented(gray, x, y, w, h, pad, False)
        return txt

    def _read_oriented(self, gray, x, y, w, h, pad=4, flip=False):
        binar = line_for_recognition(gray, (x, y, w, h), pad,
                                     target_h=int(self.p.get("recognise_height", 48)))
        if binar is None:
            return "", 0.0
        if flip:
            binar = cv2.rotate(binar, cv2.ROTATE_180)
        p = dict(self.p)
        # Uniform-block mode, not single-line. A detected box often contains a
        # line of print with a barcode row directly above or below it -- they
        # sit that close on a real label -- and single-line mode picks whichever
        # row it likes, which on a real chassis meant reading the barcode and
        # returning nonsense. Block mode returns every row and the best is
        # chosen below.
        p.update(binarize=False, denoise=False, upscale=1.0,
                 auto_rotate=False, try_quarter_turns=False,
                 psm=int(self.p.get("psm", 6)))
        lines = _read(cv2.cvtColor(binar, cv2.COLOR_GRAY2BGR), p)
        if not lines:
            return "", 0.0
        # Rows come back as separate words; group them and keep the row with the
        # most alphanumeric content, which is the printed field rather than the
        # handful of stray glyphs a barcode produces.
        joined = " ".join(t for t, _c in lines).strip()
        best_row, best_n = joined, sum(ch.isalnum() for ch in joined)
        for row in joined.split("  "):
            n = sum(ch.isalnum() for ch in row)
            if n > best_n:
                best_row, best_n = row.strip(), n
        conf = float(np.mean([c for _t, c in lines]))
        return best_row, conf
