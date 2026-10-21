"""Find text anywhere in a region and read each block separately.

The existing text tools read one fixed box as a single line. That fails on a
real label: a chassis label carrying four fields returned ':' when handed
to the recogniser whole, because it was being asked to read a whole multi-field
label as one line of text.

This tool is given a loose region and locates the text inside it, so the
operator does not have to box each field, and a field that moves between parts
is still found.

Detection works on the morphological gradient, not on brightness. Thresholding
by brightness cannot separate printing from its surroundings on a metal
chassis -- the dark metal around a label and the dark ink on it come out the
same, and an early version returned one block covering the entire region. Edge
density does separate them: text is a dense cluster of short edges, bare metal
has almost none.
"""
import time

import cv2
import numpy as np

from .base import Graphic, Tool, ToolResult, register
from .text import line_for_recognition, _engine, _read


def find_text_blocks(gray, min_h=10, min_w=24, join_div=70):
    """Rectangles that look like lines of print, top-left first."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k)
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    joiner = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(9, gray.shape[1] // max(1, join_div)), 3))
    joined = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, joiner)
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(joined)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                            stats[i, cv2.CC_STAT_AREA])
        if h < min_h or w < min_w:
            continue
        if w / float(h) < 1.5:              # a line of print is wider than tall
            continue
        if h > gray.shape[0] * 0.6:
            continue
        if area < 0.30 * w * h:
            continue
        out.extend(_split_lines(bw, (int(x), int(y), int(w), int(h)), min_h, min_w))
    out = _join_line_fragments(out)
    out.sort(key=lambda b: (b[1], b[0]))
    return out


def _join_line_fragments(boxes, gap_ratio=0.8, overlap=0.6):
    """Pieces of one line of print, put back together.

    The joining kernel above is a fixed width, so a mark printed with wide
    character spacing broke into pieces: 'AB123456' was found as 'AB1' and
    '23456', each under the minimum length, and the region reported no text at
    all. Two boxes are one line when they share most of their height and the
    gap between them is narrower than the print is tall. Separate fields on
    one row are further apart than that and stay separate."""
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for k in range(i + 1, len(boxes)):
                ax, ay, aw, ah = boxes[i]
                bx, by, bw_, bh = boxes[k]
                shared = min(ay + ah, by + bh) - max(ay, by)
                if shared < overlap * min(ah, bh):
                    continue
                gap = max(bx - (ax + aw), ax - (bx + bw_))
                if gap > gap_ratio * max(ah, bh):
                    continue
                x0, y0 = min(ax, bx), min(ay, by)
                x1, y1 = max(ax + aw, bx + bw_), max(ay + ah, by + bh)
                boxes[i] = (x0, y0, x1 - x0, y1 - y0)
                del boxes[k]
                merged = True
                break
            if merged:
                break
    return boxes


def _split_lines(bw, box, min_h, min_w):
    """One block that is really several lines of print, split at the gaps
    between them -- rows with little ink. A slightly blurred or compressed
    photograph weakens the edges, the threshold falls to the label's own
    texture, and every line on a label merged into one block the reader could
    not read: on the real chassis a blur of 1.2 px, or JPEG quality 50, left
    the serial label with no text found at all."""
    x, y, w, h = box
    if h < 2 * min_h:
        return [box]
    rows = (bw[y:y + h, x:x + w] > 0).sum(axis=1).astype(np.float32)
    if rows.max() <= 0:
        return [box]
    k = max(3, h // 40)
    rows = np.convolve(rows, np.ones(k) / k, mode="same")
    ink = rows > 0.18 * rows.max()                 # a text row, not a gap
    runs, start = [], None
    for i, on in enumerate(list(ink) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    lines = []
    for a, b in runs:
        if b - a < max(4, min_h // 2):
            continue
        sub = bw[y + a:y + b, x:x + w] > 0
        cols = np.where(sub.sum(axis=0) > 0)[0]
        if len(cols) == 0:
            continue
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        pad = max(1, (b - a) // 6)
        yy, hh = max(0, y + a - pad), min(bw.shape[0], y + b + pad) - max(0, y + a - pad)
        if x1 - x0 >= min_w:
            lines.append((x + x0, yy, x1 - x0, hh))
    return lines if len(lines) >= 2 else [box]


@register("ocr_auto")
class OcrAuto(Tool):
    """Read every line of text found inside the region.

    Returns one entry per block with its own text, confidence and position, so
    a recipe can check that an expected string appears *somewhere* on a label
    rather than in one exact box.
    """

    def run(self, image, ctx):
        t0 = time.perf_counter()
        crop, (ox, oy) = self.crop(image, ctx)
        if crop is None or crop.size == 0:
            return ToolResult(self.id, self.type_name, "error", 0.0, {},
                              self.roi_graphic(ctx, "error"),
                              "empty region", (time.perf_counter() - t0) * 1000)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        up = float(self.p.get("upscale", 2.0))
        if up != 1.0:
            gray = cv2.resize(gray, None, fx=up, fy=up,
                              interpolation=cv2.INTER_CUBIC)

        # A known orientation is read once. The chassis labels are upside
        # down in the photograph, so an automatic search read upright first --
        # garbage -- then 180; setting 180 halves the text reading, the slowest
        # step of the job (1.4 s of 1.76 s).
        fixed = str(self.p.get("turn", "auto"))
        if fixed in ("0", "90", "180", "270"):
            turns = [int(fixed)]
        elif self.p.get("try_quarter_turns", True):
            turns = [0, 180, 90, 270]
        else:
            turns = [0]

        # Orientation search is a recovery path, not the normal one. Running
        # detection and recognition on all four turns took 5.8 s of a 14 s
        # inspection on a real chassis photograph; most labels are upright, so
        # a good upright result is accepted and the rest are never tried.
        enough = float(self.p.get("accept_score", 5.0))
        best_blocks, best_score, best_turn = [], -1.0, 0
        for turn in turns:
            if best_score >= enough:
                break
            work = gray if turn == 0 else cv2.rotate(
                gray, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                       270: cv2.ROTATE_90_COUNTERCLOCKWISE}[turn])
            blocks = find_text_blocks(
                work, min_h=int(self.p.get("min_text_height", 10)),
                min_w=int(self.p.get("min_text_width", 24)))
            if not blocks:
                continue
            read = self._read_blocks(work, blocks)
            # Score an orientation by how much real text it yielded. Upright is
            # favoured slightly so a rotation has to earn the choice.
            score = sum(len(b["text"]) * b["confidence"] for b in read)
            if turn == 0:
                score *= 1.15
            if score > best_score:
                best_blocks, best_score, best_turn = read, score, turn

        # Filter on length, not confidence.
        #
        # Measured on a real chassis label, Tesseract's confidence is inversely
        # correlated with correctness on this material: the barcode rows it
        # misread as 'LAWN' and 'VAM' scored 0.38 and 0.36, while the two rows
        # it read substantially correctly scored 0.14 and 0.00. A confidence
        # filter therefore discards exactly the results worth keeping. Length
        # does separate them here -- real fields are long, noise is short.
        keep_conf = float(self.p.get("min_confidence", 0.0))
        min_chars = int(self.p.get("min_chars", 6))
        found = []
        for b in best_blocks:
            t = b["text"]
            if not t or len(t) < min_chars or b["confidence"] < keep_conf:
                continue
            alnum = sum(ch.isalnum() for ch in t)
            if alnum < 0.6 * len(t):
                continue
            found.append(b)

        # map block boxes back to whole-image coordinates for the overlay
        g = self.roi_graphic(ctx, "pass")
        H, W = gray.shape[:2]
        for b in found:
            x, y, w, h = b["box"]
            if best_turn == 180:
                x, y = W - x - w, H - y - h
            elif best_turn in (90, 270):
                x, y, w, h = y, x, h, w      # approximate, for display only
            gx, gy = ox + x / up, oy + y / up
            gw, gh = w / up, h / up
            b["image_box"] = [int(gx), int(gy), int(gw), int(gh)]
            g.append(Graphic("rect", [(gx, gy), (gx + gw, gy + gh)], "pass",
                             b["text"][:24]))

        texts = [b["text"] for b in found]
        want = (self.p.get("expect_contains") or "").strip()
        ok = bool(found)
        why = "" if found else "no text found in this region"
        if want:
            joined = " ".join(texts).upper()
            ok = want.upper() in joined
            why = "" if ok else "%r not found; read: %s" % (want, " | ".join(texts))
        mean_conf = float(np.mean([b["confidence"] for b in found])) if found else 0.0

        return ToolResult(
            self.id, self.type_name, "pass" if ok else "fail", mean_conf,
            {"blocks_found": len(found), "texts": texts,
             "quarter_turn": best_turn, "blocks": found,
             "engine": _engine()["name"]},
            g, why, (time.perf_counter() - t0) * 1000)

    def _read_blocks(self, gray, blocks, pad=3):
        out = []
        for (x, y, w, h) in blocks:
            binar = line_for_recognition(gray, (x, y, w, h), pad, scale=2.0)
            if binar is None:
                continue
            rgb = cv2.cvtColor(binar, cv2.COLOR_GRAY2BGR)
            p = dict(self.p)
            p["binarize"] = False       # already binarised for this block
            p["denoise"] = False
            p["upscale"] = 1.0
            p["auto_rotate"] = False
            p["try_quarter_turns"] = False
            lines = _read(rgb, p)
            text = " ".join(t for t, _c in lines).strip()
            conf = float(np.mean([c for _t, c in lines])) if lines else 0.0
            out.append({"box": [x, y, w, h], "text": text,
                        "confidence": round(conf, 3)})
        return out
