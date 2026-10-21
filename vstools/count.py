"""Counting and presence: screws, bolts, LEDs, generic parts."""
import time
import cv2
import numpy as np
from .base import Tool, ToolResult, Graphic, register, nms


def _gray(i):
    return cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) if i.ndim == 3 else i


@register("count_template")
class CountTemplate(Tool):
    """Multi-instance normalised-correlation match. Give it one screw head from
    the reference image and an expected count; it finds all of them in the ROI."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        tr = self.p.get("template_roi")
        if tr is None or ctx.reference is None:
            return ToolResult(self.id, self.type_name, "error",
                              message="set Sample region: x, y, w, h of one example to count, "
                                      "on the reference image")
        tx, ty, tw, th = tr
        tpl = _gray(ctx.reference)[ty:ty + th, tx:tx + tw]
        search, (ox, oy) = self.crop(image, ctx)
        sg = _gray(search)
        if sg.shape[0] < th or sg.shape[1] < tw:
            return ToolResult(self.id, self.type_name, "error", message="ROI smaller than template")

        thr = float(self.p.get("min_score", 0.7))
        res = cv2.matchTemplate(sg, tpl, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= thr)
        boxes = [[int(x), int(y), tw, th] for x, y in zip(xs, ys)]
        scores = [float(res[y, x]) for x, y in zip(xs, ys)]
        keep = nms(boxes, scores, float(self.p.get("overlap", 0.3)))

        expect = self.p.get("expect_count")
        n = len(keep)
        ok = True if expect is None else n == int(expect)
        g = self.roi_graphic(ctx, "pass" if ok else "fail")
        for i in keep:
            x, y, w, h = boxes[i]
            g.append(Graphic("rect", [(x + ox, y + oy), (x + ox + w, y + oy + h)],
                             "pass", f"{scores[i]:.2f}"))
        # where each match is, in image coordinates, for tools that fit a line
        # or circle through them; mapped back through a rotated or fixtured crop
        from .geometry import _ToImage
        to_img = _ToImage(self, ox, oy)
        centres = to_img([(boxes[i][0] + tw / 2.0, boxes[i][1] + th / 2.0) for i in keep]) \
            if keep else []
        matches = [{"x": float(cx - tw / 2.0), "y": float(cy - th / 2.0), "w": tw, "h": th,
                    "score": round(scores[i], 4)} for (cx, cy), i in zip(centres, keep)]
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail",
                          float(np.mean([scores[i] for i in keep])) if keep else 0.0,
                          {"count": n, "expected": expect, "matches": matches},
                          g, "" if ok else f"found {n}, expected {expect}",
                          (time.perf_counter() - t0) * 1000)


@register("count_blob")
class CountBlob(Tool):
    """Threshold-and-count. Good for LED lenses, holes, solder pads.
    space: gray | hsv_v | hsv_s. Set invert for dark features on light ground."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        crop, (ox, oy) = self.crop(image, ctx)
        space = self.p.get("space", "gray")
        if space == "hsv_v":
            ch = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2]
        elif space == "hsv_s":
            ch = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
        else:
            ch = _gray(crop)

        thr = self.p.get("threshold", "otsu")
        if thr == "otsu":
            _, m = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, m = cv2.threshold(ch, int(thr), 255, cv2.THRESH_BINARY)
        if self.p.get("invert"):
            m = cv2.bitwise_not(m)
        k = int(self.p.get("open_px", 3))
        if k > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

        self.record(ctx, "LastRun.BlobImage", cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
        n_lab, _, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        lo, hi = float(self.p.get("min_area", 30)), float(self.p.get("max_area", 1e9))
        blobs = []
        for i in range(1, n_lab):
            x, y, w, h, a = stats[i]
            if lo <= a <= hi:
                blobs.append({"x": int(x + ox), "y": int(y + oy), "w": int(w), "h": int(h),
                              "area": int(a), "cx": float(cent[i][0] + ox), "cy": float(cent[i][1] + oy)})
        blobs.sort(key=lambda b: (b["y"] // max(1, int(self.p.get("row_tol", 20))), b["x"]))

        expect = self.p.get("expect_count")
        n = len(blobs)
        ok = True if expect is None else n == int(expect)
        ctx.share[self.id] = blobs                      # downstream colour/glow tools read this
        g = self.roi_graphic(ctx, "pass" if ok else "fail")
        for i, b in enumerate(blobs):
            g.append(Graphic("rect", [(b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"])],
                             "pass", str(i + 1)))
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail", 1.0 if ok else 0.0,
                          {"count": n, "expected": expect, "blobs": blobs},
                          g, "" if ok else f"found {n}, expected {expect}",
                          (time.perf_counter() - t0) * 1000)


@register("presence")
class Presence(Tool):
    """Is the thing there at all? Correlation against the same ROI in the reference."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        if ctx.reference is None or self.roi is None:
            return ToolResult(self.id, self.type_name, "error", message="roi and reference required")
        x, y, w, h = self.roi
        tpl = _gray(ctx.reference)[y:y + h, x:x + w]
        pad = int(self.p.get("search_pad", 15))
        self.roi = [x - pad, y - pad, w + 2 * pad, h + 2 * pad]
        crop, (ox, oy) = self.crop(image, ctx)
        self.roi = [x, y, w, h]
        cg = _gray(crop)
        if cg.shape[0] < h or cg.shape[1] < w:
            return ToolResult(self.id, self.type_name, "error", message="search window too small")
        # A region with no texture cannot be judged by correlation. Normalised
        # correlation divides by the template's variance, so a region drawn
        # wholly inside a plain sticker -- the neat, natural way to draw it --
        # scored 1.000 against anything: the sticker present, the sticker
        # missing, random clutter. The check passed every part. For such a
        # region the question is really "is it that brightness", so it is
        # answered directly: RMS difference at the best position, 64 grey
        # levels of RMS error scoring zero.
        texture = float(tpl.std())
        if texture < float(self.p.get("min_texture", 4.0)):
            res = cv2.matchTemplate(cg, tpl, cv2.TM_SQDIFF)
            mn, _, loc, _ = cv2.minMaxLoc(res)
            rms = float(np.sqrt(max(0.0, mn) / float(w * h)))
            mx = max(0.0, 1.0 - rms / 64.0)
            mode = "brightness"
        else:
            res = cv2.matchTemplate(cg, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            mode = "pattern"
        thr = float(self.p.get("min_score", 0.65))
        ok = mx >= thr
        g = [Graphic("rect", [(loc[0] + ox, loc[1] + oy), (loc[0] + ox + w, loc[1] + oy + h)],
                     "pass" if ok else "fail", f"{self.id} {mx:.2f}")]
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail", float(mx),
                          {"match": round(float(mx), 4), "mode": mode,
                           "texture": round(texture, 1)}, g,
                          "" if ok else f"match {mx:.2f} below {thr}",
                          (time.perf_counter() - t0) * 1000)
