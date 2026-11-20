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
        tpl = _gray(ctx.reference[ty:ty + th, tx:tx + tw])
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


# ---------------------------------------------------------------- examples
# A check's OK and NG pictures from its Train tab, as the patch each holds
# where the check's region lies -- read once, kept while the files and the
# region stay the same.
_EXAMPLES = {}
# One decoded copy of each example picture, shared by every check while it is
# read. Advanced training adds the same 8 pictures to all of a job's checks
# (hard links): read per check, 233 checks decoded 1,864 twenty-megapixel
# files -- minutes -- on the first run.
_PICTURES = {}
# more than the pictures a check has: with fewer, each check threw out the
# picture the next one needed first -- 16 pictures, 141 checks, every picture
# decoded by every check (minutes). Let go after each run (release_pictures).
_PICTURES_MAX = 40
EXAMPLES_MAX = 4000         # entries; each a small patch and a thumbnail


def _picture(path):
    """(image, thumbnail) of an example picture, decoded once for every
    check that has it -- the same file under several names (a hard link)
    too."""
    import os
    from vision_station.core import imio
    try:
        st = os.stat(path)
    except OSError:
        return None, None
    key = (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size) if st.st_ino else         (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    hit = _PICTURES.get(key)
    if hit is not None:
        return hit
    img = imio.imread(path)
    got = (img, None if img is None else thumb(img))
    if len(_PICTURES) >= _PICTURES_MAX:
        _PICTURES.pop(next(iter(_PICTURES)))
    _PICTURES[key] = got
    return got


def release_pictures():
    """The decoded pictures let go (after a run): the patches stay cached."""
    _PICTURES.clear()
# How much closer to an NG example than to every good one a part must be to
# be rejected for it. An NG example that is in fact a good part is as like
# every good board as the reference is; this keeps the boards it was not
# taken from passing.
NG_MARGIN = 0.03
# The very picture an NG example was made from (or a still re-shot of it):
# mean grey difference of the two whole pictures shrunk to THUMB, in grey
# levels. Labelled NG, that picture is NG -- whatever it looks like.
SAME_PICTURE_GREY = 1.0
THUMB = (64, 48)
SAME_PATCH_SHARE = 0.01     # of its pixels may differ by more than 30 grey levels
# ... and the check's own region in it, unshrunk, nearly pixel for pixel: a
# board with only this part missing is the same picture shrunk to 64 x 48
# (0.06 grey levels), but 5-40 % of the region's pixels differ by more than
# 30 levels; a still re-shot or a 0.4 px shift of the same part, under 0.1 %.


def thumb(image):
    return cv2.resize(_gray(image), THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)


def _patch_sim(a, b):
    """Normalised correlation of two same-sized grey patches (1 = identical)."""
    if a.shape != b.shape or a.size == 0:
        return 0.0
    if float(a.std()) < 1e-6 or float(b.std()) < 1e-6:
        d = float(np.sqrt(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)))
        return max(0.0, 1.0 - d / 64.0)
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])


def _best_sim(win, patch):
    """(similarity, region): where in the search window the patch fits best,
    and how well -- normalised correlation, or for a plain patch the RMS
    difference as Part present scores a plain region."""
    ph, pw = patch.shape[:2]
    if win.shape[0] < ph or win.shape[1] < pw or patch.size == 0:
        return 0.0, None
    if float(patch.std()) < 4.0:
        r = cv2.matchTemplate(win, patch, cv2.TM_SQDIFF)
        mn, _, loc, _ = cv2.minMaxLoc(r)
        sim = max(0.0, 1.0 - float(np.sqrt(max(0.0, mn) / float(pw * ph))) / 64.0)
    else:
        r = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
        _, sim, _, loc = cv2.minMaxLoc(r)
    return float(sim), win[loc[1]:loc[1] + ph, loc[0]:loc[0] + pw]


def training_examples(folder, tool_id, roi, tpl):
    """[(label, path, patch)] -- each OK / NG training picture's patch for
    this region, found where the region's reference content is in it (a
    picture of a board lying a little differently still lines up)."""
    import glob
    import os
    if not folder:
        return []
    base = os.path.join(folder, "training", tool_id)
    files = []
    for label in ("ok", "ng"):
        for f in sorted(glob.glob(os.path.join(base, label, "*"))):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                try:
                    files.append((label, f, os.path.getmtime(f)))
                except OSError:
                    pass
    if not files:
        return []
    key = (base, tuple(roi), tpl.shape, tuple(files))
    hit = _EXAMPLES.get(key)
    if hit is not None:
        return hit
    x, y, w, h = roi
    th, tw = tpl.shape[:2]
    margin = int(max(30, 0.5 * max(w, h)))
    out = []
    for label, f, _m in files:
        img, th_img = _picture(f)
        if img is None:
            continue
        H, W = img.shape[:2]
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(W, x + w + margin), min(H, y + h + margin)
        win = _gray(img[y0:y1, x0:x1])
        if win.shape[0] < th or win.shape[1] < tw:
            continue
        method = cv2.TM_CCOEFF_NORMED if float(tpl.std()) >= 1e-6 else cv2.TM_SQDIFF
        r = cv2.matchTemplate(win, tpl, method)
        mn, mx, mnl, mxl = cv2.minMaxLoc(r)
        lx, ly = mnl if method == cv2.TM_SQDIFF else mxl
        if method != cv2.TM_SQDIFF and mx < 0.5:
            # nothing like the reference's part here (it is missing, or a
            # variant): the example is what sits where the part belongs
            lx, ly = x - x0, y - y0
            lx = max(0, min(win.shape[1] - tw, lx))
            ly = max(0, min(win.shape[0] - th, ly))
        out.append((label, f, win[ly:ly + th, lx:lx + tw].copy(), th_img))
    if len(_EXAMPLES) > EXAMPLES_MAX:
        _EXAMPLES.clear()
    _EXAMPLES[key] = out
    return out


@register("presence")
class Presence(Tool):
    """Is the thing there at all? Correlation against the same ROI in the
    reference -- and against the check's own OK and NG examples, when its
    Train tab has any: a part that looks more like an NG example than like
    any good one is rejected, one that matches an OK example is accepted."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        if ctx.reference is None or self.roi is None:
            return ToolResult(self.id, self.type_name, "error", message="roi and reference required")
        x, y, w, h = self.roi
        # the region alone made grey: the whole 20-megapixel reference was
        # converted for every check, 2-3 ms each, every cycle
        tpl = _gray(ctx.reference[max(0, y):y + h, max(0, x):x + w])
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
        # The examples from the Train tab. They changed nothing before: a
        # picture added under NG -- even the reference itself -- still
        # passed, and "Test" reported it as an escape.
        ok_best, ng_best, ng_path = float(mx), 0.0, ""
        examples = training_examples(getattr(ctx, "folder", ""), self.id, [x, y, w, h], tpl)
        same_ng = ""
        if any(e[0] == "ng" for e in examples):
            share = getattr(ctx, "share", None)
            key = ("_thumb", id(image))
            mine = share.get(key) if isinstance(share, dict) else None
            if mine is None:
                mine = thumb(image)
                if isinstance(share, dict):
                    share[key] = mine
        # A part that already matches the reference, with no NG example to
        # weigh it against, passes whatever its good examples say: they are
        # not compared then. Advanced training gives every check 8 of them,
        # and comparing all of them on every run cost 0.6 s on 233 checks.
        has_ng = any(e[0] == "ng" for e in examples)
        need_ok = not (mx >= thr and not has_ng)
        for label, path, patch, th_ in examples:
            if label == "ok" and not need_ok:
                continue
            # each example looked for in the search window, as the reference
            # is: compared at the reference's best spot, a part unlike the
            # reference was compared with whatever happened to be there
            sim, region = _best_sim(cg, patch)
            if label == "ok":
                ok_best = max(ok_best, sim)
                continue
            if sim > ng_best:
                ng_best, ng_path = sim, path
            if region is not None and th_.shape == mine.shape and \
                    float(np.mean(np.abs(th_ - mine))) < SAME_PICTURE_GREY and \
                    float(np.mean(np.abs(region.astype(np.int16) - patch.astype(np.int16))
                                  > 30)) < SAME_PATCH_SHARE:
                same_ng = path
        why = "" if ok else f"match {mx:.2f} below {thr}"
        if examples:
            ok = ok_best >= thr
            why = "" if ok else f"match {ok_best:.2f} below {thr}"
            # An NG example wins over the good ones when the part is closer to
            # it -- or is, in effect, that very picture: labelled NG, it is NG.
            import os
            if same_ng:
                ok = False
                why = "this is the picture of NG example %s" % \
                    os.path.splitext(os.path.basename(same_ng))[0]
            elif ng_best > ok_best + NG_MARGIN:
                ok = False
                why = "looks like NG example %s (%.2f) more than a good part (%.2f)" % (
                    os.path.splitext(os.path.basename(ng_path))[0], ng_best, ok_best)
            # the score: how good the part looks -- pulled down by an NG
            # example it resembles, so "Tune threshold" counts it as bad
            mx = min(ok_best, max(0.0, 1.0 - ng_best)) \
                if (why.startswith("looks like") or same_ng) else ok_best
        # The check's own outline -- box, rotated box, circle or polygon, as it
        # was drawn -- moved to where the part was found. It was a plain box
        # at the match, whatever shape the region had.
        status = "pass" if ok else "fail"
        g = []
        try:
            ex, ey = self.mapped_roi(ctx)[:2]
            dx, dy = loc[0] + ox - ex, loc[1] + oy - ey
            g = self.roi_graphic(ctx, status)
            for gr in g:
                gr.points = [(px + dx, py + dy) for px, py in gr.points]
        except Exception:
            g = []
        if not g:
            g = [Graphic("rect", [(loc[0] + ox, loc[1] + oy), (loc[0] + ox + w, loc[1] + oy + h)],
                         status, "")]
            g[0].role = "region"
        vals = {"match": round(float(mx), 4), "mode": mode, "texture": round(texture, 1)}
        if examples:
            vals.update(good_match=round(ok_best, 4), ng_match=round(ng_best, 4),
                        examples_ok=sum(1 for e in examples if e[0] == "ok"),
                        examples_ng=sum(1 for e in examples if e[0] == "ng"))
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail", float(mx),
                          vals, g, why, (time.perf_counter() - t0) * 1000)
