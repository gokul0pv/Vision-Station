"""A tool trained from example images rather than configured by hand.

Modelled on how VisionPro Deep Learning's Red Analyze is used: a region of
interest is drawn once and applied across a set of training images, the tool
learns the normal appearance of that region from OK images -- including its
tolerable variation -- and NG images are used to decide how much deviation
counts as a defect.

This is not a neural network. It is a per-pixel statistical model of the
region: the mean appearance and how much each pixel legitimately varies across
the OK set. A new part is scored by how far each pixel falls outside that
learned variation, and the verdict is the size of the largest region that does.
It trains instantly on a CPU from a handful of images, which is the point on a
line with no GPU; the ONNX anomaly tool remains for when a trained network is
worth the effort.

What the NG images are for
--------------------------
Picking a threshold by hand is where inspections go wrong: too tight and it
rejects good parts, too loose and it passes scratches. With OK and NG examples
the tool measures every one, and sets the threshold midway between the worst
OK and the best NG. It also reports the gap between them -- the margin. A small
or negative margin means the training set does not separate good from bad, and
the tool says so rather than quietly picking a number.

Few OK images
-------------
With one or two OK images there is no measured variation at all, and every
pixel would look equally certain. So when the set is small, each OK image is
augmented with sub-pixel shifts and lighting changes -- exactly the variation
the fixture and the lamp introduce on a real line. More real OK images always
beat augmentation, and the training report says how many were real.
"""
import os
import time

import cv2
import numpy as np

from .base import Graphic, Tool, ToolResult, register

MODEL_SIDE = 160          # the region is resampled so its long side is this


def _normalise(gray):
    """Zero mean, unit spread. Removes lighting gain and offset, which move
    whole-region brightness without being a defect."""
    g = gray.astype(np.float32)
    m, s = float(g.mean()), float(g.std())
    return (g - m) / (s if s > 1e-3 else 1.0)


MODEL_SIDE_MAX = 640      # ...up to this for large regions (one model pixel <= 2 image pixels)


def _model_size(roi_w, roi_h, cap=MODEL_SIDE_MAX):
    """The model's resolution. It was fixed at 160 px on the long side, so a
    1030 px region was shrunk 6.4 times and a 3 px scratch became half a pixel
    at a sixth of its contrast: on the real chassis a plainly visible scratch
    across a grey side panel scored a perfect 1.0. Now one model pixel covers
    at most two image pixels, between 160 and `cap` px on the long side.

    `cap` is the tool's Model resolution setting. 640 suits a region up to
    ~1300 px; a whole 3400 px cover panel on a 12-megapixel photograph needs
    1600 or more, or a hairline scratch is averaged away before it is seen."""
    long_side = max(roi_w, roi_h)
    side = int(min(int(cap or MODEL_SIDE_MAX), max(MODEL_SIDE, round(long_side / 2.0))))
    if roi_w >= roi_h:
        return side, max(8, int(round(side * roi_h / float(roi_w))))
    return max(8, int(round(side * roi_w / float(roi_h)))), side


def _prepare(patch, size, flatten=False):
    """The region as the model sees it. With `flatten`, slow changes of
    brightness across the region are removed before normalising: a lamp
    that lights one end of a panel more than the other on Tuesday than it
    did on Monday, or a reflection that moves with the part, is not a
    defect. A scratch, a mark or a missing label is a local change and
    survives. Measured on a chassis photographed in two sessions, the
    first session's good part scored 2253 against a threshold of 341 without
    it."""
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    if flatten:
        g = g.astype(np.float32)
        sigma = max(4.0, max(size) / 24.0)
        g = g - cv2.GaussianBlur(g, (0, 0), sigma)
    return _normalise(g)


def _augment(prepared, rng, n):
    """Plausible OK variation: what the fixture and the lamp leave behind.

    Sub-pixel shift and a little rotation, because the fixture corrects pose
    but resampling a rotated image still leaves residuals at every edge -- a
    good part rotated 3 degrees on a real chassis scored over threshold when
    only shifts were simulated. Plus mild lighting gain and gradient."""
    h, w = prepared.shape
    out = []
    for _ in range(n):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rng.uniform(-1.5, 1.5), 1.0)
        M[0, 2] += rng.uniform(-1.2, 1.2)
        M[1, 2] += rng.uniform(-1.2, 1.2)
        M = M.astype(np.float32)
        a = cv2.warpAffine(prepared, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        ramp = np.linspace(-1, 1, w, dtype=np.float32)[None, :] * rng.uniform(-0.15, 0.15)
        a = a * rng.uniform(0.9, 1.1) + ramp
        a += rng.normal(0, 0.04, a.shape).astype(np.float32)
        out.append(_normalise(a))
    return out


def deviation_score(prepared, mean, std, k):
    """(score, mask). Score is the area, in model pixels, of the largest
    connected region deviating more than k sigma from the learned appearance.
    Largest region, not total: scattered single-pixel noise is not a defect,
    one compact blob is."""
    dev = np.abs(prepared - mean) / std
    mask = (dev > k).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _l, stats, _c = cv2.connectedComponentsWithStats(mask)
    area = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    return area, mask


DEEP_MIN_OK = 4           # 'auto' trains the deep model from this many OK images


def uses_deep(params, ok_images=None):
    """Whether this tool trains the deep model: asked for, or left on 'auto'
    with the network present and enough good parts to learn from.

    The deep model sets its threshold from how good parts score against each
    other. With one or two there is nothing to compare: trained from a single
    photograph it rejected the next good part. Below DEEP_MIN_OK images
    'auto' trains the statistical model, which was built for that case."""
    from . import deep_surface
    want = str(params.get("method", "auto") or "auto")
    if want == "deep":
        if not deep_surface.available():
            raise RuntimeError("the deep model's network is not installed "
                               "(models/backbone/resnet18_features.onnx) - "
                               "set Model to statistical, or run scripts/make_backbone.py")
        return True
    if ok_images is not None and ok_images < DEEP_MIN_OK:
        return False
    return want == "auto" and deep_surface.available()


def train(tool, samples, fixture_fn, out_path, k=4.0, min_real_ok=8, seed=0):
    """Train `tool` (a Tool instance) from samples.

    samples     list of (image, "ok" | "ng")
    fixture_fn  image -> Context with the side's fixture applied, so every
                training crop is taken from the same place on the part
    Returns a report dict; writes the model to out_path (.npz).

    The tool's Model setting chooses what is trained: 'deep' is the
    pretrained-network model in deep_surface.py, 'statistical' the per-pixel
    model below, and 'auto' the deep one whenever its network is installed.
    """
    if uses_deep(tool.p, sum(1 for _i, label in samples if label == "ok")):
        from . import deep_surface
        return deep_surface.train(tool, samples, fixture_fn, out_path, seed=seed)
    rng = np.random.default_rng(seed)
    size = None
    flatten = bool(tool.p.get("flatten_lighting", False))
    ok_prep, ng_prep = [], []
    for img, label in samples:
        ctx = fixture_fn(img)
        patch, _off = tool.crop(img, ctx)
        if patch is None or patch.size == 0:
            continue
        if size is None:
            size = _model_size(patch.shape[1], patch.shape[0],
                               tool.p.get("resolution") or MODEL_SIDE_MAX)
        p = _prepare(patch, size, flatten)
        (ok_prep if label == "ok" else ng_prep).append(p)

    if not ok_prep:
        raise ValueError("training needs at least one OK image")
    real_ok = len(ok_prep)
    pool = list(ok_prep)
    if real_ok < min_real_ok:
        per = int(np.ceil((24 - real_ok) / float(real_ok)))
        for p in ok_prep:
            pool.extend(_augment(p, rng, per))

    stack = np.stack(pool)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    # A floor on the spread: a pixel that never varied in training would
    # otherwise treat the faintest noise as an infinite deviation. The spread
    # is also dilated a little, so an edge that wanders by a pixel between
    # parts is tolerated along its whole length, not just where it was seen.
    std = cv2.dilate(std, np.ones((3, 3), np.uint8))
    std = np.maximum(std, 0.12).astype(np.float32)

    ok_scores = [deviation_score(p, mean, std, k)[0] for p in ok_prep]
    ng_scores = [deviation_score(p, mean, std, k)[0] for p in ng_prep]
    worst_ok = max(ok_scores)
    # An NG image that scores no worse than a good part carries no information
    # about where the threshold should go: either it is not really defective,
    # or its defect is invisible at this lighting. Using it anyway collapsed the
    # threshold to one pixel above the best OK -- training a real chassis with
    # a pale scratch drawn on a pale surface did exactly that, and a threshold
    # that tight rejects good parts. Such images are set aside and named, so
    # the operator can check them, and the threshold comes from the rest.
    # A margin has to be worth something. An NG scoring 10 against a worst OK
    # of 6 "separates", and the threshold lands at 8 -- a handful of model
    # pixels, which the next good part under a slightly different lamp will
    # exceed. On a real chassis a side whose NG differed only at the edges
    # collapsed to a threshold of 6 this way. An NG counts as separated only
    # when it is clear of the worst OK by half again and by a dozen pixels.
    clear = max(worst_ok * 1.5, worst_ok + 12)
    unseparated = [i for i, v in enumerate(ng_scores) if v <= clear]
    separable = [v for v in ng_scores if v > clear]
    if separable:
        best_ng = min(separable)
        margin = best_ng - worst_ok
        threshold = (worst_ok + best_ng) / 2.0
    else:
        best_ng = min(ng_scores) if ng_scores else None
        margin = (best_ng - worst_ok) if ng_scores else None
        # No usable NG: allow a small blob beyond anything seen on a good part.
        threshold = max(worst_ok * 1.5, worst_ok + 12)

    np.savez_compressed(out_path, mean=mean, std=std,
                        size=np.array(size), k=np.array(k),
                        threshold=np.array(threshold), flatten=np.array(int(flatten)))
    return {"real_ok": real_ok, "ng": len(ng_prep), "augmented": len(pool) - real_ok,
            "ok_scores": ok_scores, "ng_scores": ng_scores,
            "worst_ok": worst_ok, "best_ng": best_ng, "margin": margin,
            "threshold": float(threshold),
            "unseparated_ng": unseparated,
            "separates": bool(separable) and not unseparated if ng_scores else True}


@register("trained_surface")
class TrainedSurface(Tool):
    """Learns a region's normal appearance from OK images; flags deviations."""

    _models = {}

    def _load(self, path):
        key = (os.path.abspath(path), os.path.getmtime(path))
        if key not in TrainedSurface._models:
            z = np.load(path)
            if "kind" in z.files and str(z["kind"]) == "deep":
                from . import deep_surface
                TrainedSurface._models[key] = deep_surface.load(z)
                return TrainedSurface._models[key]
            TrainedSurface._models[key] = {
                "mean": z["mean"], "std": z["std"],
                "size": tuple(int(v) for v in z["size"]),
                "k": float(z["k"]), "threshold": float(z["threshold"]),
                "flatten": bool(int(z["flatten"])) if "flatten" in z.files else False}
        return TrainedSurface._models[key]

    def model_path(self, folder):
        p = self.p.get("model") or ("models/%s.npz" % self.id)
        if os.path.isabs(p) or os.path.exists(p):
            return p                    # already resolved by the project loader
        return os.path.join(folder or "", p)

    def run(self, image, ctx):
        t0 = time.perf_counter()
        path = self.model_path(getattr(ctx, "folder", ""))
        if not os.path.exists(path):
            return ToolResult(self.id, self.type_name, "error", 0.0, {},
                              self.roi_graphic(ctx, "error"),
                              "not trained yet - add OK images and press Train",
                              (time.perf_counter() - t0) * 1000)
        m = self._load(path)
        patch, (ox, oy) = self.crop(image, ctx)
        if patch is not None and patch.size:
            # An overexposed region has lost its texture to pure white, and no
            # model can check what the camera did not record. Said as such --
            # on the real chassis a 25%-brighter photograph clipped 38% of a
            # side panel, and it was reported as a surface defect.
            clipped = float(((patch.max(axis=2) if patch.ndim == 3 else patch) >= 254).mean())
            limit = float(self.p.get("max_clipped", 0.15))
            if clipped > limit:
                return ToolResult(self.id, self.type_name, "error", 0.0,
                                  {"clipped_pct": round(100 * clipped, 1)},
                                  self.roi_graphic(ctx, "error"),
                                  "overexposed: %.0f%% of the region is clipped at white - "
                                  "reduce the exposure or lighting" % (100 * clipped),
                                  (time.perf_counter() - t0) * 1000)
        if patch is None or patch.size == 0:
            return ToolResult(self.id, self.type_name, "error", 0.0, {},
                              self.roi_graphic(ctx, "error"), "empty region",
                              (time.perf_counter() - t0) * 1000)
        if m.get("kind") == "deep":
            return self._run_deep(m, patch, ox, oy, ctx, t0)
        prep = _prepare(patch, m["size"], m["flatten"])   # as trained, whatever the setting now
        k = float(self.p.get("sensitivity_sigma") or m["k"])
        area, mask = deviation_score(prep, m["mean"], m["std"], k)
        override = float(self.p.get("max_defect_area") or 0)
        thr = override if override > 0 else m["threshold"]
        ok = area <= thr

        g = self.roi_graphic(ctx, "pass" if ok else "fail")
        if not ok:
            # outline the defect on the image, mapped back from model space
            sx = patch.shape[1] / float(m["size"][0])
            sy = patch.shape[0] / float(m["size"][1])
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
                x, y, w, h = cv2.boundingRect(c)
                g.append(Graphic("rect", [(ox + x * sx, oy + y * sy),
                                          (ox + (x + w) * sx, oy + (y + h) * sy)],
                                 "fail", "defect"))
        # score in 0..1: how far inside the threshold, for the margin report
        score = max(0.0, 1.0 - area / max(1.0, 2.0 * thr))
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail", score,
                          {"defect_area": area, "threshold": round(thr, 1)}, g,
                          "" if ok else "defect area %d over %.0f" % (area, thr),
                          (time.perf_counter() - t0) * 1000)

    def _run_deep(self, m, patch, ox, oy, ctx, t0):
        """The deep model's verdict: how far the region's features are from
        anything seen on a good part. 'Sensitivity' scales the trained
        threshold -- 0.8 rejects sooner, 1.2 later; 0 leaves it as trained."""
        from . import deep_surface
        try:
            score, thr, boxes = deep_surface.inspect(
                patch, m, float(self.p.get("deep_sensitivity") or 1.0))
        except Exception as e:
            return ToolResult(self.id, self.type_name, "error", 0.0, {},
                              self.roi_graphic(ctx, "error"),
                              "%s: %s" % (type(e).__name__, str(e)[:160]),
                              (time.perf_counter() - t0) * 1000)
        ok = score <= thr
        g = self.roi_graphic(ctx, "pass" if ok else "fail")
        for x, y, w, h in boxes:
            g.append(Graphic("rect", [(ox + x, oy + y), (ox + x + w, oy + y + h)],
                             "fail", "defect"))
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail",
                          max(0.0, 1.0 - score / max(1e-6, 2.0 * thr)),
                          {"anomaly_score": round(score, 3), "threshold": round(thr, 3),
                           "defect_area": round(score, 3), "model": "deep"}, g,
                          "" if ok else "surface differs from the good parts "
                          "(score %.2f over %.2f)" % (score, thr),
                          (time.perf_counter() - t0) * 1000)
