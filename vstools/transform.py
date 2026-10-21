"""Tools that make a new image, from VisionPro's image tools.

Affine transform    the region resampled upright and at a chosen scale
                    (CogAffineTransformTool / CogCopyRegionTool)
Polar unwrap        a ring unrolled into a strip (CogPolarUnwrapTool)
Two-image           the image combined with the reference: difference,
                    subtraction, minimum, maximum (CogIPTwoImage tools)

The first two change the geometry: a later tool reading their image draws its
region in that image's coordinates. The engine therefore gives such a tool no
fixture (the transform already followed the part) and a reference transformed
the same way, so comparisons stay like with like -- see changes_geometry.
"""
import math
import time

import cv2
import numpy as np

from .base import Tool, ToolResult, register


def _out(tool, ctx, cur, ref, t0, vals, status="pass", why=""):
    ctx.images[tool.id] = cur
    if ref is not None:
        ctx.ref_images[tool.id] = ref
    tool.record(ctx, "LastRun.OutputImage", cur)
    vals = dict(vals, width=int(cur.shape[1]), height=int(cur.shape[0]))
    return ToolResult(tool.id, tool.type_name, status, 1.0 if status == "pass" else 0.0, vals,
                      tool.roi_graphic(ctx, status, tool.id), why,
                      (time.perf_counter() - t0) * 1000)


def _on_reference(tool, ctx, fn):
    """fn() evaluated on the reference image, unmoved: no fixture."""
    if ctx.reference is None:
        return None
    saved, ctx.fixture = ctx.fixture, None
    try:
        return fn(ctx.reference)
    finally:
        ctx.fixture = saved


@register("affine_transform")
class AffineTransform(Tool):
    """The region, however it is rotated and wherever the fixture moved it,
    resampled upright into an image of its own -- a label printed at an angle
    read as if straight, a small feature enlarged for a finer tool."""
    changes_geometry = True

    def run(self, image, ctx):
        t0 = time.perf_counter()
        k = float(self.p.get("scale", 1.0)) or 1.0

        def make(img):
            patch, _o = self.crop(img, ctx)
            if patch is None or patch.size == 0:
                return None
            if k != 1.0:
                patch = cv2.resize(patch, None, fx=k, fy=k,
                                   interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
            return patch
        cur = make(image)
        if cur is None:
            return ToolResult(self.id, self.type_name, "error", message="empty region")
        return _out(self, ctx, cur, _on_reference(self, ctx, make), t0, {"scale": k})


@register("polar_unwrap")
class PolarUnwrap(Tool):
    """A ring -- between an inner and the region's outer radius -- unrolled
    into a straight strip: text printed round a cap, a bearing race, a seal.
    Angle runs left to right from the start angle, radius top (outside) to
    bottom (inside). The ring turns with the part: the fixture's rotation is
    added to the start angle, so the strip always starts at the same place on
    the part."""
    changes_geometry = True

    def _unwrap(self, img, ctx):
        if not self.roi:
            return None
        x, y, w, h = [float(v) for v in self.roi]
        c = np.array([x + w / 2.0, y + h / 2.0, 1.0])
        start = float(self.p.get("start_angle", 0.0))
        M = getattr(ctx, "fixture", None)
        if M is not None:
            c = M @ c if M.shape[0] == 3 else np.append(M @ c, 1.0)
            start += math.degrees(math.atan2(M[1, 0], M[0, 0]))
        cx, cy = float(c[0]), float(c[1])
        r_out = min(w, h) / 2.0
        r_in = r_out * float(self.p.get("inner_radius", 0.5))
        rows = max(4, int(round(r_out - r_in)))
        width = int(self.p.get("width", 0)) or max(16, int(round(2 * math.pi * (r_in + r_out) / 2)))
        sweep = float(self.p.get("sweep", 360.0))
        sign = -1.0 if self.p.get("direction", "clockwise") == "counterclockwise" else 1.0
        ang = np.radians(start + sign * sweep * np.arange(width) / float(width))
        rad = r_out - (r_out - r_in) * np.arange(rows) / float(max(1, rows - 1))
        A, R = np.meshgrid(ang, rad)
        map_x = (cx + R * np.cos(A)).astype(np.float32)
        map_y = (cy + R * np.sin(A)).astype(np.float32)
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def run(self, image, ctx):
        t0 = time.perf_counter()
        cur = self._unwrap(image, ctx)
        if cur is None:
            return ToolResult(self.id, self.type_name, "error", message="no region")
        ref = _on_reference(self, ctx, lambda img: self._unwrap(img, ctx))
        return _out(self, ctx, cur, ref, t0, {"inner_radius": float(self.p.get("inner_radius", 0.5))})


@register("image_combine")
class ImageCombine(Tool):
    """The image combined with the reference, aligned through the fixture
    first: the absolute difference shows only what changed, and a later tool
    (a blob count, a histogram) judges it. Also plain subtraction either way,
    minimum and maximum. Same size and coordinates as the camera image."""
    OPS = {"difference": lambda a, b: cv2.absdiff(a, b),
           "subtract": lambda a, b: cv2.subtract(a, b),
           "subtract_reference_from": lambda a, b: cv2.subtract(b, a),
           "minimum": lambda a, b: cv2.min(a, b),
           "maximum": lambda a, b: cv2.max(a, b),
           "average": lambda a, b: cv2.addWeighted(a, 0.5, b, 0.5, 0)}

    def run(self, image, ctx):
        t0 = time.perf_counter()
        if ctx.reference is None:
            return ToolResult(self.id, self.type_name, "error", message="needs a reference image")
        ref = ctx.reference
        M = getattr(ctx, "fixture", None)
        if M is not None:                           # line the reference up with the part
            ref = cv2.warpAffine(ref, np.asarray(M, np.float64)[:2], (image.shape[1], image.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if ref.shape != image.shape:
            ref = cv2.resize(ref, (image.shape[1], image.shape[0]))
        op = self.p.get("operation", "difference")
        out = self.OPS.get(op, self.OPS["difference"])(image, ref)
        ctx.images[self.id] = out
        self.record(ctx, "LastRun.OutputImage", out)
        diff = float(cv2.absdiff(image, ref).mean())
        return ToolResult(self.id, self.type_name, "pass", 1.0,
                          {"operation": op, "mean_difference": round(diff, 2)}, [], "",
                          (time.perf_counter() - t0) * 1000)
