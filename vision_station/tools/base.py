"""Tool contract. Every inspection tool returns the same result shape so the
runtime can aggregate a verdict and the UI can render a job it has never seen."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import cv2

REGISTRY: Dict[str, type] = {}


def register(name: str):
    def wrap(cls):
        cls.type_name = name
        REGISTRY[name] = cls
        return cls
    return wrap


@dataclass
class Graphic:
    """Overlay geometry, always in CURRENT image pixel coordinates."""
    kind: str                      # rect | poly | line | point | text
    points: List[Tuple[float, float]]
    status: str = "pass"           # drives colour in the UI
    label: str = ""
    absolute: bool = False         # True = already in image coords, never re-mapped
    role: str = ""                 # "region" for the tool's outline


@dataclass
class ToolResult:
    tool_id: str
    type_name: str
    status: str = "pass"           # pass | fail | warn | error
    score: float = 0.0             # 0..1 confidence / match quality
    values: Dict[str, Any] = field(default_factory=dict)
    graphics: List[Graphic] = field(default_factory=list)
    message: str = ""
    ms: float = 0.0
    caption: List[str] = field(default_factory=list)   # the result in words, for the picture

    def to_dict(self):
        return {
            "tool_id": self.tool_id, "type": self.type_name, "status": self.status,
            "score": round(self.score, 4), "values": self.values,
            "graphics": [{k: v for k, v in g.__dict__.items() if k != "absolute"}
                         for g in self.graphics],
            "message": self.message, "ms": round(self.ms, 2),
            "caption": list(self.caption),
        }


class Tool:
    """Subclass this. `roi` is stored in REFERENCE image coordinates and is
    mapped into the current image by the active fixture transform."""
    type_name = "base"

    def __init__(self, tool_id: str, params: Dict[str, Any]):
        self.id = tool_id
        self.p = params
        self.roi = params.get("roi")            # [x, y, w, h] in reference coords
        self.angle = float(params.get("angle", 0.0) or 0.0)   # degrees, about ROI centre
        self.shape = params.get("shape", "rect")   # "rect" | "ellipse" | "polygon"
        self.points = params.get("points") or []   # polygon vertices, reference coords
        self._crop_M = None                     # crop coords -> image coords, 2x3
        self._patch_origin = None               # where the last patch was taken
        self._crop_mask = None                  # uint8, 255 inside the region
        self.enabled = params.get("enabled", True)
        self.critical = params.get("critical", True)

    # ---- helpers -------------------------------------------------------
    def mapped_roi(self, ctx) -> Optional[Tuple[int, int, int, int]]:
        if self.shape == "polygon" and self.points:
            pts = self.polygon_points(ctx)
            if pts is not None:
                x0, y0 = pts[:, 0].min(), pts[:, 1].min()
                return (int(x0), int(y0),
                        int(pts[:, 0].max() - x0), int(pts[:, 1].max() - y0))
        if self.roi is None:
            return None
        x, y, w, h = self.roi
        corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
        M = ctx.fixture
        if M is not None:
            corners = cv2.transform(corners, M[:2])
        xs, ys = corners[:, 0, 0], corners[:, 0, 1]
        # rounded, not truncated: int() turned a fixture estimate of 1243.99
        # into 1243, a whole pixel off
        x0, y0 = int(round(float(xs.min()))), int(round(float(ys.min())))
        return (x0, y0, int(round(float(xs.max() - xs.min()))),
                int(round(float(ys.max() - ys.min()))))

    def mapped_quad(self, ctx):
        """The region's four corners in image coordinates, rotation included.
        Ordered top-left, top-right, bottom-right, bottom-left as seen by the
        tool once the patch is deskewed."""
        if self.roi is None:
            return None
        x, y, w, h = self.roi
        cx, cy = x + w / 2.0, y + h / 2.0
        a = math.radians(self.angle)
        ca, sa = math.cos(a), math.sin(a)
        pts = []
        for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
            pts.append([cx + dx * ca - dy * sa, cy + dx * sa + dy * ca])
        quad = np.float32(pts).reshape(-1, 1, 2)
        if ctx.fixture is not None:
            quad = cv2.transform(quad, ctx.fixture[:2])
        return quad.reshape(-1, 2)

    def record(self, ctx, name, image):
        """Publish a diagnostic image for the tool edit control, e.g.
        record(ctx, "LastRun.Projection", plot). Free when nobody asked."""
        if getattr(ctx, "want_records", False) and image is not None:
            ctx.records.setdefault(self.id, {})[name] = image

    def crop(self, image, ctx):
        """Returns (patch, (ox, oy)).

        For an unrotated region this is a plain slice and the offset is the
        top-left corner, which is what every tool already expects. For a rotated
        region the patch is deskewed so tools still see an upright image; the
        offset stays meaningful and the true mapping is kept in _crop_M so the
        engine can correct the graphics afterwards."""
        self._crop_M = None
        self._poly_mask = None
        if self.shape == "polygon":
            box, pts = self._polygon_bbox(ctx, image.shape[:2])
            if box is None:
                return image, (0, 0)
            x, y, w, h = box
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.int32(pts - [x, y])], 255)
            self._poly_mask = m
            return self._apply_shape(image[y:y + h, x:x + w]), (x, y)

        r = self.mapped_roi(ctx)
        if r is None:
            return image, (0, 0)

        # Derotate when the region OR the fixture carries rotation. This used to
        # test only the region's own angle, so an ordinary region under a
        # rotated fixture was cut as the axis-aligned bounding box of where it
        # landed: the tool saw the part still rotated, in an oversized box.
        # Every tool was affected whenever the part turned; it stayed hidden
        # because presence checks and template counting tolerate a few degrees,
        # and surfaced when a tool comparing pixel by pixel rejected a good part
        # rotated 3.7 degrees on a real chassis. Working in the part's own
        # coordinate space is the whole point of a fixture.
        fix_rot = 0.0
        if ctx.fixture is not None:
            fix_rot = math.degrees(math.atan2(float(ctx.fixture[1, 0]),
                                              float(ctx.fixture[0, 0])))
        # Any fixture moves the region by a fraction of a pixel -- on a line its
        # estimate is never exactly whole -- and a plain slice has to snap to
        # whole pixels. That snap jittered regions by up to a pixel from photo
        # to photo, and a Find circle starting its calipers from the region's
        # centre fitted a different edge: on the real chassis a 25%-brighter
        # photograph moved a hole 1.4 px and failed the hole spacing, with the
        # fixture itself exact to 0.01 px. Under a fixture the region is now
        # cut at its exact sub-pixel position, like a rotated one.
        shifted = False
        if ctx.fixture is not None and self.roi is not None:
            q = np.asarray(ctx.fixture, np.float64)
            tx, ty = q[0, 0] * self.roi[0] + q[0, 1] * self.roi[1] + q[0, 2], \
                q[1, 0] * self.roi[0] + q[1, 1] * self.roi[1] + q[1, 2]
            shifted = abs(tx - round(tx)) > 0.01 or abs(ty - round(ty)) > 0.01
        rotated = abs(self.angle) > 1e-6 or abs(fix_rot) > 0.05 or shifted
        if not rotated:
            x, y, w, h = r
            x, y = max(0, x), max(0, y)
            w = max(1, min(w, image.shape[1] - x))
            h = max(1, min(h, image.shape[0] - y))
            self._patch_origin = (x, y, w, h)
            return self._apply_shape(image[y:y + h, x:x + w]), (x, y)

        quad = self.mapped_quad(ctx)
        w = int(round(max(np.linalg.norm(quad[1] - quad[0]),
                          np.linalg.norm(quad[2] - quad[3]))))
        h = int(round(max(np.linalg.norm(quad[3] - quad[0]),
                          np.linalg.norm(quad[2] - quad[1]))))
        w, h = max(1, w), max(1, h)
        dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        M = cv2.getPerspectiveTransform(np.float32(quad), dst)
        patch = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)
        # inverse: crop coords -> image coords
        self._crop_M = cv2.getPerspectiveTransform(dst, np.float32(quad))
        ox, oy = int(round(quad[:, 0].min())), int(round(quad[:, 1].min()))
        return self._apply_shape(patch), (ox, oy)

    def points_bbox(self):
        p = np.float32(self.points)
        x, y = float(p[:, 0].min()), float(p[:, 1].min())
        return [int(x), int(y), int(p[:, 0].max() - x), int(p[:, 1].max() - y)]

    def polygon_points(self, ctx):
        """Polygon vertices in image coordinates, fixture applied.

        Stored absolutely rather than relative to a bounding box, because a
        polygon traced round an angled connector has no meaningful box to be
        relative to -- and re-deriving one every time the fixture moves would
        accumulate rounding error across a shift."""
        if not self.points or len(self.points) < 3:
            return None
        pts = np.float32(self.points).reshape(-1, 1, 2)
        if ctx.fixture is not None:
            pts = cv2.transform(pts, ctx.fixture[:2])
        return pts.reshape(-1, 2)

    def _polygon_bbox(self, ctx, shape_hw):
        pts = self.polygon_points(ctx)
        if pts is None:
            return None, None
        H, W = shape_hw
        x0 = int(max(0, np.floor(pts[:, 0].min())))
        y0 = int(max(0, np.floor(pts[:, 1].min())))
        x1 = int(min(W, np.ceil(pts[:, 0].max())))
        y1 = int(min(H, np.ceil(pts[:, 1].max())))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None, None
        return (x0, y0, x1 - x0, y1 - y0), pts

    def _apply_shape(self, patch):
        """For an ellipse region, neutralise everything outside the ellipse.

        The patch handed to a tool is always the bounding box, because every
        tool expects a plain rectangular array. Rather than blacking out the
        corners -- which would invent four hard edges and a large dark area that
        thresholding and defect differencing would both react to -- the outside
        is filled with the mean of the inside. That is visually neutral: it does
        not threshold as a blob, and in golden-image differencing the same fill
        appears on both sides and cancels.

        The exact mask is left on self._crop_mask so a tool that wants to be
        strict can use it directly."""
        self._crop_mask = None
        if patch is None or patch.size == 0:
            return patch
        if self.shape not in ("ellipse", "polygon"):
            return patch
        h, w = patch.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        if self.shape == "polygon":
            if len(self.points) < 3:
                return patch
            # Vertices are in reference coordinates; the patch is the bounding
            # box, so shift them into patch space. A polygon is the only way to
            # bound a part that sits at an angle without dragging in whatever
            # happens to be beside it.
            bx, by, _bw, _bh = self._patch_origin or self.points_bbox()[:2] + [0, 0]
            pts = np.int32([[int(px - bx), int(py - by)] for px, py in self.points])
            cv2.fillPoly(mask, [pts], 255)
        else:
            cv2.ellipse(mask, (w // 2, h // 2), (max(1, w // 2), max(1, h // 2)),
                        0, 0, 360, 255, -1)
        self._crop_mask = mask
        inside = mask > 0
        if not inside.any():
            return patch
        out = patch.copy()
        if patch.ndim == 3:
            fill = patch[inside].mean(axis=0)
            out[~inside] = fill.astype(patch.dtype)
        else:
            out[~inside] = patch[inside].mean().astype(patch.dtype)
        return out

    def correct_graphics(self, result, ctx):
        """Tools emit points as crop-coordinate + offset. That is exact for an
        axis-aligned region and wrong for a rotated one, so undo the offset and
        push the points through the real crop transform instead. Doing it here
        means no individual tool has to know about rotation."""
        if self._crop_M is None or not result.graphics:
            return result
        r = self.mapped_roi(ctx)
        if r is None:
            return result
        quad = self.mapped_quad(ctx)
        ox, oy = int(round(quad[:, 0].min())), int(round(quad[:, 1].min()))
        M = self._crop_M
        for g in result.graphics:
            if getattr(g, "absolute", False):
                continue
            pts = np.float32([[p[0] - ox, p[1] - oy] for p in g.points]).reshape(-1, 1, 2)
            out = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
            if g.kind == "rect" and len(out) == 2:
                # A rotated rectangle is no longer a rectangle; widen to a quad
                # so the overlay shows the true shape rather than a lying box.
                x0, y0 = g.points[0][0] - ox, g.points[0][1] - oy
                x1, y1 = g.points[1][0] - ox, g.points[1][1] - oy
                box = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
                g.kind = "poly"
                g.points = [tuple(map(float, p))
                            for p in cv2.perspectiveTransform(box, M).reshape(-1, 2)]
            else:
                g.points = [tuple(map(float, p)) for p in out]
        return result

    def roi_graphic(self, ctx, status="pass", label=""):
        out = self._roi_graphic(ctx, status, label)
        for g in out:
            g.role = "region"
        return out

    def _roi_graphic(self, ctx, status="pass", label=""):
        """The region outline, as the UI should draw it.

        Polygon first and unconditionally: a polygon carries its own vertices
        and may have no rectangle at all, so testing `roi is None` before it
        discarded the outline entirely. Two overlapping polygon branches also
        accumulated here from separate edits, one of which indexed
        `self.points` without a None check.
        """
        if self.shape == "polygon":
            pts = self.polygon_points(ctx)
            if pts is None:
                return []
            g = Graphic("poly", [tuple(map(float, p)) for p in pts], status,
                        label or self.id)
            g.absolute = True
            return [g]
        if self.roi is None:
            return []
        if self.shape == "ellipse":
            # Emitted as a polygon so it survives the fixture transform and
            # rotation without the renderer needing a separate ellipse path.
            x, y, w, h = self.roi
            cx, cy, rx, ry = x + w / 2.0, y + h / 2.0, w / 2.0, h / 2.0
            a = math.radians(self.angle)
            ca, sa = math.cos(a), math.sin(a)
            pts = []
            for i in range(48):
                th = 2.0 * math.pi * i / 48.0
                dx, dy = rx * math.cos(th), ry * math.sin(th)
                pts.append([cx + dx * ca - dy * sa, cy + dx * sa + dy * ca])
            arr = np.float32(pts).reshape(-1, 1, 2)
            if ctx.fixture is not None:
                arr = cv2.transform(arr, ctx.fixture[:2])
            g = Graphic("poly", [tuple(map(float, p)) for p in arr.reshape(-1, 2)],
                        status, label or self.id)
            g.absolute = True
            return [g]
        if abs(self.angle) > 1e-6:
            quad = self.mapped_quad(ctx)
            g = Graphic("poly", [tuple(map(float, p)) for p in quad], status,
                        label or self.id)
            g.absolute = True      # already in image coords; do not re-map
            return [g]
        x, y, w, h = self.mapped_roi(ctx)
        g = Graphic("rect", [(x, y), (x + w, y + h)], status, label or self.id)
        g.absolute = True
        return [g]

    def run(self, image: np.ndarray, ctx) -> ToolResult:
        raise NotImplementedError


def make_tool(spec: Dict[str, Any]) -> Tool:
    t = spec.get("type")
    if t not in REGISTRY:
        raise KeyError(f"unknown tool type '{t}'. known: {sorted(REGISTRY)}")
    return REGISTRY[t](spec["id"], spec.get("params", {}))


def nms(boxes, scores, iou_thr=0.3):
    """Non-maximum suppression over [x, y, w, h] boxes: indices kept, best
    first. One implementation for every tool that finds overlapping
    candidates -- it had been written out separately for template counting and
    for YOLO."""
    if len(boxes) == 0:
        return []
    b = np.asarray(boxes, np.float32)
    s = np.asarray(scores, np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    area = (x2 - x1) * (y2 - y1)
    order, keep = s.argsort()[::-1], []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def runtime():
    """The inference module (OpenVINO / onnxruntime), for any tool."""
    from vision_station.core import inference
    return inference


def abs_graphic(kind, pts, status, label=""):
    """A graphic already in image coordinates -- never re-mapped through a
    crop or fixture. One definition for every tool; it had three copies."""
    g = Graphic(kind, [tuple(map(float, p)) for p in pts], status, label)
    g.absolute = True
    return g


def passed_feature(ctx, tool_id):
    """Another tool's published feature -- a fitted circle, line or point --
    but only if that tool passed. A failed fit still publishes its best guess,
    and a fixture, distance or intersection built on it is simply wrong: a
    fixture from two holes, one of them missed, set a pose 8 px out with a
    2.6% scale error. A tool with no recorded result (used on its own, outside
    a job) is taken as it is."""
    r = getattr(ctx, "results", {}).get(tool_id)
    if r is not None and r.status != "pass":
        return None
    return ctx.share.get(tool_id)
