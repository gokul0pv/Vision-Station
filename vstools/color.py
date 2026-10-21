"""Colour work. Classify in CIELAB with a deltaE distance, not RGB thresholds —
RGB thresholds drift with every lamp-aging cycle and every camera swap, LAB
distance does not."""
import time
import cv2
import numpy as np
from .base import Tool, ToolResult, Graphic, register

# Reference LAB values (OpenCV 8-bit LAB scale: L 0-255, a/b offset by 128).
# Recalibrate these against your own lighting with a ColorChecker — these are
# only sane starting points.
DEFAULT_SWATCHES = {
    # Representative *observed* LED appearances, not pure primaries.
    #
    # Two earlier versions of this table were wrong in opposite directions. The
    # original was estimated by eye and sat 66-74 LAB units from any real green
    # or blue, so a perfect LED scored an enormous deltaE and failed the stock
    # threshold. Replacing it with fully saturated primaries was worse: a real
    # LED is never that saturated, so every colour then failed. These values are
    # the mean of a spread of plausible LED appearances, which is what the
    # camera actually records.
    "red":     (115, 190, 172), "green":   (181, 67,  183),
    "blue":    (94,  168, 56),  "amber":   (200, 145, 200),
    "yellow":  (218, 114, 204), "white":   (236, 128, 129),
    "cyan":    (210, 92,  118), "magenta": (150, 200, 90),
    "off":     (30,  129, 128),
}

# Lightness counts for less than hue when identifying an LED. Drive current,
# viewing angle and lens diffusion all move L a long way while a and b barely
# shift, so weighting L equally made brightness variation look like a colour
# change. Measured across a spread of realistic LED appearances, halving the
# L term takes the worst correct-answer distance from 28.4 to 20.4 against an
# acceptance of 28 -- the difference between marginal and comfortable. It is
# not dropped entirely because white and "off" differ only in lightness.
L_WEIGHT = 0.5


def _deltaE(a, b, l_weight=None):
    a = np.array(a, np.float32).copy(); b = np.array(b, np.float32).copy()
    a[1:] -= 128.0; b[1:] -= 128.0
    d = a - b
    d[0] *= L_WEIGHT if l_weight is None else float(l_weight)
    return float(np.linalg.norm(d))


def representative_lab(image, lab, b, clip=250, halo=1.8):
    """The colour of one LED, avoiding the blown-out core.

    A lit LED photographed at an exposure that shows anything else is
    almost always clipped at its centre: the core reads (253, 249, 255),
    which is white whatever colour the LED is. Taking the median of the
    blob therefore reports every lit LED as white. On a real switch faceplate
    this tool called all thirteen LEDs white, including obvious greens and
    reds.

    The colour survives in the halo around the core, so unclipped pixels are
    preferred: first inside the blob, and if the whole blob is clipped, from
    a ring around it. Falling back to the plain median keeps behaviour
    unchanged for LEDs that are not clipped at all.
    """
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    H, W = lab.shape[:2]
    sub_bgr = image[y:y + h, x:x + w]
    sub_lab = lab[y:y + h, x:x + w]
    if sub_lab.size == 0:
        return np.array([0, 128, 128], np.float32)

    keep = sub_bgr.max(axis=2) < clip
    if keep.sum() >= max(12, 0.05 * keep.size):
        return np.median(sub_lab[keep].reshape(-1, 3), axis=0)

    # Whole blob is clipped: read the ring just outside it.
    cx, cy = x + w / 2.0, y + h / 2.0
    rad = max(w, h) * halo / 2.0
    x0, x1 = int(max(0, cx - rad)), int(min(W, cx + rad))
    y0, y1 = int(max(0, cy - rad)), int(min(H, cy + rad))
    ring_bgr = image[y0:y1, x0:x1]
    ring_lab = lab[y0:y1, x0:x1]
    if ring_lab.size:
        m = ring_bgr.max(axis=2)
        sel = (m < clip) & (m > 40)      # lit halo, not the dark faceplate
        if sel.sum() >= 12:
            return np.median(ring_lab[sel].reshape(-1, 3), axis=0)
    return np.median(sub_lab.reshape(-1, 3), axis=0)


@register("color_classify")
class ColorClassify(Tool):
    """Reads blobs produced by an upstream count_blob tool (via `source`) and
    classifies each one, comparing to an expected sequence."""

    _representative = staticmethod(representative_lab)

    def run(self, image, ctx):
        t0 = time.perf_counter()
        blobs = ctx.share.get(self.p.get("source", ""))
        if blobs is None:
            return ToolResult(self.id, self.type_name, "error",
                              message=f"no blobs from source '{self.p.get('source')}' — run a count_blob first")

        swatches = {**DEFAULT_SWATCHES, **self.p.get("swatches", {})}
        expect = self.p.get("expect", [])
        max_de = float(self.p.get("deltaE_max", 28))
        lw = self.p.get("lightness_weight", None)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        halo = float(self.p.get("halo", 1.8))
        clip = int(self.p.get("clip_level", 250))

        found, g, fails = [], [], []
        for i, b in enumerate(blobs):
            patch = lab[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]]
            if patch.size == 0:
                continue
            med = self._representative(image, lab, b, clip, halo)
            name, de = min(((k, _deltaE(med, v, lw)) for k, v in swatches.items()),
                           key=lambda t: t[1])
            want = expect[i] if i < len(expect) else None
            ok = de <= max_de and (want is None or name == want)
            if not ok:
                fails.append(f"#{i+1} {name}" + (f" (want {want})" if want else "") + f" dE={de:.0f}")
            found.append({"index": i + 1, "color": name, "deltaE": round(de, 1),
                          "expected": want, "ok": ok})
            g.append(Graphic("rect", [(b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"])],
                             "pass" if ok else "fail", name))

        status = "fail" if fails else "pass"
        if expect and len(blobs) != len(expect):
            status, fails = "fail", fails + [f"{len(blobs)} LEDs vs {len(expect)} expected"]
        return ToolResult(self.id, self.type_name, status,
                          1.0 - min(1.0, len(fails) / max(1, len(found))),
                          {"leds": found}, g, "; ".join(fails),
                          (time.perf_counter() - t0) * 1000)


@register("led_glow")
class LedGlow(Tool):
    """Lit / not-lit per LED. Uses HSV value plus saturation so a blown-out white
    reflection off an unlit lens does not read as ON."""

    _representative = staticmethod(representative_lab)

    def run(self, image, ctx):
        t0 = time.perf_counter()
        blobs = ctx.share.get(self.p.get("source", ""))
        if blobs is None:
            crop, (ox, oy) = self.crop(image, ctx)
            blobs = [{"x": ox, "y": oy, "w": crop.shape[1], "h": crop.shape[0]}]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_on = float(self.p.get("v_on", 140))
        s_on = float(self.p.get("s_on", 60))
        expect = self.p.get("expect_state", [])   # e.g. [true, true, false, true]

        out, g, fails = [], [], []
        for i, b in enumerate(blobs):
            p = hsv[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]]
            if p.size == 0:
                continue
            v = float(np.percentile(p[:, :, 2], 90))
            s = float(np.percentile(p[:, :, 1], 60))
            lit = v >= v_on and s >= s_on
            want = expect[i] if i < len(expect) else True
            ok = lit == bool(want)
            if not ok:
                fails.append(f"#{i+1} {'on' if lit else 'off'}, want {'on' if want else 'off'}")
            out.append({"index": i + 1, "lit": lit, "v": round(v, 1), "s": round(s, 1), "ok": ok})
            g.append(Graphic("rect", [(b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"])],
                             "pass" if ok else "fail", "on" if lit else "off"))

        return ToolResult(self.id, self.type_name, "fail" if fails else "pass",
                          1.0 - min(1.0, len(fails) / max(1, len(out))),
                          {"leds": out, "lit_count": sum(1 for x in out if x["lit"])},
                          g, "; ".join(fails), (time.perf_counter() - t0) * 1000)
