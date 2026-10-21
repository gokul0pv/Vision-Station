"""Image processing, modelled on VisionPro's image-processing tools.

One operator per tool, and its output is an image that later tools can read by
choosing this tool's id under "Reads image" -- so preprocessing is part of the
recipe, visible and ordered, rather than hidden inside other tools. Chain two by
pointing the second at the first.

Every operator preserves geometry: the output is the same size as the input and
pixels stay where they were. That keeps the fixture and every region valid on
the processed image. Operators that move pixels -- a polar unwrap, say -- would
silently break every region drawn after them, so they are deliberately not here.

Operators
    smooth       Gaussian blur, for noise before edge tools
    median       removes salt-and-pepper noise and specks while keeping edges
    sharpen      unsharp mask, for soft optics
    equalize     local contrast (CLAHE), for uneven lighting
    threshold    fixed, automatic (Otsu) or adaptive black and white
    erode, dilate, open, close    morphology: shrink, grow, remove specks, fill gaps
    edges        gradient magnitude (Sobel)
    invert       dark becomes light
    channel      one colour channel as a grey image
    gain         multiply and offset brightness
"""
import time

import cv2
import numpy as np

from .base import Tool, ToolResult, register

OPERATORS = ["smooth", "median", "sharpen", "equalize", "threshold", "erode",
             "dilate", "open", "close", "edges", "invert", "channel", "gain"]


def _odd(k):
    k = max(1, int(k))
    return k if k % 2 else k + 1


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def apply(img, op, p):
    """Pure function: image in, image out, same size. Colour is kept where the
    operator makes sense on colour; otherwise the result is grey, returned as
    three identical channels so every downstream tool accepts it."""
    k = _odd(p.get("kernel", 5))
    if op == "smooth":
        out = cv2.GaussianBlur(img, (k, k), 0)
    elif op == "median":
        out = cv2.medianBlur(img, k)
    elif op == "sharpen":
        blur = cv2.GaussianBlur(img, (k, k), 0)
        amt = float(p.get("amount", 1.0))
        out = cv2.addWeighted(img, 1.0 + amt, blur, -amt, 0)
    elif op == "equalize":
        clahe = cv2.createCLAHE(clipLimit=float(p.get("amount", 2.0)),
                                tileGridSize=(8, 8))
        if img.ndim == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            out = clahe.apply(img)
    elif op == "threshold":
        g = _gray(img)
        mode = p.get("threshold_mode", "otsu")
        if mode == "fixed":
            _, out = cv2.threshold(g, int(p.get("level", 128)), 255, cv2.THRESH_BINARY)
        elif mode == "adaptive":
            out = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, _odd(max(3, k * 6)),
                                        -float(p.get("level", 0)) / 10.0)
        else:
            _, out = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif op in ("erode", "dilate", "open", "close"):
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        it = max(1, int(p.get("iterations", 1)))
        if op == "erode":
            out = cv2.erode(img, se, iterations=it)
        elif op == "dilate":
            out = cv2.dilate(img, se, iterations=it)
        else:
            out = cv2.morphologyEx(img, cv2.MORPH_OPEN if op == "open" else cv2.MORPH_CLOSE,
                                   se, iterations=it)
    elif op == "edges":
        g = _gray(img).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        out = np.clip(mag * (255.0 / max(1.0, float(np.percentile(mag, 99.5)))),
                      0, 255).astype(np.uint8)
    elif op == "invert":
        out = 255 - img
    elif op == "channel":
        if img.ndim == 2:
            out = img
        else:
            out = img[:, :, {"blue": 0, "green": 1, "red": 2}.get(p.get("channel", "green"), 1)]
    elif op == "gain":
        out = np.clip(img.astype(np.float32) * float(p.get("amount", 1.0))
                      + float(p.get("level", 0)), 0, 255).astype(np.uint8)
    else:
        raise ValueError("unknown operator %r" % op)
    if out.ndim == 2:
        out = cv2.cvtColor(np.ascontiguousarray(out), cv2.COLOR_GRAY2BGR)
    return out


@register("image_process")
class ImageProcess(Tool):
    """Produces a processed image; it has no pass or fail of its own."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        op = self.p.get("operator", "smooth")
        try:
            out = apply(image, op, self.p)
        except Exception as e:
            return ToolResult(self.id, self.type_name, "error", 0.0, {}, [],
                              "%s: %s" % (type(e).__name__, e),
                              (time.perf_counter() - t0) * 1000)
        ctx.images[self.id] = out
        return ToolResult(self.id, self.type_name, "pass", 1.0,
                          {"operator": op, "size": "%dx%d" % (out.shape[1], out.shape[0])},
                          [], "", (time.perf_counter() - t0) * 1000)
