"""Plots for tool edit control records, drawn with OpenCV.

A caliper's projection profile and gradient with the edges it found, a
histogram with its band and mean. Drawn directly rather than with a plotting
library: one less dependency on the line PC, and a few milliseconds instead of
the hundreds a figure library takes to start.
"""
import cv2
import numpy as np

BG = (246, 246, 246)
GRID = (222, 222, 222)
AXIS = (150, 150, 150)
INK = (60, 60, 60)
PROFILE = (180, 110, 40)      # blue-ish (BGR)
GRADIENT = (40, 140, 230)     # orange
PASS = (74, 158, 30)
FAIL = (58, 52, 214)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _frame(width, height, title):
    img = np.full((height, width, 3), BG, np.uint8)
    left, right, top, bottom = 46, 12, 24 if title else 10, 24
    for k in range(1, 4):
        y = top + (height - top - bottom) * k // 4
        cv2.line(img, (left, y), (width - right, y), GRID, 1)
    cv2.rectangle(img, (left, top), (width - right, height - bottom), AXIS, 1)
    if title:
        cv2.putText(img, title, (left, 16), _FONT, 0.45, INK, 1, cv2.LINE_AA)
    return img, (left, top, width - right, height - bottom)


def _xmap(n, box):
    left, _t, right, _b = box
    span = max(1, n - 1)
    return lambda i: int(round(left + (right - left) * float(i) / span))


def _ymap(lo, hi, box):
    _l, top, _r, bottom = box
    rng = (hi - lo) or 1.0
    return lambda v: int(round(bottom - (bottom - top) * (float(v) - lo) / rng))


def profile_plot(profile, gradient=None, marks=(), spans=(), width=720, height=260,
                 title=""):
    """profile: 1-D brightness along the tool. gradient: its derivative,
    drawn on its own scale. marks: [(position, 'pass'|'fail', label)] vertical
    lines at edge positions (subpixel positions allowed). spans: [(a, b,
    status)] shaded between two positions -- an edge pair's width."""
    prof = np.asarray(profile, np.float32).ravel()
    img, box = _frame(width, height, title)
    if prof.size < 2:
        return img
    X = _xmap(prof.size, box)
    for a, b, st in spans:
        x0, x1 = sorted((X(a), X(b)))
        over = img.copy()
        cv2.rectangle(over, (x0, box[1] + 1), (x1, box[3] - 1),
                      PASS if st == "pass" else FAIL, -1)
        img = cv2.addWeighted(over, 0.12, img, 0.88, 0)
    Y = _ymap(float(prof.min()), float(prof.max()), box)
    pts = np.array([[X(i), Y(v)] for i, v in enumerate(prof)], np.int32)
    cv2.polylines(img, [pts], False, PROFILE, 2, cv2.LINE_AA)
    if gradient is not None:
        g = np.asarray(gradient, np.float32).ravel()
        m = float(np.abs(g).max()) or 1.0
        Yg = _ymap(-m, m, box)
        zero = Yg(0)
        cv2.line(img, (box[0], zero), (box[2], zero), AXIS, 1, cv2.LINE_4)
        gp = np.array([[X(i), Yg(v)] for i, v in enumerate(g)], np.int32)
        cv2.polylines(img, [gp], False, GRADIENT, 1, cv2.LINE_AA)
    for pos, st, label in marks:
        x = X(pos)
        cv2.line(img, (x, box[1]), (x, box[3]), PASS if st == "pass" else FAIL, 1)
        if label:
            cv2.putText(img, str(label), (x + 3, box[1] + 12), _FONT, 0.38,
                        PASS if st == "pass" else FAIL, 1, cv2.LINE_AA)
    cv2.putText(img, "brightness", (box[0] + 4, box[3] - 6), _FONT, 0.35, PROFILE, 1,
                cv2.LINE_AA)
    if gradient is not None:
        cv2.putText(img, "edge strength", (box[0] + 80, box[3] - 6), _FONT, 0.35,
                    GRADIENT, 1, cv2.LINE_AA)
    cv2.putText(img, "0", (box[0] - 10, height - 8), _FONT, 0.35, INK, 1, cv2.LINE_AA)
    cv2.putText(img, str(prof.size - 1), (box[2] - 24, height - 8), _FONT, 0.35, INK, 1,
                cv2.LINE_AA)
    return img


def histogram_plot(pixels, band=None, mean=None, status="pass", width=720, height=260,
                   title=""):
    """pixels: the grey values the tool measured. band: (lo, hi) grey levels,
    shaded. mean: a vertical line."""
    img, box = _frame(width, height, title)
    px = np.asarray(pixels).ravel()
    if px.size == 0:
        return img
    hist = np.bincount(np.clip(px.astype(np.int64), 0, 255), minlength=256).astype(np.float32)
    X = _xmap(256, box)
    Y = _ymap(0.0, float(hist.max()) or 1.0, box)
    if band is not None:
        over = img.copy()
        cv2.rectangle(over, (X(band[0]), box[1] + 1), (X(band[1]), box[3] - 1),
                      PASS if status == "pass" else FAIL, -1)
        img = cv2.addWeighted(over, 0.12, img, 0.88, 0)
    for g in range(256):
        if hist[g]:
            cv2.line(img, (X(g), box[3] - 1), (X(g), Y(hist[g])), PROFILE, 2)
    if mean is not None:
        x = X(mean)
        cv2.line(img, (x, box[1]), (x, box[3]), PASS if status == "pass" else FAIL, 1)
        cv2.putText(img, "mean %.1f" % mean, (x + 3, box[1] + 12), _FONT, 0.38,
                    PASS if status == "pass" else FAIL, 1, cv2.LINE_AA)
    for g in (0, 128, 255):
        cv2.putText(img, str(g), (X(g) - 8, height - 8), _FONT, 0.35, INK, 1, cv2.LINE_AA)
    return img
