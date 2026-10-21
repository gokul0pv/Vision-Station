"""OCR accuracy audit.

Renders known strings, reads them back, and reports character error rate. CER is
the right measure here rather than exact-match: on a lot code, one wrong
character out of nine is a very different situation from unreadable, and an
exact-match score hides which of the two you have.

Conditions are chosen to mirror what actually degrades marking on a line: small
text, low contrast on bare metal, glare, slight defocus, and dot-matrix printing
where characters are not solid strokes.
"""
import sys, os, time
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Context
from vstools import make_tool

FONT = cv2.FONT_HERSHEY_SIMPLEX
SAMPLES = ["AB12345X", "LOT7742", "SN0098213", "REV-C", "2026-09-05"]


def render(text, scale=1.4, thick=3, fg=245, bg=35, w=520, h=120,
           blur=0.0, noise=0.0, dotted=False, glare=0.0, rotate=0.0):
    img = np.full((h, w), bg, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    org = ((w - tw) // 2, (h + th) // 2)
    cv2.putText(img, text, org, FONT, scale, int(fg), thick, cv2.LINE_AA)
    if dotted:
        # punch a dot grid through the strokes, as an inkjet coder would print
        grid = np.zeros_like(img)
        grid[1::3, 1::3] = 255
        img = np.where(grid > 0, img, bg).astype(np.uint8)
    if rotate:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=int(bg))
    if glare:
        yy, xx = np.mgrid[0:h, 0:w]
        g = np.exp(-(((xx - w * 0.65) ** 2) / (2 * (w * 0.16) ** 2) +
                     ((yy - h * 0.4) ** 2) / (2 * (h * 0.5) ** 2)))
        img = np.clip(img.astype(np.float32) + glare * 255 * g, 0, 255).astype(np.uint8)
    if blur:
        k = int(blur) * 2 + 1
        img = cv2.GaussianBlur(img, (k, k), blur)
    if noise:
        img = np.clip(img.astype(np.float32) +
                      np.random.RandomState(0).normal(0, noise, img.shape),
                      0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def cer(truth, got):
    """Levenshtein distance normalised by length of the truth."""
    a, b = truth, got or ""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(a))


def read(img, extra=None):
    p = dict({"roi": [0, 0, img.shape[1], img.shape[0]], "upscale": 2.0,
              "clahe": False, "invert": False,
              "min_confidence": 0.0, "lang": "en", "close_dots": 5,
              "charset": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-", "enabled": True,
              "critical": True, "angle": 0.0, "shape": "rect"}, **(extra or {}))
    t = make_tool({"id": "o", "type": "ocr", "params": p})
    t0 = time.perf_counter()
    r = t.run(img, Context(reference=img))
    return r, (time.perf_counter() - t0) * 1000


CONDITIONS = [
    ("clean",              {}),
    ("small text",         {"scale": 0.8, "thick": 2}),
    ("low contrast",       {"fg": 130, "bg": 70}),
    ("defocus blur 2",     {"blur": 2}),
    ("noise sigma 10",     {"noise": 10}),
    ("glare",              {"glare": 0.55}),
    ("rotated 3 deg",      {"rotate": 3}),
    ("dot matrix",         {"dotted": True}),
]


def audit(extra=None, title="OCR"):
    print("\n%s" % title)
    print("  %-18s %-14s %-14s %7s %8s" % ("condition", "truth", "read", "CER", "ms"))
    print("  " + "-" * 68)
    tot_cer, tot_ms, n = 0.0, 0.0, 0
    per_cond = {}
    for cname, kw in CONDITIONS:
        cs, ms_sum = [], 0.0
        shown = None
        for s in SAMPLES:
            img = render(s, **kw)
            r, ms = read(img, extra)
            got = (r.values.get("text") or "").strip()
            c = cer(s, got)
            cs.append(c)
            ms_sum += ms
            if shown is None:
                shown = (s, got, c)
        avg = float(np.mean(cs))
        per_cond[cname] = avg
        tot_cer += avg
        tot_ms += ms_sum / len(SAMPLES)
        n += 1
        flag = "ok" if avg < 0.02 else ("marginal" if avg < 0.15 else "BAD")
        print("  %-18s %-14s %-14s %6.1f%% %8.0f  %s"
              % (cname, shown[0], shown[1][:14], 100 * avg,
                 ms_sum / len(SAMPLES), flag))
    print("  " + "-" * 68)
    print("  mean CER %.1f%%    mean %.0f ms per read" % (100 * tot_cer / n, tot_ms / n))
    return tot_cer / n, tot_ms / n, per_cond


if __name__ == "__main__":
    audit(None, "A. Current settings")
