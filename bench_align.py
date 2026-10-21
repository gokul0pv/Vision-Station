"""Measure alignment accuracy against known transforms.

Synthesises a board, applies a transform we know exactly, then asks align to
recover it. Error is reported where it matters: how far a downstream ROI corner
would land from where it should. That is the number that becomes a false reject,
not the raw matrix difference.
"""
import sys, os, time
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Context
from vstools.align import Align


def board(w=640, h=440):
    img = np.full((h, w, 3), 35, np.uint8)
    cv2.rectangle(img, (30, 30), (w - 30, h - 30), (150, 142, 132), -1)
    # fiducials and features with real structure
    for cx, cy in ((70, 70), (w - 70, 70), (70, h - 70), (w - 70, h - 70)):
        cv2.circle(img, (cx, cy), 16, (205, 205, 210), -1)
        cv2.circle(img, (cx, cy), 16, (70, 70, 75), 2)
        cv2.line(img, (cx - 11, cy), (cx + 11, cy), (70, 70, 75), 3)
    for i in range(6):
        cv2.rectangle(img, (150 + i * 60, 150), (150 + i * 60 + 38, 205),
                      (70, 80, 95), -1)
    cv2.putText(img, "AB12345X", (170, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (245, 245, 245), 3)
    cv2.circle(img, (500, 320), 26, (60, 170, 60), -1)
    return img


def apply_pose(img, dx, dy, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, np.vstack([M, [0, 0, 1]]).astype(np.float64)


def corner_error(M_true, M_est, w, h):
    """Worst displacement of the four ROI corners, in pixels."""
    if M_est is None:
        return float("inf")
    pts = np.float32([[80, 80], [w - 80, 80], [w - 80, h - 80], [80, h - 80]])
    p = pts.reshape(-1, 1, 2)
    a = cv2.transform(p, M_true[:2].astype(np.float32)).reshape(-1, 2)
    b = cv2.transform(p, np.asarray(M_est)[:2].astype(np.float32)).reshape(-1, 2)
    return float(np.max(np.linalg.norm(a - b, axis=1)))


CASES = [
    ("still",            0,    0,   0.0),
    ("shift small",      6,   -4,   0.0),
    ("shift large",     26,   18,   0.0),
    ("rotate 1.5",       0,    0,   1.5),
    ("rotate 4",         4,   -3,   4.0),
    ("rotate 9",       -12,    7,   9.0),
    ("rotate -6",       15,   10,  -6.0),
    ("rot 3 + shift",   22,  -16,   3.2),
]


def run(params, label, noise=0):
    ref = board()
    h, w = ref.shape[:2]
    print("\n%s" % label)
    print("  %-14s %9s %9s %9s   %s" % ("case", "corner_px", "score", "ms", "verdict"))
    worst = 0.0
    times = []
    for name, dx, dy, deg in CASES:
        cur, M_true = apply_pose(ref, dx, dy, deg)
        if noise:
            cur = np.clip(cur.astype(np.int16) +
                          np.random.RandomState(0).normal(0, noise, cur.shape),
                          0, 255).astype(np.uint8)
        ctx = Context(reference=ref)
        tool = Align("a", dict(params, roi=[60, 60, w - 120, h - 120]))
        t0 = time.perf_counter()
        res = tool.run(cur, ctx)
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)
        err = corner_error(M_true, ctx.fixture, w, h)
        worst = max(worst, err if np.isfinite(err) else 999)
        flag = "ok" if err < 2.0 else ("MARGINAL" if err < 6 else "BAD")
        print("  %-14s %9.2f %9.3f %9.1f   %s"
              % (name, err, res.score, ms, flag))
    print("  worst corner error: %.2f px   mean %0.0f ms" % (worst, np.mean(times)))
    return worst


if __name__ == "__main__":
    tpl = {"method": "template", "min_score": 0.3,
           "angle_range": [-10, 10], "angle_step": 1.0}
    run(tpl, "TEMPLATE, angle sweep -10..10 step 1")
    run({"method": "features", "min_score": 0.3, "n_features": 3000}, "FEATURES (ORB)")
