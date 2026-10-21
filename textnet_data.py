"""Synthetic training data for the text detector.

Modelled on what the tool actually has to cope with on a chassis: printing on a
white label stuck to brushed metal, small and low contrast, frequently upside
down, and sitting next to barcodes.

Barcodes are the point. A gradient-based detector cannot tell a barcode from a
line of text -- both are dense clusters of short vertical edges -- and the
hand-written discriminator I measured on a real label had overlapping
distributions, so it could not be tuned reliably. A network can learn the
difference, provided the training data contains barcodes labelled as
not-text. Every negative here exists because something in the real photographs
was mistaken for text.
"""
import random

import cv2
import numpy as np

SIZE = 256
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
         cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX]
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-:+./"


def _rand_text(n=None):
    n = n or random.randint(4, 18)
    return "".join(random.choice(CHARS) for _ in range(n))


def brushed_metal(rng, size=SIZE):
    """Background like an anodised chassis: mid grey, directional grain."""
    base = rng.integers(95, 165)
    img = np.full((size, size), base, np.float32)
    grain = rng.normal(0, rng.uniform(2, 9), (size, size)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (1, 31), 0)          # horizontal brushing
    img += grain
    # broad lighting gradient
    gx, gy = rng.uniform(-30, 30), rng.uniform(-30, 30)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img += gx * xx + gy * yy
    return img


def draw_barcode(img, mask, x, y, w, h, rng):
    """A barcode, deliberately labelled as NOT text."""
    cv2.rectangle(img, (x, y), (x + w, y + h), 245, -1)
    cx = x + 2
    while cx < x + w - 2:
        bw = rng.integers(1, 5)
        if rng.random() < 0.55:
            cv2.rectangle(img, (int(cx), y), (int(cx + bw), y + h), 25, -1)
        cx += bw + rng.integers(1, 4)
    # mask deliberately left untouched: a barcode is not text


def draw_text(img, mask, x, y, text, scale, thick, rng, angle=0.0):
    canvas = np.zeros_like(img)
    m = np.zeros_like(mask)
    font = random.choice(FONTS)
    ink = float(rng.integers(10, 70))
    cv2.putText(canvas, text, (x, y), font, scale, ink, thick, cv2.LINE_AA)
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    # the target is the text line's box, slightly shrunk so lines stay separate
    pad_y = max(1, int(th * 0.12))
    cv2.rectangle(m, (x, y - th + pad_y), (x + tw, y + base - pad_y), 255, -1)
    if abs(angle) > 0.01:
        M = cv2.getRotationMatrix2D((x + tw / 2.0, y), angle, 1.0)
        canvas = cv2.warpAffine(canvas, M, (img.shape[1], img.shape[0]))
        m = cv2.warpAffine(m, M, (img.shape[1], img.shape[0]))
    drawn = canvas > 0
    img[drawn] = canvas[drawn]
    mask[m > 0] = 255


def make_sample(rng, size=SIZE):
    img = brushed_metal(rng, size)
    mask = np.zeros((size, size), np.uint8)

    n_labels = rng.integers(1, 3)
    for _ in range(n_labels):
        lw = int(rng.integers(size // 2, size - 12))
        lh = int(rng.integers(size // 5, size // 2))
        lx = int(rng.integers(4, max(5, size - lw - 4)))
        ly = int(rng.integers(4, max(5, size - lh - 4)))
        if rng.random() < 0.85:                      # a white label to print on
            cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh),
                          float(rng.integers(205, 250)), -1)

        rows = rng.integers(1, 5)
        yy = ly + int(lh / (rows + 1))
        for _r in range(rows):
            if yy > ly + lh - 6:
                break
            scale = rng.uniform(0.28, 0.62)
            thick = 1 if scale < 0.45 else int(rng.integers(1, 3))
            kind = rng.random()
            if kind < 0.28:
                bh = max(6, int(lh / (rows + 1) * 0.7))
                draw_barcode(img, mask, lx + 6, yy - bh,
                             max(24, int(lw * rng.uniform(0.35, 0.85))), bh, rng)
            else:
                draw_text(img, mask, lx + int(rng.integers(3, 14)), yy,
                          _rand_text(), scale, thick, rng,
                          angle=rng.uniform(-4, 4) if rng.random() < 0.3 else 0.0)
            yy += int(lh / (rows + 1))

    # distractors that a gradient detector mistakes for text
    for _ in range(int(rng.integers(0, 4))):
        k = rng.random()
        if k < 0.4:                                   # screw heads
            r = int(rng.integers(5, 16))
            cx, cy = int(rng.integers(r, size - r)), int(rng.integers(r, size - r))
            cv2.circle(img, (cx, cy), r, float(rng.integers(60, 120)), -1)
            cv2.circle(img, (cx, cy), r, float(rng.integers(30, 80)), 2)
        elif k < 0.7:                                 # panel seams
            y0 = int(rng.integers(0, size))
            cv2.line(img, (0, y0), (size, y0 + int(rng.integers(-6, 6))),
                     float(rng.integers(50, 110)), int(rng.integers(1, 4)))
        else:                                         # ventilation slots
            x0 = int(rng.integers(0, size - 40))
            y0 = int(rng.integers(0, size - 20))
            for k2 in range(int(rng.integers(3, 8))):
                cv2.rectangle(img, (x0 + k2 * 7, y0), (x0 + k2 * 7 + 3, y0 + 14),
                              float(rng.integers(40, 90)), -1)

    if rng.random() < 0.5:                            # the whole view rotated
        ang = rng.choice([90, 180, 270]) if rng.random() < 0.5 else rng.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((size / 2, size / 2), float(ang), 1.0)
        img = cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(mask, M, (size, size))

    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 1.4))
    img += rng.normal(0, rng.uniform(1, 7), img.shape).astype(np.float32)
    if rng.random() < 0.35:                           # glare
        gx, gy = rng.integers(0, size), rng.integers(0, size)
        yy, xx = np.mgrid[0:size, 0:size]
        g = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) /
                     (2 * (size * rng.uniform(0.12, 0.3)) ** 2)))
        img += g * rng.uniform(25, 90)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, (mask > 127).astype(np.float32)


def batch(n, rng, size=SIZE):
    xs = np.zeros((n, 1, size, size), np.float32)
    ys = np.zeros((n, 1, size, size), np.float32)
    for i in range(n):
        im, mk = make_sample(rng, size)
        xs[i, 0] = im.astype(np.float32) / 255.0
        ys[i, 0] = mk
    return xs, ys


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    tiles = []
    for _ in range(6):
        im, mk = make_sample(rng)
        vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        vis[mk > 0.5] = (0.4 * vis[mk > 0.5] +
                         0.6 * np.array([0, 255, 0])).astype(np.uint8)
        tiles.append(vis)
    top = np.hstack(tiles[:3])
    bot = np.hstack(tiles[3:])
    cv2.imwrite('/tmp/train_samples.png', np.vstack([top, bot]))
    print("wrote /tmp/train_samples.png")
