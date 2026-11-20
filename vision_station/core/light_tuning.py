"""The light level that gives the clearest teach picture, chosen by trying
levels and judging each picture -- by measurement and by the application's
own AI.

Why both: measurement alone always says "brighter" -- detail rises with
the light until something burns out -- while the part-finding network can
find fewer components in a bright picture, the glare washing them into the
board, and the text reader has its own best level. So each level is scored
on all three, each measure smoothed along the levels (one picture's luck must
not pick the level), and of the levels that score within a few per cent of
the best, the brightest is taken -- more light, less sensor noise -- but
below full power where that is as good, so the light can still be raised
later as the lamp ages (Keep adjusted).

Burnt-out white is any colour channel at white, and detail is measured on
the strongest colour channel: one saturated colour reads only part-way to
white in grey.

AI here means the application's own networks: the YOLOE part finder
(core/part_finder.py) and the PP-OCR text reader (tools/ocr_engine.py). Each
picture is shrunk to AI_SIZE for them: about four seconds a level on the GPU.
"""
import time

import cv2
import numpy as np

LEVELS = (32, 64, 96, 128, 176, 224, 255)
AI_SIZE = 2400              # long side the AI looks at
TOLERANCE = 0.03            # scores this close to the best count as as good
HEADROOM = 230              # prefer levels at most this, to leave room to grow
CLIP_LIMIT = 0.005          # burnt-out white allowed before it costs
DARK_LIMIT = 0.02           # crushed black allowed before it costs
# Text first: on the real board the text reader read
# more and more up to level 176 (392 characters at 32, 743 at 176) and less
# above, as measured detail rose; the part finder reported the most parts in
# the darkest picture (181 at 32, 153 at 128) -- on the desk too -- so it
# has the least say.
WEIGHTS = {"parts": 0.2, "text": 0.5, "detail": 0.3}
SHINY = 0.10                # burnt-out white allowed before it costs: shiny metal


CHANNELS = "BGR"


def to_mono(img, channel):
    """The picture in black and white from one colour channel, kept as
    three channels so every check reads it as it reads any picture -- for a
    job set to work in black and white (its image source's "mono"). Never
    chosen automatically: a colour camera shows a white light as strongly
    green until its white is balanced, and an automatic choice from that
    took a white light for a green one."""
    if img is None or not channel or img.ndim != 3:
        return img
    g = img[:, :, CHANNELS.index(channel)]
    return cv2.merge([g, g, g])


def picture_stats(img):
    """Mean grey, burnt-out and crushed shares, and detail (mean edge
    strength), on a quarter-size copy. Burnt out and detail are taken on
    the strongest colour channel: one saturated colour reads only part-way
    to white in grey, and grey would hide both it and
    half of its detail."""
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) \
        if max(img.shape[:2]) > 1600 else img
    if small.ndim == 3:
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        lit = small[:, :, int(np.argmax(small.reshape(-1, 3).mean(0)))]
        hot = small.max(axis=2)
    else:
        g = lit = hot = small
    lit = lit.astype(np.float32)
    gx, gy = cv2.Sobel(lit, cv2.CV_32F, 1, 0), cv2.Sobel(lit, cv2.CV_32F, 0, 1)
    return {"mean": float(g.mean()) / 255.0, "clip": float((hot >= 250).mean()),
            "dark": float((lit <= 5).mean()),
            "detail": float(np.sqrt(gx * gx + gy * gy).mean())}


def ai_stats(img):
    """What the application's AI makes of a picture: components found times
    their mean confidence, and characters read with confidence 0.8 or more.
    None for an AI that is not installed or fails."""
    f = float(AI_SIZE) / max(img.shape[:2])
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
    out = {"parts": None, "parts_found": 0, "text": None}
    try:
        from vision_station.core import part_finder
        r = part_finder.find_parts(small, None, 0.5)
        sc = [p["score"] for p in r["parts"]]
        out["parts_found"] = len(sc)
        # every part weighted by how sure the AI is
        out["parts"] = float(np.sum(sc)) if sc else 0.0
    except Exception as e:
        out["parts_error"] = "%s: %s" % (type(e).__name__, e)
    try:
        from vision_station.tools import ocr_engine
        lines = ocr_engine.read(small)
        # characters weighted by how sure the reader is
        out["text"] = float(sum(len(l["text"]) * l["confidence"] for l in lines
                                if l["confidence"] >= 0.5))
    except Exception as e:
        out["text_error"] = "%s: %s" % (type(e).__name__, e)
    return out


def score(rows):
    """Each row's score 0..1: every measure as a share of the best of the
    tries, weighted, less what burnt-out white and crushed black cost. A
    measure the AI could not give (not installed, nothing found anywhere)
    is left out and the others carry its weight."""
    # Each measure smoothed along the levels first. Picture quality changes
    # gradually with the light, and one picture's luck must not pick the
    # level: on a test scene the AI's part count went 58, 29, 47, 26 over
    # four neighbouring levels.
    rows.sort(key=lambda r: r["level"])
    for key in WEIGHTS:
        raw = [r.get(key) for r in rows]
        for i, r in enumerate(rows):
            if raw[i] is None:
                r[key + "_s"] = None
                continue
            near = [(w, raw[j]) for w, j in ((0.25, i - 1), (0.5, i), (0.25, i + 1))
                    if 0 <= j < len(rows) and raw[j] is not None]
            r[key + "_s"] = sum(w * v for w, v in near) / sum(w for w, _v in near)

    def best(key):
        vals = [r[key + "_s"] for r in rows if r.get(key + "_s") is not None]
        return max(vals) if vals and max(vals) > 0 else None
    tops = {k: best(k) for k in WEIGHTS}
    used = {k: w for k, w in WEIGHTS.items() if tops[k] is not None}
    total = sum(used.values()) or 1.0
    # White that is there at every level -- a shiny metal cage reflecting the
    # light, 3.5 % of the real board even at level 32 -- is not the light's
    # fault; only what it adds above that costs.
    always = min(r["clip"] for r in rows)
    # shiny metal is white at any useful level: 5-8 % of the board from 64
    # up, and none at 32 -- allowing only the darkest picture's 0 % made the
    # darkest picture win
    allowed = max(CLIP_LIMIT, always * 1.5, SHINY)
    for r in rows:
        s = sum(w * (r[k + "_s"] or 0.0) / tops[k] for k, w in used.items()) / total
        s -= 4.0 * max(0.0, r["clip"] - allowed) + 2.0 * max(0.0, r["dark"] - DARK_LIMIT)
        r["score"] = round(s, 4)
    return rows


def choose(rows):
    """The level to use: the brightest within TOLERANCE of the best score,
    below HEADROOM when one is."""
    top = max(r["score"] for r in rows)
    good = [r for r in rows if r["score"] >= top * (1.0 - TOLERANCE)]
    roomy = [r for r in good if r["level"] <= HEADROOM]
    if not roomy:
        # only full power is best: a level a little behind it that leaves
        # room to raise the light later is worth more than the last few %
        roomy = [r for r in rows if r["level"] <= HEADROOM
                 and r["score"] >= top * (1.0 - 2 * TOLERANCE)]
    return max(roomy or good, key=lambda r: r["level"])


def find_best_level(set_level, grab, levels=LEVELS, use_ai=True, progress=None, stop=None):
    """Try each level, judge each picture, and leave the light at the best.
    set_level(v) changes the light; grab() returns a picture exposed after
    it. Returns {"level", "rows", "ai", "seconds"}; rows hold each try's
    measures and score. RuntimeError when no picture came."""
    t0 = time.perf_counter()
    rows = []
    for i, lv in enumerate(levels):
        if stop is not None and stop():
            break
        set_level(lv)
        img = grab()
        if img is None:
            raise RuntimeError("the camera gave no picture")
        row = dict(picture_stats(img), level=int(lv))
        if use_ai:
            row.update(ai_stats(img))
        else:
            row.update(parts=None, text=None, parts_found=0)
        rows.append(row)
        if progress is not None:
            progress(i + 1, len(levels), row)
    if not rows:
        raise RuntimeError("stopped before any picture was taken")
    score(rows)
    best = choose(rows)
    set_level(best["level"])
    ai = any(r.get("parts") for r in rows) or any(r.get("text") for r in rows)
    return {"level": best["level"], "rows": rows, "ai": bool(use_ai and ai),
            "best": best, "seconds": time.perf_counter() - t0}


def describe(result):
    """One sentence for the person: what was chosen and why."""
    b = result["best"]
    bits = ["%.0f %% grey" % (100 * b["mean"])]
    if result["ai"]:
        if b.get("parts") is not None:
            bits.append("%d components found by the AI" % b.get("parts_found", 0))
        if b.get("text") is not None:
            bits.append("%d characters read" % int(b["text"]))
    if b["clip"] > CLIP_LIMIT:
        bits.append("%.1f %% burnt out" % (100 * b["clip"]))
    return "Light level %d: the clearest picture of %d tried (%s), in %.0f s." % (
        result["level"], len(result["rows"]), ", ".join(bits), result["seconds"])
