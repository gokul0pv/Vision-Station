"""Diagnostics: is the image good enough, and is the recipe robust?

Two separate questions, both worth asking before a recipe goes near a line.

**Image quality.** Most "it worked yesterday" failures are not the recipe. They
are focus drifting as a lens creeps, exposure clipping after someone changed the
lighting, or motion blur once the conveyor sped up. These are cheap to measure
and almost never measured.

**Recipe margin.** A tool that passes at 0.71 against a 0.70 threshold is not
passing, it is about to start failing intermittently at 3am. Margin analysis
finds those before they find you. This is the check that most often changes what
someone does next, and it costs nothing to run.
"""

import cv2
import numpy as np

# Where each tool type keeps its accept threshold, and which way is "good".
# higher_is_better: the score must stay ABOVE the threshold.
THRESHOLDS = {
    "align":          ("min_score", True),
    "presence":       ("min_score", True),
    "count_template": ("min_score", True),
    "ocr":            ("min_confidence", True),
    "ocv":            ("min_confidence", True),
    "yolo_detect":    ("confidence", True),
    "yolo_defect":    ("confidence", True),
    "anomaly":        ("threshold", False),
    "color_classify": ("deltaE_max", False),
}

MARGIN_THIN = 0.10      # fraction of the threshold; below this is fragile


def focus_score(gray):
    """Raw variance of the Laplacian, the usual sharpness proxy."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def focus_normalised(gray):
    """Laplacian variance divided by intensity variance.

    Raw Laplacian variance conflates sharpness with contrast: darken an image
    and the number collapses even though the lens never moved. That produces
    exactly the wrong advice - it sends someone to refocus when the real fault
    is lighting. Dividing by intensity variance cancels the contrast term, so
    this stays put when an image is merely dark and only falls when detail is
    genuinely smeared."""
    g = gray.astype(np.float64)
    var = g.var()
    if var < 1e-6:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var() / var)


def image_quality(img, roi=None):
    """Returns measurements plus plain-language findings. Never raises."""
    if img is None or img.size == 0:
        return {"ok": False, "findings": [("error", "No image to check.")], "metrics": {}}

    if roi:
        x, y, w, h = [int(v) for v in roi]
        x, y = max(0, x), max(0, y)
        sub = img[y:y + max(1, h), x:x + max(1, w)]
        if sub.size:
            img = sub

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = gray.astype(np.float32)
    total = g.size

    focus = focus_score(gray)
    focus_n = focus_normalised(gray)
    mean = float(g.mean())
    std = float(g.std())
    clipped_hi = float((gray >= 254).sum()) / total * 100.0
    clipped_lo = float((gray <= 1).sum()) / total * 100.0

    # Effective dynamic range: the span holding the middle 98% of pixels. More
    # honest than min/max, which a single hot pixel can dominate.
    lo, hi = np.percentile(gray, [1, 99])
    span = float(hi - lo)

    # Directional blur: a smeared image loses gradient energy along the
    # direction of travel, so a lopsided gradient ratio suggests motion.
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    ex, ey = float(np.abs(gx).mean()), float(np.abs(gy).mean())
    ratio = (max(ex, ey) / max(1e-6, min(ex, ey))) if min(ex, ey) > 0 else 1.0

    metrics = {"focus": round(focus, 1), "focus_normalised": round(focus_n, 4),
               "mean_brightness": round(mean, 1),
               "contrast_std": round(std, 1), "dynamic_range": round(span, 1),
               "clipped_highlights_pct": round(clipped_hi, 2),
               "clipped_shadows_pct": round(clipped_lo, 2),
               "gradient_anisotropy": round(ratio, 2)}

    f = []
    # Judge focus on the contrast-normalised figure, and only when there is
    # enough contrast to judge it at all.
    if span < 25:
        f.append(("warn", "Contrast is too low to assess focus. Fix the lighting "
                          "or exposure first, then re-check."))
    elif focus_n < 0.02:
        f.append(("bad", "Image is out of focus (%.3f). Nothing downstream will "
                         "be reliable until this is fixed." % focus_n))
    elif focus_n < 0.06:
        f.append(("warn", "Focus is soft (%.3f). Fine defects and small text "
                          "will be unreliable." % focus_n))

    if clipped_hi > 2.0:
        f.append(("bad", "%.1f%% of pixels are pure white. Detail is being "
                         "destroyed in the highlights - reduce exposure or "
                         "lighting." % clipped_hi))
    elif clipped_hi > 0.5:
        f.append(("warn", "%.1f%% of pixels are clipped white." % clipped_hi))

    if clipped_lo > 5.0:
        f.append(("warn", "%.1f%% of pixels are pure black. Anything in shadow "
                          "is invisible to the tools." % clipped_lo))

    if mean < 40:
        f.append(("warn", "Image is very dark (mean %.0f). Raise exposure or "
                          "add light rather than gain, which adds noise." % mean))
    elif mean > 215:
        f.append(("warn", "Image is very bright (mean %.0f)." % mean))

    if span < 40:
        f.append(("bad", "Very low contrast (range %.0f of 255). Thresholding "
                         "and edge finding will both be unstable." % span))
    elif span < 80:
        f.append(("warn", "Low contrast (range %.0f of 255)." % span))

    if ratio > 2.5:
        axis = "horizontally" if ex < ey else "vertically"
        f.append(("warn", "Detail is smeared %s (anisotropy %.1f). This usually "
                          "means motion blur - shorten exposure or trigger on a "
                          "stationary part." % (axis, ratio)))

    if not f:
        f.append(("good", "Image quality looks fine for inspection."))
    ok = not any(k == "bad" for k, _ in f)
    return {"ok": ok, "findings": f, "metrics": metrics}


def margin_report(result, spec):
    """How close was each tool to its own accept threshold?

    A pass at 0.71 against a 0.70 threshold is a failure that has not happened
    yet. Returns one row per tool that has a comparable threshold."""
    params = {t["id"]: (t.get("params") or {}) for t in spec.get("tools", [])}
    types = {t["id"]: t.get("type") for t in spec.get("tools", [])}
    rows = []
    for r in result.get("tools", []):
        tid = r["tool_id"]
        ttype = types.get(tid, r.get("type"))
        if ttype not in THRESHOLDS:
            continue
        key, higher_better = THRESHOLDS[ttype]
        thr = params.get(tid, {}).get(key)
        if thr in (None, ""):
            continue
        try:
            thr = float(thr)
        except (TypeError, ValueError):
            continue
        score = float(r.get("score", 0.0))
        if ttype == "color_classify":
            # score is a pass fraction, not a distance; not comparable here
            continue
        margin = (score - thr) if higher_better else (thr - score)
        rel = margin / thr if thr else 0.0
        if margin < 0:
            verdict = "failing"
        elif rel < MARGIN_THIN:
            verdict = "fragile"
        else:
            verdict = "healthy"
        rows.append({"tool_id": tid, "type": ttype, "score": round(score, 4),
                     "threshold": round(thr, 4), "margin": round(margin, 4),
                     "relative": round(rel, 3), "verdict": verdict,
                     "status": r.get("status")})
    rows.sort(key=lambda d: d["relative"])
    return rows


def timing_report(result):
    """Where the cycle time actually goes. Usually one tool dominates and it is
    rarely the one people expect."""
    tools = result.get("tools", [])
    total = float(result.get("total_ms", 0.0)) or sum(t.get("ms", 0) for t in tools)
    rows = [{"tool_id": t["tool_id"], "ms": round(float(t.get("ms", 0)), 2),
             "pct": round(100.0 * float(t.get("ms", 0)) / total, 1) if total else 0.0}
            for t in tools]
    rows.sort(key=lambda d: -d["ms"])
    return {"total_ms": round(total, 2), "rows": rows}


def stability_report(results):
    """Run the same recipe over several images of GOOD parts and this shows
    which checks wander. A tool whose score swings widely across parts that are
    all acceptable will eventually swing past its threshold."""
    if not results:
        return []
    by_tool = {}
    for res in results:
        for t in res.get("tools", []):
            by_tool.setdefault(t["tool_id"], []).append(float(t.get("score", 0.0)))
    rows = []
    for tid, scores in by_tool.items():
        if len(scores) < 2:
            continue
        arr = np.array(scores, dtype=np.float64)
        rows.append({"tool_id": tid, "n": len(scores),
                     "mean": round(float(arr.mean()), 4),
                     "min": round(float(arr.min()), 4),
                     "max": round(float(arr.max()), 4),
                     "spread": round(float(arr.max() - arr.min()), 4),
                     "std": round(float(arr.std()), 4)})
    rows.sort(key=lambda d: -d["spread"])
    return rows
