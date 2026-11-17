"""Exposure: measuring it, and finding the right one.

This exists because of a real photograph. A switch faceplate shot at an exposure
that made the panel visible had every LED core clipped at (253, 249, 255), and
`LED colour` duly reported all thirteen LEDs as white. No amount of tuning fixes
that, because the colour is not in the file.

The tempting response is to compute a correction from the histogram. It does not
work: once a region is clipped, the image carries no information about how far
past 255 the true values went. On that photograph the 90th percentile of the LED
band was already 255, so any percentile-based correction says "reduce by 6%" when
the truth is closer to a twentieth. The only honest way to find the exposure is
to change it and look again, which is what `find_exposure` does.
"""
import numpy as np

CLIP = 254          # a pixel at or above this has lost its value
TARGET_CLIPPED = 0.05     # percent of the region allowed to clip


def clipping(image, roi=None):
    """How much of a region has lost its colour to clipping."""
    img = image
    if roi:
        x, y, w, h = [int(v) for v in roi]
        x, y = max(0, x), max(0, y)
        sub = img[y:y + max(1, h), x:x + max(1, w)]
        if sub.size:
            img = sub
    if img.ndim == 3:
        m = img.max(axis=2)
    else:
        m = img
    total = float(m.size) or 1.0
    return {
        "clipped_pct": 100.0 * float((m >= CLIP).sum()) / total,
        "near_clipped_pct": 100.0 * float((m >= 245).sum()) / total,
        "peak": int(m.max()),
        "median": float(np.median(m)),
        "bright_pct": 100.0 * float((m >= 60).sum()) / total,
    }


def advise(image, roi=None, purpose="colour"):
    """Plain-language verdict on whether this exposure can support the job.

    `purpose` matters. Reading an LED's colour needs its core unclipped.
    Deciding merely whether an LED is lit tolerates a clipped core, since the
    answer only needs brightness."""
    c = clipping(image, roi)
    out = {"metrics": c, "ok": True, "findings": []}

    if purpose == "colour":
        if c["clipped_pct"] > 2.0:
            out["ok"] = False
            out["findings"].append(
                ("bad", "%.1f%% of this region is clipped. LED cores read as "
                        "white whatever colour they are, so colour cannot be "
                        "measured here. Reduce exposure until this is under "
                        "%.2f%%." % (c["clipped_pct"], TARGET_CLIPPED)))
        elif c["clipped_pct"] > TARGET_CLIPPED:
            out["findings"].append(
                ("warn", "%.2f%% clipped. Colour on the brightest features is "
                         "unreliable." % c["clipped_pct"]))
        elif c["peak"] < 90:
            out["findings"].append(
                ("warn", "Nothing in this region is brighter than %d, so colour "
                         "will be noisy. Increase exposure." % c["peak"]))
    else:
        if c["bright_pct"] < 0.05:
            out["findings"].append(
                ("warn", "Almost nothing here is lit. If LEDs should be on, "
                         "increase exposure."))

    if not out["findings"]:
        out["findings"].append(("good", "Exposure looks right for %s." % purpose))
    return out


def find_exposure(grab, set_exposure, roi=None, lo_ms=0.02, hi_ms=100.0,
                  target_pct=TARGET_CLIPPED, steps=12, settle=2, log=None):
    """Search for the longest exposure that keeps a region unclipped.

    `grab` returns a fresh frame, `set_exposure` takes milliseconds. Both are
    passed in so this can be driven by a real camera, a simulated one, or a test.

    Longest rather than shortest: an unnecessarily short exposure throws away
    signal and makes everything noisy. The goal is the brightest image that
    still has headroom, which is a bisection on a monotonic quantity -- clipping
    only ever increases with exposure.

    `settle` frames are discarded after each change because a real camera
    delivers one or two frames captured with the previous setting.
    """
    history = []

    def measure(ms):
        set_exposure(ms)
        frame = None
        for _ in range(max(1, settle)):
            frame = grab()
        if frame is None:
            return None, None
        c = clipping(frame, roi)
        history.append({"exposure_ms": round(ms, 4),
                        "clipped_pct": round(c["clipped_pct"], 4),
                        "peak": c["peak"]})
        if log:
            log("  %8.3f ms -> %6.3f%% clipped, peak %d"
                % (ms, c["clipped_pct"], c["peak"]))
        return c, frame

    c_lo, _ = measure(lo_ms)
    if c_lo is None:
        return {"ok": False, "reason": "no frames from the camera",
                "history": history}
    if c_lo["clipped_pct"] > target_pct:
        # Even the shortest exposure clips: the scene is too bright for the
        # sensor. Gain or aperture has to come down, and saying so is more use
        # than returning a number that cannot work.
        return {"ok": False, "exposure_ms": lo_ms,
                "reason": "clipping persists at the shortest exposure (%.3f ms). "
                          "Reduce gain, close the aperture, or add a filter."
                          % lo_ms,
                "history": history}

    c_hi, _ = measure(hi_ms)
    if c_hi is not None and c_hi["clipped_pct"] <= target_pct:
        return {"ok": True, "exposure_ms": hi_ms,
                "reason": "nothing clips even at the longest exposure tried",
                "clipped_pct": c_hi["clipped_pct"], "history": history}

    good, bad = lo_ms, hi_ms
    for _ in range(steps):
        mid = (good * bad) ** 0.5        # geometric: exposure spans decades
        c, _f = measure(mid)
        if c is None:
            break
        if c["clipped_pct"] <= target_pct:
            good = mid
        else:
            bad = mid
        if bad / good < 1.05:
            break

    final, _ = measure(good)
    return {"ok": True, "exposure_ms": round(good, 4),
            "clipped_pct": round(final["clipped_pct"], 4) if final else None,
            "peak": final["peak"] if final else None,
            "reason": "longest exposure keeping clipping under %.2f%%" % target_pct,
            "history": history}
