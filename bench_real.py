"""Real-image campaign: does the application hold up on real photographs?

    python bench_real.py                         the chassis job
    python bench_real.py projects/my_job         another job; photos from each
                                                 side's image source or reference

Two questions, per side of the job:

  False rejects -- a good part photographed as a line photographs it: shifted,
  turned, lit brighter or darker, slightly out of focus, noisy, compressed.
  Every one must PASS.

  Escapes -- the same photograph with a real defect planted: a label gone, a
  screw missing, a hole covered, a scratch or a stain on the surface. Every one
  must FAIL, and on the check that should catch it.

Defects are made by inpainting (a label or screw removed and the panel filled
in around it) or drawing (a scratch, a stain), in the part's own coordinates,
so they move with the part.
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ------------------------------------------------------------- line conditions
def _affine(img, dx=0, dy=0, rot=0.0, scale=1.0):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _noise(img, sigma, seed=0):
    n = np.random.default_rng(seed).normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


CONDITIONS = [
    ("as photographed", lambda im: im),
    ("shifted 40 right, 30 up", lambda im: _affine(im, 40, -30)),
    ("shifted 50 left, 35 down", lambda im: _affine(im, -50, 35)),
    ("turned 2 deg", lambda im: _affine(im, rot=2.0)),
    ("turned -2 deg", lambda im: _affine(im, rot=-2.0)),
    # Not "2% closer": one fixed camera cannot tell a part 2% closer from a part
    # 2% bigger -- either way its holes are 2% further apart in pixels -- so a
    # measurement that passed it would also pass an oversized part.
    ("25% darker", lambda im: cv2.convertScaleAbs(im, alpha=0.75)),
    ("25% brighter", lambda im: cv2.convertScaleAbs(im, alpha=1.25)),
    ("lower contrast", lambda im: cv2.convertScaleAbs(im, alpha=0.8, beta=25)),
    ("slightly out of focus", lambda im: cv2.GaussianBlur(im, (0, 0), 1.2)),
    ("sensor noise", lambda im: _noise(im, 5.0)),
    ("JPEG quality 50", lambda im: _jpeg(im, 50)),
    ("shifted, turned and darker", lambda im: cv2.convertScaleAbs(
        _affine(im, 30, 20, 1.5), alpha=0.85)),
]


# ------------------------------------------------------------- planted defects
def _remove(img, box):
    """A feature removed: the box inpainted from the panel around it."""
    x, y, w, h = [int(v) for v in box]
    m = np.zeros(img.shape[:2], np.uint8)
    m[max(0, y):y + h, max(0, x):x + w] = 255
    return cv2.inpaint(img, m, 7, cv2.INPAINT_TELEA)


def _scratch(img, box):
    x, y, w, h = [int(v) for v in box]
    out = img.copy()
    p0 = (x + w // 5, y + h // 3)
    p1 = (x + 4 * w // 5, y + 2 * h // 3)
    cv2.line(out, p0, p1, (35, 35, 38), 3, cv2.LINE_AA)
    return out


def _stain(img, box):
    """A stain darkens whatever it lands on -- by 40% here, feathered at the
    edge. A fixed stain colour was about the panel's own grey on the darker
    side panels, and so barely there at all: a test of nothing."""
    x, y, w, h = [int(v) for v in box]
    mask = np.zeros(img.shape[:2], np.float32)
    cv2.ellipse(mask, (x + w // 2, y + h // 2), (max(8, w // 10), max(6, h // 12)), 20, 0, 360,
                1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), 3)[..., None]
    return np.clip(img.astype(np.float32) * (1.0 - 0.4 * mask), 0, 255).astype(np.uint8)


def defects_for(spec, image, side):
    """(name, broken image, tools that must catch it) for one side's recipe,
    built from what the recipe actually checks."""
    tools = {t["id"]: t for t in spec.get("tools", [])}
    out = []
    for tid, t in tools.items():
        p = t["params"]
        roi = p.get("roi")
        if t["type"] == "trained_surface" or t["type"] == "defect_diff":
            out.append(("scratch on %s" % tid, _scratch(image, roi), {tid}))
            out.append(("stain on %s" % tid, _stain(image, roi), {tid}))
        elif t["type"] == "presence" and roi:
            out.append(("%s removed" % tid.replace("_present", ""), _remove(image, roi), {tid}))
        elif t["type"] == "find_circle" and roi:
            x, y, w, h = roi
            out.append(("%s covered" % tid, _remove(image, [x + w * .2, y + h * .2, w * .6, h * .6]),
                        {tid}))
        elif t["type"] == "count_template" and roi:
            res = side.inspect(image)
            r = next((q for q in res["tools"] if q["tool_id"] == tid), None)
            matches = ((r or {}).get("values") or {}).get("matches") or []
            if matches:
                m = matches[len(matches) // 2]
                out.append(("one screw missing (%s)" % tid,
                            _remove(image, [m["x"] - 4, m["y"] - 4, m["w"] + 8, m["h"] + 8]), {tid}))
    return out


# ------------------------------------------------------------- the campaign
def _photo(side):
    src = side.spec.get("image_source") or {}
    for p in src.get("paths") or []:
        if not os.path.isabs(p):
            p = os.path.join(os.path.dirname(os.path.dirname(side.folder)), p)
        img = cv2.imread(p)
        if img is not None:
            return img
    return side.reference


def run(job_folder, report=print):
    import job as jobmod
    j = jobmod.Job(job_folder)
    rows, t0 = [], time.time()
    for side in j.sides:
        img = _photo(side)
        for name, fn in CONDITIONS:
            res = side.inspect(fn(img))
            bad_tools = [t for t in res["tools"] if t["status"] != "pass"]
            failed = [t["tool_id"] for t in bad_tools]
            ok = res["verdict"] == "PASS"
            # a reject whose every cause is a named image fault -- overexposure --
            # is the software telling the truth about the image, not a false reject
            fault = (not ok) and all("overexposed" in (t.get("message") or "") for t in bad_tools)
            rows.append(("good", side.name, name, ok or fault, failed))
            report("  %-4s %-14s good  %-28s %s" % (
                "ok" if ok else "IMAGE FAULT (named)" if fault else "FALSE REJECT", side.name,
                name, "" if ok else ", ".join("%s: %s" % (t["tool_id"], t.get("message", "")[:50])
                                              for t in bad_tools)))
        for name, broken, must in defects_for(side.spec, img, side):
            for cond, fn in (("as photographed", lambda im: im),
                             ("shifted and darker", lambda im: cv2.convertScaleAbs(
                                 _affine(im, 30, -20, 1.0), alpha=0.85))):
                res = side.inspect(fn(broken))
                failed = {t["tool_id"] for t in res["tools"] if t["status"] != "pass"}
                caught = res["verdict"] != "PASS" and bool(failed & must)
                rows.append(("defect", side.name, "%s, %s" % (name, cond), caught, sorted(failed)))
                report("  %-4s %-14s bad   %-44s %s" % (
                    "ok" if caught else "ESCAPE", side.name, "%s, %s" % (name, cond),
                    ", ".join(sorted(failed)) or "(passed)"))
    good = [r for r in rows if r[0] == "good"]
    bad = [r for r in rows if r[0] == "defect"]
    fr = [r for r in good if not r[3]]
    esc = [r for r in bad if not r[3]]
    report("")
    report("good parts:   %d of %d passed   (%d false rejects)" % (len(good) - len(fr), len(good), len(fr)))
    report("defects:      %d of %d caught   (%d escapes)" % (len(bad) - len(esc), len(bad), len(esc)))
    report("time: %.0f s" % (time.time() - t0))
    return rows


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "projects", "chassis_job")
    if not os.path.isdir(folder):
        sys.exit("No job at %s. Give a job folder: python bench_real.py <job folder>\n"
                 "(the chassis job: import chassis_job/chassis_app.vsp in QuickBuild, "
                 "File > Import Package, then point this at projects/<its name>)" % folder)
    rows = run(folder)
    sys.exit(0 if all(r[3] for r in rows) else 1)
