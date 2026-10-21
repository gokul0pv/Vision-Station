"""Build the seven-side network-switch chassis job from the photographs shipped with it.

    python build_chassis_job.py

Recreates projects/chassis_job: one side per photograph, each with its own
fixture and a trained surface check. It is shipped as a script rather than as
the job folder itself because the job is around 200 MB of full-resolution
images, all reproducible from the seven photos already in
projects/chassis_example/images.

THE TRAINING SETS IT CREATES ARE DEMONSTRATION DATA. The OK images are real
photographs, but the NG images are those same photographs with a scratch drawn
on them, because no genuinely defective chassis was photographed. Replace them
with real rejects from the line before trusting any threshold: in the Training
panel, remove the demo images and add your own.
"""
import json
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from job import create_job        # noqa: E402
from schema import defaults       # noqa: E402

SRC = os.path.join(HERE, "projects", "chassis_example", "images")
JOB = os.path.join(HERE, "projects", "chassis_job")
SIDES = [("frt_top", "Front top", "FRT_TOP_Corner_09092026_163024.jpeg"),
         ("top_right", "Top right", "Top_Right_Corner_09092026_163024.jpeg"),
         ("btm_left", "Bottom left", "BTM_Left_Corner_09092026_163024.jpeg"),
         ("frt_btm", "Front bottom", "FRT_BTM_Corner_09092026_163024.jpeg"),
         ("btm_right", "Bottom right", "BTM_Right_Corner_09092026_163024.jpeg"),
         ("lh_btm", "Left side", "LH_BTM_Corner_09092026_163013.jpeg"),
         ("rh_top", "Right side", "RH_TOP_Corner_09092026_163024.jpeg")]


def tool(tid, ttype, **p):
    q = defaults(ttype)
    q.update(p)
    return {"id": tid, "type": ttype, "params": q}


def moved(im, dx, dy, deg=0.0):
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def demo_scratch(im, roi, seed):
    """A drawn scratch -- demonstration NG only, see the module docstring."""
    r = np.random.default_rng(seed)
    out = im.copy()
    x, y, w, h = roi
    x0, y0 = int(x + w * r.uniform(.2, .6)), int(y + h * r.uniform(.2, .7))
    cv2.line(out, (x0, y0), (x0 + int(w * .15), y0 + int(r.integers(-20, 20))),
             (215, 215, 215), 3, cv2.LINE_AA)
    return out


def main():
    missing = [f for _s, _n, f in SIDES if not os.path.exists(os.path.join(SRC, f))]
    if missing:
        print("missing photographs in %s:\n  %s" % (SRC, "\n  ".join(missing)))
        return 1
    shutil.rmtree(JOB, ignore_errors=True)
    job = create_job(JOB, "Network chassis - full")
    recipe = json.load(open(os.path.join(HERE, "projects", "chassis_example",
                                         "project.json")))["tools"]
    for sid, name, fn in SIDES:
        img = cv2.imread(os.path.join(SRC, fn))
        h, w = img.shape[:2]
        if sid == "frt_top":
            tools = [dict(t, id="fixture") if t["id"] == "locate_part" else t
                     for t in recipe]
            tools.append(tool("panel_surface", "trained_surface",
                              roi=[700, 550, 1000, 450]))
            # Two fastener holes and the spacing between them, measured through
            # the fixture. On the photographs the spacing held at 116.3-116.6 px
            # (std 0.1 px) across moves of 150 px, 4 degrees of rotation and a
            # lamp 10% brighter or dimmer; the band below allows for that.
            tools.append(tool("hole_a", "find_circle", roi=[1244, 756, 68, 68],
                              shape="ellipse"))
            tools.append(tool("hole_b", "find_circle", roi=[1254, 872, 68, 68],
                              shape="ellipse"))
            tools.append(tool("hole_spacing", "distance", first="hole_a",
                              second="hole_b", min_value=115.5, max_value=117.5))
        else:
            tools = [tool("fixture", "align",
                          roi=[int(w * .1), int(h * .12), int(w * .8), int(h * .76)],
                          method="features", min_score=0.25, n_features=4000),
                     tool("surface", "trained_surface",
                          roi=[int(w * .3), int(h * .35), int(w * .4), int(h * .3)])]
        side = job.add_side(sid, name, img, tools)
        tid = [t["id"] for t in tools if t["type"] == "trained_surface"][0]
        roi = [t for t in tools if t["id"] == tid][0]["params"]["roi"]
        for im, lab in ((img, "ok"), (moved(img, 30, -20, 1), "ok"),
                        (demo_scratch(img, roi, 1), "ng"),
                        (demo_scratch(moved(img, -25, 15), roi, 2), "ng")):
            side.add_training_image(tid, im, lab)
        rep = side.train_tool(tid)
        note = ("  - %d NG set aside" % len(rep["unseparated_ng_paths"])
                if rep["unseparated_ng_paths"] else "")
        print("  %-13s trained %-14s threshold %5.1f%s" % (name, tid, rep["threshold"], note))
    print("\nbuilt %s with %d sides.\nThe NG training images are demonstration data "
          "- replace them with real rejects." % (JOB, len(job.sides)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
