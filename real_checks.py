"""Screws, part present / absent and measurement, on the real chassis photograph.

    python real_checks.py [side folder] [output folder]

For the front-top side of the chassis job by default. Every check's score is
recorded across the good-part conditions of bench_real.py and across planted
defects -- each screw removed in turn, two at once, each label removed, each
hole covered, a hole moved 3 px -- and the margin between the worst good score
and the best defective one is reported: the threshold belongs in that gap, and
a check with no gap cannot be fixed by tuning. Result photographs of each case
are written to the output folder with an index.html.
"""
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_real as B          # noqa: E402


def _move_patch(img, box, dx, dy):
    """A feature moved: its patch lifted, the hole it leaves inpainted, and the
    patch put down dx, dy away."""
    x, y, w, h = [int(v) for v in box]
    patch = img[y:y + h, x:x + w].copy()
    out = B._remove(img, box)
    out[y + dy:y + dy + h, x + dx:x + dx + w] = patch
    return out


def _bare(img, box, source):
    """A label gone: its whole extent replaced by bare panel copied from a
    label-free part of the same photograph. Inpainting only the check's region
    pulled the label's own colours back in from where it runs past the region,
    and left the label's light-and-dark layout behind -- a missing label the
    correlation still half recognised."""
    x, y, w, h = [int(v) for v in box]
    sx, sy, sw, sh = [int(v) for v in source]
    out = img.copy()
    out[y:y + h, x:x + w] = cv2.resize(img[sy:sy + sh, sx:sx + sw], (w, h))
    return out


# where a label really extends, when it runs past its check's region, and a
# label-free patch of the same panel to replace it with
LABEL_EXTENT = {"caution_label_present": [0, 1075, 1185, 869]}
BARE_PANEL = [1250, 1330, 700, 270]


def scenarios(side, img):
    """(name, image, check that must fail or None for a good part)."""
    spec = {t["id"]: t for t in side.spec["tools"]}
    out = [(name, fn(img), None, []) for name, fn in B.CONDITIONS]
    res = {t["tool_id"]: t for t in side.inspect(img)["tools"]}
    matches = (res.get("panel_screws", {}).get("values") or {}).get("matches") or []
    for i, m in enumerate(matches):
        box = [m["x"] - 4, m["y"] - 4, m["w"] + 8, m["h"] + 8]
        out.append(("screw %d of %d missing" % (i + 1, len(matches)), B._remove(img, box),
                    "panel_screws", [box]))
    if len(matches) >= 2:
        b0 = [matches[0]["x"] - 4, matches[0]["y"] - 4, matches[0]["w"] + 8, matches[0]["h"] + 8]
        b1 = [matches[-1]["x"] - 4, matches[-1]["y"] - 4, matches[-1]["w"] + 8, matches[-1]["h"] + 8]
        out.append(("two screws missing", B._remove(B._remove(img, b0), b1), "panel_screws", [b0, b1]))
    for tid, t in spec.items():
        if t["type"] == "presence":
            gone = (_bare(img, LABEL_EXTENT[tid], BARE_PANEL) if tid in LABEL_EXTENT
                    else B._remove(img, t["params"]["roi"]))
            out.append(("%s removed" % tid.replace("_present", ""), gone, tid, []))
            out.append(("%s removed, shifted and darker" % tid.replace("_present", ""),
                        cv2.convertScaleAbs(B._affine(gone, 30, -20, 1.0), alpha=0.85), tid, []))
    for tid in ("hole_a", "hole_b"):
        if tid in spec:
            x, y, w, h = spec[tid]["params"]["roi"]
            out.append(("%s covered" % tid, B._remove(img, [x + w * .2, y + h * .2, w * .6, h * .6]),
                        "hole_spacing", []))
    if "hole_b" in spec:
        x, y, w, h = spec["hole_b"]["params"]["roi"]
        out.append(("hole_b 3 px out of place", _move_patch(img, [x + 6, y + 6, w - 12, h - 12], 0, 3),
                    "hole_spacing", []))
    return out


CHECKS = ("panel_screws", "serial_label_present", "caution_label_present", "sample_tag_present",
          "hole_spacing")


def measure(side, scen):
    rows = []
    for name, im, must, marks in scen:
        res = side.inspect(im, keep_images=False)
        tools = {t["tool_id"]: t for t in res["tools"]}
        row = {"name": name, "must": must, "verdict": res["verdict"], "raw": res, "image": im,
               "marks": marks}
        for c in CHECKS:
            t = tools.get(c)
            if t is None:
                continue
            v = t.get("values") or {}
            row[c] = {"status": t["status"], "score": float(t.get("score") or 0),
                      "count": v.get("count"), "value": v.get("value_px")}
        rows.append(row)
    return rows


def margins(rows, report=print):
    report("\n%-24s %-26s %-26s %s" % ("check", "worst good part", "best defective part", "margin"))
    for c in CHECKS:
        good = [r[c] for r in rows if r["must"] is None and c in r]
        bad = [r[c] for r in rows if r["must"] == c and c in r]
        if not good or not bad:
            continue
        if c == "panel_screws":
            g = min(x["count"] or 0 for x in good)
            b = max(x["count"] or 0 for x in bad)
            report("%-24s %-26s %-26s %s" % (c, "%d screws" % g, "%d screws" % b,
                                             "clear" if g > b else "NONE"))
        elif c == "hole_spacing":
            gv = [x["value"] for x in good if x["value"] is not None]
            report("%-24s %-26s %-26s %s" % (
                c, "%.2f .. %.2f px" % (min(gv), max(gv)),
                ", ".join("%.2f" % x["value"] if x["value"] is not None else x["status"]
                          for x in bad), ""))
        else:
            g = min(x["score"] for x in good)
            b = max(x["score"] for x in bad)
            report("%-24s %-26s %-26s %s" % (c, "score %.3f" % g, "score %.3f" % b,
                                             "%.3f" % (g - b) if g > b else "NONE"))


def summary(rows, report=print):
    fr = [r for r in rows if r["must"] is None and r["verdict"] != "PASS"]
    esc = [r for r in rows if r["must"] and (r.get(r["must"], {}).get("status") == "pass")]
    for r in fr:
        report("  FALSE REJECT  %s: %s" % (r["name"], [c for c in CHECKS if c in r and r[c]["status"] != "pass"]))
    for r in esc:
        report("  ESCAPE        %s (%s passed)" % (r["name"], r["must"]))
    good = [r for r in rows if r["must"] is None]
    bad = [r for r in rows if r["must"]]
    report("good parts: %d of %d passed | defects: %d of %d caught by the right check" % (
        len(good) - len(fr), len(good), len(bad) - len(esc), len(bad)))
    return fr, esc


def _crop_on(img, box, pad):
    h, w = img.shape[:2]
    x, y, bw, bh = [int(v) for v in box]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    return img[y0:min(h, y + bh + pad), x0:min(w, x + bw + pad)]


def photos(side, rows, out):
    """A result photograph per case: the whole part with its verdict, and a
    close-up of the check it is about."""
    import vsruntime
    os.makedirs(out, exist_ok=True)
    spec = {t["id"]: t for t in side.spec["tools"]}
    focus = {"panel_screws": spec["panel_screws"]["params"]["roi"],
             "hole_spacing": [1210, 720, 160, 250]}
    for tid, t in spec.items():
        if t["type"] == "presence":
            focus[tid] = t["params"]["roi"]
    items = []
    for i, r in enumerate(rows):
        full = vsruntime.draw(r["image"], r["raw"], banner=True, title=r["name"], labels=True)
        name = "%02d_%s" % (i, "".join(c if c.isalnum() else "_" for c in r["name"])[:50])
        small = cv2.resize(full, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out, name + ".jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        key = r["must"] or "panel_screws"
        # only the check this photo is about -- every tool's graphics at once
        # piled their labels on top of each other
        keep = {key} | ({"hole_a", "hole_b"} if key == "hole_spacing" else set())
        focus_raw = dict(r["raw"], tools=[
            dict(t, graphics=[g for g in t.get("graphics", [])
                              if not (key == "hole_spacing" and g.get("role") == "region")])
            for t in r["raw"]["tools"] if t["tool_id"] in keep])
        plain = vsruntime.draw(r["image"], focus_raw, banner=False, labels=True)
        for bx, by, bw, bh in r.get("marks") or []:
            c = (int(bx + bw / 2), int(by + bh / 2))
            rad = int(max(bw, bh) * 0.9)
            cv2.circle(plain, c, rad, (60, 55, 225), 6, cv2.LINE_AA)
            cv2.putText(plain, "missing", (c[0] - rad, c[1] - rad - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        1.6, (60, 55, 225), 4, cv2.LINE_AA)
        close = _crop_on(plain, focus.get(key, [0, 0, 100, 100]), 60 if key != "hole_spacing" else 70)
        k = min(1.0, 900.0 / max(close.shape[:2]))
        if key == "hole_spacing":
            k = 2.2
        close = cv2.resize(close, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out, name + "_close.jpg"), close, [cv2.IMWRITE_JPEG_QUALITY, 88])
        state = r.get(key, {})
        what = ("%s screws" % state.get("count") if key == "panel_screws" else
                "spacing %.2f px" % state["value"] if key == "hole_spacing" and state.get("value") else
                "score %.3f" % state.get("score", 0))
        ok = (r["must"] is None and r["verdict"] == "PASS") or (r["must"] and state.get("status") != "pass")
        items.append((name, r["name"], r["verdict"], key, what, ok, r["must"] is None))
    html = ["<!doctype html><meta charset='utf-8'><title>Real chassis - screws, presence, measurement</title>",
            "<style>body{font-family:Segoe UI,Arial;margin:22px;background:#f4f5f7}h1{margin:0 0 6px}"
            ".g{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}"
            "figure{margin:0;background:#fff;border:1px solid #d6dae0;border-radius:6px;overflow:hidden}"
            "img{width:100%;display:block}figcaption{padding:7px 9px;font-size:13px}"
            ".ok{color:#16a34a;font-weight:600}.bad{color:#dc2626;font-weight:600}</style>",
            "<h1>Real chassis: screws, part present / absent, measurement</h1>"]
    for title, good in (("Good parts under line conditions - must PASS", True),
                        ("Planted defects - must FAIL on the right check", False)):
        html.append("<h2>%s</h2><div class='g'>" % title)
        for name, label, verdict, key, what, ok, is_good in items:
            if is_good != good:
                continue
            html.append("<figure><a href='%s.jpg'><img src='%s_close.jpg'></a><figcaption>"
                        "<b>%s</b> - verdict %s - %s: %s - <span class='%s'>%s</span></figcaption></figure>"
                        % (name, name, label, verdict, key, what, "ok" if ok else "bad",
                           "correct" if ok else ("FALSE REJECT" if good else "ESCAPE")))
        html.append("</div>")
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return items


def main(folder=None, out=None):
    import job as jobmod
    folder = folder or os.path.join(HERE, "projects", "chassis_job", "sides", "frt_top")
    job_folder = os.path.dirname(os.path.dirname(folder))
    side = next(s for s in jobmod.Job(job_folder).sides if os.path.samefile(s.folder, folder))
    img = B._photo(side)
    rows = measure(side, scenarios(side, img))
    margins(rows)
    print()
    summary(rows)
    if out:
        photos(side, rows, out)
    return rows


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else None)
