"""Terminals: every tool's inputs and outputs, and the rules for linking them.

VisionPro shows a tool's inputs and outputs as terminals and links them by
dragging one onto another. Here the same idea, over what this application's
tools actually exchange:

    image      an image -- the Image Source, or an image-processing tool's output
    pose       a fixture -- produced by Locate part; followed by later tools
    feature    a fitted line or circle -- read by Distance / angle
    blobs      found objects -- read by LED colour and LED lit state
    value      a single result -- any number a tool measured, driving a setting

A link is legal when the types match and the source runs EARLIER than the tool
reading it: a later tool's output does not exist yet when this one runs, and
linking to it would silently read nothing. Both rules are checked here, once,
so every way of making a link -- drag and drop, a menu, the member browser --
gets the same answer.
"""
from vision_station.core.schema import SCHEMA

SOURCE = "Image Source"

# outputs each tool type produces, beyond its result values
OUTPUTS = {
    "image_process": [("OutputImage", "image")],
    "align": [("Pose", "pose")],
    "find_line": [("Feature", "feature")],
    "find_circle": [("Feature", "feature")],
    "count_blob": [("Blobs", "blobs")],
    "intersect": [("Feature", "feature")],
    "fit_shape": [("Feature", "feature")],
    "find_ellipse": [("Feature", "feature")],
    "affine_transform": [("OutputImage", "image")],
    "polar_unwrap": [("OutputImage", "image")],
    "image_combine": [("OutputImage", "image")],
    "fixture_npoint": [("Pose", "pose")],
    "construct": [("Feature", "feature")],
}

# settings whose value is a link, and the type they accept
_REF_PARAMS = {
    "distance": [("first", "feature"), ("second", "feature")],
    "color_classify": [("source", "blobs")],
    "led_glow": [("source", "blobs")],
    "intersect": [("first", "feature"), ("second", "feature")],
    "fit_shape": [("source", "blobs")],
    "construct": [("first", "feature"), ("second", "feature"), ("third", "feature")],
}

# settings that cannot sensibly be driven by another tool's number
_NOT_LINKABLE = {"enabled", "critical", "roi", "angle", "shape", "points", "capture",
                 "fixture", "links", "model", "first", "second", "source"}


def inputs(tool):
    """[(name, type, current source text)] for one tool spec."""
    p = tool.get("params", {})
    out = []
    cap = p.get("capture") or "default"
    out.append(("InputImage", "image",
                "%s.OutputImage" % (SOURCE if cap == "default" else cap)))
    if tool["type"] != "align" or p.get("fixture"):
        fx = p.get("fixture") or "auto"
        out.append(("Fixture", "pose", {"auto": "(most recent Locate part)",
                                        "none": "(none - image space)"}.get(fx, "%s.Pose" % fx)))
    for name, typ in _REF_PARAMS.get(tool["type"], []):
        ref = p.get(name) or ""
        label = {"feature": "Feature", "blobs": "Blobs"}[typ]
        out.append((name, typ, ("%s.%s" % (ref, label)) if ref else "(not linked)"))
    for param, ref in (p.get("links") or {}).items():
        out.append((param, "value", ref))
    return out


def outputs(tool, result=None):
    """[(name, type)] -- typed outputs, then the values the last run produced."""
    out = list(OUTPUTS.get(tool["type"], []))
    if result:
        out.append(("score", "value"))
        for k, v in (result.get("values") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append((k, "value"))
    return out


def linkable_params(tool):
    """Settings the member browser can turn into inputs: numeric ones from the
    tool's schema -- VisionPro only allows Add Input on writable properties."""
    out = []
    for row in SCHEMA.get(tool["type"], []):
        key, label, kind = row[0], row[1], row[2]
        if key in _NOT_LINKABLE or kind not in ("int", "float"):
            continue
        out.append((key, label))
    return out


def _input_type(tool, input_name):
    """The type an input accepts: one of its terminals, or 'value' for a
    setting the member browser can drive. None if it has no such input."""
    want = next((typ for n, typ, _s in inputs(tool) if n == input_name), None)
    if want is None and input_name in dict(linkable_params(tool)):
        want = "value"
    return want


def _index(tools, tid):
    return next((i for i, t in enumerate(tools) if t["id"] == tid), None)


def link(tools, dst_id, input_name, src_id, output_name):
    """Connect src_id.output_name to dst_id.input_name. Returns None on success
    or the reason it is not allowed; the recipe is only changed on success."""
    di = _index(tools, dst_id)
    if di is None:
        return "no tool %s" % dst_id
    dst = tools[di]
    want = _input_type(dst, input_name)
    if want is None:
        return "%s has no input %s" % (dst_id, input_name)

    if src_id == SOURCE:
        if want != "image" or output_name != "OutputImage":
            return "the Image Source only provides an image"
        dst["params"]["capture"] = "default"
        return None

    si = _index(tools, src_id)
    if si is None:
        return "no tool %s" % src_id
    if si >= di:
        return ("%s runs after %s, so its %s does not exist yet when %s runs - "
                "move it earlier first" % (src_id, dst_id, output_name, dst_id))
    src = tools[si]
    have = dict(OUTPUTS.get(src["type"], []))
    if output_name in have:
        typ = have[output_name]
    elif output_name == "score" or output_name not in have:
        typ = "value"
    if typ != want:
        return "%s.%s is %s; %s.%s needs %s" % (src_id, output_name, _a(typ),
                                               dst_id, input_name, _a(want))
    p = dst.setdefault("params", {})
    if want == "image":
        p["capture"] = src_id
    elif want == "pose":
        p["fixture"] = src_id
    elif want in ("feature", "blobs"):
        p[input_name] = src_id
    else:
        p.setdefault("links", {})[input_name] = "%s.%s" % (src_id, output_name)
    return None


def unlink(tools, dst_id, input_name):
    """Back to the default: the Image Source, the automatic fixture, no
    feature, or a fixed setting instead of a linked one."""
    di = _index(tools, dst_id)
    if di is None:
        return
    p = tools[di].setdefault("params", {})
    if input_name == "InputImage":
        p["capture"] = "default"
    elif input_name == "Fixture":
        p["fixture"] = "auto"
    elif input_name in ("first", "second", "source"):
        p[input_name] = ""
    else:
        (p.get("links") or {}).pop(input_name, None)
        if not p.get("links"):
            p.pop("links", None)


def compatible_sources(tools, dst_id, input_name, results=None):
    """Every (tool id, output) that may legally feed this input, earliest first."""
    di = _index(tools, dst_id)
    if di is None:
        return []
    dst = tools[di]
    want = _input_type(dst, input_name)
    out = []
    if want == "image":
        out.append((SOURCE, "OutputImage"))
    for t in tools[:di]:
        res = (results or {}).get(t["id"])
        for name, typ in outputs(t, res):
            if typ == want:
                out.append((t["id"], name))
    return out


def rename_references(tools, old, new):
    """Keep every link pointing at a renamed tool."""
    for t in tools:
        p = t.get("params", {})
        for key in ("capture", "fixture", "first", "second", "source"):
            if p.get(key) == old:
                p[key] = new
        for param, ref in list((p.get("links") or {}).items()):
            if ref.split(".", 1)[0] == old:
                p["links"][param] = new + "." + ref.split(".", 1)[1]


def _a(typ):
    return {"image": "an image", "pose": "a fixture", "feature": "a feature",
            "blobs": "a set of blobs", "value": "a value"}.get(typ, typ)
