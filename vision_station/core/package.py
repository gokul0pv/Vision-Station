"""Export and import Vision Station packages (.vsp).

A package is one zip file holding a whole application -- every job, its
tools, reference image, trained models and lens calibration -- plus, by
default, the runtime that executes it, so a program with nothing but the .vsp
file and OpenCV installed can run the inspection:

    manifest.json     what is inside: jobs, tools, what they need, versions
    app/              the application folder (job.json and sides, or one model)
    runtime/          engine and tools, as used by the application
    vsruntime.py      the loader, at the top level so the .vsp itself can be
                      put on sys.path and imported from

Results logs, archived images, datasets and removed jobs are left out.
Training images are left out unless asked for: they are for retraining, not
for running, and can be large.

This is Vision Station's own format. Cognex's .vpp is proprietary serialized
.NET that only VisionPro reads, so it is neither produced nor readable here.
"""
import fnmatch
import json
import os
import time
import zipfile

from vision_station import paths

FORMAT, VERSION = "vision-station-package", 1

# The runtime a package carries: the vision_station package without its user
# interface -- the engine, the tools, the camera -- as runtime/vision_station.
RUNTIME_SKIP = {"ui", "__pycache__"}
SKIP_DIRS = {"results", "dataset", "removed", "__pycache__", ".pytest_cache"}
_RUNTIME = "onnxruntime (or onnxruntime-directml / onnxruntime-gpu for a GPU)"
TEXT_TYPES = {"ocr", "ocv", "ocr_auto", "text_detect"}
NEEDS = {"yolo_detect": [_RUNTIME], "yolo_defect": [_RUNTIME], "anomaly": [_RUNTIME],
         "text_detect": [_RUNTIME], "ocr": [_RUNTIME], "ocv": [_RUNTIME], "ocr_auto": [_RUNTIME],
         "code_read": ["pyzbar and pylibdmtx (optional: OpenCV reads QR and 1D codes alone)"]}


def _describe(app_folder):
    from vision_station.core import job as jobmod
    from vision_station.core.engine import Project
    jobs = []
    if jobmod.is_job(app_folder):
        j = jobmod.Job(app_folder)
        name, kind = j.name, "job"
        for s in j.sides:
            jobs.append({"id": s.side_id, "name": s.name,
                         "tools": [{"id": t["id"], "type": t["type"]}
                                   for t in s.spec.get("tools", [])]})
    else:
        p = Project(app_folder)
        name, kind = p.spec.get("name", p.id), "model"
        jobs.append({"id": p.id, "name": name,
                     "tools": [{"id": t["id"], "type": t["type"]} for t in p.spec.get("tools", [])]})
    return name, kind, jobs


def export_package(app_folder, out_path, include_training=False, include_runtime=True):
    """Write app_folder as a .vsp package. Returns the manifest."""
    app_folder = os.path.abspath(app_folder)
    name, kind, jobs = _describe(app_folder)
    types = sorted({t["type"] for j in jobs for t in j["tools"]})
    needs = sorted({n for t in types for n in NEEDS.get(t, [])})
    manifest = {"format": FORMAT, "version": VERSION, "name": name, "kind": kind,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "jobs": jobs,
                "tool_types": types, "requires": ["opencv-python", "numpy"] + needs,
                "runtime_included": bool(include_runtime),
                "training_included": bool(include_training)}
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)       # exporting into a new folder is ordinary
    tmp = out_path + ".partial"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        for root, dirs, files in os.walk(app_folder):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                             and (include_training or d != "training"))
            for f in sorted(files):
                if fnmatch.fnmatch(f, "*.pyc") or f.endswith(".partial"):
                    continue
                # the advanced model's training weights: only a later teaching
                # run starts from them; the package runs the .onnx
                if f == "model.pt" and os.path.basename(root) == "advanced" \
                        and not include_training:
                    continue
                full = os.path.join(root, f)
                z.write(full, os.path.join("app", os.path.relpath(full, app_folder)))
        if include_runtime:
            for root, dirs, files in os.walk(paths.PACKAGE_DIR):
                dirs[:] = sorted(d for d in dirs if d not in RUNTIME_SKIP)
                for f in sorted(files):
                    if f.endswith(".py"):
                        full = os.path.join(root, f)
                        rel = os.path.relpath(full, paths.ROOT)
                        z.write(full, os.path.join("runtime", rel))
            if TEXT_TYPES & set(types):
                # The text reader's networks travel with the package: the line
                # PC then reads text with onnxruntime alone, no PaddleOCR to
                # install and nothing to download there.
                ocr_dir = paths.model("ocr")
                if os.path.isdir(ocr_dir):
                    for f in sorted(os.listdir(ocr_dir)):
                        if f.endswith((".onnx", ".txt")):
                            z.write(os.path.join(ocr_dir, f),
                                    os.path.join("runtime", "models", "ocr", f),
                                    zipfile.ZIP_STORED)
            # at the top level too, so the .vsp can go straight onto sys.path
            z.write(os.path.join(paths.PACKAGE_DIR, "core", "vsruntime.py"), "vsruntime.py")
    os.replace(tmp, out_path)          # never leave a half-written package behind
    return manifest


def import_package(path, projects_root):
    """Unpack a package's application into projects_root, under a name that
    does not overwrite anything. Returns the new application's id."""
    from vision_station.core.vsruntime import read_manifest
    m = read_manifest(path)
    base = "".join(c if c.isalnum() else "_" for c in m["name"].lower()).strip("_") or "imported"
    app_id, n = base, 2
    while os.path.exists(os.path.join(projects_root, app_id)):
        app_id, n = "%s_%d" % (base, n), n + 1
    dest = os.path.join(projects_root, app_id)
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if not info.filename.startswith("app/") or info.is_dir():
                continue
            rel = info.filename[4:]
            target = os.path.realpath(os.path.join(dest, rel))
            if not target.startswith(os.path.realpath(dest) + os.sep):
                raise ValueError("unsafe path in package: %s" % info.filename)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                out.write(src.read())
    return app_id


# ================================================================ job file
_TEMPLATE = '''"""Vision Station job file: {name}
Generated {created} from QuickBuild.

Runs every job on its taught images and shows or saves the results:

    python {py}                          everything, as taught
    python {py} --output D:\\\\results     annotated results written there
    python {py} --no-show                no windows
    python {py} --source "Front top=D:\\\\parts\\\\top"   a different image or folder

Or from your own code:

    import {module}
    results = {module}.run(show=False)
    for r in results:
        print(r.job_name, r.verdict, r.failed_tools)

The inspection itself is in {vsp}, next to this file. The settings below are
what was taught; edit them freely for this PC.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.join(HERE, {vsp!r})

# Where each job's images come from: an image file, a folder (every image in it),
# a list of files, or "camera:0". Taught in QuickBuild > Job Editor > Image Source.
# Relative paths are taken from this file's folder, so the job file and its
# images can be copied to another PC together.
SOURCES = {sources}

# Where results go. Taught in QuickBuild > File > Result Output.
OUTPUT_FOLDER = {output!r}      # annotated images are written here ("" = not saved)
SHOW = {show!r}                  # show each result in a window
ONLY_FAILURES = {only_failures!r}         # save and show rejected parts only

sys.path.insert(0, PACKAGE)                      # the package carries the runtime
import vsruntime                                 # noqa: E402

_app = None


def application():
    global _app
    if _app is None:
        _app = vsruntime.load(PACKAGE)
    return _app


def _here(src):
    """Relative paths are relative to this file, not the working directory."""
    if isinstance(src, (list, tuple)):
        return [_here(x) for x in src]
    if isinstance(src, str) and src and not src.startswith("camera:") \
            and not os.path.isabs(src):
        return os.path.join(HERE, src)
    return src


def run(sources=None, output_folder=None, show=None, only_failures=None, on_result=None):
    """Run every job; returns every Result. on_result(result) is called for
    each one -- use it to show results in your own window or send them on."""
    merged = dict(SOURCES)
    merged.update(sources or {{}})
    merged = {{k: _here(v) for k, v in merged.items()}}
    return application().run_taught(
        merged,
        show=SHOW if show is None else show,
        save_to=_here((OUTPUT_FOLDER if output_folder is None else output_folder) or "") or None,
        only_failures=ONLY_FAILURES if only_failures is None else only_failures,
        on_result=on_result)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Vision Station job file: {name}")
    ap.add_argument("--output", help="write annotated results to this folder")
    ap.add_argument("--no-show", action="store_true", help="do not open result windows")
    ap.add_argument("--only-failures", action="store_true", help="rejected parts only")
    ap.add_argument("--source", action="append", default=[],
                    help='"Job name=path" to use another image, folder or camera:N')
    a = ap.parse_args(argv)
    over = dict(s.split("=", 1) for s in a.source)
    results = run(over, a.output, False if a.no_show else None,
                  True if a.only_failures else None,
                  on_result=lambda r: print("  %-18s %-6s %s" % (
                      r.job_name, r.verdict, r.error or ", ".join(r.failed_tools))))
    verdict = application().verdict(results)
    print("{name}: %s  (%d inspection%s)" % (verdict, len(results),
                                              "" if len(results) == 1 else "s"))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
'''


def _output_settings(app_folder):
    from vision_station.core import job as jobmod
    if jobmod.is_job(app_folder):
        spec = jobmod.Job(app_folder).spec
    else:
        with open(os.path.join(app_folder, "project.json"), encoding="utf-8") as f:
            spec = json.load(f)
    o = spec.get("output") or {}
    return o.get("folder", "results"), bool(o.get("show", True)), bool(o.get("only_failures", False))


def export_job_file(app_folder, py_path, include_training=False):
    """A runnable Python job file, and the package it runs beside it. Returns
    (py_path, vsp_path, manifest)."""
    from vision_station.core import vsruntime
    py_path = os.path.abspath(py_path)
    stem = os.path.splitext(os.path.basename(py_path))[0]
    module = "".join(c if c.isalnum() or c == "_" else "_" for c in stem)
    if module[:1].isdigit():
        module = "job_" + module
    os.makedirs(os.path.dirname(py_path), exist_ok=True)
    py_path = os.path.join(os.path.dirname(py_path), module + ".py")
    vsp_path = os.path.join(os.path.dirname(py_path), module + ".vsp")
    manifest = export_package(app_folder, vsp_path, include_training)
    app = vsruntime.Application(app_folder, manifest)
    sources = {}
    for jid, name in app.jobs:
        src = app.source(jid)
        # the reference image is inside the package, so leave it to the runtime
        ref = os.path.join(app._projects[jid][1].folder, "reference.png")
        sources[name] = None if src == ref else src
    lines = ["{"]
    for name, src in sources.items():
        lines.append("    %r: %s," % (name, "None" if src is None else repr(src)))
    lines.append("}")
    out_folder, show, only_fail = _output_settings(app_folder)
    text = _TEMPLATE.format(name=manifest["name"], created=manifest["created"],
                            py=os.path.basename(py_path), module=module,
                            vsp=os.path.basename(vsp_path), sources="\n".join(lines),
                            output=out_folder, show=show, only_failures=only_fail)
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(text)
    return py_path, vsp_path, manifest
