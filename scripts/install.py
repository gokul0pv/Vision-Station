"""Installer.

This used to be batch script logic. It is Python now for one reason: batch
cannot be tested from here, so the version check, the tiered installs and the
failure handling were all shipped unexecuted. The batch file is now three lines
that call this, and everything that can actually be wrong lives here where it
can be run against a fake pip.

    python scripts/install.py            install core, then optional packages
    python scripts/install.py --core     core only
    python scripts/install.py --all      include the large OCR download without asking
    python scripts/install.py --dry-run  show what would happen and change nothing
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the project root

MIN_PYTHON = (3, 10)

CORE = [
    ("PySide6", ["PySide6"], "the application window"),
    ("OpenCV", ["opencv-contrib-python==4.10.0.84"], "all image processing"),
    ("NumPy", ["numpy<2.2"], "all image processing"),
]

def onnxruntime_package():
    """The onnxruntime build to install, or None to leave what is there.

    Exactly one build may be installed. They all provide the same
    `onnxruntime` module, so installing the plain CPU build on top of a GPU
    build silently replaces it: that is how a machine with an RTX card ended
    up reading text on the CPU -- this installer was run again and added
    `onnxruntime` beside `onnxruntime-directml`. On Windows the DirectML
    build is the default: it runs on any DirectX 12 GPU, integrated ones
    included, and on the CPU where there is none."""
    import importlib.metadata as md
    for name in ("onnxruntime-directml", "onnxruntime-gpu", "onnxruntime"):
        try:
            md.version(name)
            return None
        except md.PackageNotFoundError:
            continue
    return "onnxruntime-directml" if sys.platform == "win32" else "onnxruntime"


_ORT = onnxruntime_package()

OPTIONAL = [
    ("OpenVINO", ["openvino"],
     "fast CPU inference for a YOLO model on a PC without a GPU"),
] + ([("onnxruntime", [_ORT],
       "text reading, the deep surface check and YOLO models")] if _ORT else []) + [
    ("pyzbar", ["pyzbar"], "QR and 1D barcode reading"),
    ("pylibdmtx", ["pylibdmtx"], "DataMatrix reading"),
    ("pytest", ["pytest"], "the test suite"),
]

# No longer installed by default: the text reader runs its own ONNX models
# (get_ocr_models.py) through onnxruntime. PaddleOCR remains usable as a
# fallback where it is already installed.
LARGE = [
    ("PaddleOCR", ["paddleocr", "paddlepaddle"],
     "text reading without the reader's own models"),
]


def python_ok():
    return sys.version_info[:2] >= MIN_PYTHON


def pip_install(packages, dry_run=False, runner=None):
    """Returns (ok, output). `runner` exists so tests can drive this without
    touching the network."""
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + list(packages)
    if dry_run:
        return True, "would run: " + " ".join(cmd)
    run = runner or (lambda c: subprocess.run(c, capture_output=True, text=True))
    try:
        proc = run(cmd)
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e)
    out = (getattr(proc, "stdout", "") or "") + (getattr(proc, "stderr", "") or "")
    return proc.returncode == 0, out.strip()


def install_group(group, dry_run=False, runner=None, required=False, log=print):
    """Each package is installed on its own so one failure cannot take the rest
    with it. A machine behind a proxy will often get some and not others, and
    stopping at the first refusal would leave it worse off than continuing."""
    failures = []
    for name, packages, why in group:
        ok, out = pip_install(packages, dry_run, runner)
        if ok:
            log("    installed   %s" % name)
        else:
            failures.append((name, why, out))
            if required:
                log("    FAILED      %s  - %s" % (name, why))
            else:
                log("    skipped     %s  - %s will be unavailable" % (name, why))
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description="Install Vision Station's dependencies")
    ap.add_argument("--core", action="store_true", help="core packages only")
    ap.add_argument("--all", action="store_true",
                    help="fetch the text reader's models (145 MB) without asking")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be installed and change nothing")
    ap.add_argument("--yes", action="store_true", help="never prompt")
    a = ap.parse_args(argv)

    print("Vision Station - install")
    print("=" * 68)
    print("  Python %s" % sys.version.split()[0])

    if not python_ok():
        print()
        print("  Python %d.%d or newer is required." % MIN_PYTHON)
        print("  Install a newer Python from python.org and run this again.")
        return 2

    print()
    print("  Core packages - the application cannot run without these:")
    failed = install_group(CORE, a.dry_run, required=True)
    if failed:
        print()
        print("  Core install failed. Nothing else is worth trying yet.")
        print("  If this machine is behind a proxy, pip needs to be told about it.")
        for name, _why, out in failed:
            if out:
                print("    %s: %s" % (name, out.splitlines()[-1][:100]))
        return 1

    if a.core:
        print("\n  Core only, as asked. Run scripts/preflight.py to see what is missing.")
        return 0

    print()
    print("  Optional packages - any of these may fail harmlessly:")
    install_group(OPTIONAL, a.dry_run)

    want_large = a.all
    if not want_large and not a.yes and not a.dry_run:
        try:
            reply = input("\n  Fetch the text reader's models, for Read text and "
                          "Verify text? 145 MB. [y/N] ")
            want_large = reply.strip().lower().startswith("y")
        except EOFError:
            want_large = False
    if want_large and not a.dry_run:
        print()
        try:
            import get_ocr_models
            get_ocr_models.main([])
        except Exception as e:
            print("  The models were not fetched: %s" % e)
    elif not a.dry_run:
        print("\n  Skipped the text reader's models. Fetch them when needed with:")
        print("     python scripts/get_ocr_models.py")

    print()
    print("  Basler cameras need the Pylon SDK from baslerweb.com installed")
    print("  first, and only then:  python -m pip install pypylon")
    print("  Without it the application still runs, using ordinary webcams.")
    print()
    print("=" * 68)
    if not a.dry_run:
        try:
            from vision_station.core import preflight
            preflight.main()
        except Exception as e:
            print("  Could not run the preflight check: %s" % e)
    print("\n  Start with:  run.bat   (or: python -m vision_station)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
