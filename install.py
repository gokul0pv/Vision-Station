"""Installer.

This used to be batch script logic. It is Python now for one reason: batch
cannot be tested from here, so the version check, the tiered installs and the
failure handling were all shipped unexecuted. The batch file is now three lines
that call this, and everything that can actually be wrong lives here where it
can be run against a fake pip.

    python install.py            install core, then optional packages
    python install.py --core     core only
    python install.py --all      include the large OCR download without asking
    python install.py --dry-run  show what would happen and change nothing
"""
import argparse
import subprocess
import sys

MIN_PYTHON = (3, 10)

CORE = [
    ("PySide6", ["PySide6"], "the application window"),
    ("OpenCV", ["opencv-python"], "all image processing"),
    ("NumPy", ["numpy<2.2"], "all image processing"),
]

OPTIONAL = [
    ("OpenVINO", ["openvino"],
     "fast CPU inference for the trained-model tools - YOLO, AI surface "
     "check, AI text detection"),
    ("onnxruntime", ["onnxruntime"],
     "fallback runtime for the trained-model tools if OpenVINO is unavailable"),
    ("pyzbar", ["pyzbar"], "QR and 1D barcode reading"),
    ("pylibdmtx", ["pylibdmtx"], "DataMatrix reading"),
    ("pytest", ["pytest"], "the test suite"),
]

# Held back behind a prompt: several hundred megabytes, and plenty of lines
# never read text at all.
LARGE = [
    ("PaddleOCR", ["paddleocr", "paddlepaddle"],
     "text reading - much the best engine, especially on dot-matrix marking"),
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
                    help="include the large OCR download without asking")
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
        print("\n  Core only, as asked. Run preflight.py to see what is missing.")
        return 0

    print()
    print("  Optional packages - any of these may fail harmlessly:")
    install_group(OPTIONAL, a.dry_run)

    want_large = a.all
    if not want_large and not a.yes and not a.dry_run:
        try:
            reply = input("\n  Install PaddleOCR for text reading? Large download. [y/N] ")
            want_large = reply.strip().lower().startswith("y")
        except EOFError:
            want_large = False
    if want_large:
        print()
        install_group(LARGE, a.dry_run)
    else:
        print("\n  Skipped PaddleOCR. Text tools need an engine before they work:")
        print("     python -m pip install paddleocr paddlepaddle")

    print()
    print("  Basler cameras need the Pylon SDK from baslerweb.com installed")
    print("  first, and only then:  python -m pip install pypylon")
    print("  Without it the application still runs, using ordinary webcams.")
    print()
    print("=" * 68)
    if not a.dry_run:
        try:
            import preflight
            preflight.main()
        except Exception as e:
            print("  Could not run the preflight check: %s" % e)
    print("\n  Start with:  python app.py   (or run.bat)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
