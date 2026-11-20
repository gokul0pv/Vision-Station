"""Where things are, in one place.

From source, everything is under ROOT, the folder that holds the
vision_station package (the project folder, or the runtime folder inside an
unpacked .vsp package -- both have models/ beside the package).

As the packaged application (VisionStation.exe, built by packaging/build.py)
it is split in two:

    RESOURCES   read-only, shipped inside the application: models/, assets/,
                the demo project, the runtime sources a .vsp export copies
    APP_DIR     the folder VisionStation.exe is in: projects/ and logs/, which
                the application writes to
"""
import os
import shutil
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE_DIR)

FROZEN = bool(getattr(sys, "frozen", False))
RESOURCES = getattr(sys, "_MEIPASS", ROOT) if FROZEN else ROOT
APP_DIR = os.path.dirname(os.path.abspath(sys.executable)) if FROZEN else ROOT

MODELS = os.path.join(RESOURCES, "models")
ASSETS = os.path.join(RESOURCES, "assets")
PROJECTS = os.environ.get("VS_PROJECTS", os.path.join(APP_DIR, "projects"))
LOGS = os.path.join(APP_DIR, "logs")
DEMO = "demo_led_panel"


def model(*parts):
    """A file under models/, e.g. model("ocr", "det_small.onnx")."""
    return os.path.join(MODELS, *parts)


def demo_source():
    """The demo project as shipped: inside the application when packaged,
    in projects/ when run from source."""
    if FROZEN:
        return os.path.join(RESOURCES, "demo", DEMO)
    return os.path.join(ROOT, "projects", DEMO)


def first_run():
    """The packaged application's first start: a projects folder beside the
    program, with the demo in it so there is something to open. Nothing is
    ever overwritten."""
    if not FROZEN:
        return
    try:
        os.makedirs(PROJECTS, exist_ok=True)
        src = demo_source()
        dst = os.path.join(PROJECTS, DEMO)
        if os.path.isdir(src) and not os.listdir(PROJECTS):
            shutil.copytree(src, dst)
    except OSError as e:
        print("[paths] first-run set-up skipped: %s" % e)
