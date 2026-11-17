"""Where things are, in one place.

ROOT is the folder that holds the vision_station package: the project folder
in a source checkout, or the runtime folder inside an unpacked .vsp package.
Both have models/ beside the package, so a model is found the same way in the
application and in a package running on a line PC.
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE_DIR)

MODELS = os.path.join(ROOT, "models")
ASSETS = os.path.join(ROOT, "assets")
PROJECTS = os.environ.get("VS_PROJECTS", os.path.join(ROOT, "projects"))
LOGS = os.path.join(ROOT, "logs")


def model(*parts):
    """A file under models/, e.g. model("ocr", "det_small.onnx")."""
    return os.path.join(MODELS, *parts)
