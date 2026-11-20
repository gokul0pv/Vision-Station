# PyInstaller recipe for VisionStation.exe -- run through packaging/build.py,
# which makes the icon and version resource this file refers to.
#
# One folder, not one file: a single-file program unpacks a gigabyte of Qt,
# OpenCV and the camera driver into a temporary folder on every start.
import glob
import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
WORK = os.path.join(ROOT, "build", "app")          # icon and version file from build.py


def tree(src, dest, skip=("__pycache__", "results", "dataset")):
    """(file, folder-in-the-application) for every file under src."""
    out = []
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if f.endswith((".pyc", ".partial")):
                continue
            rel = os.path.relpath(root, src)
            out.append((os.path.join(root, f), os.path.normpath(os.path.join(dest, rel))))
    return out


datas = []
datas += tree(os.path.join(ROOT, "assets"), "assets")
for sub in ("ocr", "backbone", "parts"):                 # the networks the application runs
    if os.path.isdir(os.path.join(ROOT, "models", sub)):
        datas += tree(os.path.join(ROOT, "models", sub), os.path.join("models", sub))
datas += tree(os.path.join(ROOT, "projects", "demo_led_panel"), os.path.join("demo", "demo_led_panel"))
# The runtime's sources, which File > Export Package copies into a .vsp so a
# line PC can run it with plain Python. The user interface is not part of it.
for f in glob.glob(os.path.join(ROOT, "vision_station", "**", "*.py"), recursive=True):
    rel = os.path.relpath(os.path.dirname(f), ROOT)
    if not rel.replace("\\", "/").startswith(("vision_station/ui", "vision_station/glass")):
        datas.append((f, rel))

binaries = collect_dynamic_libs("pyzbar") + collect_dynamic_libs("pylibdmtx")
# the icon fonts (Phosphor) QtAwesome draws the interface's icons from
datas += collect_data_files("qtawesome")

# Installed on the build PC, never used by the application: Paddle (the app
# reads text with its own ONNX models), notebooks, plotting, other ML stacks.
excludes = [
    "paddle", "paddleocr", "paddlex", "torch", "torchvision", "tensorflow", "keras",
    "easyocr", "pytesseract", "openvino", "onnx", "pandas", "polars", "scipy", "sklearn",
    "matplotlib", "IPython", "jupyter", "notebook", "tkinter", "shapely", "ultralytics",
    "PyQt5", "PyQt6",
    "pytest", "PyInstaller",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel", "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtQuick",
    "PySide6.QtQml", "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtSql", "PySide6.QtTest",
]

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    # serial barcode scanners; qtawesome finds Qt through qtpy
    hiddenimports=["PySide6.QtSerialPort", "qtawesome", "qtpy"],
    excludes=excludes,
    noarchive=False,
)


def _stray_pylon(entry):
    """A Basler pylon / GenICam library PyInstaller copied into the
    application's top folder. pypylon carries its own complete pylon runtime
    in its own folder; when the pylon SDK is installed on the build PC (and on
    PATH), the dependency scan also copied the SDK's libraries -- version 12.3
    beside pypylon's 12.5 -- into the top folder, where they were found first.
    pypylon then failed to load ("The specified procedure could not be
    found"), and VisionStation.exe saw no Basler camera at all while
    python -m vision_station opened it fine. Only pypylon's own set is kept."""
    dest = entry[0].replace("\\", "/")
    if "/" in dest:
        return False                       # inside a package folder: its own
    name = dest.lower()
    return ("pylon" in name or "basler" in name or name.startswith("gx") or
            name.startswith("ux") or name.startswith("log4cpp"))


a.binaries = [b for b in a.binaries if not _stray_pylon(b)]
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VisionStation",
    icon=os.path.join(WORK, "VisionStation.ico"),
    version=os.path.join(WORK, "version.txt"),
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="VisionStation", upx=False)
