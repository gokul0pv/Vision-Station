"""Build the Windows application: dist/VisionStation/VisionStation.exe, and a
zip of that folder to copy to another PC.

    python packaging/build.py            build and zip
    python packaging/build.py --no-zip   build only

Needs PyInstaller (pip install pyinstaller) and the networks in models/
(scripts/get_ocr_models.py, scripts/make_backbone.py, scripts/make_part_model.py).
The target PC needs nothing installed: Python, Qt, OpenCV, ONNX Runtime and
the Basler camera driver are all inside the folder. A GPU is used through
DirectML when there is one.
"""
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, "build", "app")
DIST = os.path.join(ROOT, "dist")
sys.path.insert(0, ROOT)

README = """Vision Station {version}
=======================

Start:        VisionStation.exe
Classic UI:   VisionStation.exe --classic
Your work:    the projects folder beside VisionStation.exe (created on first
              start, with a demo inside); logs in the logs folder.

Settings (the gear, top right) holds the camera's IP address and every camera
setting, where result pictures are saved, the line connection, appearance and
Operator Mode. Close Pylon Viewer before starting: a GigE camera accepts one
program at a time.

Trained models and text reading run on the graphics card when there is one.
To keep them on the CPU, start from a command prompt with
    set VS_DEVICE=CPU
    VisionStation.exe
"""

VERSION_FILE = """VSVersionInfo(
  ffi=FixedFileInfo(filevers=({v0}, {v1}, {v2}, 0), prodvers=({v0}, {v1}, {v2}, 0),
                    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'Gokul PV'),
      StringStruct('FileDescription', 'Vision Station - machine-vision inspection'),
      StringStruct('FileVersion', '{version}'),
      StringStruct('InternalName', 'VisionStation'),
      StringStruct('LegalCopyright', '(c) 2026 Gokul PV - MIT licence'),
      StringStruct('OriginalFilename', 'VisionStation.exe'),
      StringStruct('ProductName', 'Vision Station'),
      StringStruct('ProductVersion', '{version}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


def version():
    import vision_station
    return vision_station.__version__


def make_icon(path):
    """The application's icon (assets/icon, made by scripts/make_icon.py) as
    the .exe's .ico -- every size drawn at its own size."""
    import shutil
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import make_icon as mk
    shutil.copyfile(mk.main(), path)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    v = version()
    os.makedirs(WORK, exist_ok=True)
    make_icon(os.path.join(WORK, "VisionStation.ico"))
    parts = (v.split(".") + ["0", "0", "0"])[:3]
    with open(os.path.join(WORK, "version.txt"), "w", encoding="utf-8") as f:
        f.write(VERSION_FILE.format(version=v, v0=parts[0], v1=parts[1], v2=parts[2]))
    for need in ("ocr", "backbone", "parts"):
        if not os.path.isdir(os.path.join(ROOT, "models", need)):
            print("  note: models/%s is missing - the application will run without it" % need)

    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
           "--distpath", DIST, "--workpath", os.path.join(ROOT, "build", "pyinstaller"),
           os.path.join(HERE, "vision_station.spec")]
    print("  building Vision Station %s ..." % v)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        return r.returncode
    app_dir = os.path.join(DIST, "VisionStation")
    with open(os.path.join(app_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(README.format(version=v))
    shutil.copy(os.path.join(ROOT, "LICENSE"), os.path.join(app_dir, "LICENSE.txt"))
    print("  application: %s" % os.path.join(app_dir, "VisionStation.exe"))

    if "--no-zip" not in argv:
        z = os.path.join(DIST, "VisionStation-%s-win64.zip" % v)
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for root, _dirs, files in os.walk(app_dir):
                for name in files:
                    full = os.path.join(root, name)
                    zf.write(full, os.path.join("VisionStation", os.path.relpath(full, app_dir)))
        print("  zip:         %s  (%.0f MB)" % (z, os.path.getsize(z) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
