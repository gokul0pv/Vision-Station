"""Start Vision Core from this folder: press Run in VS Code, or
    python run_vision_core.py

Uses the latest build (vision_core/build), else the packaged copy
(vision_core/dist/VisionCore). If Vision Core is already running, the
Studio window comes to the front instead."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for folder in ("build", os.path.join("dist", "VisionCore")):
    exe = os.path.join(HERE, "vision_core", folder, "Vision Core.exe")
    if os.path.exists(exe):
        subprocess.Popen([exe], cwd=os.path.dirname(exe))
        print("Vision Core started:", exe)
        break
else:
    sys.exit("Vision Core is not built yet - run vision_core\\build.bat first")
