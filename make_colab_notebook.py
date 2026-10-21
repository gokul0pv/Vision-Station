"""Build the Colab notebook that trains every model this application uses.

Kept as a generator rather than a hand-edited .ipynb, because notebook JSON is
miserable to review and easy to corrupt.
"""
import json
import os

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": [l + "\n" for l in text.strip().split("\n")]})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": [l + "\n" for l in text.strip().split("\n")]})


md("""
# Vision Station - train models on Colab

For a line PC with no GPU. Train here on Colab's free GPU, download the result,
drop it into the project's `models` folder. Inference on the line PC needs no GPU.

**Before you start:** Runtime -> Change runtime type -> **T4 GPU**.

Two things you can train here:

1. **A YOLO detector** from images you labelled in Vision Station's Label
   workspace -- screws, connectors, components, defects.
2. **The text detector** (`textnet`), if you want to retrain it on more varied
   data than it shipped with.

Both are exported to ONNX, which Vision Station runs with OpenVINO or
onnxruntime.
""")

code("""
import torch, subprocess
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NONE - switch the runtime to T4 GPU before training")
""")

md("""
## Part 1 - YOLO detector from your labelled images

In Vision Station: **Label** workspace, label your images, then
**Dataset -> Export for training**. Zip the exported folder and upload it below.
""")

code("""
!pip -q install ultralytics openvino
from google.colab import files
up = files.upload()                 # choose your <model>_dataset.zip
zipname = next(iter(up))
!unzip -q -o "$zipname" -d /content/dataset
import glob, os
yaml = glob.glob("/content/dataset/**/data.yaml", recursive=True)[0]
print("dataset:", yaml)
""")

md("""
The `data.yaml` paths were written for your PC. Point them at Colab instead.
""")

code("""
import yaml as Y
root = os.path.dirname(yaml)
cfg = Y.safe_load(open(yaml))
cfg["path"] = root
Y.safe_dump(cfg, open(yaml, "w"))
print(cfg)
""")

md("""
Train. `yolov8n` is the right default for a line PC: small enough to run fast on
CPU. Move to `yolov8s` only if nano is not accurate enough after adding data --
more labelled images helps far more than a bigger model.

Augmentation is set for inspection, not for photos of the world: no horizontal
flip, because a mirrored board is a different part; modest rotation, because a
fixture only allows a little; strong brightness variation, because lighting
drift is the usual enemy on a real line.
""")

code("""
from ultralytics import YOLO
m = YOLO("yolov8n.pt")
m.train(data=yaml, epochs=150, imgsz=640, batch=16, patience=40,
        degrees=8.0, translate=0.08, scale=0.25, shear=2.0,
        fliplr=0.0, flipud=0.0, hsv_h=0.01, hsv_s=0.45, hsv_v=0.35,
        mosaic=0.7, close_mosaic=15, project="/content/runs", name="yolo")
metrics = m.val()
print("mAP50    %.4f" % metrics.box.map50)
print("mAP50-95 %.4f" % metrics.box.map)
""")

md("""
Read mAP50 like this:

| mAP50 | meaning |
|---|---|
| above 0.95 | ready for the line |
| 0.85 - 0.95 | usable, expect occasional misses |
| below 0.85 | label more images - do not train longer |
""")

code("""
best = "/content/runs/yolo/weights/best.pt"
onnx_path = YOLO(best).export(format="onnx", opset=12, simplify=True, dynamic=False)
print("ONNX:", onnx_path)
files.download(onnx_path)
""")

md("""
Put the downloaded `best.onnx` in `projects/<your model>/models/`, then add a
**Detect with YOLO** tool and set its model file. Class names travel inside the
file, so you do not retype them.
""")

md("""
## Part 2 - retrain the text detector (optional)

The shipped `textnet.onnx` was trained for about fifteen minutes on a single CPU
core. A GPU allows far more steps and more varied data. Upload `textnet_data.py`
and `textnet_train.py` from your Vision Station folder.
""")

code("""
!pip -q install onnx onnxscript onnxruntime opencv-python-headless
up2 = files.upload()        # textnet_data.py and textnet_train.py
!mkdir -p models
""")

md("""
The training script uses every CPU thread it finds, but on Colab the gain is
the GPU. Patch it to train there and run longer.
""")

code("""
import re
src = open("textnet_train.py").read()
src = src.replace("model = TextNet(a.width)",
                  "model = TextNet(a.width).to(DEV)")
src = src.replace("x = torch.from_numpy(xs)", "x = torch.from_numpy(xs).to(DEV)")
src = src.replace("y = downsample_mask(torch.from_numpy(ys))",
                  "y = downsample_mask(torch.from_numpy(ys)).to(DEV)")
src = src.replace("import textnet_data as D",
                  "import textnet_data as D\\nDEV = 'cuda' if torch.cuda.is_available() else 'cpu'")
src = src.replace("logits = model(torch.from_numpy(xs))",
                  "logits = model(torch.from_numpy(xs).to(DEV)).cpu()")
src = src.replace("model.load_state_dict(torch.load(os.path.join(OUT, \\"textnet.pt\\")))",
                  "model.load_state_dict(torch.load(os.path.join(OUT, \\"textnet.pt\\"))); model = model.cpu()")
open("textnet_train_gpu.py", "w").write(src)
print("patched for", "GPU" if torch.cuda.is_available() else "CPU")
""")

code("""
!python textnet_train_gpu.py --minutes 40 --batch 32 --width 24 --resume 0
""")

code("""
files.download("models/textnet.onnx")
import os
if os.path.exists("models/textnet.onnx.data"):
    files.download("models/textnet.onnx.data")
""")

md("""
Replace `models/textnet.onnx` in your Vision Station folder with the download
(and its `.data` file if one was produced).

## Choosing precision on the line PC

After copying a model across, run `python bench_inference.py` **on the line PC
itself**. OpenVINO's fast mode uses reduced precision, and how much that buys
depends on the CPU: recent Xeons and some recent Core processors have hardware
for it, older ones gain little. The benchmark also reports whether fast mode
moves any detections on your own images -- only switch to it if they do not.
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_on_colab.ipynb"),
          "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("notebook written: %d cells" % len(cells))
