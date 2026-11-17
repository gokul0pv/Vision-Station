"""Make models/parts/parts_yoloe26s.onnx, the network behind Find parts.

The application only needs that ONNX file and onnxruntime(-directml). This
script is how it was made, for when it has to be made again (a newer YOLOE,
a bigger variant):

  1. Ultralytics YOLOE-26s-seg prompt-free weights, yoloe-26s-seg-pf.pt,
     from https://github.com/ultralytics/assets/releases/download/v8.4.0/
     (40.0 MB, SHA-256 01605da3f2e0e5325df45d5f5eef8e40b5b5c7d085399be3c33ac638258b150a)
  2. Exported to ONNX at 1280 x 1280, opset 17, up to 1000 detections, in a
     SEPARATE virtual environment holding torch, torchvision and ultralytics
     (installed --no-deps, with YOLO_AUTOINSTALL=false). Ultralytics installs
     plain `onnxruntime` when it exports, and that silently replaces the
     DirectML build the application runs on -- never export in the
     application's own Python.
  3. Cut down here: the 4585 per-name scores (the prompt-free vocabulary,
     meaningless on a circuit board) are reduced on the graph to one
     objectness score, so the output is (1, 37, 33600) -- box, score, 32 mask
     coefficients -- instead of (1, 4621, 33600). That is 125 times less data
     back from the graphics card: 40 ms a tile instead of 175 on an RTX 5060.

Usage (step 3 only, in the application's Python):
    python scripts/make_part_model.py yoloe-26s-seg-pf.onnx
Steps 1-2, in a throwaway venv:
    python -m venv --system-site-packages exportenv
    exportenv\\Scripts\\pip install --no-deps torch torchvision ultralytics
    set YOLO_AUTOINSTALL=false
    exportenv\\Scripts\\python -c "from ultralytics import YOLOE; YOLOE('yoloe-26s-seg-pf.pt').export(format='onnx', imgsz=1280, opset=17, simplify=False, max_det=1000)"

The weights are Ultralytics' and AGPL-3.0 licensed (see the model's metadata);
use within a company is fine, shipping the model inside a product sold to
others needs Ultralytics' enterprise licence.
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "models", "parts", "parts_yoloe26s.onnx")
VOCABULARY = 4585


def slim(src, dst=OUT):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(src)
    g = m.graph
    out0 = g.output[0].name
    meta = {p.key: p.value for p in m.metadata_props}

    def const(name, arr):
        g.initializer.append(numpy_helper.from_array(np.array(arr, dtype=np.int64), name))
    const("s_b0", [0])
    const("s_b1", [4])
    const("s_c1", [4 + VOCABULARY])
    const("s_m1", [4 + VOCABULARY + 32])
    const("ax1", [1])
    g.node.extend([
        helper.make_node("Slice", [out0, "s_b0", "s_b1", "ax1"], ["boxes"]),
        helper.make_node("Slice", [out0, "s_b1", "s_c1", "ax1"], ["cls"]),
        helper.make_node("ReduceMax", ["cls"], ["score"], axes=[1], keepdims=1),
        helper.make_node("Slice", [out0, "s_c1", "s_m1", "ax1"], ["coef"]),
        helper.make_node("Concat", ["boxes", "score", "coef"], ["parts"], axis=1),
    ])
    protos = g.output[1]
    anchors = g.output[0].type.tensor_type.shape.dim[2].dim_value or 33600
    del g.output[:]
    g.output.extend([helper.make_tensor_value_info("parts", TensorProto.FLOAT,
                                                   [1, 37, anchors]), protos])
    del m.metadata_props[:]
    for k, v in {"source": "Ultralytics YOLOE-26s-seg prompt-free (yoloe-26s-seg-pf.pt, assets v8.4.0)",
                 "license": meta.get("license", "AGPL-3.0"),
                 "imgsz": meta.get("imgsz", "[1280, 1280]"), "stride": meta.get("stride", "32"),
                 "layout": "parts: cx,cy,w,h, objectness, 32 mask coefficients; "
                           "output1: mask prototypes",
                 "exported_by": "Vision Station make_part_model.py"}.items():
        m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    onnx.checker.check_model(m)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    onnx.save(m, dst)
    return dst


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print("saved", slim(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else OUT))
