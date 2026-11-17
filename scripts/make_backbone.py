"""Build models/backbone/resnet18_features.onnx, the network the deep
Trained surface check sees with.

    python scripts/make_backbone.py [path to resnet18-v1-7.onnx]

Takes ResNet-18 from the ONNX model zoo (ImageNet weights; downloaded if no
path is given), keeps it up to its third stage, and returns the second and
third stages' feature maps for an image of any size. The classifier on top --
most of the file -- is cut away: 47 MB becomes 11.

Needs the `onnx` package (pip install onnx). The result runs with
onnxruntime alone.
"""
import os
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "models", "backbone", "resnet18_features.onnx")
URL = ("https://github.com/onnx/models/raw/main/validated/vision/classification/"
       "resnet/model/resnet18-v1-7.onnx")


def build(src, out=OUT):
    import onnx
    from onnx import utils

    m = onnx.load(src)
    g = m.graph
    # the activations that close stage 2 and stage 3
    wanted = []
    for stage in ("stage2", "stage3"):
        add = [n for n in g.node if n.op_type == "Add" and stage in n.name][-1]
        relu = next(n for n in g.node if n.op_type == "Relu" and add.output[0] in n.input)
        wanted.append(relu.output[0])
    data = next(i for i in g.input if i.name not in {x.name for x in g.initializer})
    dims = data.type.tensor_type.shape.dim
    for d, name in ((dims[0], "N"), (dims[2], "H"), (dims[3], "W")):
        d.ClearField("dim_value")
        d.dim_param = name
    del g.value_info[:]                       # shapes recorded for 224 x 224 only
    tmp = os.path.join(tempfile.mkdtemp(), "dyn.onnx")
    onnx.save(m, tmp)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    utils.extract_model(tmp, out, [data.name], wanted, check_model=False)
    cut = onnx.load(out)
    for o in cut.graph.output:                # any size in, any size out
        o.type.tensor_type.ClearField("shape")
    onnx.save(cut, out)
    return out, wanted


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    src = argv[0] if argv else None
    if src is None:
        src = os.path.join(tempfile.mkdtemp(), "resnet18-v1-7.onnx")
        print("  downloading ResNet-18 (47 MB) ...")
        urllib.request.urlretrieve(URL, src)
    out, names = build(src)
    print("  wrote %s (%.1f MB), outputs %s" % (out, os.path.getsize(out) / 1e6, names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
