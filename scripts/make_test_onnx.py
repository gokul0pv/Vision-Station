"""Build a tiny, real ONNX model shaped exactly like a YOLOv8 export.

The YOLO tools were the least tested code in the application -- 18% covered --
purely because testing them appeared to need a trained model. It does not. What
needs testing is the part that can realistically be wrong: reading class names
out of the model metadata, undoing the letterbox padding, non-maximum
suppression, and the count-versus-expected logic. None of that depends on the
weights being any good.

So this emits a genuine ONNX graph that ignores its input and returns a fixed
detection tensor in YOLOv8 layout, (1, 4 + num_classes, num_boxes), with the
class names embedded in metadata the way Ultralytics does. It loads through real
onnxruntime, so the whole decode path is exercised for real rather than mocked.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper


def build(path, boxes, classes, imgsz=640, n_slots=None):
    """boxes: list of (cx, cy, w, h, class_id, confidence) in model input space.

    Coordinates are in the letterboxed input space, which is what a real model
    predicts in -- the tool is responsible for mapping them back to the source
    image, and that mapping is what we want under test.
    """
    nc = len(classes)
    n = n_slots or max(8, len(boxes) + 4)
    out = np.zeros((1, 4 + nc, n), np.float32)
    for i, (cx, cy, w, h, cid, conf) in enumerate(boxes):
        out[0, 0, i] = cx
        out[0, 1, i] = cy
        out[0, 2, i] = w
        out[0, 3, i] = h
        out[0, 4 + cid, i] = conf

    const = helper.make_node(
        "Constant", [], ["output0"],
        value=helper.make_tensor("v", TensorProto.FLOAT, out.shape, out.flatten()))
    graph = helper.make_graph(
        [const], "fake_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT,
                                       [1, 3, imgsz, imgsz])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(out.shape))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 9
    # Ultralytics stores names as a str(dict) in metadata; the tool reads it back
    # with ast.literal_eval, so the format has to match exactly.
    entry = model.metadata_props.add()
    entry.key = "names"
    entry.value = str({i: c for i, c in enumerate(classes)})
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def build_anomaly(path, heat, imgsz=256):
    """An anomalib-style model: takes a normalised image, returns a heatmap.

    `heat` is a 2-D float array. As with the detector, the weights are
    irrelevant -- what needs testing is the preprocessing, the threshold
    comparison, blob extraction from the heatmap and the region offset."""
    heat = np.ascontiguousarray(np.asarray(heat, np.float32))
    const = helper.make_node(
        "Constant", [], ["output"],
        value=helper.make_tensor("h", TensorProto.FLOAT, heat.shape,
                                 heat.flatten()))
    graph = helper.make_graph(
        [const], "fake_anomaly",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                       [1, 3, imgsz, imgsz])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                       list(heat.shape))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    import sys
    p = build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_yolo.onnx",
              [(320, 320, 64, 64, 0, 0.91), (120, 160, 40, 40, 1, 0.77)],
              ["screw", "led"])
    print("wrote", p)


def build_color_classifier(path, classes):
    """A real ONNX classifier: average colour -> logits favouring the dominant
    channel. Enough to exercise the tool end to end."""
    import json
    import onnx
    from onnx import TensorProto, helper
    w = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32) * 20.0   # R, G, B -> class
    nodes = [helper.make_node("GlobalAveragePool", ["x"], ["p"]),
             helper.make_node("Flatten", ["p"], ["f"]),
             helper.make_node("Gemm", ["f", "W"], ["y"], transB=1)]
    graph = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor("W", TensorProto.FLOAT, [3, 3], w.ravel().tolist())])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    helper.set_model_props(m, {"classes": json.dumps(classes)})
    onnx.save(m, path)
    return path
