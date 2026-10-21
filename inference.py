"""One inference backend for every model in the application.

OpenVINO when it is installed, onnxruntime otherwise. Both load the same .onnx
file, so a model trained on Colab and exported once runs on either.

**Precision is a deliberate choice here, and the default is accuracy.**

Measured on an Intel Xeon with bfloat16 hardware, on the text detector:

    onnxruntime               64.9 ms
    OpenVINO, its default     15.3 ms    4.2x faster
    OpenVINO, full precision  58.9 ms    1.1x faster

The whole speedup comes from OpenVINO silently dropping to reduced precision.
Its logits then differ from the reference by up to 1.3, against 1e-5 at full
precision. On a real chassis label that happened to produce identical boxes,
but a detection sitting near its threshold can flip, and an inspection system
should not change its verdict because a runtime chose a number format.

So "accuracy" (fp32) is the default and "speed" is opt-in, to be switched on
after checking on real parts that results do not move. How much "speed" buys
depends entirely on the CPU: bfloat16 acceleration exists on recent Xeons and
some recent Core processors, and an older line PC may gain little. Run
`python bench_inference.py` on the actual machine before deciding.
"""
import os
import threading

import numpy as np

_CACHE = {}
_CACHE_LOCK = threading.Lock()


def available_backends():
    out = []
    try:
        import openvino  # noqa: F401
        out.append("openvino")
    except ImportError:
        pass
    try:
        import onnxruntime  # noqa: F401
        out.append("onnxruntime")
    except ImportError:
        pass
    return out


def choose_backend(requested="auto"):
    have = available_backends()
    if not have:
        raise ImportError("no inference runtime installed - "
                          "pip install openvino   (or onnxruntime)")
    if requested in have:
        return requested
    return have[0]            # openvino first when present


class Model:
    """A loaded model with a uniform call: model(array) -> list of outputs."""

    def __init__(self, path, backend="auto", precision="accuracy", device="CPU"):
        self._lock = threading.Lock()
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path
        self.backend = choose_backend(backend)
        self.precision = precision
        self.device = device
        self._shape = None
        if self.backend == "openvino":
            import openvino as ov
            self._core = ov.Core()
            self._model = self._core.read_model(path)
            self._compiled = None
        else:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._sess = ort.InferenceSession(
                path, opts, providers=["CPUExecutionProvider"])
            self._in = self._sess.get_inputs()[0].name

    def _compile(self, shape):
        """OpenVINO compiles for a concrete input shape; recompile only when the
        shape actually changes, which for a fixed camera is once."""
        cfg = {}
        if self.precision == "accuracy":
            cfg["INFERENCE_PRECISION_HINT"] = "f32"
        m = self._core.read_model(self.path)
        if not m.inputs[0].get_partial_shape().is_static:
            m.reshape({0: list(shape)})        # a static model keeps its own
        self._compiled = self._core.compile_model(m, self.device, cfg)
        self._req = self._compiled.create_infer_request()
        self._shape = tuple(shape)

    def __call__(self, x):
        """Locked per model. A cached model is shared by every job that uses
        it, and OpenVINO's single infer request -- and the recompile on a new
        input size -- are not safe to use from two threads at once; with jobs
        run in parallel, two jobs using the same text detector could read each
        other's results. Different models still run side by side."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        with self._lock:
            if self.backend == "openvino":
                if self._compiled is None or tuple(x.shape) != self._shape:
                    self._compile(x.shape)
                res = self._req.infer({0: x})
                return [res[o] for o in self._compiled.outputs]
            return self._sess.run(None, {self._in: x})

    def input_shape(self):
        """Input dimensions, with dynamic ones as strings on both runtimes.

        OpenVINO raises when asked for a concrete shape of a dynamic input, and
        this used to return None for such models -- the YOLO tool then fell back
        to a 640 letterbox whatever the model actually expected, and the
        benchmark fed the text detector a three-channel image."""
        if self.backend == "openvino":
            dims = []
            for d in self._model.inputs[0].get_partial_shape():
                dims.append(d.get_length() if d.is_static else "dyn")
            return dims
        return self._sess.get_inputs()[0].shape

    def metadata(self):
        """Custom metadata (YOLO stores its class names here)."""
        if self.backend == "onnxruntime":
            return dict(self._sess.get_modelmeta().custom_metadata_map or {})
        try:
            import onnx
            return {p.key: p.value for p in onnx.load(self.path).metadata_props}
        except Exception:
            return {}


def load(path, backend="auto", precision="accuracy", device="CPU"):
    """Cached: building a session or compiling a model is slow, and doing it per
    inspection would dominate the cycle time. Locked, so two jobs starting at
    once build one model, not two."""
    key = (os.path.abspath(path), backend, precision, device)
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = Model(path, backend, precision, device)
        return _CACHE[key]


def clear_cache():
    _CACHE.clear()
