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
`python benchmarks/bench_inference.py` on the actual machine before deciding.

**Device.** CPU unless asked otherwise. A tool's "Use GPU if available" option,
or the environment variable VS_DEVICE=GPU (set by run.bat on a GPU machine),
asks for the GPU. That goes through onnxruntime with CUDA (onnxruntime-gpu) or
DirectML (onnxruntime-directml, any DirectX 12 GPU), and through OpenVINO only
when it reports a GPU device of its own (Intel graphics). With no GPU runtime
available it falls back to the CPU rather than failing the inspection; the
Model's `providers` attribute says where it actually runs.
"""
import os
import threading

import numpy as np

# DirectML runs one model at a time on the device: two sessions run from two
# threads at once -- the live teach preview reading text while a run reads
# text too -- crashed the process outright. Every DirectML session takes
# this one lock; CPU and OpenVINO models keep their own.
_DML_LOCK = threading.Lock()

_CACHE = {}
_CACHE_LOCK = threading.Lock()

_GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")


def default_device():
    """'GPU' when VS_DEVICE=GPU is set, otherwise 'CPU'."""
    return "GPU" if os.environ.get("VS_DEVICE", "").strip().upper() == "GPU" else "CPU"


def resolve_device(device=None):
    if device is None or str(device).strip().lower() in ("", "auto"):
        return default_device()
    return "GPU" if str(device).strip().upper() == "GPU" else "CPU"


def ort_gpu_providers():
    """GPU execution providers this onnxruntime build offers, best first."""
    try:
        import onnxruntime as ort
        have = ort.get_available_providers()
    except Exception:
        return []
    return [p for p in _GPU_PROVIDERS if p in have]


def openvino_has_gpu():
    try:
        import openvino as ov
        return any(d.startswith("GPU") for d in ov.Core().available_devices)
    except Exception:
        return False


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
        device = resolve_device(device)
        gpu_ort = ort_gpu_providers() if device == "GPU" else []
        if device == "GPU" and gpu_ort and backend in ("auto", "onnxruntime"):
            self.backend = "onnxruntime"
        elif device == "GPU" and backend in ("auto", "openvino") and openvino_has_gpu():
            self.backend = choose_backend("openvino")
        else:
            self.backend = choose_backend(backend)
            if self.backend != "onnxruntime" or not gpu_ort:
                device = "CPU"     # no GPU route for this runtime: run, don't fail
        self.precision = precision
        self.device = device
        self._shape = None
        if self.backend == "openvino":
            import openvino as ov
            self._core = ov.Core()
            self._model = self._core.read_model(path)
            self._compiled = None
            self.providers = ["OpenVINO:" + self.device]
        else:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ["CPUExecutionProvider"]
            if self.device == "GPU":
                providers = gpu_ort + providers
                if gpu_ort[0] == "DmlExecutionProvider":
                    # DirectML requires both of these off.
                    opts.enable_mem_pattern = False
                    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._sess = ort.InferenceSession(path, opts, providers=providers)
            self.providers = self._sess.get_providers()
            self._in = self._sess.get_inputs()[0].name
            if "DmlExecutionProvider" in self.providers:
                self._lock = _DML_LOCK

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


def load(path, backend="auto", precision="accuracy", device=None):
    """Cached: building a session or compiling a model is slow, and doing it per
    inspection would dominate the cycle time. Locked, so two jobs starting at
    once build one model, not two. `device` None means default_device()."""
    device = resolve_device(device)
    key = (os.path.abspath(path), backend, precision, device)
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = Model(path, backend, precision, device)
        return _CACHE[key]


# Models on the GPU that clear_cache() let go of: kept alive until the process
# ends, never destroyed mid-run. With onnxruntime-directml 1.24 on the RTX
# 5060, freeing DirectML sessions and then running a new one -- the text
# reader's -- crashed the whole process with an access violation inside
# onnxruntime (tests.py with VS_DEVICE=GPU, every time). The application
# itself never clears its models, so this only ever holds a few test models.
_RETIRED = []


def clear_cache():
    """Forget every loaded model, so the next load() builds it again. Models
    on DirectML/CUDA are retired, not destroyed (see _RETIRED)."""
    with _CACHE_LOCK:
        _RETIRED.extend(m for m in _CACHE.values()
                        if getattr(m, "device", "CPU") == "GPU" and m.backend == "onnxruntime")
        _CACHE.clear()
