"""Run this on the line PC itself before choosing a runtime or precision.

    python bench_inference.py

For every model it finds, it reports how long each runtime takes and -- the
part that matters -- whether OpenVINO's fast (reduced precision) mode changes
the answer. The speed gain depends entirely on the processor: recent Xeons and
some recent Core CPUs have hardware for reduced precision, older ones do not.
A number measured on a development machine says nothing about the line.
"""
import glob
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inference  # noqa: E402


def median_ms(fn, n=25):
    for _ in range(4):
        fn()
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000)
    return statistics.median(xs)


def input_for(path):
    """One input per model, shared by every runtime, so the outputs can be
    compared. Built from the file itself rather than from either runtime."""
    import onnx
    dims = onnx.load(path).graph.input[0].type.tensor_type.shape.dim
    shape = [d.dim_value if d.dim_value > 0 else s
             for d, s in zip(dims, [1, 3, 512, 512])]
    return np.random.RandomState(0).rand(*shape).astype(np.float32)


def cpu_name():
    try:
        import openvino as ov
        return ov.Core().get_property("CPU", "FULL_DEVICE_NAME")
    except Exception:
        import platform
        return platform.processor() or "unknown"


def main():
    print("Vision Station - inference benchmark")
    print("=" * 72)
    print("  CPU       %s" % cpu_name())
    have = inference.available_backends()
    print("  runtimes  %s" % (", ".join(have) or "NONE - pip install openvino"))
    if not have:
        return 2

    models = sorted(set(glob.glob(os.path.join(HERE, "models", "*.onnx")) +
                        glob.glob(os.path.join(HERE, "projects", "*", "models", "*.onnx"))))
    if not models:
        print("\n  No .onnx models found in models/ or projects/*/models/.")
        return 0

    configs = [("onnxruntime", "accuracy"), ("openvino", "accuracy"),
               ("openvino", "speed")]
    configs = [c for c in configs if c[0] in have]
    advice = []

    for path in models:
        print("\n  %s" % os.path.relpath(path, HERE))
        results, ref = {}, None
        x = input_for(path)
        for be, pr in configs:
            inference.clear_cache()
            try:
                m = inference.load(path, be, pr)
                out = m(x)[0]
                ms = median_ms(lambda: m(x))
            except Exception as e:
                print("    %-11s %-8s failed: %s" % (be, pr, str(e)[:50]))
                continue
            if ref is None:
                ref = out
            diff = float(np.abs(out - ref).max()) if out.shape == ref.shape else float("nan")
            results[(be, pr)] = (ms, diff)
            print("    %-11s %-8s %8.1f ms   max difference from reference %.2e"
                  % (be, pr, ms, diff))

        base = results.get(("openvino", "accuracy")) or results.get(("onnxruntime", "accuracy"))
        fast = results.get(("openvino", "speed"))
        if base and fast:
            gain = base[0] / fast[0]
            if gain < 1.3:
                advice.append("%s: fast mode is only %.1fx faster on this CPU - "
                              "keep 'accuracy'." % (os.path.basename(path), gain))
            elif fast[1] > 0.05:
                advice.append("%s: fast mode is %.1fx faster but moves outputs by "
                              "%.2f - check real parts before switching."
                              % (os.path.basename(path), gain, fast[1]))
            else:
                advice.append("%s: fast mode is %.1fx faster and barely changes "
                              "outputs - worth switching after checking real parts."
                              % (os.path.basename(path), gain))

    print("\n" + "=" * 72)
    for a in advice:
        print("  " + a)
    print("\n  Set Inference runtime and Precision on each model tool's Settings tab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
