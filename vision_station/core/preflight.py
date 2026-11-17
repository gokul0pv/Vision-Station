"""Environment check.

Run standalone before deploying:

    python scripts/preflight.py

The app also calls this at startup and shows anything that is missing, because
the failure it prevents is the expensive one: a recipe built on a bench machine
that has PaddleOCR, deployed to a line PC that does not, where the OCR step then
falls back to a weaker engine and starts passing parts it should reject. Nothing
in the inspection result would say why.

Nothing here is fatal. Every optional package is reported with what it costs you
and the exact command to install it.
"""
import os
import platform
import shutil
import subprocess
import sys

# name, import target, what it is for, whether the app is usable without it
CHECKS = [
    ("PySide6", "PySide6", "the application window itself", True),
    ("OpenCV", "cv2", "all image processing", True),
    ("NumPy", "numpy", "all image processing", True),
    ("OpenVINO", "openvino",
     "fast CPU inference for the trained-model tools", False),
    ("onnxruntime", "onnxruntime",
     "Read text, Verify text, the deep Trained surface check and Detect with "
     "YOLO. The onnxruntime-directml build runs them on the GPU", False),
    ("pypylon", "pypylon",
     "Basler GigE and USB3 cameras. Without it only ordinary webcams work, "
     "through OpenCV", False),
    ("PaddleOCR", "paddleocr",
     "text reading when the reader's own models (models/ocr) are not there",
     False),
    ("EasyOCR", "easyocr", "an alternative text engine", False),
    ("pytesseract", "pytesseract",
     "the fallback text engine. Needs the Tesseract program as well as this "
     "package", False),
    ("pyzbar", "pyzbar", "reading QR and 1D barcodes", False),
    ("pylibdmtx", "pylibdmtx", "reading DataMatrix codes", False),
    ("pytest", "pytest", "running the test suite", False),
]

INSTALL = {
    "PySide6": "pip install PySide6",
    "OpenCV": "pip install opencv-python",
    "NumPy": "pip install numpy",
    "OpenVINO": "pip install openvino",
    "onnxruntime": "pip install onnxruntime-directml   (Windows, uses the GPU)  "
                   "or  pip install onnxruntime",
    "pypylon": "install the Basler Pylon SDK from baslerweb.com, then: pip install pypylon",
    "PaddleOCR": "pip install paddleocr paddlepaddle",
    "EasyOCR": "pip install easyocr",
    "pytesseract": "pip install pytesseract  (and install Tesseract itself)",
    "pyzbar": "pip install pyzbar",
    "pylibdmtx": "pip install pylibdmtx",
    "pytest": "pip install pytest",
}


def _present(module):
    try:
        if module == "paddleocr":
            # importing it here must not load PyTorch either -- see
            # vstools.text._keep_torch_out
            from vision_station.tools.text import _keep_torch_out
            _keep_torch_out()
        __import__(module)
        return True
    except Exception:
        # Some optional packages raise on import when a native library is
        # missing rather than failing to import at all, so anything short of
        # success counts as absent.
        return False


def check():
    rows = []
    for name, module, why, required in CHECKS:
        rows.append({"name": name, "ok": _present(module), "why": why,
                     "required": required, "fix": INSTALL.get(name, "")})
    return rows


def ocr_fallback_reasons():
    """Why each better OCR engine did not start, as the text tool recorded it."""
    try:
        from vision_station.tools.text import _ENGINE, _engine
        _engine()
        return dict(_ENGINE.get("errors", {}))
    except Exception:
        return {}


def ocr_engine():
    """Which text engine will actually be used.

    Asks the text tool itself rather than guessing from what imports. A package
    can import perfectly and still be unusable: PaddleOCR downloads its models
    on first use, so on a machine with no route to its model host it imports,
    reports itself present, and then fails over to Tesseract at runtime.
    Preflight exists to stop exactly that kind of silent substitution, so it
    has to report the engine that really loads.
    """
    st = text_reader()
    if st.get("engine") == "onnx":
        return "PP-OCR on the %s" % ("GPU" if st.get("device") == "GPU" else "CPU")
    try:
        from vision_station.tools.text import _engine
        name = _engine().get("name")
    except Exception:
        name = None
    pretty = {"paddle": "PaddleOCR", "easyocr": "EasyOCR",
              "tesseract": "Tesseract"}
    if name:
        return pretty.get(name, name)

    for label, module in (("PaddleOCR", "paddleocr"), ("EasyOCR", "easyocr"),
                          ("Tesseract", "pytesseract")):
        if _present(module):
            if label == "Tesseract" and not shutil.which("tesseract"):
                continue
            return label
    return None


def text_reader():
    """The text reader as it will actually run: {'engine', 'device', 'models',
    'error'}. Loads it, so the answer is what an inspection gets."""
    try:
        from vision_station.tools import ocr_engine
        if ocr_engine.onnx_available():
            return ocr_engine.status()
    except Exception as e:
        return {"engine": None, "device": None, "models": (),
                "error": "%s: %s" % (type(e).__name__, str(e)[:160])}
    return {"engine": None, "device": None, "models": (), "error": None}


def gpu_runtime():
    """The GPU providers the installed onnxruntime offers ([] = CPU only)."""
    try:
        from vision_station.core import inference
        return inference.ort_gpu_providers()
    except Exception:
        return []


def tesseract_binary():
    exe = shutil.which("tesseract")
    if not exe:
        return None, None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                             timeout=5)
        return exe, (out.stdout or out.stderr).splitlines()[0].strip()
    except Exception:
        return exe, "version unknown"


def missing_required():
    return [r["name"] for r in check() if r["required"] and not r["ok"]]


def summary_lines():
    """Short report for the application to show at startup."""
    out = []
    req = missing_required()
    if req:
        out.append(("bad", "Cannot run without: %s" % ", ".join(req)))
        return out
    engine = ocr_engine()
    st = text_reader()
    if st.get("engine") == "onnx" and st.get("error"):
        out.append(("warn", "Text reading fell back to the CPU: %s" % st["error"]))
    if engine is None:
        out.append(("warn", "No text reader available, so Read text and Verify "
                            "text will not work. Run: python scripts/get_ocr_models.py"))
    elif st.get("engine") != "onnx":
        out.append(("warn", "The text reader's own models are not in models/ocr, so "
                            "reading uses %s on the CPU - slower. "
                            "Run: python scripts/get_ocr_models.py" % engine))
    if engine == "Tesseract":
        why = ocr_fallback_reasons().get("paddle", "")
        if why and not why.startswith("not installed"):
            out.append(("warn", "PaddleOCR is installed but did not start (%s), so "
                                "text reading uses Tesseract." % why))
        else:
            out.append(("warn", "Text reading will use Tesseract. PaddleOCR is "
                                "substantially better, especially on dot-matrix "
                                "marking: pip install paddleocr paddlepaddle"))
    for r in check():
        if r["ok"] or r["required"] or r["name"] in ("PaddleOCR", "EasyOCR",
                                                     "pytesseract", "pytest"):
            continue
        out.append(("warn", "%s missing, so %s is unavailable. %s"
                    % (r["name"], r["why"], r["fix"])))
    try:
        from vision_station.core import part_finder
        if not part_finder.available():
            out.append(("warn", "The Find parts model (models/parts/parts_yoloe26s.onnx) is "
                                "missing, so Find parts (AI) is unavailable."))
    except Exception:
        pass
    if not out:
        out.append(("good", "Everything is installed."))
    return out


def main():
    print("Vision Station preflight")
    print("=" * 68)
    print("  Python   %s" % sys.version.split()[0])
    print("  Platform %s %s" % (platform.system(), platform.release()))
    from vision_station import paths
    print("  Folder   %s" % paths.ROOT)
    print()
    rows = check()
    width = max(len(r["name"]) for r in rows)
    for r in rows:
        mark = "yes" if r["ok"] else ("MISSING" if r["required"] else "no")
        print("  %-*s  %-8s %s" % (width, r["name"], mark, r["why"]))
    print()

    exe, ver = tesseract_binary()
    if exe:
        print("  Tesseract program: %s  (%s)" % (exe, ver))
    engine = ocr_engine()
    gpu = gpu_runtime()
    print("  GPU runtime: %s" % (", ".join(gpu) if gpu else
                                 "none - trained models and text reading run on the CPU"))
    st = text_reader()
    print("  Text reading will use: %s" % (engine or "nothing - no engine available"))
    if st.get("engine") == "onnx":
        print("     models: %s" % ", ".join(st.get("models") or ()))
        if st.get("error"):
            print("     %s" % st["error"])
    try:
        from vision_station.tools import deep_surface
        print("  Trained surface check: %s" % (
            "deep model available" if deep_surface.available()
            else "statistical model only (models/backbone is empty)"))
    except Exception:
        pass
    try:
        from vision_station.core import part_finder
        print("  Find parts (AI): %s" % (
            "model available (models/parts)" if part_finder.available()
            else "unavailable - models/parts/parts_yoloe26s.onnx is missing "
                 "(see scripts/make_part_model.py)"))
    except Exception:
        pass
    for name, why in ocr_fallback_reasons().items():
        print("     %s did not start: %s" % (name, why))
    print()

    problems = [r for r in rows if not r["ok"] and r["name"] != "pytest"]
    req = missing_required()
    if req:
        print("  CANNOT RUN. Install these first:")
        for name in req:
            print("     %s" % INSTALL[name])
        return 2
    if problems:
        print("  The application will run. These are optional:")
        for r in problems:
            print("     %-12s %s" % (r["name"], r["fix"]))
            print("     %-12s -> without it: %s" % ("", r["why"]))
    else:
        print("  Everything is installed.")
    print()
    print("  Start with:  run.bat   (or: python -m vision_station)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
