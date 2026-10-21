"""Text and codes.

`ocr` reads whatever is there. `ocv` verifies against an expected pattern and is
what you actually want on a production line — a mark either matches the expected
format or the part is rejected, and OCV is far more reliable than open-ended OCR
on laser-etched and dot-peen marks.

Engine order: PaddleOCR (best on industrial marks) -> EasyOCR -> Tesseract.
Install at least one:  pip install paddleocr paddlepaddle
"""
import re
import time
import cv2
import numpy as np
from .base import Tool, ToolResult, register

_ENGINE = {"name": None, "obj": None, "errors": {}}

# PaddleOCR 3 renamed and removed constructor arguments. This called it with
# the version 2 arguments, and version 3 rejects show_log outright -- so with
# the PaddleOCR that pip installs today, it never started, even with network,
# and every read silently fell back to Tesseract. Both are tried, newest first.
#
# On 3.x oneDNN is switched off: PaddlePaddle 3.3's oneDNN executor raises
# NotImplementedError (ConvertPirAttribute2RuntimeAttribute) on the first
# predict, after construction had succeeded -- so every text tool reported
# ERROR and every part was rejected. The document orientation and unwarping
# models are for photographed pages, not crops of a part; skipping them halves
# the start-up time.
#
# Without oneDNN the default server-size text detector took 1.2 s on a
# 640 x 160 crop, so one label of four lines cost 8 s. Finding where text is in
# a small crop does not need it: the mobile detector takes 0.17 s and finds the
# same lines. Reading is left to the full-size recogniser, which is where the
# accuracy is. Offline, without the mobile model on disk, the next entry -- the
# default models -- is used instead.
#
# Naming a model makes PaddleOCR ignore `lang`, so the fast entry is for
# English only; other languages start with the language's default models.
#
# PaddleOCR's text-line orientation model stays on. Without it, upside-down
# print reads as confident nonsense and the tools' orientation search never
# looks further. With it, a label is read correctly but the turn the tools
# record may say 0 when Paddle did the turning.
_FAST_DET = "PP-OCRv5_mobile_det"
_PADDLE_ARGS = (
    lambda lang: dict(use_textline_orientation=True, enable_mkldnn=False,
                      use_doc_orientation_classify=False, use_doc_unwarping=False,
                      text_detection_model_name=_FAST_DET) if lang == "en" else None,  # 3.x, fast
    lambda lang: dict(use_textline_orientation=True, lang=lang, enable_mkldnn=False,
                      use_doc_orientation_classify=False, use_doc_unwarping=False),   # 3.x
    lambda lang: dict(use_textline_orientation=True, lang=lang),          # early 3.x
    lambda lang: dict(use_angle_cls=True, lang=lang, show_log=False),     # 2.x
)


def _self_test(obj):
    """Read one synthetic line. An engine that constructs but cannot predict
    must not be chosen: the fallback to the next engine only happens here."""
    img = np.full((48, 200, 3), 255, np.uint8)
    cv2.putText(img, "AB12", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    _paddle_lines(obj, img)


import threading
_ENGINE_LOCK = threading.Lock()


def _engine(lang="en"):
    """The best OCR engine that actually starts. Why a better one did not is
    kept in _ENGINE["errors"], so the installation check can say so instead of
    the fallback happening silently.

    Locked: the application warms the engine up on a background thread while
    the first inspection may already need it, and two threads must not start
    two engines."""
    if _ENGINE["obj"] is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE["obj"] is not None:
            return _ENGINE
        return _start_engine(lang)


def _keep_torch_out():
    """PaddleOCR 3 imports PaddleX, which imports ModelScope, which imports
    PyTorch -- five seconds and a second Intel OpenMP runtime, PyTorch's
    libiomp5md.dll beside Paddle's own. Two OpenMP runtimes in one process
    make Intel's abort it ("OMP: Error #15"): the application closed without a
    word, about half a minute after it started, as the engine warmed up.

    ModelScope is only one of the hosts PaddleX can download a missing model
    from. A stand-in takes its place, so PyTorch is never loaded; a download
    from that one host reports that it is unavailable, and the others work.
    If PyTorch is already loaded by something else, OpenMP is told to accept
    the duplicate rather than abort."""
    import os
    import sys
    import types
    if "torch" in sys.modules:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        return
    if "modelscope" in sys.modules:
        return
    ms = types.ModuleType("modelscope")
    ms.__path__ = []                     # a package, so 'import modelscope.x' fails cleanly

    def snapshot_download(*_a, **_k):
        raise RuntimeError("downloads from ModelScope are switched off in Vision Station")
    ms.snapshot_download = snapshot_download
    sys.modules["modelscope"] = ms


def _start_engine(lang):
    errors = _ENGINE["errors"]
    try:
        _keep_torch_out()
        from paddleocr import PaddleOCR
        import warnings
        for make in _PADDLE_ARGS:
            if make(lang) is None:
                continue
            try:
                with warnings.catch_warnings():
                    # the version 2 arguments warn on version 3; trying them
                    # as a fallback is deliberate, so the warning is noise
                    warnings.simplefilter("ignore", DeprecationWarning)
                    obj = PaddleOCR(**make(lang))
                _self_test(obj)
                _ENGINE.update(name="paddle", obj=obj)
                return _ENGINE
            except Exception as e:
                # the newest API's failure is the informative one; the older
                # arguments failing on a new install is expected
                errors.setdefault("paddle", "%s: %s" % (type(e).__name__,
                                                        str(e).splitlines()[0][:160]))
    except Exception as e:
        errors["paddle"] = "not installed (%s)" % type(e).__name__
    try:
        import easyocr
        _ENGINE.update(name="easyocr", obj=easyocr.Reader([lang], gpu=False))
        return _ENGINE
    except Exception as e:
        errors["easyocr"] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160] if str(e) else "")
    try:
        import pytesseract
        _ENGINE.update(name="tesseract", obj=pytesseract)
        return _ENGINE
    except Exception as e:
        errors["tesseract"] = "not installed (%s)" % type(e).__name__
    return _ENGINE


def _paddle_lines(obj, img):
    """[(text, confidence)] from either PaddleOCR generation. Version 3 answers
    predict() with result objects carrying rec_texts and rec_scores; version 2
    answers ocr(img, cls=True) with nested [box, (text, score)] lists."""
    lines = []
    if hasattr(obj, "predict"):
        for r in obj.predict(img) or []:
            d = r
            if not hasattr(d, "get"):
                d = getattr(r, "json", None) or {}
            if "rec_texts" not in d and hasattr(d, "get") and isinstance(d.get("res"), dict):
                d = d["res"]
            texts = d.get("rec_texts") or []
            scores = d.get("rec_scores") or [0.0] * len(texts)
            lines.extend((str(t), float(c)) for t, c in zip(texts, scores) if str(t).strip())
        return lines
    for page in (obj.ocr(img, cls=True) or []):
        for det in (page or []):
            lines.append((det[1][0], float(det[1][1])))
    return lines


def _preprocess(crop, p):
    """Order matters more than any individual step here, and this order was
    chosen by measurement rather than intuition.

    Denoise first, upscale second, close third, threshold last. The previous
    pipeline upscaled then ran CLAHE with no binarisation at all, which
    amplified sensor noise into texture the recogniser read as characters: on a
    noisy sample it returned ": : OB ee" for AB12345X, a character error rate
    over 100%.

    Adaptive thresholding rather than Otsu because a global threshold gives up
    entirely under glare -- one bright patch drags the threshold past the text
    on the dim side of the part. Measured across eight degradations, this
    pipeline with a character whitelist takes mean CER from 53.9% to 6.1%."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    if p.get("denoise", True):
        g = cv2.bilateralFilter(g, 7, 50, 50)   # smooths noise, keeps stroke edges

    up = float(p.get("upscale", 2.0))
    if up != 1.0:
        g = cv2.resize(g, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)

    if p.get("clahe", False):
        g = cv2.createCLAHE(3.0, (8, 8)).apply(g)

    k = int(p.get("close_dots", 0) or 0)
    if k > 0:                        # joins dot-matrix / laser dots into strokes
        g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    if p.get("binarize", True):
        blk = int(p.get("adaptive_block", 51)) | 1
        g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, blk,
                                  float(p.get("adaptive_c", -10)))

    if p.get("invert"):
        g = cv2.bitwise_not(g)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def estimate_skew(gray, max_deg=45.0):
    """Angle of the text baseline, from the minimum-area rectangle enclosing
    the ink.

    Only the ink: thresholding first and dilating horizontally merges the
    characters of a line into one elongated mass whose long axis is the
    baseline. Measuring the whole region instead would return the angle of the
    region, which is not the same thing and is usually zero."""
    g = gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 5, 40, 40)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, -10)
    if th.mean() > 127:                 # light text on dark: invert the sense
        th = cv2.bitwise_not(th)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    pts = cv2.findNonZero(th)
    if pts is None or len(pts) < 30:
        return 0.0
    (_c, (w, h), ang) = cv2.minAreaRect(pts)
    if w < h:                            # OpenCV's angle convention flips
        ang += 90.0
    while ang > 45.0:
        ang -= 90.0
    while ang < -45.0:
        ang += 90.0
    return 0.0 if abs(ang) > max_deg else float(ang)


def _rotate_keep(img, deg):
    """Rotate about the centre, growing the canvas so nothing is cut off."""
    if abs(deg) < 0.05:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - w / 2.0
    M[1, 2] += nh / 2.0 - h / 2.0
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def read_any_orientation(crop, p, reader):
    """Read text without being told which way up it is.

    Plain OCR collapses past about five degrees of rotation: measured on
    synthetic marking, ten degrees returned nothing at all and forty-five
    degrees nothing again. Parts do not arrive aligned to the camera, so the
    tool estimates the skew, corrects it, and then tries the four right-angle
    orientations, keeping whichever reads with the highest confidence.

    The quarter turns are needed because deskew fixes the baseline angle but
    cannot tell upright text from upside-down text: both have a horizontal
    baseline.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    skew = estimate_skew(gray) if p.get("auto_rotate", True) else 0.0
    base = _rotate_keep(crop, skew) if abs(skew) > 0.2 else crop

    def attempt(img, turn):
        lines = reader(_preprocess(img, p), p)
        if not lines:
            return None, -1.0
        text = " ".join(w for w, _c in lines).strip()
        conf = float(np.mean([c for _w, c in lines]))
        # Additive, not multiplicative. Multiplying confidence by a length
        # factor collapses to zero whenever the engine reports zero confidence,
        # and Tesseract does exactly that on a noisy crop -- every orientation
        # scored 0.000, so length never broke the tie and a garbage upside-down
        # read beat a correct upright one.
        score = conf + 0.04 * min(len(text), 16)
        if turn == 0:
            score += float(p.get("upright_bias", 0.25))
        return lines, score

    def rotate(turn):
        return base if turn == 0 else cv2.rotate(
            base, {90: cv2.ROTATE_90_CLOCKWISE,
                   180: cv2.ROTATE_180,
                   270: cv2.ROTATE_90_COUNTERCLOCKWISE}[turn])

    best, best_conf, best_turn = [], -1.0, 0
    lines, score = attempt(base, 0)
    if lines:
        best, best_conf, best_turn = lines, score, 0

    # Only search the other three orientations when the upright read is poor.
    #
    # Trying all four every time made OCR 77% of the inspection cycle -- 504 ms
    # of a 652 ms total -- because it runs the recogniser four times on every
    # part. Text is upright on the overwhelming majority of parts, so the search
    # is a recovery path, not the normal one. A confident upright read is
    # accepted immediately and the other three turns are never attempted.
    good_enough = float(p.get("accept_upright_above", 0.55))
    if (p.get("try_quarter_turns", True)
            and (not lines or best_conf < good_enough + 0.25)):
        for t in (180, 90, 270):
            lines, score = attempt(rotate(t), t)
            if lines and score > best_conf:
                best, best_conf, best_turn = lines, score, t

    return best, {"skew_deg": round(skew, 2), "quarter_turn": best_turn}


def _tess_config(p):
    """Build the tesseract config, including a character whitelist.

    Restricting the alphabet is the single cheapest accuracy win available on
    industrial marking: you almost always know a lot code cannot contain
    lowercase or punctuation, and telling the recogniser removes a whole class
    of confusions."""
    cfg = p.get("tess_config") or "--psm %s" % int(p.get("psm", 7))
    wl = p.get("charset", "")
    if wl and "char_whitelist" not in cfg:
        cfg += " -c tessedit_char_whitelist=%s" % wl
    return cfg


def _only_charset(lines, p):
    """The allowed characters, for engines that take no whitelist. The
    setting reached Tesseract only: with PaddleOCR a letters-only field still
    read digits, and the control did nothing."""
    wl = p.get("charset", "")
    if not wl:
        return lines
    allowed = set(wl) | {" "}
    out = []
    for text, conf in lines:
        kept = "".join(ch for ch in text if ch in allowed).strip()
        if kept:
            out.append((kept, conf))
    return out


def _read(crop, p):
    """Preprocess, then recognise. Kept as one call for tools that do not care
    about orientation."""
    return _recognise(_preprocess(crop, p), p)


_ENGINE_CALL_LOCK = threading.Lock()


def _recognise(img, p):
    """Serialised for engines that are not safe to call from two threads at
    once (PaddleOCR, EasyOCR hold one model); Tesseract runs as a separate
    process per call and is left free, so jobs run in parallel still read
    text side by side."""
    if _engine(p.get("lang", "en"))["name"] == "tesseract":
        return _recognise_now(img, p)
    with _ENGINE_CALL_LOCK:
        return _recognise_now(img, p)


def _recognise_now(img, p):
    """Recognise text in an image that has ALREADY been preprocessed.

    Split out from _read so the orientation search can preprocess once per
    candidate rotation instead of the wrapper and the reader each doing it,
    which would upscale and threshold the same pixels twice."""
    e = _engine(p.get("lang", "en"))
    if e["obj"] is None:
        raise RuntimeError("no OCR engine installed — pip install paddleocr paddlepaddle")
    if e["name"] == "paddle":
        return _only_charset(_paddle_lines(e["obj"], img), p)
    if e["name"] == "easyocr":
        return _only_charset([(t, float(c)) for _, t, c in e["obj"].readtext(img)], p)
    # image_to_data gives per-word confidence. The previous code called
    # image_to_string and reported a hardcoded 0.5, which made min_confidence a
    # control that did nothing: set it above 0.5 and every read failed, at or
    # below and every read passed. That fake number also fed the margin report
    # in diagnostics, so a fragile OCR step looked perfectly healthy.
    cfg = _tess_config(p)
    try:
        data = e["obj"].image_to_data(img, config=cfg,
                                      output_type=e["obj"].Output.DICT)
    except Exception:
        txt = e["obj"].image_to_string(img, config=cfg).strip()
        return [(txt, 0.0)] if txt else []
    out = []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        word = (word or "").strip()
        try:
            c = float(conf)
        except (TypeError, ValueError):
            c = -1.0
        if word and c >= 0:
            out.append((word, c / 100.0))
    return out


@register("ocr")
class Ocr(Tool):
    def run(self, image, ctx):
        t0 = time.perf_counter()
        crop, _ = self.crop(image, ctx)
        try:
            lines, orient = read_any_orientation(crop, self.p, _recognise)
        except Exception as e:
            return ToolResult(self.id, self.type_name, "error", message=str(e),
                              ms=(time.perf_counter() - t0) * 1000)
        text = self.p.get("join", " ").join(t for t, _ in lines)
        conf = float(np.mean([c for _, c in lines])) if lines else 0.0
        min_conf = float(self.p.get("min_confidence", 0.5))
        ok = bool(text) and conf >= min_conf
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail", conf,
                          {"text": text, "lines": [t for t, _ in lines],
                           "confidence": round(conf, 3), "engine": _ENGINE["name"],
                           "skew_deg": orient.get("skew_deg", 0.0),
                           "quarter_turn": orient.get("quarter_turn", 0)},
                          self.roi_graphic(ctx, "pass" if ok else "fail", text or self.id),
                          "" if ok else "no text read or confidence too low",
                          (time.perf_counter() - t0) * 1000)


@register("ocv")
class Ocv(Tool):
    """Verify, don't just read. Give it a regex or an exact string."""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        crop, _ = self.crop(image, ctx)
        try:
            lines, orient = read_any_orientation(crop, self.p, _recognise)
        except Exception as e:
            return ToolResult(self.id, self.type_name, "error", message=str(e))
        text = "".join(t for t, _ in lines)
        if self.p.get("strip_spaces", True):
            text = re.sub(r"\s+", "", text)
        conf = float(np.mean([c for _, c in lines])) if lines else 0.0

        # An empty string means "not set". Testing `is not None` treated the
        # blank default from the property sheet as a demand for empty text, so
        # every OCV tool created through the UI failed every part no matter what
        # was printed on it. Only the demo recipe worked, because its JSON
        # omitted the field entirely.
        pattern = (self.p.get("expect_regex") or "").strip()
        exact = (self.p.get("expect_text") or "").strip()
        if exact:
            ok, why = text == exact, f"read '{text}', expected '{exact}'"
        elif pattern:
            ok = bool(re.fullmatch(pattern, text))
            why = f"read '{text}' does not match /{pattern}/"
        else:
            ok, why = bool(text), "no text read"
        if ok and conf < float(self.p.get("min_confidence", 0.4)):
            ok, why = False, f"low confidence {conf:.2f}"
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail", conf,
                          {"text": text, "confidence": round(conf, 3),
                           "engine": _ENGINE["name"],
                           "skew_deg": orient.get("skew_deg", 0.0),
                           "quarter_turn": orient.get("quarter_turn", 0),
                           "expected": exact or pattern},
                          self.roi_graphic(ctx, "pass" if ok else "fail", text or "no read"),
                          "" if ok else why, (time.perf_counter() - t0) * 1000)


@register("code_read")
class CodeRead(Tool):
    """DataMatrix / QR / 1D. DataMatrix is everywhere in semiconductor traceability.
    pip install pyzbar pylibdmtx"""

    def run(self, image, ctx):
        t0 = time.perf_counter()
        crop, _ = self.crop(image, ctx)
        found = []
        # A missing decoder library used to be swallowed by a bare except and
        # reported as "no code decoded", which is indistinguishable from a code
        # the camera genuinely cannot read. That sends an operator to check
        # lighting and focus for something that is actually a missing pip
        # package, so the two cases are now separated.
        available = []
        try:
            from pyzbar.pyzbar import decode as zdecode
            available.append("pyzbar")
            for d in zdecode(crop):
                found.append({"data": d.data.decode("utf-8", "replace"),
                              "symbology": d.type})
        except ImportError:
            pass
        except Exception:
            pass                      # a decode failure is a genuine no-read
        if not found and self.p.get("datamatrix", True):
            try:
                from pylibdmtx.pylibdmtx import decode as dmdecode
                available.append("pylibdmtx")
                for d in dmdecode(crop, timeout=int(self.p.get("timeout_ms", 800))):
                    found.append({"data": d.data.decode("utf-8", "replace"),
                                  "symbology": "DATAMATRIX"})
            except ImportError:
                pass
            except Exception:
                pass
        if not available:
            return ToolResult(
                self.id, self.type_name, "error", 0.0, {},
                self.roi_graphic(ctx, "error"),
                "no code reader installed - pip install pyzbar pylibdmtx",
                (time.perf_counter() - t0) * 1000)

        pattern = self.p.get("expect_regex")
        ok = bool(found) and (not pattern or bool(re.fullmatch(pattern, found[0]["data"])))
        label = found[0]["data"] if found else "no read"
        return ToolResult(self.id, self.type_name, "pass" if ok else "fail",
                          1.0 if found else 0.0,
                          {"codes": found, "data": found[0]["data"] if found else None},
                          self.roi_graphic(ctx, "pass" if ok else "fail", label),
                          "" if ok else "no code decoded or pattern mismatch",
                          (time.perf_counter() - t0) * 1000)


def line_for_recognition(gray, box, pad=3, scale=None, target_h=48):
    """One detected line of print, prepared for the recogniser: cropped with a
    margin, upscaled, smoothed, thresholded to dark print on a light ground and
    bordered. None when the box is empty.

    Both text detectors had their own copy, and the copies had drifted: only
    one turned light-on-dark print the right way round, so the classic finder
    passed white-on-black text to the recogniser inverted.

    scale: a fixed upscale factor; otherwise the line is brought to target_h
    pixels tall, since a 12 px line is below what any recogniser handles well."""
    x, y, w, h = box
    H, W = gray.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    f = scale if scale else max(1.0, target_h / float(max(1, patch.shape[0])))
    sub = cv2.resize(patch, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    sub = cv2.bilateralFilter(sub, 5, 40, 40)
    binar = cv2.adaptiveThreshold(sub, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 9)
    if binar.mean() < 110:                # recognisers want dark on light
        binar = 255 - binar
    return cv2.copyMakeBorder(binar, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
