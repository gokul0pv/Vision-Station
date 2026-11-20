"""What this PC can run, and the models and settings chosen for it.

The same application goes on PCs from a line PC with integrated graphics to
an engineering PC with an RTX 5060. Rather than one fixed choice, the PC is
looked at once -- processor, memory, graphics card and its memory -- and
everything that depends on it follows from that: where the networks run,
which text-reading models, whether a part detector can be trained here and
at what size.

    profile = detect()        what the PC has
    p = plan(profile)         what it gets
    print(describe(profile, p))

Minimum and recommended (measured on the line's RTX 5060, i7 PC, and the
models' own sizes; the camera is 20 megapixels):

    everything, on the CPU     Windows 10/11 64-bit, 4 cores, 8 GB memory
                               (16 GB recommended: a 20 MP picture is 60 MB,
                               and a run holds several). Text reading and Find
                               parts work, several times slower.
    networks on the GPU        any DirectX 12 graphics card with 2 GB of its
                               own memory or more (onnxruntime-directml);
                               4 GB or more recommended.
    training a part detector   an NVIDIA card with 4 GB or more and a driver
                               for CUDA 12.8 (R570 or newer; RTX 50 cards need
                               it); 16 GB memory. 4-6 GB: the nano model at a
                               smaller batch; 8 GB or more: the small model.
                               Without one, train on such a PC and copy
                               models/parts/parts_trained.onnx across.
"""
import json
import os
import subprocess
import sys

GB = 1024 ** 3


# ---------------------------------------------------------------- looking
def _quiet():
    if sys.platform != "win32":
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return {"startupinfo": si, "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


def _run(cmd, timeout=15):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **_quiet())
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""


def _memory_gb():
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return round(m.ullTotalPhys / GB, 1)
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GB, 1)
    except (ValueError, OSError, AttributeError):
        return 0.0


def _cpu_name():
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"])
        if out.strip():
            return out.strip()
    import platform
    return platform.processor() or "unknown"


def _gpus():
    """[{name, vendor, vram_gb, cuda}] -- NVIDIA from nvidia-smi (exact
    memory, driver), every card from the display adapters' registry entry
    (its memory as a 64-bit value: Win32_VideoController stops at 4 GB)."""
    out = []
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits"])
    for line in smi.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                drv = float(parts[2].split(".")[0])
            except ValueError:
                drv = 0.0
            out.append({"name": parts[0], "vendor": "NVIDIA",
                        "vram_gb": round(float(parts[1]) / 1024, 1),
                        "driver": parts[2], "cuda128": drv >= 570})
    if sys.platform == "win32":
        ps = ("Get-ItemProperty 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\"
              "{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' -ErrorAction SilentlyContinue | "
              "ForEach-Object { [pscustomobject]@{ name = $_.DriverDesc; "
              "mem = $_.'HardwareInformation.qwMemorySize' } } | ConvertTo-Json -Compress")
        raw = _run(["powershell", "-NoProfile", "-Command", ps])
        try:
            rows = json.loads(raw) if raw.strip() else []
        except ValueError:
            rows = []
        if isinstance(rows, dict):
            rows = [rows]
        seen = {g["name"] for g in out}
        for r in rows:
            name = (r.get("name") or "").strip()
            if not name or name in seen or "basic" in name.lower() or "remote" in name.lower():
                continue
            mem = r.get("mem")
            vram = round(float(mem) / GB, 1) if isinstance(mem, (int, float)) else 0.0
            vendor = ("NVIDIA" if "nvidia" in name.lower() else
                      "AMD" if ("amd" in name.lower() or "radeon" in name.lower()) else
                      "Intel" if "intel" in name.lower() else "other")
            out.append({"name": name, "vendor": vendor, "vram_gb": vram, "cuda128": False})
            seen.add(name)
    return out


def _gpu_runtime():
    try:
        from vision_station.core import inference
        return inference.ort_gpu_providers()
    except Exception:
        return []


def detect():
    """What this PC has."""
    return {"cpu": _cpu_name(), "cores": os.cpu_count() or 1, "memory_gb": _memory_gb(),
            "gpus": _gpus(), "gpu_runtime": _gpu_runtime()}


# ---------------------------------------------------------------- choosing
MIN_VRAM_INFER = 2.0          # GB of the card's own memory to run networks on it
MIN_VRAM_TRAIN = 4.0          # ... to train a part detector on it


def plan(profile):
    """What this PC gets: device, text models, part model, training."""
    gpus = sorted(profile.get("gpus") or [], key=lambda g: -g.get("vram_gb", 0))
    best = gpus[0] if gpus else None
    runtime = profile.get("gpu_runtime") or []
    # integrated graphics report their shared memory as small or nothing:
    # under MIN_VRAM_INFER the CPU is the safer place
    gpu_ok = bool(runtime) and best is not None and best.get("vram_gb", 0) >= MIN_VRAM_INFER
    device = "GPU" if gpu_ok else "CPU"

    nv = next((g for g in gpus if g["vendor"] == "NVIDIA"), None)
    train = {"possible": False, "why": "", "model": None, "batch": 0, "imgsz": 1280}
    if nv is None:
        train["why"] = "no NVIDIA graphics card: train on a PC with one and copy the model"
    elif nv.get("vram_gb", 0) < MIN_VRAM_TRAIN:
        train["why"] = "%s has %.0f GB; training needs %.0f GB or more" % (
            nv["name"], nv["vram_gb"], MIN_VRAM_TRAIN)
    elif not nv.get("cuda128"):
        train["why"] = "the NVIDIA driver (%s) is older than R570 (CUDA 12.8): update it" % (
            nv.get("driver") or "?")
    else:
        v = nv["vram_gb"]
        train.update(possible=True, why="on %s (%.0f GB)" % (nv["name"], v),
                     model="yolo11s.pt" if v >= 7.5 else "yolo11n.pt",
                     batch=4 if v >= 7.5 else 2 if v >= 5.5 else 1)

    mem = profile.get("memory_gb", 0)
    return {
        "device": device,
        # the larger text-reading models only where they are worth their time
        "ocr_models": "medium" if device == "GPU" and best and best.get("vram_gb", 0) >= 6
        else "small",
        "parallel_jobs": device == "CPU" and profile.get("cores", 1) >= 8 and mem >= 16,
        "train": train,
        "warnings": [w for w in (
            "%.0f GB of memory: 16 GB is recommended for 20-megapixel pictures" % mem
            if 0 < mem < 15 else "",
            "no GPU runtime for the networks: text reading and Find parts run on the "
            "processor (slower)" if device == "CPU" else "") if w],
    }


def describe(profile, p=None):
    p = p or plan(profile)
    lines = ["Processor: %s, %d threads, %.0f GB memory" % (
        profile.get("cpu", "?"), profile.get("cores", 0), profile.get("memory_gb", 0))]
    for g in profile.get("gpus") or []:
        lines.append("Graphics: %s, %.0f GB%s" % (g["name"], g.get("vram_gb", 0),
                                                 (", driver %s" % g["driver"]) if g.get("driver")
                                                 else ""))
    if not profile.get("gpus"):
        lines.append("Graphics: none found")
    lines.append("Networks run on: %s%s" % (p["device"], (" (%s)" % ", ".join(
        profile.get("gpu_runtime") or [])) if p["device"] == "GPU" else ""))
    lines.append("Text-reading models: %s" % p["ocr_models"])
    t = p["train"]
    lines.append("Part detector training: %s" % (
        "yes - %s, batch %d, %s" % (t["model"].replace(".pt", ""), t["batch"], t["why"])
        if t["possible"] else "no - " + t["why"]))
    lines += ["Note: " + w for w in p["warnings"]]
    return "\n".join(lines)


# ---------------------------------------------------------------- applying
_CACHE = {}


def current():
    """(profile, plan) for this PC, looked at once per run."""
    if "p" not in _CACHE:
        prof = detect()
        _CACHE["p"] = (prof, plan(prof))
    return _CACHE["p"]


def apply_device():
    """VS_DEVICE from the plan, unless it was set by hand (run.bat, the
    environment) -- a line PC whose integrated graphics cannot carry the
    networks is put on the CPU instead of being told to use a GPU it does
    not have."""
    if os.environ.get("VS_DEVICE", "").strip():
        return os.environ["VS_DEVICE"]
    _prof, p = current()
    os.environ["VS_DEVICE"] = p["device"]
    return p["device"]


if __name__ == "__main__":
    prof = detect()
    print(describe(prof))
