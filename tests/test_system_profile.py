"""Models and settings chosen from the PC: a line PC with integrated
graphics, a PC with an AMD card, NVIDIA cards of several sizes.

    python -m pytest tests/test_system_profile.py -q
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from vision_station.core import system_profile as S          # noqa: E402

DML = ["DmlExecutionProvider"]


def pc(gpus, runtime=DML, cores=8, mem=16):
    return {"cpu": "test", "cores": cores, "memory_gb": mem, "gpus": gpus,
            "gpu_runtime": runtime}


def nvidia(name, vram, driver="610.88"):
    return {"name": name, "vendor": "NVIDIA", "vram_gb": vram, "driver": driver,
            "cuda128": float(driver.split(".")[0]) >= 570}


def test_the_line_pc_rtx_5060():
    p = S.plan(pc([nvidia("NVIDIA GeForce RTX 5060", 8.0)], cores=28, mem=32))
    assert p["device"] == "GPU" and p["ocr_models"] == "medium"
    assert p["train"]["possible"] and p["train"]["model"] == "yolo11s.pt"
    assert p["train"]["batch"] == 4 and not p["warnings"]


def test_integrated_graphics_stay_on_the_processor():
    p = S.plan(pc([{"name": "Intel(R) UHD Graphics 730", "vendor": "Intel", "vram_gb": 0.1,
                    "cuda128": False}], mem=8))
    assert p["device"] == "CPU" and p["ocr_models"] == "small"
    assert not p["train"]["possible"] and "NVIDIA" in p["train"]["why"]
    assert any("16 GB" in w for w in p["warnings"])


def test_no_gpu_runtime_means_the_processor():
    p = S.plan(pc([nvidia("NVIDIA GeForce RTX 3060", 12.0)], runtime=[]))
    assert p["device"] == "CPU", "told to use a GPU it has no runtime for"
    assert p["train"]["possible"], "training needs the card, not the inference runtime"


def test_an_amd_card_runs_networks_but_does_not_train():
    p = S.plan(pc([{"name": "AMD Radeon RX 6600", "vendor": "AMD", "vram_gb": 8.0,
                    "cuda128": False}]))
    assert p["device"] == "GPU" and not p["train"]["possible"]


def test_small_and_old_nvidia_cards():
    p = S.plan(pc([nvidia("NVIDIA GeForce GTX 1650", 4.0)]))
    assert p["train"]["possible"] and p["train"]["model"] == "yolo11n.pt"
    assert p["train"]["batch"] == 1
    p = S.plan(pc([nvidia("NVIDIA GeForce RTX 2060", 6.0)]))
    assert (p["train"]["model"], p["train"]["batch"]) == ("yolo11n.pt", 2)
    p = S.plan(pc([nvidia("NVIDIA GeForce GT 1030", 2.0)]))
    assert not p["train"]["possible"] and "4 GB" in p["train"]["why"]
    p = S.plan(pc([nvidia("NVIDIA GeForce RTX 3070", 8.0, driver="537.13")]))
    assert not p["train"]["possible"] and "R570" in p["train"]["why"], "an old driver trained"


def test_a_device_set_by_hand_wins(monkeypatch):
    monkeypatch.setenv("VS_DEVICE", "CPU")
    monkeypatch.setattr(S, "current", lambda: (None, {"device": "GPU"}))
    assert S.apply_device() == "CPU"
    monkeypatch.delenv("VS_DEVICE")
    assert S.apply_device() == "GPU" and os.environ["VS_DEVICE"] == "GPU"


def test_the_description_says_what_was_chosen():
    prof = pc([nvidia("NVIDIA GeForce RTX 5060", 8.0)])
    text = S.describe(prof)
    assert "RTX 5060" in text and "Networks run on: GPU" in text and "yolo11s" in text


def test_on_the_processor_the_smallest_text_models_come_first(monkeypatch, tmp_path):
    from vision_station.tools import ocr_engine as E
    for n in ("det_medium.onnx", "det_small.onnx"):
        (tmp_path / n).write_bytes(b"x")
    monkeypatch.setattr(E, "MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("VS_DEVICE", "GPU")
    assert E._first(E.DET_FILES).endswith("det_medium.onnx")
    monkeypatch.setenv("VS_DEVICE", "CPU")
    assert E._first(E.DET_FILES).endswith("det_small.onnx")
