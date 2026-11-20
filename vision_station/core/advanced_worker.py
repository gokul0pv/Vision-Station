r"""Advanced training's worker: trains one job's YOLO detector and exports it.

Run by core/advanced.py with the TRAINING environment's Python (PyTorch with
CUDA and Ultralytics, kept outside the application -- see
scripts/train_parts.py), never imported by the application. It imports
nothing of Vision Station, so it runs in that environment as it is.

    python -u advanced_worker.py --data D\data.yaml --out D\runs --model yolo11s.pt
                                 --epochs 30 --imgsz 1280 --batch 4

Everything it has to say is one JSON object per line on stdout:

    {"t": "start", "epochs": 30, "batches": 75, "device": "NVIDIA GeForce RTX 5060"}
    {"t": "batch", "epoch": 3, "batch": 40, "batches": 75}
    {"t": "epoch", "epoch": 3, "epochs": 30, "metrics": {"precision": .., "recall": ..,
                                                         "map50": .., "map": ..}}
    {"t": "export", "path": "...best.onnx"}
    {"t": "done", "metrics": {...}, "onnx": "...", "seconds": 312.4}
    {"t": "error", "message": "...", "oom": false}

Ultralytics' own output goes to stderr, kept by the caller for the message
when something goes wrong.
"""
import argparse
import json
import os
import sys
import time
import traceback


def say(**kw):
    sys.stdout.write(json.dumps(kw) + "\n")
    sys.stdout.flush()


def metrics_of(m):
    m = m or {}

    def get(*keys):
        for k in keys:
            if k in m:
                try:
                    return round(float(m[k]), 4)
                except (TypeError, ValueError):
                    pass
        return None
    return {"precision": get("metrics/precision(B)", "precision"),
            "recall": get("metrics/recall(B)", "recall"),
            "map50": get("metrics/mAP50(B)", "map50"),
            "map": get("metrics/mAP50-95(B)", "map")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    t0 = time.time()
    try:
        import torch
        from ultralytics import YOLO
    except Exception as e:
        say(t="error", message="the training environment is not complete: %s" % e, oom=False)
        return 2
    if a.device != "cpu" and not torch.cuda.is_available():
        say(t="error", message="PyTorch in the training environment cannot use the graphics "
                               "card (CUDA not available)", oom=False)
        return 2
    os.makedirs(a.out, exist_ok=True)
    os.chdir(a.out)                     # starting weights download here, if they must
    state = {"batch": 0, "batches": 0, "epoch": 0}
    model = YOLO(a.model)

    def on_train_start(tr):
        try:
            state["batches"] = len(tr.train_loader)
        except Exception:
            state["batches"] = 0
        dev = "CPU"
        if a.device != "cpu":
            try:
                dev = torch.cuda.get_device_name(0)
            except Exception:
                dev = "GPU"
        say(t="start", epochs=int(tr.epochs), batches=state["batches"], device=dev)

    def on_epoch_start(tr):
        state["batch"] = 0
        state["epoch"] = int(tr.epoch) + 1

    def on_batch_end(tr):
        state["batch"] += 1
        # every batch is too chatty for a slow pipe: every 5th, and the last
        if state["batch"] % 5 == 0 or state["batch"] == state["batches"]:
            say(t="batch", epoch=state["epoch"], batch=state["batch"], batches=state["batches"])

    def on_fit_epoch_end(tr):
        say(t="epoch", epoch=int(tr.epoch) + 1, epochs=int(tr.epochs),
            metrics=metrics_of(getattr(tr, "metrics", None)))

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_batch_end", on_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    try:
        model.train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch,
                    device=a.device if a.device == "cpu" else int(a.device),
                    workers=2, project=os.path.abspath(a.out), name="train", exist_ok=True,
                    patience=a.patience, fliplr=0.0, mosaic=0.5, degrees=2.0,
                    translate=0.05, scale=0.2, hsv_v=0.3, hsv_s=0.4,
                    close_mosaic=max(1, min(10, a.epochs // 3)), seed=0,
                    deterministic=True, verbose=False, plots=False, amp=True)
        tr = model.trainer
        best = os.path.join(str(tr.save_dir), "weights", "best.pt")
        if not os.path.isfile(best):
            best = os.path.join(str(tr.save_dir), "weights", "last.pt")
        final = metrics_of(getattr(tr, "metrics", None))
        say(t="export", path="")
        # not simplified: onnxslim took a minute here for the same model at run time
        onnx = YOLO(best).export(format="onnx", imgsz=a.imgsz, opset=17, dynamic=False,
                                 simplify=False, nms=False)
        say(t="done", metrics=final, onnx=str(onnx), seconds=round(time.time() - t0, 1))
        return 0
    except Exception as e:
        msg = "%s: %s" % (type(e).__name__, e)
        oom = "out of memory" in msg.lower()
        sys.stderr.write(traceback.format_exc())
        say(t="error", message=msg[:500], oom=oom)
        return 1


if __name__ == "__main__":
    sys.exit(main())
