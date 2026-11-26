"""Model teaching run for Vision Next: Vision Station's own advanced training
(vision_station/core/advanced.py), on pictures Vision Next has just taken.

Vision Next owns the camera and the light, so it takes the pictures (the
light stepped a little brighter and darker, as Vision Station does) and starts
this with their folder. Everything after -- checking each picture is a good
part, the training tiles and moved copies, training on the graphics card in
the vision-station-train environment, testing the model, installing it in the
side's advanced/ folder -- is Vision Station's code, unchanged.

    python teach.py <side folder> <pictures folder> <control folder> [model size]

One JSON line of the session's state per second on stdout. A file named
"cancel" in the control folder cancels: the session deletes what it made.
"""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # the repository: vision_station


def say(**kw):
    print(json.dumps(kw, default=str), flush=True)


def main():
    if len(sys.argv) < 4:
        say(done=True, ok=False, error="usage: teach.py <side folder> <pictures> <control> [size]")
        return 2
    folder, pics_dir, control = sys.argv[1:4]
    size = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
    try:
        import cv2
        from vision_station.core import advanced
        from vision_station.core.engine import Project
    except Exception as e:
        say(done=True, ok=False, error="Vision Station's training code could not be loaded: %s" % e)
        return 1
    ok, why = advanced.can_train()
    if not ok:
        say(done=True, ok=False, error="Training is not possible on this PC: %s" % why)
        return 1
    pics = sorted(glob.glob(os.path.join(pics_dir, "*.png")))
    if not pics:
        say(done=True, ok=False, error="no pictures were taken")
        return 1
    queue = list(pics)

    def grab():
        return cv2.imread(queue.pop(0), cv2.IMREAD_COLOR) if queue else None

    project = Project(folder)
    s = advanced.Session(project, grab, light=None, n_pictures=len(pics), model_size=size).start()
    cancel = os.path.join(control, "cancel")
    while True:
        if os.path.exists(cancel):
            s.cancel()
        st = s.snapshot()
        st.pop("result", None) if not st.get("done") else None
        say(**st)
        if st.get("done"):
            break
        time.sleep(1.0)
    return 0 if s.state.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
