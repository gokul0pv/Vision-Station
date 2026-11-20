r"""python -m vision_station  -- start the application.

    python -m vision_station                   the application
    python -m vision_station D:\projects       ... with its projects elsewhere
    python -m vision_station --classic         the classic single-window editor
"""
import os
import sys

if not __package__:
    # run as a file (python vision_station/__main__.py, VS Code's Run
    # button) rather than with -m: the package is found from the project root
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check():
    """--check: a report on this installation in logs/check.txt -- what is
    installed, whether the GPU is used, and a real text read, part search
    and camera-driver load. For the packaged application, which has no
    console to print to."""
    import contextlib
    import io
    import traceback
    from vision_station import paths
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            from vision_station.core import preflight
            preflight.main()
        except SystemExit:
            pass
        except Exception:
            traceback.print_exc()
        print("\nThis PC")
        try:
            from vision_station.core import system_profile
            prof, p = system_profile.current()
            print("  " + system_profile.describe(prof, p).replace("\n", "\n  "))
        except Exception:
            traceback.print_exc()
        print("\nEngines")
        try:
            import cv2
            from vision_station.core import inference, part_finder
            print("  GPU providers: %s" % (inference.ort_gpu_providers() or "none"))
            demo = os.path.join(paths.demo_source(), "reference.png")
            img = cv2.imread(demo) if os.path.exists(demo) else None
            if img is None:
                print("  demo picture missing: %s" % demo)
            else:
                from vision_station.tools import ocr_engine
                lines = ocr_engine.read(img)
                print("  text read on the demo: %s  (%s)" % (
                    [x.get("text") if isinstance(x, dict) else getattr(x, "text", x)
                     for x in lines][:5] if isinstance(lines, list) else lines,
                    ocr_engine.status().get("device")))
                r = part_finder.find_parts(img, None, 0.5)
                print("  part finder: %d parts in %.0f ms on %s" % (len(r["parts"]), r["ms"],
                                                                  r["device"]))
        except Exception:
            traceback.print_exc()
        try:
            check_camera()
        except Exception:
            traceback.print_exc()
    os.makedirs(paths.LOGS, exist_ok=True)
    out = os.path.join(paths.LOGS, "check.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    return 0


def check_camera():
    """The camera driver, where its libraries came from, and -- when an
    address is set under Settings > Camera -- a real open and one frame."""
    import time
    from vision_station.hardware import camera
    if not camera.PYLON_AVAILABLE:
        print("  camera driver (pypylon): NOT LOADED - %s" % camera.PYLON_ERROR)
        return
    from pypylon import pylon
    tls = [t.GetDeviceClass() for t in pylon.TlFactory.GetInstance().EnumerateTls()]
    print("  camera driver (pypylon): loaded from %s" % os.path.dirname(pylon.__file__))
    print("  transport layers: %s" % ", ".join(tls))
    address = ""
    try:
        from PySide6.QtCore import QSettings
        from vision_station.ui.settings_dialog import load_camera_settings
        address = load_camera_settings(QSettings("VisionStation", "QuickBuild")).address
    except Exception as e:
        print("  camera settings not read: %s" % e)
    if getattr(sys, "frozen", False):          # python.exe has Windows' own rule
        from vision_station.hardware import firewall
        print("  firewall rule for %s: %s" % (os.path.basename(firewall.program()), {
            True: "yes", False: "NO - GigE pictures are blocked (Settings > Camera > Allow)",
            None: "unknown"}[firewall.has_rule()]))
    found = camera.list_devices()
    print("  cameras discovery finds: %s" % (["%s %s" % (d["model"], d["ip"]) for d in found]
                                             or "none"))
    if not (address or found):
        print("  no camera address set and none found: nothing opened")
        return
    cam = camera.PylonCamera(0, address)
    t0 = time.perf_counter()
    try:
        info = cam.open()
        cam.start()
        frame = cam.grab(5000)
        print("  camera %s opened: %s %s, frame %s in %.1f s" % (
            address or "0", (info or cam.info).get("model"), (info or cam.info).get("serial"),
            "x".join(map(str, frame.shape[1::-1])) if frame is not None else "NOT grabbed",
            time.perf_counter() - t0))
        if frame is None:
            print("    grab: %s" % (cam.last_grab_error or "no frame arrived"))
    except Exception as e:
        print("  camera %s did NOT open: %s" % (address or "0", e))
    finally:
        try:
            cam.close()
        except Exception:
            pass


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    try:
        # GPU or CPU from this PC's graphics card, unless set by hand
        from vision_station.core import system_profile
        system_profile.apply_device()
    except Exception as e:
        print("[startup] PC not profiled: %s" % e)
    if "--check" in argv[1:]:
        return check()
    if "--classic" in argv[1:]:
        from vision_station.ui.classic_window import main as classic
        return classic()
    from vision_station.ui.main_window import main as app
    return app(argv)


if __name__ == "__main__":
    sys.exit(main())



