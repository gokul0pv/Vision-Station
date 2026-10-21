"""Basler Pylon camera, with an OpenCV fallback.

Verified against pypylon 26.7's actual pylon.py — not guessed from memory. Two
things in here matter enough to explain:

1. There is no such thing as "the" InstantCamera constructor call. You attach a
   device you got from TlFactory:

       tlf = pylon.TlFactory.GetInstance()
       devices = tlf.EnumerateDevices()
       cam = pylon.InstantCamera(tlf.CreateDevice(devices[i]))

2. Exposure and gain are exposed as GenICam nodes, and the node *names* differ
   between camera families:
     - Older GigE "classic" ace cameras : ExposureTimeAbs (float, microseconds),
       GainRaw (integer, camera-specific range)
     - USB3 / ace2 / dart cameras       : ExposureTime (float, microseconds),
       Gain (float, dB)
   Code that hardcodes one set breaks silently on the other camera family — this
   is the single most common pypylon integration bug. This module checks which
   nodes exist and uses whichever is present, so the same call works on either.

Install:
    Basler Pylon SDK from baslerweb.com, then  pip install pypylon
No SDK/camera found -> falls back to OpenCV automatically.
"""

import os
import threading
import time

import cv2
import numpy as np

try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
except ImportError:
    pylon = None
    PYLON_AVAILABLE = False


class CameraError(Exception):
    pass


class CameraSettings:
    """Plain data the UI edits and the camera applies. Values are in the units
    an operator thinks in (ms, dB) regardless of which node names the camera
    uses underneath."""

    def __init__(self):
        self.auto_exposure = False
        self.exposure_ms = 10.0
        self.auto_gain = False
        self.gain_db = 0.0
        self.auto_white_balance = True
        self.pixel_format = "auto"     # "auto" | "Mono8" | "BGR8" | "RGB8"

    def copy(self):
        c = CameraSettings()
        c.__dict__.update(self.__dict__)
        return c


def list_devices():
    """Cameras Pylon can currently see, before opening any of them."""
    if not PYLON_AVAILABLE:
        return []
    out = []
    for i, d in enumerate(pylon.TlFactory.GetInstance().EnumerateDevices()):
        out.append({"index": i, "vendor": d.GetVendorName(), "model": d.GetModelName(),
                    "serial": d.GetSerialNumber()})
    return out


class PylonCamera:
    """Thin wrapper around one InstantCamera. Not thread-safe by itself —
    CameraThread below is what a UI should actually hold."""

    def __init__(self, index=0):
        self.index = index
        self.cam = None
        self.converter = None
        self.info = {}

    def open(self):
        tlf = pylon.TlFactory.GetInstance()
        devices = tlf.EnumerateDevices()
        if not devices:
            raise CameraError("No Basler camera found. Check the cable/power, and "
                              "that Pylon Viewer can see it.")
        if self.index >= len(devices):
            raise CameraError(f"Camera index {self.index} requested but only "
                              f"{len(devices)} camera(s) are connected.")
        d = devices[self.index]
        self.cam = pylon.InstantCamera(tlf.CreateDevice(d))
        self.cam.Open()
        self.info = {"vendor": d.GetVendorName(), "model": d.GetModelName(),
                     "serial": d.GetSerialNumber()}

        # Convert whatever the sensor gives us (often Bayer or Mono) to BGR8
        # once, here, instead of hand-rolling debayering per frame.
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        return self.info

    # -- node-name compatibility across camera families ------------------
    def _first_node(self, names):
        for n in names:
            if self.cam.GetNodeMap().GetNode(n) is not None:
                node = getattr(self.cam, n, None)
                if node is not None and node.IsWritable():
                    return node
        return None

    def apply_settings(self, s: CameraSettings):
        if not self.cam or not self.cam.IsOpen():
            return
        try:
            self._apply_exposure(s)
        except Exception as e:
            print(f"[camera] exposure not applied: {e}")
        try:
            self._apply_gain(s)
        except Exception as e:
            print(f"[camera] gain not applied: {e}")
        try:
            self._apply_white_balance(s)
        except Exception as e:
            print(f"[camera] white balance not applied: {e}")

    def _apply_exposure(self, s):
        auto = self._first_node(["ExposureAuto"])
        if auto is not None:
            auto.SetValue("Continuous" if s.auto_exposure else "Off")
        if s.auto_exposure:
            return
        node = self._first_node(["ExposureTime", "ExposureTimeAbs"])
        if node is None:
            raise CameraError("no writable exposure node on this camera")
        lo, hi = node.GetMin(), node.GetMax()
        node.SetValue(min(max(s.exposure_ms * 1000.0, lo), hi))   # ms -> µs

    def _apply_gain(self, s):
        """Gain is not one quantity across Basler's range.

        USB3, ace2 and dart cameras expose `Gain` in decibels. Older GigE ace
        cameras expose `GainRaw`, an integer in sensor-specific units whose
        range is typically 0-512 and which is emphatically not decibels. Writing
        a decibel figure straight into GainRaw, as this used to, meant a request
        for 3 dB became a raw value of 3 -- essentially no gain at all -- while
        the dialog cheerfully reported dB. Some models also reject a float on an
        integer node outright.

        So the unit is discovered, the value interpreted in that unit, and
        integer nodes are given integers. `gain_unit()` lets the UI label the
        field with what the camera actually wants."""
        auto = self._first_node(["GainAuto"])
        if auto is not None:
            auto.SetValue("Continuous" if s.auto_gain else "Off")
        if s.auto_gain:
            return
        node, name = self._first_node_named(["Gain", "GainAbs", "GainRaw"])
        if node is None:
            raise CameraError("no writable gain node on this camera")
        lo, hi = node.GetMin(), node.GetMax()
        val = min(max(float(s.gain_db), float(lo)), float(hi))
        if name == "GainRaw":
            val = int(round(val))
        node.SetValue(val)

    def _first_node_named(self, names):
        for n in names:
            if self.cam.GetNodeMap().GetNode(n) is not None:
                node = getattr(self.cam, n, None)
                if node is not None and node.IsWritable():
                    return node, n
        return None, None

    def gain_unit(self):
        """('dB'|'raw'|None, low, high) so the dialog can label the control
        with the unit this particular camera actually uses."""
        node, name = self._first_node_named(["Gain", "GainAbs", "GainRaw"])
        if node is None:
            return None, None, None
        unit = "raw" if name == "GainRaw" else "dB"
        return unit, node.GetMin(), node.GetMax()

    def _apply_white_balance(self, s):
        auto = self._first_node(["BalanceWhiteAuto"])
        if auto is not None:
            auto.SetValue("Continuous" if s.auto_white_balance else "Off")

    def exposure_range_us(self):
        node = self._first_node(["ExposureTime", "ExposureTimeAbs"])
        return (node.GetMin(), node.GetMax()) if node else (None, None)

    def gain_range(self):
        node, _ = self._first_node_named(["Gain", "GainAbs", "GainRaw"])
        return (node.GetMin(), node.GetMax()) if node else (None, None)

    # -- grabbing ----------------------------------------------------------
    def start(self):
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def stop(self):
        if self.cam and self.cam.IsGrabbing():
            self.cam.StopGrabbing()

    def grab(self, timeout_ms=500):
        """One frame as a BGR uint8 numpy array, or None on a soft timeout.
        Raises CameraError on a hard failure (device unplugged, etc)."""
        if not self.cam.IsGrabbing():
            return None
        try:
            res = self.cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
        except Exception as e:
            raise CameraError(f"grab failed: {e}")
        try:
            if not res.GrabSucceeded():
                return None
            converted = self.converter.Convert(res)
            arr = converted.GetArray()
            return np.ascontiguousarray(arr)
        finally:
            res.Release()

    def close(self):
        try:
            self.stop()
        except Exception:
            pass
        if self.cam:
            try:
                self.cam.Close()
            except Exception:
                pass
        self.cam = None


class CameraThread(threading.Thread):
    """What the UI actually talks to. Runs Pylon (or OpenCV, transparently) in
    the background and hands back whatever the newest frame is — this is a
    "latest frame wins" model, appropriate for live inspection, not a queue you
    have to drain."""

    def __init__(self, index=0, settings=None, prefer_pylon=True):
        super().__init__(daemon=True)
        self.index = index
        self.settings = settings or CameraSettings()
        self.prefer_pylon = prefer_pylon and PYLON_AVAILABLE
        self.alive = False
        self.backend = None            # set once we know: "pylon" | "opencv"
        self.info = {}
        self.last_error = None
        self._frame = None
        self._lock = threading.Lock()
        self._pylon_cam = None
        self._settings_dirty = False

    def run(self):
        if self.prefer_pylon:
            try:
                self._run_pylon()
                return
            except CameraError as e:
                self.last_error = str(e)
                print(f"[camera] Pylon path failed ({e}); falling back to OpenCV")
        self._run_opencv()

    def _run_pylon(self):
        self._pylon_cam = PylonCamera(self.index)
        self.info = self._pylon_cam.open()
        self.backend = "pylon"
        self._pylon_cam.apply_settings(self.settings)
        self._pylon_cam.start()
        self.alive = True
        while self.alive:
            if self._settings_dirty:
                self._pylon_cam.apply_settings(self.settings)
                self._settings_dirty = False
            try:
                frame = self._pylon_cam.grab(500)
            except CameraError as e:
                self.last_error = str(e)
                time.sleep(0.2)
                continue
            if frame is not None:
                with self._lock:
                    self._frame = frame
        self._pylon_cam.close()

    def _run_opencv(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW if os.name == "nt" else 0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.last_error = (f"No camera at index {self.index}. Check the index, "
                               "or that nothing else has it open.")
            return
        self.backend = "opencv"
        self.info = {"vendor": "", "model": f"OpenCV device {self.index}", "serial": ""}
        self.alive = True
        while self.alive:
            ok, frame = cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.02)
        cap.release()

    def update_settings(self, settings):
        """Safe to call from the UI thread while running — applied on the
        camera's own thread on the next loop iteration, not here, since Pylon
        node access from two threads at once is not guaranteed safe."""
        self.settings = settings
        self._settings_dirty = True

    def get_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def grab_sequence(self, captures, settle=3, timeout_s=3.0):
        """One frame per capture, each at its own exposure.

        A switch faceplate needs two: the LEDs clip at any exposure long enough
        to show the panel. Returns {capture_id: frame}.

        `settle` frames are discarded after each exposure change. A camera in
        free-running acquisition has one or two frames already in flight that
        were captured with the previous setting, and using those would silently
        give the new capture the old exposure -- the kind of fault that shows up
        as an intermittent false reject rather than an error."""
        out = {}
        base = self.settings
        for cap in captures:
            ms = cap.get("exposure_ms")
            if ms is not None:
                s = base.copy()
                s.auto_exposure = False
                s.exposure_ms = float(ms)
                if cap.get("gain") is not None:
                    s.auto_gain = False
                    s.gain_db = float(cap["gain"])
                self.update_settings(s)
            frame, deadline = None, time.time() + timeout_s
            seen = 0
            while time.time() < deadline:
                f = self.get_frame()
                if f is not None:
                    seen += 1
                    frame = f
                    if seen >= max(1, settle):
                        break
                time.sleep(0.02)
            if frame is not None:
                out[cap.get("id", "default")] = frame
        self.update_settings(base)
        return out

    def stop(self):
        self.alive = False
        self.join(timeout=3.0)
