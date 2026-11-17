"""Basler Pylon camera, opened and set up by this application, with an OpenCV
fallback.

Verified against pypylon's actual pylon.py and on the user's acA5472-5gc -- not
guessed from memory. Things in here that matter enough to explain:

1. A camera is opened by its ADDRESS when one is set (Settings > Camera): an
   IP address for a GigE camera, or a serial number. Opening by IP talks to
   the camera directly (unicast), so it works when GigE discovery does not --
   on this PC discovery's broadcast finds nothing while the camera answers at
   192.168.3.6, and "camera 0" then meant "no camera". With no address the
   first camera discovery finds is used, as before.

2. Exposure and gain are GenICam nodes whose NAMES differ between families:
     - Older GigE "classic" ace : ExposureTimeAbs (us), GainRaw (integer units)
     - USB3 / ace2 / dart       : ExposureTime (us), Gain (dB)
   Every setting here looks for whichever node the camera has.

3. A GigE camera admits ONE controlling connection. "The device is controlled
   by another application" was most often this application itself: a camera
   thread stopped while it was still opening (opening takes 1.5 s) set
   alive=False first, the thread then set alive=True after Open() and ran on
   for ever, holding the camera, and every later open was refused. Now a stop
   request is an Event checked after opening, and opens of one camera inside
   this process wait for the previous holder to let go. A connection left by a
   program that crashed is released by the camera after its heartbeat timeout
   (3 s by default), so a refused open is retried for a while before giving
   up -- and when it does give up, the message names the program holding it if
   one can be found.

Install:
    Basler pylon SDK from baslerweb.com, then  pip install pypylon
No SDK/camera found -> falls back to OpenCV automatically.
"""

import os
import queue
import re
import threading
import time

import cv2
import numpy as np

try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
except Exception:
    # Not only ImportError: a pypylon whose SDK libraries are missing or
    # mismatched fails part-way through its own import with AttributeError or
    # OSError, and the application then did not start at all instead of
    # falling back to an ordinary camera.
    pylon = None
    PYLON_AVAILABLE = False


class CameraError(Exception):
    pass


_IP = re.compile(r"^\s*(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\s*$")


def is_ip(text):
    m = _IP.match(text or "")
    return bool(m) and all(0 <= int(g) <= 255 for g in m.groups())


class CameraSettings:
    """Plain data the UI edits and the camera applies. Values are in the units
    an operator thinks in (ms, the camera's gain unit) regardless of which node
    names the camera uses underneath. A value of 0 / "" / None for an optional
    setting means "leave the camera's own value"."""

    # Settings the camera only accepts while it is not grabbing: changing one
    # stops acquisition, applies everything, and starts it again.
    STRUCTURAL = ("address", "pixel_format", "roi", "reverse_x", "reverse_y", "rotation",
                  "packet_size", "trigger", "trigger_source", "trigger_edge",
                  "heartbeat_ms")

    def __init__(self):
        self.address = ""              # "" = first camera found; IP or serial
        self.auto_exposure = False
        self.exposure_ms = 10.0
        self.auto_gain = False
        self.gain_db = 0.0             # in the camera's own unit (dB or raw)
        self.auto_white_balance = True
        self.pixel_format = "auto"     # "auto" | a format the camera lists
        self.gamma = None              # None = leave; 1.0 = off (linear)
        self.black_level = None        # None = leave
        self.frame_rate = 0.0          # frames/s cap; 0 = as fast as possible
        self.trigger = "free"          # "free" (continuous) | "hardware"
        self.trigger_source = "Line1"
        self.trigger_edge = "RisingEdge"
        self.trigger_delay_us = 0.0
        self.roi = None                # None = leave; "full" = whole sensor; [x, y, w, h]
        self.reverse_x = False
        self.reverse_y = False
        # Turn the picture clockwise: 0, 90, 180 or 270 degrees -- for a camera
        # mounted upside down (180) or on its side (90 / 270). 180 is done on
        # the sensor (both mirror axes, no cost); 90 and 270 in software.
        self.rotation = 0
        self.packet_size = 0           # GigE packet size (bytes); 0 = leave
        self.inter_packet_delay = 0    # GigE inter-packet delay (ticks)
        self.heartbeat_ms = 0          # GigE heartbeat; 0 = leave (3000)

    def copy(self):
        c = CameraSettings()
        c.__dict__.update(self.__dict__)
        if isinstance(self.roi, (list, tuple)):
            c.roi = list(self.roi)
        return c

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d):
        s = cls()
        for k, v in (d or {}).items():
            if k in s.__dict__:
                setattr(s, k, v)
        return s

    def structural_key(self):
        return tuple(repr(getattr(self, k, None)) for k in self.STRUCTURAL)


def turn_code(rotation, done_on_sensor=False):
    """The cv2.rotate code for a clockwise turn still to be done in software,
    or None."""
    rot = int(rotation or 0) % 360
    if rot == 90:
        return cv2.ROTATE_90_CLOCKWISE
    if rot == 270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    if rot == 180 and not done_on_sensor:
        return cv2.ROTATE_180
    return None


def list_devices():
    """Cameras pylon's discovery can see, before opening any of them. A GigE
    camera discovery misses can still be opened by its IP address."""
    if not PYLON_AVAILABLE:
        return []
    out = []
    for i, d in enumerate(pylon.TlFactory.GetInstance().EnumerateDevices()):
        ip = ""
        try:
            if d.IsIpAddressAvailable():
                ip = d.GetIpAddress()
        except Exception:
            pass
        out.append({"index": i, "vendor": d.GetVendorName(), "model": d.GetModelName(),
                    "serial": d.GetSerialNumber(), "ip": ip})
    return out


def probe_address(address, timeout_s=10.0):
    """Can a camera be reached at this address? {model, serial, ip} or raises
    CameraError. Opens nothing: the device is created, read and dropped."""
    if not PYLON_AVAILABLE:
        raise CameraError("pypylon is not installed")
    tlf = pylon.TlFactory.GetInstance()
    dev = _create_device(tlf, address, None)
    try:
        cam = pylon.InstantCamera(dev)
        info = cam.GetDeviceInfo()
        out = {"model": info.GetModelName(), "serial": info.GetSerialNumber(),
               "ip": address if is_ip(address) else ""}
        cam.DestroyDevice() if hasattr(cam, "DestroyDevice") else None
        return out
    except CameraError:
        raise
    except Exception as e:
        raise CameraError(_first_line(e))


def _first_line(e):
    return str(e).splitlines()[0] if str(e) else type(e).__name__


def _create_device(tlf, address, index):
    """A pylon device for an address (IP / serial / user name) or an index."""
    address = (address or "").strip()
    if address:
        di = pylon.DeviceInfo()
        if is_ip(address):
            di.SetIpAddress(address)
            try:
                di.SetDeviceClass("BaslerGigE")
            except Exception:
                pass
            how = "at %s" % address
        elif address.isdigit():
            di.SetSerialNumber(address)
            how = "with serial number %s" % address
        else:
            di.SetUserDefinedName(address)
            how = "named %s" % address
        try:
            return tlf.CreateDevice(di)
        except Exception as e:
            hint = ("" if is_ip(address) else
                    " Finding a camera by serial number needs GigE discovery, which "
                    "does not work on every network - enter its IP address instead.")
            raise CameraError("No Basler camera answered %s (%s).%s Check the cable, the "
                              "power and that the PC's network port is on the camera's "
                              "subnet." % (how, _first_line(e), hint))
    devices = tlf.EnumerateDevices()
    if not devices:
        raise CameraError("No Basler camera found. Enter the camera's IP address under "
                          "Settings > Camera - discovery does not reach every camera, "
                          "but an address always does.")
    index = int(index or 0)
    if index >= len(devices):
        raise CameraError("Camera %d requested but only %d camera(s) were found."
                          % (index, len(devices)))
    return tlf.CreateDevice(devices[index])


def who_holds_camera():
    """Programs on this PC that may hold a camera: Pylon Viewer, another copy
    of Vision Station. Names only; [] when nothing is found or psutil is
    missing."""
    try:
        import psutil
    except Exception:
        return []
    me = os.getpid()
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] == me:
                continue
            name = (p.info["name"] or "").lower()
            cmd = " ".join(p.info["cmdline"] or []).lower()
            if "pylonviewer" in name or "pylon viewer" in cmd:
                out.append("Pylon Viewer (process %d)" % p.info["pid"])
            elif "python" in name and any(k in cmd for k in ("vision_station", "vp_app.py",
                                                                "run_job", ".vsp")):
                out.append("another Vision Station (process %d)" % p.info["pid"])
            elif "ipconfigurator" in name or "pylonipconfig" in name:
                out.append("pylon IP Configurator (process %d)" % p.info["pid"])
        except Exception:
            continue
    return out


# Opens of one camera in this process are serialised: the previous holder must
# have closed before the next opens, or the camera refuses the second one.
_holders = {}                       # camera key -> CameraThread holding it
_holders_lock = threading.Lock()


def _camera_key(address, index):
    return ("addr", address.strip()) if (address or "").strip() else ("index", int(index or 0))


class PylonCamera:
    """Thin wrapper around one InstantCamera. Not thread-safe by itself --
    CameraThread below is what a UI should actually hold."""

    # How long a refused open is retried. A GigE connection left behind by a
    # program that ended without closing is dropped by the camera after its
    # heartbeat timeout, 3 s unless changed.
    OPEN_RETRY_S = 6.0

    def __init__(self, index=0, address=""):
        self.index = index
        self.address = address or ""
        self.cam = None
        self.converter = None
        self.info = {}
        self.last_grab_error = None
        self.is_gige = False
        self.turn_code = None          # a software turn still to do on each frame

    def open(self, stop_event=None):
        tlf = pylon.TlFactory.GetInstance()
        deadline = time.time() + self.OPEN_RETRY_S
        while True:
            dev = _create_device(tlf, self.address, self.index)
            cam = None
            try:
                cam = pylon.InstantCamera(dev)
                cam.Open()
                break
            except Exception as e:
                msg = _first_line(e)
                busy = any(k in msg.lower() for k in ("controlled by another",
                                                      "exclusive", "access denied",
                                                      "not accessible", "in use"))
                if cam is not None:
                    try:
                        cam.DestroyDevice()
                    except Exception:
                        pass
                if busy and time.time() < deadline and not (stop_event and stop_event.is_set()):
                    time.sleep(0.5)
                    continue
                holders = who_holds_camera() if busy else []
                name = self.address or "camera %d" % self.index
                if busy:
                    raise CameraError(
                        "The camera (%s) is held by another connection%s. Close it and "
                        "press Retry. If nothing else is open, the camera keeps a "
                        "connection from a program that ended abruptly for a few "
                        "seconds - wait and retry. (%s)"
                        % (name, (": " + ", ".join(holders)) if holders else
                           " - Pylon Viewer, another copy of this application, or "
                           "another PC on the camera network", msg))
                raise CameraError("The camera (%s) would not open: %s" % (name, msg))
        self.cam = cam
        d = cam.GetDeviceInfo()
        self.info = {"vendor": d.GetVendorName(), "model": d.GetModelName(),
                     "serial": d.GetSerialNumber(), "address": self.address}
        try:
            self.is_gige = "gige" in str(d.GetDeviceClass()).lower()
        except Exception:
            self.is_gige = bool(self.address and is_ip(self.address))

        # Convert whatever the sensor gives us (often Bayer or Mono) to BGR8
        # once, here, instead of hand-rolling debayering per frame.
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.last_grab_error = None
        try:
            self._use_lean_pixel_format()
        except Exception as e:
            print(f"[camera] pixel format left as it was: {e}")
        return self.info

    def _use_lean_pixel_format(self):
        """A colour camera left in a YUV 4:2:2 format sends two bytes per
        pixel; its Bayer format sends one, and the converter turns either
        into the same BGR picture. On a 20-megapixel GigE camera the YUV
        frame is 40 MB, more than the link carries in a frame period: frames
        arrived incomplete, and were shown and inspected with a solid green
        band across the bottom eighth. A camera returns to YUV whenever its
        factory settings are reloaded, so it is checked every time it opens."""
        node = getattr(self.cam, "PixelFormat", None)
        if node is None or not node.IsWritable():
            return
        current = node.GetValue()
        if not str(current).upper().startswith("YUV"):
            return
        for want in [f for f in node.Symbolics if f.startswith("Bayer") and f.endswith("8")]:
            try:
                node.SetValue(want)
                print(f"[camera] pixel format {current} -> {want} (half the data per frame)")
                return
            except Exception:
                continue

    # -- node-name compatibility across camera families ------------------
    def _node(self, names, writable=True):
        """(node, name) for the first of `names` this camera has (and can
        write, when asked), else (None, None)."""
        if self.cam is None:
            return None, None
        nm = self.cam.GetNodeMap()
        for n in names:
            if nm.GetNode(n) is None:
                continue
            node = getattr(self.cam, n, None)
            if node is None:
                continue
            # pypylon 26 hands back a PlaceholderParameter, not None, for a
            # node the camera does not have ("ExposureTime" on a GigE ace):
            # it is neither writable nor readable
            if writable and not node.IsWritable():
                continue
            readable = getattr(node, "IsReadable", None)
            if not writable and readable is not None and not readable():
                continue
            return node, n
        return None, None

    def _first_node(self, names):
        return self._node(names)[0]

    def _first_node_named(self, names):
        return self._node(names)

    def _set_number(self, names, value, integer=None):
        node, name = self._node(names)
        if node is None:
            return None
        lo, hi = node.GetMin(), node.GetMax()
        v = min(max(float(value), float(lo)), float(hi))
        if integer is None:
            integer = isinstance(node.GetValue(), int) and not isinstance(node.GetValue(), bool)
        if integer:
            v = int(round(v))
            inc = getattr(node, "GetInc", None)
            try:
                step = int(inc()) if inc else 1
            except Exception:
                step = 1
            if step > 1 and v < int(hi):           # the maximum is always allowed
                v = int(lo) + ((v - int(lo)) // step) * step
        node.SetValue(v)
        return v

    def _set_enum(self, names, value):
        node, _ = self._node(names)
        if node is None:
            return False
        options = list(getattr(node, "Symbolics", []) or [])
        if options and value not in options:
            return False
        node.SetValue(value)
        return True

    def _set_bool(self, names, value):
        node, _ = self._node(names)
        if node is None:
            return False
        node.SetValue(bool(value))
        return True

    # -- applying settings ----------------------------------------------
    def apply_settings(self, s: CameraSettings, structural=True):
        """Every setting, each on its own so one the camera lacks does not
        stop the rest. structural=False skips those that need acquisition
        stopped (see CameraSettings.STRUCTURAL)."""
        if not self.cam or not self.cam.IsOpen():
            return
        steps = []
        if structural:
            steps += [("heartbeat", self._apply_heartbeat), ("pixel format", self._apply_pixel_format),
                      ("mirror", self._apply_reverse), ("region", self._apply_roi),
                      ("packet size", self._apply_transport), ("trigger", self._apply_trigger)]
        steps += [("exposure", self._apply_exposure), ("gain", self._apply_gain),
                  ("white balance", self._apply_white_balance), ("gamma", self._apply_gamma),
                  ("black level", self._apply_black_level), ("frame rate", self._apply_frame_rate),
                  ("inter-packet delay", self._apply_packet_delay)]
        for what, fn in steps:
            try:
                fn(s)
            except Exception as e:
                print(f"[camera] {what} not applied: {e}")

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
        node.SetValue(min(max(s.exposure_ms * 1000.0, lo), hi))   # ms -> us

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

    def white_balance_once(self):
        """Balance white once on what the camera sees now, then hold it."""
        return self._set_enum(["BalanceWhiteAuto"], "Once")

    def _apply_pixel_format(self, s):
        if s.pixel_format and s.pixel_format != "auto":
            self._set_enum(["PixelFormat"], s.pixel_format)

    def _apply_gamma(self, s):
        """None leaves the camera's own gamma: the acA5472-5gc starts with it
        on, and switching it off would darken every picture against the
        reference pictures taken with it."""
        if s.gamma in (None, ""):
            return
        g = float(s.gamma)
        on = abs(g - 1.0) > 1e-3
        self._set_bool(["GammaEnable"], on)
        if on:
            self._set_enum(["GammaSelector"], "User")
            self._set_number(["Gamma"], g, integer=False)

    def _apply_black_level(self, s):
        if s.black_level is not None and s.black_level != "":
            self._set_number(["BlackLevel", "BlackLevelAbs", "BlackLevelRaw"], s.black_level)

    def _apply_frame_rate(self, s):
        rate = float(s.frame_rate or 0.0)
        self._set_bool(["AcquisitionFrameRateEnable"], rate > 0)
        if rate > 0:
            self._set_number(["AcquisitionFrameRate", "AcquisitionFrameRateAbs"], rate,
                             integer=False)

    def _apply_trigger(self, s):
        """Free run, or a frame per pulse on an input line. The trigger in
        question is the frame-start trigger; other selectors are left off."""
        sel, _ = self._node(["TriggerSelector"])
        if sel is not None:
            options = list(getattr(sel, "Symbolics", []) or [])
            for other in [o for o in options if o != "FrameStart"]:
                try:
                    sel.SetValue(other)
                    self._set_enum(["TriggerMode"], "Off")
                except Exception:
                    pass
            if "FrameStart" in options or not options:
                sel.SetValue("FrameStart")
        hw = s.trigger == "hardware"
        if hw:
            self._set_enum(["TriggerSource"], s.trigger_source or "Line1")
            self._set_enum(["TriggerActivation"], s.trigger_edge or "RisingEdge")
            self._set_number(["TriggerDelay", "TriggerDelayAbs"], float(s.trigger_delay_us or 0),
                             integer=False)
        self._set_enum(["TriggerMode"], "On" if hw else "Off")

    def _apply_reverse(self, s):
        """Mirroring and rotation. A half turn is the same as mirroring both
        ways, which the sensor does for free; it is combined with the mirror
        settings here. Quarter turns -- and a half turn on a camera without
        ReverseX/Y -- are done in software on each frame (turn_code)."""
        rot = int(getattr(s, "rotation", 0) or 0) % 360
        half = rot == 180
        ok_x = self._set_bool(["ReverseX"], bool(s.reverse_x) != half)
        ok_y = self._set_bool(["ReverseY"], bool(s.reverse_y) != half)
        hardware_half = half and ok_x and ok_y
        if half and not hardware_half:
            # undo whatever half-flip landed, then turn in software
            self._set_bool(["ReverseX"], bool(s.reverse_x))
            self._set_bool(["ReverseY"], bool(s.reverse_y))
        self.turn_code = turn_code(rot, hardware_half)

    def _apply_roi(self, s):
        """A sensor region: smaller frames transfer faster. None leaves the
        camera's own region alone -- a job's reference picture and boxes are
        at that size, and the acA5472-5gc's own 5472 x 3648 is not its whole
        5496 x 3672 sensor. "full" is the whole sensor. Offsets go to 0
        first, so the new width and height are always allowed, then out to
        where they belong."""
        if not s.roi:
            return
        self._set_number(["OffsetX"], 0)
        self._set_number(["OffsetY"], 0)
        if s.roi == "full":
            x, y = 0, 0
            w = self._max_of(["WidthMax", "Width"]) or 0
            h = self._max_of(["HeightMax", "Height"]) or 0
        else:
            x, y, w, h = [int(v) for v in s.roi]
        if w > 0:
            self._set_number(["Width"], w)
        if h > 0:
            self._set_number(["Height"], h)
        self._set_number(["OffsetX"], x)
        self._set_number(["OffsetY"], y)

    def _max_of(self, names):
        for n in names:
            node, _ = self._node([n], writable=False)
            if node is None:
                continue
            try:
                return int(node.GetValue() if n.endswith("Max") else node.GetMax())
            except Exception:
                continue
        return None

    def _apply_transport(self, s):
        if s.packet_size:
            self._set_number(["GevSCPSPacketSize"], int(s.packet_size))

    def _apply_packet_delay(self, s):
        if s.inter_packet_delay is not None:
            self._set_number(["GevSCPD"], int(s.inter_packet_delay or 0))

    def _apply_heartbeat(self, s):
        if not s.heartbeat_ms:
            return
        try:
            tl = self.cam.GetTLNodeMap()
            node = tl.GetNode("HeartbeatTimeout")
            if node is not None:
                node.SetValue(int(s.heartbeat_ms))
        except Exception as e:
            print(f"[camera] heartbeat not applied: {e}")

    def exposure_range_us(self):
        node = self._first_node(["ExposureTime", "ExposureTimeAbs"])
        return (node.GetMin(), node.GetMax()) if node else (None, None)

    def gain_range(self):
        node, _ = self._first_node_named(["Gain", "GainAbs", "GainRaw"])
        return (node.GetMin(), node.GetMax()) if node else (None, None)

    def describe(self):
        """What the Settings dialog shows: the camera's current values, the
        allowed ranges, and the options for each choice -- read from the
        camera, never assumed."""
        def num(names):
            node, name = self._node(names, writable=False)
            if node is None:
                return None
            try:
                return {"value": node.GetValue(), "min": node.GetMin(), "max": node.GetMax(),
                        "node": name}
            except Exception:
                try:
                    return {"value": node.GetValue(), "node": name}
                except Exception:
                    return None

        def enum(names):
            node, name = self._node(names, writable=False)
            if node is None:
                return None
            try:
                return {"value": node.GetValue(),
                        "options": list(getattr(node, "Symbolics", []) or []), "node": name}
            except Exception:
                return None

        unit, glo, ghi = self.gain_unit()
        d = {"info": dict(self.info), "gige": self.is_gige,
             "exposure_us": num(["ExposureTime", "ExposureTimeAbs"]),
             "gain": num(["Gain", "GainAbs", "GainRaw"]), "gain_unit": unit,
             "pixel_format": enum(["PixelFormat"]),
             "gamma": num(["Gamma"]), "gamma_enable": num(["GammaEnable"]),
             "black_level": num(["BlackLevel", "BlackLevelAbs", "BlackLevelRaw"]),
             "frame_rate": num(["AcquisitionFrameRate", "AcquisitionFrameRateAbs"]),
             "resulting_frame_rate": num(["ResultingFrameRate", "ResultingFrameRateAbs"]),
             "trigger_source": enum(["TriggerSource"]),
             "trigger_activation": enum(["TriggerActivation"]),
             "trigger_mode": enum(["TriggerMode"]),
             "width": num(["Width"]), "height": num(["Height"]),
             "width_max": self._max_of(["WidthMax", "Width"]),
             "height_max": self._max_of(["HeightMax", "Height"]),
             "offset_x": num(["OffsetX"]), "offset_y": num(["OffsetY"]),
             "packet_size": num(["GevSCPSPacketSize"]),
             "inter_packet_delay": num(["GevSCPD"]),
             "white_balance": enum(["BalanceWhiteAuto"])}
        try:
            node = self.cam.GetTLNodeMap().GetNode("HeartbeatTimeout")
            d["heartbeat_ms"] = int(node.GetValue()) if node is not None else None
        except Exception:
            d["heartbeat_ms"] = None
        return d

    # -- grabbing ----------------------------------------------------------
    def start(self):
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def stop(self):
        if self.cam and self.cam.IsGrabbing():
            self.cam.StopGrabbing()

    def grab(self, timeout_ms=500):
        """One frame as a BGR uint8 numpy array, or None on a soft timeout
        (normal while waiting for a hardware trigger). Raises CameraError on a
        hard failure (device unplugged, etc)."""
        if not self.cam.IsGrabbing():
            return None
        handling = getattr(pylon, "TimeoutHandling_Return", None)
        try:
            if handling is not None:
                res = self.cam.RetrieveResult(timeout_ms, handling)
            else:
                res = self.cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
        except Exception as e:
            raise CameraError(f"grab failed: {e}")
        if res is None:
            return None
        try:
            valid = getattr(res, "IsValid", None)
            if valid is not None and not valid():
                return None                       # timed out: no frame yet
            if not res.GrabSucceeded():
                # Kept, not swallowed: "the buffer was incompletely grabbed"
                # is a network fault the operator has to hear about, and it
                # used to look exactly like a camera that was merely slow.
                try:
                    self.last_grab_error = str(res.GetErrorDescription()).split(".")[0][:160]
                except Exception:
                    self.last_grab_error = "the camera delivered a frame that failed"
                return None
            self.last_grab_error = None
            converted = self.converter.Convert(res)
            arr = converted.GetArray()
            if self.turn_code is not None:
                return cv2.rotate(arr, self.turn_code)       # a new, contiguous array
            return np.ascontiguousarray(arr)
        finally:
            try:
                res.Release()
            except Exception:
                pass

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
            try:
                self.cam.DestroyDevice()
            except Exception:
                pass
        self.cam = None


class CameraThread(threading.Thread):
    """What the UI actually talks to. Runs Pylon (or OpenCV, transparently) in
    the background and hands back whatever the newest frame is -- this is a
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
        self._stop_event = threading.Event()
        self._calls = queue.Queue()
        # Frame bookkeeping for grab_sequence: every delivered frame bumps
        # _seq; every settings request bumps _gen, and when the camera thread
        # has applied request N it records _applied_gen = N together with the
        # _seq at that moment. "A frame taken with the new setting" can then
        # be stated exactly instead of guessed from polling.
        self._seq = 0
        self._gen = 0
        self._applied_gen = 0
        self._applied_seq = 0

    @property
    def address(self):
        return (getattr(self.settings, "address", "") or "").strip()

    @property
    def fresh_frames(self):
        """Frames to let go by before one is surely exposed after a request:
        a free-running camera has frames in flight, a triggered one does not
        expose until the next pulse."""
        return 1 if getattr(self.settings, "trigger", "free") == "hardware" else 3

    def run(self):
        try:
            self._run()
        except Exception as e:
            # Never die silently: a thread that vanished left backend None and
            # no message, and the caller saw only "no image".
            self.alive = False
            self.last_error = self.last_error or "camera stopped: %s" % e
            print("[camera] %s" % self.last_error)
        finally:
            self._release_holder()

    def _run(self):
        if self.prefer_pylon:
            try:
                self._run_pylon()
                return
            except CameraError as e:
                self.last_error = str(e)
                if self._pylon_cam is not None:
                    self._pylon_cam.close()
                if self.address or list_devices():
                    # A Basler camera was asked for, or exists, and failed:
                    # falling back to OpenCV device N would quietly inspect
                    # some other camera -- a laptop webcam -- instead of the
                    # one on the line.
                    print(f"[camera] {e}")
                    return
                print(f"[camera] Pylon path failed ({e}); falling back to OpenCV")
        if not self._stop_event.is_set():
            self._run_opencv()

    def _claim(self):
        """Wait until no other camera thread in this process holds the same
        camera -- one still closing after a stop -- then hold it."""
        key = _camera_key(self.address, self.index)
        deadline = time.time() + 10.0
        while True:
            with _holders_lock:
                other = _holders.get(key)
                if other is None or other is self or not other.is_alive():
                    _holders[key] = self
                    self._key = key
                    return
            if time.time() > deadline or self._stop_event.is_set():
                raise CameraError("the camera is still being closed by this application")
            time.sleep(0.05)

    def _release_holder(self):
        key = getattr(self, "_key", None)
        if key is None:
            return
        with _holders_lock:
            if _holders.get(key) is self:
                del _holders[key]

    def _run_pylon(self):
        self._claim()
        self._pylon_cam = PylonCamera(self.index, self.address)
        self.info = self._pylon_cam.open(self._stop_event)
        if self._stop_event.is_set():
            # Stopped while it was opening: let go now. Carrying on here is
            # how a camera used to stay held until the program exited.
            self._pylon_cam.close()
            return
        with self._lock:
            settings = self.settings
            self._settings_dirty = False
            gen = self._gen
        self._pylon_cam.apply_settings(settings)
        applied_struct = settings.structural_key()
        self._mark_applied(gen)
        self._pylon_cam.start()
        self.backend = "pylon"
        self.alive = True
        while not self._stop_event.is_set():
            self._service_calls()
            if self._settings_dirty:
                with self._lock:
                    gen, settings = self._gen, self.settings
                    self._settings_dirty = False
                if settings.structural_key() != applied_struct:
                    # Region, pixel format, trigger... are refused while
                    # grabbing: stop, apply everything, start again.
                    self._pylon_cam.stop()
                    self._pylon_cam.apply_settings(settings)
                    self._pylon_cam.start()
                    applied_struct = settings.structural_key()
                else:
                    self._pylon_cam.apply_settings(settings, structural=False)
                self._mark_applied(gen)
            try:
                frame = self._pylon_cam.grab(500)
            except CameraError as e:
                self.last_error = str(e)
                time.sleep(0.2)
                continue
            if frame is not None:
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                if self.last_error and self.last_error.startswith("frame lost"):
                    self.last_error = None
            elif getattr(self._pylon_cam, "last_grab_error", None):
                self.last_error = "frame lost - %s" % self._pylon_cam.last_grab_error
        self.alive = False
        self._pylon_cam.close()
        self._service_calls()                 # anyone still waiting gets an answer

    def _service_calls(self):
        while True:
            try:
                fn, box, done = self._calls.get_nowait()
            except queue.Empty:
                return
            try:
                # `is not None`, not truth: pypylon's InstantCamera is falsy
                pc = self._pylon_cam
                cam = pc if (pc is not None and pc.cam is not None) else None
                box["value"] = fn(cam)
            except Exception as e:
                box["error"] = e
            finally:
                done.set()

    def call(self, fn, timeout_s=5.0):
        """Run fn(pylon_camera) on the camera's own thread, between frames,
        and return its result -- node access from two threads at once is not
        safe. fn gets None when there is no pylon camera (OpenCV, closed)."""
        if not self.is_alive() or self.backend != "pylon":
            return fn(None)
        box, done = {}, threading.Event()
        self._calls.put((fn, box, done))
        if not done.wait(timeout_s):
            raise CameraError("the camera did not answer in time")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def describe(self):
        return self.call(lambda pc: pc.describe() if pc is not None else None)

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
        while not self._stop_event.is_set():
            if self._settings_dirty:
                # OpenCV cannot set exposure; acknowledge so nobody waits on it
                with self._lock:
                    gen = self._gen
                    self._settings_dirty = False
                self._mark_applied(gen)
            ok, frame = cap.read()
            if ok:
                code = turn_code(getattr(self.settings, "rotation", 0))
                if code is not None:
                    frame = cv2.rotate(frame, code)
                with self._lock:
                    self._frame = frame
                    self._seq += 1
            else:
                time.sleep(0.02)
        self.alive = False
        cap.release()

    def update_settings(self, settings):
        """Safe to call from the UI thread while running -- applied on the
        camera's own thread on the next loop iteration, not here, since Pylon
        node access from two threads at once is not guaranteed safe."""
        with self._lock:
            self.settings = settings
            self._gen += 1
            self._settings_dirty = True
            return self._gen

    def _mark_applied(self, gen):
        with self._lock:
            self._applied_gen = gen
            self._applied_seq = self._seq

    def get_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def frame_after(self, gen=None, settle=3, timeout_s=3.0):
        """A frame genuinely taken after settings request `gen` took effect.

        Frames already in flight when a setting lands still carry the old one
        -- the acA5472-5gc delivers two of them -- so `settle` new frames must
        go by after the setting was applied, and the last of them is returned.
        With gen None, the next new frame. None on timeout: better no image
        than one taken at the wrong exposure."""
        deadline = time.time() + timeout_s
        target = None
        while time.time() < deadline and self.is_alive():
            with self._lock:
                if target is None:
                    if gen is None:
                        target = self._seq + max(1, settle)
                    elif self._applied_gen >= gen:
                        target = self._applied_seq + max(1, settle)
                if target is not None and self._seq >= target \
                        and self._frame is not None:
                    return self._frame.copy()
            time.sleep(0.01)
        return None

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
                gen = self.update_settings(s)
                need_new = max(1, settle)
            else:
                gen, need_new = None, 1
            frame = self.frame_after(gen, need_new,
                                     timeout_s + (need_new + 1) * float(ms or 0) / 1000.0)
            if frame is not None:
                out[cap.get("id", "default")] = frame
            # else: left out, so the caller reports the capture as missing
            # rather than inspecting a frame taken at the wrong exposure
        self.update_settings(base)
        return out

    def stop(self, timeout_s=8.0):
        """Stop and let go of the camera. Waits for it to be closed -- an open
        in progress included -- so whoever opens it next is not refused."""
        self._stop_event.set()
        self.alive = False
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=timeout_s)
