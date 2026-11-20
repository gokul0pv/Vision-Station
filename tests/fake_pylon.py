"""A stand-in for pypylon, good enough to test our integration against.

The Basler code was written by reading the real pypylon source, but it has never
run against a camera and cannot here. The part most likely to be wrong is also
the part that silently breaks everything: GenICam node names differ between
camera families. Older GigE "classic" ace cameras expose ExposureTimeAbs and
GainRaw; USB3, ace2 and dart cameras expose ExposureTime and Gain. Code that
assumes one set fails on the other with no obvious symptom -- the camera opens,
grabs frames, and quietly ignores every exposure change.

This fake models both families, plus a camera that exposes neither, plus grab
failures and disconnects. Installing it as `pypylon.pylon` lets the real
PylonCamera class run end to end.
"""
import sys
import types

import numpy as np


class _Node:
    def __init__(self, value=0.0, lo=0.0, hi=1000000.0, writable=True,
                 kind="float", options=None, inc=1):
        self.value = value
        self.lo, self.hi = lo, hi
        self.writable = writable
        self.kind = kind
        self.inc = inc
        if options is not None:
            self.Symbolics = list(options)
        self.sets = []               # every value written, for assertions

    def GetInc(self):
        return self.inc

    def GetMin(self):
        return self.lo

    def GetMax(self):
        return self.hi

    def GetValue(self):
        return self.value

    def SetValue(self, v):
        if not self.writable:
            raise RuntimeError("node is read-only")
        if getattr(self, "Symbolics", None) is not None and v not in self.Symbolics:
            raise ValueError("%r is not one of %s" % (v, self.Symbolics))
        self.value = v
        self.sets.append(v)

    def IsWritable(self):
        return self.writable


class _NodeMap:
    def __init__(self, nodes):
        self.nodes = nodes

    def GetNode(self, name):
        return self.nodes.get(name)


class _DeviceInfo:
    def __init__(self, model="acA1920-40uc", serial="12345678",
                 vendor="Basler", ip="", device_class="BaslerUsb"):
        self._m, self._s, self._v = model, serial, vendor
        self._ip, self._cls = ip, device_class

    def IsIpAddressAvailable(self):
        return bool(self._ip)

    def GetIpAddress(self):
        return self._ip

    def GetDeviceClass(self):
        return self._cls

    def GetModelName(self):
        return self._m

    def GetSerialNumber(self):
        return self._s

    def GetVendorName(self):
        return self._v


class _GrabResult:
    def __init__(self, arr, ok=True):
        self._a, self._ok = arr, ok
        self.released = False

    def GrabSucceeded(self):
        return self._ok

    def GetArray(self):
        return self._a

    def GetErrorDescription(self):
        return "" if self._ok else "simulated grab failure"

    def Release(self):
        self.released = True


class _InstantCamera:
    def __init__(self, device):
        self.device = device
        self.nodes = device.nodes
        self._open = False
        self._grabbing = False
        self.frames = 0
        self.fail_every = device.fail_every
        self.raise_on_grab = device.raise_on_grab
        self._in_flight = []         # exposure settings of frames already taken
        for name, node in self.nodes.items():
            setattr(self, name, node)

    def Open(self):
        f = self.device.fail_open
        if f is True or (isinstance(f, int) and not isinstance(f, bool)
                         and self.device.open_attempts < f):
            self.device.open_attempts += 1
            raise RuntimeError("The device is controlled by another application.")
        self.device.open_attempts += 1
        self.device.opened += 1
        self._open = True

    def DestroyDevice(self):
        self._open = False

    def GetTLNodeMap(self):
        return _NodeMap(self.device.tl_nodes)

    def Close(self):
        self._open = False

    def IsOpen(self):
        return self._open

    def GetNodeMap(self):
        return _NodeMap(self.nodes)

    def GetDeviceInfo(self):
        return self.device.info

    def StartGrabbing(self, strategy=None):
        self._grabbing = True

    def StopGrabbing(self):
        self._grabbing = False

    def IsGrabbing(self):
        return self._grabbing

    def RetrieveResult(self, timeout, handling=None):
        if self.device.frame_period_s:
            import time
            time.sleep(self.device.frame_period_s)
        self.frames += 1
        if self.raise_on_grab:
            raise RuntimeError("device disconnected")
        ok = not (self.fail_every and self.frames % self.fail_every == 0)
        h, w = self.device.shape
        if self.device.scene is None:
            arr = np.full((h, w), (self.frames * 7) % 250, np.uint8)
        else:
            # Frames respond to the exposure setting, so an exposure search can
            # be driven end to end without hardware. Brightness is linear in
            # exposure time and clips at 255, which is how a real sensor
            # behaves and is the whole point of the exercise.
            node = self.nodes.get("ExposureTime") or self.nodes.get("ExposureTimeAbs")
            us = float(node.GetValue()) if node is not None else 5000.0
            gain = self.nodes.get("Gain")
            g = 10 ** ((float(gain.GetValue()) if gain is not None else 0.0) / 20.0)
            # A real free-running camera hands back frames that were exposed
            # before the latest setting arrived: the acA5472-5gc delivers two
            # such frames after every change. pipeline_depth models that.
            self._in_flight.append((us, g))
            if len(self._in_flight) > self.device.pipeline_depth:
                us, g = self._in_flight.pop(0)
            else:
                us, g = self._in_flight[0]
            scale = (us / float(self.device.ref_exposure_us)) * g
            arr = np.clip(self.device.scene.astype(np.float32) * scale,
                          0, 255).astype(np.uint8)
        return _GrabResult(arr, ok)


class _Device:
    """Describes a simulated camera: which nodes it has, and how it misbehaves."""

    def __init__(self, family="usb3", shape=(120, 160), fail_every=0,
                 raise_on_grab=False, model=None, scene=None,
                 ref_exposure_us=5000.0, frame_period_s=0.0, pipeline_depth=0,
                 fail_open=False, ip=None):
        self.frame_period_s = frame_period_s
        self.fail_open = fail_open          # True: always; N: the first N opens
        self.open_attempts = 0
        self.opened = 0
        self.tl_nodes = {"HeartbeatTimeout": _Node(3000, 500, 600000, kind="int")}
        self.pipeline_depth = pipeline_depth
        self.shape = shape if scene is None else scene.shape[:2]
        self.fail_every = fail_every
        self.raise_on_grab = raise_on_grab
        # `scene` is the radiance the sensor sees at ref_exposure_us. Values may
        # exceed 255: a bright LED really is many times brighter than the
        # faceplate, and that headroom is what makes reducing exposure work.
        self.scene = None if scene is None else np.asarray(scene, np.float32)
        self.ref_exposure_us = ref_exposure_us
        if family == "usb3":            # ace2 / dart / USB3
            self.nodes = {
                "ExposureTime": _Node(5000.0, 20.0, 1000000.0),
                "Gain": _Node(0.0, 0.0, 36.0),
                "ExposureAuto": _Node("Off", kind="enum"),
                "GainAuto": _Node("Off", kind="enum"),
                "BalanceWhiteAuto": _Node("Off", kind="enum"),
            }
            self.info = _DeviceInfo(model or "a2A1920-51gcPRO", "USB0001")
        elif family == "gige_classic":  # older GigE ace, as the acA5472-5gc
            self.nodes = {
                "ExposureTimeAbs": _Node(5000.0, 35.0, 999999.0),
                "GainRaw": _Node(0, 0, 512),
                "ExposureAuto": _Node("Off", kind="enum", options=["Off", "Once", "Continuous"]),
                "GainAuto": _Node("Off", kind="enum", options=["Off", "Once", "Continuous"]),
                "BalanceWhiteAuto": _Node("Off", kind="enum",
                                          options=["Off", "Once", "Continuous"]),
                "PixelFormat": _Node("BayerRG8", kind="enum",
                                     options=["Mono8", "BayerRG8", "BayerRG12", "YUV422Packed"]),
                "Width": _Node(5472, 16, 5496, inc=16), "Height": _Node(3648, 16, 3672, inc=2),
                "WidthMax": _Node(5496, writable=False), "HeightMax": _Node(3672, writable=False),
                "OffsetX": _Node(0, 0, 5480, inc=2), "OffsetY": _Node(0, 0, 3656, inc=2),
                "ReverseX": _Node(False, kind="bool"), "ReverseY": _Node(False, kind="bool"),
                "GammaEnable": _Node(True, kind="bool"),
                "GammaSelector": _Node("User", kind="enum", options=["User", "sRGB"]),
                "Gamma": _Node(1.0, 0.0, 3.99998),
                "BlackLevelRaw": _Node(0, 0, 255),
                "AcquisitionFrameRateEnable": _Node(False, kind="bool"),
                "AcquisitionFrameRateAbs": _Node(29.0, 0.000288, 100000.0),
                "ResultingFrameRateAbs": _Node(5.16, writable=False),
                "TriggerSelector": _Node("FrameStart", kind="enum",
                                         options=["AcquisitionStart", "FrameStart"]),
                "TriggerMode": _Node("Off", kind="enum", options=["Off", "On"]),
                "TriggerSource": _Node("Line1", kind="enum", options=["Line1", "Software"]),
                "TriggerActivation": _Node("RisingEdge", kind="enum",
                                           options=["RisingEdge", "FallingEdge"]),
                "TriggerDelayAbs": _Node(0.0, 0.0, 1000000.0),
                "GevSCPSPacketSize": _Node(1500, 220, 16404, inc=4),
                "GevSCPD": _Node(0, 0, 65535),
                # IP settings as the acA5472-5gc had them: the stored mask did
                # not match the network, so after a power cycle it was lost
                "GevCurrentIPAddress": _Node(0xC0A80306, kind="int", writable=False),
                "GevCurrentSubnetMask": _Node(0xFFFFFF00, kind="int", writable=False),
                "GevCurrentDefaultGateway": _Node(0, kind="int", writable=False),
                "GevPersistentIPAddress": _Node(0xC0A80306, 0, 0xFFFFFFFF, kind="int"),
                "GevPersistentSubnetMask": _Node(0xFFFF0000, 0, 0xFFFFFFFF, kind="int"),
                "GevPersistentDefaultGateway": _Node(0, 0, 0xFFFFFFFF, kind="int"),
                "GevCurrentIPConfiguration": _Node(0x4, 0, 0x7, kind="int"),
            }
            self.info = _DeviceInfo(model or "acA1300-30gm", "GIGE0001", ip=ip or "",
                                    device_class="BaslerGigE")
        elif family == "minimal":       # mono camera with almost nothing exposed
            self.nodes = {}
            self.info = _DeviceInfo(model or "acA640-90um", "MIN0001")
        elif family == "readonly":      # nodes present but locked while grabbing
            self.nodes = {
                "ExposureTime": _Node(5000.0, 20.0, 100000.0, writable=False),
                "Gain": _Node(0.0, 0.0, 36.0, writable=False),
            }
            self.info = _DeviceInfo(model or "acA2500-14gm", "RO0001")
        else:
            raise ValueError(family)


class _TlFactory:
    devices = []
    hidden = []          # reachable by IP only: discovery does not see them

    @staticmethod
    def GetInstance():
        return _TlFactory()

    def EnumerateDevices(self):
        return [d.info for d in _TlFactory.devices]

    def CreateDevice(self, info):
        for d in _TlFactory.devices:
            if d.info is info:
                return d
        if isinstance(info, DeviceInfo):
            # by address: an IP reaches the camera directly, even one that
            # discovery does not list (`hidden`)
            for d in _TlFactory.devices + _TlFactory.hidden:
                if info.ip and d.info.GetIpAddress() == info.ip:
                    return d
                if info.serial and d.info.GetSerialNumber() == info.serial and \
                        d in _TlFactory.devices:
                    return d
            raise RuntimeError("No device is available or no device contains the provided "
                               "device info properties.")
        raise RuntimeError("unknown device")


class DeviceInfo:
    """What pylon.DeviceInfo() is used for: naming a camera to CreateDevice."""

    def __init__(self):
        self.ip, self.serial, self.cls, self.name = "", "", "", ""

    def SetIpAddress(self, ip):
        self.ip = ip

    def SetSerialNumber(self, s):
        self.serial = s

    def SetDeviceClass(self, c):
        self.cls = c

    def SetUserDefinedName(self, n):
        self.name = n


class ImageFormatConverter:
    def __init__(self):
        self.OutputPixelFormat = None
        self.OutputBitAlignment = None

    def Convert(self, result):
        arr = result.GetArray()
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        return _Converted(arr)


class _Converted:
    def __init__(self, arr):
        self._a = arr

    def GetArray(self):
        return self._a


def install(devices, hidden=()):
    """Install the fake as pypylon.pylon and return the module. `hidden`
    cameras answer at their IP address but are not found by discovery -- as
    the acA5472-5gc on the user's network does."""
    _TlFactory.devices = list(devices)
    _TlFactory.hidden = list(hidden)
    pylon = types.ModuleType("pypylon.pylon")
    pylon.TlFactory = _TlFactory
    pylon.DeviceInfo = DeviceInfo
    pylon.InstantCamera = _InstantCamera
    pylon.ImageFormatConverter = ImageFormatConverter
    pylon.PixelType_BGR8packed = "BGR8"
    pylon.OutputBitAlignment_MsbAligned = "Msb"
    pylon.GrabStrategy_LatestImageOnly = "latest"
    pylon.TimeoutHandling_ThrowException = "throw"
    pkg = types.ModuleType("pypylon")
    pkg.pylon = pylon
    sys.modules["pypylon"] = pkg
    sys.modules["pypylon.pylon"] = pylon
    return pylon


def uninstall():
    sys.modules.pop("pypylon", None)
    sys.modules.pop("pypylon.pylon", None)


Device = _Device
