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
                 kind="float"):
        self.value = value
        self.lo, self.hi = lo, hi
        self.writable = writable
        self.kind = kind
        self.sets = []               # every value written, for assertions

    def GetMin(self):
        return self.lo

    def GetMax(self):
        return self.hi

    def GetValue(self):
        return self.value

    def SetValue(self, v):
        if not self.writable:
            raise RuntimeError("node is read-only")
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
                 vendor="Basler"):
        self._m, self._s, self._v = model, serial, vendor

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
        for name, node in self.nodes.items():
            setattr(self, name, node)

    def Open(self):
        self._open = True

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
            scale = (us / float(self.device.ref_exposure_us)) * g
            arr = np.clip(self.device.scene.astype(np.float32) * scale,
                          0, 255).astype(np.uint8)
        return _GrabResult(arr, ok)


class _Device:
    """Describes a simulated camera: which nodes it has, and how it misbehaves."""

    def __init__(self, family="usb3", shape=(120, 160), fail_every=0,
                 raise_on_grab=False, model=None, scene=None,
                 ref_exposure_us=5000.0):
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
        elif family == "gige_classic":  # older GigE ace
            self.nodes = {
                "ExposureTimeAbs": _Node(5000.0, 35.0, 999999.0),
                "GainRaw": _Node(0, 0, 512),
                "ExposureAuto": _Node("Off", kind="enum"),
                "GainAuto": _Node("Off", kind="enum"),
                "BalanceWhiteAuto": _Node("Off", kind="enum"),
            }
            self.info = _DeviceInfo(model or "acA1300-30gm", "GIGE0001")
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

    @staticmethod
    def GetInstance():
        return _TlFactory()

    def EnumerateDevices(self):
        return [d.info for d in _TlFactory.devices]

    def CreateDevice(self, info):
        for d in _TlFactory.devices:
            if d.info is info:
                return d
        raise RuntimeError("unknown device")


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


def install(devices):
    """Install the fake as pypylon.pylon and return the module."""
    _TlFactory.devices = list(devices)
    pylon = types.ModuleType("pypylon.pylon")
    pylon.TlFactory = _TlFactory
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
