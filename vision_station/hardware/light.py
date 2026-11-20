"""The external light, through its controller's RS232 port.

Controller: HIFLY DCS series (here a DCS2-1C200W-24PS, one channel), which
speaks the ASCII protocol many machine-vision light controllers share
(OPT's among them): 9600 baud, 8 data bits, no parity, 1 stop bit.

    $  command  channel  data(3 hex)  checksum(2 hex)

    command  1 on   2 off   3 set brightness (000-0FF)   4 read brightness
    checksum XOR of every byte before it, as two hex digits

    $3106414   channel 1 to brightness 100 (0x064)
    $4100011   read channel 1             -> $41XXXcc

The controller answers a setting with '$', a refused frame with '&', a read
with the frame carrying the value. Measured on the DCS2 here: on, off,
set and read brightness work; the DCS3's trigger, pulse-width and
auto-detect commands get no answer, and its power-level read returns 999 --
something else on this model -- so neither is sent.

The policy (Light below), chosen by the user:

    the application not running   the controller's own state: light on, at
                                  the brightness set on its panel
    Always on                     on while the application runs
    otherwise                     off; on while a Run takes its camera
                                  picture (a job whose image source is the
                                  camera), for a whole continuous run, and
                                  while Live Camera is open for aiming
    the application closes        back to how it was found: its brightness,
                                  light on

Measured with the acA5472-5gc: the frame after 'on' is fully lit; the frame
being exposed at the moment of switching is half lit. A Run waits for three
fresh frames after it starts, so switching on just before the grab is enough.
"""
import atexit
import threading
import time
from functools import reduce

BAUD = 9600
CHANNELS_MAX = 4

DEFAULTS = {
    "enabled": False,          # Vision Station controls the light
    "port": "",                # COM7...
    "channels": 1,             # how many channels the controller has (1-4)
    "set_brightness": False,   # False: leave the brightness set on the controller
    "brightness": [255, 255, 255, 255],
    "always_on": True,
    # timing: the light switched on this long before a Run's picture is
    # taken, and kept on this long after it
    "on_before_ms": 150,
    "off_after_ms": 300,
    # automatic brightness, from the camera's pictures (see auto_level)
    "auto": False,             # keep it adjusted, from every run's picture
    "target": "reference",     # "reference": as bright as the job's teach picture
    "target_percent": 45,      # "fixed": this mean grey, % of white
    "max_clipped": 0.5,        # % of the picture allowed at white (not shown: see clip_limit)
    "job_levels": {},          # Keep adjusted's level for each job, by job id
    # The controller's own setting, kept across sessions (see _connect):
    "home_levels": {},         # channel -> brightness as the controller was found
    "home_port": "",
    "in_use": False,           # True from taking the light until handing it back
}

# Settings kept by the application, not shown on the Light page: carried over
# when the page's settings are applied.
KEPT = ("job_levels", "home_levels", "home_port", "in_use")


# ---------------------------------------------------------------- auto level
# A picture's brightness is measured as its mean grey (0-1) and the share of
# pixels at white. The light's level is changed until the mean meets the
# target without more than max_clipped of the picture burnt out. Matching
# the job's teach picture keeps every picture as bright as the one the
# checks compare against -- a light that ages, or daylight through a window,
# otherwise moves every presence score and every text read.
CLIP_GREY = 250
DEADBAND = 0.06                # within 6 % of the target: leave it
MAX_STEP = 0.35                # one run changes the level by at most 35 %


def measure(img):
    """(mean grey 0-1, share of pixels burnt out 0-1) of a picture. Burnt
    out is any colour channel at white: one colour can saturate while the
    grey is only
    0.59 of white, and a grey test never saw it."""
    import cv2
    import numpy as np
    if img is None or getattr(img, "size", 0) == 0:
        return None
    small = img[::4, ::4]
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    hot = small.max(axis=2) if small.ndim == 3 else small
    return float(g.mean()) / 255.0, float(np.count_nonzero(hot >= CLIP_GREY)) / hot.size


def target_of(cfg, reference=None):
    """The mean grey to aim for: the teach picture's, or the fixed one."""
    if cfg.get("target", "reference") == "reference" and reference is not None:
        m = measure(reference)
        if m is not None:
            return m[0]
    return max(0.02, min(0.95, float(cfg.get("target_percent", 45)) / 100.0))


def clip_limit(cfg, reference=None):
    """The share of the picture allowed at white. Matching the teach
    picture, as much as it has (a shiny part is partly white whatever the
    light: a fixed 0.5 % would turn the light down on every run); else the
    setting."""
    base = float(cfg.get("max_clipped", 0.5)) / 100.0
    if cfg.get("target", "reference") == "reference" and reference is not None:
        m = measure(reference)
        if m is not None:
            return max(base, m[1] * 1.25 + 0.002)
    return base


def next_level(level, mean, clipped, target, max_clipped):
    """The level for the next picture, from this one: proportional to how
    far the mean is from the target (a constant-current light's output, and
    so the picture's grey, rises about in step with its level), a dead band
    so it does not hunt, never more than MAX_STEP in one go, and always down
    when too much of the picture is burnt out."""
    level = max(1, min(255, int(level)))
    if clipped > max_clipped:
        return max(1, min(level - 1, int(level * 0.85)))
    if abs(mean - target) <= DEADBAND * max(target, 0.05):
        return level
    ratio = target / max(mean, 0.004)
    ratio = max(1.0 - MAX_STEP, min(1.0 + MAX_STEP, ratio))
    new = int(round(level * ratio))
    if new == level:
        new += 1 if ratio > 1 else -1
    return max(1, min(255, new))


def auto_level(set_level, grab, target, max_clipped, steps=9, progress=None,
               stop=None):
    """Find the level now: a binary search over 1-255, one fresh picture per
    try. set_level(v) changes the light; grab() returns a picture exposed
    after it. Returns (level, mean, clipped, tries) -- the try nearest the
    target without burning out -- or raises RuntimeError with no pictures."""
    lo, hi = 1, 255
    tries = []
    for i in range(steps):
        if stop is not None and stop():
            break
        mid = (lo + hi) // 2
        set_level(mid)
        m = measure(grab())
        if m is None:
            raise RuntimeError("the camera gave no picture")
        tries.append((mid, m[0], m[1]))
        if progress is not None:
            progress(i + 1, steps, mid, m[0], m[1])
        if m[1] > max_clipped or m[0] > target:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1:
            break
    if not tries:
        raise RuntimeError("stopped before any picture was taken")
    # out of reach: every try too dark -- the brightest it can do; every try
    # too bright -- the dimmest. Levels next to each other often give the
    # same grey, and the nearest-mean choice below would stop short of the end.
    for end, too in ((255, lambda t: t[1] < target and t[2] <= max_clipped),
                     (1, lambda t: t[1] > target or t[2] > max_clipped)):
        if all(too(t) for t in tries) and not any(t[0] == end for t in tries) \
                and not (stop is not None and stop()):
            set_level(end)
            m = measure(grab())
            if m is not None:
                tries.append((end, m[0], m[1]))
            if all(too(t) for t in tries):
                return end, tries[-1][1], tries[-1][2], tries
    ok = [t for t in tries if t[2] <= max_clipped] or tries
    best = min(ok, key=lambda t: abs(t[1] - target))
    set_level(best[0])
    return best[0], best[1], best[2], tries


class LightError(Exception):
    pass


def frame(cmd, channel, data=0):
    """The bytes for one command: frame('3', 1, 100) -> b'$3106414'."""
    body = "$%s%d%03X" % (cmd, int(channel), max(0, min(0xFFF, int(data))))
    return (body + "%02X" % reduce(lambda a, b: a ^ b, body.encode("ascii"))).encode("ascii")


def parse_read(reply, cmd, channel):
    """The value in a read's answer, or None when it is not one."""
    try:
        text = reply.decode("ascii")
    except UnicodeDecodeError:
        return None
    i = text.find("$%s%d" % (cmd, int(channel)))
    if i < 0 or len(text) < i + 8:
        return None
    f = text[i:i + 8]
    if frame(cmd, channel, int(f[3:6], 16)).decode().upper() != f.upper():
        return None                       # checksum wrong: a garbled answer
    return int(f[3:6], 16)


def list_ports():
    """[(device, description)] of this PC's serial ports."""
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        return []
    return [(p.device, p.description or "") for p in sorted(lp.comports(),
                                                           key=lambda p: _port_order(p.device))]


def _port_order(name):
    digits = "".join(ch for ch in name if ch.isdigit())
    return (int(digits) if digits else 0, name)


def _open_serial(port, timeout):
    import serial
    return serial.Serial(port, BAUD, bytesize=8, parity="N", stopbits=1,
                         timeout=timeout, write_timeout=1.0)


class LightController:
    """One controller on one serial port. Every call is one command and its
    answer; LightError when there is no answer or the controller refuses."""

    TIMEOUT_S = 0.3                   # answers come in about 10 ms

    def __init__(self, port, opener=None):
        self.port = port
        self._opener = opener or _open_serial
        self._s = None
        self._lock = threading.Lock()

    def open(self):
        if self._s is None:
            try:
                self._s = self._opener(self.port, self.TIMEOUT_S)
            except Exception as e:
                raise LightError("%s cannot be opened: %s" % (self.port, str(e).splitlines()[0]
                                                              if str(e) else type(e).__name__))
        return self

    def close(self):
        with self._lock:
            if self._s is not None:
                try:
                    self._s.close()
                except Exception:
                    pass
            self._s = None

    def _talk(self, data, answer_len):
        with self._lock:
            self.open()
            try:
                self._s.reset_input_buffer()
                self._s.write(data)
                self._s.flush()
                got = self._s.read(answer_len)
                if 0 < len(got) < answer_len and got[:1] != b"&":
                    got += self._s.read(answer_len - len(got))
            except Exception as e:
                self.close_unlocked()
                raise LightError("the light controller on %s stopped answering: %s"
                                 % (self.port, e))
        if not got:
            raise LightError("no answer from a light controller on %s - is it switched on, "
                             "and is this its port?" % self.port)
        if got[:1] == b"&":
            raise LightError("the light controller refused %s" % data.decode())
        return got

    def close_unlocked(self):
        try:
            self._s.close()
        except Exception:
            pass
        self._s = None

    def _set(self, cmd, channel, data=0):
        got = self._talk(frame(cmd, channel, data), 1)
        if got[:1] != b"$":
            raise LightError("unexpected answer %r to %s" % (got, frame(cmd, channel, data)))

    def on(self, channel=1):
        self._set("1", channel)

    def off(self, channel=1):
        self._set("2", channel)

    def set_brightness(self, channel, value):
        self._set("3", channel, max(0, min(255, int(value))))

    def brightness(self, channel=1):
        got = self._talk(frame("4", channel), 8)
        v = parse_read(got, "4", channel)
        if v is None:
            raise LightError("unexpected answer %r to a brightness read" % got)
        return v


def find_port(ports=None, opener=None):
    """The first port where a light controller answers a brightness read --
    a read changes nothing. None when none does."""
    for port in ports if ports is not None else [p for p, _d in list_ports()]:
        c = LightController(port, opener)
        try:
            c.brightness(1)
            return port
        except LightError:
            continue
        finally:
            c.close()
    return None


class Light:
    """What the application does with the light. Holders say why it must be
    on now -- 'run', 'continuous', 'live' -- and it is on while any holds it,
    or always with Always on. Failures never stop an inspection: they are
    kept in `error` and the connection is tried again a few seconds later."""

    RETRY_S = 3.0

    def __init__(self, opener=None):
        self._opener = opener
        self.cfg = dict(DEFAULTS)
        self.ctl = None
        self.holders = set()
        self.is_on = None             # what was last sent: True, False, None = not yet
        self.on_since = 0.0           # time.perf_counter() when it was last switched on
        self.sent_level = None        # the brightness last sent
        self.error = ""
        self.found = {}               # channel -> brightness when first connected
        self._changed_brightness = False
        self._home_from_saved = False
        self._next_try = 0.0
        self._lock = threading.RLock()
        self.on_change = []           # told (is_on, error) after every change
        self.on_level = []            # told the new brightness when it is changed
        atexit.register(self.shutdown)

    # -- settings ------------------------------------------------------------
    @property
    def enabled(self):
        return bool(self.cfg.get("enabled")) and bool(self.cfg.get("port"))

    def channels(self):
        return list(range(1, max(1, min(CHANNELS_MAX, int(self.cfg.get("channels") or 1))) + 1))

    def configure(self, cfg):
        """New settings. A new port or switching control off hands the old
        controller back as it was found first."""
        given = dict(cfg or {})
        cfg = dict(DEFAULTS, **given)
        with self._lock:
            old = self.cfg
            for k in KEPT:
                if k not in given:
                    # Settings' page does not carry them: keep what was learnt
                    v = old.get(k, DEFAULTS[k])
                    cfg[k] = dict(v) if isinstance(v, dict) else v
            if not cfg.get("enabled") or cfg.get("port") != old.get("port"):
                if self.ctl is not None:
                    self.shutdown()
                if cfg.get("port") != old.get("port"):
                    self.found = {}           # another controller: its own levels
            self.cfg = cfg
            self.error = ""
            self._next_try = 0.0
            if self.enabled:
                self._connect()
                if self.ctl is not None:
                    self._apply_brightness()
                    self.is_on = None             # send the state afresh
            self._update()

    # -- holding -------------------------------------------------------------
    def hold(self, reason):
        with self._lock:
            self.holders.add(reason)
            self._update()
            return self.is_on is True

    def settle(self):
        """Wait until the light has been on for on_before_ms -- the camera's
        picture is not taken in the moment it switches. Nothing to wait for
        when it was already on long enough (Always on, a continuous run)."""
        if self.is_on is not True:
            return 0.0
        wait = float(self.cfg.get("on_before_ms", 0) or 0) / 1000.0 -             (time.perf_counter() - self.on_since)
        if wait > 0:
            time.sleep(wait)
        return max(0.0, wait)

    def release(self, reason):
        with self._lock:
            self.holders.discard(reason)
            self._update()

    def wanted(self):
        return self.enabled and (bool(self.cfg.get("always_on")) or bool(self.holders))

    # -- the controller ----------------------------------------------------
    def _connect(self):
        if self.ctl is not None:
            return True
        if time.time() < self._next_try:
            return False
        self._next_try = time.time() + self.RETRY_S
        c = LightController(self.cfg["port"], self._opener)
        try:
            now = {ch: c.brightness(ch) for ch in self.channels()}
        except LightError as e:
            c.close()
            self._fail(e)
            return False
        if not self.found:
            self._take_home(now)
        self.ctl = c
        self.error = ""
        return True

    def _take_home(self, now):
        """What to hand the controller back as. Read from the controller --
        unless the last session never handed it back (killed: VS Code's Stop,
        a closed terminal). The controller keeps any brightness it is given,
        so it then still held the application's level (224 instead of the
        operator's 0), the next session took that for the controller's own,
        and on a normal close handed back 224 with the light on. The setting
        found first is kept in the settings until it is handed back."""
        saved = {int(k): int(v) for k, v in (self.cfg.get("home_levels") or {}).items()}
        killed = bool(self.cfg.get("in_use")) and saved and \
            self.cfg.get("home_port") == self.cfg.get("port")
        self.found = dict(saved) if killed else dict(now)
        self._home_from_saved = bool(killed) and saved != now
        self.cfg["home_levels"] = {str(k): v for k, v in self.found.items()}
        self.cfg["home_port"] = self.cfg.get("port")
        self.cfg["in_use"] = True
        self._saved()

    def _fail(self, e):
        self.error = str(e)
        if self.ctl is not None:
            self.ctl.close()
        self.ctl = None
        self.is_on = None
        self.sent_level = None
        print("[light] %s" % self.error)

    def _apply_brightness(self):
        if not self.cfg.get("set_brightness"):
            return
        levels = list(self.cfg.get("brightness") or []) + [255] * CHANNELS_MAX
        try:
            for ch in self.channels():
                self.ctl.set_brightness(ch, levels[ch - 1])
            self._changed_brightness = True
            self.sent_level = int(levels[0])
        except LightError as e:
            self._fail(e)

    def _update(self):
        want = self.wanted()
        if not self.enabled:
            self._notify()
            return
        if want == self.is_on and self.ctl is not None:
            return
        if self.ctl is None:
            if not self._connect():
                self._notify()
                return
            self._apply_brightness()
            if self.ctl is None:
                self._notify()
                return
        try:
            for ch in self.channels():
                (self.ctl.on if want else self.ctl.off)(ch)
            if want and self.is_on is not True:
                self.on_since = time.perf_counter()
            self.is_on = want
            self.error = ""
        except LightError as e:
            self._fail(e)
        self._notify()

    def _notify(self):
        for fn in list(self.on_change):
            try:
                fn(self.is_on, self.error)
            except Exception:
                pass

    # -- brightness ------------------------------------------------------------
    def level(self):
        """The brightness Vision Station gives the light (channel 1's)."""
        return int((list(self.cfg.get("brightness") or []) + [255])[0])

    def send_level(self, value):
        """Every channel to this brightness now, without making it the
        setting -- a job's own level, a level being tried. The on/off state
        is put back afterwards: a brightness command must not leave a light
        on that the rule says is off. True when it was sent."""
        value = max(0, min(255, int(value)))
        with self._lock:
            if not self.enabled or not self._connect():
                return False
            try:
                for ch in self.channels():
                    self.ctl.set_brightness(ch, value)
                self._changed_brightness = True
                self.sent_level = value
                if self.is_on is False:
                    for ch in self.channels():
                        self.ctl.off(ch)
            except LightError as e:
                self._fail(e)
                return False
        return True

    def set_level(self, value):
        """Every channel to this brightness, now, and as the setting."""
        value = max(0, min(255, int(value)))
        with self._lock:
            self.cfg["brightness"] = [value] * CHANNELS_MAX
            self.cfg["set_brightness"] = True
        if not self.send_level(value):
            return False
        self._saved()
        return True

    def _saved(self):
        for fn in list(self.on_level):
            try:
                fn(self.level())
            except Exception:
                pass

    # -- each job's own level ----------------------------------------------------
    # A job whose teach picture was taken with the best light (tune) carries
    # that level in its recipe ("light_level"); Keep adjusted's corrections
    # for it are kept here, by job, in the settings. A job without one uses
    # the Brightness setting.
    def job_level(self, pid, taught=None):
        """The level a job's pictures are taken at, or None to leave the
        light's brightness alone."""
        got = (self.cfg.get("job_levels") or {}).get(pid)
        if got is not None:
            return int(got)
        if taught is not None:
            return int(taught)
        return self.level() if self.cfg.get("set_brightness") else None

    def use_job_level(self, pid, taught=None):
        """Before a job's picture: its level, sent when it is not the one
        the light already has."""
        lv = self.job_level(pid, taught)
        if lv is None or not self.enabled:
            return None
        if lv != self.sent_level:
            self.send_level(lv)
        return lv

    def forget_job(self, pid):
        """A new teach picture: Keep adjusted starts again from its level."""
        with self._lock:
            jl = dict(self.cfg.get("job_levels") or {})
            if jl.pop(pid, None) is None:
                return
            self.cfg["job_levels"] = jl
        self._saved()

    def tune(self, grab, progress=None, stop=None, use_ai=True):
        """The clearest light level for a teach picture, from the camera
        (core/light_tuning.py), the light held on meanwhile and left at the
        level chosen. Returns find_best_level's result."""
        from vision_station.core import light_tuning
        self.hold("teach")
        try:
            self.settle()
            return light_tuning.find_best_level(self.send_level, grab, use_ai=use_ai,
                                                progress=progress, stop=stop)
        finally:
            self.release("teach")

    def set_channel(self, channel, value):
        """One channel's brightness, now -- the manual slider. Measured on
        the DCS2: a brightness sent while the light is off leaves it off, and
        on/off keep the brightness."""
        value = max(0, min(255, int(value)))
        with self._lock:
            levels = (list(self.cfg.get("brightness") or []) + [255] * CHANNELS_MAX)[:CHANNELS_MAX]
            levels[int(channel) - 1] = value
            self.cfg["brightness"] = levels
            self.cfg["set_brightness"] = True
            if not self.enabled or not self._connect():
                return False
            try:
                self.ctl.set_brightness(channel, value)
                self._changed_brightness = True
            except LightError as e:
                self._fail(e)
                return False
        return True

    def auto_step(self, picture, reference=None, pid=None, taught=None):
        """Keep adjusted: the level for the next part, from this part's
        picture -- the job's own level when it has one, else the Brightness
        setting. None when nothing changed."""
        if not (self.enabled and self.cfg.get("auto")):
            return None
        m = measure(picture)
        if m is None:
            return none
        target = target_of(self.cfg, reference)
        own = pid is not None and (taught is not None or
                                   pid in (self.cfg.get("job_levels") or {}))
        now = self.job_level(pid, taught) if own else self.level()
        new = next_level(now, m[0], m[1], target, clip_limit(self.cfg, reference))
        if new == now:
            return None
        if own:
            with self._lock:
                jl = dict(self.cfg.get("job_levels") or {})
                jl[pid] = new
                self.cfg["job_levels"] = jl
            self._saved()
        else:
            self.set_level(new)
        return new

    def adjust_now(self, grab, reference=None, progress=None, stop=None):
        """Adjust now from the camera (auto_level), the light held on
        meanwhile. Returns (level, mean, clipped, tries)."""
        self.hold("adjust")
        try:
            return auto_level(self.set_level, grab, target_of(self.cfg, reference),
                              clip_limit(self.cfg, reference), progress=progress, stop=stop)
        finally:
            self.release("adjust")

    # -- by hand, from Settings ------------------------------------------------
    def test(self, on):
        """Switch the light now, whatever the policy; the next change of
        state puts the policy back."""
        with self._lock:
            if not self._connect():
                raise LightError(self.error or "not connected")
            for ch in self.channels():
                (self.ctl.on if on else self.ctl.off)(ch)
            self.is_on = None if on != self.wanted() else on

    def read(self):
        """{channel: brightness} as the controller has it now."""
        with self._lock:
            if not self._connect():
                raise LightError(self.error or "not connected")
            return {ch: self.ctl.brightness(ch) for ch in self.channels()}

    # -- the end ---------------------------------------------------------------
    def shutdown(self):
        """The controller back as it was found: its own brightness, light
        on -- its state whenever the application is not running. Safe to call
        twice; also run when Python exits."""
        with self._lock:
            ctl = self.ctl
            if ctl is None and self.enabled and self.found:
                try:
                    ctl = LightController(self.cfg["port"], self._opener).open()
                except LightError:
                    ctl = None
            if ctl is None:
                return
            handed = False
            try:
                if self._changed_brightness or getattr(self, "_home_from_saved", False):
                    for ch, v in self.found.items():
                        ctl.set_brightness(ch, v)
                for ch in self.channels():
                    ctl.on(ch)
                handed = True
            except LightError as e:
                print("[light] not handed back: %s" % e)
            finally:
                if handed:
                    self.cfg["in_use"] = False      # handed back: nothing owed
                    self._home_from_saved = False
                    self.found = {}
                    try:
                        self._saved()
                    except Exception:
                        pass
                ctl.close()
                self.ctl = None
                self.is_on = None
                self.sent_level = None
                self._changed_brightness = False


def load(qs):
    import json
    out = dict(DEFAULTS)
    if qs is None:
        return out
    raw = qs.value("light/settings")
    if raw:
        try:
            out.update(json.loads(str(raw)))
        except (TypeError, ValueError):
            pass
    return out


def save(qs, cfg):
    import json
    qs.setValue("light/settings", json.dumps(dict(DEFAULTS, **(cfg or {}))))
    qs.sync()
