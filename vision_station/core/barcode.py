"""Barcode-gated inspection: a part is inspected only after its barcode is
scanned and found valid, and its result pictures are kept in a folder named
after that barcode.

The rule is simple on purpose -- what a line actually checks before it lets
a part through: the code starts with given letters and has a given length.

    FIC + 11 characters      starts with "FIC", 11 characters in all

Scanners send the code as typed keys (USB keyboard mode) or over a serial
port, usually with a carriage return after it and sometimes with a prefix
character; both are cleaned off here, so the rule sees only the code.
"""
import json
import os
import re

DEFAULTS = {
    "enabled": False,
    "prefix": "",            # the code must start with this ("" = anything)
    "length": 0,             # the code's length in characters (0 = any length)
    "match_case": False,     # "fic" counts as "FIC" unless switched on
    "auto_run": True,        # a valid scan starts the inspection at once
    "source": "keyboard",    # "keyboard" (USB scanner typing) or "serial"
    "port": "",              # serial: COM3...
    "baud": 9600,
    "folder": "",            # where barcode folders go; "" = the results folder
}

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def clean(raw):
    """The code as scanned, without the line ends, tabs and control
    characters (AIM prefixes such as ]C1 are kept: they are rare and would
    be part of what the rule is written against)."""
    return _CONTROL.sub("", str(raw or "")).strip()


def describe(cfg):
    """The rule in words: 'starts with FIC, 11 characters'."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    parts = []
    if cfg.get("prefix"):
        parts.append("starts with %s" % cfg["prefix"])
    if int(cfg.get("length") or 0) > 0:
        parts.append("%d characters" % int(cfg["length"]))
    return ", ".join(parts) if parts else "any code"


def check(code, cfg):
    """(ok, reason) for a scanned code against the rule. The reason is
    written for the person at the line."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    code = clean(code)
    if not code:
        return False, "nothing was read"
    prefix = str(cfg.get("prefix") or "")
    length = int(cfg.get("length") or 0)
    head = code[:len(prefix)]
    same = head == prefix if cfg.get("match_case") else head.upper() == prefix.upper()
    if prefix and not same:
        return False, "it must start with %s - this one starts with %s" % (
            prefix, code[:max(1, len(prefix))])
    if length and len(code) != length:
        return False, "it must be %d characters - this one has %d" % (length, len(code))
    return True, ""


def folder_name(code):
    """A folder name for a code: characters Windows forbids become '_'."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", clean(code)).strip(" .")
    return name or "no_code"


def load(qs):
    out = dict(DEFAULTS)
    if qs is None:
        return out
    raw = qs.value("barcode/settings")
    if raw:
        try:
            out.update(json.loads(str(raw)))
        except (TypeError, ValueError):
            pass
    return out


def save(qs, cfg):
    qs.setValue("barcode/settings", json.dumps(dict(DEFAULTS, **(cfg or {}))))
    qs.sync()


def results_root(barcode_cfg, result_cfg, default=None):
    """The folder the barcode folders are made in: the result pictures'
    folder from Settings > Results, and only that -- "" when none is chosen,
    and then no picture is kept. A barcode folder of its own, and the
    program's own results folder when none was set, put pictures in two and
    three places."""
    return (result_cfg or {}).get("folder") or ""


class Gate:
    """The one code waiting to be used. A valid scan arms it; the next
    inspection takes it. A scan during an inspection waits its turn -- it is
    for the next part -- and a newer valid scan replaces an unused one."""

    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        self.pending = None
        self.last = None            # (code, verdict) of the last part inspected

    @property
    def enabled(self):
        return bool(self.cfg.get("enabled"))

    def configure(self, cfg):
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        if not self.enabled:
            self.pending = None

    def scan(self, raw):
        """(ok, code, reason). A valid code is kept for the next inspection."""
        code = clean(raw)
        ok, why = check(code, self.cfg)
        if ok:
            self.pending = code
        return ok, code, why

    def take(self):
        code, self.pending = self.pending, None
        return code


# ---------------------------------------------------------------- compare
# A Read text or Read code check with "Compare barcode with barcode scanned"
# on: what it reads on the part must be the code that was scanned, or the
# part is NG ("Barcode not good") and the rest of its checks are skipped.
# Only while the barcode check is on in Settings -- without a scanned code
# there is nothing to compare, and the check reads as it always did.
COMPARE_KEY = "compare_barcode"
COMPARE_TYPES = ("ocr", "code_read")
FIXTURE_TYPES = ("align", "fixture_npoint")


def _squeeze(text, look_alike=True):
    t = re.sub(r"\s+", "", str(text or "")).upper()
    if look_alike:
        from vision_station.tools.text import look_alike as la
        t = la(t)
    return t


def code_matches(scanned, read, look_alike=True, exact=False):
    """Does what was read on the part carry the scanned code? read: the
    texts the check read. A label may print more around the code ('S/N
    FIC12345678'), so a text containing the code matches; exact: a decoded
    QR or barcode must be the code itself. Spaces and capitals never count;
    O/0, I/1, S/5... only for printed text (look_alike)."""
    want = _squeeze(scanned, look_alike)
    if not want:
        return False
    for t in read or []:
        got = _squeeze(t, look_alike)
        if (got == want) if exact else (want in got):
            return True
    return False


def place_compare_checks(tools):
    """Every check comparing with the scanned barcode moved to just after the
    fixture -- the first Locate (or fixture from points) -- in the order they
    were in; to the top when there is no fixture. The barcode is then always
    the first thing checked on a part, and is found there in the list.
    Returns the ids moved."""
    marked = [t for t in tools if t.get("type") in COMPARE_TYPES
              and (t.get("params") or {}).get(COMPARE_KEY)]
    if not marked:
        return []
    rest = [t for t in tools if t not in marked]
    fx = next((i for i, t in enumerate(rest) if t.get("type") in FIXTURE_TYPES), None)
    at = 0 if fx is None else fx + 1
    new = rest[:at] + marked + rest[at:]
    if [t.get("id") for t in new] == [t.get("id") for t in tools]:
        return []
    moved = [t.get("id") for t in marked]
    tools[:] = new
    return moved


def result_cfg(base, root, code):
    """The picture settings for one scanned part: every picture of it, in
    <root>/<today>/<code>/ (result_images.folder_for)."""
    cfg = dict(base, folder=root, subfolder=folder_name(code), which="all")
    if not (cfg.get("annotated", True) or cfg.get("original")):
        cfg["annotated"] = True
    return cfg


def result_folder(root, code, when=None):
    """The folder a part's pictures go in: <root>/<day>/<code>."""
    from vision_station.core.result_images import day_folder
    return os.path.join(root, day_folder(when), folder_name(code))
