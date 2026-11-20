"""Windows Firewall permission for GigE cameras.

A GigE camera answers over UDP, and sends every picture as a stream of UDP
packets, to the program that asked. Windows Firewall lets those in only for
programs it has a rule for -- the rule is per program file. On a network
Windows calls Public (a camera port usually is: it has no gateway or domain)
a program without a rule never hears the camera.

That is why the camera opened from Python -- python.exe had been allowed
when Windows first asked -- and was "not accessible" from VisionStation.exe,
which had never been asked about: same code, same camera, nothing holding
it. Closing other programs could not help.

allow() adds the rule for the running program, through one administrator
prompt; has_rule() checks for it without needing administrator rights.
"""
import os
import subprocess
import sys

RULE_PREFIX = "Vision Station camera"


def program():
    """The program file the rule must name: VisionStation.exe when packaged,
    python.exe (or pythonw.exe) when run from source."""
    return os.path.abspath(sys.executable)


def rule_name(prog=None):
    return "%s (%s)" % (RULE_PREFIX, os.path.basename(prog or program()))


def _quiet():
    if sys.platform != "win32":
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return {"startupinfo": si, "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


def has_rule(prog=None):
    """True when this application's own allow rule exists (any rule other
    programs or Windows' own prompt made is not looked for: it may be for a
    different copy of the program). None when it cannot be told."""
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(["netsh", "advfirewall", "firewall", "show", "rule",
                              "name=%s" % rule_name(prog)],
                             capture_output=True, text=True, timeout=10, **_quiet())
    except Exception:
        return None
    # netsh's words are in the Windows language; its exit code is not:
    # 0 when a rule of that name exists, 1 when none does
    return out.returncode == 0


def should_offer():
    """Offer to add the rule: the packaged application, on Windows, without
    its rule. From source python.exe is used, which Windows has already asked
    about when Python first listened on the network."""
    return (sys.platform == "win32" and bool(getattr(sys, "frozen", False))
            and has_rule() is False)


def allow(prog=None, wait_s=60):
    """Add an inbound allow rule for the program, on every network profile,
    through Windows' administrator prompt. True when the rule is there
    afterwards; False when the prompt was declined or failed."""
    if sys.platform != "win32":
        return False
    prog = prog or program()
    name = rule_name(prog)
    args = ('advfirewall firewall add rule name="%s" dir=in action=allow '
            'program="%s" enable=yes profile=any' % (name, prog))
    import ctypes
    from ctypes import wintypes

    class SHELLEXECUTEINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("fMask", ctypes.c_ulong),
                    ("hwnd", wintypes.HWND), ("lpVerb", wintypes.LPCWSTR),
                    ("lpFile", wintypes.LPCWSTR), ("lpParameters", wintypes.LPCWSTR),
                    ("lpDirectory", wintypes.LPCWSTR), ("nShow", ctypes.c_int),
                    ("hInstApp", wintypes.HINSTANCE), ("lpIDList", ctypes.c_void_p),
                    ("lpClass", wintypes.LPCWSTR), ("hkeyClass", wintypes.HKEY),
                    ("dwHotKey", wintypes.DWORD), ("hIcon", wintypes.HANDLE),
                    ("hProcess", wintypes.HANDLE)]

    SEE_MASK_NOCLOSEPROCESS = 0x40
    info = SHELLEXECUTEINFO()
    info.cbSize = ctypes.sizeof(info)
    info.fMask = SEE_MASK_NOCLOSEPROCESS
    info.lpVerb = "runas"                     # the administrator prompt
    info.lpFile = "netsh"
    info.lpParameters = args
    info.nShow = 0
    if not ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(info)):
        return False                          # declined, or no rights
    if info.hProcess:
        ctypes.windll.kernel32.WaitForSingleObject(info.hProcess, int(wait_s * 1000))
        ctypes.windll.kernel32.CloseHandle(info.hProcess)
    return bool(has_rule(prog))


EXPLANATION = (
    "Windows Firewall is keeping the camera's pictures from reaching this program. "
    "A GigE camera sends its pictures over the network, and Windows lets them in "
    "only for programs it has been told to allow - Python was, this program was "
    "not. Allow it once (Windows asks for an administrator) and the camera works "
    "from then on.")
