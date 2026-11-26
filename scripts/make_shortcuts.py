"""Desktop and Start-menu shortcuts for Vision Station and Vision Glass, with
the application's icon (assets/icon/vision_station.ico).

    python scripts/make_shortcuts.py              desktop and Start menu
    python scripts/make_shortcuts.py --desktop    desktop only
    python scripts/make_shortcuts.py --remove     take them away again

They start the application with pythonw (no console window), in this
folder, exactly as run.bat and run_glass.bat do. Windows only; elsewhere
nothing happens.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICON = os.path.join(ROOT, "assets", "icon", "vision_station.ico")

SHORTCUTS = [
    ("Vision Station", "-m vision_station", "Machine-vision inspection"),
    ("Vision Glass", "-m vision_station.glass", "Vision Station - the Glass front end"),
]


def _folders(desktop=True, start=True):
    out = []
    if desktop:
        out.append(_special("Desktop"))
    if start:
        out.append(os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows",
                                "Start Menu", "Programs"))
    return [d for d in out if d and os.path.isdir(d)]


def _special(name):
    """The user's real Desktop (it may be moved to OneDrive)."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "[Environment]::GetFolderPath('%s')" % name],
                           capture_output=True, text=True, timeout=20)
        p = r.stdout.strip()
        if p:
            return p
    except Exception:
        pass
    return os.path.join(os.path.expanduser("~"), name)


def _pythonw():
    exe = sys.executable
    w = os.path.join(os.path.dirname(exe), "pythonw.exe")
    return w if os.path.isfile(w) else exe


def make(desktop=True, start=True):
    if sys.platform != "win32":
        print("  shortcuts: Windows only")
        return []
    if not os.path.isfile(ICON):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import make_icon
        make_icon.main()
    made = []
    for folder in _folders(desktop, start):
        for name, args, desc in SHORTCUTS:
            lnk = os.path.join(folder, name + ".lnk")
            ps = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}'); "
                  "$s.TargetPath = '{target}'; $s.Arguments = '{args}'; "
                  "$s.WorkingDirectory = '{cwd}'; $s.IconLocation = '{icon},0'; "
                  "$s.Description = '{desc}'; $s.Save()").format(
                lnk=lnk.replace("'", "''"), target=_pythonw().replace("'", "''"), args=args,
                cwd=ROOT.replace("'", "''"), icon=ICON.replace("'", "''"), desc=desc)
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and os.path.isfile(lnk):
                made.append(lnk)
                print("  shortcut: %s" % lnk)
            else:
                print("  shortcut not made (%s): %s" % (lnk, (r.stderr or "").strip()[:200]))
    return made


def remove():
    for folder in _folders():
        for name, _a, _d in SHORTCUTS:
            lnk = os.path.join(folder, name + ".lnk")
            if os.path.isfile(lnk):
                os.remove(lnk)
                print("  removed: %s" % lnk)


if __name__ == "__main__":
    if "--remove" in sys.argv:
        remove()
    else:
        make(desktop=True, start="--desktop" not in sys.argv)
