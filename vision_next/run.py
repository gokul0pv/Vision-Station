"""Vision Next from VS Code (F5) or any terminal: build if anything changed,
then start it. Rust and the assembler its build needs are found where they
were installed, so nothing has to be on the system PATH.

    python vision_next/run.py              the window
    python vision_next/run.py serve        no window: open http://127.0.0.1:8765 in a browser
    python vision_next/run.py --no-build   start what was built last
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
TOOLS = [os.path.join(HOME, ".cargo", "bin"),
         os.path.join(os.environ.get("LOCALAPPDATA", ""), "mingw-binutils", "bin")]


def env():
    e = dict(os.environ)
    e["PATH"] = os.pathsep.join([p for p in TOOLS if os.path.isdir(p)] + [e.get("PATH", "")])
    return e


def main():
    args = sys.argv[1:]
    build = "--no-build" not in args
    args = [a for a in args if a != "--no-build"]
    if build:
        cargo = os.path.join(TOOLS[0], "cargo.exe")
        if not os.path.exists(cargo):
            sys.exit("Rust is not installed for this user (expected %s)" % cargo)
        print("Building Vision Next (only what changed)...", flush=True)
        r = subprocess.run([cargo, "build", "--release"], cwd=HERE, env=env())
        if r.returncode != 0:
            sys.exit("The build failed: see the messages above.")
    serve = bool(args) and args[0] == "serve"
    exe = os.path.join(HERE, "target", "release", "vs-run.exe" if serve else "vision-next.exe")
    if not os.path.exists(exe):
        sys.exit("%s is not built yet: run without --no-build" % exe)
    cmd = [exe] + (args if serve else args)
    print("Starting", os.path.basename(exe), " ".join(args), flush=True)
    p = subprocess.Popen(cmd, cwd=HERE, env=env())
    try:
        # stay while the program runs: Stop in VS Code closes it
        sys.exit(p.wait())
    except KeyboardInterrupt:
        p.terminate()


if __name__ == "__main__":
    main()
