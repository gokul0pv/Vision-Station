"""Vision Core fault-injection test: the engine must answer every request and
come back when it is killed. Run from vision_core/:  python tests/fault_injection.py build
"""
import json, os, subprocess, sys, tempfile, time

BUILD = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "build")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JOB = os.path.join(ROOT, "projects", "demo_led_panel")
GOOD = os.path.join(JOB, "good.png")
PIPE = r"\\.\pipe\vision-core-engine"


def call(req, timeout=60):
    t0 = time.time()
    while True:
        try:
            with open(PIPE, "r+b", buffering=0) as f:
                f.write((json.dumps(req) + "\n").encode())
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    buf += chunk
                if not buf.strip():
                    raise OSError("empty reply")      # the engine went away mid-call
                return json.loads(buf.decode("utf-8", "replace"))
        except (OSError, ValueError):
            if time.time() - t0 > timeout:
                return {"ok": False, "error": "no engine"}
            time.sleep(0.3)


def engine_pid():
    r = call({"cmd": "ping"}, 30)
    return r.get("pid") if r.get("ok") else None


results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))


subprocess.run(["taskkill", "/F", "/IM", "vc-engine.exe"], capture_output=True)
subprocess.run(["taskkill", "/F", "/IM", "vc-supervisor.exe"], capture_output=True)
sup = subprocess.Popen([os.path.join(BUILD, "vc-supervisor.exe"), "--busy-limit-s", "120"])
pid = engine_pid()
check("supervisor starts the engine", pid is not None, "pid %s" % pid)

r = call({"cmd": "inspect", "picture": GOOD})
check("inspect without a job is refused, not crashed", r.get("ok") is False and "no job" in r.get("error", ""), r.get("error", ""))

r = call({"cmd": "open", "side": os.path.join(ROOT, "no_such_job")})
check("a missing job is refused", r.get("ok") is False, r.get("error", "")[:80])

bad_job = tempfile.mkdtemp()
open(os.path.join(bad_job, "project.json"), "w").write("{ this is not json")
r = call({"cmd": "open", "side": bad_job})
check("a corrupt project.json is refused", r.get("ok") is False, r.get("error", "")[:80])

r = call({"cmd": "open", "side": JOB})
check("the demo job opens", r.get("ok") is True, "%d checks" % len(r.get("tools", [])))

corrupt = os.path.join(tempfile.mkdtemp(), "corrupt.png")
open(corrupt, "wb").write(os.urandom(5000))
r = call({"cmd": "inspect", "picture": corrupt})
check("a corrupt picture is refused", r.get("ok") is False, r.get("error", "")[:80])

r = call({"cmd": "inspect", "picture": os.path.join(JOB, "nothing_here.png")})
check("a missing picture is refused", r.get("ok") is False)

tiny = os.path.join(tempfile.mkdtemp(), "tiny.png")
import numpy as np, cv2
cv2.imwrite(tiny, np.zeros((4, 4, 3), np.uint8))
r = call({"cmd": "inspect", "picture": tiny})
check("a 4x4 picture gives a result, every check accounted for",
      r.get("ok") is True and len(r["result"]["tools"]) >= 1, r.get("result", {}).get("verdict", r.get("error", "")))

with open(PIPE, "r+b", buffering=0) as f:
    f.write(b"\x00\xff garbage {{{\n")
    reply = f.read(65536)
check("garbage on the pipe gets an error reply", b'"ok":false' in reply.replace(b" ", b""))

r = call({"cmd": "inspect", "picture": GOOD})
check("the engine still inspects after all that", r.get("ok") is True, r.get("result", {}).get("verdict", ""))

# kill the engine: the supervisor must bring it back
old = engine_pid()
subprocess.run(["taskkill", "/F", "/PID", str(old)], capture_output=True)
t0 = time.time()
new = None
while time.time() - t0 < 30:
    new = engine_pid()
    if new and new != old:
        break
    time.sleep(0.5)
check("a killed engine is restarted", new is not None and new != old, "in %.1f s" % (time.time() - t0))
r = call({"cmd": "open", "side": JOB})
r = call({"cmd": "inspect", "picture": GOOD})
check("the restarted engine inspects", r.get("ok") is True, r.get("result", {}).get("verdict", ""))

sup.kill()
subprocess.run(["taskkill", "/F", "/IM", "vc-engine.exe"], capture_output=True)
bad = [n for n, ok in results if not ok]
print("\n%d of %d passed" % (len(results) - len(bad), len(results)))
sys.exit(1 if bad else 0)
