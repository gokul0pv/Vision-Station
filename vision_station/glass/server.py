"""Vision Glass's server: the Vision Station engine over HTTP, for the web
front end in web/ (HTML, CSS, JavaScript -- no build step).

Only on 127.0.0.1. The same projects, image sources, cameras, barcode rule,
result pictures and results database as Vision Station; nothing here changes
how Vision Station itself runs.

    GET  /api/state                 applications, jobs and their checks, counts, barcode
    POST /api/open        {app}     open an application
    POST /api/run         {pid?}    inspect once (the barcode rule applies unless pid)
    POST /api/continuous  {on}
    GET  /api/results               the last result of every job
    GET  /api/history
    POST /api/preview     {pid}     the job on its teach picture (live results)
    POST /api/tool/add    {pid, type, roi, shape?, points?}
    POST /api/tool/update {pid, id, params}
    POST /api/tool/delete {pid, ids}
    POST /api/undo        {pid}
    POST /api/save        {pid}
    POST /api/find        {pid, mode, area?, sensitivity?}
    POST /api/find/add    {pid, mode, items, names?, thresholds?, confidence?}
    POST /api/source      {pid, mode, folder?, files?, camera?}
    POST /api/reference   {pid}     teach on the last inspection's picture
    POST /api/barcode     {cfg}     and  POST /api/barcode/scan {code}
    GET  /api/schema?type=...       a check's settings, with their help
    GET  /img/<which>/<pid>.jpg?w=  which: reference | last | current
"""
import collections
import copy
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from vision_station import paths

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
FONTS = os.path.join(paths.ASSETS, "fonts")
ICONS = os.path.join(paths.ASSETS, "icon")
HISTORY_LEN = 60


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _jpeg(img, long_side, quality=88):
    h, w = img.shape[:2]
    k = min(1.0, float(long_side) / max(h, w)) if long_side else 1.0
    if k < 1.0:
        step = max(1, int(0.5 / k))
        small = img[::step, ::step] if step > 1 else img
        img = cv2.resize(small, (max(1, int(w * k)), max(1, int(h * k))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _tool_out(t):
    """A check's result for the front end: what it needs to draw and say."""
    keep = ("tool_id", "type", "status", "score", "message", "ms", "caption", "graphics")
    out = {k: t.get(k) for k in keep}
    v = t.get("values") or {}
    out["values"] = {k: v[k] for k in ("text", "lines", "expected", "count", "match",
                                       "good_match", "ng_match", "length_mm", "length_px")
                     if k in v}
    return out


class GlassStation:
    """The engine, as the front end drives it. One inspection at a time."""

    def __init__(self, root=None, settings=None):
        """settings: a QSettings -- Vision Station's own unless one is given
        (tests give a file of their own, so they never change the user's)."""
        from vision_station.core.engine import ProjectStore
        from vision_station.core import barcode as rules
        from vision_station.core.result_images import ResultImageWriter
        from PySide6.QtCore import QSettings
        self.qs = settings if settings is not None else QSettings("VisionStation", "QuickBuild")
        self.store = ProjectStore(root or paths.PROJECTS)
        self.lock = threading.RLock()
        self.run_lock = threading.Lock()
        self.app_id, self.jobs, self.projects, self.sources = None, [], {}, {}
        self.last, self.seq = {}, 0
        self.history = collections.deque(maxlen=HISTORY_LEN)
        self.counts = {"accept": 0, "reject": 0}
        self.undo = collections.defaultdict(list)
        self.continuous = False
        self.message = ""
        self.gate = rules.Gate(rules.load(self.qs))
        self.images = ResultImageWriter()
        try:
            from vision_station.ui.settings_dialog import load_result_settings
            self.images.configure(load_result_settings(self.qs))
        except Exception:
            pass
        self.logs = {}
        apps = self.applications()
        if apps:
            sizes = {a["id"]: sum(1 for x in self.store.list()
                                  if (x.get("job") or x["id"]) == a["id"]) for a in apps}
            self.open(max(apps, key=lambda a: sizes[a["id"]])["id"])

    # ---- applications ---------------------------------------------------
    def applications(self):
        seen, out = set(), []
        for x in self.store.list():
            key = x.get("job") or x["id"]
            if key not in seen:
                seen.add(key)
                out.append({"id": key, "name": x.get("job_name") or x.get("name", key)})
        return out

    def open(self, app_id):
        from vision_station.ui import main_window as V
        from vision_station.core.results import ResultStore
        with self.lock:
            self.continuous = False
            for src in self.sources.values():
                src.stop_camera()
            for lg in self.logs.values():
                try:
                    lg.close()
                except Exception:
                    pass
            entries = [x for x in self.store.list() if x.get("job") == app_id] or \
                [x for x in self.store.list() if x["id"] == app_id]
            self.app_id = app_id
            self.jobs = [{"pid": x["id"], "name": x.get("name", x["id"])} for x in entries]
            self.projects = {j["pid"]: self.store.reload(j["pid"]) for j in self.jobs}
            base = os.path.join(self.store.root, app_id)
            self.sources = {}
            for pid, p in self.projects.items():
                src = V.ImageSource.from_spec(p)
                src.camera_factory = V.open_camera
                src.base = base
                self.sources[pid] = src
            self.logs = {}
            for pid, p in self.projects.items():
                try:
                    self.logs[pid] = ResultStore(p.folder, pid, archive="none")    # pictures: Settings > Results only
                except Exception:
                    pass
            self.last, self.history = {}, collections.deque(maxlen=HISTORY_LEN)
            self.counts = {"accept": 0, "reject": 0}
            self.undo.clear()

    def job_name(self, pid):
        return next((j["name"] for j in self.jobs if j["pid"] == pid), pid)

    # ---- state ------------------------------------------------------------
    def state(self):
        from vision_station.core.schema import LABELS
        jobs = []
        for j in self.jobs:
            p = self.projects[j["pid"]]
            ref = p.reference
            jobs.append({"pid": j["pid"], "name": j["name"],
                         "tools": p.spec.get("tools", []),
                         "source": self.sources[j["pid"]].describe(),
                         "source_spec": self.sources[j["pid"]].to_spec(),
                         "size": [int(ref.shape[1]), int(ref.shape[0])] if ref is not None
                         else None,
                         "undo": len(self.undo[j["pid"]])})
        return {"apps": self.applications(), "app": self.app_id, "jobs": jobs,
                "counts": self.counts, "continuous": self.continuous, "seq": self.seq,
                "running": self.run_lock.locked(), "message": self.message,
                "barcode": dict(self.gate.cfg, pending=self.gate.pending,
                                last=list(self.gate.last) if self.gate.last else None),
                "labels": {k: v[0] for k, v in LABELS.items()}}

    # ---- running ------------------------------------------------------------
    def run_once(self, pids=None, gate=True):
        """Inspect: every job, or the ones given (an editor's run: no barcode)."""
        if not self.run_lock.acquire(blocking=False):
            return {"busy": True}
        try:
            code = None
            if gate and self.gate.enabled:
                code = self.gate.take()
                if code is None:
                    self.message = "Scan the barcode first"
                    return {"need_barcode": True}
            out = {}
            for pid in (pids or [j["pid"] for j in self.jobs]):
                p, src = self.projects.get(pid), self.sources.get(pid)
                if p is None:
                    continue
                img = src.grab()
                if img is None:
                    out[pid] = {"error": "no image: %s" % src.last_error}
                    continue
                snap = json.loads(json.dumps(p.spec))
                res = p.inspect(img, source="glass", keep_images=False,
                                tools=p.build_tools(snap), barcode=code)
                res["input_image"] = img
                if code:
                    res["barcode"] = code
                self._done(pid, res, code)
                out[pid] = self.result_of(pid)
            self.message = ""
            return {"results": out}
        finally:
            self.run_lock.release()

    def _done(self, pid, res, code):
        from vision_station.core import barcode as rules
        with self.lock:
            self.seq += 1
            res["seq"] = self.seq
            self.last[pid] = res
            ok = res["verdict"] == "PASS"
            self.counts["accept" if ok else "reject"] += 1
            self.history.appendleft({
                "seq": self.seq, "time": time.strftime("%H:%M:%S"), "pid": pid,
                "job": self.job_name(pid), "verdict": res["verdict"],
                "ms": round(res.get("total_ms", 0.0)), "barcode": code or "",
                "failed": [t["tool_id"] for t in res["tools"] if t["status"] != "pass"]})
        lg = self.logs.get(pid)
        if lg is not None:
            lg.log(res, res.get("input_image"), "")
        cfg = None
        if code:
            base = dict(self.images.cfg)
            root = rules.results_root(self.gate.cfg, base)
            cfg = rules.result_cfg(base, root, code) if root else {}
            from vision_station.core.result_images import folder_for
            self.gate.last = (code, res["verdict"], folder_for(cfg) if root else "")
        try:
            self.images.submit(res.get("input_image"), res, self.job_name(pid), cfg=cfg)
        except Exception as e:
            print("[glass] picture not saved: %s" % e)

    def result_of(self, pid):
        res = self.last.get(pid)
        if not res:
            return None
        return {"pid": pid, "seq": res.get("seq"), "verdict": res["verdict"],
                "total_ms": round(res.get("total_ms", 0.0), 1),
                "image_size": res.get("image_size"), "barcode": res.get("barcode", ""),
                "tools": [_tool_out(t) for t in res["tools"]]}

    def set_continuous(self, on):
        self.continuous = bool(on)
        if self.continuous:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.continuous:
            r = self.run_once()
            if r.get("busy") or r.get("need_barcode"):
                time.sleep(0.25)
            elif not any("error" not in v for v in (r.get("results") or {}).values()):
                time.sleep(0.5)

    # ---- the teach picture ------------------------------------------------
    def preview(self, pid):
        p = self.projects[pid]
        if p.reference is None:
            return {"tools": []}
        snap = json.loads(json.dumps(p.spec))
        res = p.inspect(p.reference, source="teach", tools=p.build_tools(snap))
        return {"pid": pid, "verdict": res["verdict"],
                "tools": [_tool_out(t) for t in res["tools"]]}

    # ---- editing ------------------------------------------------------------
    def _remember(self, pid):
        st = self.undo[pid]
        st.append(copy.deepcopy(self.projects[pid].spec.get("tools", [])))
        del st[:-50]

    def _save(self, pid):
        """The recipe written as it is, the tools rebuilt -- without reading
        the 20-megapixel reference again as Project.save does."""
        p = self.projects[pid]
        with open(p.path, "w", encoding="utf-8") as f:
            json.dump(p.spec, f, indent=2)
        p.tools = p.build_tools()

    def _tool(self, pid, tid):
        return next((t for t in self.projects[pid].spec.get("tools", []) if t["id"] == tid), None)

    def add_tool(self, pid, ty, roi=None, shape="rect", points=None):
        from vision_station.core.schema import defaults
        from vision_station.ui import main_window as V
        p = self.projects[pid]
        tools = p.spec.setdefault("tools", [])
        self._remember(pid)
        n = 1
        while any(t["id"] == "%s_%d" % (ty, n) for t in tools):
            n += 1
        params = defaults(ty)
        if roi:
            params["roi"] = [int(round(v)) for v in roi]
        params["shape"] = shape or "rect"
        if shape == "polygon" and points:
            params["points"] = [[float(a), float(b)] for a, b in points]
        spec = {"id": "%s_%d" % (ty, n), "type": ty, "params": params}
        tools.append(spec)
        text, msg = V.teach_check(p, spec)          # Verify text learns its text
        self._save(pid)
        return {"id": spec["id"], "message": msg}

    def update_tool(self, pid, tid, params):
        from vision_station.ui import main_window as V
        t = self._tool(pid, tid)
        if t is None:
            raise KeyError(tid)
        self._remember(pid)
        moved = "roi" in params and params["roi"] != t["params"].get("roi")
        for k, v in params.items():
            if v is None:
                t["params"].pop(k, None)
            else:
                t["params"][k] = v
        msg = ""
        if moved:
            _text, msg = V.teach_check(self.projects[pid], t)
        self._save(pid)
        return {"message": msg}

    def update_many(self, pid, ids, params):
        """One setting for several checks -- a group's threshold -- in one undo step."""
        self._remember(pid)
        n = 0
        for t in self.projects[pid].spec.get("tools", []):
            if t["id"] in set(ids):
                for k, v in params.items():
                    t["params"][k] = v
                n += 1
        self._save(pid)
        return {"changed": n}

    def delete_tools(self, pid, ids):
        self._remember(pid)
        tools = self.projects[pid].spec.setdefault("tools", [])
        tools[:] = [t for t in tools if t["id"] not in set(ids)]
        self._save(pid)
        return {"deleted": list(ids)}

    def undo_last(self, pid):
        st = self.undo[pid]
        if not st:
            return {"undone": False}
        self.projects[pid].spec["tools"] = st.pop()
        self._save(pid)
        return {"undone": True}

    # ---- Find parts / text ----------------------------------------------------
    def find(self, pid, mode, area=None, sensitivity=0.5):
        from vision_station.core import part_finder as pf
        img = self.projects[pid].reference
        if img is None:
            raise ValueError("this job has no teach picture")
        if mode in ("ocv", "ocr"):
            r = pf.find_text(img, area)
            items = [{"box": [int(v) for v in x["box"]], "text": x.get("text", ""),
                      "score": round(float(x.get("score", 0)), 3),
                      "keep": len(x.get("text", "").strip()) >= 2 and x.get("score", 0) >= 0.6}
                     for x in r["parts"]]
            return {"items": items, "ms": round(r.get("ms", 0)), "device": r.get("device", "")}
        r = pf.find_parts(img, area, float(sensitivity))
        items = [{"box": [int(v) for v in x["box"]], "type": int(x["type"]),
                  "type_name": pf.type_name(x["type"]), "score": round(float(x["score"]), 3),
                  "keep": True} for x in r["parts"]]
        return {"items": items, "ms": round(r.get("ms", 0)), "device": r.get("device", ""),
                "types": r.get("types", 0)}

    def find_add(self, pid, mode, items, names=None, thresholds=None, confidence=None):
        from vision_station.core import part_finder as pf
        p = self.projects[pid]
        tools = p.spec.setdefault("tools", [])
        h, w = p.reference.shape[:2]
        ids = [t["id"] for t in tools]
        self._remember(pid)
        if mode in ("ocv", "ocr"):
            new = pf.text_tools(items, mode, ids, image_size=(w, h),
                                min_confidence=confidence)
        else:
            names = {int(k): v for k, v in (names or {}).items() if v}
            thr = {int(k): float(v) for k, v in (thresholds or {}).items()}
            new = pf.presence_tools([dict(x, type=int(x["type"])) for x in items], names, ids,
                                    image_size=(w, h), thresholds=thr)
        tools.extend(new)
        self._save(pid)
        return {"added": [t["id"] for t in new]}

    # ---- the image source and the teach picture --------------------------------
    def set_source(self, pid, mode, folder="", files=None, camera=0):
        from vision_station.ui import main_window as V
        src = self.sources[pid]
        if mode == "folder":
            if not os.path.isdir(folder):
                raise ValueError("that folder does not exist")
            src.use_folder(folder)
        elif mode == "files":
            src.use_files([f for f in (files or []) if f])
        elif mode == "camera":
            src.use_camera(V.open_camera(int(camera)), int(camera))
        else:
            src.use_reference()
        p = self.projects[pid]
        p.spec["image_source"] = src.to_spec()
        self._save(pid)
        return {"source": src.describe()}

    def teach_on_last(self, pid):
        from vision_station.ui import main_window as V
        res = self.last.get(pid)
        if not res or res.get("input_image") is None:
            raise ValueError("inspect a good part first")
        p = self.projects[pid]
        img = res["input_image"]
        if p.reference is not None and p.reference.shape[:2] != img.shape[:2]:
            raise ValueError("that picture is a different size from the teach picture")
        self._remember(pid)
        moved = V.fixture_moved(res)
        if moved:
            V.move_regions(p.spec.get("tools", []), next(iter(V.spaces_for(res).values())))
        p.set_reference(img.copy())
        self._save(pid)
        return {"moved": bool(moved)}

    # ---- barcode ---------------------------------------------------------------
    def set_barcode(self, cfg):
        from vision_station.core import barcode as rules
        cfg = dict(rules.DEFAULTS, **(cfg or {}))
        rules.save(self.qs, cfg)
        self.gate.configure(cfg)
        return cfg

    def scan(self, code):
        ok, code, why = self.gate.scan(code)
        out = {"ok": ok, "code": code, "why": why}
        if ok and self.gate.cfg.get("auto_run", True):
            out["run"] = self.run_once()
        return out

    def image(self, which, pid):
        if which == "reference":
            return self.projects[pid].reference
        if which == "current":
            return self.sources[pid].current
        res = self.last.get(pid)
        return res.get("input_image") if res else None


# ================================================================ HTTP
class Handler(BaseHTTPRequestHandler):
    station = None
    server_version = "VisionGlass/1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", cache=False):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, default=_jsonable).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        st = self.station
        try:
            if u.path in ("/", "/index.html"):
                return self._file(os.path.join(WEB, "index.html"))
            if u.path.startswith("/web/"):
                return self._file(os.path.join(WEB, u.path[5:]))
            if u.path.startswith("/fonts/"):
                return self._file(os.path.join(FONTS, os.path.basename(u.path)), cache=True)
            if u.path.startswith("/icon/") or u.path == "/favicon.ico":
                # the application's icon (assets/icon): tab, sidebar mark
                name = "vision_station.ico" if u.path == "/favicon.ico" else                     os.path.basename(u.path)
                return self._file(os.path.join(ICONS, name), cache=True)
            if u.path == "/api/state":
                return self._send(200, st.state())
            if u.path == "/api/results":
                return self._send(200, {pid: st.result_of(pid) for pid in st.last})
            if u.path == "/api/history":
                return self._send(200, list(st.history))
            if u.path == "/api/schema":
                return self._send(200, _schema(q.get("type", [""])[0]))
            if u.path.startswith("/img/"):
                _, _, which, name = u.path.split("/", 3)
                pid = name.rsplit(".", 1)[0].replace("%2F", "/")
                img = st.image(which, pid)
                if img is None:
                    return self._send(404, {"error": "no picture"})
                w = int(q.get("w", ["2000"])[0])
                return self._send(200, _jpeg(img, w), "image/jpeg")
            return self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        st = self.station
        try:
            b = self._body()
            pid = b.get("pid")
            routes = {
                "/api/open": lambda: (st.open(b["app"]), st.state())[1],
                "/api/run": lambda: st.run_once([pid] if pid else None, gate=not pid),
                "/api/continuous": lambda: (st.set_continuous(b.get("on")),
                                            {"on": st.continuous})[1],
                "/api/preview": lambda: st.preview(pid),
                "/api/tool/add": lambda: st.add_tool(pid, b["type"], b.get("roi"),
                                                     b.get("shape", "rect"), b.get("points")),
                "/api/tool/update": lambda: st.update_tool(pid, b["id"], b.get("params") or {}),
                "/api/tool/delete": lambda: st.delete_tools(pid, b.get("ids") or []),
                "/api/tool/update-many": lambda: st.update_many(pid, b.get("ids") or [],
                                                                b.get("params") or {}),
                "/api/undo": lambda: st.undo_last(pid),
                "/api/find": lambda: st.find(pid, b.get("mode", "parts"), b.get("area"),
                                             b.get("sensitivity", 0.5)),
                "/api/find/add": lambda: st.find_add(pid, b.get("mode", "parts"), b["items"],
                                                     b.get("names"), b.get("thresholds"),
                                                     b.get("confidence")),
                "/api/source": lambda: st.set_source(pid, b.get("mode", "reference"),
                                                     b.get("folder", ""), b.get("files"),
                                                     b.get("camera", 0)),
                "/api/reference": lambda: st.teach_on_last(pid),
                "/api/barcode": lambda: st.set_barcode(b.get("cfg")),
                "/api/barcode/scan": lambda: st.scan(b.get("code", "")),
                "/api/reset": lambda: (st.counts.update(accept=0, reject=0),
                                       st.history.clear(), {"ok": True})[2],
            }
            fn = routes.get(u.path)
            if fn is None:
                return self._send(404, {"error": "not found"})
            return self._send(200, fn())
        except Exception as e:
            traceback.print_exc()
            return self._send(400, {"error": str(e)})

    def _file(self, path, cache=False):
        path = os.path.abspath(path)
        if not any(path.startswith(os.path.abspath(d) + os.sep) for d in (WEB, FONTS, ICONS)) \
                or not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        ctype = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "application/javascript; charset=utf-8", ".svg": "image/svg+xml",
                 ".ttf": "font/ttf", ".otf": "font/otf", ".woff2": "font/woff2",
                 ".ico": "image/x-icon", ".png": "image/png"}.get(
                     os.path.splitext(path)[1].lower(), "application/octet-stream")
        with open(path, "rb") as f:
            return self._send(200, f.read(), ctype, cache)


def _schema(ty):
    from vision_station.core.schema import ESSENTIAL, LABELS, SCHEMA, help_for
    rows = [{"key": k, "label": lab, "kind": kind, "default": dv,
             "extra": extra if isinstance(extra, (list, tuple)) else None,
             "help": help_for(ty, k)} for k, lab, kind, dv, extra in SCHEMA.get(ty, [])]
    return {"type": ty, "label": LABELS.get(ty, (ty, ""))[0],
            "description": LABELS.get(ty, ("", ""))[1], "rows": rows,
            "essential": ESSENTIAL.get(ty)}


def serve(station, port=0):
    """Start the server on 127.0.0.1 in a background thread; returns it."""
    Handler.station = station
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
