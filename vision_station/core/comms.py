"""Talking to the line: a TCP/IP interface for a PLC or robot controller.

VisionPro's Communication Explorer connects a job to the line; this is the
same role over the simplest thing every controller can do -- open a TCP socket
and exchange text lines (ending in \\n or \\r\\n, upper or lower case):

    TRIGGER                   inspect every job once
    TRIGGER <job>             inspect one job (its name or id)
      -> RESULT <job> <PASS|FAIL|ERROR> <ms> <failed tools, or ->   one per job
      -> DONE <PASS|FAIL|INCOMPLETE> <number of jobs>
    GET <job>.<tool>.<value>  a value from the last run, e.g. frt_top.hole_spacing.value_px
      -> VALUE <value>
    STATUS                    -> READY or BUSY
    CODE <barcode>            -> CODE OK <code> | CODE NG <reason>   (barcode-gated
                                 inspection: the part's code, before TRIGGER)
    PING                      -> PONG
    VERSION                   -> VISIONSTATION <protocol version>

A trigger that arrives while an inspection is running is answered BUSY, never
queued silently: a controller has to know its trigger was not acted on. With
broadcast on, every run -- from the line, a button or continuous mode -- is
also sent to every connected client, for controllers that only listen.
"""
from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QHostAddress, QTcpServer

PROTOCOL_VERSION = 1


class LineServer(QObject):
    clientsChanged = Signal(int)
    commandReceived = Signal(str)
    codeScanned = Signal(str)            # a valid barcode sent with CODE

    def __init__(self, station, port=3000, broadcast=False):
        super().__init__()
        self.station = station
        self.port = int(port)
        self.broadcast = bool(broadcast)
        self.server = QTcpServer(self)
        self.server.newConnection.connect(self._accept)
        self.clients = []
        self._buffers = {}
        self._pending = None                  # the trigger being answered
        self._run = {"results": {}, "errors": {}}
        station.jobResult.connect(self._on_result)
        station.runError.connect(self._on_error)
        station.runFinished.connect(self._on_finished)
        self.error = ""

    # ------------------------------------------------------------ server
    def start(self):
        if self.server.isListening():
            return True
        ok = self.server.listen(QHostAddress.Any, self.port)
        self.error = "" if ok else self.server.errorString()
        if ok:
            self.port = self.server.serverPort()      # port 0 -> the one chosen
        return ok

    def stop(self):
        for sock in list(self.clients):
            sock.disconnectFromHost()
        self.clients.clear()
        self.server.close()
        self.clientsChanged.emit(0)

    def _accept(self):
        while self.server.hasPendingConnections():
            sock = self.server.nextPendingConnection()
            self.clients.append(sock)
            self._buffers[sock] = b""
            sock.readyRead.connect(lambda s=sock: self._read(s))
            sock.disconnected.connect(lambda s=sock: self._gone(s))
            self.clientsChanged.emit(len(self.clients))

    def _gone(self, sock):
        if sock in self.clients:
            self.clients.remove(sock)
        self._buffers.pop(sock, None)
        if self._pending and self._pending["sock"] is sock:
            self._pending["sock"] = None      # finish the run, just nobody to tell
        sock.deleteLater()
        self.clientsChanged.emit(len(self.clients))

    def _send(self, sock, line):
        if sock is not None and sock in self.clients:
            sock.write((line + "\r\n").encode("ascii", "replace"))
            sock.flush()

    def _read(self, sock):
        self._buffers[sock] = self._buffers.get(sock, b"") + bytes(sock.readAll())
        while b"\n" in self._buffers[sock]:
            raw, self._buffers[sock] = self._buffers[sock].split(b"\n", 1)
            line = raw.decode("ascii", "replace").strip()
            if line:
                self.commandReceived.emit(line)
                self._handle(sock, line)

    # ------------------------------------------------------------ commands
    def _job(self, name):
        """A job by display name, short id or full id; None if unknown."""
        for j in self.station.jobs:
            short = j["pid"].split("/")[-1]
            if name in (j["name"], short, j["pid"]) or name.lower() in (
                    j["name"].lower(), short.lower()):
                return j["pid"]
        return None

    def _handle(self, sock, line):
        cmd, _, arg = line.partition(" ")
        cmd, arg = cmd.upper(), arg.strip()
        if cmd == "PING":
            self._send(sock, "PONG")
        elif cmd == "VERSION":
            self._send(sock, "VISIONSTATION %d" % PROTOCOL_VERSION)
        elif cmd == "STATUS":
            self._send(sock, "BUSY" if self.station.running else "READY")
        elif cmd in ("TRIGGER", "T"):
            self._trigger(sock, arg)
        elif cmd == "GET":
            self._get(sock, arg)
        elif cmd == "CODE":
            # the part's barcode from the line, for barcode-gated inspection
            gate = getattr(self.station, "barcode", None)
            if gate is None or not gate.enabled:
                self._send(sock, "ERROR barcode checking is off")
            else:
                ok, code, why = gate.scan(arg)
                self._send(sock, ("CODE OK %s" % code) if ok else ("CODE NG %s" % why))
                if ok:
                    self.codeScanned.emit(code)
        else:
            self._send(sock, "ERROR unknown command: %s" % cmd)

    def _trigger(self, sock, arg):
        if self.station.running or self._pending is not None:
            self._send(sock, "BUSY")
            return
        if arg:
            pid = self._job(arg)
            if pid is None:
                self._send(sock, "ERROR no job %s" % arg)
                return
            pids = [pid]
        else:
            pids = [j["pid"] for j in self.station.jobs]
        if not pids:
            self._send(sock, "ERROR no application open")
            return
        gate = getattr(self.station, "barcode", None)
        if gate is not None and gate.enabled and gate.pending is None:
            self._send(sock, "ERROR scan the barcode first, or send CODE <barcode>")
            return
        self._pending = {"sock": sock, "pids": pids}
        self._run = {"results": {}, "errors": {}}
        started = self.station.run_once(pids)
        if not started:
            self._finish()                     # nothing could be acquired at all

    def _get(self, sock, arg):
        parts = arg.split(".")
        if len(parts) < 3:
            self._send(sock, "ERROR use GET <job>.<tool>.<value>")
            return
        pid = self._job(parts[0])
        res = self.station.last.get(pid) if pid else None
        tool = next((t for t in (res or {}).get("tools", []) if t["tool_id"] == parts[1]), None)
        key = ".".join(parts[2:])
        if tool is None:
            self._send(sock, "ERROR no result for %s.%s" % (parts[0], parts[1]))
            return
        val = tool.get(key) if key in ("status", "score", "message") else \
            (tool.get("values") or {}).get(key)
        if val is None or isinstance(val, (list, dict)):
            self._send(sock, "ERROR no value %s" % arg)
        else:
            self._send(sock, "VALUE %s" % (("%.6g" % val) if isinstance(val, float) else val))

    # ------------------------------------------------------------ results
    @staticmethod
    def _short(pid):
        return pid.split("/")[-1]

    def _result_line(self, pid, res):
        failed = [t["tool_id"] for t in res["tools"] if t["status"] != "pass"]
        return "RESULT %s %s %.0f %s" % (self._short(pid), res["verdict"],
                                         res.get("total_ms", 0.0), ",".join(failed) or "-")

    def _on_result(self, pid, res):
        self._run["results"][pid] = res
        if self.broadcast:
            for sock in self.clients:
                if not (self._pending and sock is self._pending["sock"]):
                    self._send(sock, self._result_line(pid, res))

    def _on_error(self, pid, msg):
        self._run["errors"][pid] = msg

    def _on_finished(self):
        if self._pending is not None:
            self._finish()
        elif self.broadcast and self._run["results"]:
            verdict = self._verdict(list(self._run["results"]),
                                    self._run["results"], self._run["errors"])
            for sock in self.clients:
                self._send(sock, "DONE %s %d" % (verdict, len(self._run["results"])))
            self._run = {"results": {}, "errors": {}}

    @staticmethod
    def _verdict(pids, results, errors):
        if any(p in errors or p not in results for p in pids):
            return "INCOMPLETE"
        return "PASS" if all(results[p]["verdict"] == "PASS" for p in pids) else "FAIL"

    def _finish(self):
        pend, self._pending = self._pending, None
        results, errors = self._run["results"], self._run["errors"]
        for pid in pend["pids"]:
            if pid in results:
                self._send(pend["sock"], self._result_line(pid, results[pid]))
            else:
                why = errors.get(pid, "not inspected").replace(" ", "-")[:60]
                self._send(pend["sock"], "RESULT %s ERROR 0 %s" % (self._short(pid), why))
        verdict = self._verdict(pend["pids"], results, errors)
        self._send(pend["sock"], "DONE %s %d" % (verdict, len(pend["pids"])))
        if self.broadcast:
            for sock in self.clients:
                if sock is not pend["sock"]:
                    self._send(sock, "DONE %s %d" % (verdict, len(pend["pids"])))
        self._run = {"results": {}, "errors": {}}
