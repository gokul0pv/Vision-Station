"""Result logging and traceability.

Until now results lived only in memory and a sidecar JSON, which cannot answer
the question every real line eventually asks: what did we ship on Tuesday, and
which check was responsible for the rejects?

Two design constraints shaped this:

**Writing must not slow the cycle.** An inspection can run every 120 ms. A
synchronous commit on that path would show up as jitter in the cycle time and
occasionally as a missed frame. So results go onto a queue and a single
background thread does the committing. If the writer ever falls behind, the
queue drops the oldest entries rather than blocking the inspection - a late
inspection is worse than an incomplete log.

**Disk is finite.** Saving every image fills a drive in days. Default is to
archive only failures, capped, oldest deleted first.
"""

import json
import os
import queue
import sqlite3
import threading
import time
from datetime import datetime

import cv2

from vision_station.core import imio

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_uid     TEXT,
    ts          REAL    NOT NULL,
    iso         TEXT    NOT NULL,
    project     TEXT    NOT NULL,
    verdict     TEXT    NOT NULL,
    total_ms    REAL,
    source      TEXT,
    image_path  TEXT,
    operator    TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_ts      ON runs(ts);
CREATE INDEX IF NOT EXISTS idx_runs_verdict ON runs(verdict);

CREATE TABLE IF NOT EXISTS tool_results (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    tool_id  TEXT    NOT NULL,
    type     TEXT,
    status   TEXT,
    score    REAL,
    ms       REAL,
    message  TEXT,
    values_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_tr_run    ON tool_results(run_id);
CREATE INDEX IF NOT EXISTS idx_tr_tool   ON tool_results(tool_id, status);
"""


class ResultStore:
    def __init__(self, project_folder, project_id, archive="none",
                 max_archived=2000, queue_size=512):
        """archive: "none" | "fail" | "all" """
        self.project_id = project_id
        self.folder = project_folder
        self.dir = os.path.join(project_folder, "results")
        os.makedirs(self.dir, exist_ok=True)
        self.images_dir = os.path.join(self.dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.path = os.path.join(self.dir, "results.db")
        self.archive = archive
        self.max_archived = max_archived

        self._q = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._dropped = 0
        self._written = 0
        self._archived_since_prune = 0
        self._init_db()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    # -- setup -----------------------------------------------------------
    def _connect(self):
        c = sqlite3.connect(self.path, timeout=5.0)
        # WAL lets the UI read while the writer thread is committing.
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        return c

    def _init_db(self):
        c = self._connect()
        try:
            c.executescript(SCHEMA)
            c.commit()
        finally:
            c.close()

    # -- write path ------------------------------------------------------
    def log(self, result, image=None, operator="", image_path=None):
        """Called from the inspection thread. Never blocks, never raises.
        image_path: where the part's picture was saved (Settings > Results);
        it is recorded instead of keeping a copy here -- every picture kept
        twice, once in the chosen folder and once in the job's folder, was
        458 MB on one job in two days."""
        try:
            payload = (result, image if (image_path is None and self._want_image(result))
                       else None, operator, image_path)
            self._q.put_nowait(payload)
        except queue.Full:
            # Drop the oldest so a stalled disk cannot back-pressure inspection.
            try:
                self._q.get_nowait()
                self._q.put_nowait(payload)
            except (queue.Empty, queue.Full):
                pass
            self._dropped += 1

    def _want_image(self, result):
        if self.archive == "all":
            return True
        if self.archive == "fail":
            return result.get("verdict") != "PASS"
        return False

    def _writer(self):
        conn = self._connect()
        try:
            while not self._stop.is_set() or not self._q.empty():
                try:
                    result, image, operator, saved = self._q.get(timeout=0.25)
                except queue.Empty:
                    continue
                try:
                    self._write_one(conn, result, image, operator, saved)
                    self._written += 1
                except Exception as e:            # logging must never kill the app
                    print("[results] write failed: %s" % e)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _write_one(self, conn, result, image, operator, saved=None):
        ts = time.time()
        iso = datetime.fromtimestamp(ts).isoformat(timespec="seconds")
        img_path = saved or ""
        if image is not None:
            day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            d = os.path.join(self.images_dir, day)
            os.makedirs(d, exist_ok=True)
            name = "%s_%s_%s.jpg" % (datetime.fromtimestamp(ts).strftime("%H%M%S_%f")[:-3],
                                     result.get("verdict", "NA"),
                                     result.get("id", "")[:8])
            full = os.path.join(d, name)
            # the run is logged either way; a path is recorded only for an
            # image that was actually written
            if imio.imwrite(full, image, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
                img_path = os.path.relpath(full, self.folder)
                self._archived_since_prune += 1

        cur = conn.execute(
            "INSERT INTO runs (run_uid, ts, iso, project, verdict, total_ms, source,"
            " image_path, operator) VALUES (?,?,?,?,?,?,?,?,?)",
            (result.get("id", ""), ts, iso, self.project_id,
             result.get("verdict", ""), float(result.get("total_ms", 0.0)),
             result.get("source", ""), img_path, operator))
        rid = cur.lastrowid
        conn.executemany(
            "INSERT INTO tool_results (run_id, tool_id, type, status, score, ms,"
            " message, values_json) VALUES (?,?,?,?,?,?,?,?)",
            [(rid, t.get("tool_id", ""), t.get("type", ""), t.get("status", ""),
              float(t.get("score", 0.0)), float(t.get("ms", 0.0)),
              t.get("message", ""), json.dumps(t.get("values", {}), default=str))
             for t in result.get("tools", [])])
        conn.commit()
        # Prune on images archived, not on writes. Counting writes let the cap
        # overshoot badly whenever most runs passed and archived nothing.
        if self._archived_since_prune >= 25:
            self._prune_images()

    def _prune_images(self):
        """Oldest archived images go first once the cap is reached."""
        try:
            files = []
            for root, _dirs, names in os.walk(self.images_dir):
                for n in names:
                    if n.lower().endswith((".jpg", ".png")):
                        p = os.path.join(root, n)
                        files.append((os.path.getmtime(p), p))
            self._archived_since_prune = 0
            if len(files) <= self.max_archived:
                return
            files.sort()
            for _mt, p in files[:len(files) - self.max_archived]:
                try:
                    os.remove(p)
                except OSError:
                    pass
            self._archived_since_prune = 0
        except Exception:
            pass

    def flush(self, timeout=3.0):
        """Wait for the queue to drain. Used before showing history or exiting."""
        end = time.time() + timeout
        while not self._q.empty() and time.time() < end:
            time.sleep(0.02)
        time.sleep(0.05)
        self._prune_images()      # make the cap exact once the burst is over

    def close(self):
        self._stop.set()
        self._thread.join(timeout=4.0)

    def stats(self):
        return {"written": self._written, "dropped": self._dropped,
                "pending": self._q.qsize()}

    # -- read path -------------------------------------------------------
    def recent(self, limit=200):
        c = self._connect()
        try:
            rows = c.execute(
                "SELECT id, iso, verdict, total_ms, source, image_path FROM runs"
                " ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        finally:
            c.close()
        return [{"id": r[0], "iso": r[1], "verdict": r[2], "total_ms": r[3],
                 "source": r[4], "image_path": r[5]} for r in rows]

    def failed_tools_for(self, run_id):
        c = self._connect()
        try:
            rows = c.execute(
                "SELECT tool_id, status, score, message FROM tool_results"
                " WHERE run_id=? AND status IN ('fail','error')", (run_id,)).fetchall()
        finally:
            c.close()
        return [{"tool_id": r[0], "status": r[1], "score": r[2], "message": r[3]}
                for r in rows]

    def summary(self, since_hours=None):
        where, args = "", []
        if since_hours:
            where = " WHERE ts >= ?"
            args = [time.time() - since_hours * 3600]
        c = self._connect()
        try:
            total = c.execute("SELECT COUNT(*) FROM runs" + where, args).fetchone()[0]
            passed = c.execute("SELECT COUNT(*) FROM runs" + (where + " AND" if where
                               else " WHERE") + " verdict='PASS'", args).fetchone()[0]
            avg = c.execute("SELECT AVG(total_ms) FROM runs" + where, args).fetchone()[0]
        finally:
            c.close()
        return {"total": total, "passed": passed,
                "failed": total - passed,
                "yield_pct": (100.0 * passed / total) if total else 0.0,
                "avg_ms": round(avg or 0.0, 1)}

    def pareto(self, since_hours=None, limit=12):
        """Which check is rejecting parts, most first.

        pct_of_rejects is the share of REJECTED PARTS in which this check
        failed, so the column can total well over 100% - one bad part commonly
        trips several checks. That is the honest framing; presenting it as a
        share of blame would imply a single cause that usually is not there."""
        q = ("SELECT tr.tool_id, COUNT(*) FROM tool_results tr"
             " JOIN runs r ON r.id = tr.run_id"
             " WHERE tr.status IN ('fail','error')")
        args = []
        if since_hours:
            q += " AND r.ts >= ?"
            args.append(time.time() - since_hours * 3600)
        q += " GROUP BY tr.tool_id ORDER BY COUNT(*) DESC LIMIT ?"
        args.append(limit)
        c = self._connect()
        try:
            rows = c.execute(q, args).fetchall()
            tot = c.execute("SELECT COUNT(*) FROM runs WHERE verdict!='PASS'"
                            + (" AND ts >= ?" if since_hours else ""),
                            ([time.time() - since_hours * 3600] if since_hours else []
                             )).fetchone()[0]
        finally:
            c.close()
        return [{"tool_id": r[0], "count": r[1],
                 "pct_of_rejects": round(100.0 * r[1] / tot, 1) if tot else 0.0}
                for r in rows]

    def hourly_yield(self, hours=24):
        c = self._connect()
        try:
            rows = c.execute(
                "SELECT strftime('%Y-%m-%d %H', datetime(ts,'unixepoch','localtime')),"
                " COUNT(*), SUM(CASE WHEN verdict='PASS' THEN 1 ELSE 0 END)"
                " FROM runs WHERE ts >= ? GROUP BY 1 ORDER BY 1",
                (time.time() - hours * 3600,)).fetchall()
        finally:
            c.close()
        return [{"hour": r[0], "total": r[1], "passed": r[2],
                 "yield_pct": round(100.0 * r[2] / r[1], 1) if r[1] else 0.0}
                for r in rows]

    def export_csv(self, path, since_hours=None):
        import csv
        q = ("SELECT r.iso, r.verdict, r.total_ms, r.source, r.image_path,"
             " tr.tool_id, tr.status, tr.score, tr.message"
             " FROM runs r LEFT JOIN tool_results tr ON tr.run_id = r.id")
        args = []
        if since_hours:
            q += " WHERE r.ts >= ?"
            args.append(time.time() - since_hours * 3600)
        q += " ORDER BY r.ts DESC"
        c = self._connect()
        try:
            rows = c.execute(q, args).fetchall()
        finally:
            c.close()
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["timestamp", "verdict", "cycle_ms", "source", "image",
                         "tool", "tool_status", "score", "message"])
            wr.writerows(rows)
        return len(rows)

    def purge_older_than(self, days):
        cutoff = time.time() - days * 86400
        c = self._connect()
        try:
            ids = [r[0] for r in c.execute("SELECT id FROM runs WHERE ts < ?",
                                           (cutoff,)).fetchall()]
            c.execute("DELETE FROM tool_results WHERE run_id IN"
                      " (SELECT id FROM runs WHERE ts < ?)", (cutoff,))
            c.execute("DELETE FROM runs WHERE ts < ?", (cutoff,))
            c.commit()
            c.execute("VACUUM")
        finally:
            c.close()
        return len(ids)
