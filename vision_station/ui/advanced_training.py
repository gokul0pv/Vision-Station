"""The advanced training window: the job's own AI model being trained
(core/advanced.py), shown over the main window.

While it trains, everything behind it is blurred and takes no key or mouse:
the window is application-modal, has no close button, and Escape or Alt+F4
ask to cancel rather than close it. It shows the stage, a progress bar with
the share done, the time taken and the time left, and a Cancel button --
cancelling stops the training and deletes everything it made. At the end it
shows the training's results in per cent, or why it stopped.

    run_training(main_window, session, job_name) -> "ok" | "cancelled" | "failed"
"""
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QDialog, QGraphicsBlurEffect, QGridLayout, QHBoxLayout, QLabel,
                               QMessageBox, QProgressBar, QPushButton, QToolBar, QVBoxLayout,
                               QWidget, QDockWidget)

POLL_MS = 200


def fmt_time(s):
    if s is None:
        return "-"
    s = int(max(0, round(s)))
    if s >= 3600:
        return "%d h %02d min" % (s // 3600, (s % 3600) // 60)
    return "%02d:%02d" % (s // 60, s % 60)


def pct(v):
    return "-" if v is None else "%.1f %%" % (100.0 * float(v))


class Blur:
    """Everything of a main window blurred, and given back."""

    def __init__(self, win, radius=14):
        self.effects = []
        if win is None:
            return
        parts = [win.centralWidget(), win.menuBar() if hasattr(win, "menuBar") else None,
                 win.statusBar() if hasattr(win, "statusBar") else None]
        parts += win.findChildren(QToolBar) + win.findChildren(QDockWidget)
        for w in parts:
            if w is None or not w.isVisible() or w.parent() is not win:
                continue
            e = QGraphicsBlurEffect(w)
            e.setBlurRadius(radius)
            w.setGraphicsEffect(e)
            self.effects.append(w)

    def remove(self):
        for w in self.effects:
            try:
                w.setGraphicsEffect(None)
            except RuntimeError:
                pass
        self.effects = []


class AdvancedTrainingDialog(QDialog):
    def __init__(self, parent, session, job_name=""):
        super().__init__(parent)
        self.session = session
        self.outcome = None               # "ok" | "cancelled" | "failed"
        self._cancelling = False
        self.setObjectName("advancedTrainingDialog")
        self.setWindowTitle("Advanced training")
        self.setWindowModality(Qt.ApplicationModal)
        # no close button: it is cancelled, never just closed
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 20)
        lay.setSpacing(8)
        self.title = QLabel("Advanced training is going on")
        f = self.title.font()
        f.setPointSizeF(f.pointSizeF() * 1.35)
        f.setBold(True)
        self.title.setFont(f)
        lay.addWidget(self.title)
        self.job = QLabel(("Job: %s" % job_name) if job_name else "")
        self.job.setObjectName("rowHelp")
        lay.addWidget(self.job)
        lay.addSpacing(6)
        self.stage = QLabel("Starting...")
        f = self.stage.font()
        f.setBold(True)
        self.stage.setFont(f)
        lay.addWidget(self.stage)
        self.bar = QProgressBar()
        self.bar.setObjectName("advancedProgress")
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setTextVisible(True)
        self.bar.setFormat("0 %")
        self.bar.setMinimumHeight(26)
        lay.addWidget(self.bar)
        row = QHBoxLayout()
        self.elapsed = QLabel("Time taken 00:00")
        self.left = QLabel("Time left: estimating...")
        self.left.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.elapsed)
        row.addStretch(1)
        row.addWidget(self.left)
        lay.addLayout(row)
        self.detail = QLabel("")
        self.detail.setObjectName("rowHelp")
        self.detail.setWordWrap(True)
        lay.addWidget(self.detail)
        self.note = QLabel("Keep the part under the camera and do not move it until the "
                           "pictures are taken.")
        self.note.setWordWrap(True)
        self.note.setObjectName("rowHelp")
        lay.addWidget(self.note)
        # the results, shown at the end
        self.results = QWidget()
        self.grid = QGridLayout(self.results)
        self.grid.setContentsMargins(0, 8, 0, 0)
        self.grid.setHorizontalSpacing(24)
        self.results.hide()
        lay.addWidget(self.results)
        self.extra = QLabel("")
        self.extra.setWordWrap(True)
        self.extra.setObjectName("rowHelp")
        self.extra.hide()
        lay.addWidget(self.extra)
        lay.addSpacing(8)
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.cancelBtn = QPushButton("Cancel")
        self.cancelBtn.setMinimumWidth(110)
        self.cancelBtn.clicked.connect(self.ask_cancel)
        self.okBtn = QPushButton("OK")
        self.okBtn.setObjectName("primaryButton")
        self.okBtn.setMinimumWidth(110)
        self.okBtn.clicked.connect(self.accept)
        self.okBtn.hide()
        btns.addWidget(self.cancelBtn)
        btns.addWidget(self.okBtn)
        lay.addLayout(btns)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(POLL_MS)
        self._blur = None

    # -- showing -----------------------------------------------------------
    def showEvent(self, e):
        if self._blur is None:
            self._blur = Blur(self.parent())
        super().showEvent(e)

    def done(self, r):
        self.timer.stop()
        if self._blur is not None:
            self._blur.remove()
        super().done(r)

    def poll(self):
        st = self.session.snapshot()
        now = time.time()
        p = float(st.get("progress") or 0.0)
        self.bar.setValue(int(round(1000 * p)))
        self.bar.setFormat("%d %%" % int(100 * p))
        self.elapsed.setText("Time taken %s" % fmt_time(now - st["started"]))
        if not st["done"]:
            eta = st.get("eta_s")
            self.left.setText("Time left: about %s" % fmt_time(eta) if eta is not None
                              else "Time left: estimating...")
            if not self._cancelling:
                self.stage.setText(st.get("message") or "")
                self.detail.setText(st.get("detail") or "")
            self.note.setVisible(st.get("stage") == "capture" and not self._cancelling)
            self.cancelBtn.setEnabled(bool(st.get("can_cancel")) and not self._cancelling)
            return
        self.timer.stop()
        if st.get("cancelled"):
            self.outcome = "cancelled"
            self.reject()
        elif st.get("ok"):
            self._finished(st)
        else:
            self._failed(st)

    def _finished(self, st):
        self.outcome = "ok"
        r = st.get("result") or {}
        m = r.get("metrics") or {}
        v = r.get("validation") or {}
        self.title.setText("Training completed")
        self.stage.setText("The job's own AI model is trained and in use.")
        self.bar.setValue(1000)
        self.bar.setFormat("100 %")
        self.left.setText("Took %s" % fmt_time(r.get("seconds")))
        self.detail.setText("")
        self.note.hide()
        # short names, the meaning in the tooltip: the long ones were cut off
        rows = [("Precision", pct(m.get("precision")), "Of what the AI found, the share that is "
                 "right"),
                ("Recall", pct(m.get("recall")), "Of the parts and text, the share the AI finds"),
                ("mAP50", pct(m.get("map50")), "Detection overall"),
                ("Found on new pictures", pct(v.get("found_share")),
                 "Every learned check looked for on the pictures just taken")]
        if v.get("missing_tried"):
            rows.append(("Missing-part test", "%d of %d caught (%s)" % (
                v["missing_caught"], v["missing_tried"],
                pct(v["missing_caught"] / float(v["missing_tried"]))),
                "Checks' regions painted over: the AI must not find a part there"))
        n = r.get("learned", 0)
        tot = r.get("checks", 0)
        rows.append(("AI-confirmed checks", "%d of %d (%s)" % (
            n, tot, pct(n / float(tot)) if tot else "-"), "The others keep their classic test"))
        rows.append(("Good pictures added", "%d per check" % r.get("pictures", 0),
                     "Added to each check's Train tab as good examples"))
        for i, (k, val, tip) in enumerate(rows):
            a, b = QLabel(k), QLabel(val)
            a.setToolTip(tip)
            b.setToolTip(tip)
            b.setObjectName("advancedResult")
            f = b.font()
            f.setBold(True)
            b.setFont(f)
            self.grid.addWidget(a, i, 0)
            self.grid.addWidget(b, i, 1, Qt.AlignRight)
        self.results.show()
        notes = []
        left = dict(r.get("skipped") or {})
        for tid in r.get("not_learned") or []:
            left[tid] = "not found reliably on the new pictures"
        if left:
            items = sorted(left.items())
            notes.append("%d check%s left to the classic test: %s%s." % (
                len(items), "" if len(items) == 1 else "s",
                ", ".join("%s (%s)" % kv for kv in items[:4]),
                ", ..." if len(items) > 4 else ""))
        failing = r.get("failing") or {}
        if failing:
            notes.append("Failed on the new pictures, so not taught from them - check these "
                         "parts on the board: %s%s." % (", ".join(sorted(failing)[:6]),
                                                         ", ..." if len(failing) > 6 else ""))
        if notes:
            self.extra.setText("\n".join(notes))
            self.extra.show()
        self.cancelBtn.hide()
        self.okBtn.show()
        self.okBtn.setFocus()
        self.adjustSize()

    def _failed(self, st):
        self.outcome = "failed"
        self.title.setText("Advanced training stopped")
        self.stage.setText(st.get("error") or "It stopped.")
        self.stage.setWordWrap(True)
        self.detail.setText("Nothing was changed: everything it made was deleted, and the job "
                            "runs with its classic checks.")
        self.left.setText("")
        self.note.hide()
        self.cancelBtn.hide()
        self.okBtn.setText("Close")
        self.okBtn.show()
        self.okBtn.setFocus()
        self.adjustSize()

    # -- cancelling --------------------------------------------------------
    def ask_cancel(self):
        if self._cancelling or self.session.state.get("done"):
            return
        if QMessageBox.question(
                self, "Cancel advanced training",
                "Stop the advanced training?\n\nEverything it has made so far is deleted, and "
                "the job runs with its classic checks.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.cancel_now()

    def cancel_now(self):
        """Cancelled without asking (tests, and the confirmed button)."""
        if not self.session.cancel():
            # it finished -- or is saving the model -- while the question was
            # open: too late to cancel, and the person is told so
            self.detail.setText("The training had already finished - the model is kept. "
                                "Remove it from the job's menu if it is not wanted.")
            return False
        self._cancelling = True
        self.cancelBtn.setEnabled(False)
        self.stage.setText("Cancelling - deleting what it made...")
        self.detail.setText("")
        if not self.timer.isActive():
            self.timer.start(POLL_MS)
        return True

    def reject(self):
        if self.outcome is None and not self.session.state.get("done"):
            self.ask_cancel()                # Escape, Alt+F4: a cancel, asked
            return
        if self.outcome is None:
            self.outcome = "failed"
        super().reject()

    def accept(self):
        if self.outcome is None:
            return
        super().accept()

    def closeEvent(self, e):
        if self.outcome is None:
            e.ignore()
            self.ask_cancel()
            return
        super().closeEvent(e)


def run_training(win, session, job_name=""):
    dlg = AdvancedTrainingDialog(win, session, job_name)
    session.start()
    dlg.exec()
    return dlg.outcome or "failed"
