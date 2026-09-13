"""Calibration dialog: walks the user through the poses and saves a profile.

Opens the camera itself (the live preview must be closed first — a camera
cannot be held twice), runs detection on a worker thread, and shows the face
outline plus the pose readouts while each step is captured.  One button
walks through every pose with a countdown before each capture; the per-step
buttons remain for redoing a single one.  All the logic lives in
``modules.calibration``; this file is only the Qt around it.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import modules.globals
from modules import calibration
from modules.gettext import _
from modules.face_analyser import detect_one_face_fast, ensure_landmarks
from modules.video_capture import VideoCapturer

CAPTURE_WIDTH = 960
CAPTURE_HEIGHT = 540
# Seconds to get into the pose before a guided capture starts.
COUNTDOWN_SECONDS = 3
STEP_LABELS = {
    "neutral": "Neutral",
    "left": "Turn left",
    "right": "Turn right",
    "up": "Tilt up",
    "down": "Tilt down",
}


class _CalibrationWorker(QThread):
    """Reads the camera, detects the face, feeds the capture session."""

    frame_ready = Signal(object, object, object)   # frame, face, completed step

    def __init__(self, cap: VideoCapturer, session: calibration.CalibrationSession,
                 stop_event: threading.Event) -> None:
        super().__init__()
        self._cap = cap
        self._session = session
        self._stop = stop_event

    def run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.msleep(5)
                continue
            if modules.globals.live_mirror:
                frame = cv2.flip(frame, 1)
            face = detect_one_face_fast(frame)
            if face is not None:
                ensure_landmarks(frame, [face])
            completed = self._session.feed(face)
            self.frame_ready.emit(frame, face, completed)


def _draw_overlay(frame: np.ndarray, face, profile_kps_text: str, cue: str = "") -> np.ndarray:
    """Face outline, keypoints and the pose readout on a copy of the frame;
    ``cue`` is drawn large in the centre (countdown, "hold still")."""
    out = frame.copy()
    if cue:
        h, w = out.shape[:2]
        scale = max(1.0, h / 240.0)
        (tw, th), _ = cv2.getTextSize(cue, cv2.FONT_HERSHEY_DUPLEX, scale, 3)
        cv2.putText(out, cue, ((w - tw) // 2, h // 2 + th // 2),
                    cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 6)
        cv2.putText(out, cue, ((w - tw) // 2, h // 2 + th // 2),
                    cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 3)
    if face is None:
        cv2.putText(out, "No face", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 230), 2)
        return out
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is not None:
        hull = cv2.convexHull(np.asarray(landmarks, dtype=np.int32))
        cv2.polylines(out, [hull], True, (80, 220, 80), 2)
    for x, y in np.asarray(face.kps, dtype=np.int32):
        cv2.circle(out, (int(x), int(y)), 3, (255, 200, 60), -1)
    cv2.putText(out, profile_kps_text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 80), 2)
    return out


def _to_pixmap(frame: np.ndarray, width: int, height: int) -> QPixmap:
    h, w = frame.shape[:2]
    scale = min(width / float(w), height / float(h), 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class CalibrationDialog(QDialog):
    """Five poses, one profile.  ``on_saved`` receives the profile name."""

    def __init__(self, camera_index: int, on_saved: Callable[[str], None],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Face calibration"))
        self.setModal(False)
        self._on_saved = on_saved
        self._session = calibration.CalibrationSession()
        self._stop_event = threading.Event()
        self._worker: Optional[_CalibrationWorker] = None
        self._cap: Optional[VideoCapturer] = None
        self._last_face = None
        self._step_status: Dict[str, QLabel] = {}
        self._step_buttons: Dict[str, QPushButton] = {}
        self._flow: list = []             # steps still to capture in guided mode
        self._countdown = 0
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._on_countdown)

        screen = QApplication.primaryScreen().availableGeometry() if QApplication.primaryScreen() else None
        width = min(1100, int(screen.width() * 0.8)) if screen else 1000
        height = min(680, int(screen.height() * 0.8)) if screen else 620
        self.resize(width, height)

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        self._preview = QLabel(_("Opening camera..."))
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview.setMinimumSize(320, 180)
        self._preview.setObjectName("imageDrop")
        root.addWidget(self._preview, 3)

        root.addWidget(self._build_panel(), 2)

        self._cap = VideoCapturer(camera_index)
        if not self._cap.start(CAPTURE_WIDTH, CAPTURE_HEIGHT, 30,
                               on_wait=QApplication.processEvents):
            self._preview.setText(_("Could not open the camera — close the live preview "
                                    "or any app that is using it, then try again."))
            self._cap = None
            return
        self._worker = _CalibrationWorker(self._cap, self._session, self._stop_event)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.start()

    # ── layout ───────────────────────────────────────────────────────────

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        intro = QLabel(_(
            "Press Start: you get a few seconds to take each pose — straight, "
            "then the furthest turn left, right, up and down at which the swap "
            "should still hold.  Past those the real face is shown instead of a "
            "smeared mask.  About a minute in total."
        ))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        guided = QHBoxLayout()
        self._start = QPushButton(_("Start"))
        self._start.setToolTip(_("Walk through all five poses with a countdown before each"))
        self._start.clicked.connect(self._start_flow)
        self._skip = QPushButton(_("Skip this pose"))
        self._skip.setObjectName("secondary")
        self._skip.setEnabled(False)
        self._skip.clicked.connect(self._skip_step)
        guided.addWidget(self._start)
        guided.addWidget(self._skip)
        layout.addLayout(guided)

        self._prompt = QLabel("")
        self._prompt.setWordWrap(True)
        self._prompt.setObjectName("statusLabel")
        layout.addWidget(self._prompt)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        for row, step in enumerate(calibration.STEPS):
            grid.addWidget(QLabel(_(STEP_LABELS[step])), row, 0)
            status = QLabel("—")
            status.setMinimumWidth(24)
            grid.addWidget(status, row, 1)
            button = QPushButton(_("Capture"))
            button.setObjectName("secondary")
            button.clicked.connect(lambda _checked=False, s=step: self._arm(s))
            grid.addWidget(button, row, 2)
            self._step_status[step] = status
            self._step_buttons[step] = button
        layout.addLayout(grid)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        fade_row = QHBoxLayout()
        fade_row.addWidget(QLabel(_("Fade beyond limit")))
        self._fade = QSlider(Qt.Orientation.Horizontal)
        self._fade.setRange(10, 60)
        self._fade.setValue(int(calibration.DEFAULT_FADE_SPAN * 100))
        self._fade.setToolTip(_("How far past the captured turn the swap keeps fading "
                                "before the real face shows fully"))
        self._fade_value = QLabel(f"+{self._fade.value()}%")
        self._fade.valueChanged.connect(lambda v: self._fade_value.setText(f"+{v}%"))
        fade_row.addWidget(self._fade, 1)
        fade_row.addWidget(self._fade_value)
        layout.addLayout(fade_row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel(_("Profile name")))
        self._name = QLineEdit(modules.globals.calibration_profile or "me")
        name_row.addWidget(self._name, 1)
        layout.addLayout(name_row)

        layout.addStretch(1)

        buttons = QHBoxLayout()
        self._save = QPushButton(_("Save && use"))
        self._save.setEnabled(False)
        self._save.clicked.connect(self._on_save)
        cancel = QPushButton(_("Cancel"))
        cancel.setObjectName("secondary")
        cancel.clicked.connect(self.close)
        buttons.addWidget(self._save)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        self._set_prompt(_("Press Start for the guided capture, or Capture next to one pose."))
        return panel

    # ── guided flow ──────────────────────────────────────────────────────

    def _start_flow(self) -> None:
        if self._worker is None:
            return
        self._flow = list(calibration.STEPS)
        self._start.setEnabled(False)
        self._next_in_flow()

    def _next_in_flow(self) -> None:
        if not self._flow:
            self._start.setEnabled(True)
            self._skip.setEnabled(False)
            self._set_prompt(_("All poses done — name the profile and press Save."))
            return
        step = self._flow[0]
        self._set_prompt(_(calibration.STEP_PROMPTS[step]))
        self._skip.setEnabled(step != "neutral")
        self._countdown = COUNTDOWN_SECONDS
        self._countdown_timer.start()

    def _on_countdown(self) -> None:
        self._countdown -= 1
        if self._countdown <= 0:
            self._countdown_timer.stop()
            if self._flow:
                self._arm(self._flow[0])

    def _skip_step(self) -> None:
        if not self._flow or self._flow[0] == "neutral":
            return
        self._countdown_timer.stop()
        self._session.cancel()
        self._flow.pop(0)
        for button in self._step_buttons.values():
            button.setEnabled(True)
        self._next_in_flow()

    def _cue(self) -> str:
        if self._countdown_timer.isActive():
            return str(self._countdown)
        if self._session.armed is not None:
            return _("hold still")
        return ""

    # ── behaviour ────────────────────────────────────────────────────────

    def _set_prompt(self, text: str) -> None:
        self._prompt.setText(text)

    def _arm(self, step: str) -> None:
        if self._worker is None:
            return
        self._session.arm(step)
        self._progress.setValue(0)
        self._set_prompt(_(calibration.STEP_PROMPTS[step]) + "  " + _("Hold still..."))
        for name, button in self._step_buttons.items():
            button.setEnabled(name == step)
        self._step_buttons[step].setText(_("Capturing"))

    def _on_frame(self, frame, face, completed) -> None:
        self._last_face = face
        text = ""
        if face is not None:
            yaw, pitch = calibration.pose_proxies(face.kps)
            text = f"yaw {yaw:+.2f}  pitch {pitch:.2f}"
            neutral = self._session.captured.get("neutral")
            if neutral is not None:
                text += f"  (neutral {neutral['pose'][0]:+.2f} / {neutral['pose'][1]:.2f})"
        overlay = _draw_overlay(frame, face, text, self._cue())
        self._preview.setPixmap(_to_pixmap(overlay, self._preview.width(), self._preview.height()))

        if self._session.armed is not None:
            self._progress.setValue(int(self._session.progress() * 100))
            if face is None:
                self._set_prompt(_("No face in view — the capture waits for one."))
        if completed is not None:
            self._progress.setValue(100)
            self._step_status[completed].setText("✓")
            for name, button in self._step_buttons.items():
                button.setEnabled(True)
                button.setText(_("Redo") if name in self._session.captured else _("Capture"))
            self._save.setEnabled(self._session.can_build())
            if self._flow and self._flow[0] == completed:
                self._flow.pop(0)
                self._set_prompt(_("{step} captured.").format(step=_(STEP_LABELS[completed])))
                QTimer.singleShot(700, self._next_in_flow)
            else:
                nxt = next((s for s in calibration.STEPS if s not in self._session.captured), None)
                self._set_prompt(_("{step} captured.").format(step=_(STEP_LABELS[completed]))
                                 + ("  " + _("Next: {step}.").format(step=_(STEP_LABELS[nxt]).lower())
                                    if nxt else "  " + _("Save when ready.")))

    def _on_save(self) -> None:
        name = self._name.text().strip() or "me"
        try:
            profile = self._session.build(name, fade_span=self._fade.value() / 100.0)
            calibration.save_profile(profile)
        except (ValueError, OSError) as error:
            self._set_prompt(_("Could not save: {error}").format(error=error))
            return
        calibration.set_active(profile)
        self._on_saved(profile.name)
        self.close()

    def closeEvent(self, event) -> None:
        self._countdown_timer.stop()
        self._stop_event.set()
        if self._worker is not None:
            self._worker.wait(2000)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        event.accept()
