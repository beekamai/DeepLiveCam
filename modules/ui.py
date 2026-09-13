"""PySide6 UI for Deep-Live-Cam.

Public API kept stable for the rest of the codebase:
    init(start, destroy, lang) -> _Window
        Returned object has .mainloop() that core.py calls.
    update_status(text)
        Thread-safe; routed through Qt signal when called off-UI.
    check_and_ignore_nsfw(target, destroy=None) -> bool
"""

from __future__ import annotations

import importlib
import os
import platform
import queue
import sys
import tempfile
import threading
import time
import webbrowser
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QEventLoop,
    QObject,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QSpinBox,
    QInputDialog,
    QProgressBar,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import modules.globals
from modules import head_geometry, source_identity
from modules import calibration
from modules import enhancer_registry
from modules import swapper_registry
from modules.face_tracker import FaceTracker
from modules.reprojection import ProcessedFrame, Reprojector, RING_SIZE, collect_paste_alpha
import modules.metadata
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    add_blank_map,
    detect_many_faces_fast,
    detect_one_face_fast,
    ensure_landmarks,
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    has_valid_map,
    simplify_maps,
)
from modules.gettext import LANGUAGES, LanguageManager, _, set_language
from modules.gpu_processing import gpu_cvt_color, gpu_flip, gpu_resize
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
)
from modules import imread_unicode
from modules.video_capture import VideoCapturer
from modules.ui_calibration import CalibrationDialog

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

import json


# ─── constants ────────────────────────────────────────────────────────────

ROOT_HEIGHT = 820
ROOT_WIDTH = 640
# The window scrolls below this; it never refuses to fit a small screen.
ROOT_MIN_WIDTH = 460
ROOT_MIN_HEIGHT = 480

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 640
PREVIEW_DEFAULT_HEIGHT = 360
# Camera capture sizes offered on the Live page.
RESOLUTIONS = {
    "360p": (640, 360), "480p": (854, 480), "720p": (1280, 720),
    "1080p": (1920, 1080), "1440p": (2560, 1440),
}

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = 720
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = 870
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_SIZE = 100
SOURCE_TARGET_PREVIEW_SIZE = 200


# ─── modern dark stylesheet ───────────────────────────────────────────────

QSS = """
QMainWindow, QDialog { background-color: #1e1e1e; color: #e6e6e6; }
QWidget { color: #e6e6e6; font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif; font-size: 11pt; }

QGroupBox {
    background-color: #262626;
    border: 1px solid #333333;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 18px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #9ec5ff;
}

QPushButton {
    background-color: #2d6cdf;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton:hover  { background-color: #3a7af0; }
QPushButton:pressed{ background-color: #1d57c2; }
QPushButton:disabled { background-color: #444; color: #888; }
QPushButton#secondary {
    background-color: #3a3a3a;
}
QPushButton#secondary:hover { background-color: #4a4a4a; }
QPushButton#danger { background-color: #c2412d; }
QPushButton#danger:hover  { background-color: #d8523c; }

QComboBox {
    background-color: #2a2a2a;
    border: 1px solid #404040;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 24px;
}
QComboBox:hover { border-color: #2d6cdf; }
QComboBox QAbstractItemView {
    background-color: #2a2a2a;
    selection-background-color: #2d6cdf;
    border: 1px solid #404040;
}

QCheckBox {
    spacing: 8px;
    padding: 4px 0;
}
QCheckBox::indicator {
    width: 36px; height: 18px;
    border-radius: 9px;
    background-color: #3a3a3a;
}
QCheckBox::indicator:checked {
    background-color: #2d6cdf;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #3a3a3a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    width: 16px; height: 16px;
    margin: -5px 0;
    border-radius: 8px;
    border: 1px solid #cccccc;
}
QSlider::sub-page:horizontal {
    background: #2d6cdf;
    border-radius: 3px;
}

QLabel#imageDrop {
    background-color: #2a2a2a;
    border: 2px dashed #444;
    border-radius: 8px;
}
QLabel#statusLabel {
    color: #b9b9b9;
    font-size: 10pt;
    font-style: italic;
}
QLabel#linkLabel {
    color: #6ea8ff;
    text-decoration: underline;
}

QScrollArea { border: none; background: transparent; }

QTabWidget::pane {
    background-color: #262626;
    border: 1px solid #333333;
    border-radius: 10px;
    top: -1px;
}
QTabBar::tab {
    background-color: #2a2a2a;
    color: #b9b9b9;
    border: 1px solid #333333;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 7px 16px;
    margin-right: 2px;
    font-weight: 600;
}
QTabBar::tab:selected { background-color: #262626; color: #9ec5ff; }
QTabBar::tab:hover:!selected { background-color: #333333; color: #e6e6e6; }
QProgressBar {
    background-color: #3a3a3a;
    border: none;
    border-radius: 4px;
    height: 8px;
}
QProgressBar::chunk { background-color: #2d6cdf; border-radius: 4px; }
QLineEdit {
    background-color: #2a2a2a;
    border: 1px solid #404040;
    border-radius: 6px;
    padding: 6px 10px;
}

QFrame#card {
    background-color: #262626;
    border-radius: 10px;
}
"""


# ─── module-level state ───────────────────────────────────────────────────

REPO_URL = "https://github.com/beekamai/DeepLiveCam"
_APP: Optional[QApplication] = None
_MAIN: Optional["MainWindow"] = None
_PREVIEW: Optional["PreviewWindow"] = None
_WEBCAM_PREVIEW: Optional["WebcamPreviewWindow"] = None
_MAPPER: Optional["MapperDialog"] = None
_LIVE_MAPPER: Optional["LiveMapperDialog"] = None
_CALIBRATION: Optional[CalibrationDialog] = None
_LANG: Optional[LanguageManager] = None
_BRIDGE: Optional["_UIBridge"] = None


# Preserve original cwd state for file dialogs.
_RECENT_SOURCE_DIR: Optional[str] = None
_RECENT_TARGET_DIR: Optional[str] = None
_RECENT_OUTPUT_DIR: Optional[str] = None

# QFileDialog filter strings, built from the canonical extension sets in
# globals so every dialog stays in sync (no hand-copied lists to drift).
_IMAGE_FILE_FILTER = "Images (" + " ".join(
    f"*{ext}" for ext in modules.globals.IMAGE_EXTENSIONS
) + ")"
_MEDIA_FILE_FILTER = "Media (" + " ".join(
    f"*{ext}" for ext in (*modules.globals.IMAGE_EXTENSIONS, *modules.globals.VIDEO_EXTENSIONS)
) + ")"
_VIDEO_FILE_FILTER = "Videos (" + " ".join(
    f"*{ext}" for ext in modules.globals.VIDEO_EXTENSIONS
) + ")"


# ─── image utilities ─────────────────────────────────────────────────────


def fit_image_to_size(image, width: int, height: int):
    """BGR ndarray → BGR ndarray scaled to fit within (width, height)."""
    if width is None and height is None or width <= 0 or height <= 0:
        return image
    h, w = image.shape[:2]
    ratio_w = width / w
    ratio_h = height / h
    ratio = min(ratio_w, ratio_h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return gpu_resize(image, dsize=new_size)


def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Zero-copy BGR ndarray → QPixmap."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """PIL.Image → QPixmap."""
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimg = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def render_image_preview(image_path: str, size: Tuple[int, int]) -> QPixmap:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return _pil_to_qpixmap(image)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> Optional[QPixmap]:
    capture = cv2.VideoCapture(video_path)
    try:
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if not has_frame:
            return None
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return _pil_to_qpixmap(image)
    finally:
        capture.release()


# ─── persistence ─────────────────────────────────────────────────────────


def _accept_source_paths(paths: List[str]) -> bool:
    """Adopt ``paths`` as the source photos, keeping only those with a face.

    Several photos of one person are blended by pose (see
    ``modules/source_identity.py``).  A pick with no usable photo leaves the
    previous face in place instead of silently stopping the swap.
    """
    from modules import imread_unicode, source_identity
    from modules.face_analyser import get_many_faces

    picks = dict(getattr(modules.globals, "source_picks", {}) or {})
    chosen = []
    for path in paths:
        if path in picks:
            chosen.append(path)
            continue
        image = imread_unicode(path)
        faces = [] if image is None else (get_many_faces(image) or [])
        if len(faces) > 1 and _MAIN is not None and _APP is not None:
            # Several people in the photo: ask which one is the source.
            from modules.ui_face_picker import FacePickerDialog

            dialog = FacePickerDialog(image, faces, parent=_MAIN)
            if dialog.exec() and dialog.chosen is not None:
                picks[path] = list(dialog.chosen)
            else:
                update_status(f"{os.path.basename(path)} skipped — no face chosen.")
                continue
        chosen.append(path)
    modules.globals.source_picks = picks
    identity, skipped = source_identity.load(chosen)
    for path in skipped:
        update_status(f"No face found in {os.path.basename(path)} — skipped.")
    if identity is None:
        update_status("None of the chosen photos shows a face — keeping the previous face.")
        return False
    source_identity.set_paths(identity.paths)
    return True


def _accept_source_path(path: str) -> bool:
    return _accept_source_paths([path])


def _preload_live_models() -> None:
    """Load and warm every model the live loop will touch, on this thread.

    CUDA graphs are captured on first use; a capture that happens on one
    worker thread while another is already running on the GPU fails and
    leaves both sessions broken.  Creating the sessions here, before the
    workers exist, keeps every capture serial — and takes the multi-second
    first-frame stall out of the preview.
    """
    for key in enhancer_registry.KEYS:
        if modules.globals.fp_ui.get(key, False):
            module = sys.modules.get(f"modules.processors.frame.{key}")
            if module is None:
                try:
                    module = importlib.import_module(f"modules.processors.frame.{key}")
                except Exception as error:
                    print(f"Could not import {key}: {error}")
                    continue
            enhancer = getattr(module, "ENHANCER", None)
            if enhancer is not None:
                try:
                    enhancer.get_session()
                except Exception as error:
                    print(f"Could not preload {key}: {error}")
            elif hasattr(module, "get_face_enhancer"):
                try:
                    module.get_face_enhancer()
                except Exception as error:
                    print(f"Could not preload {key}: {error}")
    if modules.globals.occlusion_mask:
        from modules.face_occluder import get_session as get_occluder

        get_occluder()


def _settings_changed() -> None:
    """Tell the live workers that what they see has changed."""
    modules.globals.settings_epoch += 1


def _release_processor(name: str) -> None:
    """Unload a frame processor's ONNX session if that module was ever loaded."""
    module = sys.modules.get(f"modules.processors.frame.{name}")
    release = getattr(module, "release", None)
    if release is not None:
        try:
            release()
        except Exception as error:
            print(f"Could not unload {name}: {error}")


def save_switch_states():
    state = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "face_swapper_model": modules.globals.face_swapper_model,
        "enhancer_alignment": modules.globals.enhancer_alignment,
        "face_tracking": modules.globals.face_tracking,
        "occlusion_mask": modules.globals.occlusion_mask,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "mouth_mask_size": modules.globals.mouth_mask_size,
        "mouth_reveal_mode": modules.globals.mouth_reveal_mode,
        "face_outline_mask": modules.globals.face_outline_mask,
        "occlusion_interval": modules.globals.occlusion_interval,
        "calibration_profile": modules.globals.calibration_profile,
        "reference_outline": modules.globals.reference_outline,
        "pose_fade": modules.globals.pose_fade,
        "stable_alignment": modules.globals.stable_alignment,
        "blink_reveal": modules.globals.blink_reveal,
        "eye_reveal": modules.globals.eye_reveal,
        "reprojection": modules.globals.reprojection,
        "language": _LANG.current_language if _LANG is not None else "en",
        "tensorrt": modules.globals.tensorrt,
        "max_fps": modules.globals.max_fps,
        "head_pose": modules.globals.head_pose,
        "mask_forehead": modules.globals.mask_forehead,
        "mask_chin": modules.globals.mask_chin,
        "gpu_device": modules.globals.gpu_device,
        "camera_resolution": modules.globals.camera_resolution,
        "face_fader_seconds": modules.globals.face_fader_seconds,
        "head_outline": modules.globals.head_outline,
        "head_yaw_limit": modules.globals.head_yaw_limit,
        "head_pitch_limit": modules.globals.head_pitch_limit,
        "calibration_sounds": modules.globals.calibration_sounds,
        "calibration_voice": modules.globals.calibration_voice,
    }
    try:
        with open("switch_states.json", "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def _saved_language() -> Optional[str]:
    try:
        with open("switch_states.json", "r") as f:
            code = json.load(f).get("language")
    except (OSError, json.JSONDecodeError):
        return None
    return code if code in LANGUAGES else None


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            state = json.load(f)
        modules.globals.keep_fps = state.get("keep_fps", True)
        modules.globals.keep_audio = state.get("keep_audio", True)
        modules.globals.keep_frames = state.get("keep_frames", False)
        modules.globals.many_faces = state.get("many_faces", False)
        modules.globals.map_faces = state.get("map_faces", False)
        modules.globals.poisson_blend = state.get("poisson_blend", False)
        modules.globals.color_correction = state.get("color_correction", False)
        modules.globals.nsfw_filter = state.get("nsfw_filter", False)
        modules.globals.live_mirror = state.get("live_mirror", False)
        modules.globals.live_resizable = state.get("live_resizable", False)
        # Merge onto the defaults so enhancers added since the file was
        # written are present (and off) instead of missing.
        fp_ui = {key: False for key in enhancer_registry.KEYS}
        fp_ui.update(state.get("fp_ui", {}))
        modules.globals.fp_ui = fp_ui
        modules.globals.face_tracking = state.get("face_tracking", True)
        modules.globals.occlusion_mask = state.get("occlusion_mask", False)
        if state.get("enhancer_alignment") in ("legacy", "model"):
            modules.globals.enhancer_alignment = state["enhancer_alignment"]
        saved_swapper = state.get("face_swapper_model")
        if saved_swapper in swapper_registry.BY_KEY:
            modules.globals.face_swapper_model = saved_swapper
        modules.globals.show_fps = state.get("show_fps", False)
        modules.globals.face_outline_mask = state.get("face_outline_mask", True)
        modules.globals.occlusion_interval = min(3, max(1, int(state.get("occlusion_interval", 1))))
        modules.globals.reference_outline = state.get("reference_outline", True)
        modules.globals.pose_fade = state.get("pose_fade", True)
        modules.globals.stable_alignment = state.get("stable_alignment", True)
        modules.globals.blink_reveal = state.get("blink_reveal", True)
        modules.globals.eye_reveal = min(1.0, max(0.0, float(state.get("eye_reveal", 0.0))))
        modules.globals.reprojection = state.get("reprojection", True)
        modules.globals.tensorrt = state.get("tensorrt", True)
        modules.globals.max_fps = max(0, min(60, int(state.get("max_fps", 0))))
        modules.globals.head_pose = state.get("head_pose", True)
        modules.globals.mouth_reveal_mode = state.get("mouth_reveal_mode", "region")
        modules.globals.mask_forehead = min(0.8, max(0.1, float(state.get("mask_forehead", 0.35))))
        modules.globals.mask_chin = min(0.3, max(0.0, float(state.get("mask_chin", 0.0))))
        modules.globals.gpu_device = max(0, int(state.get("gpu_device", 0)))
        if state.get("camera_resolution") in RESOLUTIONS:
            modules.globals.camera_resolution = state["camera_resolution"]
        modules.globals.face_fader_seconds = max(1, min(60, int(state.get("face_fader_seconds", 5))))
        if modules.globals.mouth_reveal_mode not in ("region", "lips"):
            modules.globals.mouth_reveal_mode = "region"
        modules.globals.head_outline = state.get("head_outline", True)
        modules.globals.head_yaw_limit = max(30, min(90, int(state.get("head_yaw_limit", 55))))
        modules.globals.head_pitch_limit = max(15, min(60, int(state.get("head_pitch_limit", 35))))
        modules.globals.calibration_sounds = state.get("calibration_sounds", True)
        modules.globals.calibration_voice = state.get("calibration_voice", False)
        # A profile named on the command line wins over the remembered one.
        if calibration.active() is None and state.get("calibration_profile"):
            calibration.activate_saved(state["calibration_profile"])
        # Mouth mask always starts disabled (slider at 0) on launch,
        # regardless of the persisted value — enable it explicitly each session.
        modules.globals.mouth_mask_size = 0.0
        modules.globals.mouth_mask = False
        modules.globals.show_mouth_mask_box = False
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError):
        pass


# ─── thread-safe status bridge ───────────────────────────────────────────


class _UIBridge(QObject):
    """Single QObject that owns cross-thread signals."""

    statusChanged = Signal(str)
    busyChanged = Signal(bool, str)


def set_busy(active: bool, text: str = "") -> None:
    """Model loaders call this (any thread) around their slow parts; the
    main window shows an activity bar while at least one is in flight."""
    if _BRIDGE is not None:
        _BRIDGE.busyChanged.emit(active, _(text))


def _emit_status(text: str) -> None:
    if _BRIDGE is None:
        print(text)
        return
    _BRIDGE.statusChanged.emit(text)


# ─── public API ──────────────────────────────────────────────────────────


def update_status(text: str) -> None:
    """Thread-safe status update — uses signal if called off-UI thread."""
    _emit_status(_(text))
    if _APP is not None and QThread.currentThread() is _APP.thread():
        # On UI thread — flush events so the user sees the update during
        # long synchronous start() runs.  Input stays queued: a click
        # handled here would re-enter a model load that holds its lock
        # (a second swapper change during a TensorRT build hung the app).
        _APP.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)


def check_and_ignore_nsfw(target, destroy: Optional[Callable] = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_frame, predict_image, predict_video

    check_nsfw = None
    if isinstance(target, str):
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("Processing ignored!")
        return True
    return False


# ─── camera enumeration (unchanged from tk version) ──────────────────────


def get_available_cameras() -> Tuple[List[int], List[str]]:
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if devices:
                return list(range(len(devices))), devices
            return [], ["No cameras found"]
        except Exception as exc:
            print(f"Error detecting cameras: {exc}")
            return [], ["No cameras found"]

    if platform.system() == "Darwin":
        return [0, 1], ["Camera 0", "Camera 1"]

    # Linux probe
    indices: List[int] = []
    names: List[str] = []
    for i in range(10):
        cap = cv2.VideoCapture(f"/dev/video{i}")
        if cap.isOpened():
            indices.append(i)
            names.append(f"Camera {i}")
            cap.release()
    return (indices, names) if names else ([], ["No cameras found"])


# ─── main window ─────────────────────────────────────────────────────────


def _make_image_drop(text: str, size: Tuple[int, int]) -> QLabel:
    label = QLabel(text)
    label.setObjectName("imageDrop")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFixedSize(size[0], size[1])
    label.setText(text)
    return label


class _Switch(QWidget):
    """Compact toggle switch with label + optional tooltip."""

    toggled = Signal(bool)

    def __init__(self, text: str, initial: bool, tooltip: str = ""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._checkbox = QCheckBox(text)
        self._checkbox.setChecked(initial)
        self._checkbox.toggled.connect(self.toggled.emit)
        if tooltip:
            self._checkbox.setToolTip(tooltip)
        layout.addWidget(self._checkbox)
        layout.addStretch(1)

    def isChecked(self) -> bool:
        return self._checkbox.isChecked()

    def setChecked(self, value: bool) -> None:
        self._checkbox.setChecked(value)


def _ui_scale() -> float:
    """Scale for fixed-size widgets: 1.0 on a desktop, smaller on a laptop
    screen, capped so a 4K monitor does not get postage-stamp previews."""
    screen = QApplication.primaryScreen()
    if screen is None:
        return 1.0
    geometry = screen.availableGeometry()
    return max(0.7, min(1.4, min(geometry.width() / 1280.0, geometry.height() / 900.0)))


class MainWindow(QMainWindow):
    def __init__(self, start_cb: Callable, destroy_cb: Callable):
        super().__init__()
        load_switch_states()
        self._start_cb = start_cb
        self._destroy_cb = destroy_cb
        self._loader: Optional[_ModelLoader] = None
        self._fader_timer = QTimer(self)
        self._fader_timer.setInterval(33)
        self._fader_timer.timeout.connect(self._fader_step)

        self.setWindowTitle(
            f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
        )
        size = int(SOURCE_TARGET_PREVIEW_SIZE * _ui_scale())
        self._preview_size = (size, size)
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        width = ROOT_WIDTH if available is None else min(ROOT_WIDTH, int(available.width() * 0.9))
        height = ROOT_HEIGHT if available is None else min(ROOT_HEIGHT, int(available.height() * 0.9))
        self.setMinimumSize(ROOT_MIN_WIDTH, ROOT_MIN_HEIGHT)
        self.resize(width, height)

        # Everything sits in a scroll area, so a short screen still reaches
        # every control instead of clipping the bottom of the window.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.viewport().setAutoFillBackground(False)
        root = QWidget()
        root.setAutoFillBackground(False)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        top = QHBoxLayout()
        top.setSpacing(16)
        top.addLayout(self._build_source_column())
        self._modes = QTabWidget()
        self._modes.addTab(self._build_live_page(), _("Live"))
        self._modes.addTab(self._build_media_page(), _("Photo / Video"))
        self._modes.setToolTip(_("Live swaps your webcam; Photo / Video processes a file"))
        # Takes whatever width is left of the source column, never more —
        # and only as much height as the *current* page needs, so the short
        # Live page does not inherit the Photo / Video page's height.
        self._modes.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._modes.currentChanged.connect(self._fit_mode_tabs)
        self._fit_mode_tabs(self._modes.currentIndex())
        top.addWidget(self._modes, 1)
        layout.addLayout(top)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_models_tab(), _("Models"))
        self._tabs.addTab(self._build_mask_tab(), _("Mask"))
        self._tabs.addTab(self._build_motion_tab(), _("Motion"))
        self._tabs.addTab(self._build_output_tab(), _("Output"))
        layout.addWidget(self._tabs, 1)

        self._status_label = QLabel("")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        # Indeterminate bar shown while a model loads or a TensorRT engine
        # builds — the window stays responsive, and this says why it waits.
        self._busy_bar = QProgressBar()
        self._busy_bar.setRange(0, 0)
        self._busy_bar.setTextVisible(False)
        self._busy_bar.setFixedHeight(6)
        self._busy_bar.hide()
        self._busy_depth = 0
        layout.addWidget(self._busy_bar)

        footer = QLabel(f"DeepLiveCam {modules.metadata.version} · GitHub")
        footer.setObjectName("linkLabel")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setCursor(Qt.CursorShape.PointingHandCursor)
        footer.setToolTip(REPO_URL)
        footer.mousePressEvent = lambda _e: webbrowser.open(REPO_URL)
        layout.addWidget(footer)

        from modules.providers import install_hint

        hint = install_hint()
        if hint:
            QTimer.singleShot(0, lambda: update_status(hint))

    # ── responsive previews ──────────────────────────────────────────────

    # Two previews, the swap button and the margins must fit the window
    # width; below that the previews shrink instead of the window clipping.
    PREVIEW_MIN = 120
    # source preview, the mode tab's margins, the target preview and its
    # buttons column must share the window width
    PREVIEW_CHROME = 16 * 2 + 16 + 16 * 2 + 24

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        fit = (self.width() - self.PREVIEW_CHROME) // 2
        size = max(self.PREVIEW_MIN, min(int(SOURCE_TARGET_PREVIEW_SIZE * _ui_scale()), fit))
        if size != self._preview_size[0]:
            self._preview_size = (size, size)
            self.source_label.setFixedSize(size, size)
            self.target_label.setFixedSize(size, size)
            self._refresh_previews()

    def _refresh_previews(self) -> None:
        """Re-render the source/target thumbnails at the current size."""
        source = modules.globals.source_path
        if source and is_image(source):
            self._show_source()
        target = modules.globals.target_path
        if target and is_image(target):
            self.target_label.setPixmap(render_image_preview(target, self._preview_size))
        elif target and is_video(target):
            pm = render_video_preview(target, self._preview_size)
            if pm is not None:
                self.target_label.setPixmap(pm)

    def _fit_mode_tabs(self, index: int) -> None:
        for i in range(self._modes.count()):
            page = self._modes.widget(i)
            policy = QSizePolicy.Policy.Preferred if i == index else QSizePolicy.Policy.Ignored
            page.setSizePolicy(QSizePolicy.Policy.Ignored, policy)
        self._modes.adjustSize()

    # ── widget helpers ───────────────────────────────────────────────────

    def _switch(self, field: str, label: str, tip: str) -> _Switch:
        """A toggle bound straight to a ``modules.globals`` flag."""
        sw = _Switch(_(label), getattr(modules.globals, field), _(tip))
        sw.toggled.connect(
            lambda v, f=field: (setattr(modules.globals, f, v), _settings_changed(), save_switch_states())
        )
        return sw

    @staticmethod
    def _slider(min_v, max_v, default, denom, on_change) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(int(min_v * denom), int(max_v * denom))
        s.setValue(int(default * denom))
        s.valueChanged.connect(lambda iv: on_change(iv / denom))
        return s

    @staticmethod
    def _tab() -> Tuple[QWidget, QGridLayout]:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        return page, grid

    # ── source column ────────────────────────────────────────────────────

    def _build_source_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        self.source_label = _make_image_drop(_("Source face"), self._preview_size)
        col.addWidget(self.source_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.source_caption = QLabel("")
        self.source_caption.setObjectName("statusLabel")
        self.source_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addWidget(self.source_caption)
        row = QHBoxLayout()
        self.btn_select_source = QPushButton(_("Select a face"))
        self.btn_select_source.setToolTip(
            _("Choose the source face — pick several photos of the same person "
              "(front, turned, up, down) and they are blended by pose")
        )
        self.btn_select_source.clicked.connect(self._on_select_source)
        self.btn_random_face = QPushButton("⟳")
        self.btn_random_face.setObjectName("secondary")
        self.btn_random_face.setFixedWidth(40)
        self.btn_random_face.setToolTip(
            _("Get a random face from thispersondoesnotexist.com")
        )
        self.btn_random_face.clicked.connect(self._on_random_face)
        row.addWidget(self.btn_select_source)
        row.addWidget(self.btn_random_face)
        col.addLayout(row)

        lib_row = QHBoxLayout()
        self.cb_library = QComboBox()
        self.cb_library.setToolTip(_("Saved source sets — pick one to load its photos"))
        self._refresh_library()
        self.cb_library.currentTextChanged.connect(self._on_library_pick)
        lib_row.addWidget(self.cb_library, 1)
        self.btn_library_save = QPushButton("★")
        self.btn_library_save.setObjectName("secondary")
        self.btn_library_save.setFixedWidth(40)
        self.btn_library_save.setToolTip(_("Save the current source photos to the library"))
        self.btn_library_save.clicked.connect(self._on_library_save)
        lib_row.addWidget(self.btn_library_save)
        self.btn_library_delete = QPushButton("✕")
        self.btn_library_delete.setObjectName("secondary")
        self.btn_library_delete.setFixedWidth(40)
        self.btn_library_delete.setToolTip(_("Remove the selected set from the library"))
        self.btn_library_delete.clicked.connect(self._on_library_delete)
        lib_row.addWidget(self.btn_library_delete)
        col.addLayout(lib_row)
        col.addStretch(1)
        return col

    # ── library (saved source sets) ──────────────────────────────────────

    def _refresh_library(self) -> None:
        from modules import library

        self.cb_library.blockSignals(True)
        self.cb_library.clear()
        self.cb_library.addItem(_("Library…"))
        self.cb_library.addItems(library.names())
        self.cb_library.blockSignals(False)

    def _on_library_pick(self, name: str) -> None:
        from modules import library, source_identity

        if self.cb_library.currentIndex() <= 0:
            return
        entry = library.load(name)
        if entry is None:
            update_status(f"Library set {name} has no readable photos.")
            return
        modules.globals.source_picks = entry["picks"]
        identity, _skipped = source_identity.load(entry["paths"])
        if identity is None:
            update_status(f"No face in the photos of {name}.")
            return
        source_identity.set_paths(identity.paths)
        self._show_source()
        update_status(f"Source: {name}")

    def _on_library_save(self) -> None:
        from modules import library

        paths = modules.globals.source_paths or (
            [modules.globals.source_path] if modules.globals.source_path else [])
        if not paths:
            update_status("Select a source face first.")
            return
        default = os.path.splitext(os.path.basename(paths[0]))[0]
        name, ok = QInputDialog.getText(self, _("Save to library"), _("Name for this source set:"), text=default)
        if not ok or not name.strip():
            return
        library.save(name, paths, modules.globals.source_picks)
        self._refresh_library()
        self.cb_library.setCurrentText(name.strip())
        update_status(f"Saved {name.strip()} to the library.")

    def _on_library_delete(self) -> None:
        from modules import library

        if self.cb_library.currentIndex() <= 0:
            return
        name = self.cb_library.currentText()
        library.delete(name)
        self._refresh_library()
        update_status(f"Removed {name} from the library.")

    # ── live page: camera, Live, Calibrate ───────────────────────────────

    def _build_live_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        hint = QLabel(_("Pick a source face, choose the camera and press Live. "
                        "Calibrate once for your own face."))
        hint.setWordWrap(True)
        hint.setObjectName("statusLabel")
        hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(hint)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel(_("Camera:")))
        self._camera_indices, self._camera_names = get_available_cameras()
        self.cb_camera = QComboBox()
        if not self._camera_names or self._camera_names[0] == "No cameras found":
            self.cb_camera.addItem("No cameras found")
            self.cb_camera.setEnabled(False)
            cam_ok = False
        else:
            self.cb_camera.addItems(self._camera_names)
            cam_ok = True
        self.cb_camera.setToolTip(_("Select which camera to use for live mode"))
        cam_row.addWidget(self.cb_camera, 1)
        layout.addLayout(cam_row)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel(_("Resolution:")))
        self.cb_resolution = QComboBox()
        self.cb_resolution.addItems(list(RESOLUTIONS))
        self.cb_resolution.setCurrentText(modules.globals.camera_resolution)
        self.cb_resolution.setToolTip(_("Camera capture size: lower is faster, higher is sharper"))
        self.cb_resolution.currentTextChanged.connect(self._on_resolution_change)
        res_row.addWidget(self.cb_resolution, 1)
        layout.addLayout(res_row)

        buttons = QHBoxLayout()
        self.btn_live = QPushButton(_("Live"))
        self.btn_live.setEnabled(cam_ok)
        self.btn_live.setToolTip(_("Start real-time face swap using webcam"))
        self.btn_live.clicked.connect(self._on_live)
        self.btn_calibrate_quick = QPushButton(_("Calibrate…"))
        self.btn_calibrate_quick.setObjectName("secondary")
        self.btn_calibrate_quick.setEnabled(cam_ok)
        self.btn_calibrate_quick.setToolTip(_("A minute in front of the camera: your face outline "
                                              "and how far you can turn before the swap lets go"))
        self.btn_calibrate_quick.clicked.connect(self._on_calibrate)
        buttons.addWidget(self.btn_live, 2)
        buttons.addWidget(self.btn_calibrate_quick, 1)
        layout.addLayout(buttons)

        fader = QHBoxLayout()
        self.btn_fader = QPushButton(_("Face fader"))
        self.btn_fader.setObjectName("secondary")
        self.btn_fader.setToolTip(_("Fade the swap in (or back out) gradually while streaming"))
        self.btn_fader.clicked.connect(self._on_face_fader)
        fader.addWidget(self.btn_fader, 2)
        self.sb_fader = QSpinBox()
        self.sb_fader.setRange(1, 60)
        self.sb_fader.setSuffix(" s")
        self.sb_fader.setValue(modules.globals.face_fader_seconds)
        self.sb_fader.setToolTip(_("How long the fade takes"))
        self.sb_fader.valueChanged.connect(self._on_fader_seconds)
        fader.addWidget(self.sb_fader, 1)
        layout.addLayout(fader)
        layout.addStretch(1)
        return page

    # ── face fader ───────────────────────────────────────────────────────

    def _on_fader_seconds(self, value: int) -> None:
        modules.globals.face_fader_seconds = int(value)
        save_switch_states()

    def _on_face_fader(self) -> None:
        if self._fader_timer.isActive():
            self._fader_timer.stop()
            self.btn_fader.setText(_("Face fader"))
            return
        start = float(modules.globals.opacity)
        self._fader_target = 0.0 if start > 0.5 else 1.0
        self._fader_from = start
        self._fader_t0 = time.perf_counter()
        modules.globals.face_swapper_enabled = True
        self.btn_fader.setText(_("Stop fade"))
        self._fader_timer.start()

    def _fader_step(self) -> None:
        span = max(0.2, float(modules.globals.face_fader_seconds))
        k = min(1.0, (time.perf_counter() - self._fader_t0) / span)
        value = self._fader_from + (self._fader_target - self._fader_from) * k
        modules.globals.opacity = value
        self.s_transparency.blockSignals(True)
        self.s_transparency.setValue(int(round(value * 100)))
        self.s_transparency.blockSignals(False)
        if k >= 1.0:
            self._fader_timer.stop()
            self.btn_fader.setText(_("Face fader"))
            self._on_transparency_change(self._fader_target)

    # ── photo / video page: target, Start, Preview, output options ───────

    def _build_media_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        self.target_label = _make_image_drop(_("Target"), self._preview_size)
        layout.addWidget(self.target_label, alignment=Qt.AlignmentFlag.AlignCenter)
        tgt_row = QHBoxLayout()
        self.btn_select_target = QPushButton(_("Select a target"))
        self.btn_select_target.setToolTip(
            _("Choose the target image or video to apply face swap to")
        )
        self.btn_select_target.clicked.connect(self._on_select_target)
        self.btn_swap = QPushButton("↔")
        self.btn_swap.setObjectName("secondary")
        self.btn_swap.setFixedWidth(40)
        self.btn_swap.setToolTip(_("Swap source and target images"))
        self.btn_swap.clicked.connect(self._on_swap_paths)
        tgt_row.addWidget(self.btn_select_target, 1)
        tgt_row.addWidget(self.btn_swap)
        layout.addLayout(tgt_row)

        actions = QHBoxLayout()
        self.btn_start = QPushButton(_("Start"))
        self.btn_start.setToolTip(_("Begin processing the target image/video with selected face"))
        self.btn_start.clicked.connect(self._on_start)
        self.btn_preview = QPushButton(_("Preview"))
        self.btn_preview.setObjectName("secondary")
        self.btn_preview.setToolTip(_("Show/hide a preview of the processed output"))
        self.btn_preview.clicked.connect(self._on_toggle_preview)
        actions.addWidget(self.btn_start, 1)
        actions.addWidget(self.btn_preview, 1)
        layout.addLayout(actions)

        self.sw_keep_fps = self._switch("keep_fps", "Keep fps",
                                        "Output video keeps the original frame rate")
        self.sw_keep_audio = self._switch("keep_audio", "Keep audio",
                                          "Copy audio track from the source video to output")
        self.sw_keep_frames = self._switch("keep_frames", "Keep frames",
                                           "Keep extracted frames on disk after processing")
        self.sw_many_faces = self._switch("many_faces", "Many faces",
                                          "Swap every detected face, not just the primary one")
        # Map faces is special — closes mapper when toggled off.
        self.sw_map_faces = _Switch(_("Map faces"), modules.globals.map_faces,
                                    _("Manually assign which source face maps to which target face"))
        self.sw_map_faces.toggled.connect(self._on_map_faces_toggled)
        for w in (self.sw_keep_fps, self.sw_keep_audio, self.sw_keep_frames,
                  self.sw_many_faces, self.sw_map_faces):
            layout.addWidget(w)
        layout.addStretch(1)
        return page

    # ── models tab ───────────────────────────────────────────────────────

    def _build_models_tab(self) -> QWidget:
        page, grid = self._tab()

        grid.addWidget(QLabel(_("Face Swapper:")), 0, 0)
        self.cb_swapper = QComboBox()
        self.cb_swapper.addItems([spec.label for spec in swapper_registry.SWAPPERS])
        current_key = getattr(
            modules.globals, "face_swapper_model", swapper_registry.DEFAULT_KEY
        )
        self.cb_swapper.setCurrentText(
            swapper_registry.BY_KEY[current_key].label
            if current_key in swapper_registry.BY_KEY
            else swapper_registry.BY_KEY[swapper_registry.DEFAULT_KEY].label
        )
        self.cb_swapper.currentTextChanged.connect(self._on_swapper_change)
        self.cb_swapper.setToolTip(
            _("Face swap model. HyperSwap runs at 256px, Inswapper at 128px.")
        )
        grid.addWidget(self.cb_swapper, 0, 1)

        grid.addWidget(QLabel(_("Face Enhancer:")), 1, 0)
        self.cb_enhancer = QComboBox()
        self.cb_enhancer.addItems(
            ["None"] + [spec.label for spec in enhancer_registry.ENHANCERS]
        )
        initial = "None"
        for spec in enhancer_registry.ENHANCERS:
            if modules.globals.fp_ui.get(spec.key, False):
                initial = spec.label
                break
        self.cb_enhancer.setCurrentText(initial)
        self.cb_enhancer.currentTextChanged.connect(self._on_enhancer_change)
        self.cb_enhancer.setToolTip(_("Select a face enhancement model (None = no enhancement)"))
        grid.addWidget(self.cb_enhancer, 1, 1)

        grid.addWidget(QLabel(_("Enhancer Crop:")), 2, 0)
        self.cb_alignment = QComboBox()
        self._alignment_map = {
            _("Tight (sharper)"): "legacy",
            _("Wide (whole head)"): "model",
        }
        self.cb_alignment.addItems(list(self._alignment_map))
        for label, value in self._alignment_map.items():
            if value == getattr(modules.globals, "enhancer_alignment", "legacy"):
                self.cb_alignment.setCurrentText(label)
                break
        self.cb_alignment.currentTextChanged.connect(self._on_alignment_change)
        self.cb_alignment.setToolTip(
            _("Tight puts more pixels on the face and leaves hair untouched; "
              "wide matches how the models were trained.")
        )
        grid.addWidget(self.cb_alignment, 2, 1)

        grid.addWidget(QLabel(_("Backend:")), 5, 0)
        self.cb_backend = QComboBox()
        from modules.providers import tensorrt_available

        self._backend_map = {_("TensorRT (fastest)"): True, _("CUDA graph"): False}
        self.cb_backend.addItems(list(self._backend_map))
        if tensorrt_available():
            for label, value in self._backend_map.items():
                if value == modules.globals.tensorrt:
                    self.cb_backend.setCurrentText(label)
            self.cb_backend.setToolTip(_("TensorRT compiles each model into one engine — 2-3x faster; "
                                         "the first use of a model builds it (up to a minute)"))
        else:
            self.cb_backend.setCurrentText(_("CUDA graph"))
            self.cb_backend.setEnabled(False)
            self.cb_backend.setToolTip(_("TensorRT runtime not installed — run install-tensorrt.bat"))
        self.cb_backend.currentTextChanged.connect(self._on_backend_change)
        grid.addWidget(self.cb_backend, 5, 1)

        grid.addWidget(QLabel(_("GPU:")), 6, 0)
        self.cb_gpu = QComboBox()
        from modules.providers import list_gpus

        gpus = list_gpus()
        self.cb_gpu.addItems(gpus or [_("default")])
        self.cb_gpu.setCurrentIndex(min(modules.globals.gpu_device, max(0, len(gpus) - 1)))
        self.cb_gpu.setEnabled(len(gpus) > 1)
        self.cb_gpu.setToolTip(_("Which GPU runs the models — takes effect after a restart"))
        self.cb_gpu.currentIndexChanged.connect(self._on_gpu_change)
        grid.addWidget(self.cb_gpu, 6, 1)

        grid.addWidget(QLabel(_("Transparency")), 3, 0)
        self.s_transparency = self._slider(0.0, 1.0, 1.0, 100, self._on_transparency_change)
        self.s_transparency.setToolTip(
            _("Blend between original and swapped face (0% = original, 100% = fully swapped)")
        )
        grid.addWidget(self.s_transparency, 3, 1)

        grid.addWidget(QLabel(_("Sharpness")), 4, 0)
        self.s_sharpness = self._slider(0.0, 5.0, 0.0, 10, self._on_sharpness_change)
        self.s_sharpness.setToolTip(_("Sharpen the enhanced face output"))
        grid.addWidget(self.s_sharpness, 4, 1)

        grid.setRowStretch(7, 1)
        return page

    def _on_gpu_change(self, index: int) -> None:
        if index < 0 or index == modules.globals.gpu_device:
            return
        modules.globals.gpu_device = index
        save_switch_states()
        update_status("GPU changed — restart the application to apply.")

    # ── mask tab ─────────────────────────────────────────────────────────

    def _build_mask_tab(self) -> QWidget:
        page, grid = self._tab()

        self.sw_outline = self._switch(
            "face_outline_mask", "Face outline mask",
            "Shape the pasted region from the face landmarks instead of a fixed "
            "ellipse, so it follows the face when the head turns",
        )
        grid.addWidget(self.sw_outline, 0, 0, 1, 2)

        # Occlusion is special — the model is unloaded when switched off.
        self.sw_occlusion = _Switch(
            _("Occlusion mask"), modules.globals.occlusion_mask,
            _("Keep hands, mics and hair that cover the face out of the swap "
              "(XSeg, costs ~8 ms per frame)"),
        )
        self.sw_occlusion.toggled.connect(self._on_occlusion_toggled)
        grid.addWidget(self.sw_occlusion, 1, 0)

        self.cb_occlusion_interval = QComboBox()
        self.cb_occlusion_interval.addItems(
            [_("every frame"), _("every 2nd frame"), _("every 3rd frame")]
        )
        self.cb_occlusion_interval.setCurrentIndex(
            min(2, max(0, modules.globals.occlusion_interval - 1))
        )
        self.cb_occlusion_interval.currentIndexChanged.connect(self._on_occlusion_interval_change)
        self.cb_occlusion_interval.setToolTip(
            _("Run the occluder less often and reuse its mask in between — "
              "cheaper, with a frame or two of lag on fast hands")
        )
        grid.addWidget(self.cb_occlusion_interval, 1, 1)

        self.sw_poisson = self._switch(
            "poisson_blend", "Poisson Blend",
            "Match the swapped face's colours to the frame at the seam "
            "(~25 ms per frame)",
        )
        grid.addWidget(self.sw_poisson, 2, 0, 1, 2)

        self.sw_mask_debug = self._switch(
            "show_mask_debug", "Show mask",
            "Overlay the paste mask on the live preview: green is painted by "
            "the swap, red is held back",
        )
        grid.addWidget(self.sw_mask_debug, 3, 0, 1, 2)

        grid.addWidget(QLabel(_("Mouth Mask")), 4, 0)
        # Always starts at 0 (disabled) on launch.
        self.s_mouth = self._slider(0.0, 100.0, 0.0, 1, self._on_mouth_mask_change)
        self.s_mouth.sliderPressed.connect(self._on_mouth_mask_pressed)
        self.s_mouth.sliderReleased.connect(self._on_mouth_mask_released)
        self.cb_mouth_mode = QComboBox()
        self._mouth_modes = {_("Region (lips to chin)"): "region", _("Lips only (keeps a moustache)"): "lips"}
        self.cb_mouth_mode.addItems(list(self._mouth_modes))
        for label, key in self._mouth_modes.items():
            if key == modules.globals.mouth_reveal_mode:
                self.cb_mouth_mode.setCurrentText(label)
        self.cb_mouth_mode.setToolTip(_("Region grows from the lips towards the nose and chin; "
                                        "Lips only never rises above the upper lip, so a real "
                                        "moustache stays swapped, and grows downward for a tongue"))
        self.cb_mouth_mode.currentTextChanged.connect(self._on_mouth_mode_change)
        self.s_mouth.setToolTip(
            _("0 = use swapped mouth, 100 = expose original mouth to chin area")
        )
        mouth_row = QHBoxLayout()
        mouth_row.addWidget(self.s_mouth, 1)
        mouth_row.addWidget(self.cb_mouth_mode)
        grid.addLayout(mouth_row, 4, 1)

        self.sw_blink = self._switch(
            "blink_reveal", "Real blinks",
            "Show the real eyelids while you blink — the swap models only "
            "squint and never fully close an eye",
        )
        grid.addWidget(self.sw_blink, 5, 0, 1, 2)

        grid.addWidget(QLabel(_("Real eyes")), 6, 0)
        self.s_eyes = self._slider(0.0, 1.0, modules.globals.eye_reveal, 100, self._on_eye_reveal_change)
        self.s_eyes.setToolTip(
            _("Blend the real eyes over the swapped ones: 0 = swapped eyes, "
              "100 = your own gaze (you can cross them, the model cannot)")
        )
        grid.addWidget(self.s_eyes, 6, 1)

        grid.addWidget(QLabel(_("Forehead")), 7, 0)
        self.s_forehead = self._slider(0.1, 0.8, modules.globals.mask_forehead, 100, self._on_forehead_change)
        self.s_forehead.setToolTip(_("How far above the brows the mask reaches, as a share of the face height"))
        grid.addWidget(self.s_forehead, 7, 1)
        grid.addWidget(QLabel(_("Chin")), 8, 0)
        self.s_chin = self._slider(0.0, 0.3, modules.globals.mask_chin, 100, self._on_chin_change)
        self.s_chin.setToolTip(_("How far below the jaw landmarks the mask reaches (beard, double chin)"))
        grid.addWidget(self.s_chin, 8, 1)

        grid.setRowStretch(9, 1)
        return page

    def _on_forehead_change(self, value: float) -> None:
        modules.globals.mask_forehead = value
        save_switch_states()

    def _on_chin_change(self, value: float) -> None:
        modules.globals.mask_chin = value
        save_switch_states()

    # ── motion tab ───────────────────────────────────────────────────────

    def _build_motion_tab(self) -> QWidget:
        page, grid = self._tab()

        self.sw_tracking = self._switch(
            "face_tracking", "Face tracking",
            "Follow the face with optical flow between detections so the swap "
            "does not lag behind fast movement",
        )
        grid.addWidget(self.sw_tracking, 0, 0, 1, 2)

        self.sw_reprojection = self._switch(
            "reprojection", "Low-latency reprojection",
            "Show every camera frame with the last finished swap moved to "
            "where the face is now — a slow enhancer then delays the "
            "expression, not the position",
        )
        grid.addWidget(self.sw_reprojection, 1, 0, 1, 2)

        head = QGroupBox(_("Head pose (3D)"))
        hgrid = QGridLayout(head)
        hgrid.setHorizontalSpacing(12)
        hgrid.setVerticalSpacing(8)
        hgrid.setColumnStretch(1, 1)
        self.sw_head_pose = self._switch(
            "head_pose", "Measure head angles",
            "Fit 68 landmarks in 3D every frame (~3 ms): the real yaw and "
            "pitch fade the swap past the limits below, no calibration needed",
        )
        hgrid.addWidget(self.sw_head_pose, 0, 0, 1, 3)
        self.sw_head_outline = self._switch(
            "head_outline", "3D outline",
            "Add the 3D fit's silhouette to the mask so the far cheek stays "
            "covered when the 2D landmarks slide on a turned head",
        )
        hgrid.addWidget(self.sw_head_outline, 1, 0, 1, 3)
        hgrid.addWidget(QLabel(_("Fade past yaw")), 2, 0)
        self.s_yaw_limit = self._slider(30, 90, modules.globals.head_yaw_limit, 1, self._on_yaw_limit)
        self.s_yaw_limit.setToolTip(_("Turn (degrees) beyond which the real face shows through"))
        hgrid.addWidget(self.s_yaw_limit, 2, 1)
        self.l_yaw_limit = QLabel(f"{modules.globals.head_yaw_limit}°")
        hgrid.addWidget(self.l_yaw_limit, 2, 2)
        hgrid.addWidget(QLabel(_("Fade past pitch")), 3, 0)
        self.s_pitch_limit = self._slider(15, 60, modules.globals.head_pitch_limit, 1, self._on_pitch_limit)
        self.s_pitch_limit.setToolTip(_("Tilt (degrees) beyond which the real face shows through"))
        hgrid.addWidget(self.s_pitch_limit, 3, 1)
        self.l_pitch_limit = QLabel(f"{modules.globals.head_pitch_limit}°")
        hgrid.addWidget(self.l_pitch_limit, 3, 2)
        grid.addWidget(head, 2, 0, 1, 2)

        calib = QGroupBox(_("Calibration"))
        cgrid = QGridLayout(calib)
        cgrid.setHorizontalSpacing(12)
        cgrid.setVerticalSpacing(8)
        cgrid.setColumnStretch(1, 1)

        hint = QLabel(_(
            "A profile records your own face outline and how far you turn "
            "before the swap should hand back the real face."
        ))
        hint.setWordWrap(True)
        hint.setObjectName("statusLabel")
        cgrid.addWidget(hint, 0, 0, 1, 3)

        cgrid.addWidget(QLabel(_("Profile:")), 1, 0)
        self.cb_profile = QComboBox()
        self.cb_profile.setToolTip(_("Which calibration profile drives the outline and the fade"))
        self.cb_profile.currentTextChanged.connect(self._on_profile_change)
        cgrid.addWidget(self.cb_profile, 1, 1)
        self.btn_delete_profile = QPushButton(_("Delete"))
        self.btn_delete_profile.setObjectName("secondary")
        self.btn_delete_profile.clicked.connect(self._on_delete_profile)
        cgrid.addWidget(self.btn_delete_profile, 1, 2)

        self.btn_calibrate = QPushButton(_("Calibrate…"))
        self.btn_calibrate.setToolTip(_("Open the camera and capture a new profile"))
        self.btn_calibrate.clicked.connect(self._on_calibrate)
        cgrid.addWidget(self.btn_calibrate, 2, 0, 1, 3)

        self.sw_reference_outline = self._switch(
            "reference_outline", "Use calibrated outline",
            "Put landmarks a finger or shadow pulled away back onto the "
            "profile's outline",
        )
        cgrid.addWidget(self.sw_reference_outline, 3, 0, 1, 3)
        self.sw_pose_fade = self._switch(
            "pose_fade", "Fade out on deep turns",
            "Past the calibrated turn limits show the real face instead of a "
            "smeared swap",
        )
        cgrid.addWidget(self.sw_pose_fade, 4, 0, 1, 3)
        self.sw_stable = self._switch(
            "stable_alignment", "Expression-proof crop",
            "Place the mouth corners from the profile so pursed or smiling "
            "lips do not shrink or grow the swapped area",
        )
        cgrid.addWidget(self.sw_stable, 5, 0, 1, 3)
        grid.addWidget(calib, 3, 0, 1, 2)
        self._refresh_profiles()

        grid.setRowStretch(4, 1)
        return page

    def _on_yaw_limit(self, value: float) -> None:
        modules.globals.head_yaw_limit = int(value)
        self.l_yaw_limit.setText(f"{int(value)}°")
        save_switch_states()

    def _on_pitch_limit(self, value: float) -> None:
        modules.globals.head_pitch_limit = int(value)
        self.l_pitch_limit.setText(f"{int(value)}°")
        save_switch_states()

    # ── output tab ───────────────────────────────────────────────────────

    def _build_output_tab(self) -> QWidget:
        page, grid = self._tab()

        self.sw_color_fix = self._switch("color_correction", "Fix Blueish Cam",
                                         "Fix blue/green color cast from some webcams")
        self.sw_show_fps = self._switch("show_fps", "Show FPS",
                                        "Display frames-per-second counter on the live preview")
        self.sw_mirror = self._switch("live_mirror", "Mirror camera",
                                      "Flip the live preview horizontally, like a front camera")
        items = [self.sw_color_fix, self.sw_show_fps, self.sw_mirror]
        for i, w in enumerate(items):
            grid.addWidget(w, i // 2, i % 2)
        row = (len(items) + 1) // 2
        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel(_("Max FPS")))
        self.s_max_fps = self._slider(0, 60, modules.globals.max_fps, 1, self._on_max_fps_change)
        self.s_max_fps.setToolTip(_("Cap on swapped frames per second to spare the GPU; "
                                    "0 = every camera frame"))
        self.l_max_fps = QLabel(self._fps_text(modules.globals.max_fps))
        self.l_max_fps.setMinimumWidth(48)
        fps_row.addWidget(self.s_max_fps, 1)
        fps_row.addWidget(self.l_max_fps)
        grid.addLayout(fps_row, row, 0, 1, 2)
        row += 1
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel(_("Language:")))
        self.cb_language = QComboBox()
        self.cb_language.addItems(list(LANGUAGES.values()))
        current = _LANG.current_language if _LANG is not None else "en"
        self.cb_language.setCurrentText(LANGUAGES.get(current, "English"))
        self.cb_language.setToolTip(_("Switches the interface language"))
        self.cb_language.currentTextChanged.connect(self._on_language_change)
        lang_row.addWidget(self.cb_language, 1)
        grid.addLayout(lang_row, row, 0, 1, 2)

        self.btn_destroy = QPushButton(_("Destroy"))
        self.btn_destroy.setObjectName("danger")
        self.btn_destroy.setToolTip(_("Stop processing and close the application"))
        self.btn_destroy.clicked.connect(lambda: self._destroy_cb())
        grid.addWidget(self.btn_destroy, row + 1, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setRowStretch(row + 2, 1)
        return page

    # ── calibration ──────────────────────────────────────────────────────

    def _refresh_profiles(self) -> None:
        self.cb_profile.blockSignals(True)
        self.cb_profile.clear()
        self.cb_profile.addItem("None")
        self.cb_profile.addItems(calibration.list_profiles())
        current = calibration.active()
        self.cb_profile.setCurrentText(current.name if current is not None else "None")
        self.cb_profile.blockSignals(False)
        self.btn_delete_profile.setEnabled(current is not None)

    def _on_profile_change(self, choice: str) -> None:
        if choice == "None" or not choice:
            calibration.set_active(None)
        elif calibration.activate_saved(choice) is None:
            update_status(f"Could not load calibration profile {choice}")
            self._refresh_profiles()
            return
        self.btn_delete_profile.setEnabled(calibration.active() is not None)
        save_switch_states()
        update_status(f"Calibration profile: {choice}")

    def _on_delete_profile(self) -> None:
        current = calibration.active()
        if current is None:
            return
        calibration.delete_profile(current.name)
        self._refresh_profiles()
        save_switch_states()
        update_status(f"Deleted calibration profile {current.name}")

    def _on_profile_saved(self, name: str) -> None:
        self._refresh_profiles()
        save_switch_states()
        update_status(f"Calibration profile saved and active: {name}")

    def _on_calibrate(self) -> None:
        global _CALIBRATION
        idx = self.cb_camera.currentIndex()
        if idx < 0 or idx >= len(self._camera_indices):
            update_status("No camera available")
            return
        if _CALIBRATION is not None and _CALIBRATION.isVisible():
            _CALIBRATION.raise_()
            return
        # One camera, one owner: the live preview gives it up for the dialog.
        if _WEBCAM_PREVIEW is not None and _WEBCAM_PREVIEW.isVisible():
            _WEBCAM_PREVIEW.close()
        from modules.face_analyser import get_face_analyser

        update_status("Loading models...")
        QApplication.processEvents()
        get_face_analyser()
        update_status("Opening camera for calibration...")
        QApplication.processEvents()
        _CALIBRATION = CalibrationDialog(
            self._camera_indices[idx], self._on_profile_saved, parent=self,
        )
        _CALIBRATION.show()

    # ── slot handlers ────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_busy(self, active: bool, text: str) -> None:
        self._busy_depth = max(0, self._busy_depth + (1 if active else -1))
        if active and text:
            self._status_label.setText(text)
        elif self._busy_depth == 0:
            # The task is over; a stale "loading..." line would read as hung.
            self._status_label.setText("")
        self._busy_bar.setVisible(self._busy_depth > 0)

    def _on_select_source(self) -> None:
        global _RECENT_SOURCE_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        paths, _filter = QFileDialog.getOpenFileNames(
            self, _("Select one or more photos of the source face"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        paths = [p for p in paths if is_image(p)]
        if not paths:
            return
        if not _accept_source_paths(paths):
            if modules.globals.source_path is None:
                self.source_label.clear()
                self.source_label.setText(_("Source face"))
            return
        _RECENT_SOURCE_DIR = os.path.dirname(paths[0])
        self._show_source()

    def _show_source(self) -> None:
        paths = modules.globals.source_paths or (
            [modules.globals.source_path] if modules.globals.source_path else [])
        if not paths:
            self.source_label.clear()
            self.source_label.setText(_("Source face"))
            self.source_caption.setText("")
            return
        self.source_label.setPixmap(render_image_preview(paths[0], self._preview_size))
        self.source_label.setText("")
        self.source_caption.setText(
            _("{n} photos, blended by pose").format(n=len(paths)) if len(paths) > 1 else "")

    def _on_select_target(self) -> None:
        global _RECENT_TARGET_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an target image or video"),
            _RECENT_TARGET_DIR or "",
            _MEDIA_FILE_FILTER,
        )
        if not path:
            return
        if is_image(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            self.target_label.setPixmap(render_image_preview(path, self._preview_size))
            self.target_label.setText("")
        elif is_video(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            pm = render_video_preview(path, self._preview_size)
            if pm:
                self.target_label.setPixmap(pm)
                self.target_label.setText("")
        else:
            modules.globals.target_path = None
            self.target_label.clear()
            self.target_label.setText(_("Target"))

    def _on_random_face(self) -> None:
        if _PREVIEW is not None:
            _PREVIEW.hide()
        try:
            response = requests.get(
                "https://thispersondoesnotexist.com/",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            response.raise_for_status()
            temp_path = os.path.join(tempfile.gettempdir(), "deep_live_cam_random_face.jpg")
            with open(temp_path, "wb") as f:
                f.write(response.content)
            if not _accept_source_path(temp_path):
                return
            self._show_source()
        except Exception as exc:
            print(f"Failed to fetch random face: {exc}")

    def _on_swap_paths(self) -> None:
        global _RECENT_SOURCE_DIR, _RECENT_TARGET_DIR
        sp = modules.globals.source_path
        tp = modules.globals.target_path
        if not (sp and tp and is_image(sp) and is_image(tp)):
            return
        if len(modules.globals.source_paths) > 1:
            update_status("Swap works with a single source photo.")
            return
        modules.globals.source_path, modules.globals.target_path = tp, sp
        modules.globals.source_paths = [tp]
        _RECENT_SOURCE_DIR = os.path.dirname(tp)
        _RECENT_TARGET_DIR = os.path.dirname(sp)
        if _PREVIEW is not None:
            _PREVIEW.hide()
        self.source_label.setPixmap(render_image_preview(tp, self._preview_size))
        self.target_label.setPixmap(render_image_preview(sp, self._preview_size))
        self.source_label.setText("")
        self.target_label.setText("")

    def _on_map_faces_toggled(self, value: bool) -> None:
        modules.globals.map_faces = value
        save_switch_states()
        if not value:
            close_mapper_window()

    def _on_swapper_change(self, choice: str) -> None:
        key = swapper_registry.LABEL_TO_KEY.get(choice)
        if key is None or key == modules.globals.face_swapper_model:
            return
        modules.globals.face_swapper_model = key
        # Unload the previous model now instead of at the next swapped frame,
        # so two swap models never sit in VRAM at the same time.  The new
        # one loads in the processing worker (settings epoch): only that
        # thread may touch the GPU while Live runs.
        _release_processor("face_swapper")
        _settings_changed()
        save_switch_states()
        update_status(f"Face swapper model: {choice}")

    def _on_occlusion_toggled(self, value: bool) -> None:
        modules.globals.occlusion_mask = value
        if not value:
            from modules.face_occluder import release as release_occluder

            release_occluder()
        _settings_changed()
        save_switch_states()

    def _on_eye_reveal_change(self, value: float) -> None:
        modules.globals.eye_reveal = value
        save_switch_states()

    def _on_backend_change(self, label: str) -> None:
        value = self._backend_map.get(label)
        if value is None or value == modules.globals.tensorrt:
            return
        modules.globals.tensorrt = value
        # Models are built for one backend; drop them so the next use
        # reloads on the chosen one.
        _release_processor("face_swapper")
        for key in enhancer_registry.KEYS:
            _release_processor(key)
        from modules.face_occluder import release as release_occluder

        release_occluder()
        head_geometry.release()
        _settings_changed()
        save_switch_states()
        update_status(_("Backend: {backend}. Models reload on next use; restart Live to apply.")
                      .format(backend=label))

    @staticmethod
    def _fps_text(value: int) -> str:
        return _("no cap") if value <= 0 else f"{value} fps"

    def _on_max_fps_change(self, value: float) -> None:
        modules.globals.max_fps = int(value)
        self.l_max_fps.setText(self._fps_text(int(value)))
        save_switch_states()

    def _on_language_change(self, label: str) -> None:
        global _LANG
        code = next((c for c, name in LANGUAGES.items() if name == label), "en")
        if _LANG is not None and code == _LANG.current_language:
            return
        _LANG = set_language(code)
        save_switch_states()
        # Every label was translated at construction: build a fresh window
        # in this one's place.  Preview and live windows are separate
        # top-levels and keep running.
        QTimer.singleShot(0, _rebuild_main_window)

    def _on_occlusion_interval_change(self, index: int) -> None:
        modules.globals.occlusion_interval = index + 1
        save_switch_states()

    def _on_alignment_change(self, choice: str) -> None:
        value = self._alignment_map.get(choice)
        if value is None:
            return
        modules.globals.enhancer_alignment = value
        save_switch_states()
        update_status(f"Enhancer crop: {choice}")

    def _on_enhancer_change(self, choice: str) -> None:
        for key in enhancer_registry.KEYS:
            _update_tumbler(key, False)
        selected = enhancer_registry.LABEL_TO_KEY.get(choice)
        if selected:
            _update_tumbler(selected, True)
        # Free every enhancer that is no longer selected, then resync the
        # shared processor list so the change applies to a running preview.
        for key in enhancer_registry.KEYS:
            if key != selected:
                _release_processor(key)
        get_frame_processors_modules(modules.globals.frame_processors)
        _settings_changed()
        save_switch_states()

    def _on_transparency_change(self, value: float) -> None:
        modules.globals.opacity = value
        pct = int(value * 100)
        if pct == 0:
            modules.globals.fp_ui["face_enhancer"] = False
            update_status("Transparency set to 0% - Face swapping disabled.")
        elif pct == 100:
            modules.globals.face_swapper_enabled = True
            update_status("Transparency set to 100%.")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"Transparency set to {pct}%")

    def _on_sharpness_change(self, value: float) -> None:
        modules.globals.sharpness = value
        update_status(f"Sharpness set to {value:.1f}")

    def _on_mouth_mask_change(self, value: float) -> None:
        modules.globals.mouth_mask_size = value
        modules.globals.mouth_mask = value > 0
        if value <= 0:
            modules.globals.show_mouth_mask_box = False

    def _on_mouth_mode_change(self, label: str) -> None:
        key = self._mouth_modes.get(label)
        if key:
            modules.globals.mouth_reveal_mode = key
            save_switch_states()

    def _on_mouth_mask_pressed(self) -> None:
        if modules.globals.mouth_mask_size > 0:
            modules.globals.show_mouth_mask_box = True

    def _on_mouth_mask_released(self) -> None:
        modules.globals.show_mouth_mask_box = False

    def _on_start(self) -> None:
        if _MAPPER is not None and _MAPPER.isVisible():
            update_status("Please complete pop-up or close it.")
            return
        if modules.globals.map_faces:
            modules.globals.source_target_map = []
            if is_image(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_image()
            elif is_video(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_video()
            if modules.globals.source_target_map:
                _open_mapper_dialog(self._start_cb, modules.globals.source_target_map)
            else:
                update_status("No faces found in target")
        else:
            self._select_output_and_start()

    def _select_output_and_start(self) -> None:
        global _RECENT_OUTPUT_DIR
        if is_image(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save image output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.png"),
                _IMAGE_FILE_FILTER,
            )
        elif is_video(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save video output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.mp4"),
                _VIDEO_FILE_FILTER,
            )
        else:
            return
        if path:
            modules.globals.output_path = path
            _RECENT_OUTPUT_DIR = os.path.dirname(path)
            self._start_cb()

    def _on_toggle_preview(self) -> None:
        if _PREVIEW is None:
            return
        if _PREVIEW.isVisible():
            _PREVIEW.hide()
        elif modules.globals.source_path and modules.globals.target_path:
            _PREVIEW.init_for_target()
            _PREVIEW.refresh_frame(0)
            _PREVIEW.show()

    def _on_live(self) -> None:
        idx = self.cb_camera.currentIndex()
        if idx < 0 or idx >= len(self._camera_indices):
            update_status("No camera available")
            return
        camera_index = self._camera_indices[idx]
        if _CALIBRATION is not None and _CALIBRATION.isVisible():
            _CALIBRATION.close()
        if _LIVE_MAPPER is not None and _LIVE_MAPPER.isVisible():
            update_status("Source x Target Mapper is already open.")
            _LIVE_MAPPER.raise_()
            return
        if not modules.globals.map_faces:
            from modules.processors.frame.face_swapper import needs_source_face

            if modules.globals.source_path is None and needs_source_face():
                update_status("Please select a source image first")
                return
            if self._loader is not None and self._loader.isRunning():
                # Second click while loading: give up on opening the camera.
                # A TensorRT build cannot be interrupted, so the thread
                # finishes in the background and the models stay loaded.
                self._loader.cancelled = True
                self.btn_live.setText(_("Live"))
                update_status("Live start cancelled.")
                return
            self._loader = _ModelLoader()
            self._loader.done.connect(lambda ok, cam=camera_index: self._on_models_loaded(ok, cam))
            self.btn_live.setText(_("Loading... (click to cancel)"))
            update_status("Loading models...")
            self._loader.start()
        else:
            modules.globals.source_target_map = []
            _open_live_mapper_dialog(camera_index, modules.globals.source_target_map)

    def _on_resolution_change(self, label: str) -> None:
        if label not in RESOLUTIONS or label == modules.globals.camera_resolution:
            return
        modules.globals.camera_resolution = label
        save_switch_states()
        if _WEBCAM_PREVIEW is not None and _WEBCAM_PREVIEW.isVisible():
            # The camera negotiates its size when it opens: reopen it.
            camera_index = _WEBCAM_PREVIEW.camera_index
            _WEBCAM_PREVIEW.close()
            _open_webcam_preview(camera_index)

    def _on_models_loaded(self, ok: bool, camera_index: int) -> None:
        self.btn_live.setText(_("Live"))
        loader, self._loader = self._loader, None
        if loader is None or loader.cancelled:
            return
        if not ok:
            update_status("Could not load the models — see the console for the error.")
            return
        update_status("Opening camera...")
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        _open_webcam_preview(camera_index)

    def closeEvent(self, event):
        # Treat OS-level close as Destroy click — unless the window is being
        # swapped for a retranslated copy.
        if self._loader is not None and self._loader.isRunning():
            # An engine build cannot be interrupted; leaving the thread
            # running while Qt tears down aborts the process.
            self._loader.cancelled = True
            self._loader.wait(120_000)
        if not getattr(self, "_replaced", False):
            self._destroy_cb()
        event.accept()


def _update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, refresh frame processors so
    # toggling enhancers takes effect immediately.
    if _WEBCAM_PREVIEW is not None and _WEBCAM_PREVIEW.isVisible():
        get_frame_processors_modules(modules.globals.frame_processors)


# ─── preview window (still-image / video scrub) ──────────────────────────


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("Preview"))
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self.refresh_frame)
        layout.addWidget(self._slider)

    def init_for_target(self) -> None:
        if is_image(modules.globals.target_path):
            self._slider.hide()
        elif is_video(modules.globals.target_path):
            total = get_video_frame_total(modules.globals.target_path)
            self._slider.setRange(0, max(0, total - 1))
            self._slider.setValue(0)
            self._slider.show()

    def refresh_frame(self, frame_number: int = 0) -> None:
        from modules.processors.frame.face_swapper import needs_source_face

        target = modules.globals.target_path
        if not target:
            return
        if needs_source_face() and not modules.globals.source_path:
            return
        update_status("Processing...")
        if is_image(target):
            temp_frame = imread_unicode(target)
        else:
            temp_frame = get_video_frame(target, frame_number)
        if temp_frame is None:
            update_status("Could not read the target — unsupported file or codec.")
            return
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        source_face = None
        if needs_source_face():
            from modules import source_identity

            source_face, _skipped = source_identity.load()
            if source_face is None:
                update_status("No face in the selected source image.")
                return
        from modules.processors.frame.core import get_frame_processors_modules as _gfpm
        for fp in _gfpm(modules.globals.frame_processors):
            temp_frame = fp.process_frame(source_face, temp_frame)
        # Fit to current widget size while preserving aspect ratio.
        h, w = temp_frame.shape[:2]
        bound_w = min(PREVIEW_MAX_WIDTH, max(self.width(), PREVIEW_DEFAULT_WIDTH))
        bound_h = min(PREVIEW_MAX_HEIGHT, max(self.height(), PREVIEW_DEFAULT_HEIGHT))
        ratio = min(bound_w / w, bound_h / h)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        temp_frame = cv2.resize(temp_frame, new_size, interpolation=cv2.INTER_LANCZOS4)
        self._image_label.setPixmap(_bgr_to_qpixmap(temp_frame))
        update_status("Processing succeed!")


# ─── webcam preview window ───────────────────────────────────────────────


# Consecutive failed camera reads before Live gives up on the device.
CAPTURE_MAX_MISSES = 90


class _ModelLoader(QThread):
    """Loads everything Live needs, off the UI thread.

    Model loads and TensorRT engine builds take seconds to a minute; on the
    UI thread they looked like a hung window.  The workers are not running
    yet, so this is still the only thread on the GPU (see
    ``_preload_live_models``).
    """

    done = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False

    def run(self) -> None:
        from modules.face_analyser import get_face_analyser
        from modules.processors.frame.face_swapper import get_face_swapper

        try:
            get_face_analyser()
            get_face_swapper()
            _preload_live_models()
            self.done.emit(True)
        except Exception as error:
            print(f"[live] model load failed: {error}")
            self.done.emit(False)


class _CaptureWorker(QThread):
    """Reads frames from the camera into a bounded queue. Drops on overflow.

    Every frame gets a sequence number, and the last few raw frames are kept
    in ``ring`` so the reprojector can re-anchor its tracker on the exact
    frame a swap result came from.
    """

    def __init__(self, cap, capture_queue: queue.Queue, stop_event: threading.Event):
        super().__init__()
        self._cap = cap
        self._queue = capture_queue
        self._stop = stop_event
        self.ring: dict = {}
        self.latest_seq = -1

    def run(self) -> None:
        seq = 0
        misses = 0
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                # A failed read is usually a hiccup (USB, another app
                # probing the device); only a run of them means it is gone.
                misses += 1
                if misses >= CAPTURE_MAX_MISSES:
                    update_status("The camera stopped delivering frames — "
                                  "another application may have taken it.")
                    self._stop.set()
                    break
                time.sleep(0.01)
                continue
            misses = 0
            if modules.globals.live_mirror:
                frame = gpu_flip(frame, 1)
            if modules.globals.reprojection:
                # The processing worker swaps in place, so the ring keeps a copy.
                self.ring[seq] = frame.copy()
                for old in [k for k in self.ring if k <= seq - RING_SIZE]:
                    self.ring.pop(old, None)
            elif self.ring:
                self.ring.clear()
            self.latest_seq = seq
            item = (seq, frame)
            seq += 1
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(item)
                except queue.Full:
                    pass


class _ProcessingWorker(QThread):
    """Pulls raw frames, runs detect/swap/enhance, pushes processed frames."""

    def __init__(self, capture_queue, processed_queue, stop_event, camera_fps: float):
        super().__init__()
        self._cq = capture_queue
        self._pq = processed_queue
        self._stop = stop_event
        self._fps = camera_fps

    def run(self) -> None:
        from modules.processors.frame.face_swapper import get_face_swapper

        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        source_image = None
        last_source_path = None
        prev_time = time.time()
        fps_update_interval = 0.5
        frame_count = 0
        fps = 0.0
        tracker = FaceTracker()
        cached_target_face = None
        cached_many_faces = None
        # Detection is the most expensive stage, and optical-flow tracking
        # carries the face between runs with ~2 px error — so detect roughly
        # 4x/second when tracking is on instead of 12x/second.  Measured in
        # wall time, not processed frames: a slow enhancer must not stretch
        # the gap between detections to seconds.
        det_period = 0.25 if modules.globals.face_tracking else 0.08
        last_detection = 0.0
        last_seq = -1
        seen_epoch = modules.globals.settings_epoch
        force_detect_until = -1

        last_processed = 0.0

        while not self._stop.is_set():
            try:
                seq, frame = self._cq.get(timeout=0.05)
            except queue.Empty:
                continue
            cap = modules.globals.max_fps
            if cap > 0 and time.perf_counter() - last_processed < 1.0 / cap:
                # FPS cap: skip the swap on this frame; the display keeps
                # showing camera frames through reprojection.
                continue
            last_processed = time.perf_counter()

            temp_frame = frame
            detected_now = False

            if not modules.globals.map_faces:
                paths = tuple(source_identity.current_paths())
                if paths and paths != last_source_path:
                    last_source_path = paths
                    identity, _skipped = source_identity.load(paths)
                    if identity is not None:
                        source_image = identity
                    elif source_image is None:
                        update_status("No face in the selected source image.")
                    else:
                        update_status("No face in the selected source image — "
                                      "keeping the previous one.")

                # Flow carries the face across one or two skipped camera
                # frames; across a bigger gap (a slow enhancer) it lands tens
                # of pixels off, and detection is cheaper than that mistake.
                if modules.globals.settings_epoch != seen_epoch:
                    # A toggle changed the picture (mirror, masks, models):
                    # re-detect on this and the next couple of frames — one
                    # may still be in flight from before the change.
                    seen_epoch = modules.globals.settings_epoch
                    tracker.reset()
                    force_detect_until = seq + 3
                    # Reload whatever a toggle unloaded, here rather than
                    # in the UI handler: this is the one thread allowed on
                    # the GPU while Live runs, and the display keeps the
                    # last composed face until the model is back.
                    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
                    get_face_swapper()
                    _preload_live_models()
                detected_now = (time.time() - last_detection >= det_period
                                or seq - last_seq >= 3 or seq <= force_detect_until)
                last_seq = seq
                if detected_now:
                    last_detection = time.time()
                    if modules.globals.many_faces:
                        cached_target_face = None
                        cached_many_faces = detect_many_faces_fast(temp_frame)
                        tracker.reset()
                    else:
                        cached_target_face = detect_one_face_fast(temp_frame)
                        cached_many_faces = None
                        if modules.globals.face_tracking:
                            cached_target_face = tracker.observe(
                                temp_frame, cached_target_face
                            )
                elif (
                    modules.globals.face_tracking
                    and not modules.globals.many_faces
                    and cached_target_face is not None
                ):
                    # Detection is too slow to run every frame; optical flow
                    # carries the face across the gap so it does not lag.
                    cached_target_face = tracker.track(temp_frame, cached_target_face)

                cached_faces = None
                if cached_many_faces:
                    cached_faces = cached_many_faces
                elif cached_target_face is not None:
                    cached_faces = [cached_target_face]

                # Fast detection skips the 2d106 landmark model, but the mouth
                # mask needs it. Attach landmarks on demand (computed once per
                # detection cycle — the helper no-ops if already present).
                if cached_faces and (
                    modules.globals.mouth_mask or modules.globals.face_outline_mask
                    or modules.globals.blink_reveal or modules.globals.eye_reveal > 0
                ):
                    ensure_landmarks(temp_frame, cached_faces)
                if cached_faces:
                    # Real head angles for the fade and the 3D silhouette
                    # for the outline; ~3 ms, skipped when switched off.
                    head_geometry.estimate(temp_frame, cached_faces)

                for fp in frame_processors:
                    enhancer_key = enhancer_registry.NAME_TO_KEY.get(fp.NAME)
                    if enhancer_key is not None:
                        if modules.globals.fp_ui.get(enhancer_key, False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-SWAPPER":
                        swapped_bboxes = []
                        if modules.globals.many_faces and cached_many_faces:
                            result = temp_frame.copy()
                            for t_face in cached_many_faces:
                                result = fp.swap_face(source_image, t_face, result)
                                if hasattr(t_face, "bbox") and t_face.bbox is not None:
                                    swapped_bboxes.append(t_face.bbox.astype(int))
                            temp_frame = result
                        elif cached_target_face is not None:
                            temp_frame = fp.swap_face(
                                source_image, cached_target_face, temp_frame
                            )
                            if (
                                hasattr(cached_target_face, "bbox")
                                and cached_target_face.bbox is not None
                            ):
                                swapped_bboxes.append(cached_target_face.bbox.astype(int))
                        temp_frame = fp.apply_post_processing(temp_frame, swapped_bboxes)
                    else:
                        temp_frame = fp.process_frame(source_image, temp_frame)
            else:
                modules.globals.target_path = None
                for fp in frame_processors:
                    enhancer_key = enhancer_registry.NAME_TO_KEY.get(fp.NAME)
                    if enhancer_key is not None:
                        if modules.globals.fp_ui.get(enhancer_key, False):
                            temp_frame = fp.process_frame_v2(temp_frame)
                    else:
                        temp_frame = fp.process_frame_v2(temp_frame)

            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            if modules.globals.show_mask_debug:
                from modules.processors.frame.face_swapper import draw_mask_debug

                temp_frame = draw_mask_debug(temp_frame)

            record = ProcessedFrame(seq, temp_frame, fps=fps, detected=detected_now)
            if (modules.globals.reprojection and not modules.globals.map_faces
                    and not modules.globals.many_faces and cached_target_face is not None):
                paste = collect_paste_alpha(temp_frame.shape)
                if paste is not None:
                    record.kps = np.asarray(cached_target_face.kps, dtype=np.float32).copy()
                    bbox = getattr(cached_target_face, "bbox", None)
                    record.bbox = None if bbox is None else np.asarray(bbox, dtype=np.float32).copy()
                    record.box, record.alpha = paste

            try:
                self._pq.put_nowait(record)
            except queue.Full:
                try:
                    self._pq.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._pq.put_nowait(record)
                except queue.Full:
                    pass


class WebcamPreviewWindow(QWidget):
    def __init__(self, camera_index: int):
        super().__init__()
        self.setWindowTitle("Live Preview")
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        # closeEvent can run before the camera is up (a failed open schedules
        # close) — everything it touches must exist from the start.
        self._stop_event: Optional[threading.Event] = None
        self._timer: Optional[QTimer] = None
        self._capture_worker = None
        self._processing_worker = None
        self._cap = None
        self._reprojector = Reprojector()
        self._record: Optional[ProcessedFrame] = None
        self._shown_seq = -1
        self._seen_epoch = modules.globals.settings_epoch

        self._image_label.setText(_("Opening camera..."))
        self.camera_index = camera_index
        self._cap = VideoCapturer(camera_index)
        width, height = RESOLUTIONS.get(modules.globals.camera_resolution, RESOLUTIONS["720p"])
        # processEvents keeps the window painted and closable while the camera
        # is being opened, which can take seconds — or never finish.
        if not self._cap.start(
            width, height, 60,
            on_wait=QApplication.processEvents,
        ):
            update_status(
                "Could not open the camera. It is probably in use by another "
                "application (virtual camera, conferencing app, browser tab)."
            )
            QTimer.singleShot(0, self.close)
            return

        camera_fps = self._cap.actual_fps
        print(
            f"[webcam] Camera running at {self._cap.actual_width}x"
            f"{self._cap.actual_height}@{camera_fps:.0f}fps"
        )
        update_status("")

        self._capture_queue: queue.Queue = queue.Queue(maxsize=2)
        self._processed_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

        self._capture_worker = _CaptureWorker(
            self._cap, self._capture_queue, self._stop_event
        )
        self._processing_worker = _ProcessingWorker(
            self._capture_queue, self._processed_queue, self._stop_event, camera_fps
        )
        self._capture_worker.start()
        self._processing_worker.start()

        # Poll at ~2x camera fps so we never block but also don't burn CPU.
        poll_ms = max(1, min(16, int(500 / max(camera_fps, 1))))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(poll_ms)

    def _tick(self) -> None:
        if self._stop_event.is_set():
            self.close()
            return
        fresh = False
        while True:
            try:
                self._record = self._processed_queue.get_nowait()
                fresh = True
            except queue.Empty:
                break
        record = self._record
        if record is None:
            return
        if modules.globals.settings_epoch != self._seen_epoch:
            self._seen_epoch = modules.globals.settings_epoch
            self._reprojector.reset()
        latest_seq = self._capture_worker.latest_seq
        if modules.globals.reprojection and record.kps is not None:
            # Show every camera frame: the last finished swap moved to where
            # the face is now, instead of waiting for the next swap.
            if not fresh and latest_seq <= self._shown_seq:
                return
            bgr_frame = self._reprojector.compose(record, self._capture_worker.ring, latest_seq)
            self._shown_seq = max(latest_seq, record.seq)
        else:
            if not fresh:
                return
            bgr_frame = record.frame
            self._shown_seq = record.seq
        if modules.globals.show_fps:
            if bgr_frame is record.frame:
                bgr_frame = bgr_frame.copy()
            cv2.putText(
                bgr_frame, f"FPS: {record.fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
        bgr_frame = fit_image_to_size(bgr_frame, self.width(), self.height())
        self._image_label.setPixmap(_bgr_to_qpixmap(bgr_frame))

    def closeEvent(self, event) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._timer is not None:
            self._timer.stop()
        for worker in (self._capture_worker, self._processing_worker):
            if worker is not None:
                try:
                    worker.wait(2000)
                except Exception:
                    pass
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        global _WEBCAM_PREVIEW
        if _WEBCAM_PREVIEW is self:
            _WEBCAM_PREVIEW = None
        event.accept()


def _open_webcam_preview(camera_index: int) -> None:
    global _WEBCAM_PREVIEW
    if _WEBCAM_PREVIEW is not None:
        _WEBCAM_PREVIEW.close()
    _WEBCAM_PREVIEW = WebcamPreviewWindow(camera_index)
    _WEBCAM_PREVIEW.show()


# ─── mapper dialogs (image/video + live) ────────────────────────────────


def _make_thumb(cv2_img: np.ndarray) -> QPixmap:
    rgb = gpu_cvt_color(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize(
        (MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE), Image.LANCZOS
    )
    return _pil_to_qpixmap(image)


class MapperDialog(QDialog):
    """Source × Target mapper for image / video processing."""

    def __init__(self, start_cb: Callable, mapping: list):
        super().__init__(_MAIN)
        self._start_cb = start_cb
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_WIDTH, POPUP_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_submit = QPushButton(_("Submit"))
        btn_submit.clicked.connect(self._on_submit)
        layout.addWidget(btn_submit, alignment=Qt.AlignmentFlag.AlignCenter)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn = QPushButton(_("Select source image"))
            btn.setFixedWidth(200)
            btn.clicked.connect(lambda _c, n=row: self._select_source(n))
            grid.addWidget(btn, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px solid #555;")
            grid.addWidget(tgt_label, row, 3)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_source(self, row: int) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row]["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_submit(self) -> None:
        if has_valid_map():
            self.accept()
            _MAIN._select_output_and_start()
        else:
            self.set_status("Atleast 1 source with target is required!")


class LiveMapperDialog(QDialog):
    """Source × Target mapper for live webcam mode."""

    def __init__(self, camera_index: int, mapping: list):
        super().__init__(_MAIN)
        self._camera_index = camera_index
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_LIVE_WIDTH, POPUP_LIVE_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        for text, slot in (
            (_("Add"), self._on_add),
            (_("Clear"), self._on_clear),
            (_("Submit"), self._on_submit),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn_s = QPushButton(_("Select source image"))
            btn_s.setFixedWidth(200)
            btn_s.clicked.connect(lambda _c, n=row: self._select_face(n, "source"))
            grid.addWidget(btn_s, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            btn_t = QPushButton(_("Select target image"))
            btn_t.setFixedWidth(200)
            btn_t.clicked.connect(lambda _c, n=row: self._select_face(n, "target"))
            grid.addWidget(btn_t, row, 3)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(tgt_label, row, 4)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_face(self, row: int, kind: str) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row][kind] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_add(self) -> None:
        add_blank_map()
        self._rebuild()
        self.set_status("Please provide mapping!")

    def _on_clear(self) -> None:
        for item in self._map:
            item.pop("source", None)
            item.pop("target", None)
        self._rebuild()
        self.set_status("All mappings cleared!")

    def _on_submit(self) -> None:
        if has_valid_map():
            simplify_maps()
            self.set_status("Mappings successfully submitted!")
            self.accept()
            _open_webcam_preview(self._camera_index)
        else:
            self.set_status("At least 1 source with target is required!")


def _open_mapper_dialog(start_cb: Callable, mapping: list) -> None:
    global _MAPPER
    close_mapper_window()
    _MAPPER = MapperDialog(start_cb, mapping)
    _MAPPER.show()


def _open_live_mapper_dialog(camera_index: int, mapping: list) -> None:
    global _LIVE_MAPPER
    close_mapper_window()
    _LIVE_MAPPER = LiveMapperDialog(camera_index, mapping)
    _LIVE_MAPPER.show()


def close_mapper_window() -> None:
    global _MAPPER, _LIVE_MAPPER
    if _MAPPER is not None:
        _MAPPER.close()
        _MAPPER = None
    if _LIVE_MAPPER is not None:
        _LIVE_MAPPER.close()
        _LIVE_MAPPER = None


# ─── entry point ─────────────────────────────────────────────────────────


def _rebuild_main_window() -> None:
    """Replace the main window with one built in the current language."""
    global _MAIN
    old = _MAIN
    if old is None:
        return
    fresh = MainWindow(old._start_cb, old._destroy_cb)
    fresh.setGeometry(old.geometry())
    fresh._tabs.setCurrentIndex(old._tabs.currentIndex())
    fresh._modes.setCurrentIndex(old._modes.currentIndex())
    if _BRIDGE is not None:
        try:
            _BRIDGE.statusChanged.disconnect(old.set_status)
        except (RuntimeError, TypeError):
            pass
        _BRIDGE.statusChanged.connect(fresh.set_status)
        try:
            _BRIDGE.busyChanged.disconnect(old.set_busy)
        except (RuntimeError, TypeError):
            pass
        _BRIDGE.busyChanged.connect(fresh.set_busy)
    if modules.globals.source_path or modules.globals.target_path:
        fresh._refresh_previews()
    _MAIN = fresh
    fresh.show()
    old._replaced = True
    old.close()
    old.deleteLater()


class _Window:
    """Thin wrapper exposing .mainloop() for core.py compatibility."""

    def __init__(self, app: QApplication, main_window: MainWindow):
        self._app = app
        self._main = main_window

    def mainloop(self) -> None:
        self._main.show()
        self._app.exec()


def init(
    start: Callable[[], None], destroy: Callable[[], None], lang: str
) -> _Window:
    global _APP, _MAIN, _PREVIEW, _LANG, _BRIDGE

    # A language chosen in the UI outlives the command line default.
    saved = _saved_language()
    _LANG = set_language(saved or lang)
    if QApplication.instance() is None:
        _APP = QApplication(sys.argv)
    else:
        _APP = QApplication.instance()
    _APP.setStyleSheet(QSS)

    _BRIDGE = _UIBridge()
    _MAIN = MainWindow(start, destroy)
    _PREVIEW = PreviewWindow()

    # Route status updates onto the UI thread regardless of caller.
    _BRIDGE.statusChanged.connect(_MAIN.set_status)
    _BRIDGE.busyChanged.connect(_MAIN.set_busy)
    from modules import core

    core.BUSY_HOOK = set_busy

    return _Window(_APP, _MAIN)
