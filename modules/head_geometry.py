"""Head geometry per frame: real head angles and a 3D-consistent outline.

The five detector keypoints and the 106 landmarks are 2D: a turned head
gives pose *proxies* and a jaw contour that slides and collapses on the far
side.  A 3D landmark fit gives the actual yaw / pitch / roll and a contour
that stays a silhouette at any angle, so the swap can fade by real degrees
without a calibration profile and the mask keeps covering the visible face.

One interface, several providers.  ``Landmark3DProvider`` runs insightface's
``1k3d68`` (68 landmarks in 3D, ~3 ms per face, ships with the buffalo_l
pack already installed).  A dense mesh (MediaPipe / 3DDFA) or an iPhone
TrueDepth stream can replace it later behind the same ``HeadPose``.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

import modules.globals

NAME = "DLC.HEAD-GEOMETRY"
MODEL_FILE = "1k3d68.onnx"
# Landmark indices of the 68-point layout.
JAW = slice(0, 17)
BROWS = slice(17, 27)
# The 68 points stop at the brows; lift the brow line by this share of the
# face height to close the outline over the forehead.
FOREHEAD_RISE = 0.35
# Past the limit the swap fades to the real face over this share of it.
FADE_SPAN = 0.3

_LOCK = threading.Lock()
_PROVIDER: Optional["Landmark3DProvider"] = None


@dataclass
class HeadPose:
    yaw: float      # degrees, positive when the face turns to its left (image right)
    pitch: float    # degrees, positive looking up
    roll: float     # degrees
    landmarks: np.ndarray   # (68, 3) in frame pixels, z relative

    def outline(self) -> np.ndarray:
        """Face silhouette from the 3D fit, (N, 2) frame pixels: jaw contour
        plus the brow line mirrored upward for the forehead."""
        points = self.landmarks[:, :2]
        jaw = points[JAW]
        brows = points[BROWS]
        height = float(points[:, 1].max() - points[:, 1].min())
        forehead = brows - np.array([0.0, height * FOREHEAD_RISE], dtype=np.float32)
        return np.vstack([jaw, forehead]).astype(np.float32)


def _model_path() -> Optional[str]:
    from modules.model_downloader import ensure_insightface_pack

    ensure_insightface_pack("buffalo_l")
    path = os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l", MODEL_FILE)
    return path if os.path.isfile(path) else None


class Landmark3DProvider:
    """insightface's 68-point 3D landmark model as the geometry source."""

    def __init__(self) -> None:
        from insightface.model_zoo.landmark import Landmark
        from modules.paths import MODELS_DIR
        from modules.processors.frame._onnx_enhancer import create_onnx_session
        from modules.cuda_graph import build_static_model

        path = _model_path()
        if path is None:
            raise FileNotFoundError(f"{MODEL_FILE} not found in the buffalo_l pack")
        static = build_static_model(path, os.path.join(MODELS_DIR, "1k3d68_192x192.onnx"), {0: 1})
        session = create_onnx_session(static or path)
        self._model = Landmark(model_file=path, session=session)

    def estimate(self, frame: np.ndarray, face: Any) -> Optional[HeadPose]:
        if face is None or getattr(face, "bbox", None) is None:
            return None
        landmarks = self._model.get(frame, face)
        pose = getattr(face, "pose", None)
        if pose is None:
            return None
        pitch, yaw, roll = (float(v) for v in pose)
        return HeadPose(yaw=yaw, pitch=pitch, roll=roll,
                        landmarks=np.asarray(landmarks, dtype=np.float32))


def get_provider() -> Optional[Landmark3DProvider]:
    global _PROVIDER
    with _LOCK:
        if _PROVIDER is None:
            try:
                from modules.core import busy

                with busy("Loading 3D head pose model..."):
                    _PROVIDER = Landmark3DProvider()
                print(f"{NAME}: 3D landmark model loaded")
            except Exception as error:
                print(f"{NAME}: unavailable ({error}); head pose off")
                modules.globals.head_pose = False
                return None
    return _PROVIDER


def release() -> None:
    global _PROVIDER
    with _LOCK:
        _PROVIDER = None


def enabled() -> bool:
    return bool(getattr(modules.globals, "head_pose", True))


def estimate(frame: np.ndarray, faces: Any) -> None:
    """Attach ``head_pose`` to each face (in place) when the feature is on."""
    if not enabled() or faces is None:
        return
    if not isinstance(faces, (list, tuple)):
        faces = [faces]
    provider = get_provider()
    if provider is None:
        return
    for face in faces:
        if face is None or getattr(face, "head_pose", None) is not None:
            continue
        try:
            face.head_pose = provider.estimate(frame, face)
        except Exception as error:
            print(f"{NAME}: estimate failed ({error})")
            face.head_pose = None


def _ramp(angle: float, limit: float) -> float:
    """1 within ``limit`` degrees, 0 past ``limit * (1 + FADE_SPAN)``."""
    span = max(1.0, limit * FADE_SPAN)
    return float(np.clip((limit + span - abs(angle)) / span, 0.0, 1.0))


def pose_alpha(face: Any) -> float:
    """Swap strength from the real head angles: fades to the real face past
    the yaw / pitch limits in the settings.  1 when unavailable or off."""
    head = getattr(face, "head_pose", None)
    if head is None or not enabled():
        return 1.0
    return _ramp(head.yaw, float(modules.globals.head_yaw_limit)) * \
        _ramp(head.pitch, float(modules.globals.head_pitch_limit))
