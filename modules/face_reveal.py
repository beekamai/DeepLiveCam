"""Regions of the real face that show through the swap: eyes and mouth.

Both are built in aligned-crop space from the 106 landmarks and returned as
multipliers for the paste alpha (1 keeps the swap, 0 shows the camera), so
they ride the same paste as the outline and occlusion masks.

Eyes: the swap models never fully close an eye — a blink comes out as a
hard squint — and paint the eyes as a pair, so the person cannot cross them.
The eye region is handed back fully during a blink (closed lids are skin,
not identity) and optionally at a chosen strength all the time.

Mouth: the "Mouth Mask" slider exposes the real lips, then — as it grows —
everything from under the nose to the chin, so a moustache or beard is
either all real or all swapped, never cut through the middle.

Aperture is measured on every frame the landmarks are computed for (the
tracker drops them each frame, so that is every frame in live mode), which
keeps blink detection at camera rate regardless of how slow the swap is.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

import modules.globals

# Lid contour points per eye in the 106-point layout (the centre points 38
# and 88 are left out — they do not sit on the lid).
LEFT_EYE = (33, 34, 35, 36, 37, 39, 40, 41, 42)
RIGHT_EYE = (87, 89, 90, 91, 92, 93, 94, 95, 96)
MOUTH = tuple(range(52, 72))
NOSE = tuple(range(72, 87))
CHIN = 0
# An eye counts as shut below this fraction of its open aperture and as open
# above the upper one; in between the reveal ramps.
CLOSED_FRACTION = 0.45
OPEN_FRACTION = 0.7
# The open reference follows the widest aperture seen, decaying slowly so a
# person who squints for a while is not stuck with a stale maximum.
OPEN_DECAY = 0.998

_open_reference = [0.0, 0.0]


def reset() -> None:
    _open_reference[0] = _open_reference[1] = 0.0


def apertures(landmarks: np.ndarray) -> Tuple[float, float]:
    """Height-to-width ratio of each eye's lid contour."""
    points = np.asarray(landmarks, dtype=np.float32)
    out = []
    for idx in (LEFT_EYE, RIGHT_EYE):
        eye = points[list(idx)]
        width = float(eye[:, 0].max() - eye[:, 0].min())
        height = float(eye[:, 1].max() - eye[:, 1].min())
        out.append(height / max(width, 1e-3))
    return out[0], out[1]


def blink_alphas(landmarks: np.ndarray) -> Tuple[float, float]:
    """Per-eye reveal strength, 0 open .. 1 shut, and update the open reference."""
    result = []
    for i, aperture in enumerate(apertures(landmarks)):
        reference = max(_open_reference[i] * OPEN_DECAY, aperture)
        _open_reference[i] = reference
        ratio = aperture / max(reference, 1e-3)
        if ratio <= CLOSED_FRACTION:
            result.append(1.0)
        elif ratio >= OPEN_FRACTION:
            result.append(0.0)
        else:
            t = (ratio - CLOSED_FRACTION) / (OPEN_FRACTION - CLOSED_FRACTION)
            result.append(float(1.0 - t * t * (3.0 - 2.0 * t)))
    return result[0], result[1]


def _crop_points(face: Any, affine: np.ndarray) -> Optional[np.ndarray]:
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is None or len(landmarks) < 100:
        return None
    return cv2.transform(
        np.asarray(landmarks, dtype=np.float32).reshape(1, -1, 2), affine,
    ).reshape(-1, 2)


def eye_reveal_mask(face: Any, affine: np.ndarray, size: int) -> Optional[np.ndarray]:
    """Multiplier for the paste alpha in aligned-crop space: 1 keeps the
    swap, lower values let the real eye through.  ``None`` when nothing is
    to be revealed, so callers skip the work."""
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is None or len(landmarks) < 100:
        return None
    strength = float(getattr(modules.globals, "eye_reveal", 0.0))
    strength = min(max(strength, 0.0), 1.0)
    if getattr(modules.globals, "blink_reveal", True):
        blink = blink_alphas(landmarks)
    else:
        blink = (0.0, 0.0)
    weights = (max(strength, blink[0]), max(strength, blink[1]))
    if max(weights) <= 0.0:
        return None

    points = _crop_points(face, affine)
    mask = np.zeros((size, size), dtype=np.float32)
    grow = max(2, size // 40)
    for idx, weight in zip((LEFT_EYE, RIGHT_EYE), weights):
        if weight <= 0.0:
            continue
        hull = cv2.convexHull(points[list(idx)].astype(np.int32))
        eye = np.zeros((size, size), dtype=np.uint8)
        cv2.fillConvexPoly(eye, hull, 255)
        eye = cv2.dilate(eye, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1)))
        mask = np.maximum(mask, eye.astype(np.float32) / 255.0 * weight)
    blur = max(3, (size // 24) | 1)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return 1.0 - mask


def mouth_polygon(face: Any, affine: np.ndarray, strength: float) -> Optional[np.ndarray]:
    """Outline of the revealed mouth region in aligned-crop space.

    At 0 it is the lips' own hull; as ``strength`` grows the region widens
    and reaches up to the base of the nose and down to the chin, so what is
    revealed is bounded by facial features rather than by a fixed offset.
    """
    points = _crop_points(face, affine)
    if points is None:
        return None
    mouth = points[list(MOUTH)]
    centre = mouth.mean(axis=0)
    nose_base = float(points[list(NOSE)][:, 1].max())
    chin = float(points[CHIN, 1])
    top = float(mouth[:, 1].min())
    bottom = float(mouth[:, 1].max())
    s = min(max(strength, 0.0), 1.0)
    widened = centre + (mouth - centre) * np.array([1.0 + 0.6 * s, 1.0], dtype=np.float32)
    reach_up = top - s * 0.9 * max(top - nose_base, 0.0)
    reach_down = bottom + s * 0.9 * max(chin - bottom, 0.0)
    left = float(widened[:, 0].min())
    right = float(widened[:, 0].max())
    extra = np.array([[left, reach_up], [right, reach_up],
                      [left, reach_down], [right, reach_down]], dtype=np.float32)
    return cv2.convexHull(np.vstack([widened, extra]).astype(np.float32)).reshape(-1, 2)


def mouth_reveal_mask(face: Any, affine: np.ndarray, size: int,
                      strength: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Paste-alpha multiplier revealing the real mouth, plus its outline in
    crop space for the on-screen box."""
    if strength <= 0.0:
        return None
    polygon = mouth_polygon(face, affine, strength)
    if polygon is None:
        return None
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon.astype(np.int32), 255)
    blur = max(3, (size // 16) | 1)
    soft = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32) / 255.0
    return 1.0 - soft, polygon
