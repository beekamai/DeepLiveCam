"""Per-person face calibration: a reference outline and pose limits.

Live landmarks get pulled by fingers, shadows and strong turns, and a fixed
ellipse knows nothing about the face in front of it.  A short calibration
records, for one person:

* the neutral face outline — 106 landmarks in aligned-crop space — used to
  reject individual landmarks that stray from it while still following the
  head as a whole (a finger drags a handful of points; a turn moves all of
  them consistently);
* how far that person turns before the swap should hand the real face back:
  yaw/pitch proxies at the extremes they chose, used to fade the swap out
  smoothly instead of painting a smeared mask.

Profiles are JSON files under ``calibration/``; one is active at a time.  No
Qt in here — the dialog lives in ``modules.ui_calibration``.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import modules.globals

PROFILE_DIR = os.path.join(os.path.dirname(modules.globals.ROOT_DIR), "calibration")
REFERENCE_TEMPLATE = "arcface_128"
REFERENCE_SIZE = 128

# Calibration steps, in the order the dialog walks through them.  The
# neutral step is mandatory; the four extremes may be skipped.
STEPS = ("neutral", "left", "right", "up", "down")
STEP_PROMPTS = {
    "neutral": "Look straight at the camera, relaxed face.",
    "left": "Turn your head LEFT as far as the swap should still hold.",
    "right": "Turn your head RIGHT as far as the swap should still hold.",
    "up": "Tilt your head UP as far as the swap should still hold.",
    "down": "Tilt your head DOWN as far as the swap should still hold.",
}
# Frames averaged per capture: enough to cancel landmark jitter, short enough
# that the user does not have to hold an awkward pose.
CAPTURE_FRAMES = 12
# The captured extremes are where the swap should *still hold*: it stays
# fully on up to them and fades out over this fraction of the range beyond
# (0.3 = gone at 130 % of the extreme).
DEFAULT_FADE_SPAN = 0.3
# Pose distance (in units of the calibrated range) at which a reference
# stops contributing to the blended outline.
REFERENCE_POSE_SCALE = 0.15
# RANSAC reprojection threshold, in crop widths, for matching live landmarks
# to the reference outline.  Above it a landmark counts as pulled away.
REFERENCE_TOLERANCE = 0.02
# Fewer inliers than this (half the set) and the consensus may be the
# occluder, not the face — leave the points alone.
REFERENCE_MIN_INLIERS = 53
# 106-point indices standing in for the five detector keypoints.
KPS_FROM_LANDMARKS = (38, 88, 86, 52, 61)


@dataclass
class CalibrationProfile:
    name: str
    reference: np.ndarray                 # (106, 2) in normalised arcface_128 space
    neutral: Tuple[float, float]          # (yaw, pitch) proxies looking straight
    limits: Dict[str, float] = field(default_factory=dict)   # step -> proxy value
    fade_span: float = DEFAULT_FADE_SPAN
    created: str = ""
    reference_kps: Optional[np.ndarray] = None   # (5, 2), same space as reference
    # Every captured pose: step -> {"outline": (106, 2), "kps": (5, 2), "pose": (yaw, pitch)}.
    # The outline of a turned head differs from the neutral one beyond what an
    # affine can absorb, so the reference is blended from the nearest poses.
    references: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reference_kps is None:
            self.reference_kps = self.reference[list(KPS_FROM_LANDMARKS)].copy()
        if "neutral" not in self.references:
            self.references["neutral"] = {
                "outline": self.reference, "kps": self.reference_kps, "pose": tuple(self.neutral),
            }

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "name": self.name,
            "created": self.created,
            "reference": np.asarray(self.reference, dtype=np.float32).round(5).tolist(),
            "reference_kps": np.asarray(self.reference_kps, dtype=np.float32).round(5).tolist(),
            "neutral": [float(self.neutral[0]), float(self.neutral[1])],
            "limits": {k: float(v) for k, v in self.limits.items()},
            "fade_span": float(self.fade_span),
            "references": {
                step: {
                    "outline": np.asarray(r["outline"], dtype=np.float32).round(5).tolist(),
                    "kps": np.asarray(r["kps"], dtype=np.float32).round(5).tolist(),
                    "pose": [float(r["pose"][0]), float(r["pose"][1])],
                }
                for step, r in self.references.items()
            },
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "CalibrationProfile":
        reference = np.asarray(data["reference"], dtype=np.float32)
        if reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 100:
            raise ValueError("reference outline is malformed")
        neutral = data.get("neutral") or [0.0, 0.5]
        kps = data.get("reference_kps")
        references = {}
        for step, r in (data.get("references") or {}).items():
            outline = np.asarray(r.get("outline"), dtype=np.float32)
            if outline.shape != reference.shape:
                continue
            references[step] = {
                "outline": outline,
                "kps": np.asarray(r.get("kps"), dtype=np.float32),
                "pose": (float(r["pose"][0]), float(r["pose"][1])),
            }
        return cls(
            reference_kps=None if kps is None else np.asarray(kps, dtype=np.float32),
            name=str(data.get("name") or "profile"),
            reference=reference,
            neutral=(float(neutral[0]), float(neutral[1])),
            limits={k: float(v) for k, v in (data.get("limits") or {}).items()},
            fade_span=float(data.get("fade_span", DEFAULT_FADE_SPAN)),
            created=str(data.get("created") or ""),
            references=references,
        )

    # ── pose ─────────────────────────────────────────────────────────────

    def turn(self, kps: Any) -> float:
        """How far the head is from neutral, as a fraction of the recorded
        extremes: 0 looking straight, 1 at the limit, >1 beyond it."""
        yaw, pitch = pose_proxies(kps)
        return max(
            _turn_ratio(yaw, self.neutral[0], self.limits.get("left"), self.limits.get("right")),
            _turn_ratio(pitch, self.neutral[1], self.limits.get("up"), self.limits.get("down")),
        )

    def swap_alpha(self, kps: Any) -> float:
        """1 up to the calibrated extreme — that is where the swap was asked
        to still hold — then a smooth ramp to 0 over ``fade_span`` beyond."""
        if not self.limits:
            return 1.0
        turn = self.turn(kps)
        span = max(self.fade_span, 0.05)
        if turn <= 1.0:
            return 1.0
        if turn >= 1.0 + span:
            return 0.0
        t = (turn - 1.0) / span
        return float(1.0 - t * t * (3.0 - 2.0 * t))

    def _pose_scale(self) -> Tuple[float, float]:
        """Half-ranges of the calibrated turns, for pose distances."""
        yaw = [abs(self.limits[k] - self.neutral[0]) for k in ("left", "right") if k in self.limits]
        pitch = [abs(self.limits[k] - self.neutral[1]) for k in ("up", "down") if k in self.limits]
        return (max(yaw) if yaw else 0.3, max(pitch) if pitch else 0.4)

    def blended(self, kps: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Reference outline and keypoints for this head pose: the captured
        poses weighted by their closeness to the current one."""
        if len(self.references) == 1:
            return self.reference, self.reference_kps
        yaw, pitch = pose_proxies(kps)
        sy, sp = self._pose_scale()
        weights = []
        for r in self.references.values():
            d = ((yaw - r["pose"][0]) / sy) ** 2 + ((pitch - r["pose"][1]) / sp) ** 2
            weights.append(1.0 / (d + REFERENCE_POSE_SCALE ** 2))
        total = float(sum(weights))
        outline = sum(w * r["outline"] for w, r in zip(weights, self.references.values())) / total
        kps5 = sum(w * r["kps"] for w, r in zip(weights, self.references.values())) / total
        return outline.astype(np.float32), kps5.astype(np.float32)


def _turn_ratio(value: float, neutral: float,
                limit_a: Optional[float], limit_b: Optional[float]) -> float:
    """Distance from neutral relative to the limit lying on the same side."""
    delta = value - neutral
    ratio = 0.0
    for limit in (limit_a, limit_b):
        if limit is None:
            continue
        span = limit - neutral
        if abs(span) < 1e-4 or (span > 0) != (delta > 0):
            continue
        ratio = max(ratio, delta / span)
    return ratio


def pose_proxies(kps: Any) -> Tuple[float, float]:
    """Yaw and pitch proxies from the five detector keypoints.

    Measured along the eyes-to-mouth axis so head roll does not leak in:
    yaw is the nose tip's sideways offset, pitch its position between the
    eye line (0) and the mouth line (1), both in units of eye-mouth distance.
    Monotonic in the real angles, which is all the calibration needs.
    """
    points = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
    if len(points) < 5:
        return 0.0, 0.5
    eyes = (points[0] + points[1]) * 0.5
    mouth = (points[3] + points[4]) * 0.5
    down = mouth - eyes
    height = float(np.linalg.norm(down))
    if height < 1e-3:
        return 0.0, 0.5
    down /= height
    right = np.array([down[1], -down[0]], dtype=np.float32)
    rel = points[2] - eyes
    return float(rel @ right / height), float(rel @ down / height)


# ── reference outline ────────────────────────────────────────────────────

def _reference_affine(face: Any) -> np.ndarray:
    from insightface.utils.face_align import estimate_norm

    kps = np.asarray(face.kps, dtype=np.float32)
    return estimate_norm(kps, REFERENCE_SIZE) / REFERENCE_SIZE


def normalised_kps(face: Any) -> np.ndarray:
    """The detector keypoints in normalised arcface_128 crop space."""
    return cv2.transform(
        np.asarray(face.kps, dtype=np.float32).reshape(1, -1, 2), _reference_affine(face),
    ).reshape(-1, 2)


def normalised_landmarks(face: Any) -> Optional[np.ndarray]:
    """The face's 106 landmarks in normalised arcface_128 crop space."""
    landmarks = getattr(face, "landmark_2d_106", None)
    kps = getattr(face, "kps", None)
    if landmarks is None or kps is None or len(landmarks) < 100:
        return None
    return cv2.transform(
        np.asarray(landmarks, dtype=np.float32).reshape(1, -1, 2), _reference_affine(face),
    ).reshape(-1, 2)


def refine_outline(points: np.ndarray, frame_key: str, face: Any = None) -> np.ndarray:
    """Replace landmarks that wandered off the calibrated outline.

    An affine fit from the reference to the live points absorbs whatever the
    head did as a whole — turn, tilt, the foreshortening of a yaw — and the
    difference between alignment templates, so ``frame_key`` only names the
    caller.  The RANSAC outliers are the points something else moved: a
    finger, a shadow edge, a bad landmark frame.  Those take the reference's
    position under the same fit.  Without an active profile the points pass
    through.
    """
    profile = _ACTIVE
    if profile is None or not getattr(modules.globals, "reference_outline", True):
        return points
    reference = profile.reference
    if face is not None:
        kps = getattr(face, "raw_kps", None)
        if kps is None:
            kps = getattr(face, "kps", None)
        if kps is not None:
            reference = profile.blended(kps)[0]
    if reference.shape != points.shape:
        return points
    affine, inliers = cv2.estimateAffine2D(
        reference, points, method=cv2.RANSAC,
        ransacReprojThreshold=REFERENCE_TOLERANCE, maxIters=200, confidence=0.99,
    )
    if affine is None or inliers is None or int(inliers.sum()) < REFERENCE_MIN_INLIERS:
        return points
    fitted = cv2.transform(reference.reshape(1, -1, 2), affine).reshape(-1, 2)
    outliers = inliers.ravel() == 0
    if not outliers.any():
        return points
    refined = points.copy()
    refined[outliers] = fitted[outliers]
    return refined


# ── active profile ───────────────────────────────────────────────────────

_ACTIVE: Optional[CalibrationProfile] = None


def active() -> Optional[CalibrationProfile]:
    return _ACTIVE


def set_active(profile: Optional[CalibrationProfile]) -> None:
    global _ACTIVE
    _ACTIVE = profile
    modules.globals.calibration_profile = profile.name if profile else None


def swap_alpha(face: Any) -> float:
    """Swap strength for this face's pose under the active profile: 1 means
    swap as usual, 0 means show the real face."""
    # A face the detector lost (turned too far) is still carried by the
    # tracker for a moment, fading out — that fade applies with or without
    # a profile.
    # insightface's Face answers None for any unknown attribute.
    hold = getattr(face, "track_alpha", None)
    hold = 1.0 if hold is None else float(hold)
    # Real head angles (3D landmark fit) fade the swap past the configured
    # limits whether or not a profile is active; a profile can only narrow.
    from modules.head_geometry import pose_alpha

    hold *= pose_alpha(face)
    profile = _ACTIVE
    if profile is None or not getattr(modules.globals, "pose_fade", True):
        return hold
    # The pose has to come from the detector's own keypoints: the stabilised
    # ones are a rigid copy of the neutral face and carry no turn.
    kps = getattr(face, "raw_kps", None)
    if kps is None:
        kps = getattr(face, "kps", None)
    if kps is None:
        return hold
    return hold * profile.swap_alpha(kps)


def stable_kps(face: Any) -> Optional[np.ndarray]:
    """Five alignment keypoints whose mouth corners ignore the lips.

    The swap crop is aligned by a similarity fit of the five detector
    keypoints, two of which are the mouth corners — so lips pursed into a
    pout pull the corners inward, the fit scales *up* (+7 % for a 25 %
    pucker) and the crop covers 14 % less face: the real face shows around
    the edge.  A smile does the opposite.

    The eyes and nose stay the detector's own (they survive a finger far
    better than the 106 landmarks do); the mouth corners are the profile's
    neutral corners carried over by the similarity those three points
    define, then slid vertically onto the live corners' height so a tilted
    head keeps its foreshortening.  Returns ``None`` without a profile.
    """
    profile = _ACTIVE
    kps = getattr(face, "kps", None)
    if profile is None or kps is None:
        return None
    kps = np.asarray(kps, dtype=np.float32)
    if kps.shape != (5, 2):
        return None
    reference_kps = profile.blended(kps)[1]
    affine = cv2.estimateAffinePartial2D(
        reference_kps[:3], kps[:3], method=cv2.LMEDS,
    )[0]
    if affine is None:
        return None
    corners = cv2.transform(reference_kps[3:5].reshape(1, -1, 2), affine).reshape(-1, 2)
    # Height from the detector's own corners: they arrive smoothed by the
    # tracker, whereas the 106 landmarks are recomputed raw every frame and
    # their jitter showed up as a tremor of the whole swap.
    corners[:, 1] += float(kps[3:5, 1].mean() - corners[:, 1].mean())
    return np.vstack([kps[:3], corners]).astype(np.float32)


def stabilise(face: Any) -> Any:
    """Swap the face's alignment keypoints for the stable ones, once.

    The detector's keypoints are kept on ``raw_kps`` for the pose proxies.
    """
    if face is None or getattr(face, "raw_kps", None) is not None:
        return face
    if not getattr(modules.globals, "stable_alignment", True):
        return face
    stable = stable_kps(face)
    if stable is None:
        return face
    face.raw_kps = np.asarray(face.kps, dtype=np.float32).copy()
    face.kps = stable
    return face


# ── storage ──────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    return slug or "profile"


def profile_path(name: str) -> str:
    return os.path.join(PROFILE_DIR, f"{_slug(name)}.json")


def list_profiles() -> List[str]:
    if not os.path.isdir(PROFILE_DIR):
        return []
    names = []
    for entry in sorted(os.listdir(PROFILE_DIR)):
        if entry.lower().endswith(".json"):
            names.append(entry[:-5])
    return names


def load_profile(name: str) -> Optional[CalibrationProfile]:
    path = name if os.path.isfile(name) else profile_path(name)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return CalibrationProfile.from_json(json.load(handle))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[calibration] could not load {path}: {error}")
        return None


def save_profile(profile: CalibrationProfile) -> str:
    os.makedirs(PROFILE_DIR, exist_ok=True)
    path = profile_path(profile.name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(profile.to_json(), handle, indent=1)
    return path


def delete_profile(name: str) -> None:
    try:
        os.remove(profile_path(name))
    except OSError:
        pass
    if _ACTIVE is not None and _ACTIVE.name == name:
        set_active(None)


def activate_saved(name: Optional[str]) -> Optional[CalibrationProfile]:
    """Load ``name`` (a profile name or path) and make it the active one."""
    profile = load_profile(name) if name else None
    set_active(profile)
    return profile


# ── capture session ──────────────────────────────────────────────────────

class CalibrationSession:
    """Accumulates frames for each step and builds a profile.

    Feed every frame's face through :meth:`feed` while a step is armed; the
    step completes on its own once enough frames are in.  The dialog only
    has to arm steps and ask :meth:`build` at the end.
    """

    def __init__(self, frames_per_capture: int = CAPTURE_FRAMES) -> None:
        self.frames_per_capture = frames_per_capture
        self.captured: Dict[str, Dict[str, Any]] = {}
        self._armed: Optional[str] = None
        self._outlines: List[np.ndarray] = []
        self._kps: List[np.ndarray] = []
        self._poses: List[Tuple[float, float]] = []

    @property
    def armed(self) -> Optional[str]:
        return self._armed

    def progress(self) -> float:
        if self._armed is None:
            return 0.0
        return min(1.0, len(self._poses) / float(self.frames_per_capture))

    def arm(self, step: str) -> None:
        self._armed = step
        self._outlines = []
        self._kps = []
        self._poses = []

    def cancel(self) -> None:
        self._armed = None
        self._outlines = []
        self._kps = []
        self._poses = []

    def feed(self, face: Any) -> Optional[str]:
        """Add one frame's face; returns the step name when it just completed."""
        if self._armed is None or face is None:
            return None
        kps = getattr(face, "kps", None)
        if kps is None:
            return None
        self._poses.append(pose_proxies(kps))
        outline = normalised_landmarks(face)
        if outline is None:
            self._poses.pop()
            return None
        self._outlines.append(outline)
        self._kps.append(normalised_kps(face))
        if len(self._poses) < self.frames_per_capture:
            return None
        step = self._armed
        pose = np.median(np.asarray(self._poses, dtype=np.float32), axis=0)
        record: Dict[str, Any] = {
            "pose": (float(pose[0]), float(pose[1])),
            "outline": np.median(np.stack(self._outlines), axis=0).astype(np.float32),
            "kps": np.median(np.stack(self._kps), axis=0).astype(np.float32),
        }
        self.captured[step] = record
        self.cancel()
        return step

    def clear(self, step: str) -> None:
        self.captured.pop(step, None)

    def can_build(self) -> bool:
        return "neutral" in self.captured

    def build(self, name: str, fade_span: float = DEFAULT_FADE_SPAN) -> CalibrationProfile:
        if not self.can_build():
            raise ValueError("the neutral pose has not been captured")
        neutral = self.captured["neutral"]
        limits = {}
        for step in ("left", "right", "up", "down"):
            record = self.captured.get(step)
            if record is None:
                continue
            axis = 0 if step in ("left", "right") else 1
            limits[step] = record["pose"][axis]
        return CalibrationProfile(
            name=name,
            reference=neutral["outline"],
            reference_kps=neutral["kps"],
            neutral=neutral["pose"],
            limits=limits,
            fade_span=fade_span,
            created=time.strftime("%Y-%m-%d %H:%M"),
            references={step: dict(r) for step, r in self.captured.items()},
        )
