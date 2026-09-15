"""Per-frame face tracking and landmark smoothing for the live pipeline.

Detection costs ~20 ms at 720p, so the webcam loop only runs it every few
frames and reuses the last face in between — which makes the swapped face lag
behind and slide during motion.  This module fills those frames with sparse
optical flow on the five keypoints (~1-2 ms) and runs the result through a
One Euro filter, which smooths jitter while standing still but barely lags on
fast movement.
"""

import math
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np


# Optical flow runs on a downscaled grey frame — keypoint motion is large
# compared to the detail lost, and the cost drops quadratically.
FLOW_SCALE = 0.5
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)
# Beyond this per-frame keypoint jump the flow is treated as lost (fast pan,
# cut, someone walking through frame) and the stale face is dropped.
MAX_FLOW_JUMP = 120.0
# Box size of the high-pass applied to the flow input, in downscaled pixels.
HIGHPASS_KERNEL = (9, 9)
# Flow region padding around the face, in inter-eye distances.
ROI_MARGIN = 1.6
# Forward-backward round-trip error, in pixels, above which a point is dropped.
# Tuned against real compressed footage with the high-pass flow input: looser
# values let compression noise through and doubled the worst-frame error.
FB_ERROR_LIMIT = 1.5
# Reprojection error, in pixels, above which RANSAC treats a tracked point as
# not belonging to the face's own motion.
RANSAC_THRESHOLD = 3.0
# Largest plausible change of face scale between two consecutive frames.
MAX_SCALE_STEP = 0.3
# How much of the last velocity survives each coasted frame.
COAST_DECAY = 0.8
# Above this keypoint speed (px/frame), or this gap between the flow's
# prediction and the next detection (px), the face is moving faster than
# the flow follows reliably: the caller detects every frame until it calms.
FAST_SPEED = 8.0
FAST_DISAGREEMENT = 8.0
# Frames of current velocity to pad the flow region with, ahead of the face.
VELOCITY_LEAD = 3.0
# Extra flow points spread over the head (hairline, brows, jaw, ears): a
# similarity fitted through forty points survives a turned cheek or a finger
# over the nose where five keypoints alone would bend with it.
SUPPORT_POINTS = 40
SUPPORT_MIN = 8
SUPPORT_QUALITY = 0.01
# How long the face is carried by flow after the detector stops seeing it
# (a deep turn), fading the swap out over that time instead of cutting.
MISS_HOLD_SECONDS = 1.0


class OneEuroFilter:
    """Adaptive low-pass filter: heavy smoothing when slow, light when fast."""

    # Tuned on a synthetic pan plus a static noisy frame: this pair tracks a
    # 10 px/frame pan with ~2 px lag while cutting standing-still wobble by a
    # third. Lower beta smooths more but drags visibly behind fast motion.
    # A point at rest still wanders a fraction of a pixel after filtering,
    # and the three-point alignment turns that into a breathing crop. The
    # output trails the filtered point on a leash of this length: wander
    # inside it moves nothing, motion beyond it is followed continuously,
    # so the error never exceeds the leash and slow motion never snaps.
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.4,
                 d_cutoff: float = 1.0, deadband: float = 0.6) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.deadband = deadband
        self._prev: Optional[np.ndarray] = None
        self._prev_dx: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None
        self._held: Optional[np.ndarray] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._prev = None
        self._prev_dx = None
        self._prev_time = None
        self._held = None

    def __call__(self, value: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        now = time.perf_counter() if timestamp is None else timestamp

        if self._prev is None:
            self._prev = value
            self._prev_dx = np.zeros_like(value)
            self._prev_time = now
            self._held = value
            return value

        dt = now - self._prev_time
        if dt <= 0 or dt > 0.5:
            # First frame after a stall — restart rather than filter across it.
            self._prev = value
            self._prev_dx = np.zeros_like(value)
            self._prev_time = now
            self._held = value
            return value

        dx = (value - self._prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._prev_dx

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = np.array([self._alpha(float(c), dt) for c in cutoff.ravel()],
                         dtype=np.float32).reshape(value.shape)
        smoothed = alpha * value + (1 - alpha) * self._prev

        self._prev = smoothed
        self._prev_dx = dx_hat
        self._prev_time = now
        return self._hold(smoothed)

    def _hold(self, smoothed: np.ndarray) -> np.ndarray:
        if self.deadband <= 0 or self._held is None or self._held.shape != smoothed.shape:
            self._held = smoothed
            return smoothed
        delta = smoothed - self._held
        if smoothed.ndim == 2 and smoothed.shape[-1] == 2:
            distance = np.linalg.norm(delta, axis=-1, keepdims=True)
        else:
            distance = np.abs(delta)
        excess = np.clip(1.0 - self.deadband / np.maximum(distance, 1e-6), 0.0, 1.0)
        self._held = (self._held + delta * excess).astype(np.float32)
        return self._held


class FaceTracker:
    """Keeps one face locked to the frame between detections."""

    def __init__(self, reset_outline: bool = True) -> None:
        # The live loop's tracker restarts the outline smoothing on a fresh
        # lock; a second tracker (reprojection) must not touch that state.
        self._reset_outline = reset_outline
        self._prev_gray: Optional[np.ndarray] = None
        self._region: Optional[Tuple[int, int, int, int]] = None
        self._velocity = np.zeros(2, dtype=np.float32)   # last accepted shift/frame
        self._wide_gray: Optional[np.ndarray] = None
        self._wide_box: Optional[Tuple[int, int, int, int]] = None
        self._points: Optional[np.ndarray] = None   # kps in full-frame coords
        self._support: Optional[np.ndarray] = None  # extra flow points, frame coords
        self._bbox: Optional[np.ndarray] = None
        self.speed = 0.0          # last accepted keypoint shift, px/frame
        self.disagreement = 0.0   # flow prediction vs last detection, px
        self._filter = OneEuroFilter()
        self._face: Any = None                      # last face object handed out
        self._miss_since: Optional[float] = None    # when the detector last lost the face

    @property
    def unsettled(self) -> bool:
        """The face moves faster than the flow follows: detect every frame."""
        return self.speed > FAST_SPEED or self.disagreement > FAST_DISAGREEMENT

    def reset(self) -> None:
        self.speed = 0.0
        self.disagreement = 0.0
        self._prev_gray = None
        self._region = None
        self._velocity = np.zeros(2, dtype=np.float32)
        self._wide_gray = None
        self._wide_box = None
        self._points = None
        self._support = None
        self._bbox = None
        self._filter.reset()
        self._face = None
        self._miss_since = None

    def _plan_region(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Padded box around the current face — the only part flow looks at."""
        if self._points is None:
            return 0, 0, frame.shape[1], frame.shape[0]
        margin = max(40.0, self._face_scale() * ROI_MARGIN)
        # A fast pan needs room ahead of the face, or it leaves the region
        # every couple of frames; pad by a few frames of the current velocity.
        lead = np.abs(self._velocity) * VELOCITY_LEAD
        x0 = int(max(0, self._points[:, 0].min() - margin - lead[0]))
        y0 = int(max(0, self._points[:, 1].min() - margin - lead[1]))
        x1 = int(min(frame.shape[1], self._points[:, 0].max() + margin + lead[0]))
        y1 = int(min(frame.shape[0], self._points[:, 1].max() + margin + lead[1]))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return 0, 0, frame.shape[1], frame.shape[0]
        return x0, y0, x1, y1

    def _grey(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """Flow input for one region of the frame.

        A box high-pass removes the low-frequency component, so a shadow
        sweeping across the face barely changes what the tracker sees, while
        edges and skin texture — what the flow actually locks onto — survive.
        """
        x0, y0, x1, y1 = region
        small = cv2.resize(frame[y0:y1, x0:x1], None, fx=FLOW_SCALE, fy=FLOW_SCALE,
                           interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.addWeighted(grey, 1.0, cv2.blur(grey, HIGHPASS_KERNEL), -1.0, 128)

    def _wide_region(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Generous box around the face, kept from the previous frame so a
        re-pinned flow region can be cut out of it without losing a frame."""
        if self._points is None:
            return 0, 0, frame.shape[1], frame.shape[0]
        margin = max(80.0, self._face_scale() * ROI_MARGIN * 2.5) + float(np.abs(self._velocity).max()) * VELOCITY_LEAD
        x0 = int(max(0, self._points[:, 0].min() - margin))
        y0 = int(max(0, self._points[:, 1].min() - margin))
        x1 = int(min(frame.shape[1], self._points[:, 0].max() + margin))
        y1 = int(min(frame.shape[0], self._points[:, 1].max() + margin))
        return x0, y0, x1, y1

    def _cut_from_wide(self, region: Tuple[int, int, int, int],
                       shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Previous-frame flow input for ``region``, from the stored wide grey.

        ``shape`` is the exact (rows, cols) of this frame's flow input; the cut
        is resampled to it, since scaling rounds the two slightly differently
        and LK insists on identical pyramids.
        """
        if self._wide_gray is None or self._wide_box is None:
            return None
        wx0, wy0, wx1, wy1 = self._wide_box
        x0, y0, x1, y1 = region
        if x0 < wx0 or y0 < wy0 or x1 > wx1 or y1 > wy1:
            return None
        sx0, sy0 = int(round((x0 - wx0) * FLOW_SCALE)), int(round((y0 - wy0) * FLOW_SCALE))
        sx1, sy1 = sx0 + shape[1], sy0 + shape[0]
        cut = self._wide_gray[sy0:sy1, sx0:sx1]
        if cut.size == 0:
            return None
        if cut.shape != tuple(shape):
            cut = cv2.resize(cut, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        return cut

    def _remember_wide(self, frame: np.ndarray) -> None:
        self._wide_box = self._wide_region(frame)
        self._wide_gray = self._grey(frame, self._wide_box)

    def _inside_region(self) -> bool:
        """Is the face still comfortably inside the pinned flow region?"""
        if self._region is None or self._points is None:
            return False
        x0, y0, x1, y1 = self._region
        edge = max(8.0, self._face_scale() * 0.3)
        return bool(
            self._points[:, 0].min() > x0 + edge and
            self._points[:, 1].min() > y0 + edge and
            self._points[:, 0].max() < x1 - edge and
            self._points[:, 1].max() < y1 - edge
        )

    def _face_scale(self) -> float:
        """Inter-eye distance — the natural unit for face-sized thresholds."""
        if self._points is None:
            return 1.0
        return max(1.0, float(np.linalg.norm(self._points[1] - self._points[0])))

    def observe(self, frame: np.ndarray, face: Any,
                timestamp: Optional[float] = None) -> Any:
        """Feed a detection result; returns the face with smoothed keypoints.

        ``None`` means the detector saw nothing this time.  A face that is
        merely turned too far for the detector is still there, so the last
        one is carried on by flow for ``MISS_HOLD_SECONDS`` — its
        ``track_alpha`` fading to zero — before the lock is dropped.
        """
        if face is None:
            if self._points is None or self._face is None:
                self.reset()
                return None
            now = time.time() if timestamp is None else timestamp
            if self._miss_since is None:
                self._miss_since = now
            if now - self._miss_since >= MISS_HOLD_SECONDS:
                self.reset()
                return None
            return self.track(frame, self._face, timestamp)
        self._miss_since = None
        if self._points is not None:
            detected = np.asarray(face.kps, dtype=np.float32)
            if detected.shape == self._points.shape:
                self.disagreement = float(np.linalg.norm(detected - self._points, axis=1).mean())
        else:
            self.disagreement = 0.0
        if self._points is None and self._reset_outline:
            from modules.processors.frame._onnx_enhancer import reset_face_outline

            # Fresh lock on a face: start the outline smoothing from scratch.
            reset_face_outline()
        self._points = np.asarray(face.kps, dtype=np.float32).copy()
        self._region = self._plan_region(frame)
        self._prev_gray = self._grey(frame, self._region)
        self._support = self._find_support(self._prev_gray, self._region)
        self._remember_wide(frame)
        self._bbox = np.asarray(face.bbox, dtype=np.float32).copy()
        return self._apply(face, self._points, self._bbox, timestamp)

    def _find_support(self, gray: np.ndarray, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Corners over the head area around the keypoints, in frame coords."""
        if self._points is None:
            return None
        origin = np.array(region[:2], dtype=np.float32)
        local = (self._points - origin) * FLOW_SCALE
        centre = local.mean(axis=0)
        radius = max(8.0, float(np.linalg.norm(local[1] - local[0])) * 1.6)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.ellipse(mask, (int(centre[0]), int(centre[1])),
                    (int(radius), int(radius * 1.3)), 0, 0, 360, 255, -1)
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=SUPPORT_POINTS, qualityLevel=SUPPORT_QUALITY,
            minDistance=max(3, int(radius / 6)), mask=mask,
        )
        if corners is None or len(corners) < SUPPORT_MIN:
            return None
        return corners.reshape(-1, 2) / FLOW_SCALE + origin

    def track(self, frame: np.ndarray, face: Any,
              timestamp: Optional[float] = None) -> Any:
        """Advance the last detection to this frame with optical flow."""
        if face is None or self._prev_gray is None or self._points is None:
            return face

        if not self._inside_region():
            # The face reached the edge of the pinned region: re-pin around
            # it and take the previous frame's view of the new region from the
            # stored wide grey, so this frame's flow still happens.
            self._region = self._plan_region(frame)
            gray = self._grey(frame, self._region)
            previous = self._cut_from_wide(self._region, gray.shape[:2])
            if previous is None:
                return self._coast(frame, face, timestamp)
            self._prev_gray = previous
        else:
            gray = self._grey(frame, self._region)

        origin = np.array(self._region[:2], dtype=np.float32)
        n_kps = len(self._points)
        source = self._points if self._support is None else np.vstack([self._points, self._support])
        prev_pts = ((source - origin) * FLOW_SCALE).reshape(-1, 1, 2).astype(np.float32)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_pts, None, **LK_PARAMS,
        )

        if next_pts is None or status is None or int(status.sum()) < 3:
            return self._coast(frame, face, timestamp)

        tracked = next_pts.reshape(-1, 2) / FLOW_SCALE + origin
        valid = status.ravel() == 1

        # Forward-backward check: track the new points back and keep only those
        # that land where they started.  A hand sliding across the face drags
        # flow with it, and that asymmetry is exactly what this catches.
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, next_pts, None, **LK_PARAMS,
        )
        if back_pts is not None and back_status is not None:
            round_trip = np.linalg.norm(
                back_pts.reshape(-1, 2) / FLOW_SCALE + origin - source, axis=1,
            )
            valid &= (back_status.ravel() == 1) & (round_trip < FB_ERROR_LIMIT)

        if int(valid.sum()) < 2:
            return self._coast(frame, face, timestamp)

        # A face moves rigidly, so fit one similarity transform (rotation,
        # scale, translation) to the tracked points with RANSAC and apply it to
        # all five.  Points a finger or a mic drags away disagree with that
        # motion and are discarded as outliers, instead of bending the
        # alignment towards the occluder.
        transform, inliers = cv2.estimateAffinePartial2D(
            source[valid], tracked[valid],
            method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD,
            maxIters=200, confidence=0.99,
        )
        if transform is None:
            return self._coast(frame, face, timestamp)

        points = cv2.transform(
            self._points.reshape(1, -1, 2), transform,
        ).reshape(-1, 2).astype(np.float32)

        shift = points - self._points
        centre_shift = shift.mean(axis=0)
        scale_change = abs(float(np.sqrt(abs(np.linalg.det(transform[:, :2])))) - 1.0)
        if float(np.abs(shift).max()) > MAX_FLOW_JUMP or scale_change > MAX_SCALE_STEP:
            # A face does not grow by a third or leap across the frame in one
            # frame — the flow latched onto something else. Coast on the last
            # good velocity until detection re-anchors.
            return self._coast(frame, face, timestamp)

        self._velocity = centre_shift.astype(np.float32)
        self.speed = float(np.linalg.norm(shift, axis=1).max())
        self._points = points
        if self._support is not None:
            # Support points keep their measured positions (they follow the
            # texture); the ones the flow lost are dropped, and the set is
            # regrown from this frame when it runs thin.
            keep = valid[n_kps:]
            if inliers is not None:
                agreed = np.zeros(len(valid), dtype=bool)
                agreed[np.flatnonzero(valid)] = inliers.ravel() == 1
                keep = keep & agreed[n_kps:]
            self._support = tracked[n_kps:][keep]
            if len(self._support) < SUPPORT_MIN:
                self._support = self._find_support(gray, self._region)
        self._prev_gray = gray
        self._remember_wide(frame)
        self._bbox = self._bbox + np.concatenate([centre_shift, centre_shift])
        return self._apply(face, points, self._bbox, timestamp, carry=transform)

    def _coast(self, frame: np.ndarray, face: Any,
               timestamp: Optional[float]) -> Any:
        """Advance the face by its last velocity and re-pin the flow region.

        Used whenever this frame's flow cannot be trusted: the face keeps
        moving predictably instead of stalling, and the next frame's flow
        starts from where the face most likely is.
        """
        if self._points is None:
            return face
        self._points = self._points + self._velocity
        if self._support is not None:
            self._support = self._support + self._velocity
        self._bbox = self._bbox + np.concatenate([self._velocity, self._velocity])
        carry = np.array([[1.0, 0.0, self._velocity[0]], [0.0, 1.0, self._velocity[1]]], dtype=np.float32)
        self._velocity = self._velocity * COAST_DECAY
        self.speed = float(np.linalg.norm(self._velocity))
        self._region = self._plan_region(frame)
        self._prev_gray = self._grey(frame, self._region)
        self._remember_wide(frame)
        return self._apply(face, self._points, self._bbox, timestamp, carry=carry)

    def _apply(self, face: Any, points: np.ndarray, bbox: np.ndarray,
               timestamp: Optional[float] = None,
               carry: Optional[np.ndarray] = None) -> Any:
        """Write smoothed geometry onto the face object.

        ``carry`` is the similarity the flow found for this frame: the head
        pose and its 3D silhouette are rigid with the face, so they move by
        it instead of being re-estimated every frame (the model runs again
        on the next detection).
        """
        smoothed = self._filter(points, timestamp)
        face.kps = smoothed.astype(np.float32)
        face.bbox = np.asarray(bbox, dtype=np.float32)
        # Fresh keypoints: the calibrated alignment recomputes from them.
        face.raw_kps = None
        # The 106-point landmarks belong to the detection frame; drop them so
        # consumers recompute instead of using stale positions.
        if getattr(face, "landmark_2d_106", None) is not None:
            face.landmark_2d_106 = None
        head = getattr(face, "head_pose", None)
        if head is not None:
            face.head_pose = head.moved(carry) if carry is not None and hasattr(head, "moved") else None
        if self._miss_since is None:
            face.track_alpha = 1.0
        else:
            now = time.time() if timestamp is None else timestamp
            face.track_alpha = float(max(0.0, 1.0 - (now - self._miss_since) / MISS_HOLD_SECONDS))
        self._face = face
        return face
