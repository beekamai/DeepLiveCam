"""Late reprojection: paste the last finished swap onto the newest camera frame.

With a heavy enhancer the swap of a frame is ready 100-200 ms after the
camera took it; by then the head has moved and the shown face lags behind
the body.  Instead of waiting, the display side keeps its own optical-flow
tracker running at camera rate on the raw frames, measures where the face
went since the processed frame was captured, and moves the processed face
region there.  Position is then at most one camera frame old; only the
expression inside the region keeps the processing latency — the trade VR
headsets make with time-warp.

The processing worker publishes a :class:`ProcessedFrame`; the capture side
keeps a short ring of raw frames so the tracker can be anchored on the frame
a result came from the first time round.  After that the tracker never
looks back: it remembers where it saw the face on every frame, so a result
for frame *n* is carried to the present by the motion measured since *n*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from modules.face_tracker import FaceTracker

# Raw frames kept for the first anchor: processing latency in camera frames,
# plus a margin.  16 frames of 1280x720 BGR is ~44 MB.
RING_SIZE = 16
# Tracked positions remembered per frame; a result older than this cannot be
# carried forward and is shown as it is.
HISTORY_SIZE = 90
# A result whose frame the display tracker never saw is placed by
# interpolating its neighbours, up to this many frames apart.
HISTORY_GAP = 8
# Share of the remaining detection correction applied per shown frame: the
# worker's detection pulls the tracked position in over a few frames instead
# of snapping it.
CORRECTION_RATE = 0.5
# When the tracker must catch up over many frames (first anchor), step
# through the ring every N frames instead of one giant flow jump.
CATCH_UP_STRIDE = 2
# Below this keypoint displacement (px) the processed frame is shown as it
# is: moving the region for a pixel or two only adds a seam the processed
# frame does not have.
MIN_SHIFT = 2.5
# The moved region's edge is feathered in proportion to how far it moved —
# the further from where it was rendered, the less its border matches the
# frame underneath.  Blur radius = displacement * this, in px.
EDGE_SOFTEN = 0.4
EDGE_SOFTEN_MAX = 12


@dataclass
class ProcessedFrame:
    seq: int                                  # capture sequence number
    frame: np.ndarray                         # fully processed frame
    kps: Optional[np.ndarray] = None          # alignment keypoints used, frame coords
    bbox: Optional[np.ndarray] = None
    box: Optional[Tuple[int, int, int, int]] = None   # x0, y0, x1, y1 of the pasted region
    alpha: Optional[np.ndarray] = None        # float32 paste alpha inside ``box``
    fps: float = 0.0                          # processing rate, for the overlay
    detected: bool = False                    # keypoints come from the detector, not from flow


class _Face:
    """The minimal face the tracker needs."""

    def __init__(self, kps: np.ndarray, bbox: np.ndarray) -> None:
        self.kps = kps
        self.bbox = bbox
        self.landmark_2d_106 = None


def union_paste_alpha(shape: Tuple[int, int], pastes) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """Union of crop-space paste alphas in frame space, ROI-bounded.

    ``pastes`` yields ``(alpha_crop, affine)`` pairs: the alpha in aligned
    crop space (float 0..1 or uint8) and the frame->crop affine.  Returns the
    bounding box and the alpha inside it, or ``None`` when nothing was pasted.
    """
    h, w = shape[:2]
    boxes = []
    items = []
    for alpha, affine in pastes:
        if alpha is None or affine is None:
            continue
        size = alpha.shape[0]
        inverse = cv2.invertAffineTransform(affine)
        corners = cv2.transform(
            np.array([[[0, 0], [size, 0], [size, size], [0, size]]], dtype=np.float32), inverse,
        ).reshape(-1, 2)
        x0 = max(0, int(np.floor(corners[:, 0].min())))
        y0 = max(0, int(np.floor(corners[:, 1].min())))
        x1 = min(w, int(np.ceil(corners[:, 0].max())) + 1)
        y1 = min(h, int(np.ceil(corners[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append((x0, y0, x1, y1))
        items.append((alpha, inverse))
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    union = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    for alpha, inverse in items:
        if alpha.dtype != np.float32:
            alpha = alpha.astype(np.float32) / 255.0
        shifted = inverse.copy()
        shifted[0, 2] -= x0
        shifted[1, 2] -= y0
        warped = cv2.warpAffine(alpha, shifted, (x1 - x0, y1 - y0), borderValue=0)
        np.maximum(union, warped, out=union)
    return (x0, y0, x1, y1), union


def collect_paste_alpha(shape: Tuple[int, int]):
    """Union of the swap's and the enhancer's last pastes, then forget them
    so a frame without a swap does not inherit the previous one's region."""
    from modules.processors.frame.face_swapper import LAST_PASTE
    from modules.processors.frame._onnx_enhancer import LAST_ENHANCE

    pastes = [(LAST_PASTE["alpha"], LAST_PASTE["affine"]),
              (LAST_ENHANCE["alpha"], LAST_ENHANCE["affine"])]
    LAST_PASTE["alpha"] = LAST_PASTE["affine"] = None
    LAST_ENHANCE["alpha"] = LAST_ENHANCE["affine"] = None
    return union_paste_alpha(shape, pastes)


class Reprojector:
    """Moves the newest processed face onto the newest raw frame.

    The tracker and its history are pure optical flow; the worker's
    detections are folded in as a separate correction vector that eases
    toward its target.  Keeping the two apart matters: a correction that is
    still being applied must not leak into the motion the next correction
    is measured from, or the corrections feed on each other.
    """

    def __init__(self) -> None:
        self._tracker = FaceTracker(reset_outline=False)
        self._face: Optional[_Face] = None
        self._seq: Optional[int] = None            # frame the tracker last saw
        self._history: Dict[int, np.ndarray] = {}  # seq -> flow-tracked kps
        self._applied_seq: Optional[int] = None    # record already folded in
        self._target = np.zeros((5, 2), dtype=np.float32)    # correction wanted
        self._applied = np.zeros((5, 2), dtype=np.float32)   # correction in effect

    def reset(self) -> None:
        self._tracker.reset()
        self._face = None
        self._seq = None
        self._history.clear()
        self._applied_seq = None
        self._target[:] = 0
        self._applied[:] = 0

    def compose(self, record: ProcessedFrame, ring: Dict[int, np.ndarray],
                latest_seq: int) -> np.ndarray:
        """The frame to show: ``record.frame`` itself when it is current or
        cannot be moved, otherwise the newest raw frame with the processed
        face region warped to where the face is now.  The raw frame is drawn
        on in place."""
        if record.kps is None or record.box is None or record.alpha is None:
            return record.frame
        if latest_seq <= record.seq:
            return record.frame
        latest = ring.get(latest_seq)
        if latest is None:
            return record.frame

        if (self._face is None or self._seq is None
                or self._history_at(record.seq) is None):
            if not self._anchor(record, ring):
                return record.frame
        if latest_seq > self._seq:
            self._advance(ring, latest_seq)
        raw_now = np.asarray(self._face.kps, dtype=np.float32)
        if record.seq != self._applied_seq and record.detected:
            # Where the worker's detection would be now, by the flow measured
            # since its frame — the difference to raw flow is the correction.
            # Only real detections qualify: the worker's own tracked positions
            # lag behind the face whenever it runs slower than the camera.
            past = self._history_at(record.seq)
            if past is not None:
                motion = cv2.estimateAffinePartial2D(past, raw_now, method=cv2.LMEDS)[0]
                if motion is not None:
                    carried = cv2.transform(
                        np.asarray(record.kps, dtype=np.float32).reshape(1, -1, 2), motion,
                    ).reshape(-1, 2).astype(np.float32)
                    self._target = carried - raw_now
            self._applied_seq = record.seq
        self._applied += (self._target - self._applied) * CORRECTION_RATE
        shown_kps = raw_now + self._applied
        shift = float(np.linalg.norm(shown_kps - np.asarray(record.kps, dtype=np.float32), axis=1).max())
        if shift < MIN_SHIFT:
            return record.frame
        transform = cv2.estimateAffinePartial2D(
            np.asarray(record.kps, dtype=np.float32), shown_kps, method=cv2.LMEDS,
        )[0]
        if transform is None:
            return record.frame
        return self._paste(record, latest, transform, shift)

    def _advance(self, ring: Dict[int, np.ndarray], latest_seq: int) -> None:
        """Track to the newest frame — through the ring in strides when the
        gap is large, so no single flow step has to bridge it."""
        seqs = [s for s in range(self._seq + CATCH_UP_STRIDE, latest_seq, CATCH_UP_STRIDE) if s in ring]
        for seq in seqs + [latest_seq]:
            self._face = self._tracker.track(ring[seq], self._face)
            self._seq = seq
            self._remember(seq, self._face.kps)

    def _history_at(self, seq: int) -> Optional[np.ndarray]:
        """Flow-tracked keypoints on ``seq``, interpolated between the
        nearest frames the tracker actually saw."""
        exact = self._history.get(seq)
        if exact is not None:
            return exact
        before = [k for k in self._history if k < seq]
        after = [k for k in self._history if k > seq]
        if before and after:
            lo, hi = max(before), min(after)
            if hi - lo <= HISTORY_GAP:
                t = (seq - lo) / float(hi - lo)
                return (1.0 - t) * self._history[lo] + t * self._history[hi]
        elif before and seq - max(before) <= 1:
            return self._history[max(before)]
        return None

    def _anchor(self, record: ProcessedFrame, ring: Dict[int, np.ndarray]) -> bool:
        """Start the tracker from the record's frame — or, when the result is
        older than the ring, from the oldest frame kept, where the face is
        near enough for the next detection to correct it."""
        seq = record.seq
        anchor = ring.get(seq)
        if anchor is None:
            later = sorted(k for k in ring if k > seq)
            if not later:
                return False
            seq = later[0]
            anchor = ring[seq]
        self._tracker.reset()
        self._history.clear()
        self._target[:] = 0
        self._applied[:] = 0
        bbox = record.bbox if record.bbox is not None else self._bbox_from_kps(record.kps)
        self._face = self._tracker.observe(anchor, _Face(record.kps.copy(), bbox.copy()))
        self._seq = seq
        self._applied_seq = record.seq
        self._remember(record.seq, self._face.kps)
        if seq != record.seq:
            self._remember(seq, self._face.kps)
        return True

    def _remember(self, seq: int, kps: np.ndarray) -> None:
        self._history[seq] = np.asarray(kps, dtype=np.float32).copy()
        if len(self._history) > HISTORY_SIZE:
            for old in [k for k in self._history if k <= seq - HISTORY_SIZE]:
                self._history.pop(old, None)

    @staticmethod
    def _bbox_from_kps(kps: np.ndarray) -> np.ndarray:
        pts = np.asarray(kps, dtype=np.float32)
        centre = pts.mean(axis=0)
        half = float(np.linalg.norm(pts[1] - pts[0])) * 1.6
        return np.array([centre[0] - half, centre[1] - half * 1.2,
                         centre[0] + half, centre[1] + half * 1.4], dtype=np.float32)

    @staticmethod
    def _paste(record: ProcessedFrame, target: np.ndarray, transform: np.ndarray,
               shift: float = 0.0) -> np.ndarray:
        x0, y0, x1, y1 = record.box
        h, w = target.shape[:2]
        corners = cv2.transform(
            np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=np.float32), transform,
        ).reshape(-1, 2)
        dx0 = max(0, int(np.floor(corners[:, 0].min())))
        dy0 = max(0, int(np.floor(corners[:, 1].min())))
        dx1 = min(w, int(np.ceil(corners[:, 0].max())) + 1)
        dy1 = min(h, int(np.ceil(corners[:, 1].max())) + 1)
        if dx1 <= dx0 or dy1 <= dy0:
            return record.frame
        # source ROI -> destination ROI: shift into ROI coordinates on both ends
        local = transform.copy()
        local[:, 2] += transform[:, :2] @ np.array([x0, y0], dtype=np.float64)
        local[0, 2] -= dx0
        local[1, 2] -= dy0
        size = (dx1 - dx0, dy1 - dy0)
        moved = cv2.warpAffine(record.frame[y0:y1, x0:x1], local, size,
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpAffine(record.alpha, local, size, flags=cv2.INTER_LINEAR, borderValue=0)
        soften = int(min(EDGE_SOFTEN_MAX, shift * EDGE_SOFTEN))
        if soften >= 2:
            # Feather the edge: the region keeps its face, but its border,
            # rendered against an older frame, blends away.
            k = soften * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        roi = target[dy0:dy1, dx0:dx1]
        roi[:] = cv2.blendLinear(moved, roi, alpha, 1.0 - alpha)
        return target
