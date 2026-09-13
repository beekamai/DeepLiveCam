# Low-latency reprojection

`modules/reprojection.py`, wired into the live preview in `modules/ui.py`;
toggle **Low-latency reprojection** on the Motion tab (`reprojection` in
`switch_states.json`).

## Problem

With a heavy enhancer (GFPGAN-1.4 at 512 px is ~90 ms of model time on the
RTX 5060 Ti) the swap of a camera frame is ready 100-200 ms after it was
taken.  Showing that frame means showing the face where it *was*; the eye
reads it as the mask lagging behind the head on every fast move.

Carrying the enhancer's output across frames does not work: the residual
detail from frame *n* pasted onto frame *n+1* scores 24.9 dB against the
true enhancement — worse than not enhancing at all (26.2 dB) — because the
expression differs between frames inside the aligned crop.  So the content
cannot be predicted; the *position* can.

## What it does

* `_CaptureWorker` numbers every frame and keeps the last `RING_SIZE` raw
  frames (copies — the swap writes in place).
* `_ProcessingWorker` publishes a `ProcessedFrame`: the processed frame, the
  keypoints the swap was aligned with, and the union of the paste alphas of
  the swap and the enhancer in frame space (`collect_paste_alpha`).
* On every UI tick `Reprojector.compose` runs its own `FaceTracker` one
  optical-flow step to the newest raw frame (striding through the ring when
  the gap is large), fits the similarity from the record's keypoints to the
  shown position, and warps the pasted region there with its alpha
  (`cv2.blendLinear`).  The tracker is anchored once on the frame a result
  came from (or the oldest frame in the ring when the result is older), then
  never looks back: it remembers its flow-tracked keypoints per frame
  (interpolated for frames it skipped).
* Detections correct flow drift, but only real ones: a record flagged
  `detected` is carried to the present through the flow measured since its
  frame, and the difference to the raw flow position becomes a correction
  *target* that the applied correction eases toward (`CORRECTION_RATE` per
  shown frame).  The tracker and its history stay pure flow — feeding a
  half-applied correction back into the motion the next one is measured from
  made the corrections feed on each other (±120 px oscillation, measured).
  The worker's own flow-tracked positions are not used as corrections: at
  low worker fps its tracker bridges 8+ camera frames and was 20-40 px off.
* The worker itself now detects whenever it skipped 3+ camera frames or
  0.25 s passed — detection is 6 ms, a mis-tracked swap is a whole frame
  wasted.

Cost: ~5-10 ms per shown frame on the UI thread.  Headless on a
sinusoidally panning camera with the worker at 5-7 fps and a 60 fps
camera: ~42-50 fps shown, frame-to-frame step std 2.1 px (the pan itself
moves up to 7 px/frame), no steps above 14 px; offline sim with 8-frame
latency: 3.2 px mean error vs truth, max 7.3.

## Seams

Moving the region for a pixel or two only adds a border the processed frame
does not have, so below `MIN_SHIFT` (2.5 px) the processed frame is shown
as it is.  Above it the moved region's alpha is Gaussian-feathered in
proportion to how far it moved (`EDGE_SOFTEN` × shift, capped at 12 px):
the further from where it was rendered, the less its border matches the
frame underneath, and the more it needs to fade into it.

## What it does not do

* The expression inside the region is still as old as the processing
  latency; only position and scale are current.
* Single-face mode only (`many_faces` / `map_faces` records carry no
  keypoints and are shown as they are).
* The FPS overlay shows the *processing* rate, drawn on the display side.
