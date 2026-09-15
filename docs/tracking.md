# Face tracking in live mode

`modules/face_tracker.py`, driven from the processing worker in
`modules/ui.py`.  The detector is the expensive stage and its keypoints
jitter; the tracker carries the face between detections with optical flow
and smooths what it hands out.

## Detection cadence

Wall-clock based: every 0.25 s while tracking (0.08 s without), **plus**
whenever the worker skipped three or more camera frames — flow across a
gap that wide was 20–40 px off, and a detection costs 6 ms against a
whole mis-aligned swap.  Any UI toggle that changes the picture (mirror,
masks, models — `modules.globals.settings_epoch`) forces detection on the
next three frames and resets the tracker and the reprojector, so the swap
does not drop out for a cycle while the old state catches up.

## Flow

* Region of interest around the face, padded by a few frames of velocity;
  high-pass grey at half resolution so a sweeping shadow barely registers.
* Points: the five keypoints **plus up to forty support corners**
  (`goodFeaturesToTrack` inside an ellipse over the head — hairline, brows,
  jaw, ears).  Forward-backward check drops points the flow cannot bring
  back; RANSAC fits one similarity through what is left and moves the
  keypoints by it.  Support points keep their measured positions and are
  regrown when fewer than eight survive.  Measured: a skin-coloured hand
  crossing the face drags the keypoints 5.9 px mean / 12 px max (was 13 /
  34 with the five keypoints alone); 10–20 px/frame pans track within 2 px.
* One Euro filter on the output, followed by a 0.6 px leash
  (`OneEuroFilter.deadband`): the handed-out point trails the filtered one
  and does not move while it wanders inside the leash, so a still face
  stops breathing (measured live: keypoint wander 0.39 → 0.25 px, mean
  step per frame 0.14 → 0.05 px; fast motion unchanged, lag bounded by
  the leash).  Scale and jump limits send the tracker coasting on its last
  velocity when the flow latched onto something else.

## Hold through a missed detection

When the detector returns nothing — a deep turn, not an empty frame — the
face is still there.  `observe(frame, None)` keeps tracking it for
`MISS_HOLD_SECONDS` (1 s) with `face.track_alpha` falling from 1 to 0,
which `calibration.swap_alpha` folds into the paste alpha: the swap fades
out where the head went instead of cutting to the bare face, and comes
back the moment the detector sees the face again.  After the hold the lock
is dropped.  Measured: a face panning 5 px/frame carried 31 frames with
≤1.5 px error while fading.

## Keypoints handed out

`face.kps` — smoothed, tracked.  `face.raw_kps` — cleared every frame so
`calibration.stabilise` recomputes the expression-proof corners from fresh
keypoints (a stale stabilised set would freeze the mouth height).
`face.landmark_2d_106` is dropped each frame and recomputed on demand.
