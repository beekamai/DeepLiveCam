# Face calibration

Per-person profiles that make the live swap hold its shape under fingers and
shadows and hand the real face back on deep turns.  Logic in
`modules/calibration.py`, Qt dialog in `modules/ui_calibration.py`, entry
point on the **Motion** tab (`Calibrate…`), CLI `--calibration <name|path>`.

## What a profile holds

`calibration/<name>.json`:

| field | meaning |
|---|---|
| `reference` | 106 landmarks of the neutral face in normalised `arcface_128` crop space (median of 12 frames) |
| `neutral` | `(yaw, pitch)` proxies looking straight at the camera |
| `limits` | proxy value at each captured extreme (`left`/`right`/`up`/`down`, any may be missing) |
| `fade_start` | fraction of the extreme where the swap starts fading (default 0.75) |

## Pose proxies — why not a pose model

`pose_proxies(kps)` uses only the five detector keypoints, which every frame
has even while the tracker carries the face between detections.  Both values
are measured along the eyes→mouth axis, so head roll cancels out:

* **yaw** — nose tip's sideways offset from the eye midpoint, in eye-mouth
  distances;
* **pitch** — nose tip's position between the eye line (0) and the mouth
  line (1).

They are monotonic in the real angles (verified on a synthetic 3D head,
±60° yaw / ±40° pitch) and only weakly coupled, which is all the fade needs:
the calibration records *this person's* values at the poses they chose, so
absolute angles never enter the picture.

## Fade to the real face

The captured extremes are where the swap should *still hold* — that is what
the prompt asks for — so `swap_alpha(face)` returns 1 all the way up to the
extreme and only then ramps (smoothstep) to 0 over `fade_span` of the range
beyond it (default +30 %; the dialog's **Fade beyond limit** slider).  The
first version faded from 75 % *up to* the extreme, which read as the swap
letting go before the calibrated point.  `swap_face` and `enhance_face_onnx`
multiply their paste mask by the alpha; at 0 they return the frame untouched
**without running the model**.  Without an active profile (or with
`pose_fade` off) alpha is always 1.  The tracker's hold-through-a-miss fade
(`face.track_alpha`, see tracking.md) multiplies in as well.

## Multi-pose references

Every captured pose — neutral and the four extremes — stores its outline,
keypoints and pose proxies (`references` in the JSON; `reference` /
`reference_kps` remain the neutral copies for older files).  At runtime
`profile.blended(kps)` weights the captured poses by their closeness to the
current pose (inverse squared distance in units of the calibrated range,
`REFERENCE_POSE_SCALE` floor) and blends outline and keypoints from them.
The reference outline and the expression-proof corners therefore come from
a head that was *already turned the same way*, instead of from the neutral
face stretched by an affine.  Profiles captured before this store only the
neutral pose and behave as before; recalibrating fills the rest.

## Reference outline

`refine_outline(points, frame_key)` runs inside `face_outline_mask` before
the temporal smoothing.  It fits a full affine from the reference to the live
landmarks with RANSAC (`REFERENCE_TOLERANCE` = 2.5 % of the crop): the
affine absorbs what the head did as a whole — turn, tilt, the foreshortening
of a yaw, and the difference between the swapper's and the enhancers'
alignment templates — so the outliers are the points something else moved.
Those take the reference's position under the same fit.

Guard: fewer than 53 inliers (half the set) and the consensus may be the
occluder rather than the face, so the points are left alone.  Measured on a
skin-coloured finger crossing the face: minimum IoU against the clean mask
0.77 → 0.86, worst landmark error 14 % → 7 % of the crop.  A whole hand that
hijacks the majority of landmarks *and* the detector keypoints is beyond it —
the alignment itself is wrong then, and that is the tracker's problem.

## Expression-proof crop (`stable_kps`)

The swap crop is aligned by a similarity fit of the five detector keypoints,
two of which are the mouth corners.  Lips pursed into a pout pull the
corners inward: the fit scales up 7-8 % and the crop covers 14 % less face,
so the real face shows around the edge; a smile does the opposite.  With a
profile, `stabilise(face)` keeps the detector's eyes and nose (they survive
a finger far better than the 106 landmarks do — a landmark-only affine fit
was 4 % off in scale under a finger, the detector 1 %) and replaces the
corners with the profile's neutral corners carried by the similarity those
three points define, slid vertically onto the live lip centre so a tilted
head keeps its foreshortening.  Synthetic pucker/smile: crop scale error
+7 % → 0 % horizontally; a finger crossing the face: 2.7 → 2.3 px keypoint
error.  The detector keypoints stay on `face.raw_kps` for the pose proxies.
Toggle: **Expression-proof crop** (`stable_alignment`).

## Real blinks and eyes (`modules/face_reveal.py`)

Not calibration, but the same family: the swap models never fully close an
eye and paint both eyes as a pair (no crossing them).  `eye_reveal_mask`
hands the eye region back to the camera — fully when the lid aperture drops
below 45 % of its running maximum (a blink), and at the **Real eyes**
slider's strength otherwise.  Aperture comes from the 106 landmarks, which
live mode computes every frame, so blink detection runs at camera rate
whatever the swap costs.

The **Mouth Mask** slider goes through the same module (`mouth_reveal_mask`)
instead of the old frame-space polygon of landmarks 52-63, which cut through
a moustache: the region is the lips' hull, widened with the slider and
reaching up to the base of the nose and down to the chin at 100 %, so what
is revealed is bounded by facial features.  The yellow box while dragging is
that region's outline.  A second mode, **Lips only**, never rises above the
upper lip's edge — a real moustache stays swapped when the source has none —
and grows only a little sideways and downward, where a stuck-out tongue
goes; its feather is half as wide so the blur does not climb back over the
moustache.  The combo next to the slider picks the mode
(`modules.globals.mouth_reveal_mode`).

## Capture flow

`CalibrationSession` is Qt-free: `arm(step)`, then `feed(face)` every frame
until it returns the completed step (12 frames with a face), `build(name)`.
The dialog owns the camera itself, so the live preview is closed before it
opens (one camera, one owner) and `Live` closes the dialog.  The profile is
saved, activated and remembered in `switch_states.json`; a `--calibration`
argument wins over the remembered profile.

The user looks away from the screen while turning, so the dialog also
announces itself: `modules/audio_cues.py` plays tones on every countdown
second, on arming ("hold still"), on a capture and at the end, and — when
*Voice prompts* is on — speaks the next pose through the system
synthesiser (SAPI via PowerShell on Windows, `say` on macOS, `spd-say` /
`espeak` on Linux).  Speech is English only because that is the one voice
every OS ships; a new utterance kills the one still playing.  Both options
persist in `switch_states.json`.  The on-frame cue (countdown digits, "hold
still") is drawn with a TrueType font through PIL: OpenCV's Hershey fonts
render anything non-ASCII as `?`, which the Russian UI hit.
