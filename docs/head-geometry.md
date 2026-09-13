# Head geometry (`modules/head_geometry.py`)

The five detector keypoints give only pose *proxies* (nose offset in
eye-mouth units) and the 106 landmarks are 2D: on a turned head the far-side
jaw points slide inward or collapse, so a mask hull built from them shrinks
exactly where the swap needs cover.  A 3D fit gives real angles and a
contour that stays a silhouette.

## Provider interface

`HeadPose(yaw, pitch, roll, landmarks[68, 3])` per face per frame, attached
as `face.head_pose` by `head_geometry.estimate(frame, faces)`.  The one
provider today is `Landmark3DProvider`: insightface's `1k3d68.onnx` from
the buffalo_l pack (already downloaded for the detector), run through
`create_onnx_session` (TensorRT or CUDA graph like every other model), its
68 3D points fitted to insightface's mean shape for the Euler angles —
about 3 ms per face including the crop.  Anything that yields the same
record can replace it: a dense mesh (MediaPipe Face Mesh, 3DDFA_V2) or an
iPhone TrueDepth stream (LiveLinkFace sends head pose and 52 blendshapes
over UDP) — that is the planned third step.

The tracker drops `head_pose` together with `landmark_2d_106` when it
carries a face to a new frame, so the pose is always measured on the frame
being swapped.

## What it drives

- **Fade by real angle** (`pose_alpha`): 1 within the yaw / pitch limits
  from the Motion tab (defaults 55° / 35°), 0 past `limit × 1.3`, product of
  both axes.  `calibration.swap_alpha` multiplies it in before the profile,
  so it works without a profile and a profile can only narrow it.
- **3D outline** (`face_outline_mask`): the jaw contour plus the brow line
  mirrored upward (same forehead trick as the 2D hull) is unioned into the
  landmark mask.  On a frontal face the two hulls coincide; on a turn the
  3D contour is the visible silhouette (the far jaw projects inside the
  face), so the union keeps the far cheek covered without growing into ear
  or hair.
- The calibration dialog prints the measured yaw / pitch next to the
  proxies, which is the easiest way to see what the model thinks.

## Known behaviour

Measured on two near-frontal photos: yaw +10…+16°, pitch −12…−15°, roll
within a few degrees; an in-plane rotation of 20° moved roll by 20°.  The
model reads a slight downward pitch on frontal faces (camera above eye
level is the usual cause), so the pitch limit is effectively asymmetric —
raise it if the fade triggers when looking down.
