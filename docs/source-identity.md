# Source identity from several photos

`modules/source_identity.py`.  The swappers (inswapper, HyperSwap) see the
source person only as a 512-d ArcFace embedding — extra photos cannot show
them a profile, the generator paints turned faces from the target frame.
What several photos *do* buy:

- a steadier identity: one photo's lighting or expression no longer leaks
  into every frame (the blend averages them out);
- a pose-matched embedding: ArcFace is nearly pose-invariant, but not
  quite, and the residual is exactly what shows on turned faces, so the
  blend leans towards the photo taken at a pose like the target's.

## How the blend works

Every accepted photo is analysed once (`get_one_face`); its five keypoints
give the same yaw/pitch proxies the calibration uses (`calibration.
pose_proxies`).  Per frame the target's proxies pick weights

    w_i = BASE_WEIGHT / n + (1 - BASE_WEIGHT) * exp(-d_i² / (2 · POSE_SCALE²))

(`BASE_WEIGHT` 0.3 keeps every photo in the mix, `POSE_SCALE` 0.15 matches
the calibration references), normalised.  The unit embeddings are blended,
renormalised to the photos' mean norm and handed back inside a copy of the
nearest photo's `Face`, so `swap_face` and the swapper models are unchanged
— `swap_face` just resolves a `SourceIdentity` to a `Face` for the target it
is about to swap.  Pose proxies are quantised to 1/40 for a cache, so a run
of similar frames costs nothing.

## Where it plugs in

`modules.globals.source_paths` is the list; `source_path` mirrors the first
entry because the rest of the code checks it for "is a source chosen".  The
file dialog accepts several files, photos without a face are skipped with a
status line, the caption under the thumbnail says how many are in use.  The
live worker, the still preview and the photo/video pipeline all load
through `source_identity.load()`, and the CLI `-s` path becomes a one-photo
list.  Swap source/target is refused with more than one photo.

## What it is not

Not a profile fix: past ~60° the detector, the landmarks and the generator
all give out regardless of the source.  The next step for turned faces is a
per-frame head mesh (3DDFA_V2 / MediaPipe Face Mesh, or an iPhone TrueDepth
stream) replacing the five-pose calibration.

## Roadmap for turned faces

1. Several photos blended by pose (this document) — done.
2. A per-frame head mesh from the webcam (3DDFA_V2 or MediaPipe Face Mesh,
   2-4 ms on CPU): mask outline as the projected mesh, fade decided by the
   real angle, calibration becomes optional fine-tuning.
3. The same mesh interface fed by an iPhone TrueDepth stream (LiveLinkFace
   sends 52 blendshapes and the head pose over UDP) — more precise, no
   jitter, needs the phone.
