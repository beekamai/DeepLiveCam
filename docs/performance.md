# Performance notes

Measured on an RTX 5060 Ti 16 GB, onnxruntime-gpu 1.26, CUDA 12.8 wheels,
with nothing else on the GPU.  `nvidia-smi --query-compute-apps` must show
no other `python.exe` before any timing — the app itself is the GPU load
(56 % while running; the machine idles at 6 %), and a second instance
halves every number below.

## Where a frame goes (hyperswap_1b + GPEN-256 + XSeg + landmarks, 1280x720)

| stage | ms / frame |
|---|---|
| detect (every 0.25 s, amortised) | 1.4 |
| landmarks (every frame) | 1.4 |
| tracker flow | 1.5 |
| swap: graph replay 9.5, XSeg 4.7, warps + masks + paste ~8 | 22.8 |
| enhance: graph replay 8.5, warps + blend ~3 | 11.9 |
| **total** | **~40 → 25 fps** |

Inswapper-128 is not cheaper than HyperSwap-256 here (24.7 vs 22.8 ms for
the swap stage) — see below.

## The models are launch-bound, not compute-bound

ORT's own profile of `inswapper_128_fp16` puts the host-side node time at
7 ms while a plain `session.run` takes 17–38 ms: 273 nodes, 47 of them
`Cast` (the fp16 conversion left casts everywhere), each a separate kernel
launch plus stream synchronisation.  A CUDA graph replays the whole launch
sequence in one call: 29 → 19 ms plain-to-graph, and the same treatment
takes the detector 12 → 6 ms.  Every model in the live path runs as a graph
(`modules/cuda_graph.py`, the legacy path in `face_swapper.py` for
inswapper — its graph was silently disabled by a torch check until
2026-09-13).  CUDA EP options (`cudnn_conv_algo_search`, `prefer_nhwc`,
`use_tf32`) change nothing; the fp32 inswapper is slower than the fp16 one
(32 vs 19 ms).

## Why the stages cannot overlap on the GPU

Tried on 2026-09-13: three graph sessions (swap, GPEN, XSeg) replayed from
three threads with **per-session** locks instead of the global
`GRAPH_LOCK`.  onnxruntime 1.26 fails with `CUDA failure 900: operation not
permitted when stream is capturing` / `901: operation failed due to a
previous error during capture` and the sessions are poisoned — replays of
different sessions are not safe against each other in this build, so the
global lock stays and stages run one after another on the device.  CPU work
(warps, paste, masks) still overlaps with the device calls of the other
thread; that is the only overlap available.  A split enhancer thread was
measured earlier under the global lock: no gain.

Levers that remain: fewer CPU milliseconds around the models (paste-back
and mask warps are ~8 ms of the swap stage), running XSeg every 2nd frame
(`occlusion_interval`), and TensorRT (no wheels for this stack yet).
