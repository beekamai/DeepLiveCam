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

## TensorRT (`modules/providers.py`)

The launch-bound diagnosis points straight at kernel fusion, and the ORT
1.26 wheel already carries `TensorrtExecutionProvider` linked against
`nvinfer_10`; the runtime comes from `tensorrt-cu12-libs` on NVIDIA's own
package index (`requirements-tensorrt.txt`, `install-tensorrt.bat`; it is
not on PyPI, which is why it looked unavailable at first).  Whole-model
engines, cached on disk per GPU architecture:

| model | CUDA graph | TensorRT fp32 | TensorRT fp16 | fp16 error vs CUDA |
|---|---|---|---|---|
| hyperswap_1b_256 | 9.5 ms | 6.2 ms | **4.4 ms** | 0.2 % (used) |
| GPEN-BFR-256 | 8.5 ms | **4.8 ms** | 3.2 ms | mean 0.06, max 1.0 on a -1..1 output |
| xseg_3 | 4.7 ms | **4.3 ms** | 2.4 ms | mean 0.017, max 0.45 on a 0..1 mask |
| inswapper_128_fp16 | 19 ms | **6.2 ms** | 5.8 ms | mean 0.06, max 0.41 on a 0..1 output |
| det_10g | 12 → 6 ms (graph) | 4.1 ms | **2.5 ms** | 0.4 % (used) |

Bold is what `modules/providers.py` builds.  fp16 engines of inswapper,
GPEN and XSeg produce visible mush (the swapped face turns into blotches
even though the numbers are finite — no NaN, just a 6 % mean error), so
engines are fp32 unless the model is listed in `FP16_OK`, where the fp16
output was measured against the CUDA run.  fp32 TensorRT still fuses the
launches, which is where the time went: the full pipeline runs at 33.7 fps
(hyperswap_1b + GPEN-256 + XSeg + landmarks) against 25.2 on CUDA graphs
and 36.9 with everything in fp16.  Engine builds take 11-35 s per model.

Gotchas met on the way: provider options must be the strings `"True"` /
`"False"` — `"1"` makes ORT drop the provider silently and fall back to
CUDA (the session then reports only CUDA in `get_providers()`, which is the
check `make_session` does); plain `session.run` is faster than io-binding
for TensorRT (no graph to replay).  Every session site — `GraphSession`,
`create_onnx_session`, `OnnxSwapper`, the legacy inswapper path — asks
`make_session` first and keeps its CUDA path as the fallback, so a missing
runtime or a failed build costs nothing but a log line.  `make_session`
returns a `TrtSession` proxy whose `run` takes `GRAPH_LOCK`, so TensorRT
runs never overlap a graph replay from another thread.

## Model loads belong to the processing worker

Changing the swapper, enhancer or occluder while Live runs used to load
the new model on the UI thread.  Two things went wrong with a TensorRT
build in that window: the load ran on the GPU concurrently with the
worker's graph replays (the hazard `_preload_live_models` describes), and
`update_status` pumped the Qt event loop from inside `get_face_swapper`
while it held its non-reentrant lock — a second change in the combo box
during the 30-60 s build re-entered `release()` on the same thread and the
app hung for good.  Now the handlers only release the old model and bump
`settings_epoch`; the worker reloads on the next frame (it is the only GPU
thread during Live, and the display keeps the last composed face while it
waits), and `update_status` flushes events with `ExcludeUserInputEvents`.

Levers that remain after TensorRT: the CPU milliseconds around the models
(paste-back and mask warps are ~8 ms of the swap stage) and running XSeg
every 2nd frame (`occlusion_interval`).
