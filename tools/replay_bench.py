"""Replay a recorded webcam session through the live processing worker.

Deterministic stand for the live pipeline: the same worker and queues as
Live, fed from a video file instead of the camera. Reports throughput,
camera-paced latency, keypoint and crop-affine stability per segment, the
size of the re-anchoring snap at each detection, and (with --profile) the
time spent per stage. Record the input with tools/record_session.py.

    venv/Scripts/python tools/replay_bench.py --video session.mp4 --source me.jpg \
        --rest 12:35 --motion 36:60 --profile --tag before
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="recorded session (see tools/record_session.py)")
    p.add_argument("--source", required=True, help="source face photo")
    p.add_argument("--rest", default="12:35", help="seconds of the video with a still face, a:b")
    p.add_argument("--motion", default="36:60", help="seconds with fast motion or occlusion, a:b")
    p.add_argument("--fps", type=float, default=30.0, help="camera rate of the recording")
    p.add_argument("--queue", type=int, default=1, help="capture queue slots for the paced pass")
    p.add_argument("--profile", action="store_true", help="time the pipeline stages")
    p.add_argument("--tag", default="run")
    p.add_argument("--out", default="bench_out")
    p.add_argument("--execution-provider", default="cuda")
    return p.parse_args()


ARGS = parse()
sys.argv = ["run.py", "--execution-provider", ARGS.execution_provider]
import run  # noqa: E402,F401  (DLL preamble only; core.run is guarded by __main__)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from modules import core, ui  # noqa: E402
import modules.face_tracker as ft  # noqa: E402
import modules.processors.frame.face_swapper as fs  # noqa: E402

os.makedirs(ARGS.out, exist_ok=True)
FPS = ARGS.fps
SEGMENTS = {name: tuple(int(v) for v in getattr(ARGS, name).split(":")) for name in ("rest", "motion")}

core.parse_args()
core.limit_resources()
ui.init(lambda: None, lambda: None, "en")
if not ui._accept_source_paths([ARGS.source]):
    sys.exit(f"no face in {ARGS.source}")

KPS: dict = {}
MS: dict = {}
SNAPS: list = []
PROF: dict = {}
RECORD = {"on": True}
_cur = {"seq": -1}


def _hook_geometry() -> None:
    orig_apply = ft.FaceTracker._apply

    def apply(self, face, points, bbox, timestamp=None, **kw):
        face = orig_apply(self, face, points, bbox, timestamp, **kw)
        if RECORD["on"]:
            KPS[_cur["seq"]] = np.asarray(face.kps, dtype=float)
        return face

    ft.FaceTracker._apply = apply

    orig_observe = ft.FaceTracker.observe

    def observe(self, frame, face, timestamp=None):
        # The snap the viewer sees at a detection: how far the flow's
        # prediction for this frame is from where the detector puts the face.
        if face is not None and self._points is not None and self._prev_gray is not None and RECORD["on"]:
            probe = type("F", (), {})()
            probe.kps = self._points.copy()
            probe.bbox = self._bbox.copy()
            keep = (self._points.copy(), self._bbox.copy(), self._velocity.copy())
            tracked = self.track(frame, probe, timestamp)
            self._points, self._bbox, self._velocity = keep
            if tracked is not None:
                gap = np.linalg.norm(np.asarray(tracked.kps, dtype=float) - np.asarray(face.kps, dtype=float), axis=1)
                SNAPS.append((_cur["seq"], float(gap.mean())))
        return orig_observe(self, frame, face, timestamp)

    ft.FaceTracker.observe = observe

    orig_paste = fs._fast_paste_back

    def paste(temp_frame, bgr_fake, aimg, M, occlusion):
        if RECORD["on"]:
            MS[_cur["seq"]] = np.asarray(M, dtype=float)
        return orig_paste(temp_frame, bgr_fake, aimg, M, occlusion)

    fs._fast_paste_back = paste


def _timed(name, fn):
    def wrapped(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            if RECORD["on"]:
                cell = PROF.setdefault(name, [0.0, 0])
                cell[0] += time.perf_counter() - t
                cell[1] += 1
    return wrapped


def _hook_profile() -> None:
    import modules.face_occluder as fo
    import modules.head_geometry as hg
    import modules.processors.frame.swapper_models as sm
    from modules.processors.frame import _enhancer_processor as ep

    ui.detect_one_face_fast = _timed("detect", ui.detect_one_face_fast)
    ui.ensure_landmarks = _timed("landmarks106", ui.ensure_landmarks)
    hg.estimate = _timed("head_pose", hg.estimate)
    ft.FaceTracker.track = _timed("flow_track", ft.FaceTracker.track)
    fs.swap_face = _timed("swap_face_total", fs.swap_face)
    fs.apply_post_processing = _timed("post", fs.apply_post_processing)
    fo.get_occlusion_mask = _timed("occluder", fo.get_occlusion_mask)
    fs._fast_paste_back = _timed("paste", fs._fast_paste_back)
    fs._apply_poisson_blend = _timed("poisson", fs._apply_poisson_blend)
    for name in dir(sm):
        cls = getattr(sm, name)
        if isinstance(cls, type) and "get" in cls.__dict__:
            cls.get = _timed(f"{name}.get", cls.__dict__["get"])
    for name in dir(ep):
        cls = getattr(ep, name)
        if isinstance(cls, type) and "process_frame" in cls.__dict__:
            cls.process_frame = _timed("enhancer", cls.__dict__["process_frame"])


class _SeqQueue(queue.Queue):
    def get(self, *a, **k):
        item = super().get(*a, **k)
        _cur["seq"] = item[0]
        return item


def run_pass(paced: bool):
    cq = _SeqQueue(maxsize=ARGS.queue if paced else 2)
    pq: queue.Queue = queue.Queue(maxsize=4)
    stop = threading.Event()
    worker = ui._ProcessingWorker(cq, pq, stop, FPS)
    worker.start()
    cap = cv2.VideoCapture(ARGS.video)
    sent: dict = {}
    got: list = []
    done = threading.Event()

    def consumer():
        while not done.is_set() or not pq.empty():
            try:
                r = pq.get(timeout=0.05)
            except queue.Empty:
                continue
            got.append((r.seq, time.perf_counter(), r.detected))
            if not paced and r.seq % 150 == 75:
                cv2.imwrite(os.path.join(ARGS.out, f"frame_{ARGS.tag}_{r.seq:04d}.jpg"), r.frame)

    ct = threading.Thread(target=consumer)
    ct.start()
    t0 = time.perf_counter()
    seq = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if paced:
            target = t0 + seq / FPS
            while time.perf_counter() < target:
                time.sleep(0.0005)
            if cq.full():  # drop-oldest, as the app's capture worker does
                try:
                    cq.get_nowait()
                except queue.Empty:
                    pass
            try:
                cq.put_nowait((seq, frame))
                sent[seq] = time.perf_counter()
            except queue.Full:
                pass
        else:
            cq.put((seq, frame))
            sent[seq] = time.perf_counter()
        seq += 1
    while not cq.empty():
        time.sleep(0.01)
    time.sleep(0.5)
    done.set()
    ct.join()
    stop.set()
    worker.wait(5000)
    return sent, got, seq


def _in(name, s):
    a, b = SEGMENTS[name]
    return a * FPS <= s < b * FPS


def kps_stats(name):
    k = np.array([KPS[s] for s in sorted(KPS) if _in(name, s)])
    if len(k) < 2:
        return None
    steps = np.linalg.norm(np.diff(k, axis=0), axis=-1)
    win = int(FPS)
    wander = [np.linalg.norm(k[i:i + win] - k[i:i + win].mean(0), axis=-1).mean() for i in range(0, len(k) - win, win)]
    return dict(frames=len(k), wander_px=float(np.mean(wander)), mean_step_px=float(steps.mean()),
                p95_step_px=float(np.percentile(steps, 95)), moving_pct=float((steps.max(axis=1) > 0.02).mean() * 100))


def affine_stats(name):
    M = np.array([MS[s] for s in sorted(MS) if _in(name, s)])
    if len(M) < 2:
        return None
    scale = np.hypot(M[:, 0, 0], M[:, 0, 1])
    angle = np.degrees(np.arctan2(M[:, 0, 1], M[:, 0, 0]))
    win = int(FPS)
    sc = [scale[i:i + win].std() / scale[i:i + win].mean() * 100 for i in range(0, len(M) - win, win)]
    an = [angle[i:i + win].std() for i in range(0, len(M) - win, win)]
    return dict(swaps=len(M), scale_std_pct=float(np.mean(sc)), angle_std_deg=float(np.mean(an)))


def snap_stats(name):
    gaps = np.array([g for s, g in SNAPS if _in(name, s)])
    if len(gaps) == 0:
        return None
    return dict(detections=len(gaps), mean_px=float(gaps.mean()), p50_px=float(np.percentile(gaps, 50)),
                p90_px=float(np.percentile(gaps, 90)), max_px=float(gaps.max()))


_hook_geometry()
if ARGS.profile:
    _hook_profile()

# Pass 1: every frame, as fast as the pipeline goes — deterministic geometry.
sent, got, n = run_pass(paced=False)
WARM = int(3 * FPS)
first = sent[min(s for s in sent if s >= WARM)]
last = max(t for _, t, _ in got)
throughput = len([g for g in got if g[0] >= WARM]) / (last - first)
detections = sum(1 for g in got if g[2] and g[0] >= WARM) / (last - first)

# Pass 2: camera pacing — what the viewer actually gets.
RECORD["on"] = False
sent2, got2, _ = run_pass(paced=True)
lat = np.array([t - sent2[s] for s, t, _ in got2 if s in sent2 and s >= WARM]) * 1000
paced_fps = len([g for g in got2 if g[0] >= WARM]) / (max(t for _, t, _ in got2) - sent2[min(s for s in sent2 if s >= WARM)])

result = dict(
    tag=ARGS.tag, frames=n, throughput_fps=throughput, detections_per_s=detections, paced_fps=paced_fps,
    latency_ms=dict(p50=float(np.percentile(lat, 50)), p95=float(np.percentile(lat, 95))),
    rest=dict(kps=kps_stats("rest"), affine=affine_stats("rest"), snap=snap_stats("rest")),
    motion=dict(kps=kps_stats("motion"), affine=affine_stats("motion"), snap=snap_stats("motion")),
)
if PROF:
    result["profile_ms_per_call"] = {k: dict(ms=t * 1000 / max(c, 1), calls=c) for k, (t, c) in PROF.items()}
with open(os.path.join(ARGS.out, f"bench_{ARGS.tag}.json"), "w") as f:
    json.dump(result, f, indent=1)
print(json.dumps(result, indent=1), flush=True)
os._exit(0)  # the offscreen Qt app and model sessions need no orderly teardown
