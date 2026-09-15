"""Record a webcam session for tools/replay_bench.py.

Opens the camera the way Live does (MJPG, see modules/video_capture.py — a
plain cv2 open of a virtual camera such as Camo can take minutes and
deliver 1 fps) and writes an mp4 at the camera's rate.

    venv/Scripts/python tools/record_session.py session.mp4 --seconds 60

Suggested script for the person in front of the camera: sit still for the
first third, turn the head fast for the second, and cross the face with a
hand for the last; pass the resulting spans as --rest / --motion to the
bench.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

from modules.video_capture import VideoCapturer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    cap = VideoCapturer(args.camera)
    if not cap.start(args.width, args.height, args.fps, timeout=120.0):
        print("camera did not open", file=sys.stderr)
        return 1
    started = time.time()
    frame = None
    while time.time() - started < 30:
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 20:
            break
    if frame is None:
        print("camera delivered no picture", file=sys.stderr)
        return 1
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    print(f"recording {width}x{height} for {args.seconds:.0f} s", flush=True)
    started = time.time()
    count = 0
    while time.time() - started < args.seconds:
        ok, frame = cap.read()
        if ok:
            writer.write(frame)
            count += 1
    writer.release()
    cap.release()
    print(f"{count} frames in {time.time() - started:.1f} s -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
