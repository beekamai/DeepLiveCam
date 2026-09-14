"""Virtual camera output: the composed live frame as a webcam device.

Uses ``pyvirtualcam``; on Windows it drives the OBS Virtual Camera driver
(installed with OBS Studio) or Unity Capture, on macOS OBS, on Linux
v4l2loopback.  Discord, Zoom, browsers then list "OBS Virtual Camera" and
receive the swapped picture directly — no window capture.  The device is
opened lazily with the first frame's size and closed when Live stops.
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

import modules.globals

NAME = "DLC.VIRTUAL-CAMERA"
# Re-entrant: a failure inside send() closes the device through stop().
_LOCK = threading.RLock()
_camera = None
_size: Optional[tuple] = None
_failed = False


def available() -> bool:
    try:
        import pyvirtualcam  # noqa: F401
    except ImportError:
        return False
    return True


def active() -> bool:
    return _camera is not None


def send(frame: np.ndarray, fps: float = 30.0) -> None:
    """Push a BGR frame; opens the device on the first call.  A device that
    cannot be opened switches the feature off and reports why, once."""
    global _camera, _size, _failed
    if _failed:
        return
    h, w = frame.shape[:2]
    with _LOCK:
        if _camera is None:
            try:
                import pyvirtualcam

                _camera = pyvirtualcam.Camera(
                    width=w, height=h, fps=max(1, int(round(fps))),
                    fmt=pyvirtualcam.PixelFormat.BGR, print_fps=False,
                )
                _size = (w, h)
                print(f"{NAME}: streaming {w}x{h} to {_camera.device}")
                from modules.core import update_status

                update_status(f"Virtual camera on: {_camera.device}")
            except ImportError:
                _fail("pyvirtualcam is not installed — pip install pyvirtualcam")
                return
            except Exception as error:
                _fail(f"Virtual camera could not start ({str(error)[:120]}). "
                      "Install OBS Studio (its virtual camera driver) and make sure "
                      "OBS itself is not using it.")
                return
        try:
            if (w, h) != _size:
                frame = cv2.resize(frame, _size, interpolation=cv2.INTER_AREA)
            _camera.send(np.ascontiguousarray(frame))
        except Exception as error:
            _fail(f"Virtual camera stopped ({str(error)[:120]})")


def _fail(message: str) -> None:
    global _failed
    _failed = True
    modules.globals.virtual_camera = False
    print(f"{NAME}: {message}")
    try:
        from modules.core import update_status

        update_status(message)
    except Exception:
        pass
    stop()


def stop() -> None:
    global _camera, _size, _failed
    with _LOCK:
        if _camera is not None:
            try:
                _camera.close()
            except Exception:
                pass
            print(f"{NAME}: stopped")
        _camera = None
        _size = None
    _failed = False
