import cv2
import numpy as np
import time
from typing import Optional, Tuple, Callable
import platform
import threading

# Only import Windows-specific library if on Windows
if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph


class VideoCapturer:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.frame_callback = None
        self._current_frame = None
        self._frame_ready = threading.Event()
        self.is_running = False
        self.cap = None
        # Actual values reported by the camera after configuration
        self.actual_width: int = 0
        self.actual_height: int = 0
        self.actual_fps: float = 0.0
        self._abandoned = False
        self._start_result = False

        # Initialize Windows-specific components if on Windows
        if platform.system() == "Windows":
            self.graph = FilterGraph()
            # Verify device exists
            devices = self.graph.get_input_devices()
            if self.device_index >= len(devices):
                raise ValueError(
                    f"Invalid device index {device_index}. Available devices: {len(devices)}"
                )

    def start(self, width: int = 960, height: int = 540, fps: int = 60,
              timeout: float = 20.0, on_wait: Optional[Callable[[], None]] = None) -> bool:
        """Open the camera, giving up after ``timeout`` seconds.

        Opening a camera on Windows can block forever when another application
        holds the device (virtual cameras, conferencing apps): the backend
        reports the device as open but never delivers a frame, and a blocking
        ``read()`` never returns.  The work runs on a helper thread so a stuck
        device costs a timeout instead of a frozen UI; ``on_wait`` is called
        while waiting so the caller can keep its event loop alive.
        """
        self._abandoned = False
        self._start_result = False
        worker = threading.Thread(
            target=self._start_blocking, args=(width, height, fps), daemon=True,
        )
        worker.start()

        deadline = time.perf_counter() + timeout
        while worker.is_alive() and time.perf_counter() < deadline:
            worker.join(0.03)
            if on_wait is not None:
                on_wait()

        if worker.is_alive():
            # The thread is stuck inside OpenCV and cannot be killed; mark the
            # attempt abandoned so it releases the device when it does return.
            self._abandoned = True
            print(f"[VideoCapturer] camera {self.device_index} did not respond "
                  f"within {timeout:.0f}s — it is most likely in use by another "
                  "application", flush=True)
            return False

        return self._start_result

    def _start_blocking(self, width: int, height: int, fps: int) -> bool:
        """Initialize and start video capture"""
        try:
            if platform.system() == "Windows":
                # device_index comes from pygrabber.FilterGraph (DirectShow
                # enumeration), so open with DSHOW first to preserve mapping.
                # MSMF and DirectShow enumerate cameras in different orders, so
                # opening MSMF with a DSHOW index silently selects the wrong
                # camera. MSMF/ANY remain as fallbacks for cameras DSHOW can't
                # open.
                #
                # Pass codec + resolution + fps as construction params (OpenCV
                # 4.6+). DSHOW locks the pixel format at open time and ignores
                # later cap.set(CAP_PROP_FOURCC, ...) — without this, DSHOW
                # falls back to uncompressed YUYV at 1080p, which is USB-
                # bandwidth-limited to ~5 fps. Setting MJPG at construction
                # negotiates compressed frames from the first read.
                mjpg = cv2.VideoWriter_fourcc(*'MJPG')
                open_params = [
                    cv2.CAP_PROP_FOURCC, mjpg,
                    cv2.CAP_PROP_FRAME_WIDTH, width,
                    cv2.CAP_PROP_FRAME_HEIGHT, height,
                    cv2.CAP_PROP_FPS, fps,
                ]
                capture_methods = [
                    (self.device_index, cv2.CAP_DSHOW),
                    (self.device_index, cv2.CAP_MSMF),
                    (self.device_index, cv2.CAP_ANY),
                ]

                for dev_id, backend in capture_methods:
                    if self._abandoned:
                        break
                    try:
                        self.cap = cv2.VideoCapture(dev_id, backend, open_params)
                        # isOpened() is not proof of a working camera: a device
                        # held by another application opens and then starves.
                        # Require a real frame, retrying briefly because some
                        # cameras need a moment to deliver the first one.
                        if self.cap.isOpened() and self._await_first_frame():
                            break
                        self.cap.release()
                        self.cap = None
                    except Exception:
                        continue
            elif platform.system() == "Linux":
                self.cap = cv2.VideoCapture(f"/dev/video{self.device_index}")
            else:
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            # Belt-and-braces: also set via cap.set() for backends that honor
            # post-open changes (MSMF, V4L2). DSHOW ignores these, but the
            # construction params above already handled it.
            if platform.system() != "Windows":
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)

            # Read back resolution (usually reliable)
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # CAP_PROP_FPS is unreliable on DirectShow — often reports 30
            # even when the camera delivers 60.  Measure empirically by
            # timing a burst of frames.
            reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.actual_fps = self._measure_fps(warmup=10, sample=30,
                                                fallback=reported_fps or fps)

            print(f"[VideoCapturer] {self.actual_width}x{self.actual_height} "
                  f"@ {self.actual_fps:.1f}fps (reported={reported_fps:.0f})",
                  flush=True)

            if self._abandoned:
                # Caller gave up on us — free the device instead of holding it.
                self.cap.release()
                self.cap = None
                return False

            self.is_running = True
            self._start_result = True
            return True

        except Exception as e:
            print(f"Failed to start capture: {str(e)}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def _await_first_frame(self, attempts: int = 5, delay: float = 0.1) -> bool:
        """True once the camera delivers a frame; False if it never does."""
        for attempt in range(attempts):
            if self._abandoned:
                return False
            if self.cap.read()[0]:
                return True
            time.sleep(delay)
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera"""
        if not self.is_running or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self._current_frame = frame
            if self.frame_callback:
                self.frame_callback(frame)
            return True, frame
        return False, None

    def release(self) -> None:
        """Stop capture and release resources"""
        if self.is_running and self.cap is not None:
            self.cap.release()
            self.is_running = False
            self.cap = None

    def _measure_fps(self, warmup: int = 10, sample: int = 30,
                     fallback: float = 30.0, budget: float = 2.0) -> float:
        """Read warmup+sample frames and return measured FPS.

        This is more reliable than CAP_PROP_FPS which often lies on
        DirectShow.  Normally ~0.5-1s at startup; a virtual camera that is
        still spinning up can trickle its first frames at 1-2 fps, and then
        40 reads take longer than the open watchdog — so the probe has a
        wall-clock budget and settles for however many frames it got.
        """
        deadline = time.perf_counter() + budget
        try:
            for _ in range(warmup):
                ok, _ = self.cap.read()
                if not ok or self._abandoned:
                    # A camera that opens but delivers nothing is held by
                    # another application; don't keep reading from it.
                    return fallback
                if time.perf_counter() > deadline:
                    return fallback
            t0 = time.perf_counter()
            counted = 0
            for _ in range(sample):
                ret, _ = self.cap.read()
                if not ret or self._abandoned:
                    return fallback
                counted += 1
                if time.perf_counter() > deadline:
                    break
            elapsed = time.perf_counter() - t0
            if elapsed <= 0 or counted < 5:
                return fallback
            return counted / elapsed
        except Exception:
            return fallback

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for frame processing"""
        self.frame_callback = callback
