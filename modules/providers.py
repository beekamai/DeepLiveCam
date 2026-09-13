"""TensorRT as an optional inference backend.

The live models are launch-bound: hundreds of tiny kernels per frame, so a
128 px swap costs as much as a 256 px one.  TensorRT compiles each ONNX
graph into one fused engine and runs it 2-3.5x faster than the CUDA-graph
replay (hyperswap 9.5 -> 4.2 ms, GPEN-256 8.5 -> 3.2, XSeg 4.7 -> 2.1,
inswapper 19 -> 5.4, measured on an RTX 5060 Ti).

The price is a one-off engine build per model and GPU (16-60 s, cached in
``models/trt_cache``) and the ``tensorrt-cu12-libs`` wheel (~3.4 GB) from
NVIDIA's index — see ``install-tensorrt.bat``.  Everything here degrades to
the CUDA path when the runtime is missing or a build fails.

Engines are built in fp32: fp16 turns inswapper, GPEN and XSeg into mush
(mean error 0.06 on a 0-1 output against the CUDA run) and buys only
0.5-2 ms per model.  Models measured exact in fp16 opt in via ``FP16_OK``.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from typing import Any, List, Optional

import onnxruntime

import modules.globals
from modules.cuda_graph import GRAPH_LOCK

ENGINE_CACHE = os.path.join(os.path.dirname(modules.globals.ROOT_DIR), "models", "trt_cache")
# Workspace TensorRT may use while building; engines run within it.
WORKSPACE_BYTES = 2 << 30
# Model-file prefixes whose fp16 engines match the CUDA output (measured:
# hyperswap 0.2 % relative error, the detector and landmarks 0.4 %).
FP16_OK = ("hyperswap_", "det_10g", "2d106det")

_available: Optional[bool] = None
_supported: Optional[bool] = None
# TensorRT 10 builds engines for Turing (compute capability 7.5) and newer.
MIN_COMPUTE_CAP = 7.5


def _runtime_dirs() -> List[str]:
    root = os.path.dirname(modules.globals.ROOT_DIR)
    dirs = []
    for site in (os.path.join(sys.prefix, "Lib", "site-packages"),
                 os.path.join(root, "venv", "Lib", "site-packages")):
        d = os.path.join(site, "tensorrt_libs")
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


def tensorrt_available() -> bool:
    """The ORT build carries the provider and the TensorRT runtime is installed."""
    global _available
    if _available is None:
        has_provider = "TensorrtExecutionProvider" in onnxruntime.get_available_providers()
        dirs = [d for d in _runtime_dirs() if glob.glob(os.path.join(d, "nvinfer_10*"))]
        # The provider DLL resolves nvinfer through the DLL search path, and
        # Python ignores PATH for that — register the wheel's directory here
        # so every entry point (UI, CLI, scripts) finds it, not just run.py.
        for d in dirs:
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(d)
                except OSError:
                    pass
        _available = bool(has_provider and (dirs or sys.platform != "win32"))
    return _available


def gpu_supports_tensorrt() -> bool:
    """The first NVIDIA GPU is new enough for TensorRT 10 (per nvidia-smi)."""
    global _supported
    if _supported is None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
            _supported = float(out.strip().splitlines()[0]) >= MIN_COMPUTE_CAP
        except Exception:
            _supported = False
    return _supported


def install_hint() -> Optional[str]:
    """A status line suggesting TensorRT when the GPU could use it but the runtime is absent."""
    cuda = any(
        p == "CUDAExecutionProvider" or (isinstance(p, tuple) and p[0] == "CUDAExecutionProvider")
        for p in modules.globals.execution_providers
    )
    if cuda and not tensorrt_available() and gpu_supports_tensorrt():
        return "Your GPU supports TensorRT (~1.5x fps) — run install-tensorrt.bat"
    return None


def tensorrt_wanted() -> bool:
    """TensorRT is installed, enabled in the settings, and CUDA is the target."""
    if not getattr(modules.globals, "tensorrt", True) or not tensorrt_available():
        return False
    return any(
        p == "CUDAExecutionProvider" or (isinstance(p, tuple) and p[0] == "CUDAExecutionProvider")
        for p in modules.globals.execution_providers
    )


def tensorrt_providers(model_path: str = "") -> list:
    os.makedirs(ENGINE_CACHE, exist_ok=True)
    fp16 = os.path.basename(model_path).startswith(FP16_OK)
    return [
        ("TensorrtExecutionProvider", {
            "trt_fp16_enable": "True" if fp16 else "False",
            "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": ENGINE_CACHE,
            "trt_timing_cache_enable": "True",
            "trt_timing_cache_path": ENGINE_CACHE,
            "trt_max_workspace_size": str(WORKSPACE_BYTES),
        }),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


class TrtSession:
    """An ``InferenceSession`` whose runs take the shared GPU lock.

    TensorRT runs and CUDA-graph replays from different threads corrupt
    each other (see ``cuda_graph.GRAPH_LOCK``); routing every run through
    the same lock keeps insightface, the enhancers and the occluder safe
    without each of them knowing which backend they got.
    """

    def __init__(self, session: Any) -> None:
        self._session = session

    def run(self, output_names, input_feed, **kwargs):
        with GRAPH_LOCK:
            return self._session.run(output_names, input_feed, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)


def make_session(model_path: str, label: Optional[str] = None) -> Optional[Any]:
    """A session running the whole model on TensorRT, or ``None``.

    The first call for a model on this GPU builds the engine (up to a
    minute); the status line says so.
    """
    if not tensorrt_wanted():
        return None
    name = label or os.path.basename(model_path)
    try:
        from modules.core import update_status

        update_status(f"TensorRT: preparing engine for {name} (first time takes up to a minute)...")
    except Exception:
        pass
    try:
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        session = onnxruntime.InferenceSession(
            model_path, sess_options=options, providers=tensorrt_providers(model_path),
        )
        if session.get_providers()[0] != "TensorrtExecutionProvider":
            print(f"[tensorrt] provider not attached for {name}; using CUDA")
            return None
        # A run builds (or loads) the engine and confirms it works.
        import numpy as np

        feed = {}
        for inp in session.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            dtype = {"tensor(float16)": np.float16, "tensor(double)": np.float64}.get(inp.type, np.float32)
            feed[inp.name] = np.zeros(shape, dtype=dtype)
        with GRAPH_LOCK:
            session.run(None, feed)
        print(f"[tensorrt] {name} ready")
        return TrtSession(session)
    except Exception as error:
        print(f"[tensorrt] {name}: {str(error)[:200]} — using CUDA")
        return None
