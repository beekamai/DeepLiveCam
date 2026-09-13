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
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Any, List, Optional

import onnxruntime

import modules.globals

ENGINE_CACHE = os.path.join(os.path.dirname(modules.globals.ROOT_DIR), "models", "trt_cache")
# Workspace TensorRT may use while building; engines run within it.
WORKSPACE_BYTES = 2 << 30

_available: Optional[bool] = None


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


def tensorrt_wanted() -> bool:
    """TensorRT is installed, enabled in the settings, and CUDA is the target."""
    if not getattr(modules.globals, "tensorrt", True) or not tensorrt_available():
        return False
    return any(
        p == "CUDAExecutionProvider" or (isinstance(p, tuple) and p[0] == "CUDAExecutionProvider")
        for p in modules.globals.execution_providers
    )


def tensorrt_providers() -> list:
    os.makedirs(ENGINE_CACHE, exist_ok=True)
    return [
        ("TensorrtExecutionProvider", {
            "trt_fp16_enable": "True",
            "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": ENGINE_CACHE,
            "trt_timing_cache_enable": "True",
            "trt_timing_cache_path": ENGINE_CACHE,
            "trt_max_workspace_size": str(WORKSPACE_BYTES),
        }),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def make_session(model_path: str, label: Optional[str] = None) -> Optional[Any]:
    """A session running the whole model on TensorRT, or ``None``.

    The first call for a model on this GPU builds the engine; the status
    line says so because the UI freezes for up to a minute meanwhile.
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
            model_path, sess_options=options, providers=tensorrt_providers(),
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
        session.run(None, feed)
        print(f"[tensorrt] {name} ready")
        return session
    except Exception as error:
        print(f"[tensorrt] {name}: {str(error)[:200]} — using CUDA")
        return None
