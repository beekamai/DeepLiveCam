"""XSeg occlusion masking — keeps hands, hair and props in front of the face.

The swap pastes a rectangle of generated face back over the frame, so anything
between the camera and the face (fingers, a mic, a vape) gets painted over and
smears.  XSeg segments the visible face inside the aligned crop; multiplying
that into the paste-back alpha makes occluders survive the swap.
"""

import gc
import os
import threading
from typing import Any, Optional

import cv2
import numpy as np

import modules.globals

MODEL_FILE = "xseg_3.onnx"
OPTIMISED_MODEL_FILE = "xseg_3_cuda.onnx"
INPUT_SIZE = 256
NAME = "DLC.FACE-OCCLUDER"

_SESSION = None
_LOCK = threading.Lock()
# Last computed mask, reused when the occluder runs at a lower cadence.
_CACHE = {"mask": None, "counter": 0}
# Last mask plus the affine that produced its crop, so the face tracker can
# tell which keypoints are currently hidden behind something.
_LAST = {"mask": None, "affine": None}


def _models_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models",
    )


def _optimised_model(model_path: str) -> str:
    """Return a GPU-friendly copy of the model, building it once if needed.

    Two things keep stock XSeg slow under ONNX Runtime:

    * a dynamic batch dimension, which blocks graph specialisation;
    * ``ConvTranspose`` nodes with asymmetric pads ``[0, 0, 1, 1]``, which the
      CUDA provider refuses — all six upsampling layers then run on the CPU
      with host round-trips, costing ~75% of the total runtime.

    Fixing the batch size and moving the crop into an explicit ``Slice`` makes
    the whole graph CUDA-eligible: measured 60 ms -> 15 ms, identical mask.
    """
    optimised_path = os.path.join(_models_dir(), OPTIMISED_MODEL_FILE)
    if os.path.isfile(optimised_path):
        return optimised_path
    try:
        import onnx
        import numpy as onnx_np
        from onnx import helper, numpy_helper

        model = onnx.load(model_path)
        graph = model.graph

        for tensor in (graph.input[0], graph.output[0]):
            dim = tensor.type.tensor_type.shape.dim[0]
            dim.Clear()
            dim.dim_value = 1

        graph.initializer.extend([
            numpy_helper.from_array(
                onnx_np.array([0, 0], dtype=onnx_np.int64), "occluder_slice_starts"),
            numpy_helper.from_array(
                onnx_np.array([-1, -1], dtype=onnx_np.int64), "occluder_slice_ends"),
            numpy_helper.from_array(
                onnx_np.array([2, 3], dtype=onnx_np.int64), "occluder_slice_axes"),
        ])

        nodes = []
        for node in graph.node:
            pads = next((list(a.ints) for a in node.attribute if a.name == "pads"), None)
            if node.op_type != "ConvTranspose" or pads != [0, 0, 1, 1]:
                nodes.append(node)
                continue
            for attribute in node.attribute:
                if attribute.name == "pads":
                    del attribute.ints[:]
                    attribute.ints.extend([0, 0, 0, 0])
            cropped_output = node.output[0]
            node.output[0] = cropped_output + "_uncropped"
            nodes.append(node)
            nodes.append(helper.make_node(
                "Slice",
                inputs=[node.output[0], "occluder_slice_starts",
                        "occluder_slice_ends", "occluder_slice_axes"],
                outputs=[cropped_output],
                name=node.name + "_crop",
            ))

        del graph.node[:]
        graph.node.extend(nodes)
        onnx.checker.check_model(model)
        onnx.save(model, optimised_path)
        print(f"{NAME}: built a CUDA-friendly model at {optimised_path}")
        return optimised_path
    except Exception as error:
        print(f"{NAME}: could not rebuild the model ({error}); using it as shipped")
        return model_path


def get_session() -> Optional[Any]:
    global _SESSION
    with _LOCK:
        if _SESSION is None:
            from modules.model_downloader import ensure_model
            from modules.processors.frame._onnx_enhancer import (
                create_onnx_session, warmup_session,
            )

            model_path = ensure_model(MODEL_FILE)
            if model_path is None:
                print(f"{NAME}: could not obtain {MODEL_FILE}; occlusion disabled")
                return None
            model_path = _optimised_model(model_path)
            print(f"{NAME}: Loading ONNX model from {model_path}")
            from modules.core import busy

            with busy("Loading occlusion model..."):
                _SESSION = create_onnx_session(model_path)
                warmup_session(_SESSION)
            print(f"{NAME}: Model loaded.")
    return _SESSION


def release() -> None:
    """Drop the session so its weights leave VRAM."""
    global _SESSION
    with _LOCK:
        if _SESSION is None:
            return
        _SESSION = None
    _CACHE["mask"] = None
    _LAST["mask"] = None
    _LAST["affine"] = None
    gc.collect()
    print(f"{NAME}: model unloaded")


def _predict(aligned_crop: np.ndarray) -> Optional[np.ndarray]:
    session = get_session()
    if session is None:
        return None

    blob = cv2.resize(aligned_crop, (INPUT_SIZE, INPUT_SIZE),
                      interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    blob = blob[np.newaxis, ...]
    try:
        output = session.run(None, {session.get_inputs()[0].name: blob})[0]
    except Exception as error:
        print(f"{NAME}: inference failed ({error})")
        return None

    mask = np.asarray(output, dtype=np.float32).reshape(INPUT_SIZE, INPUT_SIZE)
    # Soften the boundary, then push the transition band so occluder edges do
    # not leave a halo of half-swapped pixels.
    mask = (cv2.GaussianBlur(mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
    return mask


def publish_affine(affine: Optional[np.ndarray]) -> None:
    """Record the crop transform that matches the most recent mask."""
    _LAST["affine"] = None if affine is None else np.asarray(affine, dtype=np.float32)


def visible_points(points: np.ndarray) -> Optional[np.ndarray]:
    """Boolean mask over frame-space points: True where the face is not covered.

    Returns ``None`` when no occlusion mask has been computed yet.
    """
    mask, affine = _LAST["mask"], _LAST["affine"]
    if mask is None or affine is None:
        return None

    crop_points = cv2.transform(
        np.asarray(points, dtype=np.float32).reshape(1, -1, 2), affine,
    ).reshape(-1, 2)
    size = mask.shape[0]
    columns = np.clip(crop_points[:, 0].round().astype(int), 0, size - 1)
    rows = np.clip(crop_points[:, 1].round().astype(int), 0, size - 1)
    return mask[rows, columns] > 0.5


def get_occlusion_mask(aligned_crop: np.ndarray) -> Optional[np.ndarray]:
    """Mask over the aligned face crop: 1 where the swap may paint, 0 where not.

    Returns ``None`` when occlusion masking is off or the model is unavailable,
    which callers treat as "no restriction".
    """
    if not getattr(modules.globals, "occlusion_mask", False):
        return None

    size = aligned_crop.shape[0]
    interval = max(1, int(getattr(modules.globals, "occlusion_interval", 1)))
    cached = _CACHE["mask"]

    if cached is not None and _CACHE["counter"] % interval != 0:
        _CACHE["counter"] += 1
        _LAST["mask"] = cached
        return cv2.resize(cached, (size, size), interpolation=cv2.INTER_LINEAR)

    mask = _predict(aligned_crop)
    _CACHE["counter"] += 1
    if mask is None:
        return None
    _CACHE["mask"] = mask
    _LAST["mask"] = mask
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
