"""Shared ONNX-based face enhancement utilities for GPEN-BFR models.

Provides session creation, pre/post processing, and the core
enhance-face-via-ONNX pipeline.
"""

import os
import platform
import threading
from functools import lru_cache
from typing import Any, Optional

import cv2
import numpy as np
import onnxruntime

import modules.globals
from modules.platform_info import OPENVINO_PROVIDER_CONFIG

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Normalised 5-point alignment templates (left eye, right eye, nose,
# left mouth, right mouth).  Each restoration model expects the crop that
# matches the alignment it was trained on — FFHQ-trained models (GPEN 512+,
# CodeFormer, RestoreFormer, GFPGAN) want ``ffhq_512``, GPEN-256 wants the
# tighter ``arcface_128`` crop.
WARP_TEMPLATES = {
    "arcface_112_v2": np.array([
        [0.34191607, 0.46157411],
        [0.65653393, 0.45983393],
        [0.50022500, 0.64050536],
        [0.37097589, 0.82469196],
        [0.63151696, 0.82325089],
    ], dtype=np.float32),
    "arcface_128": np.array([
        [0.36167656, 0.40387734],
        [0.63696719, 0.40235469],
        [0.50019687, 0.56044219],
        [0.38710391, 0.72160547],
        [0.61507734, 0.72034453],
    ], dtype=np.float32),
    "dfl_whole_face": np.array([
        [0.35342266, 0.39285716],
        [0.62797622, 0.39285716],
        [0.48660713, 0.54017860],
        [0.38839287, 0.68750011],
        [0.59821427, 0.68750011],
    ], dtype=np.float32),
    "ffhq_512": np.array([
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465],
    ], dtype=np.float32),
    # Legacy Deep-Live-Cam template (a widened arcface_112_v2)
    "dlc_legacy": np.array([
        [0.31556875, 0.4615741],
        [0.68262291, 0.4615741],
        [0.50009375, 0.6405054],
        [0.34947187, 0.8246919],
        [0.65343645, 0.8246919],
    ], dtype=np.float32),
}

DEFAULT_TEMPLATE = "ffhq_512"

# Blend region pasted back over the frame: "ellipse" keeps everything outside
# the face oval (headphones, hair, background) untouched, "box" pastes the whole
# aligned crop the way Deep-Live-Cam did originally — stronger, but the model's
# reconstruction of the surroundings comes with it.
MASK_SHAPE = "ellipse"

# Limit concurrent ONNX calls to avoid VRAM exhaustion on multi-face frames
THREAD_SEMAPHORE = threading.Semaphore(min(max(1, (os.cpu_count() or 1)), 8))


def build_provider_config(providers=None):
    """Wrap raw provider name strings with optimised CUDA / CoreML options.

    Providers that are already ``(name, options_dict)`` tuples are passed
    through unchanged.  Non-CUDA providers are left as bare strings.
    """
    if providers is None:
        providers = modules.globals.execution_providers

    config = []
    for p in providers:
        if isinstance(p, tuple):
            # Already configured – pass through
            config.append(p)
        elif p == "CUDAExecutionProvider":
            # Bare provider unless another GPU was chosen — ONNX Runtime's
            # defaults are fastest on modern GPUs (Blackwell/sm_120).
            from modules.providers import cuda_provider

            config.append(cuda_provider())
        elif p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
            config.append((
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "AllowLowPrecisionAccumulationOnGPU": 1,
                },
            ))
        elif p == "OpenVINOExecutionProvider":
            # AUTO lets OpenVINO select the best device
            config.append(OPENVINO_PROVIDER_CONFIG)
        else:
            config.append(p)
    return config


def run_inference(session: onnxruntime.InferenceSession,
                  input_name: str,
                  input_tensor: "np.ndarray",
                  extra_inputs: dict | None = None) -> "np.ndarray":
    """Run ONNX inference, using IO binding when a CUDA session is active.

    IO binding avoids redundant host↔device copies by transferring the
    input tensor directly to GPU memory and letting ONNX Runtime allocate
    the output on the device.  Falls back to the standard ``session.run``
    path for non-CUDA providers or if binding fails.
    """
    from modules.cuda_graph import GraphSession
    from modules.providers import TrtSession

    if isinstance(session, (GraphSession, TrtSession)):
        # The graph session owns its binding, and TensorRT is faster
        # without one; both lock the GPU inside their own run().
        feed = {input_name: input_tensor}
        if extra_inputs:
            feed.update(extra_inputs)
        return session.run(None, feed)[0]

    if extra_inputs:
        # IO binding is skipped for multi-input models (e.g. CodeFormer's
        # fidelity weight) — the extra tensors are tiny and the copy is free.
        feed = {input_name: input_tensor}
        feed.update(extra_inputs)
        return session.run(None, feed)[0]

    if "CUDAExecutionProvider" in session.get_providers():
        try:
            io_binding = session.io_binding()

            # Input: numpy → GPU
            from modules.providers import cuda_device

            ort_input = onnxruntime.OrtValue.ortvalue_from_numpy(
                input_tensor, "cuda", cuda_device(),
            )
            io_binding.bind_ortvalue_input(input_name, ort_input)

            # Output: allocate on GPU (avoids a CPU-side allocation)
            output_name = session.get_outputs()[0].name
            io_binding.bind_output(output_name, "cuda", cuda_device())

            session.run_with_iobinding(io_binding)

            return io_binding.get_outputs()[0].numpy()
        except Exception:
            # Fall back to standard path (e.g. ORT version mismatch,
            # unsupported op, or VRAM pressure)
            pass

    return session.run(None, {input_name: input_tensor})[0]


def _graph_session(model_path: str, providers) -> Optional[Any]:
    """Try a CUDA-graph session; these models are launch-bound, not FLOP-bound.

    Recording the kernel launch sequence once roughly halves their runtime
    (GPEN-256 24 ms -> 13 ms, XSeg 18 ms -> 8 ms).  Only single-input models
    with fully static shapes qualify, and the graph session becomes *the*
    session so the weights are not held twice in VRAM.
    """
    if not any(
        p == "CUDAExecutionProvider" or
        (isinstance(p, tuple) and p[0] == "CUDAExecutionProvider")
        for p in providers
    ):
        return None

    try:
        from modules.cuda_graph import GraphSession

        probe = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"],
        )
        inputs = probe.get_inputs()
        shape = inputs[0].shape
        del probe
        if len(inputs) != 1 or not all(isinstance(d, int) and d > 0 for d in shape):
            return None
        return GraphSession(model_path, inputs[0].name, tuple(shape))
    except Exception as error:
        print(f"CUDA graph unavailable for {os.path.basename(model_path)}: {error}")
        return None


def create_onnx_session(model_path: str) -> onnxruntime.InferenceSession:
    """Create an ONNX Runtime session with optimised provider config.

    On Apple Silicon, applies CoreML graph optimizations (Pad decomposition,
    Shape/Gather folding, Split decomposition) to reduce CPU↔ANE partition
    boundaries.
    """
    if IS_APPLE_SILICON:
        from modules.onnx_optimize import optimize_for_coreml
        # Infer input shape from the model for Shape/Gather folding
        try:
            import onnx
            m = onnx.load(model_path)
            inp = m.graph.input[0]
            dims = inp.type.tensor_type.shape.dim
            shape = tuple(d.dim_value for d in dims if d.dim_value > 0)
            input_shape = shape if len(shape) == 4 else None
        except Exception:
            input_shape = None
        model_path = optimize_for_coreml(model_path, input_shape=input_shape)

    providers = build_provider_config()
    from modules.providers import make_session

    trt_session = make_session(model_path)
    if trt_session is not None:
        return trt_session
    graph_session = _graph_session(model_path, providers)
    if graph_session is not None:
        return graph_session

    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session = onnxruntime.InferenceSession(
        model_path, sess_options=session_options, providers=providers,
    )
    return session


def warmup_session(session: onnxruntime.InferenceSession) -> None:
    """Run a dummy inference pass to trigger JIT / compile caching."""
    dtypes = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(float16)": np.float16,
    }
    try:
        input_feed = {}
        for inp in session.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            dtype = dtypes.get(inp.type, np.float32)
            input_feed[inp.name] = (
                np.zeros(shape, dtype=dtype) if shape else np.array(1.0, dtype=dtype)
            )
        session.run(None, input_feed)
    except Exception as e:
        print(f"ONNX enhancer warmup skipped (non-fatal): {e}")


def preprocess_face(face_img: np.ndarray, input_size: int) -> np.ndarray:
    """Resize, normalize, and convert a BGR face crop to ONNX input blob.

    GPEN-BFR expects [1, 3, H, W] float32 in RGB, normalized to [-1, 1].
    """
    resized = cv2.resize(face_img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob


def postprocess_face(output: np.ndarray) -> np.ndarray:
    """Convert ONNX output [1, 3, H, W] float32 back to BGR uint8 image."""
    img = output[0].transpose(1, 2, 0)
    img = ((img + 1.0) / 2.0 * 255.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _get_face_affine(face: Any, input_size: int, template: str = DEFAULT_TEMPLATE):
    """Compute affine transform to align a face to the model input space.

    Returns (M, inv_M) — forward and inverse affine matrices.
    """
    template = WARP_TEMPLATES.get(template, WARP_TEMPLATES[DEFAULT_TEMPLATE]) * input_size

    landmarks = None
    if hasattr(face, "kps") and face.kps is not None:
        landmarks = face.kps.astype(np.float32)
    elif hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        lm106 = face.landmark_2d_106
        landmarks = np.array([
            lm106[38],  # left eye
            lm106[88],  # right eye
            lm106[86],  # nose tip
            lm106[52],  # left mouth
            lm106[61],  # right mouth
        ], dtype=np.float32)

    if landmarks is None or len(landmarks) < 5:
        return None, None

    M = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)[0]
    if M is None:
        return None, None
    inv_M = cv2.invertAffineTransform(M)
    return M, inv_M


@lru_cache(maxsize=16)
def _blend_mask_u8(input_size: int, shape: str) -> np.ndarray:
    """Blend mask as uint8, so the paste-back can stay in integer math."""
    return np.clip(_blend_mask(input_size, shape) * 255.0, 0, 255).astype(np.uint8)


@lru_cache(maxsize=16)
def _blend_mask(input_size: int, shape: str = "ellipse") -> np.ndarray:
    """Feathered elliptical mask covering the face inside an aligned crop.

    A full-square mask pastes the model's reconstruction of everything in the
    crop — hair edges, headphones, background — back over the frame, so the
    blend is restricted to the face oval.
    """
    if shape == "box":
        mask = np.ones((input_size, input_size), dtype=np.float32)
        border = max(1, input_size // 16)
        ramp = np.linspace(0, 1, border, dtype=np.float32)
        mask[:border, :] = ramp[:, np.newaxis]
        mask[-border:, :] = ramp[::-1][:, np.newaxis]
        mask[:, :border] = np.minimum(mask[:, :border], ramp[np.newaxis, :])
        mask[:, -border:] = np.minimum(mask[:, -border:], ramp[::-1][np.newaxis, :])
        return mask

    mask = np.zeros((input_size, input_size), dtype=np.float32)
    center = (int(input_size * 0.5), int(input_size * 0.52))
    axes = (int(input_size * 0.42), int(input_size * 0.50))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    blur = max(3, (input_size // 12) | 1)
    return cv2.GaussianBlur(mask, (blur, blur), 0)


# Smoothed face outline in normalised crop space, one entry per alignment
# template: the swapper (arcface_128) and the enhancer (its own template) see
# the same face in *different* crop frames, so their point sets must never be
# blended with each other.
_OUTLINE = {}
# Consecutive rejections after which a "jump" is accepted as the new truth —
# otherwise a bad first lock, or a real change of pose, would be refused forever.
OUTLINE_REJECT_STREAK = 3
# Forehead synthesised above the brow line: how many top landmarks form the
# brow, and how far up (as a fraction of the landmark height) to lift them.
BROW_POINTS = 12
CHIN_POINTS = 5
FOREHEAD_RISE = 0.35
# Median per-point jump (in crop widths) above which the landmark set is
# treated as a misfire rather than real movement.
OUTLINE_REJECT = 0.035
# Weight of the new landmark set when it is accepted. Tuned against a finger
# crossing the face (steadiness) and a 40-degree turn (responsiveness).
OUTLINE_BLEND = 0.25


# Last enhancer paste (alpha in crop space + frame->crop affine), for the
# live reprojection to know which pixels it may move.
LAST_ENHANCE = {"alpha": None, "affine": None}


def reset_face_outline() -> None:
    """Forget the smoothed outlines — call when the tracked face changes."""
    _OUTLINE.clear()


def face_outline_mask(face: Any, affine: np.ndarray, input_size: int,
                      frame_key: str = "arcface_128") -> Optional[np.ndarray]:
    """Blend mask following the real face outline, in aligned-crop space.

    A fixed ellipse assumes a frontal, centred face: on a turned head it still
    covers the face, but 57-63% of what it paints is ear, hair, neck or
    background, and that share grows with the turn.  The 106-point landmarks
    give the actual silhouette, so the pasted region tracks a yawing head.

    Those landmarks are unreliable while something covers the face — a finger
    crossing the mouth made the hull lose a third of its area from one frame to
    the next, which reads as the mask contracting and snapping back.  The
    outline is therefore smoothed in aligned space (where head motion is
    already factored out) and a set that jumps too far is ignored in favour of
    the last good one.

    Returns ``None`` when landmarks are unavailable, so callers fall back to
    the ellipse.
    """
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is None or len(landmarks) < 100:
        return None

    points = cv2.transform(
        np.asarray(landmarks, dtype=np.float32).reshape(1, -1, 2), affine,
    ).reshape(-1, 2) / float(input_size)

    from modules import calibration

    # With a calibrated reference, landmarks a finger or shadow pulled away
    # are put back where the person's own outline says they belong.
    points = calibration.refine_outline(points, frame_key, face)

    state = _OUTLINE.setdefault(frame_key, {"points": None, "rejected": 0})
    previous = state["points"]
    if previous is not None and previous.shape == points.shape:
        jump = float(np.median(np.linalg.norm(points - previous, axis=1)))
        if jump > OUTLINE_REJECT and state["rejected"] < OUTLINE_REJECT_STREAK:
            state["rejected"] += 1
            points = previous
        else:
            # Either a normal frame, or the change persisted long enough to be
            # real (head turned, first lock was bad) — follow it.
            state["rejected"] = 0
            points = OUTLINE_BLEND * points + (1.0 - OUTLINE_BLEND) * previous
    state["points"] = points

    scaled = points * input_size
    # The 106-point set has no forehead: its top edge is the eyebrow line, so
    # the hull stops there and the upper eyelids sit on the mask's feather.
    # Mirror the brow line upward to give the outline a forehead.
    height = float(scaled[:, 1].max() - scaled[:, 1].min())
    order = np.argsort(scaled[:, 1])
    brow = scaled[order[:BROW_POINTS]]
    rise = float(getattr(modules.globals, "mask_forehead", FOREHEAD_RISE))
    forehead = brow - np.array([0.0, height * rise], dtype=np.float32)
    parts = [scaled, forehead]
    drop = float(getattr(modules.globals, "mask_chin", 0.0))
    if drop > 0:
        # The chin slider pushes the lowest jaw points down (beard, double chin).
        parts.append(scaled[order[-CHIN_POINTS:]] + np.array([0.0, height * drop], dtype=np.float32))
    hull = cv2.convexHull(np.vstack(parts).astype(np.int32))
    mask = np.zeros((input_size, input_size), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    # The 3D fit's contour stays a silhouette on a turned head where the
    # 2D jaw points slide inward; the union keeps the far cheek covered.
    head = getattr(face, "head_pose", None)
    if head is not None and getattr(modules.globals, "head_outline", True):
        contour = cv2.transform(head.outline().reshape(1, -1, 2), affine).reshape(-1, 2)
        cv2.fillConvexPoly(mask, cv2.convexHull(contour.astype(np.int32)), 255)
    # Grow slightly so the feather sits outside the landmarks rather than
    # eating into the jaw, then soften the edge.
    grow = max(3, input_size // 40)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    blur = max(3, (input_size // 12) | 1)
    return cv2.GaussianBlur(mask, (blur, blur), 0)


def enhance_face_onnx(
    frame: np.ndarray,
    face: Any,
    session: onnxruntime.InferenceSession,
    input_size: int,
    template: str = DEFAULT_TEMPLATE,
    weight: float | None = None,
) -> np.ndarray:
    """Enhance a single face in the frame using an ONNX face restoration model.

    The frame is modified in place and returned, matching the face swapper's
    paste-back; callers that need the original must copy it first.

    ``weight`` feeds models that expose a fidelity input (CodeFormer).
    """
    M, inv_M = _get_face_affine(face, input_size, template)
    if M is None:
        return frame

    from modules import calibration

    # Past the calibrated turn limits the real face shows through; nothing
    # to enhance there either.
    pose_alpha = calibration.swap_alpha(face)
    if pose_alpha <= 0.01:
        return frame

    face_crop = cv2.warpAffine(
        frame, M, (input_size, input_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    blob = preprocess_face(face_crop, input_size)
    extra_inputs = {}
    for model_input in session.get_inputs()[1:]:
        if model_input.name == "weight":
            extra_inputs["weight"] = np.array(
                [1.0 if weight is None else weight], dtype=np.double,
            )
    with THREAD_SEMAPHORE:
        input_name = session.get_inputs()[0].name
        output = run_inference(session, input_name, blob, extra_inputs or None)
    enhanced = postprocess_face(output)
    if enhanced.shape[0] != input_size:
        # Models that upscale (e.g. 512 in / 1024 out) need the result back in
        # the crop's coordinate space before the inverse warp.
        enhanced = cv2.resize(
            enhanced, (input_size, input_size), interpolation=cv2.INTER_AREA,
        )

    mask = None
    if modules.globals.face_outline_mask:
        mask = face_outline_mask(face, M, input_size, frame_key=template)
    if mask is None:
        mask = _blend_mask_u8(input_size, MASK_SHAPE)
    if pose_alpha < 1.0:
        mask = (mask.astype(np.float32) * pose_alpha).astype(np.uint8)
    if getattr(modules.globals, "reprojection", False):
        LAST_ENHANCE["alpha"] = mask
        LAST_ENHANCE["affine"] = M

    # Warp and blend only inside the bounding box of the aligned crop: a
    # full-frame warp plus float blend costs more than the model inference.
    h, w = frame.shape[:2]
    corners = np.array([
        [0, 0], [input_size, 0], [input_size, input_size], [0, input_size],
    ], dtype=np.float32)
    projected = cv2.transform(corners[None, :, :], inv_M)[0]
    x0 = max(0, int(np.floor(projected[:, 0].min())) - 1)
    y0 = max(0, int(np.floor(projected[:, 1].min())) - 1)
    x1 = min(w, int(np.ceil(projected[:, 0].max())) + 1)
    y1 = min(h, int(np.ceil(projected[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return frame

    roi_M = inv_M.copy()
    roi_M[0, 2] -= x0
    roi_M[1, 2] -= y0
    roi_size = (x1 - x0, y1 - y0)

    warped_enhanced = cv2.warpAffine(
        enhanced, roi_M, roi_size,
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask, roi_M, roi_size,
        flags=cv2.INTER_LINEAR, borderValue=0,
    )

    # Fused uint8 blend — the float32 round-trip costs several milliseconds on
    # a face-sized ROI, and the result is identical after rounding.
    roi = frame[y0:y1, x0:x1]
    alpha = cv2.merge([warped_mask, warped_mask, warped_mask])
    inverse_alpha = cv2.subtract(255, alpha)
    frame[y0:y1, x0:x1] = cv2.add(
        cv2.multiply(warped_enhanced, alpha, scale=1.0 / 255.0),
        cv2.multiply(roi, inverse_alpha, scale=1.0 / 255.0),
    )
    return frame
