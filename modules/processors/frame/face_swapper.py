from typing import Any, List, Optional, Tuple
import cv2
import insightface
import logging
import threading
import numpy as np
import platform
import modules.globals
import modules.processors.frame.core
from modules import imread_unicode, imwrite_unicode
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces, default_source_face
from modules.source_identity import SourceIdentity
from modules.typing import Face, Frame
from modules.utilities import (
    is_image,
    is_video,
)
from modules.cluster_analysis import find_closest_centroid
from modules.gpu_processing import gpu_gaussian_blur, gpu_sharpen, gpu_add_weighted
from modules.platform_info import OPENVINO_PROVIDER_CONFIG
from modules import swapper_registry
import os
from collections import deque
import time

FACE_SWAPPER = None
FACE_SWAPPER_KEY = None
# Alpha and crop affine of the most recent paste, for the preview overlay.
LAST_PASTE = {"alpha": None, "affine": None}
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-SWAPPER"

# --- START: Added for Interpolation ---
PREVIOUS_FRAME_RESULT = None # Stores the final processed frame from the previous step
# --- END: Added for Interpolation ---

# --- Poisson blend (ported from deep-live-cam-gumroad-edition) ---
# Root-cause fix for the "wobble": the blend mask is NOT built from the
# independently-detected 106-pt landmarks (they jitter sub-pixel every frame
# and seamlessClone is hyper-sensitive to its mask boundary). Instead it is
# derived from the swap's OWN affine transform (M) + the swapped pixels
# (bgr_fake), so the mask is locked exactly to where the swapped face was
# placed — no independent jitter source, no EMA, no lag. The mask is cached
# when the face is nearly still so an identical array is reused (zero wobble).
_ELLIPTICAL_MASK_CACHE: dict = {}
_poisson_cached_mask: Optional[np.ndarray] = None
_poisson_cached_key: Optional[tuple] = None


def _create_elliptical_mask(size: Tuple[int, int]) -> np.ndarray:
    """Fixed, heavily-blurred elliptical mask in aligned-face space.

    Geometry-based (not content-adaptive) and cached by size — identical
    every frame for the same model input size, so it contributes no jitter.
    """
    global _ELLIPTICAL_MASK_CACHE
    if size in _ELLIPTICAL_MASK_CACHE:
        return _ELLIPTICAL_MASK_CACHE[size]
    h, w = size
    center = (w // 2, h // 2)
    axes = (int(w * 0.44), int(h * 0.44))
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    if h * w < 65536:
        mask = cv2.GaussianBlur(mask, (31, 31), 12)
    else:
        mask = gpu_gaussian_blur(mask, (31, 31), 12)
    _ELLIPTICAL_MASK_CACHE[size] = mask
    return mask


POISSON_SOLVE_SCALE = 0.5   # Poisson solved at half resolution: 3x faster, <1 level of difference
POISSON_ROI_PAD = 8


def _poisson_roi_blend(swapped_frame: Frame, original_frame: Frame,
                       mask_roi: np.ndarray, x0: int, y0: int) -> bool:
    """Seamless-clone ``swapped_frame`` onto ``original_frame`` inside one
    face region, writing the result back into ``swapped_frame`` in place.

    ``mask_roi`` is the binary clone mask for the region whose top-left frame
    coordinate is ``(x0, y0)``.  The solve runs on a padded crop at
    ``POISSON_SOLVE_SCALE``; its output is a smooth colour correction, so
    resampling it back to full resolution costs under a level of error while
    the solver itself gets ~4x fewer pixels.  Returns False when the region
    cannot be blended (touches the frame border, empty mask).
    """
    h, w = swapped_frame.shape[:2]
    bx, by, bw, bh = cv2.boundingRect(mask_roi)
    if bw <= 0 or bh <= 0:
        return False
    rx0, ry0 = x0 + bx - POISSON_ROI_PAD, y0 + by - POISSON_ROI_PAD
    rx1, ry1 = x0 + bx + bw + POISSON_ROI_PAD, y0 + by + bh + POISSON_ROI_PAD
    if rx0 < 0 or ry0 < 0 or rx1 > w or ry1 > h:
        return False
    rw, rh = rx1 - rx0, ry1 - ry0
    region_mask = np.zeros((rh, rw), dtype=np.uint8)
    region_mask[POISSON_ROI_PAD:POISSON_ROI_PAD + bh, POISSON_ROI_PAD:POISSON_ROI_PAD + bw] =         mask_roi[by:by + bh, bx:bx + bw]

    src = swapped_frame[ry0:ry1, rx0:rx1]
    dst = original_frame[ry0:ry1, rx0:rx1]
    sw = max(int(rw * POISSON_SOLVE_SCALE), 16)
    sh = max(int(rh * POISSON_SOLVE_SCALE), 16)
    src_s = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
    dst_s = cv2.resize(dst, (sw, sh), interpolation=cv2.INTER_AREA)
    mask_s = cv2.resize(region_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
    mask_s = cv2.erode(mask_s, None)  # keeps the clone off the crop border
    mbx, mby, mbw, mbh = cv2.boundingRect(mask_s)
    if mbw <= 0 or mbh <= 0:
        return False
    center = (mbx + mbw // 2, mby + mbh // 2)
    blended_s = cv2.seamlessClone(src_s, dst_s, mask_s, center, cv2.NORMAL_CLONE)
    correction = cv2.subtract(blended_s, src_s, dtype=cv2.CV_16S)
    correction = cv2.resize(correction, (rw, rh), interpolation=cv2.INTER_LINEAR)
    corrected = cv2.add(src, correction, dtype=cv2.CV_8U)
    np.copyto(src, corrected, where=region_mask[:, :, None].astype(bool))
    return True


def _apply_poisson_blend(swapped_frame: Frame, original_frame: Frame,
                         target_face: Face, affine_matrix: np.ndarray = None,
                         bgr_fake: np.ndarray = None) -> Frame:
    """Poisson-blend the swapped face onto the original frame.

    Preferred path derives the blend mask from the swap's inverse affine so
    it tracks the swapped face exactly per-frame (no landmark jitter, no
    smoothing). Falls back to a cached bbox-ellipse if the affine is absent.
    Writes only the blended ellipse back so other faces are preserved.
    """
    global _poisson_cached_mask, _poisson_cached_key
    try:
        # ---- Preferred: blend ONLY the genuinely-swapped region ----
        # Use the exact paste-back mask (warped elliptical mask), eroded so
        # the Poisson seam sits on solidly-swapped pixels only.
        if affine_matrix is not None and bgr_fake is not None:
            try:
                h, w = swapped_frame.shape[:2]
                fh, fw = bgr_fake.shape[:2]
                inv = cv2.invertAffineTransform(affine_matrix)
                corners = np.array([[0, 0, 1], [fw, 0, 1], [fw, fh, 1], [0, fh, 1]],
                                   dtype=np.float32)
                t = corners @ inv.T
                px1 = max(0, int(np.floor(t[:, 0].min())))
                py1 = max(0, int(np.floor(t[:, 1].min())))
                px2 = min(w, int(np.ceil(t[:, 0].max())))
                py2 = min(h, int(np.ceil(t[:, 1].max())))
                rw, rh = px2 - px1, py2 - py1
                if rw > 8 and rh > 8:
                    roi_aff = inv.copy()
                    roi_aff[0, 2] -= px1
                    roi_aff[1, 2] -= py1
                    fm = _create_elliptical_mask((fh, fw))
                    mroi = cv2.warpAffine(fm, roi_aff, (rw, rh),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    bin_roi = np.where(mroi > 0.5, np.uint8(255), np.uint8(0))
                    k = max(3, (min(rw, rh) // 20) | 1)
                    bin_roi = cv2.erode(bin_roi,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
                    if _poisson_roi_blend(swapped_frame, original_frame, bin_roi, px1, py1):
                        return swapped_frame
            except Exception:
                pass  # fall through to the robust bbox-ellipse path below
        # ---- Fallback: bbox-ellipse (defensive, cached when still) ----
        if not hasattr(target_face, 'bbox') or target_face.bbox is None:
            return swapped_frame
        x1, y1, x2, y2 = target_face.bbox.astype(int)
        h, w = swapped_frame.shape[:2]
        x1, y1 = (max(0, x1), max(0, y1))
        x2, y2 = (min(w, x2), min(h, y2))
        if x2 <= x1 or y2 <= y1 or x2 - x1 <= 10 or (y2 - y1 <= 10):
            return swapped_frame
        padding = int(min(x2 - x1, y2 - y1) * 0.1)
        x1_p = max(0, x1 - padding)
        y1_p = max(0, y1 - padding)
        x2_p = min(w, x2 + padding)
        y2_p = min(h, y2 + padding)
        center_x = int(round((x1 + x2) / 2.0))
        center_y = int(round((y1 + y2) / 2.0))
        radius_x = max(1, int(round((x2_p - x1_p) / 2.0)))
        radius_y = max(1, int(round((y2_p - y1_p) / 2.0)))
        if not (0 <= center_x < w and 0 <= center_y < h):
            return swapped_frame
        center = (center_x, center_y)
        if center_x - radius_x < 0 or center_x + radius_x >= w or center_y - radius_y < 0 or (center_y + radius_y >= h):
            return swapped_frame
        # Reuse the cached ellipse while the face is still — identical mask,
        # zero wobble.
        rx0, ry0 = center_x - radius_x, center_y - radius_y
        mask_key = (radius_x, radius_y)
        if _poisson_cached_key == mask_key and _poisson_cached_mask is not None:
            mask = _poisson_cached_mask
        else:
            mask = np.zeros((2 * radius_y + 1, 2 * radius_x + 1), dtype=np.uint8)
            cv2.ellipse(mask, (radius_x, radius_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
            _poisson_cached_mask = mask
            _poisson_cached_key = mask_key
        _poisson_roi_blend(swapped_frame, original_frame, mask, rx0, ry0)
        return swapped_frame
    except Exception:
        return swapped_frame

# --- START: Mac M1-M5 Optimizations ---
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.machine() == 'arm64'
FRAME_CACHE = deque(maxlen=3)  # Cache for frame reuse
FACE_DETECTION_CACHE = {}  # Cache face detections
LAST_DETECTION_TIME = 0
DETECTION_INTERVAL = 0.033  # ~30 FPS detection rate for live mode
FRAME_SKIP_COUNTER = 0
ADAPTIVE_QUALITY = True
# --- END: Mac M1-M5 Optimizations ---

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)

def pre_check() -> bool:
    # Use models_dir instead of abs_dir to save to the correct location
    download_directory_path = models_dir

    # Make sure the models directory exists, catch permission errors if they occur
    try:
        os.makedirs(download_directory_path, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create directory {download_directory_path} due to permission error: {e}")
        return False

    from modules.model_downloader import ensure_any

    spec = _selected_spec()
    if spec.kind != "inswapper":
        if _resolve_model_path(spec) is None:
            update_status(
                f"Could not obtain {spec.model_file}. Place it in "
                f"{os.path.join(models_dir, spec.subdir)} manually or check "
                "your internet connection.",
                NAME,
            )
            return False
        return True

    variants = ["inswapper_128.onnx", "inswapper_128_fp16.onnx"]
    if _HAS_TORCH_CUDA:
        variants.reverse()
    if ensure_any(variants) is None:
        update_status(
            "Could not obtain the inswapper model. Place inswapper_128.onnx in "
            "the models folder manually or check your internet connection.",
            NAME,
        )
        return False
    return True


def _resolve_model_path(spec) -> Optional[str]:
    """Local path for a swap model, downloading it when it has a known source.

    DFM files live in ``models/dfm`` and are user-supplied, so they are never
    fetched — they either exist or the model is unavailable.
    """
    if spec.subdir:
        path = os.path.join(models_dir, spec.subdir, spec.model_file)
        return path if os.path.exists(path) else None

    from modules.model_downloader import ensure_model

    return ensure_model(spec.model_file)


def needs_source_face() -> bool:
    """False for models that carry their own identity (DFM) — no source photo."""
    return _selected_spec().embedding != "none"


def _selected_spec():
    """Registry entry for the swap model chosen in the UI / CLI."""
    key = getattr(modules.globals, "face_swapper_model", swapper_registry.DEFAULT_KEY)
    return swapper_registry.BY_KEY.get(
        key, swapper_registry.BY_KEY[swapper_registry.DEFAULT_KEY]
    )


def pre_start() -> bool:
    spec = _selected_spec()
    if spec.kind != "inswapper":
        if _resolve_model_path(spec) is None:
            update_status(
                f"{spec.model_file} not found in "
                f"{os.path.join(models_dir, spec.subdir)}.", NAME,
            )
            return False
        return get_face_swapper() is not None

    # Check for either model variant
    fp16_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
    fp32_path = os.path.join(models_dir, "inswapper_128.onnx")
    if not os.path.exists(fp16_path) and not os.path.exists(fp32_path):
        update_status(f"Model not found in {models_dir}. Please download inswapper_128.onnx.", NAME)
        return False

    # Try to get the face swapper to ensure it loads correctly
    if get_face_swapper() is None:
        # Error message already printed within get_face_swapper
        return False

    return True


def release() -> None:
    """Drop the swap model session so its weights leave VRAM."""
    global FACE_SWAPPER, FACE_SWAPPER_KEY
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            return
        FACE_SWAPPER = None
        FACE_SWAPPER_KEY = None
        _cuda_graph_session['recorded'] = False
        _cuda_graph_session['session'] = None
        _cuda_graph_session['io_binding'] = None
        _cuda_graph_session['ort_input'] = None
        _cuda_graph_session['ort_latent'] = None
    import gc
    gc.collect()
    update_status("Face swapper model unloaded.", NAME)


def _build_providers_config() -> list:
    """Per-provider session options shared by every swap model backend."""
    providers_config = []
    for p in modules.globals.execution_providers:
        if p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
            providers_config.append((
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "SpecializationStrategy": "FastPrediction",
                    "AllowLowPrecisionAccumulationOnGPU": 1,
                    "EnableOnSubgraphs": 1,
                },
            ))
        elif p == "CUDAExecutionProvider":
            from modules.providers import cuda_provider

            providers_config.append(cuda_provider())
        elif p == "OpenVINOExecutionProvider":
            providers_config.append(OPENVINO_PROVIDER_CONFIG)
        else:
            providers_config.append(p)
    return providers_config


def get_face_swapper() -> Any:
    global FACE_SWAPPER, FACE_SWAPPER_KEY

    spec = _selected_spec()
    from modules.core import busy

    with THREAD_LOCK, busy(f"Loading {spec.label}..."):
        if FACE_SWAPPER is not None and FACE_SWAPPER_KEY != spec.key:
            # Model switched in the UI — drop the old session (and its CUDA
            # graph) so the next frame runs the newly selected one.
            FACE_SWAPPER = None
            _cuda_graph_session['recorded'] = False
            _cuda_graph_session['session'] = None
            _cuda_graph_session['io_binding'] = None
            _cuda_graph_session['ort_input'] = None
            _cuda_graph_session['ort_latent'] = None
            import gc
            gc.collect()

        if FACE_SWAPPER is None and spec.kind != "inswapper":
            from modules.processors.frame.swapper_models import OnnxSwapper

            model_path = _resolve_model_path(spec)
            if model_path is None:
                update_status(f"Could not obtain {spec.model_file}.", NAME)
                return None
            update_status(f"Loading face swapper model from: {model_path}", NAME)
            try:
                FACE_SWAPPER = OnnxSwapper(
                    spec, model_path, _build_providers_config(),
                )
                FACE_SWAPPER_KEY = spec.key
                update_status("Face swapper model loaded successfully.", NAME)
            except Exception as error:
                update_status(f"Error loading face swapper model: {error}", NAME)
                FACE_SWAPPER = None
            return FACE_SWAPPER

        if FACE_SWAPPER is None:
            # Prefer FP16 on GPUs with Tensor Cores (Turing+) — half the
            # memory bandwidth, faster inference.  Fall back to FP32 for
            # older GPUs (e.g. GTX 16xx) where FP16 can produce NaN.
            fp32_path = os.path.join(models_dir, "inswapper_128.onnx")
            fp16_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
            use_fp16 = _HAS_TORCH_CUDA and os.path.exists(fp16_path)
            if use_fp16:
                model_path = fp16_path
            elif os.path.exists(fp32_path):
                model_path = fp32_path
            else:
                if not pre_check():
                    return None
                model_path = fp16_path if os.path.exists(fp16_path) else fp32_path
                if not os.path.exists(model_path):
                    update_status(f"No inswapper model found in {models_dir}.", NAME)
                    return None
            # On Apple Silicon, rewrite Pad(reflect) → Slice+Concat so
            # CoreML can run the entire model in a single partition on
            # the Neural Engine instead of bouncing between CPU and ANE.
            if IS_APPLE_SILICON:
                from modules.onnx_optimize import optimize_for_coreml
                model_path = optimize_for_coreml(model_path)

            update_status(f"Loading face swapper model from: {model_path}", NAME)
            try:
                from modules.providers import make_session

                providers_config = _build_providers_config()
                trt_session = make_session(model_path, "Inswapper-128")
                use_trt = trt_session is not None
                if use_trt:
                    from insightface.model_zoo.inswapper import INSwapper

                    FACE_SWAPPER = INSwapper(model_file=model_path, session=trt_session)
                else:
                    FACE_SWAPPER = insightface.model_zoo.get_model(
                        model_path,
                        providers=providers_config,
                    )
                # Set up CUDA graph session for faster inference.  Gated on
                # the provider actually in use — torch is not a dependency
                # of this path and its absence must not cost the graph.
                if not use_trt and any(
                    p == "CUDAExecutionProvider" or
                    (isinstance(p, tuple) and p[0] == "CUDAExecutionProvider")
                    for p in providers_config
                ):
                    _init_cuda_graph_session(model_path, FACE_SWAPPER)
                FACE_SWAPPER_KEY = spec.key
                update_status("Face swapper model loaded successfully.", NAME)
            except Exception as e:
                update_status(f"Error loading face swapper model: {e}", NAME)
                FACE_SWAPPER = None
                return None
    return FACE_SWAPPER


_HAS_TORCH_CUDA = False
try:
    import torch
    if torch.cuda.is_available():
        _HAS_TORCH_CUDA = True
except ImportError:
    pass

# Cache for paste-back
_paste_cache = {
    'soft_alpha': None,  # feathered alpha mask in aligned-face space
    'alpha_size': 0,
}


def _get_soft_alpha(size: int) -> np.ndarray:
    """Feathered alpha template in aligned-face space, cached.

    The legacy paste-back eroded and Gaussian-blurred the warped mask in
    output coordinates with kernels scaled to the output face size, which
    made the per-frame cost quartic in face linear size. Doing the same
    erode+blur once in aligned space and then warping the *soft* mask
    per-frame gives a visually equivalent feather at O(crop_area) cost —
    the feather radius scales naturally with the affine transform.
    """
    if _paste_cache['alpha_size'] != size:
        # Elliptical (not square) template — matches the gumroad edition's
        # _create_elliptical_mask. A full/eroded square leaves the aligned
        # crop's corners near-opaque, so the swapped square's straight edges
        # show as a visible box on the face. An ellipse (axes 0.44*size) zeroes
        # the corners and the heavy blur feathers smoothly into the original.
        center = (size // 2, size // 2)
        axes = (int(size * 0.44), int(size * 0.44))
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 12)
        _paste_cache['soft_alpha'] = mask  # uint8 [0, 255] — blended via cv2 SIMD ops
        _paste_cache['alpha_size'] = size
    return _paste_cache['soft_alpha']

# CUDA graph swap session cache
_cuda_graph_session = {
    'session': None,
    'io_binding': None,
    'ort_input': None,
    'ort_latent': None,
    'recorded': False,
}
# Serializes CUDA-graph replay. The io_binding + ort_input/ort_latent are
# shared across threads and run_with_iobinding mutates GPU-side buffers;
# concurrent calls would produce wrong output.
_cuda_graph_lock = threading.Lock()


class _CudaGraphSessionAdapter:
    """Drop-in wrapper around an ONNX Runtime session.

    Routes ``.run()`` through CUDA graph replay when a recorded graph is
    available, and transparently proxies every other attribute to the
    underlying session so insightface's INSwapper sees an unchanged API.
    """

    def __init__(self, underlying):
        # Use object.__setattr__ to bypass our own __setattr__.
        object.__setattr__(self, "_underlying", underlying)

    def run(self, output_names, input_dict, **kwargs):
        if _cuda_graph_session['recorded']:
            try:
                keys = list(input_dict.keys())
                blob = input_dict[keys[0]]
                latent = input_dict[keys[1]]
                return [_cuda_graph_swap_inference(blob, latent)]
            except Exception:
                pass
        return self._underlying.run(output_names, input_dict, **kwargs)

    def __getattr__(self, name):
        return getattr(self._underlying, name)

    def __setattr__(self, name, value):
        setattr(self._underlying, name, value)


def _init_cuda_graph_session(model_path: str, swapper):
    """Create a CUDA-graph-enabled ONNX session for the swap model.

    CUDA graphs record the GPU kernel launch sequence once, then replay it
    with near-zero CPU overhead on subsequent runs.  Requires static input
    shapes (inswapper is always 1x3x128x128 + 1x512).
    """
    import onnxruntime as ort
    try:
        from modules.providers import cuda_device

        device = cuda_device()
        providers = [('CUDAExecutionProvider', {'enable_cuda_graph': '1', 'device_id': str(device)})]
        sess = ort.InferenceSession(model_path, providers=providers)

        # Pre-allocate GPU buffers with correct shapes
        inp_shape = (1, 3, swapper.input_size[1], swapper.input_size[0])
        latent_shape = (1, 512)
        dummy_inp = np.zeros(inp_shape, dtype=np.float32)
        dummy_lat = np.zeros(latent_shape, dtype=np.float32)

        ort_input = ort.OrtValue.ortvalue_from_numpy(dummy_inp, 'cuda', device)
        ort_latent = ort.OrtValue.ortvalue_from_numpy(dummy_lat, 'cuda', device)

        io = sess.io_binding()
        io.bind_ortvalue_input(swapper.input_names[0], ort_input)
        io.bind_ortvalue_input(swapper.input_names[1], ort_latent)
        io.bind_output(swapper.output_names[0], 'cuda', device)

        # First run records the CUDA graph
        from modules.cuda_graph import GRAPH_LOCK

        with GRAPH_LOCK:
            sess.run_with_iobinding(io)

        _cuda_graph_session['session'] = sess
        _cuda_graph_session['io_binding'] = io
        _cuda_graph_session['ort_input'] = ort_input
        _cuda_graph_session['ort_latent'] = ort_latent
        _cuda_graph_session['recorded'] = True

        # Wrap swapper.session in an adapter instead of rebinding
        # session.run. insightface's INSwapper.get() reads .run via the
        # session attribute, so either works; the adapter survives any
        # later attribute reads on the session and keeps the original
        # session object untouched.
        if not isinstance(swapper.session, _CudaGraphSessionAdapter):
            swapper.session = _CudaGraphSessionAdapter(swapper.session)

        import sys
        print(f"[{NAME}] CUDA graph session initialized (swap model)")
        sys.stdout.flush()
    except Exception as e:
        print(f"[{NAME}] CUDA graph init failed, using standard session: {e}")
        _cuda_graph_session['recorded'] = False


def _cuda_graph_swap_inference(blob: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """Run swap model via CUDA graph replay — minimal CPU overhead."""
    from modules.cuda_graph import GRAPH_LOCK

    cg = _cuda_graph_session
    with _cuda_graph_lock, GRAPH_LOCK:
        cg['ort_input'].update_inplace(blob)
        cg['ort_latent'].update_inplace(latent)
        cg['session'].run_with_iobinding(cg['io_binding'])
        return cg['io_binding'].get_outputs()[0].numpy()


def _fast_paste_back(target_img: Frame, bgr_fake: np.ndarray, aimg: np.ndarray,
                     M: np.ndarray, occlusion: Optional[np.ndarray] = None) -> Frame:
    """Paste bgr_fake back onto target_img via the inverse affine of M.

    Restricts work to the face bbox in output coordinates and warps a
    precomputed feathered alpha template per-frame instead of running a
    size-scaled erode+blur on the warped mask. Cost is O(crop_area) regardless
    of how much of the frame the face occupies.
    """
    h, w = target_img.shape[:2]
    face_h, face_w = aimg.shape[:2]
    # inswapper's aligned-face space is square (128x128). _get_soft_alpha
    # caches a single NxN template keyed by N, so fail loudly if that ever
    # stops being true rather than silently mis-warping the alpha mask.
    assert face_h == face_w, f"Expected square aligned face, got {face_h}x{face_w}"
    IM = cv2.invertAffineTransform(M)

    # Bbox in output coords from the affine corners of the aligned-face square.
    corners = np.array(
        [[0, 0], [face_w, 0], [face_w, face_h], [0, face_h]], dtype=np.float32
    )
    transformed = (IM[:, :2] @ corners.T).T + IM[:, 2]
    x1 = int(np.floor(transformed[:, 0].min()))
    x2 = int(np.ceil(transformed[:, 0].max()))
    y1 = int(np.floor(transformed[:, 1].min()))
    y2 = int(np.ceil(transformed[:, 1].max()))
    if x1 >= x2 or y1 >= y2:
        return target_img

    # Small interpolation margin only — the feather is baked into the template.
    pad = 2
    y1p, y2p = max(0, y1 - pad), min(h, y2 + pad + 1)
    x1p, x2p = max(0, x1 - pad), min(w, x2 + pad + 1)

    IM_crop = IM.copy()
    IM_crop[0, 2] -= x1p
    IM_crop[1, 2] -= y1p
    crop_w, crop_h = x2p - x1p, y2p - y1p

    soft_alpha = _get_soft_alpha(face_h)
    if occlusion is not None:
        # Occluded pixels keep the original frame: scale the feathered alpha
        # down wherever XSeg says something is in front of the face.
        soft_alpha = (soft_alpha.astype(np.float32) * occlusion).astype(np.uint8)
    # Cubic upsampling of the 128/256 px result: +24 % edge sharpness on a
    # face twice that size for under a millisecond.
    bgr_fake_crop = cv2.warpAffine(bgr_fake, IM_crop, (crop_w, crop_h),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    alpha_crop = cv2.warpAffine(soft_alpha, IM_crop, (crop_w, crop_h), borderValue=0)

    target_crop = target_img[y1p:y2p, x1p:x2p]

    if _HAS_TORCH_CUDA:
        # Scale alpha to [0, 1] on device — cheaper to upload uint8 than float.
        mask_t = torch.from_numpy(alpha_crop).cuda().float().mul_(1.0 / 255.0).unsqueeze(2)
        fake_t = torch.from_numpy(bgr_fake_crop).float().cuda()
        tgt_t = torch.from_numpy(target_crop).float().cuda()
        blended = (mask_t * fake_t + (1.0 - mask_t) * tgt_t).to(torch.uint8).cpu().numpy()
        target_img[y1p:y2p, x1p:x2p] = blended
    else:
        # Fused uint8 blend via cv2 SIMD — no float32 round-trip.
        # Measured ~7-8× faster than the old numpy float32 path on a 1000×1000 crop.
        alpha_3c = cv2.merge([alpha_crop, alpha_crop, alpha_crop])
        inv_alpha = 255 - alpha_3c
        a_fake = cv2.multiply(bgr_fake_crop, alpha_3c, scale=1.0 / 255.0)
        a_tgt = cv2.multiply(target_crop, inv_alpha, scale=1.0 / 255.0)
        target_img[y1p:y2p, x1p:x2p] = cv2.add(a_fake, a_tgt)

    return target_img


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Optimized face swapping with better memory management and performance."""
    face_swapper = get_face_swapper()
    if face_swapper is None:
        update_status("Face swapper model not loaded or failed to load. Skipping swap.", NAME)
        return temp_frame

    # Safety check for faces
    if target_face is None:
        return temp_frame
    if needs_source_face():
        if source_face is None:
            return temp_frame
        if isinstance(source_face, SourceIdentity):
            # Several photos of the person: blend the embedding for this pose.
            source_face = source_face.face_for(target_face)
        if not hasattr(source_face, 'normed_embedding') or source_face.normed_embedding is None:
            return temp_frame

    # Past the calibrated turn limits the real face is shown instead of a
    # smeared swap — and the model is not even run.
    from modules import calibration

    target_face = calibration.stabilise(target_face)
    pose_alpha = calibration.swap_alpha(target_face)
    if pose_alpha <= 0.01:
        return temp_frame

    # _fast_paste_back writes in-place on the GPU path.  Only copy when
    # mouth_mask or opacity < 1 need an unmodified original.
    opacity = getattr(modules.globals, "opacity", 1.0)
    opacity = max(0.0, min(1.0, opacity))
    mouth_mask_enabled = getattr(modules.globals, "mouth_mask", False)
    poisson_blend_enabled = getattr(modules.globals, "poisson_blend", False)
    # Poisson blend's seamlessClone needs the genuine pre-swap frame as its
    # destination. Without this, original_frame aliases temp_frame, which
    # _fast_paste_back mutates in place — so seamlessClone would blend the
    # swapped face onto the already-swapped frame (no visible effect).
    needs_original = opacity < 1.0 or mouth_mask_enabled or poisson_blend_enabled
    if needs_original:
        original_frame = temp_frame.copy()
    else:
        original_frame = temp_frame

    if temp_frame.dtype != np.uint8:
        temp_frame = np.clip(temp_frame, 0, 255).astype(np.uint8)

    try:
        if not temp_frame.flags['C_CONTIGUOUS']:
            temp_frame = np.ascontiguousarray(temp_frame)

        # Use paste_back=False and our optimized paste-back
        model_mask = None
        if getattr(face_swapper, "returns_mask", False):
            # Identity-baked models (DFM) return their own face mask, which
            # beats the generic feathered square at the hairline and jaw.
            bgr_fake, M, model_mask = face_swapper.get_with_mask(
                temp_frame, target_face, source_face,
            )
        elif any("DmlExecutionProvider" in p for p in modules.globals.execution_providers):
            with modules.globals.dml_lock:
                bgr_fake, M = face_swapper.get(
                    temp_frame, target_face, source_face, paste_back=False
                )
        else:
            bgr_fake, M = face_swapper.get(
                temp_frame, target_face, source_face, paste_back=False
            )

        if bgr_fake is None:
            return original_frame

        if not isinstance(bgr_fake, np.ndarray):
            return original_frame

        # Pass a dummy aimg with correct shape — _fast_paste_back only uses aimg.shape
        # to create the white mask. Avoids redundant norm_crop2 (~0.6ms).
        _face_size = face_swapper.input_size[0]
        _aimg_dummy = np.empty((_face_size, _face_size, 3), dtype=np.uint8)

        occlusion = model_mask

        if getattr(modules.globals, "face_outline_mask", False):
            from modules.processors.frame._onnx_enhancer import face_outline_mask

            # The feathered square pastes ear, hair and neck along with the
            # face; on a turned head that is most of what it paints. The
            # landmark outline keeps the paste on the face at any angle.
            outline = face_outline_mask(target_face, M, _face_size)
            if outline is not None:
                outline = outline.astype(np.float32) / 255.0
                occlusion = outline if occlusion is None else np.minimum(
                    occlusion, outline,
                )

        if (getattr(modules.globals, "blink_reveal", True)
                or getattr(modules.globals, "eye_reveal", 0.0) > 0.0):
            from modules.face_reveal import eye_reveal_mask

            # Blinks and, optionally, the gaze come from the real eyes.
            reveal = eye_reveal_mask(target_face, M, _face_size)
            if reveal is not None:
                occlusion = reveal if occlusion is None else occlusion * reveal

        mouth_polygon = None
        if mouth_mask_enabled:
            from modules.face_reveal import mouth_reveal_mask

            # The real mouth (and, as the slider grows, the whole area from
            # nose to chin) shows through the swap.
            strength = float(getattr(modules.globals, "mouth_mask_size", 0.0)) / 100.0
            revealed = mouth_reveal_mask(
                target_face, M, _face_size, strength,
                getattr(modules.globals, "mouth_reveal_mode", "region"),
            )
            if revealed is not None:
                mouth_reveal, mouth_polygon = revealed
                occlusion = mouth_reveal if occlusion is None else occlusion * mouth_reveal

        if getattr(modules.globals, "occlusion_mask", False):
            from modules.face_occluder import get_occlusion_mask

            # The occluder must see the original face, not the swapped one —
            # warping the source frame with M costs a fraction of a millisecond.
            aligned_original = cv2.warpAffine(
                temp_frame, M, (_face_size, _face_size), borderMode=cv2.BORDER_REPLICATE,
            )
            from modules.face_occluder import publish_affine

            publish_affine(M)
            xseg_mask = get_occlusion_mask(aligned_original)
            if xseg_mask is not None:
                occlusion = xseg_mask if occlusion is None else np.minimum(
                    occlusion, xseg_mask,
                )

        if pose_alpha < 1.0:
            fade = np.full((_face_size, _face_size), pose_alpha, dtype=np.float32)
            occlusion = fade if occlusion is None else occlusion * pose_alpha

        swapped_frame = _fast_paste_back(temp_frame, bgr_fake, _aimg_dummy, M, occlusion)
        if (getattr(modules.globals, "show_mask_debug", False)
                or getattr(modules.globals, "reprojection", False)):
            LAST_PASTE["alpha"] = _get_soft_alpha(_face_size).astype(np.float32) / 255.0 * (
                1.0 if occlusion is None else occlusion)
            LAST_PASTE["affine"] = M

    except Exception as e:
        print(f"Error during face swap: {e}")
        return original_frame

    # --- Post-swap Processing (Masking, Opacity, etc.) ---
    # Now, work with the guaranteed uint8 'swapped_frame'

    if mouth_polygon is not None and getattr(modules.globals, "show_mouth_mask_box", False):
        outline = cv2.transform(
            mouth_polygon.reshape(1, -1, 2), cv2.invertAffineTransform(M),
        ).reshape(-1, 2).astype(np.int32)
        cv2.polylines(swapped_frame, [outline], True, (0, 255, 255), 2)

    # --- Poisson Blending ---
    # Mask derived from the swap's own affine (M) + swapped pixels (bgr_fake),
    # so it tracks the swapped face exactly per-frame — no landmark jitter,
    # no EMA, no lag. See _apply_poisson_blend.
    if getattr(modules.globals, "poisson_blend", False):
        swapped_frame = _apply_poisson_blend(
            swapped_frame, original_frame, target_face, M, bgr_fake
        )

    # Apply opacity blend between the original frame and the swapped frame
    if opacity >= 1.0:
        return swapped_frame.astype(np.uint8)

    # Blend the original_frame with the (potentially mouth-masked) swapped_frame
    final_swapped_frame = gpu_add_weighted(original_frame.astype(np.uint8), 1 - opacity, swapped_frame.astype(np.uint8), opacity, 0)
    return final_swapped_frame.astype(np.uint8)


# --- START: Mac M1-M5 Optimized Face Detection ---
def get_faces_optimized(frame: Frame, use_cache: bool = True) -> Optional[List[Face]]:
    """Optimized face detection for live mode on Apple Silicon"""
    global LAST_DETECTION_TIME, FACE_DETECTION_CACHE
    
    if not use_cache or not IS_APPLE_SILICON:
        # Standard detection
        if modules.globals.many_faces:
            return get_many_faces(frame)
        else:
            face = get_one_face(frame)
            return [face] if face else None
    
    # Adaptive detection rate for live mode
    current_time = time.time()
    time_since_last = current_time - LAST_DETECTION_TIME
    
    # Skip detection if too soon (adaptive frame skipping)
    if time_since_last < DETECTION_INTERVAL and FACE_DETECTION_CACHE:
        return FACE_DETECTION_CACHE.get('faces')
    
    # Perform detection
    LAST_DETECTION_TIME = current_time
    if modules.globals.many_faces:
        faces = get_many_faces(frame)
    else:
        face = get_one_face(frame)
        faces = [face] if face else None
    
    # Cache results
    FACE_DETECTION_CACHE['faces'] = faces
    FACE_DETECTION_CACHE['timestamp'] = current_time
    
    return faces
# --- END: Mac M1-M5 Optimized Face Detection ---

# --- START: Helper function for interpolation and sharpening ---
def draw_mask_debug(frame: Frame) -> Frame:
    """Overlay the last paste alpha on the frame (green painted, red held)."""
    alpha, affine = LAST_PASTE["alpha"], LAST_PASTE["affine"]
    if alpha is None or affine is None:
        return frame
    h, w = frame.shape[:2]
    size = alpha.shape[0]
    inverse = cv2.invertAffineTransform(affine)
    # Warp and tint only the crop's bounding box: a full-frame warp plus a
    # float blend of the whole frame cost ~40% of the live frame rate.
    corners = cv2.transform(
        np.array([[[0, 0], [size, 0], [size, size], [0, size]]], dtype=np.float32), inverse,
    ).reshape(-1, 2)
    x0 = max(0, int(np.floor(corners[:, 0].min())))
    y0 = max(0, int(np.floor(corners[:, 1].min())))
    x1 = min(w, int(np.ceil(corners[:, 0].max())) + 1)
    y1 = min(h, int(np.ceil(corners[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return frame
    shifted = inverse.copy()
    shifted[0, 2] -= x0
    shifted[1, 2] -= y0
    roi_size = (x1 - x0, y1 - y0)
    painted = cv2.warpAffine(alpha, shifted, roi_size, borderValue=0)
    box = cv2.warpAffine(np.ones((size, size), np.float32), shifted, roi_size, borderValue=0)
    held = np.clip(box - painted, 0, 1)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    roi[..., 1] += 90 * painted
    roi[..., 2] += 110 * held
    frame[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
    return frame


def apply_post_processing(current_frame: Frame, swapped_face_bboxes: List[np.ndarray]) -> Frame:
    """Applies sharpening and interpolation with Apple Silicon optimizations."""
    global PREVIOUS_FRAME_RESULT

    sharpness_value = getattr(modules.globals, "sharpness", 0.0)
    enable_interpolation = getattr(modules.globals, "enable_interpolation", False)

    # Skip copy when no post-processing is active
    if sharpness_value <= 0.0 and not enable_interpolation:
        PREVIOUS_FRAME_RESULT = None
        return current_frame

    processed_frame = current_frame.copy()

    # 1. Apply Sharpening (if enabled) with optimized kernel for Apple Silicon
    sharpness_value = getattr(modules.globals, "sharpness", 0.0)
    if sharpness_value > 0.0 and swapped_face_bboxes:
        height, width = processed_frame.shape[:2]
        for bbox in swapped_face_bboxes:
            # Ensure bbox is iterable and has 4 elements
            if not hasattr(bbox, '__iter__') or len(bbox) != 4:
                # print(f"Warning: Invalid bbox format for sharpening: {bbox}") # Debug
                continue
            x1, y1, x2, y2 = bbox
            # Ensure coordinates are integers and within bounds
            try:
                 x1, y1 = max(0, int(x1)), max(0, int(y1))
                 x2, y2 = min(width, int(x2)), min(height, int(y2))
            except ValueError:
                # print(f"Warning: Could not convert bbox coordinates to int: {bbox}") # Debug
                continue


            if x2 <= x1 or y2 <= y1:
                continue

            face_region = processed_frame[y1:y2, x1:x2]
            if face_region.size == 0:
                continue

            # Apply sharpening (GPU-accelerated when CUDA OpenCV is available)
            try:
                sigma = 2 if IS_APPLE_SILICON else 3
                sharpened_region = gpu_sharpen(face_region, strength=sharpness_value, sigma=sigma)
                processed_frame[y1:y2, x1:x2] = sharpened_region
            except cv2.error:
                pass


    # 2. Apply Interpolation (if enabled)
    enable_interpolation = getattr(modules.globals, "enable_interpolation", False)
    interpolation_weight = getattr(modules.globals, "interpolation_weight", 0.2)

    final_frame = processed_frame # Start with the current (potentially sharpened) frame

    if enable_interpolation and 0 < interpolation_weight < 1:
        if PREVIOUS_FRAME_RESULT is not None and PREVIOUS_FRAME_RESULT.shape == processed_frame.shape and PREVIOUS_FRAME_RESULT.dtype == processed_frame.dtype:
            # Perform interpolation
            try:
                 final_frame = gpu_add_weighted(
                    PREVIOUS_FRAME_RESULT, 1.0 - interpolation_weight,
                    processed_frame, interpolation_weight,
                    0
                 )
                 # Ensure final frame is uint8
                 final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
            except cv2.error as interp_e:
                 # print(f"Warning: OpenCV error during interpolation: {interp_e}") # Debug
                 final_frame = processed_frame # Use current frame if interpolation fails
                 PREVIOUS_FRAME_RESULT = None # Reset state if error occurs

            # Update the state for the next frame *with the interpolated result*
            PREVIOUS_FRAME_RESULT = final_frame.copy()
        else:
            # If previous frame invalid or doesn't match, use current frame and update state
            if PREVIOUS_FRAME_RESULT is not None and PREVIOUS_FRAME_RESULT.shape != processed_frame.shape:
                # print("Info: Frame shape changed, resetting interpolation state.") # Debug
                pass
            PREVIOUS_FRAME_RESULT = processed_frame.copy()
    else:
         # Interpolation is off or weight is invalid — no need to cache
         PREVIOUS_FRAME_RESULT = None


    return final_frame
# --- END: Helper function for interpolation and sharpening ---


def process_frame(source_face: Face, temp_frame: Frame, target_face: Face = None) -> Frame:
    """Process a single frame, swapping source_face onto detected target(s).

    Args:
        target_face: Pre-detected target face. When provided, skips the
            internal face detection call (saves ~30-40ms per frame).
            Ignored when many_faces mode is active.
    """
    if getattr(modules.globals, "opacity", 1.0) == 0:
        global PREVIOUS_FRAME_RESULT
        PREVIOUS_FRAME_RESULT = None
        return temp_frame

    processed_frame = temp_frame
    swapped_face_bboxes = []

    if modules.globals.many_faces:
        many_faces = get_many_faces(processed_frame)
        if many_faces:
            current_swap_target = processed_frame.copy()
            for face in many_faces:
                current_swap_target = swap_face(source_face, face, current_swap_target)
                if face is not None and hasattr(face, "bbox") and face.bbox is not None:
                    swapped_face_bboxes.append(face.bbox.astype(int))
            processed_frame = current_swap_target
    else:
        if target_face is None:
            target_face = get_one_face(processed_frame)
        if target_face:
            processed_frame = swap_face(source_face, target_face, processed_frame)
            if hasattr(target_face, "bbox") and target_face.bbox is not None:
                swapped_face_bboxes.append(target_face.bbox.astype(int))

    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)
    return final_frame


def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    """Handles complex mapping scenarios (map_faces=True) and live streams."""
    if getattr(modules.globals, "opacity", 1.0) == 0:
        # If opacity is 0, no swap happens, so no post-processing needed.
        # Also reset interpolation state if it was active.
        global PREVIOUS_FRAME_RESULT
        PREVIOUS_FRAME_RESULT = None
        return temp_frame

    processed_frame = temp_frame # Start with the input frame
    swapped_face_bboxes = [] # Keep track of where swaps happened

    # Determine source/target pairs based on mode
    source_target_pairs = []

    # Ensure maps exist before accessing them
    source_target_map = getattr(modules.globals, "source_target_map", None)
    simple_map = getattr(modules.globals, "simple_map", None)

    # Check if target is a file path (image or video) or live stream
    is_file_target = modules.globals.target_path and (is_image(modules.globals.target_path) or is_video(modules.globals.target_path))

    if is_file_target:
        # Processing specific image or video file with pre-analyzed maps
        if source_target_map:
            if modules.globals.many_faces:
                source_face = default_source_face() # Use default source for all targets
                if source_face:
                    for map_data in source_target_map:
                        if is_image(modules.globals.target_path):
                            target_info = map_data.get("target", {})
                            if target_info: # Check if target info exists
                                target_face = target_info.get("face")
                                if target_face:
                                    source_target_pairs.append((source_face, target_face))
                        elif is_video(modules.globals.target_path):
                             # Find faces for the current frame_path in video map
                             target_frames_data = map_data.get("target_faces_in_frame", [])
                             if target_frames_data: # Check if frame data exists
                                 target_frames = [f for f in target_frames_data if f and f.get("location") == temp_frame_path]
                                 for frame_data in target_frames:
                                     faces_in_frame = frame_data.get("faces", [])
                                     if faces_in_frame: # Check if faces exist
                                         for target_face in faces_in_frame:
                                             source_target_pairs.append((source_face, target_face))
            else: # Single face or specific mapping
                 for map_data in source_target_map:
                    source_info = map_data.get("source", {})
                    if not source_info:
                        continue # Skip if no source info
                    source_face = source_info.get("face")
                    if not source_face:
                        continue # Skip if no source defined for this map entry

                    if is_image(modules.globals.target_path):
                        target_info = map_data.get("target", {})
                        if target_info:
                           target_face = target_info.get("face")
                           if target_face:
                              source_target_pairs.append((source_face, target_face))
                    elif is_video(modules.globals.target_path):
                        target_frames_data = map_data.get("target_faces_in_frame", [])
                        if target_frames_data:
                           target_frames = [f for f in target_frames_data if f and f.get("location") == temp_frame_path]
                           for frame_data in target_frames:
                               faces_in_frame = frame_data.get("faces", [])
                               if faces_in_frame:
                                  for target_face in faces_in_frame:
                                      source_target_pairs.append((source_face, target_face))

    else:
        # Live stream or webcam processing (analyze faces on the fly)
        detected_faces = get_many_faces(processed_frame)
        if detected_faces:
            if modules.globals.many_faces:
                 source_face = default_source_face() # Use default source for all detected targets
                 if source_face:
                     for target_face in detected_faces:
                        source_target_pairs.append((source_face, target_face))
            elif simple_map:
                # Use simple_map (source_faces <-> target_embeddings)
                source_faces = simple_map.get("source_faces", [])
                target_embeddings = simple_map.get("target_embeddings", [])

                if source_faces and target_embeddings and len(source_faces) == len(target_embeddings):
                     # Match detected faces to the closest target embedding
                     if len(detected_faces) <= len(target_embeddings):
                          # More targets defined than detected - match each detected face
                          for detected_face in detected_faces:
                              if detected_face.normed_embedding is None:
                                  continue
                              closest_idx, _ = find_closest_centroid(target_embeddings, detected_face.normed_embedding)
                              if 0 <= closest_idx < len(source_faces):
                                  source_target_pairs.append((source_faces[closest_idx], detected_face))
                     else:
                          # More faces detected than targets defined - match each target embedding to closest detected face
                          detected_embeddings = [f.normed_embedding for f in detected_faces if f.normed_embedding is not None]
                          detected_faces_with_embedding = [f for f in detected_faces if f.normed_embedding is not None]
                          if not detected_embeddings:
                              return processed_frame # No embeddings to match

                          for i, target_embedding in enumerate(target_embeddings):
                              if 0 <= i < len(source_faces): # Ensure source face exists for this embedding
                                 closest_idx, _ = find_closest_centroid(detected_embeddings, target_embedding)
                                 if 0 <= closest_idx < len(detected_faces_with_embedding):
                                     source_target_pairs.append((source_faces[i], detected_faces_with_embedding[closest_idx]))
            else: # Fallback: if no map, use default source for the single detected face (if any)
                source_face = default_source_face()
                target_face = get_one_face(processed_frame, detected_faces) # Use faces already detected
                if source_face and target_face:
                    source_target_pairs.append((source_face, target_face))


    # Perform swaps based on the collected pairs
    current_swap_target = processed_frame.copy() # Apply swaps sequentially
    for source_face, target_face in source_target_pairs:
        if source_face and target_face:
            current_swap_target = swap_face(source_face, target_face, current_swap_target)
            if target_face is not None and hasattr(target_face, "bbox") and target_face.bbox is not None:
                swapped_face_bboxes.append(target_face.bbox.astype(int))
    processed_frame = current_swap_target # Assign final result


    # Apply sharpening and interpolation
    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)

    return final_frame


def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """
    Processes a list of frame paths (typically for video).
    Optimized with better memory management and caching.
    Iterates through frames, applies the appropriate swapping logic based on globals,
    and saves the result back to the frame path. Handles multi-threading via caller.
    """
    # Determine which processing function to use based on map_faces global setting
    use_v2 = getattr(modules.globals, "map_faces", False)
    source_face = None # Initialize source_face

    # --- Pre-load source face only if needed (Simple Mode: map_faces=False) ---
    if not use_v2 and needs_source_face():
        if not source_path or not os.path.exists(source_path):
            update_status(f"Error: Source path invalid or not provided for simple mode: {source_path}", NAME)
            # Log the error but allow proceeding; subsequent check will stop processing.
        else:
            try:
                from modules import source_identity

                paths = source_identity.current_paths() or [source_path]
                source_face, skipped = source_identity.load(paths)
                for path in skipped:
                    update_status(f"Warning: no face in source image {path}; skipped.", NAME)
                if source_face is None:
                    update_status(f"Error: no usable source image among {paths}. Swaps will be skipped.", NAME)
            except Exception as e:
                # Print the specific exception caught
                import traceback
                print(f"{NAME}: Caught exception during source image processing for {source_path}:")
                traceback.print_exc() # Print the full traceback
                update_status(f"Error during source image reading or analysis {source_path}: {e}", NAME)
                # Log general exception during the process

    total_frames = len(temp_frame_paths)
    # update_status(f"Processing {total_frames} frames. Use V2 (map_faces): {use_v2}", NAME) # Optional Debug

    # --- Stop processing entirely if in Simple Mode and source face is invalid ---
    if not use_v2 and source_face is None:
        update_status("Halting video processing: Invalid or no face detected in source image for simple mode.", NAME)
        if progress:
            # Ensure the progress bar completes if it was started
            remaining_updates = total_frames - progress.n if hasattr(progress, 'n') else total_frames
            if remaining_updates > 0:
                progress.update(remaining_updates)
        return # Exit the function entirely

    # --- Process each frame path provided in the list ---
    # Note: In the current core.py multi_process_frame, temp_frame_paths will usually contain only ONE path per call.
    for i, temp_frame_path in enumerate(temp_frame_paths):
        # update_status(f"Processing frame {i+1}/{total_frames}: {os.path.basename(temp_frame_path)}", NAME) # Optional Debug

        # Read the target frame
        temp_frame = None
        try:
            temp_frame = imread_unicode(temp_frame_path)
            if temp_frame is None:
                print(f"{NAME}: Error: Could not read frame: {temp_frame_path}, skipping.")
                if progress:
                    progress.update(1)
                continue # Skip this frame if read fails
        except Exception as read_e:
            print(f"{NAME}: Error reading frame {temp_frame_path}: {read_e}, skipping.")
            if progress:
                progress.update(1)
            continue

        # Select processing function and execute
        result_frame = None
        try:
            if use_v2:
                # V2 uses global maps and needs the frame path for lookup in video mode
                # update_status(f"Using process_frame_v2 for: {os.path.basename(temp_frame_path)}", NAME) # Optional Debug
                result_frame = process_frame_v2(temp_frame, temp_frame_path)
            else:
                # Simple mode uses the pre-loaded source_face (already checked for validity above)
                # update_status(f"Using process_frame (simple) for: {os.path.basename(temp_frame_path)}", NAME) # Optional Debug
                result_frame = process_frame(source_face, temp_frame) # source_face is guaranteed to be valid here

            # Check if processing actually returned a frame
            if result_frame is None:
                 print(f"{NAME}: Warning: Processing returned None for frame {temp_frame_path}. Using original.")
                 result_frame = temp_frame

        except Exception as proc_e:
            print(f"{NAME}: Error processing frame {temp_frame_path}: {proc_e}")
            # import traceback # Optional for detailed debugging
            # traceback.print_exc()
            result_frame = temp_frame # Use original frame on processing error

        # Write the result back to the same frame path with optimized compression
        try:
            # Use PNG compression level 3 (faster) instead of default 9
            write_success = imwrite_unicode(temp_frame_path, result_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not write_success:
                print(f"{NAME}: Error: Failed to write processed frame to {temp_frame_path}")
        except Exception as write_e:
            print(f"{NAME}: Error writing frame {temp_frame_path}: {write_e}")
        
        # Free memory immediately after processing
        del temp_frame
        if result_frame is not None:
            del result_frame

        # Update progress bar
        if progress:
            progress.update(1)
        # else: # Basic console progress (optional)
        #     if (i + 1) % 10 == 0 or (i + 1) == total_frames: # Update every 10 frames or on last frame
        #        update_status(f"Processed frame {i+1}/{total_frames}", NAME)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Processes a single target image."""
    # --- Reset interpolation state for single image processing ---
    global PREVIOUS_FRAME_RESULT
    PREVIOUS_FRAME_RESULT = None
    # ---

    use_v2 = getattr(modules.globals, "map_faces", False)

    # Read target first
    try:
        target_frame = imread_unicode(target_path)
        if target_frame is None:
            update_status(f"Error: Could not read target image: {target_path}", NAME)
            return
    except Exception as read_e:
        update_status(f"Error reading target image {target_path}: {read_e}", NAME)
        return

    result = None
    try:
        if use_v2:
            if getattr(modules.globals, "many_faces", False):
                 update_status("Processing image with 'map_faces' and 'many_faces'. Using pre-analysis map.", NAME)
            # V2 processes based on global maps, doesn't need source_path here directly
            # Assumes maps are pre-populated. Pass target_path for map lookup.
            result = process_frame_v2(target_frame, target_path)

        elif not needs_source_face():
            # DFM models hold the identity themselves — no source image.
            result = process_frame(None, target_frame)

        else: # Simple mode
            try:
                from modules import source_identity

                paths = source_identity.current_paths() or [source_path]
                source_face, skipped = source_identity.load(paths)
                for path in skipped:
                    update_status(f"Warning: no face in source image {path}; skipped.", NAME)
                if source_face is None:
                    update_status(f"Error: no usable source image among {paths}", NAME)
                    return
            except Exception as src_e:
                 update_status(f"Error reading or analyzing source image {source_path}: {src_e}", NAME)
                 return

            result = process_frame(source_face, target_frame)

        # Write the result if processing was successful
        if result is not None:
            write_success = imwrite_unicode(output_path, result)
            if write_success:
                update_status(f"Output image saved to: {output_path}", NAME)
            else:
                update_status(f"Error: Failed to write output image to {output_path}", NAME)
        else:
            # This case might occur if process_frame/v2 returns None unexpectedly
            update_status("Image processing failed (result was None).", NAME)

    except Exception as proc_e:
         update_status(f"Error during image processing: {proc_e}", NAME)
         # import traceback
         # traceback.print_exc()


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Sets up and calls the frame processing for video."""
    # --- Reset interpolation state before starting video processing ---
    global PREVIOUS_FRAME_RESULT
    PREVIOUS_FRAME_RESULT = None
    # ---

    mode_desc = "'map_faces'" if getattr(modules.globals, "map_faces", False) else "'simple'"
    if getattr(modules.globals, "map_faces", False) and getattr(modules.globals, "many_faces", False):
        mode_desc += " and 'many_faces'. Using pre-analysis map."
    update_status(f"Processing video with {mode_desc} mode.", NAME)

    # Pass the correct source_path (needed for simple mode in process_frames)
    # The core processing logic handles calling the right frame function (process_frames)
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames # Pass the newly modified process_frames
    )

def apply_color_transfer(source, target):
    """
    Apply color transfer using LAB color space. Handles potential division by zero and ensures output is uint8.
    """
    # Input validation
    if source is None or target is None or source.size == 0 or target.size == 0:
        # print("Warning: Invalid input to apply_color_transfer.")
        return source # Return original source if invalid input

    # Ensure images are 3-channel BGR uint8
    if len(source.shape) != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        # print("Warning: Source image for color transfer is not uint8 BGR.")
        # Attempt conversion if possible, otherwise return original
        try:
            if len(source.shape) == 2: # Grayscale
                source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            source = np.clip(source, 0, 255).astype(np.uint8)
            if len(source.shape) != 3 or source.shape[2] != 3:
                raise ValueError("Conversion failed")
        except Exception:
            return source
    if len(target.shape) != 3 or target.shape[2] != 3 or target.dtype != np.uint8:
        # print("Warning: Target image for color transfer is not uint8 BGR.")
        try:
            if len(target.shape) == 2: # Grayscale
                target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
            target = np.clip(target, 0, 255).astype(np.uint8)
            if len(target.shape) != 3 or target.shape[2] != 3:
                raise ValueError("Conversion failed")
        except Exception:
             return source # Return original source if target invalid

    result_bgr = source # Default to original source in case of errors

    try:
        # Convert to float32 [0, 1] range for LAB conversion
        source_float = source.astype(np.float32) / 255.0
        target_float = target.astype(np.float32) / 255.0

        source_lab = cv2.cvtColor(source_float, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_float, cv2.COLOR_BGR2LAB)

        # Compute statistics
        source_mean, source_std = cv2.meanStdDev(source_lab)
        target_mean, target_std = cv2.meanStdDev(target_lab)

        # Reshape for broadcasting
        source_mean = source_mean.reshape((1, 1, 3))
        source_std = source_std.reshape((1, 1, 3))
        target_mean = target_mean.reshape((1, 1, 3))
        target_std = target_std.reshape((1, 1, 3))

        # Avoid division by zero or very small std deviations (add epsilon)
        epsilon = 1e-6
        source_std = np.maximum(source_std, epsilon)
        # target_std = np.maximum(target_std, epsilon) # Target std can be small

        # Perform color transfer in LAB space
        result_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean

        # --- No explicit clipping needed in LAB space typically ---
        # Clipping is handled implicitly by the conversion back to BGR and then to uint8

        # Convert back to BGR float [0, 1]
        result_bgr_float = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

        # Clip final BGR values to [0, 1] range before scaling to [0, 255]
        result_bgr_float = np.clip(result_bgr_float, 0.0, 1.0)

        # Convert back to uint8 [0, 255]
        result_bgr = (result_bgr_float * 255.0).astype("uint8")

    except cv2.error as e:
         # print(f"OpenCV error during color transfer: {e}. Returning original source.") # Optional debug
         return source # Return original source if conversion fails
    except Exception as e:
         # print(f"Unexpected color transfer error: {e}. Returning original source.") # Optional debug
         # import traceback
         # traceback.print_exc()
         return source

    return result_bgr
