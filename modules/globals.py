# --- START OF FILE globals.py ---

import os
from typing import List, Dict, Any

from modules.enhancer_registry import KEYS as ENHANCER_KEYS
from modules.swapper_registry import DEFAULT_KEY as DEFAULT_SWAPPER_KEY

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

# Canonical media extensions, defined once so the file dialogs and
# has_image_extension never drift. GIF is intentionally excluded: OpenCV's
# cv2.imread/imwrite (the only image I/O this app uses) cannot decode or
# encode GIF on 4.10 or 4.11, so offering it would silently fail. WEBP works
# via the libwebp bundled with opencv-python.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mkv")

# Face Mapping Data
source_target_map: List[Dict[str, Any]] = [] # Stores detailed map for image/video processing
simple_map: Dict[str, Any] = {}             # Stores simplified map (embeddings/faces) for live/simple mode

# Paths
source_path: str | None = None
# Every accepted source photo; source_path mirrors the first one.
source_paths: list = []
target_path: str | None = None
output_path: str | None = None

# Processing Options
frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False         # Process all detected faces with default source
map_faces: bool = False          # Use source_target_map or simple_map for specific swaps
poisson_blend: bool = False      # Enable Poisson Blending for smoother face swaps
color_correction: bool = False   # Enable color correction (implementation specific)
nsfw_filter: bool = False

# Video Output Options
video_encoder: str | None = None
video_quality: int | None = None # Typically a CRF value or bitrate

# Live Mode Options
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None # Placeholder for UI element if needed
webcam_preview_running: bool = False
show_fps: bool = False

# System Configuration
max_memory: int | None = None        # Memory limit in GB? (Needs clarification)
execution_providers: List[str] = []  # e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads: int | None = None # Number of threads for CPU execution
headless: bool | None = None         # Run without UI?
log_level: str = "error"             # Logging level (e.g., 'debug', 'info', 'warning', 'error')

# Face Processor UI Toggles (Example)
fp_ui: Dict[str, bool] = {key: False for key in ENHANCER_KEYS}
face_swapper_model: str = DEFAULT_SWAPPER_KEY
# Face crop used by the ONNX enhancers: "legacy" is Deep-Live-Cam's tight crop
# (more pixels on the face, hair left alone), "model" is each model's native
# training alignment (whole head, smoother, regenerates hair).
enhancer_alignment: str = "legacy"
# Track the face with optical flow on the frames between detections, so the
# swap follows motion instead of sticking to the last detected position.
face_tracking: bool = True
# Keep hands, hair and props that cover the face out of the swap (XSeg).
occlusion_mask: bool = False
# Run the occluder every Nth frame and reuse the mask in between — the model
# costs ~30 ms, which is more than the swap itself.
occlusion_interval: int = 1
# Shape the blend region from the 106 face landmarks instead of a fixed
# ellipse, so it follows the face when the head turns.
face_outline_mask: bool = True
# Draw the paste alpha over the live preview: green = swap painted here,
# red tint = held back (outline, occluder or the model's own mask).
show_mask_debug: bool = False
# Per-person calibration (modules.calibration): the active profile's name,
# whether its reference outline gates the landmark mask, and whether the swap
# fades to the real face past the calibrated turn limits.
calibration_profile: str | None = None
reference_outline: bool = True
pose_fade: bool = True
# Align the swap crop by the profile's stable landmarks instead of the raw
# mouth corners, so pursed or smiling lips do not shrink or grow the crop.
stable_alignment: bool = True
# Hand the eye region back to the camera: always during a blink (the models
# only squint), and at this strength the rest of the time (0 = swapped eyes,
# 1 = the real eyes and gaze).
blink_reveal: bool = True
eye_reveal: float = 0.0
# Live mode: move the last finished swap onto the newest camera frame with
# the tracker, so a slow enhancer delays the expression but not the position.
reprojection: bool = True
# Bumped by the UI on any toggle that changes what the live loop sees
# (mirror, tracking, masks, models); the workers re-detect and reset their
# trackers right away instead of showing the bare face until the next cycle.
settings_epoch: int = 0
# Run the models through TensorRT when its runtime is installed (2-3x faster
# than the CUDA graph; one-off engine build per model).
tensorrt: bool = True
# Calibration announces countdown/capture by tones and, optionally, an
# English voice naming the next pose.
calibration_sounds: bool = True
calibration_voice: bool = False
# Cap on swapped frames per second (0 = every camera frame); the display
# still shows every camera frame through reprojection.
max_fps: int = 0
# 3D head pose per frame (insightface 1k3d68): fade the swap past the real
# angles and add the 3D silhouette to the mask outline.
head_pose: bool = True
# Mouth reveal shape: "region" grows from the lips to nose base / chin,
# "lips" stays on the lips (plus a stuck-out tongue below) so a moustache
# above the lip is never revealed.
mouth_reveal_mode: str = "region"
# Mask outline reach beyond the landmarks, as a share of the face height:
# up over the forehead (the landmarks stop at the brows) and down past the chin.
mask_forehead: float = 0.35
mask_chin: float = 0.0
# CUDA device index for every session and buffer (restart to apply).
gpu_device: int = 0
# Camera capture size for Live: "360p", "480p", "720p", "1080p", "1440p".
camera_resolution: str = "720p"
head_outline: bool = True
head_yaw_limit: int = 55
head_pitch_limit: int = 35

# Face Swapper Specific Options
face_swapper_enabled: bool = True # General toggle for the swapper processor
opacity: float = 1.0              # Blend factor for the swapped face (0.0-1.0)
sharpness: float = 0.0            # Sharpness enhancement for swapped face (0.0-1.0+)

# Mouth Mask Options
mouth_mask: bool = False           # Enable mouth area masking/pasting
show_mouth_mask_box: bool = False  # Visualize the mouth mask area (for debugging)
mask_feather_ratio: int = 12       # Denominator for feathering calculation (higher = smaller feather)
mask_down_size: float = 0.1        # Expansion factor for lower lip mask (relative)
mask_size: float = 1.0             # Expansion factor for upper lip mask (relative)
mouth_mask_size: float = 0.0       # Mouth mask size (0-100; 0=off, 100=mouth to chin)

# --- START: Added for Frame Interpolation ---
enable_interpolation: bool = True # Toggle temporal smoothing
interpolation_weight: float = 0  # Blend weight for current frame (0.0-1.0). Lower=smoother.
# --- END: Added for Frame Interpolation ---

# --- END OF FILE globals.py ---

import threading
dml_lock = threading.Lock()
