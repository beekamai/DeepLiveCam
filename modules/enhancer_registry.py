"""Single source of truth for the available face enhancers.

Adding an entry here (plus the matching ``modules/processors/frame/<key>.py``
wrapper) is enough for the CLI choices, the UI dropdown, the landmark
requirement check and the live loop to pick the model up.
"""

from typing import Dict, List, NamedTuple, Optional


class EnhancerSpec(NamedTuple):
    key: str              # frame-processor key / module name
    label: str            # UI dropdown label
    name: str             # processor NAME constant
    model_file: str
    input_size: int
    template: str         # alignment template, see _onnx_enhancer.WARP_TEMPLATES
    mirror_url: Optional[str] = None


ENHANCERS: List[EnhancerSpec] = [
    EnhancerSpec(
        key="face_enhancer",
        label="GFPGAN-1024",
        name="DLC.FACE-ENHANCER",
        model_file="gfpgan-1024.onnx",
        input_size=1024,
        template="ffhq_512",
    ),
    EnhancerSpec(
        key="face_enhancer_gpen256",
        label="GPEN-256",
        name="DLC.FACE-ENHANCER-GPEN256",
        model_file="GPEN-BFR-256.onnx",
        input_size=256,
        template="arcface_128",
        mirror_url="https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/GPEN-BFR/GPEN-BFR-256.onnx",
    ),
    EnhancerSpec(
        key="face_enhancer_gpen512",
        label="GPEN-512",
        name="DLC.FACE-ENHANCER-GPEN512",
        model_file="GPEN-BFR-512.onnx",
        input_size=512,
        template="ffhq_512",
        mirror_url="https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/GPEN-BFR/GPEN-BFR-512.onnx",
    ),
    EnhancerSpec(
        key="face_enhancer_gpen1024",
        label="GPEN-1024",
        name="DLC.FACE-ENHANCER-GPEN1024",
        model_file="gpen_bfr_1024.onnx",
        input_size=1024,
        template="ffhq_512",
    ),
    EnhancerSpec(
        key="face_enhancer_codeformer",
        label="CodeFormer",
        name="DLC.FACE-ENHANCER-CODEFORMER",
        model_file="codeformer.onnx",
        input_size=512,
        template="ffhq_512",
    ),
    EnhancerSpec(
        key="face_enhancer_restoreformer",
        label="RestoreFormer++",
        name="DLC.FACE-ENHANCER-RESTOREFORMER",
        model_file="restoreformer_plus_plus.onnx",
        input_size=512,
        template="ffhq_512",
    ),
    EnhancerSpec(
        key="face_enhancer_gfpgan14",
        label="GFPGAN-1.4",
        name="DLC.FACE-ENHANCER-GFPGAN14",
        model_file="gfpgan_1.4.onnx",
        input_size=512,
        template="ffhq_512",
    ),
]

BY_KEY: Dict[str, EnhancerSpec] = {spec.key: spec for spec in ENHANCERS}
KEYS: List[str] = [spec.key for spec in ENHANCERS]
LABEL_TO_KEY: Dict[str, str] = {spec.label: spec.key for spec in ENHANCERS}
NAME_TO_KEY: Dict[str, str] = {spec.name: spec.key for spec in ENHANCERS}
