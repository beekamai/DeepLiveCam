"""GFPGAN-1.4 face enhancer — ONNX face restoration (512px, ffhq_512 alignment)."""

from modules.enhancer_registry import BY_KEY
from modules.processors.frame._enhancer_processor import OnnxEnhancer

SPEC = BY_KEY["face_enhancer_gfpgan14"]

ENHANCER = OnnxEnhancer(
    name=SPEC.name,
    model_file=SPEC.model_file,
    input_size=SPEC.input_size,
    template=SPEC.template,
    mirror_url=SPEC.mirror_url,
)

globals().update(ENHANCER.as_module_api())
