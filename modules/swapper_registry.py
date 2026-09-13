"""Available face swapper models.

``inswapper_128`` runs through insightface's own INSwapper wrapper; every other
entry is driven by :class:`modules.processors.frame.swapper_models.OnnxSwapper`,
which exposes the same ``get()`` API so the swap pipeline stays unchanged.

DeepFaceLive ``.dfm`` files dropped into ``models/dfm`` are picked up
automatically: they carry one baked-in identity (no source photo), which is why
they lock onto a face far better than the one-shot models, and they output
their own face mask.
"""

import os
from typing import Dict, List, NamedTuple


class SwapperSpec(NamedTuple):
    key: str
    label: str
    model_file: str
    input_size: int
    kind: str            # 'inswapper' | 'hyperswap' | 'alphaface'
    # Normalisation of the target crop before inference.
    mean: float
    std: float
    # Whether the model output needs de-normalising with the same mean/std.
    denormalize: bool
    # Source vector: 'normed' (unit-length arcface embedding), 'raw', or
    # 'none' for models with a baked-in identity.
    embedding: str
    # Alignment template; DFM models want DeepFaceLab's whole-face crop.
    template: str = "arcface_128"
    # Relative path under models/, when the file does not sit at its root.
    subdir: str = ""
    # CUDA graph capture needs every node on the CUDA provider; models that
    # keep nodes on CPU fail the capture, so don't pay for a doomed attempt.
    cuda_graph: bool = True


SWAPPERS: List[SwapperSpec] = [
    SwapperSpec(
        key="inswapper_128",
        label="Inswapper-128",
        model_file="inswapper_128_fp16.onnx",
        input_size=128,
        kind="inswapper",
        mean=0.0,
        std=1.0,
        denormalize=False,
        embedding="raw",
    ),
    SwapperSpec(
        key="hyperswap_1a_256",
        label="HyperSwap-1A-256",
        model_file="hyperswap_1a_256.onnx",
        input_size=256,
        kind="hyperswap",
        mean=0.5,
        std=0.5,
        denormalize=True,
        embedding="normed",
    ),
    SwapperSpec(
        key="hyperswap_1b_256",
        label="HyperSwap-1B-256",
        model_file="hyperswap_1b_256.onnx",
        input_size=256,
        kind="hyperswap",
        mean=0.5,
        std=0.5,
        denormalize=True,
        embedding="normed",
    ),
    SwapperSpec(
        key="hyperswap_1c_256",
        label="HyperSwap-1C-256",
        model_file="hyperswap_1c_256.onnx",
        input_size=256,
        kind="hyperswap",
        mean=0.5,
        std=0.5,
        denormalize=True,
        embedding="normed",
    ),
    SwapperSpec(
        key="alphaface_256",
        label="AlphaFace-256",
        model_file="alphaface_256.onnx",
        input_size=256,
        kind="alphaface",
        mean=0.0,
        std=1.0,
        denormalize=False,
        embedding="raw",
        cuda_graph=False,
    ),
]

DFM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "dfm",
)


def _dfm_size(file_name: str) -> int:
    """DeepFaceLive names carry the resolution as a suffix (name_320.dfm)."""
    stem = os.path.splitext(file_name)[0]
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def discover_dfm_models() -> List[SwapperSpec]:
    """One entry per .dfm file in models/dfm."""
    if not os.path.isdir(DFM_DIR):
        return []
    specs = []
    for file_name in sorted(os.listdir(DFM_DIR)):
        if not file_name.lower().endswith(".dfm"):
            continue
        stem = os.path.splitext(file_name)[0]
        specs.append(SwapperSpec(
            key=f"dfm_{stem}",
            label=f"DFM: {stem}",
            model_file=file_name,
            # Size is read from the model on load; the name is only a hint.
            input_size=_dfm_size(file_name) or 224,
            kind="dfm",
            mean=0.0,
            std=1.0,
            denormalize=False,
            embedding="none",
            template="dfl_whole_face",
            subdir="dfm",
        ))
    return specs


SWAPPERS.extend(discover_dfm_models())

BY_KEY: Dict[str, SwapperSpec] = {spec.key: spec for spec in SWAPPERS}
KEYS: List[str] = [spec.key for spec in SWAPPERS]
LABEL_TO_KEY: Dict[str, str] = {spec.label: spec.key for spec in SWAPPERS}
DEFAULT_KEY = "inswapper_128"
