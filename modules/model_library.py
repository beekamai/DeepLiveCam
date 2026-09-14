"""Catalogue of every model the app can use, with download state.

Every model already downloads itself on first use, which stalls the first
Live for a minute per model.  The catalogue lets the user see what is on
disk, fetch models ahead of time (or all of them), and know each one's
size before committing bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from modules import enhancer_registry, swapper_registry
from modules.model_downloader import (
    MODEL_SIZES,
    ensure_insightface_pack,
    ensure_model,
    expected_size,
    is_present,
)
from modules.face_occluder import MODEL_FILE as OCCLUDER_FILE

INSIGHTFACE_PACK = "buffalo_l"


@dataclass
class Entry:
    file: str        # file name under models/ (or the pack name)
    label: str
    kind: str        # "swapper" | "enhancer" | "mask" | "detector"
    size: Optional[int]
    present: bool
    used_by: str = ""

    @property
    def size_text(self) -> str:
        if not self.size:
            return "?"
        mb = self.size / (1024 * 1024)
        return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def _pack_present() -> bool:
    import os

    dest = os.path.join(os.path.expanduser("~"), ".insightface", "models", INSIGHTFACE_PACK)
    members = [n for n in MODEL_SIZES if n.startswith(f"{INSIGHTFACE_PACK}/")]
    return bool(members) and all(is_present(m, dest) for m in members)


def catalog() -> List[Entry]:
    entries: List[Entry] = []
    pack_size = sum(v for k, v in MODEL_SIZES.items() if k.startswith(f"{INSIGHTFACE_PACK}/"))
    entries.append(Entry(INSIGHTFACE_PACK, "insightface buffalo_l (detector, landmarks, ArcFace)",
                         "detector", pack_size, _pack_present(), "everything"))
    for spec in swapper_registry.SWAPPERS:
        entries.append(Entry(spec.model_file, spec.label, "swapper",
                             expected_size(spec.model_file), is_present(spec.model_file),
                             "Face Swapper"))
    for spec in enhancer_registry.ENHANCERS:
        entries.append(Entry(spec.model_file, spec.label, "enhancer",
                             expected_size(spec.model_file), is_present(spec.model_file),
                             "Face Enhancer"))
    entries.append(Entry(OCCLUDER_FILE, "XSeg occluder", "mask",
                         expected_size(OCCLUDER_FILE), is_present(OCCLUDER_FILE), "Occlusion mask"))
    return entries


def download(entry: Entry, status: Optional[Callable[[str], None]] = None) -> bool:
    """Fetch one entry (blocking); returns True when it is on disk."""
    if status:
        status(f"Downloading {entry.label} ({entry.size_text})...")
    if entry.kind == "detector":
        ok = ensure_insightface_pack(entry.file)
    else:
        ok = ensure_model(entry.file) is not None
    entry.present = ok
    return ok
