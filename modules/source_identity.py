"""One identity from several photos of the same person.

The swap models see the source only as a 512-d ArcFace embedding, so extra
photos cannot teach them what a profile looks like — but they do make the
identity steadier (one photo's lighting or expression stops leaking into
every frame) and let the embedding lean towards the photo taken at a pose
like the target's, which measurably improves turned faces.

``SourceIdentity`` holds the analysed face of every accepted photo with its
pose proxies; ``face_for(target)`` returns a ``Face`` whose embedding is the
pose-weighted blend, so the swappers stay unchanged.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

import modules.globals
from modules import imread_unicode
from modules.calibration import pose_proxies
from modules.face_analyser import get_one_face

# Width of the pose kernel in pose-proxy units (same scale as the
# calibration references) and the share every photo keeps regardless of pose.
POSE_SCALE = 0.15
BASE_WEIGHT = 0.3


class SourceIdentity:
    def __init__(self, faces: Sequence[Any], paths: Sequence[str]) -> None:
        self.faces = list(faces)
        self.paths = list(paths)
        self._poses = np.asarray([pose_proxies(f.kps) for f in self.faces], dtype=np.float32)
        self._embeddings = np.stack([np.asarray(f.embedding, dtype=np.float32) for f in self.faces])
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        self._normed = self._embeddings / np.maximum(norms, 1e-6)
        self._scale = float(norms.mean())
        self._cache_key: Optional[Tuple[int, int]] = None
        self._cache: Optional[Any] = None

    def __len__(self) -> int:
        return len(self.faces)

    def weights(self, target_kps: Any) -> np.ndarray:
        """Per-photo weights: a floor shared by all plus a Gaussian in pose distance."""
        n = len(self.faces)
        if n == 1:
            return np.ones(1, dtype=np.float32)
        pose = np.asarray(pose_proxies(target_kps), dtype=np.float32)
        d2 = ((self._poses - pose) ** 2).sum(axis=1)
        w = BASE_WEIGHT / n + (1.0 - BASE_WEIGHT) * np.exp(-d2 / (2 * POSE_SCALE ** 2))
        return (w / w.sum()).astype(np.float32)

    def face_for(self, target_face: Any) -> Any:
        """A ``Face`` carrying the embedding blended for the target's pose."""
        if len(self.faces) == 1 or target_face is None or getattr(target_face, "kps", None) is None:
            return self.faces[0]
        # Pose proxies move slowly; quantising them makes the blend free
        # for runs of similar frames.
        pose = pose_proxies(target_face.kps)
        key = (int(round(pose[0] * 40)), int(round(pose[1] * 40)))
        if key == self._cache_key and self._cache is not None:
            return self._cache
        w = self.weights(target_face.kps)
        blended = (w[:, None] * self._normed).sum(axis=0)
        blended = blended / max(float(np.linalg.norm(blended)), 1e-6) * self._scale
        face = type(self.faces[0])(dict(self.faces[int(np.argmax(w))]))
        face.embedding = blended.astype(np.float32)
        self._cache_key, self._cache = key, face
        return face


def current_paths() -> List[str]:
    paths = list(getattr(modules.globals, "source_paths", None) or [])
    if not paths and modules.globals.source_path:
        paths = [modules.globals.source_path]
    return paths


def set_paths(paths: Sequence[str]) -> None:
    """Adopt ``paths`` as the source photos; ``source_path`` mirrors the first
    because the rest of the code checks it for "is a source selected"."""
    modules.globals.source_paths = list(paths)
    modules.globals.source_path = paths[0] if paths else None


def load(paths: Optional[Sequence[str]] = None) -> Tuple[Optional[SourceIdentity], List[str]]:
    """Analyse the photos; returns the identity (or ``None``) and the paths
    that were skipped because they could not be read or show no face."""
    paths = list(paths) if paths is not None else current_paths()
    faces, kept, skipped = [], [], []
    for path in paths:
        image = imread_unicode(path) if path and os.path.exists(path) else None
        face = None
        if image is not None:
            try:
                face = get_one_face(image)
            except Exception as error:
                print(f"[source] could not analyse {path}: {error}")
        if face is None or getattr(face, "embedding", None) is None:
            skipped.append(path)
            continue
        faces.append(face)
        kept.append(path)
    if not faces:
        return None, skipped
    return SourceIdentity(faces, kept), skipped
