import unittest

import numpy as np

from modules.source_identity import SourceIdentity


class _Face(dict):
    """Minimal stand-in for insightface's Face (dict with attribute access)."""

    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value


def _kps(yaw: float) -> np.ndarray:
    # eyes, nose, mouth corners; the nose slides sideways with yaw
    return np.array([[40, 40], [80, 40], [60 + 20 * yaw, 60], [45, 80], [75, 80]], dtype=np.float32)


def _face(embedding, yaw):
    return _Face(embedding=np.asarray(embedding, dtype=np.float32), kps=_kps(yaw))


class SourceIdentityTest(unittest.TestCase):
    def test_single_photo_is_passed_through(self):
        face = _face([1, 0, 0], 0.0)
        identity = SourceIdentity([face], ["a"])
        self.assertIs(identity.face_for(_face([0, 1, 0], 0.5)), face)

    def test_blend_leans_to_the_photo_at_the_target_pose(self):
        front = _face([1, 0, 0], 0.0)
        turned = _face([0, 1, 0], 1.0)
        identity = SourceIdentity([front, turned], ["front", "turned"])
        w = identity.weights(_kps(1.0))
        self.assertGreater(w[1], w[0])
        self.assertGreater(w[0], 0.05)          # every photo keeps a share
        blended = identity.face_for(_face(None, 1.0)).embedding
        self.assertGreater(blended[1], blended[0])
        self.assertAlmostEqual(float(np.linalg.norm(blended)), 1.0, places=5)

    def test_frontal_target_gets_the_frontal_photo(self):
        front = _face([1, 0, 0], 0.0)
        turned = _face([0, 1, 0], 1.0)
        identity = SourceIdentity([front, turned], ["front", "turned"])
        blended = identity.face_for(_face(None, 0.0)).embedding
        self.assertGreater(blended[0], blended[1])


if __name__ == "__main__":
    unittest.main()
