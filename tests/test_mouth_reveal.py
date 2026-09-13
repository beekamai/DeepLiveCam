import unittest

import numpy as np

from modules.face_reveal import MOUTH, NOSE, mouth_polygon


class _Face(dict):
    def __getattr__(self, name):
        return self.get(name)


def _face():
    pts = np.zeros((106, 2), dtype=np.float32)
    pts[:, 0] = 60
    pts[0] = (60, 118)                         # chin
    for k, i in enumerate(NOSE):
        pts[i] = (50 + k, 60)                  # nose base row at y=60
    for k, i in enumerate(MOUTH):
        a = 2 * np.pi * k / len(MOUTH)
        pts[i] = (60 + 18 * np.cos(a), 84 + 6 * np.sin(a))   # lips ellipse, y in [78, 90]
    return _Face(landmark_2d_106=pts)


class MouthRevealTest(unittest.TestCase):
    def setUp(self):
        self.affine = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    def test_region_reaches_the_nose_at_full_strength(self):
        poly = mouth_polygon(_face(), self.affine, 1.0, "region")
        self.assertLess(poly[:, 1].min(), 70)

    def test_lips_mode_never_rises_above_the_upper_lip(self):
        poly = mouth_polygon(_face(), self.affine, 1.0, "lips")
        self.assertGreaterEqual(poly[:, 1].min(), 78 - 0.01)
        self.assertGreater(poly[:, 1].max(), 90)   # still grows down for a tongue

    def test_lips_mode_at_zero_is_the_lip_hull(self):
        poly = mouth_polygon(_face(), self.affine, 0.0, "lips")
        self.assertAlmostEqual(float(poly[:, 1].max()), 90, places=3)
        self.assertAlmostEqual(float(poly[:, 0].max()), 78, places=3)


if __name__ == "__main__":
    unittest.main()
