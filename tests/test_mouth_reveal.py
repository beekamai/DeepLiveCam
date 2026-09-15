import unittest

import numpy as np

from modules.face_reveal import (
    INNER_BOTTOM,
    INNER_TOP,
    LOWER_OUTER,
    MOUTH,
    NOSE,
    OUTER_LIP,
    mouth_polygon,
    mouth_reveal_mask,
)


class _Face(dict):
    def __getattr__(self, name):
        return self.get(name)


def _face(open_gap: float = 0.0):
    """Lips: outer contour an ellipse 36 wide / 12 tall around (60, 84),
    inner contour a flatter ellipse; ``open_gap`` parts the inner lips."""
    pts = np.zeros((106, 2), dtype=np.float32)
    pts[:, 0] = 60
    pts[0] = (60, 118)                                   # chin
    for k, i in enumerate(NOSE):
        pts[i] = (50 + k, 60)                            # nose base row
    for k, i in enumerate(OUTER_LIP):
        a = 2 * np.pi * k / len(OUTER_LIP)
        pts[i] = (60 - 18 * np.cos(a), 84 + 6 * np.sin(a))
    inner = (65, 54, 60, 57, 69, 70, 62, 66)
    for k, i in enumerate(inner):
        a = 2 * np.pi * k / len(inner)
        pts[i] = (60 - 12 * np.cos(a), 84 + 2 * np.sin(a))
    pts[INNER_TOP] = (60, 84 - open_gap / 2)
    pts[INNER_BOTTOM] = (60, 84 + open_gap / 2)
    return _Face(landmark_2d_106=pts)


class MouthRevealTest(unittest.TestCase):
    def setUp(self):
        self.affine = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    def test_region_reaches_the_nose_at_full_strength(self):
        poly = mouth_polygon(_face(), self.affine, 1.0, "region")
        self.assertLess(poly[:, 1].min(), 70)

    def test_lips_mode_is_the_outer_contour_in_order(self):
        poly = mouth_polygon(_face(), self.affine, 0.0, "lips")
        self.assertEqual(poly.shape, (len(OUTER_LIP), 2))
        expected = _face().landmark_2d_106[list(OUTER_LIP)]
        np.testing.assert_allclose(poly, expected, atol=1e-4)

    def test_lips_mode_never_rises_above_the_upper_lip(self):
        top = _face().landmark_2d_106[list(MOUTH)][:, 1].min()
        for strength in (0.0, 0.5, 1.0):
            poly = mouth_polygon(_face(), self.affine, strength, "lips")
            self.assertGreaterEqual(poly[:, 1].min(), top - 1.0)

    def test_open_mouth_extends_below_the_lower_lip_for_a_tongue(self):
        closed = mouth_polygon(_face(0.0), self.affine, 0.5, "lips")
        opened = mouth_polygon(_face(12.0), self.affine, 0.5, "lips")
        self.assertGreater(opened[:, 1].max(), closed[:, 1].max() + 5)
        # only the lower lip moved; the upper edge stays put
        self.assertAlmostEqual(float(opened[:, 1].min()), float(closed[:, 1].min()), places=3)
        lower = [OUTER_LIP.index(i) for i in LOWER_OUTER]
        self.assertTrue(all(opened[j, 1] > closed[j, 1] for j in lower))

    def test_lips_mode_blends_by_strength_and_region_does_not(self):
        half = mouth_reveal_mask(_face(), self.affine, 128, 0.5, "lips")[0]
        full = mouth_reveal_mask(_face(), self.affine, 128, 1.0, "lips")[0]
        region = mouth_reveal_mask(_face(), self.affine, 128, 0.5, "region")[0]
        self.assertAlmostEqual(float(half.min()), 0.5, places=2)
        self.assertAlmostEqual(float(full.min()), 0.0, places=2)
        self.assertAlmostEqual(float(region.min()), 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
