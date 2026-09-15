import unittest

import numpy as np

import modules.globals
from modules.head_geometry import HeadPose, _ramp, pose_alpha


class _Face(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value


def _pose(yaw, pitch):
    pts = np.zeros((68, 3), dtype=np.float32)
    pts[:, 0] = np.linspace(0, 100, 68)
    pts[:17, 1] = 100
    pts[17:27, 1] = 40
    return HeadPose(yaw=yaw, pitch=pitch, roll=0.0, landmarks=pts)


class HeadGeometryTest(unittest.TestCase):
    def setUp(self):
        modules.globals.head_pose = True
        modules.globals.head_yaw_limit = 50
        modules.globals.head_pitch_limit = 30

    def test_ramp_is_full_within_limit_and_zero_past_the_span(self):
        self.assertEqual(_ramp(30, 50), 1.0)
        self.assertEqual(_ramp(-50, 50), 1.0)
        self.assertEqual(_ramp(65, 50), 0.0)
        self.assertAlmostEqual(_ramp(57.5, 50), 0.5, places=5)

    def test_pose_alpha_combines_yaw_and_pitch(self):
        face = _Face(head_pose=_pose(0, 0))
        self.assertEqual(pose_alpha(face), 1.0)
        face.head_pose = _pose(57.5, 0)
        self.assertAlmostEqual(pose_alpha(face), 0.5, places=5)
        face.head_pose = _pose(57.5, 34.5)
        self.assertAlmostEqual(pose_alpha(face), 0.25, places=5)

    def test_pose_alpha_is_one_without_a_pose_or_when_off(self):
        self.assertEqual(pose_alpha(_Face()), 1.0)
        modules.globals.head_pose = False
        self.assertEqual(pose_alpha(_Face(head_pose=_pose(80, 0))), 1.0)

    def test_outline_adds_a_forehead_above_the_brows(self):
        outline = _pose(0, 0).outline()
        self.assertEqual(outline.shape, (27, 2))
        self.assertLess(outline[17:, 1].min(), 40)

    def test_moved_carries_landmarks_by_the_similarity_and_keeps_angles(self):
        import cv2
        from modules.head_geometry import HeadPose
        pts = np.zeros((68, 3), dtype=np.float32); pts[:, 0] = np.arange(68); pts[:, 1] = 10; pts[:, 2] = 3
        pose = HeadPose(yaw=5.0, pitch=-2.0, roll=1.0, landmarks=pts)
        affine = cv2.getRotationMatrix2D((0.0, 0.0), 90, 2.0)
        moved = pose.moved(affine)
        self.assertEqual((moved.yaw, moved.pitch, moved.roll), (5.0, -2.0, 1.0))
        np.testing.assert_allclose(moved.landmarks[:, 2], 3)
        expect = cv2.transform(pts[:, :2].reshape(1, -1, 2), affine).reshape(-1, 2)
        np.testing.assert_allclose(moved.landmarks[:, :2], expect, atol=1e-4)
        np.testing.assert_allclose(pose.landmarks[:, 0], np.arange(68))


if __name__ == "__main__":
    unittest.main()
