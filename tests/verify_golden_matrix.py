import unittest
import sys
import os

# Adjust path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules to allow import of video_utils
from unittest.mock import MagicMock
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.solutions'] = MagicMock()
sys.modules['mediapipe.solutions.holistic'] = MagicMock()
sys.modules['mediapipe.solutions.face_mesh'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['librosa'] = MagicMock()

from utils.video_utils import _generate_deterministic_posture_summary, POSTURE_ENGINE_VERSION

class TestGoldenValidationMatrix(unittest.TestCase):
    
    def test_version_lock(self):
        print(f"Checking Version Lock: {POSTURE_ENGINE_VERSION}")
        self.assertEqual(POSTURE_ENGINE_VERSION, "v2.2_frozen_calibrated")

    def test_pos1_ideal(self):
        # >0.95 Align, <0.02 Motion, >0.95 Stability
        summary = _generate_deterministic_posture_summary(0.96, 0.96, 0.01)
        expected = "Excellent body alignment with very high stability and minimal movement."
        self.assertEqual(summary, expected)

    def test_pos2_slight_lean(self):
        # >0.95 Align, 0.02-0.05 Motion, 0.90-0.95 Stability
        summary = _generate_deterministic_posture_summary(0.96, 0.92, 0.03)
        # 0.92 Stability -> Very High (>=0.90)
        # 0.03 Motion -> Minimal (<0.04)
        expected = "Excellent body alignment with very high stability and minimal movement."
        self.assertEqual(summary, expected)

    def test_pos2left_severe_tilt(self):
        # <0.30 Align, low Motion, <0.40 Stability
        summary = _generate_deterministic_posture_summary(0.25, 0.35, 0.01)
        # <0.60 Align -> Significant misalignment
        # <0.60 Stab -> Low
        # <0.04 Motion -> Minimal
        expected = "Significant body alignment with low stability and minimal movement."
        self.assertEqual(summary, expected)

    def test_pos2right_moderate_tilt(self):
        # 0.60-0.80 Align, low Motion, 0.60-0.75 Stability
        summary = _generate_deterministic_posture_summary(0.70, 0.72, 0.01)
        # 0.70 Align -> Moderate
        # 0.72 Stab -> Moderate (0.70-0.79)
        # <0.04 Motion -> Minimal
        expected = "Moderate body alignment with moderate stability and minimal movement."
        self.assertEqual(summary, expected)
        
    def test_poss3_head_movement(self):
        # >0.90 Align, 0.08-0.15 Motion, 0.75-0.85 Stability
        summary = _generate_deterministic_posture_summary(0.92, 0.82, 0.12)
        # 0.92 Align -> Good (0.80-0.94)
        # 0.82 Stab -> High (0.80-0.89)
        # 0.12 Motion -> Noticeable (0.10-0.20)
        expected = "Good body alignment with high stability and noticeable movement."
        self.assertEqual(summary, expected)

    def test_poss5_fidgeting(self):
        # >0.90 Align, >0.25 Motion, <0.40 Stability
        summary = _generate_deterministic_posture_summary(0.92, 0.30, 0.26)
        # 0.92 Align -> Good
        # 0.30 Stab -> Low
        # 0.26 Motion -> Excessive (>0.20)
        expected = "Good body alignment with low stability and excessive movement."
        self.assertEqual(summary, expected)

    def test_poss6_tilt_motion(self):
        # 0.60-0.80 Align, 0.10-0.20 Motion, 0.50-0.65 Stability
        summary = _generate_deterministic_posture_summary(0.70, 0.62, 0.15)
        # 0.70 Align -> Moderate
        # 0.62 Stab -> Reduced (0.60-0.69)
        # 0.15 Motion -> Noticeable
        expected = "Moderate body alignment with reduced stability and noticeable movement."
        self.assertEqual(summary, expected)

if __name__ == '__main__':
    unittest.main()
