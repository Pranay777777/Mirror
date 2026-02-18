import unittest
from unittest.mock import MagicMock
import sys
import os

# Adjust path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.solutions'] = MagicMock()
sys.modules['mediapipe.solutions.holistic'] = MagicMock()
sys.modules['mediapipe.solutions.face_mesh'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['librosa'] = MagicMock()

# Import the function to be tested
from utils.video_utils import _generate_deterministic_posture_summary

class TestDeterministicPostureSummary(unittest.TestCase):
    
    def test_ideal_case(self):
        # Alignment >= 0.95 -> Excellent (Capitalized)
        # Stability >= 0.90 -> very high (lowercase)
        # Movement < 0.04 -> minimal (lowercase)
        summary = _generate_deterministic_posture_summary(0.96, 0.95, 0.01)
        expected = "Excellent body alignment with very high stability and minimal movement."
        self.assertEqual(summary, expected)

    def test_good_case(self):
        # Alignment 0.80-0.94 -> Good
        # Stability 0.80-0.89 -> high
        # Movement 0.04-0.10 -> mild
        summary = _generate_deterministic_posture_summary(0.85, 0.85, 0.05)
        expected = "Good body alignment with high stability and mild movement."
        self.assertEqual(summary, expected)

    def test_moderate_case(self):
        # Alignment 0.60-0.79 -> Moderate
        # Stability 0.70-0.79 -> moderate
        # Movement 0.10-0.20 -> noticeable
        summary = _generate_deterministic_posture_summary(0.70, 0.75, 0.15)
        expected = "Moderate body alignment with moderate stability and noticeable movement."
        self.assertEqual(summary, expected)

    def test_reduced_case(self):
        # Alignment < 0.60 -> Significant
        # Stability 0.60-0.69 -> reduced
        # Movement > 0.20 -> excessive
        summary = _generate_deterministic_posture_summary(0.50, 0.65, 0.25)
        expected = "Significant body alignment with reduced stability and excessive movement."
        self.assertEqual(summary, expected)
        
    def test_low_case(self):
        # Stability < 0.60 -> low
        summary = _generate_deterministic_posture_summary(0.50, 0.50, 0.30)
        expected = "Significant body alignment with low stability and excessive movement."
        self.assertEqual(summary, expected)

if __name__ == '__main__':
    unittest.main()
