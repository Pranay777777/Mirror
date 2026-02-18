import unittest
from unittest.mock import MagicMock
import sys
import os

# Adjust path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules to avoid import errors
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.solutions'] = MagicMock()
sys.modules['mediapipe.solutions.holistic'] = MagicMock()
sys.modules['mediapipe.solutions.face_mesh'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['librosa'] = MagicMock()

# Import the function to be tested
from utils.video_utils import _generate_natural_posture_summary, _interpret_posture

class TestNaturalPostureSummary(unittest.TestCase):
    
    def test_excellent_posture(self):
        # Alignment >= 0.90, Stability >= 0.85, Movement < 0.05
        # Exp: "Excellent posture. Your body alignment is well balanced and very stable with minimal movement."
        summary = _generate_natural_posture_summary(0.96, 0.95, 0.01)
        # Check actual phrases
        self.assertIn("Excellent posture", summary)
        self.assertIn("well balanced", summary)
        self.assertIn("very stable", summary)
        self.assertIn("minimal movement", summary)

    def test_good_posture_misaligned(self):
        # Alignment=0.75 (Moderate misalignment), Stability=0.88 (Stable > 0.75), Movement=0.04 (Minimal)
        # Expected: Needs Improvement (due to 0.75 align < 0.80 for Good).
        # "Needs improvement. You show moderate misalignment and generally stable with minimal movement."
        summary = _generate_natural_posture_summary(0.75, 0.88, 0.04)
        self.assertIn("Needs improvement", summary)
        self.assertIn("moderate misalignment", summary.lower())
        self.assertIn("generally stable", summary.lower())

    def test_poor_posture(self):
        # Alignment=0.50, Stability=0.60, Movement=0.30
        # Overall: Poor (< 0.65)
        # "Poor posture. Significant misalignment is present with mild instability with excessive movement."
        # Stability 0.60 -> "mild instability"
        summary = _generate_natural_posture_summary(0.50, 0.60, 0.30)
        print(f"DEBUG Poor: {summary}")
        self.assertIn("Poor posture", summary)
        self.assertIn("significant misalignment", summary.lower())
        self.assertIn("mild instability", summary.lower()) 
        self.assertIn("excessive movement", summary.lower())

    def test_unstable_posture(self):
        # Stability < 0.60 -> Unstable
        summary = _generate_natural_posture_summary(0.85, 0.55, 0.10)
        # Alignment 0.85 (Slight misalignment? NO, 0.80-0.89 is Slight) -> "There is slight misalignment"
        self.assertIn("unstable", summary.lower())

if __name__ == '__main__':
    unittest.main()
