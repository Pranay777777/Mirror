import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.temporal_features import TemporalFeatures

class TestRefactoredGaze(unittest.TestCase):
    """
    Verify Refactored Eye Contact Logic (Ratio & Direction States).
    """

    def setUp(self):
        self.tf = TemporalFeatures(fps=30.0, gaze_alignment_threshold_deg=6.0)

    def test_steady_off_camera_gaze(self):
        """
        Steady gaze at 30 degrees (off-camera) should have:
        - Low consistency (ratio ~ 0.0)
        - High confidence (signal is stable)
        - 0 switches (steady state)
        """
        # 100 frames of steady 30 deg angle, 30 deg yaw (Left)
        gaze_angles = pd.Series([30.0] * 100)
        gaze_yaws = pd.Series([30.0] * 100) # Positive = Left
        
        ratio, duration, conf, switches, max_dur, diag = self.tf._compute_eye_contact(gaze_angles, gaze_yaws)
        
        print(f"\nSteady Off-Camera: Ratio={ratio:.2f}, Conf={conf:.2f}, Switches={switches}")
        
        self.assertLess(ratio, 0.01, "Frames > threshold should have very low ratio.")
        self.assertGreater(conf, 0.8, "Steady signal should still have high confidence.")
        self.assertEqual(switches, 0, "No switches in steady state.")

    def test_camera_gaze(self):
        """
        Steady gaze at 5 degrees (on-camera) should have:
        - High consistency (ratio ~ 1.0)
        """
        gaze_angles = pd.Series([5.0] * 100)
        gaze_yaws = pd.Series([5.0] * 100) # Within threshold
        
        ratio, duration, conf, switches, max_dur, diag = self.tf._compute_eye_contact(gaze_angles, gaze_yaws)
        
        print(f"Steady Camera: Ratio={ratio:.2f}, Conf={conf:.2f}, Switches={switches}")
        # With Gaussian (sigma=8.0), 5 degrees -> prob ~ 0.82
        self.assertGreater(ratio, 0.80, "5 degrees should have high probability (>0.80)")
        self.assertLess(ratio, 0.90, "5 degrees is not perfect 1.0 anymore")

    def test_direction_switching(self):
        """
        Switching: Camera -> Left -> Camera -> Right
        Should detect transitions.
        """
        # 30 frames Camera (0 deg)
        # 30 frames Left (30 deg angle, 30 deg yaw)
        # 30 frames Camera (0 deg)
        # 30 frames Right (30 deg angle, -30 deg yaw)
        
        angles = [0.0]*30 + [30.0]*30 + [0.0]*30 + [30.0]*30
        yaws =   [0.0]*30 + [30.0]*30 + [0.0]*30 + [-30.0]*30
        
        gaze_angles = pd.Series(angles)
        gaze_yaws = pd.Series(yaws)
        
        ratio, duration, conf, switches, max_dur, diag = self.tf._compute_eye_contact(gaze_angles, gaze_yaws)
        
        # Transitions: 
        # Cam -> Left (1)
        # Left -> Cam (1)
        # Cam -> Right (1)
        # Total = 3
        
        print(f"Switching Test: Switches={switches}")
        self.assertEqual(switches, 3, "Should detect 3 state transitions.")

if __name__ == '__main__':
    unittest.main()
