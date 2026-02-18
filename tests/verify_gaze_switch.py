import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.temporal_features import TemporalFeatures

class TestGazeSwitch(unittest.TestCase):
    """
    Verify Refined Gaze Switch Detection Logic.
    """

    def test_micro_jitter_ignored(self):
        """
        Verify that micro-jitters (<8 deg) are NOT counted as switches.
        """
        tf = TemporalFeatures(fps=30.0)
        
        # Create a signal that jitters around 0 degrees, but never exceeds +/- 4 deg
        # 100 frames
        gaze_angles = []
        for i in range(100):
            # Jitter between -3 and +3
            noise = np.random.uniform(-3.0, 3.0)
            gaze_angles.append(noise)
            
        gaze_series = pd.Series(gaze_angles)
        
        # Access the private method for testing logic directly? 
        # Or wrap it in a dummy call.
        # _compute_eye_contact takes a series.
        
        score, duration, conf, switches = tf._compute_eye_contact(gaze_series)
        
        print(f"\nMicro-Jitter Test: Switches Detected = {switches}")
        self.assertEqual(switches, 0, "Micro-jitters should not trigger switch detection.")

    def test_valid_switch_detected(self):
        """
        Verify that a 15 degree shift persisting for 10 frames IS counted.
        """
        tf = TemporalFeatures(fps=30.0)
        
        # 30 frames at 0 deg, then jump to 20 deg for 30 frames
        gaze_angles = [0.0] * 30 + [20.0] * 30
        gaze_series = pd.Series(gaze_angles)
        
        score, duration, conf, switches = tf._compute_eye_contact(gaze_series)
        
        print(f"Valid Switch Test: Switches Detected = {switches}")
        self.assertEqual(switches, 1, "Should detect exactly 1 significant switch.")
        
    def test_rapid_flicker_ignored(self):
        """
        Verify that a 20 degree shift that lasts only 2 frames is IGNORED (persistence check).
        """
        tf = TemporalFeatures(fps=30.0)
        
        # 0 deg normally, but flicker to 20 deg for 2 frames
        gaze_angles = [0.0] * 30 + [20.0, 20.0] + [0.0] * 30
        gaze_series = pd.Series(gaze_angles)
        
        score, duration, conf, switches = tf._compute_eye_contact(gaze_series)
        
        print(f"Rapid Flicker Test: Switches Detected = {switches}")
        self.assertEqual(switches, 0, "Short duration flickering should be ignored.")

if __name__ == '__main__':
    unittest.main()
