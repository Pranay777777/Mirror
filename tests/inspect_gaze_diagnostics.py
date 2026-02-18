import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.temporal_features import TemporalFeatures

class InspectGazeDiagnostics(unittest.TestCase):
    """
    Inspect the new structured diagnostics output.
    """

    def setUp(self):
        self.tf = TemporalFeatures(fps=30.0, gaze_alignment_threshold_deg=10.0)

    def test_simulated_diagnostics(self):
        """
        Simulate a sequence and check diagnostics keys.
        """
        # 10 frames of data
        # 5 frames camera (0 deg)
        # 5 frames left (30 deg)
        angles = [0.0]*5 + [30.0]*5
        yaws =   [0.0]*5 + [30.0]*5
        
        gaze_angles = pd.Series(angles)
        gaze_yaws = pd.Series(yaws)
        
        ratio, duration, conf, switches, max_dur, diag = self.tf._compute_eye_contact(gaze_angles, gaze_yaws)
        
        print("\n--- Diagnostics Inspection ---")
        print(f"Keys: {diag.keys()}")
        print(f"Summary: {diag['summary']}")
        print(f"Trace (first 2 frames): {diag['frame_trace'][:2]}")
        
        # Validation
        self.assertIn("frame_trace", diag)
        self.assertIn("summary", diag)
        
        summary = diag["summary"]
        self.assertIn("mean_gaze_probability", summary)
        self.assertIn("mean_head_weight", summary)
        self.assertIn("mean_final_probability", summary)
        self.assertEqual(summary["logic_mode"], "gaussian_fusion")
        
        # Check trace content
        trace = diag["frame_trace"]
        self.assertEqual(len(trace), 10)
        self.assertTrue(0.0 <= trace[0]["gaze_probability"] <= 1.0)
        self.assertTrue(0.0 <= trace[0]["final_contact_probability"] <= 1.0)
        
        # 0 deg gaze -> close to 1.0
        self.assertGreater(trace[0]["gaze_probability"], 0.99)
        
        # 30 deg gaze -> close to 0.0
        self.assertLess(trace[-1]["gaze_probability"], 0.01)

if __name__ == '__main__':
    unittest.main()
