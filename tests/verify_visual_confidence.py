import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.temporal_features import TemporalFeatures

class TestVisualConfidence(unittest.TestCase):
    """
    Verify Refined Visual Confidence Model.
    """

    def test_ideal_video_visual_confidence(self):
        """
        Verify that a stable, perfect input yields visual_confidence >= 0.90.
        """
        tf = TemporalFeatures(fps=30.0)
        
        # Simulate 100 frames (approx 3.3 seconds) of PERFECT stability
        # To get high confidence, we need:
        # 1. High Data Completeness (all frames valid)
        # 2. Stable Gaze (low sigma) -> expected_sigma=0.3
        # 3. Stable Posture (low sigma) -> expected_sigma=0.5
        # 4. Clean Blink Signal (low sigma) -> expected_sigma=0.2
        
        for i in range(100):
            timestamp = i / 30.0
            
            # Perfect Posture
            # inclination = 0 (perfectly upright), stable
            pose = {
                'torso_inclination_deg': 0.0 + (np.random.normal(0, 0.01)), # Tiny jitter < 0.5
                'shoulder_tilt_angle': 0.0,
                'shoulder_width_raw': 0.5,
                'velocity_shoulder_midpoint_x': 0.0,
                'velocity_shoulder_midpoint_y': 0.0,
                'eye_distance_ratio': 0.3 # stable
            }
            
            # Perfect Face
            # Gaze aligned (angle=0), stable
            # Eyes open (EAR=0.3), stable
            face = {
                'gaze_alignment_angle': 0.0 + (np.random.normal(0, 0.01)), # Tiny jitter < 0.3
                'left_eye_opening_ratio': 0.3 + (np.random.normal(0, 0.005)), # Tiny jitter < 0.2
                'right_eye_opening_ratio': 0.3 + (np.random.normal(0, 0.005)),
                'mouth_opening_ratio': 0.0,
                'smile_intensity': 0.0
            }
            
            tf.add_frame(pose, face, timestamp)
            
        results = tf.finalize()
        
        # Print Diagnostics
        vc = results.get('visual_confidence', {})
        print("\n--- Ideal Video Diagnostics ---")
        print(f"Visual Confidence: {vc.get('value')}")
        print(f"Confidence Breakdown: {vc.get('confidence', 'N/A')}") # Note: 'confidence' key in output is essentially the value itself for valid result? 
        # Wait, _format_result returns {'value': X, 'confidence': C ...}
        # In finalize(), steps 759:
        # results['visual_confidence'] = self._format_result(round(final_conf, 4), 1.0, c_data, ...)
        # So results['visual_confidence']['value'] IS the weighted average confidence?
        # Let's check line 759 in temporal_features.py.
        # Yes: results['visual_confidence'] = _format_result(value=final_conf, confidence=1.0, ...)
        
        final_score = vc.get('value')
        
        # Check sub-confidences if possible (printed in stdout by code, but we can't access easily in test object unless we peek)
        # We can infer from the score.
        
        self.assertGreaterEqual(final_score, 0.90, 
                                f"Visual Confidence {final_score} should be >= 0.90 for perfect input.")
        print("SUCCESS: Ideal Visual Confidence >= 0.90")

if __name__ == '__main__':
    unittest.main()
