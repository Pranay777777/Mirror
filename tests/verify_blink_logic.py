import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add parent directory to path to import features
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.temporal_features import TemporalFeatures

class TestBlinkLogic(unittest.TestCase):
    def setUp(self):
        self.temporal = TemporalFeatures(fps=30.0)
        # Mock blink parameters to match default or expected
        self.temporal.blink_z_threshold = -2.0
        self.temporal.blink_min_duration_ms = 50.0  # Allow short for testing
        self.temporal.blink_max_duration_ms = 500.0

    def test_blink_full_recovery(self):
        """Test standard blink with full recovery (Z > -1.0)"""
        # Baseline = 1.0, std = 0.1
        # Use 0.2 for blink to ensure deep Z-score
        baseline = [1.0] * 10
        blink = [0.8, 0.5, 0.2, 0.2, 0.2, 0.5, 0.8, 1.0, 1.0]
        data = baseline + blink + baseline
        
        ear_series = pd.Series(data)
        count, events, conf = self.temporal._detect_blinks_zscore(ear_series)
        
        # With 0.2 vs 1.0, Z-score should be very low (well below -2.5)
        self.assertEqual(count, 1)
        self.assertTrue(conf > 0.5)

    def test_blink_squint_recovery(self):
        """Test blink that recovers only to squint level (Z ~ -1.5) but stabilizes."""
        # Baseline 10.0, use deep dip (2.0)
        # Recover to 8.5 (Squint) and stay there
        
        baseline = [10.0] * 20
        dip = [2.0] * 5
        squint = [8.5] * 20 
        
        data = baseline + dip + squint
        ear_series = pd.Series(data)
        
        count, events, conf = self.temporal._detect_blinks_zscore(ear_series)
        
        self.assertEqual(count, 1, f"Failed to detect squint-recovery blink.")
        
    def test_noise_rejection(self):
        """Test that shallow dips (Z > -2.5) are rejected."""
        # Add some random noise to baseline to create stable std
        np.random.seed(42)
        baseline = list(np.random.normal(10.0, 0.1, 50))
        
        # Thresholds: Mean ~10, Std ~0.1
        # Z = -2.0 => Val = 9.8
        # Z = -2.5 => Val = 9.75
        # Dip to 9.78 (Z ~ -2.2)
        
        dip = [9.78] * 3
        data = baseline[:20] + dip + baseline[20:]
        ear_series = pd.Series(data)
        
        count, events, conf = self.temporal._detect_blinks_zscore(ear_series)
        
        self.assertEqual(count, 0, "Should reject shallow noise blink")

if __name__ == '__main__':
    unittest.main()
