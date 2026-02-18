import unittest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import MagicMock, patch

# Mock dependencies
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.solutions'] = MagicMock()
sys.modules['mediapipe.solutions.holistic'] = MagicMock()
sys.modules['mediapipe.solutions.face_mesh'] = MagicMock()

# Import TemporalFeatures
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.temporal_features import TemporalFeatures

class TestPostureStabilityComposite(unittest.TestCase):
    def setUp(self):
        self.tf = TemporalFeatures(fps=30.0)

    def _add_frames(self, duration_sec, inclination_func, velocity_func, width=1.0):
        total_frames = int(duration_sec * 30)
        # Add 3s prefix (90 frames) of "noise" that gets filtered out?
        # Or just simulate valid frames directly if I want to test logic.
        # But implementation uses 3s cutoff. So I must provide >3s data.
        
        # Valid part starts at frame 90
        cutoff = 90
        
        frames = []
        for i in range(total_frames):
            if i < cutoff:
                # Filtered out frames: just put valid data or noise
                inc = 0.0
                vx, vy = 0.0, 0.0
            else:
                # Valid frames
                t = (i - cutoff) / 30.0
                inc = inclination_func(t)
                vx, vy = velocity_func(t)
                
            frame = {
                'torso_inclination_deg': inc,
                'velocity_shoulder_midpoint_x': vx,
                'velocity_shoulder_midpoint_y': vy,
                'shoulder_width_raw': width,
                # irrelevant but needed keys to avoid errors if logic checks them
                'shoulder_tilt_angle': 0.0,
                'eye_distance_ratio': 1.0 
            }
            frames.append(frame)
            self.tf.add_frame_features(frame, {}, i/30.0)
            
        return self.tf.extract_temporal_features()

    def test_baseline_stability(self):
        """Steady posture, no movement -> High Stability"""
        # Inc = 0 (std=0 -> base=1)
        # V = 0
        def inc_func(t): return 0.0
        def vel_func(t): return 0.0, 0.0
        
        results = self._add_frames(5.0, inc_func, vel_func)
        stab = results['posture_stability']['value']
        print(f"Baseline Stability: {stab}")
        self.assertGreater(stab, 0.95)

    def test_velocity_penalty(self):
        """Steady posture, HIGH movement -> Lower Stability"""
        # Inc = 0 (base=1)
        # V = 0.5 (normalized)
        # Width=1.0, so Vmag=0.5
        # Penalty: 0.3*(1-0.5)=0.15. Base 0.5. Freq ~0 (const vel). 0.2
        # Composite = 0.5 + 0.15 + 0.2 = 0.85
        # Coupling: 0.85 * (1-0.5) = 0.425
        
        def inc_func(t): return 0.0
        def vel_func(t): return 0.5, 0.0 # Constant velocity
        
        results = self._add_frames(5.0, inc_func, vel_func)
        stab = results['posture_stability']['value']
        print(f"High Velocity Stability: {stab}")
        self.assertLess(stab, 0.50, "Should be penalized heavily by velocity coupling")

    def test_oscillation_penalty(self):
        """Steady posture, low velocity, HIGH oscillation -> Lower Stability"""
        # Inc = 0
        # V = 0.1 (low penalty)
        # Freq = High (change direction every frame)
        # Oscillation Score ~ 1.0 (max changes)
        # Composite = 0.5*1 + 0.3*(0.9) + 0.2*(0) = 0.77
        # Coupling: 0.1 <= 0.15 -> No extra penalty.
        # Result ~0.77
        
        def inc_func(t): return 0.0
        def vel_func(t): 
            # Flip direction every frame
            # frame index is derived from t... hacky
            # Just use t to alternate sign
            # 30fps. t increments by 0.033. 
            frame_idx = int(t * 30)
            if frame_idx % 2 == 0:
                return 0.1, 0.1
            else:
                return -0.1, -0.1

        results = self._add_frames(5.0, inc_func, vel_func)
        stab = results['posture_stability']['value']
        print(f"High Oscillation Stability: {stab}")
        self.assertLess(stab, 0.81) # 0.5 + 0.27 + 0 = 0.77. Allow margin.
        self.assertGreater(stab, 0.70)

    def test_combined_instability(self):
        """Shaky posture + High Movement"""
        # Inc: std=10 -> base=exp(-1) ~0.37
        # V=0.3 -> Coupling penalty (0.7)
        # Freq=1.0 (shaking)
        # Term1 = 0.5 * 0.37 = 0.185
        # Term2 = 0.3 * (0.7) = 0.21
        # Term3 = 0.2 * 0 = 0
        # Sum = 0.395
        # Coupling: 0.395 * (1-0.3) = 0.276
        
        def inc_func(t): return 10.0 * np.sin(t * 10) # fast shake
        def vel_func(t): 
             # High movement shaking
             frame_idx = int(t * 30)
             val = 0.3 if frame_idx % 2 == 0 else -0.3
             return val, 0.0
             
        results = self._add_frames(5.0, inc_func, vel_func)
        stab = results['posture_stability']['value']
        print(f"Combined Instability: {stab}")
        self.assertLess(stab, 0.40)

if __name__ == '__main__':
    unittest.main()
