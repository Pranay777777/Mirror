import unittest
import numpy as np

class TestGazeStabilityFormula(unittest.TestCase):
    """
    Verify Center-Weighted Gaze Stability Formula.
    
    Formula:
    base_stability = 1 / (1 + std_angle)
    center_bias = clamp(1 - (mean_angle / 6.0), 0, 1)
    stability = base_stability * center_bias
    
    Target Ranges:
    - Perfect (Video 1): > 0.90
    - Slight Glances (Video 2): 0.60 - 0.85 (Wait, request says >0.85, and Video 3 is 0.70-0.85?)
      - User Request Re-read:
        - Video 3 (Frequent away): 0.70-0.85 ?? No, "Video 3... gaze_stability -> 0.70-0.85"
        - Video 2 (Slight): "gaze_stability -> >0.85"
        - Video 1 (Perfect): "gaze_stability -> >0.90"
      
    Let's check feasibility.
    """

    def calculate_stability(self, mean_angle, std_angle, threshold=6.0):
        base = 1.0 / (1.0 + std_angle)
        bias = max(0.0, min(1.0, 1.0 - (mean_angle / threshold)))
        return base * bias, base, bias

    def test_perfect_gaze(self):
        # Extremely stable and centered
        # Mean = 0.5 deg, Std = 0.1 deg
        mean, std = 0.5, 0.1
        stab, base, bias = self.calculate_stability(mean, std)
        print(f"\nPerfect ({mean=}, {std=}): Stability={stab:.3f} (Base={base:.3f}, Bias={bias:.3f})")
        # Base = 1/1.1 = 0.909
        # Bias = 1 - 0.5/6 = 0.916
        # Stab = 0.83
        # Expected > 0.90... This implies Perfect must be < 0.5 mean/0.1 std?
        
    def test_very_perfect_gaze(self):
        # Synthetic perfect
        mean, std = 0.1, 0.05
        stab, base, bias = self.calculate_stability(mean, std)
        print(f"Very Perfect ({mean=}, {std=}): Stability={stab:.3f}")
        # Base = 1/1.05 = 0.95
        # Bias = 1 - 0.016 = 0.98
        # Stab = 0.93 -> Matches > 0.90
        
    def test_slight_glances(self):
        # Occasional glances away
        # Mean = 2.0 deg, Std = 0.5 deg
        mean, std = 2.0, 0.5
        stab, base, bias = self.calculate_stability(mean, std)
        print(f"Slight Glances ({mean=}, {std=}): Stability={stab:.3f}")
        
    def test_frequent_away(self):
        # Looking away often
        # Mean = 5.0 deg, Std = 2.0 deg
        mean, std = 5.0, 2.0
        stab, base, bias = self.calculate_stability(mean, std)
        print(f"Frequent Away ({mean=}, {std=}): Stability={stab:.3f}")

if __name__ == '__main__':
    unittest.main()
