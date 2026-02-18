import unittest
import numpy as np

def calculate_stability_with_damping(alignment, motion):
    """
    Simulates the stability calculation logic from features/temporal_features.py
    
    Logic:
    stability_index = alignment * exp(-1.5 * motion)
    
    if motion > 0.08:
        stability_index *= (1.0 - 0.6 * motion)
        
    if motion > 0.15:
        stability_index *= (1.0 - 0.8 * motion)
        
    return clip(stability_index, 0, 1)
    """
    
    # Base calculation
    stability_index = alignment * np.exp(-1.5 * motion)
    
    # Damping 1
    if motion > 0.08:
        stability_index *= (1.0 - (0.6 * motion))
        
    # Damping 2
    if motion > 0.15:
        stability_index *= (1.0 - (0.8 * motion))
        
    return float(np.clip(stability_index, 0.0, 1.0))

class TestBiomechanicalDamping(unittest.TestCase):
    
    def test_minimal_motion(self):
        # Motion < 0.03 -> Stability remains Very High (>0.90)
        align = 0.98
        motion = 0.02
        
        stab = calculate_stability_with_damping(align, motion)
        print(f"Minimal Motion: m={motion}, align={align} -> stab={stab:.4f}")
        self.assertGreater(stab, 0.90, "Minimal motion should yield Very High stability")

    def test_noticeable_motion(self):
        # Motion 0.08-0.15 -> Stability drops to 0.65-0.80 range
        align = 0.95 # Good alignment assumed
        
        # Lower bound of range
        m_low = 0.09
        stab_low = calculate_stability_with_damping(align, m_low)
        print(f"Noticeable Motion (Low): m={m_low}, align={align} -> stab={stab_low:.4f}")
        
        # Upper bound of range
        m_high = 0.14
        stab_high = calculate_stability_with_damping(align, m_high)
        print(f"Noticeable Motion (High): m={m_high}, align={align} -> stab={stab_high:.4f}")
        
        # Check against target range (roughly)
        self.assertLess(stab_low, 0.85, "Noticeable motion should drop below High (>0.85)")
        self.assertGreater(stab_high, 0.60, "Noticeable motion should not drop too low (<0.60)")

    def test_excessive_motion(self):
        # Motion > 0.18 -> Stability drops below 0.70
        align = 0.90 # Even with decent alignment
        motion = 0.19
        
        stab = calculate_stability_with_damping(align, motion)
        print(f"Excessive Motion: m={motion}, align={align} -> stab={stab:.4f}")
        
        self.assertLess(stab, 0.70, "Excessive motion (>0.18) must drop stability below 0.70 (Reduced/Low)")

if __name__ == '__main__':
    unittest.main()
