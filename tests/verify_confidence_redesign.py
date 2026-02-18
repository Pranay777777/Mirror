import pandas as pd
import numpy as np
from features.temporal_features import TemporalFeatures

def verify_posture():
    tf = TemporalFeatures(fps=30.0)
    
    # helper to generate noisy signal
    def noise(val, scale, n):
        return np.random.normal(val, scale, n)

    # ---------------------------------------------------------
    # Scenario A: Perfect Posture (Upright, Still)
    # ---------------------------------------------------------
    n = 150
    # Torso Inclination ~0 deg (perfect upright)
    inc_a = noise(0.0, 0.5, n)
    # Shoulder Midpoint stable (x=0.5, y=0.5)
    mx_a = noise(0.5, 0.001, n)
    my_a = noise(0.5, 0.001, n)
    # Width stable
    w_a = noise(0.2, 0.001, n)
    
    print("\n=== SCENARIO A: Perfect Posture (Upright 0deg, Stable, Still) ===")
    res_a = run_scenario(tf, inc_a, mx_a, my_a, w_a)
    
    # ---------------------------------------------------------
    # Scenario B: Slouching (35 deg) + Movement
    # ---------------------------------------------------------
    # Torso Inclination ~35 deg (should result in 0 upright score as >30)
    inc_b = noise(35.0, 2.0, n)
    # Moving shoulders (drifting from 0.5 to 0.8)
    mx_b = np.linspace(0.5, 0.8, n) + noise(0, 0.01, n)
    my_b = np.linspace(0.5, 0.8, n) + noise(0, 0.01, n)
    w_b = noise(0.2, 0.01, n)
    
    print("\n=== SCENARIO B: Slouching (35deg) + Moving ===")
    res_b = run_scenario(tf, inc_b, mx_b, my_b, w_b)

    # ---------------------------------------------------------
    # Scenario C: Mild Lean (15 deg) + Stable
    # ---------------------------------------------------------
    # 15 deg -> Halfway score (0.5)
    inc_c = noise(15.0, 0.5, n)
    mx_c = noise(0.5, 0.001, n)
    my_c = noise(0.5, 0.001, n)
    w_c = noise(0.2, 0.001, n)
    
    print("\n=== SCENARIO C: Mild Lean (15deg) + Stable ===")
    res_c = run_scenario(tf, inc_c, mx_c, my_c, w_c)
    
    # Verification
    print("\n=== VERIFICATION RESULTS ===")
    
    # 1. Uprightness
    # A (0 deg) -> ~1.0
    # B (35 deg) -> 0.0
    # C (15 deg) -> ~0.5
    u_a = res_a['uprightness']
    u_b = res_b['uprightness']
    u_c = res_c['uprightness']
    
    print(f"Uprightness: A={u_a:.2f} (Exp ~1.0), B={u_b:.2f} (Exp 0.0), C={u_c:.2f} (Exp ~0.5)")
    
    if u_a > 0.9 and u_b < 0.1 and 0.4 < u_c < 0.6:
        print("[PASS] Uprightness matches formula")
    else:
        print("[FAIL] Uprightness logic incorrect")

    # 2. Stability
    # A (std ~0.5) -> exp(-0.5/10) = exp(-0.05) ~0.95
    # B (std ~2.0) -> exp(-0.2) ~0.81
    s_a = res_a['stability']
    s_b = res_b['stability']
    
    print(f"Stability: A={s_a:.2f}, B={s_b:.2f}")
    if s_a > s_b:
        print("[PASS] Stability (Low Std > High Std)")
    else:
        print("[FAIL] Stability logic incorrect")

    # 3. Movement
    # A (still) -> Low
    # B (moving) -> High
    m_a = res_a['movement']
    m_b = res_b['movement']
    
    print(f"Movement: A={m_a:.4f}, B={m_b:.4f}")
    if m_b > m_a * 10: # B is moving significantly more
        print("[PASS] Movement intensity detected correctly")
    else:
        print("[FAIL] Movement logic incorrect")

def run_scenario(tf, inclination, mx, my, w):
    tf.reset()
    for i in range(len(inclination)):
        # Construct simplified features
        pose = {
            'torso_inclination_deg': inclination[i],
            'shoulder_midpoint_x': mx[i],
            'shoulder_midpoint_y': my[i],
            'shoulder_width_raw': w[i],
            # dummies for other checks
            'shoulder_tilt_angle': 0, 
            'head_height_ratio': 0.8,
            'eye_distance_ratio': 0.1
        }
        # face dummy
        face = {
            'left_eye_opening_ratio': 0.3,
            'right_eye_opening_ratio': 0.3,
            'gaze_alignment_angle': 5.0
        }
        tf.add_frame_features(pose, face, i * (1/30.0))
        
    results = tf.extract_temporal_features()
    
    upright = results.get('posture_uprightness', {}).get('value', 0.0)
    stability = results.get('posture_stability', {}).get('value', 0.0)
    movement = results.get('overall_movement_intensity', {}).get('value', 0.0)
    
    print(f"  Uprightness: {upright:.4f}")
    print(f"  Stability:   {stability:.4f}")
    print(f"  Movement:    {movement:.4f}")
    
    # We also expect the diagnostic print from the code itself to appear in stdout
    
    return {'uprightness': upright, 'stability': stability, 'movement': movement}

if __name__ == "__main__":
    verify_posture()
