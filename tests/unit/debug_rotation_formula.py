"""debug_rotation_formula.py — Verifies the head rotation / torso inclination formula."""
import math

def torso_inclination(left_shoulder, right_shoulder):
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)

if __name__ == "__main__":
    cases = [
        ((0.3, 0.5), (0.7, 0.5), 0.0),     # level shoulders
        ((0.3, 0.45), (0.7, 0.55), 14.04),  # slight tilt
        ((0.3, 0.4), (0.7, 0.6), 26.57),    # moderate tilt
    ]
    for ls, rs, expected in cases:
        angle = torso_inclination(ls, rs)
        status = "PASS" if abs(angle - expected) < 0.5 else "FAIL"
        print(f"  [{status}] ls={ls} rs={rs} → {angle:.2f}° (expected ~{expected}°)")
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
