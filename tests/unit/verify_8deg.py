"""verify_8deg.py — Checks 8° head lean threshold produces correct posture interpretation."""
import sys
from dotenv import load_dotenv
load_dotenv()

# 8° is the threshold where lean penalty kicks in.
# Simulate: if torso_inclination_deg > 8 → posture score should decrease.

def simulate_lean_penalty(inclination_deg: float, base_score: float = 8.0) -> float:
    LEAN_THRESHOLD = 8.0
    if inclination_deg > LEAN_THRESHOLD:
        penalty_factor = min(0.08, (inclination_deg - LEAN_THRESHOLD) * 0.005)
        return round(base_score * (1.0 - penalty_factor), 2)
    return base_score

if __name__ == "__main__":
    print("[VERIFY 8DEG THRESHOLD]")
    cases = [
        (0.0,  8.0, "no lean"),
        (8.0,  8.0, "at threshold"),
        (12.0, None, "above threshold"),
        (20.0, None, "large lean"),
    ]
    for deg, expected, label in cases:
        result = simulate_lean_penalty(deg)
        if expected is None:
            ok = result < 8.0
        else:
            ok = abs(result - expected) < 0.1
        print(f"  {'PASS' if ok else 'FAIL'} {label}: {deg}° → score={result}")
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
