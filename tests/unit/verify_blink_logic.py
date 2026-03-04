"""verify_blink_logic.py — Tests blink detection logic unit: EAR threshold and timing."""

EAR_THRESHOLD = 0.21

def is_blink(ear: float) -> bool:
    return ear < EAR_THRESHOLD

if __name__ == "__main__":
    print("[BLINK LOGIC]")
    cases = [(0.10, True), (0.21, False), (0.30, False), (0.05, True)]
    for ear, expected in cases:
        result = is_blink(ear)
        print(f"  EAR={ear} → blink={result} | {'PASS' if result == expected else 'FAIL'}")
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
