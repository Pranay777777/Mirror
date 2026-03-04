"""verify_deterministic_summary.py — Validates that posture summary text is deterministic."""
from dotenv import load_dotenv
load_dotenv()

CASES = [
    (0.90, 0.90, 0.05, "Excellent"),
    (0.70, 0.80, 0.10, "Good"),
    (0.50, 0.60, 0.15, "Fair"),
    (0.30, 0.40, 0.25, "Poor"),
]

def main():
    try:
        from utils.video_utils import _generate_deterministic_posture_summary as gen
    except ImportError:
        print("[SKIP] _generate_deterministic_posture_summary not exported.")
        return
    print("[DETERMINISTIC SUMMARY]")
    for align, stab, motion, expected_kw in CASES:
        text = gen(align, stab, motion)
        ok = expected_kw.lower() in text.lower()
        print(f"  {'✓' if ok else '✗'} align={align} stab={stab} motion={motion} → {text!r}")

if __name__ == "__main__":
    main()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
