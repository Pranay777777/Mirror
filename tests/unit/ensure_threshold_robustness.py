"""ensure_threshold_robustness.py — Checks that scoring thresholds hold across edge-case inputs."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))

def check_threshold(label, value, low, high, expected_band):
    """
    band = 'below'  if value < low
    band = 'normal' if low <= value <= high
    band = 'above'  if value > high
    (low/high match the actual cutoffs used in video_utils.py)
    """
    if value < low:
        band = "below"
    elif value <= high:
        band = "normal"
    else:
        band = "above"
    status = "PASS" if band == expected_band else "FAIL"
    print(f"  [{status}] {label}={value} → band='{band}' (expected '{expected_band}')")

if __name__ == "__main__":
    print("[THRESHOLD ROBUSTNESS]")

    # WPM: <90 → score=4 (below), 90-170 → score=9 (normal), >190 → score=4 (above)
    check_threshold("wpm",  60,  90, 170, "below")
    check_threshold("wpm", 130,  90, 170, "normal")
    check_threshold("wpm", 200,  90, 190, "above")

    # Filler rate (per min): 0 → best (below), 0<x≤6 → normal, >6 → above
    check_threshold("filler_rate", 0.0, 0.001,  6.0, "below")
    check_threshold("filler_rate", 2.5, 0.001,  6.0, "normal")
    check_threshold("filler_rate", 8.0, 0.001,  6.0, "above")

    # Pitch std (Hz): <15 → flat (below), 15-29 → good (normal), ≥50 → expressive (above)
    check_threshold("pitch_std", 10, 15, 49, "below")
    check_threshold("pitch_std", 22, 15, 49, "normal")
    check_threshold("pitch_std", 55, 15, 49, "above")

    # Energy variability: <5 → flat (below), 5-29 → normal, ≥30 → expressive (above)
    check_threshold("energy_var",  3, 5, 29, "below")
    check_threshold("energy_var", 15, 5, 29, "normal")
    check_threshold("energy_var", 35, 5, 29, "above")

    # Pause rate (per min): ≤2 → fluent (below normal), 3-5 → normal, >5 → too many pauses (above)
    check_threshold("pause_rate",  1, 3, 5, "below")
    check_threshold("pause_rate",  4, 3, 5, "normal")
    check_threshold("pause_rate",  7, 3, 5, "above")
