"""verify_biomechanical_damping.py — Verifies biomechanical smoothing/damping on motion data."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import numpy as np

def apply_damping(values, alpha=0.3):
    """Exponential moving average — low α = high damping."""
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed

if __name__ == "__main__":
    raw = [0.0, 0.5, 1.0, 0.8, 0.2, 0.9, 0.1]
    damped = apply_damping(raw, alpha=0.3)
    print("[BIOMECHANICAL DAMPING]")
    for r, d in zip(raw, damped):
        print(f"  raw={r:.2f}  damped={d:.3f}")
    variance_raw = float(np.var(raw))
    variance_damped = float(np.var(damped))
    ok = variance_damped < variance_raw
    print(f"  variance_raw={variance_raw:.4f}  variance_damped={variance_damped:.4f}")
    print(f"  [{'PASS' if ok else 'FAIL'}] damping reduces variance")
