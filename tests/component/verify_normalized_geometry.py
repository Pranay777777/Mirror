"""verify_normalized_geometry.py — Smoke test for NormalizedGeometry.process() output."""
import cv2, numpy as np
from dotenv import load_dotenv
load_dotenv()

def main():
    from features.normalized_geometry import NormalizedGeometry
    geom = NormalizedGeometry()
    # Create a synthetic black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = geom.process(frame)
    print("[NORM GEOM] process() returned:", type(result).__name__)
    print("  Keys:", list(result.keys())[:10])
    ok = isinstance(result, dict)
    print(f"  is dict: {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    main()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
