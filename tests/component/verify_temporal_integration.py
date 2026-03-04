"""verify_temporal_integration.py — Verifies temporal features integrate correctly across multiple frames."""
import numpy as np
from dotenv import load_dotenv
load_dotenv()

def main():
    from features.temporal_features import TemporalFeatures
    temp = TemporalFeatures()
    # Simulate 60 frames of data
    for i in range(60):
        mock_geo = {
            "torso_inclination_deg": np.random.uniform(-5, 5),
            "shoulder_tilt_angle": np.random.uniform(-3, 3),
        }
        temp.add_frame(mock_geo, mock_geo, i / 30.0)
    results = temp.finalize()
    print("[TEMPORAL INTEGRATION]")
    print(f"  alignment_integrity : {results.get('alignment_integrity')}")
    print(f"  stability_index     : {results.get('stability_index')}")
    print(f"  valid_pose_ratio    : {results.get('valid_pose_ratio')}")
    ok = all(k in results for k in ["alignment_integrity", "stability_index"])
    print(f"  required keys present: {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    main()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
