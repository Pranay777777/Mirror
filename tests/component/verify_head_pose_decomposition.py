"""verify_head_pose_decomposition.py — Validates head pose yaw/pitch/roll decomposition."""
from dotenv import load_dotenv
load_dotenv()

def main():
    from features.head_pose_metrics import HeadPoseMetrics
    import numpy as np
    hp = HeadPoseMetrics()
    
    # Simulate forward-facing head (yaw=0, pitch=0, roll=0)
    mock_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    hp.add_frame(mock_pose, timestamp=0.0)
    
    # Simulate slight turn
    mock_pose2 = {"yaw": 15.0, "pitch": 5.0, "roll": 2.0}
    hp.add_frame(mock_pose2, timestamp=0.033)
    
    result = hp.finalize()
    print("[HEAD POSE DECOMP]", result)

if __name__ == "__main__":
    main()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
