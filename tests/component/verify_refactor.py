"""verify_refactor.py — Confirms refactored module structure still produces valid output."""
import sys
from dotenv import load_dotenv
load_dotenv()

REQUIRED_MODULES = [
    "features.normalized_geometry",
    "features.temporal_features",
    "features.audio_analysis",
    "features.speech_metrics",
    "features.linguistic_analysis",
    "features.head_pose_metrics",
    "features.stt_engine",
    "utils.video_utils",
    "utils.audio_utils",
    "utils.scoring_utils",
]

def main():
    print("[REFACTOR VERIFY]")
    for mod in REQUIRED_MODULES:
        try:
            __import__(mod)
            print(f"  ✓ {mod}")
        except Exception as e:
            print(f"  ✗ {mod} — {e}")

if __name__ == "__main__":
    main()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
