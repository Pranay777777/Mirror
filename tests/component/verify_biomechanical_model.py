"""verify_biomechanical_model.py — Validates the biomechanical posture model output range."""
import sys
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=False)
    body = result.get("body", {})
    posture = body.get("posture_score", 0)
    ok = 0.0 <= float(posture) <= 100.0
    print(f"[BIOMECH MODEL] posture_score={posture} → {'PASS' if ok else 'FAIL'} (range 0-100)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_biomechanical_model.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
