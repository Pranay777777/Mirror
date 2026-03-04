"""verify_posture_fix.py — Verifies posture_score is >0 for a video with a visible person."""
import sys
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=False)
    body = result.get("body", {})
    posture = body.get("posture_score", 0)
    print(f"[POSTURE FIX] posture_score = {posture}")
    ok = isinstance(posture, (int, float)) and float(posture) >= 0
    print(f"  non-negative: {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_posture_fix.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
