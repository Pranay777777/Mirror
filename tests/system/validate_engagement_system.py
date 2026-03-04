"""validate_engagement_system.py — Validates engagement score is within [0, 10] and has an interpretation."""
import sys
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=False)
    body = result.get("body", {})
    eng = body.get("engagement_score")
    interp = body.get("engagement_interpretation") or body.get("engagement_interp") or "(not present)"
    print(f"[ENGAGEMENT VALIDATE]")
    print(f"  engagement_score: {eng}")
    print(f"  interpretation  : {interp}")
    ok = isinstance(eng, (int, float)) and 0.0 <= eng <= 10.0
    print(f"  range check     : {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/validate_engagement_system.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
