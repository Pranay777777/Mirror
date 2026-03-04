"""verify_engagement.py — Verifies engagement_score is correct for a given video."""
import sys
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=False)
    body = result.get("body", {})
    eng = body.get("engagement_score")
    print(f"[ENGAGEMENT] engagement_score = {eng}")
    print(f"  type: {type(eng).__name__}")
    ok = isinstance(eng, (int, float)) and 0.0 <= float(eng) <= 100.0
    print(f"  range check: {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_engagement.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
