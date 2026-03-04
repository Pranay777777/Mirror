"""verify_biomechanical_model_v4.py — v4 version: checks posture, engagement, and expression are all in range."""
import sys
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=False)
    body = result.get("body", {})
    checks = {
        "posture_score":    (body.get("posture_score", -1), 0, 100),
        "engagement_score": (body.get("engagement_score", -1), 0, 100),
        "expression_score": (body.get("expression_score", -1), 0, 100),
        "overall_score":    (result.get("overall_score", -1), 0, 100),
    }
    for k, (v, lo, hi) in checks.items():
        ok = isinstance(v, (int, float)) and lo <= v <= hi
        print(f"  {'✓' if ok else '✗'} {k}={v} in [{lo},{hi}]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_biomechanical_model_v4.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
