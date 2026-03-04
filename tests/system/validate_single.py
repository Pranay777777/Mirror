"""validate_single.py — Validates a single video produces non-zero scores."""
import sys, json
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=False)
    body = result.get("body", {})
    overall = result.get("overall_score", 0)
    posture = body.get("posture_score", 0)
    engagement = body.get("engagement_score", 0)
    checks = [
        ("overall_score > 0", overall > 0),
        ("posture_score >= 0", posture >= 0),
        ("engagement_score >= 0", engagement >= 0),
        ("overall_score <= 100", overall <= 100),
    ]
    for label, ok in checks:
        print(f"  {'✓' if ok else '✗'} {label}")
    print(json.dumps(body, indent=2, default=str))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/validate_single.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
