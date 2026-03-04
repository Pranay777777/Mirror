"""verify_natural_summary.py — Validates posture summary text is human-readable."""
import sys
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=False)
    body = result.get("body", {})
    summary = body.get("posture_interpretation") or body.get("posture_summary") or "(not found)"
    print(f"[NATURAL SUMMARY] {summary!r}")
    ok = isinstance(summary, str) and len(summary) > 10 and not summary.startswith("{")
    print(f"  human-readable check: {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_natural_summary.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
