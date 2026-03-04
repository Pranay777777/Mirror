"""validate_schema.py — Validates full API response matches expected schema shape."""
import sys, json
from dotenv import load_dotenv
load_dotenv()

SCHEMA = {
    "analysis_version": str,
    "body": dict,
    "overall_score": (int, float),
}

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=False)
    passed = failed = 0
    for key, expected_type in SCHEMA.items():
        val = result.get(key)
        ok = isinstance(val, expected_type)
        print(f"  {'✓' if ok else '✗'} {key}: {type(val).__name__} (expected {expected_type})")
        if ok: passed += 1
        else: failed += 1
    print(f"\n[SCHEMA] {passed} passed, {failed} failed")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/validate_schema.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
