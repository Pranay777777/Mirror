"""final_verification_architecture.py — End-to-end architecture correctness check."""
import sys, json
from dotenv import load_dotenv
import os
load_dotenv()

REQUIRED_BODY_KEYS = ["posture_score", "engagement_score", "expression_score"]
REQUIRED_TOP_KEYS = ["analysis_version", "body", "overall_score"]

def verify(video_path: str):
    from utils.video_utils import analyze_video
    res = analyze_video(video_path, transcript=None, debug_mode=False)
    
    passed = failed = 0
    for k in REQUIRED_TOP_KEYS:
        ok = k in res
        print(f"  {'✓' if ok else '✗'} top-level: {k}")
        if ok: passed += 1
        else: failed += 1

    body = res.get("body", {})
    for k in REQUIRED_BODY_KEYS:
        ok = k in body
        print(f"  {'✓' if ok else '✗'} body: {k}")
        if ok: passed += 1
        else: failed += 1

    print(f"\n[ARCH VERIFY] {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/final_verification_architecture.py <video.mp4>")
        sys.exit(1)
    ok = verify(sys.argv[1])
    sys.exit(0 if ok else 1)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
