"""verify_confidence_redesign.py — Verifies that confidence scores are clamped to [0, 1]."""
import sys
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=True)
    debug = result.get("debug_data", {})
    ma = debug.get("multimodal_analysis", {}) if debug else {}
    cm = ma.get("confidence_metrics", {})
    for k, v in cm.items():
        ok = isinstance(v, (int, float)) and 0.0 <= v <= 1.0
        print(f"  {'✓' if ok else '✗'} {k}={v}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_confidence_redesign.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
