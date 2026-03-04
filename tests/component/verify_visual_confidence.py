"""verify_visual_confidence.py — Verifies visual_confidence score is in [0, 1]."""
import sys
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=True)
    debug = result.get("debug_data", {})
    ma = debug.get("multimodal_analysis", {}) if debug else {}
    cm = ma.get("confidence_metrics", {})
    vc = cm.get("visual_confidence")
    print(f"[VISUAL CONFIDENCE] visual_confidence = {vc}")
    ok = isinstance(vc, (int, float)) and 0.0 <= float(vc) <= 1.0
    print(f"  range [0,1]: {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_visual_confidence.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
