"""verify_gaze_stability.py — Checks gaze_stability / head_orientation metrics."""
import sys
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=True)
    debug = result.get("debug_data", {})
    ma = debug.get("multimodal_analysis", {}) if debug else {}
    eng = ma.get("engagement_analysis", {})
    metrics = eng.get("metrics", {})
    hos = metrics.get("head_orientation")
    print(f"[GAZE STABILITY]")
    print(f"  head_orientation : {hos}")
    print(f"  attention_shifts : {metrics.get('attention_shifts')}")
    print(f"  max_continuous   : {metrics.get('max_continuous')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_gaze_stability.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
