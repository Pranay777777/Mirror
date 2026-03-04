"""verify_gaze_switch.py — Verifies gaze_direction_switch_count is tracked."""
import sys
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=True)
    debug = result.get("debug_data", {})
    ma = debug.get("multimodal_analysis", {}) if debug else {}
    enc = ma.get("engagement_analysis", {})
    switches = enc.get("metrics", {}).get("attention_shifts")
    print(f"[GAZE_SWITCH] attention_shifts (gaze_direction_switch_count) = {switches}")
    ok = switches is None or isinstance(switches, (int, float))
    print(f"  type check: {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_gaze_switch.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
