"""verify_speech_v3.py — v3 speech scoring verification: WPM, filler, pause, pitch, energy."""
import sys, json
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=True)
    debug = result.get("debug_data", {})
    ma = debug.get("multimodal_analysis", {}) if debug else {}
    sa = ma.get("speech_analysis", {})
    metrics = sa.get("metrics", {})
    print("[SPEECH V3]")
    for k in ["words_per_minute", "filler_rate_per_min", "pause_rate_per_min", "rate_score", "filler_score", "pause_score", "pitch_score", "energy_score"]:
        print(f"  {k}: {metrics.get(k)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_speech_v3.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
