"""fast_verify_lean.py — Quick verify: checks sustained_lean_ratio and posture_score."""
import sys, json
from dotenv import load_dotenv
import os
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    from utils.audio_utils import process_audio
    audio = process_audio(video_path, "lean_test", os.getenv("SARVAM_API_KEY"))
    res = analyze_video(video_path, transcript=audio.get("full_transcript"), debug_mode=True)
    body = res.get("body", {})
    debug = res.get("debug_data", {})
    ma = debug.get("multimodal_analysis", {}) if debug else {}
    posture_metrics = ma.get("posture_analysis", {}).get("metrics", {})
    lean = posture_metrics.get("sustained_lean_ratio", {})
    print(f"posture_score      : {body.get('posture_score')}")
    print(f"sustained_lean_ratio: {lean}")
    print(json.dumps(res, indent=2, default=str))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/fast_verify_lean.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
