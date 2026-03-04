"""
debug_verify.py — Verifies that the full analysis pipeline runs without errors.
Usage: python debug_verify.py <video.mp4>
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import sys, json
from dotenv import load_dotenv
import os
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    from utils.audio_utils import process_audio

    print("[VERIFY] Step 1: Audio processing...")
    audio_data = process_audio(video_path, "verify", os.getenv("SARVAM_API_KEY"))
    transcript = audio_data.get("full_transcript", "")
    print(f"  transcript length: {len(transcript)} chars")

    print("[VERIFY] Step 2: Video analysis...")
    result = analyze_video(video_path, transcript=transcript, debug_mode=False)

    required_keys = ["analysis_version", "body", "overall_score"]
    for k in required_keys:
        status = "✓" if k in result else "✗ MISSING"
        print(f"  {status} {k}")

    print("[VERIFY] Done.")
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_verify.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
