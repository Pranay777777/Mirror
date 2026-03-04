"""
debug_analysis.py — Runs the full analysis pipeline with DEBUG_MODE=True and prints output.
Usage: python debug_analysis.py <video.mp4>
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
    from utils.scoring_utils import score_audio

    sarvam_key = os.getenv("SARVAM_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    print(f"[DEBUG] Analyzing: {video_path}")
    audio_data = process_audio(video_path, "debug", sarvam_key)
    transcript = audio_data.get("full_transcript")
    print(f"[DEBUG] Transcript: {transcript!r}")

    result = analyze_video(video_path, transcript=transcript, debug_mode=True)
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_analysis.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
