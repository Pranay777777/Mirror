"""
reproduce_issue.py — Reproduces a specific bug or regression for investigation.
Usage: python reproduce_issue.py <video.mp4>

Edit the `reproduce()` function body to target the specific issue being investigated.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import sys, json
from dotenv import load_dotenv
import os
load_dotenv()

def reproduce(video_path: str):
    from utils.video_utils import analyze_video
    from utils.audio_utils import process_audio

    audio_data = process_audio(video_path, "repro", os.getenv("SARVAM_API_KEY"))
    transcript = audio_data.get("full_transcript", "")
    result = analyze_video(video_path, transcript=transcript, debug_mode=True)

    # ── Example: verify engagement_score is non-zero for a visible subject ──
    body = result.get("body", {})
    eng  = body.get("engagement_score")
    print(f"[REPRO] engagement_score = {eng}")
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reproduce_issue.py <video.mp4>")
        sys.exit(1)
    reproduce(sys.argv[1])
