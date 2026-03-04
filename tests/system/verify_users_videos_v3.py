"""verify_users_videos_v3.py — v3: Same as verify_users_videos but includes speech processing via Sarvam."""
import sys, os, json
from dotenv import load_dotenv
import os as _os
load_dotenv()

TEST_VIDEOS_DIR = os.path.join("uploads", "TestVideos")

def main():
    from utils.video_utils import analyze_video
    from utils.audio_utils import process_audio
    sarvam_key = _os.getenv("SARVAM_API_KEY")

    if not os.path.exists(TEST_VIDEOS_DIR):
        print(f"[USERS V3] Directory not found: {TEST_VIDEOS_DIR}")
        return
    videos = [f for f in os.listdir(TEST_VIDEOS_DIR) if f.endswith(".mp4")]
    if not videos:
        print("[USERS V3] No .mp4 files found.")
        return

    for v in videos:
        path = os.path.join(TEST_VIDEOS_DIR, v)
        print(f"\n[USERS V3] {v}")
        try:
            audio = process_audio(path, v.replace(".mp4",""), sarvam_key)
            transcript = audio.get("full_transcript", "")
            print(f"  transcript: {len(transcript)} chars")
            result = analyze_video(path, transcript=transcript, debug_mode=False)
            body = result.get("body", {})
            print(f"  posture={body.get('posture_score')} engagement={body.get('engagement_score')} overall={result.get('overall_score')}")
        except Exception as e:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    main()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
