"""
test_sarvam.py — Tests the Sarvam AI speech-to-text integration on a local audio/video file.
Usage: python test_sarvam.py <video.mp4>
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import sys, os
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.audio_utils import process_audio
    key = os.getenv("SARVAM_API_KEY")
    if not key:
        print("[ERROR] SARVAM_API_KEY not set in .env")
        return
    print(f"[TEST SARVAM] Processing: {video_path}")
    result = process_audio(video_path, "sarvam_test", key)
    transcript = result.get("full_transcript", "")
    print(f"[TEST SARVAM] Transcript ({len(transcript)} chars): {transcript!r}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_sarvam.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
