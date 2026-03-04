"""debug_stt.py — Debugs Sarvam and Whisper STT transcription on a video."""
import sys, os
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    print("[STT DEBUG] Testing Sarvam AI...")
    from utils.audio_utils import process_audio
    sarvam_result = process_audio(video_path, "stt_debug", os.getenv("SARVAM_API_KEY"))
    print("  Sarvam transcript:", repr(sarvam_result.get("full_transcript", "")[:200]))

    print("[STT DEBUG] Testing Whisper (faster-whisper)...")
    from features.stt_engine import transcribe_audio
    whisper_result = transcribe_audio(video_path)
    print("  Whisper text:", repr(whisper_result.get("text", "")[:200]))
    print("  Whisper segments:", len(whisper_result.get("segments", [])))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/debug_stt.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
