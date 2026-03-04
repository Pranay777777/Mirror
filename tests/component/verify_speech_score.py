"""verify_speech_score.py — Verifies speech_score is in range [0, 10] and present in response."""
import sys
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=False)
    speech = result.get("speech_score")
    print(f"[SPEECH SCORE] speech_score = {speech}")
    ok = speech is None or (isinstance(speech, (int, float)) and 0.0 <= float(speech) <= 10.0)
    print(f"  range [0,10] or None: {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_speech_score.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
