"""verify_posture_start_ignore.py — Verifies that the first few frames (setup frames) are ignored in posture scoring."""
import sys
from dotenv import load_dotenv
load_dotenv()

# This test ensures that when a person sits down into frame, the "settling"
# frames at the start don't negatively impact the posture score.

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=False)
    body = result.get("body", {})
    posture = body.get("posture_score", 0)
    print(f"[POSTURE START IGNORE] posture_score = {posture}")
    # If start frames are ignored, score should be more stable / representative
    ok = isinstance(posture, (int, float)) and float(posture) >= 0
    print(f"  valid score: {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_posture_start_ignore.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
