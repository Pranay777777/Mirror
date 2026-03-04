"""single_test.py — Runs a single focused test on one video file."""
import sys, json
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    print(f"[SINGLE TEST] {video_path}")
    result = analyze_video(video_path, transcript=None, debug_mode=False)
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/single_test.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
