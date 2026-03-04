"""verify_slim_response.py — Confirms production response (debug_mode=False) doesn't include debug_data."""
import sys
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=False)
    has_debug = "debug_data" in result
    print(f"[SLIM RESPONSE] 'debug_data' present: {has_debug}")
    print(f"  slim check: {'PASS' if not has_debug else 'FAIL (debug data leaked into production response)'}")
    print(f"  keys: {list(result.keys())}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_slim_response.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
