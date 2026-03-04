"""
test_core.py — Tests the core analysis pipeline (no HTTP) on a local video.
Usage: python test_core.py <video.mp4>
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import sys, json
from dotenv import load_dotenv
import os
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video

    print(f"[TEST CORE] Analyzing: {video_path}")
    result = analyze_video(video_path, transcript=None, debug_mode=False)

    print("Keys in result:", list(result.keys()))
    body = result.get("body", {})
    print(f"  posture_score    : {body.get('posture_score')}")
    print(f"  engagement_score : {body.get('engagement_score')}")
    print(f"  overall_score    : {result.get('overall_score')}")
    print("[TEST CORE] Full output:")
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_core.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
