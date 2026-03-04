import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import sys
import logging
from utils.video_utils import analyze_video

logging.basicConfig(level=logging.INFO)

def main():
    video_path = "uploads/fhappy.mp4"
    print(f"Running analysis on {video_path}")
    try:
        print("Testing debug_mode=False (build_public_response)")
        result = analyze_video(video_path, debug_mode=False)
        print("\n--- Final Public Response ---")
        import json
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Analysis Failed: {e}")

if __name__ == "__main__":
    main()
