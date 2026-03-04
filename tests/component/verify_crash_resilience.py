"""verify_crash_resilience.py — Sends edge-case inputs to verify no unhandled exceptions."""
import sys, os
from dotenv import load_dotenv
load_dotenv()

def main():
    from utils.video_utils import analyze_video

    print("[CRASH RESILIENCE] Test 1: Non-existent file")
    try:
        analyze_video("nonexistent_file.mp4", transcript=None, debug_mode=False)
        print("  ✗ Expected exception not raised")
    except Exception as e:
        print(f"  ✓ Caught expected exception: {type(e).__name__}: {e}")

    print("[CRASH RESILIENCE] Test 2: None transcript (no speech)")
    # Only run if a test video exists
    test_video = "uploads/TestVideos/bad_english.mp4"
    if os.path.exists(test_video):
        try:
            result = analyze_video(test_video, transcript=None, debug_mode=False)
            print(f"  ✓ Returned result with keys: {list(result.keys())}")
        except Exception as e:
            print(f"  ✗ Unexpected crash: {e}")
    else:
        print(f"  SKIP: {test_video} not found")

if __name__ == "__main__":
    main()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
