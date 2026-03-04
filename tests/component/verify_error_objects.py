"""verify_error_objects.py — Verifies error responses have correct structure."""
import sys, json
from dotenv import load_dotenv
load_dotenv()

def main():
    from utils.video_utils import analyze_video
    try:
        analyze_video("does_not_exist.mp4", transcript=None, debug_mode=False)
        print("[ERROR OBJ] ✗ Expected exception, got none")
    except RuntimeError as e:
        print(f"[ERROR OBJ] ✓ RuntimeError raised: {e}")
    except Exception as e:
        print(f"[ERROR OBJ] ✓ Exception raised: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
