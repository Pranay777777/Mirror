"""verify_modes.py — Tests both debug_mode=True and debug_mode=False return valid responses."""
import sys, json
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video

    for mode in [False, True]:
        print(f"\n[VERIFY MODES] debug_mode={mode}")
        result = analyze_video(video_path, transcript=None, debug_mode=mode)
        top_keys = list(result.keys())
        print(f"  top-level keys: {top_keys}")
        has_debug = "debug_data" in result
        print(f"  has 'debug_data': {has_debug} {'(expected True)' if mode else '(expected False)'}")
        ok = has_debug == mode
        print(f"  check: {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_modes.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
