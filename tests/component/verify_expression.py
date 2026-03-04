"""verify_expression.py — Verifies expression_score is returned and in range [0, 100]."""
import sys
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=False)
    body = result.get("body", {})
    expr = body.get("expression_score")
    print(f"[EXPRESSION] expression_score = {expr}")
    ok = isinstance(expr, (int, float)) and 0.0 <= float(expr) <= 100.0
    print(f"  range [0,100]: {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_expression.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
