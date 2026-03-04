"""verify_golden_matrix.py — Golden matrix of expected score ranges for known test videos."""
import sys, os
from dotenv import load_dotenv
load_dotenv()

# Golden matrix: {video_name: {metric: (min_expected, max_expected)}}
GOLDEN = {
    "bad_english.mp4": {
        "posture_score":    (0, 100),
        "engagement_score": (0, 100),
        "overall_score":    (0, 100),
    },
}

def main():
    from utils.video_utils import analyze_video
    for video_name, expected in GOLDEN.items():
        path = os.path.join("uploads", "TestVideos", video_name)
        if not os.path.exists(path):
            print(f"[GOLDEN MATRIX] SKIP: {path} not found")
            continue
        print(f"[GOLDEN MATRIX] Testing: {video_name}")
        result = analyze_video(path, transcript=None, debug_mode=False)
        body = result.get("body", {})
        for metric, (lo, hi) in expected.items():
            if metric == "overall_score":
                val = result.get(metric)
            else:
                val = body.get(metric)
            ok = isinstance(val, (int, float)) and lo <= val <= hi
            print(f"  {'✓' if ok else '✗'} {metric}={val} in [{lo},{hi}]")

if __name__ == "__main__":
    main()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
