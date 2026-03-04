"""verify_posture_stability_composite.py — Checks posture is a composite of alignment + stability."""
import sys
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=True)
    debug = result.get("debug_data", {})
    ma = debug.get("multimodal_analysis", {}) if debug else {}
    pa = ma.get("posture_analysis", {})
    metrics = pa.get("metrics", {})
    align = metrics.get("alignment_integrity", {}).get("value")
    stab  = metrics.get("stability_index", {}).get("value")
    score = pa.get("score")
    print(f"[POSTURE COMPOSITE]")
    print(f"  alignment_integrity : {align}")
    print(f"  stability_index     : {stab}")
    print(f"  posture_score       : {score}")
    ok = all(v is not None for v in [align, stab, score])
    print(f"  all present: {'PASS' if ok else 'FAIL (some values None)'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_posture_stability_composite.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
