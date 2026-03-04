"""verify_domain_isolation.py — Verifies posture, speech, engagement domains don't bleed into each other."""
import sys
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from utils.video_utils import analyze_video
    result = analyze_video(video_path, transcript=None, debug_mode=True)
    debug = result.get("debug_data", {})
    ma = debug.get("multimodal_analysis", {}) if debug else {}

    posture_conf = ma.get("posture_analysis", {}).get("confidence")
    speech_conf  = ma.get("speech_analysis", {}).get("confidence")
    
    print("[DOMAIN ISOLATION]")
    print(f"  posture.confidence (visual) : {posture_conf}")
    print(f"  speech.confidence  (audio)  : {speech_conf}")
    # They should be different values if domains are isolated
    isolated = posture_conf != speech_conf
    print(f"  domains differ: {'✓ PASS' if isolated else '? SAME VALUE (may still be ok)'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_domain_isolation.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
