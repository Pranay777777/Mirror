import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import analyze_video 

# --------------------------------------------
# CONFIGURATION
# --------------------------------------------

VIDEO_PATH = "uploads/EYE1.mp4"

def analyze_single():
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video not found: {VIDEO_PATH}")
        return False

    print(f"\nAnalyzing {VIDEO_PATH}...", flush=True)
    try:
        # Run Analysis
        result = analyze_video(VIDEO_PATH, debug_mode=True)
        
        # Extract Metrics
        metrics = result.get("results", {}).get("metrics", {})
        engagement = metrics.get("engagement", {})
        
        gaze_stab = float(engagement.get("gaze_stability", {}).get("value", 0.0) or 0.0)
        consistency = float(engagement.get("eye_contact_consistency", {}).get("value", 0.0) or 0.0)
        interpretation = engagement.get("engagement_interpretation", "Unknown")
        
        with open("validation_details.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print("✅ Full result written to validation_details.json", flush=True)
        return True

    except Exception as e:
        print(f"❌ Exception analyzing: {e}", flush=True)
        return False

if __name__ == "__main__":
    analyze_single()
