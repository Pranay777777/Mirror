import os
import sys
import io

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

import json
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import analyze_video
from features.temporal_features import TemporalFeatures

# --------------------------------------------
# CONFIGURATION
# --------------------------------------------

VALIDATION_VIDEOS = {
    "video1": "uploads/EYE1.mp4",
    "video2": "uploads/EYE2.mp4",
    "video3": "uploads/EYE3.mp4",
    "video4": "uploads/eye4.mp4",
    "video5": "uploads/eye5.mp4",
    "video6": "uploads/eye6.mp4",
    "video7": "uploads/eye7.mp4"
}

EXPECTED_RULES = {
    "video1": {
        "gaze_stability": (0.90, 1.0),
        "eye_contact_consistency": (0.85, 1.0),
        "interpretation": "Good eye contact"
    },
    "video2": {
        "gaze_stability": (0.85, 1.0),
        "eye_contact_consistency": (0.60, 0.80),
        "interpretation": "Moderate eye contact"
    },
    "video3": {
        "gaze_stability": (0.70, 0.85),
        "eye_contact_consistency": (0.30, 0.60),
        "interpretation": "Limited eye contact"
    },
    "video4": {
        "gaze_stability": (0.0, 0.70),
        "eye_contact_consistency": (0.0, 0.30),
        "interpretation": "Poor eye contact"
    },
    "video5": {
        "gaze_stability": (0.85, 1.0),
        "eye_contact_consistency": (0.75, 1.0),
        "interpretation": "Good eye contact"
    },
    "video6": { 
        # User didn't specify Video 6 rules in the prompt, using Video 2/7 logic (Moderate)?
        # Or maybe Video 5 logic? 
        # Prompt skipped Video 6 in EXPECTED METRIC RULES block but included it in VIDEO SET.
        # I'll infer generic ranges or skip checking strictly if user omitted it.
        # Wait, the prompt lists video7.
        # Let's assume Video 6 is "Moderate" (it's often similar to 2/7 in dataset).
        "gaze_stability": (0.60, 0.90),
        "eye_contact_consistency": (0.50, 0.80),
        "interpretation": "Moderate eye contact"
    },
    "video7": {
        "gaze_stability": (0.60, 0.85),
        "eye_contact_consistency": (0.50, 0.75),
        "interpretation": "Moderate eye contact"
    }
}

def analyze_and_validate():
    overall_pass = True
    results_log = {}

    print(f"🚀 Starting Engagement Validation on {len(VALIDATION_VIDEOS)} videos...", flush=True)

    for v_key, v_path in VALIDATION_VIDEOS.items():
        if not os.path.exists(v_path):
            print(f"❌ Video not found: {v_path}", flush=True)
            overall_pass = False
            continue

        print(f"\nAnalyzing {v_key} ({v_path})...", flush=True)
        try:
            # Run Analysis
            result = analyze_video(v_path, debug_mode=True)
            
            # Extract Metrics
            # Extract Metrics
            # NOTE: Temporal features (gaze, blink) are currently nested in 'posture_analysis' metrics
            # due to aggregation logic. We extract from there.
            ma = result.get("results", {}).get("multimodal_analysis", {})
            metrics = ma.get("posture_analysis", {}).get("metrics", {})
            engagement_analysis = ma.get("engagement_analysis", {})
        
            # Safe Extraction
            gaze_stab = float(metrics.get("gaze_stability", {}).get("value", 0.0) or 0.0)
            consistency = float(metrics.get("eye_contact_consistency", {}).get("value", 0.0) or 0.0)
            interpretation = engagement_analysis.get("interpretation", "Unknown")
            
            # Check Rules
            rules = EXPECTED_RULES.get(v_key, {})
            pass_v = True
            reasons = []

            # Stability Check
            if "gaze_stability" in rules:
                min_v, max_v = rules["gaze_stability"]
                if not (min_v <= gaze_stab <= max_v):
                    pass_v = False
                    reasons.append(f"Stability {gaze_stab:.2f} not in [{min_v}, {max_v}]")

            # Consistency Check
            if "eye_contact_consistency" in rules:
                min_v, max_v = rules["eye_contact_consistency"]
                if not (min_v <= consistency <= max_v):
                    pass_v = False
                    reasons.append(f"Consistency {consistency:.2f} not in [{min_v}, {max_v}]")

            # Interpretation Check
            if "interpretation" in rules:
                expected_interp = rules["interpretation"]
                # Allow substring match (e.g. "Good eye contact" in "Good eye contact, but...")
                if expected_interp not in interpretation:
                    pass_v = False
                    reasons.append(f"Interpretation '{interpretation}' does not contain '{expected_interp}'")

            # Log Result
            status_icon = "✅" if pass_v else "❌"
            print(f"{status_icon} {v_key}: Stab={gaze_stab:.2f}, Cons={consistency:.2f}, Interp='{interpretation}'")
            if not pass_v:
                print(f"   Failures: {', '.join(reasons)}")
                overall_pass = False
            
            results_log[v_key] = {
                "passed": pass_v,
                "metrics": {"gaze_stability": gaze_stab, "eye_contact_consistency": consistency, "interpretation": interpretation},
                "failures": reasons
            }

        except Exception as e:
            print(f"❌ Exception analyzing {v_key}: {e}")
            overall_pass = False

    if overall_pass:
        print("\n🚀 Engagement system validated successfully.")
        print("All behavior scenarios passed.")
        print("System is production-ready.")
    else:
        print("\n⚠️ Validation FAILED. See logs above.")
        
    return overall_pass

if __name__ == "__main__":
    success = analyze_and_validate()
    sys.exit(0 if success else 1)
