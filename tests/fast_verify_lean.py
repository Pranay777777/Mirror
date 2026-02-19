import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import analyze_video

def run_fast_verification():
    videos = ["pran.mp4"]
    
    print(f"\nFAST VERIFICATION RUN (Lean Binary Model - WITH AUDIO)", flush=True)
    print(f"{'VIDEO':<10} | {'STABILITY':<10} | {'CONSISTENCY':<12} | {'SWITCHES':<10} | {'INTERPRETATION'}", flush=True)
    print("-" * 90, flush=True)

    for video in videos:
        path = os.path.join("uploads", video)
        if not os.path.exists(path):
            print(f"{video:<10} | FILENOTFOUND")
            continue
            
        try:
            # Suppress logging
            logging.getLogger().setLevel(logging.ERROR)
            
            result = analyze_video(path, debug_mode=True)
            
            # Extract Metrics
            metrics = result.get("results", {}).get("multimodal_analysis", {})
            if not metrics:
                 metrics = result.get("results", {}).get("metrics", {})

            posture = metrics.get("posture_analysis", {}).get("metrics", {})
            gaze_stab = posture.get("gaze_stability", {}).get("value", 0.0)
            
            engagement = metrics.get("engagement_analysis", {})
            eng_metrics = engagement.get("metrics", {})
            consistency = eng_metrics.get("eye_contact", 0.0)
            
            switch_count = posture.get("gaze_direction_switch_count", {}).get("value", 0)
            
            interpretation = engagement.get("interpretation", "N/A")
            
            print(f"{video:<10} | {gaze_stab:10.4f} | {consistency:12.4f} | {switch_count:10} | {interpretation}")
            
        except Exception as e:
            print(f"{video:<10} | ERROR: {e}")

if __name__ == "__main__":
    run_fast_verification()
