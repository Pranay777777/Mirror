import os
import sys
import json
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import analyze_video

def run_verification():
    videos = ["EYE1.mp4", "EYE2.mp4", "EYE3.mp4", "eye4.mp4", "eye5.mp4", "eye6.mp4", "eye7.mp4"]
    
    with open("verification_report.txt", "w", encoding="utf-8") as f:
        f.write("\nEXTENDED VERIFICATION RUN\n")
        f.write(f"{'VIDEO':<10} | {'STABILITY':<10} | {'CONSISTENCY':<12} | {'AWAY_RATIO':<10} | {'SWITCHES':<10} | {'INTERPRETATION'}\n")
        f.write("-" * 95 + "\n")

        for video in videos:
            path = os.path.join("uploads", video)
            if not os.path.exists(path):
                f.write(f"{video:<10} | FILENOTFOUND\n")
                continue
                
            try:
                # Suppress logging for clean output
                logging.getLogger().setLevel(logging.ERROR)
                
                result = analyze_video(path, debug_mode=True)
                
                # Extract Metrics
                metrics = result.get("results", {}).get("multimodal_analysis", {})
                if not metrics:
                     metrics = result.get("results", {}).get("metrics", {})

                # Navigate to values
                posture = metrics.get("posture_analysis", {}).get("metrics", {})
                gaze_stab = posture.get("gaze_stability", {}).get("value", 0.0)
                
                engagement = metrics.get("engagement_analysis", {})
                eng_metrics = engagement.get("metrics", {})
                consistency = eng_metrics.get("eye_contact", 0.0)
                
                # Switch Count
                switch_count = posture.get("gaze_direction_switch_count", {}).get("value", 0)
                
                # Interpretation
                interpretation = engagement.get("interpretation", "N/A")
                
                # Print Row
                line = f"{video:<10} | {gaze_stab:10.4f} | {consistency:12.4f} | {switch_count:10} | {interpretation}\n"
                f.write(line)
                print(line, end='', flush=True)
                
            except Exception as e:
                f.write(f"{video:<10} | ERROR: {e}\n")

if __name__ == "__main__":
    run_verification()
