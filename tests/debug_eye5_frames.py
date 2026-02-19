import sys
import os
import logging
import io

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import analyze_video

def run_debug():
    video = "eye5.mp4"
    path = os.path.join("uploads", video)
    
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    # Redirect stdout to file with UTF-8 encoding
    original_stdout = sys.stdout
    with open("eye5_frames.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        try:
            # Suppress logging
            logging.getLogger().setLevel(logging.ERROR)
            
            print(f"Analyzing {video} for frame-level debug data...")
            result = analyze_video(path, debug_mode=True)
            print("Analysis complete.")
            
        except Exception as e:
            print(f"Error analyzing {video}: {e}")
        finally:
            sys.stdout = original_stdout

if __name__ == "__main__":
    run_debug()
