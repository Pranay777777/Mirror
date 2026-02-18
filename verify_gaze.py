import sys
import os
from utils.video_utils import analyze_video

# Force stdout encoding to utf-8 for Windows
sys.stdout.reconfigure(encoding='utf-8')

video_path = "uploads/pran.mp4"
if not os.path.exists(video_path):
    print(f"File not found: {video_path}")
    sys.exit(1)

print(f"Running analysis on {video_path}...")
result = analyze_video(video_path)
print("Analysis complete.")
