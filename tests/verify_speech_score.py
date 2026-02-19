import sys, os, logging
import numpy as np
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock Librosa for speed? 
# No, we need REAL audio analysis for reliability and variety!
# So we CANNOT mock librosa.load completely if we want to test the full pipeline.
# BUT, running full analysis is slow.
# I will use the actual video_utils pipeline, but I might need to wait a bit.
# The user wants "Deterministic v1 Implementation" verified.
# I should just run it.

from utils.video_utils import analyze_video

logging.getLogger().setLevel(logging.ERROR)

videos = ["EYE1.mp4", "EYE2.mp4", "videoB1.mp4"]
results = []

print(f"{'Video':<15} | {'Score':<5} | {'Mode':<25} | {'Rel':<4} | {'Rate':<4} | {'Fill':<4} | {'Var':<4}")
print("-" * 85)

for vid in videos:
    path = os.path.join("uploads", vid)
    if not os.path.exists(path):
        print(f"{vid}: FILE NOT FOUND")
        continue
        
    try:
        # We need transcript for rate/fillers. 
        # If I don't provide transcript, wpm is 0 -> No Transcript Fallback (5.0)
        # To test the SCORING logic, I should provide a dummy transcript for videoB1.
        # For EYE1/EYE2, I expect fallback or low reliability.
        
        transcript = None
        if vid == "videoB1.mp4":
            # Dummy transcript to simulate speech
            transcript = "This is a test transcript with enough words to calculate a speaking rate that falls into a specific category for verification purposes."
            
        r = analyze_video(path, transcript=transcript, debug_mode=True)
        sa = r["results"]["multimodal_analysis"].get("speech_analysis", {})
        score = sa.get("score")
        metrics = sa.get("metrics", {})
        
        mode = metrics.get("speech_logic_mode", "?")
        rel = r["results"]["multimodal_analysis"]["confidence_metrics"].get("audio_confidence", 0.0)
        
        rate = metrics.get("rate_score", "-")
        fill = metrics.get("filler_score", "-")
        var = metrics.get("variety_score", "-")
        
        print(f"{vid:<15} | {score:<5} | {mode:<25} | {rel:<4} | {rate:<4} | {fill:<4} | {var:<4}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"{vid}: ERROR {e}")

print("\nDONE")
