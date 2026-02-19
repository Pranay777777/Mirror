import sys, os, logging
import numpy as np
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import analyze_video

logging.getLogger().setLevel(logging.ERROR)

videos = [
    "good_english.mp4", 
    "bad_english.mp4", 
    "english_women_bad.mov", 
    "english_women_good.mov"
]

lines = []
header = f"{'Video':<25} | {'Score':<5} | {'Rel':<4} | {'Var':<4} | {'PitchStd':<8} | {'EgyVar':<6} | {'Mode'}"
lines.append(header)
print(header)
print("-" * 100)
lines.append("-" * 100)

for vid in videos:
    path = os.path.join("uploads", vid)
    if not os.path.exists(path):
        msg = f"{vid:<25} | FILE NOT FOUND"
        print(msg)
        lines.append(msg)
        continue

    try:
        # Run without transcript
        r = analyze_video(path, transcript=None, debug_mode=True)
        
        sa = r["results"]["multimodal_analysis"].get("speech_analysis", {})
        score = sa.get("score")
        metrics = sa.get("metrics", {})
        mode = metrics.get("speech_logic_mode", "?")
        
        # Audio metrics
        cm = r["results"]["multimodal_analysis"]["confidence_metrics"]
        rel = cm.get("audio_confidence", 0.0)
        
        # Variety components
        af = sa.get("audio_features", {})
        am = af.get("metrics", {})
        pitch_std = am.get("pitch_std_hz", 0.0)
        egy_var = am.get("energy_variability_score", 0.0)
        
        var_score = metrics.get("variety_score", "-")
        
        output_line = f"{vid:<25} | {score:<5} | {rel:<4} | {var_score:<4} | {pitch_std:<8} | {egy_var:<6} | {mode}"
        lines.append(output_line)
        print(output_line)

    except Exception as e:
        # import traceback
        # traceback.print_exc()
        result_line = f"{vid:<25} | ERROR: {str(e)[:50]}..."
        lines.append(result_line)
        print(result_line)

with open("verify_users_out.txt", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

print("\nDONE")
