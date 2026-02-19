import sys, os, logging
import numpy as np
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

mock_librosa = MagicMock()
mock_librosa.load.return_value = (np.zeros(16000), 16000)
sys.modules["librosa"] = mock_librosa

import features.audio_analysis
features.audio_analysis.AudioAnalyzer = MagicMock()
features.audio_analysis.AudioAnalyzer.return_value.finalize.return_value = {}
import features.speech_metrics
features.speech_metrics.SpeechMetrics = MagicMock()
features.speech_metrics.SpeechMetrics.return_value.finalize.return_value = {}
import features.linguistic_analysis
features.linguistic_analysis.LinguisticAnalyzer = MagicMock()
features.linguistic_analysis.LinguisticAnalyzer.return_value.finalize.return_value = {}

from utils.video_utils import analyze_video
import utils.video_utils
utils.video_utils.librosa = mock_librosa

logging.getLogger().setLevel(logging.ERROR)

videos = ["fhappy.mp4", "fneutral.mp4", "fsad.mp4"]
lines = []
lines.append(f"{'Video':<15} | {'score':>6} | {'happy':>6} | {'neutral':>7} | {'low_pos':>7} | {'mean_sr':>7}")
lines.append("-" * 65)

for vid in videos:
    path = os.path.join("uploads", vid)
    if not os.path.exists(path):
        lines.append(f"{vid}: FILE NOT FOUND")
        continue
    try:
        r = analyze_video(path, transcript="test", debug_mode=True)
        e = r["results"]["multimodal_analysis"].get("expression_analysis", {})
        lines.append(f"{vid:<15} | {e.get('expression_score','?'):>6} | {e.get('happy_ratio','?'):>6} | {e.get('neutral_ratio','?'):>7} | {e.get('low_positive_ratio','?'):>7} | {e.get('mean_smile_ratio','?'):>7}")
    except Exception as ex:
        import traceback
        lines.append(f"{vid}: ERROR - {ex}\n{traceback.format_exc()}")

with open("expr_v2_results.txt", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")
        print(line, flush=True)

print("\nDONE", flush=True)
