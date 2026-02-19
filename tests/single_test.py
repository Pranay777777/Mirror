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

videos = ["EYE1.mp4", "eye5.mp4", "eye7.mp4"]
output_lines = []

for vid in videos:
    path = os.path.join("uploads", vid)
    if not os.path.exists(path):
        output_lines.append(f"{vid}: FILE NOT FOUND")
        continue
    try:
        r = analyze_video(path, transcript="test", debug_mode=True)
        mm = r["results"]["multimodal_analysis"]
        pm = mm["posture_analysis"]["metrics"]
        ecc = pm.get("eye_contact_consistency", {})
        gs = pm.get("gaze_stability", {})
        eng = mm["engagement_analysis"]
        diag = ecc.get("diagnostics", {})
        
        line = (
            f"{vid}: ECR={ecc.get('value', 'N/A')}, "
            f"GSS={gs.get('value', 'N/A')}, "
            f"Interp={eng.get('interpretation', 'N/A')}, "
            f"MeanGood={diag.get('mean_good_gaze', 'N/A')}, "
            f"MeanBad={diag.get('mean_bad_gaze', 'N/A')}, "
            f"StdGaze={diag.get('std_gaze', 'N/A')}"
        )
        output_lines.append(line)
    except Exception as e:
        import traceback
        output_lines.append(f"{vid}: ERROR - {e}")
        output_lines.append(traceback.format_exc())

# Write to file (UTF-8)
with open("threshold_results.txt", "w", encoding="utf-8") as f:
    f.write("THRESHOLD ROBUSTNESS RESULTS\n")
    f.write("Bands: GOOD >= 0.75 | MODERATE 0.40-0.75 | BAD < 0.40\n")
    f.write("=" * 80 + "\n")
    for line in output_lines:
        f.write(line + "\n")

print("Done. Results written to threshold_results.txt", flush=True)
