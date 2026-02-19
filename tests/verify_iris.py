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

videos = ["EYE1.mp4", "EYE2.mp4", "EYE3.mp4", "eye5.mp4", "eye7.mp4"]
lines = []

for vid in videos:
    path = os.path.join("uploads", vid)
    if not os.path.exists(path):
        lines.append(f"{vid}: FILE NOT FOUND")
        continue
    try:
        r = analyze_video(path, transcript="test", debug_mode=True)
        mm = r["results"]["multimodal_analysis"]
        pm = mm["posture_analysis"]["metrics"]
        ecc = pm.get("eye_contact_consistency", {}).get("value", "N/A")
        diag = pm.get("eye_contact_consistency", {}).get("diagnostics", {})
        interp = mm["engagement_analysis"].get("interpretation", "N/A")
        ecr_s = diag.get("ECR_strong", "?")
        ecr_soft = diag.get("ECR_soft", "?")
        lines.append(f"{vid}: ECR_soft={ecr_soft}  ECR_strong={ecr_s}  Interp={interp}")
    except Exception as e:
        import traceback
        lines.append(f"{vid}: ERROR - {e}\n{traceback.format_exc()}")

with open("v3_results.txt", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

print("DONE", flush=True)
