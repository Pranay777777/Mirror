import sys, os, logging
import numpy as np
import pandas as pd
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
lines.append(f"{'Video':<10} | {'p90_cd':>10} | {'p95_cd':>10} | {'max_cd':>10} | {'mean_cd':>10} | {'std_rx':>10}")
lines.append("-" * 75)

for vid in videos:
    path = os.path.join("uploads", vid)
    if not os.path.exists(path):
        lines.append(f"{vid}: FILE NOT FOUND")
        continue
    try:
        r = analyze_video(path, transcript="test", debug_mode=True)
        # Extract raw face_history from temporal features to get iris_ratio_x series
        mm = r["results"]["multimodal_analysis"]
        pm = mm["posture_analysis"]["metrics"]
        diag = pm.get("eye_contact_consistency", {}).get("diagnostics", {})

        # We need raw frame data - re-extract from the analyzer
        # Instead, let's compute from a fresh run using just the geometry module
        from features.normalized_geometry import NormalizedGeometry
        from features.temporal_features import TemporalFeatures
        import cv2

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ng = NormalizedGeometry()
        ratios = []

        import mediapipe as mp
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            if results.face_landmarks:
                ir = ng.compute_iris_ratio(results.face_landmarks)
                if ir is not None:
                    ratios.append(ir)

        cap.release()
        holistic.close()

        if ratios:
            s = pd.Series(ratios)
            cd = (s - 0.5).abs()
            p90 = cd.quantile(0.90)
            p95 = cd.quantile(0.95)
            mx = cd.max()
            mn = cd.mean()
            std = s.std()
            lines.append(f"{vid:<10} | {p90:10.4f} | {p95:10.4f} | {mx:10.4f} | {mn:10.4f} | {std:10.4f}")
        else:
            lines.append(f"{vid:<10} | NO IRIS DATA")

    except Exception as e:
        import traceback
        lines.append(f"{vid}: ERROR - {e}")

with open("iris_percentile_stats.txt", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")
        print(line, flush=True)

print("\nDONE", flush=True)
