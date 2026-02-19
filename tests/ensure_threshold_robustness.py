import sys
import os
import logging
import numpy as np
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock Librosa BEFORE any imports that use it
mock_librosa = MagicMock()
mock_librosa.load.return_value = (np.zeros(16000), 16000)
sys.modules["librosa"] = mock_librosa

# Mock Audio components at module level  
import features.audio_analysis
features.audio_analysis.AudioAnalyzer = MagicMock()
features.audio_analysis.AudioAnalyzer.return_value.finalize.return_value = {
    "metrics": {}, "confidence": 0.0, "reliability_score": 0.0, "diagnostics": {}
}

import features.speech_metrics
features.speech_metrics.SpeechMetrics = MagicMock()
features.speech_metrics.SpeechMetrics.return_value.finalize.return_value = {
    "metrics": {}, "confidence": 0.0
}

import features.linguistic_analysis
features.linguistic_analysis.LinguisticAnalyzer = MagicMock()
features.linguistic_analysis.LinguisticAnalyzer.return_value.finalize.return_value = {}

from utils.video_utils import analyze_video
import utils.video_utils
# Patch librosa reference that was already bound during import
utils.video_utils.librosa = mock_librosa

def verify_robustness():
    videos = [
        ("EYE1.mp4", "Good"),   # Expect >= 0.75 -> Good
        ("eye5.mp4", "Bad"),    # Expect < 0.40 -> Limited
        ("eye7.mp4", "Bad"),    # Expect < 0.40 -> Limited
    ]
    
    print(f"\nTHRESHOLD ROBUSTNESS CHECK", flush=True)
    print(f"  Bands: GOOD >= 0.75 | MODERATE 0.40-0.75 | BAD < 0.40", flush=True)
    print(f"{'VIDEO':<10} | {'CONSIS':<8} | {'STAB':<8} | {'MEAN_GOOD':<10} | {'MEAN_BAD':<10} | {'INTERP':<30} | {'RESULT'}", flush=True)
    print("-" * 115, flush=True)

    all_passed = True

    for filename, expected_type in videos:
        video_path = os.path.join("uploads", filename)
        if not os.path.exists(video_path):
            print(f"{filename:<10} | FILENOTFOUND", flush=True)
            continue
            
        try:
            logging.getLogger().setLevel(logging.ERROR)
            result = analyze_video(video_path, transcript="test", debug_mode=True)
            
            # Navigate response structure safely
            results = result.get('results', {})
            multimodal = results.get('multimodal_analysis', {})
            
            # Engagement analysis contains interpretation
            engagement = multimodal.get('engagement_analysis', {})
            interp = engagement.get('interpretation', 'N/A')
            
            # Posture metrics contain the temporal results (where eye_contact_consistency lives)
            posture_metrics = multimodal.get('posture_analysis', {}).get('metrics', {})
            
            # Eye contact consistency
            ecc_obj = posture_metrics.get('eye_contact_consistency', {})
            consistency = ecc_obj.get('value', 0.0)
            if consistency is None: consistency = 0.0
            
            # Gaze stability
            gs_obj = posture_metrics.get('gaze_stability', {})
            stability = gs_obj.get('value', 0.0)
            if stability is None: stability = 0.0
            
            # Diagnostics (distribution stats)
            diag = ecc_obj.get('diagnostics', {})
            mean_good = diag.get('mean_good_gaze', -1.0)
            mean_bad = diag.get('mean_bad_gaze', -1.0)
            
            # Validation
            passed = False
            if expected_type == "Good":
                passed = (consistency >= 0.75) and ("Good" in interp)
            elif expected_type == "Bad":
                passed = (consistency < 0.40) and ("Limited" in interp)
            
            status = "PASS" if passed else "FAIL"
            if not passed: all_passed = False
            
            print(f"{filename:<10} | {consistency:<8.4f} | {stability:<8.4f} | {mean_good:<10.2f} | {mean_bad:<10.2f} | {interp:<30} | {status}", flush=True)

        except Exception as e:
            import traceback
            print(f"{filename:<10} | ERROR: {e}", flush=True)
            traceback.print_exc()
            all_passed = False

    print("-" * 115, flush=True)
    if all_passed:
        print("SUCCESS: All thresholds verified.", flush=True)
    else:
        print("FAILURE: Threshold calibration needs adjustment.", flush=True)

if __name__ == "__main__":
    verify_robustness()
