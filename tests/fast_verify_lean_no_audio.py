import os
import sys
import logging
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock AudioAnalyzer
import features.audio_analysis
features.audio_analysis.AudioAnalyzer = MagicMock()
features.audio_analysis.AudioAnalyzer.return_value.finalize.return_value = {
    "metrics": {},
    "confidence": 0.0,
    "reliability_score": 0.0,
    "diagnostics": {}
}

# Mock Librosa
import librosa
sys.modules['librosa'] = MagicMock()
sys.modules['librosa'].load.return_value = (None, 16000)

# Mock SpeechMetrics
import features.speech_metrics
features.speech_metrics.SpeechMetrics = MagicMock()
features.speech_metrics.SpeechMetrics.return_value.finalize.return_value = {
    "metrics": {},
    "confidence": 0.0
}

# Mock LinguisticAnalyzer
import features.linguistic_analysis
features.linguistic_analysis.LinguisticAnalyzer = MagicMock()
features.linguistic_analysis.LinguisticAnalyzer.return_value.finalize.return_value = {}

from utils.video_utils import analyze_video

def run_fast_verification():
    videos = ["EYE1.mp4", "eye5.mp4", "eye7.mp4", "pran.mp4"]
    
    print(f"\nFAST VERIFICATION RUN (Lean Binary Model - No Audio - 300 Frame Limit)", flush=True)
    print(f"{'VIDEO':<10} | {'STABILITY':<10} | {'CONSISTENCY':<12} | {'SWITCHES':<10} | {'INTERPRETATION'}", flush=True)
    print("-" * 90, flush=True)

    for video in videos:
        path = os.path.join("uploads", video)
        if not os.path.exists(path):
            print(f"{video:<10} | FILENOTFOUND")
            continue
            
        try:
            print(f"Processing {video}...", flush=True)
            # Suppress logging
            logging.getLogger().setLevel(logging.ERROR)
            
            # Run analysis
            print("Calling analyze_video...", flush=True)
            result = analyze_video(path, debug_mode=True)
            print(f"Analysis complete. Result keys: {list(result.keys()) if result else 'None'}", flush=True)
            
            # Extract Metrics
            metrics = result.get("results", {}).get("multimodal_analysis", {})
            if not metrics:
                 print("No multimodal_analysis found. Checking root metrics...", flush=True)
                 metrics = result.get("results", {}).get("metrics", {})

            print(f"Metrics keys: {list(metrics.keys()) if metrics else 'None'}", flush=True)

            posture = metrics.get("posture_analysis", {}).get("metrics", {})
            gaze_stab = posture.get("gaze_stability", {}).get("value", 0.0)
            
            engagement = metrics.get("engagement_analysis", {})
            eng_metrics = engagement.get("metrics", {})
            consistency = eng_metrics.get("eye_contact", 0.0)
            
            # Consistency might be inside "eye_contact_consistency" object in recent schema?
            # "eye_contact" key in metrics map is simplified.
            # Let's check posture -> eye_contact_consistency -> value
            ecc = posture.get("eye_contact_consistency", {}).get("value", 0.0)
            if ecc > 0: consistency = ecc
            
            switch_count = posture.get("gaze_direction_switch_count", {}).get("value", 0)
            
            interpretation = engagement.get("interpretation", "N/A")
            
            print(f"{video:<10} | {gaze_stab:10.4f} | {consistency:12.4f} | {switch_count:10} | {interpretation}", flush=True)
            
        except Exception as e:
            print(f"{video:<10} | ERROR: {e}", flush=True)

if __name__ == "__main__":
    run_fast_verification()
