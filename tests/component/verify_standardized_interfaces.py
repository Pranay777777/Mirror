"""verify_standardized_interfaces.py — Confirms all analyzers have add_frame and finalize methods."""
from dotenv import load_dotenv
load_dotenv()

REQUIRED_INTERFACE = ["add_frame", "finalize"]

CLASSES = [
    ("features.temporal_features", "TemporalFeatures"),
    ("features.audio_analysis",    "AudioAnalyzer"),
    ("features.speech_metrics",    "SpeechMetrics"),
    ("features.linguistic_analysis","LinguisticAnalyzer"),
    ("features.head_pose_metrics", "HeadPoseMetrics"),
]

def main():
    print("[STANDARDIZED INTERFACES]")
    for mod_name, cls_name in CLASSES:
        mod = __import__(mod_name, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        instance = cls()
        for method in REQUIRED_INTERFACE:
            ok = hasattr(instance, method)
            print(f"  {'✓' if ok else '✗'} {cls_name}.{method}")

if __name__ == "__main__":
    main()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
