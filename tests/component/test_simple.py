"""
test_simple.py — Minimal smoke test: imports all modules and checks they load correctly.
Usage: python test_simple.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
def main():
    errors = []
    modules = [
        ("utils.video_utils", "analyze_video"),
        ("utils.audio_utils", "process_audio"),
        ("utils.scoring_utils", "score_audio"),
        ("features.normalized_geometry", "NormalizedGeometry"),
        ("features.temporal_features", "TemporalFeatures"),
        ("features.audio_analysis", "AudioAnalyzer"),
        ("features.speech_metrics", "SpeechMetrics"),
        ("features.linguistic_analysis", "LinguisticAnalyzer"),
        ("features.head_pose_metrics", "HeadPoseMetrics"),
    ]
    for mod, attr in modules:
        try:
            m = __import__(mod, fromlist=[attr])
            getattr(m, attr)
            print(f"  ✓  {mod}.{attr}")
        except Exception as e:
            print(f"  ✗  {mod}.{attr} — {e}")
            errors.append((mod, str(e)))

    if errors:
        print(f"\n[FAILED] {len(errors)} import(s) failed.")
    else:
        print("\n[PASSED] All modules imported successfully.")

if __name__ == "__main__":
    main()
