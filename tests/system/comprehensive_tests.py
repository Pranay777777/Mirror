"""comprehensive_tests.py — Runs a battery of checks across all subsystems."""
import sys, json
from dotenv import load_dotenv
import os
load_dotenv()

TESTS = []

def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator

@test("imports")
def test_imports():
    from utils.video_utils import analyze_video
    from utils.audio_utils import process_audio
    from utils.scoring_utils import score_audio
    return True

@test("normalized_geometry_instantiation")
def test_geom():
    from features.normalized_geometry import NormalizedGeometry
    g = NormalizedGeometry()
    return hasattr(g, "process")

@test("temporal_features_instantiation")
def test_temp():
    from features.temporal_features import TemporalFeatures
    t = TemporalFeatures()
    return hasattr(t, "finalize")

def run():
    passed = failed = 0
    for name, fn in TESTS:
        try:
            result = fn()
            status = "PASS" if result else "FAIL"
            if result: passed += 1
            else: failed += 1
        except Exception as e:
            status = f"ERROR: {e}"
            failed += 1
        print(f"  [{status}] {name}")
    print(f"\nResults: {passed} passed, {failed} failed")

if __name__ == "__main__":
    run()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
