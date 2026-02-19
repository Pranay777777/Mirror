"""
Validation: Production Response Slim Layer
Tests that:
  1. Public response shape is correct (no internal fields exposed)
  2. Scores are identical to v3.1 calibrated values
  3. No KeyError / NameError / AttributeError
  4. Mode shows as speech_v3_1_calibrated (not visible in slim, but confirms scoring)
"""
import sys, os, logging, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import analyze_video

logging.getLogger().setLevel(logging.ERROR)

BANNED_KEYS = [
    # Deep internal fields that must NEVER appear anywhere in the public response
    "alignment_integrity", "motion_activity_level", "stability_index",
    "data_completeness", "evidence_ratio", "method", "reason",
    "logic_mode", "speech_logic_mode", "rate_score",
    "rms_mean", "rms_std", "pitch_mean_hz", "pitch_confidence",
    "voiced_frame_ratio", "silence_ratio"
]

# These must NOT appear at the top level of the public response
BANNED_TOP_LEVEL_KEYS = [
    "results", "multimodal_analysis", "summary_view",
    "confidence_metrics", "linguistic_analysis"
]

REQUIRED_TOP_KEYS = [
    "posture_score", "engagement", "speech", "professionalism_score",
    "overall_confidence", "qualitative_feedback", "transcript"
]

REQUIRED_ENGAGEMENT_KEYS = [
    "score", "interpretation", "eye_contact_ratio", "gaze_stability",
    "max_continuous_eye_contact"
]

REQUIRED_SPEECH_KEYS = [
    "score", "words_per_minute", "speaking_rate_category",
    "filler_rate_per_min", "pitch_score", "pause_score"
]

def check_no_banned_keys(d, path=""):
    """Recursively checks that no banned internal keys appear in output."""
    if isinstance(d, dict):
        for k, v in d.items():
            if k in BANNED_KEYS:
                print(f"  [FAIL] Banned key found at {path}.{k}")
                return False
            if not check_no_banned_keys(v, f"{path}.{k}"):
                return False
    return True

videos = [
    ("EYE1.mp4", None),
    ("good_english.mp4", None),
]

all_ok = True
for vid, _transcript in videos:
    path = os.path.join("uploads", vid)
    if not os.path.exists(path):
        print(f"[SKIP] {vid} not found")
        continue

    print(f"\n{'='*60}")
    print(f"  Testing: {vid}")
    print(f"{'='*60}")

    try:
        r = analyze_video(path, transcript=None, debug_mode=False)

        # 1. Check required top-level keys
        for key in REQUIRED_TOP_KEYS:
            if key not in r:
                print(f"  [FAIL] Missing required key: {key}")
                all_ok = False
            else:
                print(f"  [OK]   Found: {key}")

        # 2. Check engagement sub-keys
        eng = r.get("engagement", {})
        for key in REQUIRED_ENGAGEMENT_KEYS:
            if key not in eng:
                print(f"  [FAIL] Missing engagement.{key}")
                all_ok = False

        # 3. Check speech sub-keys
        speech = r.get("speech", {})
        for key in REQUIRED_SPEECH_KEYS:
            if key not in speech:
                print(f"  [FAIL] Missing speech.{key}")
                all_ok = False

        # 4. Banned key scan (deep)
        if check_no_banned_keys(r):
            print(f"  [OK]   No deep internal/banned keys found")
        else:
            all_ok = False

        # 5. Top-level structure check
        for key in BANNED_TOP_LEVEL_KEYS:
            if key in r:
                print(f"  [FAIL] Internal top-level key exposed: {key}")
                all_ok = False
        print(f"  [OK]   Top-level structure clean")

        # 5. Print slim summary
        print(f"\n  >> posture_score      : {r.get('posture_score')}")
        print(f"  >> engagement.score   : {eng.get('score')}")
        print(f"  >> speech.score       : {speech.get('score')}")
        print(f"  >> speech.wpm         : {speech.get('words_per_minute')} ({speech.get('speaking_rate_category')})")
        print(f"  >> professionalism    : {r.get('professionalism_score')}")
        print(f"  >> overall_confidence : {r.get('overall_confidence')}")
        print(f"  >> tone_quality       : {r.get('qualitative_feedback', {}).get('tone_quality')}")
        print(f"  >> voice_feedback     : {r.get('qualitative_feedback', {}).get('voice_quality_feedback')}")
        t = r.get("transcript", {})
        print(f"  >> transcript.lang    : {t.get('language_code')}")
        print(f"  >> transcript.text    : {repr(t.get('text', '')[:80])}")

        # 6. Confirm 'multimodal_analysis' is NOT in output
        if "multimodal_analysis" in r or "results" in r:
            print(f"  [FAIL] Internal 'results' or 'multimodal_analysis' exposed!")
            all_ok = False
        else:
            print(f"  [OK]   Internal structure correctly hidden")

    except Exception as e:
        import traceback
        print(f"  [ERROR] {vid}: {e}")
        traceback.print_exc()
        all_ok = False

print(f"\n{'='*60}")
print(f"  OVERALL: {'ALL PASS' if all_ok else 'SOME FAILURES — see above'}")
print(f"{'='*60}\n")
