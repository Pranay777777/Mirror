import requests
import json

path = 'uploads/EYE2.mp4'
url = 'http://127.0.0.1:8000/analyze'

print(f"Testing: {path}")
with open(path, 'rb') as f:
    files = {'file': ('EYE2.mp4', f, 'video/mp4')}
    resp = requests.post(url, files=files, timeout=600)

r = resp.json()
if "results" not in r:
    print("WARNING: API returned no results:")
    print(json.dumps(r, indent=2))
    exit(1)

res = r.get('results', {})
ma = res.get('multimodal_analysis', {})
body = ma.get('body', {})
speech = res.get('speech_score', {})
final = ma.get('final_score', 'N/A')

transcript_obj = res.get('transcript', {})
if isinstance(transcript_obj, dict):
    transcript_text = transcript_obj.get('text', '')
else:
    transcript_text = str(transcript_obj)

print("\n=== VALIDATION RESULTS ===")
print(f"engagement_score (0-100): {body.get('engagement_score', 'N/A')}")
print(f"head_orientation_score (0-100): {body.get('head_orientation_score', 'N/A')}")
print(f"body_total (0-100): {body.get('body_total', 'N/A')}")
print(f"final_score (0-100): {final}")
print(f"transcript word_count: {len(transcript_text.split()) if transcript_text.strip() else 0}")
print(f"transcript: '{transcript_text[:100]}'")

# Internal analysis block (debug)
ma = res.get('multimodal_analysis', {})
ea = ma.get('engagement_analysis', {})
sa = ma.get('speech_analysis', {})
print(f"\n--- Internal ---")
print(f"engagement_analysis.score (0-10): {ea.get('score', 'N/A')}")
print(f"engagement_analysis.interpretation: {ea.get('interpretation', 'N/A')}")

sv = res.get('summary_view', {})
print(f"summary_view.engagement_score (0-10): {sv.get('engagement_score', 'N/A')}")

# Speech metrics
sm = sa.get('metrics', {}) if sa else {}
print(f"\n--- Speech ---")
print(f"word_count: {sm.get('word_count', 'N/A')}")
print(f"words_per_minute: {sm.get('words_per_minute', 'N/A')}")

# Audio metrics for voiced_ratio
af = sa.get('audio_features', {}) if sa else {}
am = af.get('metrics', {}) if af else {}
print(f"voiced_ratio: {am.get('voiced_ratio', 'N/A')}")
print(f"speech_total (0-100): {speech.get('speech_total', 'N/A')}")
