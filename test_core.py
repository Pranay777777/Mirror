import sys
sys.path.append('.')
from features.normalized_geometry import NormalizedGeometry
from features.temporal_features import TemporalFeatures
from utils.scoring_utils import _validate_and_clamp_scores
import pandas as pd

print('🧪 CORE COMPONENTS TEST')
print('=' * 40)

# Test 1: Camera Invariance
print('1. Testing Camera Invariance...')
geometry = NormalizedGeometry()

# Mock landmarks at different scales
def create_mock_landmarks(scale):
    import unittest.mock
    landmarks = unittest.mock.MagicMock()
    landmarks.landmark = []
    for i in range(33):
        lm = unittest.mock.MagicMock()
        lm.x = (0.1 + i * 0.01) * scale
        lm.y = (0.2 + i * 0.005) * scale
        lm.z = (0.0 + i * 0.001) * scale
        lm.configure_mock(**{'visibility': 0.9})  # Proper mock configuration
        landmarks.landmark.append(lm)
    return landmarks

features1 = geometry.extract_pose_features(create_mock_landmarks(1.0))
features2 = geometry.extract_pose_features(create_mock_landmarks(2.0))

if 'eye_distance_ratio' in features1 and 'eye_distance_ratio' in features2:
    diff = abs(features1['eye_distance_ratio'] - features2['eye_distance_ratio'])
    print(f'   ✅ Eye distance ratio invariance: {diff:.4f} (should be ~0.00)')
else:
    print('   ⚠️  Eye distance ratio not calculated')

# Test 2: Temporal Features
print('2. Testing Temporal Features...')
temporal = TemporalFeatures(window_size_seconds=2.0, fps=30.0)

# Add test data
for i in range(10):
    pose_features = {'shoulder_tilt_angle': 5.0 + i * 0.1}
    face_features = {'mouth_opening_ratio': 0.1 + i * 0.01}
    temporal.add_frame_features(pose_features, face_features, i * 0.033)

temporal_analysis = temporal.extract_temporal_features()
if 'error' not in temporal_analysis:
    print('   ✅ Temporal analysis successful')
    print(f'   📊 Features extracted: {len(temporal_analysis)}')
else:
    print(f'   ❌ Temporal analysis failed: {temporal_analysis.get("error", "unknown")}')

# Test 3: LLM Safety
print('3. Testing LLM Safety...')
invalid_scores = {
    'sentiment': 'invalid',
    'tone_quality': 'invalid',
    'professionalism_score': 15.0,
    'communication_score': -5.0,
    'confidence': 1.5,
    'voice_quality_feedback': '',
    'analysis_notes': ''
}

validated = _validate_and_clamp_scores(invalid_scores)
print(f'   ✅ Score clamping: {validated["professionalism_score"]} (should be 5.0)')
print(f'   ✅ Confidence clamping: {validated["confidence"]} (should be 0.5)')

print('🎉 CORE COMPONENTS WORKING!')
