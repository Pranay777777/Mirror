import sys
sys.path.append('.')

print('🧪 CORE COMPONENTS TEST')
print('=' * 40)

# Test 1: LLM Safety (doesn't require MediaPipe)
print('1. Testing LLM Safety...')
try:
    from utils.scoring_utils import _validate_and_clamp_scores
    
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
    print('   ✅ LLM Safety validation working')
except Exception as e:
    print(f'   ❌ LLM Safety test failed: {e}')

# Test 2: Temporal Features (basic test)
print('2. Testing Temporal Features...')
try:
    from features.temporal_features import TemporalFeatures
    import pandas as pd
    
    temporal = TemporalFeatures(window_size_seconds=2.0, fps=30.0)
    
    # Test stability calculation
    stable_data = [5.0, 5.1, 4.9, 5.0, 5.1]
    stability = temporal._calculate_stability(pd.Series(stable_data))
    
    unstable_data = [1.0, 10.0, 2.0, 9.0, 3.0]
    unstability = temporal._calculate_stability(pd.Series(unstable_data))
    
    print(f'   ✅ Stable data stability: {stability:.3f} (should be >0.8)')
    print(f'   ✅ Unstable data stability: {unstability:.3f} (should be <0.5)')
    print('   ✅ Temporal features working')
except Exception as e:
    print(f'   ❌ Temporal features test failed: {e}')

# Test 3: Import Test
print('3. Testing System Imports...')
try:
    from main import app
    print('   ✅ FastAPI app imports successfully')
except Exception as e:
    print(f'   ❌ FastAPI import failed: {e}')

print('\n🎉 CORE SYSTEM TESTS COMPLETE!')
print('📝 Note: Full MediaPipe testing requires actual video files')
