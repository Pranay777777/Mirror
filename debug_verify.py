import sys
import os
from unittest.mock import MagicMock, patch

# Pre-mock mediapipe to avoid loading TensorFlow
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.solutions'] = MagicMock()
sys.modules['mediapipe.solutions.holistic'] = MagicMock()
sys.modules['mediapipe.solutions.face_mesh'] = MagicMock()

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@patch('utils.video_utils.cv2')
@patch('utils.video_utils.NormalizedGeometry')
@patch('utils.video_utils.TemporalFeatures')
@patch('utils.video_utils.AudioAnalyzer')
@patch('utils.video_utils.SpeechMetrics')
@patch('utils.video_utils.LinguisticAnalyzer')
@patch('utils.video_utils.HeadPoseMetrics')
@patch('utils.video_utils.librosa')
def run_debug(mock_librosa, mock_head, mock_ling, mock_speech, mock_audio, mock_temp, mock_geo, mock_cv2):
    print("Setting up mocks...")
    
    # Setup common mocks
    mock_cap = MagicMock()
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
    mock_cap.get.return_value = 30.0
    mock_librosa.load.return_value = (MagicMock(), 16000)

    mock_ling.return_value.analyze_answer_structure.return_value = {}
    mock_speech.return_value.analyze_speaking_rate.return_value = {}
    mock_speech.return_value.analyze_fillers.return_value = {}
    mock_speech.return_value.analyze_pauses.return_value = {}
    mock_audio.return_value.analyze_vocal_features.return_value = {'metrics': {}, 'reliability_score': 0.8}

    mock_temp_instance = MagicMock()
    mock_temp.return_value = mock_temp_instance

    mock_head_instance = MagicMock()
    mock_head.return_value = mock_head_instance

    def check_structure():
        print("Checking Response Structure...")
        # Reset mocks
        mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
        
        # Mock temporal results
        mock_temp_instance.extract_temporal_features.return_value = {
            "posture_stability": {"value": 0.9, "confidence": 1.0, "reason": "test"},
            "posture_uprightness": {"value": 0.9, "confidence": 1.0, "reason": "test"},
            "posture_confidence": {"value": 0.9},
            "data_completeness": {"value": 1.0},
            "overall_movement_intensity": {"value": 0.0},
            "gaze_stability": {"value": 0.9},
            "eye_contact_consistency": {"value": 0.9},
            "expression_dynamics": {"value": 0.5},
            "smile_intensity": {"value": 0.5},
            "expression_variability": {"value": 0.5},
            "blink_rate": {"value": 15},
            "gaze_confidence": {"value": 0.9}
        }
        mock_head_instance.aggregate_metrics.return_value = {'roll_std': 5}
        
        with patch('utils.video_utils.multimodal_confidence_fusion', return_value=0.9):
            # 1. Production Mode
            print("\n--- Production Mode ---")
            mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
            from utils.video_utils import analyze_video
            result_prod = analyze_video("dummy.mp4", debug_mode=False)
            payload_prod = result_prod["results"]["multimodal_analysis"]
            
            if "temporal_features" in payload_prod:
                print("FAIL: temporal_features present in production output!")
            else:
                print("PASS: temporal_features absent")
                
            if "vocal_diagnostics" in payload_prod:
                print("FAIL: vocal_diagnostics present in production output!")
            else:
                print("PASS: vocal_diagnostics absent")
                
            posture = payload_prod["posture_analysis"]
            if "reason" in posture["stability_score"]:
                print("FAIL: reason field present in posture stability!")
            else:
                print("PASS: reason field cleaned")
                
            # 2. Debug Mode
            print("\n--- Debug Mode ---")
            mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
            result_debug = analyze_video("dummy.mp4", debug_mode=True)
            payload_debug = result_debug["results"]["multimodal_analysis"]
            
            if "temporal_features" in payload_debug:
                 print("PASS: temporal_features present in debug output")
            else:
                 print("FAIL: temporal_features MISSING in debug output")

    check_structure()

if __name__ == "__main__":
    run_debug()
