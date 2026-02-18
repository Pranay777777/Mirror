import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.solutions'] = MagicMock()
sys.modules['mediapipe.solutions.holistic'] = MagicMock()
sys.modules['mediapipe.solutions.face_mesh'] = MagicMock()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestPostureStartIgnore(unittest.TestCase):
    
    @patch('utils.video_utils.cv2') # Bottom arg
    @patch('utils.video_utils.NormalizedGeometry')
    @patch('utils.video_utils.TemporalFeatures')
    @patch('utils.video_utils.AudioAnalyzer')
    @patch('utils.video_utils.SpeechMetrics')
    @patch('utils.video_utils.LinguisticAnalyzer')
    @patch('utils.video_utils.HeadPoseMetrics')
    @patch('utils.video_utils.librosa') # Top arg
    def test_start_ignore_logic(self, mock_librosa, mock_head, mock_ling, mock_speech, mock_audio, mock_temp, mock_geo, mock_cv2):
        # Setup mocks
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.side_effect = [True, False]
        mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
        mock_cap.get.return_value = 30.0
        mock_librosa.load.return_value = (MagicMock(), 16000)
        
        # FIX: Mock content analysis
        mock_ling.return_value.analyze_answer_structure.return_value = {}
        mock_ling.return_value.analyze_content.return_value = {}
        
        mock_head.return_value.aggregate_metrics.return_value = {'roll_std': 5.0} 
        mock_audio.return_value.analyze_vocal_features.return_value = {'metrics': {}, 'reliability_score': 0.8}
        
        # Configure Geometry to avoid crash 
        mock_geo_instance = MagicMock()
        mock_geo.return_value = mock_geo_instance
        mock_geo_instance.analyze_frame.return_value = {
            'face_detection_confidence': 0.9,
            'face_size_ratio': 0.3,
            'head_pose': {'roll': 0, 'pitch': 0, 'yaw': 0},
            'face_landmarks': [],
            'mp_results': MagicMock()
        }
        
        # FIX: Configure TemporalFeatures to return valid dict
        mock_temp_instance = MagicMock()
        mock_temp.return_value = mock_temp_instance
        # Minimum required structure
        mock_temp_instance.extract_temporal_features.return_value = {
            "alignment_integrity": {"value": 0.9},
            "stability_index": {"value": 0.9},
            "motion_activity_level": {"value": 0.0},
            "posture_confidence": {"value": 0.9},
            "data_completeness": {"value": 1.0},
            "gaze_stability": {"value": 0.9},
            "eye_contact_consistency": {"value": 0.9},
            "blink_rate": {"value": 15},
            "gaze_confidence": {"value": 0.9},
            "expression_dynamics": {"value": 0.5},
            "smile_intensity": {"value": 0.5},
            "expression_variability": {"value": 0.5}
        }

        # Verify arguments passed to TemporalFeatures constructor
        from utils.video_utils import analyze_video
        
        # Run with default
        print("Running analyze_video (default)...")
        analyze_video("dummy.mp4")
        # Check constructor call (mock_temp is the class)
        # analyze_video default is 1000
        mock_temp.assert_called_with(fps=30.0, ignore_start_ms=1000)
        
        # Run with explicit
        print("Running analyze_video (explicit)...")
        mock_cap.isOpened.side_effect = [True, False] 
        mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
        analyze_video("dummy.mp4", ignore_start_ms=5000)
        mock_temp.assert_called_with(fps=30.0, ignore_start_ms=5000)
        
        print("Verified TemporalFeatures instantiated with correct ignore_start_ms.")

if __name__ == '__main__':
    unittest.main()
