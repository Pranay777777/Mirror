import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.solutions'] = MagicMock()
sys.modules['mediapipe.solutions.holistic'] = MagicMock()
sys.modules['mediapipe.solutions.face_mesh'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['librosa'] = MagicMock()

# Manually add utils to path to allow direct import if package fails
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import analyze_video

class TestEngagementLogic(unittest.TestCase):
    
    @patch('utils.video_utils.cv2') 
    @patch('utils.video_utils.NormalizedGeometry')
    @patch('utils.video_utils.TemporalFeatures')
    @patch('utils.video_utils.AudioAnalyzer')
    @patch('utils.video_utils.SpeechMetrics')
    @patch('utils.video_utils.LinguisticAnalyzer')
    @patch('utils.video_utils.HeadPoseMetrics')
    @patch('utils.video_utils.librosa') 
    def test_eye_contact_buckets(self, mock_librosa, mock_head, mock_ling, mock_speech, mock_audio, mock_temp, mock_geo, mock_cv2):
        
        # Setup mocks
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
        mock_cap.get.return_value = 30.0
        mock_librosa.load.return_value = (MagicMock(), 16000)
        
        # Mocks to prevent crashes
        mock_ling.return_value.analyze_answer_structure.return_value = {}
        mock_ling.return_value.analyze_content.return_value = {}
        mock_audio.return_value.analyze_vocal_features.return_value = {'metrics': {}, 'reliability_score': 0.8}
        
        mock_geo_instance = MagicMock()
        mock_geo.return_value = mock_geo_instance
        mock_geo_instance.analyze_frame.return_value = {}

        def check_interpretation(consistency_val, expected_str):
            print(f"Testing Consistency: {consistency_val} -> Expect: '{expected_str}'")
            # Reset mocks
            mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
            
            mock_temp_instance = MagicMock()
            mock_temp.return_value = mock_temp_instance
            mock_temp_instance.extract_temporal_features.return_value = {
                # Alignment/Stab required for flow
                "alignment_integrity": {"value": 0.9},
                "stability_index": {"value": 0.9},
                "motion_activity_level": {"value": 0.0},
                
                # Target Metric
                "eye_contact_consistency": {"value": consistency_val},
                
                # Others
                "posture_confidence": {"value": 0.9},
                "gaze_stability": {"value": 0.9},
                "smile_intensity": {"value": 0.5},
            }
            
            with patch('utils.video_utils.multimodal_confidence_fusion', return_value=0.9):
                from utils.video_utils import analyze_video
                result = analyze_video("dummy.mp4", debug_mode=False)
                
                # Access interpretation
                interp = result["results"]["multimodal_analysis"]["engagement_analysis"]["interpretation"]
                self.assertEqual(interp, expected_str)

        # Test Cases
        check_interpretation(0.95, "Strong eye contact")
        check_interpretation(0.85, "Good eye contact")
        check_interpretation(0.65, "Moderate eye contact")
        check_interpretation(0.46, "Limited eye contact")
        check_interpretation(0.40, "Poor eye contact")

if __name__ == '__main__':
    unittest.main()
