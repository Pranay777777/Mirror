import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mocks to prevent actual heavy imports
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.solutions'] = MagicMock()
sys.modules['mediapipe.solutions.holistic'] = MagicMock()
sys.modules['mediapipe.solutions.face_mesh'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['librosa'] = MagicMock()

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need utils/scoring_utils.py to be importable or mocked
# It seems present now.

class TestCrashResilience(unittest.TestCase):
    
    @patch('utils.video_utils.cv2') 
    @patch('utils.video_utils.NormalizedGeometry')
    @patch('utils.video_utils.TemporalFeatures')
    @patch('utils.video_utils.AudioAnalyzer')
    @patch('utils.video_utils.SpeechMetrics')
    @patch('utils.video_utils.LinguisticAnalyzer')
    @patch('utils.video_utils.HeadPoseMetrics')
    @patch('utils.video_utils.librosa')
    def test_missing_data_resilience(self, mock_librosa, mock_head, mock_ling, mock_speech, mock_audio, mock_temp, mock_geo, mock_cv2):
        
        # Setup basic strict mocks
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, MagicMock()), (False, None)] # 1 frame
        mock_cap.get.return_value = 30.0
        mock_librosa.load.return_value = (MagicMock(), 16000)
        
        # Mock analyzers avoiding None where critical (Audio/Linguistic need valid return usually)
        mock_ling.return_value.analyze_content.return_value = {} # Empty dict
        mock_audio.return_value.analyze_vocal_features.return_value = {'metrics': {}, 'reliability_score': 0.8}
        
        mock_geo_instance = MagicMock()
        mock_geo.return_value = mock_geo_instance
        mock_geo_instance.analyze_frame.return_value = {}

        # The TEMPORAL ANALYZER is key. It returns the problematic structure.
        mock_temp_instance = MagicMock()
        mock_temp.return_value = mock_temp_instance
        
        # Test Case 1: Partial Data (Missing some keys)
        mock_temp_instance.extract_temporal_features.return_value = {
            "alignment_integrity": {"value": 0.8},
            # Missing stability, motion, eye contact
        }
        
        from utils.video_utils import analyze_video # deferred import
        
        try:
            print("Running Partial Data Test...")
            res = analyze_video("dummy.mp4")
            print("Partial Data Test Passed (No Crash).")
            # Verify basic structure
            self.assertIn("results", res)
            self.assertIn("summary_view", res["results"])
            # verify defaults
            # stability missing -> 0.5 default
            # motion missing -> 0.0 default
            # eye -> 0.5 default
            # engagement -> (0.5*4 + 0.5*3 + 0*3) = 3.5
            self.assertAlmostEqual(res["results"]["summary_view"]["engagement_score"], 3.5)
            
        except Exception as e:
            self.fail(f"Partial Data Test crashed: {e}")

        # Test Case 2: None Values for Keys
        mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
        mock_temp_instance.extract_temporal_features.return_value = {
            "alignment_integrity": {"value": None},
            "stability_index": {"value": None},
            "motion_activity_level": {"value": None},
            "eye_contact_consistency": {"value": None},
            "gaze_stability": {"value": None},
            "smile_intensity": {"value": None},
            "posture_confidence": {"value": None},
        }
        
        try:
            print("Running None Data Test...")
            res = analyze_video("dummy.mp4")
            print("None Data Test Passed (No Crash).")
            # All defaults -> 0.5 or 0.0
            # Posture: (0.5*6 + 0.5*4) = 5.0
            self.assertAlmostEqual(res["results"]["summary_view"]["posture_score"], 5.0)
            
        except Exception as e:
            self.fail(f"None Data Test crashed: {e}")

if __name__ == '__main__':
    unittest.main()
