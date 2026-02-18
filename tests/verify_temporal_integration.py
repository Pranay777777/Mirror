import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

# Mocks
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.solutions'] = MagicMock()
sys.modules['mediapipe.solutions.holistic'] = MagicMock()
sys.modules['mediapipe.solutions.face_mesh'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['librosa'] = MagicMock()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestTemporalIntegration(unittest.TestCase):
    
    @patch('utils.video_utils.cv2') 
    @patch('utils.video_utils.NormalizedGeometry')
    @patch('utils.video_utils.TemporalFeatures')
    @patch('utils.video_utils.AudioAnalyzer')
    @patch('utils.video_utils.SpeechMetrics')
    @patch('utils.video_utils.LinguisticAnalyzer')
    @patch('utils.video_utils.HeadPoseMetrics')
    @patch('utils.video_utils.librosa')
    def test_analyze_video_integration(self, mock_librosa, mock_head, mock_ling, mock_speech, mock_audio, mock_temp_cls, mock_geo_cls, mock_cv2):
        
        # Setup mocks
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        # Return 2 frames to test loop and timestamp increment
        mock_cap.read.side_effect = [(True, MagicMock()), (True, MagicMock()), (False, None)]
        mock_cap.get.return_value = 30.0 # FPS
        mock_librosa.load.return_value = (MagicMock(), 16000)
        
        # Mock Instances
        mock_geo = mock_geo_cls.return_value
        mock_temp = mock_temp_cls.return_value
        
        # NormalizedGeometry.analyze_frame returns a dict (simulated)
        dummy_features = {'some_feature': 0.5, 'head_pose': (0,0,0)}
        mock_geo.analyze_frame.return_value = dummy_features
        
        # TemporalFeatures.extract_temporal_features must return a dict-like structure
        # to prevent scoring logic from crashing on MagicMock comparisons/conversions
        mock_temp.extract_temporal_features.return_value = {
            "alignment_integrity": {"value": 0.8},
            "stability_index": {"value": 0.8},
            "eye_contact_consistency": {"value": 0.8},
            "gaze_stability": {"value": 0.8},
            "smile_intensity": {"value": 0.5},
            "posture_confidence": {"value": 0.9}
        }
        
        # Call the function
        from utils.video_utils import analyze_video
        
        print("Running analyze_video integration test...")
        try:
            analyze_video("dummy.mp4")
        except Exception as e:
            self.fail(f"analyze_video crashed: {e}")
            
        # VERIFY the call to add_frame_features
        # Expecting: add_frame_features(geo_results, geo_results, timestamp)
        # Check call args
        self.assertTrue(mock_temp.add_frame_features.called, "add_frame_features was not called")
        
        # Check arguments of the first call
        args, _ = mock_temp.add_frame_features.call_args_list[0]
        # args should be (features, features, timestamp)
        self.assertEqual(len(args), 3, "add_frame_features incorrectly called (wrong arg count)")
        self.assertEqual(args[0], dummy_features)
        self.assertEqual(args[1], dummy_features)
        self.assertIsInstance(args[2], float) # timestamp
        
        print("Verified add_frame_features called with 3 arguments.")

if __name__ == '__main__':
    unittest.main()
