import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import analyze_video

class TestErrorObjects(unittest.TestCase):
    @patch('utils.video_utils.cv2.VideoCapture')
    @patch('utils.video_utils.NormalizedGeometry')
    @patch('utils.video_utils.TemporalFeatures')
    @patch('utils.video_utils.AudioAnalyzer')
    @patch('utils.video_utils.SpeechMetrics')
    @patch('utils.video_utils.LinguisticAnalyzer')
    @patch('utils.video_utils.HeadPoseMetrics')
    @patch('utils.video_utils.librosa.load')
    def test_error_objects(self, mock_load, mock_head, mock_ling, mock_speech, mock_audio, mock_temp, mock_geo, mock_cap):
        # 1. Setup mocks to FAIL (return None/Empty)
        mock_cap_instance = MagicMock()
        mock_cap.return_value = mock_cap_instance
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.read.side_effect = [(True, MagicMock()), (False, None)]
        mock_cap_instance.get.return_value = 30.0

        mock_temp_instance = MagicMock()
        mock_temp.return_value = mock_temp_instance
        mock_temp_instance.extract_temporal_features.return_value = {} # Empty temporal
        
        # Audio failure
        mock_load.side_effect = Exception("Audio load failed")
        
        # Linguistic setup (to avoid crash)
        mock_ling_instance = MagicMock()
        mock_ling.return_value = mock_ling_instance
        mock_ling_instance.analyze_answer_structure.return_value = {}

        # Speech metrics return empty
        mock_speech_instance = MagicMock()
        mock_speech.return_value = mock_speech_instance
        mock_speech_instance.analyze_speaking_rate.return_value = {}
        mock_speech_instance.analyze_fillers.return_value = {}
        mock_speech_instance.analyze_pauses.return_value = {}
        
        # Head pose returns empty
        mock_head_instance = MagicMock()
        mock_head.return_value = mock_head_instance
        mock_head_instance.aggregate_metrics.return_value = {}

        # Run analysis
        result = analyze_video("test.mp4")
        payload = result["results"]["multimodal_analysis"]
        
        # Verify Audio Error Object
        self.assertIn("vocal_diagnostics", payload)
        self.assertEqual(payload["vocal_diagnostics"]["error"], True)
        self.assertEqual(payload["vocal_diagnostics"]["failed_module"], "audio_analysis")
        
        # Verify Speech Error Object
        self.assertIn("speech_analysis", payload)
        self.assertEqual(payload["speech_analysis"]["error"], True)
        self.assertEqual(payload["speech_analysis"]["failed_module"], "speech_metrics")
        
        # Verify Head Pose Error Object
        self.assertIn("head_pose_diagnostics", payload)
        self.assertEqual(payload["head_pose_diagnostics"]["error"], True)
        self.assertEqual(payload["head_pose_diagnostics"]["failed_module"], "head_pose_metrics")

        print("✅ Error objects verified successfully!")

if __name__ == '__main__':
    unittest.main()
