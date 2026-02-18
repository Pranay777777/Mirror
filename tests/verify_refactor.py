import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import analyze_video

class TestAnalyzeVideoRefactor(unittest.TestCase):
    @patch('utils.video_utils.cv2.VideoCapture')
    @patch('utils.video_utils.NormalizedGeometry')
    @patch('utils.video_utils.TemporalFeatures')
    @patch('utils.video_utils.AudioAnalyzer')
    @patch('utils.video_utils.SpeechMetrics')
    @patch('utils.video_utils.LinguisticAnalyzer')
    @patch('utils.video_utils.HeadPoseMetrics')
    @patch('utils.video_utils.librosa.load')
    def test_analyze_video_structure(self, mock_load, mock_head, mock_ling, mock_speech, mock_audio, mock_temp, mock_geo, mock_cap):
        # Setup mocks
        mock_cap_instance = MagicMock()
        mock_cap.return_value = mock_cap_instance
        mock_cap_instance.isOpened.return_value = True
        # Mock reading one frame then stop
        mock_cap_instance.read.side_effect = [(True, MagicMock()), (False, None)]
        mock_cap_instance.get.return_value = 30.0 # FPS
        
        # Mock temporal features return
        mock_temp_instance = MagicMock()
        mock_temp.return_value = mock_temp_instance
        mock_temp_instance.extract_temporal_features.return_value = {
            "posture_stability": {"value": 0.8},
            "posture_confidence": {"value": 0.9},
            "eye_contact_consistency": {"value": 0.7, "method": "gaze_angle", "evidence_ratio": 1.0},
        }
        
        mock_load.return_value = (None, None) # No audio
        
        # Run analysis
        result = analyze_video("dummy_path.mp4", transcript="Hello world")
        
        # Verify structure
        self.assertIn("analysis_version", result)
        self.assertIn("results", result)
        
        results = result["results"]
        self.assertIn("multimodal_analysis", results)
        self.assertIn("qualitative_feedback", results)
        self.assertIn("transcript", results)
        self.assertIn("summary_view", results)
        
        # Verify multimodal payload
        mm = results["multimodal_analysis"]
        self.assertIn("analysis_metadata", mm)
        self.assertIn("temporal_features", mm)
        self.assertIn("posture_analysis", mm)
        
        # Verify summary view
        sv = results["summary_view"]
        self.assertIn("posture_score", sv)
        self.assertIn("overall_confidence", sv)
        
        print("✅ Response structure verified successfully!")

if __name__ == '__main__':
    unittest.main()
