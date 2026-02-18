import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import analyze_video

class TestDualResponseModes(unittest.TestCase):
    @patch('utils.video_utils.cv2.VideoCapture')
    @patch('utils.video_utils.NormalizedGeometry')
    @patch('utils.video_utils.TemporalFeatures')
    @patch('utils.video_utils.AudioAnalyzer')
    @patch('utils.video_utils.SpeechMetrics')
    @patch('utils.video_utils.LinguisticAnalyzer')
    @patch('utils.video_utils.HeadPoseMetrics')
    @patch('utils.video_utils.librosa.load')
    def test_analyze_video_modes(self, mock_load, mock_head, mock_ling, mock_speech, mock_audio, mock_temp, mock_geo, mock_cap):
        # Helper to setup fresh mocks
        def setup_mocks():
            mock_cap_instance = MagicMock()
            mock_cap.return_value = mock_cap_instance
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.read.side_effect = [(True, MagicMock()), (False, None)]
            mock_cap_instance.get.return_value = 30.0
            
            mock_temp_instance = MagicMock()
            mock_temp.return_value = mock_temp_instance
            mock_temp_instance.extract_temporal_features.return_value = {
                "posture_stability": {"value": 0.8},
                "posture_confidence": {"value": 0.9}
            }
            
            mock_ling_instance = MagicMock()
            mock_ling.return_value = mock_ling_instance
            mock_ling_instance.analyze_answer_structure.return_value = {
                 "lexical_diversity": 0.6,
                 "is_english_analysis": True
            }

            mock_speech_instance = MagicMock()
            mock_speech.return_value = mock_speech_instance
            mock_speech_instance.analyze_speaking_rate.return_value = {}
            mock_speech_instance.analyze_fillers.return_value = {}
            mock_speech_instance.analyze_pauses.return_value = {}
            
            return mock_cap_instance, mock_temp_instance

        mock_load.return_value = (None, None)

        # 1. Test Debug Mode (Default/True)
        setup_mocks() # Setup for first call
        result_debug = analyze_video("test.mp4", debug_mode=True)
        self.assertIn("multimodal_analysis", result_debug["results"])
        self.assertIsNotNone(result_debug["results"]["multimodal_analysis"])
        
        # 2. Test Production Mode (False)
        setup_mocks() # Setup for second call (reset side effects)
        result_prod = analyze_video("test.mp4", debug_mode=False)
        results = result_prod["results"]
        
        self.assertIn("summary_view", results)
        self.assertIn("overall_confidence", results)
        self.assertIn("summary_view", results)
        self.assertIn("overall_confidence", results)
        # Production mode now returns FULL multimodal_analysis
        self.assertIsNotNone(results["multimodal_analysis"])
        
        print("✅ Dual response modes verified successfully!")

class TestPostureInterpretation(unittest.TestCase):
        
    def test_interpret_posture(self):
        # Need to access the private function from utils
        from utils.video_utils import _interpret_posture
        
        # Helper to create mock results
        def make_res(uprightness, stability, movement):
            return {
                "posture_uprightness": {"value": uprightness},
                "posture_stability": {"value": stability},
                "overall_movement_intensity": {"value": movement}
            }

        # 1. Poor posture (< 0.6)
        res = make_res(0.5, 0.9, 0.0)
        self.assertEqual(_interpret_posture(res), "Poor posture (significant misalignment)")
        
        # 2. Moderate misalignment (< 0.8)
        res = make_res(0.75, 0.9, 0.0)
        self.assertEqual(_interpret_posture(res), "Moderate misalignment")
        
        # 3. Unstable (< 0.75)
        # Uprightness must be >= 0.8 to reach this check
        res = make_res(0.9, 0.7, 0.0)
        self.assertEqual(_interpret_posture(res), "Unstable posture")
        
        # 4. Excessive movement (> 0.06)
        res = make_res(0.9, 0.9, 0.07)
        self.assertEqual(_interpret_posture(res), "Excessive movement detected")
        
        # 5. Excellent posture
        # Uprightness >= 0.85, Stability >= 0.85, Movement <= 0.03
        res = make_res(0.9, 0.9, 0.02)
        self.assertEqual(_interpret_posture(res), "Excellent posture")
        
        # 6. Stable posture (Default)
        # E.g. Uprightness 0.82 (passes 1,2), Stability 0.8 (passes 3), Movement 0.04 (passes 4), but fails Excellent
        res = make_res(0.82, 0.8, 0.04)
        self.assertEqual(_interpret_posture(res), "Stable posture")


if __name__ == '__main__':
    unittest.main()
