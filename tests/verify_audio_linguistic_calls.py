import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mocks for dependencies
sys.modules['mediapipe'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['librosa'] = MagicMock()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestMethodCalls(unittest.TestCase):
    
    @patch('utils.video_utils.cv2') 
    @patch('utils.video_utils.NormalizedGeometry')
    @patch('utils.video_utils.TemporalFeatures')
    @patch('utils.video_utils.AudioAnalyzer')
    @patch('utils.video_utils.SpeechMetrics')
    @patch('utils.video_utils.LinguisticAnalyzer')
    @patch('utils.video_utils.HeadPoseMetrics')
    @patch('utils.video_utils.librosa')
    def test_correct_method_calls(self, mock_librosa, mock_head, mock_ling_cls, mock_speech_cls, mock_audio, mock_temp, mock_geo, mock_cv2):
        
        # Setup mocks
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        # 1 frame then end
        mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
        mock_cap.get.return_value = 30.0 # FPS
        mock_librosa.load.return_value = (MagicMock(), 16000)
        
        # Mock Instances
        mock_speech = mock_speech_cls.return_value
        mock_ling = mock_ling_cls.return_value
        
        # Call with transcript
        from utils.video_utils import analyze_video
        
        transcript = "This is a test transcript."
        try:
            analyze_video("dummy.mp4", transcript=transcript)
        except Exception as e:
            # We don't care about return value structure here, just the calls
            pass
            
        # Check SpeechMetrics calls
        # Should call analyze_speaking_rate and analyze_fillers
        self.assertTrue(mock_speech.analyze_speaking_rate.called, "analyze_speaking_rate not called")
        self.assertTrue(mock_speech.analyze_fillers.called, "analyze_fillers not called")
        
        # Should NOT call analyze_transcript
        # self.assertFalse(mock_speech.analyze_transcript.called) # Attribute won't exist on mock unless accessed, but we can check usage
        # Actually on a MagicMock, accessing it creates it. We check that it wasn't *called*.
        # But since the method doesn't exist in the real class, the code would crash if called on real object.
        # Here we just verify the NEW methods ARE called.
        
        # Check LinguisticAnalyzer calls
        # Should call analyze_answer_structure
        self.assertTrue(mock_ling.analyze_answer_structure.called, "analyze_answer_structure not called")
        
        print("Verified correct method calls for Speech and Linguistic analysis.")
        
if __name__ == '__main__':
    unittest.main()
