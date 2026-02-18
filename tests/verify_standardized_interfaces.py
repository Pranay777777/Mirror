import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import traceback

# Mocks
sys.modules['mediapipe'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['librosa'] = MagicMock()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestStandardizedInterfaces(unittest.TestCase):
    
    @patch('utils.video_utils.cv2') 
    @patch('utils.video_utils.NormalizedGeometry')
    @patch('utils.video_utils.TemporalFeatures')
    @patch('utils.video_utils.AudioAnalyzer')
    @patch('utils.video_utils.SpeechMetrics')
    @patch('utils.video_utils.LinguisticAnalyzer')
    @patch('utils.video_utils.HeadPoseMetrics')
    @patch('utils.video_utils.librosa')
    def test_standardized_flow(self, mock_librosa, mock_head_cls, mock_ling_cls, mock_speech_cls, mock_audio, mock_temp_cls, mock_geo, mock_cv2):
        
        # Setup mocks
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        # 2 frames
        mock_cap.read.side_effect = [(True, MagicMock()), (True, MagicMock()), (False, None)]
        mock_cap.get.return_value = 30.0
        mock_librosa.load.return_value = (MagicMock(), 16000)
        
        # Mock Instances
        mock_temp = mock_temp_cls.return_value
        mock_head = mock_head_cls.return_value
        mock_speech = mock_speech_cls.return_value
        mock_geo = mock_geo.return_value
        mock_audio = mock_audio.return_value
        mock_ling = mock_ling_cls.return_value
        
        # Setup valid return values to avoid TypeErrors in logic
        # Strict Interface: return values must be set on NEW method names
        mock_geo.process.return_value = {'head_pose': (0,0,0)} 
        
        # Finalize/Analyze returns
        mock_temp.finalize.return_value = {} 
        mock_head.finalize.return_value = {}
        mock_speech.finalize.return_value = {} # Was analyze
        mock_audio.finalize.return_value = {} # Was analyze_vocal_features
        mock_ling.finalize.return_value = {} # Was analyze_answer_structure
        
        # Ensure instances have the required methods (pass Interface Guard)
        # MagicMock has them by default, but let's be safe against spec if used.
        
        from utils.video_utils import analyze_video
        
        # Execute
        try:
            analyze_video("dummy.mp4", transcript="test transcript")
        except Exception as e:
            with open("traceback.log", "w") as f:
                traceback.print_exc(file=f)
            self.fail(f"analyze_video failed with standardized interfaces: {e}")
            
        # Verify TemporalFeatures calls
        self.assertTrue(mock_temp.add_frame.called, "TemporalFeatures.add_frame not called")
        self.assertTrue(mock_temp.finalize.called, "TemporalFeatures.finalize not called")
        
        # Verify HeadPoseMetrics calls
        self.assertTrue(mock_head.add_frame.called, "HeadPoseMetrics.add_frame not called")
        self.assertTrue(mock_head.finalize.called, "HeadPoseMetrics.finalize not called")
        
        # Verify SpeechMetrics calls - Strict: finalize
        self.assertTrue(mock_speech.finalize.called, "SpeechMetrics.finalize not called")
        
        # Verify AudioAnalyzer calls - Strict: finalize
        self.assertTrue(mock_audio.finalize.called, "AudioAnalyzer.finalize not called")

        # Verify LinguisticAnalyzer calls - Strict: finalize
        self.assertTrue(mock_ling.finalize.called, "LinguisticAnalyzer.finalize not called")
        
        # Verify NormalizedGeometry calls - Strict: process
        self.assertTrue(mock_geo.process.called, "NormalizedGeometry.process not called")
        
        print("Verified strict interface calls: process, add_frame, finalize.")

    def test_interface_guard_enforcement(self):
        # This test ensures the guard actually raises error if method is missing
        
        from utils.video_utils import analyze_video
        
        with patch('utils.video_utils.SpeechMetrics') as mock_bad_speech_cls:
            # Create a mock that MISSES finalize
            mock_bad_instance = MagicMock()
            del mock_bad_instance.finalize # Explicitly delete if it exists on MagicMock? 
            # MagicMock creates on access. To simulate missing, we access 'finalize' and raise AttributeError?
            # Or use spec.
            mock_bad_instance = MagicMock(spec=['add_frame']) # Only add_frame, no finalize
            mock_bad_speech_cls.return_value = mock_bad_instance
            
            with self.assertRaisesRegex(RuntimeError, "SpeechMetrics missing required method: finalize"):
                analyze_video("dummy.mp4", transcript="test")
                
        print("Verified interface guard catches missing methods.")

if __name__ == '__main__':
    unittest.main()
