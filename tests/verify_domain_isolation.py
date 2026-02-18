import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import analyze_video

class TestDomainIsolation(unittest.TestCase):
    """
    Verify that visual confidence is isolated from audio/speech reliability.
    """

    @patch('utils.video_utils.cv2')
    @patch('utils.video_utils.NormalizedGeometry')
    @patch('utils.video_utils.TemporalFeatures')
    @patch('utils.video_utils.AudioAnalyzer')
    @patch('utils.video_utils.SpeechMetrics')
    @patch('utils.video_utils.LinguisticAnalyzer')
    @patch('utils.video_utils.HeadPoseMetrics')
    @patch('utils.video_utils.librosa')
    def test_silence_does_not_affect_posture_confidence(self, mock_librosa, mock_head, mock_ling_cls, mock_speech_cls, mock_audio_cls, mock_temp_cls, mock_geo_cls, mock_cv2):
        
        # Setup mocks
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, True, False] # Open, Read 1 frame, Close
        mock_cap.read.return_value = (True, MagicMock())
        mock_cap.get.return_value = 30.0 # FPS
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_librosa.load.return_value = (None, 22050) # Mock audio load
        
        # 1. Visual Data = PERFECT
        mock_temp = mock_temp_cls.return_value
        mock_temp.finalize.return_value = {
            'visual_confidence': {'value': 0.95}, # High visual confidence
            'alignment_integrity': {'value': 0.95},
            'stability_index': {'value': 0.90},
            'motion_activity_level': {'value': 0.05}
        }
        
        # 2. Audio Data = TERRIBLE (Silence)
        mock_audio = mock_audio_cls.return_value
        mock_audio.finalize.return_value = {
            'reliability_score': 0.0, # Complete silence/noise
            'diagnostics': {'voiced_ratio': 0.0}
        }
        
        # 3. Speech/Ling = Empty
        mock_speech = mock_speech_cls.return_value
        mock_speech.finalize.return_value = {}
        
        mock_ling = mock_ling_cls.return_value
        mock_ling.finalize.return_value = {}
        
        mock_geo = mock_geo_cls.return_value
        mock_geo.process.return_value = {}
        
        mock_head_inst = mock_head.return_value
        mock_head_inst.finalize.return_value = {}

        # Run Analysis
        print("Running analyze_video with Perfect Vision + Zero Audio...")
        result = analyze_video("dummy.mp4", transcript=None)
        
        # Checks
        # analyze_video returns query response wrapper
        # response = { "results": { "multimodal_analysis": ... } }
        mm = result['results']['multimodal_analysis']
        conf = mm['confidence_metrics']
        posture = mm['posture_analysis']
        
        print("\n--- Domain Isolation Results ---")
        print(f"Visual Confidence: {conf['visual_confidence']}")
        print(f"Audio Confidence:  {conf['audio_confidence']}")
        print(f"Posture Confidence: {posture['confidence']}")
        print(f"Overall Confidence: {conf['overall_confidence']}")
        
        # Assertions
        # 1. Posture Confidence MUST match Visual Confidence (Isolation)
        self.assertEqual(posture['confidence'], conf['visual_confidence'], 
                         "Posture analysis confidence must explicitly match visual_confidence")
        
        # 2. Posture Confidence must be HIGH (>0.85) despite Audio=0
        self.assertGreater(posture['confidence'], 0.85, 
                           "Posture confidence dropped despite perfect visual data! Domain contamination detected.")
        
        # 3. Audio Confidence must be 0
        self.assertEqual(conf['audio_confidence'], 0.0)
        
        # 4. Overall Confidence should be dragged down by audio, verifying fusion works
        # Visual(0.95)*0.5 + Audio(0.0)*0.3 + Speech(0.0)*0.2 = 0.475
        self.assertAlmostEqual(conf['overall_confidence'], 0.475, delta=0.05,
                               msg="Overall confidence fusion calculation incorrect")
        
        print("\nSUCCESS: Domain Isolation Verified.")

if __name__ == '__main__':
    unittest.main()
