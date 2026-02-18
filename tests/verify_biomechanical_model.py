import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Pre-mock mediapipe before imports
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.solutions'] = MagicMock()
sys.modules['mediapipe.solutions.holistic'] = MagicMock()
sys.modules['mediapipe.solutions.face_mesh'] = MagicMock()

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestBiomechanicalModel(unittest.TestCase):
    
    @patch('utils.video_utils.cv2')
    @patch('utils.video_utils.NormalizedGeometry')
    @patch('utils.video_utils.TemporalFeatures')
    @patch('utils.video_utils.AudioAnalyzer')
    @patch('utils.video_utils.SpeechMetrics')
    @patch('utils.video_utils.LinguisticAnalyzer')
    @patch('utils.video_utils.HeadPoseMetrics')
    @patch('utils.video_utils.librosa')
    def verify_interpretation(self, alignment, movement, stability, expected_text, 
                              mock_librosa, mock_head, mock_ling, mock_speech, mock_audio, mock_temp, mock_geo, mock_cv2):
        
        # Setup common mocks
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
        mock_cap.get.return_value = 30.0
        mock_librosa.load.return_value = (MagicMock(), 16000)
        
        # Mock TemporalFeatures to return specific biomechanical values
        mock_temp_instance = MagicMock()
        mock_temp.return_value = mock_temp_instance
        mock_temp_instance.extract_temporal_features.return_value = {
            "alignment_integrity": {"value": alignment},
            "motion_activity_level": {"value": movement},
            "stability_index": {"value": stability},
            # Required placeholders
            "gaze_stability": {"value": 0.9},
            "eye_contact_consistency": {"value": 0.9},
            "expression_dynamics": {"value": 0.5},
            "smile_intensity": {"value": 0.5},
            "expression_variability": {"value": 0.5},
            "blink_rate": {"value": 15},
            "gaze_confidence": {"value": 0.9},
            "data_completeness": {"value": 1.0},
            "posture_confidence": {"value": 0.9}
        }
        
        # Mock HeadPoseMetrics
        mock_head_instance = MagicMock()
        mock_head.return_value = mock_head_instance
        # Default low roll_std unless testing lateral tilt
        mock_head_instance.aggregate_metrics.return_value = {'roll_std': 5.0}
        
        # Other mocks
        mock_ling.return_value.analyze_answer_structure.return_value = {}
        mock_ling.return_value.analyze_content.return_value = {}
        mock_audio.return_value.analyze_vocal_features.return_value = {'metrics': {}, 'reliability_score': 0.8}
        
        with patch('utils.video_utils.multimodal_confidence_fusion', return_value=0.9):
            from utils.video_utils import analyze_video
            result = analyze_video("dummy.mp4", debug_mode=False)
            
            # Check interpretation text
            posture_data = result["results"]["multimodal_analysis"]["posture_analysis"]
            interpretation = posture_data["interpretation"]
            
            print(f"A:{alignment} M:{movement} S:{stability} -> {interpretation}")
            self.assertIn(expected_text, interpretation)
            
            # Check keys
            self.assertIn("alignment_integrity", posture_data)
            self.assertIn("motion_activity_level", posture_data)
            self.assertIn("stability_index", posture_data)
            self.assertNotIn("posture_stability", posture_data)
            self.assertNotIn("movement_intensity", posture_data)

    def test_hierarchy_1_poor_alignment(self):
        # A=0.6, M=0.0, S=1.0. A < 0.65 -> "significant lateral misalignment"
        # S >= 0.85 -> "stable"
        # M < 0.05 -> ""
        # Expected: "Stable posture with significant lateral misalignment"
        self.verify_interpretation(0.5, 0.0, 1.0, "Stable posture with significant lateral misalignment")

    def test_hierarchy_2_excessive_movement(self):
        # A=0.9 (Well-aligned), M=0.25 (Excessive), S=0.6 (Unstable)
        # S < 0.65 -> "unstable"
        # M >= 0.12 -> "with excessive movement"
        # Expected: "Unstable and well-aligned posture with excessive movement"
        self.verify_interpretation(0.9, 0.25, 0.6, "Unstable and well-aligned posture with excessive movement")

    def test_hierarchy_3_unstable(self):
        # A=0.9, M=0.10 (Noticeable), S=0.70 (Moderately Stable)
        # S >= 0.65 -> "moderately stable"
        # M < 0.12, >= 0.05 -> "with noticeable movement"
        # Expected: "Moderately stable and well-aligned posture with noticeable movement"
        self.verify_interpretation(0.9, 0.10, 0.70, "Moderately stable and well-aligned posture with noticeable movement")

    def test_hierarchy_4_stable(self):
        # A=0.9, M=0.04 (None), S=0.9 (Stable)
        # M < 0.05 -> ""
        # Expected: "Stable and well-aligned posture"
        self.verify_interpretation(0.9, 0.04, 0.9, "Stable and well-aligned posture")

if __name__ == '__main__':
    unittest.main()
