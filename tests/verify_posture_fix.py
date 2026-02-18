import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.solutions'] = MagicMock()
sys.modules['mediapipe.solutions.holistic'] = MagicMock()
sys.modules['mediapipe.solutions.face_mesh'] = MagicMock()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestPostureScoring(unittest.TestCase):
    @patch('utils.video_utils.cv2') 
    @patch('utils.video_utils.NormalizedGeometry')
    @patch('utils.video_utils.TemporalFeatures')
    @patch('utils.video_utils.AudioAnalyzer')
    @patch('utils.video_utils.SpeechMetrics')
    @patch('utils.video_utils.LinguisticAnalyzer')
    @patch('utils.video_utils.HeadPoseMetrics')
    @patch('utils.video_utils.librosa') 
    def test_posture_scoring_logic(self, mock_librosa, mock_head, mock_ling, mock_speech, mock_audio, mock_temp, mock_geo, mock_cv2):
        print("Running scoring logic test...")
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.side_effect = [True, False] 
        mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
        mock_cap.get.return_value = 30.0
        mock_librosa.load.return_value = (MagicMock(), 16000)
        
        # FIX: Mock analyze_content to return dict, avoiding MagicMock * float crash
        mock_ling.return_value.analyze_answer_structure.return_value = {}
        mock_ling.return_value.analyze_content.return_value = {} 
        
        mock_geo_instance = MagicMock()
        mock_geo.return_value = mock_geo_instance
        mock_geo_instance.analyze_frame.return_value = {
            'face_detection_confidence': 0.9,
            'face_size_ratio': 0.3,
            'head_pose': {'roll': 0, 'pitch': 0, 'yaw': 0},
            'face_landmarks': [],
            'mp_results': MagicMock()
        }

        def check_score(case_name, alignment, stability, roll_std, expected_score, tolerance=0.5):
            print(f"Running score check: {case_name}")
            mock_cap.isOpened.side_effect = [True, False] 
            mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
            
            mock_temp_instance = MagicMock()
            mock_temp.return_value = mock_temp_instance
            mock_temp_instance.extract_temporal_features.return_value = {
                "alignment_integrity": {"value": alignment},
                "stability_index": {"value": stability},
                "motion_activity_level": {"value": 0.0},
                "posture_confidence": {"value": 0.9},
                "data_completeness": {"value": 1.0},
                "gaze_stability": {"value": 0.9},
                "eye_contact_consistency": {"value": 0.9},
                "expression_dynamics": {"value": 0.5},
                "smile_intensity": {"value": 0.5},
                "expression_variability": {"value": 0.5},
                "blink_rate": {"value": 15},
                "gaze_confidence": {"value": 0.9}
            }
            
            mock_head_instance = MagicMock()
            mock_head.return_value = mock_head_instance
            mock_head_instance.aggregate_metrics.return_value = {'roll_std': roll_std}
            
            with patch('utils.video_utils.multimodal_confidence_fusion', return_value=0.9):
                from utils.video_utils import analyze_video
                result = analyze_video("dummy.mp4", debug_mode=False)
                score = result["results"]["summary_view"]["posture_score"]
                print(f"[{case_name}] Align:{alignment} Stab:{stability} RollStd:{roll_std} -> Score: {score}")
                self.assertAlmostEqual(score, expected_score, delta=tolerance, msg=f"Failed {case_name}")
        
        check_score("Case 1 (Slight)", 0.8, 0.9, 25, 6.7)
        check_score("Case 2 (Moderate)", 0.75, 0.9, 45, 5.9)
        check_score("Case 3 (Severe)", 0.6, 0.9, 70, 4.4)
        check_score("Case 4 (Ideal)", 0.96, 0.96, 5, 9.6)

    @patch('utils.video_utils.cv2') 
    @patch('utils.video_utils.NormalizedGeometry')
    @patch('utils.video_utils.TemporalFeatures')
    @patch('utils.video_utils.AudioAnalyzer')
    @patch('utils.video_utils.SpeechMetrics')
    @patch('utils.video_utils.LinguisticAnalyzer')
    @patch('utils.video_utils.HeadPoseMetrics')
    @patch('utils.video_utils.librosa')
    def test_interpretation_text(self, mock_librosa, mock_head, mock_ling, mock_speech, mock_audio, mock_temp, mock_geo, mock_cv2):
        print("Testing Interpretation Text...")
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_librosa.load.return_value = (MagicMock(), 16000)

        # FIX: Mock content analysis to avoid crash
        mock_ling.return_value.analyze_answer_structure.return_value = {}
        mock_ling.return_value.analyze_content.return_value = {}
        mock_audio.return_value.analyze_vocal_features.return_value = {'metrics': {}, 'reliability_score': 0.8}
        
        mock_geo_instance = MagicMock()
        mock_geo.return_value = mock_geo_instance
        mock_geo_instance.analyze_frame.return_value = {
            'face_detection_confidence': 0.9,
            'face_size_ratio': 0.3,
            'head_pose': {'roll': 0, 'pitch': 0, 'yaw': 0},
            'face_landmarks': [],
            'mp_results': MagicMock()
        }

        def check_text(case_name, alignment, stability, roll_std, expected_phrases):
            print(f"Checking Text: {case_name}")
            mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
            
            mock_temp_instance = MagicMock()
            mock_temp.return_value = mock_temp_instance
            mock_temp_instance.extract_temporal_features.return_value = {
                "alignment_integrity": {"value": alignment},
                "stability_index": {"value": stability},
                "motion_activity_level": {"value": 0.0},
                "posture_confidence": {"value": 0.9},
                "data_completeness": {"value": 1.0},
                "gaze_stability": {"value": 0.9},
                "eye_contact_consistency": {"value": 0.9},
                "expression_dynamics": {"value": 0.5},
                "smile_intensity": {"value": 0.5},
                "expression_variability": {"value": 0.5},
                "blink_rate": {"value": 15},
                "gaze_confidence": {"value": 0.9}
            }
            
            mock_head_instance = MagicMock()
            mock_head.return_value = mock_head_instance
            mock_head_instance.aggregate_metrics.return_value = {'roll_std': roll_std}
            
            with patch('utils.video_utils.multimodal_confidence_fusion', return_value=0.9):
                from utils.video_utils import analyze_video
                result = analyze_video("dummy.mp4", debug_mode=False)
                summary = result["results"]["summary_view"]["concise_summary_text"]
                score = result["results"]["summary_view"]["posture_score"]
                
                print(f"[{case_name}] Score: {score} | Summary: {repr(summary)}")
                
                for phrase in expected_phrases:
                    self.assertIn(phrase, summary, f"[{case_name}] Missing '{phrase}' in '{summary}'")

        # Deterministic checks
        # Template: "{Alignment} body alignment with {stability} stability and {movement} movement."
        # Stability Bands:
        # 0.90+ very high
        # 0.80-0.89 high
        # 0.70-0.79 moderate
        # 0.60-0.69 reduced
        # <0.60 low
        
        # Excellent: 0.95 Align -> Excellent, 0.95 Stab -> very high, <0.04 Mov -> minimal
        check_text("Excellent", 0.95, 0.95, 5, ["Excellent body alignment", "very high stability"])
        
        # VeryGood (Old) -> 0.85 Align -> Good (0.80-0.94), 0.85 Stab -> high (0.80+)
        check_text("VeryGood", 0.85, 0.85, 5, ["Good body alignment", "high stability"])
        
        # Good (Old) -> 0.82 Align -> Good (>=0.80), 0.80 Stab -> high (>=0.80)
        check_text("Good", 0.82, 0.80, 5, ["Good body alignment", "high stability"])
        
        # Fair (Old) -> 0.75 Align -> Moderate (0.60-0.79), 0.75 Stab -> moderate (0.70-0.79)
        check_text("Fair", 0.75, 0.75, 5, ["Moderate body alignment", "moderate stability"])
        
        # Reduced Case (New) -> 0.65 Stab -> reduced (0.60-0.69)
        check_text("ReducedStab", 0.75, 0.65, 5, ["reduced stability"])

        # NeedsImp (Old) -> 0.55 Align -> Significant (<0.60), 0.9 Stab -> very high
        check_text("NeedsImp", 0.55, 0.9, 5, ["Significant body alignment", "very high stability"])
        
        # LatImb (Old) -> 0.8 Align -> Good, 0.9 Stab -> very high
        check_text("LatImb", 0.8, 0.9, 70, ["Good body alignment", "very high stability"])

    @patch('utils.video_utils.cv2') 
    @patch('utils.video_utils.NormalizedGeometry')
    @patch('utils.video_utils.TemporalFeatures')
    @patch('utils.video_utils.AudioAnalyzer')
    @patch('utils.video_utils.SpeechMetrics')
    @patch('utils.video_utils.LinguisticAnalyzer')
    @patch('utils.video_utils.HeadPoseMetrics')
    @patch('utils.video_utils.librosa') 
    def test_response_prod(self, mock_librosa, mock_head, mock_ling, mock_speech, mock_audio, mock_temp, mock_geo, mock_cv2):
        # Configure Geometry to avoid crash (COMPLETE)
        mock_geo_instance = MagicMock()
        mock_geo.return_value = mock_geo_instance
        mock_geo_instance.analyze_frame.return_value = {
            'face_detection_confidence': 0.9,
            'face_size_ratio': 0.3,
            'head_pose': {'roll': 0, 'pitch': 0, 'yaw': 0},
            'face_landmarks': [],
            'mp_results': MagicMock()
        }

        # Setup common mocks
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
        mock_cap.get.return_value = 30.0
        mock_librosa.load.return_value = (MagicMock(), 16000)

        # FIX: Mock content analysis to avoid crash
        mock_ling.return_value.analyze_answer_structure.return_value = {}
        mock_ling.return_value.analyze_content.return_value = {}
        mock_audio.return_value.analyze_vocal_features.return_value = {'metrics': {}, 'reliability_score': 0.8}
        
        mock_temp.return_value.extract_temporal_features.return_value = {
            "alignment_integrity": {"value": 0.9},
            "stability_index": {"value": 0.9},
            "motion_activity_level": {"value": 0.0},
            "posture_confidence": {"value": 0.9},
            "data_completeness": {"value": 1.0},
            "gaze_stability": {"value": 0.9},
            "eye_contact_consistency": {"value": 0.9},
            "blink_rate": {"value": 15},
            "gaze_confidence": {"value": 0.9},
             "expression_dynamics": {"value": 0.5},
            "smile_intensity": {"value": 0.5},
            "expression_variability": {"value": 0.5},
        }
        
        mock_head.return_value.aggregate_metrics.return_value = {'roll_std': 5}

        with patch('utils.video_utils.multimodal_confidence_fusion', return_value=0.9):
            from utils.video_utils import analyze_video
            result_prod = analyze_video("dummy.mp4", debug_mode=False)
            payload_prod = result_prod["results"]["multimodal_analysis"]
            
            posture = payload_prod["posture_analysis"]
            self.assertIn("alignment_integrity", posture)
            self.assertIn("stability_index", posture)

if __name__ == '__main__':
    unittest.main()
