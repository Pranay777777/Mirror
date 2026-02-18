import sys
import os
import json
import unittest
import jsonschema
from unittest.mock import MagicMock, patch

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import analyze_video

class TestSchemaValidation(unittest.TestCase):
    @patch('utils.video_utils.cv2.VideoCapture')
    @patch('utils.video_utils.NormalizedGeometry')
    @patch('utils.video_utils.TemporalFeatures')
    @patch('utils.video_utils.AudioAnalyzer')
    @patch('utils.video_utils.SpeechMetrics')
    @patch('utils.video_utils.LinguisticAnalyzer')
    @patch('utils.video_utils.HeadPoseMetrics')
    @patch('utils.video_utils.librosa.load')
    def test_output_schema(self, mock_load, mock_head, mock_ling, mock_speech, mock_audio, mock_temp, mock_geo, mock_cap):
        # 1. Setup mocks for a SUCCESSFUL run
        mock_cap_instance = MagicMock()
        mock_cap.return_value = mock_cap_instance
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.read.side_effect = [(True, MagicMock()), (False, None)]
        mock_cap_instance.get.side_effect = lambda prop: 30.0 if prop == 5 else (640 if prop == 3 else (480 if prop == 4 else 100)) # FPS, W, H, Frames

        mock_temp_instance = MagicMock()
        mock_temp.return_value = mock_temp_instance
        mock_temp_instance.extract_temporal_features.return_value = {
            "posture_stability": {"value": 0.8},
            "posture_confidence": {"value": 0.9}
        }
        
        mock_ling_instance = MagicMock()
        mock_ling.return_value = mock_ling_instance
        # ensure linguistic results are dicts
        mock_ling_instance.analyze_answer_structure.return_value = {"lexical_diversity": 0.5}

        mock_speech_instance = MagicMock()
        mock_speech.return_value = mock_speech_instance
        # ensure speech results are populated to avoid fallback-to-error if possible, 
        # or handle empty dictionaries gracefully.
        mock_speech_instance.analyze_speaking_rate.return_value = {"speaking_pace": "Balanced"}
        mock_speech_instance.analyze_fillers.return_value = {}
        mock_speech_instance.analyze_pauses.return_value = {}
        
        mock_head_instance = MagicMock()
        mock_head.return_value = mock_head_instance
        mock_head_instance.aggregate_metrics.return_value = {"pitch_std": 5.0}

        mock_load.return_value = (MagicMock(), 22050)
        
        mock_audio_instance = MagicMock()
        mock_audio.return_value = mock_audio_instance
        mock_audio_instance.analyze_vocal_features.return_value = {
            "metrics": {"pitch": 100},
            "reliability_score": 0.8
        }

        # 2. Load Schema
        schema_path = os.path.join(os.path.dirname(__file__), '..', 'schemas', 'analysis_schema.json')
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        # 3. Run analysis (Debug Mode = True for full schema check)
        print("Running analysis...")
        result = analyze_video("test.mp4", debug_mode=True)
        
        # 4. Validate
        print("Validating JSON schema...")
        try:
            jsonschema.validate(instance=result, schema=schema)
            print("✅ Schema validation successful!")
        except jsonschema.ValidationError as e:
            self.fail(f"Schema Validation Failed: {e.message}")

if __name__ == '__main__':
    unittest.main()
