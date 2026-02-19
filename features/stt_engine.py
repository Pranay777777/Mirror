import logging
import os
from typing import Dict, List, Any
# from faster_whisper import WhisperModel # Import inside function to avoid startup cost/errors if not installed? 
# No, let's import at top, but wrap in try-except for safety during dev/test if env issues arise.

logger = logging.getLogger(__name__)

class STTEngine:
    _model = None

    @classmethod
    def get_model(cls, model_size="base", device="cpu", compute_type="int8"):
        if cls._model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Loading faster-whisper model: {model_size} on {device}")
                cls._model = WhisperModel(model_size, device=device, compute_type=compute_type)
            except ImportError:
                logger.error("faster-whisper not installed. Please run 'pip install faster-whisper'")
                raise
            except Exception as e:
                logger.error(f"Failed to load faster-whisper model: {e}")
                raise
        return cls._model

def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    """
    Transcribes audio using faster-whisper.
    Returns:
    {
        "text": str,
        "language": str,
        "confidence": float,
        "segments": [ {"start": float, "end": float, "text": str}, ... ]
    }
    """
    if not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {audio_path}")
        return {"text": "", "language": "unknown", "confidence": 0.0, "segments": []}

    try:
        model = STTEngine.get_model()
        
        # Run transcription with VAD filter enabled
        segments_generator, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
        
        segments = []
        full_text_parts = []
        total_confidence = 0.0
        count = 0
        
        for segment in segments_generator:
            text = segment.text.strip()
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": text
            })
            full_text_parts.append(text)
            # segment.avg_logprob is log probability. confidence = exp(avg_logprob)
            # But faster-whisper might not expose confidence directly in the same way?
            # It has segment.avg_logprob.
            # We can approximate confidence or just rely on VAD.
            # Let's use 1.0 as placeholder or exp if needed, but 'info' has language probability.
            # Actually segment has 'no_speech_prob'. 
            # Let's just track that we got segments.
            count += 1
            
        final_text = " ".join(full_text_parts)
        
        return {
            "text": final_text,
            "language": info.language,
            "confidence": info.language_probability, # Using language prob as proxy for now
            "segments": segments
        }
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {"text": "", "language": "error", "confidence": 0.0, "segments": []}
