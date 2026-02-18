import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(self):
        self.sr = 22050 # Default sample rate for librosa

    def add_frame(self, data: dict, timestamp: float):
        """No-op for audio analyzer which processes full audio track."""
        pass

    def finalize(self, y: np.ndarray, sr: int):
        """
        Analyzes audio for vocal energy and pitch features.
        y: audio waveform
        sr: sample rate
        Returns metrics dict and reliability score.
        """
        try:
            if y is None or len(y) == 0:
                return None
            
            # --- 1. Vocal Energy (RMS) ---
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))
            
            # Avoid division by zero
            mean_safe = rms_mean if rms_mean > 1e-6 else 1e-6
            energy_variability = rms_std / mean_safe

            # --- 2. Pitch (F0) Analysis ---
            # using pyin for robustness (slow but accurate)
            # Limit frequency range to human voice (50-500Hz)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('A1'), # ~55Hz 
                fmax=librosa.note_to_hz('C6'), # ~1046Hz
                sr=sr
            )
            
            # Filter for voiced segments only
            if f0 is not None:
                voiced_f0 = f0[voiced_flag]
            else:
                voiced_f0 = []
            
            # --- Reliability Calculation (Task 2) ---
            # 1. Voiced Ratio: Proportion of frames with detectable pitch
            total_frames = len(f0) if f0 is not None else 1
            voiced_count = np.sum(voiced_flag)
            voiced_ratio = voiced_count / total_frames if total_frames > 0 else 0.0
            
            # 2. Pitch Detection Confidence (prob)
            pitch_confidence = 0.0
            if f0 is not None:
                # Average probability of voiced frames
                # If no voiced frames, pitch confidence is low (unless silence)
                if voiced_count > 0:
                     pitch_confidence = float(np.mean(voiced_probs[voiced_flag]))
            
            # 3. Energy Stability (Consistency)
            # We want high reliability for consistent volume, but natural speech has variance.
            # Penalize only extreme dropout or noise.
            # Use Signal-to-Noise proxy?
            # Simple approach: 1.0 - (std / (mean + epsilon)) clamped
            # But high dynamic range (expressive) shouldn't be penalized?
            # User said: "Energy consistency". "Don't penalize neutral tone".
            # Maybe just check if mean energy is above noise floor?
            # If rms_mean is very low, reliability is low.
            # RMS of -60dB is ~0.001.
            volume_score = min(1.0, rms_mean / 0.01) # Satures at 0.01 (-40dB approx)
            
            # 4. Silence Ratio
            # High silence -> low data -> lower reliability?
            # Or silence is valid? 
            # User said: "Compute silence percentage."
            silence_thresh = 0.001 # -60dB
            silence_count = np.sum(rms < silence_thresh)
            silence_ratio = silence_count / len(rms) if len(rms) > 0 else 0.0
            non_silence_ratio = 1.0 - silence_ratio
            
            # Weighted Average
            # If silence is high, we rely on non-silence parts?
            # Reliability = Quality of the SPEECH segments.
            
            # Adjusted weights:
            # Voiced Ratio: 0.4 (Human speech should have pitch)
            # Pitch Confidence: 0.3 (Algorithm certainty)
            # Volume/Non-Silence: 0.3 (Is there signal?)
            
            # Note: Voiced Ratio might be low for unvoiced speech (whisper/sibilants). 
            # But usually >30% for normal speech.
            # Normalize voiced_ratio: expect > 0.3. 
            voiced_score = min(1.0, voiced_ratio / 0.3)
            
            final_reliability = (
                0.4 * voiced_score +
                0.3 * pitch_confidence +
                0.3 * volume_score
            )
            
            # Clamp
            final_reliability = max(0.0, min(1.0, final_reliability))

            reliability_details = {
                'voiced_ratio': round(voiced_ratio, 3),
                'pitch_valid_ratio': round(pitch_confidence, 3),
                'energy_variance': round(energy_variability, 3),
                'silence_ratio': round(silence_ratio, 3),
                'final_audio_reliability': round(final_reliability, 3)
            }
            
            pitch_mean = 0.0
            pitch_std = 0.0
            pitch_variability = 0.0

            if len(voiced_f0) > 0:
                pitch_mean = float(np.mean(voiced_f0))
                pitch_std = float(np.std(voiced_f0))
                mean_pitch_safe = pitch_mean if pitch_mean > 1e-6 else 1e-6
                pitch_variability = pitch_std / mean_pitch_safe
            
            return {
                'metrics': {
                    'rms_mean': round(rms_mean, 4),
                    'rms_std': round(rms_std, 4),
                    'pitch_mean_hz': round(pitch_mean, 2),
                    'pitch_std_hz': round(pitch_std, 2),
                    'energy_variability_score': round(energy_variability, 4),
                    'pitch_variability_score': round(pitch_variability, 4)
                },
                'reliability_score': round(final_reliability, 4),
                'diagnostics': reliability_details
            }

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return None
