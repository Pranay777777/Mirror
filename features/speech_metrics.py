import logging
import math
import librosa
import numpy as np

logger = logging.getLogger(__name__)

class SpeechMetrics:
    def __init__(self):
        pass

    def __init__(self):
        pass

    def add_frame(self, data: dict, timestamp: float):
        """No-op for speech metrics."""
        pass

    def finalize(self, transcript: str, duration: float) -> dict:
        """
        Unified analysis wrapper for Speech Metrics.
        """
        results = {}
        if not transcript:
            return results
            
        rate_metrics = self.analyze_speaking_rate(transcript, duration)
        filler_metrics = self.analyze_fillers(transcript, duration)
        
        if rate_metrics:
            results.update(rate_metrics)
        if filler_metrics:
            results.update(filler_metrics)
            
        return results

    def analyze_speaking_rate(self, transcript_text, duration_seconds):
        """
        Computes words per minute (WPM).
        """
        if not transcript_text or duration_seconds <= 0:
            return None

        words = transcript_text.split()
        word_count = len(words)
        wpm = (word_count / duration_seconds) * 60.0
        
        category = "Balanced"
        if wpm < 120:
            category = "Too Slow"
        elif wpm > 180:
             category = "Too Fast"

        return {
            'words_per_minute': round(wpm, 1),
            'word_count': word_count,
            'speaking_rate_category': category
        }

    def analyze_fillers(self, transcript_text, duration_seconds=None):
        """
        Detects filler words frequency.
        """
        if not transcript_text:
            return None
            
        fillers = ["um", "uh", "like", "basically", "you know", "actually", "literally", "sort of", "kind of", "i mean"]
        text_lower = transcript_text.lower()
        
        filler_count = 0
        details = {}
        
        # Simple string counting - prone to false positives (e.g. "I like it")
        # Ideal solution uses tokenization or regex with word boundaries
        # For MVP, we'll accept some error or use basic split
        words = text_lower.split()
        
        # Only counting unigrams for now to match words list easily
        # For "you know", handle separately? 
        # Let's do a simple count for now.
        for f in fillers:
            count = text_lower.count(f) # this counts substrings
            if count > 0:
                filler_count += count
                details[f] = count
                
        rate = 0.0
        if duration_seconds and duration_seconds > 0:
            rate = (filler_count / duration_seconds) * 60.0
        
        return {
            'filler_count': filler_count,
            'filler_rate_per_min': round(rate, 2),
            'filler_breakdown': details
        }
    
    def analyze_pauses(self, y: np.ndarray, sr: int, top_db=25, min_silence_duration=0.5):
        """
        Analyzes pauses using silence detection.
        y: audio waveform
        sr: sample rate
        top_db: threshold below max db to consider silent.
        """
        try:
            if y is None or len(y) == 0:
                return None
            
            # Detect non-silent intervals
            non_silent_intervals = librosa.effects.split(y, top_db=top_db)
            
            # Calculate gaps (pauses)
            pauses = []
            
            # Total duration
            total_dur = len(y) / sr
            
            if len(non_silent_intervals) > 0:
                # Check gap before first speech? Usually ignore.
                
                for i in range(len(non_silent_intervals) - 1):
                    end_prev = non_silent_intervals[i][1]
                    start_next = non_silent_intervals[i+1][0]
                    gap_samples = start_next - end_prev
                    gap_sec = gap_samples / sr
                    
                    if gap_sec >= min_silence_duration:
                        pauses.append(gap_sec)
            
            avg_pause = 0.0
            long_pauses = 0
            
            if pauses:
                avg_pause = float(np.mean(pauses))
                long_pauses = len([p for p in pauses if p > 1.5])
            
            return {
                'avg_pause_duration': round(avg_pause, 2),
                'long_pause_count': long_pauses,
                'total_pauses': len(pauses)
            }
            
        except Exception as e:
            logger.error(f"Pause analysis failed: {e}")
            return None
