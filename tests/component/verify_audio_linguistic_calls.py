"""verify_audio_linguistic_calls.py — Confirms AudioAnalyzer and LinguisticAnalyzer are called correctly."""
from dotenv import load_dotenv
load_dotenv()

SAMPLE_TRANSCRIPT = "Hello, my name is Pranay. I am demonstrating confident communication skills today."

def main():
    import librosa, numpy as np
    from features.audio_analysis import AudioAnalyzer
    from features.linguistic_analysis import LinguisticAnalyzer

    # Audio
    y = np.zeros(16000, dtype=np.float32)  # 1-second silence
    sr = 16000
    audio_r = AudioAnalyzer().finalize(y, sr)
    print("[AUDIO] reliability_score:", audio_r.get("reliability_score"))
    print("[AUDIO] metrics:", list(audio_r.get("metrics", {}).keys()))

    # Linguistic
    ling_r = LinguisticAnalyzer().finalize(SAMPLE_TRANSCRIPT)
    print("[LING] keys:", list(ling_r.keys()))
    print("[LING] sentiment:", ling_r.get("sentiment"))

if __name__ == "__main__":
    main()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
