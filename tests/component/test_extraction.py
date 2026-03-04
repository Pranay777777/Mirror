"""
test_extraction.py — Tests video/audio feature extraction in isolation.
Usage: python test_extraction.py <video.mp4>
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import sys, json, cv2
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from features.normalized_geometry import NormalizedGeometry
    from features.temporal_features import TemporalFeatures
    from features.audio_analysis import AudioAnalyzer
    import librosa

    print(f"[EXTRACTION TEST] {video_path}")

    # Video feature extraction
    geom = NormalizedGeometry()
    temp = TemporalFeatures()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        geo = geom.process(frame)
        temp.add_frame(geo, geo, idx / fps)
        idx += 1
    cap.release()
    temporal = temp.finalize()
    print("[TEMPORAL FEATURES]:", json.dumps(temporal, indent=2, default=str))

    # Audio extraction
    y, sr = librosa.load(video_path, sr=16000)
    audio_analyzer = AudioAnalyzer()
    audio_res = audio_analyzer.finalize(y, sr)
    print("[AUDIO FEATURES]:", json.dumps(audio_res, indent=2, default=str))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_extraction.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
