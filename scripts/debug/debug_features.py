"""
debug_features.py — Dumps temporal features output for a given video.
Usage: python debug_features.py <video.mp4>
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import sys, json, cv2
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from features.normalized_geometry import NormalizedGeometry
    from features.temporal_features import TemporalFeatures

    geom = NormalizedGeometry()
    temp = TemporalFeatures()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        geo = geom.process(frame)
        ts = frame_idx / fps
        temp.add_frame(geo, geo, ts)
        frame_idx += 1
    cap.release()
    results = temp.finalize()
    print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_features.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
