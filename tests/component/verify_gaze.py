"""
verify_gaze.py — Verifies gaze/head-orientation metrics on a video.
Usage: python verify_gaze.py <video.mp4>
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
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        geo = geom.process(frame)
        temp.add_frame(geo, geo, idx / fps)
        idx += 1
    cap.release()
    results = temp.finalize()
    gaze_keys = [k for k in results if "gaze" in k or "head_orientation" in k or "eye" in k]
    print("[GAZE VERIFY] Relevant keys:", gaze_keys)
    print(json.dumps({k: results[k] for k in gaze_keys}, indent=2, default=str))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_gaze.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
