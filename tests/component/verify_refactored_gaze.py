"""verify_refactored_gaze.py — Validates refactored gaze metrics output fields."""
import sys, cv2
from dotenv import load_dotenv
load_dotenv()

EXPECTED_GAZE_KEYS = ["gaze_direction_switch_count", "max_continuous_eye_contact", "head_orientation_score"]

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
    print("[REFACTORED GAZE]")
    for k in EXPECTED_GAZE_KEYS:
        present = k in results
        print(f"  {'✓' if present else '✗'} {k}: {results.get(k)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_refactored_gaze.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
