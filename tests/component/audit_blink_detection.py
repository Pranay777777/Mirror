"""audit_blink_detection.py — Audits per-frame blink detection accuracy."""
import sys, cv2
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
    blink = results.get("blink_rate", {})
    print("[BLINK AUDIT]", blink)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/audit_blink_detection.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
