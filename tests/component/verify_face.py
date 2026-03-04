"""
verify_face.py — Verifies face detection metrics (blink, expression) on a video.
Usage: python verify_face.py <video.mp4>
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
    print("[FACE VERIFY] expression_score:", results.get("expression_score"))
    print("[FACE VERIFY] blink_rate:", results.get("blink_rate"))
    print(json.dumps({k: v for k, v in results.items() if "express" in k or "blink" in k or "face" in k}, indent=2, default=str))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_face.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
