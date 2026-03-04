"""debug_eye5_frames.py — Dumps eye/iris landmark coordinates for the first 5 detected frames."""
import sys, cv2
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from features.normalized_geometry import NormalizedGeometry
    geom = NormalizedGeometry()
    cap = cv2.VideoCapture(video_path)
    found = 0
    idx = 0
    while cap.isOpened() and found < 5:
        ret, frame = cap.read()
        if not ret: break
        geo = geom.process(frame)
        iris = geo.get("iris_data") or geo.get("left_iris") or geo.get("eye_data")
        if iris is not None:
            print(f"  Frame {idx} iris_data: {iris}")
            found += 1
        idx += 1
    cap.release()
    if found == 0:
        print("[EYE5] No iris/eye data detected in this video.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/debug_eye5_frames.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
