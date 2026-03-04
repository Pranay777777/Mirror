"""verify_iris.py — Verifies iris tracking data is captured over a video."""
import sys, cv2
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from features.normalized_geometry import NormalizedGeometry
    geom = NormalizedGeometry()
    cap = cv2.VideoCapture(video_path)
    iris_count = total = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        total += 1
        geo = geom.process(frame)
        iris = geo.get("iris_relative_position") or geo.get("left_iris_x") or geo.get("iris_data")
        if iris is not None:
            iris_count += 1
    cap.release()
    print(f"[IRIS] frames with iris data: {iris_count}/{total} ({iris_count/max(total,1):.2%})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_iris.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
