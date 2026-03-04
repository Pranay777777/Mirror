"""iris_percentiles.py — Computes iris position percentile statistics on a video."""
import sys, cv2, json
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from features.normalized_geometry import NormalizedGeometry
    import numpy as np
    geom = NormalizedGeometry()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    iris_values = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        geo = geom.process(frame)
        v = geo.get("iris_relative_position") or geo.get("left_iris_x")
        if v is not None:
            iris_values.append(float(v))
        idx += 1
    cap.release()
    if iris_values:
        arr = np.array(iris_values)
        print(f"[IRIS PERCENTILES] n={len(arr)}")
        for p in [5, 25, 50, 75, 95]:
            print(f"  p{p:02d}: {np.percentile(arr, p):.4f}")
    else:
        print("[IRIS PERCENTILES] No iris data found.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/iris_percentiles.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
