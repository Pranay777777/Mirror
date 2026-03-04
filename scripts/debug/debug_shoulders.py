"""
debug_shoulders.py — Debugs shoulder detection and tilt angle over time.
Usage: python debug_shoulders.py <video.mp4>
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import sys, cv2, json
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from features.normalized_geometry import NormalizedGeometry
    geom = NormalizedGeometry()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    tilt_values = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = geom.process(frame)
        tilt = result.get("shoulder_tilt_angle")
        if tilt is not None:
            tilt_values.append(round(float(tilt), 3))
        frame_idx += 1
    cap.release()
    if tilt_values:
        import statistics
        print(f"[SHOULDERS] frames={frame_idx}, shoulder_detected={len(tilt_values)}")
        print(f"  mean={statistics.mean(tilt_values):.3f}")
        print(f"  stdev={statistics.stdev(tilt_values) if len(tilt_values) > 1 else 0:.3f}")
        print(f"  min={min(tilt_values):.3f}, max={max(tilt_values):.3f}")
    else:
        print("[SHOULDERS] No shoulder data detected in this video.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_shoulders.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
