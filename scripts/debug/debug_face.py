"""
debug_face.py — Debugs face/landmark detection on a video frame by frame.
Usage: python debug_face.py <video.mp4>
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import sys, cv2
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from features.normalized_geometry import NormalizedGeometry
    geom = NormalizedGeometry()
    cap = cv2.VideoCapture(video_path)
    detected = 0
    total = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total += 1
        result = geom.process(frame)
        face_data = result.get("face_landmarks") or result.get("face")
        if face_data:
            detected += 1
    cap.release()
    ratio = detected / max(total, 1)
    print(f"[FACE DEBUG] Detected: {detected}/{total} frames | ratio={ratio:.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_face.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
