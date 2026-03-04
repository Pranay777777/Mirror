"""
diagnose_coverage.py — Diagnoses landmark coverage (pose/face) across every frame.
Usage: python diagnose_coverage.py <video.mp4>
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
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    pose_count = face_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = geom.process(frame)
        if result.get("torso_inclination_deg") is not None:
            pose_count += 1
        if result.get("expression_score") is not None or result.get("blink_detected") is not None:
            face_count += 1
        frame_idx += 1
    cap.release()
    total = max(frame_idx, 1)
    print(f"[COVERAGE] Total frames: {frame_idx}")
    print(f"  Pose coverage : {pose_count}/{total} = {pose_count/total:.2%}")
    print(f"  Face coverage : {face_count}/{total} = {face_count/total:.2%}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_coverage.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
