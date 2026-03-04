"""verify_head_pose_fix.py — Confirms head pose is not returning None for valid frames."""
import sys, cv2
from dotenv import load_dotenv
load_dotenv()

def main(video_path: str):
    from features.normalized_geometry import NormalizedGeometry
    geom = NormalizedGeometry()
    cap = cv2.VideoCapture(video_path)
    has_head = not_head = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        geo = geom.process(frame)
        hp = geo.get("head_pose")
        if hp is not None:
            has_head += 1
        else:
            not_head += 1
    cap.release()
    print(f"[HEAD POSE FIX] has_head={has_head}, no_head={not_head}")
    if has_head > 0:
        print("  ✓ head_pose data is present in at least some frames")
    else:
        print("  ✗ head_pose was None for ALL frames — check NormalizedGeometry")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_head_pose_fix.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
