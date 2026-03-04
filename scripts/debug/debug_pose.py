"""
debug_pose.py — Debugs pose/torso detection frame by frame.
Usage: python debug_pose.py <video.mp4>
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
    pose_hits = 0
    samples = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = geom.process(frame)
        torso = result.get("torso_inclination_deg")
        if torso is not None:
            pose_hits += 1
            if frame_idx % 30 == 0:
                samples.append({"frame": frame_idx, "torso_inclination_deg": torso})
        frame_idx += 1
    cap.release()
    print(f"[POSE DEBUG] pose_hits={pose_hits}/{frame_idx} ({pose_hits/max(frame_idx,1):.2%})")
    print("Samples:", json.dumps(samples[:10], indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_pose.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
