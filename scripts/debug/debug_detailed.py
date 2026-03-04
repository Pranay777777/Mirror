"""
debug_detailed.py — Prints per-frame geometry debug info for a video.
Usage: python debug_detailed.py <video.mp4>
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
    frame_idx = 0
    print(f"[DEBUG DETAILED] Video: {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = geom.process(frame)
        if frame_idx % 30 == 0:  # Print every 30th frame
            print(f"  Frame {frame_idx}: {result}")
        frame_idx += 1
    cap.release()
    print(f"[DONE] Processed {frame_idx} frames.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_detailed.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
