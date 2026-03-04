"""
diagnose_video.py — Quick video metadata diagnostics.
Usage: python diagnose_video.py <video.mp4>
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import sys, cv2

def main(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fc     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = fc / fps if fps > 0 else 0
    print(f"[VIDEO DIAGNOSTICS] {video_path}")
    print(f"  Resolution : {width}x{height}")
    print(f"  FPS        : {fps}")
    print(f"  Frame count: {fc}")
    print(f"  Duration   : {duration:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_video.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
