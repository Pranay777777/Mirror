"""
debug_mediapipe.py — Tests MediaPipe landmark detection on a single video frame.
Usage: python debug_mediapipe.py <video.mp4>
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import sys, cv2
import mediapipe as mp

def main(video_path: str):
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Could not read frame from video.")
        return
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        results = holistic.process(rgb)
        print("[MEDIAPIPE] pose_landmarks:", "DETECTED" if results.pose_landmarks else "NOT DETECTED")
        print("[MEDIAPIPE] face_landmarks:", "DETECTED" if results.face_landmarks else "NOT DETECTED")
        print("[MEDIAPIPE] left_hand_landmarks:", "DETECTED" if results.left_hand_landmarks else "NOT DETECTED")
        print("[MEDIAPIPE] right_hand_landmarks:", "DETECTED" if results.right_hand_landmarks else "NOT DETECTED")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_mediapipe.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
