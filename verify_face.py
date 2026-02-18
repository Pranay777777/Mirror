
import cv2
import mediapipe as mp
import pandas as pd
from features.normalized_geometry import NormalizedGeometry

def verify_face(video_path):
    print(f"🕵️ Verifying Face Extraction for: {video_path}")
    
    mp_holistic = mp.solutions.holistic
    geometry = NormalizedGeometry()
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞️ Total Frames: {total_frames}")
    
    frames_with_face_lms = 0
    frames_with_face_feats = 0
    frames_with_pose_lms = 0
    frames_with_pose_feats = 0
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=2,
        refine_face_landmarks=True
    ) as holistic:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            # Check MediaPipe Output
            if results.face_landmarks:
                frames_with_face_lms += 1
                feats = geometry.extract_face_features(results.face_landmarks)
                if feats and 'left_eye_opening_ratio' in feats:
                    frames_with_face_feats += 1
            
            if results.pose_landmarks:
                frames_with_pose_lms += 1
                # Check what my new logic returns
                p_feats = geometry.extract_pose_features(results.pose_landmarks)
                if p_feats and 'error' not in p_feats:
                     frames_with_pose_feats += 1
            
            if frame_idx % 100 == 0:
                print(f"   Processed {frame_idx}...")

    cap.release()
    
    print("\n📊 RESULTS:")
    print(f"   MediaPipe Face Detected: {frames_with_face_lms}/{frame_idx} ({(frames_with_face_lms/frame_idx)*100:.1f}%)")
    print(f"   Geometry Face Extracted: {frames_with_face_feats}/{frame_idx} ({(frames_with_face_feats/frame_idx)*100:.1f}%)")
    print(f"   MediaPipe Pose Detected: {frames_with_pose_lms}/{frame_idx}")
    print(f"   Geometry Pose Extracted: {frames_with_pose_feats}/{frame_idx}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "uploads/short_test.mp4"
    verify_face(path)
