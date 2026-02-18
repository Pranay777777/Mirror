
import cv2
import mediapipe as mp
from features.normalized_geometry import NormalizedGeometry

def diagnose_coverage(video_path):
    print(f"🔍 COVERAGE DIAGNOSIS for: {video_path}")
    
    mp_holistic = mp.solutions.holistic
    geometry = NormalizedGeometry()
    
    cap = cv2.VideoCapture(video_path)
    
    # OpenCV metadata
    total_frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames_cv / fps if fps > 0 else 0
    
    print(f"\n📊 VIDEO METADATA:")
    print(f"   OpenCV Frame Count: {total_frames_cv}")
    print(f"   FPS: {fps}")
    print(f"   Duration: {duration:.1f}s")
    
    frames_read = 0
    frames_with_pose_lm = 0
    frames_with_face_lm = 0
    frames_with_valid_pose = 0
    frames_with_valid_face = 0
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=2,
        refine_face_landmarks=True
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"\n   cap.read() returned False at frame {frames_read + 1}. Exiting loop.")
                break
                
            frames_read += 1
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            if results.pose_landmarks:
                frames_with_pose_lm += 1
                feats = geometry.extract_pose_features(results.pose_landmarks)
                if feats and 'error' not in feats:
                    frames_with_valid_pose += 1
            
            if results.face_landmarks:
                frames_with_face_lm += 1
                feats = geometry.extract_face_features(results.face_landmarks)
                if feats and 'left_eye_opening_ratio' in feats:
                    frames_with_valid_face += 1
            
            if frames_read % 200 == 0:
                print(f"   ... {frames_read} frames read")

    cap.release()
    
    print(f"\n📊 FRAME COUNTS:")
    print(f"   OpenCV reported:       {total_frames_cv}")
    print(f"   Frames actually read:  {frames_read}")
    print(f"   Frames w/ Pose LM:     {frames_with_pose_lm} ({frames_with_pose_lm/frames_read*100:.1f}%)")
    print(f"   Frames w/ Valid Pose:  {frames_with_valid_pose} ({frames_with_valid_pose/frames_read*100:.1f}%)")
    print(f"   Frames w/ Face LM:     {frames_with_face_lm} ({frames_with_face_lm/frames_read*100:.1f}%)")
    print(f"   Frames w/ Valid Face:  {frames_with_valid_face} ({frames_with_valid_face/frames_read*100:.1f}%)")
    
    print(f"\n🏁 DIAGNOSIS CONCLUSION:")
    if frames_read < total_frames_cv * 0.9:
        print(f"   [BUG] Loop terminated early! Only {frames_read}/{total_frames_cv} frames read.")
    else:
        print(f"   [OK] All frames were read. The 12% 'successful_frames' is detection failure, not loop termination.")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "uploads/short_test.mp4"
    diagnose_coverage(path)
