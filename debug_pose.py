"""
Debug pose landmarks specifically.
"""

import cv2
import mediapipe as mp
import sys

def debug_pose_landmarks(video_path):
    """Debug pose landmark detection specifically."""
    
    print(f"🔍 DEBUGGING POSE LANDMARKS: {video_path}")
    print("=" * 60)
    
    mp_holistic = mp.solutions.holistic
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Test first 10 frames
    frame_count = 0
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=1,
        enable_segmentation=False
    ) as holistic:
        
        while cap.isOpened() and frame_count < 10:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert color space
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            
            print(f"\n📹 Frame {frame_count}:")
            
            # Check pose landmarks
            if results.pose_landmarks:
                print(f"   ✅ Pose landmarks detected: {len(results.pose_landmarks.landmark)} points")
                
                # Check specific landmarks we need
                left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
                left_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP.value]
                right_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP.value]
                
                print(f"   Left Shoulder: visibility={getattr(left_shoulder, 'visibility', 'N/A')}, presence={getattr(left_shoulder, 'presence', 'N/A')}")
                print(f"   Right Shoulder: visibility={getattr(right_shoulder, 'visibility', 'N/A')}, presence={getattr(right_shoulder, 'presence', 'N/A')}")
                print(f"   Left Hip: visibility={getattr(left_hip, 'visibility', 'N/A')}, presence={getattr(left_hip, 'presence', 'N/A')}")
                print(f"   Right Hip: visibility={getattr(right_hip, 'visibility', 'N/A')}, presence={getattr(right_hip, 'presence', 'N/A')}")
                
            else:
                print("   ❌ No pose landmarks detected")
            
            # Check face landmarks
            if results.face_landmarks:
                print(f"   ✅ Face landmarks detected: {len(results.face_landmarks.landmark)} points")
            else:
                print("   ❌ No face landmarks detected")
    
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_pose.py <path_to_video>")
        sys.exit(1)
    
    debug_pose_landmarks(sys.argv[1])
