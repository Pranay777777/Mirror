"""
Debug MediaPipe object types.
"""

import cv2
import mediapipe as mp
import sys

def debug_mediapipe(video_path):
    """Debug what MediaPipe is actually returning."""
    
    print(f"🔍 DEBUGGING MEDIAPIPE: {video_path}")
    print("=" * 60)
    
    mp_holistic = mp.solutions.holistic
    
    cap = cv2.VideoCapture(video_path)
    
    # Test first 3 frames
    frame_count = 0
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=1,
        enable_segmentation=False
    ) as holistic:
        
        while cap.isOpened() and frame_count < 3:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert color space
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            
            print(f"\n📹 Frame {frame_count}:")
            print(f"   Pose landmarks type: {type(results.pose_landmarks)}")
            print(f"   Pose landmarks: {results.pose_landmarks}")
            
            if results.pose_landmarks:
                print(f"   Landmarks type: {type(results.pose_landmarks.landmark)}")
                print(f"   Landmarks length: {len(results.pose_landmarks.landmark)}")
                print(f"   First landmark: {results.pose_landmarks.landmark[0]}")
                print(f"   First landmark type: {type(results.pose_landmarks.landmark[0])}")
            else:
                print("   ❌ No pose landmarks")
    
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_mediapipe.py <path_to_video>")
        sys.exit(1)
    
    debug_mediapipe(sys.argv[1])
