"""
Debug shoulder landmark extraction specifically.
"""

import cv2
import mediapipe as mp
import sys

def debug_shoulder_extraction(video_path):
    """Debug shoulder landmark extraction specifically."""
    
    print(f"🔍 DEBUGGING SHOULDER EXTRACTION: {video_path}")
    print("=" * 60)
    
    mp_holistic = mp.solutions.holistic
    
    cap = cv2.VideoCapture(video_path)
    
    # Test first frame
    frame_count = 0
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=1,
        enable_segmentation=False
    ) as holistic:
        
        while cap.isOpened() and frame_count < 1:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert color space
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            
            print(f"\n📹 Frame {frame_count}:")
            
            if results.pose_landmarks:
                print(f"   ✅ Pose landmarks detected: {len(results.pose_landmarks.landmark)} points")
                
                # Test shoulder extraction manually
                try:
                    left_shoulder_idx = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
                    right_shoulder_idx = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value
                    
                    print(f"   Left shoulder index: {left_shoulder_idx}")
                    print(f"   Right shoulder index: {right_shoulder_idx}")
                    
                    left_shoulder_lm = results.pose_landmarks.landmark[left_shoulder_idx]
                    right_shoulder_lm = results.pose_landmarks.landmark[right_shoulder_idx]
                    
                    print(f"   Left shoulder: x={left_shoulder_lm.x}, y={left_shoulder_lm.y}, z={left_shoulder_lm.z}, visibility={left_shoulder_lm.visibility}")
                    print(f"   Right shoulder: x={right_shoulder_lm.x}, y={right_shoulder_lm.y}, z={right_shoulder_lm.z}, visibility={right_shoulder_lm.visibility}")
                    
                    # Test visibility check
                    left_vis = left_shoulder_lm.visibility
                    right_vis = right_shoulder_lm.visibility
                    
                    print(f"   Left visibility check: {left_vis} < 0.1 = {left_vis < 0.1}")
                    print(f"   Right visibility check: {right_vis} < 0.1 = {right_vis < 0.1}")
                    
                    if left_vis >= 0.1 and right_vis >= 0.1:
                        print("   ✅ BOTH SHOULDERS SHOULD BE VISIBLE!")
                    else:
                        print("   ❌ Shoulders filtered out by visibility check")
                        
                except Exception as e:
                    print(f"   ❌ Shoulder extraction error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("   ❌ No pose landmarks detected")
    
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_shoulders.py <path_to_video>")
        sys.exit(1)
    
    debug_shoulder_extraction(sys.argv[1])
