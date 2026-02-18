"""
Debug extraction with detailed logging.
"""

import sys
sys.path.append('.')

from features.normalized_geometry import NormalizedGeometry
import mediapipe as mp
import cv2

def debug_extraction_detailed(video_path):
    """Debug extraction with detailed logging."""
    
    print(f"🔍 DETAILED EXTRACTION DEBUG: {video_path}")
    print("=" * 60)
    
    geometry = NormalizedGeometry()
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
                
                # Test each landmark extraction manually
                try:
                    print("\n   Testing individual landmark extraction:")
                    
                    left_shoulder = geometry._get_landmark_safe(results.pose_landmarks, mp_holistic.PoseLandmark.LEFT_SHOULDER)
                    print(f"   Left shoulder: {left_shoulder}")
                    
                    right_shoulder = geometry._get_landmark_safe(results.pose_landmarks, mp_holistic.PoseLandmark.RIGHT_SHOULDER)
                    print(f"   Right shoulder: {right_shoulder}")
                    
                    nose = geometry._get_landmark_safe(results.pose_landmarks, mp_holistic.PoseLandmark.NOSE)
                    print(f"   Nose: {nose}")
                    
                    print(f"\n   Condition checks:")
                    print(f"   left_shoulder exists: {left_shoulder is not None}")
                    print(f"   right_shoulder exists: {right_shoulder is not None}")
                    print(f"   nose exists: {nose is not None}")
                    print(f"   Both shoulders: {left_shoulder is not None and right_shoulder is not None}")
                    print(f"   All three: {left_shoulder is not None and right_shoulder is not None and nose is not None}")
                    
                    # Test full extraction
                    pose_features = geometry.extract_pose_features(results.pose_landmarks)
                    if 'error' in pose_features:
                        print(f"   ❌ Full extraction error: {pose_features['error']}")
                    else:
                        print(f"   ✅ Full extraction successful: {len(pose_features)} features")
                        
                except Exception as e:
                    print(f"   ❌ Extraction exception: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("   ❌ No pose landmarks detected")
    
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_detailed.py <path_to_video>")
        sys.exit(1)
    
    debug_extraction_detailed(sys.argv[1])
