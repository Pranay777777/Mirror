"""
Test landmark extraction directly.
"""

import sys
sys.path.append('.')

from features.normalized_geometry import NormalizedGeometry
import mediapipe as mp
import cv2

def test_landmark_extraction(video_path):
    """Test landmark extraction directly."""
    
    print(f"🔍 TESTING LANDMARK EXTRACTION: {video_path}")
    print("=" * 60)
    
    geometry = NormalizedGeometry()
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
            
            if results.pose_landmarks:
                print(f"   ✅ Pose landmarks detected: {len(results.pose_landmarks.landmark)} points")
                
                # Test extraction
                try:
                    pose_features = geometry.extract_pose_features(results.pose_landmarks)
                    if 'error' in pose_features:
                        print(f"   ❌ Extraction error: {pose_features['error']}")
                    else:
                        print(f"   ✅ Extraction successful: {len(pose_features)} features")
                        print(f"   Features: {list(pose_features.keys())}")
                except Exception as e:
                    print(f"   ❌ Extraction exception: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("   ❌ No pose landmarks detected")
    
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_extraction.py <path_to_video>")
        sys.exit(1)
    
    test_landmark_extraction(sys.argv[1])
