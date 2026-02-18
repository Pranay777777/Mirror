"""
Debug what face features are actually extracted.
"""

import sys
sys.path.append('.')

from features.normalized_geometry import NormalizedGeometry
import mediapipe as mp
import cv2

def debug_face_features(video_path):
    """Debug face feature extraction."""
    
    print(f"🔍 DEBUGGING FACE FEATURES: {video_path}")
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
            
            if results.face_landmarks:
                print(f"   ✅ Face landmarks detected: {len(results.face_landmarks.landmark)} points")
                
                try:
                    face_features = geometry.extract_face_features(results.face_landmarks)
                    if 'error' in face_features:
                        print(f"   ❌ Face extraction error: {face_features['error']}")
                    else:
                        print(f"   ✅ Face extraction successful: {len(face_features)} features")
                        print(f"   Available features: {list(face_features.keys())}")
                        
                        # Check for eye opening ratios
                        if 'left_eye_opening_ratio' in face_features:
                            print(f"   ✅ left_eye_opening_ratio: {face_features['left_eye_opening_ratio']}")
                        else:
                            print(f"   ❌ left_eye_opening_ratio: MISSING")
                            
                        if 'right_eye_opening_ratio' in face_features:
                            print(f"   ✅ right_eye_opening_ratio: {face_features['right_eye_opening_ratio']}")
                        else:
                            print(f"   ❌ right_eye_opening_ratio: MISSING")
                        
                except Exception as e:
                    print(f"   ❌ Face extraction exception: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("   ❌ No face landmarks detected")
    
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_face.py <path_to_video>")
        sys.exit(1)
    
    debug_face_features(sys.argv[1])
