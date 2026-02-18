"""
Quick video diagnostic tool to check MediaPipe landmark detection.
"""

import cv2
import mediapipe as mp
import sys

def diagnose_video(video_path):
    """Diagnose video for MediaPipe landmark detection."""
    
    print(f"🔍 DIAGNOSING VIDEO: {video_path}")
    print("=" * 50)
    
    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video Info:")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f} seconds")
    
    # Check first few frames
    frame_count = 0
    pose_detected = 0
    face_detected = 0
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=1,
        enable_segmentation=False
    ) as holistic:
        
        while cap.isOpened() and frame_count < 100:  # Check first 100 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert color space
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            
            # Check detections
            if results.pose_landmarks and len(results.pose_landmarks.landmark) > 0:
                pose_detected += 1
                if frame_count <= 10:
                    print(f"✅ Frame {frame_count}: POSE detected")
            
            if results.face_landmarks and len(results.face_landmarks.landmark) > 0:
                face_detected += 1
                if frame_count <= 10:
                    print(f"✅ Frame {frame_count}: FACE detected")
            
            # Show first 10 frames status
            if frame_count <= 10:
                if not results.pose_landmarks:
                    print(f"❌ Frame {frame_count}: NO POSE detected")
                if not results.face_landmarks:
                    print(f"❌ Frame {frame_count}: NO FACE detected")
    
    cap.release()
    
    # Results
    print("\n📊 DETECTION RESULTS:")
    print("=" * 50)
    print(f"Frames analyzed: {frame_count}")
    print(f"Pose detected: {pose_detected}/{frame_count} ({pose_detected/frame_count*100:.1f}%)")
    print(f"Face detected: {face_detected}/{frame_count} ({face_detected/frame_count*100:.1f}%)")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("=" * 50)
    
    if pose_detected < 50:
        print("❌ LOW POSE DETECTION:")
        print("   - Person may not be fully visible")
        print("   - Try better lighting")
        print("   - Ensure person is facing camera")
        print("   - Check if person is too far/close")
    
    if face_detected < 50:
        print("❌ LOW FACE DETECTION:")
        print("   - Face may be obscured or turned away")
        print("   - Improve lighting on face")
        print("   - Ensure frontal face angle")
        print("   - Check resolution too low")
    
    if pose_detected >= 70 and face_detected >= 70:
        print("✅ GOOD DETECTION RATES!")
        print("   - Video should work well with analysis system")
        print("   - Expect high confidence scores")
    
    print(f"\n🎯 OVERALL: {'GOOD' if pose_detected >= 50 and face_detected >= 50 else 'NEEDS IMPROVEMENT'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_video.py <path_to_video>")
        sys.exit(1)
    
    diagnose_video(sys.argv[1])
