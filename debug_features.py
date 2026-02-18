import sys
import cv2
import mediapipe as mp
import numpy as np
from features.normalized_geometry import NormalizedGeometry

def debug_features(video_path):
    print(f"🔍 DEBUGGING FEATURES FOR: {video_path}")
    
    mp_holistic = mp.solutions.holistic
    geometry = NormalizedGeometry()
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    
    frame_count = 0
    
    # buffers for stats
    smiles = []
    left_eyes = []
    mouth_centers_y = []
    eyebrow_centers_y = []
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        while cap.isOpened() and frame_count < 50: # Check first 50 frames
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # ... process ...
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            if results.face_landmarks:
                feats = geometry.extract_face_features(results.face_landmarks)
                
                if feats and 'error' not in feats:
                    if 'smile_intensity' in feats:
                        smiles.append(feats['smile_intensity'])
                    if 'left_eye_opening_ratio' in feats:
                        left_eyes.append(feats['left_eye_opening_ratio'])
    
    cap.release()
    
    print("\n📊 STATS:", flush=True)
    print(f"Frames processed: {frame_count}", flush=True)
    
    if smiles:
        print(f"\n😊 Smile Intensity:", flush=True)
        print(f"   Max: {max(smiles):.6f}", flush=True)
        print(f"   Mean: {sum(smiles)/len(smiles):.6f}", flush=True)
        print(f"   Count > 0: {sum(1 for s in smiles if s > 0)}", flush=True)
    
    if left_eyes:
        print(f"\n👁️ Left Eye EAR:", flush=True)
        print(f"   Max: {max(left_eyes):.6f}", flush=True)
        print(f"   Min: {min(left_eyes):.6f}", flush=True)
        print(f"   Mean: {sum(left_eyes)/len(left_eyes):.6f}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_features.py <video>")
    else:
        debug_features(sys.argv[1])
