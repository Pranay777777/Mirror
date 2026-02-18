import sys
import os
import cv2
import pandas as pd
import numpy as np
import logging

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.normalized_geometry import NormalizedGeometry
from features.temporal_features import TemporalFeatures
from utils.face_preprocessing import FacePreprocessor
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def audit_blink_detection(video_path):
    print(f"Starting Blink Detection Audit on: {video_path}")
    
    # Initialize Geometry
    geometry = NormalizedGeometry()
    
    # Initialize TemporalFeatures with AUDIT overrides
    # - Disable smoothing (smoothing_window=1 effectively disables it?)
    #   Wait, _smooth method uses window=5 by default but checks self.smoothing_window.
    # - blink_z_threshold = -1.5
    # - blink_min_consecutive = 1
    
    temporal = TemporalFeatures(
        fps=30.0, # Will update later
        blink_z_threshold=-1.5,
        blink_min_consecutive=1,
        smoothing_window=1 # Effectively disables smoothing
    )
    
    # Setup MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    temporal.fps = fps
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    geometry.set_frame_dimensions(frame_width, frame_height)
    
    print(f"Video FPS: {fps}, Resolution: {frame_width}x{frame_height}")
    
    ear_history = []
    frames_count = 0
    
    print("\n--- Frame-by-Frame Log (First 200) ---")
    print("Frame | Raw EAR | Smoothed EAR | Z-Score | Blink Candidate")
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_count += 1
            
            # Extract EAR
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = face_mesh.process(image)
            
            raw_ear = np.nan
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                face_features = geometry.extract_face_features(face_landmarks)
                
                if 'left_eye_opening_ratio' in face_features and 'right_eye_opening_ratio' in face_features:
                    l = face_features['left_eye_opening_ratio']
                    r = face_features['right_eye_opening_ratio']
                    raw_ear = (l + r) / 2.0
            
            ear_history.append(raw_ear)
            
            # We need history to compute Z-Score
            # But the audit request says "Log for each frame... z-score".
            # Z-score depends on mean/std of the *whole video* (in adaptive mode) or run-time?
            # The current implementation `_detect_blinks_zscore` calculates z-score on the *entire series* passed to it.
            # So we can't compute the final z-score frame-by-frame until we have the whole history?
            # Or should we compute z-score based on history *so far*?
            # The user code `_detect_blinks_zscore` uses `clean.mean()` of the passed series.
            # I will collect all EARs first, then compute Z-scores validly, then print the log.
            # But the user asked to "Log for each frame".
            # I'll collect all, then print the log for the first 200.
            
    cap.release()
    
    # Convert to Series
    ear_series = pd.Series(ear_history)
    clean_series = ear_series.dropna()
    
    if len(clean_series) < 5:
        print("Insufficient data for z-score analysis.")
        return

    # Calculate Z-Scores (Adaptive)
    mean_ear = clean_series.mean()
    std_ear = clean_series.std()
    z_scores = (ear_series - mean_ear) / std_ear
    
    # Identify Candidates (Threshold -1.5)
    candidates = z_scores < -1.5
    
    # ---------------------------------------------------------
    # DEBUG BLINK DETECTOR (Shadow Copy of Production Logic)
    # ---------------------------------------------------------
    def debug_detect_blinks_zscore(self, ear_series: pd.Series):
        print("\n--- Running Debug Blink Aggregation ---")
        clean = ear_series.dropna()
        if len(clean) < 5:
            return 0, [], 0.0

        mean_ear = clean.mean()
        std_ear = clean.std()
        if std_ear < 1e-8:
            return 0, [], 0.0

        z_scores = (clean - mean_ear) / std_ear

        # Find runs below z-threshold
        below = z_scores < self.blink_z_threshold
        blink_events = []
        run_start = None

        print(f"Debug Config: Z-Thresh={self.blink_z_threshold}, MinConsec={self.blink_min_consecutive}, FPS={self.fps}")

        for i in range(len(below)):
            idx = below.index[i]
            if below.iloc[i]:
                if run_start is None:
                    run_start = i
                    print(f"Frame {i}: Blink Start (Z={z_scores.iloc[i]:.4f})")
            else:
                if run_start is not None:
                    run_len = i - run_start
                    print(f"Frame {i}: Blink End Candidate (Len={run_len})")
                    
                    if run_len >= self.blink_min_consecutive:
                        duration_ms = (run_len / self.fps) * 1000.0
                        print(f"  -> Duration: {duration_ms:.2f}ms (Min: {self.blink_min_duration_ms}ms)")
                        
                        if self.blink_min_duration_ms <= duration_ms <= self.blink_max_duration_ms:
                            # Verify EAR recovery: value after run must be > mean - 1 std
                            recovery_ok = True
                            if i < len(clean):
                                recovery_limit = mean_ear - std_ear
                                recovery_ok = clean.iloc[i] > recovery_limit
                                print(f"  -> Recovery Check: Val={clean.iloc[i]:.4f} > Limit={recovery_limit:.4f}? {recovery_ok}")
                                
                            if recovery_ok:
                                print(f"  -> BLINK REGISTERED! Frames {run_start} to {i-1}")
                                blink_events.append({
                                    "start_idx": run_start,
                                    "end_idx": i - 1,
                                    "duration_ms": duration_ms,
                                })
                            else:
                                print("  -> Rejected: Recovery failed")
                        else:
                            print("  -> Rejected: Duration out of bounds")
                    else:
                        print("  -> Rejected: Too short (consecutive frames)")
                        
                    run_start = None

        # Handle run that extends to end of series
        if run_start is not None:
            run_len = len(below) - run_start
            print(f"End of Video: Blink End Candidate (Len={run_len})")
            if run_len >= self.blink_min_consecutive:
                duration_ms = (run_len / self.fps) * 1000.0
                if self.blink_min_duration_ms <= duration_ms <= self.blink_max_duration_ms:
                     print(f"  -> BLINK REGISTERED! Frames {run_start} to {len(below)-1}")
                     blink_events.append({
                        "start_idx": run_start,
                        "end_idx": len(below) - 1,
                        "duration_ms": duration_ms,
                    })

        blink_count = len(blink_events)
        print(f"\nFINAL BLINK COUNT: {blink_count}")
        
        # Duration for Rate
        duration_min = len(clean) / self.fps / 60.0
        print(f"Duration (min): {duration_min:.4f}")
        if duration_min > 0:
            print(f"Blink Rate: {blink_count / duration_min:.2f} per min")

        # Confidence (simplified for debug)
        return blink_count, blink_events, 1.0

    # Monkey patch
    temporal._detect_blinks_zscore = debug_detect_blinks_zscore.__get__(temporal, TemporalFeatures)
    
    # Run Detection
    temporal._detect_blinks_zscore(ear_series)

    # Convert to Series
    # ear_series = pd.Series(ear_history) # Already converted above
    pass # Cleanup placeholder logic below as we injected above
        
    """
    # Printing
    for i in range(min(200, len(ear_history))):
        ear_val = ear_history[i]
        z_val = z_scores.iloc[i] if i in z_scores else np.nan
        is_cand = candidates.iloc[i] if i in candidates else False
        
        # Smoothed is same as raw here
        print(f"{i:5d} | {ear_val:.6f} | {ear_val:.6f} | {z_val:.4f} | {is_cand}")
    """
    
    print(f"\n--- Summary from Audit ---")
    print(f"Total Frames: {frames_count}")
    print(f"Mean EAR: {mean_ear:.6f}, Std EAR: {std_ear:.6f}")
    print(f"Candidates Found: {candidates.sum()}")

if __name__ == "__main__":
    video_path = "uploads/videoB1.mp4"
    if not os.path.exists(video_path):
        # Fallback search
        import glob
        matches = glob.glob(f"**/{os.path.basename(video_path)}", recursive=True)
        if matches:
            video_path = matches[0]
            
    audit_blink_detection(video_path)
