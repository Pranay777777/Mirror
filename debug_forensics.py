
import pandas as pd
import numpy as np
from features.normalized_geometry import NormalizedGeometry
from features.temporal_features import TemporalFeatures
import mediapipe as mp

class MockLandmark:
    def __init__(self, x, y, z, visibility=0.9):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class MockLandmarks:
    def __init__(self, landmarks_dict):
        # MediaPipe landmarks are lists/protobufs. We mock list access.
        # landmarks_dict: {index: MockLandmark}
        self.landmark = [MockLandmark(0,0,0,0) for _ in range(500)] # valid size
        for idx, lm in landmarks_dict.items():
            self.landmark[idx] = lm

def test_scenario(name, pose_lms, face_lms):
    print(f"\n==========================================")
    print(f"SCENARIO: {name}")
    print(f"==========================================")
    
    geo = NormalizedGeometry()
    temp = TemporalFeatures()
    
    # Extract
    if pose_lms:
        p_feat = geo.extract_pose_features(pose_lms)
    else:
        p_feat = {} # Simulate missing
        
    if face_lms:
        f_feat = geo.extract_face_features(face_lms)
    else:
        f_feat = {}
        
    print(f"1) Extraction Result:")
    print(f"   Pose keys: {list(p_feat.keys())}")
    print(f"   Face keys: {list(f_feat.keys())}")
    
    # Check for Forbidden Defaults
    defaults = ['shoulder_tilt_angle', 'eye_distance_ratio']
    for d in defaults:
        if d in p_feat:
            print(f"   [CHECK] {d}: {p_feat[d]}")
            
    # Temporal Processing (simulate 10 frames of this)
    for _ in range(10):
        temp.add_frame_features(p_feat, f_feat, 0.1)
        
    results = temp.extract_temporal_features()
    
    print(f"2) Temporal Results:")
    for k, v in results.items():
        if isinstance(v, dict):
            val = v.get('value')
            reason = v.get('reason')
            print(f"   METRIC: {k}")
            print(f"     Value: {val}")
            print(f"     Reason: {reason}")
            
            # DIAGNOSIS
            if val is not None and val == 0.0 and "default" in name.lower():
                 print(f"     [FAILURE] Zero returned for missing data!")
            elif val is not None and reason == "insufficient_data":
                 print(f"     [SUCCESS] Correctly identified insufficient data.")

# Define Indices
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

def run_forensics():
    # Scenario A: Head Only (Shoulders Missing)
    # This triggers Tier 2 in normalized_geometry
    head_only = MockLandmarks({
        NOSE: MockLandmark(0.5, 0.2, 0),
        LEFT_EYE: MockLandmark(0.45, 0.15, 0),
        RIGHT_EYE: MockLandmark(0.55, 0.15, 0)
    })
    
    test_scenario("A: Head Only (Shoulders Missing)", head_only, None)

    # Scenario B: Shoulders Only (Head Missing)
    # This triggers Tier 3
    shoulders_only = MockLandmarks({
        LEFT_SHOULDER: MockLandmark(0.4, 0.5, 0),
        RIGHT_SHOULDER: MockLandmark(0.6, 0.5, 0)
    })
    
    test_scenario("B: Shoulders Only (Head Missing)", shoulders_only, None)

    # Scenario C: Empty/None
    test_scenario("C: No Landmarks", None, None)

if __name__ == "__main__":
    run_forensics()
