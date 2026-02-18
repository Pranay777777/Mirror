"""
Camera-invariant geometric features for human pose analysis.

All features are dimensionless ratios or angles, ensuring:
- Same human = same score across camera distances
- No dependency on resolution, zoom, or camera position
- Scientifically defensible normalization

v2.0 — Adaptive behavioral modeling:
- 6-point EAR for robust blink detection
- Head pose estimation (yaw, pitch, roll) via solvePnP
- Gaze vector from iris landmarks + head pose
- Enhanced expression features (eyebrow raise, jaw opening)
"""

import math
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
import mediapipe as mp

mp_holistic = mp.solutions.holistic

# Generic 3D face model points (canonical coordinates in mm).
# Used for solvePnP head-pose estimation.
# Order: nose tip, chin, left eye outer, right eye outer, left mouth, right mouth
_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -63.6, -12.5),      # Chin
    (-43.3, 32.7, -26.0),     # Left eye outer corner
    (43.3, 32.7, -26.0),      # Right eye outer corner
    (-28.9, -28.9, -24.1),    # Left mouth corner
    (28.9, -28.9, -24.1),     # Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe Face Mesh landmark indices
_POSE_LANDMARK_IDS = [1, 199, 33, 263, 61, 291]

# 6-point EAR landmark indices (MediaPipe Face Mesh)
# Left eye: outer(33), top1(160), top2(159), inner(133), bot1(144), bot2(145)
_LEFT_EYE_EAR = [33, 160, 159, 133, 144, 145]
# Right eye: inner(362), top1(385), top2(386), outer(263), bot1(380), bot2(374)
_RIGHT_EYE_EAR = [362, 385, 386, 263, 380, 374]

# Iris center landmarks (available when refine_face_landmarks=True)
_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 473


class NormalizedGeometry:
    """Extract camera-invariant geometric features from MediaPipe landmarks."""
    
    def __init__(self):
        self.confidence_threshold = 0.1  # Lowered from 0.5 to be more lenient
        # Approximate camera matrix (updated per-frame from image dimensions)
        self._camera_matrix = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        # Initialize MediaPipe Holistic here to encapsulate it
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,  # Critical for iris/gaze
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame: np.ndarray) -> Dict:
        """
        Process a single video frame using MediaPipe Holistic and extract features.
        
        Args:
            frame: BGR numpy array from cv2.VideoCapture
            
        Returns:
            Dict containing combined 'pose_features' and 'face_features'.
            Also includes 'head_pose' to support legacy checks.
        """
        if frame is None:
            return {}
            
        # Update camera matrix based on frame dimensions
        h, w = frame.shape[:2]
        if self._camera_matrix is None:
            self.set_frame_dimensions(w, h)
            
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process
        results = self.holistic.process(image)
        
        # Extract features
        pose_features = self.extract_pose_features(results.pose_landmarks)
        face_features = self.extract_face_features(results.face_landmarks)
        
        # Combine results
        # We merge them into a single flat dictionary, or keep them separate?
        # looking at video_utils usage:
        # temporal_analyzer.add_frame_features(geo_results)
        # if 'head_pose' in geo_results: ...
        
        combined = {}
        combined.update(pose_features)
        combined.update(face_features)
        
        # Synthesize 'head_pose' key for compatibility if face features yielded it
        if 'head_yaw' in face_features:
            combined['head_pose'] = (
                face_features['head_yaw'],
                face_features['head_pitch'],
                face_features['head_roll']
            )
            
        return combined

    def set_frame_dimensions(self, width: int, height: int):
        """Set camera intrinsics from frame dimensions (call once per video)."""
        focal_length = width  # Approximate: focal_length ≈ image width
        cx, cy = width / 2.0, height / 2.0
        self._camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
    def _get_landmark_safe(self, landmarks, landmark_type) -> Optional[Dict]:
        """Safely extract landmark with visibility check."""
        try:
            # Direct access to landmarks
            lm = landmarks.landmark[landmark_type.value]
            
            # Get visibility from protobuf object
            visibility = lm.visibility if hasattr(lm, 'visibility') else 1.0
            
            # More lenient visibility check for different body parts
            if landmark_type in [mp_holistic.PoseLandmark.LEFT_HIP, mp_holistic.PoseLandmark.RIGHT_HIP]:
                threshold = 0.01  # Very lenient for hips
            else:
                threshold = 0.1  # Normal threshold for other landmarks
            
            if visibility < threshold:
                return None
            
            # Extract coordinates from protobuf object
            return {'x': lm.x, 'y': lm.y, 'z': lm.z}
                
        except Exception:
            return None
    
    def _calculate_distance_3d(self, p1: Dict, p2: Dict) -> float:
        """Calculate 3D Euclidean distance between two points."""
        if not p1 or not p2:
            return 0.0
        return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)
    
    def _calculate_distance_2d(self, p1: Dict, p2: Dict) -> float:
        """Calculate 2D Euclidean distance (ignoring depth)."""
        if not p1 or not p2:
            return 0.0
        return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
    
    def _calculate_angle_3d(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calculate angle at p2 formed by p1-p2-p3 (in degrees)."""
        if not all([p1, p2, p3]):
            return 0.0
        
        # Create vectors
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))

    # ------------------------------------------------------------------
    # 6-point Eye Aspect Ratio
    # ------------------------------------------------------------------
    @staticmethod
    def _ear_6point(landmarks, indices: List[int]) -> float:
        """
        Compute 6-point Eye Aspect Ratio.

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

        indices order: [p1_outer, p2_top1, p3_top2, p4_inner, p5_bot1, p6_bot2]
        """
        pts = []
        for idx in indices:
            lm = landmarks.landmark[idx]
            pts.append(np.array([lm.x, lm.y]))

        p1, p2, p3, p4, p5, p6 = pts
        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        if horizontal < 1e-8:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    # ------------------------------------------------------------------
    # Head Pose Estimation
    # ------------------------------------------------------------------
    def _estimate_head_pose(self, face_landmarks) -> Optional[Tuple[float, float, float]]:
        """
        Estimate head pose (yaw, pitch, roll) via solvePnP.

        Returns:
            Tuple of (yaw, pitch, roll) in degrees, or None on failure.
        """
        if self._camera_matrix is None:
            return None

        # Gather 2D image points from face mesh
        image_points = []
        for idx in _POSE_LANDMARK_IDS:
            lm = face_landmarks.landmark[idx]
            # MediaPipe returns normalised coords; convert to pixel coords
            # using the focal length stored in the camera matrix as a proxy
            # for image width/height.
            w = self._camera_matrix[0, 2] * 2  # image width
            h = self._camera_matrix[1, 2] * 2  # image height
            image_points.append([lm.x * w, lm.y * h])
        image_points = np.array(image_points, dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_3D, image_points,
            self._camera_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        # Decompose rotation matrix into Euler angles
        # Decompose rotation matrix into Euler angles (X->Y->Z: Pitch-Yaw-Roll)
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[0, 1] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.atan2(rmat[1, 2], rmat[2, 2])
            yaw = math.atan2(-rmat[0, 2], sy)
            roll = math.atan2(rmat[0, 1], rmat[0, 0])
        else:
            # Singular case (Yaw = +/- 90 degrees)
            pitch = math.atan2(rmat[1, 0], rmat[1, 1])
            yaw = math.atan2(-rmat[0, 2], sy)
            roll = 0.0

        return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))

    # ------------------------------------------------------------------
    # Gaze Vector Estimation
    # ------------------------------------------------------------------
    def _estimate_gaze(self, face_landmarks) -> Optional[float]:
        """
        Estimate gaze alignment angle (degrees) from iris position + head pose.

        Returns:
            Dict with keys 'gaze_angle', 'gaze_yaw', 'gaze_pitch',
            or None if unavailable.
        """
        n_lm = len(face_landmarks.landmark)
        if n_lm < 474:
            return None

        # --- Iris offset within eye (horizontal + vertical) ---
        def _iris_offset(iris_idx, eye_indices):
            """Return (h_offset, v_offset) as fraction: 0 = centre, ±1 = edge."""
            iris = face_landmarks.landmark[iris_idx]
            outer = face_landmarks.landmark[eye_indices[0]]
            inner = face_landmarks.landmark[eye_indices[3]]
            top1 = face_landmarks.landmark[eye_indices[1]]
            bot2 = face_landmarks.landmark[eye_indices[5]]

            eye_w = math.sqrt((outer.x - inner.x) ** 2 + (outer.y - inner.y) ** 2)
            eye_h = math.sqrt((top1.x - bot2.x) ** 2 + (top1.y - bot2.y) ** 2)
            cx = (outer.x + inner.x) / 2
            cy = (top1.y + bot2.y) / 2

            h_off = (iris.x - cx) / (eye_w + 1e-8)
            v_off = (iris.y - cy) / (eye_h + 1e-8)
            return h_off, v_off

        lh, lv = _iris_offset(_LEFT_IRIS_CENTER, _LEFT_EYE_EAR)
        rh, rv = _iris_offset(_RIGHT_IRIS_CENTER, _RIGHT_EYE_EAR)

        # Average offsets (both eyes for robustness)
        h_offset = (lh + rh) / 2.0
        v_offset = (lv + rv) / 2.0

        # Convert iris offset to approximate gaze deviation in degrees.
        # Empirical mapping: full offset ≈ 30° deviation.
        iris_yaw_deg = h_offset * 30.0
        iris_pitch_deg = v_offset * 30.0

        # Combine with head pose if available
        head_pose = self._estimate_head_pose(face_landmarks)
        if head_pose is not None:
            head_yaw, head_pitch, _ = head_pose
            total_yaw = head_yaw + iris_yaw_deg
            total_pitch = head_pitch + iris_pitch_deg
        else:
            total_yaw = iris_yaw_deg
            total_pitch = iris_pitch_deg

        # Convert to radians for vector calculation
        yaw_rad = math.radians(total_yaw)
        pitch_rad = math.radians(total_pitch)

        # Gaze vector from Euler angles (assuming standard PnP coordinate system)
        # Yaw=0, Pitch=0 -> (0, 0, 1) [looking away]
        # Yaw=180, Pitch=0 -> (0, 0, -1) [looking at camera]
        # v = [sin(yaw)cos(pitch), -sin(pitch), cos(yaw)cos(pitch)]
        gaze_vector = np.array([
            math.sin(yaw_rad) * math.cos(pitch_rad),
            -math.sin(pitch_rad),
            math.cos(yaw_rad) * math.cos(pitch_rad)
        ])

        # Camera forward vector (looking from camera to subject is Z, so subject looking at camera is -Z)
        # We want the angle between Gaze and "Vector TO Camera"
        # Since camera is at origin and subject at +Z, vector TO camera is (0,0,-1)
        camera_vector = np.array([0.0, 0.0, -1.0])

        # Dot product
        dot_product = np.dot(gaze_vector, camera_vector)
        # Clamp for safety
        dot_product = max(-1.0, min(1.0, dot_product))
        
        # Angle in degrees
        gaze_angle = math.degrees(math.acos(dot_product))
        # Helper function to normalize angle to [-180, 180]
        def normalize_angle(angle):
            return (angle + 180) % 360 - 180

        # Normalized yaw relative to camera
        # Camera is at -Z (effectively at 180 deg yaw in our coords)
        # So deviation from camera is (total_yaw - 180)
        # 180 -> 0 (Camera)
        # 270 (-90) -> +90 (Left/Right?)
        # 90 -> -90
        
        rel_yaw = normalize_angle(total_yaw - 180)
        rel_pitch = normalize_angle(total_pitch)
        
        return {
            'gaze_angle': gaze_angle,
            'gaze_yaw': rel_yaw,     # 0=Camera, +ve=Left, -ve=Right (approx)
            'gaze_pitch': rel_pitch  # 0=Level, +ve=Down, -ve=Up
        }

    # ------------------------------------------------------------------
    # Pose Features (unchanged core, preserved for backward compat)
    # ------------------------------------------------------------------
    def extract_pose_features(self, pose_landmarks) -> Dict[str, float]:
        """
        Extract camera-invariant pose features using 3-tier landmark strategy.
        
        FRAME ACCEPTANCE POLICY:
        - VALID: Face detected AND (both shoulders OR head landmark)
        - INVALID: Face NOT detected OR (shoulders + head BOTH missing)
        - Missing hips MUST NEVER invalidate a frame
        
        Returns:
            Dict with normalized geometric features
        """
        features = {}
        
        # Extract key landmarks safely - use correct access
        left_shoulder = self._get_landmark_safe(pose_landmarks, mp_holistic.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = self._get_landmark_safe(pose_landmarks, mp_holistic.PoseLandmark.RIGHT_SHOULDER)
        left_hip = self._get_landmark_safe(pose_landmarks, mp_holistic.PoseLandmark.LEFT_HIP)
        right_hip = self._get_landmark_safe(pose_landmarks, mp_holistic.PoseLandmark.RIGHT_HIP)
        left_ear = self._get_landmark_safe(pose_landmarks, mp_holistic.PoseLandmark.LEFT_EAR)
        right_ear = self._get_landmark_safe(pose_landmarks, mp_holistic.PoseLandmark.RIGHT_EAR)
        nose = self._get_landmark_safe(pose_landmarks, mp_holistic.PoseLandmark.NOSE)
        
        # FRAME VALIDATION LOGIC
        face_detected = nose is not None  # Face proxy via nose landmark
        
        # Check if we have core upper body landmarks
        shoulders_detected = (left_shoulder is not None) or (right_shoulder is not None)
        head_detected = nose is not None
        
        # FRAME ACCEPTANCE: Face OR shoulders (relaxed to allow more frames through)
        if not (face_detected or shoulders_detected):
            return {'error': 'no_face_or_upper_body_landmarks'}
        
        # TIER 1: PRIMARY - Shoulders + Head (sufficient for posture)
        if shoulders_detected and head_detected:
            # 1. Shoulder Width Normalization
            if left_shoulder and right_shoulder:
                shoulder_width = self._calculate_distance_3d(left_shoulder, right_shoulder)
            else:
                shoulder_width = 0.2  # Default estimate
                
            if left_hip and right_hip:
                hip_width = self._calculate_distance_3d(left_hip, right_hip)
            else:
                hip_width = 0.15  # Default estimate
            
            # Body width ratio (camera invariant)
            features['shoulder_hip_ratio'] = shoulder_width / (hip_width + 1e-8)
            
            # 2. Shoulder Tilt Angle (2D Image Plane - Refinement v2.2)
            if left_shoulder and right_shoulder:
                dx = right_shoulder['x'] - left_shoulder['x']
                dy = right_shoulder['y'] - left_shoulder['y']
                # Angle relative to horizontal
                raw_angle = math.degrees(math.atan2(dy, dx))
                
                # Normalize angle to 0-90 range (User Fix v2.4)
                # Fixes 170 deg issue (wrapping)
                norm_angle = abs(raw_angle)
                if norm_angle > 90.0:
                    norm_angle = 180.0 - norm_angle
                
                features['shoulder_tilt_angle'] = norm_angle
            else:
                features['shoulder_tilt_angle'] = 0.0  # Neutral
            
            # 3. Head Position Normalization
            if left_shoulder and right_shoulder and nose:
                shoulder_center = {
                    'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
                    'y': (left_shoulder['y'] + right_shoulder['y']) / 2,
                    'z': (left_shoulder['z'] + right_shoulder['z']) / 2
                }
                head_height = self._calculate_distance_3d(nose, shoulder_center)
                features['head_height_ratio'] = head_height / (shoulder_width + 1e-8)
            
            # 4. Eye Engagement (if ears visible)
            if left_ear and right_ear and left_shoulder and right_shoulder:
                eye_distance = self._calculate_distance_3d(left_ear, right_ear)
                features['eye_distance_ratio'] = eye_distance / (shoulder_width + 1e-8)
            
            # 5. Hip-based refinement (TIER 2 - optional)
            if left_hip and right_hip:
                hip_width = self._calculate_distance_3d(left_hip, right_hip)
                features['shoulder_hip_ratio'] = shoulder_width / (hip_width + 1e-8)
                
                shoulder_angle = self._calculate_angle_3d(left_shoulder, right_shoulder, {'x': right_shoulder['x'], 'y': right_shoulder['y'] + 0.1, 'z': right_shoulder['z']})
                hip_angle = self._calculate_angle_3d(left_hip, right_hip, {'x': right_hip['x'], 'y': right_hip['y'] + 0.1, 'z': right_hip['z']})
                features['upper_body_alignment'] = abs(shoulder_angle - hip_angle)
            else:
                features['shoulder_hip_ratio'] = 1.0  # Neutral
                features['upper_body_alignment'] = 0.0  # Neutral
            
            # 6. Facial Symmetry (if available)
            if left_ear and right_ear and nose:
                ear_center = {
                    'x': (left_ear['x'] + right_ear['x']) / 2,
                    'y': (left_ear['y'] + right_ear['y']) / 2
                }
                nose_offset = self._calculate_distance_2d(nose, ear_center)
                if 'eye_distance_ratio' in features:
                    features['facial_symmetry_offset'] = nose_offset / (features['eye_distance_ratio'] + 1e-8)
                    
            # 7. Torso Inclination & Keypoints (Redesign v2.1)
            # Needed for biomechanical uprightness and movement scoring
            if left_shoulder and right_shoulder:
                mid_shoulder = {
                    'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
                    'y': (left_shoulder['y'] + right_shoulder['y']) / 2,
                    'z': (left_shoulder['z'] + right_shoulder['z']) / 2
                }
                
                # Export raw midpoint and width for velocity calc
                features['shoulder_midpoint_x'] = mid_shoulder['x']
                features['shoulder_midpoint_y'] = mid_shoulder['y']
                features['shoulder_width_raw'] = shoulder_width
                
                if left_hip and right_hip:
                    mid_hip = {
                        'x': (left_hip['x'] + right_hip['x']) / 2,
                        'y': (left_hip['y'] + right_hip['y']) / 2,
                        'z': (left_hip['z'] + right_hip['z']) / 2
                    }
                    
                    # 2D Angle Calculation - Image Plane Only (Refinement v2.2)
                    # User requested: arctan2(abs(dx), abs(dy)) to measure lean relative to vertical
                    dx = mid_shoulder['x'] - mid_hip['x']
                    dy = mid_shoulder['y'] - mid_hip['y']
                    
                    # Compute angle from vertical axis
                    # If perfectly upright, dx=0, dy large. atan2(0, dy) = 0.
                    # If leaning, dx increases.
                    # We use abs() to measure magnitude of lean regardless of direction
                    angle_rad = math.atan2(abs(dx), abs(dy))
                    features['torso_inclination_deg'] = math.degrees(angle_rad)
                else:
                    # Fallback: assume upright if hips missing but shoulders present
                    features['torso_inclination_deg'] = 0.0
        
        # TIER 2: FALLBACK - Head only (minimal but usable)
        elif head_detected:
            features['shoulder_tilt_angle'] = 0.0  # Neutral (unknown)
            features['head_height_ratio'] = 0.5   # Neutral estimate
            features['shoulder_hip_ratio'] = 1.0  # Neutral estimate
            features['head_only_mode'] = True
        
        # TIER 3: EMERGENCY - Shoulders only
        elif shoulders_detected:
            if left_shoulder and right_shoulder:
                shoulder_vector = {
                    'x': right_shoulder['x'] - left_shoulder['x'],
                    'y': right_shoulder['y'] - left_shoulder['y'],
                    'z': right_shoulder['z'] - left_shoulder['z']
                }
                horizontal_magnitude = math.sqrt(shoulder_vector['x']**2 + shoulder_vector['z']**2)
                shoulder_tilt_angle = math.degrees(math.atan2(abs(shoulder_vector['y']), horizontal_magnitude + 1e-8))
                features['shoulder_tilt_angle'] = shoulder_tilt_angle
                
                features['head_height_ratio'] = 0.5   # Neutral estimate
                features['shoulder_hip_ratio'] = 1.0  # Neutral estimate
                features['shoulders_only_mode'] = True
            
        return features

    # ------------------------------------------------------------------
    # Face Features (v2.0 — adaptive)
    # ------------------------------------------------------------------
    def extract_face_features(self, face_landmarks) -> Dict[str, float]:
        """
        Extract camera-invariant facial features.
        
        v2.0 additions:
        - 6-point EAR (robust blink detection)
        - Head pose (yaw, pitch, roll)
        - Gaze alignment angle
        - Eyebrow raise ratio
        - Jaw opening ratio
        
        Returns:
            Dict with normalized facial geometric features
        """
        features = {}
        
        if not face_landmarks or len(face_landmarks.landmark) < 468:
            return {'error': 'insufficient_face_landmarks'}
        
        # ---- 1. 6-point EAR ----
        left_ear_val = self._ear_6point(face_landmarks, _LEFT_EYE_EAR)
        right_ear_val = self._ear_6point(face_landmarks, _RIGHT_EYE_EAR)
        features['left_eye_opening_ratio'] = left_ear_val
        features['right_eye_opening_ratio'] = right_ear_val

        # ---- 2. Head Pose ----
        head_pose = self._estimate_head_pose(face_landmarks)
        if head_pose is not None:
            features['head_yaw'] = head_pose[0]
            features['head_pitch'] = head_pose[1]
            features['head_roll'] = head_pose[2]

        # ---- 3. Gaze Alignment and Direction ----
        gaze_data = self._estimate_gaze(face_landmarks)
        if gaze_data is not None:
            features['gaze_alignment_angle'] = gaze_data['gaze_angle']
            features['gaze_yaw'] = gaze_data['gaze_yaw']
            features['gaze_pitch'] = gaze_data['gaze_pitch']

        # ---- Extract key facial landmarks for remaining features ----
        left_eyebrow = {'x': face_landmarks.landmark[70].x, 'y': face_landmarks.landmark[70].y, 'z': face_landmarks.landmark[70].z}
        right_eyebrow = {'x': face_landmarks.landmark[300].x, 'y': face_landmarks.landmark[300].y, 'z': face_landmarks.landmark[300].z}

        nose_tip = {'x': face_landmarks.landmark[1].x, 'y': face_landmarks.landmark[1].y, 'z': face_landmarks.landmark[1].z}
        upper_lip = {'x': face_landmarks.landmark[13].x, 'y': face_landmarks.landmark[13].y, 'z': face_landmarks.landmark[13].z}
        lower_lip = {'x': face_landmarks.landmark[14].x, 'y': face_landmarks.landmark[14].y, 'z': face_landmarks.landmark[14].z}

        mouth_left = {'x': face_landmarks.landmark[61].x, 'y': face_landmarks.landmark[61].y, 'z': face_landmarks.landmark[61].z}
        mouth_right = {'x': face_landmarks.landmark[291].x, 'y': face_landmarks.landmark[291].y, 'z': face_landmarks.landmark[291].z}

        # Chin (landmark 199)
        chin = {'x': face_landmarks.landmark[199].x, 'y': face_landmarks.landmark[199].y, 'z': face_landmarks.landmark[199].z}

        # Left/right eye centres (for eyebrow raise)
        left_eye_center = {
            'x': (face_landmarks.landmark[33].x + face_landmarks.landmark[133].x) / 2,
            'y': (face_landmarks.landmark[33].y + face_landmarks.landmark[133].y) / 2,
        }
        right_eye_center = {
            'x': (face_landmarks.landmark[362].x + face_landmarks.landmark[263].x) / 2,
            'y': (face_landmarks.landmark[362].y + face_landmarks.landmark[263].y) / 2,
        }

        # Face height as reference (eyebrow centre to lip centre)
        face_height = self._calculate_distance_2d(
            {'x': (left_eyebrow['x'] + right_eyebrow['x']) / 2, 'y': (left_eyebrow['y'] + right_eyebrow['y']) / 2},
            {'x': (upper_lip['x'] + lower_lip['x']) / 2, 'y': (upper_lip['y'] + lower_lip['y']) / 2}
        )

        # ---- 4. Mouth Opening Ratio ----
        mouth_opening = self._calculate_distance_2d(upper_lip, lower_lip)
        if face_height > 0:
            features['mouth_opening_ratio'] = mouth_opening / face_height
        else:
            features['mouth_opening_ratio'] = 0.0

        # ---- 5. Smile Intensity (Mouth Corner Lift) ----
        if mouth_left and mouth_right and upper_lip and face_height > 0:
            lip_center_y = (upper_lip['y'] + lower_lip['y']) / 2
            corner_avg_y = (mouth_left['y'] + mouth_right['y']) / 2
            smile_lift = (lip_center_y - corner_avg_y) / face_height
            features['smile_intensity'] = max(0.0, smile_lift * 5.0)
        else:
            features['smile_intensity'] = 0.0

        # ---- 6. Eyebrow Raise Ratio (NEW) ----
        # Vertical distance from eyebrow to eye centre, normalised by face height.
        if face_height > 0:
            left_brow_raise = abs(left_eyebrow['y'] - left_eye_center['y']) / face_height
            right_brow_raise = abs(right_eyebrow['y'] - right_eye_center['y']) / face_height
            features['eyebrow_raise_ratio'] = (left_brow_raise + right_brow_raise) / 2.0
        else:
            features['eyebrow_raise_ratio'] = 0.0

        # ---- 7. Jaw Opening Ratio (NEW) ----
        # Distance chin to nose normalised by face height.
        if face_height > 0:
            jaw_dist = self._calculate_distance_2d(chin, nose_tip)
            features['jaw_opening_ratio'] = jaw_dist / face_height
        else:
            features['jaw_opening_ratio'] = 0.0
        
        return features

    def get_feature_confidence(self, features: Dict) -> float:
        """
        Calculate confidence score for extracted features.
        
        Returns:
            Float between 0.0 and 1.0
        """
        if 'error' in features:
            return 0.0
        
        valid_features = sum(1 for v in features.values() if isinstance(v, (int, float)) and v > 0)
        total_features = len([k for k, v in features.items() if isinstance(v, (int, float))])
        
        if total_features == 0:
            return 0.0
        
        return min(1.0, valid_features / total_features)
