import cv2
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)

class HeadPoseMetrics:
    def __init__(self):
        # 3D model points (generic human face)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # Internal storage for stateful usage
        self.pose_history = []

    def add_frame(self, pose_data, timestamp: float = 0.0):
        """
        Add a single frame's pose data to history.
        Handles both dict {'pitch', 'yaw', 'roll'} and tuple (yaw, pitch, roll)
        timestamp: Unused in this version but required for interface standardization.
        """
        if not pose_data:
            return
            
        if isinstance(pose_data, dict):
            self.pose_history.append(pose_data)
        elif isinstance(pose_data, (list, tuple)) and len(pose_data) == 3:
            # NormalizedGeometry returns (yaw, pitch, roll)
            self.pose_history.append({
                'yaw': pose_data[0],
                'pitch': pose_data[1],
                'roll': pose_data[2]
            })

    def compute_head_pose(self, face_landmarks, image_shape):
        """
        Computes 3D head pose (yaw, pitch, roll) using solvePnP.
        landmarks: mediapipe normalized landmarks
        image_shape: (height, width)
        """
        if not face_landmarks:
            return None
            
        h, w = image_shape
        
        # Extract 2D image points
        # MediaPipe Indices:
        # Nose tip: 1
        # Chin: 152
        # Left Eye Left Corner: 33 (subject's right)
        # Right Eye Right Corner: 263 (subject's left)
        # Left Mouth Corner: 61
        # Right Mouth Corner: 291
        
        image_points = np.array([
            (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),     # Nose tip
            (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h), # Chin
            (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),   # Left eye outer
            (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h), # Right eye outer
            (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),   # Left Mouth
            (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h)  # Right mouth
        ], dtype=np.float64)
        
        # Camera internals
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, 
            image_points, 
            camera_matrix, 
            dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
            
        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[0, 1] * rmat[0, 1])
        
        singular = sy < 1e-6
        if not singular:
            pitch = math.atan2(rmat[1, 2], rmat[2, 2])
            yaw = math.atan2(-rmat[0, 2], sy)
            roll = math.atan2(rmat[0, 1], rmat[0, 0])
        else:
            pitch = math.atan2(rmat[1, 0], rmat[1, 1])
            yaw = math.atan2(-rmat[0, 2], sy)
            roll = 0.0
            
        return {
            'pitch': float(math.degrees(pitch)),
            'yaw': float(math.degrees(yaw)),
            'roll': float(math.degrees(roll))
        }

    def _smooth_signal(self, signal, window_size=5):
        """Simple rolling mean smoothing."""
        try:
            signal = np.array(signal, dtype=np.float64).flatten()
            if len(signal) < window_size:
                return signal
            return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
        except Exception as e:
            logger.error(f"Smoothing error: {e}")
            return np.array(signal)

    def finalize(self, pose_history=None):
        """
        Computes variability and gesture frequency.
        If pose_history is None, uses self.pose_history.
        """
        if pose_history is None:
            pose_history = self.pose_history
            
        if not pose_history:
            return None
            
        # Extract raw signals
        # Sanitize data: Ensure float scalars
        pitches = []
        yaws = []
        rolls = []
        
        for p in pose_history:
            try:
                # Ensure values are float scalars
                pitch = float(p.get('pitch', 0.0))
                yaw = float(p.get('yaw', 0.0))
                roll = float(p.get('roll', 0.0))
                
                pitches.append(pitch)
                yaws.append(yaw)
                rolls.append(roll)
            except (ValueError, TypeError):
                continue
                
        # Coerce to 1D float arrays
        pitches = np.asarray(pitches, dtype=float).reshape(-1)
        yaws = np.asarray(yaws, dtype=float).reshape(-1)
        rolls = np.asarray(rolls, dtype=float).reshape(-1)
        
        # Smooth signals (Task 1)
        # Using valid mode reduces length, but that's fine for stats
        
        # Unwrap angles to handle boundary crossings (e.g. 179 -> -179)
        # Convert to radians -> unwrap -> convert back to degrees
        pitches_rad = np.deg2rad(pitches)
        yaws_rad = np.deg2rad(yaws)
        rolls_rad = np.deg2rad(rolls)
        
        pitches_unwrapped = np.rad2deg(np.unwrap(pitches_rad))
        yaws_unwrapped = np.rad2deg(np.unwrap(yaws_rad))
        rolls_unwrapped = np.rad2deg(np.unwrap(rolls_rad))
        
        smoothed_pitch = self._smooth_signal(pitches_unwrapped, window_size=5)
        smoothed_yaw = self._smooth_signal(yaws_unwrapped, window_size=5)
        smoothed_roll = self._smooth_signal(rolls_unwrapped, window_size=5)
        
        # Use smoothed for std dev (variability)
        metrics = {
            'pitch_std': float(np.std(smoothed_pitch)) if len(smoothed_pitch) > 0 else 0.0,
            'yaw_std': float(np.std(smoothed_yaw)) if len(smoothed_yaw) > 0 else 0.0,
            'roll_std': float(np.std(smoothed_roll)) if len(smoothed_roll) > 0 else 0.0,
            'smoothed_signal_used': True
        }
        
        # Robust Gesture Detection
        # Thresholds
        MIN_AMPLITUDE = 5.0 # degrees
        MIN_DURATION = 5 # frames at ~30fps -> ~150ms
        
        def count_gestures(signal, center_val=0.0):
            if len(signal) < MIN_DURATION: return 0
            
            # Center signal
            centered = signal - np.mean(signal)
            
            # Find peaks/valleys
            # Simple approach: Zero crossings with amplitude check
            # Better: State machine. 
            # State: Neutral, Positive, Negative.
            
            count = 0
            in_gesture = False
            last_sign = 0
            gesture_frames = 0
            max_excursion = 0.0
            
            crossings = 0
            # Iterate
            for val in centered:
                if abs(val) > MIN_AMPLITUDE:
                    curr_sign = np.sign(val)
                    if not in_gesture:
                        in_gesture = True
                        last_sign = curr_sign
                        gesture_frames = 1
                        max_excursion = abs(val)
                    else:
                        if curr_sign != last_sign:
                            # Crossing within gesture
                            crossings += 1
                            last_sign = curr_sign
                        gesture_frames += 1
                        max_excursion = max(max_excursion, abs(val))
                else:
                    # Below threshold
                    if in_gesture:
                        # End of gesture candidate
                        # Check duration and complexity (needs at least one crossing for full nod?)
                        # Or just persistent excursion?
                        # A nod is Down-Up-Neutral. A shake is Left-Right-Neutral.
                        # So at least 2 crossings (start->max, max->min, min->end?)
                        # Let's count significant direction changes?
                        pass
                        in_gesture = False
                        gesture_frames = 0
            
            # Fallback: Zero crossing rate on strong signal segments
            # Mask low amplitude
            strong_mask = np.abs(centered) > MIN_AMPLITUDE
            if np.sum(strong_mask) < MIN_DURATION:
                return 0
                
            # Count zero crossings on masked signal? No, discontinuous.
            # Let's use simple peak detection on smoothed signal.
            # Local maxima > Threshold + Local minima < -Threshold
            # Find peaks
            peaks = (np.diff(np.sign(np.diff(centered))) < 0).nonzero()[0] + 1 # Indices of local max
            valleys = (np.diff(np.sign(np.diff(centered))) > 0).nonzero()[0] + 1 # Indices of local min
            
            # Filter by amplitude
            valid_peaks = [p for p in peaks if centered[p] > MIN_AMPLITUDE]
            valid_valleys = [v for v in valleys if centered[v] < -MIN_AMPLITUDE]
            
            # Each pair of peak+valley is roughly one cycle (nod/shake)
            cycles = min(len(valid_peaks), len(valid_valleys))
            return cycles

        duration_sec = len(pose_history) / 30.0 # Approx
        if duration_sec == 0: duration_sec = 1
        
        nod_count = count_gestures(smoothed_pitch)
        shake_count = count_gestures(smoothed_yaw)
        
        # Cap Frequency (Biologically Impossible > 0.5 per sec usually for sustained, but bursts ok)
        # User said: <= 5 per 10 seconds (0.5 Hz avg)
        # If count > 0.5 * duration, cap it?
        # Or just report raw and let interpretation handle? 
        # User said "Cap biologically impossible frequencies".
        
        max_freq = 0.5 # Hz
        max_allowed = max(1.0, duration_sec * max_freq)
        
        metrics['nod_count'] = min(nod_count, int(max_allowed * 1.5)) # Allow burst
        metrics['shake_count'] = min(shake_count, int(max_allowed * 1.5))
        
        metrics['nod_frequency'] = round(metrics['nod_count'] / duration_sec * 60, 1) # per min
        metrics['shake_frequency'] = round(metrics['shake_count'] / duration_sec * 60, 1) # per min
        
        return metrics
