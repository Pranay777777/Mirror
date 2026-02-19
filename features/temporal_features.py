"""
Temporal feature extraction for behavioral analysis.

Transforms frame-level features into time-aware behavioral metrics.
Captures dynamics, patterns, and stability over sliding windows.

v3.0 — Lean Binary Architecture:
- Z-score adaptive blink detection (no fixed EAR thresholds)
- Lean binary eye contact (gaze < 12° threshold, no Gaussian)
- Adaptive expression modeling with per-video normalisation
- Full-video accumulation with smoothing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import deque
import math
try:
    import traycer
    if not hasattr(traycer, 'trace'):
        raise AttributeError("traycer.trace not available")
except (ImportError, AttributeError):
    import types
    traycer = types.ModuleType('traycer')
    traycer.trace = lambda f: f


class TemporalFeatures:
    """Extract temporal behavioral features from geometric measurements.
    
    v2.0 — All parameters are configurable; no hardcoded magic constants.
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        # Blink detection
        blink_z_threshold: float = -2.0,
        blink_min_duration_ms: float = 100.0,
        blink_max_duration_ms: float = 300.0,
        blink_min_consecutive: int = 2,
        # Expression
        expression_evidence_min: float = 0.3,
        # Smoothing
        smoothing_window: int = 5,
        # Start noise filtering
        ignore_start_ms: float = 3000.0,
    ):
        """
        Initialize temporal feature extractor.
        
        All thresholds are configurable — no hardcoded constants.
        """
        self.fps = fps

        # Blink parameters
        self.blink_z_threshold = blink_z_threshold
        self.blink_min_duration_ms = blink_min_duration_ms
        self.blink_max_duration_ms = blink_max_duration_ms
        self.blink_min_consecutive = blink_min_consecutive

        # Expression parameters
        self.expression_evidence_min = expression_evidence_min

        # Smoothing
        self.smoothing_window = smoothing_window
        self.ignore_start_ms = ignore_start_ms

        # Feature history buffers — NO maxlen; accumulate entire video
        self.pose_history: List[Dict] = []
        self.face_history: List[Dict] = []
        self.timestamps: List[float] = []
        
        # Velocity calculation
        self.prev_pose = None
        self.prev_face = None
        self.prev_timestamp = None

    # ------------------------------------------------------------------
    # Frame ingestion
    # ------------------------------------------------------------------
    def add_frame(self, pose_features: Dict, face_features: Dict, timestamp: float):
        """Add frame features to temporal buffers.
        
        Args:
            pose_features: Dictionary of pose features for this frame
            face_features: Dictionary of face features for this frame  
            timestamp: Frame timestamp in seconds
        """
        # Calculate velocities BEFORE appending
        if pose_features is not None and 'error' not in pose_features:
            if self.prev_pose and self.prev_timestamp:
                dt = timestamp - self.prev_timestamp
                if dt > 0:
                    pose_velocity = self._calculate_feature_velocity(pose_features, self.prev_pose, dt)
                    pose_features.update({f'velocity_{k}': v for k, v in pose_velocity.items()})

            self.pose_history.append(pose_features)
            self.timestamps.append(timestamp)
            
            if face_features is not None and 'error' not in face_features:
                self.face_history.append(face_features)
            else:
                self.face_history.append({})
        
        # Update previous values
        self.prev_pose = pose_features.copy() if pose_features else None
        self.prev_face = face_features.copy() if face_features else None
        self.prev_timestamp = timestamp
    
    def _calculate_feature_velocity(self, current: Dict, previous: Dict, dt: float) -> Dict:
        """Calculate velocity for numeric features."""
        velocity = {}
        for key in current:
            if key in previous and isinstance(current[key], (int, float)) and isinstance(previous[key], (int, float)):
                if 'error' not in key:
                    velocity[key] = abs(current[key] - previous[key]) / dt
        return velocity

    # ------------------------------------------------------------------
    # Helper: probabilistic result format
    # ------------------------------------------------------------------
    @staticmethod
    def _format_result(value, confidence: float, evidence_ratio: float,
                       method: str, reason: str = "calculated") -> Dict:
        """Return a metric in the probabilistic output format."""
        return {
            "value": value,
            "confidence": confidence,
            "evidence_ratio": evidence_ratio,
            "method": method,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Smoothing utility
    # ------------------------------------------------------------------
    def _smooth(self, series: pd.Series) -> pd.Series:
        """Apply moving-average smoothing."""
        if len(series) < self.smoothing_window:
            return series
        return series.rolling(window=self.smoothing_window, center=True, min_periods=1).mean()

    # ------------------------------------------------------------------
    # NEW Helper: Signal Confidence
    # ------------------------------------------------------------------
    def _calculate_signal_confidence(self, series: pd.Series, threshold: float = None, expected_sigma: float = 0.5) -> float:
        """
        Calculate confidence based on normalized exponential decay model (linear decay approximation).
        
        Formula:
        sigma_norm = min(sigma / expected_sigma, 1.0)
        stability = 1.0 - sigma_norm
        
        signal_conf = (0.6 * stability) + (0.4 * margin_score)
        """
        if len(series) < 2:
            return 0.0
            
        # 1. Stability (Normalized Linear Decay)
        sigma = series.std()
        sigma_norm = min(sigma / expected_sigma, 1.0)
        stability = 1.0 - sigma_norm
        
        # 2. Margin from threshold (if applicable)
        margin_score = 1.0
        if threshold is not None:
            mean_val = series.mean()
            if threshold > 0:
                dist = abs(mean_val - threshold)
                margin = dist / threshold
                margin_score = min(1.0, margin)
        
        # Combined score
        final_conf = (stability * 0.6) + (margin_score * 0.4)
        return float(np.clip(final_conf, 0.0, 1.0))


    # ==================================================================
    # PHASE 2 — Adaptive Z-Score Blink Detection
    # ==================================================================
    def _detect_blinks_zscore(self, ear_series: pd.Series) -> Tuple[int, List[dict], float]:
        """
        Detect blinks using robust adaptive z-score method (v2.1).
        
        Improvements:
        - Two-stage recovery: Full (Z > -1.0) OR Stable Squint (Z > -2.0 & slope ~ 0)
        - Prevents false negatives from slow reopening
        - Filters noise with deep amplitude check
        """
        clean = ear_series.dropna()
        if len(clean) < 5:
            return 0, [], 0.0

        mean_ear = clean.mean()
        std_ear = clean.std()
        if std_ear < 1e-8:
            return 0, [], 0.0

        z_scores = (clean - mean_ear) / std_ear

        # Parameters
        Z_START = self.blink_z_threshold       # -2.0 (Trigger closure)
        Z_RECOVER_FULL = -1.0                  # Clear open state
        Z_RECOVER_PARTIAL = -2.0               # Minimal recovery threshold
        MIN_AMPLITUDE_Z = -2.5                 # Noise suppression (must dip at least this deep)
        STABILITY_WINDOW = max(1, int(0.05 * self.fps)) # ~50ms to check for plateau

        blink_events = []
        state = "OPEN"
        start_idx = 0
        min_z_in_blink = 0
        
        # Convert to numpy for faster, cleaner indexing if needed, but pandas is fine
        # Iterate through z-scores
        for i in range(len(z_scores)):
            z = z_scores.iloc[i]
            idx = z_scores.index[i] # Original index if needed
            
            if state == "OPEN":
                if z < Z_START:
                    state = "BLINKING"
                    start_idx = i
                    min_z_in_blink = z
                    
            elif state == "BLINKING":
                min_z_in_blink = min(min_z_in_blink, z)
                duration_ms = ((i - start_idx) / self.fps) * 1000.0
                
                # --- Robust Recovery Logic ---
                
                # Condition 1: Full Statistical Recovery (Original strict check)
                is_full_open = (z > Z_RECOVER_FULL)
                
                # Condition 2: Plateaued at safe level (Gradual reopening fix)
                # If Z is above trigger (-2.0) but stuck (rate of change ~ 0)
                is_stable_squint = False
                if z > Z_RECOVER_PARTIAL and i > (start_idx + STABILITY_WINDOW):
                    # Check recent slope (opening velocity) over stability window
                    # We look back STABILITY_WINDOW frames
                    window_start = i - STABILITY_WINDOW
                    if window_start >= 0:
                        recent_z = z_scores.iloc[window_start : i + 1] # +1 to include current
                        if len(recent_z) > 1:
                            # Simple slope: (end - start) / frames
                            slope = (recent_z.iloc[-1] - recent_z.iloc[0])
                            # If slope is very small (stopped opening) or negative (closing again?), 
                            # we consider it "settled" if it's strictly positive but small? 
                            # Actually, if it's < 0.1, it means it's not changing much.
                            # We want to catch "stopped opening".
                            if abs(slope) < 0.2: # Relaxed slope check
                                is_stable_squint = True
                
                if is_full_open or is_stable_squint:
                    # Validate event
                    # 1. Was it a real blink? (Deep enough?)
                    # 2. Was it long enough (but not too long?)
                    if min_z_in_blink < MIN_AMPLITUDE_Z: 
                         # Using MIN_AMPLITUDE ensures we don't count noise 
                         # hovering between -2.0 and -2.2 as a blink.
                         if self.blink_min_duration_ms <= duration_ms <= self.blink_max_duration_ms:
                             blink_events.append({
                                 "start_idx": start_idx,
                                 "end_idx": i, # inclusive
                                 "duration_ms": duration_ms
                             })
                    
                    state = "OPEN"
                    
                # Safety Timeout
                if duration_ms > (self.blink_max_duration_ms * 1.5): # Allow some buffer before force-reset
                    state = "OPEN" # Reset without saving
        
        # Handle end of video while blinking -> discard
        
        blink_count = len(blink_events)

        # Confidence: data coverage + signal stability
        coverage = len(clean) / max(len(ear_series), 1)
        # Expected sigma for blinks = 0.2 (User defined)
        signal_conf = self._calculate_signal_confidence(clean, expected_sigma=0.2)
        confidence = coverage * 0.5 + signal_conf * 0.5
        confidence = min(1.0, max(0.0, confidence))

        return blink_count, blink_events, round(float(confidence), 3)

    # ==================================================================
    # PHASE X - Global Camera Offset Correction
    # ==================================================================
    def _estimate_camera_offset(self, gaze_angles: pd.Series, gaze_yaws: pd.Series, total_frames: int) -> float:
        """
        Estimate the camera-screen angular bias.
        Criteria:
        - Head yaw near 0 (looking at screen)
        - Early part of video (first 30% or 300 frames)
        - Low motion (implied by stable yaw)
        """
        if gaze_angles.empty or gaze_yaws is None:
             return 0.0
             
        # 1. Define "Early Part"
        limit = min(300, int(total_frames * 0.3))
        if limit < 10: limit = total_frames # Fallback for short videos
        
        early_angles = gaze_angles.iloc[:limit]
        early_yaws = gaze_yaws.iloc[:limit]
        
        # 2. Filter for Head Yaw near 0 (±10 degrees)
        # This assumes the user is facing the screen (where the camera usually is)
        valid_mask = (early_yaws.abs() < 10.0)
        
        valid_angles = early_angles[valid_mask]
        
        if len(valid_angles) < 5:
             # Fallback: If strict yaw filter yields no data (e.g. head tracking failure -90),
             # but we have gaze data, try to use median of all early gaze angles.
             # This assumes the user spends most of the early video looking at the screen.
             # We check if we have enough early data at all.
             if len(early_angles) > 10:
                  valid_angles = early_angles
             else:
                  return 0.0
             
        # 3. Compute Median Gaze Angle
        # User defined: "median(gaze_angle)"
        offset = float(valid_angles.median())
        
        # Safety Clamp (Don't correct more than 15 degrees)
        offset = max(0.0, min(15.0, offset))
        
        return offset


    # ==================================================================
    # PHASE 4 — Adaptive Expression Modeling
    # ==================================================================
    def _compute_expression_metrics(self, face_df: pd.DataFrame) -> Dict:
        """
        Compute adaptive expression metrics.

        Returns dict with smile_intensity, expression_variability,
        expression_dynamics sub-metrics.
        """
        results = {}
        evidence_cols = ['smile_intensity', 'mouth_opening_ratio', 'eyebrow_raise_ratio', 'jaw_opening_ratio']
        available = [c for c in evidence_cols if c in face_df.columns]
        total_frames = len(face_df)

        # --- Smile Intensity (z-score normalised per video) ---
        if 'smile_intensity' in face_df.columns:
            smile = face_df['smile_intensity'].dropna()
            evidence = len(smile) / max(total_frames, 1)
            if len(smile) >= 5:
                mean_s = smile.mean()
                std_s = smile.std()
                if std_s > 1e-8:
                    z_smile = (smile - mean_s) / std_s
                    # Fraction of frames above +1 std (above baseline)
                    above_baseline = (z_smile > 1.0).sum() / len(smile)
                    
                    # Confidence: Evidence + Signal Stability
                    # We don't use margin here because "neutral" (low smile) is a valid state
                    # Confidence: Evidence + Signal Stability
                    # We don't use margin here because "neutral" (low smile) is a valid state
                    # Expression default sigma = 0.5 (safe assumption)
                    signal_conf = self._calculate_signal_confidence(smile, expected_sigma=0.5)
                    confidence = evidence * 0.5 + signal_conf * 0.5
                    confidence = evidence * 0.5 + signal_conf * 0.5
                    
                    results['smile_intensity'] = self._format_result(
                        round(float(above_baseline), 4), confidence, evidence,
                        "adaptive_zscore"
                    )
                else:
                    results['smile_intensity'] = self._format_result(
                        0.0, evidence, evidence, "adaptive_zscore", "static_signal"
                    )
            else:
                results['smile_intensity'] = self._format_result(
                    None, 0.0, evidence, "adaptive_zscore", "insufficient_samples"
                )
        else:
            results['smile_intensity'] = self._format_result(
                None, 0.0, 0.0, "adaptive_zscore", "missing_data"
            )

        # --- Expression Variability (std of action units) ---
        if len(available) >= 2:
            sub_df = face_df[available].dropna()
            evidence = len(sub_df) / max(total_frames, 1)
            if len(sub_df) >= 5:
                # Normalise each column to [0,1] range per-video
                normed = (sub_df - sub_df.min()) / (sub_df.max() - sub_df.min() + 1e-8)
                variability = float(normed.std(axis=0).mean())
                
                # Confidence based on data coverage
                # Variability itself is the metric, so we don't punish for low variability
                results['expression_variability'] = self._format_result(
                    round(variability, 4), evidence, evidence,
                    "action_unit_std"
                )
            else:
                results['expression_variability'] = self._format_result(
                    None, 0.0, evidence, "action_unit_std", "insufficient_samples"
                )
        else:
            results['expression_variability'] = self._format_result(
                None, 0.0, 0.0, "action_unit_std", "missing_data"
            )

        # --- Expression Dynamics (rate of change) ---
        if 'mouth_opening_ratio' in face_df.columns:
            mouth = face_df['mouth_opening_ratio'].dropna()
            evidence = len(mouth) / max(total_frames, 1)
            if len(mouth) >= 5:
                data_range = mouth.max() - mouth.min()
                if data_range > 1e-8:
                    total_variation = mouth.diff().abs().sum()
                    dynamics = min(float(total_variation / data_range), 1.0)
                else:
                    dynamics = 0.0
                
                # Confidence: Evidence + Signal Stability of the mouth signal
                # Confidence: Evidence + Signal Stability of the mouth signal
                # Expression default sigma = 0.5
                signal_conf = self._calculate_signal_confidence(mouth, expected_sigma=0.5)
                confidence = evidence * 0.5 + signal_conf * 0.5
                
                results['expression_dynamics'] = self._format_result(
                    round(dynamics, 4), confidence, evidence,
                    "temporal_variation"
                )
            else:
                results['expression_dynamics'] = self._format_result(
                    None, 0.0, evidence, "temporal_variation", "insufficient_samples"
                )
        else:
            results['expression_dynamics'] = self._format_result(
                None, 0.0, 0.0, "temporal_variation", "missing_data"
            )

        return results

    # ==================================================================
    # Main extraction
    # ==================================================================
    @traycer.trace
    def finalize(self) -> Dict:
        """
        Extract temporal behavioral features from accumulated frame data.
        
        Returns:
            Dict with temporal behavioral metrics in probabilistic format:
            {
                "metric_name": {
                    "value": float | None,
                    "confidence": float,
                    "evidence_ratio": float,
                    "method": str,
                    "reason": str
                }
            }
        """
        results = {}
        
        if len(self.pose_history) < 2:
            return {'error': 'insufficient_temporal_data'}
        
        pose_df = pd.DataFrame(self.pose_history)
        face_df = pd.DataFrame(self.face_history)
        total_frames = len(pose_df)

        # ── 1. Biomechanical Posture Model (Redesign v3.0) ──
        # Metrics:
        # 1) alignment_integrity (Geometry only, full video)
        # 2) motion_activity_level (Velocity + Acceleration, valid frames)
        # 3) stability_index (Composite: 0.4*A + 0.3*(1-V) + 0.3*(1-Acc))

        cutoff_frame = int(self.fps * (self.ignore_start_ms / 1000.0))
        valid_pose_df = pose_df
        if len(pose_df) > cutoff_frame:
             valid_pose_df = pose_df.iloc[cutoff_frame:].reset_index(drop=True)

        # A. Alignment Integrity (Full Video)
        alignment_integrity = 0.0
        evidence_alignment = 0.0
        
        if 'torso_inclination_deg' in pose_df.columns:
            inclination_full = pose_df['torso_inclination_deg'].dropna()
            evidence_alignment = len(inclination_full) / max(total_frames, 1)
            
            # Smoothing (3-frame rolling mean) to reduce jitter
            smoothed_inc_full = inclination_full.rolling(3, min_periods=1).mean()
            mean_inc = smoothed_inc_full.abs().mean()
            
            total_lateral = mean_inc
            tilt_mean = 0.0
            
            if 'shoulder_tilt_angle' in pose_df.columns:
                tilt_series = pose_df['shoulder_tilt_angle'].dropna().abs()
                if not tilt_series.empty:
                    tilt_mean = tilt_series.mean()
                    total_lateral = max(tilt_mean, mean_inc)
            
            # Alignment Score: 1 - min((angle/20)^2, 1)
            penalty = (total_lateral / 20.0) ** 2
            alignment_integrity = 1.0 - min(penalty, 1.0)
            
            results['alignment_integrity'] = self._format_result(
                round(alignment_integrity, 4), 1.0, evidence_alignment, "geometric_alignment"
            )
        else:
             results['alignment_integrity'] = self._format_result(None, 0.0, 0.0, "geometric_alignment", "missing_torso_data")

        # B. Motion Activity Level (Valid Frames)
        # Includes Velocity (V) and Acceleration (Acc)
        motion_activity = 0.0
        norm_v_score = 0.0
        norm_a_score = 0.0
        evidence_motion = 0.0
        
        if 'velocity_shoulder_midpoint_x' in valid_pose_df.columns and \
           'velocity_shoulder_midpoint_y' in valid_pose_df.columns and \
           'shoulder_width_raw' in valid_pose_df.columns:
            
            vx = valid_pose_df['velocity_shoulder_midpoint_x']
            vy = valid_pose_df['velocity_shoulder_midpoint_y']
            width = valid_pose_df['shoulder_width_raw']
            
            # 1. Normalized Velocity
            v_mag = (vx**2 + vy**2)**0.5
            norm_v = v_mag / width.replace(0, np.nan)
            norm_v_score = float(norm_v.mean())
            
            # 2. Normalized Acceleration (Jitter)
            # acc = diff(velocity) / dt (dt is constant 1/fps, so just diff)
            ax = vx.diff().fillna(0)
            ay = vy.diff().fillna(0)
            a_mag = (ax**2 + ay**2)**0.5
            norm_a = a_mag / width.replace(0, np.nan)
            norm_a_score = float(norm_a.mean())
            
            # Motion Activity Level (v4.0 Biomechanical Model)
            # Normalize to [0,1] with non-linear amplification of jerky movements
            # motion = clip((V^1.2) + (0.7 * A^1.5), 0, 1)
            
            # Clip intermediate terms for safety (though normalization usually keeps them < 1)
            v_term = np.clip(norm_v_score, 0, 1) ** 1.2
            a_term = np.clip(norm_a_score, 0, 1) ** 1.5
            
            motion_activity = np.clip(v_term + (0.7 * a_term), 0.0, 1.0)
            
            evidence_motion = len(norm_v.dropna()) / max(len(valid_pose_df), 1)
            
            results['motion_activity_level'] = self._format_result(
                round(float(motion_activity), 4), 1.0, evidence_motion, "biomechanical_v4"
            )
        else:
             results['motion_activity_level'] = self._format_result(None, 0.0, 0.0, "velocity_acceleration_fusion", "missing_velocity_data")

        # C. Stability Index (v4.0 Exponential Suppression)
        # stability = alignment * exp(-1.5 * motion)
        # Ensures that even perfect alignment (1.0) drops rapidly if motion is high.
        
        if results['alignment_integrity'] and results['motion_activity_level']:
            # Using raw values calculated above:
            # alignment_integrity (float)
            # motion_activity (float)
            
            # Base Stability Calculation (Biomechanical Model v4.0)
            # stability = alignment * exp(-1.5 * motion)
            stability_index = alignment_integrity * np.exp(-1.5 * motion_activity)
            
            # Additional Motion-Based Damping (to ensure < 0.70 when motion > 0.18)
            # Apply progressive penalties for sustained motion
            if motion_activity > 0.08:
                stability_index *= (1.0 - (0.6 * motion_activity))
            
            if motion_activity > 0.15:
                stability_index *= (1.0 - (0.8 * motion_activity))
            
            # Clamp result [0,1]
            stability_index = float(np.clip(stability_index, 0.0, 1.0))
            
            results['stability_index'] = self._format_result(
                round(stability_index, 4), 1.0, min(evidence_alignment, evidence_motion), "biomechanical_v4_exponential"
            )
            
            # --- Posture Confidence Calculation (Injected) ---
            # We don't have a single time-series for "posture", but we have 'torso_inclination_deg'.
            # User requirement: expected_sigma = 0.5 for posture.
            # We will use torso inclination stability as the proxy for posture confidence signal.
            
            posture_conf_val = 0.0
            if 'torso_inclination_deg' in pose_df.columns:
                 inc_clean = pose_df['torso_inclination_deg'].dropna()
                 # Expected sigma for posture = 0.5
                 posture_conf_val = self._calculate_signal_confidence(inc_clean, expected_sigma=0.5)
            
            # Save this confidence into the stability_index metric so it can be retrieved later
            results['stability_index']['confidence'] = posture_conf_val
        else:
            results['stability_index'] = self._format_result(None, 0.0, 0.0, "biomechanical_composite_v3", "insufficient_component_data")
            
        # Removed legacy metrics: posture_stability, posture_uprightness, overall_movement_intensity

        # ── 5. Gaze Analysis (Lean Iris-Only Pipeline v3 — 3-Band) ──
        # center_distance = abs(ratio_x - 0.5)
        # strong_contact = center_distance < 0.05
        # soft_contact   = center_distance < 0.10
        
        if 'iris_ratio_x' in face_df.columns:
            ratio_x = face_df['iris_ratio_x']
            evidence = ratio_x.notna().sum() / max(total_frames, 1)

            if evidence >= 0.1 and ratio_x.notna().sum() >= 5:
                clean_ratio = ratio_x.dropna()
                n = max(len(clean_ratio), 1)
                
                # 1. Center Distance
                center_distance = (clean_ratio - 0.5).abs()
                
                # 2. 3-Band Contact Classification
                strong_mask = (center_distance < 0.05)
                soft_mask   = (center_distance < 0.10)
                
                ecr_strong = strong_mask.sum() / n
                ecr_soft   = soft_mask.sum() / n
                
                # 3. Consistency = ECR_soft (primary metric)
                consistency_val = ecr_soft
                
                # 4. Stability (Inverse Std of ratio_x — unchanged)
                std_ratio = clean_ratio.std()
                stability_val = 1.0 / (1.0 + std_ratio)
                
                # 5. Switch Count (using soft_mask)
                switches = (soft_mask.astype(int).diff().abs() > 0).sum()
                
                # 6. Max Continuous Duration (using soft_mask)
                soft_frames = soft_mask.sum()
                if soft_frames > 0:
                    mask_int = soft_mask.astype(int)
                    groups = (mask_int != mask_int.shift()).cumsum()
                    run_lengths = mask_int.groupby(groups).sum()
                    contact_runs = run_lengths[mask_int.groupby(groups).first() == 1]
                    max_duration_frames = contact_runs.max() if not contact_runs.empty else 0
                    max_dur_s = max_duration_frames / self.fps
                else:
                    max_dur_s = 0.0
                
                # Diagnostics
                diag = {
                    "ECR_strong": round(float(ecr_strong), 4),
                    "ECR_soft": round(float(ecr_soft), 4),
                    "strong_ratio": round(float(ecr_strong), 4),
                    "soft_ratio": round(float(ecr_soft), 4),
                    "mean_ratio_x": round(float(clean_ratio.mean()), 4),
                    "std_ratio_x": round(float(std_ratio), 4),
                    "mean_center_distance": round(float(center_distance.mean()), 4),
                    "logic_mode": "iris_ratio_v3"
                }

                # Store Results
                results['eye_contact_consistency'] = self._format_result(
                    round(float(consistency_val), 4), evidence, evidence, "iris_ratio_3band"
                )
                results['eye_contact_consistency']['diagnostics'] = diag
                
                # Stability
                results['gaze_stability'] = self._format_result(
                    round(float(stability_val), 4), evidence, evidence, "inv_std_iris_ratio"
                )
                
                # Switch Count
                results['gaze_direction_switch_count'] = self._format_result(
                    int(switches), evidence, evidence, "binary_switch_count"
                )
                
                # Duration
                results['max_continuous_eye_contact'] = self._format_result(
                    round(float(max_dur_s), 2), evidence, evidence, "max_run_binary"
                )
                results['eye_contact_duration'] = results['max_continuous_eye_contact']
                
                # Stable Duration (rolling std of center_distance)
                window = min(10, len(center_distance))
                rolling_std = center_distance.rolling(window=window, center=True).std()
                stable_frames_count = (rolling_std <= 0.05).sum()
                stable_dur_s = stable_frames_count / self.fps
                results['stable_gaze_duration'] = self._format_result(
                     round(float(stable_dur_s), 2), evidence, evidence, "rolling_std_iris_ratio"
                )

            else:
                 # Insufficient Data
                 results['eye_contact_consistency'] = self._format_result(0.0, 0.0, evidence, "insufficient_data")
                 results['gaze_stability'] = self._format_result(0.0, 0.0, evidence, "insufficient_data")
                 results['gaze_direction_switch_count'] = self._format_result(0, 0.0, evidence, "insufficient_data")
                 results['max_continuous_eye_contact'] = self._format_result(0.0, 0.0, evidence, "insufficient_data")
                 results['stable_gaze_duration'] = self._format_result(0.0, 0.0, evidence, "insufficient_data")

        else:
            # Fallback (No Iris Ratio Data)
            results['eye_contact_consistency'] = self._format_result(None, 0.0, 0.0, "iris_ratio", "missing_iris_data")
            results['gaze_stability'] = self._format_result(None, 0.0, 0.0, "iris_ratio", "missing_iris_data")
            results['gaze_direction_switch_count'] = self._format_result(0, 0.0, 0.0, "attention_shifts", "missing_iris_data")
            results['max_continuous_eye_contact'] = self._format_result(0.0, 0.0, 0.0, "contiguous_segment", "missing_iris_data")
            results['stable_gaze_duration'] = self._format_result(0.0, 0.0, 0.0, "rolling_std_iris_ratio", "missing_iris_data")

        # ── 6. Overall Movement (Redesign v2.1) ──
        # Normalized velocity of shoulder midpoint
        # v = displacement / width / dt
        if 'velocity_shoulder_midpoint_x' in valid_pose_df.columns and \
           'velocity_shoulder_midpoint_y' in valid_pose_df.columns and \
           'shoulder_width_raw' in valid_pose_df.columns:
            
            vx = valid_pose_df['velocity_shoulder_midpoint_x']
            vy = valid_pose_df['velocity_shoulder_midpoint_y']
            width = valid_pose_df['shoulder_width_raw']
            
            # aligned indices? they verify pandas alignment
            # Calculate magnitude
            v_mag = (vx**2 + vy**2)**0.5
            
            # Normalize by width (avoid div by zero/small)
            # Use width from same frame
            norm_v = v_mag / width.replace(0, np.nan)
            movement_score = float(norm_v.mean())
            
            # Evidence based on valid frames only? Or total? 
            # If we slice, the metric represents the sliced period.
            # Evidence ratio usually means % of video successfully tracked.
            # If we intentionally ignore start, that's policy, not missing data.
            # But the 'evidence' is technically the count of valid samples.
            evidence_movement = len(norm_v.dropna()) / max(len(valid_pose_df), 1)

            results['overall_movement_intensity'] = self._format_result(
                round(movement_score, 4), 1.0, evidence_movement, "normalized_shoulder_velocity"
            )
        else:
             # Fallback: use old method if new features missing
            vel_cols = [c for c in valid_pose_df.columns if c.startswith('velocity_')]
            if vel_cols:
                vel_mag = valid_pose_df[vel_cols].abs().mean(axis=1)
                val = float(vel_mag.mean())
                evidence_movement = len(vel_mag) / max(len(valid_pose_df), 1)
                results['overall_movement_intensity'] = self._format_result(
                    round(val, 4), 1.0, evidence_movement, "velocity_mean_fallback"
                )
            else:
                results['overall_movement_intensity'] = self._format_result(
                    None, 0.0, 0.0, "velocity_mean", "no_velocity_data"
                )

        # ── 7. Confidence Indicators ──
        # Redesigned to be weighted average of component certainties
        # Components:
        # 1. Data Coverage (completeness)
        # 2. Gaze Confidence
        # 3. Posture Confidence
        # 4. Blink Confidence
        
        # Retrieve component confidences (default to 0 if missing)
        c_gaze = 0.0
        if 'eye_contact_consistency' in results:
             c_gaze = results['eye_contact_consistency'].get('confidence', 0.0)
             
        c_posture = 0.0
        # Fix: Read from 'stability_index', not legacy 'posture_stability'
        if 'stability_index' in results:
             c_posture = results['stability_index'].get('confidence', 0.0)
             
        c_blink = 0.0
        if 'blink_rate' in results:
             c_blink = results['blink_rate'].get('confidence', 0.0)
             
        # Calculate overall data completeness (already computed)
        c_data = self._calculate_data_completeness(pose_df, face_df)
        
        # Weighted Average (Visual Domain Only)
        # Weights:
        # Gaze: 30% (Critical for interaction)
        # Posture: 30% (Available in most frames)
        # Blink: 20% (Subtle but important)
        # Data: 20% (Base reliability)
        final_conf = (c_gaze * 0.3) + (c_posture * 0.3) + (c_blink * 0.2) + (c_data * 0.2)
        
        # Add composite confidence metric
        results['visual_confidence'] = self._format_result(
            round(final_conf, 4), 1.0, c_data, "weighted_visual_composite"
        )
        
        # Diagnostic print of confidence breakdown
        print(f"VISUAL CONFIDENCE DIAGNOSTICS: Gaze={c_gaze:.2f}, Posture={c_posture:.2f}, "
              f"Blink={c_blink:.2f}, Data={c_data:.2f} -> Final={final_conf:.4f}")
              
        print({
            "gaze_certainty": c_gaze,
            "posture_certainty": c_posture,
            "blink_certainty": c_blink,
            "data_certainty": c_data,
            "weights_used": {"gaze": 0.3, "posture": 0.3, "blink": 0.2, "data": 0.2},
            "visual_confidence_before_rounding": final_conf
        })

        # ── 7b. Posture Audit Diagnostics ──
        if 'torso_inclination_deg' in pose_df.columns:
            inc = pose_df['torso_inclination_deg'].dropna()
            
            tilt_mean = 0.0
            if 'shoulder_tilt_angle' in pose_df.columns:
                tilt_mean = pose_df['shoulder_tilt_angle'].dropna().abs().mean()
            
            # Recalculate total logic for diagnostic print (approximate since we used smoothed above)
            total_lateral_diag = max(tilt_mean, inc.mean())
            
            print(f"\nPOSTURE DIAGNOSTICS (v2.3):")
            print({
                "torso_vertical_mean_deg": round(float(inc.mean()), 2),
                "shoulder_tilt_mean_deg": round(float(tilt_mean), 2),
                "total_lateral_tilt_deg": round(float(total_lateral_diag), 2),
                "upright_score": results.get('posture_uprightness', {}).get('value'),
                "stability_score": results.get('posture_stability', {}).get('value'),
                "movement_score": results.get('overall_movement_intensity', {}).get('value')
            })

        # ── 8. Data Quality ──
        results['data_completeness'] = self._format_result(
            round(c_data, 4), 1.0, 1.0, "feature_coverage"
        )

        return results

    # ==================================================================
    # Stability metric (shared helper)
    # ==================================================================
    def _calculate_stability(self, series: pd.Series) -> Tuple[Optional[float], float, str]:
        """Calculate stability via coefficient of variation. Returns (value, confidence, reason)."""
        if len(series) < 5:
            return None, 0.0, "insufficient_samples"

        mean_val = series.mean()
        if mean_val == 0:
            return 1.0, 1.0, "perfectly_stable_at_zero"

        cv = series.std() / abs(mean_val)
        stability = 1.0 / (1.0 + cv)
        return round(float(stability), 4), 1.0, "calculated"

    # ==================================================================
    # Data completeness
    # ==================================================================
    def _calculate_data_completeness(self, pose_df: pd.DataFrame, face_df: pd.DataFrame) -> float:
        """Calculate data completeness ratio (0.0 to 1.0)."""
        if len(pose_df) == 0:
            return 0.0

        pose_features = ['shoulder_tilt_angle', 'head_height_ratio', 'eye_distance_ratio', 'shoulder_hip_ratio']
        pose_completeness = sum(1 for f in pose_features if f in pose_df.columns) / len(pose_features)

        face_features = ['left_eye_opening_ratio', 'right_eye_opening_ratio', 'mouth_opening_ratio',
                         'smile_intensity', 'iris_ratio_x', 'eyebrow_raise_ratio']
        face_completeness = sum(1 for f in face_features if f in face_df.columns) / len(face_features)

        return pose_completeness * 0.5 + face_completeness * 0.5

    # ==================================================================
    # PHASE 7 — Calibration Diagnostics
    # ==================================================================
    def print_calibration_diagnostics(self):
        """
        Print comprehensive calibration diagnostics to server logs.
        Read-only — does NOT modify any state.
        """
        print("\n" + "=" * 70)
        print("  CALIBRATION DIAGNOSTICS: Adaptive Behavioral Modeling v2.0")
        print("=" * 70)

        face_df = pd.DataFrame(self.face_history) if self.face_history else pd.DataFrame()
        pose_df = pd.DataFrame(self.pose_history) if self.pose_history else pd.DataFrame()
        total = len(pose_df)

        # --- Frame counts ---
        face_valid = sum(1 for f in self.face_history if f) if self.face_history else 0
        print(f"\n--- FRAME COUNTS ---")
        print(f"  Pose frames:  {total}")
        print(f"  Face frames:  {face_valid} / {len(self.face_history)}")

        # --- EAR distribution ---
        has_ear = ('left_eye_opening_ratio' in face_df.columns and
                   'right_eye_opening_ratio' in face_df.columns) if len(face_df) > 0 else False
        if has_ear:
            avg_ear = ((face_df['left_eye_opening_ratio'] + face_df['right_eye_opening_ratio']) / 2).dropna()
            if len(avg_ear) > 0:
                print(f"\n--- EAR DISTRIBUTION (avg of both eyes) ---")
                print(f"  count:  {len(avg_ear)}")
                print(f"  min:    {avg_ear.min():.6f}")
                print(f"  p5:     {avg_ear.quantile(0.05):.6f}")
                print(f"  median: {avg_ear.median():.6f}")
                print(f"  mean:   {avg_ear.mean():.6f}")
                print(f"  p95:    {avg_ear.quantile(0.95):.6f}")
                print(f"  max:    {avg_ear.max():.6f}")
                print(f"  std:    {avg_ear.std():.6f}")

                # Z-score distribution
                std_e = avg_ear.std()
                if std_e > 1e-8:
                    z = (avg_ear - avg_ear.mean()) / std_e
                    print(f"\n--- EAR Z-SCORE DISTRIBUTION ---")
                    print(f"  min z:  {z.min():.3f}")
                    print(f"  p5 z:   {z.quantile(0.05):.3f}")
                    print(f"  median: {z.median():.3f}")
                    print(f"  p95 z:  {z.quantile(0.95):.3f}")
                    print(f"  max z:  {z.max():.3f}")
                    below = (z < self.blink_z_threshold).sum()
                    print(f"  frames below z={self.blink_z_threshold}: {below} / {len(z)}")

        # --- Iris ratio distribution ---
        if 'iris_ratio_x' in face_df.columns and len(face_df) > 0:
            ratio_x = face_df['iris_ratio_x'].dropna()
            if len(ratio_x) > 0:
                print(f"\n--- IRIS RATIO_X DISTRIBUTION ---")
                print(f"  count:   {len(ratio_x)}")
                print(f"  min:     {ratio_x.min():.4f}")
                print(f"  median:  {ratio_x.median():.4f}")
                print(f"  mean:    {ratio_x.mean():.4f}")
                print(f"  max:     {ratio_x.max():.4f}")
                print(f"  std:     {ratio_x.std():.4f}")
                centered = ((ratio_x - 0.5).abs() < 0.18).sum()
                print(f"  frames centered (<0.18): {centered} / {len(ratio_x)} ({centered/len(ratio_x)*100:.1f}%)")

        # --- Smile intensity distribution ---
        if 'smile_intensity' in face_df.columns and len(face_df) > 0:
            smile = face_df['smile_intensity'].dropna()
            if len(smile) > 0:
                print(f"\n--- SMILE INTENSITY DISTRIBUTION ---")
                print(f"  count:  {len(smile)}")
                print(f"  min:    {smile.min():.6f}")
                print(f"  median: {smile.median():.6f}")
                print(f"  mean:   {smile.mean():.6f}")
                print(f"  max:    {smile.max():.6f}")
                print(f"  std:    {smile.std():.6f}")
                std_s = smile.std()
                if std_s > 1e-8:
                    z_s = (smile - smile.mean()) / std_s
                    above = (z_s > 1.0).sum()
                    print(f"  z > 1.0 (above baseline): {above} / {len(smile)}")

        # --- Summary ---
        print(f"\n--- PARAMETERS IN USE ---")
        print(f"  blink_z_threshold:      {self.blink_z_threshold}")
        print(f"  blink_min_duration_ms:  {self.blink_min_duration_ms}")
        print(f"  blink_max_duration_ms:  {self.blink_max_duration_ms}")
        print(f"  blink_min_consecutive:  {self.blink_min_consecutive}")
        print(f"  gaze_align_threshold:   {self.gaze_alignment_threshold_deg}°")
        print(f"  smoothing_window:       {self.smoothing_window}")
        print(f"  fps:                    {self.fps}")
        print("=" * 70 + "\n")

    # ==================================================================
    # Reset
    # ==================================================================
    def reset(self):
        """Reset all temporal buffers."""
        self.pose_history.clear()
        self.face_history.clear()
        self.timestamps.clear()
        self.prev_pose = None
        self.prev_face = None
        self.prev_timestamp = None
