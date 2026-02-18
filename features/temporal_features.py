"""
Temporal feature extraction for behavioral analysis.

Transforms frame-level features into time-aware behavioral metrics.
Captures dynamics, patterns, and stability over sliding windows.

v2.0 — Adaptive behavioral modeling:
- Z-score adaptive blink detection (no fixed EAR thresholds)
- Gaze-angle eye contact (cumulative, not contiguous)
- Adaptive expression modeling with per-video normalisation
- Probabilistic output with evidence_ratio and method
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
        # Eye contact
        gaze_alignment_threshold_deg: float = 6.0,
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

        # Eye contact parameters
        self.gaze_alignment_threshold_deg = gaze_alignment_threshold_deg

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

        return blink_count, blink_events, confidence

    # ==================================================================
    # PHASE 3 — Gaze-Angle Eye Contact
    # ==================================================================
    def _compute_eye_contact(self, gaze_angles: pd.Series, gaze_yaws: pd.Series = None) -> Tuple[float, float, float, int]:
        """
        Compute eye contact score and direction switches.

        Returns:
            (ratio, total_duration, confidence, switch_count, max_continuous_duration, diagnostics)
        """
        if len(gaze_angles) < 3:
            return 0.0, 0.0, 0.0, 0
            
        clean_angle = gaze_angles.dropna()
        # Ensure alignment of indices if yaw is provided
        if gaze_yaws is not None:
             clean_yaw = gaze_yaws.loc[clean_angle.index]
        else:
             clean_yaw = pd.Series([0.0]*len(clean_angle), index=clean_angle.index)

        total = len(clean_angle)
        
        # 0. Normalize Head Yaw (if provided)
        # Map to [-90, +90] range.
        if gaze_yaws is not None:
             # Standard normalization to [-180, 180] first
             clean_yaw = (clean_yaw + 180) % 360 - 180
             # Then clamp to [-90, 90] as valid range
             clean_yaw = clean_yaw.clip(-90, 90)
        else:
             # Default to 0 if not provided
             clean_yaw = pd.Series([0.0]*len(clean_angle), index=clean_angle.index)
        
        # 1. Probabilistic Gaze Model (v5.0 Gaussian)
        # Gaze Probability: exp(-(gaze^2) / (2 * sigma^2))
        sigma = 8.0 # User defined webcam tolerance
        gaze_sq = clean_angle.pow(2)
        gaze_prob = np.exp(-gaze_sq / (2 * (sigma ** 2)))
        gaze_prob = gaze_prob.clip(0.0, 1.0)
        
        # Head Pose Soft Confidence
        # head_weight = exp(-(yaw^2) / (2 * head_sigma^2))
        head_sigma = 35.0
        yaw_sq = clean_yaw.pow(2)
        head_weight = np.exp(-yaw_sq / (2 * (head_sigma ** 2)))
        head_weight = head_weight.clip(0.3, 1.0) # Never zero-out
        
        # Final Contact Probability
        # weighted combination
        final_contact_prob = (gaze_prob * 0.75) + (gaze_prob * head_weight * 0.25)
        
        # consistency = mean probability
        eye_contact_ratio = final_contact_prob.mean()
        
        # Binary Classification for duration/switches
        is_contact = final_contact_prob >= 0.5
        
        # Diagnostics Trace (First 50 frames)
        frame_trace = []
        limit = min(50, len(clean_angle))
        
        # Safe extraction of aligned data for tracing
        # We'll re-iterate or use index lookup
        trace_indices = clean_angle.index[:limit]
        
        for idx in trace_indices:
             current_angle = clean_angle.loc[idx]
             current_yaw = clean_yaw.loc[idx]
             g_prob = gaze_prob.loc[idx]
             h_weight = head_weight.loc[idx]
             f_prob = final_contact_prob.loc[idx]
             is_c = is_contact.loc[idx]
             
             frame_trace.append({
                 "frame_idx": int(idx),
                 "gaze_angle": round(float(current_angle), 2),
                 "gaze_probability": round(float(g_prob), 4),
                 "head_yaw": round(float(current_yaw), 2),
                 "head_weight": round(float(h_weight), 4),
                 "final_contact_probability": round(float(f_prob), 4),
                 "binary_contact": bool(is_c),
                 "logic_mode": "gaussian_fusion"
             })
             
        contact_count = is_contact.sum()
        # eye_contact_ratio is already taking mean probability above
        # But wait, original code used `contact_count / total` for ratio.
        # User requirement: "eye_contact_consistency = mean(final_contact_probability across frames)"
        # So I stick with `final_contact_prob.mean()`.
        
        duration_s = contact_count / self.fps # Accumulate duration of "binary contact" frames for "duration" metric?
        # User said "BUT store raw probability for consistency computation."
        # User didn't specify strict change for 'duration_s'. 
        # Usually duration refers to time spent in contact. 
        # I'll use binary contact count for duration to be physically interpretable.
        
        # 2. Confidence Calculation

        # 2. Confidence Calculation
        # Confidence is primarily data coverage. 
        # We DO NOT based it on variance because steady off-camera gaze (low variance) 
        # should not be "high confidence eye contact".
        # But wait, "confidence" here is "confidence in the metric measurement", not "confidence of the person".
        # If the person looks away steadily, we are CONFIDENT they are looking away.
        # So low variance IS high confidence in the measurement.
        # The user said: "Do NOT base consistency on signal variance." -> This refers to the SCORE, not confidence.
        # The score is now purely ratio-based.
        # But for 'visual_confidence' metric later, we want high reliability.
        coverage = total / max(len(gaze_angles), 1)
        
        # We keep the signal confidence logic for the METRIC VALIDITY, but we ensure output score is low.
        # Expected sigma for gaze = 0.3
        signal_conf = self._calculate_signal_confidence(clean_angle, threshold=self.gaze_alignment_threshold_deg, expected_sigma=0.3)
        
        confidence = coverage * 0.5 + signal_conf * 0.5
        confidence = min(1.0, max(0.0, confidence))

        # 3. Direction State Detection & Switch Counting
        # States: "camera", "left", "right"
        # We need smoothing to avoid micro-jitter transitions
        
        # Smooth the yaw signal
        smoothed_yaw = clean_yaw.rolling(window=5, center=True, min_periods=1).median()
        smoothed_angle = clean_angle.rolling(window=5, center=True, min_periods=1).median()
        
        # Determine State per frame
        # Camera: angle < threshold
        # Left: angle >= threshold AND yaw > 0
        # Right: angle >= threshold AND yaw < 0  (assuming standard signs)
        
        states = []
        threshold = self.gaze_alignment_threshold_deg
        
        for i in range(len(smoothed_angle)):
            ang = smoothed_angle.iloc[i]
            yaw = smoothed_yaw.iloc[i]
            
            if ang < threshold:
                states.append(0) # Camera
            elif yaw > 0:
                states.append(1) # Left (approx)
            else:
                states.append(-1) # Right (approx)
                
        # Count transitions between different states
        state_series = pd.Series(states)
        # Shift to find changes
        switches = (state_series != state_series.shift()).sum() - 1 # -1 because first item compares to NaN/None
        switches = max(0, int(switches))
        
        # 4. Max Continuous Duration (Sustained Attention)
        # Identify contiguous blocks where angle < threshold
        # is_contact is a boolean series
        # We want the longest run of True
        is_contact_int = is_contact.astype(int)
        # Group by value change
        groups = (is_contact_int != is_contact_int.shift()).cumsum()
        # Filter for groups where value is 1 (True)
        # Calculate size of each group
        runs = is_contact_int.groupby(groups)
        
        max_duration_frames = 0
        for g, frame_indices in runs.groups.items():
             # Check if this group corresponds to "True" (contact)
             # Get the value of the first index in the group
             first_idx = frame_indices[0]
             val = is_contact_int.loc[first_idx]
             if val == 1:
                  duration = len(frame_indices)
                  if duration > max_duration_frames:
                       max_duration_frames = duration
                       
        max_duration_s = max_duration_frames / self.fps
                       
        max_duration_s = max_duration_frames / self.fps
        
        # Diagnostics Summary
        diagnostics = {
             "frame_trace": frame_trace,
             "summary": {
                  "mean_gaze_probability": round(float(gaze_prob.mean()), 4),
                  "mean_head_weight": round(float(head_weight.mean()), 4),
                  "mean_final_probability": round(float(final_contact_prob.mean()), 4),
                  "frames_above_0.5": int(is_contact.sum()),
                  "logic_mode": "gaussian_fusion"
             }
        }
                       
        return eye_contact_ratio, duration_s, confidence, switches, max_duration_s, diagnostics

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

        # ── 2. Gaze Stability (from eye_distance_ratio) ──
        if 'eye_distance_ratio' in pose_df.columns:
            edr = pose_df['eye_distance_ratio'].dropna()
            evidence = len(edr) / total_frames
            if len(edr) >= 5:
                # Center-Weighted Gaze Stability (v4.1)
                # base_stability = 1 / (1 + std_angle)
                # center_bias = 1 - (mean_angle / effective_threshold)
                # center_bias = clamp(center_bias, 0, 1)
                # gaze_stability = base_stability * center_bias
                
                mean_angle = edr.abs().mean()
                std_angle = edr.std()
                
                base_stability = 1.0 / (1.0 + std_angle)
                
                # Use class threshold (6.0)
                eff_threshold = self.gaze_alignment_threshold_deg
                center_bias = 1.0 - (mean_angle / eff_threshold)
                center_bias = max(0.0, min(1.0, center_bias))
                
                val = base_stability * center_bias
                
                # Confidence: Based on evidence ratio and signal stability
                conf = evidence * 0.5 + base_stability * 0.5 # rudimentary conf
                conf = min(1.0, max(0.0, conf))
                
                results['gaze_stability'] = self._format_result(val, conf, evidence, "center_weighted_stability")
                
                # Stable gaze duration (cumulative, rolling std <= threshold)
                window = min(10, len(edr))
                rolling_std = edr.rolling(window=window, center=True).std()
                stable_frames = (rolling_std <= 0.05).sum()
                stable_dur = stable_frames / self.fps
                results['stable_gaze_duration'] = self._format_result(
                    round(float(stable_dur), 2), conf, evidence, "rolling_std"
                )
            else:
                results['gaze_stability'] = self._format_result(None, 0.0, evidence, "cv_stability", "insufficient_samples")
                results['stable_gaze_duration'] = self._format_result(None, 0.0, evidence, "rolling_std", "insufficient_samples")
        else:
            results['gaze_stability'] = self._format_result(None, 0.0, 0.0, "cv_stability", "missing_eye_landmarks")
            results['stable_gaze_duration'] = self._format_result(None, 0.0, 0.0, "rolling_std", "missing_eye_landmarks")

        # ── 3. Expression Metrics (Phase 4) ──
        expr = self._compute_expression_metrics(face_df)
        results.update(expr)

        # ── 4. Blink Rate (Phase 2 — adaptive z-score) ──
        has_left = 'left_eye_opening_ratio' in face_df.columns
        has_right = 'right_eye_opening_ratio' in face_df.columns
        if has_left and has_right:
            left = face_df['left_eye_opening_ratio']
            right = face_df['right_eye_opening_ratio']
            avg_ear = ((left + right) / 2).dropna()
            evidence = len(avg_ear) / max(total_frames, 1)

            if evidence >= 0.1 and len(avg_ear) >= 5:
                # Smooth EAR before detection
                smoothed = self._smooth(avg_ear)
                blink_count, blink_events, blink_conf = self._detect_blinks_zscore(smoothed)

                duration_min = len(avg_ear) / self.fps / 60.0
                if duration_min > 0:
                    rate = blink_count / duration_min
                    results['blink_rate'] = self._format_result(
                        round(float(rate), 2), blink_conf, evidence, "adaptive_zscore"
                    )
                else:
                    results['blink_rate'] = self._format_result(
                        None, 0.0, evidence, "adaptive_zscore", "insufficient_duration"
                    )
            else:
                results['blink_rate'] = self._format_result(
                    None, 0.0, evidence, "adaptive_zscore", "insufficient_eye_data"
                )
        else:
            results['blink_rate'] = self._format_result(
                None, 0.0, 0.0, "adaptive_zscore", "missing_eye_landmarks"
            )

        # ── 5. Eye Contact (Phase 3 — gaze angle) ──
        if 'gaze_alignment_angle' in face_df.columns:
            gaze = face_df['gaze_alignment_angle']
            evidence = gaze.notna().sum() / max(total_frames, 1)

            if evidence >= 0.1 and gaze.notna().sum() >= 5:
                # Get yaw if available, else None
                gaze_yaw = face_df.get('gaze_yaw', None)
                
                # Get yaw if available, else None
                gaze_yaw = face_df.get('gaze_yaw', None)
                
                score, cum_dur, conf, switches, max_dur, diag = self._compute_eye_contact(gaze, gaze_yaw)
                results['eye_contact_consistency'] = self._format_result(
                    round(float(score), 4), conf, evidence, "gaze_angle"
                )
                # Attach diagnostics to result metadata/details if supported, 
                # or just ensure we unpacked it safely.
                # User requested "safely ignore diagnostics if not needed".
                # But we might want to store it in results['eye_contact_consistency']['diagnostics']?
                # The _format_result returns a dict. We can extend it.
                if isinstance(results['eye_contact_consistency'], dict):
                     results['eye_contact_consistency']['diagnostics'] = diag
                results['eye_contact_duration'] = self._format_result(
                    round(float(cum_dur), 2), conf, evidence, "gaze_angle"
                )
                results['max_continuous_eye_contact'] = self._format_result(
                    round(float(max_dur), 2), conf, evidence, "contiguous_segment"
                )
                results['gaze_direction_switch_count'] = self._format_result(
                    int(switches), conf, evidence, "attention_shifts"
                )
                results['gaze_direction_switch_count'] = self._format_result(
                    int(switches), conf, evidence, "attention_shifts"
                )
            else:
                results['eye_contact_consistency'] = self._format_result(
                    None, 0.0, evidence, "gaze_angle", "insufficient_gaze_data"
                )
                results['eye_contact_duration'] = self._format_result(
                    None, 0.0, evidence, "gaze_angle", "insufficient_gaze_data"
                )
                results['max_continuous_eye_contact'] = self._format_result(
                    None, 0.0, evidence, "contiguous_segment", "insufficient_gaze_data"
                )
                results['gaze_direction_switch_count'] = self._format_result(
                    0, 0.0, evidence, "attention_shifts", "insufficient_data"
                )
        else:
            # Fallback: use EAR stability if gaze angle not available
            if has_left and has_right:
                avg_ear = ((face_df['left_eye_opening_ratio'] + face_df['right_eye_opening_ratio']) / 2).dropna()
                evidence = len(avg_ear) / max(total_frames, 1)
                if len(avg_ear) >= 5:
                    val, conf, reason = self._calculate_stability(avg_ear)
                    results['eye_contact_consistency'] = self._format_result(
                        val, conf, evidence, "ear_stability_fallback", reason
                    )
                else:
                    results['eye_contact_consistency'] = self._format_result(
                        None, 0.0, evidence, "ear_stability_fallback", "insufficient_samples"
                    )
            else:
                results['eye_contact_consistency'] = self._format_result(
                    None, 0.0, 0.0, "gaze_angle", "missing_eye_landmarks"
                )
                results['eye_contact_duration'] = self._format_result(
                    None, 0.0, 0.0, "gaze_angle", "missing_gaze_data"
                )
                results['max_continuous_eye_contact'] = self._format_result(
                    None, 0.0, 0.0, "contiguous_segment", "missing_data"
                )
                results['gaze_direction_switch_count'] = self._format_result(
                    0, 0.0, 0.0, "attention_shifts", "missing_data"
                )

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
                         'smile_intensity', 'gaze_alignment_angle', 'eyebrow_raise_ratio']
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

        # --- Gaze angle distribution ---
        if 'gaze_alignment_angle' in face_df.columns and len(face_df) > 0:
            gaze = face_df['gaze_alignment_angle'].dropna()
            if len(gaze) > 0:
                print(f"\n--- GAZE ANGLE DISTRIBUTION ---")
                print(f"  count:   {len(gaze)}")
                print(f"  min:     {gaze.min():.2f}°")
                print(f"  median:  {gaze.median():.2f}°")
                print(f"  mean:    {gaze.mean():.2f}°")
                print(f"  max:     {gaze.max():.2f}°")
                print(f"  std:     {gaze.std():.2f}°")
                aligned = (gaze < self.gaze_alignment_threshold_deg).sum()
                print(f"  frames < {self.gaze_alignment_threshold_deg}°: {aligned} / {len(gaze)} ({aligned/len(gaze)*100:.1f}%)")

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
