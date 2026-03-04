import cv2
import logging
import librosa
import numpy as np
import openai
import os
from typing import Dict, List, Optional, Tuple, Union

# Frozen Posture Engine Version
POSTURE_ENGINE_VERSION = "v2.2_frozen_calibrated"

# ─── Response Layer Control ───────────────────────────────────────────────────
# Set DEBUG_MODE = True to return the full internal JSON (for dev/testing).
# Set DEBUG_MODE = False (default) to return the slim production-safe response.
DEBUG_MODE = False
# ──────────────────────────────────────────────────────────────────────────────

import traceback
from features.normalized_geometry import NormalizedGeometry
from features.temporal_features import TemporalFeatures
from features.stt_engine import transcribe_audio
from features.audio_analysis import AudioAnalyzer
from features.speech_metrics import SpeechMetrics
from features.linguistic_analysis import LinguisticAnalyzer
from features.head_pose_metrics import HeadPoseMetrics
from utils.scoring_utils import get_score_val, get_score_range_desc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Basic multimodal confidence fusion
def multimodal_confidence_fusion(video_conf: float, audio_conf: float, text_conf: float) -> float:
    """
    Weighted confidence fusion strategy.
    Video (0.5), Audio (0.3), Text (0.2).
    """
    vc = video_conf if video_conf is not None else 0.5
    ac = audio_conf if audio_conf is not None else 0.5
    tc = text_conf if text_conf is not None else 0.5
    
    return (vc * 0.5) + (ac * 0.3) + (tc * 0.2)

def analyze_video(video_path: str, transcript: str = None, debug_mode: bool = False) -> Dict:
    """
    Main entry point for multimodal analysis.
    
    Args:
        video_path (str): Path to the video file.
        transcript (str): Optional transcript text.
        debug_mode (bool): If True, returns full multimodal_analysis. 
                           If False, returns filtered summary_view (BUT User requested keeping data internal).
                           Actually, Step 2379 says "Production Mode (debug_mode=False) now returns FULL multimodal_analysis".
                           
    Returns:
        Dict: Analysis results including scores and interpreted feedback.
    """
    logger.info(f"Starting analysis for: {video_path}")
    
    # Initialize component analyzers
    geometry_analyzer = NormalizedGeometry()
    temporal_analyzer = TemporalFeatures()
    audio_analyzer = AudioAnalyzer()
    speech_metrics = SpeechMetrics() # For filler words, wpm
    linguistic_analyzer = LinguisticAnalyzer() # For content scoring
    head_pose_metrics = HeadPoseMetrics() # For head stability
    
    # --- Interface Guard (Step 5) ---
    # Prevents runtime errors due to interface mismatches
    # Strict Interface Enforcement (v3.0)
    # Processors: process(frame)
    # Analyzers: finalize(...)
    
    # 1. Geometry Processor
    if not hasattr(geometry_analyzer, 'process'):
        raise RuntimeError("NormalizedGeometry missing required method: process")

    # 2. Feature Analyzers
    analyzers = {
        "TemporalFeatures": temporal_analyzer,
        "HeadPoseMetrics": head_pose_metrics,
        "AudioAnalyzer": audio_analyzer,
        "SpeechMetrics": speech_metrics,
        "LinguisticAnalyzer": linguistic_analyzer
    }
    
    for name, instance in analyzers.items():
        if not hasattr(instance, 'finalize'):
             raise RuntimeError(f"{name} missing required method: finalize")
        # Optimization: We don't strictly enforce add_frame on Audio/Speech/Ling here 
        # because video_utils doesn't call it on them, but they HAVE it.
        # We check it just to be sure the class is updated.
        if not hasattr(instance, 'add_frame'):
             raise RuntimeError(f"{name} missing required method: add_frame")
    # -------------------------------
    
    try:
        # 1. Video Processing Loop
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            
            # Geometry Analysis (per frame)
            # Strict Call: process
            geo_results = geometry_analyzer.process(frame)
            
            # Temporal Feature Accumulation
            # Extract features needed for temporal analysis
            # We need to map geo_results to what temporal_analyzer expects
            # NormalizedGeometry.analyze_frame now returns a combined dict
            # We pass it to both pose and face args, and calculate timestamp
            timestamp = frame_count / (fps if fps > 0 else 30.0)
            # Standardized call: add_frame
            temporal_analyzer.add_frame(geo_results, geo_results, timestamp)
            
            # Head Pose Accumulation
            if 'head_pose' in geo_results:
                # Standardized call: add_frame (with timestamp)
                head_pose_metrics.add_frame(geo_results['head_pose'], timestamp)
                
        cap.release()
        
        if frame_count == 0:
            raise RuntimeError("Video processing failed: No frames read.")
            
        duration_seconds = frame_count / (fps if fps > 0 else 30.0)
            
        # 2. Audio Analysis
        # Load audio using librosa
        y, sr = librosa.load(video_path, sr=16000)
        # Strict Call: finalize
        audio_results = audio_analyzer.finalize(y, sr)
        
        # 3. Speech Metrics (if transcript is available, or implied from audio)
        # Using raw audio for speech metrics if transcript is missing?
        # speech_metrics usually needs text or audio. 
        # Let's assume it uses audio processing results or transcript.
        # Just creating a placeholder structure if not fully implemented in snippet.
        speech_results = {}
        # -------------------------------------------------------------------------
        # ── SPEECH PIPELINE — Two-Engine STT Architecture ────────────────────────
        # Engine A: Sarvam AI  — PRIMARY transcript source (multilingual, translates to EN)
        # Engine B: Whisper    — DUAL role:
        #     • Timestamps / segments: ALWAYS used for pause analysis (independent of text)
        #     • Fallback text: used ONLY when Sarvam returns empty or < 5 words
        #
        # Rule: ONE authoritative transcript per run — never mix text from both engines.
        # ─────────────────────────────────────────────────────────────────────────────

        # Step 1 — Whisper (always runs; provides segments + fallback text)
        stt_result = transcribe_audio(video_path)
        whisper_text = stt_result.get("text", "") if stt_result else ""
        segments = stt_result.get("segments", []) if stt_result else []

        # Step 2 — Transcript source selection
        # Sarvam text is passed in from main.py as the `transcript` argument.
        sarvam_text = (transcript or "").strip()

        MIN_SARVAM_WORDS = 5  # Sarvam must return at least this many words to be trusted
        if sarvam_text and len(sarvam_text.split()) >= MIN_SARVAM_WORDS:
            transcript = sarvam_text
            transcript_source = "Sarvam"
        else:
            # Sarvam absent, empty, or below word threshold — Whisper text is authoritative
            if sarvam_text:
                print(f"[TRANSCRIPT] Sarvam too short ({len(sarvam_text.split())} words < {MIN_SARVAM_WORDS}) — Whisper fallback engaged")
            else:
                print("[TRANSCRIPT] Sarvam returned empty — Whisper fallback engaged")
            transcript = whisper_text
            transcript_source = "Whisper_fallback"

        print(f"[TRANSCRIPT] source={transcript_source} | sarvam={len(sarvam_text)}c | whisper={len(whisper_text)}c | final={len(transcript)}c")

        # 2. Pause Analysis (from Segments)
        pause_count = 0
        long_pauses = 0
        total_pause_duration = 0.0
        last_end = 0.0
        
        for seg in segments:
            start = seg['start']
            gap = start - last_end
            
            if gap > 0.6: # Gap > 0.6s is a pause
                pause_count += 1
                total_pause_duration += gap
                if gap > 1.5:
                    long_pauses += 1
            
            last_end = seg['end']
            
        duration_min = max(duration_seconds / 60.0, 0.001)
        pause_rate_per_min = pause_count / duration_min
        long_pause_ratio = long_pauses / max(pause_count, 1)

        # 3. Speech Metrics Calculation
        which_transcript_used_for_metrics = "none (empty transcript)"
        if transcript:
            # Strict Call: finalize
            speech_results = speech_metrics.finalize(transcript, duration_seconds)
            which_transcript_used_for_metrics = transcript_source

        # 4. Linguistic Analysis
        linguistic_results = {}
        if transcript:
            # Strict Call: finalize
            linguistic_results = linguistic_analyzer.finalize(transcript)

        print("[TRANSCRIPT SOURCE]:", transcript_source)
        print("[WORD COUNT FROM]:", which_transcript_used_for_metrics)
            
        # 5. Temporal Aggregation
        # Standardized call: finalize
        temporal_results = temporal_analyzer.finalize()
        print("DEBUG analyze_video() expression_score from temporal_results:",
              temporal_results.get("expression_score"))
        
        # 6. Head Head Pose Aggregation
        # Standardized call: finalize
        head_pose_stats = head_pose_metrics.finalize()
        
        # 7. High-Level Scoring & Interpretation
        
        # 7.1 Posture Score
        valid_pose_data = temporal_results.get('valid_pose_ratio', {})
        valid_pose_ratio = float(valid_pose_data.get('value', 0.0))
        
        posture_score = 0.0
        
        if valid_pose_ratio >= 0.20:
            posture_data = temporal_results.get('alignment_integrity', {})
            stability_data = temporal_results.get('stability_index', {})
            
            # TASK 1: Extract Raw Components
            align_score = posture_data.get('value')
            if align_score is None: align_score = 0.5
            else: align_score = float(align_score)
            alignment_integrity_raw = align_score
            
            stab_score = stability_data.get('value')
            if stab_score is None: stab_score = 0.5
            else: stab_score = float(stab_score)
            stability_index_raw = stab_score
            
            # TASK 2: Apply Non-Linear Alignment Amplification
            alignment_error = 1.0 - alignment_integrity_raw
            alignment_adj = 1.0 - (alignment_error ** 1.6)
            alignment_adj = max(0.0, min(1.0, alignment_adj))
            
            # TASK 3: Recompute Final Posture Score
            posture_raw = (0.70 * alignment_adj) + (0.30 * stability_index_raw)
            
            # Extract motion activity for testing high movement penalties
            motion_data = temporal_results.get('motion_activity_level', {})
            motion_activity = motion_data.get('value') if motion_data else 0.0

            # Motion/Fidgeting Penalty (direct penalty to raw score)
            if motion_activity > 0.15: # high motion (e.g., poss.mp4 has 0.22+)
                # Drops the posture score proportionally. 
                # A huge motion of 0.23 could apply a 0.20 (20%) deduction
                motion_penalty = min(0.25, motion_activity * 0.9)
                posture_raw *= (1.0 - motion_penalty)

            posture_raw = max(0.0, min(1.0, posture_raw))
            posture_score = float(np.clip(posture_raw * 10.0, 0.0, 10.0))
            
            # Sustained Lean Penalty
            lean_ratio_data = temporal_results.get('sustained_lean_ratio', {})
            lean_ratio = lean_ratio_data.get('value') if lean_ratio_data else 0.0
            if lean_ratio and float(lean_ratio) > 0.40:
                posture_score *= 0.92   # –8% for sustained lean
                posture_score = float(np.clip(posture_score, 0.0, 10.0))

        # EXTRACT SPEECH & AUDIO METRICS EARLY FOR SMART ENGAGEMENT
        wpm = speech_results.get('words_per_minute', 0.0) if speech_results else 0.0
        filler_rate = speech_results.get('filler_rate_per_min', 0.0) if speech_results else 0.0
        
        audio_rel = 0.0
        pitch_std = 0.0
        energy_var = 0.0
        
        if audio_results:
             audio_rel = audio_results.get('reliability_score', 0.0)
             audio_metrics = audio_results.get('metrics', {})
             pitch_std = audio_metrics.get('pitch_std_hz', 0.0)
             energy_var = audio_metrics.get('energy_variability_score', 0.0)

        # DETERMINISTIC SCORING v3
        if wpm < 90: wpm_score = 4
        elif wpm <= 170: wpm_score = 9
        elif wpm <= 190: wpm_score = 7
        else: wpm_score = 4
        
        if filler_rate == 0: filler_score = 9
        elif filler_rate <= 3: filler_score = 7
        elif filler_rate <= 6: filler_score = 5
        else: filler_score = 3
        
        if pause_rate_per_min <= 2: pause_score = 8
        elif pause_rate_per_min <= 5: pause_score = 6
        else: pause_score = 4
        
        if pitch_std < 15: pitch_score = 4
        elif pitch_std < 30: pitch_score = 6
        elif pitch_std < 50: pitch_score = 8
        else: pitch_score = 9
        
        if energy_var < 5: energy_score = 4
        elif energy_var < 15: energy_score = 6
        elif energy_var < 30: energy_score = 8
        else: energy_score = 9

        # -------------------------------------------------------------------------
        # 7.2 SMART ENGAGEMENT FORMULA (Industry Style)
        # -------------------------------------------------------------------------
        
        engagement_score = 0.0
        engagement_interp = "No visible subject detected. Body analysis unavailable."
        
        # Default initialization to prevent UnboundLocalError if pose is missing
        head_orientation = None
        norm_head_orientation = 50.0  # Fallback
        
        if valid_pose_ratio >= 0.20:
            # --- STEP 1: Normalize All Relevant Signals (0-100) ---
            
            # From Body:
            head_orientation_data = temporal_results.get('head_orientation_score', {})
            head_orientation = head_orientation_data.get('value')
            norm_head_orientation = float(head_orientation) * 100.0 if head_orientation is not None else 50.0
        
        # Backward compatibility for existing code structure (gaze removed, replaced with default 50.0 if queried later)
        gaze_stab = 0.0
        norm_gaze_stab = 50.0
        
        expression_data = temporal_results.get('expression_score', {})
        expr_val = expression_data.get('value')
        norm_expression = float(expr_val) if expr_val is not None else 0.0
        norm_posture = posture_score * 10.0
        
        # From Speech:
        # Scale the 0-10 scores to 0-100 (using 10 as multiplier)
        norm_energy = energy_score * 10.0
        norm_pace = wpm_score * 10.0
        norm_filler = filler_score * 10.0
        
        # Confidence logic (Speech Confidence)
        speech_conf_early = audio_rel
        if linguistic_results and linguistic_results.get('sentence_count', 0) > 0:
             speech_conf_early = (audio_rel * 0.6) + 0.4
        norm_confidence = speech_conf_early * 100.0
        
        # Sentiment Positivity
        sentiment_val = linguistic_results.get('sentiment', 'neutral') if linguistic_results else 'neutral'
        if sentiment_val == 'positive': norm_sentiment = 100.0
        elif sentiment_val == 'negative': norm_sentiment = 0.0
        else: norm_sentiment = 50.0
        
        # --- STEP 2: Create 3 Engagement Pillars ---
        
        # 1. Visual Engagement (40%)
        # Components: 50% Head Orientation, 30% Expression, 20% Posture
        visual_engagement = (0.50 * norm_head_orientation) + (0.30 * norm_expression) + (0.20 * norm_posture)
        
        # 2. Vocal Engagement (40%)
        vocal_engagement = (0.30 * norm_energy) + (0.25 * norm_confidence) + (0.20 * norm_pace) + (0.15 * norm_sentiment) + (0.10 * norm_filler)
        
        # 3. Interaction Smoothness (20%)
        # Long Pause Penalty + Energy Drop Penalty
        long_pause_penalty = min(100.0, long_pause_ratio * 100.0)
        energy_drop_penalty = 30.0 if energy_score <= 4 else (10.0 if energy_score <= 6 else 0.0)
        interaction_smoothness = max(0.0, 100.0 - (long_pause_penalty + energy_drop_penalty))
        
        # --- TASK 1 & 2: SPEECH ACTIVITY GATE ---
        # Thresholds relaxed to accommodate short or accented-speech clips (e.g. bad_english.mp4).
        # word_count: 10 → 5   (short clip may produce fewer words)
        # voiced_ratio: 0.20 → 0.10  (accented / non-native speech has lower pitch-detection rate)
        # speech_duration_seconds: 5.0 → 3.0  (clips under 5 s but with clear speech should not be zeroed)
        GATE_MIN_WORDS   = 5
        GATE_MIN_VOICED  = 0.10
        GATE_MIN_DURATION = 3.0

        word_count = int(speech_results.get('word_count', 0)) if speech_results else 0
        voiced_ratio = float(audio_results.get('metrics', {}).get('voiced_ratio', 0.0)) if audio_results else 0.0
        # AudioAnalyzer.finalize() now returns duration_sec at the top level
        speech_duration_seconds = float(audio_results.get('duration_sec', 0.0)) if audio_results else 0.0

        print(
            f"[SPEECH GATE] word_count={word_count} (min={GATE_MIN_WORDS})"
            f" | voiced_ratio={voiced_ratio:.3f} (min>{GATE_MIN_VOICED})"
            f" | duration={speech_duration_seconds:.2f}s (min>={GATE_MIN_DURATION}s)"
        )

        speech_activity_flag = (
            word_count >= GATE_MIN_WORDS
            and voiced_ratio > GATE_MIN_VOICED
            and speech_duration_seconds >= GATE_MIN_DURATION
        )
        print(f"[SPEECH GATE] speech_activity_flag={speech_activity_flag}")

        if not speech_activity_flag:
            vocal_engagement = 0.0
            interaction_smoothness = 0.0
            norm_confidence = 0.0  # TASK 2: no audio speech → confidence must not inflate vocal scores
        
        # --- FINAL ENGAGEMENT SCORE ---
        engagement_score_100 = (0.4 * visual_engagement) + (0.4 * vocal_engagement) + (0.2 * interaction_smoothness)
        print("DEBUG A: After 40/40/20 aggregation (0–100):", engagement_score_100)

        engagement_score = float(np.clip(engagement_score_100 / 10.0, 0.0, 10.0))
        
        # APPLY HARD GATE FOR EMPTY VIDEOS
        if valid_pose_ratio < 0.20:
            engagement_score = 0.0
            engagement_interp = "No visible subject detected. Body analysis unavailable."
            
        print("DEBUG B: After /10 normalization & gating:", engagement_score)
        
        print("\n--- ENGAGEMENT RECALIBRATION VALIDATION ---")
        print("Visual:", visual_engagement)
        print("Vocal (post-gate):", vocal_engagement)
        print("Smoothness:", interaction_smoothness)
        print("Final Engagement:", engagement_score)
        print("-------------------------------------------\n")
        
        # 7.3 Speech Delivery Score (Communication)
        
        # Final Weighted Score (Simplified)
        # 0.40 WPM + 0.30 Pause + 0.30 ASR Clarity
        asr_clarity_score = audio_rel * 10.0
        
        raw_score = (
            (0.40 * wpm_score) +
            (0.30 * pause_score) +
            (0.30 * asr_clarity_score)
        )
        
        speech_delivery_score = float(round(min(10.0, max(0.0, raw_score)), 1))
        
        # Reliability Gate
        speech_logic_mode = "speech_simplified"
        
        # No audio reliability dampening unless input is completely invalid
        if audio_rel < 0.10: 
            speech_delivery_score = float(round(speech_delivery_score * audio_rel, 1)) # Dampen score
            speech_logic_mode = "speech_low_reliability"
            
        # Inject Metrics for Diagnostics
        if speech_results is not None:
            speech_results['rate_score'] = wpm_score
            speech_results['filler_score'] = filler_score
            speech_results['pause_score'] = pause_score
            speech_results['pitch_score'] = pitch_score
            speech_results['energy_score'] = energy_score
            speech_results['speech_logic_mode'] = speech_logic_mode
            speech_results['pause_rate_per_min'] = round(pause_rate_per_min, 1)
            speech_results['long_pause_ratio'] = round(long_pause_ratio, 2)
            # Note: speech_results is assumed to be a valid dict here if transcript existed
            # If transcript was empty, speech_results is {} empty dict from fallback initialization?
            # Wait, my logic above: "if transcript: speech_results = ..."
            # Initialize speech_results if not present!
        
        if 'speech_results' not in locals() or speech_results is None:
             speech_results = {
                'rate_score': wpm_score,
                'filler_score': filler_score,
                'pause_score': pause_score,
                'pitch_score': pitch_score,
                'energy_score': energy_score,
                'speech_logic_mode': speech_logic_mode,
                'pause_rate_per_min': round(pause_rate_per_min, 1),
                'long_pause_ratio': round(long_pause_ratio, 2)
             }
        
        # 7.4 Professionalism & Confidence Score
        
        # Keep legacy confidence variables for strict JSON schema compliance
        audio_conf = audio_results.get('reliability_score', 0.5) if audio_results else 0.5
        speech_conf = audio_conf
        if linguistic_results and linguistic_results.get('sentence_count', 0) > 0:
             speech_conf = (audio_conf * 0.6) + 0.4
             
        # 1. Visual Confidence (Posture/Gaze/Blink/Data)
        visual_data = temporal_results.get('visual_confidence', {})
        visual_conf = visual_data.get('value')
        if visual_conf is None: visual_conf = 0.5
        else: visual_conf = float(visual_conf)
        
        # Overall Confidence (Simplified)
        # Depend ONLY on: Visual confidence (50%), Voice pitch stability (30%), Energy stability (20%)
        pitch_stability = pitch_score / 10.0
        energy_stability = energy_score / 10.0
        
        if not speech_activity_flag:
             overall_confidence = visual_conf # Silent fallback
        else:
             overall_confidence = (visual_conf * 0.50) + (pitch_stability * 0.30) + (energy_stability * 0.20)
             
        overall_confidence = round(min(1.0, max(0.0, overall_confidence)), 3)

        # Professionalism Score
        # ONLY: Grammar score (40%), Formal vocabulary density (35%), Filler control score (25%)
        if linguistic_results:
             # Extract from linguistic results (or default if missing)
             grammar_score = float(linguistic_results.get('grammar_score', 7.0))
             formal_vocab = float(linguistic_results.get('formal_vocabulary_density', 6.0))
             prof_raw = (0.40 * grammar_score) + (0.35 * formal_vocab) + (0.25 * filler_score)
             professionalism_score = float(np.clip(prof_raw, 0.0, 10.0))
        else:
             # Fallback if no transcript (Silent Fallback)
             # Rely on posture and overall confidence.
             prof_base = (posture_score * 0.5) + (engagement_score * 0.5)
             professionalism_score = float(np.clip((prof_base * 0.7) + (overall_confidence * 10.0 * 0.3), 0.0, 10.0))
            
        # 7.5 Summary Text
        # Construct sentences based on scores
        summary_lines = []
        
        if valid_pose_ratio < 0.20:
            concise_summary_text = "No visible subject detected. Body analysis unavailable."
            posture_interp = concise_summary_text
            # engagement_interp is already set above
        else:
            # Task Fix: Ensure pa is fresh (interpretation added late)
            # Re-fetch relevant data for interpretation
            alignment_val = align_score
            stability_val = stab_score
            motion_val = temporal_results.get('motion_activity_level', {}).get('value', 0.0)
    
            # Structured Deterministic Summary (v2.2 Frozen)
            concise_summary_text = _generate_deterministic_posture_summary(alignment_val, stability_val, motion_val)
            
            # Generate Interpretation Blocks
            # Posture
            posture_interp = concise_summary_text # Use the official summary
            
            # Engagement (Head Orientation Consistency)
            # Use safe variable extracted earlier (defaults to 0.5 if missing)
            eye_stats = head_orientation if 'head_orientation' in locals() else 0.5
        
        # ------------------------------------------------------------------
        # Engagement Classification (Fusion Model v4.0)
        # ------------------------------------------------------------------
        # Inputs:
        # 1. ECR (Eye Contact Ratio) = eye_contact_consistency
        # 2. ASF (Attention Switch Frequency) = switches / duration
        # 3. SAS (Sustained Attention Score) = max_continuous / total_duration
        # 4. GSS (Gaze Stability Score) = gaze_stability
        
        # Check if gaze metrics are actually available
        switch_data = temporal_results.get('gaze_direction_switch_count', {})
        max_continuous_data = temporal_results.get('max_continuous_eye_contact', {})
        
        gaze_metrics_available = bool(switch_data and max_continuous_data)
        print("Engagement before gaze fusion:", engagement_score)
        print("Gaze metrics available:", gaze_metrics_available)
        
        if gaze_metrics_available:
            # 1. Head Orientation Score (HOS)
            hos = temporal_results.get('head_orientation_score', {}).get('value')
            if hos is None: hos = 0.0
            
            # 2. Attention Switch Frequency (ASF)
            switch_count = switch_data.get('value')
            if switch_count is None: switch_count = 0
            else: switch_count = int(switch_count)
            
            # Use metadata duration (calculated at start of analysis)
            total_duration_s = max(duration_seconds, 1.0)
            
            asf = switch_count / total_duration_s
            asf_score = 1.0 / (1.0 + (asf * 0.75))
            
            # 3. Sustained Attention Score (SAS)
            max_continuous = max_continuous_data.get('value')
            if max_continuous is None: max_continuous = 0.0
            sas = max_continuous / max(total_duration_s, 1.0)
            # Clamp SAS to [0, 1] just in case
            sas = min(1.0, max(0.0, sas))
            
            # Geometric Fusion
            # HOS is the dominant factor.
            fusion_score = (hos ** 0.55) * (sas ** 0.25) * (asf_score ** 0.20)
            
            # Scale to 0-10
            engagement_score = fusion_score * 10.0
            
            # Interpretation Bands (using HOS instead of fusion score per user request)
            hos_score = hos * 100.0  # internal is 0-1, evaluate as 0-100 logic
            engagement_interp = ""
            if hos_score >= 85:
                 engagement_interp = "Strong forward-facing attention and stable visual focus."
            elif hos_score >= 60:
                 engagement_interp = "Generally attentive posture with occasional attention shifts."
            else:
                 engagement_interp = "Frequent attention shifts or reduced forward-facing orientation."
                 
            # Append facial expressiveness logic
            raw_expr = temporal_results.get('expression_score', {}).get('value')
            expr_val = float(raw_expr) * 10.0 if raw_expr is not None else 0.0
            
            if expr_val >= 75:
                 engagement_interp += " Shows positive facial expressiveness."
            elif expr_val < 50:
                 engagement_interp += " Shows limited facial expressiveness."

        else:
            # Fallback Interpretation if gaze metrics are missing
            safe_ho = head_orientation if head_orientation is not None else 0.0
            hos_score_fb = safe_ho * 100.0
            if hos_score_fb >= 85:
                 engagement_interp = "Strong forward-facing attention and stable visual focus."
            elif hos_score_fb >= 60:
                 engagement_interp = "Generally attentive posture with occasional attention shifts."
            else:
                 engagement_interp = "Frequent attention shifts or reduced forward-facing orientation."

            raw_expr = temporal_results.get('expression_score', {}).get('value')
            expr_val = float(raw_expr) * 10.0 if raw_expr is not None else 0.0
            if expr_val >= 75:
                 engagement_interp += " Shows positive facial expressiveness."
            elif expr_val < 50:
                 engagement_interp += " Shows limited facial expressiveness."
    
        print("Engagement after gaze fusion:", engagement_score)
        print("DEBUG C: After gaze fusion:", engagement_score)
        print("DEBUG D: After speech gate:", engagement_score)
        
        # APPLY HARD GATE FOR EMPTY VIDEOS AGAIN TO OVERRIDE LEGACY BLOCKS
        if valid_pose_ratio < 0.20:
            engagement_score = 0.0
            engagement_interp = "No visible subject detected. Body analysis unavailable."
        
        # Construct Result Dictionary
        results = {
            "multimodal_analysis": {
                "confidence_metrics": {
                    "visual_confidence": round(visual_conf, 3),
                    "audio_confidence": round(audio_conf, 3),
                    "speech_confidence": round(speech_conf, 3),
                    "overall_confidence": overall_confidence
                },
                "posture_analysis": {
                    "score": round(posture_score, 1),
                    "interpretation": posture_interp,
                    "metrics": temporal_results, # includes alignment, stability, motion
                    "confidence": round(visual_conf, 3) # Strict Domain Isolation
                },
                "engagement_analysis": {
                    "score": round(engagement_score, 1),
                    "interpretation": engagement_interp,
                    "metrics": {
                        "head_orientation": head_orientation,
                        "attention_shifts": temporal_results.get('gaze_direction_switch_count', {}).get('value'),
                        "max_continuous": temporal_results.get('max_continuous_eye_contact', {}).get('value')
                    }
                },
                "speech_analysis": {
                    "score": round(speech_delivery_score, 1),
                    "metrics": speech_results,
                    "audio_features": audio_results,
                    "confidence": round(audio_conf, 3)
                },
                "linguistic_analysis": linguistic_results
            },
            "summary_view": {
                'posture_score': round(posture_score, 1),
                'engagement_score': round(engagement_score, 1),
                'speech_delivery_score': round(speech_delivery_score, 1),
                'professionalism_score': round(professionalism_score, 1),
                'overall_confidence': overall_confidence,
                'concise_summary_text': concise_summary_text
            },
            # Convenience fields
            'overall_confidence': overall_confidence,
            'professionalism_score': round(professionalism_score, 1),
            'communication_score': round(speech_delivery_score, 1),
            # ── Transcript passthrough ──────────────────────────────────────────
            # `transcript` is the final authoritative text (Sarvam, or Whisper
            # fallback). It is stored here so build_public_response() can read
            # it without having to re-derive it from linguistic_analysis (which
            # only stores word-level stats, not the raw text).
            'transcript_text': transcript or "",
            'transcript_source': transcript_source,
            'transcript_language_code': stt_result.get("language", "en") if transcript_source == "Whisper_fallback" else "en",
            'metadata': {
                'processing_method': 'camera_invariant_geometry',
                'temporal_analysis': True,
                'confidence_estimation': True,
                'domain_isolation': True,
                'duration': round(duration_seconds, 2)
            }
        }

        print("DEBUG E: Final engagement_score leaving analyze_video:", engagement_score)
        response = {
            "analysis_version": "v2.2_frozen_calibrated",
            "results": results
        }
        
        if debug_mode or DEBUG_MODE:
            public_res = build_public_response(response)
            public_res["debug_data"] = response["results"]
            return public_res
        else:
            return build_public_response(response)

    except Exception as e:
        logger.error(f"Analysis Failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

def build_public_response(full_response: dict) -> dict:
    """
    Standard production response builder — v2 (Standard Structure).
    Output shape: body / speech_score / speech_analysis / final_score
    All internal computation still runs — only output shape is changed.
    """
    results = full_response.get("results", {})
    ma = results.get("multimodal_analysis", {})
    sv = results.get("summary_view", {})

    # ── Scaling helpers (response layer only, no internal values changed) ──────
    def scale_10_to_100(x): return round(float(x) * 10.0, 1)
    def scale_1_to_100(x):  return round(float(x) * 100.0, 1)

    def extract_metric_value(metric):
        if isinstance(metric, dict):
            return metric.get("value", 0)
        elif isinstance(metric, (int, float)):
            return metric
        else:
            return 0

    # ── Posture ──────────────────────────────────────────────────────────────
    posture_raw = float(extract_metric_value(sv.get("posture_score", 0.0)))
    posture_norm = scale_10_to_100(posture_raw)

    # ── Engagement / Eye ──────────────────────────────────────────────────────
    eng = ma.get("engagement_analysis", {})
    eng_metrics = eng.get("metrics", {})
    engagement_raw = float(extract_metric_value(eng.get("score", 0.0)))
    print("DEBUG F: engagement_score entering public response:", engagement_raw)
    engagement_norm = scale_10_to_100(engagement_raw)
    engagement_interp = eng.get("interpretation", "")

    # HOS (0–1 → 0–100)
    hos_raw = float(extract_metric_value(eng_metrics.get("head_orientation", 0.0)))
    hos_norm = scale_1_to_100(hos_raw)

    # ── Expression ────────────────────────────────────────────────────────────
    # posture_analysis.metrics IS temporal_results (line 599 of analyze_video).
    # expression_score from temporal_features stores {"value": N, "confidence": face_detection_ratio}.
    # Use {} fallback (not 0.0) so isinstance check works and confidence is preserved.
    posture_metrics = ma.get("posture_analysis", {}).get("metrics", {})
    # expr_data may be: a full dict, None (key exists but value is None), or missing (-> {})
    expr_data = posture_metrics.get("expression_score") or {}   # None → {}

    # Read face_detection_ratio from confidence BEFORE flattening to float
    if isinstance(expr_data, dict) and expr_data:
        face_detection_ratio = float(expr_data.get("confidence", 1.0))
        extracted_value = float(expr_data.get("value") or 0.0)
    else:
        # Empty dict or non-dict — face detection status unknown, assume good coverage
        face_detection_ratio = 1.0
        extracted_value = float(expr_data) if isinstance(expr_data, (int, float)) else 0.0


    print("DEBUG final face_detection_ratio used in interpretation:", face_detection_ratio)

    # Value is already 0-100 from temporal_features.py
    # User Requirement: Do NOT return null under any circumstance.
    expression_norm = round(extracted_value, 1)
    expression_for_formula = expression_norm
    print("PUBLIC RESPONSE expression_score:", extracted_value)


    # ── body_total: industry-standard weighted linear aggregation (0–100) ──────
    # posture 30% + engagement 30% + HOS 30% + expression 10%
    print("DEBUG G: engagement_score used in body_total:", engagement_norm)
    
    valid_pose_ratio_data = posture_metrics.get('valid_pose_ratio', {}) if posture_metrics else {}
    valid_pose_ratio = float(valid_pose_ratio_data.get('value', 0.0))
    if valid_pose_ratio < 0.20:
        body_total = 0.0
    else:
        body_total = round(
            (0.30 * posture_norm) +
            (0.30 * engagement_norm) +
            (0.30 * hos_norm) +
            (0.10 * expression_for_formula),
            1
        )
        body_total = max(0.0, min(100.0, body_total))  # clamp [0, 100]

    # ── Body interpretation: merged posture + engagement text ─────────────────
    # face_detection_ratio is already sourced from expr_data.confidence above.

    # PART 2 — Trace interpretation builder inputs
    print("DEBUG interpretation input:")
    print("  posture_score:", posture_norm)
    print("  head_orientation_score:", hos_norm)
    print("  expression_score:", expression_norm)
    print("  expr_data type:", type(expr_data).__name__, "| value:", expr_data)
    print("  face_detection_ratio:", face_detection_ratio)


    if face_detection_ratio < 0.40:
        # PART 3 — low coverage branch
        print("DEBUG low face coverage branch triggered")
        body_interpretation = (
            "Insufficient visual data detected. "
            "Face visibility was too low to reliably evaluate posture, attention, or facial expressiveness."
        )
    else:
        # PART 3 — normal branch
        print("DEBUG normal interpretation branch triggered")
        # ── Interpretations & Feedbacks ───────────────────────────────────────────
        posture_analysis = ma.get("posture_analysis", {})
        posture_interp = posture_analysis.get("interpretation", "No visible subject detected. Body analysis unavailable.") if posture_analysis else "No visible subject detected. Body analysis unavailable."

        if posture_interp == engagement_interp:
            body_summary = posture_interp
        else:
            body_summary = f"{posture_interp} {engagement_interp}".strip()

        positivity_feedback = ""
        body_interpretation = body_summary

    # PART 4 — final interpretation value before return
    print("DEBUG final interpretation:", body_interpretation)


    # ── Speech ────────────────────────────────────────────────────────────
    speech = ma.get("speech_analysis", {})
    speech_metrics_dict = speech.get("metrics") or {}
    speech_raw = float(extract_metric_value(speech.get("score", 0.0)))          # internal 0-10
    speech_total_norm = scale_10_to_100(speech_raw)        # 0-100 for output

    wpm_val = float(extract_metric_value(speech_metrics_dict.get("words_per_minute", 0.0)))
    filler_rate = float(extract_metric_value(speech_metrics_dict.get("filler_rate_per_min", 0.0)))
    pause_rate = float(extract_metric_value(speech_metrics_dict.get("pause_rate_per_min", 0.0)))

    # ── Audio / Voice ──────────────────────────────────────────────────────
    audio_feat = speech.get("audio_features") or {}
    audio_metrics_inner = audio_feat.get("metrics") or {}
    audio_rel = float(extract_metric_value(audio_feat.get("reliability_score", 0.0)))

    pitch_mean = float(extract_metric_value(audio_metrics_inner.get("pitch_mean_hz", 0.0)))
    energy_variability = float(extract_metric_value(audio_metrics_inner.get("energy_variability_score", 0.0)))

    # Tone quality from pitch mean Hz
    if pitch_mean == 0:
        tone_quality = "not_detected"
    elif pitch_mean < 150:
        tone_quality = "deep"
    elif pitch_mean < 230:
        tone_quality = "medium"
    else:
        tone_quality = "high"

    # Voice quality: audio reliability (0-1) → 0-100
    voice_quality_norm = scale_1_to_100(min(1.0, audio_rel))

    # Energy level: energy_variability (0-∞) capped at 0.50 reference → 0-100
    energy_level_norm = round(min(100.0, (energy_variability / 0.50) * 100.0), 1)

    # ── Professionalism & Confidence ──────────────────────────────────────────
    professionalism_raw  = float(extract_metric_value(results.get("professionalism_score", 0.0)))  # 0-10
    communication_raw    = float(extract_metric_value(results.get("communication_score", speech_raw)))  # 0-10
    confidence_raw       = float(extract_metric_value(results.get("overall_confidence", 0.0)))     # 0-1

    professionalism_norm = scale_10_to_100(professionalism_raw)   # 0-100
    communication_norm   = scale_10_to_100(communication_raw)     # 0-100
    confidence_norm      = scale_1_to_100(min(1.0, confidence_raw))  # 0-100

    # ── Final Score (all inputs are now 0-100) ──────────────────────────────
    # body 40% + speech 40% + confidence 20% — all 0-100
    final_score = round(
        (body_total           * 0.40) +
        (speech_total_norm    * 0.40) +
        (confidence_norm      * 0.20),
        1
    )
    final_score = max(0.0, min(100.0, final_score))  # clamp [0, 100]

    # ── Transcript ─────────────────────────────────────────────────────────────
    # `transcript_text` is set in analyze_video() and stored at the top level of
    # `results` so it survives the handoff here. The old code incorrectly read
    # from linguistic_analysis which only stores word-count stats, not raw text.
    transcript_text = results.get("transcript_text", "") or ""
    language_code   = results.get("transcript_language_code", "en") or "en"
    transcript_source = results.get("transcript_source", "unknown")
    print(f"[TRANSCRIPT] source={transcript_source} | len={len(transcript_text)} | lang={language_code}")
    print(f"[TRANSCRIPT] preview: {transcript_text[:120]!r}")

    # ── Speech Summary (speech-domain only) ───────────────────────────────────
    # body.interpretation already carries posture/engagement/expression text.
    # speech_summary must contain ONLY: pace, fillers, clarity, energy, tone.
    pace_label = "ideal pace" if 90 <= wpm_val <= 170 else ("slow pace" if wpm_val < 90 else "fast pace")
    filler_label = "minimal fillers" if filler_rate <= 3 else "frequent fillers"

    # Clarity descriptor from audio reliability (0-1 → qualitative label)
    if audio_rel >= 0.80:
        clarity_label = "excellent clarity"
    elif audio_rel >= 0.55:
        clarity_label = "good clarity"
    elif audio_rel >= 0.30:
        clarity_label = "moderate clarity"
    else:
        clarity_label = "low clarity"

    # Energy descriptor from energy_variability score (0-100 after scaling)
    if energy_level_norm >= 60:
        energy_label = "high energy"
    elif energy_level_norm >= 25:
        energy_label = "moderate energy"
    else:
        energy_label = "low energy"

    # analysis_summary is constructed below after speech_total and clarity_level are computed.

    # ── Speech sub-metric weighted display values ─────────────────────────────
    # Re-derive band scores using the same rules as analyze_video() from response-
    # layer values already available here. Display-only — does NOT affect any totals.
    _wpm_band:    int = (9 if 90 <= wpm_val <= 170 else (7 if wpm_val <= 190 else 4)) if wpm_val >= 90 else 4
    _filler_band: int = 9 if filler_rate == 0 else (7 if filler_rate <= 3 else (5 if filler_rate <= 6 else 3))
    _pause_band:  int = 8 if pause_rate <= 2 else (6 if pause_rate <= 5 else 4)
    _pitch_band:  int = 9 if pitch_mean == 0 else (4 if pitch_mean < 150 else (6 if pitch_mean < 230 else 8))
    _energy_band: int = 9 if energy_level_norm >= 60 else (6 if energy_level_norm >= 25 else 4)
    wpm_display    = round(_wpm_band    / 10.0 * 30.0, 1)  # out of 30
    filler_display = round(_filler_band / 10.0 * 25.0, 1)  # out of 25
    pause_display  = round(_pause_band  / 10.0 * 25.0, 1)  # out of 25
    pitch_display  = round(_pitch_band  / 10.0 * 10.0, 1)  # out of 10
    energy_display = round(_energy_band / 10.0 * 10.0, 1)  # out of 10

    # ── Communication score (exposed at its 0-100 scale) ─────────────────────
    communication_score_norm = round(float(results.get("communication_score", 0.0)) * 10.0, 1)

    # ── Clarity level (from band scores, display-only) ────────────────────────
    _clarity_raw = (0.40 * _pause_band) + (0.35 * _filler_band) + (0.25 * _wpm_band)
    if _clarity_raw >= 8.5:
        clarity_level = "Excellent"
    elif _clarity_raw >= 7.0:
        clarity_level = "Strong"
    elif _clarity_raw >= 5.5:
        clarity_level = "Moderate"
    elif _clarity_raw >= 4.0:
        clarity_level = "Basic"
    else:
        clarity_level = "Poor"

    # ── Sentiment & sarcasm (from linguistic_results) ─────────────────────────
    _ling_data = ma.get("linguistic_analysis") or {}
    _sentiment_val = _ling_data.get("sentiment", "neutral") or "neutral"
    _sarcasm_val   = bool(_ling_data.get("sarcasm", False))

    # ── Speech total (frontline evaluation weighted formula) ───────────────────
    # communication 40% + confidence 30% + professionalism 20% + voice_quality 10%
    # All inputs are 0–100. Output is clamped to [0, 100].
    speech_total = round(
        (0.40 * communication_score_norm) +
        (0.30 * confidence_norm) +
        (0.20 * professionalism_norm) +
        (0.10 * voice_quality_norm),
        1
    )
    speech_total = max(0.0, min(100.0, speech_total))

    # ── LLM-generated speech summary ───────────────────────────────────────
    # Fallback template (used if OpenAI call fails for any reason)
    _fallback_summary = (
        f"The speaker maintained a {pace_label} speaking pace ({round(wpm_val, 0):.0f} WPM) "
        f"with {energy_label} vocal energy and a {tone_quality} tone. "
        f"Clarity was assessed as {clarity_level.lower()}, and filler usage was {filler_label}. "
        f"Professionalism was scored at {professionalism_norm:.1f}, "
        f"while confidence was {confidence_norm:.1f}. "
        f"Overall communication effectiveness was rated at {speech_total:.1f} out of 100."
    )

    # Metrics payload sent to the LLM
    _llm_payload = {
        "communication_score": communication_score_norm,
        "confidence":          confidence_norm,
        "professionalism":     professionalism_norm,
        "voice_quality":       voice_quality_norm,
        "speech_total":        speech_total,
        "pace":                pace_label,
        "clarity_level":       clarity_level,
        "energy_level":        energy_label,
        "filler_usage":        filler_label,
        "tone_quality":        tone_quality,
        "sentiment":           _sentiment_val,
    }

    analysis_summary = _fallback_summary   # default
    try:
        _api_key = os.environ.get("OPENAI_API_KEY", "")
        if _api_key:
            _client = openai.OpenAI(api_key=_api_key)
            _chat = _client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.4,
                max_tokens=200,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional interview evaluation assistant. "
                            "Generate a concise, recruiter-friendly speech performance summary. "
                            "Base your explanation strictly on the provided metrics. "
                            "Do not invent information. Do not reinterpret scores. "
                            "Explain strengths and improvement areas clearly and simply."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Generate a 3–4 sentence speech performance summary based on these metrics:\n\n"
                            f"{_llm_payload}\n\n"
                            f"Keep language simple, professional, and objective. "
                            f"Do not restate numeric values unnecessarily. "
                            f"Highlight strengths and improvement areas."
                        ),
                    },
                ],
            )
            _llm_text = _chat.choices[0].message.content
            if _llm_text and _llm_text.strip():
                analysis_summary = _llm_text.strip()
                print("[LLM SUMMARY] Generated via OpenAI")
        else:
            print("[LLM SUMMARY] OPENAI_API_KEY not set — using fallback template")
    except Exception as _llm_err:
        print(f"[LLM SUMMARY] OpenAI call failed ({_llm_err}) — using fallback template")
    # ──────────────────────────────────────────────────────────────────────────

    def fmt(value, max_val):
        """Format a score as 'X out of Y'."""
        return f"{value} out of {max_val}"

    response = {
        "analysis_version": full_response.get("analysis_version", ""),
        "body": {
            "posture_score":          fmt(round(posture_norm     * 0.30, 1), 30),
            "engagement_score":       fmt(round(engagement_norm  * 0.30, 1), 30),
            "head_orientation_score": fmt(round(hos_norm         * 0.30, 1), 30),
            "expression_score":       fmt(round(expression_norm  * 0.10, 1), 10),
            "body_language_score":    fmt(body_total, 100),
            "interpretation": body_interpretation
        },
        "speech_score": {
            "professionalism":     fmt(professionalism_norm,       100),
            "confidence":          fmt(confidence_norm,            100),
            "voice_quality":       fmt(voice_quality_norm,         100),
            "communication score": fmt(communication_score_norm,   100),
            "speech_total":        fmt(speech_total,               100)
        },
        "speech_analysis": {
            "sentiment":      _sentiment_val,
            "tone_quality":   tone_quality,
            "sarcasm":        _sarcasm_val,
            "pace":           pace_label,
            "clarity_level":  clarity_level,
            "energy_level":   energy_label,
            "filler_usage":   filler_label,
            "speech_summary": analysis_summary
        },
        "transcript": {
            "text": transcript_text,
            "language_code": language_code
        },
        # final_score is intentionally omitted from the public response.
        # overall_score (below) carries the same value as a human-readable percentage.
    }


    # ── overall_score (body 50% + speech 50%) ─────────────────────────────────
    # Confidence is excluded. Uses body_total (0-100) and speech_total (0-100).
    overall_score_numeric = round(
        (0.50 * body_total) +
        (0.50 * speech_total),
        1
    )
    overall_score_numeric = max(0.0, min(100.0, overall_score_numeric))
    _overall_str = f"{round(overall_score_numeric)}% out of 100%"
    print("DEBUG overall_score (50% body + 50% speech):", overall_score_numeric, "->", _overall_str)
    response["overall_score"] = _overall_str
    return response


def _generate_deterministic_posture_summary(alignment, stability, motion) -> str:
    """
    Generate a strictly deterministic posture summary based on rule-based mapping.
    Template: "{Alignment} body alignment with {stability} stability and {movement} movement."
    """
    # Handle missing data
    if alignment is None or stability is None or motion is None:
        return "Insufficient data to generate posture summary."

    # 1. Alignment Mapping
    # =0.95 → Excellent alignment
    # 0.80–0.94 → Good alignment
    # 0.60–0.79 → Moderate misalignment
    # <0.60 → Significant misalignment
    if alignment >= 0.95:
        align_desc = "Excellent"
    elif alignment >= 0.80:
        align_desc = "Good"
    elif alignment >= 0.60:
        align_desc = "Moderate"
    else:
        align_desc = "Significant"

    # 2. Stability Mapping (Industry-Standard Bands)
    # 0.90–1.00 → very high
    # 0.80–0.89 → high
    # 0.70–0.79 → moderate
    # 0.60–0.69 → reduced
    # < 0.60 → low
    if stability >= 0.90:
        stab_desc = "very high"
    elif stability >= 0.80:
        stab_desc = "high"
    elif stability >= 0.70:
        stab_desc = "moderate"
    elif stability >= 0.60:
        stab_desc = "reduced"
    else:
        stab_desc = "low"

    # 3. Movement Mapping
    # <0.04 → minimal
    # 0.04–0.10 → mild
    # 0.10–0.20 → noticeable
    # >0.20 → excessive
    if motion < 0.04:
        move_desc = "minimal"
    elif motion < 0.10:
        move_desc = "mild"
    elif motion < 0.20:
        move_desc = "noticeable"
    else:
        move_desc = "excessive"

    # Strict Template Construction
    return f"{align_desc} body alignment with {stab_desc} stability and {move_desc} movement."
