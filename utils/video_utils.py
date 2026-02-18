import cv2
import logging
import librosa
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# Frozen Posture Engine Version
POSTURE_ENGINE_VERSION = "v2.2_frozen_calibrated"

import traceback
from features.normalized_geometry import NormalizedGeometry
from features.temporal_features import TemporalFeatures
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
        if transcript:
            # Strict Call: finalize
            speech_results = speech_metrics.finalize(transcript, duration_seconds)
            
        # 4. Linguistic Analysis
        linguistic_results = {}
        if transcript:
            # Strict Call: finalize
            linguistic_results = linguistic_analyzer.finalize(transcript)
            
        # 5. Temporal Aggregation
        # Standardized call: finalize
        temporal_results = temporal_analyzer.finalize()
        
        # 6. Head Pose Aggregation
        # Standardized call: finalize
        head_pose_stats = head_pose_metrics.finalize()
        
        # 7. High-Level Scoring & Interpretation
        
        # 7.1 Posture Score
        # Driven by stability_index and alignment_integrity
        # range 0-10
        
        posture_data = temporal_results.get('alignment_integrity', {})
        stability_data = temporal_results.get('stability_index', {})
        
        # Safe extraction: handle None via get_score_val or explicit check
        align_score = posture_data.get('value')
        if align_score is None: align_score = 0.5
        else: align_score = float(align_score)
        
        stab_score = stability_data.get('value')
        if stab_score is None: stab_score = 0.5
        else: stab_score = float(stab_score)
        
        # 60% Alignment, 40% Stability
        posture_score = (align_score * 6.0) + (stab_score * 4.0)
        
        # 7.2 Engagement Score
        # Driven by eye contact, gaze stability, smile
        eye_contact_data = temporal_results.get('eye_contact_consistency', {})
        eye_contact = eye_contact_data.get('value')
        if eye_contact is None: eye_contact = 0.5
        else: eye_contact = float(eye_contact)
        
        gaze_data = temporal_results.get('gaze_stability', {})
        gaze_stab = gaze_data.get('value')
        if gaze_stab is None: gaze_stab = 0.5
        else: gaze_stab = float(gaze_stab)
        
        smile_data = temporal_results.get('smile_intensity', {})
        smile = smile_data.get('value')
        if smile is None: smile = 0.0
        else: smile = float(smile)
        
        engagement_score = (eye_contact * 4.0) + (gaze_stab * 3.0) + (smile * 3.0)
        
        # 7.3 Speech Delivery Score (Communication)
        # Driven by wpm (pace), filler words, pitch variety
        # Default fallback
        speech_delivery_score = 5.0
        # If we had real metrics:
        # speech_delivery_score = ...
        
        # 7.4 Professionalism Score
        # Combination of all + confidence + linguistic quality
        
        # --- Domain Isolation (v3.0) ---
        # 1. Visual Confidence (Posture/Gaze/Blink/Data)
        visual_data = temporal_results.get('visual_confidence', {})
        visual_conf = visual_data.get('value')
        if visual_conf is None: visual_conf = 0.5
        else: visual_conf = float(visual_conf)
        
        # 2. Audio Confidence (Reliability)
        audio_conf = audio_results.get('reliability_score', 0.5)
        
        # 3. Speech Confidence (Audio + Linguistic Validity)
        # Speech is confident if audio is reliable AND we found intelligible sentences
        speech_conf = audio_conf
        if linguistic_results:
             sent_count = linguistic_results.get('sentence_count', 0)
             if sent_count > 0:
                 speech_conf = (audio_conf * 0.6) + 0.4 # Bonus for intelligible text
        
        # 4. Overall Confidence (Multimodal Fusion)
        # Visual 50%, Audio 30%, Speech 20%
        overall_confidence = (visual_conf * 0.5) + (audio_conf * 0.3) + (speech_conf * 0.2)
        overall_confidence = round(overall_confidence, 3)

        
        prof_base = (posture_score * 0.4) + (engagement_score * 0.3) + (speech_delivery_score * 0.3)
        
        # Linguistic modifier
        ling_factor = 0.0
        if linguistic_results:
             # Basic sentiment/quality mapping
             ling_factor = linguistic_results.get('professionalism_rating', 5.0)
             
        # Audio modifier (reliability)
        audio_rel = audio_results.get('reliability_score', 0.5)
        
        # English bonus
        if linguistic_results.get('is_english_analysis'):
            bonus = 0.0
            if linguistic_results.get('star_structure_flag'): bonus += 1.0
            if linguistic_results.get('example_usage_flag'): bonus += 0.5
            ling_factor = min(10.0, ling_factor + bonus)
            
        # Combine (Audio 40%, Linguistic 40%, Confidence 20%)
        if linguistic_results:
            professionalism_score = (prof_base * 0.4) + (ling_factor * 0.4) + (overall_confidence * 10 * 0.2)
        else:
            # Fallback if no transcript
            professionalism_score = (prof_base * 0.7) + (overall_confidence * 10 * 0.3)
            
        # 7.5 Summary Text
        # Construct sentences based on scores
        summary_lines = []
        
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
        
        # Engagement (Eye Contact Consistency)
        # Use safe variable extracted earlier (defaults to 0.5 if missing)
        eye_stats = eye_contact
        
        # ------------------------------------------------------------------
        # Engagement Classification (Fusion Model v4.0)
        # ------------------------------------------------------------------
        # Inputs:
        # 1. ECR (Eye Contact Ratio) = eye_contact_consistency
        # 2. ASF (Attention Switch Frequency) = switches / duration
        # 3. SAS (Sustained Attention Score) = max_continuous / total_duration
        # 4. GSS (Gaze Stability Score) = gaze_stability
        
        # 1. Eye Contact Ratio (ECR)
        ecr = temporal_results.get('eye_contact_consistency', {}).get('value')
        if ecr is None: ecr = 0.0
        
        # 2. Attention Switch Frequency (ASF)
        switch_data = temporal_results.get('gaze_direction_switch_count', {})
        switch_count = switch_data.get('value')
        if switch_count is None: switch_count = 0
        else: switch_count = int(switch_count)
        
        # Use metadata duration (calculated at start of analysis)
        total_duration_s = max(duration_seconds, 1.0)
        
        asf = switch_count / total_duration_s
        asf_score = 1.0 / (1.0 + (asf * 0.75))
        
        # 3. Sustained Attention Score (SAS)
        max_continuous = temporal_results.get('max_continuous_eye_contact', {}).get('value')
        if max_continuous is None: max_continuous = 0.0
        sas = max_continuous / max(total_duration_s, 1.0)
        # Clamp SAS to [0, 1] just in case
        sas = min(1.0, max(0.0, sas))
        
        # 4. Gaze Stability Score (GSS)
        gss = temporal_results.get('gaze_stability', {}).get('value')
        if gss is None: gss = 0.0
        
        # Geometric Fusion
        # Score = (ECR^0.40) * (SAS^0.25) * (ASF^0.20) * (GSS^0.15)
        # Add epsilon to prevent zeroing out entirely? 
        # User requirement: "Ensure switch_count meaningfully penalizes engagement."
        # Multiplication does that. If ASF is high -> ASF_score low -> Total low.
        
        # ECR is the dominant factor.
        fusion_score = (ecr ** 0.40) * (sas ** 0.25) * (asf_score ** 0.20) * (gss ** 0.15)
        
        # Scale to 0-10
        engagement_score = fusion_score * 10.0
        
        # Interpretation Bands
        if fusion_score >= 0.85:
             engagement_interp = "Strong eye contact with sustained attention."
        elif fusion_score >= 0.70:
             engagement_interp = "Good eye contact with minor attention shifts."
        elif fusion_score >= 0.55:
             engagement_interp = "Moderate eye contact with noticeable attention shifts."
        elif fusion_score >= 0.40:
             engagement_interp = "Limited eye contact due to frequent attention shifts."
        else:
             engagement_interp = "Poor eye contact and unstable attention focus."
             
        # Map to legacy 0-4 level for backward compatibility if needed, or just use score
        # We'll rely on the text interpretation.
        
        # Update results
        # We replace the simple 'eye_contact_consistency' interpretation
        # But we still need to return the metric structure.
            
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
                        "eye_contact": eye_contact,
                        "gaze_stability": gaze_stab,
                        "smile_intensity": smile
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
            'metadata': {
                'processing_method': 'camera_invariant_geometry',
                'temporal_analysis': True,
                'confidence_estimation': True,
                'domain_isolation': True,
                'duration': round(duration_seconds, 2)
            }
        }

        response = {
            "analysis_version": "v2.2_frozen_calibrated",
            "results": results
        }
        
        return response

    except Exception as e:
        logger.error(f"Analysis Failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

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
