import logging

logger = logging.getLogger(__name__)

def multimodal_confidence_fusion(scores: dict) -> float:
    """
    Fuses reliability scores from multiple modalities into a single global confidence score.
    scores: dict like {'posture': 0.8, 'gaze': 0.7, 'speech': 0.9, ...}
    Returns float 0.0-1.0
    """
    if not scores:
        return 0.0
        
    # Weights for each modality
    # Adjust based on importance
    weights = {
        'posture': 0.25,
        'gaze': 0.25,
        'speech': 0.20, # reliability of transcript/wpm
        'audio': 0.15, # reliability of signal
        'head_pose': 0.15
    }
    
    total_weight = 0.0
    weighted_sum = 0.0
    
    for key, weight in weights.items():
        score = scores.get(key)
        
        # If score is present, include it
        if score is not None and isinstance(score, (int, float)):
            # Clamp score
            s = max(0.0, min(1.0, float(score)))
            weighted_sum += s * weight
            total_weight += weight
            
    if total_weight == 0:
        return 0.0
        
    # Normalize by total weight of available modalities
    final_score = weighted_sum / total_weight
    return round(final_score, 4)
