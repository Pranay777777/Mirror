import json
import logging
import requests
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def score_audio(transcript_text: str, openai_key: str) -> Dict:
    """
    Send transcript to OpenAI with structured rubric and safety constraints.
    
    Returns a dict with validated, clamped scores and confidence estimation.
    On API error or malformed response returns an error dict.
    """
    api_url = "https://api.openai.com/v1/chat/completions"
    # Structured evaluation rubric with hard constraints
    system_prompt = """
You are a professional communication analyst. Evaluate the transcript using this EXACT rubric:

Return ONLY a JSON object with these exact fields:
{
  "sentiment": "positive|neutral|negative",
  "tone_quality": "professional|casual|inappropriate", 
  "professionalism_score": 0.0-10.0,
  "communication_score": 0.0-10.0,
  "voice_quality_feedback": "brief constructive feedback",
  "confidence": 0.0-1.0,
  "analysis_notes": "brief explanation of scores"
}

SCORING CRITERIA:
- Professionalism: clarity, structure, appropriate language
- Communication: engagement, coherence, effectiveness
- Confidence: how certain you are of this evaluation

IMPORTANT:
- ALL scores MUST be between 0.0 and 10.0 inclusive
- Confidence MUST be between 0.0 and 1.0 inclusive
- If transcript is too short/unclear, set confidence < 0.5
- Do not invent information not present in transcript
- Do not self-grade or meta-analyze
"""

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript_text}
        ]
    }

    headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        logger.exception("OpenAI request failed")
        return {
            "error": "request_exception", 
            "details": str(e),
            "confidence": 0.0,
            "professionalism_score": 0.0,
            "communication_score": 0.0
        }

    # Check HTTP status
    if response.status_code != 200:
        body = None
        try:
            body = response.json()
        except Exception:
            body = response.text
        logger.error("OpenAI returned non-200: %s %s", response.status_code, body)
        return {
            "error": "http_error", 
            "status_code": response.status_code, 
            "details": body,
            "confidence": 0.0,
            "professionalism_score": 0.0,
            "communication_score": 0.0
        }

    # Parse expected structure
    try:
        resp_json = response.json()
    except ValueError:
        logger.error("OpenAI returned non-json response: %s", response.text)
        return {
            "error": "invalid_json", 
            "details": response.text,
            "confidence": 0.0,
            "professionalism_score": 0.0,
            "communication_score": 0.0
        }

    if "choices" not in resp_json or not resp_json["choices"]:
        logger.error("OpenAI response missing choices: %s", resp_json)
        return {
            "error": "no_choices", 
            "details": resp_json,
            "confidence": 0.0,
            "professionalism_score": 0.0,
            "communication_score": 0.0
        }

    try:
        content = resp_json["choices"][0]["message"]["content"]
    except Exception as e:
        logger.exception("Failed to extract content from OpenAI response")
        return {
            "error": "extract_content_failed", 
            "details": str(e), 
            "response": resp_json,
            "confidence": 0.0,
            "professionalism_score": 0.0,
            "communication_score": 0.0
        }

    try:
        parsed = json.loads(content)
        return _validate_and_clamp_scores(parsed)
    except json.JSONDecodeError:
        logger.error("OpenAI content is not valid JSON: %s", content)
        return {
            "error": "invalid_json_in_content", 
            "raw_content": content,
            "confidence": 0.0,
            "professionalism_score": 0.0,
            "communication_score": 0.0
        }

def _validate_and_clamp_scores(parsed: Dict) -> Dict:
    # Validate and clamp scores
    if "sentiment" not in parsed or parsed["sentiment"] not in ["positive", "neutral", "negative"]:
        logger.error("Invalid sentiment: %s", parsed.get("sentiment"))
        parsed["sentiment"] = "neutral"

    if "tone_quality" not in parsed or parsed["tone_quality"] not in ["professional", "casual", "inappropriate"]:
        logger.error("Invalid tone quality: %s", parsed.get("tone_quality"))
        parsed["tone_quality"] = "casual"

    if "professionalism_score" not in parsed or not isinstance(parsed["professionalism_score"], (int, float)) or parsed["professionalism_score"] < 0.0 or parsed["professionalism_score"] > 10.0:
        logger.error("Invalid professionalism score: %s", parsed.get("professionalism_score"))
        parsed["professionalism_score"] = 5.0

    if "communication_score" not in parsed or not isinstance(parsed["communication_score"], (int, float)) or parsed["communication_score"] < 0.0 or parsed["communication_score"] > 10.0:
        logger.error("Invalid communication score: %s", parsed.get("communication_score"))
        parsed["communication_score"] = 5.0

    if "confidence" not in parsed or not isinstance(parsed["confidence"], (int, float)) or parsed["confidence"] < 0.0 or parsed["confidence"] > 1.0:
        logger.error("Invalid confidence: %s", parsed.get("confidence"))
        parsed["confidence"] = 0.5

    if "voice_quality_feedback" not in parsed or not isinstance(parsed["voice_quality_feedback"], str):
        logger.error("Invalid voice quality feedback: %s", parsed.get("voice_quality_feedback"))
        parsed["voice_quality_feedback"] = ""

    if "analysis_notes" not in parsed or not isinstance(parsed["analysis_notes"], str):
        logger.error("Invalid analysis notes: %s", parsed.get("analysis_notes"))
        parsed["analysis_notes"] = ""

    return parsed

def get_score_val(metric_data) -> float:
    """
    Extract numeric value from metric dictionary or return float directly.
    """
    if isinstance(metric_data, dict):
        return float(metric_data.get('value', 0.0))
    if metric_data is None:
        return 0.0
    try:
        return float(metric_data)
    except (ValueError, TypeError):
        return 0.0

def get_score_range_desc(score: float) -> str:
    """
    Return a descriptive string for a score (0-10 or 0-1).
    Heuristic: if <=1.0 assume normalized, if >1.0 assume 0-10.
    """
    if score is None: return "Unknown"
    
    val = float(score)
    # Normalized 0-1
    if val <= 1.0:
        if val >= 0.8: return "High"
        if val >= 0.5: return "Moderate"
        return "Low"
    else:
        # Scale 0-10
        if val >= 8.0: return "High"
        if val >= 5.0: return "Moderate"
        return "Low"