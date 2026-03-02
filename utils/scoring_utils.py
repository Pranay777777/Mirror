import json
import logging
import requests
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert communication analyst specializing in interview and professional speech evaluation.

Your task is to evaluate a transcript of spoken communication and return ONLY a valid JSON object in the exact format specified below.

IMPORTANT RULES:
- Analyze ONLY the transcript text.
- Do NOT mention limitations.
- Do NOT include explanations outside the JSON.
- Do NOT wrap output in code fences.
- Return STRICT JSON.
- All numeric scores must be between 0.0 and 10.0.
- Be consistent and deterministic in scoring.

You must evaluate the transcript across two categories:

---------------------------------------
1) SPEECH ANALYSIS (Perceptual / Semantic)
---------------------------------------

sentiment:
  - "positive" → confident, optimistic, solution-oriented
  - "neutral" → factual, balanced, emotionally stable
  - "negative" → defensive, pessimistic, frustrated

tone_quality:
  - "professional" → structured, respectful, workplace-appropriate
  - "casual" → informal but acceptable
  - "inappropriate" → slang-heavy, disrespectful, unprofessional

sarcasm:
  - true → if irony or mocking intent is clearly present
  - false → otherwise

---------------------------------------
2) TECHNICAL DELIVERY METRICS (Text-Inferred)
---------------------------------------

pace:
  - "slow" → overly wordy, repetitive, dragging sentences
  - "balanced" → natural flow, structured progression
  - "fast" → rushed, fragmented ideas, abrupt transitions

clarity_score (0-10):
  Evaluate logical structure, coherence, and ease of understanding.

energy_level:
  - "low" → flat, passive, weak phrasing
  - "moderate" → steady and composed
  - "high" → dynamic, impactful, enthusiastic

filler_usage:
  - "low" → clean speech, minimal unnecessary phrases
  - "moderate" → occasional fillers
  - "high" → frequent unnecessary phrases or redundancy

---------------------------------------
3) SCORING
---------------------------------------

professionalism_score (0-10):
  Based on vocabulary, structure, tone, appropriateness.

communication_score (0-10):
  Based on clarity, engagement, coherence, and impact.

overall_speech_score (0-10):
  Weighted combination:
    40% communication
    40% professionalism
    20% clarity_score

confidence (0-1):
  Model confidence in this evaluation.

---------------------------------------
RESPONSE FORMAT (STRICT JSON ONLY)
---------------------------------------

{
  "speech_analysis": {
    "sentiment": "positive | neutral | negative",
    "tone_quality": "professional | casual | inappropriate",
    "sarcasm": false
  },
  "technical_metrics": {
    "pace": "slow | balanced | fast",
    "clarity_score": 0.0,
    "energy_level": "low | moderate | high",
    "filler_usage": "low | moderate | high"
  },
  "scores": {
    "professionalism_score": 0.0,
    "communication_score": 0.0,
    "overall_speech_score": 0.0
  },
  "confidence": 0.0
}"""


def _error_response(error_code: str, details) -> Dict:
    """Return a schema-consistent result dict on failure."""
    return {
        "error": error_code,
        "details": str(details),
        "speech_analysis": {
            "sentiment": "neutral",
            "tone_quality": "casual",
            "sarcasm": False,
        },
        "technical_metrics": {
            "pace": "balanced",
            "clarity_score": 0.0,
            "energy_level": "moderate",
            "filler_usage": "moderate",
        },
        "scores": {
            "professionalism_score": 0.0,
            "communication_score": 0.0,
            "overall_speech_score": 0.0,
        },
        "confidence": 0.0,
    }


def _validate_and_clamp_scores(parsed: Dict) -> Dict:
    """Validate and apply safe defaults for the nested speech evaluation schema."""

    # ── speech_analysis ──────────────────────────────────────────────────────
    sa = parsed.get("speech_analysis")
    if not isinstance(sa, dict):
        sa = {}
        parsed["speech_analysis"] = sa

    VALID_SENTIMENT = {"positive", "neutral", "negative"}
    if sa.get("sentiment") not in VALID_SENTIMENT:
        logger.warning("Invalid sentiment '%s', defaulting to 'neutral'", sa.get("sentiment"))
        sa["sentiment"] = "neutral"

    VALID_TONE = {"professional", "casual", "inappropriate"}
    if sa.get("tone_quality") not in VALID_TONE:
        logger.warning("Invalid tone_quality '%s', defaulting to 'casual'", sa.get("tone_quality"))
        sa["tone_quality"] = "casual"

    if not isinstance(sa.get("sarcasm"), bool):
        logger.warning("Invalid sarcasm value '%s', defaulting to False", sa.get("sarcasm"))
        sa["sarcasm"] = False

    # ── technical_metrics ────────────────────────────────────────────────────
    tm = parsed.get("technical_metrics")
    if not isinstance(tm, dict):
        tm = {}
        parsed["technical_metrics"] = tm

    VALID_PACE = {"slow", "balanced", "fast"}
    if tm.get("pace") not in VALID_PACE:
        logger.warning("Invalid pace '%s', defaulting to 'balanced'", tm.get("pace"))
        tm["pace"] = "balanced"

    cs = tm.get("clarity_score")
    try:
        cs_f = float(cs)
        tm["clarity_score"] = round(cs_f, 2) if 0.0 <= cs_f <= 10.0 else 5.0
    except (TypeError, ValueError):
        logger.warning("Invalid clarity_score '%s', defaulting to 5.0", cs)
        tm["clarity_score"] = 5.0

    VALID_ENERGY = {"low", "moderate", "high"}
    if tm.get("energy_level") not in VALID_ENERGY:
        logger.warning("Invalid energy_level '%s', defaulting to 'moderate'", tm.get("energy_level"))
        tm["energy_level"] = "moderate"

    VALID_FILLER = {"low", "moderate", "high"}
    if tm.get("filler_usage") not in VALID_FILLER:
        logger.warning("Invalid filler_usage '%s', defaulting to 'moderate'", tm.get("filler_usage"))
        tm["filler_usage"] = "moderate"

    # ── scores ───────────────────────────────────────────────────────────────
    sc = parsed.get("scores")
    if not isinstance(sc, dict):
        sc = {}
        parsed["scores"] = sc

    for field, default in [
        ("professionalism_score", 5.0),
        ("communication_score", 5.0),
        ("overall_speech_score", 5.0),
    ]:
        val = sc.get(field)
        try:
            v = float(val)
            sc[field] = round(v, 2) if 0.0 <= v <= 10.0 else default
        except (TypeError, ValueError):
            logger.warning("Invalid %s '%s', defaulting to %.1f", field, val, default)
            sc[field] = default

    # ── confidence ───────────────────────────────────────────────────────────
    conf = parsed.get("confidence")
    try:
        c = float(conf)
        parsed["confidence"] = round(c, 3) if 0.0 <= c <= 1.0 else 0.5
    except (TypeError, ValueError):
        logger.warning("Invalid confidence '%s', defaulting to 0.5", conf)
        parsed["confidence"] = 0.5

    return parsed


def score_audio(transcript_text: str, openai_key: str) -> Dict:
    """
    Send transcript to OpenAI GPT-4o-mini with a structured speech evaluation rubric.

    Returns a nested dict with:
      - speech_analysis  (sentiment, tone_quality, sarcasm)
      - technical_metrics (pace, clarity_score, energy_level, filler_usage)
      - scores           (professionalism_score, communication_score, overall_speech_score)
      - confidence       (0–1)

    On any failure returns an error dict with the same top-level keys zeroed out.
    """
    api_url = "https://api.openai.com/v1/chat/completions"

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": transcript_text},
        ],
    }
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        logger.exception("OpenAI request failed")
        return _error_response("request_exception", str(e))

    if response.status_code != 200:
        body = None
        try:
            body = response.json()
        except Exception:
            body = response.text
        logger.error("OpenAI returned non-200: %s %s", response.status_code, body)
        return _error_response("http_error", body)

    try:
        resp_json = response.json()
    except ValueError:
        logger.error("OpenAI returned non-JSON response: %s", response.text)
        return _error_response("invalid_json", response.text)

    if "choices" not in resp_json or not resp_json["choices"]:
        logger.error("OpenAI response missing choices: %s", resp_json)
        return _error_response("no_choices", resp_json)

    try:
        content = resp_json["choices"][0]["message"]["content"]
    except Exception as e:
        logger.exception("Failed to extract content from OpenAI response")
        return _error_response("extract_content_failed", str(e))

    try:
        parsed = json.loads(content)
        return _validate_and_clamp_scores(parsed)
    except json.JSONDecodeError:
        logger.error("OpenAI content is not valid JSON: %s", content)
        return _error_response("invalid_json_in_content", content)


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
    if score is None:
        return "Unknown"

    val = float(score)
    if val <= 1.0:
        if val >= 0.8: return "High"
        if val >= 0.5: return "Moderate"
        return "Low"
    else:
        if val >= 8.0: return "High"
        if val >= 5.0: return "Moderate"
        return "Low"