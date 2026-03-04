from dotenv import load_dotenv
import os
import shutil
import uuid
import logging
import traceback
try:
    import traycer
    traycer.start()
except Exception:
    pass  # traycer observability is optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from utils.video_utils import analyze_video
from utils.audio_utils import process_audio
from utils.scoring_utils import score_audio

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
app = FastAPI()

# basic logging to console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/analyze")
async def start_analysis(file: UploadFile = File(...), debug_mode: bool = None):
    try:
        # Determine debug mode: Query param > Env Var > Default (False for production safety)
        if debug_mode is None:
             debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"
        
        # Save video temporarily (safe filename + unique id)
        os.makedirs("uploads", exist_ok=True)
        # Prevent path traversal
        original_name = os.path.basename(file.filename or "upload.mp4")
        unique_name = f"{uuid.uuid4().hex}_{original_name}"
        video_path = os.path.join("uploads", unique_name)

        # Stream to disk in chunks to avoid large memory spikes
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        prefix = os.path.splitext(original_name)[0]
        
        # 1. Transcription & Audio Processing (Sarvam AI)
        # Move this BEFORE video analysis so we can pass transcript to metrics
        transcript_text = None
        audio_data = {}
        try:
             audio_data = process_audio(video_path, prefix, os.getenv("SARVAM_API_KEY"))
             transcript_text = audio_data.get("full_transcript")
        except Exception as e:
             logger.warning(f"Audio processing failed: {e}")

        # 2. Multimodal Body Language Analysis (Video + Audio + Text)
        result = analyze_video(video_path, transcript=transcript_text, debug_mode=debug_mode)

        # 3. Tone & Quality Scoring (OpenAI with structured output)
        audio_scores = {}
        # Only compute OpenAI scores if we have a transcript
        if transcript_text:
            try:
                audio_scores = score_audio(transcript_text, os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                logger.warning(f"OpenAI scoring failed: {e}")

        print("API MODULE PATH:", __file__)
        print("API analysis_version:", result.get("analysis_version"))
        print("API engagement_score:", result.get("body", {}).get("engagement_score"))
        print("API speech_activity_flag:", result.get("speech_activity_flag"))
        print("API gaze_metrics_available:", result.get("gaze_metrics_available"))

        if audio_scores:
            # Remove internal scoring sub-fields not intended for public output
            audio_scores.pop("scores", None)
            audio_scores.pop("confidence", None)

            # ── Build qualitative summary from available fields ───────────────
            sa  = audio_scores.get("speech_analysis", {})
            tm  = audio_scores.get("technical_metrics", {})
            sentiment    = sa.get("sentiment", "neutral")
            tone         = sa.get("tone_quality", "neutral")
            pace         = tm.get("pace", "balanced")
            clarity      = tm.get("clarity_score", 5.0)
            energy       = tm.get("energy_level", "moderate")
            fillers      = tm.get("filler_usage", "moderate")

            clarity_desc = (
                "excellent clarity" if float(clarity) >= 8
                else "good clarity" if float(clarity) >= 6
                else "moderate clarity" if float(clarity) >= 4
                else "low clarity"
            )
            filler_desc = (
                "minimal filler words" if fillers == "low"
                else "frequent filler words" if fillers == "high"
                else "occasional filler words"
            )
            audio_scores["summary"] = (
                f"The speaker demonstrated a {sentiment} sentiment with a {tone} tone. "
                f"Delivery was {pace}-paced with {clarity_desc} and {energy} energy. "
                f"Speech contained {filler_desc}."
            )

            # qualitative_feedback injection removed — speech_analysis in build_public_response() handles this now
            # if "results" in result:
            #     result["results"]["qualitative_feedback"] = audio_scores
            # else:
            #     result["qualitative_feedback"] = audio_scores


        print("API overall_score before return:", result.get("overall_score"))
        print("API final_score before return:", result.get("final_score"))

        # ── Enforce final key order ───────────────────────────────────────────
        # Desired: analysis_version → body → speech_score → speech_analysis → transcript → overall_score
        ordered = {}
        for key in ("analysis_version", "body", "speech_score", "speech_analysis",
                    "transcript", "overall_score"):
            if key in result:
                ordered[key] = result[key]
        # carry through any remaining keys not in the fixed list
        for key, val in result.items():
            if key not in ordered:
                ordered[key] = val
        return ordered


    except Exception as e:
        # Log full stack trace for debugging in server logs
        logger.exception("Analysis failed")
        traceback.print_exc()
        # Return sanitized error to client
        return JSONResponse(status_code=500, content={"status": "error", "detail": "internal server error"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)