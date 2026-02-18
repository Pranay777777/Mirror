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
from dotenv import load_dotenv

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
        body_data = analyze_video(video_path, transcript=transcript_text, debug_mode=debug_mode)

        # 3. Tone & Quality Scoring (OpenAI with structured output)
        audio_scores = {}
        # Only compute OpenAI scores if we have a transcript
        if transcript_text:
            try:
                audio_scores = score_audio(transcript_text, os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                logger.warning(f"OpenAI scoring failed: {e}")

        # Final JSON Response Body - Scientifically Defensible Format
        # Task 4: Clean JSON Structure
        
        # Check if analyze_video returned the new structured response
        if "results" in body_data and "multimodal_analysis" in body_data["results"]:
             # Inject the external scoring and transcript into the results
             body_data["results"]["qualitative_feedback"] = audio_scores
             body_data["results"]["transcript"] = transcript_text
             
             # Ensure summary_view is present (it should be)
             if "summary_view" not in body_data["results"]:
                 body_data["results"]["summary_view"] = {}
                 
             # Wrappers
             return {
                 "status": "success",
                 "filename": original_name,
                 "analysis_version": body_data.get("analysis_version", "v2.1_stabilized"),
                 "results": body_data["results"],
                 "metadata": {
                     "processing_method": "camera_invariant_geometry",
                     "temporal_analysis": True,
                     "confidence_estimation": True
                 }
             }
        else:
            # LEGACY PATH: If analyze_video returns flat dict (shouldn't happen with new code, but safe fallback)
            summary_view = body_data.pop('summary_view', None)
            if summary_view is None:
                 summary_view = {} 
            
            return {
                "status": "success",
                "filename": original_name,
                "analysis_version": "v2.1_stabilized",
                "results": {
                    "summary_view": summary_view,
                    "multimodal_analysis": body_data,
                    "qualitative_feedback": audio_scores,
                    "transcript": transcript_text
                },
                "metadata": {
                    "processing_method": "camera_invariant_geometry",
                    "temporal_analysis": True,
                    "confidence_estimation": True
                }
            }

    except Exception as e:
        # Log full stack trace for debugging in server logs
        logger.exception("Analysis failed")
        traceback.print_exc()
        # Return sanitized error to client
        return JSONResponse(status_code=500, content={"status": "error", "detail": "internal server error"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)