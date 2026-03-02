import os
import re
import tempfile
import json
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip
from pydub import AudioSegment
from sarvamai import SarvamAI

def _extract_sarvam_transcript(resp) -> str:
    """
    Safely extract the clean speech text from a Sarvam STT response object.

    Priority:
      1. resp.transcript  (correct field per Sarvam SDK)
      2. resp.text        (legacy / alternate SDK versions)
      3. str(resp)        (last-resort, with sanitization to strip metadata)

    Returns an empty string if nothing usable is found.
    """
    text = ""

    if hasattr(resp, "transcript") and resp.transcript:
        text = str(resp.transcript)
    elif hasattr(resp, "text") and resp.text:
        text = str(resp.text)
    else:
        # Absolute fallback — stringify and try to extract the transcript field
        raw = str(resp)
        # Pattern: transcript='...' or transcript="..."
        m = re.search(r'transcript=["\']([^"\']*)["\']', raw)
        if m:
            text = m.group(1)
        else:
            text = raw  # give up and pass raw; will be caught by length check below

    text = text.strip()

    # Safety: if the extracted text still looks like a stringified object,
    # reject it and return empty so the fallback logic can engage.
    if text.startswith("request_id=") or text.startswith("SpeechToText"):
        print("[SARVAM PARSER] WARNING: extracted text looks like a stringified object — rejecting.")
        return ""

    return text


def process_audio(video_path, prefix, sarvam_key):
    # Extract audio using a context manager to release the file lock
    with VideoFileClip(video_path) as video:
        audio_file = f"uploads/{prefix}_audio.mp3"
        video.audio.write_audiofile(audio_file, logger=None)
    
    client = SarvamAI(api_subscription_key=sarvam_key)

    def split_audio_to_chunks(audio_path, chunk_length_ms=29000):
        audio = AudioSegment.from_file(audio_path)
        chunks = []
        start = 0
        while start < len(audio):
            end = start + chunk_length_ms
            chunk = audio[start:end]
            # Windows Fix: Close the file descriptor immediately
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            chunk.export(temp_path, format="wav")
            chunks.append(temp_path)
            start = end
        return chunks

    chunks = split_audio_to_chunks(audio_file)
    final_text = []
    
    for chunk_path in chunks:
        with open(chunk_path, "rb") as f:
            resp = client.speech_to_text.translate(file=f, model="saaras:v2.5")
            text = _extract_sarvam_transcript(resp)
            print(f"[SARVAM CHUNK] extracted ({len(text)} chars): {text[:120]!r}")
            final_text.append(text)
        os.remove(chunk_path)

    full_transcript = " ".join(t for t in final_text if t).strip()
    print(f"[SARVAM FINAL] full_transcript ({len(full_transcript)} chars): {full_transcript[:200]!r}")
    return {"full_transcript": full_transcript}
