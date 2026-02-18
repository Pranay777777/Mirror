import os
import tempfile
import json
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip
from pydub import AudioSegment
from sarvamai import SarvamAI

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
            text = resp.text if hasattr(resp, "text") else str(resp)
            final_text.append(text)
        os.remove(chunk_path) 

    full_transcript = " ".join(final_text)
    return {"full_transcript": full_transcript} 