import sys, os, logging
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.stt_engine import transcribe_audio

# Enable logging
logging.basicConfig(level=logging.INFO)

video_path = os.path.join("uploads", "videoB1.mp4")
print(f"Testing STT on: {video_path}")

if not os.path.exists(video_path):
    print("File not found!")
    sys.exit(1)

try:
    t0 = time.time()
    result = transcribe_audio(video_path)
    dt = time.time() - t0
    
    print(f"Time: {dt:.2f}s")
    print("Result Keys:", result.keys())
    print("Text:", result.get("text"))
    print("Language:", result.get("language"))
    print("Segments:", len(result.get("segments", [])))
    
    for i, seg in enumerate(result.get("segments", [])[:5]):
        print(f"Seg {i}: {seg}")

except Exception as e:
    import traceback
    traceback.print_exc()
