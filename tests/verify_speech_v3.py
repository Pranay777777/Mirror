import sys, os, logging
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import analyze_video

logging.getLogger().setLevel(logging.ERROR)

videos = ["videoB1.mp4", "good_english.mp4"]
results = []

print(f"{'Video':<15} | {'Score':<5} | {'Rel':<4} | {'Rates':<20} | {'Mode'}", flush=True)
print("-" * 100, flush=True)

for vid in videos:
    path = os.path.join("uploads", vid)
    if not os.path.exists(path):
        print(f"{vid:<15} | FILE NOT FOUND")
        continue

    try:
        t0 = time.time()
        # Run without transcript argument to force local Whisper usage
        r = analyze_video(path, transcript=None, debug_mode=True)
        dt = time.time() - t0
        
        sa = r["results"]["multimodal_analysis"].get("speech_analysis", {})
        score = sa.get("score")
        metrics = sa.get("metrics", {})
        
        mode = metrics.get("speech_logic_mode", "?")
        
        # Audio metrics
        cm = r["results"]["multimodal_analysis"]["confidence_metrics"]
        rel = cm.get("audio_confidence", 0.0)
        
        # Breakdown
        rate = metrics.get("rate_score", "-")
        fill = metrics.get("filler_score", "-")
        pause = metrics.get("pause_score", "-")
        pitch = metrics.get("pitch_score", "-")
        egy = metrics.get("energy_score", "-")
        
        rates_str = f"R:{rate} F:{fill} P:{pause} Pi:{pitch} E:{egy}"
        
        print(f"{vid:<15} | {score:<5} | {rel:<4} | {rates_str:<20} | {mode} ({dt:.1f}s)")
        
        # Also print transcript snippet
        transcript = r["results"].get("transcript", "")
        if transcript:
            print(f"   Transcript: {transcript[:50]}...")
        else:
            print("   Transcript: [None]")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"{vid:<15} | ERROR: {str(e)[:50]}...")

print("\nDONE")
