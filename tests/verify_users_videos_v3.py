import sys, os, logging
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import analyze_video

logging.getLogger().setLevel(logging.ERROR)

videos = [
    "good_english.mp4", 
    "bad_english.mp4", 
    "english_women_bad.mov", 
    "english_women_good.mov"
]
results = []

print(f"{'Video':<25} | {'Score':<5} | {'Rel':<4} | {'Rates (W/F/P/Pi/E)':<25} | {'Mode'}", flush=True)
print("-" * 100, flush=True)

for vid in videos:
    path = os.path.join("uploads", vid)
    if not os.path.exists(path):
        print(f"{vid:<25} | FILE NOT FOUND", flush=True)
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
        
        # Raw metrics for context
        wpm = metrics.get("words_per_minute", 0)
        pitch_hz = 0
        if sa.get("audio_features"):
             pitch_hz = sa["audio_features"]["metrics"].get("pitch_std_hz", 0)
        
        rates_str = f"R:{rate} F:{fill} P:{pause} Pi:{pitch} E:{egy}"
        
        msg_line = f"{vid:<25} | {score:<5} | {rel:<4} | {rates_str:<25} | {mode}"
        print(msg_line, flush=True)
        results.append(msg_line)
        
        # Also print transcript snippet
        transcript = r["results"].get("transcript", "")
        if transcript:
            trans_line = f"   Transcript: \"{transcript[:60]}...\" (WPM: {wpm:.1f}, PitchStd: {pitch_hz:.1f})"
        else:
            trans_line = f"   Transcript: [None] (WPM: {wpm:.1f}, PitchStd: {pitch_hz:.1f})"
        print(trans_line, flush=True)
        print("", flush=True)
        results.append(trans_line)
        results.append("")

    except Exception as e:
        # import traceback
        # traceback.print_exc()
        msg = f"{vid:<25} | ERROR: {str(e)[:50]}..."
        print(msg, flush=True)
        results.append(msg)

with open("verify_v3_full.txt", "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

print("\nDONE (Written to verify_v3_full.txt)", flush=True)
