import os
import sys
import traceback
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Import utils
from utils.video_utils import analyze_video
from utils.audio_utils import process_audio
from utils.scoring_utils import score_audio

def reproduce(video_path):
    print(f"🎬 Reproducing analysis for: {video_path}")
    
    try:
        if not os.path.exists(video_path):
            print(f"❌ File not found: {video_path}")
            return

        print("1️⃣  Starting Body Language Analysis...")
        body_data = analyze_video(video_path)
        if "error" in body_data:
            print(f"⚠️  Body Language Error: {body_data['error']}")
        else:
            print("✅ Body Language Analysis Complete")

        print("2️⃣  Starting Transcription...")
        prefix = os.path.splitext(os.path.basename(video_path))[0]
        # Check API KEY
        sarvam_key = os.getenv("SARVAM_API_KEY")
        if not sarvam_key:
            print("❌ Missing SARVAM_API_KEY")
            return
            
        audio_data = process_audio(video_path, prefix, sarvam_key)
        print("✅ Transcription Complete")
        
        print("3️⃣  Starting Scoring...")
        transcript = audio_data.get("full_transcript", "")
        print(f"   Transcript length: {len(transcript)} chars")
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("❌ Missing OPENAI_API_KEY")
            return
            
        audio_scores = score_audio(transcript, openai_key)
        print("✅ Scoring Complete")
        
        print("\n🎉 SUCCESS! No exceptions raised.")
        print(audio_scores)

    except Exception:
        print("\n❌ CAUGHT EXCEPTION:")
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reproduce_issue.py <video_path>")
    else:
        reproduce(sys.argv[1])
