"""
test_openai.py — Tests the OpenAI scoring utility directly with a sample transcript.
Usage: python test_openai.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
from dotenv import load_dotenv
import os, json
load_dotenv()

SAMPLE_TRANSCRIPT = (
    "Good morning everyone. Today I want to talk about effective communication. "
    "It is important to speak clearly, maintain eye contact, and demonstrate confidence. "
    "In conclusion, practice makes perfect."
)

def main():
    from utils.scoring_utils import score_audio
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("[ERROR] OPENAI_API_KEY not set in .env")
        return
    print(f"[TEST OPENAI] Scoring transcript ({len(SAMPLE_TRANSCRIPT)} chars)...")
    result = score_audio(SAMPLE_TRANSCRIPT, key)
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
