import os
from dotenv import load_dotenv
from utils.scoring_utils import score_audio

load_dotenv()

def test_openai():
    key = os.getenv("OPENAI_API_KEY")
    print(f"Testing OpenAI Key: {key[:5]}...{key[-5:] if key else 'None'}")
    
    if not key:
        print("❌ No key found")
        return

    print("Sending request to scoring_utils...")
    result = score_audio("This is a test transcript.", key)
    print("Result:", result)
    
    if "error" in result:
        print(f"❌ OpenAI Error: {result}")
    else:
        print("✅ OpenAI Success")

if __name__ == "__main__":
    test_openai()
