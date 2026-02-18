import os
from dotenv import load_dotenv
from sarvamai import SarvamAI
import tempfile

load_dotenv()

def test_sarvam():
    key = os.getenv("SARVAM_API_KEY")
    print(f"Testing Sarvam Key: {key[:5]}...{key[-5:] if key else 'None'}")
    
    if not key:
        print("❌ No key found")
        return

    client = SarvamAI(api_subscription_key=key)
    
    # Create valid dummy wav file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        temp_path = temp.name
    
    # Generate 1 sec silent wav using wave
    import wave
    with wave.open(temp_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(b'\x00' * 44100 * 2)

    try:
        print("Sending request...")
        with open(temp_path, "rb") as f:
            resp = client.speech_to_text.translate(file=f, model="saaras:v2.5")
            print("✅ Response:", resp)
    except Exception as e:
        print(f"❌ Sarvam Error: {e}")
    finally:
        os.remove(temp_path)

if __name__ == "__main__":
    test_sarvam()
