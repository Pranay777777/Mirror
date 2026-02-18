import sys
import requests
import time
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Test video analysis API")
    parser.add_argument("file", help="Path to the video file to analyze")
    parser.add_argument("--url", default="http://127.0.0.1:8000/analyze", help="URL of the analysis endpoint")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.file):
        print(f"Error: File not found: '{args.file}'")
        print('Hint: Use quotes around paths with spaces, e.g., "C:\\My Documents\\video.mp4"')
        sys.exit(1)

    path = args.file
    url = args.url
    
    print(f"🎬 Starting analysis of: {path}")
    print("⏱️  This may take 1-5 minutes depending on video length and quality...")
    
    start_time = time.time()
    
    with open(path, "rb") as f:
        filename = os.path.basename(path)
        files = {"file": (filename, f, "video/mp4")}
        try:
            print("📤 Uploading and processing...")
            resp = requests.post(url, files=files, timeout=600)  # 10 minute timeout
            resp.raise_for_status() # Raise HTTPError for bad responses (4xx, 5xx)
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            print(f"⏰ Request timed out after {elapsed:.1f} seconds")
            print("💡 Try with a shorter video (<30 seconds) or check server logs")
            sys.exit(1)
        except requests.RequestException as e:
            print(f"❌ Request failed: {e}")
            if 'resp' in locals():
                 print(f"Response status: {resp.status_code}")
                 print(f"Response text: {resp.text}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            sys.exit(1)
    
    elapsed = time.time() - start_time
    print(f"⏱️  Analysis completed in {elapsed:.1f} seconds")
    print(f"📊 Status: {resp.status_code}")
    
    try:
        result = resp.json()
        import json
        print("\n🎯 FULL RESPONSE JSON:")
        print("=" * 50)
        print(json.dumps(result, indent=2, default=str))
        
    except Exception as e:
        print("📄 Raw Response:")
        print(resp.text)
    
    print("\n" + "=" * 50)
    print("🎉 Analysis complete! Results are scientifically defensible.")

if __name__ == "__main__":
    main()
