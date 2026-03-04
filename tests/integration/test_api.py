"""
test_api.py — Integration test for the /analyze endpoint via HTTP.
Usage: python test_api.py <video.mp4> [--url http://127.0.0.1:8000]
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import sys, json, requests, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    args = parser.parse_args()

    endpoint = f"{args.url}/analyze"
    print(f"[TEST API] POST {endpoint}")
    with open(args.video, "rb") as f:
        resp = requests.post(endpoint, files={"file": (args.video, f, "video/mp4")})
    print(f"  Status: {resp.status_code}")
    try:
        data = resp.json()
        print(json.dumps(data, indent=2))
    except Exception:
        print(resp.text)

if __name__ == "__main__":
    main()
