"""
test_post.py — Send a video to the running /analyze endpoint and display the full JSON response.

Usage:
    python test_post.py <path_to_video.mp4>
    python test_post.py uploads/TestVideos/bad_Hindi2.mp4

The server must be running:
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload
"""
import sys
import json
import time
import requests

SERVER_URL = "http://127.0.0.1:8000/analyze"
TIMEOUT_SECONDS = 600  # 10 minutes — large videos take time


def main(video_path: str):
    print(f"\n📹 Sending video: {video_path}")
    print(f"🌐 Server: {SERVER_URL}")

    _start = time.time()
    try:
        with open(video_path, "rb") as f:
            filename = video_path.replace("\\", "/").split("/")[-1]
            files = {"file": (filename, f, "video/mp4")}
            resp = requests.post(SERVER_URL, files=files, timeout=TIMEOUT_SECONDS)
    except FileNotFoundError:
        print(f"❌ ERROR: Video file not found: {video_path}")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to server. Make sure uvicorn is running.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out. Video may be too large or server is slow.")
        sys.exit(1)

    elapsed = time.time() - _start
    mins, secs = divmod(int(elapsed), 60)
    print(f"✅ Analysis completed in {mins}m {secs}s ({elapsed:.1f}s total)\n")

    if resp.status_code != 200:
        print(f"❌ ERROR: Server returned HTTP {resp.status_code}")
        print(resp.text[:500])
        sys.exit(1)

    try:
        result = resp.json()
    except Exception:
        print("❌ ERROR: Server response is not valid JSON")
        print(resp.text[:500])
        sys.exit(1)

    # ── Pretty print ──────────────────────────────────────────────────────────
    print("🎯 FULL RESPONSE JSON:")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 60)

    print("\n🎉 Analysis complete! Results are scientifically defensible.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to bad_Hindi2.mp4 if no argument given
        default_video = "uploads/TestVideos/bad_Hindi2.mp4"
        print(f"No video path provided — using default: {default_video}")
        main(default_video)
    else:
        main(sys.argv[1])
