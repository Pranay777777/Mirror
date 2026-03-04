"""verify_users_videos.py — Tests the pipeline against a set of known user-provided test videos."""
import sys, os, json
from dotenv import load_dotenv
load_dotenv()

TEST_VIDEOS_DIR = os.path.join("uploads", "TestVideos")

def main():
    from utils.video_utils import analyze_video
    if not os.path.exists(TEST_VIDEOS_DIR):
        print(f"[USERS VIDEOS] Directory not found: {TEST_VIDEOS_DIR}")
        return
    videos = [f for f in os.listdir(TEST_VIDEOS_DIR) if f.endswith(".mp4")]
    if not videos:
        print("[USERS VIDEOS] No .mp4 files found in TestVideos/")
        return
    for v in videos:
        path = os.path.join(TEST_VIDEOS_DIR, v)
        print(f"\n[USERS VIDEOS] {v}")
        try:
            result = analyze_video(path, transcript=None, debug_mode=False)
            body = result.get("body", {})
            print(f"  posture={body.get('posture_score')} engagement={body.get('engagement_score')} overall={result.get('overall_score')}")
        except Exception as e:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    main()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
