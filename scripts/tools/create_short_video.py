"""
create_short_video.py — Trims a source video to a short clip for testing.
Usage: python create_short_video.py <input.mp4> <output.mp4> [duration_sec]
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import sys
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip

def trim(src: str, dst: str, duration: float = 15.0):
    with VideoFileClip(src) as clip:
        short = clip.subclip(0, min(duration, clip.duration))
        short.write_videofile(dst, logger=None)
    print(f"Created {dst} ({duration}s)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_short_video.py <input.mp4> <output.mp4> [duration_sec]")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    trim(src, dst, dur)
