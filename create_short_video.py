from moviepy.editor import VideoFileClip
import sys

def shorten_video(input_path, output_path, duration=10):
    print(f"✂️ Cutting {input_path} to {duration}s...")
    with VideoFileClip(input_path) as video:
        # Cut first 'duration' seconds
        short_video = video.subclip(0, min(duration, video.duration))
        short_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print(f"✅ Created {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_short_video.py <input> <output>")
    else:
        shorten_video(sys.argv[1], sys.argv[2])
