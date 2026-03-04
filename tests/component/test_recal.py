import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import sys
import logging
import warnings
warnings.filterwarnings('ignore')

from utils.video_utils import analyze_video

def main():
    print("\nTesting poss2.mp4 (High movement)")
    try:
        # Instead of just running analyze_video, let's look at the logs we get or we just use the parse_logs.py approach
        analyze_video("uploads/poss2.mp4")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
