import traceback
import sys
import json
from utils.video_utils import analyze_video

def run():
    f = open('py_err.txt', 'w', encoding='utf-8')
    sys.stdout = f
    sys.stderr = f
    try:
        r = analyze_video('uploads/fhappy.mp4', debug_mode=False)
        print("FINAL INTERPRETATION STRING:")
        # The key in the public response is just "interpretation"
        print("Body Interp:", r.get("body", {}).get("interpretation", "MISSING"))
    except Exception as e:
        traceback.print_exc()
    finally:
        f.close()

if __name__ == '__main__':
    run()
