import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
from utils.video_utils import analyze_video
import json

def debug_columns():
    res = analyze_video('uploads/EYE1.mp4', debug_mode=True)
    temp = res.get('results', {}).get('multimodal_analysis', {}).get('posture_analysis', {}).get('metrics', {})
    print("ALL KEYS IN TEMP:", list(temp.keys()))
    if 'eye_contact_consistency' in temp:
        print("ECR:", temp['eye_contact_consistency'])
    else:
        print("ECR KEY NOT IN TEMP")

debug_columns()
