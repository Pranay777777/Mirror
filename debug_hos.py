import os
import json
import warnings
warnings.filterwarnings('ignore')

from utils.video_utils import analyze_video 

res = analyze_video('uploads/EYE2.mp4', debug_mode=True)
print("\n--- RES KEYS ---")
print(res.keys())
if 'results' in res:
    print("--- RES['results'] KEYS ---")
    print(res['results'].keys())
    if 'multimodal_analysis' in res['results']:
        print("--- MULTI KEYS ---")
        print(res['results']['multimodal_analysis'].keys())
        if 'body' in res['results']['multimodal_analysis']:
            print(json.dumps(res['results']['multimodal_analysis']['body'], indent=2))
        else:
            print("NO BODY KEY. Keys:", res['results']['multimodal_analysis'].keys())
