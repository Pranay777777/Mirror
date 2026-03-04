import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import json
from utils.video_utils import analyze_video

def test_video(filepath):
    print(f"Testing {filepath}...")
    try:
        res = analyze_video(filepath, debug_mode=True)
        # In debug mode, metrics are inside posture_analysis -> metrics
        temporal = res.get('results', {}).get('multimodal_analysis', {}).get('posture_analysis', {}).get('metrics', {})
        ecr_data = temporal.get('eye_contact_consistency', {})
        diag = ecr_data.get('diagnostics', {})
        
        return {
            "ECR_old": diag.get('ECR_raw'),
            "ECR_new": diag.get('ECR_penalized'),
            "gaze_switches_per_min": diag.get('gaze_switches_per_min'),
            "sustained_contact_ratio": diag.get('sustained_contact_ratio')
        }
    except Exception as e:
        print(f"Error on {filepath}: {e}")
        return {}

def main():
    results = {}
    results['uploads/EYE1.mp4'] = test_video('uploads/EYE1.mp4')
    results['uploads/EYE3.mp4'] = test_video('uploads/EYE3.mp4')
    results['uploads/poss2.mp4'] = test_video('uploads/poss2.mp4')
    
    with open('test_ecr_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("DONE")

main()
