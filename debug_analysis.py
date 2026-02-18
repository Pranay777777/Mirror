"""
Debug script to check what features are actually being extracted.
"""

import sys
sys.path.append('.')

from utils.video_utils import analyze_video
import json

def debug_video_analysis(video_path):
    """Debug video analysis to see what's happening."""
    
    print(f"🔍 DEBUGGING VIDEO ANALYSIS: {video_path}")
    print("=" * 60)
    
    # Run analysis
    results = analyze_video(video_path)
    
    print("\n📊 ANALYSIS METADATA:")
    print(json.dumps(results.get('analysis_metadata', {}), indent=2))
    
    print("\n🔬 TEMPORAL FEATURES:")
    temporal = results.get('temporal_features', {})
    print(json.dumps(temporal, indent=2))
    
    print("\n🧍 POSTURE ANALYSIS:")
    posture = results.get('posture_analysis', {})
    print(json.dumps(posture, indent=2))
    
    print("\n👁️ ENGAGEMENT ANALYSIS:")
    engagement = results.get('engagement_analysis', {})
    print(json.dumps(engagement, indent=2))
    
    print("\n😊 EXPRESSION ANALYSIS:")
    expression = results.get('expression_analysis', {})
    print(json.dumps(expression, indent=2))
    
    print("\n🎯 CONFIDENCE METRICS:")
    confidence = results.get('confidence_metrics', {})
    print(json.dumps(confidence, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_analysis.py <path_to_video>")
        sys.exit(1)
    
    debug_video_analysis(sys.argv[1])
