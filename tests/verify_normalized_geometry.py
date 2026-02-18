import unittest
import numpy as np
import cv2
import sys
import os

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.normalized_geometry import NormalizedGeometry

class TestNormalizedGeometry(unittest.TestCase):
    def test_analyze_frame_structure(self):
        print("Initializing NormalizedGeometry...")
        geom = NormalizedGeometry()
        
        # Create a dummy black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        print("Calling analyze_frame...")
        results = geom.analyze_frame(frame)
        
        print(f"Results keys: {results.keys()}")
        
        self.assertIsInstance(results, dict)
        # We expect at least some keys, or empty dict if no holistic landmarks found (which is expected on black frame)
        # But crucially, it should NOT raise AttributeError.
        
        # To test actual features, we'd need a real face image, but for now we just want to verify method existence and signature.
        
    def test_analyze_frame_method_exists(self):
        geom = NormalizedGeometry()
        self.assertTrue(hasattr(geom, 'analyze_frame'), "analyze_frame method missing")

if __name__ == '__main__':
    unittest.main()
