import unittest
import sys
import os
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.head_pose_metrics import HeadPoseMetrics

class TestHeadPoseFix(unittest.TestCase):
    def test_stateful_metrics(self):
        metrics = HeadPoseMetrics()
        
        # Test add_frame_data with tuple (yaw, pitch, roll)
        # This mirrors what NormalizedGeometry returns and video_utils passes
        # Note: NormalizedGeometry returns (yaw, pitch, roll) in that order
        # HeadPoseMetrics.add_frame_data should map index 0->yaw, 1->pitch, 2->roll
        
        # Frame 1: 10 deg yaw, 5 deg pitch, 0 roll
        metrics.add_frame_data((10.0, 5.0, 0.0))
        
        # Frame 2: 12 deg yaw, 6 deg pitch, 0 roll
        metrics.add_frame_data((12.0, 6.0, 0.0))
        
        # Frame 3: 8 deg yaw, 4 deg pitch, 0 roll
        metrics.add_frame_data((8.0, 4.0, 0.0))
        
        # Check internal state size
        self.assertEqual(len(metrics.pose_history), 3)
        self.assertEqual(metrics.pose_history[0]['yaw'], 10.0)
        
        # Test aggregate_metrics without arguments
        stats = metrics.aggregate_metrics()
        
        # Verify structure
        self.assertIsNotNone(stats)
        self.assertIn('yaw_std', stats)
        self.assertIn('pitch_std', stats)
        self.assertIn('roll_std', stats)
        self.assertIn('nod_count', stats)
        
        print(f"Stats computed: {stats}")

if __name__ == '__main__':
    unittest.main()
