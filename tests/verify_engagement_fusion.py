import unittest
import numpy as np

class TestEngagementFusion(unittest.TestCase):
    """
    Verify Engagement Fusion Formula Logic.
    
    Formula:
    Score = (ECR^0.40) * (SAS^0.25) * (ASF_score^0.20) * (GSS^0.15)
    
    ASF_score = 1 / (1 + ASF*0.75)
    """

    def calculate_score(self, ecr, switch_count, max_continuous_s, total_duration_s, gss):
        asf = switch_count / max(total_duration_s, 1.0)
        asf_score = 1.0 / (1.0 + (asf * 0.75))
        
        sas = max_continuous_s / max(total_duration_s, 1.0)
        sas = min(1.0, max(0.0, sas))
        
        # Geometric Fusion
        score = (ecr ** 0.40) * (sas ** 0.25) * (asf_score ** 0.20) * (gss ** 0.15)
        return score * 10.0, score

    def test_ideal_engagement(self):
        """
        Ideal: 
        ECR=1.0 (Always looking)
        Switches=0
        MaxContinuous=TotalDuration
        GSS=1.0 (Stable)
        """
        score_10, score_norm = self.calculate_score(1.0, 0, 10.0, 10.0, 1.0)
        print(f"\nIdeal Engagement: {score_10:.2f} / 10.0")
        self.assertAlmostEqual(score_10, 10.0, places=2)

    def test_distracted_engagement(self):
        """
        Distracted:
        ECR=0.5 (Looking half time)
        Switches=10 (Frequent shifts)
        MaxContinuous=1.0s (Short bursts)
        Total=10.0s
        GSS=0.5 (Unstable)
        """
        # ASF = 10 / 10 = 1.0
        # ASF_score = 1 / (1 + 0.75) = 1/1.75 = 0.57
        # SAS = 1.0 / 10.0 = 0.1
        # GSS = 0.5
        # ECR = 0.5
        
        # Score = (0.5^0.4) * (0.1^0.25) * (0.57^0.2) * (0.5^0.15)
        #       = 0.757 * 0.562 * 0.893 * 0.901
        #       = 0.34
        
        score_10, score_norm = self.calculate_score(0.5, 10, 1.0, 10.0, 0.5)
        print(f"Distracted Engagement: {score_10:.2f} / 10.0")
        self.assertLess(score_10, 4.0, "Should be poor/limited.")

    def test_switch_penalty(self):
        """
        Verify that switches penalize the score even if ECR is high.
        ECR=0.9
        Switches=20 (Very high frequency)
        Total=10s
        """
        # ASF = 2.0
        # ASF_score = 1/2.5 = 0.4
        score_10, score_norm = self.calculate_score(0.9, 20, 5.0, 10.0, 0.9)
        print(f"High Switch Penalty: {score_10:.2f} / 10.0")
        
        # Compare with same metrics but 0 switches
        score_clean, _ = self.calculate_score(0.9, 0, 5.0, 10.0, 0.9)
        print(f"Clean (0 switches): {score_clean:.2f} / 10.0")
        
        self.assertLess(score_10, score_clean * 0.9, "Switches should significantly reduce score.")

if __name__ == '__main__':
    unittest.main()
