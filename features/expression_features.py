"""
Geometric Facial Expression Scoring (v2.0 — Calibrated)

Pure landmark-based expression classification using MediaPipe FaceMesh.
No CNN. No external emotion models.

3-band calibrated scoring:
    smile_ratio >= 0.61  → happy   (1.0)
    smile_ratio >= 0.55  → neutral (0.6)
    smile_ratio <  0.55  → low     (0.3)

expression_score = mean(frame_scores) * 100
"""

from typing import Dict, List, Optional
import math


class ExpressionFeatures:
    """Accumulates per-frame expression landmark ratios and classifies."""

    # Calibrated thresholds (from fhappy/fneutral/fsad dataset)
    SMILE_HIGH = 0.61
    SMILE_MID = 0.55

    def __init__(self):
        self._smile_ratios: List[float] = []
        self._brow_distances: List[float] = []
        self._eye_openness: List[float] = []
        self._frame_labels: List[str] = []
        self._frame_scores: List[float] = []

    def add_frame(self, geo_results: Dict, timestamp: float = 0.0) -> None:
        """
        Accumulate expression ratios from a single frame's geometry output.

        Args:
            geo_results: Dict from NormalizedGeometry.process(), must contain
                         'expr_smile_ratio', 'expr_brow_distance', 'expr_eye_openness'.
            timestamp: Frame timestamp (unused, kept for interface consistency).
        """
        smile = geo_results.get('expr_smile_ratio')
        brow = geo_results.get('expr_brow_distance')
        eye = geo_results.get('expr_eye_openness')

        if smile is None:
            return  # Skip frames without face data

        self._smile_ratios.append(float(smile))
        if brow is not None:
            self._brow_distances.append(float(brow))
        if eye is not None:
            self._eye_openness.append(float(eye))

        # Per-frame 3-band classification
        if smile >= self.SMILE_HIGH:
            self._frame_labels.append('happy')
            self._frame_scores.append(1.0)
        elif smile >= self.SMILE_MID:
            self._frame_labels.append('neutral')
            self._frame_scores.append(0.6)
        else:
            self._frame_labels.append('low_positive')
            self._frame_scores.append(0.3)

    def finalize(self) -> Dict:
        """
        Compute aggregate expression analysis.

        Returns:
            Dict with expression_score, ratios, and diagnostics.
        """
        n = len(self._frame_labels)
        if n == 0:
            return {
                "expression_score": 0.0,
                "happy_ratio": 0.0,
                "neutral_ratio": 0.0,
                "low_positive_ratio": 0.0,
                "mean_smile_ratio": 0.0,
                "std_smile_ratio": 0.0,
                "smile_high": self.SMILE_HIGH,
                "smile_mid": self.SMILE_MID,
                "logic_mode": "geometric_expression_v2_calibrated"
            }

        happy_count = self._frame_labels.count('happy')
        neutral_count = self._frame_labels.count('neutral')
        low_count = self._frame_labels.count('low_positive')

        expression_score = (sum(self._frame_scores) / n) * 100.0

        mean_smile = sum(self._smile_ratios) / len(self._smile_ratios)
        # std
        if len(self._smile_ratios) > 1:
            variance = sum((x - mean_smile) ** 2 for x in self._smile_ratios) / (len(self._smile_ratios) - 1)
            std_smile = math.sqrt(variance)
        else:
            std_smile = 0.0

        return {
            "expression_score": round(float(expression_score), 2),
            "happy_ratio": round(happy_count / n, 4),
            "neutral_ratio": round(neutral_count / n, 4),
            "low_positive_ratio": round(low_count / n, 4),
            "mean_smile_ratio": round(float(mean_smile), 4),
            "std_smile_ratio": round(float(std_smile), 4),
            "smile_high": self.SMILE_HIGH,
            "smile_mid": self.SMILE_MID,
            "logic_mode": "geometric_expression_v2_calibrated"
        }
