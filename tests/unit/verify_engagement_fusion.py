"""verify_engagement_fusion.py — Tests the ECR/SAS/ASF gaze fusion formula."""

def fusion(hos: float, sas: float, asf_score: float) -> float:
    """Geometric fusion of engagement signals."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
    return (hos ** 0.55) * (sas ** 0.25) * (asf_score ** 0.20)

if __name__ == "__main__":
    print("[ENGAGEMENT FUSION]")
    cases = [
        (1.0, 1.0, 1.0, "full engagement"),
        (0.9, 0.8, 0.9, "high engagement"),
        (0.5, 0.5, 0.5, "moderate engagement"),
        (0.1, 0.2, 0.3, "low engagement"),
    ]
    for hos, sas, asf, label in cases:
        score = fusion(hos, sas, asf) * 10.0
        print(f"  {label}: hos={hos} sas={sas} asf={asf} → eng_score={score:.2f}/10")
