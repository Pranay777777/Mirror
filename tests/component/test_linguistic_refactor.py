"""test_linguistic_refactor.py — Validates that LinguisticAnalyzer returns expected fields."""
from dotenv import load_dotenv
load_dotenv()

SAMPLE = "Good morning. I believe effective communication requires clarity and confidence. Thank you."

def main():
    from features.linguistic_analysis import LinguisticAnalyzer
    la = LinguisticAnalyzer()
    result = la.finalize(SAMPLE)
    required = ["sentence_count", "grammar_score", "formal_vocabulary_density", "sentiment"]
    for k in required:
        status = "✓" if k in result else "✗ MISSING"
        print(f"  {status} {k}: {result.get(k)}")

if __name__ == "__main__":
    main()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
