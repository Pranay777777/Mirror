import sys
import os
import logging

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from features.linguistic_analysis import LinguisticAnalyzer
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_linguistic_metrics():
    print("\n--- Testing Linguistic Analyzer ---")
    analyzer = LinguisticAnalyzer()
    
    # Test 1: English
    text_en = "This is a test. For example, I used the STAR method to solve a problem. It was a challenge."
    res_en = analyzer.analyze_answer_structure(text_en, "en-US")
    print("\nEnglish Result:")
    for k, v in res_en.items():
        print(f"  {k}: {v}")
        
    assert res_en['sentence_count'] == 3
    assert res_en['star_structure_flag'] == True
    assert res_en['example_usage_flag'] == True
    assert res_en['is_english_analysis'] == True
    
    # Test 2: Non-English (Spanish)
    text_es = "Esto es una prueba. Por ejemplo, use el metodo STAR."
    res_es = analyzer.analyze_answer_structure(text_es, "es-MX")
    print("\nSpanish Result:")
    for k, v in res_es.items():
        print(f"  {k}: {v}")
        
    assert res_es['sentence_count'] >= 2 # "prueba. Por"
    assert res_es['star_structure_flag'] is None
    assert res_es['example_usage_flag'] is None
    assert res_es['is_english_analysis'] == False
    
    print("\nLinguistic Analysis: PASSED")

if __name__ == "__main__":
    try:
        test_linguistic_metrics()
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
