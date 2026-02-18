import logging
import re

logger = logging.getLogger(__name__)

class LinguisticAnalyzer:
    def __init__(self):
        pass

    def add_frame(self, data: dict, timestamp: float):
        """No-op for linguistic analysis."""
        pass

    def finalize(self, transcript_text, language_code="en-US"):
        """
        Finalizes linguistic structure of the answer.
        Splits metrics into Universal (all languages) and English-specific.
        """
        if not transcript_text:
            return None
            
        # --- Universal Metrics (Language Agnostic) ---
        # 1. Sentence Parsing
        # Split by .!? followed by space or end of string
        sentences = re.split(r'[.!?]+(?:\s+|$)', transcript_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        sentence_count = len(sentences)
        
        # 2. Word Statistics
        # Basic word tokenization (approximation for non-space languages, but functional)
        words = re.findall(r'\b\w+\b', transcript_text.lower())
        total_words = len(words)
        unique_words = len(set(words))
        
        avg_sentence_length = 0.0
        if sentence_count > 0:
            avg_sentence_length = total_words / sentence_count
            
        lexical_diversity = 0.0
        if total_words > 0:
            lexical_diversity = unique_words / total_words
            
        # 3. Repetition Ratio (1 - lexical_diversity)
        repetition_ratio = 1.0 - lexical_diversity
        
        # --- English-Specific Metrics ---
        star_found = False
        example_found = False
        is_english = language_code and language_code.lower().startswith("en")
        
        if is_english:
            # 4. STAR Structure Detection
            star_keywords = ["situation", "task", "action", "result", "context", "challenge"]
            star_found = any(k in words for k in star_keywords)
            
            # 5. Example Usage
            example_phrases = ["for example", "for instance", "such as", "like when", "an example"]
            text_lower = transcript_text.lower()
            example_found = any(p in text_lower for p in example_phrases)
        
        return {
            'sentence_count': sentence_count,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'lexical_diversity': round(lexical_diversity, 2),
            'repetition_ratio': round(repetition_ratio, 2),
            'star_structure_flag': star_found if is_english else None,
            'example_usage_flag': example_found if is_english else None,
            'is_english_analysis': is_english
        }
