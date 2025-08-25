import spacy
import pandas as pd
from typing import List, Tuple  # Pastikan import Tuple juga

class FrenchTextProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("fr_core_news_sm")
        except OSError:
            raise Exception("Model French not found. Run: python -m spacy download fr_core_news_sm")
    
    def extract_lemmas(self, text: str) -> List[str]:
        """Extract lemmas from French text"""
        doc = self.nlp(text)
        return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    
    def check_word_count(self, text: str, max_words: int = 5) -> bool:
        """Check if text has 5 words or less"""
        return len(text.split()) <= max_words
    
    def find_repetitions(self, transitions: pd.DataFrame, article_paragraphs: List[str]) -> List[Tuple[str, int]]:
        """Find repeated lemmas across article"""
        all_lemmas = []
        
        # Extract lemmas from all paragraphs
        for paragraph in article_paragraphs:
            all_lemmas.extend(self.extract_lemmas(paragraph))
        
        # Extract lemmas from transitions
        for _, row in transitions.iterrows():
            all_lemmas.extend(self.extract_lemmas(row['transition_text']))
        
        # Count lemma frequency
        lemma_counts = pd.Series(all_lemmas).value_counts()
        return [(lemma, count) for lemma, count in lemma_counts.items() if count > 2]