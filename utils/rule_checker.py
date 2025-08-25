# rule_checker.py
from typing import Dict, Any, List
from .text_processor import FrenchTextProcessor

class RuleChecker:
    def __init__(self):
        self.text_processor = FrenchTextProcessor()
        self.concluding_phrases = [
            "en conclusion", "finalement", "pour terminer", 
            "pour finir", "en guise de conclusion", "en définitive"
        ]
    
    def check_transition_rules(self, transition_text: str, paragraph_index: int, 
                              total_paragraphs: int, article_paragraphs: List[str]) -> Dict[str, Any]:
        """Check all rules for a transition"""
        results = {
            "word_count_pass": False,
            "position_pass": False,
            "repetition_issues": [],
            "coherence_pass": False,
            "variety_pass": False,
            "rules_broken": []
        }
        
        # Rule 1: Word count check (2-8 words ideal for transitions)
        word_count = len(transition_text.split())
        results["word_count_pass"] = 2 <= word_count <= 8
        if not results["word_count_pass"]:
            results["rules_broken"].append(f"Word count ({word_count}) not in ideal range 2-8")
        
        # Rule 2: Position check for concluding transitions
        is_concluding = any(phrase in transition_text.lower() for phrase in self.concluding_phrases)
        results["position_pass"] = not (is_concluding and paragraph_index != total_paragraphs - 2)
        if not results["position_pass"]:
            results["rules_broken"].append("Concluding transition in wrong position")
        
        # Rule 3: Check for repetition issues
        results["repetition_issues"] = self._check_repetitions(transition_text, article_paragraphs)
        if results["repetition_issues"]:
            results["rules_broken"].append("Repetition issues detected")
        
        # Rule 4: Coherence check (transition should make sense contextually)
        results["coherence_pass"] = self._check_coherence(transition_text, article_paragraphs, paragraph_index)
        if not results["coherence_pass"]:
            results["rules_broken"].append("Coherence issue detected")
        
        return results
    
    def _check_repetitions(self, transition_text: str, article_paragraphs: List[str]) -> List[str]:
        """Check for repetitive words/phrases"""
        issues = []
        transition_lemmas = self.text_processor.extract_lemmas(transition_text)
        
        # Check repetition within the transition itself
        if len(transition_lemmas) != len(set(transition_lemmas)):
            issues.append("Repetition within transition")
        
        # Check repetition with article content
        all_article_lemmas = []
        for paragraph in article_paragraphs:
            all_article_lemmas.extend(self.text_processor.extract_lemmas(paragraph))
        
        # Count frequency of lemmas from transition in article
        for lemma in transition_lemmas:
            count = all_article_lemmas.count(lemma)
            if count > 3:  # Allow some repetition but not excessive
                issues.append(f"Word '{lemma}' overused in article")
        
        return issues
    
    def _check_coherence(self, transition_text: str, article_paragraphs: List[str], paragraph_index: int) -> bool:
        """Basic coherence check"""
        # Simple check: transition should not be too generic
        generic_phrases = ["également", "aussi", "de plus", "par ailleurs"]
        if any(phrase in transition_text.lower() for phrase in generic_phrases):
            # Check if it's appropriately used
            return len(transition_text.split()) > 2  # Not just a generic word
        
        return True