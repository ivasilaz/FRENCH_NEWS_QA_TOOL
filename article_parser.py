# article_parser.py
import re
import os
from typing import List, Dict, Any
import pandas as pd

class ArticleParser:
    def __init__(self):
        self.patterns = {
            'title': r"Titre:\s*(.+)",
            'chapeau': r"Chapeau:\s*(.+)",
            'article': r"Article:(.+?)Transitions générées:",
            'transitions': r"Transitions générées:\s*(.+)",
            'transition_items': r"(\d+\.\s*.+)"
        }
    
    def parse_article_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a single article file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            article_data = {
                'file_name': os.path.basename(file_path),
                'title': self._extract_pattern(content, 'title'),
                'chapeau': self._extract_pattern(content, 'chapeau'),
                'article_text': self._extract_pattern(content, 'article'),
                'transitions': self._extract_transitions(content)
            }
            
            # Extract paragraphs from article text
            if article_data['article_text']:
                article_data['paragraphs'] = self._extract_paragraphs(article_data['article_text'])
            
            return article_data
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {str(e)}")
            return None
    
    def _extract_pattern(self, content: str, pattern_key: str) -> str:
        """Extract text using regex pattern"""
        match = re.search(self.patterns[pattern_key], content, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _extract_transitions(self, content: str) -> List[str]:
        """Extract transitions from content"""
        transitions_text = self._extract_pattern(content, 'transitions')
        if not transitions_text:
            return []
        
        # Split transitions by numbered items
        transition_items = re.findall(self.patterns['transition_items'], transitions_text)
        transitions = []
        for item in transition_items:
            # Remove the numbering
            transition = re.sub(r'^\d+\.\s*', '', item).strip()
            transitions.append(transition)
        
        return transitions
    
    def _extract_paragraphs(self, article_text: str) -> List[str]:
        """Extract paragraphs from article text"""
        # Split by double newlines and clean up
        paragraphs = [p.strip() for p in article_text.split('\n\n') if p.strip()]
        return paragraphs
    
    def parse_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Parse all article files in a directory"""
        articles = []
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                article_data = self.parse_article_file(file_path)
                if article_data:
                    articles.append(article_data)
        
        return articles

# Update generated.py to use real data
def generate_sample_data_from_articles(articles_dir: str) -> Tuple[pd.DataFrame, list]:
    """Generate dataset from parsed articles"""
    parser = ArticleParser()
    articles = parser.parse_directory(articles_dir)
    
    transitions_data = []
    
    for article in articles:
        article_id = hash(article['title']) % 1000  # Simple ID generation
        
        if 'paragraphs' in article and article['transitions']:
            # Create transitions between paragraphs
            for i, transition_text in enumerate(article['transitions']):
                if i < len(article['paragraphs']) - 1:
                    transitions_data.append({
                        "article_id": article_id,
                        "paragraph_index": i,
                        "transition_text": transition_text,
                        "previous_paragraph": article['paragraphs'][i] if i > 0 else "",
                        "next_paragraph": article['paragraphs'][i + 1],
                        "is_concluding": (i == len(article['paragraphs']) - 2),
                        "article_title": article['title']
                    })
    
    return pd.DataFrame(transitions_data), articles