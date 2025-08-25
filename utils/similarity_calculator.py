# similarity_calculator.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict

class SimilarityCalculator:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def calculate_similarity(self, transition: str, previous_para: str, next_para: str) -> Dict[str, float]:
        """Calculate similarity scores between transition and paragraphs"""
        # Encode all texts
        texts_to_encode = [transition, previous_para, next_para]
        embeddings = self.model.encode(texts_to_encode)
        
        # Calculate similarities
        sim_next = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
        sim_prev = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Calculate coherence score (how well transition connects both paragraphs)
        sim_bridge = cosine_similarity([embeddings[1]], [embeddings[2]])[0][0]
        
        return {
            "similarity_next": round(sim_next, 3),
            "similarity_prev": round(sim_prev, 3),
            "similarity_difference": round(sim_next - sim_prev, 3),
            "coherence_score": round(sim_bridge, 3),
            "overall_score": round((sim_next + sim_bridge) / 2, 3)
        }