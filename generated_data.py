import pandas as pd
import numpy as np
from typing import Tuple  # Tambahkan import Tuple

def generate_sample_data() -> Tuple[pd.DataFrame, list]:  # Tambahkan type hint
    # Data artikel contoh dalam bahasa Prancis
    articles = [
        {
            "id": 1,
            "title": "L'économie française en 2024",
            "paragraphs": [
                "La croissance économique de la France montre des signes positifs cette année.",
                "Cependant, les défis persistent dans le secteur industriel.",
                "Par conséquent, le gouvernement annonce de nouvelles mesures de soutien.",
                "En outre, les exportations augmentent régulièrement depuis le trimestre dernier.",
                "En conclusion, l'avenir semble prometteur malgré certains obstacles."
            ]
        },
        {
            "id": 2, 
            "title": "Les élections européennes",
            "paragraphs": [
                "Les campagnes électorales battent leur plein à travers l'Europe.",
                "Tout d'abord, les citoyens expriment leurs préoccupations environnementales.",
                "De plus, les questions migratoires dominent les débats dans plusieurs pays.",
                "Cependant, la participation électorale reste une inquiétude majeure.",
                "Finalement, les résultats détermineront l'orientation future de l'Union."
            ]
        }
    ]
    
    # Generate sample transitions dataset
    transitions_data = []
    transition_id = 1
    
    for article in articles:
        for i in range(len(article["paragraphs"]) - 1):
            transition_text = article["paragraphs"][i].split()[-3:]  # Last few words as transition
            transition_text = " ".join(transition_text)
            
            transitions_data.append({
                "article_id": article["id"],
                "paragraph_index": i,
                "transition_text": transition_text,
                "previous_paragraph": article["paragraphs"][i],
                "next_paragraph": article["paragraphs"][i + 1],
                "is_concluding": (i == len(article["paragraphs"]) - 2)
            })
            
            transition_id += 1
    
    return pd.DataFrame(transitions_data), articles

if __name__ == "__main__":
    df, articles = generate_sample_data()
    df.to_csv("test_data/sample_transitions.csv", index=False)
    print("Sample data generated successfully!")