import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# ==================== ARTICLE PARSER ====================
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
            st.error(f"Error parsing file {file_path}: {str(e)}")
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
        # Further split by single newlines if they represent paragraph breaks
        refined_paragraphs = []
        for paragraph in paragraphs:
            if '\n' in paragraph and len(paragraph) > 200:  # Likely multiple paragraphs
                lines = [line.strip() for line in paragraph.split('\n') if line.strip()]
                refined_paragraphs.extend(lines)
            else:
                refined_paragraphs.append(paragraph)
        return refined_paragraphs
    
    def parse_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Parse all article files in a directory"""
        articles = []
        
        if not os.path.exists(directory_path):
            st.error(f"Directory {directory_path} not found!")
            return articles
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                article_data = self.parse_article_file(file_path)
                if article_data:
                    articles.append(article_data)
        
        return articles

# ==================== FRENCH TEXT PROCESSOR ====================
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
    
    def check_word_count(self, text: str, max_words: int = 8, min_words: int = 2) -> bool:
        """Check if text has appropriate word count"""
        word_count = len(text.split())
        return min_words <= word_count <= max_words
    
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
        return [(lemma, count) for lemma, count in lemma_counts.items() if count > 3]

# ==================== RULE CHECKER ====================
class RuleChecker:
    def __init__(self):
        self.text_processor = FrenchTextProcessor()
        self.concluding_phrases = [
            "en conclusion", "finalement", "pour terminer", 
            "pour finir", "en guise de conclusion", "en définitive",
            "pour conclure", "en somme"
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
        
        # Rule 4: Coherence check
        results["coherence_pass"] = self._check_coherence(transition_text, article_paragraphs, paragraph_index)
        if not results["coherence_pass"]:
            results["rules_broken"].append("Coherence issue detected")
        
        # Rule 5: Variety check (avoid generic transitions)
        results["variety_pass"] = self._check_variety(transition_text)
        if not results["variety_pass"]:
            results["rules_broken"].append("Transition too generic")
        
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
    
    def _check_variety(self, transition_text: str) -> bool:
        """Check if transition has variety (not too generic)"""
        generic_transitions = [
            "également", "aussi", "de plus", "par ailleurs", 
            "ensuite", "puis", "alors", "donc"
        ]
        
        # If it's just a single generic word, fail the check
        words = transition_text.lower().split()
        if len(words) == 1 and words[0] in generic_transitions:
            return False
        
        return True

# ==================== SIMILARITY CALCULATOR ====================
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

# ==================== DATA GENERATION ====================
def generate_sample_data() -> Tuple[pd.DataFrame, list]:
    """Generate sample data for demonstration"""
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
        }
    ]
    
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

def generate_sample_data_from_articles(articles_dir: str) -> Tuple[pd.DataFrame, list]:
    """Generate dataset from parsed articles"""
    parser = ArticleParser()
    articles = parser.parse_directory(articles_dir)
    
    transitions_data = []
    
    for idx, article in enumerate(articles):
        article_id = idx + 1  # Simple sequential ID
        
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
                        "article_title": article['title'],
                        "source_file": article['file_name']
                    })
    
    return pd.DataFrame(transitions_data), articles

# ==================== STREAMLIT APP ====================
# Set page config
st.set_page_config(
    page_title="QA Tool for French News Transitions",
    page_icon="📰",
    layout="wide"
)

# Initialize classes dengan progress indicator
@st.cache_resource(show_spinner=False)
def load_models():
    # Buat placeholder untuk progress
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Loading French language model (spaCy)...")
        text_processor = FrenchTextProcessor()
        progress_bar.progress(25)
        time.sleep(0.5)
        
        status_text.text("Initializing rule checker...")
        rule_checker = RuleChecker()
        progress_bar.progress(50)
        time.sleep(0.5)
        
        status_text.text("Loading similarity model (Sentence Transformers)...")
        similarity_calculator = SimilarityCalculator()
        progress_bar.progress(75)
        time.sleep(0.5)
        
        status_text.text("Finalizing model initialization...")
        progress_bar.progress(100)
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        progress_placeholder.empty()
        
        return text_processor, rule_checker, similarity_calculator
        
    except Exception as e:
        progress_bar.empty()
        status_text.error(f"Error loading models: {str(e)}")
        st.error("""
        Please make sure you have:
        1. Installed required packages: pip install -r requirements.txt
        2. Downloaded spaCy French model: python -m spacy download fr_core_news_sm
        """)
        return None, None, None

# Load data function
@st.cache_data
def load_data():
    return generate_sample_data()

def display_article_analysis(selected_article, article_transitions, text_processor, rule_checker, similarity_calculator):
    """Display article analysis in tab 1"""
    st.header(f"Article: {selected_article['title']}")
    
    # Display article metadata
    if 'file_name' in selected_article:
        st.caption(f"Source: {selected_article['file_name']}")
    
    # Display article paragraphs
    for i, paragraph in enumerate(selected_article["paragraphs"]):
        st.markdown(f"**Paragraph {i+1}:** {paragraph}")
    
    # Analyze transitions
    results = []
    for _, row in article_transitions.iterrows():
        # Check rules
        rule_results = rule_checker.check_transition_rules(
            row["transition_text"], 
            row["paragraph_index"],
            len(selected_article["paragraphs"]),
            selected_article["paragraphs"]
        )
        
        # Calculate similarity
        similarity_results = similarity_calculator.calculate_similarity(
            row["transition_text"],
            row["previous_paragraph"],
            row["next_paragraph"]
        )
        
        # Combine results
        result = {
            "article_id": row["article_id"],
            "para_idx": row["paragraph_index"],
            "transition_text": row["transition_text"],
            "word_count_pass": rule_results["word_count_pass"],
            "position_pass": rule_results["position_pass"],
            "coherence_pass": rule_results["coherence_pass"],
            "variety_pass": rule_results["variety_pass"],
            "similarity_next": similarity_results["similarity_next"],
            "similarity_prev": similarity_results["similarity_prev"],
            "similarity_difference": similarity_results["similarity_difference"],
            "coherence_score": similarity_results["coherence_score"],
            "overall_score": similarity_results["overall_score"],
            "all_rules_pass": (rule_results["word_count_pass"] and 
                              rule_results["position_pass"] and 
                              rule_results["coherence_pass"] and
                              rule_results["variety_pass"] and
                              (similarity_results["similarity_difference"] > 0)),
            "failure_reason": "; ".join(rule_results["rules_broken"]) if rule_results["rules_broken"] else "None"
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

def display_transition_details(results_df):
    """Display transition details in tab 2"""
    st.header("Detailed Transition Analysis")
    
    for _, result in results_df.iterrows():
        with st.expander(f"Transition {result['para_idx'] + 1}: {result['transition_text']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Similarity with Next Paragraph", f"{result['similarity_next']:.3f}")
                st.metric("Similarity with Previous Paragraph", f"{result['similarity_prev']:.3f}")
                st.metric("Similarity Difference", f"{result['similarity_difference']:.3f}", 
                         delta="Good" if result['similarity_difference'] > 0 else "Needs Improvement")
            
            with col2:
                st.metric("Coherence Score", f"{result['coherence_score']:.3f}")
                st.metric("Overall Score", f"{result['overall_score']:.3f}")
                st.metric("Word Count Check", "PASS" if result['word_count_pass'] else "FAIL")
            
            with col3:
                st.metric("Position Check", "PASS" if result['position_pass'] else "FAIL")
                st.metric("Coherence Check", "PASS" if result['coherence_pass'] else "FAIL")
                st.metric("Variety Check", "PASS" if result['variety_pass'] else "FAIL")
            
            if result['failure_reason'] != "None":
                st.error(f"**Failure Reasons:** {result['failure_reason']}")
            else:
                st.success("All checks passed! ✅")

def display_summary_statistics(results_df):
    """Display summary statistics in tab 3"""
    st.header("Summary Statistics")
    
    # Calculate overall metrics
    total_transitions = len(results_df)
    compliant_transitions = len(results_df[results_df["all_rules_pass"]])
    compliance_rate = (compliant_transitions / total_transitions) * 100 if total_transitions > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transitions", total_transitions)
    col2.metric("Compliant Transitions", compliant_transitions)
    col3.metric("Compliance Rate", f"{compliance_rate:.1f}%")
    
    # Visualizations
    if total_transitions > 0:
        # Similarity difference chart
        fig = px.bar(results_df, x="para_idx", y="similarity_difference",
                    title="Similarity Difference by Transition Position",
                    labels={"para_idx": "Paragraph Index", "similarity_difference": "Similarity Difference"})
        st.plotly_chart(fig)
        
        # Score distribution
        fig_scores = px.box(results_df, y="overall_score", 
                           title="Distribution of Overall Transition Scores")
        st.plotly_chart(fig_scores)
        
        # Most common issues
        st.subheader("Common Issues")
        word_count_failures = len(results_df[~results_df["word_count_pass"]])
        position_failures = len(results_df[~results_df["position_pass"]])
        coherence_failures = len(results_df[~results_df["coherence_pass"]])
        variety_failures = len(results_df[~results_df["variety_pass"]])
        similarity_failures = len(results_df[results_df["similarity_difference"] <= 0])
        
        issues_df = pd.DataFrame({
            "Issue Type": ["Word Count", "Position", "Coherence", "Variety", "Similarity"],
            "Count": [word_count_failures, position_failures, coherence_failures, variety_failures, similarity_failures]
        })
        
        if issues_df["Count"].sum() > 0:
            fig_issues = px.bar(issues_df, x="Issue Type", y="Count", 
                               title="Distribution of Compliance Issues",
                               color="Issue Type")
            st.plotly_chart(fig_issues)
        else:
            st.success("No compliance issues found! All transitions passed all checks. 🎉")

def main():
    st.title("📰 QA Tool for French News Transitions")
    st.markdown("Advanced tool for evaluating transition phrases in French news articles")
    
    # Load models dengan progress indicator
    with st.spinner("Loading NLP models. This may take a few minutes..."):
        text_processor, rule_checker, similarity_calculator = load_models()
    
    # Check if models loaded successfully
    if text_processor is None or rule_checker is None or similarity_calculator is None:
        st.error("Failed to load models. Please check the error messages above.")
        return
    
    # Sidebar for data source selection
    st.sidebar.header("Configuration")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Sample Data", "Real Articles Directory"]
    )
    
    transitions_df = pd.DataFrame()
    articles = []
    
    if data_source == "Real Articles Directory":
        articles_dir = st.sidebar.text_input("Path to articles directory:", "articles/")
        if st.sidebar.button("Load Articles") or st.session_state.get('articles_loaded', False):
            if os.path.exists(articles_dir):
                with st.spinner("Parsing articles and generating transitions..."):
                    transitions_df, articles = generate_sample_data_from_articles(articles_dir)
                if len(articles) > 0:
                    st.session_state.transitions_df = transitions_df
                    st.session_state.articles = articles
                    st.session_state.articles_loaded = True
                    st.success(f"✅ Loaded {len(articles)} articles with {len(transitions_df)} transitions")
                else:
                    st.error("No articles found in the directory or parsing failed.")
            else:
                st.error("Directory not found. Please create an 'articles/' folder and add your TXT files.")
    else:
        # Load sample data
        transitions_df, articles = load_data()
        st.session_state.transitions_df = transitions_df
        st.session_state.articles = articles
    
    # Check if we have data
    if 'transitions_df' not in st.session_state or st.session_state.transitions_df.empty:
        st.warning("Please load data using the sidebar options.")
        return
    
    transitions_df = st.session_state.transitions_df
    articles = st.session_state.articles
    
    # Article selection
    st.sidebar.header("Article Selection")
    article_options = [f"{art.get('title', f'Article {i+1}')} ({art.get('file_name', 'No file')})" 
                      for i, art in enumerate(articles)]
    
    selected_article_idx = st.sidebar.selectbox(
        "Select Article", 
        options=range(len(articles)),
        format_func=lambda x: article_options[x]
    )
    
    # Get selected article
    selected_article = articles[selected_article_idx]
    article_transitions = transitions_df[transitions_df["article_id"] == selected_article_idx + 1]
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Article Analysis", "Transition Details", "Summary Statistics"])
    
    with tab1:
        if not article_transitions.empty:
            results_df = display_article_analysis(selected_article, article_transitions, text_processor, rule_checker, similarity_calculator)
            
            # Display results table
            st.subheader("Transition Analysis Results")
            st.dataframe(results_df)
            
            # Download button
            if not results_df.empty:
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="transition_analysis_results.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No transitions found for this article.")
    
    with tab2:
        if not article_transitions.empty and 'results_df' in locals():
            display_transition_details(results_df)
        else:
            st.warning("No transition data to display")
    
    with tab3:
        if not article_transitions.empty and 'results_df' in locals():
            display_summary_statistics(results_df)
        else:
            st.warning("No data available for summary statistics")

if __name__ == "__main__":
    main()