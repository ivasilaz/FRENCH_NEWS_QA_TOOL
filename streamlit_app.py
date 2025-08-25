import re
import math
import streamlit as st
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
MIN_WORDS = 80
MAX_WORDS = 160
CONCLUSION_MARKERS = [
    "en conclusion", "pour conclure", "en résumé", "pour résumer",
    "en somme", "en définitive", "finalement", "pour finir"
]

st.set_page_config(page_title="French News QA – Verification Demo", layout="centered")

st.title("French News QA – Verification Demo")
st.caption("Checks: repetition • thematic cohesion • word limit • concluding placement")

# ========== Replace this with your real QA engine ==========
def generate_answer_stub(question: str, context: str) -> str:
    # TODO: import & call your real engine here, e.g. from app.py / article_parser.py
    # from your_module import qa_infer
    # return qa_infer(question, context)
    return ("En résumé, " + context)[:120] + "."

# ===========================================================

def sentences_fr(text: str) -> List[str]:
    # simple French-ish splitter (avoid heavy deps for a demo)
    chunks = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    return [c.strip() for c in chunks if c.strip()]

def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))

def repetition_score(text: str) -> Tuple[float, float]:
    """Returns (type_token_ratio, repeated_ngram_ratio). Higher TTR = less repetition."""
    tokens = [t.lower() for t in re.findall(r"\w+", text, flags=re.UNICODE)]
    if not tokens:
        return 1.0, 0.0
    ttr = len(set(tokens)) / len(tokens)

    # n-gram repetition (bigrams + trigrams)
    def ngram_counts(n):
        grams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        return grams
    bigrams = ngram_counts(2)
    trigrams = ngram_counts(3)
    rep_big = sum(1 for g in set(bigrams) if bigrams.count(g) > 1)
    rep_tri = sum(1 for g in set(trigrams) if trigrams.count(g) > 1)
    total = max(len(set(bigrams)) + len(set(trigrams)), 1)
    rep_ratio = (rep_big + rep_tri) / total
    return ttr, rep_ratio

def thematic_cohesion(text: str) -> float:
    """Average cosine similarity between consecutive sentences using TF-IDF."""
    sents = sentences_fr(text)
    if len(sents) < 2:
        return 1.0
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(sents)
    sims = []
    for i in range(len(sents)-1):
        sim = cosine_similarity(X[i], X[i+1])[0,0]
        sims.append(sim)
    return float(sum(sims) / len(sims))

def has_concluding_placement(text: str) -> bool:
    """Checks that the last sentence looks like a conclusion."""
    sents = sentences_fr(text)
    if not sents:
        return False
    last = sents[-1].lower()
    # Heuristic: contains marker OR summarizes (starts with hence/therefore-ish)
    marker_hit = any(m in last for m in CONCLUSION_MARKERS)
    # backup heuristic: length and contains wrap-up phrases
    wrap_hit = bool(re.search(r"\b(en bref|ainsi|donc|par conséquent)\b", last))
    return marker_hit or wrap_hit

def status_badge(ok: bool) -> str:
    return "✅ PASS" if ok else "❌ FAIL"

with st.form("qa_form"):
    st.subheader("Input")
    question = st.text_input("Question (FR or EN):", value="Quel est le point principal de l'article ?")
    context = st.text_area("Article / Contexte (FR):", height=200, placeholder="Collez ici l'article français…")
    mode = st.radio("Réponse", ["Générer avec mon moteur (démo)", "Je colle la réponse manuellement"], horizontal=True)
    manual_answer = ""
    if mode == "Je colle la réponse manuellement":
        manual_answer = st.text_area("Votre réponse (FR):", height=140)
    submitted = st.form_submit_button("Run QA Checks")

if submitted:
    if mode == "Générer avec mon moteur (démo)":
        answer = generate_answer_stub(question, context)
    else:
        answer = manual_answer.strip()
    st.markdown("### Output")
    st.write(answer if answer else "_(no answer)_")

    wc = word_count(answer)
    ttr, rep_ratio = repetition_score(answer)
    cohesion = thematic_cohesion(answer)
    concluding_ok = has_concluding_placement(answer)

    # RULES (tune thresholds as needed)
    word_limit_ok = (MIN_WORDS <= wc <= MAX_WORDS)
    repetition_ok = (ttr >= 0.45) and (rep_ratio <= 0.12)     # fewer repeats → pass
    cohesion_ok = (cohesion >= 0.22)                          # higher cohesion → pass

    st.markdown("### Checks")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Word limit ({MIN_WORDS}-{MAX_WORDS})**: {wc} → {status_badge(word_limit_ok)}")
        st.write(f"**Repetition (TTR ≥ 0.45 & n-gram rep ≤ 0.12)**: TTR={ttr:.2f}, rep={rep_ratio:.2f} → {status_badge(repetition_ok)}")
    with col2:
        st.write(f"**Thematic cohesion (avg cosine ≥ 0.22)**: {cohesion:.2f} → {status_badge(cohesion_ok)}")
        st.write(f"**Concluding placement (last sentence)**: {status_badge(concluding_ok)}")

    st.divider()
    st.caption("Notes: TTR = type-token ratio. Cohesion uses TF-IDF cosine similarity between consecutive sentences.")

st.markdown("---")
st.caption("Demo app for verification. Replace the stub with your real QA engine call before sharing.")
