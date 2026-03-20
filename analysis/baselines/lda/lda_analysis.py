"""
lda/lda_analysis.py — Train LDA baseline on free-text identity responses.

Trains tomotopy LDA with K=32, alpha=1.0, eta=0.1, 2000 iterations, seed=42.
Preprocessing: lowercase + remove punctuation/numbers (no stopword removal).
Post-training: top-50 high-frequency stopwords removed from displayed word lists.

Pre-trained models are provided in lda_models/. Run with --overwrite to retrain.

Usage:
    python lda/lda_analysis.py
    python lda/lda_analysis.py --overwrite
"""

import os
import argparse
import numpy as np
import pandas as pd
import tomotopy as tp
import spacy
from collections import Counter
from pathlib import Path

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_FILE = ROOT / "data" / "in_your_own_words.csv"

TOPIC_OUTPUT_DIR = Path(__file__).parent / "lda_topics"
MODEL_OUTPUT_DIR = Path(__file__).parent / "lda_models"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_simple(text):
    """Lowercase, tokenize, remove punctuation and numbers."""
    doc = nlp(text.lower())
    return [
        token.text.strip()
        for token in doc
        if not token.is_punct and not token.is_space and not token.like_num
    ]


def get_stopwords_from_corpus(*doc_lists):
    """Collect stop tokens across all corpora and return the top-50 as a set."""
    stop_words = []
    for docs in doc_lists:
        for d in docs:
            doc = nlp(d.lower())
            stop_words.extend(token.text for token in doc if token.is_stop)
    top_counts = Counter(stop_words).most_common(50)
    return {word for word, _ in top_counts}


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def fit_lda(
    tokenized_docs,
    K=32,
    alpha=1.0,
    eta=0.1,
    iterations=2000,
    seed=42,
    min_doc_tokens=1,
    print_every=200,
):
    lda_model = tp.LDAModel(k=K, alpha=alpha, eta=eta, seed=seed)

    skipped = 0
    for doc in tokenized_docs:
        if doc is None or len(doc) < min_doc_tokens:
            skipped += 1
            continue
        lda_model.add_doc(doc)
    if skipped:
        print(f"  Skipped {skipped} documents with <{min_doc_tokens} tokens")

    for i in range(iterations):
        lda_model.train(1)
        if print_every and ((i + 1) % print_every == 0 or i == iterations - 1):
            print(f"  Iteration {i+1}/{iterations} — LL per word: {lda_model.ll_per_word:.4f}")

    return lda_model


def load_or_train(model_path, tokenized_docs, overwrite=False):
    if not overwrite and os.path.exists(model_path):
        print(f"  Loading model from {model_path}")
        return tp.LDAModel.load(model_path)
    print(f"  Training model → {model_path}")
    model = fit_lda(tokenized_docs)
    model.save(model_path)
    return model


# ---------------------------------------------------------------------------
# Topic display
# ---------------------------------------------------------------------------
def get_topic_table(lda_model, top_n=10, stoplist=None):
    """Return a DataFrame of topics with top words (post-training stopword filter)."""
    if stoplist is None:
        stoplist = set()

    rows = []
    for k in range(lda_model.k):
        n_docs = sum(1 for doc in lda_model.docs if np.argmax(doc.get_topic_dist()) == k)
        raw_words_probs = lda_model.get_topic_words(k, top_n=top_n + len(stoplist))

        filtered_words = []
        for word, _ in raw_words_probs:
            if word not in stoplist:
                filtered_words.append(word)
            if len(filtered_words) == top_n:
                break

        rows.append({
            "Topic": k,
            "n_documents": n_docs,
            "Top Words": ", ".join(filtered_words),
        })

    return pd.DataFrame(rows).sort_values("Topic").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train LDA baseline on identity responses.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Retrain models even if checkpoints exist")
    return parser.parse_args()


def main():
    args = parse_args()

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Survey data not found: {DATA_FILE}\nSee data/README.md for access instructions.")

    TOPIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    race_docs = df.race_open.fillna("").tolist()
    gender_docs = df.gender_open.fillna("").tolist()
    so_docs = df.sexuality_open.fillna("").tolist()

    print("Preprocessing documents...")
    tokenized_race = [preprocess_simple(d) for d in race_docs]
    tokenized_gender = [preprocess_simple(d) for d in gender_docs]
    tokenized_so = [preprocess_simple(d) for d in so_docs]

    print("Computing post-training stopword list (top 50)...")
    stoplist = get_stopwords_from_corpus(race_docs, gender_docs, so_docs)
    print(f"  Stoplist size: {len(stoplist)}")

    identities = [
        ("race", tokenized_race, "lda_race_model.bin"),
        ("gender", tokenized_gender, "lda_gender_model.bin"),
        ("sexual_orientation", tokenized_so, "lda_so_model.bin"),
    ]

    for identity, tokenized_docs, model_file in identities:
        print(f"\n{'─'*60}")
        print(f"Identity: {identity}")
        print("─" * 60)

        model_path = str(MODEL_OUTPUT_DIR / model_file)
        model = load_or_train(model_path, tokenized_docs, overwrite=args.overwrite)

        topic_df = get_topic_table(model, top_n=10, stoplist=stoplist)

        out_path = TOPIC_OUTPUT_DIR / f"{identity}_lda_topics.csv"
        topic_df.to_csv(out_path, index=False)
        print(f"\n  Saved topics: {out_path.relative_to(Path(__file__).parent)}")
        print(topic_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
