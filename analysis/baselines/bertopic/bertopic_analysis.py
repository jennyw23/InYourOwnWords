"""
bertopic/bertopic_analysis.py — BERTopic baseline on free-text identity responses.

Uses HDBSCAN (min_cluster_size=7) with pre-computed OpenAI embeddings, then
relabels clusters with GPT-4o-mini representations.

Pre-trained models are provided in bertopic_models/. Run with --overwrite to retrain. Retraining may lead to slight differences in clusters and topic labels due to the non-deterministic nature of HDBSCAN and GPT.

Requires:
    OPENAI_API_KEY environment variable (set in .env at repo root)

Usage:
    python bertopic/bertopic_analysis.py
    python bertopic/bertopic_analysis.py --overwrite
"""

import os
import argparse
import numpy as np
import pandas as pd
import openai
import hdbscan
from bertopic import BERTopic
from bertopic.representation import OpenAI as OpenAIRepresentation
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_FILE = ROOT / "data" / "in_your_own_words.csv"
EMB_DIR = ROOT / "data" / "embeddings"

TOPIC_OUTPUT_DIR = Path(__file__).parent / "bertopic_topics"
MODEL_OUTPUT_DIR = Path(__file__).parent / "bertopic_models"

MIN_CLUSTER_SIZE = 7


# ---------------------------------------------------------------------------
# Clustering + GPT relabeling
# ---------------------------------------------------------------------------
def fit_bertopic(docs, embeddings, min_cluster_size=MIN_CLUSTER_SIZE):
    cluster_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    topic_model = BERTopic(hdbscan_model=cluster_model)
    topic_model.fit_transform(docs, embeddings=embeddings)
    topic_df = topic_model.get_topic_info()
    print(f"  Topics found: {topic_df.Topic.nunique()} (including noise cluster -1)")
    return topic_model, topic_df


def add_gpt_labels(topic_model, docs):
    """Relabel BERTopic clusters using GPT-4o-mini representations; same
    model used for In Your Own Words."""
    client = openai.OpenAI()
    representation_model = OpenAIRepresentation(client, model="gpt-4o-mini", chat=True)
    topic_model.update_topics(docs, representation_model=representation_model)
    return topic_model, topic_model.get_topic_info()


def load_or_train(model_dir, docs, embeddings, overwrite=False):
    if not overwrite and os.path.exists(model_dir):
        print(f"  Loading model from {model_dir}")
        return BERTopic.load(model_dir), None
    print(f"  Training model → {model_dir}")
    model, topic_df = fit_bertopic(docs, embeddings)
    model.save(model_dir)
    return model, topic_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="BERTopic baseline on identity responses.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Retrain models even if checkpoints exist")
    return parser.parse_args()


def main():
    args = parse_args()

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Survey data not found: {DATA_FILE}\nSee data/README.md for access instructions.")
    if not EMB_DIR.is_dir():
        raise FileNotFoundError(f"Embeddings directory not found: {EMB_DIR}\nSee data/README.md for access instructions.")

    TOPIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    race_docs = df.race_open.fillna("").tolist()
    gender_docs = df.gender_open.fillna("").tolist()
    so_docs = df.sexuality_open.fillna("").tolist()

    print("Loading embeddings...")
    race_embeddings = np.load(EMB_DIR / "race_embeddings.npy")
    gender_embeddings = np.load(EMB_DIR / "gender_embeddings.npy")
    so_embeddings = np.load(EMB_DIR / "sexual_orientation_embeddings.npy")

    identities = [
        ("race", race_docs, race_embeddings),
        ("gender", gender_docs, gender_embeddings),
        ("sexual_orientation", so_docs, so_embeddings),
    ]

    for identity, docs, embeddings in identities:
        print(f"\n{'─'*60}")
        print(f"Identity: {identity}  (min_cluster_size={MIN_CLUSTER_SIZE})")
        print("─" * 60)

        model_dir = str(MODEL_OUTPUT_DIR / f"bertopic_{identity}_model")
        gpt_model_dir = model_dir + "_gpt_topics"

        # Step 1: cluster
        model, _ = load_or_train(model_dir, docs, embeddings, overwrite=args.overwrite)

        # Step 2: GPT relabeling
        if not args.overwrite and os.path.exists(gpt_model_dir):
            print(f"  Loading GPT-labeled model from {gpt_model_dir}")
            gpt_model = BERTopic.load(gpt_model_dir)
        else:
            print("  Adding GPT-4o-mini topic labels...")
            gpt_model, _ = add_gpt_labels(model, docs)
            gpt_model.representation_model = None  # strip unpicklable object before saving
            gpt_model.save(gpt_model_dir)

        topic_df = gpt_model.get_topic_info().drop(columns=["Name", "Representative_Docs"])

        out_path = TOPIC_OUTPUT_DIR / f"bertopic_{identity}_gpt_topics.csv"
        topic_df.to_csv(out_path, index=False)
        print(f"\n  Saved topics: {out_path.relative_to(Path(__file__).parent)}")
        print(topic_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
