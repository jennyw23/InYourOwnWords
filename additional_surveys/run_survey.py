"""
Run the framework on one additional survey dataset (Supplementary H).

For each M in {16, 32, 64} and K in {1, 4}: train an SAE, interpret every dimension
with GPT-4o (3 candidates), and score fidelity with GPT-4.1-mini. The configuration
with the highest mean fidelity is then annotated on every response to get theme
prevalence. Requires an OpenAI key; finished runs are skipped on re-run.

Outputs (results/):
    {name}_M={M}_K={K}_recall_specificity_hmean_interpretation_fidelity.csv
    {name}_M={M}_K={K}_prevalence_comparison.csv      (best configuration only)

Usage:
    python additional_surveys/run_survey.py --dataset anes
    python additional_surveys/run_survey.py --dataset estate
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "src"))
load_dotenv(REPO / ".env")

import hypothesaes.annotate as _annotate_mod
import hypothesaes.embedding as _embed_mod
import hypothesaes.interpret_neurons as _interp_mod
import hypothesaes.quickstart as _qs
import hypothesaes.sae as _sae
from hypothesaes.annotate import annotate, annotate_texts_with_concepts
from hypothesaes.embedding import get_openai_embeddings
from hypothesaes.interpret_neurons import (
    InterpretConfig, LLMConfig, NeuronInterpreter, SamplingConfig, ScoringConfig,
)
from hypothesaes.quickstart import train_sae

from fidelity import FIDELITY_METRIC, patch_hypothesaes

DATASETS = {
    "anes": {
        "name": "anes_mip",
        "csv": "cleaned_data/anes_mip_open_ended.csv",
        "text_col": "mip_text",
        "instructions": """All of the texts are responses to the question:
"What do you think are the most important problems facing this country?" -- an open-ended
item from the American National Election Studies (ANES) time series survey.
Features should describe a specific aspect of the response.
""",
    },
    "estate": {
        "name": "estate_why_reduce_inequality",
        "csv": "cleaned_data/stantcheva_estate_tax_opinion_cleaned.csv",
        "text_col": "estate_why_reduce_inequality_text",
        "instructions": """All of the texts are responses to the question:
"Why?" -- a follow-up asking respondents to explain why they think increasing the federal
estate tax is a good or bad way to reduce wealth inequality.
Features should describe a specific aspect of the response.
""",
    },
}
M_VALUES, K_VALUES = [16, 32, 64], [1, 4]
EMBEDDING_MODEL = "text-embedding-3-large"
INTERPRETER_MODEL, ANNOTATOR_MODEL = "gpt-4o", "gpt-4.1-mini"

RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "sae_cache"
SCORE_COL = f"{FIDELITY_METRIC}_fidelity_score"

# keep caches inside this folder; each hypothesaes module binds CACHE_DIR at import
_interp_mod.CACHE_DIR = _annotate_mod.CACHE_DIR = str(CACHE_DIR / "annotation_cache")
_embed_mod.CACHE_DIR = str(CACHE_DIR / "embedding_cache")
patch_hypothesaes()


def _load_model_compat(path, device="cuda" if torch.cuda.is_available() else "cpu"):
    ckpt = torch.load(path, pickle_module=pickle, weights_only=False)
    model = _sae.SparseAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model


_sae.load_model = _qs.load_model = _load_model_compat


def run_name(name, M, K):
    return f"{name}_M={M}_K={K}"


def fidelity_path(name, M, K):
    return RESULTS_DIR / f"{run_name(name, M, K)}_{FIDELITY_METRIC}_interpretation_fidelity.csv"


def load_texts_and_embeddings(cfg):
    df = pd.read_csv(ROOT / cfg["csv"])
    texts = df[cfg["text_col"]].dropna().tolist()
    emb_path = CACHE_DIR / f"{cfg['name']}_{EMBEDDING_MODEL}_embeddings.npy"
    if emb_path.exists():
        embeddings = np.load(emb_path)
    else:
        text2emb = get_openai_embeddings(texts=texts, model=EMBEDDING_MODEL,
                                         cache_name=f"{cfg['name']}_{EMBEDDING_MODEL}")
        embeddings = np.stack([text2emb[t] for t in texts])
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embeddings)
    print(f"{cfg['name']}: {len(texts)} responses, embeddings {embeddings.shape}")
    return texts, embeddings


def interpret_and_score(sae, texts, embeddings, M, cache_name, instructions):
    """Best of 3 GPT-4o candidates per dimension, chosen by fidelity."""
    activations = np.asarray(sae.get_activations(embeddings.astype(np.float32)))
    interpreter = NeuronInterpreter(cache_name=cache_name, interpreter_model=INTERPRETER_MODEL,
                                    annotator_model=ANNOTATOR_MODEL,
                                    n_workers_interpretation=10, n_workers_annotation=30)
    config = InterpretConfig(
        sampling=SamplingConfig(n_examples=20, max_words_per_example=256),
        llm=LLMConfig(temperature=0.7, max_interpretation_tokens=50),
        n_candidates=3,
        task_specific_instructions=instructions,
    )
    interpretations = interpreter.interpret_neurons(texts=texts, activations=activations,
                                                    neuron_indices=list(range(M)), config=config)
    metrics = interpreter.score_interpretations(texts=texts, activations=activations,
                                                interpretations=interpretations,
                                                config=ScoringConfig(n_examples=100))
    rows = []
    for idx in range(M):
        best = max(interpretations[idx], key=lambda i: metrics[idx][i][FIDELITY_METRIC])
        rows.append({"neuron_idx": idx, "source_sae": (sae.m_total_neurons, sae.k_active_neurons),
                     "interpretation": best, SCORE_COL: metrics[idx][best][FIDELITY_METRIC]})
    return pd.DataFrame(rows)


def annotate_prevalence(texts, fidelity_df, cache_name, chunk_size=2000, n_workers=200):
    """Share of all responses the annotator labels as expressing each theme.

    Annotates in chunks because hypothesaes only writes its cache at the end of each call.
    """
    concepts = fidelity_df["interpretation"].tolist()
    tasks = [(t, c) for c in concepts for t in texts]
    cache_path = os.path.join(_annotate_mod.CACHE_DIR, f"{cache_name}_hypothesis-eval.json")
    for start in range(0, len(tasks), chunk_size):
        annotate(tasks=tasks[start:start + chunk_size], model=ANNOTATOR_MODEL,
                 cache_path=cache_path, n_workers=n_workers)
    arrays = annotate_texts_with_concepts(texts=texts, concepts=concepts, model=ANNOTATOR_MODEL,
                                          cache_name=cache_name, n_workers=n_workers, show_progress=False)
    return pd.DataFrame({
        "neuron_idx": fidelity_df["neuron_idx"],
        "interpretation": concepts,
        "pct_annotated_llm": [float(arrays[c].mean()) for c in concepts],
    })


def main(dataset):
    cfg = DATASETS[dataset]
    name = cfg["name"]
    RESULTS_DIR.mkdir(exist_ok=True)
    texts, embeddings = load_texts_and_embeddings(cfg)

    means = {}
    for M in M_VALUES:
        for K in K_VALUES:
            path = fidelity_path(name, M, K)
            if not path.exists():
                sae = train_sae(embeddings=embeddings, M=M, K=K,
                                checkpoint_dir=str(CACHE_DIR / "checkpoints" / run_name(name, M, K)))
                interpret_and_score(sae, texts, embeddings, M, run_name(name, M, K),
                                    cfg["instructions"]).to_csv(path, index=False)
            means[(M, K)] = pd.read_csv(path)[SCORE_COL].mean()
            print(f"  M={M} K={K}: mean fidelity {means[(M, K)]:.3f}")

    best_M, best_K = max(means, key=means.get)
    print(f"Best configuration: M={best_M}, K={best_K}")
    sae = train_sae(embeddings=embeddings, M=best_M, K=best_K,
                    checkpoint_dir=str(CACHE_DIR / "checkpoints" / run_name(name, best_M, best_K)))
    best_fid = pd.read_csv(fidelity_path(name, best_M, best_K))
    prevalence = annotate_prevalence(texts, best_fid, run_name(name, best_M, best_K))

    activations = np.asarray(sae.get_activations(embeddings.astype(np.float32)))
    prevalence["pct_active_activation_based"] = (activations > 0).mean(axis=0)[prevalence["neuron_idx"]]
    out = RESULTS_DIR / f"{run_name(name, best_M, best_K)}_prevalence_comparison.csv"
    prevalence.sort_values("pct_annotated_llm", ascending=False).to_csv(out, index=False)
    print(f"Saved: {out.relative_to(REPO)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    main(parser.parse_args().dataset)
