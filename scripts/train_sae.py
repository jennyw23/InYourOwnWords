"""
scripts/train_sae.py — Train Sparse Autoencoders on free-text identity responses.

Pre-trained SAE weights and fidelity scores are already provided in data/, so this script is only shown to reproduce the training pipeline from scratch.

Pre-computed embeddings are required and are distributed separately from the
survey data (see data/README.md for access instructions).  The embeddings
directory must contain the following .npy files (all N=1004 respondents):

    response_ids.npy                          (1004,)       — shared ResponseId order
    race_embeddings.npy                       (1004, 3072)
    gender_embeddings.npy                     (1004, 3072)
    sexual_orientation_embeddings.npy         (1004, 3072)

These were produced with OpenAI text-embedding-3-large (3072 dimensions).

What this script does:
  1. Loads pre-computed embeddings from the embeddings directory.
  2. Trains an SAE with M=32 neurons and K=4 active neurons per input.
  3. Interprets each SAE neuron via an LLM (GPT-4o by default).
  4. Scores interpretation fidelity with an annotator LLM (GPT-4.1-mini).
  5. Saves fidelity scores to data/fidelity/{identity}_interpretation_fidelity.csv.
  6. Saves all 3 candidate interpretations per neuron, scored under both the recall/specificity
     (v1) and recall/precision (v2) fidelity formula, to
     data/f1_metric_comparison/{identity}_all_candidates.csv.
  7. Times the SAE-only stages (train/load checkpoint, compute activations —
     NOT the LLM interpretation/annotation steps) and appends the results,
     along with hardware info, to data/runtime_log.csv — use this to report
     the SAE's approximate runtime and hardware requirements.

Usage:
    # Run all three identities (reuses checkpoint if found)
    python scripts/train_sae.py --embeddings data/embeddings/

    # Run one identity
    python scripts/train_sae.py --embeddings data/embeddings/ --identity race

    # Retrain from scratch, overwriting existing checkpoints
    python scripts/train_sae.py --embeddings data/embeddings/ --identity gender --overwrite
"""

import sys
import os
import time
import json
import platform
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMBED_DIR = ROOT / "data" / "embeddings"
RUNTIME_LOG_PATH = DATA_DIR / "runtime_log.csv"
sys.path.insert(0, str(ROOT / "src"))

import hypothesaes.sae as _sae
import hypothesaes.quickstart as _qs
import torch
from hypothesaes.quickstart import train_sae
from hypothesaes.interpret_neurons import (
    NeuronInterpreter, InterpretConfig, ScoringConfig, LLMConfig, SamplingConfig,
    sample_top_zero,
)
from hypothesaes.annotate import annotate, CACHE_DIR


# ---------------------------------------------------------------------------
# Checkpoint compatibility: old .pt files were saved before hypothesaes added
# the `threshold` buffer; load with strict=False so the default (0.0) is used.
# ---------------------------------------------------------------------------
def _load_model_compat(path, device="cuda" if torch.cuda.is_available() else "cpu"):
    ckpt = torch.load(path, pickle_module=pickle, weights_only=False)
    model = _sae.SparseAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"Loaded model from {path}")
    return model

_sae.load_model = _load_model_compat
_qs.load_model  = _load_model_compat


# ---------------------------------------------------------------------------
# Timing / hardware logging — for reporting runtime & hardware requirements
# ---------------------------------------------------------------------------
def _device_summary():
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Silicon (MPS)"
    else:
        device = "cpu"
        device_name = platform.processor() or platform.machine()
    return {
        "device": device,
        "device_name": device_name,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }


_DEVICE_INFO = _device_summary()


def log_timing(identity, stage, seconds, **extra):
    """Print and append one timing row to data/runtime_log.csv.

    Fixed schema regardless of what `extra` each call passes (M, K,
    n_candidates, interpreter_model, ...) — those vary by stage, so they're
    JSON-encoded into a single `extra` column rather than becoming ragged
    columns across appended rows.
    """
    print(f"  [timing] {identity} — {stage}: {seconds:.1f}s")
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "identity": identity,
        "stage": stage,
        "seconds": round(seconds, 3),
        "device": _DEVICE_INFO["device"],
        "device_name": _DEVICE_INFO["device_name"],
        "cpu_count": _DEVICE_INFO["cpu_count"],
        "platform": _DEVICE_INFO["platform"],
        "extra": json.dumps(extra) if extra else "",
    }
    RUNTIME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(RUNTIME_LOG_PATH, mode="a", index=False, header=not RUNTIME_LOG_PATH.exists())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_M = 32
DEFAULT_K = 4

IDENTITY_TEXT_COLS = {
    "race":               "race_open",
    "gender":             "gender_open",
    "sexual_orientation": "sexuality_open",
}

IDENTITY_TASK_INSTRUCTIONS = {
    "race": """All of the texts are responses to the question:
In at least 2-3 sentences, how would you describe your race and/or ethnicity?
Features should describe a specific aspect of the response. For example:
- "mentions relationship to ..."
- "self-describes as ...."
- "discusses ... they ..."
""",
    "gender": """All of the texts are responses to the question:
In at least 2-3 sentences, how would you describe your gender identity?
Features should describe a specific aspect of the response. For example:
- "mentions relationship to ..."
- "self-describes as ..."
- "discusses ... they ..."
""",
    "sexual_orientation": """All of the texts are responses to the question:
In at least 2-3 sentences, how would you describe your sexual orientation?
Features should describe a specific aspect of the response. For example:
- "mentions relationship to ..."
- "self-describes as ..."
- "discusses ... they ..."
""",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAE on free-text identity responses and score interpretation fidelity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--identity",
        choices=["race", "gender", "sexual_orientation", "all"],
        default="all",
        help="Which identity to process (default: all three)",
    )
    parser.add_argument("--M", type=int, default=DEFAULT_M,
                        help="Number of SAE neurons (default: 32)")
    parser.add_argument("--K", type=int, default=DEFAULT_K,
                        help="Active neurons per input — sparsity (default: 4)")
    parser.add_argument(
        "--interpreter-model", default="gpt-4o",
        help="LLM for generating neuron interpretations (default: gpt-4o)",
    )
    parser.add_argument(
        "--annotator-model", default="gpt-4.1-mini",
        help="LLM for scoring interpretation fidelity (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing SAE checkpoints and retrain from scratch",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_embeddings(emb_dir, identity):
    """Load pre-computed embeddings from the embeddings directory.

    Returns (response_ids, embeddings).
    """
    emb_dir = Path(emb_dir)
    ids_path = emb_dir / "response_ids.npy"
    emb_path = emb_dir / f"{identity}_embeddings.npy"

    for p in (ids_path, emb_path):
        if not p.exists():
            sys.exit(
                f"Embeddings file not found: {p}\n"
                "See data/README.md for instructions on requesting access."
            )

    response_ids = np.load(ids_path, allow_pickle=True)
    embeddings   = np.load(emb_path).astype(np.float32)
    print(f"  Loaded {identity} embeddings: {embeddings.shape}")
    return response_ids, embeddings


def load_survey_texts(survey_path, identity, response_ids):
    """Return texts in the same order as response_ids."""
    df = pd.read_csv(survey_path, usecols=["ResponseId", IDENTITY_TEXT_COLS[identity]])
    df = df.set_index("ResponseId")
    texts = [df.loc[rid, IDENTITY_TEXT_COLS[identity]] for rid in response_ids]
    return texts


# ---------------------------------------------------------------------------
# TODO: RELABEL AS NOT "BUG" BEFORE PUSHING
# F1 metric comparison: old (buggy, pre-fix) vs new (fixed) formula
#
# Old bug: precision was computed as 1 - mean(annotations[labels==0]), which is
# specificity (TN / (TN+FP)), not precision (TP / (TP+FP)). Recall was correct
# and is unchanged between the two formulas.
# ---------------------------------------------------------------------------
def _old_f1_metrics(annotations, labels):
    annotations = np.asarray(annotations, dtype=float)
    labels = np.asarray(labels)
    if not (1 in labels and 0 in labels):
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    true_pos = np.mean(annotations[labels == 1])
    false_pos = np.mean(annotations[labels == 0])
    precision = 1 - false_pos
    f1 = 2 * true_pos * precision / (true_pos + precision) if (true_pos + precision) > 0 else 0.0
    return {"recall": true_pos, "precision": precision, "f1": f1}


def score_all_candidates_both_metrics(interpreter, texts, activations, interpretations,
                                       scoring_config, cache_name):
    """Score every candidate interpretation (not just the winner) under both the
    old (buggy) and new (fixed) F1 formulas, reusing the annotation cache that
    `score_interpretations` just populated (no extra API calls).

    Returns a DataFrame with one row per (neuron_idx, candidate) pair.
    """
    cache_path = str(Path(CACHE_DIR) / f"{cache_name}_interp-scoring.json")
    rows = []
    for neuron_idx, candidates in interpretations.items():
        sample = sample_top_zero(
            texts=texts,
            activations=activations,
            neuron_idx=neuron_idx,
            n_examples=scoring_config.n_examples,
            max_words_per_example=scoring_config.max_words_per_example,
            random_seed=neuron_idx,
        )
        eval_texts = sample["positive_texts"] + sample["negative_texts"]
        eval_activations = np.array(sample["positive_activations"] + sample["negative_activations"])
        labels = np.concatenate([
            np.ones(len(sample["positive_texts"])),
            np.zeros(len(sample["negative_texts"])),
        ])

        for candidate_num, interp in enumerate(candidates, start=1):
            if interp is None:
                continue
            tasks = [(text, interp) for text in eval_texts]
            cached = annotate(tasks=tasks, cache_path=cache_path, use_cache_only=True, show_progress=False)
            annotations = np.array([cached[interp][text] for text in eval_texts])

            new_m = interpreter._compute_metrics(annotations=annotations, labels=labels, activations=eval_activations)
            old_m = _old_f1_metrics(annotations, labels)

            rows.append({
                "neuron_idx": neuron_idx,
                "candidate_num": candidate_num,
                "interpretation": interp,
                "new_precision": new_m["precision"],
                "new_recall": new_m["recall"],
                "new_f1": new_m["f1"],
                "old_precision": old_m["precision"],
                "old_recall": old_m["recall"],
                "old_f1": old_m["f1"],
            })

    df = pd.DataFrame(rows).sort_values(["neuron_idx", "candidate_num"]).reset_index(drop=True)

    flips = 0
    for neuron_idx, group in df.groupby("neuron_idx"):
        best_new = group.loc[group["new_f1"].idxmax(), "interpretation"]
        best_old = group.loc[group["old_f1"].idxmax(), "interpretation"]
        if best_new != best_old:
            flips += 1
    n_neurons = df["neuron_idx"].nunique()
    print(f"  Theme selection changed by the metric fix: {flips}/{n_neurons} neurons")

    return df


# ---------------------------------------------------------------------------
# Interpret SAE neurons and score fidelity
# ---------------------------------------------------------------------------
def generate_interpretations_and_fidelity_scores(
    M,
    texts,
    embeddings,
    sae,
    cache_name,
    *,
    identity=None,
    interpreter_model="gpt-4o",
    annotator_model="gpt-4.1-mini",
    n_examples_for_interpretation=20,
    max_words_per_example=256,
    interpret_temperature=0.7,
    max_interpretation_tokens=50,
    n_candidate_interpretations=1,
    n_scoring_examples=100,
    scoring_metric="f1",
    n_workers_interpretation=10,
    n_workers_annotation=30,
    task_specific_instructions=None,
):
    identity_label = identity or cache_name
    embeddings = np.array(embeddings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)

    print(f"Embeddings shape: {embeddings.shape}")

    if not isinstance(sae, list):
        sae = [sae]

    t0 = time.perf_counter()
    activations_list = []
    neuron_source_sae_info = []
    for s in sae:
        activations_list.append(s.get_activations(X))
        neuron_source_sae_info += [(s.m_total_neurons, s.k_active_neurons)] * s.m_total_neurons
    activations = np.concatenate(activations_list, axis=1)
    log_timing(identity_label, "compute_activations", time.perf_counter() - t0, n_responses=embeddings.shape[0])

    print(f"Activations shape (from {len(sae)} SAEs): {activations.shape}")

    selected_neurons, scores = list(range(M)), [0] * M

    print(f"\nStep 2: Interpreting selected neurons")
    interpreter = NeuronInterpreter(
        cache_name=cache_name,
        interpreter_model=interpreter_model,
        annotator_model=annotator_model,
        n_workers_interpretation=n_workers_interpretation,
        n_workers_annotation=n_workers_annotation,
    )

    interpret_config = InterpretConfig(
        sampling=SamplingConfig(
            n_examples=n_examples_for_interpretation,
            max_words_per_example=max_words_per_example,
        ),
        llm=LLMConfig(
            temperature=interpret_temperature,
            max_interpretation_tokens=max_interpretation_tokens,
        ),
        n_candidates=n_candidate_interpretations,
        task_specific_instructions=task_specific_instructions,
    )

    interpretations = interpreter.interpret_neurons(
        texts=texts,
        activations=activations,
        neuron_indices=selected_neurons,
        config=interpret_config,
    )

    results = []
    all_candidates_df = None
    if n_scoring_examples == 0:
        for idx, score in zip(selected_neurons, scores):
            try:
                results.append({
                    "neuron_idx": idx,
                    "source_sae": neuron_source_sae_info[idx],
                    "interpretation": interpretations[idx][0],
                })
            except Exception:
                results.append({
                    "neuron_idx": idx,
                    "source_sae": neuron_source_sae_info[idx],
                    "interpretation": "",
                })
    else:
        print(f"\nStep 3: Scoring Interpretations")
        scoring_config = ScoringConfig(n_examples=n_scoring_examples)
        metrics = interpreter.score_interpretations(
            texts=texts,
            activations=activations,
            interpretations=interpretations,
            config=scoring_config,
        )

        for idx, score in zip(selected_neurons, scores):
            best_interp = max(
                interpretations[idx],
                key=lambda interp: metrics[idx][interp][scoring_metric],
            )
            best_score = metrics[idx][best_interp][scoring_metric]
            results.append({
                "neuron_idx": idx,
                "source_sae": neuron_source_sae_info[idx],
                "interpretation": best_interp,
                f"{scoring_metric}_fidelity_score": best_score,
            })

        print(f"\nStep 4: Comparing old vs. new F1 across all {n_candidate_interpretations} candidates per neuron")
        all_candidates_df = score_all_candidates_both_metrics(
            interpreter=interpreter,
            texts=texts,
            activations=activations,
            interpretations=interpretations,
            scoring_config=scoring_config,
            cache_name=cache_name,
        )

    return pd.DataFrame(results), all_candidates_df


# ---------------------------------------------------------------------------
# Core: train + interpret
# ---------------------------------------------------------------------------
def train_and_interpret(survey_path, emb_dir, identity, M, K,
                        interpreter_model, annotator_model, overwrite=False):
    print(f"\n{'─'*60}")
    print(f"Identity: {identity}  M={M}  K={K}")
    print("─" * 60)

    response_ids, embeddings = load_embeddings(emb_dir, identity)
    texts = load_survey_texts(survey_path, identity, response_ids)
    cache_name     = f"M={M}_K={K}_{identity}"
    checkpoint_dir = str(DATA_DIR / "checkpoints" / cache_name)
    checkpoint_path = Path(checkpoint_dir) / f"SAE_M={M}_K={K}.pt"
    reused_checkpoint = checkpoint_path.exists() and not overwrite

    # Train SAE — reuses checkpoint unless --overwrite
    t0 = time.perf_counter()
    sae = train_sae(
        embeddings=embeddings,
        M=M,
        K=K,
        checkpoint_dir=checkpoint_dir,
        overwrite_checkpoint=overwrite,
    )
    stage = "load_checkpoint (reused, no training)" if reused_checkpoint else "train_sae (from scratch)"
    log_timing(identity, stage, time.perf_counter() - t0, M=M, K=K, n_responses=embeddings.shape[0])

    # Interpret neurons and score fidelity
    fidelity_df, all_candidates_df = generate_interpretations_and_fidelity_scores(
        M=M,
        texts=texts,
        embeddings=embeddings,
        sae=sae,
        cache_name=cache_name,
        identity=identity,
        n_candidate_interpretations=3,
        interpreter_model=interpreter_model,
        annotator_model=annotator_model,
        task_specific_instructions=IDENTITY_TASK_INSTRUCTIONS[identity],
    )

    fidelity_dir = DATA_DIR / "fidelity"
    fidelity_dir.mkdir(parents=True, exist_ok=True)
    out_path = fidelity_dir / f"{identity}_interpretation_fidelity.csv"
    fidelity_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.relative_to(ROOT)}")
    print(f"  Median F1 fidelity: {fidelity_df['f1_fidelity_score'].median():.3f}")

    if all_candidates_df is not None:
        comparison_dir = DATA_DIR / "f1_metric_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        candidates_path = comparison_dir / f"{identity}_all_candidates.csv"
        all_candidates_df.to_csv(candidates_path, index=False)
        print(f"  Saved: {candidates_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    identities = (
        ["race", "gender", "sexual_orientation"]
        if args.identity == "all"
        else [args.identity]
    )

    survey_path = DATA_DIR / "in_your_own_words.csv"

    if not survey_path.exists():
        sys.exit(f"Survey data not found at {survey_path}.\nSee data/README.md for access instructions.")
    if not EMBED_DIR.is_dir():
        sys.exit(f"Embeddings directory not found: {EMBED_DIR}.\nSee data/README.md for access instructions.")

    print(f"Hardware: {_DEVICE_INFO['device']} ({_DEVICE_INFO['device_name']}), "
          f"{_DEVICE_INFO['cpu_count']} CPUs, {_DEVICE_INFO['platform']}")

    for identity in identities:
        try:
            train_and_interpret(
                survey_path, EMBED_DIR, identity, args.M, args.K,
                args.interpreter_model, args.annotator_model,
                args.overwrite,
            )
        except Exception as e:
            print(f"Error processing {identity}: {e}")

    print(f"\nDone. SAE timing log: {RUNTIME_LOG_PATH.relative_to(ROOT)}")
    print("Run scripts/annotate_themes.py next to regenerate theme annotation indicators.")
