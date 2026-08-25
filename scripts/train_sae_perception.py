"""
scripts/train_sae_perception.py — Train Sparse Autoencoders on perceived identity responses.

Trains SAEs on responses to the perception question ("How does your self-identified
[identity] compare to how you believe others perceive your [identity]?"), filtering
out respondents who answered "Mostly the same" on the closed-form perception item,
as done in the paper.

Pre-computed perception embeddings are required and are distributed separately from
the survey data (see data/README.md for access instructions).  The embeddings
directory must contain:

    response_ids.npy                                (1004,)
    race_perceive_embeddings.npy                    (1004, 3072)
    gender_perceive_embeddings.npy                  (1004, 3072)
    sexual_orientation_perceive_embeddings.npy      (1004, 3072)

These were produced with OpenAI text-embedding-3-large (3072 dimensions).

What this script does:
  1. Loads perception embeddings and survey data.
  2. Filters out "Mostly the same" responses on the closed-form perception item.
  3. Trains an SAE with M=32 neurons and K=4 active neurons per input.
  4. Interprets each SAE neuron via an LLM (GPT-4o by default).
  5. Scores interpretation fidelity with an annotator LLM (GPT-4.1-mini).
  6. Saves fidelity scores to data/fidelity/{identity}_perceive_interpretation_fidelity.csv.

Usage:
    python scripts/train_sae_perception.py

    python scripts/train_sae_perception.py --identity race

    python scripts/train_sae_perception.py --identity gender --overwrite
"""

import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMBED_DIR = ROOT / "data" / "embeddings"
sys.path.insert(0, str(ROOT / "src"))

import hypothesaes.sae as _sae
import hypothesaes.quickstart as _qs
import torch
from hypothesaes.quickstart import train_sae
from hypothesaes.interpret_neurons import NeuronInterpreter, InterpretConfig, ScoringConfig, LLMConfig, SamplingConfig


# ---------------------------------------------------------------------------
# Checkpoint compatibility (matches train_sae.py)
# ---------------------------------------------------------------------------
def _load_model_compat(path, device="cuda" if torch.cuda.is_available() else "cpu"):
    ckpt = torch.load(path, pickle_module=pickle, weights_only=False)
    model = _sae.SparseAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model

_sae.load_model = _load_model_compat
_qs.load_model  = _load_model_compat


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_M = 32
DEFAULT_K = 4

IDENTITY_CONFIG = {
    "race": {
        "text_col":   "race_perceive_open",
        "closed_col": "race_perceive_closed",
        "emb_file":   "race_perceive_embeddings.npy",
        "task_instructions": """All of the texts are responses to the question:
How does your self-identified race and/or ethnicity compare to how you believe others perceive your race and/or ethnicity?
Features should describe a specific aspect of the response. For example:
- "mentions relationship to ..."
- "self-describes as ..."
- "discusses ... they ..."
""",
    },
    "gender": {
        "text_col":   "gend_perceive_open",
        "closed_col": "gend_perceive_closed",
        "emb_file":   "gender_perceive_embeddings.npy",
        "task_instructions": """All of the texts are responses to the question:
How does your self-identified gender identity compare to how you believe others perceive your gender identity?
Features should describe a specific aspect of the response. For example:
- "mentions relationship to ..."
- "self-describes as ..."
- "discusses ... they ..."
""",
    },
    "sexual_orientation": {
        "text_col":   "sex_perceive_open",
        "closed_col": "sex_perceive_closed",
        "emb_file":   "sexual_orientation_perceive_embeddings.npy",
        "task_instructions": """All of the texts are responses to the question:
How does your self-identified sexual orientation compare to how you believe others perceive your sexual orientation?
Features should describe a specific aspect of the response. For example:
- "mentions relationship to ..."
- "self-describes as ..."
- "discusses ... they ..."
""",
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAE on perceived identity responses and score interpretation fidelity.",
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
def load_perception_data(survey_path, emb_dir, identity):
    """Load perception texts and embeddings, filtered to drop 'Mostly the same'.

    Returns (texts, embeddings, n_original) where texts and embeddings are
    aligned arrays for respondents who did NOT answer 'Mostly the same'.
    """
    cfg = IDENTITY_CONFIG[identity]
    emb_dir = Path(emb_dir)

    ids_path = emb_dir / "response_ids.npy"
    emb_path = emb_dir / cfg["emb_file"]
    for p in (ids_path, emb_path):
        if not p.exists():
            sys.exit(
                f"Embeddings file not found: {p}\n"
                "See data/README.md for instructions on requesting access."
            )

    response_ids = np.load(ids_path, allow_pickle=True)
    embeddings   = np.load(emb_path).astype(np.float32)

    df = pd.read_csv(survey_path, usecols=["ResponseId", cfg["text_col"], cfg["closed_col"]])
    df = df.set_index("ResponseId")

    closed_vals  = np.array([df.loc[rid, cfg["closed_col"]] for rid in response_ids])
    keep_mask    = closed_vals != "Mostly the same"

    n_original   = len(response_ids)
    n_kept       = keep_mask.sum()
    print(f"  Filtering 'Mostly the same': {n_kept}/{n_original} responses kept "
          f"({n_kept / n_original * 100:.1f}%)")

    kept_ids     = response_ids[keep_mask]
    kept_embeds  = embeddings[keep_mask]
    kept_texts   = [str(df.loc[rid, cfg["text_col"]]) for rid in kept_ids]

    # Handle any zero/missing embeddings
    emb_dim = kept_embeds.shape[1]
    for i, emb in enumerate(kept_embeds):
        if np.all(emb == 0):
            kept_texts[i] = ""

    print(f"  Loaded {identity} perception embeddings: {kept_embeds.shape}")
    return kept_texts, kept_embeds


# ---------------------------------------------------------------------------
# Interpret SAE neurons and score fidelity (mirrors train_sae.py)
# ---------------------------------------------------------------------------
def generate_interpretations_and_fidelity_scores(
    M, texts, embeddings, sae, cache_name,
    interpreter_model, annotator_model, task_specific_instructions,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)

    activations = sae.get_activations(X)
    selected_neurons = list(range(M))

    interpreter = NeuronInterpreter(
        cache_name=cache_name,
        interpreter_model=interpreter_model,
        annotator_model=annotator_model,
        n_workers_interpretation=10,
        n_workers_annotation=30,
    )
    interpret_config = InterpretConfig(
        sampling=SamplingConfig(n_examples=20, max_words_per_example=256),
        llm=LLMConfig(temperature=0.7, max_interpretation_tokens=50),
        n_candidates=3,
        task_specific_instructions=task_specific_instructions,
    )
    interpretations = interpreter.interpret_neurons(
        texts=texts,
        activations=activations,
        neuron_indices=selected_neurons,
        config=interpret_config,
    )

    scoring_config = ScoringConfig(n_examples=100)
    metrics = interpreter.score_interpretations(
        texts=texts,
        activations=activations,
        interpretations=interpretations,
        config=scoring_config,
    )

    results = []
    for idx in selected_neurons:
        best_interp = max(
            interpretations[idx],
            key=lambda interp: metrics[idx][interp]["f1"],
        )
        results.append({
            "neuron_idx": idx,
            "interpretation": best_interp,
            "f1_fidelity_score": metrics[idx][best_interp]["f1"],
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Core: train + interpret one identity
# ---------------------------------------------------------------------------
def train_and_interpret(survey_path, emb_dir, identity, M, K,
                        interpreter_model, annotator_model, overwrite=False):
    print(f"\n{'─'*60}")
    print(f"Identity: {identity} (perception, drop 'Mostly the same')  M={M}  K={K}")
    print("─" * 60)

    texts, embeddings = load_perception_data(survey_path, emb_dir, identity)

    cache_name     = f"perception_drop_same_M={M}_K={K}_{identity}"
    checkpoint_dir = str(DATA_DIR / "checkpoints" / cache_name)

    sae = train_sae(
        embeddings=embeddings,
        M=M,
        K=K,
        checkpoint_dir=checkpoint_dir,
        overwrite_checkpoint=overwrite,
    )

    fidelity_df = generate_interpretations_and_fidelity_scores(
        M=M,
        texts=texts,
        embeddings=embeddings,
        sae=sae,
        cache_name=cache_name,
        interpreter_model=interpreter_model,
        annotator_model=annotator_model,
        task_specific_instructions=IDENTITY_CONFIG[identity]["task_instructions"],
    )

    fidelity_dir = DATA_DIR / "fidelity"
    fidelity_dir.mkdir(parents=True, exist_ok=True)
    out_path = fidelity_dir / f"{identity}_perceive_interpretation_fidelity.csv"
    fidelity_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.relative_to(ROOT)}")
    print(f"  Median F1 fidelity: {fidelity_df['f1_fidelity_score'].median():.3f}")


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

    for identity in identities:
        train_and_interpret(
            survey_path, EMBED_DIR, identity, args.M, args.K,
            args.interpreter_model, args.annotator_model,
            args.overwrite,
        )

    print("\nDone.")
