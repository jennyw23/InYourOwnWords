"""
Rescores the open-weight themes under the paper's fidelity metric (harmonic mean of recall and
specificity) from the annotation cache written by run_pipeline.py. No API calls.

Rebuilds each neuron's scoring sample exactly as hypothesaes did (top 50 / random 50 zero-activation
texts, seed = neuron index) and checks that the precision/recall F1 it gets back matches the saved
results before writing anything.

Usage:
    python open_weight/rescore_fidelity.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hypothesaes.annotate import generate_cache_key  # noqa: E402
from hypothesaes.interpret_neurons import ScoringConfig, sample_top_zero  # noqa: E402
from hypothesaes.sae import SparseAutoencoder  # noqa: E402

from run_pipeline import IDENTITIES, M, K, OUT_DIR, FIDELITY_DIR, TEXT_COLS, DATA_DIR  # noqa: E402
from src.fidelity import FIDELITY_METRIC, compute_metrics  # noqa: E402


def activations_for(identity):
    ckpt = torch.load(OUT_DIR / "checkpoints" / f"M={M}_K={K}_{identity}" / f"SAE_M={M}_K={K}.pt",
                      pickle_module=pickle, weights_only=False, map_location="cpu")
    sae = SparseAutoencoder(**ckpt["config"])
    sae.load_state_dict(ckpt["state_dict"], strict=False)
    X = torch.tensor(np.load(OUT_DIR / "embeddings" / f"{identity}_embeddings.npy"), dtype=torch.float32)
    return sae.get_activations(X)


def candidates_for(cache, eval_texts):
    """Interpretations that have a cached annotation for every text in this neuron's sample."""
    keys = {generate_cache_key("", t)[3:] for t in eval_texts}  # strip the empty concept and '|||'
    hits = {}
    for k in cache:
        concept, _, text_key = k.partition("|||")
        if text_key in keys:
            hits[concept] = hits.get(concept, 0) + 1
    return [c for c, n in hits.items() if n == len(keys)]


def rescore(identity):
    texts = pd.read_csv(DATA_DIR / "in_your_own_words.csv", usecols=[TEXT_COLS[identity]])[TEXT_COLS[identity]].tolist()
    acts = activations_for(identity)
    cache = json.load(open(OUT_DIR / "annotation_cache" / f"qwen_M={M}_K={K}_{identity}_interp-scoring.json"))
    saved = pd.read_csv(FIDELITY_DIR / f"{identity}_interpretation_fidelity.csv").set_index("neuron_idx")
    cfg = ScoringConfig(n_examples=100)

    rows = []
    for idx, row in saved.iterrows():
        s = sample_top_zero(texts=texts, activations=acts, neuron_idx=idx, n_examples=cfg.n_examples,
                            max_words_per_example=cfg.max_words_per_example, random_seed=idx)
        eval_texts = s["positive_texts"] + s["negative_texts"]
        labels = np.r_[np.ones(len(s["positive_texts"])), np.zeros(len(s["negative_texts"]))]
        eval_acts = np.array(s["positive_activations"] + s["negative_activations"])

        def score(interp):
            ann = np.array([int(cache[generate_cache_key(interp, t)]) for t in eval_texts])
            return compute_metrics(ann, labels, eval_acts)

        m = score(row["interpretation"])
        if not np.isclose(m["f1"], row["f1_fidelity_score"]):
            sys.exit(f"{identity} neuron {idx}: rebuilt F1 {m['f1']:.4f} != saved {row['f1_fidelity_score']:.4f}")

        cands = {c: score(c)[FIDELITY_METRIC] for c in candidates_for(cache, eval_texts)}
        best = max(cands, key=cands.get)
        rows.append({"neuron_idx": idx, "interpretation": row["interpretation"],
                     "recall": m["recall"], "specificity": m["specificity"], "precision": m["precision"],
                     "precision_recall_f1": m["f1"], "recall_specificity_hmean": m[FIDELITY_METRIC],
                     "n_candidates": len(cands), "best_rs_interpretation": best,
                     "best_rs_hmean": cands[best]})
    out = pd.DataFrame(rows)
    out.to_csv(FIDELITY_DIR / f"{identity}_interpretation_fidelity_recall_specificity.csv", index=False)
    return out


if __name__ == "__main__":
    all_rows = pd.concat([rescore(i).assign(identity=i) for i in IDENTITIES])
    flips = (all_rows["interpretation"] != all_rows["best_rs_interpretation"]).sum()
    print("All 96 rebuilt precision/recall F1 scores match the saved results.")
    print(f"Median fidelity, recall/specificity, themes as reported: {all_rows['recall_specificity_hmean'].median():.3f}")
    print(f"Median fidelity, precision/recall F1, themes as reported: {all_rows['precision_recall_f1'].median():.3f}")
    print(f"Median fidelity, recall/specificity, best of candidates: {all_rows['best_rs_hmean'].median():.3f}"
          f"  ({flips}/96 neurons would pick a different theme; candidates per neuron: "
          f"{sorted(all_rows['n_candidates'].unique().tolist())})")
    print(all_rows.groupby("identity")[["recall_specificity_hmean", "precision_recall_f1"]].median().round(3))
