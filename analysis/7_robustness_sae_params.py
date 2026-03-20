"""
analysis/6_robustness_sae_params.py — Robustness check: SAE themes at M=16 and M=64.

Loads pre-computed interpretation fidelity scores for M=16 and M=64 SAEs and
prints the filtered theme lists for each identity, mirroring what the main
analysis uses at M=32.

To regenerate the fidelity CSVs from scratch, run:
    python scripts/train_sae_robustness.py --embeddings data/embeddings/

Run from InYourOwnWords/:
    python analysis/6_robustness_sae_params.py
"""

import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from sae_helper import filter_interpretations

FIDELITY_DIR       = DATA_DIR / "fidelity"
FIDELITY_THRESHOLD = 0.50
IDENTITIES         = ["race", "gender", "sexual_orientation"]
M_VALUES           = [16, 64]


def load_fidelity(identity, M):
    path = FIDELITY_DIR / f"{identity}_interpretation_fidelity_m{M}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Fidelity file not found: {path}\n"
            "Run scripts/train_sae_robustness.py to generate it."
        )
    return pd.read_csv(path)


def print_themes(identity, M, fidelity_df):
    # filter_interpretations expects annotation_df with theme columns;
    # here we just want to see which themes survive the fidelity threshold,
    # so pass fidelity_df itself as a stand-in with the interpretation column.
    kept = fidelity_df[fidelity_df["f1_fidelity_score"] >= FIDELITY_THRESHOLD]
    print(f"\n  {identity} — M={M}  ({len(kept)}/{len(fidelity_df)} themes pass F1 ≥ {FIDELITY_THRESHOLD})")
    for _, row in kept.iterrows():
        print(f"    [{row['neuron_idx']:2d}]  {row['f1_fidelity_score']:.3f}  {row['interpretation']}")


if __name__ == "__main__":
    for M in M_VALUES:
        print(f"\n{'─'*70}")
        print(f"SAE dimensions: M={M}, K=4")
        print("─" * 70)
        for identity in IDENTITIES:
            try:
                fidelity_df = load_fidelity(identity, M)
            except FileNotFoundError as e:
                print(f"  {identity}: {e}")
                continue
            print_themes(identity, M, fidelity_df)
    print()
