"""
Section 8: Perceived identity themes (perception discordance analysis).

Loads pre-computed SAE interpretation fidelity files for perceived identity
responses and reports all 32 candidate themes per identity with fidelity scores.
Respondents who answered "Mostly the same" on the closed-form perception item
were excluded before training (see scripts/train_sae_perception.py).

To retrain the perception SAEs from scratch, see scripts/train_sae_perception.py.

Run from InYourOwnWords/:
    python analysis/8_perception_themes.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

FIDELITY_DIR = DATA_DIR / "fidelity"
IDENTITIES   = ["race", "gender", "sexual_orientation"]


def load_fidelity(identity):
    path = FIDELITY_DIR / f"{identity}_perceive_interpretation_fidelity.csv"
    return pd.read_csv(path)


def report_themes(identity, fidelity_df):
    print(f"\n  {identity.upper().replace('_', ' ')} PERCEPTION ({len(fidelity_df)} themes)")
    print(f"  {'#':>3}  {'Theme':<80}  {'F1':>6}")
    print(f"  {'─'*3}  {'─'*80}  {'─'*6}")
    for _, row in fidelity_df.iterrows():
        idx   = int(row["neuron_idx"]) + 1
        interp = row["interpretation"]
        score  = row["f1_fidelity_score"]
        truncated = (interp[:77] + "…") if len(interp) > 80 else interp
        print(f"  {idx:>3}  {truncated:<80}  {score:>6.3f}")


def fidelity_summary(all_fidelity_dfs):
    all_scores = []
    for identity, fid_df in all_fidelity_dfs.items():
        scores = fid_df["f1_fidelity_score"].tolist()
        all_scores.extend(scores)
        print(f"  {identity:25s}  n={len(scores)}  "
              f"mean={np.mean(scores):.3f}  median={np.median(scores):.3f}")
    print(f"\n  Overall median F1 (all {len(all_scores)} themes): {np.median(all_scores):.3f}")


if __name__ == "__main__":
    print("\nLoading perception interpretation fidelity scores …\n")

    all_fidelity = {}

    for identity in IDENTITIES:
        try:
            fid_df = load_fidelity(identity)
        except FileNotFoundError:
            print(f"  WARNING: fidelity file not found for {identity} — "
                  "run scripts/train_sae_perception.py first")
            continue
        all_fidelity[identity] = fid_df
        report_themes(identity, fid_df)

    print("\n" + "─" * 60)
    print("Fidelity score summary")
    print("─" * 60)
    fidelity_summary(all_fidelity)
    print()
