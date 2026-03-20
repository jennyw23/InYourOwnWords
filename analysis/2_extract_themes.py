"""
Section 2: Computationally extracting interpretable themes.

Loads the pre-trained SAE interpretation fidelity files and reports:
  - All 32 candidate themes per identity with fidelity scores
  - Themes excluded (fidelity < 0.50 or prompt-shaped)
  - Final theme counts (26 race, 27 gender, 28 sexual orientation)
  - Overall median fidelity score

To retrain the SAE from scratch, see scripts/train_sae.py.
To re-run LLM annotation, see scripts/annotate_themes.py.

Run from InYourOwnWords/:
    python analysis/2_extract_themes.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from sae_helper import _interpretations_map_M_32

FIDELITY_DIR  = DATA_DIR / "fidelity"
FIDELITY_THRESHOLD = 0.50
IDENTITIES = ["race", "gender", "sexual_orientation"]

# Prompt-shaped themes excluded regardless of fidelity score
PROMPT_SHAPED = _interpretations_map_M_32()


# ──────────────────────────────────────────────────────────────────────────────

def load_fidelity(identity):
    path = FIDELITY_DIR / f"{identity}_interpretation_fidelity.csv"
    return pd.read_csv(path)


def report_themes(identity, fidelity_df):
    """Print all themes, marking excluded ones, and return included theme list."""
    print(f"\n  {identity.upper().replace('_', ' ')} ({len(fidelity_df)} candidates)")
    print(f"  {'Theme':<80}  {'Fidelity':>8}  {'Status'}")
    print(f"  {'─'*80}  {'─'*8}  {'─'*12}")

    prompt_shaped = set(PROMPT_SHAPED.get(identity, []))
    included = []

    for _, row in fidelity_df.iterrows():
        interp = row["interpretation"]
        score  = row["f1_fidelity_score"]

        if score < FIDELITY_THRESHOLD:
            status = f"excluded (fidelity={score:.2f})"
        elif interp in prompt_shaped:
            status = "excluded (prompt-shaped)"
        else:
            status = "included"
            included.append(interp)

        truncated = (interp[:77] + "…") if len(interp) > 80 else interp
        print(f"  {truncated:<80}  {score:>8.2f}  {status}")

    return included


def fidelity_summary(all_fidelity_dfs):
    """Print median fidelity across all 96 themes."""
    all_scores = []
    for identity, fid_df in all_fidelity_dfs.items():
        scores = fid_df["f1_fidelity_score"].tolist()
        all_scores.extend(scores)
        print(f"  {identity:25s}  n={len(scores)}  "
              f"mean={np.mean(scores):.3f}  median={np.median(scores):.3f}")
    overall = np.median(all_scores)
    print(f"\n  Overall median fidelity (all {len(all_scores)} themes): {overall:.3f}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nLoading interpretation fidelity scores …\n")

    all_fidelity = {}
    final_counts  = {}

    for identity in IDENTITIES:
        try:
            fid_df = load_fidelity(identity)
        except FileNotFoundError:
            print(f"  WARNING: fidelity file not found for {identity} — skipping")
            continue
        all_fidelity[identity] = fid_df
        included = report_themes(identity, fid_df)
        final_counts[identity] = len(included)

    print("\n" + "─" * 60)
    print("Theme counts after filtering")
    print("─" * 60)
    for identity, n in final_counts.items():
        print(f"  {identity:25s}  {n} themes included")

    print("\n" + "─" * 60)
    print("Fidelity score summary")
    print("─" * 60)
    fidelity_summary(all_fidelity)
    print()
