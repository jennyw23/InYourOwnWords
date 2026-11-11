"""
Section 11: Running the framework with open-weight models (Supplementary I).

Compares the open-weight pipeline (Qwen3-Embedding-8B + Qwen3-32B) with the OpenAI
pipeline, from files already on disk (no API calls):
  - median interpretation fidelity across all 96 themes
  - share of themes with a "same or related" counterpart in the other pipeline
    (manual labels in open_weight/results/)
  - Table S31: annotation agreement with the human validation sample

To regenerate the open-weight outputs, see open_weight/run_pipeline.py, open_weight/rescore_fidelity.py
and open_weight/match_themes.py.

Run from InYourOwnWords/:
    python analysis/11_open_weight_models.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OPEN_WEIGHT_DIR = ROOT / "open_weight" / "results"

IDENTITIES = ["race", "gender", "sexual_orientation"]
ROWS_PER_THEME, THEMES_PER_IDENTITY = 100, 8
MODELS = {"GPT-4.1-mini": "gpt-4.1-mini-identity-specific", "Qwen3-32B": "qwen3-32b-identity-specific"}


def median_fidelity():
    print("Interpretation fidelity, median across 96 themes")
    # Qwen recall/specificity scores come from open_weight/rescore_fidelity.py.
    sources = [
        ("OpenAI, recall/specificity", DATA_DIR / "fidelity", "_interpretation_fidelity.csv", "f1_fidelity_score"),
        ("Qwen, precision/recall F1", OPEN_WEIGHT_DIR, "_interpretation_fidelity.csv", "f1_fidelity_score"),
        ("Qwen, recall/specificity", OPEN_WEIGHT_DIR, "_interpretation_fidelity_recall_specificity.csv",
         "recall_specificity_hmean"),
    ]
    print(f"  {'':<28} {'all':>6}" + "".join(f"{i:>20}" for i in IDENTITIES))
    for label, folder, suffix, col in sources:
        try:
            by_id = {i: pd.read_csv(folder / f"{i}{suffix}")[col].to_numpy() for i in IDENTITIES}
        except FileNotFoundError as e:
            print(f"  {label:<28} file not found: {e.filename}")
            continue
        overall = np.median(np.concatenate(list(by_id.values())))
        print(f"  {label:<28} {overall:>6.3f}" + "".join(f"{np.median(v):>20.3f}" for v in by_id.values()))
    print()


def correspondence():
    print("Themes with a same or related counterpart (manual labels)")
    for label, fname in [("Qwen -> OpenAI", "theme_match_labels.csv"),
                         ("OpenAI -> Qwen", "reverse_manual_review.csv")]:
        labels = pd.read_csv(OPEN_WEIGHT_DIR / fname)["label"].dropna().astype(str).str.strip().str.lower()
        if labels.empty:
            print(f"  {label}: no labels in {fname}")
            continue
        share = labels.isin(["same", "related"]).mean()
        print(f"  {label}: {share:.0%} same or related, {1 - share:.0%} different (n={len(labels)})")
    print()


def kappas(model_file):
    """Pooled and median-across-themes kappa per identity, plus the LLM positive rate."""
    rows = {}
    for identity in IDENTITIES:
        human = pd.read_csv(DATA_DIR / "validation" / f"annotated-{identity}-sample.csv")
        llm = pd.read_csv(DATA_DIR / "validation" / f"annotated-{identity}-{model_file}.csv")
        per_theme, H, L = [], [], []
        for t in range(THEMES_PER_IDENTITY):
            block = slice(t * ROWS_PER_THEME, (t + 1) * ROWS_PER_THEME)
            col = human.columns[t + 2]
            h, l = human.iloc[block][col].to_numpy(), llm.iloc[block][col].to_numpy()
            mask = ~(pd.isna(h) | pd.isna(l))
            h, l = h[mask].astype(int), l[mask].astype(int)
            per_theme.append(cohen_kappa_score(h, l))
            H.append(h)
            L.append(l)
        H, L = np.concatenate(H), np.concatenate(L)
        rows[identity] = {"median": np.median(per_theme), "pooled": cohen_kappa_score(H, L),
                          "human_pos": H.mean(), "llm_pos": L.mean()}
    return pd.DataFrame(rows).T


def table_s31():
    print("Table S31 — agreement with human labels, identity-specific prompt")
    if not (DATA_DIR / "validation" / "annotated-race-sample.csv").exists():
        print("  skipped: needs the validation sample in data/validation/ (data request)\n")
        return
    results = {label: kappas(f) for label, f in MODELS.items()}
    for summary in ["median", "pooled"]:
        print(f"  kappa {summary} across 8 themes")
        print(f"    {'Identity':<20}" + "".join(f"{m:>14}" for m in MODELS))
        for identity in IDENTITIES:
            print(f"    {identity:<20}" + "".join(f"{results[m].loc[identity, summary]:>14.3f}" for m in MODELS))
    human = results["GPT-4.1-mini"]["human_pos"].mean()
    print(f"  Mean positive rate: human {human:.1%}, "
          + ", ".join(f"{m} {results[m]['llm_pos'].mean():.1%}" for m in MODELS))
    print()


if __name__ == "__main__":
    print()
    median_fidelity()
    correspondence()
    table_s31()
