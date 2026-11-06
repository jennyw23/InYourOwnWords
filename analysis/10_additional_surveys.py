"""
Section 10: Applying the framework to other surveys (Supplementary H).

Reads the results in additional_surveys/results/ (no API calls) and reports:
  - Table S30: mean fidelity for each (M, K), per dataset
  - Figures S13 and S14: theme prevalence and fidelity for the best configuration

To regenerate the results, see additional_surveys/README.md.

Run from InYourOwnWords/:
    python analysis/10_additional_surveys.py
"""

import re
import textwrap
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "additional_surveys" / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

SCORE_COL = "recall_specificity_hmean_fidelity_score"
DATASETS = {
    "anes_mip": ("ANES most important problem", "Figure S13"),
    "estate_why_reduce_inequality": ("Stantcheva (2021) estate tax", "Figure S14"),
}


def fidelity_by_config(name):
    rows = []
    for path in RESULTS_DIR.glob(f"{name}_M=*_K=*_recall_specificity_hmean_interpretation_fidelity.csv"):
        M, K = map(int, re.search(r"_M=(\d+)_K=(\d+)_", path.name).groups())
        rows.append({"M": M, "K": K, "fidelity": pd.read_csv(path)[SCORE_COL].mean()})
    return pd.DataFrame(rows).sort_values("fidelity", ascending=False).reset_index(drop=True)


def plot_prevalence(df, save_path, label_width=50):
    """Horizontal bars of LLM-annotated prevalence, with each theme's fidelity beside its label."""
    fonts = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = next((f for f in ["Helvetica Light", "Helvetica Neue", "Helvetica"] if f in fonts),
                                       "DejaVu Sans")

    df = df.sort_values("pct_annotated_llm").reset_index(drop=True)
    labels = df["interpretation"].apply(lambda s: textwrap.fill(s, width=label_width))
    n = len(df)
    max_lines = max(label.count("\n") + 1 for label in labels)
    fig, ax = plt.subplots(figsize=(9 + max(0, label_width - 45) * 0.08,
                                    max(4, 0.3 * (1 + (max_lines - 1) * 0.5) * n)))

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=12)
    ax.tick_params(axis="y", pad=32)  # room for the fidelity column
    ax.barh(range(n), df["pct_annotated_llm"], color="#4C72B0")
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlabel("Share of responses annotated as expressing each theme", fontsize=12)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(axis="x", labelsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()  # before adding text outside the axes

    trans = ax.get_yaxis_transform()
    for i, score in enumerate(df[SCORE_COL]):
        ax.annotate(f"{score:.2f}", xy=(0, i), xycoords=trans, xytext=(-10, -1.5), textcoords="offset points",
                    ha="right", va="center", fontsize=11, clip_on=False)
    ax.annotate("Fidelity", xy=(0, n - 0.5), xycoords=trans, xytext=(-2, 0), textcoords="offset points",
                ha="right", va="bottom", fontsize=11, fontweight="bold", clip_on=False)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.relative_to(ROOT)}")


if __name__ == "__main__":
    print("\nTable S30 — mean fidelity by hyperparameter configuration")
    print("─" * 60)
    for name, (label, figure) in DATASETS.items():
        table = fidelity_by_config(name)
        print(f"\n  {label}")
        for r in table.itertuples():
            print(f"    M={r.M:<3} K={r.K}  {r.fidelity:.3f}")

        best = table.iloc[0]
        stem = f"{name}_M={int(best.M)}_K={int(best.K)}"
        prevalence = pd.read_csv(RESULTS_DIR / f"{stem}_prevalence_comparison.csv")
        fidelity = pd.read_csv(RESULTS_DIR / f"{stem}_recall_specificity_hmean_interpretation_fidelity.csv")
        df = prevalence.merge(fidelity[["neuron_idx", SCORE_COL]], on="neuron_idx")

        print(f"\n  {figure}: {label}, M={int(best.M)} K={int(best.K)}")
        for r in df.sort_values("pct_annotated_llm", ascending=False).itertuples():
            print(f"    {r.pct_annotated_llm:6.1%}  fidelity={getattr(r, SCORE_COL):.2f}  {r.interpretation}")
        plot_prevalence(df, FIGURES_DIR / f"{name}_prevalence.png")
