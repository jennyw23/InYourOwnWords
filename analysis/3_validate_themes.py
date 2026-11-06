"""
Section 3: Validating extracted themes (Supplementary C.1).

Compares LLM annotations with human annotations on 8 themes x 100 responses per
identity axis (24 themes). Reports the median across themes, as in the paper:
  - Table S2: Cohen's kappa and positive rates for each annotation model and prompt
  - Table S3: kappa, precision, and recall for GPT-4.1-mini + identity-specific prompt
  - Figure S6: per-theme kappa for that configuration

Run from InYourOwnWords/:
    python analysis/3_validate_themes.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DIR = ROOT / "data" / "validation"
FIGURES_DIR = Path(__file__).parent / "figures"

ROWS_PER_THEME = 100
THEMES_PER_IDENTITY = 8
IDENTITIES = ["race", "gender", "sexual_orientation"]

# Rows of Table S2, in paper order
MODELS = [
    ("gpt-4.1-mini", "default"),
    ("gpt-4.1",      "default"),
    ("gpt-4.1-mini", "identity-specific"),
    ("gpt-4.1",      "identity-specific"),
]
CHOSEN = ("gpt-4.1-mini", "identity-specific")


def paired_labels(identity, model, prompt):
    """(theme, human, llm) for each theme's 100-row block, NaNs dropped pairwise."""
    human_df = pd.read_csv(VALIDATION_DIR / f"annotated-{identity}-sample.csv")
    llm_df = pd.read_csv(VALIDATION_DIR / f"annotated-{identity}-{model}-{prompt}.csv")
    out = []
    for i in range(THEMES_PER_IDENTITY):
        rows = slice(i * ROWS_PER_THEME, (i + 1) * ROWS_PER_THEME)
        theme = human_df.columns[i + 2]
        h = human_df.iloc[rows][theme].to_numpy()
        l = llm_df.iloc[rows][theme].to_numpy()
        mask = ~(pd.isna(h) | pd.isna(l))
        out.append((theme, h[mask].astype(int), l[mask].astype(int)))
    return out


def clopper_pearson(successes, trials):
    if trials == 0:
        return np.nan, np.nan
    return proportion_confint(successes, trials, method="beta")


def theme_reliability(model, prompt):
    """One row per theme: kappa, precision, recall (human labels as reference) with 95% CIs."""
    rows = []
    for identity in IDENTITIES:
        for theme, h, l in paired_labels(identity, model, prompt):
            tp = int(((h == 1) & (l == 1)).sum())
            n_llm, n_human = int(l.sum()), int(h.sum())
            p_lo, p_hi = clopper_pearson(tp, n_llm)
            r_lo, r_hi = clopper_pearson(tp, n_human)
            rows.append({
                "identity": identity, "theme": theme,
                "kappa": cohen_kappa_score(h, l),
                "precision": precision_score(h, l, zero_division=0), "precision_lo": p_lo, "precision_hi": p_hi,
                "recall": recall_score(h, l, zero_division=0), "recall_lo": r_lo, "recall_hi": r_hi,
                "n_llm_positive": n_llm, "n_human_positive": n_human,
                "n": len(h), "human_sum": int(h.sum()), "llm_sum": int(l.sum()),
            })
    return pd.DataFrame(rows)


def table_s2():
    print("─" * 78)
    print("Table S2 — annotation agreement by model and prompt (median kappa across 8 themes)")
    print("─" * 78)
    print(f"  {'Model':<14}{'Prompt':<19}{'Race':>7}{'Gender':>8}{'SO':>7}{'Human+%':>9}{'LLM+%':>8}{'Δ pp':>7}")
    for model, prompt in MODELS:
        df = theme_reliability(model, prompt)
        k = df.groupby("identity")["kappa"].median()
        # positive rates pool each identity's 800 labels, then average across identities
        sums = df.groupby("identity")[["human_sum", "llm_sum", "n"]].sum()
        pos = pd.Series({"human_pos": (sums.human_sum / sums.n).mean(), "llm_pos": (sums.llm_sum / sums.n).mean()})
        marker = "  <- selected" if (model, prompt) == CHOSEN else ""
        print(f"  {model:<14}{prompt:<19}{k['race']:>7.3f}{k['gender']:>8.3f}{k['sexual_orientation']:>7.3f}"
              f"{100 * pos['human_pos']:>9.1f}{100 * pos['llm_pos']:>8.1f}"
              f"{round(100 * (pos['llm_pos'] - pos['human_pos']), 6):>+7.1f}{marker}")
    print()


def table_s3(df):
    print("─" * 78)
    print(f"Table S3 — {CHOSEN[0]}, {CHOSEN[1]} prompt (median across 8 themes)")
    print("─" * 78)
    med = df.groupby("identity", sort=False)[["kappa", "precision", "recall"]].median()
    for identity, r in med.iterrows():
        print(f"  {identity:<22} kappa={r['kappa']:.3f}  precision={r['precision']:.3f}  recall={r['recall']:.3f}")
    print(f"\n  Median kappa across all 24 themes: {df['kappa'].median():.3f}\n")

    print("  Per-theme estimates (95% Clopper-Pearson CIs)")
    for r in df.itertuples():
        print(f"    {r.identity:<19} {r.theme[:60]:<60} κ={r.kappa:5.2f}  "
              f"P={r.precision:.2f} [{r.precision_lo:.2f}, {r.precision_hi:.2f}] (n={r.n_llm_positive})  "
              f"R={r.recall:.2f} [{r.recall_lo:.2f}, {r.recall_hi:.2f}] (n={r.n_human_positive})")
    print()


def plot_kappa_figure(df, out_path=FIGURES_DIR / "kappa_by_theme.png"):
    """Figure S6: per-theme kappa, colored by identity axis."""
    colors = dict(zip(IDENTITIES, plt.cm.Set2.colors))
    labels = {"race": "Race", "gender": "Gender", "sexual_orientation": "Sexual Orientation"}

    fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.8), 6))
    bars = ax.bar(range(len(df)), df["kappa"], color=df["identity"].map(colors))
    for bar, k in zip(bars, df["kappa"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{k:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["theme"], rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("Cohen's Kappa")
    ax.set_title(f"Per-theme kappa ({CHOSEN[0]}, {CHOSEN[1]} prompt)")
    ax.set_ylim(0, 1.05)
    ax.legend(handles=[mpatches.Patch(color=colors[i], label=labels[i]) for i in IDENTITIES], title="Identity")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved kappa figure: {out_path}")


if __name__ == "__main__":
    print()
    table_s2()
    chosen = theme_reliability(*CHOSEN)
    table_s3(chosen)
    plot_kappa_figure(chosen)
