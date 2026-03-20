"""
Section 3: Validating extracted themes.

Reports inter-rater agreement between LLM annotations (GPT-4.1-mini,
identity-specific prompt) and human annotations on a random sample of
8 themes x 100 responses per identity axis (24 themes total).

Two kappa methods are used (following the paper):
  - Per-theme kappa: kappa computed for each 100-row theme block; overall median cited in text
  - Diagonal kappa: single kappa over all 800 concatenated diagonal values per identity;
                used for the per-identity values and Table S7

Run from InYourOwnWords/:
    python analysis/3_validate_themes.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))
FIGURES_DIR    = Path(__file__).parent / "figures"

VALIDATION_DIR    = DATA_DIR / "validation"
ROWS_PER_THEME    = 100
THEMES_PER_IDENTITY = 8
IDENTITIES        = ["race", "gender", "sexual_orientation"]

# All model x prompt combinations in the validation study
MODELS = [
    ("gpt-4o-mini",  "default"),
    ("gpt-4.1-mini", "default"),
    ("gpt-4.1",      "default"),
    ("gpt-4.1-mini", "identity-specific"),
    ("gpt-4.1",      "identity-specific"),
]
CHOSEN_MODEL  = "gpt-4.1-mini"
CHOSEN_PROMPT = "identity-specific"

KAPPA_LABELS = {
    (0.00, 0.20): "slight",
    (0.20, 0.40): "fair",
    (0.40, 0.60): "moderate",
    (0.60, 0.80): "substantial",
    (0.80, 1.01): "almost perfect",
}


def _kappa_label(k):
    for (lo, hi), label in KAPPA_LABELS.items():
        if lo <= k < hi:
            return label
    return ""


def _per_theme_kappas(identity, model, prompt):
    """Compute kappa for each 100-row theme block (Method 1)."""
    llm_path   = VALIDATION_DIR / f"annotated-{identity}-{model}-{prompt}.csv"
    human_path = VALIDATION_DIR / f"annotated-{identity}-sample.csv"
    llm_df     = pd.read_csv(llm_path)
    human_df   = pd.read_csv(human_path)

    kappas = []
    for i in range(THEMES_PER_IDENTITY):
        lo, hi    = i * ROWS_PER_THEME, (i + 1) * ROWS_PER_THEME
        theme_col = human_df.columns[i + 2]
        llm_vals  = llm_df.iloc[lo:hi][theme_col].values
        hum_vals  = human_df.iloc[lo:hi][theme_col].values
        mask      = ~(pd.isna(llm_vals) | pd.isna(hum_vals))
        k = cohen_kappa_score(llm_vals[mask].astype(int), hum_vals[mask].astype(int))
        kappas.append(k)
    return kappas


def _diagonal_kappa(identity, model, prompt):
    """Compute single kappa over 800 concatenated diagonal values (Method 2)."""
    llm_path   = VALIDATION_DIR / f"annotated-{identity}-{model}-{prompt}.csv"
    human_path = VALIDATION_DIR / f"annotated-{identity}-sample.csv"
    llm_df     = pd.read_csv(llm_path)
    human_df   = pd.read_csv(human_path)

    llm_labels, hum_labels = [], []
    for i in range(THEMES_PER_IDENTITY):
        lo, hi    = i * ROWS_PER_THEME, (i + 1) * ROWS_PER_THEME
        theme_col = human_df.columns[i + 2]
        llm_block = llm_df.iloc[lo:hi][theme_col].values
        hum_block = human_df.iloc[lo:hi][theme_col].values
        mask      = ~(pd.isna(llm_block) | pd.isna(hum_block))
        llm_labels.extend(llm_block[mask].tolist())
        hum_labels.extend(hum_block[mask].tolist())

    llm_arr = np.array(llm_labels, dtype=int)
    hum_arr = np.array(hum_labels, dtype=int)
    k       = cohen_kappa_score(llm_arr, hum_arr)
    pos_rate = np.mean(llm_arr)
    return k, pos_rate


# ──────────────────────────────────────────────────────────────────────────────

def overall_median_kappa():
    """Overall median kappa across all 24 themes (chosen model)."""
    print("─" * 60)
    print("Per-theme kappa — chosen model "
          f"({CHOSEN_MODEL}, {CHOSEN_PROMPT} prompt)")
    print("─" * 60)

    all_kappas = []
    for identity in IDENTITIES:
        try:
            kappas = _per_theme_kappas(identity, CHOSEN_MODEL, CHOSEN_PROMPT)
        except FileNotFoundError as e:
            print(f"  {identity}: file not found — {e}")
            continue
        all_kappas.extend(kappas)
        ks = [f"{k:.2f}" for k in kappas]
        print(f"  {identity:25s}  [{', '.join(ks)}]  "
              f"median={np.median(kappas):.3f}")

    if all_kappas:
        med = np.median(all_kappas)
        print(f"\n  Overall median kappa (24 themes): {med:.3f}  ({_kappa_label(med)})")
    print()


def per_identity_kappa():
    """Per-identity diagonal kappa for the chosen model (cited in main text)."""
    print("─" * 60)
    print("Per-identity kappa — chosen model "
          f"({CHOSEN_MODEL}, {CHOSEN_PROMPT} prompt)")
    print("─" * 60)
    for identity in IDENTITIES:
        try:
            k, pos = _diagonal_kappa(identity, CHOSEN_MODEL, CHOSEN_PROMPT)
        except FileNotFoundError as e:
            print(f"  {identity}: file not found — {e}")
            continue
        print(f"  {identity:25s}  kappa={k:.3f}  ({_kappa_label(k)})"
              f"  LLM positive rate={pos:.3f}")
    print()


def annotation_agreement_table():
    """Full annotation agreement table: all model × prompt × identity combinations."""
    print("─" * 60)
    print("Annotation agreement by model and prompt (Table S7)")
    print("─" * 60)
    header = f"  {'Model':<15} {'Prompt':<20} {'Race kappa':>8} {'Gender kappa':>10} {'SO kappa':>8} {'LLM+ rate':>10}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    human_pos_rates = []
    for identity in IDENTITIES:
        try:
            human_path = VALIDATION_DIR / f"annotated-{identity}-sample.csv"
            hum_df = pd.read_csv(human_path)
            all_vals = []
            for i in range(THEMES_PER_IDENTITY):
                lo, hi = i * ROWS_PER_THEME, (i + 1) * ROWS_PER_THEME
                theme_col = hum_df.columns[i + 2]
                all_vals.extend(hum_df.iloc[lo:hi][theme_col].dropna().astype(int).tolist())
            human_pos_rates.append(np.mean(all_vals))
        except FileNotFoundError:
            pass
    human_avg_pos = np.mean(human_pos_rates) if human_pos_rates else float("nan")

    for model, prompt in MODELS:
        kappas, pos_rates = {}, []
        for identity in IDENTITIES:
            try:
                k, pos = _diagonal_kappa(identity, model, prompt)
                kappas[identity] = k
                pos_rates.append(pos)
            except FileNotFoundError:
                kappas[identity] = float("nan")

        avg_pos = np.mean(pos_rates) if pos_rates else float("nan")
        marker  = " <- chosen" if (model, prompt) == (CHOSEN_MODEL, CHOSEN_PROMPT) else ""
        print(f"  {model:<15} {prompt:<20} "
              f"{kappas.get('race', float('nan')):>8.3f} "
              f"{kappas.get('gender', float('nan')):>10.3f} "
              f"{kappas.get('sexual_orientation', float('nan')):>8.3f} "
              f"{avg_pos:>10.3f}{marker}")

    print(f"\n  Human annotator average positive rate: {human_avg_pos:.3f}")
    print()


def plot_kappa_figure(out_path=None):
    """Bar chart of per-theme kappas for the chosen model, color-coded by identity."""
    if out_path is None:
        out_path = FIGURES_DIR / "kappa_by_theme.png"

    colors = plt.cm.Set2.colors
    identity_colors = {
        "race":               colors[0],
        "gender":             colors[1],
        "sexual_orientation": colors[2],
    }
    identity_labels = {
        "race":               "Race",
        "gender":             "Gender",
        "sexual_orientation": "Sexual Orientation",
    }

    all_kappas, all_labels, all_colors = [], [], []

    for identity in IDENTITIES:
        human_path = VALIDATION_DIR / f"annotated-{identity}-sample.csv"
        try:
            human_df = pd.read_csv(human_path)
            kappas   = _per_theme_kappas(identity, CHOSEN_MODEL, CHOSEN_PROMPT)
        except FileNotFoundError as e:
            print(f"  Skipping {identity}: {e}")
            continue

        theme_cols = [human_df.columns[i + 2] for i in range(THEMES_PER_IDENTITY)]
        all_kappas.extend(kappas)
        all_labels.extend(theme_cols)
        all_colors.extend([identity_colors[identity]] * len(kappas))

    if not all_kappas:
        print("No data to plot.")
        return

    x = range(len(all_kappas))
    fig, ax = plt.subplots(figsize=(max(12, len(all_kappas) * 0.8), 6))
    bars = ax.bar(x, all_kappas, color=all_colors)

    for bar, k in zip(bars, all_kappas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{k:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(all_labels, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("Cohen's Kappa")
    ax.set_title(f"Per-theme kappa ({CHOSEN_MODEL}, {CHOSEN_PROMPT} prompt)")
    ax.set_ylim(0, 1.05)

    legend_handles = [
        mpatches.Patch(color=identity_colors[i], label=identity_labels[i])
        for i in IDENTITIES if identity_colors.get(i)
    ]
    ax.legend(handles=legend_handles, title="Identity")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved kappa figure: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    overall_median_kappa()
    per_identity_kappa()
    annotation_agreement_table()
    plot_kappa_figure()
