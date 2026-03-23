"""
Section 4: Theme alignment with standardized categories.

For each extracted theme, regresses the binary theme indicator on the
standardized identity categories (OLS) and reports R².  Themes that map closely
to existing categories have high R²; themes that cut across categories have low R².

Produces:
  - Per-theme R² sorted table (one per identity)
  - Median R² per identity
  - Table S5: four R² metrics (OLS, adjusted, McFadden, Cox-Snell)
  - Stacked bar-chart figures saved to analysis/figures/

Run from InYourOwnWords/:
    python analysis/4_themes_by_category.py
"""

import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from data_helper import (multiracial_category_hispa_not, crosstab_gender,
                         group_bi_pan_gay_lesb_queer)
from regression_helper import create_identity_closed_indicators
from sae_helper import filter_interpretations

DATA_PATH      = DATA_DIR / "in_your_own_words.csv"
ANNOTATION_DIR = DATA_DIR / "annotations"
FIDELITY_DIR   = DATA_DIR / "fidelity"
FIGURES_DIR    = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

FIDELITY_THRESHOLD = 0.50
IDENTITIES = ["race", "gender", "sexual_orientation"]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_survey():
    df = pd.read_csv(DATA_PATH)
    df["race"]               = df["race_closed"].apply(multiracial_category_hispa_not)
    df["gender"]             = df.apply(crosstab_gender, axis=1)
    df["sexual_orientation"] = df["sexuality_closed"].apply(group_bi_pan_gay_lesb_queer)
    df = create_identity_closed_indicators(df, "race", col="race_closed")
    df = create_identity_closed_indicators(df, "sexual_orientation",
                                           col="sexuality_closed")
    return df


def load_annotations(identity):
    ann_path = ANNOTATION_DIR / f"{identity}_annotations.csv"
    fid_path = FIDELITY_DIR   / f"{identity}_interpretation_fidelity.csv"
    ann = pd.read_csv(ann_path)
    fid = pd.read_csv(fid_path)
    ann = ann.drop(columns=[c for c in ann.columns
                             if c in ("ResponseId", "Unnamed: 0")], errors="ignore")
    ann = filter_interpretations(ann, fid, FIDELITY_THRESHOLD, identity)
    return ann


def category_formula(identity, df):
    if identity == "race":
        cols = sorted(c for c in df.columns if c.startswith("race_indicator_"))
        return " + ".join(cols)
    elif identity == "gender":
        return "C(describe_gender) * C(gender_trans, Treatment('No'))"
    else:
        cols = sorted(c for c in df.columns if c.startswith("sexual_orientation_indicator_"))
        return " + ".join(cols)


# ──────────────────────────────────────────────────────────────────────────────
# R² computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_ols_r2(theme_col, y, formula, df):
    combined = pd.concat(
        [df.reset_index(drop=True), y.rename("_y").reset_index(drop=True)], axis=1
    ).dropna(subset=["_y"])
    try:
        m = sm.OLS.from_formula(f"_y ~ {formula}", data=combined).fit()
        return m.rsquared, m.rsquared_adj
    except Exception:
        return float("nan"), float("nan")


def compute_pseudo_r2(theme_col, y, formula, df):
    combined = pd.concat(
        [df.reset_index(drop=True), y.rename(theme_col).reset_index(drop=True)], axis=1
    ).dropna(subset=[theme_col])
    try:
        glm = smf.glm(
            f'Q("{theme_col}") ~ {formula}',
            data=combined,
            family=sm.families.Binomial(),
        ).fit(disp=0, method="lbfgs")
        return glm.pseudo_rsquared(kind="mcf"), glm.pseudo_rsquared(kind="cs")
    except Exception:
        return float("nan"), float("nan")


def compute_all_r2(annotations, df, formula):
    rows = []
    for theme in annotations.columns:
        y = annotations[theme]
        r2, adj_r2     = compute_ols_r2(theme, y, formula, df)
        mcf, cox_snell = compute_pseudo_r2(theme, y, formula, df)
        rows.append({"theme": theme, "r2": r2, "adj_r2": adj_r2,
                     "mcfadden": mcf, "cox_snell": cox_snell})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def print_r2_table(r2_df, identity):
    print(f"\n  {identity.upper().replace('_', ' ')} — R² per theme (sorted)")
    print(f"  {'Theme':<80}  {'R²':>6}")
    print(f"  {'─'*80}  {'─'*6}")
    for _, row in r2_df.sort_values("r2").iterrows():
        truncated = (row["theme"][:77] + "…") if len(row["theme"]) > 80 else row["theme"]
        print(f"  {truncated:<80}  {row['r2']:>6.3f}")


def print_median_r2(results):
    print("\n" + "─" * 60)
    print("Median R² (OLS) per identity")
    print("─" * 60)
    for identity, r2_df in results.items():
        med = r2_df["r2"].median()
        mn  = r2_df["r2"].mean()
        n   = len(r2_df)
        print(f"  {identity:25s}  median={med:.4f}  mean={mn:.3f}  n={n} themes")


def print_robust_r2(results):
    print("\n" + "─" * 60)
    print("Median R² by metric (Table S5)")
    print("─" * 60)
    metrics = [
        ("r2",        "OLS R²"),
        ("adj_r2",    "Adj R²"),
        ("mcfadden",  "McFadden"),
        ("cox_snell", "Cox-Snell"),
    ]
    header = f"  {'Metric':<12}" + "".join(
        f"  {i.upper().replace('_', ' '):>22}" for i in IDENTITIES
    )
    print(header)
    print("  " + "─" * (len(header) - 2))
    for col, label in metrics:
        row = f"  {label:<12}"
        for identity in IDENTITIES:
            val = results[identity][col].median()
            row += f"  {val:>22.3f}"
        print(row)
    print()


def _set_figure_font():
    """Use Helvetica if available, else DejaVu Sans (matches notebook)."""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    preferred = ["Helvetica Light", "Helvetica Neue", "Helvetica"]
    available = {f.name for f in fm.fontManager.ttflist}
    font = next((f for f in preferred if f in available), "DejaVu Sans")
    plt.rcParams["font.family"] = font


def save_figure(identity, annotations, df):
    """Save stacked bar chart for this identity to figures/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from make_figures import (improved_race_themes_barchart,
                                  improved_gender_themes_barchart,
                                  improved_sexual_orientation_themes_barchart)
        from sae_helper import calculate_proportions_identity, count_activated_themes

        _set_figure_font()

        proportions = calculate_proportions_identity(annotations, df, identity)
        n_activated = count_activated_themes(annotations)

        fn_map = {
            "race":               improved_race_themes_barchart,
            "gender":             improved_gender_themes_barchart,
            "sexual_orientation": improved_sexual_orientation_themes_barchart,
        }
        kwargs = dict(
            sort_by="min_max_any_group",
            r2_dict=None,
            n_activated=n_activated,
            has_custom_bbox=(1, 1),
        )
        if identity == "sexual_orientation":
            kwargs["figwidth"] = 18

        fig = fn_map[identity](proportions, identity, **kwargs)
        out = FIGURES_DIR / f"{identity}_themes_barchart.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved figure: {out.relative_to(ROOT)}")
    except Exception as e:
        print(f"  [figure skipped for {identity}: {e}]")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nLoading data …")
    df = load_survey()
    print(f"  N = {len(df)} respondents")

    results = {}
    for identity in IDENTITIES:
        try:
            ann = load_annotations(identity)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping {identity}")
            continue

        formula = category_formula(identity, df)
        print(f"\nComputing R² for {len(ann.columns)} {identity} themes …")
        r2_df = compute_all_r2(ann, df, formula)
        results[identity] = r2_df

        print_r2_table(r2_df, identity)
        save_figure(identity, ann, df)

    print_median_r2(results)
    print_robust_r2(results)
