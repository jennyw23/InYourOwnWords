"""
Section 4: Theme alignment with standardized categories.

For each extracted theme, regresses the binary theme indicator on the
standardized identity categories (OLS) and reports R².  Themes that map closely
to existing categories have high R²; themes that cut across categories have low R².

Produces:
  - Per-theme R² sorted table (one per identity)
  - Median R² per identity
  - Table S10: four R² metrics (OLS, adjusted, McFadden, Cox-Snell)
  - Stacked bar-chart figures saved to analysis/figures/ (Figures S7–S9, and the three
    Figure 2 panels as figure2_{identity}.png)

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
    print("Median R² by metric (Table S10)")
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
# Figure 2: seven selected themes per identity, one panel each (combined by hand)
# ──────────────────────────────────────────────────────────────────────────────

FIG2_THEMES = {
    "race": [
        "mentions being Black or African American",
        "mentions being born in the United States or America",
        "mentions being first, second, or third generation American or immigrant",
        "mentions speaking or understanding multiple languages or specific non-English languages",
        "mentions traditional or cultural food as an important aspect of their identity",
        "describes beliefs or values associated with their race",
        "mentions feelings of not fully belonging or being out of place related to race/ethnicity",
    ],
    "gender": [
        "mentions enjoyment or participation in activities or items stereotypically associated with femininity (e.g., makeup, fashion, dressing up, heels, etc.)",
        "mentions traditionally masculine activities or traits such as sports, fixing things, outdoor activities, or being a provider/protector",
        "mentions roles or responsibilities within a family or household",
        "mentions fluidity or fluctuation in gender identity",
        "explicitly rejects or feels disconnected from the concept of gender as a meaningful personal identity",
        "mentions experiences of questioning or struggling with their gender identity",
        "mentions being male or a man in a direct and explicit manner",
    ],
    "sexual_orientation": [
        "mentions attraction not based on gender or sex",
        "mentions attraction to more than two genders or a spectrum of genders",
        "mentions uncertainty or questioning about their sexual orientation",
        "mentions age or life stage when they realized or came out",
        "explicitly states never questioning their sexual orientation",
        "mentions romantic orientation as distinct from sexual orientation",
        "mentions difficulty or hesitation in forming romantic or sexual connections",
    ],
}

# Shortened labels used in the figure.
FIG2_LABELS = {
    "mentions speaking or understanding multiple languages or specific non-English languages": "mentions speaking or understanding multiple languages",
    "mentions being first, second, or third generation American or immigrant": "mentions being immigrant or immigrant generation",
    "mentions traditional or cultural food as an important aspect of their identity": "mentions cultural food as important to identity",
    "mentions feelings of not fully belonging or being out of place related to race/ethnicity": "mentions a lack of racial or ethnic belonging",
    "mentions being born in the United States or America": "mentions being born in the United States",
    "mentions experiences of questioning or struggling with their gender identity": "mentions questioning gender identity",
    "explicitly rejects or feels disconnected from the concept of gender as a meaningful personal identity": "mentions rejecting gender as part of identity",
    "mentions roles or responsibilities within a family or household": "mentions responsibilities within family",
    "mentions traditionally masculine activities or traits such as sports, fixing things, outdoor activities, or being a provider/protector": "mentions traditionally masculine activities or traits",
    "mentions enjoyment or participation in activities or items stereotypically associated with femininity (e.g., makeup, fashion, dressing up, heels, etc.)": "mentions stereotypically feminine activities or items",
    "mentions being male or a man in a direct and explicit manner": "directly mentions being male or a man",
    "mentions attraction to more than two genders or a spectrum of genders": "mentions attraction to multiple genders",
    "mentions uncertainty or questioning about their sexual orientation": "mentions questioning sexual orientation",
    "mentions age or life stage when they realized or came out": "mentions age of realization or coming out",
    "explicitly states never questioning their sexual orientation": "mentions never questioning sexual orientation",
    "mentions romantic orientation as distinct from sexual orientation": "distinguishes between romantic & sexual orientation",
    "mentions difficulty or hesitation in forming romantic or sexual connections": "mentions reluctance to form intimate connections",
}

FIG2_MIN_N = 10
# As in the paper: respondents in sexual orientation categories with fewer than FIG2_MIN_N
# people are dropped, and that same filtered sample is used for all three panels.
FIG2_FILTER_COLUMN = "sexual_orientation"

FIG2_PANELS = {  # plotting function name, figwidth, legend anchor
    "race":               ("improved_race_themes_barchart",               18,   (1.1, 1.025)),
    "gender":             ("improved_gender_themes_barchart_crosstab",    18,   (1.1, 1.0)),
    "sexual_orientation": ("improved_sexual_orientation_themes_barchart", 18.5, (1.1, 1.05)),
}


def _fig2_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20,
                         "xtick.labelsize": 20, "ytick.labelsize": 25,
                         "legend.fontsize": 25, "legend.title_fontsize": 25})


def save_figure2_panels(annotations_by_identity, df):
    """Save the three Figure 2 panels to figures/figure2_{identity}.png."""
    import matplotlib
    matplotlib.use("Agg")
    import make_figures
    from sae_helper import calculate_proportions_identity

    _set_figure_font()
    counts = df[FIG2_FILTER_COLUMN].value_counts()
    print(f"\nFigure 2: dropping {FIG2_FILTER_COLUMN} categories with n < {FIG2_MIN_N}: "
          f"{counts[counts < FIG2_MIN_N].to_dict()}")
    filtered = df[df[FIG2_FILTER_COLUMN].isin(counts[counts >= FIG2_MIN_N].index)]

    for identity, ann in annotations_by_identity.items():
        fn_name, figwidth, bbox = FIG2_PANELS[identity]
        subset = ann[FIG2_THEMES[identity]].rename(columns=FIG2_LABELS)
        proportions = calculate_proportions_identity(subset, filtered, identity)
        fig = getattr(make_figures, fn_name)(
            proportions, identity, sort_by="min_max_any_group", r2_dict=None, n_activated=None,
            custom_style=_fig2_style, no_legend=False, textwrap_width=60,
            figwidth=figwidth, figheight=6, has_custom_bbox=bbox,
        )
        out = FIGURES_DIR / f"figure2_{identity}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved figure: {out.relative_to(ROOT)}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nLoading data …")
    df = load_survey()
    print(f"  N = {len(df)} respondents")

    results, annotations = {}, {}
    for identity in IDENTITIES:
        try:
            ann = load_annotations(identity)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping {identity}")
            continue
        annotations[identity] = ann

        formula = category_formula(identity, df)
        print(f"\nComputing R² for {len(ann.columns)} {identity} themes …")
        r2_df = compute_all_r2(ann, df, formula)
        results[identity] = r2_df

        print_r2_table(r2_df, identity)
        save_figure(identity, ann, df)

    print_median_r2(results)
    print_robust_r2(results)

    if len(annotations) == len(IDENTITIES):
        save_figure2_panels(annotations, df)
