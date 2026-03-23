"""
Section 5: Free-text themes help explain life outcomes (Table 1).

For each identity axis, runs nested F-tests comparing:
  - Base model:  outcome ~ standardized identity categories
  - Full model:  outcome ~ standardized categories + free-text theme indicators

Outcomes: identity importance, physical health, mental health,
          life satisfaction, everyday discrimination, income.

Statistical significance assessed with Benjamini-Hochberg FDR correction
applied across all 18 tests (6 outcomes x 3 identity axes).

Run from InYourOwnWords/:
    python analysis/5_themes_explain_outcomes.py
"""

import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from data_helper import (multiracial_category_hispa_not, crosstab_gender,
                         group_bi_pan_gay_lesb_queer)
from regression_helper import (create_identity_closed_indicators,
                                clean_outcome_vars,
                                benjamini_hochberg_correction)
from sae_helper import filter_interpretations

DATA_PATH      = DATA_DIR / "in_your_own_words.csv"
ANNOTATION_DIR = DATA_DIR / "annotations"
FIDELITY_DIR   = DATA_DIR / "fidelity"
FIGURES_DIR    = Path(__file__).parent / "figures"

FIDELITY_THRESHOLD = 0.50
IDENTITIES = ["race", "gender", "sexual_orientation"]

# Outcomes to display per identity in the coefficient figures
SIGNIFICANT_OUTCOMES = {
    "race":               ["race_importance", "physical_health", "mental_health"],
    "gender":             ["gender_importance", "physical_health", "mental_health",
                           "life_satisfaction"],
    "sexual_orientation": ["sexual_orientation_importance", "mental_health"],
}

# Font setup (prefer Helvetica, fall back to DejaVu Sans)
_preferred_fonts = ["Helvetica Light", "Helvetica Neue", "Helvetica"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
_plot_font = next((f for f in _preferred_fonts if f in _available_fonts), "DejaVu Sans")
plt.rcParams["font.family"] = _plot_font


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
    ann = pd.read_csv(ANNOTATION_DIR / f"{identity}_annotations.csv")
    fid = pd.read_csv(FIDELITY_DIR   / f"{identity}_interpretation_fidelity.csv")
    ann = ann.drop(columns=[c for c in ann.columns
                             if c in ("ResponseId", "Unnamed: 0")], errors="ignore")
    return filter_interpretations(ann, fid, FIDELITY_THRESHOLD, identity)


def base_formula(identity, df):
    if identity == "race":
        cols = sorted(c for c in df.columns if c.startswith("race_indicator_"))
        return " + ".join(cols)
    elif identity == "gender":
        return "C(describe_gender) * C(gender_trans, Treatment('No'))"
    else:
        cols = sorted(c for c in df.columns
                      if c.startswith("sexual_orientation_indicator_"))
        return " + ".join(cols)


# ──────────────────────────────────────────────────────────────────────────────
# Nested F-test
# ──────────────────────────────────────────────────────────────────────────────

def _safe_colname(name):
    s = (name.replace(" ", "_").replace("'", "").replace('"', "")
             .replace(",", "").replace("(", "").replace(")", "")
             .replace("/", "_").replace("-", "_").replace(".", "")
             .replace("?", "").replace("!", "").replace(":", ""))
    if s and s[0].isdigit():
        s = "t_" + s
    return s[:60]


def nested_f_test(df, outcome, base_vars, theme_cols):
    data = df.dropna(subset=[outcome]).copy()

    rename = {}
    seen   = set()
    for col in theme_cols:
        safe = _safe_colname(col)
        while safe in seen:
            safe += "_"
        seen.add(safe)
        rename[col] = safe
    data = data.rename(columns=rename)

    theme_terms  = " + ".join(rename.values())
    base_formula_ = f"{outcome} ~ {base_vars}"
    full_formula  = f"{outcome} ~ {base_vars} + {theme_terms}"

    try:
        base_m = sm.OLS.from_formula(base_formula_, data=data).fit()
        print(base_m.summary())
        full_m = sm.OLS.from_formula(full_formula,  data=data).fit()
    except Exception as e:
        return {"outcome": outcome, "adj_r2_base": np.nan, "adj_r2_full": np.nan,
                "ratio": np.nan, "f_stat": np.nan, "p_value": np.nan, "n": 0}

    f_stat, p_value, _ = full_m.compare_f_test(base_m)
    adj_base = base_m.rsquared_adj
    adj_full = full_m.rsquared_adj
    ratio    = adj_full / adj_base if adj_base > 0 else np.nan

    return {
        "outcome":     outcome,
        "adj_r2_base": round(adj_base, 4),
        "adj_r2_full": round(adj_full, 4),
        "ratio":       round(ratio, 2),
        "f_stat":      f_stat,
        "p_value":     p_value,
        "n":           int(base_m.nobs),
    }


def significance_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def run_identity(df, annotations, identity, bformula):
    df, outcomes = clean_outcome_vars(df.copy(), identity)
    theme_cols   = annotations.columns.tolist()
    combined     = pd.concat(
        [df.reset_index(drop=True), annotations.reset_index(drop=True)], axis=1
    )

    rows = []
    for outcome in outcomes:
        r = nested_f_test(combined, outcome, bformula, theme_cols)
        r["identity"] = identity
        rows.append(r)

    results = pd.DataFrame(rows)
    valid   = results["p_value"].notna()
    p_vals  = results.loc[valid, "p_value"].tolist()
    df_p    = pd.DataFrame({"p-value": p_vals})
    df_p    = benjamini_hochberg_correction(df_p)
    results.loc[valid, "p_bh"] = df_p["BH_corrected_p"].values
    results["stars"] = results["p_bh"].apply(
        lambda p: significance_stars(p) if pd.notna(p) else ""
    )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Theme-level coefficient estimation
# ──────────────────────────────────────────────────────────────────────────────

USE_HC3 = True  # set False for classic homoskedastic SEs


def get_outcome_by_theme_identity_control(df, annotation_df, theme,
                                          outcome, base_vars, sd_type="HC3"):
    """OLS of outcome ~ base_vars + theme; returns (model, coef_dict)."""
    combined = pd.concat(
        [df.reset_index(drop=True), annotation_df.reset_index(drop=True)], axis=1
    )
    formula = f'{outcome} ~ {base_vars} + Q("{theme}")'
    model  = smf.ols(formula, data=combined).fit()

    key      = f'Q("{theme}")'
    idx      = list(model.params.index).index(key)

    if USE_HC3:
        robust = model.get_robustcov_results(cov_type='HC3')
        coef   = robust.params[idx]
        se     = robust.bse[idx]
        pval   = robust.pvalues[idx]
    else:
        coef   = model.params[key]
        se     = model.bse[key]
        pval   = model.pvalues[key]

    coef_dict  = {
        "theme":    theme,
        "outcome":  outcome,
        "coef":     coef,
        "std_err":  se,
        "t_value":  coef / se,
        "p_value":  pval,
        "ci_lower": coef - 1.96 * se,
        "ci_upper": coef + 1.96 * se,
    }
    return model, coef_dict


def compute_theme_effect_coefs(df, annotation_df, identity, base_vars):
    """
    For every (theme, outcome) pair run a univariate OLS controlling for
    standardized identity categories.  Returns a tidy DataFrame of
    coefficients with columns: theme, outcome, coef, std_err, t_value,
    p_value, ci_lower, ci_upper.
    """
    df, outcome_cols = clean_outcome_vars(df.copy(), identity, convert_to_z_scores=True)
    themes = annotation_df.columns.tolist()

    rows = []
    for outcome in outcome_cols:
        for theme in themes:
            _, info = get_outcome_by_theme_identity_control(
                df, annotation_df, theme, outcome, base_vars
            )
            rows.append(info)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_point_estimates(
    effect_df,
    outcome_columns,
    sort_by=None,
    figsize=None,
    textwrap_width=80,
    fontsize=(20, 25),
):
    """
    One subplot per outcome showing per-theme point estimates ± 1.96 SE.
    Themes that straddle zero are shown in gray; significant themes are black.
    Sorted by coefficient magnitude in the sort_by outcome column.
    """
    n_cols = len(outcome_columns)
    if figsize is None:
        figsize = (max(5 * n_cols, 22), 18)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharey=True)
    label_fontsize, theme_fontsize = fontsize

    effect_df = effect_df.copy()
    effect_df["theme"] = effect_df["theme"].apply(
        lambda t: textwrap.fill(t, width=textwrap_width)
    )

    if n_cols == 1:
        axes = [axes]

    theme_sort_order = None
    if sort_by is not None and sort_by in outcome_columns:
        sort_df = effect_df[effect_df["outcome"] == sort_by].sort_values("coef")
        theme_sort_order = sort_df["theme"].tolist()

    for i, outcome in enumerate(outcome_columns):
        sub = effect_df[effect_df["outcome"] == outcome].copy()
        if theme_sort_order is not None:
            sub["theme"] = pd.Categorical(
                sub["theme"], categories=theme_sort_order, ordered=True
            )
            sub = sub.sort_values("theme")

        y_pos = np.arange(len(sub))
        for j, row in enumerate(sub.itertuples()):
            spanning_zero = row.ci_lower <= 0 <= row.ci_upper
            color = "black" if spanning_zero else "black" # remove color contrast
            alpha = 0.5 if spanning_zero else 1.0
            axes[i].errorbar(
                row.coef, j,
                xerr=1.96 * row.std_err,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=3,
                capsize=5,
                alpha=alpha,
            )

        n_unique = len(effect_df["theme"].unique())
        for y in range(n_unique - 1):
            axes[i].axhline(y + 0.5, color="lightgray", linestyle="-", linewidth=1)

        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(sub["theme"], fontsize=theme_fontsize)
        axes[i].set_title(
            " ".join(outcome.split("_")).title(), fontsize=label_fontsize
        )
        axes[i].axvline(0, color="gray", linestyle="--", linewidth=1)
        axes[i].tick_params(axis="x", labelsize=theme_fontsize)

    plt.tight_layout()
    return fig


def make_coefficient_figures(df=None, fig_dir=None):
    """
    For each identity axis, compute per-theme OLS coefficients (controlling for
    standardized identity categories) and save a point-estimate figure.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-loaded survey dataframe (output of load_survey()).  Loaded
        automatically when omitted.
    fig_dir : Path or str, optional
        Directory to save figures.  Defaults to FIGURES_DIR.
    """
    if df is None:
        df = load_survey()
    if fig_dir is None:
        fig_dir = FIGURES_DIR
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for identity in IDENTITIES:
        try:
            ann = load_annotations(identity)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping {identity}")
            continue

        bformula  = base_formula(identity, df)
        sort_col  = f"{identity}_importance"
        sig_cols  = SIGNIFICANT_OUTCOMES[identity]

        print(f"Computing theme coefficients for {identity} …")
        coef_df = compute_theme_effect_coefs(df.copy(), ann, identity, bformula)

        fig = plot_point_estimates(
            coef_df,
            outcome_columns=sig_cols,
            sort_by=sort_col,
            figsize=(25, 25),
            fontsize=(30, 30),
        )

        out = fig_dir / f"point_estimates_{identity}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

OUTCOME_LABELS = {
    "race_importance":              "Identity importance",
    "gender_importance":            "Identity importance",
    "sexual_orientation_importance":"Identity importance",
    "physical_health":              "Physical health",
    "mental_health":                "Mental health",
    "life_satisfaction":            "Life satisfaction",
    "personal_discrimination_index":"Everyday discrimination",
    "income":                       "Income",
}


def print_table(all_results):
    print("\n" + "─" * 70)
    print("Table 1 — Free-text themes improve explained variance in life outcomes")
    print("(adj R²: base model → full model, relative increase, BH-corrected sig.)")
    print("─" * 70)
    header = f"  {'Identity':<22} {'Outcome':<28} {'Base':>6} {'Full':>6} {'Ratio':>7} {'Sig':>4}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for identity in IDENTITIES:
        if identity not in all_results:
            continue
        results = all_results[identity]
        for _, row in results.iterrows():
            label = OUTCOME_LABELS.get(row["outcome"], row["outcome"])
            print(f"  {identity:<22} {label:<28} "
                  f"{row['adj_r2_base']:>6.2f} {row['adj_r2_full']:>6.2f} "
                  f"{row['ratio']:>6.1f}x {row['stars']:>4}")
        print()

    # Cross-identity BH correction (18 tests together)
    all_rows = pd.concat(all_results.values(), ignore_index=True)
    valid    = all_rows["p_value"].notna()
    p_vals   = all_rows.loc[valid, "p_value"].tolist()
    df_p     = pd.DataFrame({"p-value": p_vals})
    df_p     = benjamini_hochberg_correction(df_p)
    all_rows.loc[valid, "p_bh_all"] = df_p["BH_corrected_p"].values
    all_rows["stars_all"] = all_rows["p_bh_all"].apply(
        lambda p: significance_stars(p) if pd.notna(p) else ""
    )
    sig_count = (all_rows["stars_all"] != "").sum()
    print(f"  {sig_count} of {len(all_rows)} outcome x identity tests significant "
          f"after BH correction (alpha=0.05, 18 tests pooled)")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nLoading data …")
    df = load_survey()
    print(f"  N = {len(df)} respondents")

    all_results = {}
    for identity in IDENTITIES:
        try:
            ann = load_annotations(identity)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping {identity}")
            continue

        bformula = base_formula(identity, df)
        print(f"\nRunning nested F-tests for {identity} ({ann.shape[1]} themes) …")
        all_results[identity] = run_identity(df, ann, identity, bformula)

    print_table(all_results)
    print()

    make_coefficient_figures(df=df)
