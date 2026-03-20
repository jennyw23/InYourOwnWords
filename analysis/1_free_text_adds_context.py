"""
Section 1: Minority respondents share more context.

Corresponds to the paper section "Minority respondents share more context."

Run from InYourOwnWords/:
    python d02_analysis/1_free_text_adds_context.py

Or run all scripts at once:
    python run_all.py
"""

import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH = DATA_DIR / "in_your_own_words.csv"

FREE_TEXT_USEFUL_COLS = {
    "race":               "race_details",
    "gender":             "gender_details",
    "sexual_orientation": "sexuality_details",
}
OPEN_TEXT_COLS = {
    "race":               "race_open",
    "gender":             "gender_open",
    "sexual_orientation": "sexuality_open",
}
PERCEIVE_TEXT_COLS = {
    "race":               "race_perceive_open",
    "gender":             "gend_perceive_open",
    "sexual_orientation": "sex_perceive_open",
}
REFERENCE_GROUPS = {
    "race":               "White",
    "gender":             "Cisgender Man/Woman",
    "sexual_orientation": "Straight or heterosexual",
}
MIN_GROUP_SIZE = 15


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_survey(path):
    df = pd.read_csv(path)
    from data_helper import (multiracial_category_hispa_not,
                             crosstab_gender,
                             group_bi_pan_gay_lesb_queer)
    df["race"]               = df["race_closed"].apply(multiracial_category_hispa_not)
    df["gender"]             = df.apply(crosstab_gender, axis=1)
    df["sexual_orientation"] = df["sexuality_closed"].apply(group_bi_pan_gay_lesb_queer)
    return df


def _collapse_gender(val):
    if val in ("Cisgender Man", "Cisgender Woman"):
        return "Cisgender Man/Woman"
    return val


def _binarize_minority(series, reference):
    return (series != reference).astype(int)


def _filter_small_groups(df, identity_col, min_n=MIN_GROUP_SIZE):
    counts = df[identity_col].value_counts()
    valid  = counts[counts >= min_n].index
    removed = df[~df[identity_col].isin(valid)]
    if len(removed):
        small = counts[counts < min_n].to_dict()
        print(f"  [{identity_col}] Dropping groups with n < {min_n}: {small}")
    return df[df[identity_col].isin(valid)].copy()


def _percent_yes(series):
    valid = series.dropna()
    return 100 * (valid == "Yes").mean()


def _word_count(text):
    return len(str(text).split())


def _logit_odds_ratio(df, outcome_col, predictor_col, cov_type="HC2"):
    data  = df[[outcome_col, predictor_col]].dropna()
    model = smf.logit(f"{outcome_col} ~ {predictor_col}", data=data).fit(
        cov_type=cov_type, disp=False
    )
    coef = model.params[predictor_col]
    ci   = model.conf_int().loc[predictor_col]
    return {
        "odds_ratio": np.exp(coef),
        "ci_low":     np.exp(ci[0]),
        "ci_high":    np.exp(ci[1]),
        "p_value":    model.pvalues[predictor_col],
    }


def _ols_coefficient(df, outcome_col, predictor_col, cov_type="HC2"):
    data  = df[[outcome_col, predictor_col]].dropna()
    model = smf.ols(f"{outcome_col} ~ {predictor_col}", data=data).fit(cov_type=cov_type)
    coef  = model.params[predictor_col]
    ci    = model.conf_int().loc[predictor_col]
    return {
        "coef":    coef,
        "ci_low":  ci[0],
        "ci_high": ci[1],
        "p_value": model.pvalues[predictor_col],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ──────────────────────────────────────────────────────────────────────────────

def word_length_statistics(df):
    """Descriptive word-count statistics for each identity axis (self-description)."""
    print("─" * 60)
    print("Word-count statistics — self-description responses")
    print("─" * 60)
    for identity, col in OPEN_TEXT_COLS.items():
        wc = df[col].apply(_word_count)
        print(f"  {identity:25s}  mean={wc.mean():.1f}  median={wc.median():.0f}"
              f"  max={wc.max()}")
    print()


def perception_word_length_statistics(df):
    """Descriptive word-count statistics for perception responses (how others see you)."""
    print("─" * 60)
    print("Word-count statistics — perception responses")
    print("─" * 60)
    for identity, col in PERCEIVE_TEXT_COLS.items():
        wc = df[col].apply(_word_count)
        print(f"  {identity:25s}  mean={wc.mean():.1f}  median={wc.median():.0f}"
              f"  max={wc.max()}")
    print()


def free_text_adds_context(df):
    """Percentage of respondents who say their free-text adds important context."""
    print("─" * 60)
    print("% saying free-text adds important context")
    print("─" * 60)
    for identity, col in FREE_TEXT_USEFUL_COLS.items():
        pct = _percent_yes(df[col])
        print(f"  {identity:25s}  {pct:.1f}%")
    print()


def minority_more_likely(df):
    """Odds that minority respondents say free-text adds important context (vs. majority)."""
    print("─" * 60)
    print("Minority respondents more likely to say free-text adds context")
    print("(logistic regression, minority vs. majority reference group)")
    print("─" * 60)
    for identity, useful_col in FREE_TEXT_USEFUL_COLS.items():
        work = df[[identity, useful_col]].copy()
        if identity == "gender":
            work[identity] = work[identity].map(_collapse_gender)
        # work = _filter_small_groups(work, identity)
        work["minority"]   = _binarize_minority(work[identity], REFERENCE_GROUPS[identity])
        work["useful_bin"] = (work[useful_col] == "Yes").astype(int)
        r = _logit_odds_ratio(work, "useful_bin", "minority")
        print(f"  {identity:25s}  OR={r['odds_ratio']:.2f}×  "
              f"95% CI=[{r['ci_low']:.2f}, {r['ci_high']:.2f}]  "
              f"p={r['p_value']:.2e}")
    print()


def minority_write_more(df):
    """Percentage more words written by minority respondents vs. majority reference group."""
    print("─" * 60)
    print("Minority respondents write longer responses")
    print("(OLS, word count normalized by majority group mean)")
    print("─" * 60)
    for identity, text_col in OPEN_TEXT_COLS.items():
        work = df[[identity, text_col]].copy()
        if identity == "gender":
            work[identity] = work[identity].map(_collapse_gender)
        # work = _filter_small_groups(work, identity)
        work["minority"]   = _binarize_minority(work[identity], REFERENCE_GROUPS[identity])
        work["word_count"] = work[text_col].apply(_word_count)
        ref_mean           = work.loc[work["minority"] == 0, "word_count"].mean()
        work["normalized"] = work["word_count"] / ref_mean
        r = _ols_coefficient(work, "normalized", "minority")
        print(f"  {identity:25s}  {r['coef']*100:+.1f}%  "
              f"95% CI=[{r['ci_low']*100:.1f}%, {r['ci_high']*100:.1f}%]  "
              f"p={r['p_value']:.2e}")
    print()


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nLoading {DATA_PATH.name} …")
    df = _load_survey(DATA_PATH)
    print(f"  N = {len(df)} respondents\n")

    word_length_statistics(df)
    perception_word_length_statistics(df)
    free_text_adds_context(df)
    minority_more_likely(df)
    minority_write_more(df)
