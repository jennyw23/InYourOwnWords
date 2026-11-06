#!/usr/bin/env python3
"""
Run the full In Your Own Words analysis pipeline.

Usage:
    python run_all.py              # run all scripts; those missing restricted data are skipped
    python run_all.py 1 2 5        # run only scripts 1, 2, and 5
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    ("1", "analysis/1_free_text_adds_context.py",
     "Minority respondents share more context"),
    ("2", "analysis/2_extract_themes.py",
     "Computationally extracting interpretable themes"),
    ("3", "analysis/3_validate_themes.py",
     "Validating extracted themes"),
    ("4", "analysis/4_themes_by_category.py",
     "Theme alignment with standardized categories"),
    ("5", "analysis/5_themes_explain_outcomes.py",
     "Themes help explain life outcomes (Table S11)"),
    ("6", "analysis/6_baseline_comparisons.py",
     "Comparison to LDA and BERTopic baseline topic models"),
    ("7", "analysis/7_robustness_sae_params.py",
     "Robustness to SAE hyperparameters"),
    ("8", "analysis/8_perception_themes.py",
     "Perceived identity themes (perception discordance analysis)"),
    ("9", "analysis/9_api_cost.py",
     "OpenAI API cost of interpretation and annotation"),
    ("10", "analysis/10_additional_surveys.py",
     "Other surveys: ANES and estate tax (Supplementary H)"),
    ("11", "analysis/11_open_weight_models.py",
     "Open-weight models (Supplementary I)"),
]


# signs that a script stopped because restricted data isn't present
MISSING_DATA = ("FileNotFoundError", "requires data access", "data request")


def main():
    requested = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    status = []

    for num, path, description in SCRIPTS:
        if requested and num not in requested:
            continue

        script = ROOT / path
        print(f"\n{'=' * 70}")
        print(f"  [{num}] {description}")
        print(f"      {path}")
        print("=" * 70, flush=True)

        # stdout streams as usual; stderr is captured to tell missing data from real errors
        result = subprocess.run([sys.executable, str(script)], cwd=str(script.parent),
                                stderr=subprocess.PIPE, text=True)
        sys.stderr.write(result.stderr)
        if result.returncode == 0:
            status.append((num, path, "ok"))
        elif any(m in result.stderr for m in MISSING_DATA):
            status.append((num, path, "skipped: needs data from the data request"))
        else:
            status.append((num, path, f"FAILED (exit code {result.returncode})"))

    print(f"\n{'=' * 70}\n  Summary\n{'=' * 70}")
    for num, path, outcome in status:
        print(f"  [{num:>2}] {path:<42} {outcome}")
    if any(o.startswith("FAILED") for _, _, o in status):
        sys.exit(1)


if __name__ == "__main__":
    main()
