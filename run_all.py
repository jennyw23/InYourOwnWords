#!/usr/bin/env python3
"""
Run the full In Your Own Words analysis pipeline.

Usage:
    python run_all.py              # run all scripts
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
     "Themes help explain life outcomes (Table 1)"),
    ("6", "analysis/6_baseline_comparisons.py",
     "Comparison to LDA and BERTopic baseline topic models"),
    ("7", "analysis/7_robustness_sae_params.py",
     "Robustness to SAE hyperparameters"),
    ("8", "analysis/8_perception_themes.py",
     "Perceived identity themes (perception discordance analysis)"),
]


def main():
    requested = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    for num, path, description in SCRIPTS:
        if requested and num not in requested:
            continue

        script = ROOT / path
        print(f"\n{'=' * 70}")
        print(f"  [{num}] {description}")
        print(f"      {path}")
        print("=" * 70)

        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(script.parent),
        )
        if result.returncode != 0:
            print(f"\nERROR: script {num} exited with code {result.returncode}")
            sys.exit(1)

    print(f"\n{'=' * 70}")
    print("  All analyses complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
