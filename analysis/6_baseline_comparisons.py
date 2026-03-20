"""
analysis/6_baseline_comparisons.py — Run LDA and BERTopic baseline topic models.

Delegates to the standalone scripts in analysis/baselines/:
  - baselines/lda/lda_analysis.py        (tomotopy LDA, K=32)
  - baselines/bertopic/bertopic_analysis.py  (HDBSCAN min_cluster_size=7 + GPT-4o-mini labels)

Pre-trained models are included in baselines/lda/lda_models/ and
baselines/bertopic/bertopic_models/, so the scripts will load existing
checkpoints unless --overwrite is passed.

TopicGPT (Pham et al., 2024) is run separately via
baselines/topicgpt/topicgpt_run.py, as it requires many OpenAI API calls.
Pre-computed TopicGPT outputs are stored in
baselines/topicgpt/data/output/default_prompt_full_dataset/.

Usage:
    python analysis/6_baseline_comparisons.py
    python analysis/6_baseline_comparisons.py --overwrite
"""

import subprocess
import argparse
import sys
from pathlib import Path

BASELINES_DIR = Path(__file__).resolve().parent / "baselines"

SCRIPTS = [
    BASELINES_DIR / "lda" / "lda_analysis.py",
    BASELINES_DIR / "bertopic" / "bertopic_analysis.py",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run LDA and BERTopic baseline analyses.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Retrain models from scratch, ignoring existing checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()

    for script in SCRIPTS:
        print(f"\n{'═'*60}")
        print(f"Running: {script.relative_to(BASELINES_DIR.parent)}")
        print("═" * 60)

        cmd = [sys.executable, str(script)]
        if args.overwrite:
            cmd.append("--overwrite")

        result = subprocess.run(cmd, cwd=script.parent)
        if result.returncode != 0:
            print(f"\nError: {script.name} exited with code {result.returncode}")
            sys.exit(result.returncode)

    print("\nDone.")


if __name__ == "__main__":
    main()
