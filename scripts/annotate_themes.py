"""
scripts/annotate_themes.py — Annotate every response for every theme.

Labels each free-text response (rows, in survey order) against each SAE theme
(columns, in data/fidelity order) with GPT-4.1-mini and the identity-specific
prompt in prompts/annotate.txt (Figure S5). Writes data/annotations/{identity}_annotations.csv.

Usage:
    python scripts/annotate_themes.py
    python scripts/annotate_themes.py --identity race --model gpt-4.1-mini
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from hypothesaes.annotate import annotate_texts_with_concepts
from prompts import use_identity_specific_prompt

IDENTITY_TEXT_COLS = {
    "race": "race_open",
    "gender": "gender_open",
    "sexual_orientation": "sexuality_open",
}
MAX_WORDS_PER_EXAMPLE = 128


def annotate_identity(identity, model, n_workers):
    survey = pd.read_csv(DATA_DIR / "in_your_own_words.csv")
    texts = survey[IDENTITY_TEXT_COLS[identity]].astype(str).tolist()
    themes = pd.read_csv(DATA_DIR / "fidelity" / f"{identity}_interpretation_fidelity.csv")["interpretation"].tolist()

    print(f"{identity}: {len(texts)} responses x {len(themes)} themes with {model}")
    annotations = annotate_texts_with_concepts(
        texts=texts,
        concepts=themes,
        max_words_per_example=MAX_WORDS_PER_EXAMPLE,
        cache_name=f"annotate_{identity}_{model}",
        n_workers=n_workers,
        model=model,
    )

    out_dir = DATA_DIR / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{identity}_annotations.csv"
    pd.DataFrame(annotations)[themes].to_csv(out_path, index=False)
    print(f"  Saved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--identity", choices=[*IDENTITY_TEXT_COLS, "all"], default="all")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--n-workers", type=int, default=50)
    args = parser.parse_args()

    if not (DATA_DIR / "in_your_own_words.csv").exists():
        sys.exit("Survey data not found. See data/README.md for access instructions.")

    use_identity_specific_prompt()
    identities = list(IDENTITY_TEXT_COLS) if args.identity == "all" else [args.identity]
    for identity in identities:
        annotate_identity(identity, args.model, args.n_workers)
