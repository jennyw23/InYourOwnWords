"""
Matches each Qwen theme to its closest OpenAI theme, and each OpenAI theme to its closest
Qwen theme, by cosine similarity of the theme text (text-embedding-3-large). Writes two
review sheets whose `label` column is filled in by hand (same / related / different);
labels are kept for pairs that haven't changed.

Usage (needs OPENAI_KEY_SAE in .env):
    python open_weight/match_themes.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve().parent
OPENAI_DIR = HERE.parent / "data" / "fidelity"
RESULTS_DIR = HERE / "results"
IDENTITIES = ["race", "gender", "sexual_orientation"]


def embed(strings):
    from hypothesaes.embedding import get_openai_embeddings

    t2e = get_openai_embeddings(strings, model="text-embedding-3-large", cache_name=None, show_progress=False)
    E = np.vstack([t2e[s] for s in strings])
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def load(folder, identity):
    df = pd.read_csv(folder / f"{identity}_interpretation_fidelity.csv").sort_values("neuron_idx")
    return df["interpretation"].fillna("").tolist(), df["f1_fidelity_score"].tolist()


def match_identity(identity):
    new, _ = load(RESULTS_DIR, identity)
    old, old_f1 = load(OPENAI_DIR, identity)
    E = embed(new + old)
    S = E[: len(new)] @ E[len(new):].T  # [qwen, openai]

    best_old, best_new = S.argmax(1), S.argmax(0)
    forward = pd.DataFrame({
        "identity": identity, "new_idx": range(len(new)), "new_interpretation": new,
        "best_old_idx": best_old, "best_old_interpretation": [old[i] for i in best_old], "best_cos": S.max(1),
    })
    reverse = pd.DataFrame({
        "identity": identity, "old_idx": range(len(old)), "old_interpretation": old, "old_f1": old_f1,
        "best_new_idx": best_new, "best_new_interpretation": [new[i] for i in best_new], "best_cos": S.max(0),
    })
    print(f"{identity}: median cosine {np.median(forward['best_cos']):.3f} (Qwen -> OpenAI)")
    return forward, reverse


def write_sheet(sheet, keys, name):
    path = RESULTS_DIR / name
    if path.exists():
        sheet = sheet.merge(pd.read_csv(path)[keys + ["label"]], on=keys, how="left")
    else:
        sheet["label"] = ""
    sheet.to_csv(path, index=False)
    print(f"saved {path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    pairs = [match_identity(i) for i in IDENTITIES]
    write_sheet(pd.concat([f for f, _ in pairs], ignore_index=True),
                ["identity", "new_interpretation", "best_old_interpretation"], "theme_match_labels.csv")
    write_sheet(pd.concat([r for _, r in pairs], ignore_index=True),
                ["identity", "old_interpretation", "best_new_interpretation"], "reverse_manual_review.csv")
