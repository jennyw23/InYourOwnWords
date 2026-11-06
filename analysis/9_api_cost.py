"""
Section 9: OpenAI API cost of the interpretation and annotation stages.

Token usage was not logged, so costs are reconstructed: real prompts are filled in
with real responses and themes, counted with tiktoken, and priced at OpenAI list
prices (fetched 2026-08-04). Reported in the paper's "Model access" paragraph.

Run from InYourOwnWords/:
    python analysis/9_api_cost.py
"""

import random
import statistics
from pathlib import Path

import hypothesaes
import pandas as pd
import tiktoken

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROMPTS_DIR = Path(hypothesaes.__file__).parent / "prompts"

ENC = tiktoken.get_encoding("o200k_base")  # GPT-4o / GPT-4.1 family

# USD per 1M tokens
PRICES = {
    "gpt-4o":       {"in": 2.50, "out": 10.00},
    "gpt-4o-mini":  {"in": 0.15, "out": 0.60},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
}

IDENTITY_COLS = {"race": "race_open", "gender": "gender_open", "sexual_orientation": "sexuality_open"}
QUESTION = {
    "race": "how would you describe your race and/or ethnicity?",
    "gender": "how would you describe your gender identity?",
    "sexual_orientation": "how would you describe your sexual orientation?",
}
M, N_CANDIDATES, N_INTERP_EXAMPLES, N_SCORING_EXAMPLES = 32, 3, 10, 100


def ntok(text):
    return len(ENC.encode(text))


def truncate(text, max_words):
    words = str(text).split()
    return " ".join(words[:max_words]) if len(words) > max_words else str(text)


def cost(model, in_tok, out_tok=0):
    return in_tok / 1e6 * PRICES[model]["in"] + out_tok / 1e6 * PRICES[model]["out"]


def task_instructions(identity):
    return (f"All of the texts are responses to the question:\nIn at least 2-3 sentences, {QUESTION[identity]}\n"
            "Features should describe a specific aspect of the response. For example:\n"
            '- "mentions relationship to ..."\n- "self-describes as ..."\n- "discusses ... they ..."\n')


def interpretation_tokens(df, themes, seed):
    """GPT-4o calls: 32 neurons x 3 candidates per identity, each with 10 positive + 10 negative examples."""
    template = (PROMPTS_DIR / "interpret-neuron-binary.txt").read_text()
    rng, total = random.Random(seed), 0
    for identity, col in IDENTITY_COLS.items():
        texts = [truncate(t, 256) for t in df[col].astype(str)]
        for _ in range(M * N_CANDIDATES):
            total += ntok(template.format(task_specific_instructions=task_instructions(identity),
                                          positive_texts=rng.sample(texts, N_INTERP_EXAMPLES),
                                          negative_texts=rng.sample(texts, N_INTERP_EXAMPLES)))
    return total


def annotation_tokens(template, df, themes, max_words, n_per_theme=None, seed=0):
    """(calls, input tokens) for annotating responses against themes with one prompt template."""
    rng, calls, total = random.Random(seed), 0, 0
    for identity, col in IDENTITY_COLS.items():
        texts = [truncate(t, max_words) for t in df[col].astype(str)]
        for theme in themes[identity]:
            sample = texts if n_per_theme is None else rng.sample(texts, n_per_theme)
            for text in sample:
                total += ntok(template.format(hypothesis=theme, text=text))
                calls += 1
    return calls, total


def output_tokens_per_call(template):
    """Range of answer lengths from the template's own few-shot outputs (real completions weren't logged)."""
    outs = [ntok(l[len("Output: "):]) for l in template.splitlines()
            if l.startswith("Output: ") and len(l) > len("Output: ")]
    return min(outs), statistics.mean(outs), max(outs)


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "in_your_own_words.csv")
    themes = {i: pd.read_csv(DATA_DIR / "fidelity" / f"{i}_interpretation_fidelity.csv")["interpretation"].astype(str).tolist()
              for i in IDENTITY_COLS}
    stock = (PROMPTS_DIR / "annotate.txt").read_text()
    identity_specific = (DATA_DIR / "validation" / "annotate.txt").read_text()

    # Interpretation (GPT-4o)
    samples = [interpretation_tokens(df, themes, s) for s in range(10)]
    in_tok = statistics.mean(samples)
    out_tok = sum(ntok(f'- "{t}"') for v in themes.values() for t in v) * N_CANDIDATES
    interp_cost = cost("gpt-4o", in_tok, out_tok)
    print(f"Interpretation (gpt-4o): {M * N_CANDIDATES * 3} calls, {in_tok:,.0f}±{statistics.stdev(samples):,.0f} in, "
          f"{out_tok:,} out tokens -> ${interp_cost:.2f}")

    # Full annotation (GPT-4.1-mini), 1,004 responses x 32 themes x 3 identities
    print("\nFull annotation (gpt-4.1-mini):")
    totals = {}
    for label, template in [("stock hypothesaes prompt", stock), ("identity-specific prompt", identity_specific)]:
        calls, tok = annotation_tokens(template, df, themes, max_words=128)
        lo, mean, hi = output_tokens_per_call(template)
        c_lo, c_mean, c_hi = (cost("gpt-4.1-mini", tok, calls * o) for o in (lo, mean, hi))
        totals[label] = c_mean
        print(f"  {label:<26} {calls:,} calls, {tok:,} in tokens -> ${c_lo:.2f}–${c_hi:.2f} (mean ${c_mean:.2f})")

    # Fidelity scoring: 100 examples per candidate interpretation
    calls, tok = annotation_tokens(stock, df, {i: v * N_CANDIDATES for i, v in themes.items()},
                                   max_words=256, n_per_theme=N_SCORING_EXAMPLES)
    _, mean, _ = output_tokens_per_call(stock)
    fid = {m: cost(m, tok, calls * mean) for m in ("gpt-4o-mini", "gpt-4.1-mini")}
    print(f"\nFidelity scoring: {calls:,} calls, {tok:,} in tokens -> "
          f"${fid['gpt-4o-mini']:.2f} (gpt-4o-mini) / ${fid['gpt-4.1-mini']:.2f} (gpt-4.1-mini)")

    print("\nTotal, interpretation + full annotation:")
    for label, c in totals.items():
        print(f"  {label:<26} ${interp_cost + c:.2f}")
