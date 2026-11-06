"""
Runs the pipeline with open-weight models in place of OpenAI: `Qwen/Qwen3-Embedding-8B` embeddings, a new SAE (M=32, K=4), and `Qwen/Qwen3-32B` (thinking disabled) for interpretation, fidelity scoring, and annotation.

These models can be run entirely locally. The results here were produced by serving them through an open-weight model provider, [DeepInfra](https://deepinfra.com). To run them yourself instead, serve the same models with any OpenAI-compatible server (e.g. vLLM) and set `DEEPINFRA_BASE_URL` to its address.

Usage (needs DEEPINFRA_API_KEY in .env):
    python open_weight/run_pipeline.py [--identity gender]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
OUT_DIR = HERE / "output"
FIDELITY_DIR = HERE / "results"

sys.path.insert(0, str(HERE))
import deepinfra  # noqa: E402

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
LLM = "Qwen/Qwen3-32B"
M, K = 32, 4
IDENTITIES = ["race", "gender", "sexual_orientation"]
TEXT_COLS = {"race": "race_open", "gender": "gender_open", "sexual_orientation": "sexuality_open"}
# same instructions as scripts/train_sae.py
TASK_INSTRUCTIONS = {
    "race": """All of the texts are responses to the question:
In at least 2-3 sentences, how would you describe your race and/or ethnicity?
Features should describe a specific aspect of the response. For example:
- "mentions relationship to ..."
- "self-describes as ...."
- "discusses ... they ..."
""",
    "gender": """All of the texts are responses to the question:
In at least 2-3 sentences, how would you describe your gender identity?
Features should describe a specific aspect of the response. For example:
- "mentions relationship to ..."
- "self-describes as ..."
- "discusses ... they ..."
""",
    "sexual_orientation": """All of the texts are responses to the question:
In at least 2-3 sentences, how would you describe your sexual orientation?
Features should describe a specific aspect of the response. For example:
- "mentions relationship to ..."
- "self-describes as ..."
- "discusses ... they ..."
""",
}
MAX_ANNOTATION_TOKENS = 50  # Yes/No plus a short reason; DeepInfra's default max_tokens can exceed the context


def load_sae(embeddings, identity, overwrite):
    import hypothesaes.quickstart as qs
    import hypothesaes.sae as sae_mod

    def load_model(path, device="cuda" if torch.cuda.is_available() else "cpu"):
        # older checkpoints lack the `threshold` buffer; torch>=2.6 needs weights_only=False
        ckpt = torch.load(path, pickle_module=pickle, weights_only=False)
        model = sae_mod.SparseAutoencoder(**ckpt["config"]).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        return model

    sae_mod.load_model = qs.load_model = load_model
    return qs.train_sae(embeddings=embeddings, M=M, K=K, overwrite_checkpoint=overwrite,
                        checkpoint_dir=str(OUT_DIR / "checkpoints" / f"M={M}_K={K}_{identity}"))


def interpret_and_score(texts, embeddings, sae, identity):
    from hypothesaes.interpret_neurons import (
        InterpretConfig, LLMConfig, NeuronInterpreter, SamplingConfig, ScoringConfig,
    )

    X = torch.tensor(embeddings, dtype=torch.float32).to(next(sae.parameters()).device)
    activations = sae.get_activations(X)
    interpreter = NeuronInterpreter(cache_name=f"qwen_M={M}_K={K}_{identity}", interpreter_model=LLM,
                                    annotator_model=LLM, n_workers_interpretation=8, n_workers_annotation=16)
    config = InterpretConfig(
        sampling=SamplingConfig(n_examples=20, max_words_per_example=256),
        llm=LLMConfig(temperature=0.7, max_interpretation_tokens=300,
                      llm_kwargs={"extra_body": deepinfra.no_thinking(LLM)}),
        n_candidates=3,
        task_specific_instructions=TASK_INSTRUCTIONS[identity],
    )
    neurons = list(range(M))
    interpretations = interpreter.interpret_neurons(texts=texts, activations=activations,
                                                    neuron_indices=neurons, config=config)
    metrics = interpreter.score_interpretations(
        texts=texts, activations=activations, interpretations=interpretations,
        config=ScoringConfig(n_examples=100), max_output_tokens=MAX_ANNOTATION_TOKENS,
        extra_body=deepinfra.no_thinking(LLM),
    )
    # hypothesaes' precision/recall F1, not the paper's recall/specificity metric
    rows = []
    for idx in neurons:
        best = max(interpretations[idx], key=lambda c: metrics[idx][c]["f1"])
        rows.append({"neuron_idx": idx, "interpretation": best, "f1_fidelity_score": metrics[idx][best]["f1"]})
    return pd.DataFrame(rows)


def run_identity(identity, overwrite):
    print(f"\n── {identity} ──")
    df = pd.read_csv(DATA_DIR / "in_your_own_words.csv", usecols=["ResponseId", TEXT_COLS[identity]])
    texts = df[TEXT_COLS[identity]].tolist()

    emb_path = OUT_DIR / "embeddings" / f"{identity}_embeddings.npy"
    if emb_path.exists():
        embeddings = np.load(emb_path)
    else:
        embeddings = deepinfra.embed(texts, EMBEDDING_MODEL)
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embeddings)
        np.save(emb_path.parent / "response_ids.npy", df["ResponseId"].to_numpy())

    sae = load_sae(embeddings, identity, overwrite)
    fidelity = interpret_and_score(texts, embeddings, sae, identity)
    FIDELITY_DIR.mkdir(parents=True, exist_ok=True)
    out = FIDELITY_DIR / f"{identity}_interpretation_fidelity.csv"
    fidelity.to_csv(out, index=False)
    print(f"  saved {out.relative_to(ROOT)}; median fidelity {fidelity['f1_fidelity_score'].median():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", choices=IDENTITIES + ["all"], default="all")
    parser.add_argument("--overwrite", action="store_true", help="retrain SAEs even if a checkpoint exists")
    args = parser.parse_args()

    if not (DATA_DIR / "in_your_own_words.csv").exists():
        sys.exit("This script requires data access: data/in_your_own_words.csv is provided with the "
                 "data request; see data/README.md.")
    deepinfra.patch_hypothesaes(cache_dir=str(OUT_DIR / "annotation_cache"))
    for identity in IDENTITIES if args.identity == "all" else [args.identity]:
        run_identity(identity, args.overwrite)
