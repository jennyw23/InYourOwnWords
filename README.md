# **In Your Own Words**: Computationally identifying interpretable themes in free-text survey data

This code accompanies the paper `In Your Own Words:
Computationally identifying interpretable themes in free-text survey data`. If using either the data or the code, please cite the paper:

````
Add citation here
````

## Data availability
If you would like to request access to the *In Your Own Words* dataset, follow the instructions on the project website. We grant data access for research use. More information about the available variables are described in the [codebook](https://docs.google.com/spreadsheets/d/1UIqBnTKMBFFVU_sbcRfJyRT9xyuPz09e8LgbQtjbbIs/edit?gid=0#gid=0) and [data request form](https://docs.google.com/forms/d/e/1FAIpQLSdn29f5FPxeCmHPTNaXFTpF7kFQmIbziAqANjIK-WnTBY6ymA/viewform). 

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

An OpenAI API key is required for the LLM-based steps (SAE training, BERTopic relabeling, TopicGPT). Set it in a `.env` file at the repo root:

```
OPENAI_API_KEY=sk-...
```

## Reproducing the analysis

**Pre-computed files provided** — you can reproduce all paper analyses without re-running the LLM-based steps. Once you have placed the data files in `data/` (see below), run:

```bash
python run_all.py
```

This executes all eight analysis scripts in order. You can also run individual scripts by number:

```bash
python run_all.py 1 2 5
```

**Re-running from scratch** — the data request form (see Data availability above) provides everything needed to reproduce the full pipeline, including raw survey responses and pre-computed embeddings. With those in place:

| Step | Script | What it does |
|------|--------|--------------|
| Train self-description SAEs | `scripts/train_sae.py` | Trains SAEs on free-text identity responses; generates and scores theme interpretations |
| Train perception SAEs | `scripts/train_sae_perception.py` | Same, for perceived-identity responses (dropping "Mostly the same") |
| Train robustness SAEs | `scripts/train_sae_robustness.py` | Repeats training at M=16 and M=64 for robustness checks |
| Annotate themes | `scripts/annotate_themes.py` | Re-runs LLM annotation of theme indicators across all respondents |
| Run LDA / BERTopic | `analysis/6_baseline_comparisons.py` | Trains LDA and BERTopic baseline models |
| Run TopicGPT | `analysis/baselines/topicgpt/topicgpt_run.py` | Runs TopicGPT (requires OpenAI API; pre-computed outputs already included) |

## Data

Survey data is available upon request for research use — see the Data availability section above. The data request form provides:

- **`data/in_your_own_words.csv`** — raw survey responses (required for all scripts)
- **`data/embeddings/`** — pre-computed OpenAI embeddings for all identity and perception questions (required for SAE training and BERTopic)

The following pre-computed intermediate files are included in this repository and are sufficient to reproduce all paper analyses without re-running LLM steps:

- **`data/annotations/`** — LLM-annotated binary theme indicators (1,004 respondents × themes per identity)
- **`data/fidelity/`** — SAE interpretation fidelity scores (self-description and perception themes, M=16/32/64)
- **`data/checkpoints/`** — pre-trained SAE model weights
- **`data/validation/`** — human and LLM annotations used for inter-rater reliability (Table S7)

See [`data/README.md`](data/README.md) for the full directory layout and variable descriptions.

## Repository structure

```
├── run_all.py                          # Runs all analysis scripts in order
├── analysis/
│   ├── 1_free_text_adds_context.py     # Minority respondents share more context
│   ├── 2_extract_themes.py             # Reporting extracted themes and fidelity scores
│   ├── 3_validate_themes.py            # Validating themes against human annotations
│   ├── 4_themes_by_category.py         # Theme alignment with standardized categories
│   ├── 5_themes_explain_outcomes.py    # Themes predict life outcomes (Table 1)
│   ├── 6_baseline_comparisons.py       # LDA and BERTopic baseline models
│   ├── 7_robustness_sae_params.py      # Robustness to SAE hyperparameters
│   ├── 8_perception_themes.py          # Perceived identity themes
│   └── baselines/
│       ├── lda/                        # LDA implementation (tomotopy, K=32)
│       ├── bertopic/                   # BERTopic implementation (HDBSCAN + GPT labels)
│       └── topicgpt/                   # TopicGPT implementation (Pham et al., 2024)
├── scripts/
│   ├── train_sae.py                    # Train SAEs on self-description responses
│   ├── train_sae_perception.py         # Train SAEs on perceived-identity responses
│   ├── train_sae_robustness.py         # Train SAEs at alternative hyperparameters
│   └── annotate_themes.py              # LLM annotation of theme indicators
├── src/
│   ├── data_helper.py                  # Data loading and identity grouping utilities
│   ├── sae_helper.py                   # SAE loading and theme filtering helpers
│   ├── regression_helper.py            # Regression utilities
│   └── make_figures.py                 # Figure generation
└── data/                               # See data/README.md
```