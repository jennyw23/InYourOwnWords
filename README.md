# **In Your Own Words**: Computationally identifying interpretable themes in free-text survey data

This code accompanies the paper `In Your Own Words:
Computationally identifying interpretable themes in free-text survey data`. If using either the data or the code, please cite the paper:

````
@article{InYourOwnWords2026, 
      title={In your own words: computationally identifying interpretable themes in free-text survey data}, 
      author={Jenny S Wang and Aliya Saperstein and Emma Pierson},
      journal={arXiv preprint arXiv:2603.26930},
      url={https://arxiv.org/abs/2603.26930},
      year={2026}
}
````

## Quickstart — try it on your own data

Want to run the SAE pipeline on your own free-text dataset? See [`quickstart/`](quickstart/), which provides a self-contained notebook (`train_and_interpret_sae.ipynb`) that works with any CSV. A worked example on the estate tax item from Stantcheva (2021) is included (`ex_train_and_interpret_sae.ipynb`).

## Data availability
If you would like to request access to the *In Your Own Words* dataset, follow the instructions on [Harvard Dataverse](https://doi.org/10.7910/DVN/YHUVXQ). The instructions will ask you to request access through Harvard Dataverse, and fill out a [Google Form](https://forms.gle/uUWFmsV8ZCz5Nq8V6) for details about your planned use case. We grant data access for research use. 

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

API keys are only needed to re-run LLM steps. Copy `.env.example` to `.env` at the repo root and fill in what you need: `OPENAI_KEY_SAE` for SAE interpretation, fidelity scoring, annotation, and embeddings (read by hypothesaes); `OPENAI_API_KEY` for the BERTopic and TopicGPT baselines; `DEEPINFRA_API_KEY` for the open-weight models in `open_weight/`.

## Reproducing the analysis

**Pre-computed files provided** — once the files from the data request are in `data/` (see below), you can reproduce all paper analyses without API calls:

```bash
python run_all.py
```

This runs the analysis scripts in order. You can also run individual scripts by number:

```bash
python run_all.py 1 2 5
```

| # | Script | Paper |
|---|--------|-------|
| 1 | `analysis/1_free_text_adds_context.py` | Minority respondents report free text adds context (main text) |
| 2 | `analysis/2_extract_themes.py` | Extracted themes and fidelity; Tables S4–S6 |
| 3 | `analysis/3_validate_themes.py` | Annotation agreement; Tables S2–S3, Figure S6 |
| 4 | `analysis/4_themes_by_category.py` | Themes vs. standardized categories; Tables S7–S10, Figures S7–S9 |
| 5 | `analysis/5_themes_explain_outcomes.py` | Themes and life outcomes; Table S11, Figures S10–S12 |
| 6 | `analysis/6_baseline_comparisons.py` | LDA and BERTopic baselines; Tables S15–S20 (TopicGPT, S21–S23, is in `analysis/baselines/topicgpt/topicgpt_topics/`) |
| 7 | `analysis/7_robustness_sae_params.py` | M=16 and M=64; Tables S24–S29 |
| 8 | `analysis/8_perception_themes.py` | Perceived identity themes; Tables S12–S14 |
| 9 | `analysis/9_api_cost.py` | OpenAI API cost (Methods, "Model access") |
| 10 | `analysis/10_additional_surveys.py` | ANES and estate tax surveys; Table S30, Figures S13–S14 |
| 11 | `analysis/11_open_weight_models.py` | Open-weight models; Table S31 |

SAE runtime (Methods) comes from `scripts/benchmark_sae_timing.ipynb`, with results in `data/sae_timing_benchmark.csv` and `data/runtime_log.csv`.

**Re-running from scratch** — the data request also provides the raw survey responses and pre-computed embeddings needed to rerun the LLM steps:

| Step | Script | What it does |
|------|--------|--------------|
| Train self-description SAEs | `scripts/train_sae.py` | Trains SAEs on free-text identity responses; generates and scores theme interpretations |
| Train perception SAEs | `scripts/train_sae_perception.py` | Same, for perceived-identity responses (dropping "Mostly the same") |
| Train robustness SAEs | `scripts/train_sae_robustness.py` | Repeats training at M=16 and M=64 for robustness checks |
| Annotate themes | `scripts/annotate_themes.py` | LLM annotation of every theme for every respondent |
| Run LDA / BERTopic | `analysis/6_baseline_comparisons.py` | Trains LDA and BERTopic baseline models |
| Run TopicGPT | `analysis/baselines/topicgpt/topicgpt_run.py` | Runs TopicGPT (pre-computed outputs included in the data request) |
| Other surveys | `additional_surveys/` | ANES and estate tax data (Supplementary H); see its README |
| Open-weight models | `open_weight/` | Qwen embeddings, interpretation, and theme matching (Supplementary I); usage in each script's docstring |

The data request provides the files that contain the survey text or could be used to recover it:

- **`data/in_your_own_words.csv`** — raw survey responses (required for all scripts)
- **`data/embeddings/`** — pre-computed OpenAI embeddings for all identity and perception questions (required for SAE training and BERTopic)
- **`data/validation/`** — human and LLM annotations used for inter-rater reliability (Tables S2, S3, S31), and the identity-specific annotation prompt, whose examples are real responses
- **`analysis/baselines/`** — saved LDA and BERTopic models (both store responses) and the TopicGPT inputs

Included in the repo, since they contain no text:

- **`data/annotations/`** — LLM-annotated binary theme indicators (1,004 respondents × 32 themes per identity)
- **`data/fidelity/`** — SAE interpretation fidelity scores (self-description and perception themes, M=16/32/64)
- **`data/checkpoints/`** — pre-trained SAE model weights
