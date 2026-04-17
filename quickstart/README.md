This directory contains code to run the **In Your Own Words** framework on any text dataset with categorical grouping variables.

## Files

| File | Description |
|------|-------------|
| `train_and_interpret_sae.ipynb` | Template notebook — fill in your own dataset and run the full pipeline |
| `ex_train_and_interpret_sae.ipynb` | Worked example using the Women's E-Commerce Clothing Reviews dataset |
| `quickstart_figures.py` | Plotting utilities for visualizing theme distributions across groups |

## Pipeline overview

Both notebooks run the same 7-step pipeline:

1. **Load** — read texts from a CSV file
2. **Embed** — encode texts with an OpenAI embedding model (`text-embedding-3-large`); embeddings are cached to avoid re-computing
3. **Train SAE** — fit a sparse autoencoder (M=32 dimensions, K=4 active per response) on the embeddings; checkpoint is saved and reused on subsequent runs
4. **Interpret** — use an LLM to generate candidate theme labels for each SAE dimension, then score each candidate's fidelity (F1 against LLM annotations) and keep the best
5. **Save** — write the best interpretation and its F1 fidelity score for each dimension to a CSV
6. **Annotate** — for themes above a fidelity threshold (default F1 ≥ 0.5), classify every response as activating that theme (1) or not (0) using an LLM annotator
7. **Analyze** — visualize which themes are more or less prevalent across any categorical variable (e.g. department, rating, demographic group) using a stacked bar chart

## Example notebook

`ex_train_and_interpret_sae.ipynb` walks through the full pipeline on a 500-response sample from the [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) dataset (downloaded via `kagglehub`). Step 7 demonstrates theme analysis across three categorical variables:

- **Department Name** — Tops, Dresses, Bottoms, Intimate, Jackets, Trend
- **Rating** — 1–5 stars (what do satisfied vs. dissatisfied reviewers mention?)
- **Division Name** — General, General Petite, Initmates

## Requirements

- Set `OPENAI_API_KEY` in a `.env` file or as an environment variable
- Install dependencies: `hypothesaes`, `openai`, `torch`, `pandas`, `matplotlib`, `python-dotenv`
- For the example notebook: `kagglehub`
