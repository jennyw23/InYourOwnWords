This directory contains code to run the **In Your Own Words** framework on any text dataset with categorical grouping variables.

## Files

| File | Description |
|------|-------------|
| `train_and_interpret_sae.ipynb` | Template notebook — fill in your own dataset and run the full pipeline |
| `ex_train_and_interpret_sae.ipynb` | Worked example on the estate tax item from Stantcheva (2021) |
| `data/stantcheva_estate_tax.csv` | Data for the example: 500 free-text answers plus three grouping columns |
| `quickstart_figures.py` | Plotting utilities for visualizing theme distributions across groups |

## Pipeline overview

Both notebooks run the same 7-step pipeline:

1. **Load** — read texts from a CSV file
2. **Embed** — encode texts with an OpenAI embedding model (`text-embedding-3-large`); embeddings are cached to avoid re-computing
3. **Train SAE** — fit a sparse autoencoder (M=32 dimensions, K=4 active per response) on the embeddings; checkpoint is saved and reused on subsequent runs
4. **Interpret** — use an LLM to generate candidate theme labels for each SAE dimension, then score each candidate's fidelity (harmonic mean of recall and specificity against LLM annotations) and keep the best
5. **Save** — write the best interpretation and its fidelity score for each dimension to a CSV (column `f1_fidelity_score`, a name kept for compatibility)
6. **Annotate** — for themes above a fidelity threshold (default fidelity ≥ 0.5), classify every response as activating that theme (1) or not (0) using an LLM annotator
7. **Analyze** — visualize which themes are more or less prevalent across any categorical variable (e.g. party, demographic group) using a stacked bar chart

## Example notebook

`ex_train_and_interpret_sae.ipynb` runs the pipeline on the estate tax item from Stantcheva (2021), "Understanding Tax Policy: How Do People Reason?", *QJE* 136(4), also analyzed in Supplementary H. Respondents answered "Do you think that increasing the federal estate tax is a good or bad way to reduce wealth inequality?" and then "Why?". The CSV is a random sample of 500 of the 2,422 non-empty answers, with only the columns the notebook uses. The notebook fits M=16 themes (K=4) to keep API costs low. The full survey is in the [replication package](https://socialeconomicslab.org/resources-and-data/).

- `why` — the free-text answer
- `estate_tax_view` — the closed-ended answer it explains, from "one of the best ways" to "one of the worst ways"
- `political_party` — Democrat, Republican, Independent, Non-Affiliated, Other
- `gender` — Female, Male, Other
- `response_id` — row in `data_estate_raw.dta`

**Data source:** Stantcheva, Stefanie. 2021. "Understanding Tax Policy: How Do People Reason?" *Quarterly Journal of Economics* 136(4): 2309–2369. Data from the [Social Economics Lab](https://socialeconomicslab.org/resources-and-data/).

## Requirements

- Set `OPENAI_API_KEY` in a `.env` file or as an environment variable
- Install dependencies: `hypothesaes`, `openai`, `torch`, `pandas`, `matplotlib`, `python-dotenv`
