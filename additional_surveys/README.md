# Other surveys (Supplementary H)

Applies the framework to two public surveys with shorter free-text responses:

- **ANES most important problem.** "What do you think are the most important problems facing this country?" from the American National Election Studies 2020 and 2024 Time Series (10,789 non-empty responses; median 2 words).
- **Stantcheva (2021) estate tax.** The "Why?" follow-up to "Do you think that increasing the federal estate tax is a good or bad way to reduce wealth inequality?" from the "Understanding Tax Policy" survey (2,422 non-empty responses; median 11 words).

`results/` holds the fidelity scores for every (M, K) configuration and theme prevalence for the best one. These are enough to reproduce Table S30 and Figures S13–S14 without API calls:

```bash
python analysis/10_additional_surveys.py
```

## Re-running from scratch

1. Download the raw files into `additional_surveys/raw_data/` (not redistributed here):
   - From [electionstudies.org](https://electionstudies.org), the redacted open-ended responses for the 2020 and 2024 Time Series: `anes_timeseries_2020_redactedopenends_excel_20211118.xlsx` and `anes_timeseries_2024_redactedopenends_excel_20250923.xlsx`.
   - From the replication package for Stantcheva (2021), "Understanding Tax Policy: How Do People Reason?", *QJE* 136(4): `data_estate_raw.dta`.
2. Extract the free-text items:
   ```bash
   python additional_surveys/clean_data.py
   ```
3. Embed responses (`text-embedding-3-large`), train SAEs for M ∈ {16, 32, 64} and K ∈ {1, 4}, interpret and score each one, then annotate every response for the best configuration's themes. This needs an OpenAI key in `.env`:
   ```bash
   python additional_surveys/run_survey.py --dataset anes
   python additional_surveys/run_survey.py --dataset estate
   ```
   Interpretation uses GPT-4o (3 candidates per dimension). Fidelity scoring and annotation use GPT-4.1-mini with the stock hypothesaes annotation prompt. Fidelity is the harmonic mean of recall and specificity, the same definition as the main analysis. Results in `results/` are not recomputed, so delete them to rerun a configuration.

Several recovered themes describe response style rather than content (e.g., "Capitalized first letter"). As in the paper, all themes are reported unfiltered.
