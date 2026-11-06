# Data

The survey data (`in_your_own_words.csv`) is available upon request for research use from [Harvard Dataverse](https://doi.org/10.7910/DVN/YHUVXQ)

Files that contain the survey text, or could be used to recover it, come with the data request: the survey responses, the embeddings (`embeddings/`), and the validation sample and annotation prompt (`validation/`). Place them here before running the analysis scripts. The fidelity scores, annotation indicators, and SAE checkpoints contain no text and are included in the repo.

## Directory layout

```
data/
├── in_your_own_words.csv                              # Survey responses (N = 1,004)
├── codebook.csv                                       # Full variable descriptions
│
├── annotations/                                       # LLM-annotated theme indicators
│   ├── race_annotations.csv                           # 1,004 rows × 32 themes (binary); analyses keep 26
│   ├── gender_annotations.csv                         # 1,004 rows × 32 themes; analyses keep 27
│   └── sexual_orientation_annotations.csv             # 1,004 rows × 32 themes; analyses keep 28
│
├── fidelity/                                          # Fidelity (harmonic mean of recall and specificity)
│   ├── race_interpretation_fidelity.csv               # M=32 self-description themes
│   ├── gender_interpretation_fidelity.csv
│   ├── sexual_orientation_interpretation_fidelity.csv
│   ├── race_interpretation_fidelity_m16.csv           # robustness: M=16 and M=64 variants
│   ├── ...
│   ├── race_perceive_interpretation_fidelity.csv      # perception themes
│   ├── gender_perceive_interpretation_fidelity.csv    #   (respondents who did not answer
│   └── sexual_orientation_perceive_interpretation_fidelity.csv  #   "Mostly the same")
│
├── embeddings/                                        # Pre-computed embeddings (text-embedding-3-large)
│   ├── response_ids.npy                               # Shared ResponseId order (N = 1,004)
│   ├── race_embeddings.npy                            # Self-description embeddings
│   ├── gender_embeddings.npy
│   ├── sexual_orientation_embeddings.npy
│   ├── race_perceive_embeddings.npy                   # Perception-question embeddings
│   ├── gender_perceive_embeddings.npy
│   └── sexual_orientation_perceive_embeddings.npy
│
├── checkpoints/                                       # Pre-trained SAE model weights
│   ├── M=32_K=4_race/                                 # Self-description SAEs
│   ├── M=16_K=4_race/                                 # robustness variants
│   ├── ...                                            # (M=16, M=32, M=64 for all identities)
│   ├── perception_race_drop_same/                     # Perception SAEs (M=32, K=4)
│   └── ...
│
└── validation/                                        # Human and LLM annotations (Tables S2, S3, S31)
    ├── annotated-race-sample.csv                      # Human-annotated (100 rows × 8 themes)
    ├── annotated-gender-sample.csv
    ├── annotated-sexual_orientation-sample.csv
    ├── annotated-race-gpt-4.1-mini-identity-specific.csv
    ├── ...                                            # (other model/prompt combos for Table S2)
    ├── annotated-race-qwen3-32b-identity-specific.csv # Qwen3-32B (Table S31)
    ├── ...
    └── annotate.txt                                   # Identity-specific annotation prompt (Figure S5)
```

## Survey variables

Key columns in `in_your_own_words.csv`:

| Column | Description |
|--------|-------------|
| `race_open` | Free-text race/ethnicity description |
| `gender_open` | Free-text gender description |
| `sexuality_open` | Free-text sexual orientation description |
| `race_details` | "Does your free-text add important information?" (Yes/No) for race |
| `gender_details` | Same for gender |
| `sexuality_details` | Same for sexual orientation |
| `race_closed` | Standardized race/ethnicity (multi-select) |
| `describe_gender` | Standardized gender (Man / Woman / Some other way) |
| `gender_trans` | Transgender status (Yes / No / Prefer not to answer) |
| `sexuality_closed` | Standardized sexual orientation (multi-select) |
| `identity_import_1` | Importance of race identity (5-point scale) |
| `identity_import_2` | Importance of gender identity |
| `identity_import_3` | Importance of SO identity |
| `physical_health` | Self-rated physical health (Poor–Excellent) |
| `mental_health` | Self-rated mental health |
| `life_satisfaction` | Life satisfaction (1–10) |
| `discrim_personal_1`–`_5` | Everyday discrimination frequency items |
| `income` | Household income bracket |

See `codebook.csv` for full variable descriptions.
