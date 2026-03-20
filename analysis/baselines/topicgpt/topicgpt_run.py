"""
topicgpt/topicgpt_run.py — Run TopicGPT baseline on free-text identity responses.

Uses the official TopicGPT implementation (Pham et al., 2024) with default
prompt templates and no early stopping, so all responses are processed during
theme generation.  Level-2 subthemes are generated for each identity.

Requires:
    OPENAI_API_KEY environment variable (set in .env at repo root)

Pre-computed outputs are stored in data/output/default_prompt_full_dataset/.
Re-running will overwrite those files.

Usage:
    python analysis/baselines/topicgpt/topicgpt_run.py
"""

import json
import yaml
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from topicgpt_python import generate_topic_lvl1, generate_topic_lvl2

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_FILE = ROOT / "data" / "in_your_own_words.csv"
DATA_DIR = Path(__file__).parent / "data" / "input"


def generate_data(data_dir, df):
    """Write per-identity .jsonl files if the input directory does not exist."""
    if data_dir.exists():
        return
    data_dir.mkdir(parents=True)
    for col, name in [
        ("race_open", "race_open"),
        ("gender_open", "gender_open"),
        ("sexuality_open", "sexuality_open"),
    ]:
        convert_to_jsonl(df[col].fillna("").astype(str), data_dir, name)


def convert_to_jsonl(docs, data_dir, name):
    path = data_dir / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i, text in enumerate(docs):
            f.write(json.dumps({"id": str(i), "text": text, "label": ""}) + "\n")


def run_topicgpt(config, identity, dataset_size):
    print(f"Running TopicGPT for identity: {identity}")
    generate_topic_lvl1(
        "openai",
        "gpt-4o-mini",
        config[f"{identity}_data"],
        config["generation"]["prompt"],
        config["generation"]["seed"],
        config["generation"][f"{identity}_output"],
        config["generation"][f"{identity}_topic_output"],
        verbose=config["verbose"],
        early_stop=dataset_size,
    )
    if config["generate_subtopics"]:
        generate_topic_lvl2(
            "openai",
            "gpt-4o",
            config["generation"][f"{identity}_topic_output"],
            config["generation"][f"{identity}_output"],
            config["generation_2"]["prompt"],
            config["generation_2"][f"{identity}_output"],
            config["generation_2"][f"{identity}_topic_output"],
            verbose=True,
        )
    print(f"Completed TopicGPT for identity: {identity}")


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Survey data not found: {DATA_FILE}\nSee data/README.md for access instructions."
        )

    df = pd.read_csv(DATA_FILE)
    generate_data(DATA_DIR, df)

    config_path = Path(__file__).parent / "config_full.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for identity in ("race", "gender", "sexual_orientation"):
        run_topicgpt(config, identity, dataset_size=len(df))


if __name__ == "__main__":
    main()
