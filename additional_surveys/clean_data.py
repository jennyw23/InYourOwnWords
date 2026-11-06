"""
Extract the free-text items used in Supplementary H from the raw survey files.

  - ANES 2020 (V202211) and 2024 (V242171): most important problem facing the country
  - Stantcheva (2021) estate tax survey: "Why?" follow-up on whether raising the
    federal estate tax is a good way to reduce wealth inequality (Q06107/Q06207/Q06307,
    one column per randomized arm)

See README.md for where to download the raw files. Writes cleaned_data/*.csv.

Usage:
    python additional_surveys/clean_data.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw_data"
OUT_DIR = ROOT / "cleaned_data"

ANES_FILES = {
    # ANES labels the 2020 sheet V202011, but its header row is V202211 (most important problem)
    2020: ("anes_timeseries_2020_redactedopenends_excel_20211118.xlsx", "V202011"),
    2024: ("anes_timeseries_2024_redactedopenends_excel_20250923.xlsx", "V242171"),
}
ESTATE_FILE = "data_estate_raw.dta"
ESTATE_ARMS = {"Q06107": "arm 1", "Q06207": "arm 2", "Q06307": "arm 3"}


def clean_text(s, missing=("", "nan")):
    s = s.astype(str).str.strip()
    return s.mask(s.isin(missing))


def clean_anes():
    frames = []
    for year, (fname, sheet) in ANES_FILES.items():
        df = pd.read_excel(RAW_DIR / fname, sheet_name=sheet)
        df.columns = ["case_id", "mip_text"]
        df["mip_text"] = clean_text(df["mip_text"], missing=("", "nan", "-1", "-9", "-8", "-7"))
        df["year"] = year
        print(f"ANES {year}: {df['mip_text'].notna().sum()} non-missing of {len(df)}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)[["year", "case_id", "mip_text"]]
    out.to_csv(OUT_DIR / "anes_mip_open_ended.csv", index=False)


def clean_estate():
    raw = pd.read_stata(RAW_DIR / ESTATE_FILE, convert_categoricals=False)
    raw.index.name = "raw_row_id"
    raw = raw.reset_index()

    # the three arms are mutually exclusive; combine into one text column
    text = pd.Series(pd.NA, index=raw.index, dtype="object")
    arm = pd.Series(pd.NA, index=raw.index, dtype="object")
    for col, label in ESTATE_ARMS.items():
        values = clean_text(raw[col])
        mask = values.notna()
        text[mask], arm[mask] = values[mask], label

    out = pd.DataFrame({
        "raw_row_id": raw["raw_row_id"],
        "estate_mechanisms_arm": arm,
        "estate_why_reduce_inequality_text": text,
    })
    path = OUT_DIR / "stantcheva_estate_tax_opinion_cleaned.csv"
    out.to_csv(path, index=False)
    # count after reading back: pandas parses answers like "NA" or "None" as missing
    n = pd.read_csv(path)["estate_why_reduce_inequality_text"].notna().sum()
    print(f"Estate tax: {n} non-missing of {len(out)}")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    clean_anes()
    clean_estate()
