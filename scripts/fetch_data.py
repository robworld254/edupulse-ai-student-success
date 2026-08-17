from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.config import DATA_PATH


def main() -> None:
    parser = ArgumentParser(description="Fetch UCI Dataset 697 without silently replacing raw data.")
    parser.add_argument("--force", action="store_true", help="Replace an existing raw file explicitly.")
    args = parser.parse_args()
    if DATA_PATH.exists() and not args.force:
        print(f"Raw dataset already exists and was left unchanged: {DATA_PATH}")
        return
    dataset = fetch_ucirepo(id=697)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        target = y.iloc[:, 0].astype(str)
    else:
        target = pd.Series(y, name="Target").astype(str)
    X.columns = [str(c).strip() for c in X.columns]
    if "Nacionality" in X.columns and "Nationality" not in X.columns:
        X = X.rename(columns={"Nacionality": "Nationality"})
    df = X.copy()
    df["Target"] = target.to_numpy()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = DATA_PATH.with_suffix(".download.csv")
    df.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(DATA_PATH)
    print(f"Saved {len(df):,} rows to {DATA_PATH}")
    print("Source: UCI ML Repository dataset 697 — Predict Students' Dropout and Academic Success")
    print("License: CC BY 4.0 | DOI: 10.24432/C5MC89")


if __name__ == "__main__":
    main()
