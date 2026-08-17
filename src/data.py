from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .config import CLASS_ORDER, DATA_PATH, TARGET


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(column).replace("\ufeff", "").strip() for column in out.columns]
    if "Nacionality" in out.columns and "Nationality" not in out.columns:
        out = out.rename(columns={"Nacionality": "Nationality"})
    return out


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found at {source}. Run `python -m scripts.fetch_data` first.")
    df = normalize_columns(pd.read_csv(source, sep=None, engine="python", encoding="utf-8-sig"))
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}'.")
    return df


def file_sha256(path: Path | str = DATA_PATH) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(df: pd.DataFrame) -> dict[str, object]:
    target_values = sorted(df[TARGET].dropna().astype(str).unique().tolist())
    expected = sorted(CLASS_ORDER)
    source_features = df.drop(columns=[TARGET])
    numeric = source_features.apply(pd.to_numeric, errors="coerce")
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "original_feature_count": len(df.columns) - 1,
        "duplicates": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "target_classes": target_values,
        "target_classes_valid": target_values == expected,
        "unexpected_non_numeric_feature_cells": int(numeric.isna().sum().sum() - source_features.isna().sum().sum()),
    }
