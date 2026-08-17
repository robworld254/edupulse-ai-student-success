from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import CLASS_ORDER, INPUT_FEATURES, MODEL_PATH, RESULTS_PATH


def load_results(path: Path = RESULTS_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Training results are missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(path: Path = MODEL_PATH) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Trained model is missing: {path}")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated",
            category=DeprecationWarning,
        )
        return joblib.load(path)


def support_priority(dropout_probability: float, risk_bands: dict[str, Any]) -> str:
    higher = float(risk_bands["higher_threshold"])
    monitor = float(risk_bands["monitor_threshold"])
    if dropout_probability >= higher:
        return "Higher Priority"
    if dropout_probability >= monitor:
        return "Monitor"
    return "Lower Priority"


def validate_batch(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in INPUT_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    clean = frame[INPUT_FEATURES].copy()
    for column in INPUT_FEATURES:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if clean.isna().any().any():
        bad = clean.columns[clean.isna().any()].tolist()
        raise ValueError(f"Non-numeric or missing values found in: {', '.join(bad)}")
    if (clean[["units_registered", "units_passed", "assessments_completed"]] < 0).any().any():
        raise ValueError("Unit and assessment counts cannot be negative.")
    if (clean["units_passed"] > clean["units_registered"]).any():
        raise ValueError("Units passed cannot exceed units registered.")
    if (~clean["average_mark_pct"].between(0, 100)).any():
        raise ValueError("average_mark_pct must be between 0 and 100.")
    for column in ["tuition_up_to_date", "outstanding_fee_balance", "scholarship_support"]:
        if (~clean[column].isin([0, 1])).any():
            raise ValueError(f"{column} must contain only 0 or 1.")
    return clean


def predict_batch(frame: pd.DataFrame, model: Any, risk_bands: dict[str, Any]) -> pd.DataFrame:
    clean = validate_batch(frame)
    probabilities = model.predict_proba(clean)
    predictions = model.predict(clean)
    result = frame.copy().reset_index(drop=True)
    class_indices = {str(label): index for index, label in enumerate(model.classes_)}
    for label in CLASS_ORDER:
        result[f"probability_{label.lower()}"] = probabilities[:, class_indices[label]]
    result["predicted_outcome"] = predictions
    result["support_priority"] = [
        support_priority(value, risk_bands) for value in result["probability_dropout"].to_numpy()
    ]
    sums = result[[f"probability_{label.lower()}" for label in CLASS_ORDER]].sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise RuntimeError("Model probability outputs do not sum to one.")
    return result
