from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import ENGINEERED_FEATURES, INPUT_FEATURES, TARGET

RAW_REQUIRED = [
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Tuition fees up to date",
    "Debtor",
    "Scholarship holder",
]

WIDE_SAFE_FEATURES = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Daytime/evening attendance",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Tuition fees up to date",
    "Debtor",
    "Scholarship holder",
]


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(num, errors="coerce").fillna(0.0)
    denominator = pd.to_numeric(den, errors="coerce").fillna(0.0)
    values = np.divide(
        numerator,
        denominator,
        out=np.zeros(len(numerator), dtype=float),
        where=denominator.to_numpy() != 0,
    )
    return pd.Series(values, index=numerator.index, dtype=float)


def build_compact_inputs(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Map immutable UCI source columns to the seven-field deployment contract."""
    missing = [column for column in RAW_REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required Semester 1 fields: {missing}")

    out = pd.DataFrame(index=df.index)
    out["units_registered"] = pd.to_numeric(df["Curricular units 1st sem (enrolled)"], errors="coerce").clip(lower=0)
    out["units_passed"] = pd.to_numeric(df["Curricular units 1st sem (approved)"], errors="coerce").clip(lower=0)
    # The source grade is 0–20. Multiplication by five is a linear display
    # conversion to a familiar 0–100 scale; it does not claim Kenyan equivalence.
    out["average_mark_pct"] = (pd.to_numeric(df["Curricular units 1st sem (grade)"], errors="coerce") * 5.0).clip(
        0, 100
    )
    out["assessments_completed"] = pd.to_numeric(df["Curricular units 1st sem (evaluations)"], errors="coerce").clip(
        lower=0
    )
    out["tuition_up_to_date"] = pd.to_numeric(df["Tuition fees up to date"], errors="coerce").clip(0, 1)
    out["outstanding_fee_balance"] = pd.to_numeric(df["Debtor"], errors="coerce").clip(0, 1)
    out["scholarship_support"] = pd.to_numeric(df["Scholarship holder"], errors="coerce").clip(0, 1)
    if include_target and TARGET in df.columns:
        out[TARGET] = df[TARGET].astype(str)
    return out


def engineer_compact_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic, divide-by-zero-safe features from the UI contract."""
    missing = [column for column in INPUT_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Assessment input is missing required fields: {missing}")

    registered = pd.to_numeric(df["units_registered"], errors="coerce").fillna(0).clip(lower=0)
    passed = pd.to_numeric(df["units_passed"], errors="coerce").fillna(0).clip(lower=0)
    passed = pd.Series(np.minimum(passed, registered), index=df.index)
    assessments = pd.to_numeric(df["assessments_completed"], errors="coerce").fillna(0).clip(lower=0)

    out = pd.DataFrame(index=df.index)
    out["units_registered"] = registered
    out["units_passed"] = passed
    out["units_not_passed"] = (registered - passed).clip(lower=0)
    out["pass_rate"] = _safe_divide(passed, registered).clip(0, 1)
    out["average_mark_pct"] = pd.to_numeric(df["average_mark_pct"], errors="coerce").fillna(0).clip(0, 100)
    out["assessments_completed"] = assessments
    out["assessments_per_unit"] = _safe_divide(assessments, registered).clip(0, 10)
    for column in ["tuition_up_to_date", "outstanding_fee_balance", "scholarship_support"]:
        out[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).clip(0, 1)
    return out[ENGINEERED_FEATURES]


def engineer_semester1_features(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    compact = build_compact_inputs(df, include_target=include_target)
    engineered = engineer_compact_inputs(compact)
    if include_target and TARGET in compact.columns:
        engineered[TARGET] = compact[TARGET]
    return engineered


class Semester1FeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible feature engineering embedded in the saved pipeline."""

    def fit(self, X: pd.DataFrame, y: object = None) -> Semester1FeatureTransformer:
        self.feature_names_in_ = np.asarray(INPUT_FEATURES, dtype=object)
        self.n_features_in_ = len(INPUT_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=INPUT_FEATURES)
        return engineer_compact_inputs(X)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(ENGINEERED_FEATURES, dtype=object)


def build_assessment_row(
    *,
    units_registered: float,
    units_passed: float,
    average_mark_pct: float,
    assessments_completed: float,
    tuition_up_to_date: int,
    outstanding_fee_balance: int,
    scholarship_support: int,
) -> pd.DataFrame:
    registered = max(float(units_registered), 0.0)
    row = {
        "units_registered": registered,
        "units_passed": min(max(float(units_passed), 0.0), registered),
        "average_mark_pct": float(np.clip(average_mark_pct, 0, 100)),
        "assessments_completed": max(float(assessments_completed), 0.0),
        "tuition_up_to_date": int(bool(tuition_up_to_date)),
        "outstanding_fee_balance": int(bool(outstanding_fee_balance)),
        "scholarship_support": int(bool(scholarship_support)),
    }
    return pd.DataFrame([row], columns=INPUT_FEATURES)
