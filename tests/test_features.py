import numpy as np
import pandas as pd

from src.config import ENGINEERED_FEATURES, INPUT_FEATURES
from src.features import (
    Semester1FeatureTransformer,
    build_assessment_row,
    engineer_compact_inputs,
    engineer_semester1_features,
)


def _raw_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Curricular units 1st sem (enrolled)": [6],
            "Curricular units 1st sem (evaluations)": [8],
            "Curricular units 1st sem (approved)": [4],
            "Curricular units 1st sem (grade)": [12.0],
            "Tuition fees up to date": [1],
            "Debtor": [0],
            "Scholarship holder": [1],
            "Target": ["Enrolled"],
        }
    )


def test_engineer_semester1_features_and_grade_conversion():
    output = engineer_semester1_features(_raw_row())
    assert output.loc[0, "pass_rate"] == 4 / 6
    assert output.loc[0, "average_mark_pct"] == 60.0
    assert output.loc[0, "units_not_passed"] == 2
    assert output.loc[0, "Target"] == "Enrolled"


def test_divide_by_zero_is_finite():
    row = build_assessment_row(
        units_registered=0,
        units_passed=0,
        average_mark_pct=0,
        assessments_completed=0,
        tuition_up_to_date=0,
        outstanding_fee_balance=0,
        scholarship_support=0,
    )
    output = engineer_compact_inputs(row)
    assert np.isfinite(output.to_numpy()).all()
    assert output.loc[0, "pass_rate"] == 0
    assert output.loc[0, "assessments_per_unit"] == 0


def test_ui_mapping_caps_ranges_and_has_exact_contract():
    row = build_assessment_row(
        units_registered=6,
        units_passed=9,
        average_mark_pct=120,
        assessments_completed=8,
        tuition_up_to_date=1,
        outstanding_fee_balance=0,
        scholarship_support=0,
    )
    assert row.columns.tolist() == INPUT_FEATURES
    assert row.loc[0, "units_passed"] == 6
    assert row.loc[0, "average_mark_pct"] == 100


def test_sklearn_transformer_outputs_engineered_contract():
    row = build_assessment_row(
        units_registered=6,
        units_passed=5,
        average_mark_pct=65,
        assessments_completed=8,
        tuition_up_to_date=1,
        outstanding_fee_balance=0,
        scholarship_support=0,
    )
    transformer = Semester1FeatureTransformer().fit(row)
    assert transformer.transform(row).columns.tolist() == ENGINEERED_FEATURES
