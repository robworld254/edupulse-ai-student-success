import pandas as pd
import pytest

from src.inference import load_model, load_results, predict_batch, validate_batch
from src.localization import DEMO_PROFILES, FEE_STATUS


def _profile_frame(name: str) -> pd.DataFrame:
    profile = DEMO_PROFILES[name]
    tuition, debtor = FEE_STATUS[profile["fee_status"]]
    return pd.DataFrame(
        [
            {
                "units_registered": profile["units_registered"],
                "units_passed": profile["units_passed"],
                "average_mark_pct": profile["average_mark_pct"],
                "assessments_completed": profile["assessments_completed"],
                "tuition_up_to_date": tuition,
                "outstanding_fee_balance": debtor,
                "scholarship_support": int(profile["scholarship"] == "Yes"),
            }
        ]
    )


@pytest.mark.parametrize(
    ("name", "expected_priority"),
    [
        ("Strong Progress", "Lower Priority"),
        ("Monitor", "Monitor"),
        ("Higher Support Need", "Higher Priority"),
    ],
)
def test_demo_profiles_are_valid_and_predict(name, expected_priority):
    frame = _profile_frame(name)
    result = predict_batch(frame, load_model(), load_results()["risk_bands"])
    assert len(result) == 1
    assert result.loc[0, "support_priority"] == expected_priority


def test_batch_validation_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_batch(pd.DataFrame({"units_registered": [6]}))


def test_batch_validation_rejects_impossible_units():
    frame = _profile_frame("Strong Progress")
    frame.loc[0, "units_passed"] = 9
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_batch(frame)
