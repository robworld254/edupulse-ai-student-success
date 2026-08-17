import numpy as np

from src.config import IDENTITY_FIELDS, INPUT_FEATURES, MODEL_PATH, SEMESTER_2_MARKER
from src.features import build_assessment_row
from src.inference import load_model, load_results


def _demo_row():
    return build_assessment_row(
        units_registered=6,
        units_passed=4,
        average_mark_pct=55,
        assessments_completed=6,
        tuition_up_to_date=1,
        outstanding_fee_balance=0,
        scholarship_support=0,
    )


def test_saved_artifact_loads_and_predicts():
    assert MODEL_PATH.exists()
    model = load_model()
    assert model.predict(_demo_row())[0] in {"Dropout", "Enrolled", "Graduate"}
    pipeline = model.calibrated_classifiers_[0].estimator
    assert list(pipeline.named_steps) == [
        "feature_engineering",
        "imputer",
        "scaler",
        "model",
    ]


def test_probability_outputs_sum_to_one():
    probabilities = load_model().predict_proba(_demo_row())
    assert probabilities.shape == (1, 3)
    assert np.isclose(probabilities.sum(), 1.0)


def test_semester_two_leakage_is_impossible():
    results = load_results()
    contract = results["leakage_control"]["pipeline_input_contract"]
    assert contract == INPUT_FEATURES
    assert not any(SEMESTER_2_MARKER.lower() in name.lower() for name in contract)
    assert results["leakage_control"]["semester_2_allowed"] is False


def test_identity_fields_cannot_enter_pipeline():
    contract = set(load_results()["leakage_control"]["pipeline_input_contract"])
    assert contract.isdisjoint(IDENTITY_FIELDS)
    assert load_results()["leakage_control"]["identity_allowed"] is False
