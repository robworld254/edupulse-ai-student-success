from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
DELIVERABLES = ROOT / "deliverables"

DATA_PATH = DATA_RAW / "student_dropout_success.csv"
MODEL_PATH = ARTIFACTS / "edupulse_model.joblib"
RESULTS_PATH = ARTIFACTS / "results.json"
METRICS_CSV_PATH = ARTIFACTS / "metrics.csv"
MODEL_COMPARISON_PATH = ARTIFACTS / "model_comparison.csv"
FEATURE_SET_COMPARISON_PATH = ARTIFACTS / "feature_set_comparison.csv"
CALIBRATION_COMPARISON_PATH = ARTIFACTS / "calibration_comparison.csv"
IMPORTANCE_PATH = ARTIFACTS / "feature_importance.csv"
PREDICTIONS_PATH = ARTIFACTS / "holdout_predictions.csv"
CONFUSION_MATRIX_PATH = ARTIFACTS / "confusion_matrix.csv"
PER_CLASS_METRICS_PATH = ARTIFACTS / "per_class_metrics.csv"
CLASS_DISTRIBUTION_PATH = ARTIFACTS / "class_distribution.csv"
TRAINING_METADATA_PATH = ARTIFACTS / "training_metadata.json"
SELECTED_FEATURES_PATH = ARTIFACTS / "selected_features.json"

TARGET = "Target"
RANDOM_STATE = 42
MODEL_VERSION = "3.0.0"
DATASET_ID = 697
DATASET_DOI = "10.24432/C5MC89"
DATASET_LICENSE = "CC BY 4.0"
DATASET_TITLE = "Predict Students' Dropout and Academic Success"

# These seven fields are the complete contract accepted by the deployed artifact.
INPUT_FEATURES = [
    "units_registered",
    "units_passed",
    "average_mark_pct",
    "assessments_completed",
    "tuition_up_to_date",
    "outstanding_fee_balance",
    "scholarship_support",
]

ENGINEERED_FEATURES = [
    "units_registered",
    "units_passed",
    "units_not_passed",
    "pass_rate",
    "average_mark_pct",
    "assessments_completed",
    "assessments_per_unit",
    "tuition_up_to_date",
    "outstanding_fee_balance",
    "scholarship_support",
]

FRIENDLY_FEATURE_NAMES = {
    "units_registered": "Units registered",
    "units_passed": "Units passed",
    "units_not_passed": "Units not passed",
    "pass_rate": "Pass rate",
    "average_mark_pct": "Semester 1 average mark",
    "assessments_completed": "Assessments completed",
    "assessments_per_unit": "Assessment activity",
    "tuition_up_to_date": "Tuition fees up to date",
    "outstanding_fee_balance": "Outstanding fee balance",
    "scholarship_support": "Scholarship / sponsorship",
}

CLASS_ORDER = ["Dropout", "Enrolled", "Graduate"]
SEMESTER_2_MARKER = "2nd sem"
IDENTITY_FIELDS = {"student_name", "registration_number", "school", "programme_name"}
SENSITIVE_SOURCE_FIELDS = {
    "Gender",
    "Nationality",
    "Marital Status",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Age at enrollment",
}
