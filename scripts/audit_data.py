from __future__ import annotations

import json

from src.config import ARTIFACTS, CLASS_ORDER, TARGET
from src.data import load_dataset

BINARY_COLUMNS = [
    "Tuition fees up to date",
    "Debtor",
    "Scholarship holder",
]
COUNT_COLUMNS = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (without evaluations)",
]
GRADE_COLUMN = "Curricular units 1st sem (grade)"


def audit_dataset() -> dict[str, object]:
    frame = load_dataset()
    numeric = frame.drop(columns=[TARGET]).apply(lambda series: series.astype(float))
    invalid_binary = sum(int((~frame[column].isin([0, 1])).sum()) for column in BINARY_COLUMNS)
    negative_counts = int((frame[COUNT_COLUMNS] < 0).sum().sum())
    invalid_grades = int((~frame[GRADE_COLUMN].between(0, 20)).sum())
    summary_columns = COUNT_COLUMNS + [GRADE_COLUMN]
    summary = frame[summary_columns].describe().loc[["min", "mean", "std", "max"]]
    return {
        "rows": len(frame),
        "all_predictors_numeric": not numeric.isna().any().any(),
        "target_values": sorted(frame[TARGET].unique().tolist()),
        "target_values_valid": sorted(frame[TARGET].unique().tolist()) == sorted(CLASS_ORDER),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "invalid_binary_cells": invalid_binary,
        "negative_semester1_count_cells": negative_counts,
        "semester1_grades_outside_0_20": invalid_grades,
        "key_numeric_distributions": summary.to_dict(),
    }


def main() -> None:
    audit = audit_dataset()
    path = ARTIFACTS / "data_quality_audit.json"
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
