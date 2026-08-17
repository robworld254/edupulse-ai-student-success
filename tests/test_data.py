import pandas as pd

from src.config import CLASS_ORDER, DATA_PATH
from src.data import file_sha256, load_dataset, normalize_columns, validate_dataset


def test_normalize_columns_and_alias():
    frame = pd.DataFrame({" Nacionality ": [1], " Target ": ["Graduate"]})
    normalized = normalize_columns(frame)
    assert "Nationality" in normalized.columns
    assert "Target" in normalized.columns


def test_real_dataset_loads_and_has_expected_classes():
    frame = load_dataset()
    assert DATA_PATH.exists()
    assert len(frame) == 4424
    assert sorted(frame["Target"].unique()) == sorted(CLASS_ORDER)


def test_dataset_quality_facts_are_reproducible():
    summary = validate_dataset(load_dataset())
    assert summary["original_feature_count"] == 36
    assert summary["duplicates"] == 0
    assert summary["missing_cells"] == 0
    assert summary["target_classes_valid"] is True
    assert len(file_sha256()) == 64
