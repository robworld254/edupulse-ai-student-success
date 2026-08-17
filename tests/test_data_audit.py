from scripts.audit_data import audit_dataset


def test_data_ranges_types_and_categories_are_valid():
    audit = audit_dataset()
    assert audit["all_predictors_numeric"] is True
    assert audit["target_values_valid"] is True
    assert audit["invalid_binary_cells"] == 0
    assert audit["negative_semester1_count_cells"] == 0
    assert audit["semester1_grades_outside_0_20"] == 0
