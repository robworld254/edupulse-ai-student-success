from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACTS,
    CALIBRATION_COMPARISON_PATH,
    CLASS_DISTRIBUTION_PATH,
    CLASS_ORDER,
    CONFUSION_MATRIX_PATH,
    DATASET_DOI,
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_TITLE,
    ENGINEERED_FEATURES,
    FEATURE_SET_COMPARISON_PATH,
    IMPORTANCE_PATH,
    INPUT_FEATURES,
    METRICS_CSV_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    PER_CLASS_METRICS_PATH,
    PREDICTIONS_PATH,
    RANDOM_STATE,
    RESULTS_PATH,
    SELECTED_FEATURES_PATH,
    TARGET,
    TRAINING_METADATA_PATH,
)
from src.data import file_sha256, load_dataset, validate_dataset
from src.explainability import global_permutation_importance
from src.features import WIDE_SAFE_FEATURES, build_compact_inputs
from src.modeling import (
    compare_calibration,
    compare_feature_sets,
    compare_models,
    derive_risk_thresholds,
    evaluate,
    fit_calibrated,
    tune_model,
)


def _jsonable(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "streamlit", "plotly"]
    installed: dict[str, str] = {}
    for name in names:
        try:
            installed[name] = version(name)
        except PackageNotFoundError:
            installed[name] = "not installed"
    return installed


def _write_plots(class_distribution: pd.DataFrame, confusion: pd.DataFrame) -> None:
    colors = ["#8F1D2C", "#D6A84D", "#4D9D45"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(class_distribution["class"], class_distribution["count"], color=colors)
    ax.set(title="UCI research cohort outcome distribution", ylabel="Records")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    image = ax.imshow(confusion.to_numpy(), cmap="Reds")
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            ax.text(column, row, int(confusion.iloc[row, column]), ha="center", va="center")
    ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Final holdout confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    started = datetime.now(UTC)
    raw = load_dataset()
    data_summary = validate_dataset(raw)
    if not data_summary["target_classes_valid"]:
        raise ValueError(f"Unexpected target classes: {data_summary['target_classes']}")
    print("Dataset audit:", data_summary)

    all_indices = raw.index.to_numpy()
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.20,
        stratify=raw[TARGET].astype(str),
        random_state=RANDOM_STATE,
    )
    raw_train = raw.loc[train_indices]
    raw_test = raw.loc[test_indices]
    compact_train = build_compact_inputs(raw_train, include_target=True)
    compact_test = build_compact_inputs(raw_test, include_target=True)
    X_train = compact_train[INPUT_FEATURES]
    y_train = compact_train[TARGET]
    X_test = compact_test[INPUT_FEATURES]
    y_test = compact_test[TARGET]

    feature_sets = {
        "A — wider leakage-safe Semester 1": (raw_train[WIDE_SAFE_FEATURES].copy(), False),
        "B — compact raw inputs": (X_train.copy(), False),
        "C — engineered compact inputs": (X_train.copy(), True),
    }
    feature_comparison = compare_feature_sets(feature_sets, y_train, RANDOM_STATE)
    print("\nFeature-set comparison (training CV only):\n", feature_comparison.to_string(index=False))

    comparison = compare_models(X_train, y_train, RANDOM_STATE)
    print("\nModel comparison (training CV only):\n", comparison.table.to_string(index=False))

    tuned_rows: list[dict[str, object]] = []
    tuned_models: dict[str, object] = {}
    tuned_params: dict[str, dict[str, object]] = {}
    for name in comparison.table.head(2)["model"].astype(str):
        tuned, params, score = tune_model(name, comparison.pipelines[name], X_train, y_train, RANDOM_STATE)
        tuned_models[name] = tuned
        tuned_params[name] = params
        baseline_score = float(comparison.table.loc[comparison.table["model"] == name, "macro_f1"].iloc[0])
        tuned_rows.append(
            {
                "model": name,
                "untuned_cv_macro_f1": baseline_score,
                "tuned_cv_macro_f1": score,
                "improvement": score - baseline_score,
                "best_params": json.dumps(_jsonable(params), sort_keys=True),
            }
        )
        print(f"Tuned {name}: CV Macro F1={score:.4f}; parameters={params}")

    tuning_table = pd.DataFrame(tuned_rows).sort_values("tuned_cv_macro_f1", ascending=False).reset_index(drop=True)
    selected_name = str(tuning_table.iloc[0]["model"])
    selected_pipeline = tuned_models[selected_name]
    selected_params = tuned_params[selected_name]

    calibration, calibration_method, oof_probabilities, oof_classes = compare_calibration(
        selected_pipeline, X_train, y_train, RANDOM_STATE
    )
    print("\nCalibration comparison (nested training CV):\n", calibration.to_string(index=False))
    dropout_index = oof_classes.index("Dropout")
    risk_bands = derive_risk_thresholds(y_train, oof_probabilities[:, dropout_index])

    final_model = fit_calibrated(selected_pipeline, X_train, y_train, method=calibration_method)
    # This is the single final evaluation of the untouched test partition.
    holdout = evaluate(final_model, X_test, y_test)
    predictions = final_model.predict(X_test)
    probabilities = final_model.predict_proba(X_test)

    baselines = {column: float(pd.to_numeric(X_train[column], errors="coerce").median()) for column in INPUT_FEATURES}
    importance = global_permutation_importance(final_model, X_test, y_test, random_state=RANDOM_STATE)

    class_distribution = raw[TARGET].value_counts().reindex(CLASS_ORDER).rename_axis("class").reset_index(name="count")
    class_distribution["share"] = class_distribution["count"] / len(raw)

    report = pd.DataFrame(holdout["classification_report"]).T
    per_class = report.loc[CLASS_ORDER, ["precision", "recall", "f1-score", "support"]].copy()
    per_class.index.name = "class"
    per_class = per_class.reset_index().rename(columns={"f1-score": "f1"})
    confusion = pd.DataFrame(holdout["confusion_matrix"], index=CLASS_ORDER, columns=CLASS_ORDER)
    confusion.index.name = "actual"

    pred_df = X_test.reset_index(drop=True).copy()
    pred_df["actual"] = y_test.reset_index(drop=True)
    pred_df["predicted"] = predictions
    for index, label in enumerate(final_model.classes_):
        pred_df[f"prob_{str(label).lower()}"] = probabilities[:, index]

    completed = datetime.now(UTC)
    packages = _package_versions()
    dataset_section = {
        **data_summary,
        "id": DATASET_ID,
        "title": DATASET_TITLE,
        "doi": DATASET_DOI,
        "license": DATASET_LICENSE,
        "source": "UCI Machine Learning Repository",
        "geographic_context": "Portuguese higher education",
        "sha256": file_sha256(),
        "class_distribution": {str(row["class"]): int(row["count"]) for _, row in class_distribution.iterrows()},
    }
    results = {
        "project": {
            "name": "EduPulse AI",
            "full_title": "EduPulse AI: Student Success Early-Warning and Intervention Platform",
            "model_version": MODEL_VERSION,
            "context": "Academic prototype localized for a Kabarak University context",
            "prediction_point": "End of Semester 1",
        },
        "reproducibility": {
            "random_seed": RANDOM_STATE,
            "training_started_utc": started.isoformat(),
            "training_completed_utc": completed.isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": packages,
            "command": "python -m scripts.train",
        },
        "dataset": dataset_section,
        "split": {
            "method": "Stratified random 80/20 split before any model selection",
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_fraction": 0.20,
        },
        "feature_selection": {
            "selected_set": "C — engineered compact inputs",
            "selection_reason": (
                "Chosen for the usability, interpretability, data-minimization and predictive-performance trade-off. "
                "The wider set remains an experiment only and is not accepted by the deployed model."
            ),
            "required_ui_inputs": INPUT_FEATURES,
            "engineered_model_features": ENGINEERED_FEATURES,
            "comparison": feature_comparison.to_dict(orient="records"),
            "grade_conversion": "UCI 0–20 Semester 1 grade multiplied by 5 for a 0–100 display scale.",
            "excluded": (
                "All Semester 2, identity, school/programme, demographic, parental, age, nationality, "
                "macroeconomic and Portuguese course-code fields."
            ),
        },
        "model_selection": {
            "primary_metric": "Macro F1",
            "cv": "5-fold stratified cross-validation on training data only",
            "selected_model": selected_name,
            "selected_cv_macro_f1": float(tuning_table.iloc[0]["tuned_cv_macro_f1"]),
            "best_params": _jsonable(selected_params),
            "benchmarks": comparison.table.to_dict(orient="records"),
            "tuning": tuning_table.to_dict(orient="records"),
        },
        "calibration": {
            "selected_method": calibration_method,
            "comparison": calibration.to_dict(orient="records"),
            "wording": "Displayed values are model-estimated probabilities, not guarantees.",
        },
        "risk_bands": risk_bands,
        "holdout": holdout,
        "feature_baselines": baselines,
        "leakage_control": {
            "semester_2_allowed": False,
            "identity_allowed": False,
            "pipeline_input_contract": INPUT_FEATURES,
        },
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    RESULTS_PATH.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    TRAINING_METADATA_PATH.write_text(
        json.dumps(
            _jsonable(
                {
                    "project": results["project"],
                    "reproducibility": results["reproducibility"],
                    "dataset": dataset_section,
                    "split": results["split"],
                    "model_selection": results["model_selection"],
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    SELECTED_FEATURES_PATH.write_text(json.dumps(_jsonable(results["feature_selection"]), indent=2), encoding="utf-8")
    pd.DataFrame([{key: value for key, value in holdout.items() if isinstance(value, (int, float))}]).to_csv(
        METRICS_CSV_PATH, index=False
    )
    comparison.table.to_csv(MODEL_COMPARISON_PATH, index=False)
    feature_comparison.to_csv(FEATURE_SET_COMPARISON_PATH, index=False)
    calibration.to_csv(CALIBRATION_COMPARISON_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    pred_df.to_csv(PREDICTIONS_PATH, index=False)
    confusion.to_csv(CONFUSION_MATRIX_PATH)
    per_class.to_csv(PER_CLASS_METRICS_PATH, index=False)
    class_distribution.to_csv(CLASS_DISTRIBUTION_PATH, index=False)
    _write_plots(class_distribution, confusion)

    print(f"\nSelected model: {selected_name}")
    print(f"Selected calibration: {calibration_method}")
    print(f"Holdout Macro F1: {holdout['macro_f1']:.4f}")
    print(f"Holdout balanced accuracy: {holdout['balanced_accuracy']:.4f}")
    print(f"Holdout accuracy: {holdout['accuracy']:.4f}")
    print(f"Saved complete pipeline: {MODEL_PATH}")
    print(f"Canonical results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
