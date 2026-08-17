from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

from .config import CLASS_ORDER
from .features import Semester1FeatureTransformer


@dataclass
class ModelComparison:
    table: pd.DataFrame
    pipelines: dict[str, Pipeline]


def make_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def candidate_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=4000, class_weight="balanced", random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Support Vector Machine": SVC(
            C=2.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            random_state=random_state,
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def make_pipeline(estimator: Any, *, engineered: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if engineered:
        steps.append(("feature_engineering", Semester1FeatureTransformer()))
    steps.extend(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    return Pipeline(steps)


def _dropout_recall(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
    return float(recall_score(y, estimator.predict(X), labels=["Dropout"], average="macro", zero_division=0))


def scoring() -> dict[str, Any]:
    return {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "dropout_recall": _dropout_recall,
    }


def compare_models(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> ModelComparison:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows: list[dict[str, float | str]] = []
    pipelines: dict[str, Pipeline] = {}
    for name, estimator in candidate_models(random_state).items():
        pipeline = make_pipeline(estimator)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring(), n_jobs=-1)
        row: dict[str, float | str] = {"model": name}
        for metric in scoring():
            row[metric] = float(np.mean(scores[f"test_{metric}"]))
            row[f"{metric}_std"] = float(np.std(scores[f"test_{metric}"]))
        rows.append(row)
        pipelines[name] = pipeline
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return ModelComparison(table=table, pipelines=pipelines)


def compare_feature_sets(
    feature_sets: dict[str, tuple[pd.DataFrame, bool]], y: pd.Series, random_state: int = 42
) -> pd.DataFrame:
    """Fairly compare feature sets using one fixed, interpretable classifier."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows: list[dict[str, object]] = []
    for name, (X, engineered) in feature_sets.items():
        estimator = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=random_state)
        scores = cross_validate(
            make_pipeline(estimator, engineered=engineered),
            X,
            y,
            cv=cv,
            scoring=scoring(),
            n_jobs=-1,
        )
        rows.append(
            {
                "feature_set": name,
                "input_count": int(X.shape[1]),
                "model_feature_count": 10 if engineered else int(X.shape[1]),
                **{metric: float(np.mean(scores[f"test_{metric}"])) for metric in scoring()},
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def tune_model(
    name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> tuple[Pipeline, dict[str, Any], float]:
    spaces: dict[str, dict[str, list[Any]]] = {
        "Logistic Regression": {"model__C": [0.03, 0.07, 0.1, 0.3, 0.7, 1.0, 2.0, 5.0, 10.0]},
        "Random Forest": {
            "model__n_estimators": [350, 500, 700],
            "model__max_depth": [None, 8, 12, 18],
            "model__min_samples_leaf": [1, 2, 3, 5, 8],
            "model__max_features": ["sqrt", 0.6, 0.8, 1.0],
        },
        "Extra Trees": {
            "model__n_estimators": [350, 500, 700],
            "model__max_depth": [None, 8, 12, 18],
            "model__min_samples_leaf": [1, 2, 3, 5, 8],
            "model__max_features": ["sqrt", 0.6, 0.8, 1.0],
        },
        "Support Vector Machine": {
            "model__C": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0],
            "model__gamma": ["scale", 0.01, 0.03, 0.06, 0.1, 0.2],
        },
        "Histogram Gradient Boosting": {
            "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
            "model__max_iter": [180, 250, 350],
            "model__max_leaf_nodes": [15, 23, 31, 47],
            "model__l2_regularization": [0.0, 0.5, 1.0, 2.0],
        },
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    combinations = int(np.prod([len(values) for values in spaces[name].values()]))
    search = RandomizedSearchCV(
        estimator=clone(pipeline),
        param_distributions=spaces[name],
        n_iter=min(16, combinations),
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def multiclass_brier(y_true: pd.Series, probabilities: np.ndarray, classes: list[str]) -> float:
    binary = label_binarize(y_true, classes=classes)
    return float(np.mean(np.sum((probabilities - binary) ** 2, axis=1)))


def expected_calibration_error(
    y_true: pd.Series, probabilities: np.ndarray, classes: list[str], bins: int = 10
) -> float:
    predicted_index = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    truth = np.asarray([classes.index(str(value)) for value in y_true])
    correct = predicted_index == truth
    error = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lower, upper in pairwise(edges):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            error += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return error


def compare_calibration(
    tuned_pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> tuple[pd.DataFrame, str, np.ndarray, list[str]]:
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    classes = sorted(y.unique().tolist())
    rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    for method in ["sigmoid", "isotonic"]:
        calibrated = CalibratedClassifierCV(clone(tuned_pipeline), method=method, cv=3)
        probabilities = cross_val_predict(calibrated, X, y, cv=outer_cv, method="predict_proba", n_jobs=-1)
        predictions[method] = probabilities
        rows.append(
            {
                "method": method,
                "cv_log_loss": float(log_loss(y, probabilities, labels=classes)),
                "cv_multiclass_brier": multiclass_brier(y, probabilities, classes),
                "cv_expected_calibration_error": expected_calibration_error(y, probabilities, classes),
            }
        )
    table = pd.DataFrame(rows).sort_values(["cv_log_loss", "cv_multiclass_brier"]).reset_index(drop=True)
    best_method = str(table.iloc[0]["method"])
    return table, best_method, predictions[best_method], classes


def fit_calibrated(
    best_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, method: str
) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(best_pipeline, method=method, cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def derive_risk_thresholds(y_true: pd.Series, dropout_probabilities: np.ndarray) -> dict[str, float | str]:
    binary = (y_true.astype(str) == "Dropout").astype(int)
    precision, recall, thresholds = precision_recall_curve(binary, dropout_probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    monitor = float(thresholds[int(np.nanargmax(f1))])
    eligible = np.flatnonzero((precision[:-1] >= 0.75) & (thresholds >= monitor))
    if len(eligible):
        higher = float(thresholds[int(eligible[0])])
        high_method = "smallest OOF threshold at or above monitor with at least 0.75 precision"
    else:
        higher = float(np.quantile(dropout_probabilities, 0.85))
        high_method = "85th percentile fallback because 0.75 precision was not achieved"
    higher = max(higher, monitor + 0.01)
    return {
        "monitor_threshold": round(monitor, 6),
        "higher_threshold": round(min(higher, 0.99), 6),
        "method": (
            "Monitor maximizes binary Dropout F1 on out-of-fold training probabilities; Higher is the " + high_method
        ),
    }


def evaluate(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = list(model.classes_)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
        "log_loss": float(log_loss(y_test, probabilities, labels=classes)),
        "multiclass_brier": multiclass_brier(y_test, probabilities, classes),
        "roc_auc_ovr_weighted": float(
            roc_auc_score(y_test, probabilities, labels=classes, multi_class="ovr", average="weighted")
        ),
        "classes": classes,
        "classification_report": classification_report(
            y_test, predictions, labels=CLASS_ORDER, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=CLASS_ORDER).tolist(),
    }
