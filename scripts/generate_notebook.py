from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "EduPulse_AI_Final_End_to_End.ipynb"
PROJECT_METADATA = json.loads((ROOT / "project_metadata.json").read_text(encoding="utf-8"))


def markdown(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


def main() -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "authors": [
            {
                "name": PROJECT_METADATA["student_name"],
                "registration_number": PROJECT_METADATA["registration_number"],
            }
        ],
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        markdown(
            f"""
# EduPulse AI — Final End-to-End Machine Learning Notebook

**Student Success Early-Warning and Intervention Platform**<br>
AI Mini Group Project<br>
Academic prototype localized for a Kabarak University context.<br>
**Student:** {PROJECT_METADATA["student_name"]}<br>
**Registration number:** {PROJECT_METADATA["registration_number"]}<br>
**Unit:** {PROJECT_METADATA["unit_code"]} · **Lecturer:** {PROJECT_METADATA["lecturer_name"]}

This notebook is an executable academic narrative. It uses the real UCI Machine Learning Repository Dataset 697 and the canonical outputs from the reproducible `python -m scripts.train` workflow. It does not claim Kabarak University data, deployment or validation.
"""
        ),
        markdown(
            """
## 1. Problem Definition & Objective

Universities may recognize difficulty only after a student's position has deteriorated. The objective is supervised three-class classification of **Dropout**, **Enrolled** and **Graduate** outcomes from information genuinely available at the **end of Semester 1**. Outputs support human review; they must never automate punitive decisions.
"""
        ),
        code(
            """
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RESULTS = json.loads((ROOT / "artifacts" / "results.json").read_text(encoding="utf-8"))
RESULTS["project"]
"""
        ),
        markdown(
            """
## 2. Data Acquisition

The source is UCI Dataset 697, *Predict Students' Dropout and Academic Success*, DOI `10.24432/C5MC89`, licensed CC BY 4.0. It describes Portuguese higher education and is not Kabarak University administrative data. The raw local file is treated as immutable and identified by SHA-256 in the training metadata.
"""
        ),
        code(
            """
from src.data import load_dataset, validate_dataset

raw = load_dataset()
audit = validate_dataset(raw)
audit, raw.shape, raw["Target"].value_counts()
"""
        ),
        markdown(
            """
## 3. Data Cleaning & Preprocessing

The audit verifies headers, data types, target values, missingness and duplicates. The dataset is already clean: zero missing cells and zero duplicate rows. Median imputation remains inside the sklearn pipeline as a defensive deployment safeguard, followed by standardization. The raw file is not rewritten during training.
"""
        ),
        code(
            """
quality = pd.Series({
    "Rows": len(raw),
    "Original predictors": raw.shape[1] - 1,
    "Missing cells": int(raw.isna().sum().sum()),
    "Duplicate rows": int(raw.duplicated().sum()),
    "Target classes": raw["Target"].nunique(),
})
quality.to_frame("Value")
"""
        ),
        markdown(
            """
## 4. Exploratory Data Analysis

The Graduate class is largest and Enrolled smallest, so accuracy alone can hide weak minority-class performance. Semester 1 academic indicators shift across outcomes, but overlap makes multivariable modelling necessary. All findings are associations, not causes.
"""
        ),
        code(
            """
from src.features import engineer_semester1_features

research = engineer_semester1_features(raw, include_target=True)
order = ["Dropout", "Enrolled", "Graduate"]
colors = ["#8F1D2C", "#D6A84D", "#4D9D45"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
counts = research["Target"].value_counts().reindex(order)
axes[0].bar(order, counts, color=colors)
axes[0].set_title("Target distribution")
axes[0].set_ylabel("Records")
axes[1].boxplot([research.loc[research.Target == label, "average_mark_pct"] for label in order], tick_labels=order, showfliers=False)
axes[1].set_title("Semester 1 average")
axes[1].set_ylabel("Converted mark (%)")
axes[2].boxplot([research.loc[research.Target == label, "pass_rate"] * 100 for label in order], tick_labels=order, showfliers=False)
axes[2].set_title("Semester 1 pass rate")
axes[2].set_ylabel("Units passed (%)")
plt.tight_layout()
plt.show()
"""
        ),
        code(
            """
fee_outcomes = pd.crosstab(
    research["tuition_up_to_date"].map({1: "Up to date", 0: "Not up to date"}),
    research["Target"],
    normalize="index",
).reindex(columns=order)
fee_outcomes.plot(kind="bar", stacked=True, color=colors, figsize=(8, 4))
plt.ylabel("Outcome share")
plt.xlabel("Tuition status")
plt.title("Tuition status and final outcome")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
fee_outcomes
"""
        ),
        markdown(
            """
## 5. Feature Engineering

The deployed UI accepts seven concepts. The saved pipeline derives `units_not_passed`, `pass_rate` and `assessments_per_unit` with divide-by-zero protection. The UCI 0–20 grade is multiplied by five for an exact 0–100 display conversion. All Semester 2, identity and sensitive/proxy fields are excluded.
"""
        ),
        code(
            """
from src.config import ENGINEERED_FEATURES, INPUT_FEATURES
from src.features import build_compact_inputs, engineer_compact_inputs

compact = build_compact_inputs(raw, include_target=False)
engineered = engineer_compact_inputs(compact)
print("Required UI inputs:", INPUT_FEATURES)
print("Engineered model features:", ENGINEERED_FEATURES)
engineered.describe().T
"""
        ),
        code(
            """
feature_comparison = pd.read_csv(ROOT / "artifacts" / "feature_set_comparison.csv")
feature_comparison
"""
        ),
        markdown(
            """
## 6. Model Building

The methodology creates one stratified 80/20 split with seed 42. Five-fold stratified cross-validation on training data compares Logistic Regression, Random Forest, Extra Trees, Support Vector Machine and Histogram Gradient Boosting. The two strongest candidates are tuned with bounded randomized searches. Run `python -m scripts.train` from the repository root to reproduce all artifacts.
"""
        ),
        code(
            """
model_comparison = pd.read_csv(ROOT / "artifacts" / "model_comparison.csv")
model_comparison[["model", "macro_f1", "balanced_accuracy", "accuracy", "weighted_f1", "dropout_recall"]]
"""
        ),
        code(
            """
pd.DataFrame(RESULTS["model_selection"]["tuning"])
"""
        ),
        markdown(
            """
## 7. Model Evaluation

Macro F1 is primary because it weights each imbalanced class equally. The final test partition is evaluated once after selection. Reported evidence includes Accuracy, Balanced Accuracy, Macro and Weighted F1, per-class metrics, confusion matrix, log loss and multiclass ROC-AUC. Probability calibration is selected through nested training cross-validation.
"""
        ),
        code(
            """
metrics = RESULTS["holdout"]
pd.Series({name: metrics[name] for name in [
    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
    "log_loss", "multiclass_brier", "roc_auc_ovr_weighted",
]}, name="Final holdout")
"""
        ),
        code(
            """
report = pd.DataFrame(metrics["classification_report"]).T
report.loc[["Dropout", "Enrolled", "Graduate"], ["precision", "recall", "f1-score", "support"]]
"""
        ),
        code(
            """
matrix = pd.DataFrame(metrics["confusion_matrix"], index=metrics["classes"], columns=metrics["classes"])
fig, ax = plt.subplots(figsize=(5, 4))
image = ax.imshow(matrix, cmap="Reds")
for row in range(3):
    for column in range(3):
        ax.text(column, row, matrix.iloc[row, column], ha="center", va="center")
ax.set_xticks(range(3), matrix.columns)
ax.set_yticks(range(3), matrix.index)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Final holdout confusion matrix")
plt.colorbar(image, ax=ax)
plt.tight_layout()
plt.show()
matrix
"""
        ),
        markdown(
            """
## 8. Results Interpretation & Insights

Global permutation importance measures the decrease in holdout Macro F1 when an input is shuffled. It describes model reliance, not causation. Individual portal explanations compare the estimate with research-cohort medians. The Enrolled class remains the most difficult and is reported as a limitation.
"""
        ),
        code(
            """
importance = pd.read_csv(ROOT / "artifacts" / "feature_importance.csv").sort_values("importance")
importance.plot.barh(x="feature", y="importance", xerr="std", legend=False, color="#8F1D2C", figsize=(8, 4.5))
plt.xlabel("Decrease in Macro F1 when permuted")
plt.ylabel("")
plt.title("Global permutation importance")
plt.tight_layout()
plt.show()
importance.sort_values("importance", ascending=False)
"""
        ),
        markdown(
            """
## 9. Deployment

The Streamlit portal has four concise pages: Dashboard, Student Assessment, Cohort Analytics and Model Performance. `START_EDUPULSE.bat` performs setup and trains only when required; `RUN_EDUPULSE.bat` fast-starts existing artifacts. The complete sklearn pipeline includes feature engineering, imputation, scaling, classification and calibration.
"""
        ),
        code(
            """
from src.features import build_assessment_row
from src.inference import load_model, support_priority

model = load_model()
demo = build_assessment_row(
    units_registered=6,
    units_passed=2,
    average_mark_pct=41,
    assessments_completed=3,
    tuition_up_to_date=0,
    outstanding_fee_balance=1,
    scholarship_support=0,
)
probabilities = dict(zip(model.classes_, model.predict_proba(demo)[0]))
{
    "probabilities": probabilities,
    "support_priority": support_priority(probabilities["Dropout"], RESULTS["risk_bands"]),
}
"""
        ),
        markdown(
            """
## 10. Conclusion

EduPulse AI is a reproducible, leakage-safe and transparent academic prototype. The compact engineered feature set retained the strongest validation trade-off, the final model provides calibrated estimated probabilities, and every intervention remains supportive and human-reviewed. The Portuguese research source is not evidence of Kabarak validity; governance, privacy review, local validation, fairness testing and prospective monitoring are required before operational use.
"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, OUTPUT)
    print(f"Generated and executed {OUTPUT}")


if __name__ == "__main__":
    main()
