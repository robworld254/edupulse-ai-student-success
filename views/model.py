from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import FRIENDLY_FEATURE_NAMES, IMPORTANCE_PATH, RESULTS_PATH
from src.inference import load_results
from src.ui import (
    GOLD,
    GREEN,
    MAROON,
    context_strip,
    footer_note,
    kpi_card,
    loading_state,
    page_header,
    plotly_config,
    section_title,
)


@st.cache_data(show_spinner=False)
def model_evidence() -> tuple[dict, pd.DataFrame]:
    return load_results(), pd.read_csv(IMPORTANCE_PATH)


page_header(
    "Model Performance",
    "Transparent selection evidence, final holdout results and operational safeguards.",
    "Model assurance",
)
context_strip("Training-fold selection · Untouched final holdout · Structural leakage controls", "Validated evidence")

if not RESULTS_PATH.exists():
    st.error("Model results are unavailable. Run START_EDUPULSE.bat to complete setup.")
    st.stop()

loader = loading_state("Loading model assurance evidence", "Verifying evaluation artifacts")
try:
    results, importance_data = model_evidence()
finally:
    loader.empty()

holdout = results["holdout"]
selection = results["model_selection"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Selected model", selection["selected_model"], "Highest tuned CV Macro F1", MAROON)
with c2:
    kpi_card("CV Macro F1", f"{selection['selected_cv_macro_f1']:.3f}", "Training folds", GOLD)
with c3:
    kpi_card("Holdout Macro F1", f"{holdout['macro_f1']:.3f}", "Primary metric", GREEN)
with c4:
    kpi_card("Balanced accuracy", f"{holdout['balanced_accuracy']:.3f}", "Final holdout", GREEN)

st.write("")
left, right = st.columns([1.08, 0.92], gap="large")
with left, st.container(border=True):
    section_title("Five-model benchmark", "Comparable cross-validation results on training folds.")
    benchmark = pd.DataFrame(selection["benchmarks"])
    chart = benchmark.melt(
        id_vars="model",
        value_vars=["macro_f1", "balanced_accuracy", "accuracy"],
        var_name="metric",
        value_name="score",
    )
    chart["metric"] = chart["metric"].replace(
        {"macro_f1": "Macro F1", "balanced_accuracy": "Balanced accuracy", "accuracy": "Accuracy"}
    )
    figure = px.bar(
        chart,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        color_discrete_sequence=[MAROON, GOLD, GREEN],
    )
    figure.update_traces(marker_cornerradius=3)
    figure.update_layout(
        height=375,
        xaxis_title="",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1],
        legend_title_text="",
        margin={"l": 8, "r": 8, "t": 6, "b": 6},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch", config=plotly_config())

with right, st.container(border=True):
    section_title("Final confusion matrix", "Actual outcomes against model classifications.", "20% holdout")
    matrix = holdout["confusion_matrix"]
    labels = holdout["classes"]
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            text=matrix,
            texttemplate="%{text}",
            colorscale=[[0, "#FBF6F7"], [1, MAROON]],
            showscale=False,
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>Records %{z}<extra></extra>",
        )
    )
    figure.update_layout(
        height=375,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        margin={"l": 8, "r": 8, "t": 6, "b": 6},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch", config=plotly_config())

st.write("")
selection_tab, class_tab, reliance_tab, governance_tab = st.tabs(
    ["Feature selection", "Class performance", "Model reliance", "Governance"]
)

with selection_tab:
    section_title("Feature-set experiment", "The compact seven-field set was selected through training-fold evidence.")
    feature_sets = pd.DataFrame(results["feature_selection"]["comparison"])
    st.dataframe(
        feature_sets[
            ["feature_set", "input_count", "macro_f1", "balanced_accuracy", "accuracy", "dropout_recall"]
        ].style.format(
            {
                "macro_f1": "{:.3f}",
                "balanced_accuracy": "{:.3f}",
                "accuracy": "{:.3f}",
                "dropout_recall": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(results["feature_selection"]["selection_reason"])

with class_tab:
    section_title("Per-class holdout performance", "Precision, recall and F1 for each final outcome.")
    report = pd.DataFrame(holdout["classification_report"]).T
    class_rows = [label for label in labels if label in report.index]
    table = report.loc[class_rows, ["precision", "recall", "f1-score", "support"]].copy()
    table.columns = ["Precision", "Recall", "F1", "Support"]
    st.dataframe(
        table.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "Support": "{:.0f}"}),
        width="stretch",
    )

with reliance_tab:
    section_title("Global permutation importance", "Mean holdout Macro F1 decrease when each feature is permuted.")
    importance = importance_data.head(7).copy()
    importance["Feature"] = importance["feature"].map(FRIENDLY_FEATURE_NAMES).fillna(importance["feature"])
    importance = importance.sort_values("importance")
    figure = px.bar(
        importance,
        x="importance",
        y="Feature",
        orientation="h",
        error_x="std",
        color_discrete_sequence=[MAROON],
    )
    figure.update_traces(marker_cornerradius=4)
    figure.update_layout(
        height=360,
        xaxis_title="Mean decrease in holdout Macro F1",
        yaxis_title="",
        margin={"l": 8, "r": 8, "t": 6, "b": 6},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch", config=plotly_config())
    st.caption("Model reliance does not establish causation or justify adverse action.")

with governance_tab:
    section_title("Calibration and controls", "How model selection and operational risk are constrained.")
    governance_left, governance_right = st.columns(2, gap="large")
    with governance_left:
        st.markdown(
            f"""
            **Selection metric**
            Macro F1 gives each outcome class equal influence.

            **Data split**
            {results["split"]["train_rows"]:,} training records and {results["split"]["test_rows"]:,} untouched test records, seed 42.

            **Calibration**
            {results["calibration"]["selected_method"].title()} calibration selected through nested training cross-validation. Holdout log loss: {holdout["log_loss"]:.3f}.
            """
        )
    with governance_right:
        st.markdown(
            f"""
            **Support bands**
            Monitor at {results["risk_bands"]["monitor_threshold"]:.1%}; Higher Priority at {results["risk_bands"]["higher_threshold"]:.1%}.

            **Leakage boundary**
            The artifact accepts seven Semester 1 fields. Semester 2, identity, school, programme and sensitive demographics are excluded.

            **Decision boundary**
            Outputs prioritize supportive human review; they are not automated decisions.
            """
        )
    with st.expander("Technical parameters and calibration comparison", expanded=False):
        st.json(selection["best_params"], expanded=False)
        st.dataframe(pd.DataFrame(results["calibration"]["comparison"]), width="stretch", hide_index=True)

footer_note()
