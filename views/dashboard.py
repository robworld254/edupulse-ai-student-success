from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import CLASS_DISTRIBUTION_PATH, MODEL_PATH, RESULTS_PATH
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
def dashboard_data() -> tuple[dict, pd.DataFrame]:
    return load_results(), pd.read_csv(CLASS_DISTRIBUTION_PATH)


page_header(
    "Student Success Dashboard",
    "A concise Semester 1 view of model readiness, evidence quality and support operations.",
    "Executive overview",
)
context_strip(
    "Kabarak-context academic prototype powered by public UCI research records",
    "Verified local artifacts",
)

if not RESULTS_PATH.exists() or not MODEL_PATH.exists():
    st.error("Training artifacts are unavailable. Run START_EDUPULSE.bat to complete setup.")
    st.stop()

loader = loading_state("Loading intelligence overview", "Reading verified model artifacts")
try:
    results, distribution = dashboard_data()
finally:
    loader.empty()

holdout = results["holdout"]
dataset = results["dataset"]
dropout = holdout["classification_report"]["Dropout"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Model status", "Ready", f"Version {results['project']['model_version']}", GREEN)
with c2:
    kpi_card("Selected model", results["model_selection"]["selected_model"], "Cross-validated selection", MAROON)
with c3:
    kpi_card("Macro F1", f"{holdout['macro_f1']:.3f}", "Untouched holdout", GOLD)
with c4:
    kpi_card("Dropout recall", f"{dropout['recall']:.1%}", "Support-sensitive class", GREEN)

st.write("")
left, right = st.columns([0.9, 1.1], gap="large")
with left, st.container(border=True):
    section_title(
        "Research cohort outcomes",
        "Final outcomes in the model-development dataset.",
        f"n = {dataset['rows']:,}",
    )
    colors = {"Dropout": MAROON, "Enrolled": GOLD, "Graduate": GREEN}
    figure = px.pie(
        distribution,
        names="class",
        values="count",
        hole=0.67,
        color="class",
        color_discrete_map=colors,
    )
    figure.update_traces(textposition="outside", textinfo="percent+label", marker_line_width=0)
    figure.update_layout(
        height=330,
        margin={"l": 12, "r": 12, "t": 6, "b": 6},
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch", config=plotly_config())
    st.caption("Research distribution; not current Kabarak University statistics.")

with right, st.container(border=True):
    section_title(
        "Validation snapshot",
        "Performance on the final partition excluded from model selection.",
        "20% holdout",
    )
    metrics = pd.DataFrame(
        {
            "Metric": ["Macro F1", "Balanced accuracy", "Accuracy", "Weighted F1"],
            "Score": [
                holdout["macro_f1"],
                holdout["balanced_accuracy"],
                holdout["accuracy"],
                holdout["weighted_f1"],
            ],
        }
    )
    figure = px.bar(
        metrics,
        x="Score",
        y="Metric",
        orientation="h",
        text=metrics["Score"].map(lambda value: f"{value:.3f}"),
        color_discrete_sequence=[MAROON],
        range_x=[0, 1],
    )
    figure.update_traces(marker_cornerradius=4, textposition="outside")
    figure.update_layout(
        height=330,
        xaxis_tickformat=".0%",
        xaxis_title="Holdout score",
        yaxis_title="",
        margin={"l": 8, "r": 28, "t": 6, "b": 6},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch", config=plotly_config())

section_title("Operational overview", "The system boundary and evidence trail at a glance.")
a, b, c = st.columns(3)
with a:
    st.markdown(
        "<div class='panel'><div class='panel-title'>Prediction point</div><div class='panel-value'>End of Semester 1</div><div class='compact-note'>Seven academic and financial-support inputs. Semester 2 data are excluded.</div></div>",
        unsafe_allow_html=True,
    )
with b:
    st.markdown(
        f"<div class='panel'><div class='panel-title'>Research data</div><div class='panel-value'>{dataset['rows']:,} records</div><div class='compact-note'>UCI Dataset 697 with no missing cells or duplicate records.</div></div>",
        unsafe_allow_html=True,
    )
with c:
    trained = results["reproducibility"]["training_completed_utc"][:10]
    st.markdown(
        f"<div class='panel'><div class='panel-title'>Last trained</div><div class='panel-value'>{trained}</div><div class='compact-note'>Seed 42, stratified validation, calibrated probabilities and required human review.</div></div>",
        unsafe_allow_html=True,
    )

st.write("")
st.page_link("views/assessment.py", label="Open student assessment")
footer_note()
