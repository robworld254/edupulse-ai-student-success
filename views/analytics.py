from __future__ import annotations

from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PREDICTIONS_PATH
from src.data import load_dataset
from src.features import engineer_semester1_features
from src.inference import load_model, load_results, predict_batch
from src.ui import (
    GOLD,
    GREEN,
    MAROON,
    context_strip,
    footer_note,
    loading_state,
    page_header,
    plotly_config,
    section_title,
)


@st.cache_data(show_spinner=False)
def analytics_data() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    research = engineer_semester1_features(load_dataset(), include_target=True)
    return research, load_results(), pd.read_csv(PREDICTIONS_PATH)


@st.cache_resource(show_spinner=False)
def batch_model():
    return load_model()


page_header(
    "Cohort Analytics",
    "Research-cohort patterns and a validated workspace for multi-record assessment.",
    "Evidence explorer",
)
context_strip("Public UCI research cohort · Not current Kabarak University statistics", "Research context")

loader = loading_state("Building cohort analytics", "Preparing research evidence")
try:
    research, results, predictions = analytics_data()
except (FileNotFoundError, ValueError) as error:
    loader.empty()
    st.error(f"Required research artifacts are unavailable: {error}")
    st.stop()
finally:
    loader.empty()

colors = {"Dropout": MAROON, "Enrolled": GOLD, "Graduate": GREEN}
left, right = st.columns(2, gap="large")
with left, st.container(border=True):
    section_title("Semester 1 average", "Distribution by final outcome.", "Converted to 0–100")
    figure = px.box(
        research,
        x="Target",
        y="average_mark_pct",
        color="Target",
        color_discrete_map=colors,
        points=False,
    )
    figure.update_layout(
        height=340,
        showlegend=False,
        xaxis_title="",
        yaxis_title="Semester 1 average (%)",
        margin={"l": 8, "r": 8, "t": 6, "b": 6},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch", config=plotly_config())
    st.caption("Outcome overlap means marks alone are insufficient.")

with right, st.container(border=True):
    section_title("Semester 1 pass rate", "Distribution by final outcome.", "Units passed")
    plot_frame = research.assign(pass_rate_pct=research["pass_rate"] * 100)
    figure = px.box(
        plot_frame,
        x="Target",
        y="pass_rate_pct",
        color="Target",
        color_discrete_map=colors,
        points=False,
    )
    figure.update_layout(
        height=340,
        showlegend=False,
        xaxis_title="",
        yaxis_title="Units passed (%)",
        margin={"l": 8, "r": 8, "t": 6, "b": 6},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch", config=plotly_config())
    st.caption("The Enrolled class remains the most ambiguous.")

left, right = st.columns(2, gap="large")
with left, st.container(border=True):
    section_title("Tuition status and outcome", "Outcome composition by payment status.")
    fee = research.assign(fee_status=research["tuition_up_to_date"].map({1: "Up to date", 0: "Not up to date"}))
    aggregate = fee.groupby(["fee_status", "Target"]).size().reset_index(name="records")
    aggregate["share"] = aggregate.groupby("fee_status")["records"].transform(lambda values: values / values.sum())
    figure = px.bar(
        aggregate,
        x="fee_status",
        y="share",
        color="Target",
        color_discrete_map=colors,
        barmode="stack",
    )
    figure.update_layout(
        height=325,
        xaxis_title="",
        yaxis_title="Outcome share",
        yaxis_tickformat=".0%",
        legend_title_text="",
        margin={"l": 8, "r": 8, "t": 6, "b": 6},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch", config=plotly_config())

with right, st.container(border=True):
    section_title("Estimated dropout risk", "Probability distribution on the final holdout.")
    figure = px.histogram(
        predictions,
        x="prob_dropout",
        color="actual",
        color_discrete_map=colors,
        nbins=24,
        barmode="overlay",
        opacity=0.68,
    )
    figure.update_layout(
        height=325,
        xaxis_title="Estimated dropout probability",
        xaxis_tickformat=".0%",
        yaxis_title="Holdout records",
        legend_title_text="Actual outcome",
        margin={"l": 8, "r": 8, "t": 6, "b": 6},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch", config=plotly_config())

st.write("")
with st.container(border=True):
    section_title(
        "Batch assessment",
        "Upload records using the exact seven-field schema. Optional references are retained, never modelled.",
        "Maximum 5,000 rows",
    )
    template = pd.DataFrame(
        [
            {
                "reference_number": "DEMO-001",
                "units_registered": 6,
                "units_passed": 5,
                "average_mark_pct": 64,
                "assessments_completed": 8,
                "tuition_up_to_date": 1,
                "outstanding_fee_balance": 0,
                "scholarship_support": 0,
            }
        ]
    )
    action, upload_column = st.columns([0.34, 0.66], gap="large")
    with action:
        st.download_button(
            "Download CSV template",
            template.to_csv(index=False),
            file_name="EduPulse_AI_Assessment_Template.csv",
            mime="text/csv",
            width="stretch",
        )
    with upload_column:
        upload = st.file_uploader("Upload completed CSV", type=["csv"], label_visibility="collapsed")

    if upload is not None:
        batch_progress = st.progress(10, text="Reading uploaded records")
        try:
            uploaded = pd.read_csv(upload)
            batch_progress.progress(35, text="Validating assessment schema")
            if len(uploaded) > 5000:
                raise ValueError("The demonstration batch limit is 5,000 rows.")
            batch_progress.progress(65, text="Running calibrated batch assessment")
            batch_result = predict_batch(uploaded, batch_model(), results["risk_bands"])
            batch_progress.progress(100, text="Batch assessment complete")
            st.success(f"Validated and assessed {len(batch_result):,} records.")
            st.dataframe(batch_result.head(100), width="stretch", hide_index=True)
            buffer = StringIO()
            batch_result.to_csv(buffer, index=False)
            st.download_button(
                "Download batch results",
                buffer.getvalue(),
                file_name="edupulse_batch_results.csv",
                mime="text/csv",
            )
        except (ValueError, pd.errors.ParserError) as error:
            batch_progress.empty()
            st.error(f"Batch validation failed: {error}")

footer_note()
