from __future__ import annotations

import streamlit as st

from src.config import FRIENDLY_FEATURE_NAMES, MODEL_PATH, MODEL_VERSION, RESULTS_PATH
from src.explainability import local_sensitivity
from src.features import build_assessment_row
from src.inference import load_model, load_results, support_priority
from src.localization import DEMO_PROFILES, FEE_STATUS, KABARAK_SCHOOLS, SUPPORT_SERVICES
from src.reporting import assessment_html
from src.ui import (
    context_strip,
    footer_note,
    loading_state,
    page_header,
    section_title,
    workflow_progress,
)


@st.cache_resource(show_spinner=False)
def assessment_model():
    return load_model()


@st.cache_data(show_spinner=False)
def assessment_results() -> dict:
    return load_results()


page_header(
    "Student Assessment",
    "A focused Semester 1 assessment that turns model evidence into a human support review.",
    "Support workflow",
)
context_strip("Seven model inputs · Optional references excluded · Processed locally", "Privacy by design")

if not MODEL_PATH.exists() or not RESULTS_PATH.exists():
    st.error("The trained pipeline is unavailable. Run START_EDUPULSE.bat to complete setup.")
    st.stop()

loader = loading_state("Preparing assessment workspace", "Loading calibrated model")
try:
    model = assessment_model()
    results = assessment_results()
finally:
    loader.empty()
baselines = results["feature_baselines"]
risk_bands = results["risk_bands"]

defaults = {
    "units_registered": 6,
    "units_passed": 5,
    "average_mark_pct": 64,
    "assessments_completed": 8,
    "fee_status": "Up to date",
    "scholarship": "No",
    "student_name": "",
    "registration_number": "",
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def apply_profile(name: str) -> None:
    profile = DEMO_PROFILES[name]
    for key, value in profile.items():
        st.session_state[key] = value


def reset_form() -> None:
    for key, value in defaults.items():
        st.session_state[key] = value


workflow_slot = st.empty()
workflow_progress(1, workflow_slot)

section_title("Demonstration profiles", "Use a synthetic profile or enter a custom Semester 1 snapshot.")
p1, p2, p3, reset_column = st.columns([1, 1, 1, 0.65])
with p1:
    st.button("Strong Progress", on_click=apply_profile, args=("Strong Progress",), width="stretch")
with p2:
    st.button("Monitor", on_click=apply_profile, args=("Monitor",), width="stretch")
with p3:
    st.button(
        "Higher Support Need",
        on_click=apply_profile,
        args=("Higher Support Need",),
        width="stretch",
    )
with reset_column:
    st.button("Reset", on_click=reset_form, width="stretch")
st.caption("Synthetic examples only; no real student records are included.")

with st.expander("Student reference (optional and excluded from prediction)", expanded=False):
    reference_left, reference_middle, reference_right = st.columns(3)
    with reference_left:
        student_name = st.text_input("Student name", key="student_name")
    with reference_middle:
        registration_number = st.text_input(
            "Registration number", key="registration_number", placeholder="e.g. CM/M/1234/01/26"
        )
    with reference_right:
        school = st.selectbox("School", KABARAK_SCHOOLS, index=6)

with st.form("assessment_form", clear_on_submit=False):
    left, middle, right = st.columns(3, gap="large")
    with left:
        units_registered = st.number_input(
            "Units registered", min_value=0, max_value=26, step=1, key="units_registered"
        )
        if st.session_state["units_passed"] > units_registered:
            st.session_state["units_passed"] = int(units_registered)
        units_passed = st.number_input(
            "Units passed",
            min_value=0,
            max_value=int(units_registered),
            step=1,
            key="units_passed",
        )
    with middle:
        average_mark_pct = st.number_input(
            "Semester 1 average mark (%)",
            min_value=0,
            max_value=100,
            step=1,
            key="average_mark_pct",
            help="The research source uses 0–20; the interface displays the exact linear conversion to 0–100.",
        )
        assessments_completed = st.number_input(
            "Assessments completed",
            min_value=0,
            max_value=45,
            step=1,
            key="assessments_completed",
        )
    with right:
        fee_status = st.selectbox("Fee status", list(FEE_STATUS), key="fee_status")
        scholarship = (
            st.segmented_control(
                "Scholarship / sponsorship",
                ["No", "Yes"],
                key="scholarship",
                selection_mode="single",
            )
            or "No"
        )
    submitted = st.form_submit_button("Run Assessment", type="primary", width="stretch")

if submitted:
    workflow_progress(2, workflow_slot)
    task_progress = st.progress(12, text="Validating assessment inputs")
    tuition, debtor = FEE_STATUS[fee_status]
    row = build_assessment_row(
        units_registered=units_registered,
        units_passed=units_passed,
        average_mark_pct=average_mark_pct,
        assessments_completed=assessments_completed,
        tuition_up_to_date=tuition,
        outstanding_fee_balance=debtor,
        scholarship_support=int(scholarship == "Yes"),
    )
    task_progress.progress(48, text="Running calibrated prediction")
    probabilities_array = model.predict_proba(row)[0]
    probability = {str(label): float(value) for label, value in zip(model.classes_, probabilities_array)}
    dropout_probability = probability["Dropout"]
    predicted = str(model.classes_[int(probabilities_array.argmax())])
    priority = support_priority(dropout_probability, risk_bands)
    title_map = {
        "Higher Priority": "Prompt supportive review recommended",
        "Monitor": "Continued monitoring recommended",
        "Lower Priority": "Routine support and monitoring",
    }
    ring_map = {
        "Higher Priority": "#8F1D2C",
        "Monitor": "#D09A28",
        "Lower Priority": "#4D9D45",
    }
    score = round(dropout_probability * 100)
    task_progress.progress(76, text="Generating support evidence")
    st.markdown(
        f"""
        <div class='risk-shell' style='--ring:{ring_map[priority]}'>
          <div class='risk-scoreboard' style='--score:{score};--ring:{ring_map[priority]}'>
            <div class='score'>{score}%</div>
            <div class='caption'>Estimated dropout probability</div>
            <div class='risk-meter'><span></span></div>
          </div>
          <div class='risk-detail'>
            <div class='band'>{priority}</div><h2>{title_map[priority]}</h2>
            <p>Highest-probability outcome: <strong>{predicted}</strong>. This is a model-estimated probability for prioritizing a human support review—not a prediction guarantee.</p>
            <div class='prob-row'>
              <div class='prob'><div class='name'>Dropout</div><div class='num'>{probability["Dropout"]:.1%}</div></div>
              <div class='prob'><div class='name'>Enrolled</div><div class='num'>{probability["Enrolled"]:.1%}</div></div>
              <div class='prob'><div class='name'>Graduate</div><div class='num'>{probability["Graduate"]:.1%}</div></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sensitivity = local_sensitivity(model, row, baselines)
    signal_rows = sensitivity.reindex(sensitivity["risk_change"].abs().sort_values(ascending=False).index).head(4)
    signal_text: list[str] = []
    for _, signal in signal_rows.iterrows():
        feature = str(signal["feature"])
        direction = "higher" if float(signal["risk_change"]) > 0 else "lower"
        signal_text.append(
            f"{FRIENDLY_FEATURE_NAMES.get(feature, feature)} was associated with {direction} estimated dropout risk relative to the research-cohort median."
        )

    actions: list[tuple[str, str]] = []
    pass_rate = units_passed / units_registered if units_registered else 0
    if pass_rate < 0.65 or average_mark_pct < 55:
        actions.append(
            (SUPPORT_SERVICES["academic"], "Review unit performance, study planning and academic support options.")
        )
    if assessments_completed < max(units_registered, 4):
        actions.append((SUPPORT_SERVICES["academic"], "Check assessment participation and any missed coursework."))
    if fee_status != "Up to date":
        actions.append(
            (SUPPORT_SERVICES["finance"], "Review fee status, payment arrangements and available funding support.")
        )
    if priority == "Higher Priority":
        actions.append(
            (
                SUPPORT_SERVICES["wellbeing"],
                "Offer a confidential check-in and connect the student to appropriate support.",
            )
        )
    if not actions:
        actions.append(
            (SUPPORT_SERVICES["academic"], "Continue routine academic monitoring and normal student support.")
        )

    task_progress.progress(100, text="Assessment ready for human review")
    workflow_progress(3, workflow_slot)

    st.write("")
    signals_column, pathways_column = st.columns(2, gap="large")
    with signals_column, st.container(border=True):
        section_title("Key model signals", "Largest local sensitivity shifts.")
        for signal in signal_text:
            st.markdown(f"<div class='path-card'><div class='action'>{signal}</div></div>", unsafe_allow_html=True)
        st.caption("Model associations, not causal explanations.")
    with pathways_column, st.container(border=True):
        section_title("Suggested support pathway", "Actions for staff consideration.")
        for service, action in actions[:4]:
            st.markdown(
                f"<div class='path-card'><div class='service'>{service}</div><div class='action'>{action}</div></div>",
                unsafe_allow_html=True,
            )

    input_summary = {
        "Units registered": int(units_registered),
        "Units passed": int(units_passed),
        "Semester 1 average mark": f"{average_mark_pct}%",
        "Assessments completed": int(assessments_completed),
        "Fee status": fee_status,
        "Scholarship / sponsorship": scholarship,
    }
    report = assessment_html(
        registration_number=registration_number,
        student_name=student_name,
        school=school,
        risk_band=priority,
        dropout_probability=dropout_probability,
        predicted_outcome=predicted,
        probabilities=probability,
        inputs=input_summary,
        actions=[f"{service}: {action}" for service, action in actions[:4]],
        signals=signal_text,
        model_version=results["project"].get("model_version", MODEL_VERSION),
    )
    safe_reference = (registration_number or "student").replace("/", "-").replace("\\", "-")
    st.download_button(
        "Download Assessment Summary",
        data=report,
        file_name=f"edupulse_assessment_{safe_reference}.html",
        mime="text/html",
    )

footer_note()
