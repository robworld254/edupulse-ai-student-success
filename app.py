import streamlit as st

from src.ui import apply_branding, sidebar_brand, sidebar_footer

st.set_page_config(
    page_title="EduPulse AI | Student Success Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_branding()
sidebar_brand()

pages = [
    st.Page("views/dashboard.py", title="Dashboard", default=True),
    st.Page("views/assessment.py", title="Student Assessment"),
    st.Page("views/analytics.py", title="Cohort Analytics"),
    st.Page("views/model.py", title="Model Performance"),
]

pg = st.navigation(pages)
sidebar_footer()
pg.run()
