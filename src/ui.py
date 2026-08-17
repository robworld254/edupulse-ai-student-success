from __future__ import annotations

import html
from typing import Any

import streamlit as st

MAROON = "#7A2634"
MAROON_DARK = "#4D1821"
GOLD = "#B58A3A"
GREEN = "#39715B"
INK = "#20201E"
MUTED = "#686761"


def apply_branding() -> None:
    st.markdown(
        """
        <style>
        :root {
          --brand:#7A2634; --brand-dark:#4D1821; --gold:#B58A3A; --success:#39715B;
          --ink:#20201E; --muted:#686761; --quiet:#8C8982; --canvas:#F7F6F3;
          --surface:#FCFBF9; --sidebar:#EFECE6; --line:#D9D5CE; --line-dark:#BDB8AE;
        }

        html, body, [class*="css"] {
          font-family:"Aptos","Segoe UI Variable Text","Segoe UI",Arial,sans-serif;
          font-feature-settings:"tnum","ss01"; -webkit-font-smoothing:antialiased;
        }
        .stApp { background:var(--canvas); color:var(--ink); }
        .block-container { max-width:1320px; padding:2rem 2.5rem 3.5rem; }
        [data-testid="stToolbar"], #MainMenu, footer { visibility:hidden; }
        [data-testid="stHeader"] { background:transparent; height:2.4rem; }
        [data-testid="stDecoration"] { display:none; }

        /* Editorial navigation rail */
        [data-testid="stSidebar"] {
          background:var(--sidebar); border-right:1px solid #CFC9BF; box-shadow:none;
          transition:min-width .2s ease,max-width .2s ease,transform .2s ease;
        }
        [data-testid="stSidebarContent"] { padding-top:0; position:relative; }
        [data-testid="stSidebarNav"] { padding:3.15rem 1.15rem 0; }
        [data-testid="stSidebarNavItems"] { gap:.15rem; counter-reset:nav-item; }
        [data-testid="stSidebarNavLink"] {
          counter-increment:nav-item; display:grid; grid-template-columns:1.8rem 1fr; align-items:center;
          min-height:45px; padding:.25rem .55rem; border:0; border-left:2px solid transparent;
          border-radius:0; background:transparent; transition:border-color .14s ease,color .14s ease;
        }
        [data-testid="stSidebarNavLink"]:before {
          content:"0" counter(nav-item); color:#9A958C; font-size:.59rem; font-weight:680;
          letter-spacing:.06em; font-variant-numeric:tabular-nums;
        }
        [data-testid="stSidebarNavLink"]:hover { background:transparent; border-left-color:#B8B1A6; }
        [data-testid="stSidebarNavLink"] span { color:#504E49; font-size:.82rem; font-weight:600; letter-spacing:-.012em; }
        [data-testid="stSidebarNavLink"][aria-current="page"] { background:transparent; border-left-color:var(--brand); }
        [data-testid="stSidebarNavLink"][aria-current="page"]:before { color:var(--brand); }
        [data-testid="stSidebarNavLink"][aria-current="page"] span { color:var(--brand); font-weight:720; }

        [data-testid="stSidebarCollapseButton"] { visibility:visible !important; opacity:1 !important; }
        [data-testid="stSidebarCollapseButton"] * { visibility:visible !important; }
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stExpandSidebarButton"] {
          visibility:visible !important; opacity:1 !important; width:29px; height:29px;
          color:#5D5952; background:transparent; border:1px solid #C7C1B7; border-radius:2px;
          box-shadow:none; transition:border-color .14s ease,color .14s ease;
        }
        [data-testid="stExpandSidebarButton"] * { visibility:visible !important; }
        [data-testid="stSidebarCollapseButton"] button:hover,
        [data-testid="stExpandSidebarButton"]:hover { color:var(--brand); border-color:var(--brand); background:transparent; }

        .ep-brand {
          position:fixed; top:1.1rem; left:1.25rem; width:225px; z-index:1002;
          display:flex; align-items:center; gap:.72rem; padding:0 0 .9rem;
          border-bottom:1px solid #D2CCC2;
        }
        .ep-mark {
          width:30px; height:36px; flex:none; display:flex; align-items:center;
          border-left:3px solid var(--brand); color:var(--brand); font-size:.72rem;
          font-weight:800; letter-spacing:.02em; padding-left:.5rem;
        }
        .ep-product { color:var(--ink); font-weight:760; font-size:.94rem; letter-spacing:-.025em; line-height:1.1; }
        .ep-context { color:#77736C; font-size:.61rem; line-height:1.3; margin-top:.25rem; letter-spacing:.025em; }
        .sidebar-bottom {
          position:fixed; left:1.25rem; bottom:1.25rem; width:220px; z-index:1002;
          padding-top:.75rem; border-top:1px solid #D2CCC2; color:#77736C;
          font-size:.62rem; line-height:1.55; letter-spacing:.01em;
        }
        .sidebar-edition { color:var(--brand); font-size:.58rem; font-weight:760; letter-spacing:.11em; text-transform:uppercase; margin-bottom:.28rem; }

        /* Page structure */
        .page-head { margin:.15rem 0 1.1rem; padding-bottom:1.25rem; border-bottom:1px solid var(--line); }
        .page-head .overline { color:var(--brand); font-size:.61rem; line-height:1; font-weight:760; letter-spacing:.14em; text-transform:uppercase; }
        .page-head h1 { margin:.52rem 0 .42rem; color:var(--ink); font-size:clamp(1.8rem,2.45vw,2.42rem); line-height:1.04; font-weight:690; letter-spacing:-.045em; }
        .page-head p { margin:0; color:var(--muted); font-size:.86rem; line-height:1.5; max-width:760px; }
        .context-strip {
          display:flex; align-items:center; justify-content:space-between; gap:1rem;
          margin:0 0 1.2rem; padding:.55rem 0; border-top:1px solid var(--line);
          border-bottom:1px solid var(--line); color:#605E59; font-size:.7rem;
        }
        .context-main { min-width:0; }
        .context-label { color:var(--brand); white-space:nowrap; font-size:.58rem; font-weight:740; letter-spacing:.1em; text-transform:uppercase; }
        .section-head { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin:.25rem 0 .8rem; }
        .section-title { color:var(--ink); font-size:.94rem; font-weight:690; line-height:1.25; letter-spacing:-.02em; }
        .section-description { color:var(--muted); font-size:.7rem; line-height:1.45; margin-top:.2rem; }
        .section-meta { color:#817D75; font-size:.62rem; font-weight:630; white-space:nowrap; }

        /* Fixed-grid metrics, never floating cards */
        .kpi {
          height:128px; box-sizing:border-box; padding:.88rem .95rem .82rem;
          background:transparent; border:0; border-top:2px solid var(--kpi-accent,var(--brand));
          border-bottom:1px solid var(--line); border-radius:0; box-shadow:none; overflow:hidden;
        }
        .kpi .label { color:#77736C; font-size:.58rem; font-weight:730; letter-spacing:.1em; text-transform:uppercase; }
        .kpi .value { color:var(--ink); font-size:1.55rem; font-weight:690; line-height:1.03; letter-spacing:-.04em; margin:.48rem 0 .28rem; overflow-wrap:anywhere; }
        .kpi .note { color:var(--muted); font-size:.65rem; line-height:1.3; }
        .panel { height:100%; padding:.9rem .95rem; background:transparent; border:0; border-top:1px solid var(--line-dark); border-bottom:1px solid var(--line); border-radius:0; box-shadow:none; }
        .panel-title { color:#79756E; font-size:.58rem; font-weight:730; letter-spacing:.1em; text-transform:uppercase; }
        .panel-value { color:var(--ink); font-size:1.02rem; font-weight:670; letter-spacing:-.025em; margin:.4rem 0 .28rem; }
        .compact-note { color:var(--muted); font-size:.7rem; line-height:1.48; }
        [data-testid="stVerticalBlockBorderWrapper"] {
          background:rgba(255,255,255,.42); border-color:var(--line) !important;
          border-radius:2px !important; box-shadow:none; padding:.12rem .18rem;
        }

        /* Controls */
        div[data-testid="stForm"] { background:rgba(255,255,255,.46); border:1px solid var(--line); border-radius:2px; padding:1.05rem 1.15rem; box-shadow:none; }
        label[data-testid="stWidgetLabel"] p { color:#3F3D39; font-size:.72rem; font-weight:630; }
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { background:#FFF; border-color:#CFCAC1; border-radius:2px; min-height:42px; }
        div[data-baseweb="input"] > div:focus-within, div[data-baseweb="select"] > div:focus-within { border-color:var(--brand); box-shadow:0 0 0 2px rgba(122,38,52,.08); }
        [data-testid="stExpander"] { border:1px solid var(--line); border-radius:2px; background:rgba(255,255,255,.3); overflow:hidden; }
        [data-testid="stExpander"] summary { min-height:43px; }
        .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button, [data-testid="stPageLink"] a {
          min-height:41px; border-radius:2px; font-size:.73rem; font-weight:640;
          border:1px solid #C7C1B8; background:transparent; color:#34322F; box-shadow:none; transition:border-color .14s ease,color .14s ease,background .14s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stPageLink"] a:hover { border-color:var(--brand); color:var(--brand); background:#F4F0EC; transform:none; box-shadow:none; }
        .stButton > button:focus-visible, .stFormSubmitButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline:2px solid rgba(122,38,52,.2); outline-offset:2px; }
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] { border-color:var(--brand); background:var(--brand); color:#FFF; box-shadow:none; }
        .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover { border-color:var(--brand-dark); background:var(--brand-dark); color:#FFF; box-shadow:none; }
        [data-testid="stFileUploaderDropzone"] { border:1px dashed #BFB9AF; border-radius:2px; background:transparent; padding:1rem; }
        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:2px; overflow:hidden; }
        [data-baseweb="tab-list"] { gap:1rem; border-bottom:1px solid var(--line); }
        [data-baseweb="tab"] { border-radius:0; padding:.65rem .1rem; font-size:.72rem; }

        /* Quiet process feedback */
        .loading-shell { padding:.75rem 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line); background:transparent; }
        .loading-label { display:flex; justify-content:space-between; color:#4E4B46; font-size:.68rem; font-weight:630; }
        .loading-label span:last-child { color:#88837A; font-weight:520; }
        .loading-track { height:2px; overflow:hidden; background:#DEDAD3; margin-top:.62rem; }
        .loading-track:after { content:""; display:block; width:36%; height:100%; background:var(--brand); animation:loader 1.05s ease-in-out infinite; }
        @keyframes loader { 0%{transform:translateX(-110%)} 100%{transform:translateX(340%)} }
        [data-testid="stProgress"] p { color:#5F5B55; font-size:.65rem; font-weight:620; margin-bottom:.3rem; }
        [data-testid="stProgressBarTrack"] { height:4px; background:#DEDAD3; border-radius:0; overflow:hidden; }
        [data-testid="stProgressBarTrack"] > div { background:var(--brand); border-radius:0; }
        .workflow { display:grid; grid-template-columns:repeat(3,1fr); margin:.1rem 0 1.1rem; border-top:1px solid var(--line); border-bottom:1px solid var(--line); background:transparent; }
        .workflow-step { position:relative; padding:.7rem .8rem .7rem 2rem; color:#89847C; font-size:.65rem; font-weight:620; border-right:1px solid var(--line); }
        .workflow-step:last-child { border-right:0; }
        .workflow-step:before { content:attr(data-step); position:absolute; left:.58rem; top:.65rem; color:#9A958D; font-size:.58rem; font-weight:700; }
        .workflow-step.active { color:var(--brand); background:transparent; }
        .workflow-step.complete { color:#504D47; }
        .workflow-step.active:before, .workflow-step.complete:before { color:var(--brand); }

        /* Assessment result as an evidence sheet, not a dashboard gauge */
        .risk-shell { display:grid; grid-template-columns:190px 1fr; gap:1.45rem; align-items:stretch; padding:1.15rem 0; border-top:2px solid var(--ring,var(--brand)); border-bottom:1px solid var(--line); background:transparent; }
        .risk-scoreboard { padding:.2rem 1.25rem .2rem .15rem; border-right:1px solid var(--line); }
        .risk-scoreboard .score { color:var(--ink); font-size:3.4rem; line-height:.95; font-weight:650; letter-spacing:-.075em; }
        .risk-scoreboard .caption { color:#77736C; font-size:.57rem; font-weight:730; letter-spacing:.1em; text-transform:uppercase; margin-top:.55rem; }
        .risk-meter { height:3px; background:#DDD8D0; margin-top:1rem; }
        .risk-meter span { display:block; width:calc(var(--score)*1%); height:100%; background:var(--ring,var(--brand)); }
        .risk-detail .band { color:var(--ring,var(--brand)); font-size:.58rem; font-weight:760; letter-spacing:.11em; text-transform:uppercase; }
        .risk-detail h2 { color:var(--ink); font-size:1.42rem; font-weight:670; line-height:1.18; letter-spacing:-.035em; margin:.38rem 0 .4rem; }
        .risk-detail p { color:var(--muted); font-size:.75rem; line-height:1.55; margin:0; max-width:850px; }
        .prob-row { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-top:.9rem; }
        .prob { padding:.55rem 0 0; border:0; border-top:1px solid var(--line); background:transparent; }
        .prob .name { color:#79756E; font-size:.56rem; font-weight:730; letter-spacing:.09em; text-transform:uppercase; }
        .prob .num { color:var(--ink); font-size:1rem; font-weight:680; letter-spacing:-.025em; margin-top:.2rem; }
        .path-card { padding:.62rem 0; background:transparent; border:0; border-bottom:1px solid var(--line); border-radius:0; margin:0; }
        .path-card .service { color:var(--brand); font-size:.56rem; font-weight:740; letter-spacing:.09em; text-transform:uppercase; }
        .path-card .action { color:#514E49; font-size:.7rem; line-height:1.48; margin-top:.16rem; }

        [data-testid="stAlert"] { border-radius:2px; border-width:1px; font-size:.74rem; }
        [data-testid="stCaptionContainer"] { color:#817D75; font-size:.66rem; line-height:1.45; }
        hr { border-color:var(--line); margin:1.35rem 0; }
        .footer-note { margin-top:1.8rem; padding-top:.75rem; border-top:1px solid var(--line); color:#88837B; font-size:.61rem; line-height:1.5; }
        .footer-note strong { color:#66625C; }

        @media(max-width:900px) {
          .block-container { padding:1.1rem 1rem 2.5rem; }
          .context-label { display:none; }
          .risk-shell { grid-template-columns:1fr; }
          .risk-scoreboard { border-right:0; border-bottom:1px solid var(--line); padding-bottom:1rem; }
          .prob-row { grid-template-columns:1fr; gap:.5rem; }
          .workflow-step { padding-left:.65rem; text-align:center; }
          .workflow-step:before { display:none; }
        }
        @media(prefers-reduced-motion:reduce) { *, *:before, *:after { animation-duration:.01ms !important; transition-duration:.01ms !important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="ep-brand">
          <div class="ep-mark">EP</div>
          <div>
            <div class="ep-product">EduPulse AI</div>
            <div class="ep-context">Student Success Intelligence</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_footer() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-bottom">
          <div class="sidebar-edition">Research build 3.0</div>
          Semester 1 student-support model<br>Kabarak academic context
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", eyebrow: str = "Intelligence workspace") -> None:
    subtitle_html = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="page-head">
          <div class="overline">{html.escape(eyebrow)}</div>
          <h1>{html.escape(title)}</h1>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, note: str = "", accent: str = MAROON) -> None:
    st.markdown(
        f"""
        <div class="kpi" style="--kpi-accent:{html.escape(accent)}">
          <div class="label">{html.escape(label)}</div>
          <div class="value">{html.escape(value)}</div>
          <div class="note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def context_strip(text: str, label: str = "Research context") -> None:
    st.markdown(
        f"""
        <div class="context-strip">
          <div class="context-main">{html.escape(text)}</div>
          <div class="context-label">{html.escape(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str, description: str = "", meta: str = "") -> None:
    description_html = f'<div class="section-description">{html.escape(description)}</div>' if description else ""
    meta_html = f'<div class="section-meta">{html.escape(meta)}</div>' if meta else ""
    st.markdown(
        f"""
        <div class="section-head">
          <div><div class="section-title">{html.escape(text)}</div>{description_html}</div>
          {meta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def loading_state(label: str, detail: str = "Preparing workspace") -> Any:
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div class="loading-shell">
          <div class="loading-label"><span>{html.escape(label)}</span><span>{html.escape(detail)}</span></div>
          <div class="loading-track"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return placeholder


def workflow_progress(active: int = 1, target: Any | None = None) -> None:
    labels = ["Review inputs", "Run model", "Plan support"]
    steps = []
    for index, label in enumerate(labels, start=1):
        state = "complete" if index < active else "active" if index == active else ""
        steps.append(f'<div class="workflow-step {state}" data-step="{index}">{html.escape(label)}</div>')
    renderer = target if target is not None else st
    renderer.markdown(f'<div class="workflow">{"".join(steps)}</div>', unsafe_allow_html=True)


def footer_note() -> None:
    st.markdown(
        """
        <div class="footer-note"><strong>Research-use boundary.</strong> Public UCI data power this academic prototype; no Kabarak University student records were used. Local validation is required before operational use.</div>
        """,
        unsafe_allow_html=True,
    )


def plotly_config() -> dict[str, bool]:
    return {"displayModeBar": False, "responsive": True, "scrollZoom": False, "doubleClick": False}


source_strip = context_strip
