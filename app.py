"""Streamlit interface for food demand prediction and waste reduction."""

from __future__ import annotations

from datetime import date, datetime
from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st

from data_processing import DAYS, MEALS, MENUS, YES_NO_VALUES
from database import get_prediction_history, init_db, save_prediction
from food_logic import calculate_food_requirement
from model import predict_demand, prediction_engine_status
from seed_demo_history import seed_demo_history_if_empty


APP_TITLE = "Smart Food Demand Prediction and Waste Reduction System"
CHART_COLORS = ["#087A5B", "#63D6A8", "#9AA6F3", "#F1AA72", "#DB6B61"]

st.set_page_config(page_title=APP_TITLE, page_icon="🍃", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --forest: #075B48;
        --forest-dark: #044A3C;
        --green: #087A5B;
        --mint: #63D6A8;
        --mint-soft: #DDF8EC;
        --ink: #151A2E;
        --text: #374057;
        --muted: #6B7488;
        --canvas: #F8F8FC;
        --soft: #F0F2FB;
        --line: #E3E7EF;
    }
    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
            "Segoe UI", sans-serif;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] {
        background: rgba(248, 248, 252, 0.82);
        backdrop-filter: blur(12px);
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1440px;
        padding: 1rem 2.6rem 4rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--forest) 0%, var(--forest-dark) 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebarContent"] { padding: 1.15rem 0.8rem 1rem; }
    [data-testid="stSidebar"] * { color: #F4FFF9; }
    .sidebar-brand { padding: 0.25rem 0.45rem 1rem; }
    .brand-row { display: flex; align-items: center; gap: 0.7rem; }
    .brand-mark {
        width: 2.15rem; height: 2.15rem; display: grid; place-items: center;
        border-radius: 0.65rem; background: rgba(73,225,166,.2);
        border: 1px solid rgba(131,244,202,.18); color: #82F0C4;
        font-weight: 900; font-size: 1.05rem;
    }
    .brand-name { color: white; font-size: 1.05rem; font-weight: 780; }
    .brand-tagline { color: #9FD3C0; font-size: 0.77rem; margin-top: 0.45rem; }
    .nav-label {
        color: #76B9A3; font-size: 0.68rem; font-weight: 800;
        letter-spacing: .14em; text-transform: uppercase; padding: .35rem .55rem .2rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: .28rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: .62rem; padding: .54rem .62rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255,255,255,.07);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: rgba(0,36,29,.46); box-shadow: inset 3px 0 0 #65E7B4;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label p {
        font-size: .91rem; font-weight: 590;
    }
    .engine-card {
        border: 1px solid rgba(137,235,198,.16); border-radius: .68rem;
        background: rgba(0,42,34,.36); padding: .72rem .78rem; margin-top: .55rem;
    }
    .engine-card small { display:block; color:#92CDB7; line-height:1.35; }
    .engine-card strong { color:white; font-size:.82rem; }

    /* Shared page structure */
    .topbar {
        display:flex; align-items:center; justify-content:space-between;
        min-height:2.7rem; border-bottom:1px solid var(--line);
        margin:-.2rem -2.6rem 1.65rem; padding:0 2.6rem .75rem;
    }
    .topbar-name { color:var(--text); font-size:.82rem; font-weight:650; }
    .date-pill {
        background:#F0F2F7; color:var(--text); border-radius:99rem;
        padding:.38rem .72rem; font-size:.76rem; font-weight:600;
    }
    .page-kicker, .section-eyebrow {
        color:var(--green); font-size:.7rem; font-weight:820;
        letter-spacing:.13em; text-transform:uppercase;
    }
    .page-title {
        color:var(--ink); font-size:clamp(2rem,3.2vw,2.75rem); line-height:1.04;
        letter-spacing:-.045em; margin:.36rem 0 0;
    }
    .page-description { color:var(--muted); font-size:.98rem; margin:.55rem 0 1.65rem; }
    .section-heading {
        color:var(--ink); font-size:1.2rem; font-weight:760;
        letter-spacing:-.02em; margin:.16rem 0 .85rem;
    }
    .surface-card {
        background:white; border:1px solid var(--line); border-radius:1rem;
        box-shadow:0 8px 24px rgba(24,35,54,.045); padding:1.25rem;
    }

    /* Dashboard and result cards */
    .hero-card {
        background:radial-gradient(circle at 86% 14%,rgba(112,230,186,.30),transparent 26rem),
            linear-gradient(118deg,#246F5C 0%,#0C6B52 52%,#07523F 100%);
        border-radius:1.05rem; padding:2rem 2.15rem; color:white;
        box-shadow:0 8px 20px rgba(3,70,54,.16); margin-bottom:1.45rem;
    }
    .hero-kicker {
        display:inline-flex; background:rgba(129,239,199,.17); color:#A9F1D6;
        border:1px solid rgba(167,245,216,.15); border-radius:99rem;
        padding:.32rem .7rem; font-size:.7rem; font-weight:800;
        letter-spacing:.12em; text-transform:uppercase;
    }
    .hero-card h1 {
        color:white; font-size:clamp(2rem,4vw,3rem); line-height:1.06;
        letter-spacing:-.045em; max-width:1040px; margin:.8rem 0 .55rem;
    }
    .hero-card p {
        color:#C7E3D8; font-size:.98rem; line-height:1.55; max-width:850px; margin:0;
    }
    .metric-card {
        background:white; border:1px solid var(--line); border-radius:1rem;
        box-shadow:0 7px 22px rgba(24,35,54,.045); padding:1.05rem 1.1rem 1rem;
        min-height:11.25rem; display:flex; flex-direction:column;
        justify-content:space-between; margin-bottom:.55rem;
    }
    .metric-head { display:flex; align-items:flex-start; justify-content:space-between; gap:.7rem; }
    .metric-eyebrow {
        color:#7C857F; font-size:.65rem; font-weight:820;
        letter-spacing:.09em; text-transform:uppercase;
    }
    .metric-label { color:var(--ink); font-size:1rem; font-weight:720; line-height:1.25; margin-top:.2rem; }
    .metric-icon {
        display:grid; place-items:center; width:2.25rem; height:2.25rem;
        border-radius:.62rem; background:var(--mint-soft); color:var(--green);
        font-size:1rem; font-weight:850; flex:0 0 auto;
    }
    .metric-value {
        color:var(--ink); font-size:2.35rem; line-height:1; letter-spacing:-.055em;
        font-weight:780; margin-top:1rem;
    }
    .metric-value.green { color:var(--green); }
    .metric-unit { color:#A0A8A4; font-size:.76rem; font-weight:650; margin-left:.25rem; }
    .metric-note { color:#737E78; font-size:.72rem; line-height:1.35; margin-top:.7rem; }
    .record-title { color:var(--ink); font-size:1.35rem; font-weight:780; margin:.15rem 0 .9rem; }
    .record-banner {
        display:flex; align-items:center; justify-content:space-between; gap:1rem;
        background:var(--soft); border-radius:.8rem; padding:1rem 1.1rem; margin:.7rem 0 1rem;
    }
    .record-menu { color:var(--ink); font-size:1.25rem; font-weight:760; }
    .record-meta { color:var(--muted); font-size:.78rem; margin-top:.18rem; }
    .record-day { color:var(--text); font-size:.82rem; font-weight:700; text-align:right; }
    .stat-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.55rem; }
    .mini-stat { background:#F7F8FC; border:1px solid #ECEEF4; border-radius:.7rem; padding:.72rem; }
    .mini-label { color:#7B8494; font-size:.67rem; font-weight:650; }
    .mini-value { color:var(--ink); font-size:1rem; font-weight:780; margin-top:.18rem; }
    .summary-row {
        display:flex; justify-content:space-between; align-items:center; gap:1rem;
        background:var(--soft); border-radius:.65rem; padding:.66rem .78rem;
        margin:.45rem 0; font-size:.78rem;
    }
    .summary-row span:first-child { color:#7A8291; }
    .summary-row strong { color:var(--ink); text-align:right; }
    .source-note { color:#768078; font-size:.7rem; margin-top:.85rem; }
    .risk-badge { border-radius:99rem; padding:.18rem .56rem; font-size:.7rem; font-weight:780; }
    .risk-low { background:#CFF7E6; color:#087A5B; }
    .risk-medium { background:#FFF0C9; color:#946312; }
    .risk-high { background:#FFE0DB; color:#A44035; }

    /* Native Streamlit controls */
    [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
        background:rgba(255,255,255,.98); border-color:var(--line)!important;
        border-radius:1rem!important; box-shadow:0 7px 22px rgba(24,35,54,.04);
    }
    [data-testid="stForm"] { padding:1.25rem 1.3rem 1.35rem; }
    [data-testid="stWidgetLabel"] p { color:var(--text); font-size:.82rem; font-weight:680; }
    [data-baseweb="select"] > div, [data-testid="stNumberInputContainer"],
    [data-testid="stDateInput"] > div > div, [data-testid="stTextInputRootElement"] {
        background:#F1F4F2!important; border-color:transparent!important;
        border-radius:.72rem!important; min-height:3rem;
    }
    .stButton > button, [data-testid="stFormSubmitButton"] > button {
        border-radius:.68rem; min-height:2.85rem; font-weight:740; border:1px solid #0A634E;
    }
    .stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] {
        background:var(--forest); color:white;
    }
    [data-testid="stDataFrame"] {
        background:white; border:1px solid var(--line); border-radius:.9rem;
        overflow:hidden; box-shadow:0 6px 18px rgba(24,35,54,.035);
    }
    [data-testid="stPlotlyChart"] {
        background:white; border:1px solid var(--line); border-radius:.95rem;
        padding:.35rem; box-shadow:0 6px 18px rgba(24,35,54,.035);
    }
    [data-testid="stAlert"] { border-radius:.75rem; }
    h1, h2, h3 { color:var(--ink); }
    @media (max-width:900px) {
        [data-testid="stMainBlockContainer"] { padding:.8rem 1rem 3rem; }
        .topbar { margin:0 -1rem 1.2rem; padding:0 1rem .7rem; }
        .hero-card { padding:1.5rem; }
        .stat-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe(value: object) -> str:
    """Return HTML-safe display text."""
    return escape(str(value))


def page_topbar(page_name: str) -> None:
    today = date.today().strftime("%b %d, %Y")
    st.markdown(
        f'<div class="topbar"><span class="topbar-name">{safe(page_name)}</span>'
        f'<span class="date-pill">Today · {safe(today)}</span></div>',
        unsafe_allow_html=True,
    )


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="page-kicker">{safe(kicker)}</div>'
        f'<h1 class="page-title">{safe(title)}</h1>'
        f'<p class="page-description">{safe(description)}</p>',
        unsafe_allow_html=True,
    )


def section_heading(eyebrow: str, title: str) -> None:
    st.markdown(
        f'<div class="section-eyebrow">{safe(eyebrow)}</div>'
        f'<div class="section-heading">{safe(title)}</div>',
        unsafe_allow_html=True,
    )


def metric_card(eyebrow, label, value, unit, note, icon, green=False) -> None:
    value_class = "metric-value green" if green else "metric-value"
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-head"><div><div class="metric-eyebrow">{safe(eyebrow)}</div>
          <div class="metric-label">{safe(label)}</div></div><div class="metric-icon">{safe(icon)}</div></div>
          <div><div class="{value_class}">{safe(value)}<span class="metric-unit">{safe(unit)}</span></div>
          <div class="metric-note">{safe(note)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_class(risk: object) -> str:
    normalised = str(risk).strip().lower()
    return {"high": "risk-high", "medium": "risk-medium"}.get(normalised, "risk-low")


def style_chart(figure, height: int = 370):
    figure.update_layout(
        template="plotly_white",
        colorway=CHART_COLORS,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font={"color": "#374057", "size": 12},
        title_font={"size": 18, "color": "#151A2E"},
        title_x=0.03,
        height=height,
        margin={"l": 45, "r": 24, "t": 65, "b": 42},
        legend_title_text="",
        hoverlabel={"bgcolor": "#FFFFFF", "font_color": "#151A2E"},
    )
    figure.update_xaxes(gridcolor="#EEF0F5", linecolor="#DDE2EA", zeroline=False)
    figure.update_yaxes(gridcolor="#EEF0F5", linecolor="#DDE2EA", zeroline=False)
    return figure


def load_history() -> pd.DataFrame:
    try:
        return get_prediction_history()
    except Exception as exc:
        st.error("Prediction history could not be loaded.")
        with st.expander("Technical detail"):
            st.code(str(exc))
        return pd.DataFrame()


def recent_table(history: pd.DataFrame, rows: int = 6) -> pd.DataFrame:
    recent = history.head(rows)[
        ["prediction_date", "meal", "menu", "predicted_attendance",
         "recommended_food", "expected_wastage", "wastage_risk", "record_source"]
    ].copy()
    recent.columns = ["Date", "Meal", "Menu", "Predicted", "Recommended",
                      "Potential Surplus", "Risk", "Source"]
    return recent


try:
    init_db()
except Exception as exc:
    st.error("The local database could not be initialized.")
    with st.expander("Technical detail"):
        st.code(str(exc))
    st.stop()

try:
    seed_demo_history_if_empty(count=300)
except Exception as exc:
    st.warning(
        "The application is ready, but its synthetic demonstration history "
        "could not be created automatically."
    )
    with st.expander("Demo data technical detail"):
        st.code(str(exc))

engine_name, engine_detail = prediction_engine_status()

st.sidebar.markdown(
    """
    <div class="sidebar-brand"><div class="brand-row"><div class="brand-mark">◒</div>
    <div class="brand-name">Food Demand</div></div>
    <div class="brand-tagline">Smarter preparation · Less waste</div></div>
    <div class="nav-label">Navigate</div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Navigate", ("Dashboard", "Demand Prediction", "Analytics", "History"),
    label_visibility="collapsed", key="main_page",
)
st.sidebar.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
st.sidebar.markdown(
    f'<div class="engine-card"><small>Prediction method</small>'
    f'<strong>{safe(engine_name)}</strong></div>',
    unsafe_allow_html=True,
)
if engine_name != "Random Forest model":
    st.sidebar.warning("The trained model is unavailable, so the fallback is active.")
    with st.sidebar.expander("Technical detail"):
        st.code(engine_detail)


if page == "Dashboard":
    page_topbar("Dashboard · Overview")
    st.markdown(
        f"""
        <section class="hero-card"><div class="hero-kicker">Campus dining intelligence</div>
        <h1>{safe(APP_TITLE)}</h1><p>Forecast meal attendance, calculate a 5% preparation
        buffer, and review saved planning records in one consistent workspace.</p></section>
        """,
        unsafe_allow_html=True,
    )
    history = load_history()
    columns = st.columns(4)
    if history.empty:
        specs = (
            ("Forecast target", "Predicted attendance", "—", "students", "No saved prediction yet", "P", False),
            ("Production quota", "Recommended portions", "—", "meals", "Includes the planning buffer", "R", True),
            ("Waste ceiling", "Potential surplus", "—", "portions", "Estimated, not measured waste", "S", False),
            ("Planning risk", "Buffer risk", "—", "", "Calculated after a prediction", "✓", True),
        )
        for column, spec in zip(columns, specs):
            with column:
                metric_card(*spec)
        st.info("Create and save a prediction to populate the dashboard.")
    else:
        latest = history.iloc[0]
        specs = (
            ("Forecast target", "Predicted attendance", int(latest["predicted_attendance"]),
             "students", f"{latest['meal']} service", "P", False),
            ("Production quota", "Recommended portions", int(latest["recommended_food"]),
             "meals", "5% planning buffer applied", "R", True),
            ("Waste ceiling", "Potential surplus", int(latest["expected_wastage"]),
             "portions", "Estimate above predicted demand", "S", False),
            ("Planning risk", "Buffer risk", latest["wastage_risk"], "",
             "Based on the preparation buffer", "✓", True),
        )
        for column, spec in zip(columns, specs):
            with column:
                metric_card(*spec)

        left, right = st.columns([1.65, 1], gap="large")
        previous = int(latest["previous_attendance"])
        predicted = int(latest["predicted_attendance"])
        change_text = f"{predicted - previous:+d} vs previous"
        risk = safe(latest["wastage_risk"])
        with left:
            st.markdown(
                f"""
                <div class="surface-card"><div class="section-eyebrow">Operational blueprint</div>
                <div class="record-title">Latest saved prediction</div>
                <div class="record-banner"><div><div class="record-menu">{safe(latest['menu'])}</div>
                <div class="record-meta">{safe(latest['meal'])} service · {safe(latest['prediction_date'])}</div></div>
                <div class="record-day">DAY OF WEEK<br>{safe(latest['day'])}</div></div>
                <div class="stat-grid">
                <div class="mini-stat"><div class="mini-label">Previous attendance</div><div class="mini-value">{previous}</div></div>
                <div class="mini-stat"><div class="mini-label">Predicted attendance</div><div class="mini-value">{predicted}</div></div>
                <div class="mini-stat"><div class="mini-label">Recommended</div><div class="mini-value">{int(latest['recommended_food'])} meals</div></div>
                <div class="mini-stat"><div class="mini-label">Demand change</div><div class="mini-value">{safe(change_text)}</div></div>
                </div><div class="source-note">Prediction method: {safe(engine_name)} · Record source: {safe(latest['record_source'])}</div></div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f"""
                <div class="surface-card"><div class="section-eyebrow">Model inputs</div>
                <div class="record-title">Prediction summary</div>
                <div class="summary-row"><span>Date and day</span><strong>{safe(latest['prediction_date'])} · {safe(latest['day'])}</strong></div>
                <div class="summary-row"><span>Meal service</span><strong>{safe(latest['meal'])}</strong></div>
                <div class="summary-row"><span>Featured menu</span><strong>{safe(latest['menu'])}</strong></div>
                <div class="summary-row"><span>Holiday</span><strong>{safe(latest['holiday'])}</strong></div>
                <div class="summary-row"><span>Examination period</span><strong>{safe(latest['exam_period'])}</strong></div>
                <div class="summary-row"><span>Potential surplus</span><strong>{int(latest['expected_wastage'])} portions</strong></div>
                <div class="summary-row"><span>Buffer risk</span><strong><span class="risk-badge {risk_class(risk)}">{risk}</span></strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
        section_heading("Saved records", "Recent predictions")
        st.dataframe(recent_table(history), width="stretch", hide_index=True, height=260)
        if (history["record_source"] == "Synthetic demo").any():
            st.caption("Some records are labelled Synthetic demo and exist only to demonstrate the interface and analytics.")


elif page == "Demand Prediction":
    page_topbar("Demand Prediction")
    page_header("Plan a meal", "Demand Prediction",
                "Enter the supported meal conditions to estimate attendance and recommended portions.")
    with st.form("prediction_form"):
        left, right = st.columns(2, gap="large")
        with left:
            prediction_date = st.date_input("Prediction date", value=date.today())
            day = st.selectbox("Day", DAYS)
            meal = st.selectbox("Meal", MEALS)
            menu = st.selectbox("Menu", MENUS)
        with right:
            holiday = st.selectbox("Holiday", YES_NO_VALUES)
            exam_period = st.selectbox("Examination period", YES_NO_VALUES)
            previous_attendance = st.number_input(
                "Previous attendance", min_value=0, max_value=5000, value=500, step=10,
                help="Attendance from a comparable previous meal service.",
            )
        submitted = st.form_submit_button("Predict demand", type="primary", width="stretch")

    if submitted:
        try:
            predicted = predict_demand(day, meal, menu, holiday, exam_period, previous_attendance)
            recommended, surplus, risk = calculate_food_requirement(predicted)
            st.session_state["prediction"] = {
                "prediction_date": prediction_date, "day": day, "meal": meal, "menu": menu,
                "holiday": holiday, "exam_period": exam_period,
                "previous_attendance": int(previous_attendance),
                "predicted_attendance": predicted, "recommended_food": recommended,
                "expected_wastage": surplus, "wastage_risk": risk,
            }
            st.session_state["prediction_saved"] = False
        except Exception as exc:
            st.error("The prediction could not be created. Check the entered values.")
            with st.expander("Technical detail"):
                st.code(str(exc))

    if "prediction" in st.session_state:
        result = st.session_state["prediction"]
        st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
        section_heading("Forecast ready", "Prediction result")
        columns = st.columns(4)
        specs = (
            ("Forecast target", "Predicted attendance", result["predicted_attendance"], "students", f"{result['meal']} service", "P", False),
            ("Production quota", "Recommended portions", result["recommended_food"], "meals", "5% planning buffer included", "R", True),
            ("Waste ceiling", "Potential surplus", result["expected_wastage"], "portions", "Estimate above predicted demand", "S", False),
            ("Planning risk", "Buffer risk", result["wastage_risk"], "", "Based on the buffer ratio", "✓", True),
        )
        for column, spec in zip(columns, specs):
            with column:
                metric_card(*spec)
        st.success(
            f"Estimated demand is **{result['predicted_attendance']} students**. "
            f"The recommended preparation quantity is **{result['recommended_food']} portions**."
        )
        st.caption(
            "Recommended portions include a 5% planning buffer. Potential surplus is the "
            "difference between recommended portions and predicted attendance; it is not measured food waste."
        )
        if st.session_state.pop("prediction_save_notice", False):
            st.success("Prediction saved successfully.")
        already_saved = st.session_state.get("prediction_saved", False)
        if st.button("Save prediction", type="primary", width="stretch", disabled=already_saved):
            try:
                save_prediction(
                    result["prediction_date"].strftime("%Y-%m-%d"), result["day"], result["meal"],
                    result["menu"], result["holiday"], result["exam_period"],
                    result["previous_attendance"], result["predicted_attendance"],
                    result["recommended_food"], result["expected_wastage"],
                    result["wastage_risk"], datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                st.session_state["prediction_saved"] = True
                st.session_state["prediction_save_notice"] = True
                st.rerun()
            except Exception as exc:
                st.error("The prediction could not be saved.")
                with st.expander("Technical detail"):
                    st.code(str(exc))
        elif already_saved:
            st.info("This result is already saved. Submit the form again to create a new result.")


elif page == "Analytics":
    page_topbar("Analytics")
    page_header("Explore patterns", "Analytics",
                "Review demand, preparation, and potential-surplus patterns across saved records.")
    history = load_history()
    if history.empty:
        st.info("Save predictions to generate analytics.")
    else:
        if (history["record_source"] == "Synthetic demo").any():
            st.caption("Charts may include Synthetic demo records. They are not measured campus activity.")
        with st.container(border=True):
            section_heading("Refine the view", "Analytics filters")
            meals = sorted(history["meal"].dropna().unique())
            menus = sorted(history["menu"].dropna().unique())
            sources = sorted(history["record_source"].dropna().unique())
            if len(sources) > 1:
                c1, c2, c3 = st.columns(3)
            else:
                c1, c2 = st.columns(2)
            meal_filter = c1.multiselect("Meal", meals, default=meals)
            menu_filter = c2.multiselect("Menu", menus, default=menus)
            source_filter = c3.multiselect("Record source", sources, default=sources) if len(sources) > 1 else sources
            if len(sources) == 1:
                st.caption(
                    f"Record source filter hidden because all saved records currently come from {sources[0]}."
                )
        filtered = history[
            history["meal"].isin(meal_filter) & history["menu"].isin(menu_filter)
            & history["record_source"].isin(source_filter)
        ].copy()
        if filtered.empty:
            st.warning("No records match the selected filters.")
        else:
            filtered["prediction_date"] = pd.to_datetime(filtered["prediction_date"], errors="coerce")
            filtered = filtered.dropna(subset=["prediction_date"]).sort_values("prediction_date")
            metrics = st.columns(4)
            values = (
                ("Saved records", len(filtered)),
                ("Average demand (students)", round(filtered["predicted_attendance"].mean())),
                ("Average preparation (portions)", round(filtered["recommended_food"].mean())),
                ("Potential surplus (portions)", int(filtered["expected_wastage"].sum())),
            )
            for col, (label, value) in zip(metrics, values):
                col.metric(label, value)
            left, right = st.columns(2, gap="large")
            fig1 = px.line(filtered, x="prediction_date", y="predicted_attendance", color="meal",
                           markers=True, title="Predicted Attendance Trend",
                           labels={"prediction_date":"Date", "predicted_attendance":"Students", "meal":"Meal"})
            left.plotly_chart(style_chart(fig1), width="stretch")
            meal_data = filtered.groupby("meal", as_index=False)["predicted_attendance"].mean()
            fig2 = px.bar(meal_data, x="meal", y="predicted_attendance", color="meal",
                          title="Average Demand by Meal",
                          labels={"meal":"Meal", "predicted_attendance":"Average students"})
            fig2.update_layout(showlegend=False)
            right.plotly_chart(style_chart(fig2), width="stretch")
            comparison = filtered.melt(id_vars=["prediction_date"],
                value_vars=["predicted_attendance", "recommended_food"],
                var_name="Measure", value_name="Portions")
            comparison["Measure"] = comparison["Measure"].map({
                "predicted_attendance":"Predicted demand",
                "recommended_food":"Recommended preparation"})
            fig3 = px.bar(comparison, x="prediction_date", y="Portions", color="Measure",
                          barmode="group", title="Demand vs Recommended Preparation",
                          labels={"prediction_date":"Date"})
            left.plotly_chart(style_chart(fig3), width="stretch")
            surplus_data = filtered.groupby("menu", as_index=False)["expected_wastage"].sum()
            surplus_data = surplus_data.sort_values("expected_wastage")
            fig4 = px.bar(surplus_data, x="expected_wastage", y="menu", orientation="h",
                          title="Potential Surplus by Menu",
                          labels={"expected_wastage":"Potential surplus portions", "menu":"Menu"})
            right.plotly_chart(style_chart(fig4), width="stretch")


elif page == "History":
    page_topbar("Prediction History")
    history = load_history()
    total = 0 if history.empty else len(history)
    page_header(f"Saved records · {total} total", "Prediction History",
                "Inspect saved planning results and distinguish demo records from user predictions.")
    if history.empty:
        st.info("No predictions have been saved yet.")
    else:
        risk_options = sorted(history["wastage_risk"].dropna().unique())
        source_options = sorted(history["record_source"].dropna().unique())
        with st.container(border=True):
            widths = [2.2, 1]
            if len(risk_options) > 1:
                widths.append(1)
            if len(source_options) > 1:
                widths.append(1.15)
            filter_columns = iter(st.columns(widths))
            c1, c2 = next(filter_columns), next(filter_columns)
            term = c1.text_input("Search", placeholder="Search by date, day, meal, or menu")
            meal_choice = c2.selectbox("Meal", ["All meals", *sorted(history["meal"].dropna().unique())])
            risk_choice = None
            source_choice = None
            if len(risk_options) > 1:
                risk_choice = next(filter_columns).selectbox("Buffer risk", ["All risks", *risk_options])
            if len(source_options) > 1:
                source_choice = next(filter_columns).selectbox("Source", ["All sources", *source_options])
            if len(risk_options) == 1:
                st.caption(
                    f"Risk filter hidden because all saved records currently have "
                    f"{risk_options[0]} buffer risk."
                )
            if len(source_options) == 1:
                st.caption(
                    f"Source filter hidden because all saved records currently come from {source_options[0]}."
                )
        filtered = history.copy()
        if term.strip():
            blob = filtered[["prediction_date", "day", "meal", "menu"]].astype(str).agg(" ".join, axis=1)
            filtered = filtered[blob.str.contains(term.strip(), case=False, na=False)]
        if meal_choice != "All meals":
            filtered = filtered[filtered["meal"] == meal_choice]
        if risk_choice and risk_choice != "All risks":
            filtered = filtered[filtered["wastage_risk"] == risk_choice]
        if source_choice and source_choice != "All sources":
            filtered = filtered[filtered["record_source"] == source_choice]
        if filtered.empty:
            st.warning("No records match the selected filters.")
        else:
            display = filtered[["prediction_date", "day", "meal", "menu", "holiday",
                "exam_period", "previous_attendance", "predicted_attendance",
                "recommended_food", "expected_wastage", "wastage_risk", "record_source"]].copy()
            display.columns = ["Date", "Day", "Meal", "Menu", "Holiday", "Exam Period",
                "Previous Attendance", "Predicted Attendance", "Recommended Portions",
                "Potential Surplus", "Risk", "Source"]
            st.dataframe(display, width="stretch", hide_index=True, height=560,
                column_config={"Date":st.column_config.TextColumn(width="small"),
                               "Menu":st.column_config.TextColumn(width="medium"),
                               "Source":st.column_config.TextColumn(width="medium")})
            st.caption(
                f"Showing {len(filtered):,} of {len(history):,} saved records. "
                "Synthetic demo records are labelled in the Source column."
            )
