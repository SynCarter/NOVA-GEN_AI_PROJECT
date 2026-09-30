import os
import re
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.engine import (
    analyze_question,
    profile_data,
    auto_insights
)

from src.ai import (
    explain_with_local_llm,
    llm_available
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="NOVA | AI Data Intelligence Studio — S Larens",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MODERN DESIGN SYSTEM & CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --primary: #8b5cf6;
        --primary-glow: rgba(139, 92, 246, 0.35);
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --accent-amber: #f59e0b;
        --bg-dark: #070b14;
        --card-bg: rgba(15, 23, 42, 0.72);
        --card-border: rgba(255, 255, 255, 0.08);
        --card-hover: rgba(30, 41, 59, 0.85);
        --text-main: #f1f5f9;
        --text-muted: #94a3b8;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Background glow effect */
    .stApp {
        background-color: #070b14;
        background-image: 
            radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.08) 0px, transparent 50%),
            radial-gradient(at 50% 50%, rgba(15, 23, 42, 0.5) 0px, transparent 100%);
    }

    /* Top Brand Hero */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 20px -5px rgba(139, 92, 246, 0.2);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }

    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: 4px;
        margin-bottom: 0;
    }

    .student-badge {
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.4);
        border-radius: 12px;
        padding: 8px 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .student-avatar {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, #8b5cf6, #3b82f6);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 0 12px rgba(139, 92, 246, 0.5);
    }

    .student-info-name {
        font-weight: 700;
        font-size: 0.95rem;
        color: #f8fafc;
        line-height: 1.2;
    }

    .student-info-roll {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #38bdf8;
    }

    /* Glass Cards */
    .nova-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 20px;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 16px;
    }

    .nova-card:hover {
        border-color: rgba(139, 92, 246, 0.4);
        box-shadow: 0 8px 24px -8px rgba(139, 92, 246, 0.25);
    }

    /* Metric KPI Cards */
    .metric-card-custom {
        background: linear-gradient(145deg, rgba(20, 28, 48, 0.85) 0%, rgba(12, 18, 33, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        transition: all 0.25s ease;
    }

    .metric-card-custom:hover {
        transform: translateY(-2px);
        border-color: rgba(139, 92, 246, 0.5);
        box-shadow: 0 10px 25px -10px rgba(139, 92, 246, 0.3);
    }

    .metric-card-custom::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #8b5cf6, #06b6d4);
        opacity: 0.8;
    }

    .metric-label-custom {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 6px;
    }

    .metric-value-custom {
        font-size: 1.85rem;
        font-weight: 800;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }

    .metric-subtitle-custom {
        font-size: 0.76rem;
        color: #38bdf8;
        margin-top: 6px;
    }

    /* Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 600;
    }

    .pill-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .pill-purple {
        background: rgba(139, 92, 246, 0.15);
        color: #c084fc;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }

    .pill-cyan {
        background: rgba(6, 182, 212, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }

    .pill-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Verified Answer Glass Card */
    .answer-card {
        background: linear-gradient(135deg, rgba(24, 24, 43, 0.9) 0%, rgba(13, 20, 36, 0.95) 100%);
        border: 1px solid rgba(139, 92, 246, 0.4);
        border-left: 5px solid #8b5cf6;
        border-radius: 12px;
        padding: 22px;
        margin: 16px 0;
        box-shadow: 0 12px 32px -8px rgba(0, 0, 0, 0.6), 0 0 18px -4px rgba(139, 92, 246, 0.2);
    }

    .answer-title {
        font-size: 0.84rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #a78bfa;
        font-weight: 700;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .answer-text {
        font-size: 1.28rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.45;
    }

    /* AI Explanation Card */
    .ai-card {
        background: linear-gradient(135deg, rgba(14, 26, 47, 0.85) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-left: 5px solid #06b6d4;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
    }

    .ai-title {
        font-size: 0.84rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #38bdf8;
        font-weight: 700;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .ai-text {
        font-size: 0.96rem;
        color: #e2e8f0;
        line-height: 1.6;
    }

    /* Landing Hero */
    .landing-hero {
        text-align: center;
        padding: 40px 20px 30px;
        max-width: 900px;
        margin: 0 auto;
    }

    .landing-title {
        font-size: 2.75rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 10%, #c4b5fd 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin-bottom: 12px;
    }

    .landing-desc {
        font-size: 1.15rem;
        color: #94a3b8;
        line-height: 1.6;
        margin-bottom: 28px;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 20px;
        margin: 32px 0;
    }

    .feature-box {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 24px;
        text-align: left;
        transition: all 0.25s ease;
    }

    .feature-box:hover {
        border-color: rgba(139, 92, 246, 0.4);
        transform: translateY(-3px);
        box-shadow: 0 12px 28px -10px rgba(139, 92, 246, 0.25);
    }

    .feature-icon {
        font-size: 1.75rem;
        margin-bottom: 12px;
    }

    .feature-head {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 8px;
    }

    .feature-body {
        font-size: 0.88rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    /* Custom Streamlit Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0px 0px;
        padding: 8px 18px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.92rem;
        border: none;
        background: transparent;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: rgba(139, 92, 246, 0.15) !important;
        border-bottom: 3px solid #8b5cf6 !important;
    }

    /* Footer */
    .footer-container {
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px 0 10px;
        margin-top: 48px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        color: #64748b;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "df" not in st.session_state:
    st.session_state["df"] = None

if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None

if "query_history" not in st.session_state:
    st.session_state["query_history"] = []

if "active_question" not in st.session_state:
    st.session_state["active_question"] = ""

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def is_identifier_column(dataframe: pd.DataFrame, column: str) -> bool:
    """Identify ID-like columns to exclude them from mathematical aggregations."""
    name = str(column).lower().strip()
    id_keywords = ["id", "uuid", "guid", "hash", "identifier", "code", "index"]

    name_based_id = (
        name in id_keywords
        or any(name.endswith("_" + word) or name.startswith(word + "_") for word in id_keywords)
    )

    unique_count = dataframe[column].nunique(dropna=True)
    unique_ratio = unique_count / max(len(dataframe), 1)
    high_cardinality = (unique_ratio > 0.85 and unique_count > 20)

    return name_based_id or high_cardinality

def get_color_sequence(theme_choice: str):
    """Return Plotly color sequences based on the selected theme."""
    palettes = {
        "Electric Neon (Violet/Cyan)": ["#8b5cf6", "#06b6d4", "#ec4899", "#3b82f6", "#10b981", "#f59e0b"],
        "Cyber Emerald (Mint/Teal)": ["#10b981", "#14b8a6", "#06b6d4", "#3b82f6", "#8b5cf6", "#a855f7"],
        "Sunset Flare (Rose/Amber)": ["#f43f5e", "#fb923c", "#f59e0b", "#e11d48", "#c084fc", "#38bdf8"],
        "Deep Cosmos (Indigo/Blue)": ["#6366f1", "#3b82f6", "#0ea5e9", "#2dd4bf", "#a855f7", "#ec4899"],
    }
    return palettes.get(theme_choice, palettes["Electric Neon (Violet/Cyan)"])

def calculate_health_score(dataframe: pd.DataFrame) -> int:
    """Calculate overall dataset data health score (0-100%)."""
    if dataframe.empty:
        return 0
    total_cells = dataframe.size
    missing_cells = dataframe.isna().sum().sum()
    duplicate_rows = dataframe.duplicated().sum()

    cell_completeness = max(0, 1 - (missing_cells / max(total_cells, 1)))
    row_uniqueness = max(0, 1 - (duplicate_rows / max(len(dataframe), 1)))

    score = int((cell_completeness * 0.7 + row_uniqueness * 0.3) * 100)
    return max(5, min(score, 100))

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    # Student / Developer Profile Badge
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(30, 27, 75, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
                border: 1px solid rgba(139, 92, 246, 0.4); border-radius: 14px; padding: 16px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
            <div style="width: 42px; height: 42px; border-radius: 10px; background: linear-gradient(135deg, #8b5cf6, #06b6d4);
                        display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.1rem; color: #fff;
                        box-shadow: 0 0 15px rgba(139, 92, 246, 0.5);">
                SL
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.05rem; color: #f8fafc; letter-spacing: -0.01em;">S Larens</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #38bdf8; font-weight: 600;">Roll: 23053071</div>
            </div>
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
            ✦ GenAI Capstone: AI Data Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🗄️ Data Management")

    uploaded = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"],
        help="Upload any structured tabular CSV dataset for instant AI analytics."
    )

    if uploaded is not None:
        try:
            st.session_state["df"] = pd.read_csv(uploaded)
            st.session_state["dataset_name"] = uploaded.name
        except Exception as exc:
            st.error(f"Error parsing uploaded CSV: {exc}")

    # One-click Sample Dataset loader button
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_sales.csv")
    if os.path.exists(sample_path):
        if st.button("⚡ Load Sample Sales Dataset", use_container_width=True):
            st.session_state["df"] = pd.read_csv(sample_path)
            st.session_state["dataset_name"] = "sample_sales.csv (Demo)"
            st.rerun()

    if st.session_state["df"] is not None:
        if st.button("🔄 Unload / Clear Dataset", use_container_width=True):
            st.session_state["df"] = None
            st.session_state["dataset_name"] = None
            st.session_state["active_question"] = ""
            st.rerun()

    st.divider()

    st.markdown("### 🎨 Visual Studio Theme")
    chart_theme = st.selectbox(
        "Palette Preset",
        [
            "Electric Neon (Violet/Cyan)",
            "Cyber Emerald (Mint/Teal)",
            "Sunset Flare (Rose/Amber)",
            "Deep Cosmos (Indigo/Blue)"
        ],
        index=0
    )
    theme_colors = get_color_sequence(chart_theme)

    st.divider()

    st.markdown("### ⚙️ Engine Telemetry")
    if llm_available():
        st.markdown("""
        <div class="status-pill pill-green" style="width: 100%; justify-content: center; margin-bottom: 8px;">
            ● Local FLAN-T5 GenAI Active
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-pill pill-purple" style="width: 100%; justify-content: center; margin-bottom: 8px;">
            ✦ Deterministic Analytical Engine Ready
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 0.78rem; color: #64748b; line-height: 1.4; margin-top: 10px;">
        Deterministic Pandas Core guarantees zero numerical hallucinations. All mathematical outputs are verified prior to natural language synthesis.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TOP HERO BANNER
# ============================================================
st.markdown("""
<div class="hero-container">
    <div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span class="status-pill pill-purple">✦ NOVA DATA LAB</span>
            <span class="status-pill pill-cyan">v2.5 Interactive Edition</span>
        </div>
        <h1 class="hero-title">NOVA · AI Data Analyst</h1>
        <p class="hero-subtitle">Conversational natural-language query routing, verified Pandas calculation, and deep visual intelligence.</p>
    </div>
    <div class="student-badge">
        <div class="student-avatar">SL</div>
        <div>
            <div class="student-info-name">S Larens</div>
            <div class="student-info-roll">Roll No: 23053071</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LANDING SCREEN (WHEN NO DATASET IS LOADED)
# ============================================================
if st.session_state["df"] is None:
    st.markdown("""
    <div class="landing-hero">
        <div class="landing-title">Empower Your Data with Grounded GenAI</div>
        <div class="landing-desc">
            Transform raw tabular datasets into interactive dashboards, deep statistical insights, and instant conversational answers without writing a single line of SQL or Python.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Call-to-action cards
    c_btn1, c_btn2 = st.columns([1, 1])
    with c_btn1:
        if st.button("⚡ Click Here to Launch Sample Sales Dataset", use_container_width=True, type="primary"):
            if os.path.exists(sample_path):
                st.session_state["df"] = pd.read_csv(sample_path)
                st.session_state["dataset_name"] = "sample_sales.csv (Demo)"
                st.rerun()

    with c_btn2:
        st.info("👈 Or drag & drop your own CSV file into the sidebar uploader.")

    st.markdown("""
    <div class="feature-grid">
        <div class="feature-box">
            <div class="feature-icon">🔍</div>
            <div class="feature-head">Deterministic Accuracy</div>
            <div class="feature-body">
                Numerical queries are calculated directly by Pandas before language model synthesis, preventing mathematical hallucinations.
            </div>
        </div>
        <div class="feature-box">
            <div class="feature-icon">💬</div>
            <div class="feature-head">Natural Language Querying</div>
            <div class="feature-body">
                Ask in plain English: "Average unit price in East", "Top 5 revenue products", or "What are the strongest correlations?".
            </div>
        </div>
        <div class="feature-box">
            <div class="feature-icon">📊</div>
            <div class="feature-head">Interactive Visual Studio</div>
            <div class="feature-body">
                Explore distributions, outlier box plots, density violins, 2D/3D scatter spaces, and correlation heatmaps.
            </div>
        </div>
        <div class="feature-box">
            <div class="feature-icon">🛡️</div>
            <div class="feature-head">Data Health Audit</div>
            <div class="feature-body">
                Automated profiling calculates data health scores, pinpoints missing cells, identifies duplicates, and flags outliers.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📐 View System Architecture & Verification Flow", expanded=True):
        st.markdown("""
        ```mermaid
        flowchart LR
            User([User Query / Upload]) --> UI[Streamlit Modern Frontend]
            UI --> Router[Natural Language Operation Router]
            Router --> Pandas[Authoritative Pandas Calculation Engine]
            Pandas --> Result[Deterministic Verified Result]
            Result --> LLM[Local FLAN-T5 GenAI Explainer]
            Result --> Plotly[Interactive Plotly Visual Studio]
            Result --> Table[Supporting Data Table]
            LLM --> FinalUI[Visual Dashboard Output]
            Plotly --> FinalUI
            Table --> FinalUI
        ```
        """, unsafe_allow_html=True)
        st.caption("Developed by S Larens | Roll No: 23053071")

    st.stop()

# ============================================================
# ACTIVE DATASET PROCESSING
# ============================================================
df = st.session_state["df"]

if df.empty:
    st.error("The uploaded dataset contains zero rows. Please upload a valid CSV file.")
    st.stop()

# Identify columns
all_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
all_categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

num_cols = [c for c in all_numeric_cols if not is_identifier_column(df, c)]
cat_cols = [c for c in all_categorical_cols if not is_identifier_column(df, c)]
id_cols = [c for c in df.columns if is_identifier_column(df, c)]

# Profiling and health
profile = profile_data(df)
missing_cells = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())
rows_with_missing = int(df.isna().any(axis=1).sum())
health_score = calculate_health_score(df)
memory_kb = df.memory_usage(deep=True).sum() / 1024

# ============================================================
# EXECUTIVE METRICS STRIP
# ============================================================
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

with kpi1:
    st.markdown(f"""
    <div class="metric-card-custom">
        <div class="metric-label-custom">Total Rows</div>
        <div class="metric-value-custom">{len(df):,}</div>
        <div class="metric-subtitle-custom">Records Loaded</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card-custom">
        <div class="metric-label-custom">Dimensions</div>
        <div class="metric-value-custom">{len(df.columns)}</div>
        <div class="metric-subtitle-custom">{len(num_cols)} Num · {len(cat_cols)} Cat</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    health_color = "#10b981" if health_score >= 85 else ("#f59e0b" if health_score >= 60 else "#ef4444")
    st.markdown(f"""
    <div class="metric-card-custom">
        <div class="metric-label-custom">Health Score</div>
        <div class="metric-value-custom" style="color: {health_color};">{health_score}%</div>
        <div class="metric-subtitle-custom">Automated Quality Index</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    miss_color = "#10b981" if missing_cells == 0 else "#f59e0b"
    st.markdown(f"""
    <div class="metric-card-custom">
        <div class="metric-label-custom">Missing Cells</div>
        <div class="metric-value-custom" style="color: {miss_color};">{missing_cells:,}</div>
        <div class="metric-subtitle-custom">{rows_with_missing} Rows Affected</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    dup_color = "#10b981" if duplicate_rows == 0 else "#f59e0b"
    st.markdown(f"""
    <div class="metric-card-custom">
        <div class="metric-label-custom">Duplicates</div>
        <div class="metric-value-custom" style="color: {dup_color};">{duplicate_rows:,}</div>
        <div class="metric-subtitle-custom">Redundant Rows</div>
    </div>
    """, unsafe_allow_html=True)

with kpi6:
    st.markdown(f"""
    <div class="metric-card-custom">
        <div class="metric-label-custom">Memory Usage</div>
        <div class="metric-value-custom">{memory_kb:.1f} <span style="font-size: 1rem; color: #94a3b8;">KB</span></div>
        <div class="metric-subtitle-custom">{st.session_state.get('dataset_name', 'Active Dataset')}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ============================================================
# PRIMARY INTERACTIVE TABS
# ============================================================
tabs = st.tabs([
    "📊 Executive Overview",
    "🤖 Ask NOVA (AI Analyst)",
    "📈 Visual Analytics Studio",
    "💡 Automated Insights",
    "🛡️ Data Quality & Audit",
    "📄 Capstone Report & Export"
])

# ============================================================
# TAB 1: EXECUTIVE OVERVIEW & DATA DNA
# ============================================================
with tabs[0]:
    st.markdown("### 📊 Interactive Dataset Explorer")
    
    col_left, col_right = st.columns([1.65, 1])

    with col_left:
        # Search & Column Filter Controls
        ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1, 0.8])
        with ctrl1:
            search_query = st.text_input("🔍 Quick Search Rows", placeholder="Filter by keyword...", key="overview_search")
        with ctrl2:
            selected_preview_cols = st.multiselect("Visible Columns", options=df.columns.tolist(), default=df.columns.tolist()[:8], key="overview_cols")
        with ctrl3:
            row_limit = st.selectbox("Rows to Display", [10, 25, 50, 100, "All"], index=1, key="overview_limit")

        # Filter dataframe
        filtered_df = df.copy()
        if search_query:
            mask = filtered_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False, na=False)).any(axis=1)
            filtered_df = filtered_df[mask]

        display_cols = selected_preview_cols if selected_preview_cols else df.columns.tolist()
        num_rows = len(filtered_df) if row_limit == "All" else int(row_limit)

        st.dataframe(
            filtered_df[display_cols].head(num_rows),
            use_container_width=True,
            height=380
        )

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            csv_data = filtered_df[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Filtered View (CSV)",
                data=csv_data,
                file_name=f"exported_{st.session_state.get('dataset_name', 'data')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d_col2:
            st.caption(f"Displaying {min(num_rows, len(filtered_df))} of {len(filtered_df)} matched rows.")

    with col_right:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px;">
            <div style="font-weight: 700; font-size: 1rem; color: #f8fafc; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                🧬 Dataset Fingerprint & Architecture
            </div>
        """, unsafe_allow_html=True)

        fig_breakdown = go.Figure(data=[go.Pie(
            labels=["Analytical Numeric", "Analytical Categorical", "Identifiers/Other"],
            values=[len(num_cols), len(cat_cols), len(id_cols)],
            hole=.6,
            marker=dict(colors=[theme_colors[0], theme_colors[1], "#475569"])
        )])
        fig_breakdown.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", size=11)
        )
        st.plotly_chart(fig_breakdown, use_container_width=True)

        st.markdown(f"""
            <div style="font-size: 0.84rem; color: #94a3b8; line-height: 1.5; margin-top: 8px;">
                • <b>File:</b> {st.session_state.get('dataset_name', 'Uploaded File')}<br>
                • <b>Integrity:</b> {'All rows complete' if missing_cells == 0 else f'{missing_cells} empty cells require attention'}<br>
                • <b>Cardinality:</b> {len(cat_cols)} groupable dimensions detected
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Column Intelligence / DNA Table
    st.markdown("### 🧬 Column Intelligence & Schema Profiler")
    
    col_search = st.text_input("Filter Columns in Profiler", placeholder="Search column name or data type...", key="profile_search")
    
    filtered_profile = profile.copy()
    if col_search:
        p_mask = filtered_profile.astype(str).apply(lambda row: row.str.contains(col_search, case=False, na=False)).any(axis=1)
        filtered_profile = filtered_profile[p_mask]

    # Add missing percentage column
    filtered_profile["Missing %"] = (filtered_profile["Missing"] / len(df) * 100).round(1).astype(str) + "%"

    st.dataframe(
        filtered_profile,
        use_container_width=True,
        height=300
    )

# ============================================================
# TAB 2: ASK NOVA (CONVERSATIONAL AI ANALYST)
# ============================================================
with tabs[1]:
    st.markdown("### 🤖 Ask NOVA Anything in Plain English")
    st.caption("Ask analytical questions. NOVA maps natural language to verified Pandas calculations, generates visual proof, and provides grounded AI explanations.")

    # Generate dynamic suggestion chips based on columns
    suggested_queries = []
    if num_cols:
        primary_metric = num_cols[0]
        suggested_queries.append(f"What is the average {primary_metric}?")
        suggested_queries.append(f"What is the total {primary_metric}?")
        suggested_queries.append(f"Top 5 {primary_metric} records")
    
    if cat_cols and num_cols:
        primary_cat = cat_cols[0]
        primary_metric = num_cols[0]
        suggested_queries.append(f"What is the average {primary_metric} in {primary_cat}?")
        suggested_queries.append(f"Total {primary_metric} by {primary_cat}")

    if len(num_cols) >= 2:
        suggested_queries.append("What are the strongest correlations?")
    
    suggested_queries.append("How many missing values are there?")
    suggested_queries.append("How many duplicate rows are there?")

    # Suggestion Chips UI
    st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #94a3b8; margin-bottom: 8px;'>💡 QUICK QUERY SUGGESTIONS:</div>", unsafe_allow_html=True)
    chip_cols = st.columns(min(4, len(suggested_queries[:4])))
    for i, s_query in enumerate(suggested_queries[:4]):
        with chip_cols[i % len(chip_cols)]:
            if st.button(f"▸ {s_query}", key=f"chip_{i}", use_container_width=True):
                st.session_state["active_question"] = s_query

    # Query Input Box
    query_col, action_col = st.columns([4, 1])
    with query_col:
        user_question = st.text_input(
            "Natural Language Query",
            value=st.session_state.get("active_question", ""),
            placeholder="e.g., What is the average unit price in East? or Top 5 revenue records",
            label_visibility="collapsed",
            key="nl_query_input"
        )
    with action_col:
        execute_button = st.button("🚀 Analyze Query", type="primary", use_container_width=True)

    query_to_run = user_question.strip() if execute_button or user_question else ""

    if query_to_run:
        if query_to_run not in st.session_state["query_history"]:
            st.session_state["query_history"].insert(0, query_to_run)
            if len(st.session_state["query_history"]) > 8:
                st.session_state["query_history"].pop()

        with st.spinner("⚡ Parsing question with Natural Language Router & Computing Pandas result..."):
            result = analyze_question(df, query_to_run)

        # ----------------------------------------------------
        # VERIFIED ANSWER CARD
        # ----------------------------------------------------
        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-title">
                <span>✓ VERIFIED CALCULATION RESULT</span>
                <span class="status-pill pill-green">authoritative</span>
                <span class="status-pill pill-purple">operation: {result.get('operation', 'pandas.compute')}</span>
            </div>
            <div class="answer-text">
                {result.get('answer', 'No calculation could be resolved.')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ----------------------------------------------------
        # VISUAL & TABULAR EVIDENCE
        # ----------------------------------------------------
        has_chart = result.get("chart") is not None
        if has_chart or has_table:
            st.markdown("#### 📈 Visual & Empirical Evidence")
            if has_chart and has_table:
                ev_left, ev_right = st.columns([1.2, 1])
                with ev_left:
                    chart_fig = result["chart"]
                    chart_fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15,23,42,0.6)",
                        font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
                        margin=dict(l=30, r=30, t=50, b=30)
                    )
                    st.plotly_chart(chart_fig, use_container_width=True)
                with ev_right:
                    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #94a3b8; margin-bottom: 6px;'>Supporting Breakdown:</div>", unsafe_allow_html=True)
                    st.dataframe(result["table"], use_container_width=True, height=320)
            elif has_chart:
                chart_fig = result["chart"]
                chart_fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
                    margin=dict(l=30, r=30, t=50, b=30)
                )
                st.plotly_chart(chart_fig, use_container_width=True)
            elif has_table:
                st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #94a3b8; margin-bottom: 6px;'>Supporting Breakdown:</div>", unsafe_allow_html=True)
                st.dataframe(result["table"], use_container_width=True, height=320)

        # ----------------------------------------------------
        # AI EXPLANATION
        # ----------------------------------------------------
        with st.spinner("🤖 Synthesizing AI explanation with local model..."):
            explanation = explain_with_local_llm(
                query_to_run,
                result.get("answer", ""),
                result.get("details", "")
            )

        st.markdown(f"""
        <div class="ai-card">
            <div class="ai-title">
                <span>🤖 AI REASONING & EXPLANATION</span>
                <span class="status-pill pill-cyan">{'FLAN-T5-small' if llm_available() else 'Deterministic Safe AI'}</span>
            </div>
            <div class="ai-text">
                {explanation}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ----------------------------------------------------
        # EXECUTION TELEMETRY & TRACE
        # ----------------------------------------------------
        with st.expander("🔍 Inspect Execution Trace & Telemetry"):
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("**Routed Operation:**")
                st.code(result.get("operation", "unknown"), language="text")
            with t_col2:
                st.markdown("**Mathematical Operation Details:**")
                st.code(result.get("details", "N/A"), language="text")

    # Query History Bar
    if st.session_state["query_history"]:
        with st.expander("🕒 Recent Query History", expanded=False):
            for h_query in st.session_state["query_history"]:
                if st.button(f"↺ {h_query}", key=f"hist_{h_query}", use_container_width=True):
                    st.session_state["active_question"] = h_query
                    st.rerun()

# ============================================================
# TAB 3: VISUAL ANALYTICS STUDIO
# ============================================================
with tabs[2]:
    st.markdown("### 📈 Interactive Visual Analytics Studio")
    st.caption("Generate publication-ready interactive charts with dynamic parameters, category groupings, and hyper-dimensional projections.")

    if not num_cols and not cat_cols:
        st.warning("No analytical columns available to visualize.")
    else:
        vis_mode = st.selectbox(
            "Select Chart Mode",
            [
                "Distribution & Histogram",
                "Box Plot & Outlier Range",
                "Violin Density Plot",
                "2D Interactive Scatter Studio",
                "3D Hyper-Dimensional Scatter",
                "Category Breakdown Studio",
                "Composition & Donut Chart",
                "Correlation Heatmap Matrix"
            ]
        )

        st.write("")

        # 1. DISTRIBUTION
        if vis_mode == "Distribution & Histogram":
            if not num_cols:
                st.warning("No numeric columns available.")
            else:
                p1, p2, p3 = st.columns([1.2, 1, 1])
                with p1:
                    target_col = st.selectbox("Numeric Column", num_cols, key="dist_num")
                with p2:
                    bins = st.slider("Bin Count", min_value=5, max_value=60, value=25, key="dist_bins")
                with p3:
                    color_by = st.selectbox("Group by Category (Optional)", [None] + cat_cols, key="dist_cat")

                fig = px.histogram(
                    df,
                    x=target_col,
                    nbins=bins,
                    color=color_by,
                    marginal="box",
                    title=f"Distribution Analysis · {target_col}",
                    color_discrete_sequence=theme_colors,
                    template="plotly_dark"
                )
                fig.update_layout(
                    height=560,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    font=dict(family="Plus Jakarta Sans", color="#cbd5e1")
                )
                st.plotly_chart(fig, use_container_width=True)

        # 2. BOX PLOT
        elif vis_mode == "Box Plot & Outlier Range":
            if not num_cols:
                st.warning("No numeric columns available.")
            else:
                p1, p2, p3 = st.columns([1.2, 1, 1])
                with p1:
                    box_y = st.selectbox("Value Dimension (Y-Axis)", num_cols, key="box_y")
                with p2:
                    box_x = st.selectbox("Group by Category (X-Axis)", [None] + cat_cols, key="box_x")
                with p3:
                    show_pts = st.selectbox("Points Display", ["outliers", "all", False], key="box_pts")

                fig = px.box(
                    df,
                    x=box_x,
                    y=box_y,
                    points=show_pts,
                    color=box_x if box_x else None,
                    title=f"Box Plot & Outlier Spread · {box_y}",
                    color_discrete_sequence=theme_colors,
                    template="plotly_dark"
                )
                fig.update_layout(
                    height=560,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    font=dict(family="Plus Jakarta Sans", color="#cbd5e1")
                )
                st.plotly_chart(fig, use_container_width=True)

        # 3. VIOLIN PLOT
        elif vis_mode == "Violin Density Plot":
            if not num_cols:
                st.warning("No numeric columns available.")
            else:
                p1, p2 = st.columns(2)
                with p1:
                    viol_y = st.selectbox("Value Dimension", num_cols, key="viol_y")
                with p2:
                    viol_x = st.selectbox("Category Split", [None] + cat_cols, key="viol_x")

                fig = px.violin(
                    df,
                    x=viol_x,
                    y=viol_y,
                    box=True,
                    points="outliers",
                    color=viol_x if viol_x else None,
                    title=f"Density Violin Plot · {viol_y}",
                    color_discrete_sequence=theme_colors,
                    template="plotly_dark"
                )
                fig.update_layout(
                    height=560,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    font=dict(family="Plus Jakarta Sans", color="#cbd5e1")
                )
                st.plotly_chart(fig, use_container_width=True)

        # 4. 2D SCATTER
        elif vis_mode == "2D Interactive Scatter Studio":
            if len(num_cols) < 2:
                st.warning("At least two numeric columns required for 2D scatter.")
            else:
                p1, p2, p3, p4 = st.columns(4)
                with p1:
                    scat_x = st.selectbox("X-Axis", num_cols, index=0, key="scat_x")
                with p2:
                    scat_y = st.selectbox("Y-Axis", num_cols, index=min(1, len(num_cols)-1), key="scat_y")
                with p3:
                    scat_color = st.selectbox("Color Category", [None] + cat_cols, key="scat_col")
                with p4:
                    trend = st.checkbox("Fit OLS Trendline", value=len(df) >= 3, key="scat_trend")

                try:
                    fig = px.scatter(
                        df,
                        x=scat_x,
                        y=scat_y,
                        color=scat_color,
                        trendline="ols" if trend else None,
                        title=f"Bivariate Scatter: {scat_y} vs {scat_x}",
                        color_discrete_sequence=theme_colors,
                        template="plotly_dark"
                    )
                except Exception:
                    fig = px.scatter(
                        df,
                        x=scat_x,
                        y=scat_y,
                        color=scat_color,
                        title=f"Bivariate Scatter: {scat_y} vs {scat_x}",
                        color_discrete_sequence=theme_colors,
                        template="plotly_dark"
                    )

                fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=1, color="rgba(255,255,255,0.3)")))
                fig.update_layout(
                    height=600,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    font=dict(family="Plus Jakarta Sans", color="#cbd5e1")
                )
                st.plotly_chart(fig, use_container_width=True)

        # 5. 3D SCATTER
        elif vis_mode == "3D Hyper-Dimensional Scatter":
            if len(num_cols) < 3:
                st.warning("At least 3 numeric dimensions required for 3D scatter visualization.")
            else:
                p1, p2, p3, p4 = st.columns(4)
                with p1:
                    x_3d = st.selectbox("X Dimension", num_cols, index=0, key="3d_x")
                with p2:
                    y_3d = st.selectbox("Y Dimension", num_cols, index=1, key="3d_y")
                with p3:
                    z_3d = st.selectbox("Z Dimension", num_cols, index=2, key="3d_z")
                with p4:
                    col_3d = st.selectbox("Cluster Color", [None] + cat_cols, key="3d_color")

                fig = px.scatter_3d(
                    df,
                    x=x_3d,
                    y=y_3d,
                    z=z_3d,
                    color=col_3d,
                    title=f"3D Space: {x_3d} × {y_3d} × {z_3d}",
                    color_discrete_sequence=theme_colors,
                    template="plotly_dark"
                )
                fig.update_traces(marker=dict(size=6, opacity=0.85))
                fig.update_layout(
                    height=680,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

        # 6. CATEGORY BREAKDOWN
        elif vis_mode == "Category Breakdown Studio":
            if not cat_cols:
                st.warning("No categorical columns available.")
            else:
                p1, p2, p3 = st.columns(3)
                with p1:
                    cat_target = st.selectbox("Category Dimension", cat_cols, key="cat_break_target")
                with p2:
                    agg_metric = st.selectbox("Metric Column", ["Record Count"] + num_cols, key="cat_break_metric")
                with p3:
                    top_n = st.slider("Top N Categories", min_value=3, max_value=25, value=10, key="cat_top_n")

                if agg_metric == "Record Count":
                    plot_data = df[cat_target].astype(str).value_counts().head(top_n).reset_index()
                    plot_data.columns = [cat_target, "Count"]
                    fig = px.bar(
                        plot_data,
                        x=cat_target,
                        y="Count",
                        color=cat_target,
                        title=f"Record Frequency by {cat_target}",
                        color_discrete_sequence=theme_colors,
                        template="plotly_dark"
                    )
                else:
                    plot_data = df.groupby(cat_target)[agg_metric].sum().sort_values(ascending=False).head(top_n).reset_index()
                    fig = px.bar(
                        plot_data,
                        x=cat_target,
                        y=agg_metric,
                        color=cat_target,
                        title=f"Total {agg_metric} by {cat_target}",
                        color_discrete_sequence=theme_colors,
                        template="plotly_dark"
                    )

                fig.update_layout(
                    height=580,
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    font=dict(family="Plus Jakarta Sans", color="#cbd5e1")
                )
                st.plotly_chart(fig, use_container_width=True)

        # 7. DONUT / PIE
        elif vis_mode == "Composition & Donut Chart":
            if not cat_cols:
                st.warning("No categorical columns available.")
            else:
                p1, p2 = st.columns(2)
                with p1:
                    pie_cat = st.selectbox("Category", cat_cols, key="donut_cat")
                with p2:
                    pie_val = st.selectbox("Weight by Metric", ["Equal Weights (Count)"] + num_cols, key="donut_val")

                if pie_val == "Equal Weights (Count)":
                    plot_data = df[pie_cat].astype(str).value_counts().head(10).reset_index()
                    plot_data.columns = [pie_cat, "Count"]
                    fig = px.pie(
                        plot_data,
                        names=pie_cat,
                        values="Count",
                        hole=0.55,
                        title=f"Proportional Composition · {pie_cat}",
                        color_discrete_sequence=theme_colors,
                        template="plotly_dark"
                    )
                else:
                    plot_data = df.groupby(pie_cat)[pie_val].sum().reset_index()
                    fig = px.pie(
                        plot_data,
                        names=pie_cat,
                        values=pie_val,
                        hole=0.55,
                        title=f"Share of {pie_val} by {pie_cat}",
                        color_discrete_sequence=theme_colors,
                        template="plotly_dark"
                    )

                fig.update_layout(
                    height=580,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans", color="#cbd5e1")
                )
                st.plotly_chart(fig, use_container_width=True)

        # 8. CORRELATION HEATMAP
        elif vis_mode == "Correlation Heatmap Matrix":
            if len(num_cols) < 2:
                st.warning("At least two numeric dimensions required for correlation analysis.")
            else:
                corr_matrix = df[num_cols].corr()

                fig = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    title="Pearson Correlation Coefficient Matrix",
                    color_continuous_scale="Purples",
                    zmin=-1,
                    zmax=1,
                    template="plotly_dark"
                )
                fig.update_layout(
                    height=600,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans", color="#cbd5e1")
                )
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4: AUTOMATED INSIGHTS ENGINE
# ============================================================
with tabs[3]:
    st.markdown("### 💡 Automated Intelligence & Pattern Detection")
    st.caption("Heuristic reasoning engine scans dataset distributions, identifies skewness, spots category dominance, and detects high-impact anomalies.")

    insights_list = auto_insights(df)

    if not insights_list:
        st.info("No anomalous statistical patterns detected in this dataset.")
    else:
        for idx, (ins_title, ins_body) in enumerate(insights_list):
            st.markdown(f"""
            <div class="nova-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; display: flex; align-items: center; gap: 8px;">
                        <span>✦</span> {ins_title}
                    </div>
                    <span class="status-pill pill-purple">Automated Finding #{idx+1}</span>
                </div>
                <div style="color: #cbd5e1; font-size: 0.94rem; line-height: 1.55;">
                    {ins_body}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 5: DATA QUALITY & AUDIT
# ============================================================
with tabs[4]:
    st.markdown("### 🛡️ Data Quality & Hygiene Assessment")
    st.caption("Deep integrity check identifying missing records, redundant duplicates, and column completeness.")

    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        st.markdown(f"""
        <div class="metric-card-custom">
            <div class="metric-label-custom">Overall Quality Index</div>
            <div class="metric-value-custom" style="color: {'#10b981' if health_score >= 80 else '#f59e0b'};">{health_score} / 100</div>
            <div class="metric-subtitle-custom">Composite Integrity Rating</div>
        </div>
        """, unsafe_allow_html=True)
    with q_col2:
        st.markdown(f"""
        <div class="metric-card-custom">
            <div class="metric-label-custom">Cell Completeness</div>
            <div class="metric-value-custom">{100 - (missing_cells / max(df.size, 1) * 100):.1f}%</div>
            <div class="metric-subtitle-custom">{missing_cells} Empty Cells Detected</div>
        </div>
        """, unsafe_allow_html=True)
    with q_col3:
        st.markdown(f"""
        <div class="metric-card-custom">
            <div class="metric-label-custom">Uniqueness Rate</div>
            <div class="metric-value-custom">{100 - (duplicate_rows / max(len(df), 1) * 100):.1f}%</div>
            <div class="metric-subtitle-custom">{duplicate_rows} Duplicate Rows Found</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Visual Missingness Breakdown
    missing_by_col = df.isna().sum().reset_index()
    missing_by_col.columns = ["Column", "Missing Count"]
    missing_by_col = missing_by_col[missing_by_col["Missing Count"] > 0]

    if not missing_by_col.empty:
        st.markdown("#### ⚠️ Missing Values by Column")
        fig_miss = px.bar(
            missing_by_col,
            x="Column",
            y="Missing Count",
            title="Null Cells Per Dimension",
            color_discrete_sequence=["#f43f5e"],
            template="plotly_dark"
        )
        fig_miss.update_layout(
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(family="Plus Jakarta Sans", color="#cbd5e1")
        )
        st.plotly_chart(fig_miss, use_container_width=True)
    else:
        st.success("✅ Perfect Data Completeness: Zero missing cells detected across all columns.")

    if duplicate_rows > 0:
        st.warning(f"⚠️ {duplicate_rows} redundant duplicate rows were detected in the dataset.")
        with st.expander("Inspect Duplicate Records"):
            st.dataframe(df[df.duplicated(keep=False)], use_container_width=True)
    else:
        st.success("✅ Zero Duplicate Records: Every row in this dataset is unique.")

# ============================================================
# TAB 6: CAPSTONE REPORT & EXPORT
# ============================================================
with tabs[5]:
    st.markdown("### 📄 Capstone Project Dossier & Executive Summary")
    st.caption("Official project metadata, evaluation credentials, and automated executive export.")

    # Student Credentials Card
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(30, 27, 75, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
                border: 1px solid rgba(139, 92, 246, 0.4); border-radius: 14px; padding: 24px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="font-size: 0.82rem; text-transform: uppercase; color: #a78bfa; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 4px;">
                    PROJECT DEVELOPER & CANDIDATE
                </div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">S Larens</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; color: #38bdf8; font-weight: 600; margin-top: 4px;">
                    Roll Number: 23053071
                </div>
            </div>
            <div style="text-align: right;">
                <span class="status-pill pill-green" style="font-size: 0.85rem; padding: 6px 16px;">Verified Submission</span>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 8px;">
                    GenAI Capstone: NOVA AI Data Analyst
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Executive Summary Report Generator
    report_content = f"""# Executive Analytical Report — NOVA AI Data Analyst
**Developer:** S Larens  
**Roll No:** 23053071  
**Generated On:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Dataset:** {st.session_state.get('dataset_name', 'Active Dataset')}

---

## 1. Dataset Dimensions & Architecture
- **Total Records:** {len(df):,}
- **Total Features:** {len(df.columns)}
- **Analytical Numerical Dimensions ({len(num_cols)}):** {', '.join(num_cols) if num_cols else 'None'}
- **Analytical Categorical Dimensions ({len(cat_cols)}):** {', '.join(cat_cols) if cat_cols else 'None'}
- **Data Health Score:** {health_score}/100
- **Missing Cells:** {missing_cells}
- **Duplicate Rows:** {duplicate_rows}

---

## 2. Key Statistical Insights
"""
    for title, desc in insights_list:
        report_content += f"- **{title}:** {desc}\n"

    report_content += f"""
---

## 3. Technology Architecture
- **Execution Engine:** Authoritative Pandas DataFrame Processor (Deterministic)
- **Conversational Interface:** Natural Language Pattern Router
- **GenAI Explainer:** Local FLAN-T5 Model (`google/flan-t5-small`) with Safe Fallback
- **Visualization:** Plotly Express Interactive Studio
- **Platform:** Streamlit Python Web Framework
"""

    st.markdown("#### 📋 Executive Dataset Summary Preview")
    st.markdown(report_content)

    st.download_button(
        "📥 Download Executive Markdown Report (.md)",
        data=report_content.encode("utf-8"),
        file_name=f"NOVA_Report_S_Larens_23053071.md",
        mime="text/markdown",
        use_container_width=True,
        type="primary"
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer-container">
    <div>
        ✦ <b>NOVA</b> · AI Data Intelligence Studio · Developed by <b>S Larens</b> (Roll No: <b>23053071</b>)
    </div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">
        Deterministic Pandas Core · Plotly Express · Local GenAI
    </div>
</div>
""", unsafe_allow_html=True)
