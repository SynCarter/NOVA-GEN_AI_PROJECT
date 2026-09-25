import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NOVA | AI Data Analyst",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# APP TITLE
# ============================================================

st.markdown("# ✦ NOVA · AI DATA INTELLIGENCE")

st.title("AI Data Analyst")

st.write(
    "Upload a dataset. Ask questions in natural language. "
    "Get calculation-backed answers, visual evidence, "
    "interactive analytics and AI-assisted explanations."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Workspace")

    uploaded = st.file_uploader(
        "Upload CSV dataset",
        type=["csv"]
    )

    st.divider()

    st.subheader("Analysis modes")

    st.write("⌕ **Ask the Data**")
    st.write("◈ **Explore**")
    st.write("◌ **Insights**")
    st.write("⌁ **Data Quality**")

    st.divider()

    if llm_available():
        st.success("Local GenAI model available")
    else:
        st.info(
            "Local model will be loaded when an "
            "AI explanation is requested."
        )

    st.caption(
        "Free stack · Python · Pandas · Plotly · Streamlit"
    )


# ============================================================
# LANDING SCREEN
# ============================================================

if uploaded is None:

    st.info(
        "Upload a CSV from the sidebar to begin analysis."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("01 · Understand")
        st.write(
            "Automatically inspect rows, columns, data types, "
            "missing values, duplicates and dataset structure."
        )

    with c2:
        st.subheader("02 · Ask")
        st.write(
            "Ask natural-language questions about averages, "
            "totals, rankings, groups, filters and correlations."
        )

    with c3:
        st.subheader("03 · Visualize")
        st.write(
            "Explore distributions, relationships, categories, "
            "correlations and interactive 3D visualizations."
        )

    st.divider()

    st.subheader("Designed for arbitrary CSV datasets")

    st.write(
        "NOVA adapts its analytical suggestions and visualizations "
        "to the columns in the uploaded dataset. Identifier-like "
        "columns such as order_id and transaction_id are excluded "
        "from analytical charts."
    )

    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================

try:

    df = pd.read_csv(uploaded)

except Exception as exc:

    st.error(
        f"Could not read the uploaded CSV: {exc}"
    )

    st.stop()


# ============================================================
# EMPTY DATASET CHECK
# ============================================================

if df.empty:

    st.error(
        "The uploaded CSV contains no rows."
    )

    st.stop()


# ============================================================
# SMART IDENTIFIER DETECTION
# ============================================================

def is_identifier_column(dataframe, column):
    """
    Detect columns that are likely identifiers rather than
    useful analytical variables.

    Examples:
        order_id
        transaction_id
        customer_id
        user_id
        uuid
        guid
    """

    name = str(column).lower().strip()

    id_keywords = [
        "id",
        "uuid",
        "guid",
        "hash",
        "identifier"
    ]

    # Explicit identifier naming
    name_based_id = (
        name in id_keywords
        or any(
            name.endswith("_" + word)
            or name.startswith(word + "_")
            for word in id_keywords
        )
    )

    unique_count = dataframe[column].nunique(
        dropna=True
    )

    unique_ratio = (
        unique_count /
        max(len(dataframe), 1)
    )

    # A high-cardinality field where nearly every row
    # has a unique value is usually an identifier.
    high_cardinality = (
        unique_ratio > 0.80
        and unique_count > 20
    )

    return (
        name_based_id
        or high_cardinality
    )


# ============================================================
# RAW COLUMN TYPES
# ============================================================

all_numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

all_categorical_cols = df.select_dtypes(
    exclude=np.number
).columns.tolist()


# ============================================================
# ANALYTICAL NUMERIC COLUMNS
# ============================================================

num_cols = [
    column
    for column in all_numeric_cols
    if not is_identifier_column(
        df,
        column
    )
]


# ============================================================
# ANALYTICAL CATEGORICAL COLUMNS
# ============================================================

cat_cols = [
    column
    for column in all_categorical_cols
    if not is_identifier_column(
        df,
        column
    )
]


# ============================================================
# DATA QUALITY
# ============================================================

profile = profile_data(df)

missing = int(
    df.isna().sum().sum()
)

duplicates = int(
    df.duplicated().sum()
)

rows_with_missing = int(
    df.isna().any(axis=1).sum()
)


# ============================================================
# DATASET HEADER
# ============================================================

st.subheader("Dataset Overview")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Rows",
        f"{len(df):,}"
    )

with m2:
    st.metric(
        "Columns",
        f"{len(df.columns):,}"
    )

with m3:
    st.metric(
        "Missing cells",
        f"{missing:,}"
    )

with m4:
    st.metric(
        "Duplicate rows",
        f"{duplicates:,}"
    )


# ============================================================
# DATASET INFO
# ============================================================

with st.expander(
    "Dataset information",
    expanded=False
):

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write(
            f"**Analytical numeric fields:** {len(num_cols)}"
        )

    with c2:
        st.write(
            f"**Analytical categorical fields:** {len(cat_cols)}"
        )

    with c3:
        memory_kb = (
            df.memory_usage(
                deep=True
            ).sum() / 1024
        )

        st.write(
            f"**Memory:** {memory_kb:.1f} KB"
        )


# ============================================================
# MAIN TABS
# ============================================================

tabs = st.tabs(
    [
        "✦ Overview",
        "⌕ Ask the Data",
        "◈ Explore",
        "◌ Insights",
        "⌁ Data Quality"
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tabs[0]:

    st.header("Dataset Overview")

    left, right = st.columns(
        [1.6, 1]
    )

    with left:

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(15),
            use_container_width=True,
            height=400
        )

    with right:

        st.subheader("Dataset Fingerprint")

        st.metric(
            "Rows × Columns",
            f"{len(df):,} × {len(df.columns)}"
        )

        st.metric(
            "Analytical Numeric Fields",
            len(num_cols)
        )

        st.metric(
            "Analytical Categorical Fields",
            len(cat_cols)
        )

        if missing == 0:
            st.success(
                "No missing cells detected."
            )
        else:
            st.warning(
                f"{missing:,} missing cells detected."
            )

        if duplicates == 0:
            st.success(
                "No duplicate rows detected."
            )
        else:
            st.warning(
                f"{duplicates:,} duplicate rows detected."
            )

    st.divider()

    st.subheader("Column Intelligence")

    st.dataframe(
        profile,
        use_container_width=True,
        height=350
    )


# ============================================================
# TAB 2 — ASK THE DATA
# ============================================================

with tabs[1]:

    st.header(
        "Ask Questions in Plain English"
    )

    st.write(
        "NOVA converts your question into a structured "
        "analytical operation, executes it with Pandas, "
        "and then explains the verified result."
    )

    # --------------------------------------------------------
    # DYNAMIC QUESTION GENERATION
    # --------------------------------------------------------

    examples = []

    if num_cols:

        metric = num_cols[0]

        examples.append(
            f"What is the average {metric}?"
        )

        examples.append(
            f"What is the total {metric}?"
        )

        examples.append(
            f"What is the maximum {metric}?"
        )

    if cat_cols and num_cols:

        category = cat_cols[0]
        metric = num_cols[0]

        examples.append(
            f"What is the average {metric} "
            f"for each {category}?"
        )

    if cat_cols:

        category = cat_cols[0]

        examples.append(
            f"How many records are in each {category}?"
        )

    if len(num_cols) >= 2:

        examples.append(
            "What are the strongest correlations?"
        )

    examples = examples[:5]

    # --------------------------------------------------------
    # QUESTION INPUT
    # --------------------------------------------------------

    question = st.text_input(
        "Your question",
        placeholder=(
            "Example: What is the average amount?"
        ),
        key="question_input"
    )

    # --------------------------------------------------------
    # SUGGESTED QUESTIONS
    # --------------------------------------------------------

    if examples:

        st.subheader(
            "Suggested questions"
        )

        suggestion_columns = st.columns(
            min(4, len(examples))
        )

        for i, example in enumerate(
            examples
        ):

            with suggestion_columns[
                i % len(suggestion_columns)
            ]:

                if st.button(
                    example,
                    key=f"suggestion_{i}",
                    use_container_width=True
                ):

                    question = example

    # --------------------------------------------------------
    # RUN ANALYSIS
    # --------------------------------------------------------

    if question:

        with st.spinner(
            "Analyzing dataset..."
        ):

            result = analyze_question(
                df,
                question
            )

        st.divider()

        st.subheader(
            "Verified Answer"
        )

        st.success(
            result.get(
                "answer",
                "No answer generated."
            )
        )

        # ----------------------------------------------------
        # SUPPORTING TABLE
        # ----------------------------------------------------

        if result.get("table") is not None:

            st.subheader(
                "Supporting Evidence"
            )

            st.dataframe(
                result["table"],
                use_container_width=True
            )

        # ----------------------------------------------------
        # SUPPORTING CHART
        # ----------------------------------------------------

        if result.get("chart") is not None:

            st.subheader(
                "Visual Evidence"
            )

            st.plotly_chart(
                result["chart"],
                use_container_width=True
            )

        # ----------------------------------------------------
        # AI EXPLANATION
        # ----------------------------------------------------

        st.subheader(
            "AI Explanation"
        )

        with st.spinner(
            "Generating explanation..."
        ):

            try:

                explanation = (
                    explain_with_local_llm(
                        question,
                        result.get(
                            "answer",
                            ""
                        ),
                        result.get(
                            "details",
                            ""
                        )
                    )
                )

                st.info(
                    explanation
                )

            except Exception as exc:

                st.warning(
                    "The verified analytical result is "
                    "available, but the local AI explanation "
                    f"could not be generated: {exc}"
                )

        # ----------------------------------------------------
        # ANALYSIS TRACE
        # ----------------------------------------------------

        with st.expander(
            "Analysis Trace"
        ):

            st.code(
                result.get(
                    "operation",
                    "unknown"
                ),
                language="text"
            )

            st.write(
                result.get(
                    "details",
                    ""
                )
            )


# ============================================================
# TAB 3 — EXPLORE
# ============================================================

with tabs[2]:

    st.header(
        "Interactive Visual Analytics"
    )

    st.write(
        "Choose a visualization and explore the analytical "
        "columns in the uploaded dataset."
    )

    if not num_cols:

        st.warning(
            "No suitable analytical numeric columns "
            "were detected."
        )

        st.info(
            "Identifier columns such as order_id and "
            "transaction_id are intentionally excluded."
        )

    else:

        visualization = st.selectbox(
            "Visualization",
            [
                "Distribution",
                "Box Plot",
                "Violin Plot",
                "Scatter Plot",
                "3D Scatter",
                "Category Breakdown",
                "Pie Chart",
                "Correlation Heatmap"
            ],
            key="visualization"
        )

        # ====================================================
        # DISTRIBUTION
        # ====================================================

        if visualization == "Distribution":

            column = st.selectbox(
                "Numeric column",
                num_cols,
                key="distribution_column"
            )

            fig = px.histogram(
                df,
                x=column,
                nbins=30,
                title=f"Distribution · {column}",
                template="plotly_dark",
                color_discrete_sequence=[
                    "#6366f1"
                ]
            )

            fig.update_layout(
                height=560,
                margin=dict(
                    l=30,
                    r=30,
                    t=70,
                    b=30
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ====================================================
        # BOX PLOT
        # ====================================================

        elif visualization == "Box Plot":

            column = st.selectbox(
                "Numeric column",
                num_cols,
                key="box_column"
            )

            fig = px.box(
                df,
                y=column,
                title=f"Box Plot · {column}",
                template="plotly_dark",
                points="outliers",
                color_discrete_sequence=[
                    "#22d3ee"
                ]
            )

            fig.update_layout(
                height=560,
                margin=dict(
                    l=30,
                    r=30,
                    t=70,
                    b=30
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ====================================================
        # VIOLIN PLOT
        # ====================================================

        elif visualization == "Violin Plot":

            column = st.selectbox(
                "Numeric column",
                num_cols,
                key="violin_column"
            )

            fig = px.violin(
                df,
                y=column,
                box=True,
                points="outliers",
                title=f"Distribution Shape · {column}",
                template="plotly_dark",
                color_discrete_sequence=[
                    "#a78bfa"
                ]
            )

            fig.update_layout(
                height=560,
                margin=dict(
                    l=30,
                    r=30,
                    t=70,
                    b=30
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ====================================================
        # 2D SCATTER
        # ====================================================

        elif visualization == "Scatter Plot":

            if len(num_cols) < 2:

                st.warning(
                    "At least two analytical numeric "
                    "columns are required."
                )

            else:

                c1, c2 = st.columns(2)

                with c1:

                    x = st.selectbox(
                        "X axis",
                        num_cols,
                        index=0,
                        key="scatter_x"
                    )

                with c2:

                    y = st.selectbox(
                        "Y axis",
                        num_cols,
                        index=1,
                        key="scatter_y"
                    )

                color_column = None

                if cat_cols:

                    color_enabled = st.checkbox(
                        "Color points by category",
                        value=True,
                        key="scatter_color_enabled"
                    )

                    if color_enabled:

                        color_column = st.selectbox(
                            "Category",
                            cat_cols,
                            key="scatter_category"
                        )

                # Try OLS trendline because statsmodels
                # is installed. Fall back gracefully if
                # unavailable.

                try:

                    fig = px.scatter(
                        df,
                        x=x,
                        y=y,
                        color=color_column,
                        title=f"{y} vs {x}",
                        template="plotly_dark",
                        trendline=(
                            "ols"
                            if len(df) >= 3
                            else None
                        ),
                        color_discrete_sequence=(
                            px.colors.qualitative.Bold
                        )
                    )

                except Exception:

                    fig = px.scatter(
                        df,
                        x=x,
                        y=y,
                        color=color_column,
                        title=f"{y} vs {x}",
                        template="plotly_dark",
                        color_discrete_sequence=(
                            px.colors.qualitative.Bold
                        )
                    )

                fig.update_traces(
                    marker=dict(
                        size=8,
                        opacity=0.75
                    )
                )

                fig.update_layout(
                    height=620,
                    margin=dict(
                        l=30,
                        r=30,
                        t=70,
                        b=30
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # ====================================================
        # 3D SCATTER
        # ====================================================

        elif visualization == "3D Scatter":

            if len(num_cols) < 3:

                st.warning(
                    "At least three analytical numeric "
                    "columns are required for 3D visualization."
                )

            else:

                c1, c2, c3 = st.columns(3)

                with c1:

                    x3 = st.selectbox(
                        "X axis",
                        num_cols,
                        index=0,
                        key="three_d_x"
                    )

                with c2:

                    y3 = st.selectbox(
                        "Y axis",
                        num_cols,
                        index=1,
                        key="three_d_y"
                    )

                with c3:

                    z3 = st.selectbox(
                        "Z axis",
                        num_cols,
                        index=2,
                        key="three_d_z"
                    )

                color3 = None

                if cat_cols:

                    color_enabled_3d = st.checkbox(
                        "Color 3D points by category",
                        value=True,
                        key="three_d_color_enabled"
                    )

                    if color_enabled_3d:

                        color3 = st.selectbox(
                            "3D category",
                            cat_cols,
                            key="three_d_category"
                        )

                fig = px.scatter_3d(
                    df,
                    x=x3,
                    y=y3,
                    z=z3,
                    color=color3,
                    title=(
                        f"3D Relationship · "
                        f"{x3} × {y3} × {z3}"
                    ),
                    template="plotly_dark",
                    color_discrete_sequence=(
                        px.colors.qualitative.Bold
                    )
                )

                fig.update_traces(
                    marker=dict(
                        size=5,
                        opacity=0.78
                    )
                )

                fig.update_layout(
                    height=720,
                    margin=dict(
                        l=0,
                        r=0,
                        t=70,
                        b=0
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # ====================================================
        # CATEGORY BREAKDOWN
        # ====================================================

        elif visualization == "Category Breakdown":

            if not cat_cols:

                st.warning(
                    "No useful categorical columns "
                    "were detected."
                )

            else:

                category = st.selectbox(
                    "Category",
                    cat_cols,
                    key="category_breakdown"
                )

                counts = (
                    df[category]
                    .astype(str)
                    .value_counts()
                    .head(15)
                    .reset_index()
                )

                counts.columns = [
                    category,
                    "Count"
                ]

                fig = px.bar(
                    counts,
                    x=category,
                    y="Count",
                    title=f"Records by {category}",
                    template="plotly_dark",
                    color=category,
                    color_discrete_sequence=(
                        px.colors.qualitative.Bold
                    )
                )

                fig.update_layout(
                    height=600,
                    margin=dict(
                        l=30,
                        r=30,
                        t=70,
                        b=30
                    ),
                    showlegend=False
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # ====================================================
        # PIE CHART
        # ====================================================

        elif visualization == "Pie Chart":

            if not cat_cols:

                st.warning(
                    "No useful categorical columns "
                    "were detected."
                )

            else:

                category = st.selectbox(
                    "Category",
                    cat_cols,
                    key="pie_category"
                )

                counts = (
                    df[category]
                    .astype(str)
                    .value_counts()
                    .head(8)
                    .reset_index()
                )

                counts.columns = [
                    category,
                    "Count"
                ]

                fig = px.pie(
                    counts,
                    names=category,
                    values="Count",
                    title=f"Composition · {category}",
                    template="plotly_dark",
                    color_discrete_sequence=(
                        px.colors.qualitative.Bold
                    )
                )

                fig.update_layout(
                    height=600,
                    margin=dict(
                        l=30,
                        r=30,
                        t=70,
                        b=30
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # ====================================================
        # CORRELATION HEATMAP
        # ====================================================

        elif visualization == "Correlation Heatmap":

            if len(num_cols) < 2:

                st.warning(
                    "At least two analytical numeric "
                    "columns are required."
                )

            else:

                corr = df[
                    num_cols
                ].corr()

                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    title="Numeric Feature Correlations",
                    color_continuous_scale="Turbo",
                    zmin=-1,
                    zmax=1
                )

                fig.update_layout(
                    height=680,
                    margin=dict(
                        l=30,
                        r=30,
                        t=70,
                        b=30
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


# ============================================================
# TAB 4 — INSIGHTS
# ============================================================

with tabs[3]:

    st.header(
        "Automated Insights"
    )

    insights = auto_insights(
        df
    )

    if not insights:

        st.info(
            "No automated insights could be generated "
            "for this dataset."
        )

    else:

        for title, body in insights:

            with st.container(
                border=True
            ):

                st.subheader(
                    title
                )

                st.write(
                    body
                )


# ============================================================
# TAB 5 — DATA QUALITY
# ============================================================

with tabs[4]:

    st.header(
        "Data Quality"
    )

    q1, q2, q3 = st.columns(3)

    with q1:

        st.metric(
            "Rows with missing values",
            f"{rows_with_missing:,}"
        )

    with q2:

        st.metric(
            "Analytical numeric fields",
            len(num_cols)
        )

    with q3:

        st.metric(
            "Analytical categorical fields",
            len(cat_cols)
        )

    st.divider()

    st.subheader(
        "Column-level quality profile"
    )

    st.dataframe(
        profile,
        use_container_width=True,
        height=400
    )

    st.divider()

    if missing == 0 and duplicates == 0:

        st.success(
            "Dataset quality check passed: "
            "no missing cells or duplicate rows detected."
        )

    else:

        if missing > 0:

            st.warning(
                f"The dataset contains {missing:,} missing cells."
            )

        if duplicates > 0:

            st.warning(
                f"The dataset contains {duplicates:,} duplicate rows."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NOVA · AI Data Analyst · "
    "Calculation-backed analytics with Pandas, "
    "interactive Plotly visualization and local AI explanation."
)
