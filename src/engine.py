import re
import numpy as np
import pandas as pd
import plotly.express as px


def profile_data(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Type": [str(df[c].dtype) for c in df.columns],
        "Non-null": [int(df[c].notna().sum()) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })


def _column(df, q):
    q = q.lower()

    # Exact column-name match first
    for c in df.columns:
        if c.lower() in q:
            return c

    # Match individual words from column names
    for c in df.columns:
        tokens = [
            x for x in re.split(r"[_\s-]+", c.lower())
            if len(x) > 2
        ]
        if any(x in q for x in tokens):
            return c

    return None


def _categorical_columns(df):
    """
    Return useful categorical columns while excluding
    identifier-like and nearly-unique fields.
    """

    candidates = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    id_keywords = [
        "id",
        "uuid",
        "guid",
        "hash",
        "identifier"
    ]

    useful = []

    for c in candidates:

        name = c.lower().strip()

        looks_like_id = (
            name in id_keywords
            or any(
                name.endswith("_" + word)
                or name.startswith(word + "_")
                for word in id_keywords
            )
        )

        unique_ratio = (
            df[c].nunique(dropna=True)
            / max(len(df), 1)
        )

        nearly_unique = (
            unique_ratio > 0.80
            and df[c].nunique(dropna=True) > 20
        )

        if not looks_like_id and not nearly_unique:
            useful.append(c)

    return useful


def _group_column(df, q):
    """
    Detect a categorical column when the question asks:
    - for each category
    - by category
    - per category
    - grouped by category
    """

    q = q.lower()

    grouping_phrases = [
        "for each",
        "for every",
        "by ",
        "per ",
        "group by",
        "grouped by",
        "each ",
        "every ",
    ]

    wants_grouping = any(p in q for p in grouping_phrases)

    if not wants_grouping:
        return None

    categorical = _categorical_columns(df)

    # Prefer a column explicitly mentioned in the question
    for c in categorical:
        if c.lower() in q:
            return c

        tokens = [
            x for x in re.split(r"[_\s-]+", c.lower())
            if len(x) > 2
        ]

        if any(token in q for token in tokens):
            return c

    # If "each category" is used, look for a column
    # whose name is category-like.
    for c in categorical:
        if any(word in c.lower() for word in [
            "category",
            "type",
            "class",
            "group",
            "segment"
        ]):
            return c

    # Last resort: first categorical field
    if categorical:
        return categorical[0]

    return None


def _filter(df, q):
    """
    Detect a categorical value used as a filter.
    Example:
    'average amount in Electronics'
    """

    for c in _categorical_columns(df):
        values = df[c].dropna().astype(str).unique()

        for v in sorted(values, key=len, reverse=True):
            if str(v).lower() in q.lower():
                return c, v

    return None, None


def _chart(table, x, y, title):
    fig = px.bar(
        table,
        x=x,
        y=y,
        title=title,
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def _grouped_result(df, group_col, value_col, operation):
    """
    Perform a real GROUP BY operation and return
    answer + table + visualization + trace.
    """

    if operation == "mean":
        grouped = (
            df.groupby(group_col, dropna=False)[value_col]
            .mean()
            .reset_index(name=f"Average {value_col}")
        )
        result_col = f"Average {value_col}"
        label = "average"

    elif operation == "sum":
        grouped = (
            df.groupby(group_col, dropna=False)[value_col]
            .sum()
            .reset_index(name=f"Total {value_col}")
        )
        result_col = f"Total {value_col}"
        label = "total"

    elif operation == "max":
        grouped = (
            df.groupby(group_col, dropna=False)[value_col]
            .max()
            .reset_index(name=f"Maximum {value_col}")
        )
        result_col = f"Maximum {value_col}"
        label = "maximum"

    elif operation == "min":
        grouped = (
            df.groupby(group_col, dropna=False)[value_col]
            .min()
            .reset_index(name=f"Minimum {value_col}")
        )
        result_col = f"Minimum {value_col}"
        label = "minimum"

    elif operation == "count":
        grouped = (
            df.groupby(group_col, dropna=False)[value_col]
            .count()
            .reset_index(name=f"Count of {value_col}")
        )
        result_col = f"Count of {value_col}"
        label = "count"

    else:
        return None

    grouped = grouped.sort_values(
        result_col,
        ascending=False
    ).reset_index(drop=True)

    return {
        "answer": (
            f"Here is the **{label} {value_col}** "
            f"for each **{group_col}**."
        ),
        "table": grouped,
        "chart": _chart(
            grouped,
            group_col,
            result_col,
            f"{label.title()} {value_col} by {group_col}"
        ),
        "details": (
            f"Grouped by **{group_col}** and calculated "
            f"{label} of **{value_col}** using Pandas."
        ),
        "operation": (
            f"groupby({group_col}).{operation}({value_col})"
        )
    }


def analyze_question(df, question):

    q = question.lower().strip()

    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # ---------------------------------------------------------
    # DATA QUALITY
    # ---------------------------------------------------------

    if "missing" in q or "null" in q:

        miss = df.isna().sum().sort_values(
            ascending=False
        )

        miss = miss[miss > 0]

        if miss.empty:
            return {
                "answer": "There are **0 missing cells**.",
                "details": (
                    "Pandas isna().sum() returned zero "
                    "across all columns."
                ),
                "operation": "quality.missing"
            }

        tab = (
            miss.rename("Missing")
            .reset_index()
            .rename(columns={"index": "Column"})
        )

        return {
            "answer": (
                f"There are **{int(miss.sum()):,} "
                f"missing cells**. "
                f"**{miss.index[0]}** has the most "
                f"({int(miss.iloc[0]):,})."
            ),
            "table": tab,
            "details": "Counted missing cells per column.",
            "operation": "quality.missing"
        }

    if "duplicate" in q:

        n = int(df.duplicated().sum())

        return {
            "answer": f"There are **{n:,} duplicate rows**.",
            "details": "Pandas duplicated().sum().",
            "operation": "quality.duplicates"
        }

    # ---------------------------------------------------------
    # CORRELATION
    # ---------------------------------------------------------

    if "correlation" in q or "correlat" in q:

        if len(numeric) < 2:
            return {
                "answer": (
                    "At least two numeric columns are "
                    "needed for correlation analysis."
                ),
                "details": (
                    "Fewer than two numeric columns "
                    "were detected."
                ),
                "operation": "statistics.correlation"
            }

        corr = df[numeric].corr()

        pairs = []

        for i, a in enumerate(numeric):
            for b in numeric[i + 1:]:
                pairs.append(
                    (a, b, float(corr.loc[a, b]))
                )

        pairs.sort(
            key=lambda z: abs(z[2]),
            reverse=True
        )

        tab = pd.DataFrame(
            pairs[:10],
            columns=[
                "Column A",
                "Column B",
                "Correlation"
            ]
        )

        a, b, v = pairs[0]

        return {
            "answer": (
                f"The strongest numeric relationship is "
                f"**{a} ↔ {b}** with correlation "
                f"**{v:.3f}**."
            ),
            "table": tab,
            "details": (
                "Pearson correlation matrix computed "
                "with Pandas."
            ),
            "operation": "statistics.correlation"
        }

    # ---------------------------------------------------------
    # DETECT GROUPING
    # ---------------------------------------------------------

    group_col = _group_column(df, q)

    # ---------------------------------------------------------
    # DETECT FILTER
    # ---------------------------------------------------------

    fcol, fval = _filter(df, q)

    # Do not treat the grouping column itself as a filter.
    if group_col == fcol and fval is not None:
        fcol, fval = None, None

    if fcol:
        work = df[
            df[fcol]
            .astype(str)
            .str.lower()
            == fval.lower()
        ].copy()
    else:
        work = df.copy()

    # ---------------------------------------------------------
    # DETECT NUMERIC COLUMN
    # ---------------------------------------------------------

    col = _column(df, q)

    if col not in numeric:
        col = numeric[0] if numeric else None

    note = (
        f" for **{fcol} = {fval}**"
        if fcol else ""
    )

    # ---------------------------------------------------------
    # COUNT BY GROUP
    # ---------------------------------------------------------

    if group_col and any(
        x in q for x in [
            "how many",
            "number of",
            "count",
            "records"
        ]
    ):

        grouped = (
            work.groupby(group_col, dropna=False)
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
            .reset_index(drop=True)
        )

        return {
            "answer": (
                f"Here is the **record count** "
                f"for each **{group_col}**."
            ),
            "table": grouped,
            "chart": _chart(
                grouped,
                group_col,
                "Count",
                f"Records by {group_col}"
            ),
            "details": (
                f"Grouped by {group_col} and counted "
                f"rows with Pandas groupby().size()."
            ),
            "operation": (
                f"groupby({group_col}).count(rows)"
            )
        }

    # ---------------------------------------------------------
    # GROUPED AVERAGE
    # ---------------------------------------------------------

    if group_col and any(
        x in q for x in [
            "average",
            "mean",
            "avg"
        ]
    ):

        if col is None:
            return {
                "answer": (
                    "No numeric field was identified "
                    "for the average."
                ),
                "details": "No numeric columns detected.",
                "operation": "groupby.mean"
            }

        return _grouped_result(
            work,
            group_col,
            col,
            "mean"
        )

    # ---------------------------------------------------------
    # GROUPED TOTAL
    # ---------------------------------------------------------

    if group_col and any(
        x in q for x in [
            "total",
            "sum"
        ]
    ):

        if col is None:
            return {
                "answer": (
                    "No numeric field was identified "
                    "for the total."
                ),
                "details": "No numeric columns detected.",
                "operation": "groupby.sum"
            }

        return _grouped_result(
            work,
            group_col,
            col,
            "sum"
        )

    # ---------------------------------------------------------
    # GROUPED MAXIMUM / HIGHEST
    # ---------------------------------------------------------

    if group_col and any(
        x in q for x in [
            "maximum",
            "max",
            "highest",
            "largest"
        ]
    ):

        if col is None:
            return {
                "answer": (
                    "No numeric field was identified "
                    "for the maximum."
                ),
                "details": "No numeric columns detected.",
                "operation": "groupby.max"
            }

        return _grouped_result(
            work,
            group_col,
            col,
            "max"
        )

    # ---------------------------------------------------------
    # GROUPED MINIMUM / LOWEST
    # ---------------------------------------------------------

    if group_col and any(
        x in q for x in [
            "minimum",
            "min",
            "lowest",
            "smallest"
        ]
    ):

        if col is None:
            return {
                "answer": (
                    "No numeric field was identified "
                    "for the minimum."
                ),
                "details": "No numeric columns detected.",
                "operation": "groupby.min"
            }

        return _grouped_result(
            work,
            group_col,
            col,
            "min"
        )

    # ---------------------------------------------------------
    # OVERALL AVERAGE
    # ---------------------------------------------------------

    if any(
        x in q for x in [
            "average",
            "mean",
            "avg"
        ]
    ):

        if col is None:
            return {
                "answer": (
                    "No numeric field was identified "
                    "for the average."
                ),
                "details": "No numeric columns detected.",
                "operation": "aggregate.mean"
            }

        value = work[col].mean()

        return {
            "answer": (
                f"The average **{col}** is "
                f"**{value:,.2f}**{note}."
            ),
            "details": (
                f"Calculated on {len(work):,} matching "
                f"rows with Pandas mean()."
            ),
            "operation": (
                f"aggregate.mean("
                f"column={col}, "
                f"filter={fcol}:{fval})"
            )
        }

    # ---------------------------------------------------------
    # OVERALL TOTAL
    # ---------------------------------------------------------

    if any(
        x in q for x in [
            "total",
            "sum"
        ]
    ):

        if col is None:
            return {
                "answer": (
                    "No numeric field was identified "
                    "for the total."
                ),
                "details": "No numeric columns detected.",
                "operation": "aggregate.sum"
            }

        value = work[col].sum()

        return {
            "answer": (
                f"The total **{col}** is "
                f"**{value:,.2f}**{note}."
            ),
            "details": (
                f"Calculated on {len(work):,} matching "
                f"rows with Pandas sum()."
            ),
            "operation": (
                f"aggregate.sum("
                f"column={col}, "
                f"filter={fcol}:{fval})"
            )
        }

    # ---------------------------------------------------------
    # OVERALL MAXIMUM
    # ---------------------------------------------------------

    if any(
        x in q for x in [
            "maximum",
            "max",
            "highest",
            "largest"
        ]
    ):

        if col is None:
            return {
                "answer": (
                    "No numeric field was identified."
                ),
                "details": "No numeric columns detected.",
                "operation": "aggregate.max"
            }

        value = work[col].max()

        return {
            "answer": (
                f"The maximum **{col}** is "
                f"**{value:,.2f}**{note}."
            ),
            "details": (
                f"Calculated on {len(work):,} matching "
                f"rows with Pandas max()."
            ),
            "operation": (
                f"aggregate.max("
                f"column={col}, "
                f"filter={fcol}:{fval})"
            )
        }

    # ---------------------------------------------------------
    # OVERALL MINIMUM
    # ---------------------------------------------------------

    if any(
        x in q for x in [
            "minimum",
            "min",
            "lowest",
            "smallest"
        ]
    ):

        if col is None:
            return {
                "answer": (
                    "No numeric field was identified."
                ),
                "details": "No numeric columns detected.",
                "operation": "aggregate.min"
            }

        value = work[col].min()

        return {
            "answer": (
                f"The minimum **{col}** is "
                f"**{value:,.2f}**{note}."
            ),
            "details": (
                f"Calculated on {len(work):,} matching "
                f"rows with Pandas min()."
            ),
            "operation": (
                f"aggregate.min("
                f"column={col}, "
                f"filter={fcol}:{fval})"
            )
        }

    # ---------------------------------------------------------
    # TOP ROWS
    # ---------------------------------------------------------

    if "top" in q:

        n = 10

        m = re.search(
            r"top\s+(\d+)",
            q
        )

        if m:
            n = min(
                max(int(m.group(1)), 1),
                50
            )

        if col is None:
            return {
                "answer": (
                    "A numeric field is required "
                    "for ranking."
                ),
                "details": "No numeric columns detected.",
                "operation": "ranking.nlargest"
            }

        out = work.nlargest(n, col)

        return {
            "answer": (
                f"Here are the top **{len(out)}** "
                f"rows by **{col}**{note}."
            ),
            "table": out,
            "chart": _chart(
                out.reset_index(),
                "index",
                col,
                f"Top {len(out)} rows · {col}"
            ),
            "details": (
                f"Used Pandas nlargest({n}, {col})."
            ),
            "operation": (
                f"ranking.nlargest("
                f"n={n}, column={col}, "
                f"filter={fcol}:{fval})"
            )
        }

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------

    if numeric:

        tab = (
            work[numeric]
            .describe()
            .T
            .reset_index()
            .rename(columns={"index": "Column"})
        )

        return {
            "answer": (
                f"I generated a statistical snapshot "
                f"for {len(numeric)} numeric fields{note}."
            ),
            "table": tab,
            "details": "Fallback to Pandas describe().",
            "operation": "statistics.describe"
        }

    return {
        "answer": (
            "Try asking for an average, total, "
            "maximum, minimum, top rows, missing "
            "values, duplicates or correlations."
        ),
        "details": "No supported operation detected.",
        "operation": "fallback.help"
    }


def auto_insights(df):

    out = []

    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if numeric:

        means = (
            df[numeric]
            .mean()
            .sort_values(ascending=False)
        )

        out.append(
            (
                "Highest average numeric field",
                f"**{means.index[0]}** has the "
                f"largest mean value at "
                f"**{means.iloc[0]:,.2f}** among "
                f"numeric fields."
            )
        )

        spread = (
            df[numeric]
            .std()
            .sort_values(ascending=False)
        )

        out.append(
            (
                "Largest variability",
                f"**{spread.index[0]}** has the "
                f"largest standard deviation at "
                f"**{spread.iloc[0]:,.2f}**, indicating "
                f"the widest absolute spread."
            )
        )

        if len(numeric) > 1:

            corr = df[numeric].corr()

            pairs = [
                (
                    a,
                    b,
                    abs(float(corr.loc[a, b]))
                )
                for i, a in enumerate(numeric)
                for b in numeric[i + 1:]
            ]

            if pairs:

                a, b, v = max(
                    pairs,
                    key=lambda x: x[2]
                )

                out.append(
                    (
                        "Strongest relationship",
                        f"**{a}** and **{b}** have the "
                        f"strongest absolute Pearson "
                        f"correlation at **{v:.3f}**."
                    )
                )

    out.append(
        (
            "Data quality",
            f"The dataset contains "
            f"**{int(df.isna().sum().sum()):,} "
            f"missing cells** and "
            f"**{int(df.duplicated().sum()):,} "
            f"duplicate rows**."
        )
    )

    return out
