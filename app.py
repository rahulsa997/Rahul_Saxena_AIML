"""Streamlit app comparing 5 from-scratch classifiers on the Breast Cancer dataset."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Resolved from this file so the app behaves the same locally and when deployed.
MODEL_DIR = Path(__file__).resolve().parent / "model"

# The classifier modules live in model/, so that folder must be importable.
sys.path.insert(0, str(MODEL_DIR))

# Malignant (label 0) is the positive class, so recall means "cancers caught".
POS_LABEL = 0

# Backs up hue on the ROC chart, where 5 lines sit too close for colour alone.
MODEL_DASH = {
    "Logistic Regression": "solid",
    "Decision Tree": "dash",
    "kNN": "dot",
    "Naive Bayes": "dashdot",
    "Random Forest (Ensemble)": "longdash",
}

# BITS Pilani brand: maroon on cream owns the page chrome (headers, buttons,
# sidebar). Chart series keep their own validated hues -- see "models" below.
BITS_MAROON = "#7A0000"
BITS_MAROON_DARK = "#5C0000"
BITS_MAROON_ON_DARK = "#d9736e"  # lightened so it stays readable on a dark surface

# Dark mode is a re-stepped palette, not an inverted one; flipping the light
# values would leave several hues unreadable.
PALETTES = {
    "light": {
        "surface": "#fdfbf7",
        "plane": "#f9f6f0",
        "sidebar": "#f4ece1",
        "tile_from": "#fdfbf7",
        "tile_to": "#f4ece1",
        "ink": "#222222",
        "ink_secondary": "#5c5145",
        "ink_muted": "#8a7d6d",
        "gridline": "#e6d5c3",
        "border": "#e6d5c3",
        "brand": BITS_MAROON,
        "brand_hover": BITS_MAROON_DARK,
        "diverge_mid": "#f0e6d8",
        # Series hues stay blue/orange/aqua/yellow/magenta: they are ordered for
        # colour-blind separation, which maroon shades could not preserve.
        "models": {
            "Logistic Regression": "#2a78d6",       # blue
            "Decision Tree": "#eb6834",             # orange
            "kNN": "#1baf7a",                       # aqua
            "Naive Bayes": "#eda100",               # yellow
            "Random Forest (Ensemble)": "#e87ba4",  # magenta
        },
        "classes": {"malignant": "#eb6834", "benign": "#2a78d6"},
        # Single-hue ramp in the brand maroon, light to dark.
        "seq_brand": ["#f7e6e6", "#b8433f", BITS_MAROON],
        "diverge_low": "#b8433f",
        "diverge_high": "#2a6fb0",
        "plotly_template": "plotly_white",
    },
    "dark": {
        "surface": "#1f1a18",
        "plane": "#151110",
        "sidebar": "#241d1a",
        "tile_from": "#2a2320",
        "tile_to": "#1f1a18",
        "ink": "#f5f0ea",
        "ink_secondary": "#c9bcae",
        "ink_muted": "#9a8d7e",
        "gridline": "#3a302c",
        "border": "#3a302c",
        "brand": BITS_MAROON_ON_DARK,
        "brand_hover": "#e89b96",
        "diverge_mid": "#3a302c",
        "models": {
            "Logistic Regression": "#3987e5",       # blue
            "Decision Tree": "#d95926",             # orange
            "kNN": "#199e70",                       # aqua
            "Naive Bayes": "#c98500",               # yellow
            "Random Forest (Ensemble)": "#d55181",  # magenta
        },
        "classes": {"malignant": "#d95926", "benign": "#3987e5"},
        # Runs dark to light so the low end recedes toward the dark surface.
        "seq_brand": ["#4a1010", "#a8403c", "#e8a5a1"],
        "diverge_low": "#e08b87",
        "diverge_high": "#3987e5",
        "plotly_template": "plotly_dark",
    },
}

# Fixed in both modes; always paired with an icon and label, never colour alone.
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

st.set_page_config(
    page_title="Breast Cancer Classification Explorer",
    page_icon="🩺",
    layout="wide",
)

# Read before anything renders; the sidebar toggle binds to this same key.
st.session_state.setdefault("dark_mode", False)
THEME = PALETTES["dark" if st.session_state["dark_mode"] else "light"]

MODEL_COLORS = THEME["models"]
CLASS_COLORS = THEME["classes"]
INK_PRIMARY = THEME["ink"]
INK_MUTED = THEME["ink_muted"]
GRIDLINE = THEME["gridline"]
SURFACE = THEME["surface"]
BRAND = THEME["brand"]

st.markdown(
    f"""
<style>
    /* Streamlit paints its own chrome, so surfaces are repainted rather than inherited. */
    [data-testid="stAppViewContainer"], .stApp {{
        background-color: {THEME["plane"]};
        color: {THEME["ink"]};
    }}
    [data-testid="stHeader"] {{ background-color: {THEME["plane"]}; }}
    [data-testid="stSidebar"] {{
        background-color: {THEME["sidebar"]};
        border-right: 1px solid {THEME["border"]};
    }}

    /* BITS maroon carries the headings. */
    h1, h2, h3 {{ color: {BRAND} !important; }}
    h4, h5, h6 {{ color: {THEME["ink"]}; }}
    h1 {{ margin-bottom: 4px; }}
    p, li, label, .stMarkdown {{ color: {THEME["ink"]}; }}

    [data-testid="stMetricValue"] {{ color: {BRAND}; }}
    [data-testid="stMetricLabel"] {{ color: {THEME["ink_secondary"]}; }}

    .stButton>button {{
        background-color: {BRAND};
        color: #ffffff;
        border-radius: 4px;
        border: none;
    }}
    .stButton>button:hover {{
        background-color: {THEME["brand_hover"]};
        color: #ffffff;
    }}

    /* Expanders, tabs and captions carry their own backgrounds. */
    [data-testid="stExpander"] {{
        background-color: {THEME["surface"]};
        border: 1px solid {THEME["border"]};
        border-radius: 8px;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] p {{ color: {THEME["ink"]}; }}

    [data-testid="stTabs"] button {{ color: {THEME["ink_secondary"]}; }}
    [data-testid="stTabs"] button[aria-selected="true"] {{
        color: {BRAND};
        font-weight: 600;
    }}
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background-color: {BRAND}; }}

    [data-testid="stCaptionContainer"], .stCaption, small {{
        color: {THEME["ink_secondary"]} !important;
    }}

    .stat-tile {{
        background: linear-gradient(135deg, {THEME["tile_from"]} 0%, {THEME["tile_to"]} 100%);
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        border: 1px solid {THEME["border"]};
        border-top: 3px solid {BRAND};
    }}
    .stat-number {{ font-size: 32px; font-weight: 700; color: {BRAND}; }}
    .stat-label {{ font-size: 13px; color: {THEME["ink_secondary"]}; margin-top: 4px; }}

    .verdict {{
        border-radius: 10px;
        padding: 22px 24px;
        color: #ffffff;
        margin-bottom: 12px;
    }}
    .verdict-label {{ font-size: 13px; opacity: 0.92; letter-spacing: 0.04em; }}
    .verdict-value {{ font-size: 34px; font-weight: 700; line-height: 1.2; }}

    .badge {{
        display: inline-block;
        border-radius: 6px;
        padding: 10px 16px;
        font-weight: 600;
        font-size: 15px;
        color: #ffffff;
    }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Train all five models once and hold them; no saved .pkl files to load."""
    import decision_tree
    import knn
    import logistic_regression
    import naive_bayes
    import random_forest
    from data_prep import prepare_data

    # save_test_csv=False -- a running app should never write to disk.
    data = prepare_data(save_test_csv=False)

    # Order fixes the sidebar dropdown and the comparison-table row order.
    modules = (logistic_regression, decision_tree, knn, naive_bayes, random_forest)

    models = {}
    for module in modules:
        estimator = module.build()
        estimator.fit(data["X_train"], data["y_train"])
        models[module.NAME] = estimator

    # Overview-tab figures, measured rather than typed in, so they cannot drift.
    all_targets = pd.concat([data["y_train"], data["y_test"]])
    counts = all_targets.value_counts()
    total = int(len(all_targets))
    stats = {
        "n_samples": total,
        "n_features": len(data["feature_columns"]),
        "n_train": int(len(data["y_train"])),
        "n_test": int(len(data["y_test"])),
        "class_counts": {
            data["class_names"][int(label)]: int(n) for label, n in counts.items()
        },
        "total": total,
    }

    return data["scaler"], data["feature_columns"], data["class_names"], models, stats


@st.cache_data(show_spinner=False)
def compute_all_metrics(X_scaled, y_true, _models):
    """Score all five models on the uploaded data, never from a saved results file."""
    y_binary = (np.asarray(y_true) == POS_LABEL).astype(int)

    rows = []
    for name, mdl in _models.items():
        pred = mdl.predict(X_scaled)
        proba = mdl.predict_proba(X_scaled)[:, POS_LABEL]
        try:
            auc_score = roc_auc_score(y_binary, proba)
        except ValueError:
            auc_score = float("nan")
        rows.append(
            {
                "ML Model Name": name,
                "Accuracy": accuracy_score(y_true, pred),
                "AUC": auc_score,
                "Precision": precision_score(y_true, pred, pos_label=POS_LABEL, zero_division=0),
                "Recall": recall_score(y_true, pred, pos_label=POS_LABEL, zero_division=0),
                "F1": f1_score(y_true, pred, pos_label=POS_LABEL, zero_division=0),
                "MCC": matthews_corrcoef(y_true, pred),
            }
        )
    return pd.DataFrame(rows).round(4)


@st.cache_data(show_spinner=False)
def compute_permutation_importance(model_key, _model, X_scaled, y_true):
    """Score drop when each feature is shuffled; works for every estimator."""
    result = permutation_importance(
        _model, X_scaled, y_true, n_repeats=10, random_state=42, scoring="accuracy"
    )
    return result.importances_mean


with st.spinner("Training the five models (first load only)…"):
    scaler, feature_columns, class_names, models, dataset_stats = load_artifacts()


def explain(title, body):
    """Collapsible plain-language reading, attached under a chart or metric."""
    with st.expander(f"❓ {title}"):
        st.markdown(body)


def style_fig(fig, height=380, showlegend=False):
    """Shared chart chrome: recessive axes, themed surface, room to breathe."""
    fig.update_layout(
        template=THEME["plotly_template"],
        height=height,
        showlegend=showlegend,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_PRIMARY),
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(font_size=13),
        legend=dict(font_color=INK_PRIMARY),
    )
    fig.update_xaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, linecolor=GRIDLINE)
    fig.update_yaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, linecolor=GRIDLINE)
    return fig


def label_of(target_value):
    """0/1 -> 'malignant'/'benign', using the names saved at training time."""
    return class_names[int(target_value)]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🩺 Controls")
st.sidebar.markdown(
    "Upload the **test data CSV** (the `test_data.csv` in this repo, or any CSV "
    f"with the same {len(feature_columns)} feature columns plus a `target` column)."
)

uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
model_name = st.sidebar.selectbox("Model", list(models.keys()))

st.sidebar.markdown("---")
# Toggling reruns the script, re-deriving every colour from the other palette.
st.sidebar.toggle("🌙 Dark mode", key="dark_mode")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Target: 0 = malignant, 1 = benign\n\n"
    "Dataset: Breast Cancer Wisconsin (Diagnostic), UCI ML Repository"
)

# Validate the upload once. A flag rather than st.stop(), which would halt the
# whole script and blank every later tab.
data = None
load_error = None

if uploaded_file is not None:
    try:
        candidate = pd.read_csv(uploaded_file)
    except Exception as exc:
        load_error = f"Could not read the uploaded file: {exc}"
    else:
        if "target" not in candidate.columns:
            load_error = (
                "The uploaded CSV needs a `target` column (the true labels) so the "
                "app can score the predictions against reality."
            )
        else:
            missing = [c for c in feature_columns if c not in candidate.columns]
            if missing:
                load_error = f"Uploaded CSV is missing required feature columns: {missing}"
            else:
                data = candidate

has_data = data is not None

if has_data:
    X = data[feature_columns]
    y_true = data["target"]
    X_scaled = scaler.transform(X)

    model = models[model_name]
    y_pred = model.predict(X_scaled)
    # Probability of malignant, the positive class.
    y_proba = model.predict_proba(X_scaled)[:, POS_LABEL]
    y_binary = (np.asarray(y_true) == POS_LABEL).astype(int)


def needs_upload():
    """Shown in each data-driven tab when there is nothing loaded yet."""
    if load_error:
        st.error(load_error)
    else:
        st.info("👈 Upload a CSV from the sidebar to fill in this tab — try `test_data.csv`.")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.title("🩺 Breast Cancer Classification Explorer")
st.caption(
    "Five machine-learning models, one dataset, and a plain-language reading of "
    "everything they produce."
)

tab_overview, tab_data, tab_eval, tab_single, tab_compare = st.tabs(
    ["🏠 Overview", "🔬 Data Explorer", "🎯 Live Evaluation", "👤 Single Prediction", "🏆 Model Comparison"]
)

# =========================================================================
# TAB 1 — OVERVIEW
# =========================================================================
with tab_overview:
    st.header("What this project does")
    st.markdown(
        """
Doctors can take a fine-needle sample from a breast lump and photograph the cells
under a microscope. Software then measures the shape of each cell nucleus — how
big it is, how bumpy its edge is, how irregular it looks.

**The question this project answers:** using only those measurements, can a
computer tell a *malignant* (cancerous) lump from a *benign* (harmless) one?

Five different machine-learning models are trained on the same data and scored
the same way, so their results can be compared fairly. This app lets you look at
the data, watch the models make individual calls, and see which one comes out on
top.
"""
    )

    st.subheader("📋 The dataset at a glance")
    st.caption(
        "These describe the **full source dataset the models were trained on**, "
        "measured when the app started — not the file you upload. The upload is "
        "summarised in the Data Explorer tab."
    )

    total = dataset_stats["total"]
    benign = dataset_stats["class_counts"].get("benign", 0)
    malignant = dataset_stats["class_counts"].get("malignant", 0)

    c1, c2, c3, c4 = st.columns(4)
    tiles = [
        (c1, f"{total}", "Patient samples"),
        (c2, f"{dataset_stats['n_features']}", "Measurements each"),
        (c3, f"{benign}", f"Benign ({benign / total:.0%})"),
        (c4, f"{malignant}", f"Malignant ({malignant / total:.0%})"),
    ]
    for col, number, label in tiles:
        col.markdown(
            f'<div class="stat-tile"><div class="stat-number">{number}</div>'
            f'<div class="stat-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"Split {dataset_stats['n_train']} rows for training / "
        f"{dataset_stats['n_test']} held back for testing, stratified so both "
        "halves keep the same class balance."
    )

    explain(
        "Where does this data come from, and what are the 30 measurements?",
        """
The **Breast Cancer Wisconsin (Diagnostic)** dataset from the UCI Machine
Learning Repository — a long-standing public research dataset.

Ten physical properties are measured for each cell nucleus:

| Property | In plain terms |
|---|---|
| radius, perimeter, area | how big the nucleus is |
| texture | how much the greyscale varies inside it |
| smoothness | how even the edge is |
| compactness, concavity, concave points | how dented or lobed the outline is |
| symmetry | how balanced the shape is |
| fractal dimension | how ragged the boundary is at fine detail |

Each one is reported three ways — the **mean** across all nuclei in the image,
the **error** (how much it varied), and the **worst** (the most extreme nucleus
found). 10 properties × 3 versions = **30 features**.

The intuition doctors already have holds up in the numbers: malignant nuclei
tend to be *larger* and *more irregularly shaped*. You can see this yourself in
the Data Explorer tab.
""",
    )

    st.subheader("🤖 The five models")
    st.markdown(
        """
| Model | How it decides, in one sentence |
|---|---|
| **Logistic Regression** | Draws one straight dividing line through the measurements and asks which side a case falls on. |
| **Decision Tree** | Plays twenty questions — "is the worst radius above 16.8?" — until it reaches a verdict. |
| **k-Nearest Neighbours** | Finds the 7 most similar past patients and goes with the majority. |
| **Naive Bayes** | Uses probability, assuming each measurement is an independent clue. |
| **Random Forest** | Grows 200 different decision trees and lets them vote. |
"""
    )

    st.subheader("📖 How to use this app")
    st.markdown(
        """
1. **Upload `test_data.csv`** using the sidebar — this is data the models have
   never seen, held back during training.
2. **🔬 Data Explorer** — see what the measurements actually look like.
3. **🎯 Live Evaluation** — score the selected model, with every metric explained.
4. **👤 Single Prediction** — watch the model judge one patient at a time.
5. **🏆 Model Comparison** — all five models head to head.

Switch models any time using the sidebar dropdown; every tab updates.
"""
    )

# =========================================================================
# TAB 2 — DATA EXPLORER
# =========================================================================
with tab_data:
    st.header("🔬 What does the data look like?")

    if not has_data:
        needs_upload()
    else:
        counts = y_true.map(label_of).value_counts()

        left, right = st.columns([1, 1.3])

        with left:
            st.subheader("Class balance")
            st.caption("How many of each kind of case are in the file you uploaded.")
            donut = go.Figure(
                go.Pie(
                    labels=counts.index.tolist(),
                    values=counts.values.tolist(),
                    hole=0.55,
                    marker=dict(
                        colors=[CLASS_COLORS[c] for c in counts.index],
                        line=dict(color=SURFACE, width=2),
                    ),
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>%{value} cases (%{percent})<extra></extra>",
                )
            )
            donut.update_layout(
                annotations=[
                    dict(
                        text=f"{len(data)}<br>cases",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=18, color=INK_PRIMARY),
                    )
                ]
            )
            st.plotly_chart(style_fig(donut, height=340), use_container_width=True)

        with right:
            st.subheader("Feature distribution")
            st.caption(
                "Pick a measurement. The two humps show how malignant and benign "
                "cases differ on it — the further apart they sit, the more useful "
                "that measurement is for telling them apart."
            )
            feature = st.selectbox("Measurement", feature_columns, index=feature_columns.index("mean radius"))

            plot_df = data[[feature, "target"]].copy()
            plot_df["Diagnosis"] = plot_df["target"].map(label_of)

            hist = px.histogram(
                plot_df,
                x=feature,
                color="Diagnosis",
                nbins=30,
                barmode="overlay",
                opacity=0.65,
                color_discrete_map=CLASS_COLORS,
            )
            hist.update_traces(marker_line_color=SURFACE, marker_line_width=1)
            hist.update_layout(legend=dict(orientation="h", y=1.12, x=0))
            hist.update_yaxes(title="Number of cases")
            st.plotly_chart(style_fig(hist, height=340, showlegend=True), use_container_width=True)

            malignant_mean = plot_df.loc[plot_df["target"] == 0, feature].mean()
            benign_mean = plot_df.loc[plot_df["target"] == 1, feature].mean()
            direction = "higher" if malignant_mean > benign_mean else "lower"
            st.markdown(
                f"On **{feature}**, malignant cases average **{malignant_mean:.3f}** and "
                f"benign cases average **{benign_mean:.3f}** — malignant runs {direction}."
            )

        st.subheader("How the measurements relate to each other")
        st.caption(
            "Deep blue means two measurements rise and fall together; deep red means "
            "one goes up as the other goes down; pale means they are unrelated."
        )

        corr = data[feature_columns].corr()
        heat = px.imshow(
            corr,
            zmin=-1,
            zmax=1,
            color_continuous_scale=[
                [0.0, THEME["diverge_low"]],
                [0.5, THEME["diverge_mid"]],
                [1.0, THEME["diverge_high"]],
            ],
            aspect="auto",
        )
        heat.update_traces(
            hovertemplate="<b>%{x}</b><br><b>%{y}</b><br>correlation: %{z:.2f}<extra></extra>"
        )
        heat.update_xaxes(tickangle=-45, tickfont_size=9)
        heat.update_yaxes(tickfont_size=9)
        st.plotly_chart(style_fig(heat, height=620), use_container_width=True)

        explain(
            "Why does this heatmap matter?",
            """
Look at the bright blue block where **radius**, **perimeter** and **area** meet.
Those three sit near a correlation of **1.0** — which makes sense, because they
are three ways of describing the same thing: how big the nucleus is. Measure the
radius and you already know the area.

This is not a quirk — it explains a result you will see later. **Naive Bayes
assumes every measurement is an independent clue.** This heatmap is the proof
that the assumption is false here, and it is why Naive Bayes lands mid-table
despite being a perfectly reasonable model.

It also explains why **Logistic Regression** and **Random Forest** do well: both
cope gracefully with overlapping, redundant measurements.
""",
        )

        st.subheader("Browse the raw numbers")
        f1, f2 = st.columns([1, 2])
        with f1:
            class_filter = st.radio("Show", ["All", "Malignant only", "Benign only"], index=0)
        with f2:
            shown_cols = st.multiselect(
                "Columns to display",
                feature_columns,
                default=feature_columns[:6],
                help="30 columns at once is unreadable — pick the ones you care about.",
            )

        table_df = data.copy()
        if class_filter == "Malignant only":
            table_df = table_df[table_df["target"] == 0]
        elif class_filter == "Benign only":
            table_df = table_df[table_df["target"] == 1]

        table_df = table_df[shown_cols + ["target"]].copy()
        table_df["Diagnosis"] = table_df["target"].map(label_of)
        table_df = table_df.drop(columns=["target"])

        st.dataframe(table_df, use_container_width=True, height=320)
        st.caption(f"Showing {len(table_df)} of {len(data)} rows.")

# =========================================================================
# TAB 3 — LIVE EVALUATION
# =========================================================================
with tab_eval:
    st.header(f"🎯 How well does {model_name} do?")

    if not has_data:
        needs_upload()
    else:
        try:
            auc = roc_auc_score(y_binary, y_proba)
        except ValueError:
            auc = float("nan")  # only one class present in the uploaded sample

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        # Class balance of the source dataset, for the MCC explanation below.
        _total = dataset_stats["total"]
        benign_pct = dataset_stats["class_counts"].get("benign", 0) / _total
        malignant_pct = dataset_stats["class_counts"].get("malignant", 0) / _total

        st.caption(
            f"Scored live on the {len(data)} rows you uploaded — nothing here is "
            "hardcoded."
        )

        cols = st.columns(6)
        scores = [
            ("Accuracy", accuracy),
            ("AUC", auc),
            ("Precision", precision),
            ("Recall", recall),
            ("F1 Score", f1),
            ("MCC", mcc),
        ]
        for col, (label, value) in zip(cols, scores):
            col.metric(label, "N/A" if np.isnan(value) else f"{value:.4f}")

        explain(
            "What do these six numbers actually mean?",
            f"""
Every score below runs from 0 to 1, and higher is better.

**Malignant is the "positive" class here** — the thing we are trying to detect.
So Precision, Recall and F1 all describe how well the model finds *cancer*, not
how well it clears healthy patients.

**Accuracy — {accuracy:.4f}**
Out of every case, how many did the model label correctly? Simple, but it can
flatter a model when one class is much more common than the other.

**Recall — {recall:.4f}**
Of all the cases that really were cancer, how many did the model catch?
*This is the one that matters most in screening.* A miss here means a patient
with cancer is told they are fine.

**Precision — {precision:.4f}**
When the model said "cancer", how often was it right? Low precision means false
alarms — frightening and expensive, but not dangerous the way a miss is.

**F1 Score — {f1:.4f}**
A single number balancing precision and recall. Useful when you want one figure
instead of two that trade off against each other.

**AUC — {auc:.4f}**
Take one random cancer case and one random healthy case. How often does the
model rate the cancer case as more suspicious? **0.5 is a coin flip; 1.0 is
perfect.** AUC judges the model's ranking, independent of where you draw the
cut-off.

**MCC — {mcc:.4f}**
Matthews Correlation Coefficient — a balanced score that stays honest when the
classes are uneven, which they are here ({benign_pct:.0%} benign / {malignant_pct:.0%} malignant). It only
scores high when the model does well on *both* classes, so it is the hardest of
the six to fake.
""",
        )

        st.subheader("Where exactly did it get things right and wrong?")
        left, right = st.columns([1, 1])

        cm = confusion_matrix(y_true, y_pred)

        with left:
            st.caption("Rows are the truth; columns are what the model guessed.")
            cm_fig = px.imshow(
                cm,
                x=[f"predicted {c}" for c in class_names],
                y=[f"actually {c}" for c in class_names],
                color_continuous_scale=THEME["seq_brand"],
                text_auto=True,
                aspect="auto",
            )
            cm_fig.update_traces(
                textfont_size=22,
                hovertemplate="%{y}<br>%{x}<br><b>%{z} cases</b><extra></extra>",
            )
            cm_fig.update_coloraxes(showscale=False)
            st.plotly_chart(style_fig(cm_fig, height=360), use_container_width=True)

        with right:
            st.caption("Per-class breakdown of the same result.")
            report = classification_report(
                y_true,
                y_pred,
                target_names=[str(c) for c in class_names],
                output_dict=True,
                zero_division=0,
            )
            report_df = pd.DataFrame(report).transpose().round(4)
            st.dataframe(report_df, use_container_width=True, height=360)

        # Spell the matrix out in words, with this run's actual numbers.
        # Row 0 is malignant, the positive class, so row 0 holds TP and FN.
        if cm.shape == (2, 2):
            true_pos, false_neg = cm[0, 0], cm[0, 1]
            false_pos, true_neg = cm[1, 0], cm[1, 1]
            st.markdown(
                f"""
**Reading the grid above, in words:**

- ✅ **{true_pos}** cases were malignant and correctly flagged as malignant.
- ✅ **{true_neg}** cases were benign and correctly cleared as benign.
- ⚠️ **{false_neg}** malignant cases were wrongly cleared as benign — **these are
  the dangerous mistakes.** A patient with cancer would have been sent home.
- ⚠️ **{false_pos}** benign cases were wrongly flagged as malignant — a false
  alarm, causing worry and follow-up tests, but nobody is harmed.

The two errors are *not* equally bad, which is why **recall** ({recall:.4f}) —
the share of real cancers caught — deserves more attention than raw accuracy.
"""
            )

# =========================================================================
# TAB 4 — SINGLE PREDICTION
# =========================================================================
with tab_single:
    st.header("👤 One patient at a time")

    if not has_data:
        needs_upload()
    else:
        st.caption(
            "Aggregate scores hide what a model actually does. Pick a single case "
            "and watch it decide."
        )

        row_idx = st.slider("Case number", 0, len(data) - 1, 0)

        row_true = int(y_true.iloc[row_idx])
        row_pred = int(y_pred[row_idx])
        # y_proba holds P(malignant), the positive class.
        proba_malignant = float(y_proba[row_idx])
        proba = {"malignant": proba_malignant, "benign": 1.0 - proba_malignant}
        pred_label = label_of(row_pred)
        true_label = label_of(row_true)
        correct = row_pred == row_true

        left, right = st.columns([1, 1])

        with left:
            st.markdown(
                f'<div class="verdict" style="background-color:{CLASS_COLORS[pred_label]}">'
                f'<div class="verdict-label">{model_name.upper()} PREDICTS</div>'
                f'<div class="verdict-value">{pred_label.capitalize()}</div></div>',
                unsafe_allow_html=True,
            )

            badge_color = STATUS_GOOD if correct else STATUS_CRITICAL
            badge_text = (
                f"✓ Correct — this case really is {true_label}"
                if correct
                else f"✕ Wrong — this case is actually {true_label}"
            )
            st.markdown(
                f'<span class="badge" style="background-color:{badge_color}">{badge_text}</span>',
                unsafe_allow_html=True,
            )

            st.markdown("**How confident is it?**")
            conf = go.Figure(
                go.Bar(
                    x=[proba["malignant"] * 100, proba["benign"] * 100],
                    y=["malignant", "benign"],
                    orientation="h",
                    marker=dict(
                        color=[CLASS_COLORS["malignant"], CLASS_COLORS["benign"]],
                        line=dict(color=SURFACE, width=2),
                    ),
                    text=[f"{proba['malignant'] * 100:.1f}%", f"{proba['benign'] * 100:.1f}%"],
                    textposition="auto",
                    hovertemplate="<b>%{y}</b>: %{x:.1f}% confidence<extra></extra>",
                )
            )
            conf.update_xaxes(range=[0, 100], title="Confidence (%)")
            st.plotly_chart(style_fig(conf, height=220), use_container_width=True)

            confidence_pct = max(proba.values()) * 100
            if confidence_pct > 95:
                verdict_note = "The model is very sure about this one."
            elif confidence_pct > 75:
                verdict_note = "Reasonably confident, but not certain."
            else:
                verdict_note = (
                    "This is a borderline case — the model is close to a coin flip, "
                    "and different models may well disagree about it."
                )
            st.info(f"**{confidence_pct:.1f}% confidence.** {verdict_note}")

            st.caption(
                "💡 Change the model in the sidebar to see how a different one judges "
                "this same patient. Borderline cases are where they part ways."
            )

        with right:
            st.markdown("**This patient's measurements**")
            st.caption(
                "Each value is shown beside the average for that measurement across "
                "the whole file, so you can see what stands out."
            )
            case = data.iloc[row_idx]
            comparison = pd.DataFrame(
                {
                    "Measurement": feature_columns,
                    "This case": [round(float(case[c]), 4) for c in feature_columns],
                    "File average": [round(float(data[c].mean()), 4) for c in feature_columns],
                }
            )
            comparison["vs average"] = np.where(
                comparison["This case"] > comparison["File average"], "▲ above", "▼ below"
            )
            st.dataframe(comparison, use_container_width=True, height=460, hide_index=True)

        explain(
            "Why does confidence matter more than the label?",
            """
Every one of these models produces a *probability*, not a yes/no. The label you
see is just that probability pushed through a 50% cut-off.

A case predicted benign at **99%** and one predicted benign at **51%** get the
exact same label — but they are wildly different situations. The second one is
a case a doctor would want to look at again.

This is also what **AUC** measures: how well the model *ranks* cases by
suspicion, ignoring where the cut-off happens to sit. And it is why a hospital
might deliberately lower the threshold below 50% — catching more cancers at the
cost of more false alarms is usually the right trade.
""",
        )

# =========================================================================
# TAB 5 — MODEL COMPARISON
# =========================================================================
with tab_compare:
    st.header("🏆 All five models, head to head")

    if not has_data:
        needs_upload()
    else:
        metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        metrics_df = compute_all_metrics(X_scaled, y_true.values, models)

        st.subheader("Every metric, every model")
        st.caption(
            f"All five models scored on the {len(data)} rows you uploaded. Hover any "
            "bar for the exact value; click a model in the legend to hide it."
        )

        bar = go.Figure()
        for _, row in metrics_df.iterrows():
            name = row["ML Model Name"]
            bar.add_trace(
                go.Bar(
                    name=name,
                    x=metric_cols,
                    y=[row[c] for c in metric_cols],
                    marker=dict(color=MODEL_COLORS[name], line=dict(color=SURFACE, width=2)),
                    hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
                )
            )
        bar.update_layout(barmode="group", legend=dict(orientation="h", y=-0.18, x=0))
        bar.update_yaxes(range=[0.8, 1.0], title="Score")
        st.plotly_chart(style_fig(bar, height=440, showlegend=True), use_container_width=True)

        best_row = metrics_df.loc[metrics_df["MCC"].idxmax()]
        st.success(
            f"🥇 **Best overall: {best_row['ML Model Name']}** — highest MCC "
            f"({best_row['MCC']:.4f}), the metric that is hardest to fake on "
            "uneven classes."
        )

        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        explain(
            "Are these the same numbers as in the README?",
            """
They will match exactly **when you upload `test_data.csv`**, because that is the
same held-out split the models were scored on at training time.

Upload a different file and these numbers will change — everything on this page
is recomputed from whatever you provide, not read back from a saved results
file. That is the point: the table reflects your data, not a frozen number.
""",
        )

        st.markdown("---")
        st.subheader("ROC curves — the ranking test")
        st.caption(
            "Each line traces the trade-off between catching cancers (up) and "
            "raising false alarms (right) as the cut-off moves. The closer a line "
            "hugs the top-left corner, the better. The grey diagonal is random guessing."
        )

        roc_fig = go.Figure()
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random guessing",
                line=dict(color=INK_MUTED, width=1.5, dash="dot"),
                hoverinfo="skip",
            )
        )
        for name, mdl in models.items():
            proba_all = mdl.predict_proba(X_scaled)[:, POS_LABEL]
            fpr, tpr, _ = roc_curve(y_binary, proba_all)
            model_auc = roc_auc_score(y_binary, proba_all)
            roc_fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{name} (AUC {model_auc:.3f})",
                    line=dict(color=MODEL_COLORS[name], width=2, dash=MODEL_DASH[name]),
                    hovertemplate=(
                        f"<b>{name}</b><br>false alarms: %{{x:.3f}}"
                        "<br>cancers caught: %{y:.3f}<extra></extra>"
                    ),
                )
            )
        roc_fig.update_xaxes(title="False alarm rate", range=[-0.02, 1.02])
        roc_fig.update_yaxes(title="Cancers caught (recall)", range=[-0.02, 1.02])
        roc_fig.update_layout(
            legend=dict(orientation="v", y=0.02, x=0.45, bgcolor=SURFACE, bordercolor=GRIDLINE, borderwidth=1)
        )
        st.plotly_chart(style_fig(roc_fig, height=520, showlegend=True), use_container_width=True)

        explain(
            "How do I read an ROC curve?",
            """
Imagine sliding the cut-off from "call everything benign" to "call everything
malignant". At each setting you get two numbers: how many real cancers you
caught, and how many false alarms you raised. Plotting every setting traces
one of these curves.

- **Top-left corner** = catching every cancer with zero false alarms. Perfect.
- **The grey diagonal** = pure guesswork.
- **Area under the curve (AUC)** = the number in the legend. It compresses the
  whole curve into one score.

Because these lines are so close together, hue alone would not be enough to tell
them apart — so each model also gets its own dash pattern.
""",
        )

        st.markdown("---")
        st.subheader(f"What is {model_name} actually paying attention to?")
        st.caption(
            "Each measurement is scrambled in turn to see how much the model's "
            "accuracy suffers. A bigger drop means the model relies on it more."
        )

        with st.spinner("Measuring feature importance…"):
            importances = compute_permutation_importance(
                model_name, models[model_name], X_scaled, y_true.values
            )

        imp_df = (
            pd.DataFrame({"Measurement": feature_columns, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(12)
            .sort_values("Importance")
        )

        imp_fig = go.Figure(
            go.Bar(
                x=imp_df["Importance"],
                y=imp_df["Measurement"],
                orientation="h",
                marker=dict(color=MODEL_COLORS[model_name], line=dict(color=SURFACE, width=2)),
                hovertemplate="<b>%{y}</b><br>accuracy drop when scrambled: %{x:.4f}<extra></extra>",
            )
        )
        imp_fig.update_xaxes(title="Drop in accuracy when this measurement is scrambled")
        st.plotly_chart(style_fig(imp_fig, height=460), use_container_width=True)

        explain(
            "Why measure importance this way?",
            """
This is **permutation importance**. Take a trained model, shuffle one
measurement's values across all the patients so it becomes meaningless noise,
and re-score. If accuracy collapses, that measurement was load-bearing. If
nothing happens, the model was ignoring it.

The advantage is that it works for **every** model. Decision trees and random
forests can report their own importances directly, but kNN and Naive Bayes
offer no such thing — scrambling the input works regardless of what is inside.

Switch models in the sidebar and watch the ranking change. Models that score
similarly can still be reading the data in quite different ways.
""",
        )

st.markdown("---")
st.caption(
    "Built by **Rahul Saxena** (2025ac05155@wilp.bits-pilani.ac.in) for ML Assignment 2 — "
    "Work Integrated Learning Programmes Division, BITS Pilani (M.Tech AIML/DSE)."
)
