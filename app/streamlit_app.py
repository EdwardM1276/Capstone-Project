import json
from pathlib import Path

import pandas as pd
import streamlit as st

from fraud_detection.config import DEFAULT_MAX_ROWS, FIGURE_DIR, MODEL_DIR, REPORT_DIR
from fraud_detection.models import MODEL_NAMES
from fraud_detection.prediction import load_best_model, predict_transaction

STRATEGY_LABELS = {
    "class_weight": "Class weighting",
    "smote": "SMOTE",
    "smote_tomek": "SMOTE with Tomek links",
}
CURRENT_RESULT_COLUMNS = {
    "model",
    "imbalance_strategy",
    "validation_weighted_cost",
    "validation_precision_at_recall_target",
    "validation_recall",
    "threshold",
    "pr_auc",
    "precision",
    "recall",
    "fraud_test_rows",
    "pr_auc_ci_low",
    "pr_auc_ci_high",
    "cost_per_1000_transactions",
    "prediction_rows_per_second",
    "pareto_efficient",
}

st.set_page_config(
    page_title="PaySim | Fraud Review",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --paper: #f4f7fb;
        --panel: #ffffff;
        --ink: #10233f;
        --muted: #43566f;
        --rule: #cbd7e5;
        --blue: #155eef;
        --signal: #b54708;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: var(--paper); }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--rule);
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #000000 !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] *,
    [data-testid="stSelectbox"] label,
    [data-testid="stSelectbox"] [data-baseweb="select"],
    [data-testid="stSelectbox"] [data-baseweb="select"] *,
    [data-baseweb="popover"] [role="option"] {
        color: #000000 !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapsedControl"] *,
    button[aria-label*="sidebar" i],
    button[aria-label*="sidebar" i] * {
        color: #155eef !important;
        fill: #155eef !important;
        stroke: #155eef !important;
    }
    [data-testid="stSidebarCollapseButton"]:hover svg,
    [data-testid="stSidebarCollapseButton"]:focus svg,
    [data-testid="stSidebarCollapseButton"]:active svg,
    [data-testid="stSidebarCollapsedControl"]:hover svg,
    [data-testid="stSidebarCollapsedControl"]:focus svg,
    [data-testid="stSidebarCollapsedControl"]:active svg,
    button[aria-label*="sidebar" i]:hover svg,
    button[aria-label*="sidebar" i]:focus svg,
    button[aria-label*="sidebar" i]:active svg {
        color: #155eef !important;
        fill: #155eef !important;
        stroke: #155eef !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #ffffff !important;
        border-color: #64748b !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] svg {
        fill: #000000 !important;
    }
    html, body, [class*="css"] {
        color: var(--ink);
        font-family: "Segoe UI", Tahoma, sans-serif;
        font-size: 16px;
    }
    h1, h2, h3 { color: var(--ink); font-family: Georgia, "Times New Roman", serif; }
    h1 { font-size: 2.1rem; }
    h2 { font-size: 1.55rem; }
    h3 { font-size: 1.2rem; }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--rule);
        border-radius: 3px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--blue); }
    [data-testid="stAlert"] { border-radius: 3px; }
    button[kind="primary"] {
        background: var(--blue);
        border: 1px solid var(--blue);
        border-radius: 3px;
    }
    button[kind="primary"] p { color: white; }
    div[data-baseweb="tab-list"] { gap: 0.3rem; }
    button[data-baseweb="tab"] { border-radius: 3px 3px 0 0; }
    div[data-baseweb="tab-list"] button[data-baseweb="tab"],
    div[data-baseweb="tab-list"] button[data-baseweb="tab"] p {
        color: #10233f !important;
        opacity: 1 !important;
    }
    div[data-baseweb="tab-list"] button[aria-selected="true"],
    div[data-baseweb="tab-list"] button[aria-selected="true"] p {
        color: #155eef !important;
    }
    div[data-baseweb="tab-highlight"] { background-color: #155eef !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _current_results() -> tuple[pd.DataFrame, dict] | None:
    summary = _read_json(REPORT_DIR / "training_summary.json")
    status = _read_json(REPORT_DIR / "run_status.json")
    comparison_path = REPORT_DIR / "model_comparison.csv"
    if summary is None or status is None or not comparison_path.is_file():
        return None
    if (
        status.get("status") != "complete"
        or status.get("run_id") != summary.get("run_id")
        or summary.get("split_strategy") != "chronological_by_step"
    ):
        return None
    try:
        results = pd.read_csv(comparison_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    if not CURRENT_RESULT_COLUMNS.issubset(results.columns):
        return None
    if results.empty or not results["imbalance_strategy"].isin(STRATEGY_LABELS).all():
        return None
    if len(results) != len(MODEL_NAMES) * len(STRATEGY_LABELS):
        return None
    if set(results["model"]) != set(MODEL_NAMES):
        return None
    if set(results["imbalance_strategy"]) != set(STRATEGY_LABELS):
        return None
    if results.duplicated(["model", "imbalance_strategy"]).any():
        return None
    if not results["test_rows"].eq(summary.get("test_rows")).all():
        return None
    if not results["fraud_test_rows"].eq(summary.get("fraud_test_rows")).all():
        return None
    return results, summary


def _current_model_available() -> bool:
    metadata = _read_json(MODEL_DIR / "best_model.json")
    summary = _read_json(REPORT_DIR / "training_summary.json")
    status = _read_json(REPORT_DIR / "run_status.json")
    return bool(
        metadata
        and summary
        and status
        and status.get("status") == "complete"
        and metadata.get("run_id") == summary.get("run_id") == status.get("run_id")
        and metadata.get("split_strategy") == "chronological_by_step"
        and (MODEL_DIR / "best_model.joblib").is_file()
    )


@st.cache_resource
def get_model(model_path: str, modified_ns: int):
    return load_best_model(Path(model_path))


def _training_command() -> str:
    return f"python -m fraud_detection.cli train --max-rows {DEFAULT_MAX_ROWS}"


def _show_stale_run_notice() -> None:
    st.info(
        "This workspace has no results from the current chronological evaluation "
        "and pre-transaction feature workflow. Previous results are hidden to avoid "
        "mixing incompatible runs."
    )
    st.code(_training_command(), language="powershell")


def render_overview() -> None:
    st.title("Transaction risk, evaluated over time")
    st.write(
        "A PaySim fraud-classification study comparing five models and three "
        "imbalance treatments with chronological validation and a final future period."
    )

    current = _current_results()
    quality = _read_json(REPORT_DIR / "data_quality.json")
    if current is None:
        st.info("The current evaluation has not been run yet.")
        st.code(_training_command(), language="powershell")
        if quality:
            distribution = quality.get("class_distribution", {})
            col_a, col_b = st.columns(2)
            col_a.metric("Source transactions", f"{quality.get('row_count', 0):,}")
            col_b.metric("Fraud cases", f"{int(distribution.get('1', 0)):,}")
        st.subheader("Evaluation policy")
        st.write(
            "Each strategy uses the same sampled rows and fixed model settings. "
            "Thresholds are chosen on the middle time window to meet the recall "
            "target. The latest window is reserved for final reporting."
        )
        return

    results, summary = current
    distribution = quality.get("class_distribution", {}) if quality else {}
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Transactions evaluated", f"{summary.get('sampled_rows', 0):,}")
    col_b.metric("Source fraud cases", f"{int(distribution.get('1', 0)):,}")
    col_c.metric("Fraud cases in final window", f"{summary.get('fraud_test_rows', 0):,}")

    st.subheader("Selected operating point")
    best = results.sort_values(
        ["validation_weighted_cost", "validation_precision_at_recall_target"],
        ascending=[True, False],
    ).iloc[0]
    st.write(
        f"{best['model']} with {STRATEGY_LABELS[best['imbalance_strategy']]} "
        f"was selected using validation cost. Final-window PR-AUC was "
        f"{best['pr_auc']:.3f}; recall was {best['recall']:.1%}."
    )
    st.caption(
        f"Time ranges by step: train {summary['split_step_ranges']['training']}, "
        f"validation {summary['split_step_ranges']['validation']}, "
        f"final test {summary['split_step_ranges']['test']}. The final test did "
        "not select the model or threshold."
    )


def render_prediction() -> None:
    st.title("Score a transaction")
    st.write(
        "Enter only fields available before the payment completes. The saved "
        "threshold was selected on the validation period."
    )
    if not _current_model_available():
        st.info(
            "No compatible model has been trained with the current chronological "
            "split and pre-transaction feature schema."
        )
        st.code(_training_command(), language="powershell")
        return

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            step = st.number_input("PaySim hour step", min_value=0, value=1, step=1)
            transaction_type = st.selectbox(
                "Transaction type",
                ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
            )
            amount = st.number_input(
                "Transaction amount", min_value=0.0, value=1000.0, step=100.0
            )
        with right:
            oldbalance_org = st.number_input(
                "Origin balance before", min_value=0.0, value=5000.0, step=100.0
            )
            oldbalance_dest = st.number_input(
                "Destination balance before", min_value=0.0, value=1000.0, step=100.0
            )
        submitted = st.form_submit_button("Score transaction", type="primary")

    if not submitted:
        return

    model_path = MODEL_DIR / "best_model.joblib"
    transaction = {
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "oldbalanceDest": oldbalance_dest,
    }
    try:
        result = predict_transaction(
            get_model(str(model_path), model_path.stat().st_mtime_ns), transaction
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        st.error(f"Could not score this transaction: {error}")
        return

    probability_col, threshold_col = st.columns(2)
    probability_col.metric("Fraud score", f"{result['fraud_probability']:.1%}")
    threshold_col.metric("Decision threshold", f"{result['threshold']:.1%}")
    if result["prediction"] == 1:
        st.error("Flag for review")
    else:
        st.success("Below the review threshold")
    st.caption(
        "Scores are estimates from synthetic PaySim data, not production decisions."
    )


def render_model_results() -> None:
    st.title("Compare treatments within each algorithm")
    current = _current_results()
    if current is None:
        _show_stale_run_notice()
        return
    results, summary = current
    st.caption(
        f"{summary['sampled_rows']:,} rows evaluated. The final period contains "
        f"{summary['fraud_test_rows']:,} fraud cases. Models are ranked by "
        "validation cost, not by the final test results."
    )
    model_order = [model for model in MODEL_NAMES if model in results["model"].unique()]
    model_tabs = st.tabs(model_order)
    display_columns = [
        "model",
        "validation_precision_at_recall_target",
        "validation_recall",
        "validation_weighted_cost",
        "threshold",
        "precision",
        "recall",
        "pr_auc",
        "pr_auc_ci_low",
        "pr_auc_ci_high",
        "cost_per_1000_transactions",
        "fit_seconds",
        "prediction_rows_per_second",
    ]
    metric_choices = {
        "Final test PR-AUC": "pr_auc",
        "Final test recall": "recall",
        "Final test precision": "precision",
        "Validation precision at recall target": "validation_precision_at_recall_target",
        "Validation weighted cost": "validation_weighted_cost",
        "Final test cost per 1,000": "cost_per_1000_transactions",
        "Scoring throughput": "prediction_rows_per_second",
    }
    for tab, model in zip(model_tabs, model_order):
        with tab:
            st.subheader(model)
            model_results = results.loc[results["model"] == model].copy()
            model_results["treatment"] = model_results["imbalance_strategy"].map(
                STRATEGY_LABELS
            )
            metric_label = st.selectbox(
                "Compare treatments by",
                list(metric_choices),
                key=f"metric_{model}",
            )
            metric_column = metric_choices[metric_label]
            chart = model_results.set_index("treatment")[[metric_column]]
            st.bar_chart(chart, color="#155eef", horizontal=True)
            strategy_results = model_results.sort_values(
                ["validation_weighted_cost", "validation_precision_at_recall_target"],
                ascending=[True, False],
            )
            st.dataframe(
                strategy_results[[
                    column for column in display_columns if column in strategy_results
                ]].style.format(precision=3),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "All three treatments share the same rows and time windows. "
                "Lower cost is preferable; higher scores and throughput are preferable."
            )


def render_visualizations() -> None:
    st.title("Consolidated comparative visuals")
    current = _current_results()
    if current is None:
        _show_stale_run_notice()
        return
    _, summary = current
    performance_path = FIGURE_DIR / "performance_dashboard.png"
    pareto_path = FIGURE_DIR / "pareto_dashboard.png"
    score_map_path = FIGURE_DIR / "model_strategy_pr_auc_heatmap.png"
    risk_map_path = FIGURE_DIR / "probability_band_risk_map.png"
    visual_tabs = st.tabs(["Performance by algorithm", "Pareto frontiers", "High-score risk"])
    with visual_tabs[0]:
        if performance_path.is_file():
            st.image(str(performance_path), caption="Six holdout and operating metrics, grouped by algorithm")
        else:
            st.warning("The performance dashboard is missing. Rerun the current training pipeline.")
    with visual_tabs[1]:
        if pareto_path.is_file():
            st.image(str(pareto_path), caption="Final-period quality, speed, recall, and cost trade-offs")
        else:
            st.warning("The Pareto dashboard is missing. Rerun the current training pipeline.")
    with visual_tabs[2]:
        if risk_map_path.is_file():
            st.image(str(risk_map_path), caption="Observed fraud rate among high-score transactions")
        st.dataframe(
            pd.read_csv(REPORT_DIR / "probability_bands.csv"),
            width="stretch",
            hide_index=True,
        )
    if score_map_path.is_file():
        st.image(str(score_map_path), caption="PR-AUC matrix across all 15 pairings")
    st.caption(
        f"Run {summary['run_id']} completed {summary['completed_at']}. "
        "Pareto dominance is calculated on the final period for display, not model selection."
    )


page = st.sidebar.radio(
    "Project sections",
    ["Overview", "Model comparison", "Visual analysis", "Transaction scoring"],
)
if page == "Overview":
    render_overview()
elif page == "Model comparison":
    render_model_results()
elif page == "Visual analysis":
    render_visualizations()
else:
    render_prediction()