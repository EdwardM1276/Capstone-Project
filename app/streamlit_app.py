import hashlib
import json
import math
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from fraud_detection.config import (
    DEFAULT_MAX_ROWS,
    FIGURE_DIR,
    MODEL_DIR,
    REPORT_DIR,
)
from fraud_detection.models import MODEL_NAMES
from fraud_detection.prediction import load_best_model, predict_transaction
from fraud_detection.review import (
    build_uploaded_alert_queue,
    calculate_uploaded_batch_impact,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #0f172a;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(21,94,239,0.08), transparent 45%),
                radial-gradient(circle at top right, rgba(16,185,129,0.08), transparent 45%),
                #f5f7fb;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        }

        section[data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }

        .app-title {
            font-size: 2.35rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.15rem;
        }

        .app-subtitle {
            font-size: 1.02rem;
            color: #475569;
            margin-bottom: 1.4rem;
        }

        .metric-card {
            background: white;
            border-radius: 18px;
            padding: 1.2rem 1.3rem;
            border: 1px solid rgba(148,163,184,0.18);
            box-shadow: 0 10px 28px rgba(15,23,42,0.07);
            height: 100%;
        }

        .metric-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748b;
        }

        .metric-value {
            font-size: 1.9rem;
            font-weight: 800;
            color: #155eef;
            margin-top: 0.25rem;
        }

        .metric-note {
            font-size: 0.82rem;
            color: #64748b;
            margin-top: 0.25rem;
        }

        .panel {
            background: white;
            border-radius: 20px;
            padding: 1.5rem;
            border: 1px solid rgba(148,163,184,0.16);
            box-shadow: 0 10px 28px rgba(15,23,42,0.06);
            margin-bottom: 1.2rem;
        }

        .panel-title {
            font-size: 1.12rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.8rem;
        }

        .risk-high {
            background: linear-gradient(135deg, #dc2626, #b91c1c);
            color: white;
            border-radius: 18px;
            padding: 1.4rem;
        }

        .risk-low {
            background: linear-gradient(135deg, #16a34a, #15803d);
            color: white;
            border-radius: 18px;
            padding: 1.4rem;
        }

        .footer-note {
            text-align: center;
            color: #64748b;
            font-size: 0.84rem;
            margin-top: 2rem;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 8px 22px rgba(15,23,42,0.05);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: #e2e8f0;
            border-radius: 999px;
            padding: 0.45rem 1rem;
        }

        .stTabs [aria-selected="true"] {
            background: #155eef !important;
        }

        .stTabs [aria-selected="true"] p {
            color: white !important;
        }
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


def _page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="app-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def _metric_card(
    label: str,
    value: str,
    note: str = "",
    colour: str = "#155eef",
) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{colour};">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    _page_header(
        "Transaction risk, evaluated over time",
        "A PaySim fraud-classification study comparing models and imbalance treatments "
        "using chronological validation and a final future period.",
    )

    current = _current_results()
    quality = _read_json(REPORT_DIR / "data_quality.json")

    if current is None:
        st.info("The current evaluation has not been run yet.")
        st.code(_training_command(), language="powershell")

        if quality:
            distribution = quality.get("class_distribution", {})

            col_a, col_b = st.columns(2)

            with col_a:
                _metric_card(
                    "Source transactions",
                    f"{quality.get('row_count', 0):,}",
                    "Rows in the sampled PaySim dataset",
                )

            with col_b:
                _metric_card(
                    "Fraud cases",
                    f"{int(distribution.get('1', 0)):,}",
                    "Confirmed fraudulent transactions",
                    "#dc2626",
                )

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Evaluation policy</div>', unsafe_allow_html=True)
        st.write(
            "Each strategy uses the same sampled rows and fixed model settings. "
            "Review thresholds are chosen on the middle time window to meet the recall "
            "target. The latest window is reserved for final reporting."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        return

    results, summary = current
    quality = _read_json(REPORT_DIR / "data_quality.json")
    distribution = quality.get("class_distribution", {}) if quality else {}

    best = results.sort_values(
        ["validation_weighted_cost", "validation_precision_at_recall_target"],
        ascending=[True, False],
    ).iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        _metric_card(
            "Transactions evaluated",
            f"{summary.get('sampled_rows', 0):,}",
            "Rows used in the chronological evaluation",
        )

    with col2:
        _metric_card(
            "Source fraud cases",
            f"{int(distribution.get('1', 0)):,}",
            "Fraud cases in the sampled dataset",
            "#dc2626",
        )

    with col3:
        _metric_card(
            "Fraud cases in final window",
            f"{summary.get('fraud_test_rows', 0):,}",
            "Fraud cases available for final testing",
            "#b54708",
        )

    with col4:
        _metric_card(
            "Final PR-AUC",
            f"{best['pr_auc']:.3f}",
            f"{best['model']} · {STRATEGY_LABELS[best['imbalance_strategy']]}",
            "#16a34a",
        )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Selected operating point</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Winning model", best["model"])

    with col_b:
        st.metric("Imbalance strategy", STRATEGY_LABELS[best["imbalance_strategy"]])

    with col_c:
        st.metric("Final recall", f"{best['recall']:.1%}")

    st.success(
        f"{best['model']} with {STRATEGY_LABELS[best['imbalance_strategy']]} was selected "
        f"using validation cost. Final-window PR-AUC was {best['pr_auc']:.3f} and "
        f"recall was {best['recall']:.1%}."
    )

    st.caption(
        f"Time ranges by step: train {summary['split_step_ranges']['training']}, "
        f"validation {summary['split_step_ranges']['validation']}, "
        f"final test {summary['split_step_ranges']['test']}. The final test did "
        "not select the model or review threshold."
    )

    st.markdown('</div>', unsafe_allow_html=True)


def render_fraud_review() -> None:
    _page_header(
        "Fraud review & cost simulator",
        "Score a single transaction or upload a CSV, then review prioritised "
        "alerts and estimate the operational impact of fraud review.",
    )

    render_prediction()

    st.divider()

    render_alert_triage()


def render_prediction() -> None:
    if not _current_model_available():
        st.info(
            "No compatible saved model is available for the current evaluation."
        )
        st.code(_training_command(), language="powershell")
        return

    model_path = MODEL_DIR / "best_model.joblib"

    try:
        model_version = model_path.stat().st_mtime_ns
        trained_model = get_model(str(model_path), model_version)
    except Exception as error:
        st.error(f"Could not load the saved model: {error}")
        return

    required_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
    ]

    allowed_types = [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    ]

    def score_transaction(transaction):
        result = predict_transaction(trained_model, transaction)

        probability = float(result["fraud_probability"])
        threshold = float(result["threshold"])

        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("The model returned an invalid fraud score.")

        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("The saved review threshold is invalid.")

        return probability, threshold

    mode = st.radio(
        "Choose how to score transactions",
        ["Single transaction", "Upload transactions"],
        horizontal=True,
        key="transaction_scoring_mode",
    )
    st.session_state["transaction_batch_active"] = False

    if mode == "Single transaction":
        st.subheader("Enter transaction details")
        st.caption(
            "After changing a number, press Enter or click outside the "
            "field to update the result."
        )

        left, right = st.columns(2)

        with left:
            step = st.number_input(
                "PaySim hour step",
                min_value=0,
                value=1,
                step=1,
                key="single_score_step",
            )

            transaction_type = st.selectbox(
                "Transaction type",
                allowed_types,
                key="single_score_type",
            )

            amount = st.number_input(
                "Transaction amount",
                min_value=0.0,
                value=1000.0,
                step=100.0,
                format="%.2f",
                key="single_score_amount",
            )

        with right:
            oldbalance_org = st.number_input(
                "Sender balance before",
                min_value=0.0,
                value=5000.0,
                step=100.0,
                format="%.2f",
                key="single_score_origin",
            )

            oldbalance_dest = st.number_input(
                "Recipient balance before",
                min_value=0.0,
                value=1000.0,
                step=100.0,
                format="%.2f",
                key="single_score_destination",
            )

        transaction = {
            "step": int(step),
            "type": transaction_type,
            "amount": float(amount),
            "oldbalanceOrg": float(oldbalance_org),
            "oldbalanceDest": float(oldbalance_dest),
        }

        try:
            probability, threshold = score_transaction(transaction)
        except Exception as error:
            st.error(f"Could not score this transaction: {error}")
            return

        flagged = probability >= threshold

        st.divider()
        st.subheader("Fraud-risk result")

        score_col, threshold_col = st.columns(2)
        score_col.metric("Fraud score", f"{probability:.4%}")
        threshold_col.metric("Review threshold", f"{threshold:.4%}")

        st.progress(probability)

        if flagged:
            st.warning(
                "Flag for review — this transaction meets or exceeds "
                "the model's review threshold."
            )
        else:
            st.success(
                "Below the review threshold — this transaction has "
                "not been flagged by the model."
            )

        st.caption(
            "A model score is not proof of fraud. Being below the review "
            "threshold does not guarantee that a transaction is legitimate."
        )

        return

    st.subheader("Upload several transactions")

    st.write(
        "Upload one CSV file with one transaction per row. "
        "The app will automatically score every valid row and show the results."
    )

    st.caption(
        "Required columns: step, type, amount, oldbalanceOrg, oldbalanceDest. "
        "Other columns are ignored when predicting."
    )

    template = pd.DataFrame(
        [
            {
                "step": 600,
                "type": "TRANSFER",
                "amount": 10000.0,
                "oldbalanceOrg": 12000.0,
                "oldbalanceDest": 0.0,
            },
            {
                "step": 601,
                "type": "PAYMENT",
                "amount": 250.0,
                "oldbalanceOrg": 5000.0,
                "oldbalanceDest": 1000.0,
            },
        ]
    )

    st.download_button(
        "Download example CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="transaction_upload_template.csv",
        mime="text/csv",
        key="download_transaction_template",
    )

    st.info(
        "Use synthetic or authorised data only. Avoid uploading customer "
        "names, account numbers, or other sensitive personal information."
    )

    uploaded_file = st.file_uploader(
        "Choose a transaction CSV",
        type=["csv"],
        key="transaction_csv_upload",
    )

    if uploaded_file is None:
        return

    file_bytes = uploaded_file.getvalue()

    if len(file_bytes) > 10 * 1024 * 1024:
        st.error(
            "This page accepts files up to 10 MB. "
            "Split your CSV into smaller files."
        )
        return

    try:
        transactions = pd.read_csv(
            BytesIO(file_bytes),
            nrows=5001,
        )
        transactions.columns = transactions.columns.str.strip()
    except Exception as error:
        st.error(f"Could not read the CSV: {error}")
        return

    if transactions.empty:
        st.warning("The CSV contains no transaction rows.")
        return

    if len(transactions) > 5000:
        st.error(
            "This page supports up to 5,000 transactions per upload. "
            "Split your file into smaller batches. No rows have been scored."
        )
        return

    if transactions.columns.duplicated().any():
        st.error("The CSV contains duplicate column names.")
        return

    missing = [
        column
        for column in required_columns
        if column not in transactions.columns
    ]

    if missing:
        st.error(
            "Missing required columns: " + ", ".join(missing)
        )
        st.write("Use the downloadable template to correct the file.")
        return

    model_inputs = transactions[required_columns].copy()

    model_inputs["type"] = (
        model_inputs["type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    numeric_columns = [
        "step",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
    ]

    for column in numeric_columns:
        model_inputs[column] = pd.to_numeric(
            model_inputs[column],
            errors="coerce",
        )

    batch_key = (
        hashlib.sha256(file_bytes).hexdigest(),
        str(model_path),
        model_version,
    )

    if st.session_state.get("transaction_batch_key") != batch_key:
        scored_rows = []
        progress = st.progress(0.0)
        status = st.empty()

        for position, (_, row) in enumerate(model_inputs.iterrows(), start=1):
            output = {
                "CSV row": position + 1,
                "step": row["step"],
                "type": row["type"],
                "amount": row["amount"],
                "oldbalanceOrg": row["oldbalanceOrg"],
                "oldbalanceDest": row["oldbalanceDest"],
                "Fraud score (%)": None,
                "Review threshold (%)": None,
                "Decision": "Not scored",
                "Error": "",
            }

            try:
                for column in numeric_columns:
                    value = float(row[column])

                    if not math.isfinite(value) or value < 0:
                        raise ValueError(
                            f"{column} must be a non-negative, finite number."
                        )

                if float(row["step"]) % 1 != 0:
                    raise ValueError("step must be a whole number.")

                if pd.isna(row["type"]) or row["type"] not in allowed_types:
                    raise ValueError(
                        "type must be CASH_IN, CASH_OUT, DEBIT, "
                        "PAYMENT, or TRANSFER."
                    )

                transaction = {
                    "step": int(row["step"]),
                    "type": str(row["type"]),
                    "amount": float(row["amount"]),
                    "oldbalanceOrg": float(row["oldbalanceOrg"]),
                    "oldbalanceDest": float(row["oldbalanceDest"]),
                }

                probability, threshold = score_transaction(transaction)

                output["Fraud score (%)"] = probability * 100
                output["Review threshold (%)"] = threshold * 100
                output["Decision"] = (
                    "Flag for review"
                    if probability >= threshold
                    else "Below review threshold"
                )

            except Exception as error:
                output["Error"] = str(error)

            scored_rows.append(output)

            progress.progress(position / len(model_inputs))
            status.caption(
                f"Checked {position:,} of {len(model_inputs):,} transactions."
            )

        progress.empty()
        status.empty()

        st.session_state["transaction_batch_results"] = pd.DataFrame(
            scored_rows
        )
        st.session_state["transaction_batch_key"] = batch_key
        st.session_state["analyst_feedback"] = []

    results = st.session_state["transaction_batch_results"].copy()
    st.session_state["transaction_batch_source_name"] = uploaded_file.name
    st.session_state["transaction_batch_active"] = True

    flagged_count = int(
        results["Decision"].eq("Flag for review").sum()
    )
    below_count = int(
        results["Decision"].eq("Below review threshold").sum()
    )
    failed_count = int(
        results["Decision"].eq("Not scored").sum()
    )

    st.divider()
    st.subheader("Upload results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions uploaded", f"{len(results):,}")
    col2.metric("Flagged for review", f"{flagged_count:,}")
    col3.metric("Below review threshold", f"{below_count:,}")
    col4.metric("Not scored", f"{failed_count:,}")

    if failed_count:
        st.warning(
            f"{failed_count:,} rows could not be scored. "
            "Check the Error column, correct those rows, and upload again. "
            "Failed rows are not classified as safe."
        )
    else:
        st.success("All uploaded transactions were scored successfully.")

    result_filter = st.selectbox(
        "Show",
        [
            "All transactions",
            "Flag for review",
            "Below review threshold",
            "Not scored",
        ],
        key="batch_result_filter",
    )

    display_results = results.copy()

    if result_filter != "All transactions":
        display_results = display_results[
            display_results["Decision"] == result_filter
        ]

    display_results = display_results.sort_values(
        "Fraud score (%)",
        ascending=False,
        na_position="last",
    )

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fraud score (%)": st.column_config.NumberColumn(
                "Fraud score (%)",
                format="%.4f",
            ),
            "Review threshold (%)": st.column_config.NumberColumn(
                "Review threshold (%)",
                format="%.4f",
            ),
        },
    )

    st.download_button(
        "Download all transaction results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="transaction_scoring_results.csv",
        mime="text/csv",
        key="download_all_scoring_results",
    )

    st.caption(
        "These results use the saved model and its saved review threshold. "
        "They are not confirmed fraud labels. Download results before "
        "ending your session."
    )


def render_model_results() -> None:
    _page_header(
        "Compare treatments within each algorithm",
        "Compare imbalance-handling strategies for every model using validation "
        "and final-period results.",
    )

    current = _current_results()

    if current is None:
        _show_stale_run_notice()
        return

    results, summary = current

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.caption(
        f"{summary['sampled_rows']:,} rows evaluated. The final period contains "
        f"{summary['fraud_test_rows']:,} fraud cases. Models are ranked by "
        "validation cost, not by the final test results."
    )

    model_order = [
        model for model in MODEL_NAMES if model in results["model"].unique()
    ]
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
            st.markdown(f"### {model}")

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
                strategy_results[
                    [column for column in display_columns if column in strategy_results]
                ].style.format(precision=3),
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "All three treatments share the same rows and time windows. "
                "Lower cost is preferable; higher scores and throughput are preferable."
            )

    st.markdown('</div>', unsafe_allow_html=True)


def render_visualizations() -> None:
    _page_header(
        "Consolidated comparative visuals",
        "Explore model performance, trade-offs, and high-score fraud risk.",
    )

    current = _current_results()

    if current is None:
        _show_stale_run_notice()
        return

    _, summary = current

    performance_path = FIGURE_DIR / "performance_dashboard.png"
    pareto_path = FIGURE_DIR / "pareto_dashboard.png"
    score_map_path = FIGURE_DIR / "model_strategy_pr_auc_heatmap.png"
    risk_map_path = FIGURE_DIR / "probability_band_risk_map.png"

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    visual_tabs = st.tabs(
        [
            "Performance by algorithm",
            "Pareto frontiers",
            "High-score risk",
        ]
    )

    with visual_tabs[0]:
        if performance_path.is_file():
            st.image(
                str(performance_path),
                caption="Six holdout and operating metrics, grouped by algorithm",
                use_container_width=True,
            )
        else:
            st.warning(
                "The performance dashboard is missing. Rerun the current training pipeline."
            )

    with visual_tabs[1]:
        if pareto_path.is_file():
            st.image(
                str(pareto_path),
                caption="Final-period quality, speed, recall, and cost trade-offs",
                use_container_width=True,
            )
        else:
            st.warning(
                "The Pareto dashboard is missing. Rerun the current training pipeline."
            )

    with visual_tabs[2]:
        if risk_map_path.is_file():
            st.image(
                str(risk_map_path),
                caption="Observed fraud rate among high-score transactions",
                use_container_width=True,
            )

        probability_bands_path = REPORT_DIR / "probability_bands.csv"

        if probability_bands_path.is_file():
            st.dataframe(
                pd.read_csv(probability_bands_path),
                width="stretch",
                hide_index=True,
            )
        else:
            st.warning("Probability-band results were not found.")

    if score_map_path.is_file():
        st.image(
            str(score_map_path),
            caption="PR-AUC matrix across all 15 pairings",
            use_container_width=True,
        )

    st.caption(
        f"Run {summary['run_id']} completed {summary['completed_at']}. "
        "Pareto dominance is calculated on the final period for display, "
        "not model selection."
    )

    st.markdown('</div>', unsafe_allow_html=True)


def render_explainability() -> None:
    _page_header(
        "Explainability and XAI",
        "Understand how the winning fraud-detection model makes predictions "
        "using SHAP and LIME.",
    )

    metadata = _read_json(MODEL_DIR / "best_model.json")
    interpretability = _read_json(
        ROOT_DIR / "evidence_deliverable_5" / "interpretability_summary.json"
    )

    if metadata is None:
        st.info("No compatible saved model was found.")
        st.code(_training_command(), language="powershell")
        return

    test_metrics = metadata.get("metrics", {})
    validation_metrics = metadata.get("validation_metrics", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        _metric_card(
            "Winning model",
            metadata.get("model", "Not available"),
            STRATEGY_LABELS.get(
                metadata.get("imbalance_strategy", ""),
                metadata.get("imbalance_strategy", "Not available"),
            ),
        )

    with col2:
        _metric_card(
            "Review threshold",
            f"{metadata.get('threshold', 0):.4f}",
            "Selected on validation data",
            "#b54708",
        )

    with col3:
        _metric_card(
            "Test recall",
            f"{test_metrics.get('recall', 0):.1%}",
            "Fraud cases detected",
            "#16a34a",
        )

    with col4:
        _metric_card(
            "Test PR-AUC",
            f"{test_metrics.get('pr_auc', 0):.4f}",
            "Fraud-ranking performance",
        )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Winning model details</div>', unsafe_allow_html=True)

    winning_model = pd.DataFrame(
        [
            ["Winning model", metadata.get("model", "Not available")],
            [
                "Imbalance strategy",
                STRATEGY_LABELS.get(
                    metadata.get("imbalance_strategy"),
                    metadata.get("imbalance_strategy", "Not available"),
                ),
            ],
            [
                "Selection metric",
                metadata.get("selection_metric", "Not available"),
            ],
            ["Review threshold", f"{metadata.get('threshold', 0):.4f}"],
            [
                "Validation precision",
                f"{validation_metrics.get('precision', 0):.4f}",
            ],
            [
                "Validation recall",
                f"{validation_metrics.get('recall', 0):.4f}",
            ],
            ["Test precision", f"{test_metrics.get('precision', 0):.4f}"],
            ["Test recall", f"{test_metrics.get('recall', 0):.4f}"],
            ["Test F1-score", f"{test_metrics.get('f1', 0):.4f}"],
            ["Test ROC-AUC", f"{test_metrics.get('roc_auc', 0):.4f}"],
            ["Test PR-AUC", f"{test_metrics.get('pr_auc', 0):.4f}"],
            [
                "Chronological split",
                metadata.get("split_strategy", "Not available"),
            ],
        ],
        columns=["Item", "Value"],
    )

    st.dataframe(winning_model, width="stretch", hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    shap_tab, lime_tab, insight_tab = st.tabs(
        [
            "SHAP global explanation",
            "LIME local explanation",
            "Business interpretation",
        ]
    )

    shap_path = ROOT_DIR / "evidence_deliverable_5" / "figures" / "shap_summary_plot.png"
    lime_path = ROOT_DIR / "evidence_deliverable_5" / "figures" / "lime_local_explanation.png"

    with shap_tab:
        if shap_path.is_file():
            st.image(
                str(shap_path),
                caption="SHAP summary plot for the winning XGBoost model",
                use_container_width=True,
            )
        else:
            st.warning(
                "SHAP image not found. Run scripts/create_interpretability_evidence.py."
            )

        if interpretability:
            top_features = interpretability.get("top_shap_features", [])

            if top_features:
                shap_table = pd.DataFrame(
                    [
                        {
                            "Feature": item["feature"],
                            "Mean absolute SHAP value": round(
                                item["mean_abs_shap"], 4
                            ),
                        }
                        for item in top_features
                    ]
                )

                st.markdown(
                    '<div class="panel-title">Most influential features</div>',
                    unsafe_allow_html=True,
                )
                st.bar_chart(
                    shap_table.set_index("Feature"),
                    color="#155eef",
                    horizontal=True,
                )
                st.dataframe(shap_table, width="stretch", hide_index=True)

        st.info(
            "SHAP shows which features most often influence the model’s fraud "
            "predictions. The strongest feature was transaction_to_origin_balance, "
            "followed by amount, type_TRANSFER, hour, and oldbalanceDest. These "
            "explanations describe model behaviour; they do not prove that a "
            "feature causes fraud."
        )

    with lime_tab:
        if lime_path.is_file():
            st.image(
                str(lime_path),
                caption="LIME local explanation for one transaction predicted as fraud",
                use_container_width=True,
            )
        else:
            st.warning(
                "LIME image not found. Run scripts/create_interpretability_evidence.py."
            )

        if interpretability:
            probability = interpretability.get(
                "explained_transaction_predicted_probability"
            )
            true_label = interpretability.get("explained_transaction_true_label")

            if probability is not None:
                st.metric(
                    "Explained transaction fraud score",
                    f"{float(probability):.1%}",
                )

            if true_label == 0:
                st.warning(
                    "The explained transaction was actually legitimate. It is a "
                    "false positive, showing why human review is important before "
                    "blocking a transaction."
                )

        st.caption(
            "LIME explains one individual prediction. It is a local approximation "
            "of model behaviour, not proof that the transaction was fraudulent."
        )

    with insight_tab:
        st.markdown(
            '<div class="panel-title">What the model learned</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            - **Transaction size relative to sender balance** is the strongest fraud signal.
            - Large **TRANSFER** transactions receive higher fraud scores.
            - The **transaction amount** strongly influences the prediction.
            - The **hour** of the transaction affects the fraud score.
            - The recipient’s starting balance also contributes to the prediction.
            """
        )

        st.markdown(
            '<div class="panel-title">How to use these explanations</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            SHAP explains the model’s behaviour across many transactions.
            LIME explains why one specific transaction received its fraud probability.

            These explanations support transparency, but they do not prove that a
            transaction is fraudulent. They should support human decision-making.
            """
        )

    st.markdown('</div>', unsafe_allow_html=True)


def render_alert_triage() -> None:
    st.subheader("Human-in-the-loop alert triage")

    current = _current_results()

    if current is None:
        _show_stale_run_notice()
        return

    results, _ = current

    if not st.session_state.get("transaction_batch_active"):
        st.info(
            "Upload and score a CSV above to create a queue from those transactions. "
            "The analyst workspace will review alerts from that same scored batch."
        )
        return

    scored_batch = st.session_state.get("transaction_batch_results")
    batch_key = st.session_state.get("transaction_batch_key")
    model_info = _read_json(MODEL_DIR / "best_model.json")
    if scored_batch is None or batch_key is None:
        st.error("The active upload has no scored transaction batch. Upload and score it again.")
        return
    if model_info is None:
        st.error("The saved model metadata is missing. The uploaded batch cannot be matched to its model.")
        return

    model = str(model_info["model"])
    strategy = str(model_info["imbalance_strategy"])
    selected_rows = results.loc[
        (results["model"] == model)
        & (results["imbalance_strategy"] == strategy)
    ]
    if selected_rows.empty:
        st.error(
            "The active upload's saved model does not match the current evaluation. "
            "Refresh the model results and score the CSV again."
        )
        return
    batch_id = hashlib.sha256(repr(batch_key).encode("utf-8")).hexdigest()[:16]
    source_name = st.session_state.get("transaction_batch_source_name", "uploaded CSV")
    alert_queue = build_uploaded_alert_queue(scored_batch)
    scored_count = int(
        pd.to_numeric(scored_batch["Fraud score (%)"], errors="coerce").notna().sum()
    )
    failed_count = len(scored_batch) - scored_count

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">Scored upload for analyst review</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        f"{source_name}: {scored_count:,} of {len(scored_batch):,} row(s) scored, "
        f"{len(alert_queue):,} sent for review. Scores use {model} with "
        f"{STRATEGY_LABELS.get(strategy, strategy)} at a "
        f"{float(model_info['threshold']):.2%} review threshold. "
        "The analyst queue below contains only rows flagged from this upload."
    )
    if failed_count:
        st.warning(
            f"{failed_count:,} row(s) were not scored and are excluded from the queue. "
            "They must be corrected and uploaded again; they are not treated as safe."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">Priority alert queue</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Alerts are ranked using fraud probability, transaction-value risk, and "
        "model uncertainty. Higher-priority alerts should be reviewed first."
    )

    risk_filter_options = ["All alerts"] + [
        f"{band} risk"
        for band in ("High", "Medium", "Low")
        if alert_queue["risk_band"].eq(band).any()
    ]
    risk_filter = st.selectbox(
        "Filter by risk band",
        risk_filter_options,
        key=f"triage_risk_filter_{batch_id}",
    )

    if risk_filter == "High risk":
        visible_alerts = alert_queue.loc[alert_queue["risk_band"] == "High"]
    elif risk_filter == "Medium risk":
        visible_alerts = alert_queue.loc[alert_queue["risk_band"] == "Medium"]
    elif risk_filter == "Low risk":
        visible_alerts = alert_queue.loc[alert_queue["risk_band"] == "Low"]
    else:
        visible_alerts = alert_queue

    st.dataframe(
        visible_alerts[
            [
                "alert_id",
                "type",
                "amount",
                "fraud_probability",
                "risk_band",
                "priority_score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "amount": st.column_config.NumberColumn(
                "Amount",
                format="R %.2f",
            ),
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud probability",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "priority_score": st.column_config.ProgressColumn(
                "Review priority",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
        },
    )

    if visible_alerts.empty:
        st.info("No flagged transactions match this risk-band filter.")
        impact = calculate_uploaded_batch_impact(
            scored_batch, [], review_cost_per_alert=5, false_confirmation_cost=5,
            loss_multiplier=1,
        )
        st.metric(
            "Estimated exposure without triage",
            f"R {impact['exposure_without_triage']:,.2f}",
        )
        st.caption(
            "Exposure is estimated by weighting each transaction amount by its model "
            "fraud score. This is a scenario estimate, not observed loss."
        )
        return

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">Analyst review workspace</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Select an alert, read the explanation, and record a human decision."
    )

    selected_alert_id = st.selectbox(
        "Select alert to review",
        visible_alerts["alert_id"].tolist(),
        key=f"triage_alert_{batch_id}_{risk_filter}",
    )

    alert = alert_queue.loc[
        alert_queue["alert_id"] == selected_alert_id
    ].iloc[0]
    st.caption(
        f"Reviewing {selected_alert_id} from {source_name}, "
        f"CSV row {int(alert['CSV row']):,}."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        _metric_card(
            "Fraud probability",
            f"{alert['fraud_probability']:.1%}",
            "Model-based fraud risk",
            "#dc2626" if alert["risk_band"] == "High" else "#155eef",
        )

    with col2:
        _metric_card(
            "Risk band",
            alert["risk_band"],
            "Model risk category",
            "#dc2626" if alert["risk_band"] == "High" else "#155eef",
        )

    with col3:
        _metric_card(
            "Review priority",
            f"{alert['priority_score']:.2f}",
            "Higher means review sooner",
            "#b54708",
        )

    with col4:
        _metric_card(
            "Transaction amount",
            f"R {alert['amount']:,.0f}",
            "Transaction value",
        )

    st.markdown(
        '<div class="panel-title">Transaction details</div>',
        unsafe_allow_html=True,
    )

    detail_col1, detail_col2, detail_col3 = st.columns(3)

    with detail_col1:
        st.metric("Transaction type", alert["type"])

    with detail_col2:
        st.metric(
            "Sender balance before",
            f"R {alert['oldbalanceOrg']:,.0f}",
        )

    with detail_col3:
        st.metric(
            "Recipient balance before",
            f"R {alert['oldbalanceDest']:,.0f}",
        )

    amount_ratio = (
        alert["amount"] / alert["oldbalanceOrg"]
        if alert["oldbalanceOrg"] > 0
        else 0
    )

    explanation = (
        f"This {alert['type']} transaction was assigned a fraud probability of "
        f"{alert['fraud_probability']:.1%}. "
    )

    if alert["type"] in ["TRANSFER", "CASH_OUT"]:
        explanation += (
            "TRANSFER and CASH-OUT transactions are generally more relevant to "
            "fraud risk than other transaction types. "
        )

    if amount_ratio >= 0.8:
        explanation += (
            "The transaction amount is very high compared with the sender’s "
            "starting balance, which increases the model’s concern. "
        )
    elif amount_ratio >= 0.4:
        explanation += (
            "The transaction amount is moderately high compared with the sender’s "
            "starting balance. "
        )
    else:
        explanation += (
            "The transaction amount is small compared with the sender’s starting "
            "balance. "
        )

    if alert["risk_band"] == "High":
        explanation += (
            "Because the fraud probability is 5% or higher, this alert is classified "
            "as high risk and should be reviewed urgently."
        )
    elif alert["risk_band"] == "Medium":
        explanation += (
            "Because the fraud probability is between 1% and 5%, this alert is "
            "classified as medium risk and should be monitored or reviewed."
        )
    else:
        explanation += (
            "Because the fraud probability is below 1%, this alert is classified "
            "as low risk."
        )

    st.info(explanation)

    st.caption(
        "This explanation describes patterns learned by the model. It does not prove "
        "that the transaction is fraudulent."
    )

    st.markdown(
        '<div class="panel-title">Record analyst decision</div>',
        unsafe_allow_html=True,
    )

    decision = st.radio(
        "What should happen to this alert?",
        [
            "Confirm fraud",
            "False positive",
            "Needs further investigation",
        ],
        horizontal=True,
        key=f"decision_{batch_id}_{selected_alert_id}",
    )

    analyst_notes = st.text_area(
        "Analyst notes",
        placeholder="Write a short note explaining the decision.",
        key=f"notes_{batch_id}_{selected_alert_id}",
    )

    if st.button(
        "Save decision",
        type="primary",
        key=f"save_{batch_id}_{selected_alert_id}",
    ):
        feedback = st.session_state.setdefault("analyst_feedback", [])
        feedback = [
            item
            for item in feedback
            if not (
                item.get("batch_id") == batch_id
                and item.get("alert_id") == selected_alert_id
            )
        ]
        feedback.append(
            {
                "batch_id": batch_id,
                "source_name": source_name,
                "CSV row": int(alert["CSV row"]),
                "alert_id": selected_alert_id,
                "decision": decision,
                "fraud_probability": alert["fraud_probability"],
                "risk_band": alert["risk_band"],
                "priority_score": alert["priority_score"],
                "analyst_notes": analyst_notes,
            }
        )
        st.session_state["analyst_feedback"] = feedback

        st.success(
            f"Decision saved for {selected_alert_id}: {decision}."
        )

    batch_feedback = [
        item
        for item in st.session_state.get("analyst_feedback", [])
        if item.get("batch_id") == batch_id
    ]

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">Cost-impact simulator</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Estimate potential exposure and review costs from the scored transactions "
        "in this CSV batch."
    )

    cost_inputs = st.columns(3)
    review_cost = cost_inputs[0].number_input(
        "Analyst cost per reviewed alert (R)",
        min_value=0.0,
        value=5.0,
        step=1.0,
        key=f"triage_review_cost_{batch_id}",
    )
    false_confirmation_cost = cost_inputs[1].number_input(
        "Cost per wrongly confirmed alert (R)",
        min_value=0.0,
        value=5.0,
        step=1.0,
        key=f"triage_false_confirmation_cost_{batch_id}",
    )
    loss_multiplier = cost_inputs[2].number_input(
        "Loss per fraud amount (multiplier)",
        min_value=0.0,
        value=1.0,
        step=0.1,
        key=f"triage_loss_multiplier_{batch_id}",
    )
    impact = calculate_uploaded_batch_impact(
        scored_batch,
        batch_feedback,
        review_cost,
        false_confirmation_cost,
        loss_multiplier,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Estimated exposure without triage",
        f"R {impact['exposure_without_triage']:,.2f}",
    )
    metric_columns[1].metric(
        "Exposure associated with confirmed alerts",
        f"R {impact['exposure_associated_with_confirmed']:,.2f}",
    )
    metric_columns[2].metric(
        "Estimated cost after triage",
        f"R {impact['remaining_exposure'] + impact['total_cost_after_triage']:,.2f}",
    )
    metric_columns[3].metric(
        "Estimated net value",
        f"R {impact['net_value']:,.2f}",
    )
    st.write(
        f"Reviewed {impact['reviewed_count']:,} of {len(alert_queue):,} flagged alerts. "
        f"Estimated analyst cost: R {impact['analyst_cost']:,.2f}; "
        f"estimated cost of false confirmations: "
        f"R {impact['false_confirmation_cost']:,.2f}."
    )
    st.caption(
        "Exposure is estimated by weighting each transaction amount by its model "
        "fraud score. Confirmed fraud decisions are assumed to prevent the associated "
        "exposure. These are scenario estimates, not observed losses or a forecast."
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">Analyst feedback summary</div>',
        unsafe_allow_html=True,
    )

    feedback = batch_feedback

    if not feedback:
        st.info(
            "No analyst decisions have been saved yet. Select an alert and save a "
            "decision to create feedback."
        )
    else:
        feedback_df = pd.DataFrame(feedback)
        public_feedback = feedback_df.drop(columns="batch_id")

        confirmed = (feedback_df["decision"] == "Confirm fraud").sum()
        false_positives = (feedback_df["decision"] == "False positive").sum()
        investigate = (
            feedback_df["decision"] == "Needs further investigation"
        ).sum()

        summary_columns = st.columns(3)

        with summary_columns[0]:
            st.metric("Total reviewed", len(feedback_df))

        with summary_columns[1]:
            st.metric("Confirmed fraud", confirmed)

        with summary_columns[2]:
            st.metric("False positives", false_positives)

        st.metric("Needs investigation", investigate)

        st.dataframe(
            public_feedback,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = public_feedback.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download analyst feedback as CSV",
            data=csv_data,
            file_name="analyst_feedback.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.info(
            "In a production system, analyst feedback could be used as new labelled "
            "data for future model retraining. This helps the model learn from "
            "confirmed fraud cases and reduce false positives."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        "This workspace supports human decision-making. It does not automatically "
        "block transactions or replace fraud analysts."
    )


with st.sidebar:
    st.markdown("## PaySim Fraud Review")
    st.caption("Fraud detection, model comparison, and explainable AI")

    st.divider()

    page = st.radio(
        "Project sections",
        [
            "Overview",
            "Model comparison",
            "Visual analysis",
            "Explainability",
            "Fraud review & cost simulator",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Chronological validation by transaction step")
    st.caption("Model outputs support human fraud review.")


if page == "Overview":
    render_overview()
elif page == "Model comparison":
    render_model_results()
elif page == "Visual analysis":
    render_visualizations()
elif page == "Explainability":
    render_explainability()
else:
    render_fraud_review()


st.markdown(
    '<div class="footer-note">PaySim Fraud Review Dashboard · Model outputs support '
    'fraud review and should not replace human judgement.</div>',
    unsafe_allow_html=True,
)