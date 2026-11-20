import hashlib
import json
import re
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
    CONFIRM_FRAUD,
    MAX_UPLOAD_BYTES,
    calculate_cost_impact,
    explain_alert,
    load_review_transactions,
    score_review_batch,
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
            background: #d7c2c7 !important;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: #d7c2c7 !important;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        section[data-testid="stSidebar"] {
            background: #d7c2c7;
        }

        section[data-testid="stSidebar"] * {
            color: #33282b !important;
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


def _metric_card(label: str, value: str, note: str = "", colour: str = "#155eef") -> None:
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
            "Thresholds are chosen on the middle time window to meet the recall "
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
        "not select the model or threshold."
    )

    st.markdown('</div>', unsafe_allow_html=True)


def render_prediction() -> None:
    _page_header(
        "Score a transaction",
        "Enter only fields available before the payment completes. The saved "
        "threshold was selected on the validation period.",
    )

    if not _current_model_available():
        st.info(
            "No compatible model has been trained with the current chronological "
            "split and pre-transaction feature schema."
        )
        st.code(_training_command(), language="powershell")
        return

    with st.form("prediction_form"):
        st.markdown('<div class="panel-title">Transaction details</div>', unsafe_allow_html=True)

        left, right = st.columns(2)

        with left:
            step = st.number_input("PaySim hour step", min_value=0, value=1, step=1)
            transaction_type = st.selectbox(
                "Transaction type",
                ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
            )
            amount = st.number_input(
                "Transaction amount",
                min_value=0.0,
                value=1000.0,
                step=100.0,
            )

        with right:
            oldbalance_org = st.number_input(
                "Origin balance before",
                min_value=0.0,
                value=5000.0,
                step=100.0,
            )
            oldbalance_dest = st.number_input(
                "Destination balance before",
                min_value=0.0,
                value=1000.0,
                step=100.0,
            )

        submitted = st.form_submit_button(
            "Assess fraud risk",
            type="primary",
            use_container_width=True,
        )

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
            get_model(str(model_path), model_path.stat().st_mtime_ns),
            transaction,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        st.error(f"Could not score this transaction: {error}")
        return

    probability = float(result["fraud_probability"])
    threshold = float(result["threshold"])

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Fraud-risk result</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if result["prediction"] == 1:
            st.markdown(
                f"""
                <div class="risk-high">
                    <div style="font-size:0.85rem; font-weight:700; opacity:0.9;">Model decision</div>
                    <div style="font-size:1.6rem; font-weight:800;">Flag for review</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="risk-low">
                    <div style="font-size:0.85rem; font-weight:700; opacity:0.9;">Model decision</div>
                    <div style="font-size:1.6rem; font-weight:800;">Below review threshold</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        st.metric("Fraud score", f"{probability:.1%}")

    with col3:
        st.metric("Decision threshold", f"{threshold:.1%}")
        st.progress(min(probability, 1.0))

    if result["prediction"] == 1:
        st.error("This transaction should be sent to a fraud analyst for review.")
    else:
        st.success("This transaction is below the review threshold.")

    st.caption(
        "Scores are estimates from synthetic PaySim data, not production decisions."
    )

    st.markdown('</div>', unsafe_allow_html=True)


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
            " Performance by algorithm",
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
            st.warning("The performance dashboard is missing. Rerun the current training pipeline.")

    with visual_tabs[1]:
        if pareto_path.is_file():
            st.image(
                str(pareto_path),
                caption="Final-period quality, speed, recall, and cost trade-offs",
                use_container_width=True,
            )
        else:
            st.warning("The Pareto dashboard is missing. Rerun the current training pipeline.")

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
        "Pareto dominance is calculated on the final period for display, not model selection."
    )

    st.markdown('</div>', unsafe_allow_html=True)


def render_explainability() -> None:
    _page_header(
        "Explainability and XAI",
        "Understand how the winning fraud-detection model makes predictions using "
        "SHAP and LIME.",
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
            "Decision threshold",
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
            ["Decision threshold", f"{metadata.get('threshold', 0):.4f}"],
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
            " Business interpretation",
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
                caption="LIME explanation for one transaction predicted as fraud",
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
    _page_header(
        "Fraud review and cost simulator",
        "Score one transaction or a CSV batch, prioritise alerts by risk, record analyst decisions, and estimate the value of review.",
    )

    current = _current_results()
    if current is None:
        _show_stale_run_notice()
        return
    results, _ = current

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Choose a model and strategy</div>', unsafe_allow_html=True)
    available_models = [name for name in MODEL_NAMES if name in results["model"].unique()]
    model = st.selectbox("Model", available_models, key="review_model")
    model_results = results.loc[results["model"] == model]
    available_strategies = model_results["imbalance_strategy"].tolist()
    strategy = st.selectbox(
        "Imbalance strategy",
        available_strategies,
        format_func=lambda value: STRATEGY_LABELS[value],
        key="review_strategy",
    )
    selected = model_results.loc[model_results["imbalance_strategy"] == strategy].iloc[0]
    threshold = float(selected["threshold"])
    st.caption("The selected saved model and strategy will score the uploaded or single transaction.")
    st.markdown('</div>', unsafe_allow_html=True)

    source_mode = st.radio(
        "Transaction source",
        ["Upload a CSV", "Single transaction"],
        horizontal=True,
        key="review_source_mode",
    )
    input_data = None
    input_fingerprint = None
    source_name = ""

    if source_mode == "Upload a CSV":
        uploaded_file = st.file_uploader(
            "Choose a transaction CSV (maximum 200 MB)",
            type=["csv"],
            help=(
                "Required columns: step, type, amount, oldbalanceOrg, oldbalanceDest. "
                "An isFraud column is optional and is used only for aggregate cost evaluation."
            ),
            key="review_csv_upload",
        )
        if uploaded_file is not None:
            raw_csv = uploaded_file.getvalue()
            if len(raw_csv) > MAX_UPLOAD_BYTES:
                st.error("The CSV exceeds the 200 MB upload limit.")
            else:
                input_fingerprint = hashlib.sha256(raw_csv).hexdigest()
                source_name = uploaded_file.name
                cached_source = st.session_state.get("review_source_cache")
                if cached_source and cached_source["fingerprint"] == input_fingerprint:
                    input_data = cached_source["data"]
                else:
                    try:
                        input_data = load_review_transactions(BytesIO(raw_csv))
                    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as error:
                        st.error(f"Could not use this CSV: {error}")
                    else:
                        st.session_state["review_source_cache"] = {
                            "fingerprint": input_fingerprint,
                            "data": input_data,
                        }

    else:
        single_columns = st.columns(5)
        with single_columns[0]:
            step = st.number_input("Step", min_value=0, value=1, step=1, key="review_single_step")
        with single_columns[1]:
            transaction_type = st.selectbox(
                "Transaction type",
                ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
                key="review_single_type",
            )
        with single_columns[2]:
            amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0, key="review_single_amount")
        with single_columns[3]:
            origin_balance = st.number_input(
                "Sender balance before",
                min_value=0.0,
                value=2000.0,
                step=100.0,
                key="review_single_origin",
            )
        with single_columns[4]:
            destination_balance = st.number_input(
                "Recipient balance before",
                min_value=0.0,
                value=0.0,
                step=100.0,
                key="review_single_destination",
            )
        input_data = pd.DataFrame(
            [{
                "step": step,
                "type": transaction_type,
                "amount": amount,
                "oldbalanceOrg": origin_balance,
                "oldbalanceDest": destination_balance,
            }]
        )
        source_name = "Single transaction"
        input_fingerprint = hashlib.sha256(
            input_data.to_json(orient="records", double_precision=15).encode("utf-8")
        ).hexdigest()

    artifact_slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    model_path = MODEL_DIR / f"{artifact_slug}_{strategy}.joblib"
    score_clicked = st.button(
        "Score transactions and build priority queue",
        type="primary",
        disabled=input_data is None,
        key="score_review_batch",
    )
    if score_clicked and input_data is not None and input_fingerprint is not None:
        if not model_path.is_file():
            st.error(f"The selected model artifact is missing: {model_path.name}")
            return
        model_object = get_model(str(model_path), model_path.stat().st_mtime_ns)
        scored_data = score_review_batch(input_data, model_object, threshold)
        batch_id = hashlib.sha256(
            f"{input_fingerprint}:{model}:{strategy}".encode("utf-8")
        ).hexdigest()[:16]
        previous_batch = st.session_state.get("review_batch")
        if (
            previous_batch is None
            or previous_batch["batch_id"] != batch_id
            or "review_probability" not in previous_batch["data"]
        ):
            st.session_state["review_decisions"] = {}
        st.session_state["review_batch"] = {
            "batch_id": batch_id,
            "source_fingerprint": input_fingerprint,
            "source_name": source_name,
            "model": model,
            "strategy": strategy,
            "threshold": threshold,
            "data": scored_data,
        }

    batch = st.session_state.get("review_batch")
    if batch is None:
        st.info("Choose a transaction source and score it to create the alert queue.")
        return
    if "review_probability" not in batch["data"]:
        st.info("This alert queue was scored by an earlier version. Score the transactions again to refresh it.")
        return
    if (
        input_fingerprint != batch["source_fingerprint"]
        or model != batch["model"]
        or strategy != batch["strategy"]
    ):
        st.info("The input or model selection has changed. Score it to refresh the queue and start a new review batch.")
        return

    transactions = batch["data"]
    decisions = st.session_state.setdefault("review_decisions", {})
    counts = transactions["risk_band"].value_counts()
    band_counts = {band: int(counts.get(band, 0)) for band in ("High", "Medium", "Low")}
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Batch overview and priority queue</div>', unsafe_allow_html=True)
    st.caption(
        f"{len(transactions):,} transaction(s) from {batch['source_name']}. "
        "Risk bands use the project's documented cutoffs: Low below 1%, Medium 1% to below 5%, High 5% or higher."
    )
    overview_columns = st.columns(4)
    overview_columns[0].metric("Transactions", f"{len(transactions):,}")
    overview_columns[1].metric("High risk", f"{band_counts['High']:,}")
    overview_columns[2].metric("Medium risk", f"{band_counts['Medium']:,}")
    overview_columns[3].metric("Low risk", f"{band_counts['Low']:,}")

    risk_filter = st.selectbox(
        "Filter by risk band",
        ["All risk bands", "High", "Medium", "Low"],
        key=f"risk_band_filter_{batch['batch_id']}",
    )
    if risk_filter == "All risk bands":
        alert_queue = transactions.copy()
    else:
        alert_queue = transactions.loc[transactions["risk_band"] == risk_filter].copy()
    visible_columns = [
        "alert_id",
        "type",
        "amount",
        "fraud_probability",
        "review_probability",
        "risk_band",
    ]
    st.caption(
        "Alerts are grouped by risk band, then ordered by fraud probability. "
        "Review probability is a prioritisation estimate, not a separately trained model score. "
        "Select a row to open its analyst review."
    )
    queue_event = st.dataframe(
        alert_queue[visible_columns],
        width="stretch",
        height=420,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"queue_{batch['batch_id']}_{risk_filter}",
        column_config={
            "alert_id": st.column_config.TextColumn("Alert ID"),
            "type": st.column_config.TextColumn("Type"),
            "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud probability", min_value=0.0, max_value=1.0, format="%.1%"
            ),
            "review_probability": st.column_config.ProgressColumn(
                "Review probability", min_value=0.0, max_value=1.0, format="%.1%"
            ),
            "risk_band": st.column_config.TextColumn("Risk band"),
        },
    )
    export_data = alert_queue[visible_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered alert queue CSV",
        data=export_data,
        file_name="scored_alert_queue.csv",
        mime="text/csv",
        key=f"download_queue_{batch['batch_id']}_{risk_filter}",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Analyst review workspace</div>', unsafe_allow_html=True)
    selected_rows = queue_event.selection.rows
    if not selected_rows:
        st.info("Select an alert in the priority queue above to read its explanation and record a decision.")
    else:
        alert = alert_queue.iloc[selected_rows[0]]
        alert_id = str(alert["alert_id"])
        detail_columns = st.columns(4)
        detail_columns[0].metric("Fraud score", f"{float(alert['fraud_probability']):.1%}")
        detail_columns[1].metric("Risk band", str(alert["risk_band"]))
        detail_columns[2].metric("Review probability", f"{float(alert['review_probability']):.1%}")
        detail_columns[3].metric("Amount", f"{float(alert['amount']):,.2f}")
        st.markdown("**Transaction details**")
        st.write(
            f"Step {int(alert['step'])} | {alert['type']} | "
            f"Sender balance before: {float(alert['oldbalanceOrg']):,.2f} | "
            f"Recipient balance before: {float(alert['oldbalanceDest']):,.2f}"
        )
        st.info(explain_alert(alert))
        st.caption("This is a model-based explanation, not proof of fraud. Outcome labels, when present in a sample file, are not shown to analysts.")
        decision_key = f"decision_{batch['batch_id']}_{alert_id}"
        decision = st.selectbox(
            "Analyst decision",
            ["Select a decision", CONFIRM_FRAUD, "False positive", "Needs further investigation"],
            key=decision_key,
        )
        notes = st.text_area(
            "Analyst notes",
            placeholder="Briefly explain the decision in plain language.",
            key=f"notes_{batch['batch_id']}_{alert_id}",
        )
        if st.button("Save analyst decision", type="primary", key=f"save_{batch['batch_id']}_{alert_id}"):
            if decision == "Select a decision":
                st.warning("Choose a decision before saving.")
            else:
                decisions[alert_id] = {
                    "alert_id": alert_id,
                    "step": int(alert["step"]),
                    "type": str(alert["type"]),
                    "amount": float(alert["amount"]),
                    "fraud_probability": float(alert["fraud_probability"]),
                    "review_probability": float(alert["review_probability"]),
                    "risk_band": str(alert["risk_band"]),
                    "decision": decision,
                    "analyst_notes": notes.strip(),
                }
                st.success(f"Decision saved for {alert_id}.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Cost-impact simulator</div>', unsafe_allow_html=True)
    st.caption("Compare the estimated fraud exposure without triage against exposure remaining after confirmed fraud decisions, plus review operating costs.")
    cost_columns = st.columns(3)
    review_cost = cost_columns[0].number_input(
        "Analyst cost per reviewed alert (value units)", min_value=0.0, value=5.0, step=1.0,
        key=f"review_cost_{batch['batch_id']}",
    )
    false_alert_cost = cost_columns[1].number_input(
        "Cost per wrongly confirmed alert (value units)", min_value=0.0, value=5.0, step=1.0,
        key=f"false_alert_cost_{batch['batch_id']}",
    )
    loss_multiplier = cost_columns[2].number_input(
        "Loss per fraud amount (multiplier)", min_value=0.0, value=1.0, step=0.1,
        key=f"loss_multiplier_{batch['batch_id']}",
    )
    impact = calculate_cost_impact(
        transactions, decisions, review_cost, false_alert_cost, loss_multiplier
    )
    if impact["has_ground_truth"]:
        st.caption("This CSV contains isFraud labels. They are used only here for aggregate benchmark estimates, never as model inputs or in the analyst queue.")
    else:
        st.caption("No isFraud labels were supplied. Exposure figures are probability-weighted estimates from model scores, not observed losses.")
    with_costs = float(impact["remaining_exposure"] + impact["total_process_cost"])
    impact_columns = st.columns(4)
    impact_columns[0].metric("Exposure without review", f"{float(impact['uncontrolled_exposure']):,.2f}")
    impact_columns[1].metric("Exposure identified", f"{float(impact['captured_exposure']):,.2f}")
    impact_columns[2].metric("Estimated cost with review", f"{with_costs:,.2f}")
    impact_columns[3].metric("Estimated net savings", f"{float(impact['net_savings']):,.2f}")
    st.write(
        f"Decisions recorded: {impact['reviewed_count']:,} of {len(transactions):,}. "
        f"Analyst handling cost: {float(impact['analyst_cost']):,.2f}; "
        f"wrong-confirmation cost: {float(impact['false_alert_cost']):,.2f}; "
        f"remaining exposure: {float(impact['remaining_exposure']):,.2f}."
    )
    st.caption("The scenario assumes confirmed fraud is stopped, uses transaction amount as loss exposure, and applies the editable cost assumptions above. It is an educational estimate, not a financial forecast.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Analyst feedback summary</div>', unsafe_allow_html=True)
    decision_counts = pd.Series(
        [entry["decision"] for entry in decisions.values()], dtype="string"
    ).value_counts()
    confirmed_count = int(decision_counts.get(CONFIRM_FRAUD, 0))
    false_positive_count = int(decision_counts.get("False positive", 0))
    investigate_count = int(decision_counts.get("Needs further investigation", 0))
    summary_columns = st.columns(4)
    summary_columns[0].metric("Reviewed", len(decisions))
    summary_columns[1].metric("Confirmed fraud", confirmed_count)
    summary_columns[2].metric("False positives", false_positive_count)
    summary_columns[3].metric("Needs investigation", investigate_count)
    st.bar_chart(pd.DataFrame({"Transactions": band_counts}), color="#155eef")
    if decisions:
        feedback = pd.DataFrame(list(decisions.values()))
        st.write(
            f"The team reviewed {len(decisions):,} transaction(s) from this batch: "
            f"{confirmed_count:,} were marked for fraud action, "
            f"{false_positive_count:,} were marked as false alerts, and "
            f"{investigate_count:,} need more investigation. "
            "These are analyst decisions, not a claim that the model or the decisions are always correct."
        )
        st.dataframe(
            feedback[[
                "alert_id", "step", "type", "amount", "risk_band",
                "fraud_probability", "review_probability", "decision", "analyst_notes",
            ]],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download analyst feedback CSV",
            data=feedback.to_csv(index=False).encode("utf-8"),
            file_name="analyst_feedback.csv",
            mime="text/csv",
            key=f"download_feedback_{batch['batch_id']}",
        )
    else:
        st.info("No analyst decisions have been recorded for this batch yet. Select an alert, review the explanation, and save a decision to build the feedback summary.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("This demonstration supports human review. It does not automatically block payments or replace an analyst's judgement.")


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
            "Transaction scoring",
            "Alert triage & cost simulator",
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
elif page == "Alert triage & cost simulator":
    render_alert_triage()
else:
    render_prediction()


st.markdown(
    '<div class="footer-note">PaySim Fraud Review Dashboard · Model outputs support fraud review and should not replace human judgement.</div>',
    unsafe_allow_html=True,
)