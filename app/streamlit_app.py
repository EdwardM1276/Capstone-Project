import json
from pathlib import Path

import pandas as pd
import streamlit as st

from fraud_detection.config import DEFAULT_MAX_ROWS, FIGURE_DIR, MODEL_DIR, REPORT_DIR
from fraud_detection.models import MODEL_NAMES
from fraud_detection.prediction import load_best_model, predict_transaction


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
            " Pareto frontiers",
            " High-score risk",
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
            " SHAP global explanation",
            " LIME local explanation",
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
        "Alert triage and cost simulator",
        "Estimate review workload and the impact of false positives and missed fraud "
        "using validation-period model metrics.",
    )

    st.info(
        "This page estimates how many transactions may need review and compares the "
        "cost of false alerts with the cost of missed fraud."
    )

    current = _current_results()

    if current is None:
        _show_stale_run_notice()
        return

    results, _ = current

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Select model and strategy</div>', unsafe_allow_html=True)

    model_columns = st.columns(2)

    with model_columns[0]:
        model = st.selectbox(
            "Model",
            MODEL_NAMES,
            key="triage_model",
        )

    model_results = results.loc[results["model"] == model]

    with model_columns[1]:
        strategy = st.selectbox(
            "Imbalance strategy",
            model_results["imbalance_strategy"].tolist(),
            format_func=lambda value: STRATEGY_LABELS[value],
            key="triage_strategy",
        )

    selected = model_results.loc[
        model_results["imbalance_strategy"] == strategy
    ].iloc[0]

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Simulation settings</div>', unsafe_allow_html=True)

    inputs = st.columns(3)

    with inputs[0]:
        transaction_volume = st.number_input(
            "Transactions to simulate",
            min_value=1,
            value=10_000,
            step=1_000,
        )

    with inputs[1]:
        fraud_prevalence = st.slider(
            "Expected fraud prevalence (%)",
            min_value=0.01,
            max_value=10.0,
            value=0.13,
            step=0.01,
        ) / 100

    with inputs[2]:
        review_capacity = st.number_input(
            "Analyst review capacity",
            min_value=1,
            value=200,
            step=25,
        )

    cost_inputs = st.columns(2)

    with cost_inputs[0]:
        false_positive_cost = st.number_input(
            "Cost per false alert (R)",
            min_value=0.0,
            value=5.0,
            step=1.0,
        )

    with cost_inputs[1]:
        missed_fraud_cost = st.number_input(
            "Cost per missed fraud (R)",
            min_value=0.0,
            value=100.0,
            step=10.0,
        )

    precision = float(selected["validation_precision_at_recall_target"])
    recall = float(selected["validation_recall"])

    expected_fraud = transaction_volume * fraud_prevalence
    true_positives = expected_fraud * recall
    false_negatives = expected_fraud - true_positives

    estimated_alerts = min(
        float(transaction_volume),
        true_positives / precision if precision > 0 else float(transaction_volume),
    )

    false_positives = max(0.0, estimated_alerts - true_positives)

    estimated_cost = (
        false_positives * false_positive_cost
        + false_negatives * missed_fraud_cost
    )

    st.markdown('<div class="panel-title">Estimated triage workload</div>', unsafe_allow_html=True)

    metrics = st.columns(4)

    with metrics[0]:
        st.metric("Alerts for review", f"{estimated_alerts:,.0f}")

    with metrics[1]:
        st.metric("False alerts", f"{false_positives:,.0f}")

    with metrics[2]:
        st.metric("Missed fraud", f"{false_negatives:,.0f}")

    with metrics[3]:
        st.metric("Estimated total cost", f"R {estimated_cost:,.2f}")

    if estimated_alerts > review_capacity:
        st.warning(
            f"Estimated alerts exceed capacity by "
            f"{estimated_alerts - review_capacity:,.0f} transactions."
        )
    else:
        st.success("Estimated alert volume is within analyst review capacity.")

    st.caption(
        f"Estimates use validation precision ({precision:.1%}) and recall "
        f"({recall:.1%}) for {model} with {STRATEGY_LABELS[strategy]}. "
        "These are scenario estimates, not a scored transaction queue or production forecast."
    )

    st.markdown('</div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown("##  PaySim Fraud Review")
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
            " Alert triage & cost simulator",
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