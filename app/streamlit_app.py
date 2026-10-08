import json

import pandas as pd
import streamlit as st

from fraud_detection.config import FIGURE_DIR, MODEL_DIR, REPORT_DIR
from fraud_detection.prediction import load_best_model, predict_transaction

st.set_page_config(
    page_title="PaySim Fraud Detection",
    page_icon=None,
    layout="wide",
)

@st.cache_resource
def get_model():
    return load_best_model(MODEL_DIR / "best_model.joblib")


def render_overview() -> None:
    st.title("Mobile Money Fraud Detection")
    st.write(
        "This application demonstrates the Deliverable 3 modelling plan as a "
        "reproducible PaySim transaction-classification workflow."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Project objective")
        st.write(
            "Compare supervised machine-learning models that identify potentially "
            "fraudulent mobile-money transactions while reporting minority-class "
            "performance rather than relying on accuracy alone."
        )
    with right:
        st.subheader("Business context")
        st.write(
            "Payment providers process large transaction volumes. A fraud score can "
            "help prioritise transactions for review; this demonstration is not an "
            "automated payment-decision service."
        )
    st.subheader("Implementation")
    st.write(
        "The solution uses the PaySim synthetic dataset, automated data checks, "
        "engineered balance and time features, stratified evaluation, and a saved "
        "model. Customer identifiers are excluded from prediction."
    )
    metadata_path = MODEL_DIR / "best_model.json"
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        st.info(
            f"Selected model: {metadata.get('model', 'Unknown')} "
            f"using {metadata.get('imbalance_strategy', 'unknown')} "
            f"(selection metric: {metadata.get('selection_metric', 'PR-AUC')})."
        )


def render_prediction() -> None:
    st.title("Transaction prediction")
    st.caption(
        "Enter transaction details to receive the saved model's predicted class "
        "and fraud probability. Identifiers are not required."
    )
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            step = st.number_input("Time step (hour)", min_value=0, value=1, step=1)
            transaction_type = st.selectbox(
                "Transaction type",
                ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
            )
            amount = st.number_input(
                "Transaction amount", min_value=0.0, value=1000.0, step=100.0
            )
            oldbalance_org = st.number_input(
                "Originator balance before", min_value=0.0, value=5000.0, step=100.0
            )
        with right:
            newbalance_orig = st.number_input(
                "Originator balance after", min_value=0.0, value=4000.0, step=100.0
            )
            oldbalance_dest = st.number_input(
                "Destination balance before", min_value=0.0, value=1000.0, step=100.0
            )
            newbalance_dest = st.number_input(
                "Destination balance after", min_value=0.0, value=2000.0, step=100.0
            )
        submitted = st.form_submit_button("Score transaction")

    if submitted:
        transaction = {
            "step": step,
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalance_org,
            "newbalanceOrig": newbalance_orig,
            "oldbalanceDest": oldbalance_dest,
            "newbalanceDest": newbalance_dest,
            "isFlaggedFraud": 0,
        }
        try:
            result = predict_transaction(get_model(), transaction)
        except FileNotFoundError as error:
            st.error(str(error))
            st.info(
                "From the project root, run "
                "`python -m fraud_detection.cli train --max-rows 100000` "
                "after installing the project."
            )
            return
        except (ValueError, RuntimeError) as error:
            st.error(f"Prediction failed: {error}")
            return

        st.metric("Fraud probability", f"{result['fraud_probability']:.2%}")
        if result["prediction"] == 1:
            st.error("Model prediction: potentially fraudulent")
        else:
            st.success("Model prediction: likely legitimate")
        st.caption(
            "The class label uses a 0.50 probability threshold. This output is a "
            "model demonstration and should not be used as the sole basis for "
            "financial decisions."
        )


def render_model_results() -> None:
    st.title("Model comparison")
    comparison_path = REPORT_DIR / "model_comparison.csv"
    if not comparison_path.is_file():
        st.info("Run the training workflow to generate model-comparison results.")
        return
    results = pd.read_csv(comparison_path)
    st.dataframe(
        results.sort_values("pr_auc", ascending=False),
        width="stretch",
        hide_index=True,
    )
    best = results.sort_values(["pr_auc", "recall", "f1"], ascending=False).iloc[0]
    st.write(
        f"Selected by PR-AUC: **{best['model']}** "
        f"({best['imbalance_strategy']}); PR-AUC **{best['pr_auc']:.4f}**, "
        f"recall **{best['recall']:.4f}**, precision **{best['precision']:.4f}**."
    )
    summary_path = REPORT_DIR / "training_summary.json"
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as summary_file:
            summary = json.load(summary_file)
        st.caption(
            f"Evaluation used {summary['test_rows']:,} stratified test rows from "
            f"{summary['sampled_rows']:,} modelled rows, including "
            f"{summary['fraud_test_rows']:,} fraud cases."
        )
    st.caption(
        "Precision, recall, F1, ROC-AUC and PR-AUC are shown. PR-AUC is the "
        "primary selection metric for the rare fraud class."
    )


def render_visualizations() -> None:
    st.title("Evaluation visualisations")
    images = sorted(FIGURE_DIR.glob("*.png"))
    if not images:
        st.info("Run the preprocessing and training workflow to create figures.")
        return
    for image in images:
        st.subheader(image.stem.replace("_", " ").title())
        st.image(str(image), width="stretch")


page = st.sidebar.radio(
    "Project sections",
    ["Overview", "Fraud prediction", "Model comparison", "Visualisations"],
)
if page == "Overview":
    render_overview()
elif page == "Fraud prediction":
    render_prediction()
elif page == "Model comparison":
    render_model_results()
else:
    render_visualizations()
