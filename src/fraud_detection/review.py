from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.config import (
    HIGH_FRAUD_PROBABILITY_MIN,
    LOW_FRAUD_PROBABILITY_MAX,
    TARGET_COLUMN,
)
from fraud_detection.evaluation import probability_band
from fraud_detection.features import engineer_features

REVIEW_INPUT_COLUMNS = (
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
)
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_TRANSACTION_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
CONFIRM_FRAUD = "Confirm fraud"


def load_review_transactions(source: BytesIO | bytes | str) -> pd.DataFrame:
    """Read and validate the pre-transaction fields needed by the scoring model."""
    if isinstance(source, bytes):
        source = BytesIO(source)
    data = pd.read_csv(source)
    if data.empty:
        raise ValueError("The CSV must contain at least one transaction.")

    missing = sorted(set(REVIEW_INPUT_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(
            "The CSV is missing required scoring columns: " + ", ".join(missing)
        )
    if TARGET_COLUMN in data.columns:
        labels = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError("Optional isFraud values must contain only 0 or 1.")
        data[TARGET_COLUMN] = labels.astype("int8")

    for column in ("step", "amount", "oldbalanceOrg", "oldbalanceDest"):
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite numeric values.")
        data[column] = values

    if (data["step"] < 0).any() or (data["step"] % 1 != 0).any():
        raise ValueError("step must contain non-negative whole numbers.")
    if any((data[column] < 0).any() for column in ("amount", "oldbalanceOrg", "oldbalanceDest")):
        raise ValueError("amount and pre-transaction balances must be non-negative.")
    if data["type"].isna().any() or not data["type"].isin(ALLOWED_TRANSACTION_TYPES).all():
        raise ValueError(
            "type must contain CASH_IN, CASH_OUT, DEBIT, PAYMENT, or TRANSFER."
        )

    return data.reset_index(drop=True)


def score_review_batch(
    transactions: pd.DataFrame,
    model: Any,
    threshold: float,
) -> pd.DataFrame:
    """Score a batch without using optional labels or other extra CSV columns."""
    if not 0 <= threshold <= 1:
        raise ValueError("The decision threshold must be between 0 and 1.")
    generated_columns = {
        "alert_id",
        "fraud_probability",
        "decision_threshold",
        "predicted_fraud",
        "risk_band",
        "priority_score",
        "review_probability",
        "risk_band_order",
    }
    data = transactions.drop(columns=generated_columns, errors="ignore").copy()
    features = engineer_features(data.loc[:, REVIEW_INPUT_COLUMNS])
    probabilities = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    if len(probabilities) != len(data) or not np.isfinite(probabilities).all():
        raise ValueError("The model returned invalid fraud probabilities.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("The model returned probabilities outside [0, 1].")

    data.insert(0, "alert_id", [f"TXN-{index:06d}" for index in range(1, len(data) + 1)])
    data["fraud_probability"] = probabilities
    data["decision_threshold"] = threshold
    data["predicted_fraud"] = (probabilities >= threshold).astype("int8")
    data["risk_band"] = probability_band(
        probabilities,
        LOW_FRAUD_PROBABILITY_MAX,
        HIGH_FRAUD_PROBABILITY_MIN,
    )
    largest_amount = float(data["amount"].max())
    amount_risk = (
        data["amount"] / largest_amount
        if largest_amount > 0
        else pd.Series(0.0, index=data.index)
    )
    uncertainty = 1 - np.abs(probabilities - 0.5) * 2
    data["review_probability"] = (
        0.60 * probabilities + 0.25 * amount_risk + 0.15 * uncertainty
    )
    data["risk_band_order"] = data["risk_band"].map(
        {"High": 0, "Medium": 1, "Low": 2}
    )
    return data.sort_values(
        ["risk_band_order", "fraud_probability", "review_probability", "step", "alert_id"],
        ascending=[True, False, False, True, True],
        kind="stable",
    ).drop(columns="risk_band_order").reset_index(drop=True)


def explain_alert(alert: pd.Series) -> str:
    amount = float(alert["amount"])
    origin_balance = float(alert["oldbalanceOrg"])
    balance_share = amount / origin_balance if origin_balance > 0 else None
    ratio_text = (
        "The amount is compared with a zero starting sender balance."
        if balance_share is None
        else f"The amount is {balance_share:.0%} of the sender's starting balance."
    )
    band_guidance = {
        "High": "This falls in the high-risk band (5% or higher); review it promptly.",
        "Medium": "This falls in the medium-risk band (1% to below 5%).",
        "Low": "This falls in the low-risk band (below 1%).",
    }[str(alert["risk_band"])]
    return (
        f"The model assigned this {alert['type']} transaction a "
        f"{float(alert['fraud_probability']):.1%} fraud score. {ratio_text} "
        f"{band_guidance} Transaction type, amount, balances, and time-related "
        "features contribute to the model's learned score; none alone proves fraud."
    )


def calculate_cost_impact(
    transactions: pd.DataFrame,
    decisions: dict[str, dict[str, Any]],
    review_cost_per_alert: float,
    false_confirmation_cost: float,
    loss_multiplier: float,
) -> dict[str, float | int | bool]:
    """Estimate fraud exposure and decision costs for the active uploaded batch."""
    if min(review_cost_per_alert, false_confirmation_cost, loss_multiplier) < 0:
        raise ValueError("Cost assumptions must be non-negative.")
    if not {"alert_id", "amount", "fraud_probability"}.issubset(transactions.columns):
        raise ValueError("Scored transactions are missing cost-simulation fields.")

    batch_alert_ids = set(transactions["alert_id"])
    saved = {
        alert_id: entry["decision"]
        for alert_id, entry in decisions.items()
        if alert_id in batch_alert_ids
    }
    confirmed_ids = {
        alert_id for alert_id, decision in saved.items() if decision == CONFIRM_FRAUD
    }
    confirmed = transactions["alert_id"].isin(confirmed_ids)
    reviewed_count = len(saved)
    has_labels = TARGET_COLUMN in transactions.columns

    if has_labels:
        actual_fraud = transactions[TARGET_COLUMN].astype(bool)
        uncontrolled_exposure = float(
            (transactions.loc[actual_fraud, "amount"] * loss_multiplier).sum()
        )
        captured_exposure = float(
            (
                transactions.loc[actual_fraud & confirmed, "amount"]
                * loss_multiplier
            ).sum()
        )
        false_confirmations = int((~actual_fraud & confirmed).sum())
    else:
        expected_loss = transactions["amount"] * transactions["fraud_probability"]
        uncontrolled_exposure = float(expected_loss.sum() * loss_multiplier)
        captured_exposure = float(expected_loss.loc[confirmed].sum() * loss_multiplier)
        false_confirmations = float(
            (1 - transactions.loc[confirmed, "fraud_probability"]).sum()
        )

    remaining_exposure = max(0.0, uncontrolled_exposure - captured_exposure)
    analyst_cost = reviewed_count * review_cost_per_alert
    false_alert_cost = false_confirmations * false_confirmation_cost
    total_process_cost = analyst_cost + false_alert_cost
    return {
        "has_ground_truth": has_labels,
        "reviewed_count": reviewed_count,
        "confirmed_count": len(confirmed_ids),
        "false_confirmations": false_confirmations,
        "uncontrolled_exposure": uncontrolled_exposure,
        "captured_exposure": captured_exposure,
        "remaining_exposure": remaining_exposure,
        "analyst_cost": analyst_cost,
        "false_alert_cost": false_alert_cost,
        "total_process_cost": total_process_cost,
        "net_savings": captured_exposure - total_process_cost,
    }
