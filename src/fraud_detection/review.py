from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.config import (
    HIGH_FRAUD_PROBABILITY_MIN,
    LOW_FRAUD_PROBABILITY_MAX,
)

SCORED_UPLOAD_COLUMNS = {
    "CSV row",
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "Fraud score (%)",
    "Decision",
}
CONFIRM_FRAUD = "Confirm fraud"


def build_uploaded_alert_queue(scored_transactions: pd.DataFrame) -> pd.DataFrame:
    """Create review alerts from the rows flagged by the active CSV scoring run."""
    missing = sorted(SCORED_UPLOAD_COLUMNS - set(scored_transactions.columns))
    if missing:
        raise ValueError(
            "Scored upload is missing required columns: " + ", ".join(missing)
        )

    flagged = scored_transactions.loc[
        scored_transactions["Decision"] == "Flag for review"
    ].copy()
    if "Error" in flagged.columns:
        flagged = flagged.loc[flagged["Error"].fillna("").eq("")]

    flagged["CSV row"] = pd.to_numeric(flagged["CSV row"], errors="coerce")
    flagged["Fraud score (%)"] = pd.to_numeric(
        flagged["Fraud score (%)"], errors="coerce"
    )
    flagged["amount"] = pd.to_numeric(flagged["amount"], errors="coerce")
    valid = (
        np.isfinite(flagged["CSV row"])
        & np.isfinite(flagged["Fraud score (%)"])
        & np.isfinite(flagged["amount"])
        & flagged["CSV row"].ge(2)
        & flagged["Fraud score (%)"].between(0, 100)
        & flagged["amount"].ge(0)
    )
    flagged = flagged.loc[valid].copy()

    flagged["CSV row"] = flagged["CSV row"].astype("int64")
    flagged["fraud_probability"] = flagged["Fraud score (%)"] / 100
    flagged["alert_id"] = (
        "TXN-" + (flagged["CSV row"] - 1).astype(str).str.zfill(6)
    )
    flagged["risk_band"] = np.select(
        [
            flagged["fraud_probability"] >= HIGH_FRAUD_PROBABILITY_MIN,
            flagged["fraud_probability"] >= LOW_FRAUD_PROBABILITY_MAX,
        ],
        ["High", "Medium"],
        default="Low",
    )

    largest_amount = float(flagged["amount"].max()) if not flagged.empty else 0.0
    amount_risk = (
        flagged["amount"] / largest_amount
        if largest_amount > 0
        else pd.Series(0.0, index=flagged.index)
    )
    uncertainty = 1 - (flagged["fraud_probability"] - 0.5).abs() * 2
    flagged["priority_score"] = (
        0.60 * flagged["fraud_probability"]
        + 0.25 * amount_risk
        + 0.15 * uncertainty
    )
    return flagged.sort_values(
        ["priority_score", "fraud_probability", "CSV row"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def calculate_uploaded_batch_impact(
    scored_transactions: pd.DataFrame,
    analyst_feedback: list[dict[str, Any]],
    review_cost_per_alert: float,
    false_confirmation_cost: float,
    loss_multiplier: float,
) -> dict[str, float | int]:
    """Estimate exposure and review costs from the active scored upload."""
    if min(review_cost_per_alert, false_confirmation_cost, loss_multiplier) < 0:
        raise ValueError("Cost assumptions must be non-negative.")
    if not SCORED_UPLOAD_COLUMNS.issubset(scored_transactions.columns):
        raise ValueError("Scored upload is missing required cost-simulation columns.")

    data = scored_transactions.copy()
    data["CSV row"] = pd.to_numeric(data["CSV row"], errors="coerce")
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    data["Fraud score (%)"] = pd.to_numeric(
        data["Fraud score (%)"], errors="coerce"
    )
    valid = (
        np.isfinite(data["CSV row"])
        & np.isfinite(data["amount"])
        & np.isfinite(data["Fraud score (%)"])
        & data["CSV row"].ge(2)
        & data["amount"].ge(0)
        & data["Fraud score (%)"].between(0, 100)
        & data["Decision"].isin(["Flag for review", "Below review threshold"])
    )
    data = data.loc[valid].copy()
    data["alert_id"] = (
        "TXN-" + (data["CSV row"].astype("int64") - 1).astype(str).str.zfill(6)
    )
    data["expected_exposure"] = (
        data["amount"] * data["Fraud score (%)"] / 100 * loss_multiplier
    )

    decisions = {
        str(item["alert_id"]): str(item["decision"])
        for item in analyst_feedback
        if "alert_id" in item and "decision" in item
    }
    batch_alert_ids = set(data["alert_id"])
    flagged_alert_ids = set(
        data.loc[data["Decision"] == "Flag for review", "alert_id"]
    )
    confirmed_ids = {
        alert_id
        for alert_id, decision in decisions.items()
        if decision == CONFIRM_FRAUD and alert_id in flagged_alert_ids
    }
    confirmed = data["alert_id"].isin(confirmed_ids)
    exposure_without_triage = float(data["expected_exposure"].sum())
    exposure_associated_with_confirmed = float(
        data.loc[confirmed, "expected_exposure"].sum()
    )
    remaining_exposure = max(
        0.0, exposure_without_triage - exposure_associated_with_confirmed
    )
    expected_false_confirmations = float(
        (1 - data.loc[confirmed, "Fraud score (%)"] / 100).sum()
    )
    reviewed_ids = {
        alert_id
        for alert_id in decisions
        if alert_id in batch_alert_ids and alert_id in flagged_alert_ids
    }
    analyst_cost = len(reviewed_ids) * review_cost_per_alert
    false_confirmation_cost_total = (
        expected_false_confirmations * false_confirmation_cost
    )
    total_cost_after_triage = analyst_cost + false_confirmation_cost_total

    return {
        "reviewed_count": len(reviewed_ids),
        "confirmed_count": len(confirmed_ids),
        "exposure_without_triage": exposure_without_triage,
        "exposure_associated_with_confirmed": exposure_associated_with_confirmed,
        "remaining_exposure": remaining_exposure,
        "analyst_cost": analyst_cost,
        "false_confirmation_cost": false_confirmation_cost_total,
        "total_cost_after_triage": total_cost_after_triage,
        "net_value": exposure_associated_with_confirmed - total_cost_after_triage,
    }
