import pandas as pd
import pytest

from fraud_detection.review import (
    build_uploaded_alert_queue,
    calculate_uploaded_batch_impact,
)


def test_uploaded_alert_queue_uses_flagged_rows_and_keeps_csv_identity():
    scored = pd.DataFrame(
        [
            {
                "CSV row": 4,
                "step": 3,
                "type": "TRANSFER",
                "amount": 10.0,
                "oldbalanceOrg": 12.0,
                "oldbalanceDest": 0.0,
                "Fraud score (%)": 70.0,
                "Decision": "Flag for review",
                "Error": "",
            },
            {
                "CSV row": 2,
                "step": 1,
                "type": "CASH_OUT",
                "amount": 100.0,
                "oldbalanceOrg": 120.0,
                "oldbalanceDest": 0.0,
                "Fraud score (%)": 80.0,
                "Decision": "Flag for review",
                "Error": "",
            },
            {
                "CSV row": 3,
                "step": 2,
                "type": "PAYMENT",
                "amount": 5.0,
                "oldbalanceOrg": 20.0,
                "oldbalanceDest": 10.0,
                "Fraud score (%)": 1.0,
                "Decision": "Below review threshold",
                "Error": "",
            },
            {
                "CSV row": 5,
                "step": None,
                "type": "DEBIT",
                "amount": None,
                "oldbalanceOrg": None,
                "oldbalanceDest": None,
                "Fraud score (%)": None,
                "Decision": "Not scored",
                "Error": "invalid amount",
            },
        ]
    )

    queue = build_uploaded_alert_queue(scored)

    assert queue["alert_id"].tolist() == ["TXN-000001", "TXN-000003"]
    assert queue["CSV row"].tolist() == [2, 4]
    assert queue["risk_band"].tolist() == ["High", "High"]
    assert queue["fraud_probability"].tolist() == pytest.approx([0.8, 0.7])


def test_uploaded_alert_queue_is_empty_when_no_rows_are_flagged():
    scored = pd.DataFrame(
        [
            {
                "CSV row": 2,
                "step": 1,
                "type": "PAYMENT",
                "amount": 5.0,
                "oldbalanceOrg": 20.0,
                "oldbalanceDest": 10.0,
                "Fraud score (%)": 1.0,
                "Decision": "Below review threshold",
            }
        ]
    )

    assert build_uploaded_alert_queue(scored).empty


def test_uploaded_alert_queue_rejects_missing_scoring_output_columns():
    with pytest.raises(ValueError, match="CSV row"):
        build_uploaded_alert_queue(pd.DataFrame({"Decision": []}))


def test_uploaded_batch_impact_uses_only_the_active_batch_decisions():
    scored = pd.DataFrame(
        [
            {
                "CSV row": 2,
                "step": 1,
                "type": "TRANSFER",
                "amount": 100.0,
                "oldbalanceOrg": 120.0,
                "oldbalanceDest": 0.0,
                "Fraud score (%)": 80.0,
                "Decision": "Flag for review",
            },
            {
                "CSV row": 3,
                "step": 2,
                "type": "PAYMENT",
                "amount": 50.0,
                "oldbalanceOrg": 100.0,
                "oldbalanceDest": 10.0,
                "Fraud score (%)": 2.0,
                "Decision": "Below review threshold",
            },
            {
                "CSV row": 4,
                "step": 3,
                "type": "CASH_OUT",
                "amount": 10.0,
                "oldbalanceOrg": 12.0,
                "oldbalanceDest": 0.0,
                "Fraud score (%)": 70.0,
                "Decision": "Flag for review",
            },
        ]
    )
    feedback = [
        {"alert_id": "TXN-000001", "decision": "Confirm fraud"},
        {"alert_id": "TXN-000003", "decision": "False positive"},
        {"alert_id": "TXN-999999", "decision": "Confirm fraud"},
    ]

    impact = calculate_uploaded_batch_impact(
        scored,
        feedback,
        review_cost_per_alert=5,
        false_confirmation_cost=10,
        loss_multiplier=2,
    )

    assert impact["reviewed_count"] == 2
    assert impact["confirmed_count"] == 1
    assert impact["exposure_without_triage"] == pytest.approx(176)
    assert impact["exposure_associated_with_confirmed"] == pytest.approx(160)
    assert impact["remaining_exposure"] == pytest.approx(16)
    assert impact["false_confirmation_cost"] == pytest.approx(2)
    assert impact["net_value"] == pytest.approx(148)
