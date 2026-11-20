import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import REQUIRED_COLUMNS
from fraud_detection.data import load_dataset
from fraud_detection.review import (
    calculate_cost_impact,
    explain_alert,
    load_review_transactions,
    score_review_batch,
)
from scripts.create_review_samples import create_samples, training_sample_indices


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)
        self.feature_columns = None

    def predict_proba(self, features):
        self.feature_columns = list(features.columns)
        return np.column_stack((1 - self.probabilities, self.probabilities))


def full_paysim_rows(count=20):
    rows = []
    for index in range(count):
        fraud = int(index % 5 == 0)
        rows.append(
            [
                index // 2 + 1,
                "TRANSFER" if fraud else "PAYMENT",
                float(index + 1),
                f"origin-{index}",
                float(index + 20),
                float(index + 19),
                f"destination-{index}",
                float(index + 30),
                float(index + 31),
                fraud,
                0,
            ]
        )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_upload_validation_accepts_scoring_schema_and_optional_label():
    rows = full_paysim_rows(2)
    loaded = load_review_transactions(rows.to_csv(index=False).encode("utf-8"))

    assert len(loaded) == 2
    assert loaded["isFraud"].tolist() == [1, 0]
    assert loaded["step"].tolist() == [1, 1]


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("amount", -1, "non-negative"),
        ("step", 1.5, "whole numbers"),
        ("type", "UNKNOWN", "type must contain"),
        ("isFraud", 2, "only 0 or 1"),
    ],
)
def test_upload_validation_rejects_invalid_values(column, value, message):
    row = full_paysim_rows(1)
    if column == "step":
        row[column] = row[column].astype(float)
    row.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        load_review_transactions(row.to_csv(index=False).encode("utf-8"))


def test_upload_validation_requires_model_inputs():
    rows = full_paysim_rows(1).drop(columns=["oldbalanceDest"])

    with pytest.raises(ValueError, match="oldbalanceDest"):
        load_review_transactions(rows.to_csv(index=False).encode("utf-8"))


def test_scoring_uses_only_pre_transaction_features_and_orders_alerts_by_risk():
    rows = full_paysim_rows(4)
    rows["newbalanceOrig"] = 0.0
    rows["alert_id"] = "source-id"
    rows["fraud_probability"] = 0.99
    model = FixedProbabilityModel([0.049, 0.06, 0.009, 0.7])

    scored = score_review_batch(rows, model, threshold=0.05)

    assert model.feature_columns is not None
    assert "isFraud" not in model.feature_columns
    assert "newbalanceOrig" not in model.feature_columns
    assert scored["alert_id"].nunique() == 4
    assert scored["risk_band"].tolist() == ["High", "High", "Medium", "Low"]
    assert scored["fraud_probability"].tolist() == [0.7, 0.06, 0.049, 0.009]
    assert scored["predicted_fraud"].tolist() == [1, 1, 0, 0]
    assert 0 <= scored["review_probability"].min() <= scored["review_probability"].max() <= 1
    assert scored["review_probability"].iloc[0] == pytest.approx(0.76)
    assert "none alone proves fraud" in explain_alert(scored.iloc[0])


def test_cost_impact_uses_labels_only_for_ground_truth_summary():
    transactions = pd.DataFrame(
        {
            "alert_id": ["TXN-1", "TXN-2", "TXN-3"],
            "amount": [100.0, 200.0, 300.0],
            "fraud_probability": [0.9, 0.2, 0.8],
            "isFraud": [1, 0, 1],
        }
    )
    decisions = {
        "TXN-1": {"decision": "Confirm fraud"},
        "TXN-2": {"decision": "Confirm fraud"},
    }

    impact = calculate_cost_impact(transactions, decisions, 5, 7, 1)

    assert impact["has_ground_truth"] is True
    assert impact["reviewed_count"] == 2
    assert impact["uncontrolled_exposure"] == 400
    assert impact["captured_exposure"] == 100
    assert impact["remaining_exposure"] == 300
    assert impact["false_confirmations"] == 1
    assert impact["total_process_cost"] == 17
    assert impact["net_savings"] == 83


def test_cost_impact_uses_probability_weighting_without_labels():
    transactions = pd.DataFrame(
        {
            "alert_id": ["TXN-1", "TXN-2"],
            "amount": [100.0, 200.0],
            "fraud_probability": [0.5, 0.25],
        }
    )
    decisions = {"TXN-1": {"decision": "Confirm fraud"}}

    impact = calculate_cost_impact(transactions, decisions, 2, 4, 1)

    assert impact["has_ground_truth"] is False
    assert impact["uncontrolled_exposure"] == 100
    assert impact["captured_exposure"] == 50
    assert impact["remaining_exposure"] == 50
    assert impact["false_confirmations"] == 0.5
    assert impact["total_process_cost"] == 4
    assert impact["net_savings"] == 46


def test_sample_generator_excludes_training_sample_and_keeps_chronology(tmp_path):
    source = tmp_path / "paysim.csv"
    output = tmp_path / "samples"
    rows = full_paysim_rows()
    rows.to_csv(source, index=False)
    training_indices, _ = training_sample_indices(source, sample_size=4)
    training_rows = set(rows.loc[training_indices, "nameOrig"])
    trained = load_dataset(source, max_rows=4)
    expected_training_rows = set(
        zip(rows.loc[training_indices, "step"], rows.loc[training_indices, "amount"])
    )
    loaded_training_rows = set(zip(trained["step"], trained["amount"]))
    assert expected_training_rows == loaded_training_rows

    manifest = create_samples(
        source,
        output,
        file_count=2,
        rows_per_file=4,
        seed=17,
        training_sample_rows=4,
    )

    sampled_frames = [
        pd.read_csv(output / item["file"]) for item in manifest["files"]
    ]
    for sampled in sampled_frames:
        assert len(sampled) == 4
        assert sampled["step"].is_monotonic_increasing
        assert not training_rows.intersection(sampled["nameOrig"])
    assert sum(len(sampled) for sampled in sampled_frames) == 8
    sample_identity = {
        (step, amount)
        for sampled in sampled_frames
        for step, amount in zip(sampled["step"], sampled["amount"])
    }
    assert len(sample_identity) == 8
    assert not sample_identity.intersection(loaded_training_rows)
