import json

import pandas as pd

from fraud_detection.config import REQUIRED_COLUMNS
from fraud_detection.prediction import load_best_model, predict_transaction
from fraud_detection.training import run_training


def test_training_uses_temporal_windows_and_validation_threshold(tmp_path):
    rows = []
    for step in range(1, 16):
        for index in range(100):
            is_fraud = int(index < 40)
            amount = float(1000 + index if is_fraud else 10 + index)
            rows.append(
                {
                    "step": step,
                    "type": "TRANSFER" if is_fraud else "PAYMENT",
                    "amount": amount,
                    "nameOrig": f"sender-{step}-{index}",
                    "oldbalanceOrg": amount + 500.0,
                    "newbalanceOrig": 500.0,
                    "nameDest": f"receiver-{step}-{index}",
                    "oldbalanceDest": 100.0,
                    "newbalanceDest": 100.0 + amount,
                    "isFraud": is_fraud,
                    "isFlaggedFraud": 0,
                }
            )
    source = tmp_path / "transactions.csv"
    pd.DataFrame(rows, columns=REQUIRED_COLUMNS).to_csv(source, index=False)

    results = run_training(
        input_path=source,
        model_names=["Logistic Regression"],
        strategies=["class_weight"],
        output_dir=tmp_path / "models",
        report_dir=tmp_path / "reports",
        figure_dir=tmp_path / "figures",
    )

    summary = json.loads((tmp_path / "reports" / "training_summary.json").read_text())
    assert len(results) == 1
    assert summary["split_strategy"] == "chronological_by_step"
    assert summary["split_step_ranges"] == {
        "training": [1, 9],
        "validation": [10, 12],
        "test": [13, 15],
    }
    assert summary["full_dataset_used"] is True
    assert results.iloc[0]["validation_recall"] >= 0.90
    assert results.iloc[0]["threshold"] != 0.5
    assert (tmp_path / "models" / "best_model.joblib").is_file()
    model = load_best_model(tmp_path / "models" / "best_model.joblib")
    prediction = predict_transaction(
        model,
        {
            "step": 16,
            "type": "TRANSFER",
            "amount": 1200.0,
            "oldbalanceOrg": 1500.0,
            "oldbalanceDest": 100.0,
        },
    )
    assert prediction["threshold"] == summary["threshold"]


def test_prediction_rejects_saved_model_with_old_feature_schema():
    class OldSchemaModel:
        feature_names_in_ = ["step", "isFlaggedFraud"]

    try:
        predict_transaction(
            OldSchemaModel(),
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 100.0,
                "oldbalanceOrg": 500.0,
                "oldbalanceDest": 0.0,
            },
        )
    except RuntimeError as error:
        assert "outdated feature schema" in str(error)
    else:
        raise AssertionError("Expected the old model feature schema to be rejected.")