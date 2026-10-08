from pathlib import Path

import joblib
import pandas as pd

from fraud_detection.config import DECISION_THRESHOLD, FEATURE_COLUMNS, MODEL_DIR
from fraud_detection.features import engineer_features


def load_best_model(path: str | Path = MODEL_DIR / "best_model.joblib"):
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run the training workflow first."
        )
    return joblib.load(model_path)


def predict_transaction(model, transaction: dict | pd.DataFrame) -> dict:
    data = transaction if isinstance(transaction, pd.DataFrame) else pd.DataFrame([transaction])
    features = engineer_features(data)
    if list(features.columns) != FEATURE_COLUMNS:
        raise RuntimeError("Prediction features do not match the training feature schema.")
    probability = float(model.predict_proba(features)[:, 1][0])
    return {
        "prediction": int(probability >= DECISION_THRESHOLD),
        "fraud_probability": probability,
        "threshold": DECISION_THRESHOLD,
    }
