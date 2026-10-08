import numpy as np
import pandas as pd

from fraud_detection.config import FEATURE_COLUMNS, TARGET_COLUMN


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build proposal features and remove raw identifiers and the target."""
    required = {
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Cannot engineer features; missing columns: {', '.join(missing)}")

    features = data.copy()
    for column in required - {"type"}:
        features[column] = pd.to_numeric(features[column], errors="raise")

    features["origin_balance_change"] = (
        features["oldbalanceOrg"] - features["newbalanceOrig"]
    )
    features["destination_balance_change"] = (
        features["newbalanceDest"] - features["oldbalanceDest"]
    )
    features["hour"] = features["step"] % 24
    features["log_amount"] = np.log1p(features["amount"].clip(lower=0))
    features["origin_balance_error"] = (
        features["oldbalanceOrg"] - features["amount"] - features["newbalanceOrig"]
    )
    features["destination_balance_error"] = (
        features["newbalanceDest"] - features["oldbalanceDest"] - features["amount"]
    )
    features["transaction_to_origin_balance"] = features["amount"] / (
        features["oldbalanceOrg"] + 1.0
    )
    features["account_drained"] = (
        (features["oldbalanceOrg"] > 0) & (features["newbalanceOrig"] == 0)
    ).astype("int8")
    if "isFlaggedFraud" not in features:
        features["isFlaggedFraud"] = 0

    unknown_features = sorted(set(FEATURE_COLUMNS) - set(features.columns))
    if unknown_features:
        raise RuntimeError(f"Feature engineering did not create: {unknown_features}")
    if TARGET_COLUMN in features.columns:
        features = features.drop(columns=[TARGET_COLUMN])
    return features.loc[:, FEATURE_COLUMNS]
