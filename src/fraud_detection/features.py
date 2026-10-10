import numpy as np
import pandas as pd

from fraud_detection.config import FEATURE_COLUMNS, TARGET_COLUMN


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build features available before transaction completion."""
    required = {
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Cannot engineer features; missing columns: {', '.join(missing)}")

    features = data.copy()
    numeric_columns = ("step", "amount", "oldbalanceOrg", "oldbalanceDest")
    for column in numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="raise")

    numeric_values = features.loc[:, numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Prediction inputs must contain only finite numeric values.")
    if (features["step"] < 0).any() or (features["step"] % 1 != 0).any():
        raise ValueError("Time steps must be non-negative whole numbers.")
    if (features[["amount", "oldbalanceOrg", "oldbalanceDest"]] < 0).any().any():
        raise ValueError("Amounts and pre-transaction balances must be non-negative.")
    allowed_types = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
    if features["type"].isna().any() or not features["type"].isin(allowed_types).all():
        raise ValueError("Transaction type must be a supported PaySim category.")

    features["hour"] = features["step"] % 24
    features["log_amount"] = np.log1p(features["amount"])
    features["transaction_to_origin_balance"] = features["amount"] / (
        features["oldbalanceOrg"] + 1.0
    )

    unknown_features = sorted(set(FEATURE_COLUMNS) - set(features.columns))
    if unknown_features:
        raise RuntimeError(f"Feature engineering did not create: {unknown_features}")
    if TARGET_COLUMN in features.columns:
        features = features.drop(columns=[TARGET_COLUMN])
    return features.loc[:, FEATURE_COLUMNS]
