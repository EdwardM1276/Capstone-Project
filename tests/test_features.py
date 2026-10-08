import numpy as np
import pandas as pd

from fraud_detection.config import FEATURE_COLUMNS
from fraud_detection.features import engineer_features


def test_engineered_features_and_identifier_exclusion():
    data = pd.DataFrame(
        {
            "step": [25, 48],
            "type": ["TRANSFER", "PAYMENT"],
            "amount": [181.0, 25.0],
            "nameOrig": ["sender-1", "sender-2"],
            "oldbalanceOrg": [181.0, 100.0],
            "newbalanceOrig": [0.0, 75.0],
            "nameDest": ["receiver-1", "receiver-2"],
            "oldbalanceDest": [0.0, 50.0],
            "newbalanceDest": [0.0, 75.0],
            "isFraud": [1, 0],
            "isFlaggedFraud": [0, 0],
        }
    )

    features = engineer_features(data)

    assert list(features.columns) == FEATURE_COLUMNS
    assert "nameOrig" not in features.columns
    assert "nameDest" not in features.columns
    assert "isFraud" not in features.columns
    assert features["hour"].tolist() == [1, 0]
    assert features["origin_balance_change"].tolist() == [181.0, 25.0]
    assert features["destination_balance_change"].tolist() == [0.0, 25.0]
    assert features["account_drained"].tolist() == [1, 0]
    assert np.isclose(features.loc[0, "log_amount"], np.log1p(181.0))
    assert features.loc[0, "origin_balance_error"] == 0
    assert np.isclose(features.loc[0, "transaction_to_origin_balance"], 181 / 182)
