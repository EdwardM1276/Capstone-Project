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
            "oldbalanceOrg": [181.0, 100.0],
            "oldbalanceDest": [0.0, 50.0],
        }
    )

    features = engineer_features(data)

    assert list(features.columns) == FEATURE_COLUMNS
    assert not {"step", "newbalanceOrig", "newbalanceDest", "isFlaggedFraud", "isFraud"}.intersection(features.columns)
    assert features["hour"].tolist() == [1, 0]
    assert np.isclose(features.loc[0, "log_amount"], np.log1p(181.0))
    assert np.isclose(features.loc[0, "transaction_to_origin_balance"], 181 / 182)


def test_feature_engineering_does_not_require_post_transaction_fields():
    data = pd.DataFrame(
        {
            "step": [1],
            "type": ["TRANSFER"],
            "amount": [10.0],
            "oldbalanceOrg": [20.0],
            "oldbalanceDest": [0.0],
        }
    )

    features = engineer_features(data)

    assert len(features) == 1
