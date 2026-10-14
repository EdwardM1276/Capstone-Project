import numpy as np
import pandas as pd

from fraud_detection.config import FEATURE_COLUMNS
from fraud_detection.evaluation import chronological_split_indices
from fraud_detection.features import engineer_features
from fraud_detection.models import make_model_pipeline


def make_audit_data():
    """Create small artificial data for testing pipeline safeguards."""
    n = 24
    labels = np.array([0] * 16 + [1] * 8)

    data = pd.DataFrame(
        {
            "step": np.arange(1, n + 1),
            "type": ["PAYMENT"] * 16 + ["TRANSFER"] * 8,
            "amount": np.arange(10, 10 + n, dtype=float),
            "oldbalanceOrg": np.arange(100, 100 + n, dtype=float),
            "oldbalanceDest": np.arange(200, 200 + n, dtype=float),
            "newbalanceOrig": np.full(n, 999999.0),
            "newbalanceDest": np.full(n, 888888.0),
            "nameOrig": [f"sender-{i}" for i in range(n)],
            "nameDest": [f"receiver-{i}" for i in range(n)],
            "isFlaggedFraud": labels,
            "isFraud": labels,
        }
    )
    return data, labels


def test_forbidden_columns_are_excluded_from_predictors():
    data, _ = make_audit_data()
    features = engineer_features(data)

    forbidden = {
        "isFraud",
        "isFlaggedFraud",
        "newbalanceOrig",
        "newbalanceDest",
        "nameOrig",
        "nameDest",
        "step",
    }

    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert forbidden.isdisjoint(features.columns)

    changed = data.copy()
    for column in forbidden:
        if column != "step":
            changed[column] = "changed-for-audit"

    pd.testing.assert_frame_equal(
        features,
        engineer_features(changed),
    )


def test_chronological_partitions_are_disjoint_and_complete():
    steps = np.repeat(np.arange(1, 16), 3)
    steps = np.random.default_rng(42).permutation(steps)

    train, validation, test = chronological_split_indices(steps)

    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)

    combined = np.concatenate([train, validation, test])
    assert len(combined) == len(steps)
    assert len(np.unique(combined)) == len(steps)

    assert steps[train].max() < steps[validation].min()
    assert steps[validation].max() < steps[test].min()

    assert set(steps[train]).isdisjoint(steps[validation])
    assert set(steps[train]).isdisjoint(steps[test])
    assert set(steps[validation]).isdisjoint(steps[test])


def test_preprocessing_statistics_use_training_data_only():
    data, labels = make_audit_data()
    X_train = engineer_features(data)

    pipeline = make_model_pipeline(
        "Logistic Regression",
        "class_weight",
        positive_weight=2.0,
    )
    pipeline.fit(X_train, labels)

    preprocessor = pipeline.named_steps["preprocessor"]
    numeric = preprocessor.named_transformers_["numeric"]
    imputer = numeric.named_steps["imputer"]
    scaler = numeric.named_steps["scaler"]

    numeric_columns = list(preprocessor.transformers_[0][2])
    training_values = X_train[numeric_columns].to_numpy(dtype=float)

    np.testing.assert_allclose(
        imputer.statistics_,
        np.median(training_values, axis=0),
    )
    np.testing.assert_allclose(
        scaler.mean_,
        training_values.mean(axis=0),
    )

    original_medians = imputer.statistics_.copy()
    original_means = scaler.mean_.copy()
    original_categories = [
        category.copy()
        for category in preprocessor.named_transformers_["categorical"]
        .named_steps["onehot"].categories_
    ]

    held_out = data.iloc[:3].copy()
    held_out["amount"] = 1_000_000.0
    held_out["oldbalanceOrg"] = 2_000_000.0
    held_out["oldbalanceDest"] = 3_000_000.0
    held_out["type"] = "CASH_IN"

    probabilities = pipeline.predict_proba(engineer_features(held_out))

    assert probabilities.shape == (3, 2)
    np.testing.assert_array_equal(imputer.statistics_, original_medians)
    np.testing.assert_array_equal(scaler.mean_, original_means)

    current_categories = (
        preprocessor.named_transformers_["categorical"]
        .named_steps["onehot"].categories_
    )
    for before, after in zip(original_categories, current_categories):
        np.testing.assert_array_equal(before, after)


def test_smote_is_not_called_during_held_out_prediction(monkeypatch):
    data, labels = make_audit_data()
    X_train = engineer_features(data)

    pipeline = make_model_pipeline(
        "Logistic Regression",
        "smote",
        positive_weight=2.0,
    )

    sampler = pipeline.named_steps["sampler"]
    original_fit_resample = sampler.fit_resample
    calls = []

    def record_training_resample(X, y, **kwargs):
        calls.append((X.shape[0], len(y)))
        return original_fit_resample(X, y, **kwargs)

    monkeypatch.setattr(sampler, "fit_resample", record_training_resample)

    pipeline.fit(X_train, labels)
    assert calls == [(len(X_train), len(labels))]

    def reject_prediction_resampling(*args, **kwargs):
        raise AssertionError("SMOTE was called during held-out prediction.")

    monkeypatch.setattr(
        sampler,
        "fit_resample",
        reject_prediction_resampling,
    )

    held_out = data.iloc[:5].copy()
    held_out["amount"] = held_out["amount"] + 500.0
    probabilities = pipeline.predict_proba(engineer_features(held_out))

    assert probabilities.shape == (5, 2)
    assert np.isfinite(probabilities).all()
    assert calls == [(len(X_train), len(labels))]