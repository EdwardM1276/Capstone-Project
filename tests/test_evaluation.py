import numpy as np

from fraud_detection.evaluation import evaluate_predictions


def test_metrics_use_probabilities_and_fixed_threshold():
    y_true = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8])

    labels, metrics = evaluate_predictions(y_true, probabilities)

    assert labels.tolist() == [0, 1, 0, 1]
    assert metrics == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "roc_auc": 1.0,
        "pr_auc": 1.0,
    }
