import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_detection.config import DECISION_THRESHOLD


def evaluate_predictions(
    y_true,
    probabilities,
    threshold: float = DECISION_THRESHOLD,
) -> tuple[np.ndarray, dict[str, float]]:
    labels = (probabilities >= threshold).astype("int8")
    metrics = {
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }
    return labels, metrics
