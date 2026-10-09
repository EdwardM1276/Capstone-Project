import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_detection.config import (
    DECISION_THRESHOLD,
    HIGH_FRAUD_PROBABILITY_MIN,
    LOW_FRAUD_PROBABILITY_MAX,
)

PROBABILITY_BANDS = (
    (0.0, LOW_FRAUD_PROBABILITY_MAX, "Low"),
    (LOW_FRAUD_PROBABILITY_MAX, HIGH_FRAUD_PROBABILITY_MIN, "Medium"),
    (HIGH_FRAUD_PROBABILITY_MIN, 1.0, "High"),
)
MINIMUM_RECALL = 0.90
FALSE_NEGATIVE_COST = 10.0
FALSE_POSITIVE_COST = 1.0


def chronological_split_indices(
    steps,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(steps)
    test_fraction = 1.0 - train_fraction - validation_fraction
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or test_fraction <= 0:
        raise ValueError("Train, validation, and test fractions must all be positive.")
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("Step values must be a finite one-dimensional array.")

    step_values = np.unique(values)
    if len(step_values) < 3:
        raise ValueError("At least three distinct time steps are required.")
    train_end_index = int(np.ceil(len(step_values) * train_fraction)) - 1
    validation_end_index = (
        int(np.ceil(len(step_values) * (train_fraction + validation_fraction))) - 1
    )
    if train_end_index >= validation_end_index or validation_end_index >= len(step_values) - 1:
        raise ValueError("Time steps cannot form three non-empty chronological partitions.")

    train_end = step_values[train_end_index]
    validation_end = step_values[validation_end_index]
    train_indices = np.flatnonzero(values <= train_end)
    validation_indices = np.flatnonzero((values > train_end) & (values <= validation_end))
    test_indices = np.flatnonzero(values > validation_end)
    return train_indices, validation_indices, test_indices


def select_threshold_for_recall(
    y_true,
    probabilities,
    minimum_recall: float = MINIMUM_RECALL,
) -> tuple[float, dict[str, float]]:
    if not 0 < minimum_recall <= 1:
        raise ValueError("Minimum recall must be in (0, 1].")
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    feasible = np.flatnonzero(recall[:-1] >= minimum_recall)
    if len(feasible) == 0 or len(thresholds) == 0:
        raise ValueError("Validation data cannot satisfy the requested recall target.")
    best_precision = precision[feasible].max()
    best_indices = feasible[precision[feasible] == best_precision]
    best_index = best_indices[-1]
    threshold = float(thresholds[best_index])
    labels, metrics = evaluate_predictions(y_true, probabilities, threshold)
    metrics["false_positives"] = float(((np.asarray(y_true) == 0) & (labels == 1)).sum())
    metrics["false_negatives"] = float(((np.asarray(y_true) == 1) & (labels == 0)).sum())
    return threshold, metrics


def weighted_error_cost(
    y_true,
    labels,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
) -> float:
    if false_negative_cost < 0 or false_positive_cost < 0:
        raise ValueError("Error costs must be non-negative.")
    truth = np.asarray(y_true)
    predictions = np.asarray(labels)
    false_negatives = int(((truth == 1) & (predictions == 0)).sum())
    false_positives = int(((truth == 0) & (predictions == 1)).sum())
    return float(
        false_negative_cost * false_negatives
        + false_positive_cost * false_positives
    )


def hourly_bootstrap_intervals(
    y_true,
    labels,
    steps,
    probabilities=None,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> dict[str, float]:
    truth = np.asarray(y_true)
    predictions = np.asarray(labels)
    step_values, inverse = np.unique(np.asarray(steps), return_inverse=True)
    if len(truth) != len(predictions) or len(truth) != len(inverse):
        raise ValueError("Labels, predictions, and steps must have equal lengths.")
    if n_bootstrap < 1 or len(step_values) < 2:
        raise ValueError("At least two hours and one bootstrap replicate are required.")

    true_positive = np.bincount(
        inverse, weights=((truth == 1) & (predictions == 1)), minlength=len(step_values)
    )
    false_positive = np.bincount(
        inverse, weights=((truth == 0) & (predictions == 1)), minlength=len(step_values)
    )
    false_negative = np.bincount(
        inverse, weights=((truth == 1) & (predictions == 0)), minlength=len(step_values)
    )
    rng = np.random.default_rng(random_state)
    sampled_hours = rng.integers(0, len(step_values), size=(n_bootstrap, len(step_values)))
    sampled_tp = true_positive[sampled_hours].sum(axis=1)
    sampled_fp = false_positive[sampled_hours].sum(axis=1)
    sampled_fn = false_negative[sampled_hours].sum(axis=1)
    sampled_precision = np.divide(
        sampled_tp,
        sampled_tp + sampled_fp,
        out=np.zeros(n_bootstrap, dtype=float),
        where=(sampled_tp + sampled_fp) > 0,
    )
    sampled_recall = np.divide(
        sampled_tp,
        sampled_tp + sampled_fn,
        out=np.zeros(n_bootstrap, dtype=float),
        where=(sampled_tp + sampled_fn) > 0,
    )
    intervals = {
        "precision_ci_low": float(np.quantile(sampled_precision, 0.025)),
        "precision_ci_high": float(np.quantile(sampled_precision, 0.975)),
        "recall_ci_low": float(np.quantile(sampled_recall, 0.025)),
        "recall_ci_high": float(np.quantile(sampled_recall, 0.975)),
    }
    if probabilities is not None:
        scores = np.asarray(probabilities)
        if len(scores) != len(truth):
            raise ValueError("Probabilities must have the same length as labels.")
        grouped_indices = [np.flatnonzero(inverse == hour) for hour in range(len(step_values))]
        sampled_pr_auc = []
        for hours in sampled_hours:
            sampled_indices = np.concatenate([grouped_indices[hour] for hour in hours])
            sampled_truth = truth[sampled_indices]
            if np.unique(sampled_truth).size == 2:
                sampled_pr_auc.append(
                    average_precision_score(sampled_truth, scores[sampled_indices])
                )
        if not sampled_pr_auc:
            raise ValueError("Hourly bootstrap samples did not contain both classes.")
        intervals["pr_auc_ci_low"] = float(np.quantile(sampled_pr_auc, 0.025))
        intervals["pr_auc_ci_high"] = float(np.quantile(sampled_pr_auc, 0.975))
    return intervals


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
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }
    return labels, metrics


def probability_band(
    probabilities,
    low_upper_bound: float = LOW_FRAUD_PROBABILITY_MAX,
    high_lower_bound: float = HIGH_FRAUD_PROBABILITY_MIN,
) -> np.ndarray:
    values = np.asarray(probabilities)
    if not 0 <= low_upper_bound < high_lower_bound <= 1:
        raise ValueError("Risk band cutoffs must satisfy 0 <= low < high <= 1.")
    return np.select(
        [
            values < low_upper_bound,
            (values >= low_upper_bound) & (values < high_lower_bound),
        ],
        ["Low", "Medium"],
        default="High",
    )


def mark_pareto_frontier(results):
    """Mark rows not dominated on PR-AUC and prediction throughput."""
    frontier = np.ones(len(results), dtype=bool)
    quality = results["pr_auc"].to_numpy()
    throughput = results["prediction_rows_per_second"].to_numpy()
    for index, (score, speed) in enumerate(zip(quality, throughput)):
        dominates = (quality >= score) & (throughput >= speed) & (
            (quality > score) | (throughput > speed)
        )
        dominates[index] = False
        frontier[index] = not dominates.any()
    return frontier
