import numpy as np
import pandas as pd

from fraud_detection.evaluation import (
    chronological_split_indices,
    evaluate_predictions,
    hourly_bootstrap_intervals,
    mark_pareto_frontier,
    probability_band,
    select_threshold_for_recall,
    weighted_error_cost,
)


def test_metrics_use_probabilities_and_fixed_threshold():
    y_true = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8])

    labels, metrics = evaluate_predictions(y_true, probabilities)

    assert labels.tolist() == [0, 1, 0, 1]
    assert {key: value for key, value in metrics.items() if key != "brier_score"} == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "roc_auc": 1.0,
        "pr_auc": 1.0,
    }
    assert np.isclose(metrics["brier_score"], 0.025)


def test_probability_bands_use_documented_boundaries():
    probabilities = np.array([0.0, 0.009, 0.01, 0.049, 0.05, 1.0])

    assert probability_band(probabilities).tolist() == [
        "Low",
        "Low",
        "Medium",
        "Medium",
        "High",
        "High",
    ]
    assert probability_band(probabilities, 0.02, 0.1).tolist() == [
        "Low",
        "Low",
        "Low",
        "Medium",
        "Medium",
        "High",
    ]


def test_probability_bands_reject_overlapping_cutoffs():
    with np.testing.assert_raises(ValueError):
        probability_band(np.array([0.1]), 0.1, 0.1)


def test_pareto_frontier_marks_undominated_quality_speed_pairs():
    results = pd.DataFrame(
        {
            "pr_auc": [0.80, 0.90, 0.75],
            "prediction_rows_per_second": [100.0, 50.0, 75.0],
        }
    )

    assert mark_pareto_frontier(results).tolist() == [True, True, False]


def test_chronological_split_keeps_each_hour_in_one_partition():
    steps = np.repeat(np.arange(1, 6), 2)

    train, validation, test = chronological_split_indices(
        steps, train_fraction=0.6, validation_fraction=0.2
    )

    assert steps[train].tolist() == [1, 1, 2, 2, 3, 3]
    assert steps[validation].tolist() == [4, 4]
    assert steps[test].tolist() == [5, 5]
    assert steps[train].max() < steps[validation].min()
    assert steps[validation].max() < steps[test].min()


def test_threshold_meets_recall_floor_and_maximizes_validation_precision():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.1])

    threshold, metrics = select_threshold_for_recall(y_true, probabilities, 2 / 3)

    assert threshold == 0.8
    assert metrics["recall"] >= 2 / 3
    assert metrics["precision"] == 1.0


def test_weighted_cost_and_hourly_bootstrap_are_reported():
    y_true = np.array([1, 1, 0, 0])
    labels = np.array([1, 0, 1, 0])
    steps = np.array([1, 1, 2, 2])

    assert weighted_error_cost(y_true, labels) == 11.0
    intervals = hourly_bootstrap_intervals(
        y_true,
        labels,
        steps,
        probabilities=np.array([0.9, 0.6, 0.8, 0.1]),
        n_bootstrap=100,
    )
    assert 0 <= intervals["precision_ci_low"] <= intervals["precision_ci_high"] <= 1
    assert 0 <= intervals["recall_ci_low"] <= intervals["recall_ci_high"] <= 1
    assert 0 <= intervals["pr_auc_ci_low"] <= intervals["pr_auc_ci_high"] <= 1
