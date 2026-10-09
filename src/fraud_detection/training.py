import json
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud_detection.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_MAX_ROWS,
    FIGURE_DIR,
    HIGH_FRAUD_PROBABILITY_MIN,
    LOW_FRAUD_PROBABILITY_MAX,
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET_COLUMN,
)
from fraud_detection.data import load_dataset, validate_csv
from fraud_detection.evaluation import (
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    MINIMUM_RECALL,
    chronological_split_indices,
    evaluate_predictions,
    hourly_bootstrap_intervals,
    mark_pareto_frontier,
    probability_band,
    select_threshold_for_recall,
    weighted_error_cost,
)
from fraud_detection.features import engineer_features
from fraud_detection.models import (
    IMBALANCE_STRATEGIES,
    MODEL_NAMES,
    fit_kwargs,
    make_model_pipeline,
)
from fraud_detection.visualization import (
    save_class_distribution,
    save_model_comparison_figures,
)

DEFAULT_MODELS = list(MODEL_NAMES)
DEFAULT_STRATEGIES = list(IMBALANCE_STRATEGIES)


def run_training(
    input_path: str | Path = DEFAULT_DATA_PATH,
    max_rows: int | None = DEFAULT_MAX_ROWS,
    model_names: list[str] | None = None,
    strategies: list[str] | None = None,
    low_risk_upper_bound: float = LOW_FRAUD_PROBABILITY_MAX,
    high_risk_lower_bound: float = HIGH_FRAUD_PROBABILITY_MIN,
    minimum_recall_target: float = MINIMUM_RECALL,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
    output_dir: str | Path = MODEL_DIR,
    report_dir: str | Path = REPORT_DIR,
    figure_dir: str | Path = FIGURE_DIR,
) -> pd.DataFrame:
    if not 0 <= low_risk_upper_bound < high_risk_lower_bound <= 1:
        raise ValueError("Risk band cutoffs must satisfy 0 <= low < high <= 1.")
    if not 0 < minimum_recall_target <= 1:
        raise ValueError("Minimum recall target must be in (0, 1].")
    if false_negative_cost < 0 or false_positive_cost < 0:
        raise ValueError("False-positive and false-negative costs must be non-negative.")
    models = model_names or DEFAULT_MODELS
    imbalance_strategies = strategies or DEFAULT_STRATEGIES
    if not models or any(name not in MODEL_NAMES for name in models):
        raise ValueError(f"Model names must be selected from {MODEL_NAMES}.")
    if not imbalance_strategies or any(
        strategy not in IMBALANCE_STRATEGIES for strategy in imbalance_strategies
    ):
        raise ValueError(
            f"Imbalance strategies must be selected from {IMBALANCE_STRATEGIES}."
        )

    output_path = Path(output_dir)
    reports_path = Path(report_dir)
    figures_path = Path(figure_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    started_at = datetime.now(timezone.utc).isoformat()
    run_status_path = reports_path / "run_status.json"
    run_status_path.write_text(
        json.dumps({"run_id": run_id, "status": "running", "started_at": started_at}, indent=2),
        encoding="utf-8",
    )

    source_path = Path(input_path)
    quality_path = reports_path / "data_quality.json"
    quality = None
    if quality_path.is_file():
        cached_quality = json.loads(quality_path.read_text(encoding="utf-8"))
        source_stat = source_path.stat()
        if (
            cached_quality.get("passed") is True
            and cached_quality.get("source") == str(source_path)
            and cached_quality.get("source_size_bytes") == source_stat.st_size
            and cached_quality.get("source_modified_ns") == source_stat.st_mtime_ns
        ):
            quality = cached_quality
    if quality is None:
        quality = validate_csv(input_path, quality_path)
    data = load_dataset(input_path, max_rows=max_rows, random_state=RANDOM_STATE)
    save_class_distribution(
        quality["class_distribution"], figures_path / "class_distribution.png"
    )
    y = pd.to_numeric(data.pop(TARGET_COLUMN), errors="raise").astype("int8")
    steps = data["step"].to_numpy()
    X = engineer_features(data)
    train_indices, validation_indices, test_indices = chronological_split_indices(steps)
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_validation, y_validation = X.iloc[validation_indices], y.iloc[validation_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    steps_test = steps[test_indices]
    for partition_name, labels in (
        ("training", y_train),
        ("validation", y_validation),
        ("test", y_test),
    ):
        fraud_count = int(labels.sum())
        if labels.nunique() != 2:
            raise ValueError(f"The chronological {partition_name} partition lacks one class.")
        if partition_name != "training" and fraud_count < 30:
            raise ValueError(
                f"The chronological {partition_name} partition has only {fraud_count} "
                "fraud cases; use more rows before drawing reliable conclusions."
            )
    positive_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    result_rows = []
    probability_band_rows = []
    model_files = {}

    for strategy in imbalance_strategies:
        for model_name in models:
            key = f"{model_name} [{strategy}]"
            print(f"Training {key}...")
            pipeline = make_model_pipeline(model_name, strategy, positive_weight)
            sample_kwargs = fit_kwargs(model_name, strategy, y_train)
            fit_started = time.perf_counter()
            pipeline.fit(X_train, y_train, **sample_kwargs)
            fit_seconds = time.perf_counter() - fit_started

            validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]
            threshold, validation_metrics = select_threshold_for_recall(
                y_validation, validation_probabilities, minimum_recall_target
            )
            validation_labels = (
                validation_probabilities >= threshold
            ).astype("int8")
            validation_cost = weighted_error_cost(
                y_validation,
                validation_labels,
                false_negative_cost,
                false_positive_cost,
            )

            predict_started = time.perf_counter()
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            predict_seconds = time.perf_counter() - predict_started
            labels, metrics = evaluate_predictions(
                y_test, probabilities, threshold
            )
            false_positives = int(((y_test.to_numpy() == 0) & (labels == 1)).sum())
            false_negatives = int(((y_test.to_numpy() == 1) & (labels == 0)).sum())
            test_cost = weighted_error_cost(
                y_test, labels, false_negative_cost, false_positive_cost
            )
            intervals = hourly_bootstrap_intervals(
                y_test, labels, steps_test, probabilities=probabilities
            )
            bands = probability_band(
                probabilities, low_risk_upper_bound, high_risk_lower_bound
            )
            risk_bands = (
                (0.0, low_risk_upper_bound, "Low"),
                (low_risk_upper_bound, high_risk_lower_bound, "Medium"),
                (high_risk_lower_bound, 1.0, "High"),
            )
            for lower, upper, band_name in risk_bands:
                selected = bands == band_name
                probability_band_rows.append(
                    {
                        "model": model_name,
                        "imbalance_strategy": strategy,
                        "probability_band": band_name,
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "transaction_count": int(selected.sum()),
                        "observed_fraud_rate": (
                            float(y_test.to_numpy()[selected].mean())
                            if selected.any()
                            else np.nan
                        ),
                    }
                )
            result_rows.append(
                {
                    "model": model_name,
                    "imbalance_strategy": strategy,
                    **metrics,
                    "validation_precision_at_recall_target": validation_metrics["precision"],
                    "validation_recall": validation_metrics["recall"],
                    "validation_pr_auc": validation_metrics["pr_auc"],
                    "validation_weighted_cost": validation_cost,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "weighted_cost": test_cost,
                    "cost_per_1000_transactions": test_cost / len(y_test) * 1000,
                    **intervals,
                    "test_rows": int(len(y_test)),
                    "fraud_test_rows": int(y_test.sum()),
                    "validation_rows": int(len(y_validation)),
                    "validation_fraud_rows": int(y_validation.sum()),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "prediction_rows_per_second": (
                        float(len(y_test) / predict_seconds)
                        if predict_seconds > 0
                        else np.inf
                    ),
                    "threshold": threshold,
                    "validation_threshold_recall_target": minimum_recall_target,
                }
            )
            safe_name = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            model_file = output_path / f"{safe_name}.joblib"
            pipeline.fraud_decision_threshold_ = threshold
            joblib.dump(pipeline, model_file)
            model_files[key] = model_file
            print(
                f"  PR-AUC={metrics['pr_auc']:.4f} "
                f"recall={metrics['recall']:.4f} "
                f"precision={metrics['precision']:.4f}"
            )

    results = pd.DataFrame(result_rows)
    results["pareto_efficient"] = mark_pareto_frontier(results)
    results = results.sort_values(
        ["validation_weighted_cost", "validation_precision_at_recall_target"],
        ascending=[True, False],
    )
    results.to_csv(reports_path / "model_comparison.csv", index=False)
    strategy_summary = results.groupby("imbalance_strategy", as_index=False).agg(
        mean_pr_auc=("pr_auc", "mean"),
        mean_test_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_validation_cost=("validation_weighted_cost", "mean"),
        mean_test_weighted_cost=("weighted_cost", "mean"),
        mean_fit_seconds=("fit_seconds", "mean"),
        mean_prediction_rows_per_second=("prediction_rows_per_second", "mean"),
        model_count=("model", "count"),
    )
    strategy_summary.to_csv(reports_path / "imbalance_strategy_comparison.csv", index=False)
    probability_bands = pd.DataFrame(probability_band_rows)
    probability_bands.to_csv(reports_path / "probability_bands.csv", index=False)
    save_model_comparison_figures(results, figures_path, probability_bands)

    best_row = results.iloc[0]
    best_key = f"{best_row['model']} [{best_row['imbalance_strategy']}]"
    best_model_path = output_path / "best_model.joblib"
    shutil.copyfile(model_files[best_key], best_model_path)
    metadata = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model": str(best_row["model"]),
        "imbalance_strategy": str(best_row["imbalance_strategy"]),
        "selection_metric": "validation_weighted_cost_at_recall_target",
        "metrics": {
            metric: float(best_row[metric])
            for metric in (
                "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"
            )
        },
        "threshold": float(best_row["threshold"]),
        "random_state": RANDOM_STATE,
        "split_strategy": "chronological_by_step",
        "split_fractions": {"training": 0.60, "validation": 0.20, "test": 0.20},
        "split_step_ranges": {
            "training": [int(steps[train_indices].min()), int(steps[train_indices].max())],
            "validation": [
                int(steps[validation_indices].min()),
                int(steps[validation_indices].max()),
            ],
            "test": [int(steps[test_indices].min()), int(steps[test_indices].max())],
        },
        "minimum_recall_target": minimum_recall_target,
        "error_costs": {
            "false_negative": false_negative_cost,
            "false_positive": false_positive_cost,
        },
        "validation_metrics": {
            "precision": float(best_row["validation_precision_at_recall_target"]),
            "recall": float(best_row["validation_recall"]),
            "weighted_cost": float(best_row["validation_weighted_cost"]),
        },
        "source_rows": int(quality["row_count"]),
        "sampled_rows": int(len(data)),
        "full_dataset_used": (
            max_rows is None or max_rows <= 0 or len(data) >= quality["row_count"]
        ),
        "probability_bands": [
            {"lower_bound": 0.0, "upper_bound": low_risk_upper_bound, "label": "Low"},
            {
                "lower_bound": low_risk_upper_bound,
                "upper_bound": high_risk_lower_bound,
                "label": "Medium",
            },
            {"lower_bound": high_risk_lower_bound, "upper_bound": 1.0, "label": "High"},
        ],
        "training_rows": int(len(y_train)),
        "training_fraud_rows": int(y_train.sum()),
        "validation_rows": int(len(y_validation)),
        "validation_fraud_rows": int(y_validation.sum()),
        "test_rows": int(len(y_test)),
        "fraud_test_rows": int(y_test.sum()),
        "best_model_file": best_model_path.name,
    }
    (output_path / "best_model.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (reports_path / "training_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    current_figures = {
        "class_distribution.png",
        "model_strategy_pr_auc_heatmap.png",
        "performance_dashboard.png",
        "pareto_dashboard.png",
        "probability_band_risk_map.png",
    }
    for obsolete_figure in figures_path.glob("*.png"):
        if obsolete_figure.name not in current_figures:
            obsolete_figure.unlink()
    run_status_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "complete",
                "started_at": started_at,
                "completed_at": metadata["completed_at"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Best model: {best_key} "
        f"(validation cost={best_row['validation_weighted_cost']:.0f}, "
        f"test recall={best_row['recall']:.4f})"
    )
    return results
