import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from fraud_detection.config import (
    DEFAULT_DATA_PATH,
    DECISION_THRESHOLD,
    FIGURE_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from fraud_detection.data import load_dataset, validate_csv
from fraud_detection.evaluation import evaluate_predictions
from fraud_detection.features import engineer_features
from fraud_detection.models import (
    IMBALANCE_STRATEGIES,
    MODEL_NAMES,
    fit_kwargs,
    make_model_pipeline,
)
from fraud_detection.visualization import (
    save_class_distribution,
    save_evaluation_figures,
    save_feature_importance,
)

DEFAULT_MODELS = list(MODEL_NAMES)
DEFAULT_STRATEGIES = list(IMBALANCE_STRATEGIES)


def _parameter_distributions(model_name: str) -> dict:
    if model_name == "Random Forest":
        return {
            "classifier__n_estimators": randint(100, 351),
            "classifier__max_depth": [None, 12, 20, 30],
            "classifier__min_samples_leaf": randint(1, 6),
            "classifier__min_samples_split": randint(2, 11),
            "classifier__max_features": ["sqrt", "log2", 0.7],
        }
    if model_name == "XGBoost":
        return {
            "classifier__n_estimators": randint(100, 351),
            "classifier__max_depth": randint(3, 10),
            "classifier__learning_rate": uniform(0.02, 0.28),
            "classifier__subsample": uniform(0.65, 0.35),
            "classifier__colsample_bytree": uniform(0.65, 0.35),
            "classifier__min_child_weight": randint(1, 11),
        }
    raise ValueError(f"Hyperparameter search is not configured for {model_name}.")


def _tune_model(
    model_name: str,
    strategy: str,
    positive_weight: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune_rows: int,
    iterations: int,
) -> dict:
    if tune_rows > 0 and len(y_train) > tune_rows:
        X_tune, _, y_tune, _ = train_test_split(
            X_train,
            y_train,
            train_size=tune_rows,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
    else:
        X_tune, y_tune = X_train, y_train

    search = RandomizedSearchCV(
        estimator=make_model_pipeline(model_name, strategy, positive_weight),
        param_distributions=_parameter_distributions(model_name),
        n_iter=iterations,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=False,
        error_score="raise",
    )
    search.fit(X_tune, y_tune)
    return {
        "best_params": search.best_params_,
        "best_cv_pr_auc": float(search.best_score_),
        "tuning_rows": int(len(y_tune)),
        "iterations": iterations,
    }


def run_training(
    input_path: str | Path = DEFAULT_DATA_PATH,
    max_rows: int | None = None,
    model_names: list[str] | None = None,
    strategies: list[str] | None = None,
    tune_rows: int = 100_000,
    search_iterations: int = 5,
    output_dir: str | Path = MODEL_DIR,
    report_dir: str | Path = REPORT_DIR,
    figure_dir: str | Path = FIGURE_DIR,
) -> pd.DataFrame:
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
    X = engineer_features(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    positive_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    prediction_results = {}
    fitted_models = {}
    result_rows = []
    tuning_results = {}

    for strategy in imbalance_strategies:
        for model_name in models:
            key = f"{model_name} [{strategy}]"
            print(f"Training {key}...")
            best_params = {}
            tuning = None
            if model_name in ("Random Forest", "XGBoost"):
                tuning = _tune_model(
                    model_name,
                    strategy,
                    positive_weight,
                    X_train,
                    y_train,
                    tune_rows,
                    search_iterations,
                )
                best_params = tuning["best_params"]
                tuning_results[key] = tuning

            pipeline = make_model_pipeline(model_name, strategy, positive_weight)
            if best_params:
                pipeline.set_params(**best_params)
            sample_kwargs = fit_kwargs(model_name, strategy, y_train)
            pipeline.fit(X_train, y_train, **sample_kwargs)

            probabilities = pipeline.predict_proba(X_test)[:, 1]
            labels, metrics = evaluate_predictions(
                y_test, probabilities, DECISION_THRESHOLD
            )
            result_rows.append(
                {
                    "model": model_name,
                    "imbalance_strategy": strategy,
                    **metrics,
                    "test_rows": int(len(y_test)),
                    "fraud_test_rows": int(y_test.sum()),
                    "threshold": DECISION_THRESHOLD,
                    "best_cv_pr_auc": (
                        tuning["best_cv_pr_auc"] if tuning is not None else np.nan
                    ),
                }
            )
            prediction_results[key] = {
                "labels": labels,
                "probabilities": probabilities,
            }
            fitted_models[key] = pipeline
            safe_name = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            joblib.dump(pipeline, output_path / f"{safe_name}.joblib")
            save_feature_importance(pipeline, key, figures_path)
            print(
                f"  PR-AUC={metrics['pr_auc']:.4f} "
                f"recall={metrics['recall']:.4f} "
                f"precision={metrics['precision']:.4f}"
            )

    results = pd.DataFrame(result_rows).sort_values(
        ["pr_auc", "recall", "f1"], ascending=False
    )
    results.to_csv(reports_path / "model_comparison.csv", index=False)
    save_evaluation_figures(y_test, prediction_results, figures_path)

    best_row = results.iloc[0]
    best_key = f"{best_row['model']} [{best_row['imbalance_strategy']}]"
    best_model_path = output_path / "best_model.joblib"
    joblib.dump(fitted_models[best_key], best_model_path)
    metadata = {
        "model": str(best_row["model"]),
        "imbalance_strategy": str(best_row["imbalance_strategy"]),
        "selection_metric": "pr_auc",
        "metrics": {
            metric: float(best_row[metric])
            for metric in ("precision", "recall", "f1", "roc_auc", "pr_auc")
        },
        "threshold": DECISION_THRESHOLD,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "source_rows": int(quality["row_count"]),
        "sampled_rows": int(len(data)),
        "training_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "fraud_test_rows": int(y_test.sum()),
        "tuning": tuning_results,
        "best_model_file": best_model_path.name,
    }
    (output_path / "best_model.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (reports_path / "training_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Best model: {best_key} "
        f"(PR-AUC={best_row['pr_auc']:.4f}, Recall={best_row['recall']:.4f})"
    )
    return results
