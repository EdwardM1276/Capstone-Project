from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

sns.set_theme(style="whitegrid")


def save_class_distribution(data: pd.DataFrame | dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, pd.DataFrame):
        counts = data["isFraud"].value_counts().reindex([0, 1], fill_value=0)
    else:
        counts = pd.Series(
            {label: int(data.get(str(label), 0)) for label in (0, 1)}
        ).reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=["Legitimate", "Fraud"], y=counts.values, color="#315a7d", ax=ax)
    ax.set(title="PaySim transaction class distribution", xlabel="", ylabel="Transactions")
    for index, count in enumerate(counts.values):
        ax.text(index, count, f"{count:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def save_evaluation_figures(
    y_true,
    predictions: dict[str, dict],
    output_dir: str | Path,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for label, result in predictions.items():
        safe_name = "_".join(part.lower() for part in label.replace("/", " ").split())
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_true, result["labels"], display_labels=["Legitimate", "Fraud"], ax=ax
        )
        ax.set_title(f"Confusion matrix: {label}")
        fig.tight_layout()
        fig.savefig(destination / f"{safe_name}_confusion_matrix.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_true, result["probabilities"], ax=ax)
        ax.set_title(f"ROC curve: {label}")
        fig.tight_layout()
        fig.savefig(destination / f"{safe_name}_roc_curve.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(y_true, result["probabilities"], ax=ax)
        ax.set_title(f"Precision-recall curve: {label}")
        fig.tight_layout()
        fig.savefig(destination / f"{safe_name}_precision_recall_curve.png", dpi=160)
        plt.close(fig)


def save_feature_importance(
    fitted_pipeline,
    label: str,
    output_dir: str | Path,
    limit: int = 20,
) -> Path | None:
    classifier = fitted_pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = abs(classifier.coef_[0])
    else:
        return None
    names = fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()
    ranking = (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(limit)
        .sort_values("importance")
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    filename = "_".join(part.lower() for part in label.replace("/", " ").split())
    path = destination / f"{filename}_feature_importance.png"
    fig, ax = plt.subplots(figsize=(9, max(4, len(ranking) * 0.3)))
    sns.barplot(data=ranking, x="importance", y="feature", color="#315a7d", ax=ax)
    ax.set(title=f"Feature importance: {label}", xlabel="Importance", ylabel="")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
