"""Create a confusion matrix for the saved final fraud-detection model."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from fraud_detection.data import load_dataset
from fraud_detection.evaluation import chronological_split_indices
from fraud_detection.features import engineer_features


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "paysim.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "best_model.json"
OUTPUT_DIR = ROOT / "evidence_deliverable_5" / "figures"
IMAGE_PATH = OUTPUT_DIR / "confusion_matrix.png"
COUNTS_PATH = ROOT / "evidence_deliverable_5" / "confusion_matrix_counts.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)

    threshold = float(metadata["threshold"])
    expected_test_start = int(metadata["split_step_ranges"]["test"][0])
    expected_test_end = int(metadata["split_step_ranges"]["test"][1])

    data = load_dataset(
        path=DATA_PATH,
        max_rows=200_000,
        random_state=42,
    )

    steps = data["step"].to_numpy()
    train_indices, validation_indices, test_indices = (
        chronological_split_indices(steps)
    )

    test_data = data.iloc[test_indices].copy()
    test_steps = test_data["step"].to_numpy()

    if test_steps.min() != expected_test_start or test_steps.max() != expected_test_end:
        raise ValueError(
            "The reproduced test period does not match the saved model metadata. "
            f"Expected steps {expected_test_start}-{expected_test_end}, "
            f"but found {test_steps.min()}-{test_steps.max()}."
        )

    model = joblib.load(MODEL_PATH)
    X_test = engineer_features(test_data)
    y_test = test_data["isFraud"].to_numpy()

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    counts = {
        "model": metadata["model"],
        "imbalance_strategy": metadata["imbalance_strategy"],
        "threshold": threshold,
        "test_rows": int(len(test_data)),
        "fraud_test_rows": int(y_test.sum()),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    with COUNTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(counts, file, indent=2)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted legitimate", "Predicted fraud"],
        yticklabels=["Actual legitimate", "Actual fraud"],
    )
    plt.title(
        "Confusion Matrix - Final XGBoost Model\n"
        f"Decision threshold: {threshold:.4f}"
    )
    plt.tight_layout()
    plt.savefig(IMAGE_PATH, dpi=300)
    plt.close()

    print(json.dumps(counts, indent=2))
    print(f"Saved image: {IMAGE_PATH}")
    print(f"Saved counts: {COUNTS_PATH}")


if __name__ == "__main__":
    main()