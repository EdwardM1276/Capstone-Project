"""Create SHAP and LIME interpretability evidence for the saved final model."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

from fraud_detection.data import load_dataset
from fraud_detection.evaluation import chronological_split_indices
from fraud_detection.features import engineer_features


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "paysim.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "best_model.json"
OUTPUT_DIR = ROOT / "evidence_deliverable_5" / "figures"
SHAP_PATH = OUTPUT_DIR / "shap_summary_plot.png"
LIME_PATH = OUTPUT_DIR / "lime_local_explanation.png"
SUMMARY_PATH = ROOT / "evidence_deliverable_5" / "interpretability_summary.json"

SHAP_SAMPLE_SIZE = 500


def to_dense(values):
    """Convert sparse transformed features to a dense NumPy array."""
    if hasattr(values, "toarray"):
        return values.toarray()
    return values


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)

    threshold = float(metadata["threshold"])

    data = load_dataset(
        path=DATA_PATH,
        max_rows=200_000,
        random_state=42,
    )

    steps = data["step"].to_numpy()
    _, _, test_indices = chronological_split_indices(steps)
    test_data = data.iloc[test_indices].copy()

    model = joblib.load(MODEL_PATH)
    X_test = engineer_features(test_data)
    y_test = test_data["isFraud"].to_numpy()

    # Separate preprocessing from the final classifier.
    preprocessor = model[:-1]
    classifier = model[-1]

    X_transformed = to_dense(preprocessor.transform(X_test))
    feature_names = list(preprocessor.get_feature_names_out())

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    fraud_positions = [i for i, value in enumerate(predictions) if value == 1]
    if not fraud_positions:
        raise ValueError("No transactions were predicted as fraud in the test sample.")

    # Use a reproducible sample for SHAP to keep the run manageable.
    shap_positions = fraud_positions[:SHAP_SAMPLE_SIZE]
    X_shap = X_transformed[shap_positions]

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_shap)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(SHAP_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    # Explain one transaction predicted as fraud.
    explanation_position = fraud_positions[0]
    X_local = X_transformed[explanation_position]

    lime_explainer = LimeTabularExplainer(
        X_transformed,
        feature_names=feature_names,
        class_names=["Legitimate", "Fraud"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    lime_explanation = lime_explainer.explain_instance(
        X_local,
        classifier.predict_proba,
        num_features=10,
        labels=(1,),
    )

    fig = lime_explanation.as_pyplot_figure(label=1)
    fig.set_size_inches(9, 6)
    fig.tight_layout()
    fig.savefig(LIME_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    mean_abs_shap = abs(shap_values).mean(axis=0)
    ranked = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda item: item[1],
        reverse=True,
    )

    summary = {
        "model": metadata["model"],
        "imbalance_strategy": metadata["imbalance_strategy"],
        "threshold": threshold,
        "shap_sample_size": int(len(shap_positions)),
        "explained_transaction_index": int(explanation_position),
        "explained_transaction_true_label": int(y_test[explanation_position]),
        "explained_transaction_predicted_probability": float(
            probabilities[explanation_position]
        ),
        "top_shap_features": [
            {"feature": feature, "mean_abs_shap": float(value)}
            for feature, value in ranked[:10]
        ],
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved SHAP plot: {SHAP_PATH}")
    print(f"Saved LIME plot: {LIME_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()