import argparse
import json

from fraud_detection.config import DEFAULT_DATA_PATH, FIGURE_DIR, REPORT_DIR
from fraud_detection.data import validate_csv
from fraud_detection.training import (
    DEFAULT_MODELS,
    DEFAULT_STRATEGIES,
    run_training,
)
from fraud_detection.visualization import save_class_distribution


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess, train, and evaluate PaySim fraud detection models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Validate PaySim data.")
    preprocess.add_argument("--input", default=str(DEFAULT_DATA_PATH))
    preprocess.add_argument("--report", default=str(REPORT_DIR / "data_quality.json"))

    train = subparsers.add_parser("train", help="Train and compare fraud classifiers.")
    train.add_argument("--input", default=str(DEFAULT_DATA_PATH))
    train.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Use a stratified sample for modelling; 0 trains on the entire dataset.",
    )
    train.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    train.add_argument(
        "--imbalance-strategies",
        nargs="+",
        choices=DEFAULT_STRATEGIES,
        default=["class_weight"],
    )
    train.add_argument("--tune-rows", type=int, default=100_000)
    train.add_argument("--search-iterations", type=int, default=5)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "preprocess":
        report = validate_csv(args.input, args.report)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_class_distribution(
            report["class_distribution"], FIGURE_DIR / "class_distribution.png"
        )
        print(json.dumps(report, indent=2))
        print(f"Data quality report: {args.report}")
        return

    results = run_training(
        input_path=args.input,
        max_rows=args.max_rows if args.max_rows > 0 else None,
        model_names=args.models,
        strategies=args.imbalance_strategies,
        tune_rows=args.tune_rows,
        search_iterations=args.search_iterations,
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
