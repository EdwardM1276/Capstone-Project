from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(
    style="whitegrid",
    rc={
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "axes.edgecolor": "#cbd7e5",
        "axes.labelcolor": "#10233f",
        "text.color": "#10233f",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "grid.color": "#e2e8f0",
    },
)


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


def save_model_comparison_figures(
    results: pd.DataFrame,
    output_dir: str | Path,
    probability_bands: pd.DataFrame | None = None,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    score_matrix = results.pivot(
        index="model", columns="imbalance_strategy", values="pr_auc"
    )
    strategy_order = [
        strategy
        for strategy in ("class_weight", "smote", "smote_tomek")
        if strategy in score_matrix.columns
    ]
    score_matrix = score_matrix[strategy_order]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        score_matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        linewidths=0.8,
        linecolor="#ffffff",
        annot_kws={"size": 11, "weight": "bold"},
        cbar_kws={"label": "PR-AUC"},
        ax=ax,
    )
    ax.set(title="Final-period PR-AUC by algorithm and treatment", xlabel="Imbalance treatment", ylabel="Algorithm")
    fig.tight_layout()
    fig.savefig(destination / "model_strategy_pr_auc_heatmap.png", dpi=160)
    plt.close(fig)

    model_order = [
        model
        for model in (
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "Feedforward Neural Network",
            "LightGBM",
        )
        if model in results["model"].unique()
    ]
    short_names = {
        "Logistic Regression": "Logistic\nRegression",
        "Random Forest": "Random\nForest",
        "XGBoost": "XGBoost",
        "Feedforward Neural Network": "Feedforward\nNeural Net",
        "LightGBM": "LightGBM",
    }
    strategy_colors = {
        "class_weight": "#155eef",
        "smote": "#0e7490",
        "smote_tomek": "#b54708",
    }
    metrics = [
        ("pr_auc", "Final PR-AUC", "Higher is better"),
        ("recall", "Final recall", "Higher is better"),
        (
            "validation_precision_at_recall_target",
            "Validation precision at recall floor",
            "Higher is better",
        ),
        ("cost_per_1000_transactions", "Final cost per 1,000", "Lower is better"),
        ("fit_seconds", "Fit duration (seconds)", "Lower is faster"),
        ("prediction_rows_per_second", "Scoring throughput (rows/s)", "Higher is faster"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    x_positions = np.arange(len(model_order))
    bar_width = 0.22
    for ax, (metric, title, note) in zip(axes.flat, metrics):
        for strategy_index, strategy in enumerate(strategy_order):
            subset = (
                results.loc[results["imbalance_strategy"] == strategy]
                .set_index("model")
                .reindex(model_order)
            )
            values = subset[metric].to_numpy(dtype=float)
            offset = (strategy_index - (len(strategy_order) - 1) / 2) * bar_width
            ax.bar(
                x_positions + offset,
                values,
                width=bar_width,
                color=strategy_colors[strategy],
                label={
                    "class_weight": "Class weighting",
                    "smote": "SMOTE",
                    "smote_tomek": "SMOTE + Tomek Links",
                }[strategy],
                zorder=3,
            )
            if metric == "pr_auc" and {"pr_auc_ci_low", "pr_auc_ci_high"}.issubset(subset.columns):
                lower = values - subset["pr_auc_ci_low"].to_numpy(dtype=float)
                upper = subset["pr_auc_ci_high"].to_numpy(dtype=float) - values
                ax.errorbar(
                    x_positions + offset,
                    values,
                    yerr=np.vstack([lower, upper]),
                    fmt="none",
                    ecolor="#10233f",
                    elinewidth=0.8,
                    capsize=2,
                    zorder=4,
                )
        ax.set_title(title, loc="left", weight="bold", fontsize=13)
        ax.text(0, 1.01, note, transform=ax.transAxes, fontsize=10, color="#43566f")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([short_names[item] for item in model_order], fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", alpha=0.65, zorder=0)
        if metric == "prediction_rows_per_second":
            ax.set_yscale("log")
            ax.set_ylabel("Rows per second, log scale")
        elif metric == "fit_seconds":
            ax.set_ylabel("Seconds")
        elif metric == "cost_per_1000_transactions":
            ax.set_ylabel("Weighted cost")
        else:
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(strategy_order), frameon=False)
    fig.suptitle("Deliverable 4 | All treatments compared within each algorithm", x=0.04, ha="left", weight="bold", fontsize=19)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(destination / "performance_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    marker_by_model = {
        "Logistic Regression": "o",
        "Random Forest": "s",
        "XGBoost": "^",
        "Feedforward Neural Network": "D",
        "LightGBM": "P",
    }
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    axes[0].set_title("Ranking quality and scoring speed", loc="left", weight="bold", fontsize=14)
    axes[1].set_title("Fraud capture and weighted error cost", loc="left", weight="bold", fontsize=14)
    costs = results["cost_per_1000_transactions"].to_numpy(dtype=float)
    recalls = results["recall"].to_numpy(dtype=float)
    cost_frontier = np.ones(len(results), dtype=bool)
    for index, (cost, recall) in enumerate(zip(costs, recalls)):
        dominates = (costs <= cost) & (recalls >= recall) & (
            (costs < cost) | (recalls > recall)
        )
        dominates[index] = False
        cost_frontier[index] = not dominates.any()

    for index, row in enumerate(results.itertuples(index=False)):
        color = strategy_colors.get(row.imbalance_strategy, "#475569")
        marker = marker_by_model.get(row.model, "o")
        edge_color = "#10233f" if row.pareto_efficient else "#ffffff"
        axes[0].scatter(
            row.prediction_rows_per_second,
            row.pr_auc,
            color=color,
            marker=marker,
            edgecolor=edge_color,
            linewidth=1.6,
            s=100,
            zorder=3,
        )
        axes[0].annotate(
            short_names.get(row.model, row.model).replace("\n", " "),
            (row.prediction_rows_per_second, row.pr_auc),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
        )
        axes[1].scatter(
            costs[index],
            row.recall,
            color=color,
            marker=marker,
            edgecolor="#10233f" if cost_frontier[index] else "#ffffff",
            linewidth=1.6,
            s=100,
            zorder=3,
        )
        axes[1].annotate(
            short_names.get(row.model, row.model).replace("\n", " "),
            (costs[index], row.recall),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
        )
    axes[0].set_xlabel("Final test transactions scored per second")
    axes[0].set_ylabel("Final test PR-AUC")
    axes[1].set_xlabel("Weighted cost per 1,000 final test transactions")
    axes[1].set_ylabel("Final test recall")
    axes[0].grid(alpha=0.65)
    axes[1].grid(alpha=0.65)

    from matplotlib.lines import Line2D

    strategy_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            color=color,
            label={
                "class_weight": "Class weighting",
                "smote": "SMOTE",
                "smote_tomek": "SMOTE + Tomek Links",
            }[strategy],
        )
        for strategy, color in strategy_colors.items()
        if strategy in strategy_order
    ]
    model_handles = [
        Line2D(
            [0], [0], marker=marker, linestyle="", color="#334155", label=short_names[model].replace("\n", " ")
        )
        for model, marker in marker_by_model.items()
        if model in model_order
    ]
    fig.legend(
        handles=strategy_handles + model_handles,
        loc="lower center",
        ncol=min(8, len(strategy_handles) + len(model_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.suptitle("Pareto analysis | Quality, efficiency, recall, and cost", x=0.06, ha="left", weight="bold", fontsize=18)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(destination / "pareto_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if probability_bands is not None and not probability_bands.empty:
        high_risk = probability_bands.loc[
            probability_bands["probability_band"] == "High"
        ]
        risk_matrix = high_risk.pivot(
            index="model", columns="imbalance_strategy", values="observed_fraud_rate"
        ).reindex(index=model_order, columns=strategy_order)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.heatmap(
            risk_matrix,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            vmin=0,
            vmax=1,
            linewidths=0.8,
            linecolor="#ffffff",
            cbar_kws={"label": "Observed fraud rate"},
            ax=ax,
        )
        ax.set(
            title="Observed fraud rate in the high-score band",
            xlabel="Imbalance treatment",
            ylabel="Algorithm",
        )
        fig.tight_layout()
        fig.savefig(destination / "probability_band_risk_map.png", dpi=160)
        plt.close(fig)


def save_calibration_figures(
    y_true,
    predictions: dict[str, dict],
    output_dir: str | Path,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    strategies = sorted(
        {label.rsplit(" [", 1)[1].rstrip("]") for label in predictions}
    )
    for strategy in strategies:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "--", color="#555555", label="Perfect calibration")
        for label, result in predictions.items():
            if not label.endswith(f"[{strategy}]"):
                continue
            observed, predicted = calibration_curve(
                y_true, result["probabilities"], n_bins=10, strategy="quantile"
            )
            ax.plot(predicted, observed, marker="o", label=label)
        ax.set(
            title=f"Probability calibration: {strategy}",
            xlabel="Mean predicted fraud probability",
            ylabel="Observed fraud rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(destination / f"calibration_{strategy}.png", dpi=160)
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
