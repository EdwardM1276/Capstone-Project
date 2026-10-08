from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from fraud_detection.config import RANDOM_STATE
from fraud_detection.preprocessing import make_preprocessor

IMBALANCE_STRATEGIES = ("class_weight", "smote", "smote_tomek")
MODEL_NAMES = (
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Feedforward Neural Network",
    "LightGBM",
)


def make_sampler(strategy: str):
    if strategy == "class_weight":
        return None
    if strategy == "smote":
        return SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    if strategy == "smote_tomek":
        return SMOTETomek(
            random_state=RANDOM_STATE,
            smote=SMOTE(k_neighbors=3, random_state=RANDOM_STATE),
        )
    raise ValueError(
        f"Unknown imbalance strategy {strategy!r}; choose from {IMBALANCE_STRATEGIES}."
    )


def make_estimator(name: str, strategy: str, positive_weight: float):
    use_class_weight = strategy == "class_weight"
    if name == "Logistic Regression":
        return LogisticRegression(
            class_weight="balanced" if use_class_weight else None,
            max_iter=500,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            class_weight="balanced_subsample" if use_class_weight else None,
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "XGBoost":
        return XGBClassifier(
            eval_metric="logloss",
            learning_rate=0.1,
            max_depth=6,
            n_estimators=200,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            scale_pos_weight=positive_weight if use_class_weight else 1,
            tree_method="hist",
        )
    if name == "Feedforward Neural Network":
        return MLPClassifier(
            early_stopping=True,
            hidden_layer_sizes=(64, 32),
            learning_rate_init=0.001,
            max_iter=150,
            random_state=RANDOM_STATE,
        )
    if name == "LightGBM":
        return LGBMClassifier(
            class_weight="balanced" if use_class_weight else None,
            learning_rate=0.1,
            n_estimators=200,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}.")


def make_model_pipeline(name: str, strategy: str, positive_weight: float) -> Pipeline:
    scale = name in ("Logistic Regression", "Feedforward Neural Network")
    steps: list[tuple[str, object]] = [("preprocessor", make_preprocessor(scale))]
    sampler = make_sampler(strategy)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("classifier", make_estimator(name, strategy, positive_weight)))
    return Pipeline(steps)


def fit_kwargs(name: str, strategy: str, y_train):
    if name == "Feedforward Neural Network" and strategy == "class_weight":
        from sklearn.utils.class_weight import compute_sample_weight

        return {"classifier__sample_weight": compute_sample_weight("balanced", y_train)}
    return {}
