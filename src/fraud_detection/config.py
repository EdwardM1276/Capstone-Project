from pathlib import Path

RANDOM_STATE = 42
TRAINING_WORKFLOW_VERSION = 2
DEFAULT_MAX_ROWS = 200_000
DECISION_THRESHOLD = 0.5
LOW_FRAUD_PROBABILITY_MAX = 0.01
HIGH_FRAUD_PROBABILITY_MIN = 0.05
TARGET_COLUMN = "isFraud"
CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "hour",
    "log_amount",
    "transaction_to_origin_balance",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    TARGET_COLUMN,
    "isFlaggedFraud",
]

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "paysim.csv"
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
