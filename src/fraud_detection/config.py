from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
DECISION_THRESHOLD = 0.5
TARGET_COLUMN = "isFraud"
CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "origin_balance_change",
    "destination_balance_change",
    "hour",
    "log_amount",
    "origin_balance_error",
    "destination_balance_error",
    "transaction_to_origin_balance",
    "account_drained",
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
