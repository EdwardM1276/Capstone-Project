import json
import heapq
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

from fraud_detection.config import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATA_PATH,
    NUMERIC_FEATURES,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)

CSV_CHUNK_SIZE = 250_000
MODEL_INPUT_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    TARGET_COLUMN,
]


def validate_columns(columns: list[str]) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    unexpected = sorted(set(columns) - set(REQUIRED_COLUMNS))
    if missing or unexpected:
        issues = []
        if missing:
            issues.append(f"missing columns: {', '.join(missing)}")
        if unexpected:
            issues.append(f"unexpected columns: {', '.join(unexpected)}")
        raise ValueError("Dataset schema does not match PaySim: " + "; ".join(issues))


def validate_dataframe(data: pd.DataFrame) -> dict:
    """Return data quality counts and fail on structural or invalid-value errors."""
    validate_columns(data.columns.tolist())
    numeric = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        TARGET_COLUMN,
        "isFlaggedFraud",
    ]
    missing = data[REQUIRED_COLUMNS].isna().sum().astype(int).to_dict()
    invalid_types = {}
    for column in numeric:
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid_types[column] = int((converted.isna() & data[column].notna()).sum())
    non_finite = {
        column: int(
            (~np.isfinite(pd.to_numeric(data[column], errors="coerce").dropna())).sum()
        )
        for column in numeric
    }
    negative_amounts = int((pd.to_numeric(data["amount"], errors="coerce") < 0).sum())
    negative_balances = {
        column: int((pd.to_numeric(data[column], errors="coerce") < 0).sum())
        for column in (
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        )
    }
    invalid_labels = int(
        (~pd.to_numeric(data[TARGET_COLUMN], errors="coerce").isin([0, 1])).sum()
    )
    invalid_flags = int(
        (~pd.to_numeric(data["isFlaggedFraud"], errors="coerce").isin([0, 1])).sum()
    )
    invalid_steps = int((pd.to_numeric(data["step"], errors="coerce") < 0).sum())
    invalid_steps += int(
        (
            pd.to_numeric(data["step"], errors="coerce").dropna()
            % 1
            != 0
        ).sum()
    )
    allowed_types = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
    invalid_transaction_types = int((~data["type"].isin(allowed_types)).sum())
    duplicate_records = int(data.duplicated(subset=REQUIRED_COLUMNS).sum())
    report = {
        "row_count": int(len(data)),
        "column_count": int(len(data.columns)),
        "columns": data.columns.tolist(),
        "missing_values": missing,
        "invalid_numeric_values": invalid_types,
        "non_finite_numeric_values": non_finite,
        "negative_transaction_amounts": negative_amounts,
        "negative_balances": negative_balances,
        "invalid_target_labels": invalid_labels,
        "invalid_flag_labels": invalid_flags,
        "negative_time_steps": invalid_steps,
        "invalid_transaction_types": invalid_transaction_types,
        "duplicate_records": duplicate_records,
        "transaction_types": sorted(data["type"].dropna().astype(str).unique().tolist()),
        "class_distribution": {
            str(label): int(count)
            for label, count in data[TARGET_COLUMN].value_counts(dropna=False).items()
        },
    }
    problems = []
    if any(missing.values()):
        problems.append("missing required values")
    if any(invalid_types.values()):
        problems.append("invalid numeric types")
    if any(non_finite.values()):
        problems.append("non-finite numeric values")
    if negative_amounts:
        problems.append("negative transaction amounts")
    if any(negative_balances.values()):
        problems.append("negative account balances")
    if invalid_labels:
        problems.append("target labels outside {0, 1}")
    if invalid_flags:
        problems.append("isFlaggedFraud labels outside {0, 1}")
    if invalid_steps:
        problems.append("invalid transaction time steps")
    if invalid_transaction_types:
        problems.append("unknown transaction types")
    if duplicate_records:
        problems.append("duplicate records")
    if problems:
        raise ValueError(
            "Data quality validation failed: "
            + ", ".join(problems)
            + ". See the generated quality report for counts."
        )
    report["passed"] = True
    return report


def validate_csv(path: str | Path, report_path: str | Path | None = None) -> dict:
    """Validate the entire CSV in chunks without loading customer identifiers."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PaySim dataset not found: {source}")

    header = pd.read_csv(source, nrows=0).columns.tolist()
    validate_columns(header)
    row_count = 0
    missing = {column: 0 for column in REQUIRED_COLUMNS}
    invalid_numeric = {
        column: 0
        for column in (
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            TARGET_COLUMN,
            "isFlaggedFraud",
        )
    }
    non_finite_numeric = {column: 0 for column in invalid_numeric}
    negative_amounts = 0
    negative_balances = {
        column: 0
        for column in (
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        )
    }
    invalid_labels = 0
    negative_steps = 0
    invalid_flags = 0
    invalid_transaction_types = 0
    class_counts = {0: 0, 1: 0}
    transaction_types: set[str] = set()
    duplicate_count = 0
    with tempfile.TemporaryDirectory(prefix="paysim-validation-") as temporary_dir:
        hash_paths = []
        for chunk in pd.read_csv(source, chunksize=CSV_CHUNK_SIZE):
            row_count += len(chunk)
            for column in REQUIRED_COLUMNS:
                missing[column] += int(chunk[column].isna().sum())
            for column in invalid_numeric:
                converted = pd.to_numeric(chunk[column], errors="coerce")
                invalid_numeric[column] += int(
                    (converted.isna() & chunk[column].notna()).sum()
                )
                non_finite_numeric[column] += int(
                    (~np.isfinite(converted.dropna())).sum()
                )
            amount = pd.to_numeric(chunk["amount"], errors="coerce")
            negative_amounts += int((amount < 0).sum())
            for column in negative_balances:
                negative_balances[column] += int(
                    (pd.to_numeric(chunk[column], errors="coerce") < 0).sum()
                )
            labels = pd.to_numeric(chunk[TARGET_COLUMN], errors="coerce")
            invalid_labels += int((~labels.isin([0, 1])).sum())
            steps = pd.to_numeric(chunk["step"], errors="coerce")
            negative_steps += int((steps < 0).sum()) + int(
                ((steps.dropna() % 1) != 0).sum()
            )
            flags = pd.to_numeric(chunk["isFlaggedFraud"], errors="coerce")
            invalid_flags += int((~flags.isin([0, 1])).sum())
            invalid_transaction_types += int(
                (~chunk["type"].isin({"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"})).sum()
            )
            counts = labels.value_counts()
            class_counts[0] += int(counts.get(0, 0))
            class_counts[1] += int(counts.get(1, 0))
            transaction_types.update(chunk["type"].dropna().astype(str).unique())
            hashes = hash_pandas_object(chunk[REQUIRED_COLUMNS], index=False).to_numpy(
                dtype=np.uint64
            )
            hash_path = Path(temporary_dir) / f"hashes-{len(hash_paths):05d}.npy"
            np.save(hash_path, np.sort(hashes), allow_pickle=False)
            hash_paths.append(hash_path)
        sorted_chunks = (
            np.load(path, mmap_mode="r", allow_pickle=False) for path in hash_paths
        )
        previous = None
        for value in heapq.merge(*(iter(chunk) for chunk in sorted_chunks)):
            if previous is not None and value == previous:
                duplicate_count += 1
            previous = value

    source_stat = source.stat()
    report = {
        "source": str(source),
        "source_size_bytes": source_stat.st_size,
        "source_modified_ns": source_stat.st_mtime_ns,
        "row_count": row_count,
        "column_count": len(header),
        "columns": header,
        "missing_values": missing,
        "invalid_numeric_values": invalid_numeric,
        "non_finite_numeric_values": non_finite_numeric,
        "negative_transaction_amounts": negative_amounts,
        "negative_balances": negative_balances,
        "invalid_target_labels": invalid_labels,
        "invalid_flag_labels": invalid_flags,
        "negative_time_steps": negative_steps,
        "invalid_transaction_types": invalid_transaction_types,
        "duplicate_records": duplicate_count,
        "transaction_types": sorted(transaction_types),
        "class_distribution": {str(k): v for k, v in class_counts.items()},
    }
    problems = []
    if any(missing.values()):
        problems.append("missing required values")
    if any(invalid_numeric.values()):
        problems.append("invalid numeric types")
    if any(non_finite_numeric.values()):
        problems.append("non-finite numeric values")
    if negative_amounts:
        problems.append("negative transaction amounts")
    if any(negative_balances.values()):
        problems.append("negative account balances")
    if invalid_labels:
        problems.append("target labels outside {0, 1}")
    if invalid_flags:
        problems.append("isFlaggedFraud labels outside {0, 1}")
    if negative_steps:
        problems.append("invalid transaction time steps")
    if invalid_transaction_types:
        problems.append("unknown transaction types")
    if duplicate_count:
        problems.append("duplicate records")
    report["passed"] = not problems
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if problems:
        raise ValueError(
            "Data quality validation failed: "
            + ", ".join(problems)
            + ". See the generated quality report for counts."
        )
    return report


def load_dataset(
    path: str | Path = DEFAULT_DATA_PATH,
    max_rows: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load model inputs, optionally taking a reproducible stratified sample."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PaySim dataset not found: {source}")
    columns = pd.read_csv(source, nrows=0).columns.tolist()
    validate_columns(columns)

    if max_rows is None or max_rows <= 0:
        data = pd.read_csv(source, usecols=MODEL_INPUT_COLUMNS)
        data["type"] = data["type"].astype("category")
        return data

    totals = {}
    for chunk in pd.read_csv(source, usecols=[TARGET_COLUMN], chunksize=CSV_CHUNK_SIZE):
        for label, count in chunk[TARGET_COLUMN].value_counts().items():
            key = int(label)
            totals[key] = totals.get(key, 0) + int(count)
    if len(totals) < 2:
        raise ValueError("The dataset must contain both fraud and legitimate rows.")
    if max_rows < len(totals):
        raise ValueError("The requested sample must include at least one row per class.")
    quotas = _allocate_sample_quotas(totals, max_rows)
    remaining = totals.copy()
    remaining_quotas = quotas.copy()
    sampled_chunks = []
    rng = np.random.default_rng(random_state)

    for chunk in pd.read_csv(source, usecols=MODEL_INPUT_COLUMNS, chunksize=CSV_CHUNK_SIZE):
        chosen = []
        labels = pd.to_numeric(chunk[TARGET_COLUMN], errors="raise").astype(int)
        for label in totals:
            candidates = chunk.loc[labels == label]
            count_left = remaining[label]
            quota_left = remaining_quotas[label]
            if count_left == 0 or quota_left == 0:
                take = 0
            elif count_left == len(candidates):
                take = quota_left
            else:
                take = min(
                    len(candidates),
                    int(round(quota_left * len(candidates) / count_left)),
                )
            if take:
                seed = int(rng.integers(0, np.iinfo(np.int32).max))
                chosen.append(candidates.sample(n=take, random_state=seed))
                remaining_quotas[label] -= take
            remaining[label] -= len(candidates)
        if chosen:
            sampled_chunks.append(pd.concat(chosen, ignore_index=True))

    data = pd.concat(sampled_chunks, ignore_index=True)
    if len(data) != min(max_rows, sum(totals.values())):
        raise RuntimeError(
            f"Stratified sample returned {len(data)} rows; expected "
            f"{min(max_rows, sum(totals.values()))}."
        )
    data["type"] = data["type"].astype("category")
    return data


def _allocate_sample_quotas(class_counts: dict[int, int], sample_size: int) -> dict[int, int]:
    total = sum(class_counts.values())
    exact = {label: sample_size * count / total for label, count in class_counts.items()}
    quotas = {label: min(class_counts[label], int(value)) for label, value in exact.items()}
    left = sample_size - sum(quotas.values())
    for label in sorted(exact, key=lambda key: exact[key] - int(exact[key]), reverse=True):
        if left == 0:
            break
        if quotas[label] < class_counts[label]:
            quotas[label] += 1
            left -= 1
    return quotas
