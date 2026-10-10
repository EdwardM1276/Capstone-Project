import json

import pandas as pd
import pytest

from fraud_detection.config import REQUIRED_COLUMNS
from fraud_detection.data import load_dataset, validate_csv, validate_dataframe


def valid_rows():
    return pd.DataFrame(
        [
            [1, "PAYMENT", 25.0, "C1", 100.0, 75.0, "M1", 0.0, 0.0, 0, 0],
            [2, "TRANSFER", 10.0, "C2", 50.0, 40.0, "C3", 0.0, 10.0, 1, 0],
        ],
        columns=REQUIRED_COLUMNS,
    )


def test_valid_data_quality_report():
    report = validate_dataframe(valid_rows())

    assert report["passed"] is True
    assert report["row_count"] == 2
    assert report["duplicate_records"] == 0
    assert report["class_distribution"] == {"0": 1, "1": 1}


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("amount", -1.0, "negative transaction amounts"),
        ("newbalanceOrig", -1.0, "negative account balances"),
        ("isFraud", 2, "target labels outside {0, 1}"),
        ("step", None, "missing required values"),
    ],
)
def test_invalid_data_fails_validation(column, value, message):
    data = valid_rows()
    data.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_dataframe(data)


def test_duplicate_records_fail_validation():
    data = pd.concat([valid_rows(), valid_rows().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate records"):
        validate_dataframe(data)


def test_unexpected_columns_fail_schema_validation():
    data = valid_rows().assign(unexpected_feature=1)

    with pytest.raises(ValueError, match="unexpected columns"):
        validate_dataframe(data)


def test_streaming_validation_detects_duplicates_across_chunks(
    tmp_path, monkeypatch
):
    import fraud_detection.data as data_module

    monkeypatch.setattr(data_module, "CSV_CHUNK_SIZE", 2)
    data = pd.concat(
        [valid_rows(), valid_rows().iloc[[0]], valid_rows().iloc[[1]]],
        ignore_index=True,
    )
    source = tmp_path / "transactions.csv"
    report_path = tmp_path / "quality.json"
    data.to_csv(source, index=False)

    with pytest.raises(ValueError, match="duplicate records"):
        validate_csv(source, report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["duplicate_records"] == 2
    assert report["source"] == "transactions.csv"
    assert report["passed"] is False


def test_stratified_sampling_is_reproducible(tmp_path):
    data = pd.concat([valid_rows()] * 4, ignore_index=True)
    source = tmp_path / "transactions.csv"
    data.to_csv(source, index=False)

    first = load_dataset(source, max_rows=4, random_state=42)
    second = load_dataset(source, max_rows=4, random_state=42)

    assert first.equals(second)
    assert first["isFraud"].value_counts().to_dict() == {0: 2, 1: 2}
