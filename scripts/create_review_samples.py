"""Create reproducible review batches disjoint from the 200,000-row training sample."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.config import RANDOM_STATE, REQUIRED_COLUMNS, TARGET_COLUMN
from fraud_detection.data import (
    CSV_CHUNK_SIZE,
    MODEL_INPUT_COLUMNS,
    _allocate_sample_quotas,
)


def training_sample_indices(
    source: Path,
    sample_size: int = 200_000,
) -> tuple[np.ndarray, dict[int, int]]:
    """Reproduce load_dataset's stratified sample and return original CSV row IDs."""
    totals: dict[int, int] = {}
    for chunk in pd.read_csv(
        source, usecols=[TARGET_COLUMN], chunksize=CSV_CHUNK_SIZE
    ):
        for label, count in chunk[TARGET_COLUMN].value_counts().items():
            key = int(label)
            totals[key] = totals.get(key, 0) + int(count)
    if len(totals) < 2:
        raise ValueError("The PaySim source must contain both fraud classes.")
    if sample_size < len(totals):
        raise ValueError("The training sample must include both fraud classes.")

    quotas = _allocate_sample_quotas(totals, sample_size)
    remaining = totals.copy()
    remaining_quotas = quotas.copy()
    rng = np.random.default_rng(RANDOM_STATE)
    selected_indices: list[np.ndarray] = []

    for chunk in pd.read_csv(
        source,
        usecols=MODEL_INPUT_COLUMNS,
        chunksize=CSV_CHUNK_SIZE,
    ):
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
                selected_indices.append(
                    candidates.sample(n=take, random_state=seed).index.to_numpy()
                )
                remaining_quotas[label] -= take
            remaining[label] -= len(candidates)

    indices = np.concatenate(selected_indices).astype("int64", copy=False)
    if len(indices) != min(sample_size, sum(totals.values())):
        raise RuntimeError("Could not reproduce the configured training sample size.")
    return indices, totals


def _allocate_review_indices(
    source: Path,
    training_indices: np.ndarray,
    totals: dict[int, int],
    file_count: int,
    rows_per_file: int,
    seed: int,
) -> list[np.ndarray]:
    training_labels: dict[int, int] = {label: 0 for label in totals}
    training_lookup = np.sort(training_indices)
    class_indices: dict[int, list[np.ndarray]] = {label: [] for label in totals}
    for chunk in pd.read_csv(
        source,
        usecols=[TARGET_COLUMN],
        chunksize=CSV_CHUNK_SIZE,
    ):
        chunk_indices = chunk.index.to_numpy()
        positions = np.searchsorted(training_lookup, chunk_indices)
        in_training = (positions < len(training_lookup)) & (
            training_lookup[np.minimum(positions, len(training_lookup) - 1)]
            == chunk_indices
        )
        labels = pd.to_numeric(chunk[TARGET_COLUMN], errors="raise").astype(int)
        for label in totals:
            class_rows = labels.to_numpy() == label
            training_labels[label] += int((in_training & class_rows).sum())
            available = chunk_indices[class_rows & ~in_training]
            if len(available):
                class_indices[label].append(available)

    remaining_counts = {
        label: totals[label] - training_labels[label] for label in totals
    }
    requested = file_count * rows_per_file
    if sum(remaining_counts.values()) < requested:
        raise ValueError("Not enough non-training PaySim rows for the requested files.")
    quotas = _allocate_sample_quotas(remaining_counts, requested)
    rng = np.random.default_rng(seed)
    selected_classes = []
    for label, pieces in class_indices.items():
        if quotas[label] == 0:
            continue
        candidates = np.concatenate(pieces)
        selected_count = quotas[label]
        selected_classes.append(
            rng.choice(candidates, size=selected_count, replace=False)
        )
    selected = np.concatenate(selected_classes)
    rng.shuffle(selected)
    owners = np.arange(len(selected), dtype="int64") % file_count
    result = [
        np.sort(selected[owners == index]).astype("int64", copy=False)
        for index in range(file_count)
    ]
    if any(len(indices) != rows_per_file for indices in result):
        raise RuntimeError("The selected review files do not have the requested row count.")
    combined = np.concatenate(result)
    if len(np.unique(combined)) != len(combined):
        raise RuntimeError("Review samples overlap each other.")
    if np.intersect1d(combined, training_indices).size:
        raise RuntimeError("A review sample overlaps the training sample.")
    return result


def create_samples(
    source: Path,
    output_dir: Path,
    file_count: int = 12,
    rows_per_file: int = 100_000,
    seed: int = 43,
    training_sample_rows: int = 200_000,
) -> dict:
    if file_count < 1 or rows_per_file < 1:
        raise ValueError("File count and rows per file must be positive.")
    if not source.is_file():
        raise FileNotFoundError(f"PaySim source not found: {source}")
    columns = pd.read_csv(source, nrows=0).columns.tolist()
    if set(columns) != set(REQUIRED_COLUMNS):
        raise ValueError("The source columns do not match the full PaySim schema.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Sample output directory is not empty: {output_dir}. "
            "Move its contents or choose a new output directory."
        )

    training_indices, totals = training_sample_indices(source, training_sample_rows)
    sample_indices = _allocate_review_indices(
        source,
        training_indices,
        totals,
        file_count,
        rows_per_file,
        seed,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="paysim-samples-",
        dir=output_dir.parent,
    ) as temporary:
        temporary_dir = Path(temporary)
        names = [f"transactions_{index:02d}.csv" for index in range(1, file_count + 1)]
        temp_paths = [temporary_dir / name for name in names]
        row_counts = np.zeros(file_count, dtype="int64")
        for index, name in enumerate(names):
            pd.DataFrame(columns=columns).to_csv(temp_paths[index], index=False)

        selected = np.concatenate(sample_indices)
        owners = np.concatenate(
            [
                np.full(len(indices), index, dtype="int8")
                for index, indices in enumerate(sample_indices)
            ]
        )
        order = np.argsort(selected, kind="stable")
        selected = selected[order]
        owners = owners[order]

        for chunk in pd.read_csv(source, chunksize=CSV_CHUNK_SIZE):
            chunk_indices = chunk.index.to_numpy()
            start = int(chunk_indices[0]) if len(chunk_indices) else 0
            end = int(chunk_indices[-1]) + 1 if len(chunk_indices) else start
            left = int(np.searchsorted(selected, start))
            right = int(np.searchsorted(selected, end))
            if left == right:
                continue
            matched_indices = selected[left:right] - start
            matched_owners = owners[left:right]
            for dataset_index in np.unique(matched_owners):
                mask = matched_owners == dataset_index
                rows = chunk.iloc[matched_indices[mask]]
                rows.to_csv(
                    temp_paths[int(dataset_index)],
                    mode="a",
                    header=False,
                    index=False,
                )
                row_counts[int(dataset_index)] += len(rows)

        if not np.all(row_counts == rows_per_file):
            raise RuntimeError(
                f"Generated row counts do not match the request: {row_counts.tolist()}."
            )

        files = []
        for dataset_index, name in enumerate(names):
            data = pd.read_csv(temp_paths[dataset_index])
            if (data["step"].diff().dropna() < 0).any():
                raise ValueError("The source rows are not chronological by step.")
            files.append(
                {
                    "file": name,
                    "rows": int(len(data)),
                    "fraud_rows": int(data[TARGET_COLUMN].sum()),
                    "first_step": int(data["step"].min()),
                    "last_step": int(data["step"].max()),
                    "size_bytes": int(temp_paths[dataset_index].stat().st_size),
                }
            )

        manifest = {
            "source_file": source.name,
            "source_rows": int(sum(totals.values())),
            "training_sample_rows_excluded": int(len(training_indices)),
            "training_sample_seed": RANDOM_STATE,
            "review_sample_seed": seed,
            "rows_per_file": rows_per_file,
            "files": files,
        }
        for path in temp_paths:
            path.replace(output_dir / path.name)
        (output_dir / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original PaySim CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sample_datasets"),
    )
    parser.add_argument("--file-count", type=int, default=12)
    parser.add_argument("--rows-per-file", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--training-sample-rows", type=int, default=200_000)
    args = parser.parse_args()
    manifest = create_samples(
        args.source,
        args.output_dir,
        args.file_count,
        args.rows_per_file,
        args.seed,
        args.training_sample_rows,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
