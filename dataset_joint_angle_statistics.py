#!/usr/bin/env python3
"""Compute per-joint position and motion statistics from LeRobot v2 Parquet data.

This script reads only metadata and Parquet columns; it never opens video data.
It intentionally contains no robot safety limits or deployment-specific values.
Frame-to-frame deltas are calculated within each episode file, never across an
episode boundary. When both default columns are selected, the output also
contains same-frame and one-frame-lag commanded-versus-measured joint gaps.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_COLUMNS = ("action", "observation.state")
SAME_FRAME_GAP = "commanded_action_minus_measured_state_same_frame"
NEXT_FRAME_GAP = "commanded_action_t_minus_measured_state_t_plus_1"
FIELDNAMES = (
    "source",
    "joint_index",
    "joint_name",
    "sample_count",
    "finite_count",
    "nonfinite_count",
    "min",
    "p01",
    "p05",
    "median",
    "p95",
    "p99",
    "max",
    "mean",
    "std",
    "abs_mean",
    "abs_p95",
    "abs_p99",
    "abs_max",
    "delta_sample_count",
    "delta_finite_count",
    "delta_nonfinite_count",
    "abs_delta_mean",
    "abs_delta_p95",
    "abs_delta_p99",
    "abs_delta_max",
    "sample_rate_hz",
    "abs_velocity_mean_rad_s",
    "abs_velocity_p95_rad_s",
    "abs_velocity_p99_rad_s",
    "abs_velocity_max_rad_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate per-joint statistics from one or more compatible "
            "LeRobot v2 dataset split roots."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        type=Path,
        help="Dataset split root(s), each containing meta/info.json and data/.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        choices=DEFAULT_COLUMNS,
        default=list(DEFAULT_COLUMNS),
        help=(
            "Parquet vector columns to analyse. Selecting both default columns also "
            "adds paired same-frame and one-frame-lag command/state gaps."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional per-dataset Parquet-file limit for a quick smoke test.",
    )
    return parser.parse_args()


def load_info(dataset: Path) -> dict[str, Any]:
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    with info_path.open(encoding="utf-8") as handle:
        info = json.load(handle)
    if not isinstance(info, dict):
        raise ValueError(f"Expected a JSON object in {info_path}")
    return info


def feature_contract(
    dataset: Path,
    info: dict[str, Any],
    column: str,
) -> tuple[tuple[str, ...], int]:
    try:
        feature = info["features"][column]
        shape = feature["shape"]
        names = feature["names"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{dataset}: metadata does not define names and shape for {column!r}"
        ) from exc

    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or not isinstance(shape[0], int)
        or shape[0] <= 0
    ):
        raise ValueError(f"{dataset}: expected one-dimensional shape for {column!r}")

    if (
        isinstance(names, list)
        and len(names) == 1
        and isinstance(names[0], list)
    ):
        names = names[0]
    if not isinstance(names, list) or len(names) != shape[0]:
        raise ValueError(
            f"{dataset}: {column!r} has shape {shape}, but its names do not match"
        )

    joint_names = tuple(str(name) for name in names)
    if len(set(joint_names)) != len(joint_names):
        raise ValueError(f"{dataset}: duplicate joint names in {column!r}")
    return joint_names, shape[0]


def dataset_sample_rate(dataset: Path, info: dict[str, Any]) -> float:
    try:
        sample_rate_hz = float(info["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{dataset}: metadata does not define a numeric fps") from exc
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError(f"{dataset}: fps must be finite and positive")
    return sample_rate_hz


def parquet_files(dataset: Path, max_files: int | None) -> list[Path]:
    files = sorted((dataset / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {dataset / 'data'}")
    if max_files is not None:
        if max_files <= 0:
            raise ValueError("--max-files must be positive")
        files = files[:max_files]
    return files


def vector_array(
    table: pa.Table,
    column: str,
    width: int,
    path: Path,
) -> np.ndarray:
    chunked = table[column]
    if chunked.null_count:
        raise ValueError(f"{path}: {column!r} contains null vectors")
    array = chunked.combine_chunks()

    if pa.types.is_fixed_size_list(array.type):
        if array.type.list_size != width:
            raise ValueError(
                f"{path}: {column!r} width is {array.type.list_size}, expected {width}"
            )
        flat = array.values.to_numpy(zero_copy_only=False)
        values = np.asarray(flat).reshape(len(array), width)
    else:
        values = np.asarray(array.to_pylist())
        if values.shape != (len(array), width):
            raise ValueError(
                f"{path}: {column!r} shape is {values.shape}, "
                f"expected ({len(array)}, {width})"
            )

    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{path}: {column!r} is not numeric")
    return values.astype(np.float64, copy=False)


def collect_values(
    datasets: list[Path],
    columns: list[str],
    max_files: int | None,
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    float,
    int,
    int,
]:
    expected_names: dict[str, tuple[str, ...]] = {}
    blocks: dict[str, list[np.ndarray]] = {column: [] for column in columns}
    delta_blocks: dict[str, list[np.ndarray]] = {column: [] for column in columns}
    paired_sources_enabled = all(column in columns for column in DEFAULT_COLUMNS)
    if paired_sources_enabled:
        blocks[SAME_FRAME_GAP] = []
        blocks[NEXT_FRAME_GAP] = []
        delta_blocks[SAME_FRAME_GAP] = []
        delta_blocks[NEXT_FRAME_GAP] = []
    expected_sample_rate_hz: float | None = None
    total_files = 0
    total_rows = 0

    for raw_dataset in datasets:
        dataset = raw_dataset.expanduser().resolve()
        info = load_info(dataset)
        sample_rate_hz = dataset_sample_rate(dataset, info)
        if expected_sample_rate_hz is None:
            expected_sample_rate_hz = sample_rate_hz
        elif not np.isclose(sample_rate_hz, expected_sample_rate_hz):
            raise ValueError(
                f"{dataset}: fps {sample_rate_hz} differs from the preceding "
                f"dataset roots ({expected_sample_rate_hz})"
            )
        widths: dict[str, int] = {}
        for column in columns:
            names, width = feature_contract(dataset, info, column)
            if column in expected_names and expected_names[column] != names:
                raise ValueError(
                    f"{dataset}: joint ordering for {column!r} differs from "
                    "the preceding dataset roots"
                )
            expected_names[column] = names
            widths[column] = width

        if paired_sources_enabled:
            action_names = expected_names["action"]
            state_names = expected_names["observation.state"]
            if action_names != state_names:
                raise ValueError(
                    f"{dataset}: action and observation.state joint ordering differs"
                )
            expected_names[SAME_FRAME_GAP] = action_names
            expected_names[NEXT_FRAME_GAP] = action_names

        for path in parquet_files(dataset, max_files):
            table = pq.read_table(path, columns=columns)
            row_count = table.num_rows
            file_values: dict[str, np.ndarray] = {}
            for column in columns:
                values = vector_array(table, column, widths[column], path)
                if values.shape[0] != row_count:
                    raise ValueError(f"{path}: inconsistent row count for {column!r}")
                blocks[column].append(values)
                file_values[column] = values
                if row_count > 1:
                    delta_blocks[column].append(np.diff(values, axis=0))

            if paired_sources_enabled:
                same_frame_gap = (
                    file_values["action"] - file_values["observation.state"]
                )
                blocks[SAME_FRAME_GAP].append(same_frame_gap)
                if row_count > 1:
                    delta_blocks[SAME_FRAME_GAP].append(
                        np.diff(same_frame_gap, axis=0)
                    )

                    next_frame_gap = (
                        file_values["action"][:-1]
                        - file_values["observation.state"][1:]
                    )
                    blocks[NEXT_FRAME_GAP].append(next_frame_gap)
                    if row_count > 2:
                        delta_blocks[NEXT_FRAME_GAP].append(
                            np.diff(next_frame_gap, axis=0)
                        )
            total_files += 1
            total_rows += row_count

    combined = {
        column: np.concatenate(column_blocks, axis=0)
        for column, column_blocks in blocks.items()
    }
    combined_deltas: dict[str, np.ndarray] = {}
    for column, column_delta_blocks in delta_blocks.items():
        if not column_delta_blocks:
            raise ValueError(
                f"No within-episode frame transitions found for {column!r}"
            )
        combined_deltas[column] = np.concatenate(column_delta_blocks, axis=0)

    if expected_sample_rate_hz is None:
        raise ValueError("No datasets were provided")
    return (
        expected_names,
        combined,
        combined_deltas,
        expected_sample_rate_hz,
        total_files,
        total_rows,
    )


def statistics_rows(
    names_by_column: dict[str, tuple[str, ...]],
    values_by_column: dict[str, np.ndarray],
    deltas_by_column: dict[str, np.ndarray],
    sample_rate_hz: float,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    quantiles = (0.01, 0.05, 0.50, 0.95, 0.99)

    for source, values in values_by_column.items():
        names = names_by_column[source]
        deltas = deltas_by_column[source]
        for joint_index, joint_name in enumerate(names):
            samples = values[:, joint_index]
            finite = samples[np.isfinite(samples)]
            if finite.size == 0:
                raise ValueError(
                    f"{source} joint {joint_index} ({joint_name}) has no finite samples"
                )
            p01, p05, median, p95, p99 = np.quantile(
                finite,
                quantiles,
                method="linear",
            )
            finite_abs = np.abs(finite)
            abs_p95, abs_p99 = np.quantile(
                finite_abs,
                (0.95, 0.99),
                method="linear",
            )
            delta_samples = deltas[:, joint_index]
            finite_abs_deltas = np.abs(delta_samples[np.isfinite(delta_samples)])
            if finite_abs_deltas.size == 0:
                raise ValueError(
                    f"{source} joint {joint_index} ({joint_name}) has no finite deltas"
                )
            delta_p95, delta_p99 = np.quantile(
                finite_abs_deltas,
                (0.95, 0.99),
                method="linear",
            )
            delta_mean = float(np.mean(finite_abs_deltas))
            delta_max = float(np.max(finite_abs_deltas))
            rows.append(
                {
                    "source": source,
                    "joint_index": joint_index,
                    "joint_name": joint_name,
                    "sample_count": int(samples.size),
                    "finite_count": int(finite.size),
                    "nonfinite_count": int(samples.size - finite.size),
                    "min": float(np.min(finite)),
                    "p01": float(p01),
                    "p05": float(p05),
                    "median": float(median),
                    "p95": float(p95),
                    "p99": float(p99),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                    "abs_mean": float(np.mean(finite_abs)),
                    "abs_p95": float(abs_p95),
                    "abs_p99": float(abs_p99),
                    "abs_max": float(np.max(finite_abs)),
                    "delta_sample_count": int(delta_samples.size),
                    "delta_finite_count": int(finite_abs_deltas.size),
                    "delta_nonfinite_count": int(
                        delta_samples.size - finite_abs_deltas.size
                    ),
                    "abs_delta_mean": delta_mean,
                    "abs_delta_p95": float(delta_p95),
                    "abs_delta_p99": float(delta_p99),
                    "abs_delta_max": delta_max,
                    "sample_rate_hz": sample_rate_hz,
                    "abs_velocity_mean_rad_s": delta_mean * sample_rate_hz,
                    "abs_velocity_p95_rad_s": float(delta_p95) * sample_rate_hz,
                    "abs_velocity_p99_rad_s": float(delta_p99) * sample_rate_hz,
                    "abs_velocity_max_rad_s": delta_max * sample_rate_hz,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    names, values, deltas, sample_rate_hz, file_count, row_count = collect_values(
        args.datasets,
        list(dict.fromkeys(args.columns)),
        args.max_files,
    )
    rows = statistics_rows(names, values, deltas, sample_rate_hz)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    nonfinite = sum(int(row["nonfinite_count"]) for row in rows)
    nonfinite_deltas = sum(int(row["delta_nonfinite_count"]) for row in rows)
    print(f"Scanned {file_count} Parquet files and {row_count} frames.")
    print(
        f"Wrote {len(rows)} per-joint position, paired-gap, and within-episode "
        f"delta rows "
        f"to {output}"
    )
    print(f"Non-finite scalar values: {nonfinite}")
    print(f"Non-finite delta values: {nonfinite_deltas}")


if __name__ == "__main__":
    main()
