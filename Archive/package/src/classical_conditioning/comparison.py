"""Streaming comparisons between versioned tabular analysis artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.artifacts import sha256_file as _sha256_file


@dataclass(frozen=True)
class ComparisonResult:
    left: Path
    right: Path
    output: Path
    row_counts_equal: bool
    common_column_count: int


@dataclass(frozen=True)
class RouteComparisonResult:
    recording_id: str
    output: Path
    legacy_final_rows: int
    candidate_frame_rows: int


def _null_mask(array: pa.Array) -> np.ndarray:
    return array.is_null().to_numpy(zero_copy_only=False)


def compare_parquet_artifacts(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    batch_size: int = 250_000,
) -> ComparisonResult:
    """Compare row-aligned Parquet artifacts without loading them fully."""
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("Comparison tolerances must be non-negative.")
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")

    left_path = left_path.resolve()
    right_path = right_path.resolve()
    output_path = output_path.resolve()
    if output_path in {left_path, right_path}:
        raise ValueError("Comparison output must differ from both input artifacts.")
    left_hash = _sha256_file(left_path)
    right_hash = _sha256_file(right_path)
    left = pq.ParquetFile(left_path)
    right = pq.ParquetFile(right_path)
    try:
        left_columns = left.schema_arrow.names
        right_columns = right.schema_arrow.names
        common_columns = [
            column for column in left_columns if column in set(right_columns)
        ]
        row_counts_equal = left.metadata.num_rows == right.metadata.num_rows
        column_results: dict[str, dict[str, Any]] = {}

        for column in common_columns:
            left_field = left.schema_arrow.field(column)
            right_field = right.schema_arrow.field(column)
            result: dict[str, Any] = {
                "left_type": str(left_field.type),
                "right_type": str(right_field.type),
                "types_equal": left_field.type == right_field.type,
                "compared_rows": 0,
                "null_mismatch_count": 0,
                "value_mismatch_count": 0,
            }
            integer = pa.types.is_integer(left_field.type) and pa.types.is_integer(
                right_field.type
            )
            floating = pa.types.is_floating(
                left_field.type
            ) and pa.types.is_floating(right_field.type)
            numeric = integer or floating
            if floating:
                result.update(
                    {
                        "maximum_absolute_difference": 0.0,
                        "sum_absolute_difference": 0.0,
                        "finite_pair_count": 0,
                    }
                )

            left_column = left.read(columns=[column]).column(0).combine_chunks()
            right_column = right.read(columns=[column]).column(0).combine_chunks()
            if isinstance(left_column, pa.DictionaryArray) or isinstance(
                right_column, pa.DictionaryArray
            ):
                left_dictionary = (
                    left_column.dictionary.to_pylist()
                    if isinstance(left_column, pa.DictionaryArray)
                    else None
                )
                right_dictionary = (
                    right_column.dictionary.to_pylist()
                    if isinstance(right_column, pa.DictionaryArray)
                    else None
                )
                result["left_categories"] = left_dictionary
                result["right_categories"] = right_dictionary
                result["categories_equal"] = left_dictionary == right_dictionary
                result["category_ordering_equal"] = (
                    getattr(left_field.type, "ordered", None)
                    == getattr(right_field.type, "ordered", None)
                )
            compared_length = min(len(left_column), len(right_column))
            for start in range(0, compared_length, batch_size):
                length = min(batch_size, compared_length - start)
                left_array = left_column.slice(start, length)
                right_array = right_column.slice(start, length)
                result["compared_rows"] += len(left_array)
                left_null = _null_mask(left_array)
                right_null = _null_mask(right_array)
                null_mismatch = left_null != right_null
                result["null_mismatch_count"] += int(np.count_nonzero(null_mismatch))
                valid = ~(left_null | right_null)

                if integer:
                    left_values = left_array.to_pylist()
                    right_values = right_array.to_pylist()
                    unequal = valid & np.fromiter(
                        (
                            left_value != right_value
                            for left_value, right_value in zip(
                                left_values,
                                right_values,
                            )
                        ),
                        dtype=bool,
                        count=len(left_values),
                    )
                    result["value_mismatch_count"] += int(np.count_nonzero(unequal))
                    if np.any(unequal):
                        maximum = max(
                            abs(int(left_value) - int(right_value))
                            for left_value, right_value, is_unequal in zip(
                                left_values, right_values, unequal
                            )
                            if is_unequal
                        )
                        result["maximum_absolute_integer_difference"] = max(
                            int(result.get("maximum_absolute_integer_difference", 0)),
                            maximum,
                        )
                elif floating:
                    left_values = left_array.to_numpy(
                        zero_copy_only=False
                    ).astype(np.float64, copy=False)
                    right_values = right_array.to_numpy(
                        zero_copy_only=False
                    ).astype(np.float64, copy=False)
                    finite = valid & np.isfinite(left_values) & np.isfinite(right_values)
                    if np.any(finite):
                        differences = np.abs(left_values[finite] - right_values[finite])
                        result["maximum_absolute_difference"] = max(
                            result["maximum_absolute_difference"],
                            float(np.max(differences)),
                        )
                        result["sum_absolute_difference"] += float(np.sum(differences))
                        result["finite_pair_count"] += int(np.count_nonzero(finite))
                        close = np.isclose(
                            left_values[finite],
                            right_values[finite],
                            atol=absolute_tolerance,
                            rtol=relative_tolerance,
                            equal_nan=True,
                        )
                        result["value_mismatch_count"] += int(
                            np.count_nonzero(~close)
                        )
                    nonfinite_equal = (
                        (np.isnan(left_values) & np.isnan(right_values))
                        | (np.isposinf(left_values) & np.isposinf(right_values))
                        | (np.isneginf(left_values) & np.isneginf(right_values))
                    )
                    result["value_mismatch_count"] += int(
                        np.count_nonzero(valid & ~finite & ~nonfinite_equal)
                    )
                else:
                    left_values = np.asarray(left_array.to_pylist(), dtype=object)
                    right_values = np.asarray(right_array.to_pylist(), dtype=object)
                    result["value_mismatch_count"] += int(
                        np.count_nonzero(valid & (left_values != right_values))
                    )

            if floating:
                finite_pairs = result["finite_pair_count"]
                result["mean_absolute_difference"] = (
                    result["sum_absolute_difference"] / finite_pairs
                    if finite_pairs
                    else None
                )
                del result["sum_absolute_difference"]
            column_results[column] = result

        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "left": {
                "path": str(left_path),
                "sha256": left_hash,
                "rows": left.metadata.num_rows,
                "columns": left_columns,
            },
            "right": {
                "path": str(right_path),
                "sha256": right_hash,
                "rows": right.metadata.num_rows,
                "columns": right_columns,
            },
            "row_counts_equal": row_counts_equal,
            "columns_only_left": [
                column for column in left_columns if column not in set(right_columns)
            ],
            "columns_only_right": [
                column for column in right_columns if column not in set(left_columns)
            ],
            "tolerances": {
                "absolute": absolute_tolerance,
                "relative": relative_tolerance,
            },
            "column_results": column_results,
        }
    finally:
        left.close()
        right.close()

    if _sha256_file(left_path) != left_hash or _sha256_file(right_path) != right_hash:
        raise RuntimeError("A compared artifact changed during comparison.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    return ComparisonResult(
        left=left_path,
        right=right_path,
        output=output_path,
        row_counts_equal=row_counts_equal,
        common_column_count=len(common_columns),
    )


def write_preprocessing_route_comparison(
    project_dir: Path,
    recording_id: str,
    *,
    overwrite: bool = False,
) -> RouteComparisonResult:
    """Combine legacy and candidate summaries into an explicit route report."""
    project_dir = project_dir.resolve()
    quality_dir = project_dir / "Quality checks" / recording_id
    legacy_path = quality_dir / "legacy-v1_preprocessing_summary.json"
    candidate_path = quality_dir / "candidate-v1_activity_summary.json"
    intake_path = quality_dir / "acquisition_summary.json"
    for path in (legacy_path, candidate_path, intake_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing comparison input: {path}")

    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    intake = json.loads(intake_path.read_text(encoding="utf-8"))
    for label, content in (
        ("legacy", legacy),
        ("candidate", candidate),
        ("intake", intake),
    ):
        if content.get("recording_id") != recording_id:
            raise ValueError(f"{label} summary recording identity mismatch.")

    output = quality_dir / "legacy-vs-candidate_preprocessing.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"Route comparison already exists: {output}")

    payload = {
        "recording_id": recording_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "candidate_comparison",
        "inputs": {
            "intake_summary": {
                "path": str(intake_path),
                "sha256": _sha256_file(intake_path),
            },
            "legacy_summary": {
                "path": str(legacy_path),
                "sha256": _sha256_file(legacy_path),
            },
            "candidate_summary": {
                "path": str(candidate_path),
                "sha256": _sha256_file(candidate_path),
            },
        },
        "coverage": {
            "source_camera_rows": intake["camera"]["statistics"]["frames"]["row_count"],
            "source_tracking_rows": intake["tracking"]["statistics"]["frames"]["row_count"],
            "legacy_stage_rows": legacy["stage_row_counts"],
            "legacy_final_trial_samples": legacy["artifact"]["rows"],
            "candidate_overlapping_frame_rows": candidate["row_count"],
            "candidate_valid_derivatives": candidate["valid_derivative_count"],
            "candidate_gap_invalid_derivatives": candidate["invalid_gap_count"],
        },
        "tracking_semantics": candidate["tracking_semantics"],
        "differences": [
            {
                "area": "tracking fields",
                "previous": "Uses angle0..angle15 only and drops measured x/y.",
                "candidate": "Uses measured x/y for physical point and segment motion.",
                "classification": "candidate improvement",
            },
            {
                "area": "tracking rows",
                "previous": "Unconditionally removes the final tracking row.",
                "candidate": "Retains every valid row and intersects explicitly with camera frames.",
                "classification": "engineering correction candidate",
            },
            {
                "area": "timing",
                "previous": "Interpolates/extrapolates to a nominal 700 FPS grid.",
                "candidate": "Uses measured camera elapsed time and invalidates cross-gap derivatives.",
                "classification": "scientific decision required",
            },
            {
                "area": "activity",
                "previous": "Absolute derivative of one distal cumulative-angle signal.",
                "candidate": (
                    "Stores segment-speed sum, angular RMS, whole-tail XY RMS, "
                    "whole-tail XY mean, and curvature-change RMS side by side."
                ),
                "classification": "scientific validation required",
            },
            {
                "area": "filtering",
                "previous": "Centered 10-frame temporal mean; spatial parameter has no effect.",
                "candidate": "No smoothing in candidate-v1 so raw metric behavior can be validated first.",
                "classification": "smoothing decision pending",
            },
            {
                "area": "movement state",
                "previous": "Primary-threshold bout detector; configured secondary threshold unused.",
                "candidate": "No movement threshold or bouts yet.",
                "classification": "detector calibration pending",
            },
        ],
        "legacy_known_limitations": legacy["known_limitations"],
        "candidate_known_limitations": candidate["known_limitations"],
        "next_decisions": [
            "Validate smoothing choices against synthetic signals and reviewed traces.",
            "Calibrate metric-specific movement-state and bout detectors.",
            "Align candidate metrics to protocol trials without interpolation.",
            "Compare total activity, movement probability, and conditional intensity.",
            "Do not select a paper metric from this single fish.",
        ],
    }
    temporary = output.with_suffix(output.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    return RouteComparisonResult(
        recording_id=recording_id,
        output=output,
        legacy_final_rows=int(legacy["artifact"]["rows"]),
        candidate_frame_rows=int(candidate["row_count"]),
    )
