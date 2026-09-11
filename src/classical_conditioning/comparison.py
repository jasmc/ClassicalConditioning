"""Streaming comparisons between supported tabular analysis artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


def compare_parquet_artifacts(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    batch_size: int = 250_000,
) -> ComparisonResult:
    """Compare row-aligned Parquet artifacts without importing archived routes."""
    if absolute_tolerance < 0 or relative_tolerance < 0 or batch_size < 1:
        raise ValueError("Comparison tolerances must be non-negative and batch_size positive.")
    left_path, right_path, output_path = (
        left_path.resolve(), right_path.resolve(), output_path.resolve()
    )
    if output_path in {left_path, right_path}:
        raise ValueError("Comparison output must differ from both input artifacts.")
    left_hash, right_hash = _sha256_file(left_path), _sha256_file(right_path)
    left, right = pq.ParquetFile(left_path), pq.ParquetFile(right_path)
    try:
        left_columns, right_columns = left.schema_arrow.names, right.schema_arrow.names
        common = [column for column in left_columns if column in set(right_columns)]
        column_results = {}
        for column in common:
            left_field, right_field = left.schema_arrow.field(column), right.schema_arrow.field(column)
            left_column = left.read(columns=[column]).column(0).combine_chunks()
            right_column = right.read(columns=[column]).column(0).combine_chunks()
            left_values = left_column.to_pylist()
            right_values = right_column.to_pylist()
            compared = min(len(left_values), len(right_values))
            left_array, right_array = np.asarray(left_values[:compared], dtype=object), np.asarray(right_values[:compared], dtype=object)
            left_null = np.fromiter((value is None for value in left_array), dtype=bool, count=compared)
            right_null = np.fromiter((value is None for value in right_array), dtype=bool, count=compared)
            valid = ~(left_null | right_null)
            integer = pa.types.is_integer(left_field.type) and pa.types.is_integer(right_field.type)
            floating = pa.types.is_floating(left_field.type) and pa.types.is_floating(right_field.type)
            mismatches = 0
            result = {
                "left_type": str(left_field.type), "right_type": str(right_field.type),
                "types_equal": left_field.type == right_field.type,
                "compared_rows": compared,
                "null_mismatch_count": int(np.count_nonzero(left_null != right_null)),
                "value_mismatch_count": 0,
            }
            if integer:
                unequal = [index for index in range(compared) if valid[index] and left_array[index] != right_array[index]]
                mismatches = len(unequal)
                if unequal:
                    result["maximum_absolute_integer_difference"] = max(abs(int(left_array[index]) - int(right_array[index])) for index in unequal)
            elif floating:
                left_float = np.asarray([float(value) if value is not None else np.nan for value in left_array])
                right_float = np.asarray([float(value) if value is not None else np.nan for value in right_array])
                finite = valid & np.isfinite(left_float) & np.isfinite(right_float)
                differences = np.abs(left_float[finite] - right_float[finite])
                result["maximum_absolute_difference"] = float(np.max(differences)) if differences.size else 0.0
                result["mean_absolute_difference"] = float(np.mean(differences)) if differences.size else None
                mismatches = int(np.count_nonzero(~np.isclose(left_float[finite], right_float[finite], atol=absolute_tolerance, rtol=relative_tolerance)))
                mismatches += int(np.count_nonzero(valid & ~finite & ~((np.isnan(left_float) & np.isnan(right_float)) | (np.isposinf(left_float) & np.isposinf(right_float)) | (np.isneginf(left_float) & np.isneginf(right_float)))))
            else:
                mismatches = sum(left_array[index] != right_array[index] for index in range(compared) if valid[index])
            result["value_mismatch_count"] = int(mismatches)
            if pa.types.is_dictionary(left_field.type) or pa.types.is_dictionary(right_field.type):
                left_categories = (
                    left_column.dictionary.to_pylist()
                    if isinstance(left_column, pa.DictionaryArray)
                    else None
                )
                right_categories = (
                    right_column.dictionary.to_pylist()
                    if isinstance(right_column, pa.DictionaryArray)
                    else None
                )
                result["left_categories"] = left_categories
                result["right_categories"] = right_categories
                result["categories_equal"] = left_categories == right_categories
                result["category_ordering_equal"] = (
                    getattr(left_field.type, "ordered", None)
                    == getattr(right_field.type, "ordered", None)
                )
            column_results[column] = result
        payload = {
            "left": {"path": str(left_path), "sha256": left_hash, "rows": left.metadata.num_rows, "columns": left_columns},
            "right": {"path": str(right_path), "sha256": right_hash, "rows": right.metadata.num_rows, "columns": right_columns},
            "row_counts_equal": left.metadata.num_rows == right.metadata.num_rows,
            "columns_only_left": [column for column in left_columns if column not in set(right_columns)],
            "columns_only_right": [column for column in right_columns if column not in set(left_columns)],
            "tolerances": {"absolute": absolute_tolerance, "relative": relative_tolerance},
            "column_results": column_results,
        }
    finally:
        left.close(); right.close()
    if _sha256_file(left_path) != left_hash or _sha256_file(right_path) != right_hash:
        raise RuntimeError("A compared artifact changed during comparison.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ComparisonResult(left_path, right_path, output_path, payload["row_counts_equal"], len(common))
