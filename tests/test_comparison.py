from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.comparison import (
    compare_parquet_artifacts,
)


class ComparisonTests(unittest.TestCase):
    def test_reports_numeric_categorical_and_null_differences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_path = root / "left.parquet"
            right_path = root / "right.parquet"
            report_path = root / "comparison.json"
            pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "value": [1.0, np.nan, 3.0],
                    "label": ["a", "b", "c"],
                    "left_only": [1, 1, 1],
                }
            ).to_parquet(left_path, index=False)
            pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "value": [1.0, 2.0, 3.1],
                    "label": ["a", "x", "c"],
                    "right_only": [2, 2, 2],
                }
            ).to_parquet(right_path, index=False)

            result = compare_parquet_artifacts(
                left_path,
                right_path,
                report_path,
                absolute_tolerance=0.01,
                batch_size=2,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertTrue(result.row_counts_equal)
        self.assertEqual(report["columns_only_left"], ["left_only"])
        self.assertEqual(report["columns_only_right"], ["right_only"])
        self.assertEqual(
            report["column_results"]["value"]["null_mismatch_count"],
            1,
        )
        self.assertEqual(
            report["column_results"]["value"]["value_mismatch_count"],
            1,
        )
        self.assertEqual(
            report["column_results"]["label"]["value_mismatch_count"],
            1,
        )

    def test_tolerance_can_accept_small_numeric_difference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_path = root / "left.parquet"
            right_path = root / "right.parquet"
            report_path = root / "comparison.json"
            pd.DataFrame({"value": [1.0]}).to_parquet(left_path, index=False)
            pd.DataFrame({"value": [1.0001]}).to_parquet(right_path, index=False)
            compare_parquet_artifacts(
                left_path,
                right_path,
                report_path,
                absolute_tolerance=0.001,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(
            report["column_results"]["value"]["value_mismatch_count"],
            0,
        )

    def test_large_integer_difference_is_not_lost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_path = root / "left.parquet"
            right_path = root / "right.parquet"
            report_path = root / "comparison.json"
            pd.DataFrame({"id": [2**53]}).to_parquet(left_path, index=False)
            pd.DataFrame({"id": [2**53 + 1]}).to_parquet(right_path, index=False)
            compare_parquet_artifacts(left_path, right_path, report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
        result = report["column_results"]["id"]
        self.assertEqual(result["value_mismatch_count"], 1)
        self.assertEqual(result["maximum_absolute_integer_difference"], 1)

    def test_rejects_output_that_would_replace_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            other = Path(directory) / "other.parquet"
            pd.DataFrame({"value": [1]}).to_parquet(path, index=False)
            pd.DataFrame({"value": [1]}).to_parquet(other, index=False)
            with self.assertRaisesRegex(ValueError, "must differ"):
                compare_parquet_artifacts(path, other, path)

    def test_nullable_large_integer_difference_is_not_lost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_path = root / "left.parquet"
            right_path = root / "right.parquet"
            report_path = root / "comparison.json"
            pd.DataFrame(
                {"id": pd.array([2**53, None], dtype="Int64")}
            ).to_parquet(left_path, index=False)
            pd.DataFrame(
                {"id": pd.array([2**53 + 1, None], dtype="Int64")}
            ).to_parquet(right_path, index=False)
            compare_parquet_artifacts(left_path, right_path, report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(
            report["column_results"]["id"]["value_mismatch_count"],
            1,
        )

    def test_reports_categorical_dictionary_difference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_path = root / "left.parquet"
            right_path = root / "right.parquet"
            report_path = root / "comparison.json"
            pd.DataFrame(
                {
                    "label": pd.Categorical(
                        ["a"],
                        categories=["a", "b"],
                        ordered=True,
                    )
                }
            ).to_parquet(left_path, index=False)
            pd.DataFrame(
                {
                    "label": pd.Categorical(
                        ["a"],
                        categories=["a", "c"],
                        ordered=True,
                    )
                }
            ).to_parquet(right_path, index=False)
            compare_parquet_artifacts(left_path, right_path, report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
        result = report["column_results"]["label"]
        self.assertFalse(result["categories_equal"])
        self.assertEqual(result["value_mismatch_count"], 0)

    def test_mixed_signed_integer_types_compare_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_path = root / "left.parquet"
            right_path = root / "right.parquet"
            report_path = root / "comparison.json"
            pq.write_table(
                pa.table({"id": pa.array([-1], type=pa.int64())}),
                left_path,
            )
            pq.write_table(
                pa.table({"id": pa.array([2**64 - 1], type=pa.uint64())}),
                right_path,
            )
            compare_parquet_artifacts(left_path, right_path, report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
        result = report["column_results"]["id"]
        self.assertEqual(result["value_mismatch_count"], 1)
        self.assertEqual(result["maximum_absolute_integer_difference"], 2**64)

    def test_rejects_input_changed_during_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_path = root / "left.parquet"
            right_path = root / "right.parquet"
            report_path = root / "comparison.json"
            pd.DataFrame({"value": [1]}).to_parquet(left_path, index=False)
            pd.DataFrame({"value": [1]}).to_parquet(right_path, index=False)
            with patch(
                "classical_conditioning.comparison._sha256_file",
                side_effect=["left-before", "right-before", "left-after", "right-before"],
            ):
                with self.assertRaisesRegex(RuntimeError, "changed during comparison"):
                    compare_parquet_artifacts(left_path, right_path, report_path)
            self.assertFalse(report_path.exists())


if __name__ == "__main__":
    unittest.main()
