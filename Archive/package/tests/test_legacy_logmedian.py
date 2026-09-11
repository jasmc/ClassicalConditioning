from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.artifacts import sha256_file
from classical_conditioning.analysis.legacy_logmedian import (
    SCALED_VIGOR_COLUMN,
    VIGOR_COLUMN,
    LegacyLogMedianConfig,
    _iter_contiguous_trial_groups,
    _transform_groups_to_parquet,
    build_legacy_logmedian,
    transform_legacy_logmedian,
)


class LegacyLogMedianTests(unittest.TestCase):
    @staticmethod
    def _historical_reference(
        samples: pd.DataFrame,
        config: LegacyLogMedianConfig,
    ) -> pd.DataFrame:
        result = samples.copy()
        result["Fish"] = result["Fish"].astype("string")
        result["Trial number"] = result["Trial number"].astype("int32")
        result["Trial time (frame) [700 FPS]"] = result[
            "Trial time (frame) [700 FPS]"
        ].astype("int32")
        result[VIGOR_COLUMN] = result[VIGOR_COLUMN].astype("float32")
        result[SCALED_VIGOR_COLUMN] = result[SCALED_VIGOR_COLUMN].astype("float32")
        result.loc[~result["Bout"], VIGOR_COLUMN] = np.nan
        grouping = ["Fish", "Trial type", "Trial number"]
        result = result.sort_values(
            grouping + ["Trial time (frame) [700 FPS]"]
        )

        def historical_rolling(values: pd.Series) -> np.ndarray:
            source = values.to_numpy()
            output = np.empty(len(source), dtype=source.dtype)
            for index in range(len(source)):
                start = max(0, index - config.rolling_window_frames + 1)
                output[index] = np.nanmedian(source[start : index + 1])
            return output

        result[VIGOR_COLUMN] = result.groupby(
            grouping,
            observed=True,
            sort=False,
        )[VIGOR_COLUMN].transform(historical_rolling)
        result = result.groupby(
            grouping,
            observed=True,
            group_keys=False,
            sort=False,
        ).nth(slice(None, None, config.downsample_factor))
        vigor = result[VIGOR_COLUMN].to_numpy(dtype="float64")
        result[VIGOR_COLUMN] = np.where(vigor > 0, np.log(vigor), np.nan)
        baseline = (
            result.loc[
                result["Trial time (frame) [700 FPS]"]
                < -config.baseline_window_frames
            ]
            .groupby(grouping, observed=True)[VIGOR_COLUMN]
            .median()
        )
        result = result.merge(
            baseline.rename("_bl_median"),
            on=grouping,
            how="left",
        )
        result[SCALED_VIGOR_COLUMN] = (
            result[VIGOR_COLUMN] - result["_bl_median"]
        )
        return result.drop(columns="_bl_median").reset_index(drop=True)

    def test_reproduces_historical_operation_order(self) -> None:
        samples = pd.DataFrame(
            {
                "Fish": ["fish-1"] * 8,
                "Trial type": ["CS"] * 8,
                "Trial number": [1] * 8,
                "Trial time (frame) [700 FPS]": np.arange(-14, 2, 2),
                VIGOR_COLUMN: [1.0, 100.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0],
                SCALED_VIGOR_COLUMN: np.zeros(8),
                "Bout": [True, False, True, True, True, True, True, True],
            }
        )
        config = LegacyLogMedianConfig(
            rolling_window_frames=3,
            downsample_factor=2,
            baseline_window_frames=10,
        )

        result = transform_legacy_logmedian(samples, config)

        self.assertEqual(
            result["Trial time (frame) [700 FPS]"].tolist(),
            [-14, -10, -6, -2],
        )
        expected_log_vigor = np.log([1.0, 5.0, 16.0, 36.0])
        np.testing.assert_allclose(result[VIGOR_COLUMN], expected_log_vigor)
        baseline = expected_log_vigor[0]
        np.testing.assert_allclose(
            result[SCALED_VIGOR_COLUMN],
            expected_log_vigor - baseline,
        )

    def test_matches_executable_historical_process_on_shared_fixture(self) -> None:
        samples = pd.DataFrame(
            {
                "Fish": ["fish-1"] * 10,
                "Trial type": ["CS"] * 5 + ["US"] * 5,
                "Trial number": [1] * 5 + [1] * 5,
                "Trial time (frame) [700 FPS]": [-4, -3, -2, -1, 0] * 2,
                VIGOR_COLUMN: [
                    1.00000006,
                    7.0,
                    3.0,
                    0.0,
                    9.0,
                    2.0,
                    4.0,
                    8.0,
                    16.0,
                    32.0,
                ],
                SCALED_VIGOR_COLUMN: np.zeros(10),
                "Bout": [True, False, True, True, True] * 2,
            }
        )
        config = LegacyLogMedianConfig(
            rolling_window_frames=3,
            downsample_factor=2,
            baseline_window_frames=1,
        )

        actual = transform_legacy_logmedian(samples, config)
        expected = self._historical_reference(samples, config)

        pd.testing.assert_frame_equal(actual, expected)

    def test_keeps_trial_types_separate_when_numbers_overlap(self) -> None:
        samples = pd.DataFrame(
            {
                "Fish": ["fish-1"] * 4,
                "Trial type": ["CS", "CS", "US", "US"],
                "Trial number": [1, 1, 1, 1],
                "Trial time (frame) [700 FPS]": [-2, 0, -2, 0],
                VIGOR_COLUMN: [1.0, 4.0, 9.0, 16.0],
                SCALED_VIGOR_COLUMN: np.zeros(4),
                "Bout": [True] * 4,
            }
        )
        config = LegacyLogMedianConfig(
            rolling_window_frames=1,
            downsample_factor=1,
            baseline_window_frames=1,
        )

        result = transform_legacy_logmedian(samples, config)

        cs = result[result["Trial type"] == "CS"]
        us = result[result["Trial type"] == "US"]
        np.testing.assert_allclose(cs[SCALED_VIGOR_COLUMN], [0.0, np.log(4.0)])
        np.testing.assert_allclose(
            us[SCALED_VIGOR_COLUMN],
            [0.0, np.log(16.0) - np.log(9.0)],
        )

    def test_constructs_historical_fish_identifier_from_day_and_fish_number(
        self,
    ) -> None:
        samples = pd.DataFrame(
            {
                "Day": ["20221115", "20221115"],
                "Fish no.": ["04", "04"],
                "Trial type": ["CS", "CS"],
                "Trial number": [1, 1],
                "Trial time (frame) [700 FPS]": [-2, 0],
                VIGOR_COLUMN: [1.0, 4.0],
                SCALED_VIGOR_COLUMN: [0.0, 0.0],
                "Bout": [True, True],
            }
        )

        result = transform_legacy_logmedian(
            samples,
            LegacyLogMedianConfig(
                rolling_window_frames=1,
                downsample_factor=1,
                baseline_window_frames=1,
            ),
        )

        self.assertEqual(result["Fish"].tolist(), ["20221115_04"] * 2)

    def test_nonpositive_values_and_missing_baselines_remain_missing(self) -> None:
        samples = pd.DataFrame(
            {
                "Fish": ["fish-1", "fish-1"],
                "Trial type": ["CS", "CS"],
                "Trial number": [1, 1],
                "Trial time (frame) [700 FPS]": [0, 1],
                VIGOR_COLUMN: [0.0, -1.0],
                SCALED_VIGOR_COLUMN: [0.0, 0.0],
                "Bout": [True, True],
            }
        )

        result = transform_legacy_logmedian(
            samples,
            LegacyLogMedianConfig(
                rolling_window_frames=1,
                downsample_factor=1,
                baseline_window_frames=1,
            ),
        )

        self.assertTrue(result[VIGOR_COLUMN].isna().all())
        self.assertTrue(result[SCALED_VIGOR_COLUMN].isna().all())

    def test_rejects_ambiguous_bout_state(self) -> None:
        samples = pd.DataFrame(
            {
                "Fish": ["fish-1"],
                "Trial type": ["CS"],
                "Trial number": [1],
                "Trial time (frame) [700 FPS]": [-2],
                VIGOR_COLUMN: [1.0],
                SCALED_VIGOR_COLUMN: [0.0],
                "Bout": [np.nan],
            }
        )

        with self.assertRaisesRegex(TypeError, "complete boolean"):
            transform_legacy_logmedian(samples)

    def test_builder_verifies_lineage_and_publishes_lossless_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            recording_id = "recording-1"
            source_dir = project_dir / "Processed data" / recording_id
            quality_dir = project_dir / "Quality checks" / recording_id
            metadata_dir = project_dir / "Metadata"
            source_dir.mkdir(parents=True)
            quality_dir.mkdir(parents=True)
            metadata_dir.mkdir(parents=True)
            source_path = source_dir / "samples_legacy-v1.parquet"
            source = pd.DataFrame(
                {
                    "Fish": ["fish-1"] * 4,
                    "Trial type": ["CS"] * 4,
                    "Trial number": [1] * 4,
                    "Trial time (frame) [700 FPS]": [-2, -1, 0, 1],
                    VIGOR_COLUMN: [1.0, 4.0, 9.0, 16.0],
                    SCALED_VIGOR_COLUMN: np.zeros(4),
                    "Bout": [True] * 4,
                }
            )
            pq.write_table(
                pa.Table.from_pandas(source, preserve_index=False),
                source_path,
                compression="zstd",
            )
            source_hash = sha256_file(source_path)
            summary_path = quality_dir / "legacy-v1_preprocessing_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "recipe": "legacy-paper-v1",
                        "recording_id": recording_id,
                        "artifact": {"sha256": source_hash},
                    }
                ),
                encoding="utf-8",
            )
            marker_path = metadata_dir / f"{recording_id}_legacy-v1_complete.json"
            marker_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "recipe": "legacy-paper-v1",
                        "recording_id": recording_id,
                        "samples_sha256": source_hash,
                        "summary_sha256": sha256_file(summary_path),
                    }
                ),
                encoding="utf-8",
            )

            result = build_legacy_logmedian(
                project_dir,
                recording_id,
                config=LegacyLogMedianConfig(
                    rolling_window_frames=1,
                    downsample_factor=1,
                    baseline_window_frames=1,
                ),
            )

            self.assertEqual(result.row_count, 4)
            self.assertTrue(result.samples_path.is_file())
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            marker = json.loads(
                result.completion_marker_path.read_text(encoding="utf-8")
            )
            self.assertTrue(summary["artifact"]["compression_lossless"])
            self.assertEqual(
                sha256_file(result.samples_path),
                marker["samples_sha256"],
            )
            self.assertEqual(
                sha256_file(result.summary_path),
                marker["summary_sha256"],
            )
            self.assertEqual(summary["streaming"]["maximum_source_group_rows"], 4)
            self.assertEqual(summary["trial_counts"], {"CS": 1})

    def test_streaming_rejects_trial_groups_that_reappear(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "noncontiguous.parquet"
            frame = pd.DataFrame(
                {
                    "Fish": ["fish-1"] * 3,
                    "Trial type": ["CS"] * 3,
                    "Trial number": [1, 2, 1],
                    "Trial time (frame) [700 FPS]": [0, 0, 1],
                }
            )
            pq.write_table(
                pa.Table.from_pandas(frame, preserve_index=False),
                path,
            )

            with self.assertRaisesRegex(ValueError, "not contiguous"):
                list(
                    _iter_contiguous_trial_groups(
                        pq.ParquetFile(path),
                        batch_rows=2,
                    )
                )

    def test_streamed_transform_matches_in_memory_across_batch_boundaries(self) -> None:
        source = pd.DataFrame(
            {
                "Fish": ["fish-1"] * 7,
                "Trial type": ["CS"] * 7,
                "Trial number": [1, 1, 1, 1, 2, 2, 2],
                "Trial time (frame) [700 FPS]": [-3, -2, -1, 0, -2, -1, 0],
                VIGOR_COLUMN: [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0],
                SCALED_VIGOR_COLUMN: np.zeros(7),
                "Bout": [True, False, True, True, True, True, True],
            }
        )
        config = LegacyLogMedianConfig(
            rolling_window_frames=2,
            downsample_factor=2,
            baseline_window_frames=1,
            read_batch_rows=3,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "source.parquet"
            output_path = Path(temporary_directory) / "output.parquet"
            pq.write_table(
                pa.Table.from_pandas(source, preserve_index=False),
                source_path,
            )

            row_count, trial_counts, maximum_group_rows = (
                _transform_groups_to_parquet(source_path, output_path, config)
            )
            actual = pq.read_table(output_path).to_pandas()

        expected = transform_legacy_logmedian(source, config)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(row_count, len(expected))
        self.assertEqual(trial_counts, {"CS": 2})
        self.assertEqual(maximum_group_rows, 4)

    def test_streaming_rejects_nonmonotonic_trial_time(self) -> None:
        source = pd.DataFrame(
            {
                "Fish": ["fish-1"] * 2,
                "Trial type": ["CS"] * 2,
                "Trial number": [1, 1],
                "Trial time (frame) [700 FPS]": [0, -1],
                VIGOR_COLUMN: [1.0, 2.0],
                SCALED_VIGOR_COLUMN: [0.0, 0.0],
                "Bout": [True, True],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "source.parquet"
            output_path = Path(temporary_directory) / "output.parquet"
            pq.write_table(
                pa.Table.from_pandas(source, preserve_index=False),
                source_path,
            )

            with self.assertRaisesRegex(ValueError, "not strictly increasing"):
                _transform_groups_to_parquet(
                    source_path,
                    output_path,
                    LegacyLogMedianConfig(read_batch_rows=1),
                )

    def test_streaming_rejects_duplicate_trial_time(self) -> None:
        source = pd.DataFrame(
            {
                "Fish": ["fish-1"] * 2,
                "Trial type": ["CS"] * 2,
                "Trial number": [1, 1],
                "Trial time (frame) [700 FPS]": [0, 0],
                VIGOR_COLUMN: [1.0, 2.0],
                SCALED_VIGOR_COLUMN: [0.0, 0.0],
                "Bout": [True, True],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "source.parquet"
            output_path = Path(temporary_directory) / "output.parquet"
            pq.write_table(
                pa.Table.from_pandas(source, preserve_index=False),
                source_path,
            )

            with self.assertRaisesRegex(ValueError, "not strictly increasing"):
                _transform_groups_to_parquet(
                    source_path,
                    output_path,
                    LegacyLogMedianConfig(read_batch_rows=1),
                )


if __name__ == "__main__":
    unittest.main()
