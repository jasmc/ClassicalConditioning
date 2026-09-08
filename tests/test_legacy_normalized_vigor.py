from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.legacy_normalized_vigor import (
    BASELINE_COLUMN,
    CONFIG_SHA256,
    NORMALIZED_COLUMN,
    RECIPE_ID,
    RESPONSE_COLUMN,
    LegacyNormalizedVigorConfig,
    aggregate_legacy_normalized_vigor,
    build_legacy_normalized_vigor,
)
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.cli import build_parser
from classical_conditioning.exceptions import ConfigurationError


def make_samples(
    *,
    alignment: str,
    times_s: list[float],
    vigor: list[float],
    trial_number: int,
) -> pd.DataFrame:
    size = len(times_s)
    return pd.DataFrame(
        {
            "Strain": ["strain"] * size,
            "Age (dpf)": ["6"] * size,
            "Exp.": ["delay"] * size,
            "ProtocolRig": ["rig"] * size,
            "Day": ["20221115"] * size,
            "Fish no.": ["04"] * size,
            "Fish": ["20221115_04"] * size,
            "Block name": ["wrong"] * size,
            "Trial number": [trial_number] * size,
            "Trial time (frame) [700 FPS]": np.asarray(times_s) * 700,
            "Vigor (deg/ms)": vigor,
            "US beg": [0] * size,
            "Trial type": [alignment] * size,
        }
    )


class LegacyNormalizedVigorTests(unittest.TestCase):
    def test_cs_includes_zero_in_baseline_and_response(self) -> None:
        samples = make_samples(
            alignment="CS",
            times_s=[-2.0, 0.0, 1.0],
            vigor=[2.0, 10.0, 4.0],
            trial_number=5,
        )
        result = aggregate_legacy_normalized_vigor(
            samples,
            alignment="CS",
            config=LegacyNormalizedVigorConfig(
                baseline_duration_s=2.0,
                response_window_s=(0.0, 1.0),
            ),
        )

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[BASELINE_COLUMN].iloc[0], 6.0)
        self.assertAlmostEqual(result[RESPONSE_COLUMN].iloc[0], 7.0)
        self.assertAlmostEqual(result[NORMALIZED_COLUMN].iloc[0], 7.0 / 6.0)
        self.assertEqual(result["Block name"].astype(str).iloc[0], "Pre-train")

    def test_us_windows_share_negative_response_endpoint(self) -> None:
        samples = make_samples(
            alignment="US",
            times_s=[-3.0, -1.0, 0.0],
            vigor=[2.0, 10.0, 4.0],
            trial_number=18,
        )
        result = aggregate_legacy_normalized_vigor(
            samples,
            alignment="US",
            config=LegacyNormalizedVigorConfig(
                baseline_duration_s=2.0,
                response_window_s=(0.0, 1.0),
            ),
        )

        self.assertAlmostEqual(result[BASELINE_COLUMN].iloc[0], 6.0)
        self.assertAlmostEqual(result[RESPONSE_COLUMN].iloc[0], 7.0)
        self.assertAlmostEqual(result[NORMALIZED_COLUMN].iloc[0], 7.0 / 6.0)
        self.assertEqual(result["Block name"].astype(str).iloc[0], "Train 1")

    def test_zero_baseline_is_not_guarded(self) -> None:
        samples = make_samples(
            alignment="CS",
            times_s=[-1.0, 0.0, 1.0],
            vigor=[0.0, 0.0, 4.0],
            trial_number=5,
        )
        result = aggregate_legacy_normalized_vigor(
            samples,
            alignment="CS",
            config=LegacyNormalizedVigorConfig(
                baseline_duration_s=1.0,
                response_window_s=(0.0, 1.0),
            ),
        )

        self.assertTrue(np.isinf(result[NORMALIZED_COLUMN].iloc[0]))

    def test_negative_time_us_marker_masks_vigor_before_windows(self) -> None:
        samples = make_samples(
            alignment="CS",
            times_s=[-1.0, 0.0, 1.0],
            vigor=[100.0, 2.0, 4.0],
            trial_number=5,
        )
        samples.loc[0, "US beg"] = 1
        result = aggregate_legacy_normalized_vigor(
            samples,
            alignment="CS",
            config=LegacyNormalizedVigorConfig(
                baseline_duration_s=1.0,
                response_window_s=(0.0, 1.0),
            ),
        )

        self.assertAlmostEqual(result[BASELINE_COLUMN].iloc[0], 2.0)
        self.assertAlmostEqual(result[RESPONSE_COLUMN].iloc[0], 3.0)

    def test_us_baseline_is_truncated_by_outer_trial_window(self) -> None:
        samples = make_samples(
            alignment="US",
            times_s=[-24.0, -21.0, -9.0, 0.0],
            vigor=[100.0, 2.0, 10.0, 4.0],
            trial_number=18,
        )

        result = aggregate_legacy_normalized_vigor(
            samples,
            alignment="US",
        )

        self.assertAlmostEqual(result[BASELINE_COLUMN].iloc[0], 6.0)
        self.assertAlmostEqual(result[RESPONSE_COLUMN].iloc[0], 7.0)

    def test_frozen_config_rejects_corrected_boundaries_and_filters(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "both endpoints"):
            LegacyNormalizedVigorConfig(interval_closure="left")
        with self.assertRaisesRegex(ConfigurationError, "fish exclusion"):
            LegacyNormalizedVigorConfig(apply_fish_discard=True)
        with self.assertRaisesRegex(ConfigurationError, "missing-window"):
            LegacyNormalizedVigorConfig(apply_nan_fraction_filter=True)

    def test_builder_verifies_inputs_and_publishes_both_alignments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            recording_id = "recording-1"
            processed_dir = project_dir / "Processed data" / recording_id
            quality_dir = project_dir / "Quality checks" / recording_id
            metadata_dir = project_dir / "Metadata"
            processed_dir.mkdir(parents=True)
            quality_dir.mkdir(parents=True)
            metadata_dir.mkdir(parents=True)

            sources = {
                "CS": make_samples(
                    alignment="CS",
                    times_s=[-15.0, 0.0, 9.0],
                    vigor=[2.0, 10.0, 4.0],
                    trial_number=5,
                ),
                "US": make_samples(
                    alignment="US",
                    times_s=[-21.0, -9.0, 0.0],
                    vigor=[2.0, 10.0, 4.0],
                    trial_number=18,
                ),
            }
            source_paths = {
                alignment: processed_dir
                / f"samples_legacy-standard-main-v1_{alignment}.parquet"
                for alignment in sources
            }
            source_hashes = {}
            for alignment, source in sources.items():
                pq.write_table(
                    pa.Table.from_pandas(source, preserve_index=False),
                    source_paths[alignment],
                    compression="zstd",
                )
                source_hashes[alignment] = sha256_file(source_paths[alignment])
            source_summary_path = (
                quality_dir / "legacy-standard-main-v1_summary.json"
            )
            source_summary_path.write_text(
                json.dumps(
                    {
                        "recipe": "legacy-standard-main-v1",
                        "recording_id": recording_id,
                        "experiment": "allDelay",
                        "artifacts": {
                            alignment: {
                                "path": str(source_paths[alignment]),
                                "sha256": source_hashes[alignment],
                            }
                            for alignment in sources
                        },
                    }
                ),
                encoding="utf-8",
            )
            source_marker_path = (
                metadata_dir
                / f"{recording_id}_legacy-standard-main-v1_complete.json"
            )
            source_marker_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "recipe": "legacy-standard-main-v1",
                        "recording_id": recording_id,
                        "artifact_sha256": source_hashes,
                        "summary_sha256": sha256_file(source_summary_path),
                    }
                ),
                encoding="utf-8",
            )

            result = build_legacy_normalized_vigor(
                project_dir,
                recording_id,
            )

            self.assertEqual(result.row_counts, {"CS": 1, "US": 1})
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            marker = json.loads(
                result.completion_marker_path.read_text(encoding="utf-8")
            )
            self.assertEqual(summary["config_sha256"], CONFIG_SHA256)
            self.assertFalse(summary["config"]["apply_fish_discard"])
            self.assertFalse(summary["config"]["apply_nan_fraction_filter"])
            for alignment, path in result.artifact_paths.items():
                self.assertEqual(
                    sha256_file(path),
                    marker["artifact_sha256"][alignment],
                )
                output = pq.read_table(path).to_pandas()
                expected = aggregate_legacy_normalized_vigor(
                    sources[alignment],
                    alignment=alignment,
                )
                pd.testing.assert_frame_equal(output, expected)
            with self.assertRaises(FileExistsError):
                build_legacy_normalized_vigor(project_dir, recording_id)

            source_paths["US"].unlink()
            source_summary = json.loads(
                source_summary_path.read_text(encoding="utf-8")
            )
            del source_summary["artifacts"]["US"]
            source_summary_path.write_text(
                json.dumps(source_summary),
                encoding="utf-8",
            )
            source_marker = json.loads(
                source_marker_path.read_text(encoding="utf-8")
            )
            del source_marker["artifact_sha256"]["US"]
            source_marker["summary_sha256"] = sha256_file(source_summary_path)
            source_marker_path.write_text(
                json.dumps(source_marker),
                encoding="utf-8",
            )

            replaced = build_legacy_normalized_vigor(
                project_dir,
                recording_id,
                overwrite=True,
            )

            self.assertEqual(set(replaced.artifact_paths), {"CS"})
            self.assertFalse(result.artifact_paths["US"].exists())

    def test_cli_exposes_explicit_stage_five_window_recipe(self) -> None:
        args = build_parser().parse_args(
            [
                "legacy-normalized-vigor",
                "--project-dir",
                ".",
                "--recording-id",
                "recording",
            ]
        )
        self.assertEqual(args.recipe, RECIPE_ID)
        self.assertEqual(args.experiment, "allDelay")


if __name__ == "__main__":
    unittest.main()
