from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.legacy_scaled_vigor import (
    CONFIG_SHA256,
    RECIPE_ID,
    SCALED_VIGOR_COLUMN,
    LegacyScaledVigorConfig,
    aggregate_legacy_scaled_vigor,
    build_legacy_scaled_vigor,
    build_legacy_scaled_vigor_cohort,
    compute_legacy_time_bins,
)
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.cli import build_parser
from classical_conditioning.exceptions import ConfigurationError


class LegacyScaledVigorTests(unittest.TestCase):
    def test_time_bins_match_executable_zero_edge_and_extension(self) -> None:
        bins = compute_legacy_time_bins((-20.0, 20.0), 0.5)

        self.assertIn(0.0, bins)
        self.assertLessEqual(min(bins), -21.0)
        self.assertGreaterEqual(max(bins), 22.0)
        np.testing.assert_allclose(np.diff(bins), 0.5)

    def test_reproduces_two_stage_aggregation_and_fish_count(self) -> None:
        times = [-3.75, -3.25, -0.75, -0.25, 0.25, 0.75]
        values = {
            "fish-1": [0.0, 1.0, 10.0, 20.0, 30.0, 40.0],
            "fish-2": [100.0, 2.0, 10.0, 20.0, 30.0, 40.0],
        }
        rows = []
        for fish, fish_values in values.items():
            for time, value in zip(times, fish_values, strict=True):
                rows.append(
                    {
                        "Exp.": "delay",
                        "Trial time (frame) [700 FPS]": time * 700,
                        "Trial number": 1,
                        "Block name": "Train",
                        SCALED_VIGOR_COLUMN: value,
                        "Fish": fish,
                        "Bout": True,
                    }
                )

        tables = aggregate_legacy_scaled_vigor(
            pd.DataFrame(rows),
            bin_width_s=1.0,
        )

        first_bin = tables.line.loc[
            np.isclose(tables.line["Trial time (s)"], -3.5),
            SCALED_VIGOR_COLUMN,
        ].iloc[0]
        self.assertAlmostEqual(first_bin, 25.75)
        self.assertTrue(np.allclose(tables.line["Count"], 2.0))
        self.assertTrue(np.allclose(tables.count_heatmap["Count"], 2.0))
        scaled = tables.scaled_heatmap.set_index("Trial time (s)")[
            SCALED_VIGOR_COLUMN
        ]
        self.assertAlmostEqual(float(scaled.loc[-3.5]), 1.0)
        self.assertAlmostEqual(float(scaled.loc[-0.5]), 0.0)
        self.assertAlmostEqual(float(scaled.loc[0.5]), 1.0)

    def test_masks_rest_and_applies_exact_fish_exclusion(self) -> None:
        frame = pd.DataFrame(
            {
                "Exp.": ["delay"] * 4,
                "Trial time (frame) [700 FPS]": [-700, 0, -700, 0],
                "Trial number": [1] * 4,
                "Block name": ["Train"] * 4,
                SCALED_VIGOR_COLUMN: [1.0, 100.0, 3.0, 5.0],
                "Fish": ["keep", "keep", "discard", "discard"],
                "Bout": [True, False, True, True],
            }
        )

        tables = aggregate_legacy_scaled_vigor(
            frame,
            bin_width_s=1.0,
            discarded_fish_ids=("discard",),
        )

        self.assertEqual(tables.fish_count, 1)
        at_zero = tables.line.loc[
            np.isclose(tables.line["Trial time (s)"], 0.5)
        ]
        self.assertTrue(at_zero[SCALED_VIGOR_COLUMN].isna().all())
        self.assertTrue(np.allclose(at_zero["Count"], 0.0))

    def test_frozen_config_rejects_scientific_changes(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "quantiles"):
            LegacyScaledVigorConfig(
                heatmap_lower_quantile=0.9,
                heatmap_upper_quantile=0.1,
            )

    def test_builder_verifies_standard_main_lineage_and_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            recording_id = "recording-1"
            processed_dir = project_dir / "Processed data" / recording_id
            quality_dir = project_dir / "Quality checks" / recording_id
            metadata_dir = project_dir / "Metadata"
            processed_dir.mkdir(parents=True)
            quality_dir.mkdir(parents=True)
            metadata_dir.mkdir(parents=True)

            source = pd.DataFrame(
                {
                    "Exp.": ["delay"] * 8,
                    "Trial time (frame) [700 FPS]": [
                        -1400,
                        -700,
                        0,
                        700,
                    ]
                    * 2,
                    "Trial number": [5] * 8,
                    "Block name": ["Pre-train"] * 8,
                    SCALED_VIGOR_COLUMN: [
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        9.0,
                        10.0,
                        11.0,
                        12.0,
                    ],
                    "Fish": ["keep"] * 4 + ["discard"] * 4,
                    "Bout": [True] * 8,
                }
            )
            source_paths = {
                alignment: processed_dir
                / f"samples_legacy-standard-main-v1_{alignment}.parquet"
                for alignment in ("CS", "US")
            }
            source_hashes = {}
            for alignment, source_path in source_paths.items():
                alignment_source = source.assign(
                    **{
                        "Trial number": (
                            5 if alignment == "CS" else 18
                        ),
                        "Block name": (
                            "Pre-train" if alignment == "CS" else "Train 1"
                        ),
                    }
                )
                pq.write_table(
                    pa.Table.from_pandas(
                        alignment_source,
                        preserve_index=False,
                    ),
                    source_path,
                    compression="zstd",
                )
                source_hashes[alignment] = sha256_file(source_path)
            source_summary_path = (
                quality_dir / "legacy-standard-main-v1_summary.json"
            )
            source_summary_path.write_text(
                json.dumps(
                    {
                        "recipe": "legacy-standard-main-v1",
                        "recording_id": recording_id,
                        "artifacts": {
                            alignment: {
                                "path": str(source_paths[alignment]),
                                "sha256": source_hashes[alignment],
                            }
                            for alignment in ("CS", "US")
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
            discard_path = (
                project_dir / "Processed data" / "Discarded_fish_IDs.txt"
            )
            discard_path.write_text("discard\n", encoding="utf-8")

            result = build_legacy_scaled_vigor(project_dir, recording_id)

            self.assertEqual(len(result.artifact_paths), 12)
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            marker = json.loads(
                result.completion_marker_path.read_text(encoding="utf-8")
            )
            self.assertEqual(summary["config_sha256"], CONFIG_SHA256)
            self.assertEqual(summary["fish_counts"], {"CS": 1, "US": 1})
            self.assertEqual(
                summary["fish_exclusion"]["fish_ids"],
                ["discard"],
            )
            for key, path in result.artifact_paths.items():
                self.assertEqual(
                    sha256_file(path),
                    marker["artifact_sha256"][key],
                )
                self.assertTrue(
                    summary["artifacts"][key]["compression_lossless"]
                )
            line = pq.read_table(
                result.artifact_paths["CS_1000ms_line"]
            ).to_pandas()
            self.assertAlmostEqual(
                line.loc[
                    np.isclose(line["Trial time (s)"], -1.5),
                    SCALED_VIGOR_COLUMN,
                ].iloc[0],
                2.0,
            )
            us_line = pq.read_table(
                result.artifact_paths["US_1000ms_line"]
            ).to_pandas()
            self.assertEqual(set(us_line["Trial number"]), {18})
            self.assertEqual(set(us_line["Block name"].astype(str)), {"Train 1"})
            with self.assertRaises(FileExistsError):
                build_legacy_scaled_vigor(project_dir, recording_id)

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

            replaced = build_legacy_scaled_vigor(
                project_dir,
                recording_id,
                overwrite=True,
            )

            self.assertEqual(len(replaced.artifact_paths), 6)
            self.assertTrue(
                all(key.startswith("CS_") for key in replaced.artifact_paths)
            )
            self.assertFalse(
                any(
                    path.exists()
                    for key, path in result.artifact_paths.items()
                    if key.startswith("US_")
                )
            )

    def test_cli_exposes_explicit_stage_four_recipe(self) -> None:
        args = build_parser().parse_args(
            [
                "legacy-scaled-vigor",
                "--project-dir",
                ".",
                "--recording-id",
                "recording",
            ]
        )
        self.assertEqual(args.recipe, RECIPE_ID)

    def test_cohort_builder_pools_recordings_before_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            (project_dir / "Metadata").mkdir(parents=True)
            sources = {}
            recording_conditions = (
                ("recording-a", "delay"),
                ("recording-b", "delay"),
                ("recording-c", "control"),
            )
            for index, (recording_id, condition) in enumerate(
                recording_conditions
            ):
                source_dir = project_dir / recording_id
                source_dir.mkdir()
                source_path = source_dir / "CS.parquet"
                frame = pd.DataFrame(
                    {
                        "Exp.": [condition] * 4,
                        "Trial time (frame) [700 FPS]": [-1400, -700, 0, 700],
                        "Trial number": [5] * 4,
                        "Block name": ["Pre-train"] * 4,
                        SCALED_VIGOR_COLUMN: [
                            1.0 + index,
                            2.0 + index,
                            3.0 + index,
                            4.0 + index,
                        ],
                        "Fish": [f"fish-{index}"] * 4,
                        "Bout": [True] * 4,
                    }
                )
                pq.write_table(
                    pa.Table.from_pandas(frame, preserve_index=False),
                    source_path,
                    compression="zstd",
                )
                summary_path = source_dir / "summary.json"
                summary_path.write_text("{}", encoding="utf-8")
                stat = source_path.stat()
                sources[recording_id] = SimpleNamespace(
                    data_paths={"CS": source_path},
                    summary={"experiment": "allDelay"},
                    summary_path=summary_path,
                    marker={"artifact_sha256": {"CS": sha256_file(source_path)}},
                    data_states={"CS": (stat.st_size, stat.st_mtime_ns)},
                )

            with patch(
                "classical_conditioning.analysis.legacy_scaled_vigor._verify_standard_main",
                side_effect=lambda _, recording_id: sources[recording_id],
            ):
                result = build_legacy_scaled_vigor_cohort(
                    project_dir,
                    [recording_id for recording_id, _ in recording_conditions],
                    analysis_id="cohort-a",
                )

            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                result.recording_ids,
                ("recording-a", "recording-b", "recording-c"),
            )
            self.assertEqual(
                summary["fish_counts"],
                {"control_CS": 1, "delay_CS": 2},
            )
            self.assertEqual(
                summary["pooling"]["order"],
                "recordings_then_exact_time_then_time_bin",
            )
            self.assertTrue(
                summary["pooling"]["performed_before_nonlinear_heatmap_scaling"]
            )
            self.assertTrue(
                summary["pooling"]["performed_separately_within_condition"]
            )
            self.assertEqual(len(result.artifact_paths), 12)


if __name__ == "__main__":
    unittest.main()
