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

from classical_conditioning.analysis.legacy_normalized_vigor import (
    BASELINE_COLUMN,
    NORMALIZED_COLUMN,
    RECIPE_ID as NORMALIZED_RECIPE_ID,
    RESPONSE_COLUMN,
)
from classical_conditioning.analysis.legacy_statistics import (
    CONFIG_SHA256,
    LegacyStatisticsConfig,
    build_legacy_statistics,
    build_legacy_block_medians,
    fit_legacy_mixed_model,
    prepare_legacy_model_input,
    run_legacy_nonparametric_tests,
)
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.cli import build_parser
from classical_conditioning.exceptions import (
    ConfigurationError,
    SchemaValidationError,
)


def outcome_rows(
    fish: str,
    condition: str,
    block: str,
    trial_numbers: list[int],
    value: float,
) -> list[dict[str, object]]:
    return [
        {
            "Strain": "strain",
            "Age (dpf)": "6",
            "Exp.": condition,
            "ProtocolRig": "rig",
            "Day": fish.split("_")[0],
            "Fish no.": fish.split("_")[1],
            "Fish": fish,
            "Block name": block,
            "Trial number": trial,
            BASELINE_COLUMN: 2.0,
            RESPONSE_COLUMN: 2.0 * value,
            NORMALIZED_COLUMN: value,
            "Trial type": "CS",
        }
        for trial in trial_numbers
    ]


class LegacyStatisticsTests(unittest.TestCase):
    def _write_normalized_sources(
        self,
        project_dir: Path,
        recording_id: str,
    ) -> None:
        processed_dir = project_dir / "Processed data" / recording_id
        quality_dir = project_dir / "Quality checks" / recording_id
        metadata_dir = project_dir / "Metadata"
        processed_dir.mkdir(parents=True)
        quality_dir.mkdir(parents=True)
        metadata_dir.mkdir()
        frames = {
            "CS": pd.DataFrame(
                outcome_rows(
                    "day_A",
                    "delay",
                    "Pre-train",
                    list(range(5, 11)),
                    1.0,
                )
            ),
            "US": pd.DataFrame(
                outcome_rows(
                    "day_A",
                    "delay",
                    "Pre-train",
                    list(range(18, 24)),
                    1.0,
                )
            ),
        }
        paths = {
            alignment: processed_dir
            / f"{NORMALIZED_RECIPE_ID}_{alignment}.parquet"
            for alignment in frames
        }
        hashes = {}
        for alignment, frame in frames.items():
            pq.write_table(
                pa.Table.from_pandas(frame, preserve_index=False),
                paths[alignment],
                compression="zstd",
            )
            hashes[alignment] = sha256_file(paths[alignment])
        summary_path = quality_dir / f"{NORMALIZED_RECIPE_ID}_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "recipe": NORMALIZED_RECIPE_ID,
                    "recording_id": recording_id,
                    "artifacts": {
                        alignment: {
                            "path": str(path),
                            "sha256": hashes[alignment],
                        }
                        for alignment, path in paths.items()
                    },
                }
            ),
            encoding="utf-8",
        )
        marker_path = (
            metadata_dir
            / f"{recording_id}_{NORMALIZED_RECIPE_ID}_complete.json"
        )
        marker_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "recipe": NORMALIZED_RECIPE_ID,
                    "recording_id": recording_id,
                    "artifact_sha256": hashes,
                    "summary_sha256": sha256_file(summary_path),
                }
            ),
            encoding="utf-8",
        )

    def test_model_input_preserves_any_one_block_threshold_bug(self) -> None:
        rows = []
        rows.extend(
            outcome_rows(
                "day_A",
                "delay",
                "Block A",
                list(range(1, 7)),
                1.0,
            )
        )
        rows.extend(outcome_rows("day_A", "delay", "Block B", [7], 1.1))
        rows.extend(
            outcome_rows(
                "day_B",
                "control",
                "Block A",
                list(range(1, 6)),
                1.0,
            )
        )
        rows.extend(
            outcome_rows(
                "day_B",
                "control",
                "Block B",
                list(range(7, 12)),
                1.0,
            )
        )

        result = prepare_legacy_model_input(pd.DataFrame(rows))

        self.assertEqual(set(result["Fish_ID"]), {"day_A"})
        self.assertEqual(int((result["Block_name"] == "Block B").sum()), 1)

    def test_block_medians_use_frozen_selected_five_trial_blocks(self) -> None:
        rows = []
        for fish, offset in (("day_A", 0.0), ("day_B", 0.2)):
            rows.extend(
                outcome_rows(fish, "delay", "Pre-train", [5, 6], 1.0 + offset)
            )
            rows.extend(
                outcome_rows(fish, "delay", "Test 1", [65, 66], 1.2 + offset)
            )
            rows.extend(
                outcome_rows(fish, "delay", "Test 3", [90, 91], 1.4 + offset)
            )

        result = build_legacy_block_medians(
            pd.DataFrame(rows),
            alignment="CS",
        )

        self.assertEqual(
            set(result["Block name"]),
            {"Early Pre-Train", "Early Test", "Late Test"},
        )
        self.assertEqual(len(result), 6)

    def test_nonparametric_routes_and_holm_families_are_explicit(self) -> None:
        rows = []
        for condition, condition_offset in (("control", 0.0), ("delay", 0.5)):
            for fish_index in range(4):
                for block_index, block in enumerate(
                    ("Early Pre-Train", "Early Test", "Late Test")
                ):
                    rows.append(
                        {
                            "Fish": f"{condition}_{fish_index}",
                            "Block name": block,
                            "Exp.": condition,
                            NORMALIZED_COLUMN: (
                                1.0
                                + condition_offset
                                + block_index * 0.1
                                + fish_index * 0.01
                            ),
                        }
                    )

        result = run_legacy_nonparametric_tests(
            pd.DataFrame(rows),
            condition_order=("control", "delay"),
        )

        self.assertEqual(
            set(result["route"]),
            {
                "block-line-within-mann-whitney",
                "block-box-between-mann-whitney",
                "block-box-within-wilcoxon",
            },
        )
        self.assertTrue(result["p_adjusted"].notna().all())
        self.assertTrue(
            result["correction_family"].str.contains(
                "mann-whitney|wilcoxon"
            ).all()
        )

    def test_line_route_compares_adjacent_surviving_blocks(self) -> None:
        rows = [
            {
                "Fish": f"fish-{index}",
                "Block name": block,
                "Exp.": "delay",
                NORMALIZED_COLUMN: value + index * 0.01,
            }
            for index in range(3)
            for block, value in (
                ("Early Pre-Train", 1.0),
                ("Late Test", 1.5),
            )
        ]

        result = run_legacy_nonparametric_tests(
            pd.DataFrame(rows),
            condition_order=("delay",),
        )
        line_result = result.loc[
            result["route"] == "block-line-within-mann-whitney"
        ]

        self.assertEqual(len(line_result), 1)
        self.assertEqual(line_result.iloc[0]["block_left"], "Early Pre-Train")
        self.assertEqual(line_result.iloc[0]["block_right"], "Late Test")

    def test_mixed_model_failures_are_returned_as_strings(self) -> None:
        frame = pd.DataFrame({"Fish_ID": ["fish"], "value": [1.0]})
        with patch(
            "statsmodels.formula.api.mixedlm",
            side_effect=RuntimeError("fit failed"),
        ):
            result, error = fit_legacy_mixed_model(
                frame,
                "value ~ 1",
                "Fish_ID",
            )

        self.assertIsNone(result)
        self.assertEqual(error, "fit failed")

    def test_frozen_config_rejects_corrected_inference_settings(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "fish exclusion"):
            LegacyStatisticsConfig(apply_fish_discard=True)
        with self.assertRaisesRegex(ConfigurationError, "Holm"):
            LegacyStatisticsConfig(nonparametric_correction="fdr_bh")
        with self.assertRaisesRegex(ConfigurationError, "bootstrap"):
            LegacyStatisticsConfig(trajectory_bootstrap_resamples=1000)

    def test_builder_publishes_authenticated_tables_and_failure_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            self._write_normalized_sources(project_dir, "recording")

            result = build_legacy_statistics(
                project_dir,
                ["recording"],
                analysis_id="synthetic",
            )

            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            marker = json.loads(
                result.completion_marker_path.read_text(encoding="utf-8")
            )
            self.assertEqual(summary["config_sha256"], CONFIG_SHA256)
            self.assertFalse(summary["validation"]["cohort_scale_validated"])
            self.assertEqual(summary["validation"]["recording_count"], 1)
            self.assertEqual(summary["validation"]["condition_count"], 1)
            self.assertEqual(result.model_errors[0]["Type"], "Global")
            for name, path in result.artifact_paths.items():
                self.assertEqual(
                    sha256_file(path),
                    marker["artifact_sha256"][name],
                )
            with self.assertRaises(FileExistsError):
                build_legacy_statistics(
                    project_dir,
                    ["recording"],
                    analysis_id="synthetic",
                )

    def test_cli_accepts_explicit_recording_list(self) -> None:
        args = build_parser().parse_args(
            [
                "legacy-statistics",
                "--project-dir",
                "paper",
                "--recording-id",
                "fish-a",
                "--recording-id",
                "fish-b",
                "--analysis-id",
                "cohort-a",
            ]
        )

        self.assertEqual(args.recording_id, ["fish-a", "fish-b"])
        self.assertEqual(args.recipe, "legacy-statistics-v1")

    def test_builder_rejects_recording_without_requested_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            recording_id = "recording"
            self._write_normalized_sources(project_dir, recording_id)
            summary_path = (
                project_dir
                / "Quality checks"
                / recording_id
                / f"{NORMALIZED_RECIPE_ID}_summary.json"
            )
            marker_path = (
                project_dir
                / "Metadata"
                / f"{recording_id}_{NORMALIZED_RECIPE_ID}_complete.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            del summary["artifacts"]["CS"]
            del marker["artifact_sha256"]["CS"]
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            marker["summary_sha256"] = sha256_file(summary_path)
            marker_path.write_text(json.dumps(marker), encoding="utf-8")

            with self.assertRaisesRegex(
                SchemaValidationError,
                "absent for recordings",
            ):
                build_legacy_statistics(
                    project_dir,
                    [recording_id],
                    analysis_id="synthetic",
                )


if __name__ == "__main__":
    unittest.main()
