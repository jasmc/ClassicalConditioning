from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from classical_conditioning.analysis.legacy_runner import (
    RUNNER_RECIPE_ID,
    run_legacy_analysis_pipeline,
)
from classical_conditioning.cli import build_parser


class LegacyRunnerTests(unittest.TestCase):
    @patch("classical_conditioning.analysis.legacy_runner.sha256_file", return_value="digest")
    @patch("classical_conditioning.analysis.legacy_runner._verify_statistics")
    @patch("classical_conditioning.analysis.legacy_runner._verify_scaled_vigor")
    @patch("classical_conditioning.analysis.legacy_runner._verify_recording_stage")
    @patch("classical_conditioning.analysis.legacy_runner.build_legacy_statistics")
    @patch("classical_conditioning.analysis.legacy_runner.build_legacy_normalized_vigor")
    @patch("classical_conditioning.analysis.legacy_runner.build_legacy_scaled_vigor_cohort")
    @patch("classical_conditioning.analysis.legacy_runner.build_legacy_standard_main")
    def test_pipeline_runs_each_frozen_stage_and_writes_manifest(
        self,
        mock_standard_main,
        mock_scaled_vigor,
        mock_normalized_vigor,
        mock_statistics,
        mock_verify_recording,
        mock_verify_scaled,
        mock_verify_statistics,
        mock_sha256,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            (project_dir / "Metadata").mkdir(parents=True)

            result = run_legacy_analysis_pipeline(
                project_dir,
                ["recording-a", "recording-b"],
                analysis_id="cohort-a",
            )

            self.assertEqual(mock_standard_main.call_count, 2)
            self.assertEqual(mock_scaled_vigor.call_count, 1)
            self.assertEqual(mock_normalized_vigor.call_count, 2)
            self.assertEqual(mock_statistics.call_count, 1)
            self.assertEqual(mock_verify_recording.call_count, 4)
            self.assertEqual(mock_verify_scaled.call_count, 1)
            self.assertEqual(mock_verify_statistics.call_count, 1)
            self.assertEqual(result.recording_ids, ("recording-a", "recording-b"))

            manifest_path = (
                project_dir
                / "Metadata"
                / f"cohort-a_{RUNNER_RECIPE_ID}_manifest.json"
            )
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["recording_ids"], ["recording-a", "recording-b"])
            self.assertEqual(manifest["status"], "complete")

    @patch("classical_conditioning.analysis.legacy_runner.sha256_file", return_value="digest")
    @patch("classical_conditioning.analysis.legacy_runner._verify_statistics")
    @patch("classical_conditioning.analysis.legacy_runner._verify_scaled_vigor")
    @patch("classical_conditioning.analysis.legacy_runner._verify_recording_stage")
    @patch("classical_conditioning.analysis.legacy_runner.build_legacy_statistics")
    @patch("classical_conditioning.analysis.legacy_runner.build_legacy_normalized_vigor")
    @patch("classical_conditioning.analysis.legacy_runner.build_legacy_scaled_vigor_cohort")
    @patch("classical_conditioning.analysis.legacy_runner.build_legacy_standard_main")
    def test_pipeline_skips_existing_stage_markers(
        self,
        mock_standard_main,
        mock_scaled_vigor,
        mock_normalized_vigor,
        mock_statistics,
        mock_verify_recording,
        mock_verify_scaled,
        mock_verify_statistics,
        mock_sha256,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            metadata_dir = project_dir / "Metadata"
            metadata_dir.mkdir(parents=True)

            for recording_id in ("recording-a", "recording-b"):
                for recipe_id in (
                    "legacy-standard-main-v1",
                    "legacy-normalized-vigor-v1",
                ):
                    (metadata_dir / f"{recording_id}_{recipe_id}_complete.json").write_text(
                        "{}",
                        encoding="utf-8",
                    )
            (
                metadata_dir / "cohort-a_legacy-scaled-vigor-v1_complete.json"
            ).write_text("{}", encoding="utf-8")

            run_legacy_analysis_pipeline(
                project_dir,
                ["recording-a", "recording-b"],
                analysis_id="cohort-a",
            )

            self.assertEqual(mock_standard_main.call_count, 0)
            self.assertEqual(mock_scaled_vigor.call_count, 0)
            self.assertEqual(mock_normalized_vigor.call_count, 0)
            self.assertEqual(mock_statistics.call_count, 1)
            self.assertEqual(mock_verify_recording.call_count, 4)
            self.assertEqual(mock_verify_scaled.call_count, 1)
            self.assertEqual(mock_verify_statistics.call_count, 1)

    def test_cli_accepts_runner_command(self) -> None:
        args = build_parser().parse_args(
            [
                "legacy-runner",
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
        self.assertEqual(args.recipe, "legacy-runner-v1")
        self.assertFalse(args.skip_statistics)


if __name__ == "__main__":
    unittest.main()
