from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from classical_conditioning.analysis.candidate_runner import (
    CORRECTED_RUNNER_RECIPE_ID,
    RUNNER_RECIPE_ID,
    _verify_movement,
    run_candidate_development_pipeline,
)
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.cli import build_parser
from classical_conditioning.exceptions import ArtifactIntegrityError


class CandidateRunnerTests(unittest.TestCase):
    @patch(
        "classical_conditioning.analysis.candidate_runner._verify_comparison",
        return_value="comparison-marker",
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner._verify_temporal",
        return_value="temporal-marker",
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner._verify_trial_outcomes",
        return_value="trial-marker",
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner._verify_movement",
        return_value="movement-marker",
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner._verify_metrics",
        return_value="metric-marker",
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner.build_candidate_metric_comparison"
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner.build_candidate_temporal_profiles"
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner.build_candidate_trial_outcomes"
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner.build_candidate_movement_state"
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner.build_candidate_activity_metrics"
    )
    def test_runs_each_recording_then_one_cohort_comparison(
        self,
        mock_metrics,
        mock_movement,
        mock_trial,
        mock_temporal,
        mock_comparison,
        mock_verify_metrics,
        mock_verify_movement,
        mock_verify_trial,
        mock_verify_temporal,
        mock_verify_comparison,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)

            result = run_candidate_development_pipeline(
                project_dir,
                ["recording-a", "recording-b"],
                analysis_id="candidate-a",
            )

            self.assertEqual(mock_metrics.call_count, 2)
            self.assertEqual(mock_movement.call_count, 2)
            self.assertEqual(mock_temporal.call_count, 2)
            self.assertEqual(mock_trial.call_count, 2)
            self.assertEqual(mock_comparison.call_count, 1)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["paper_approved"])
            self.assertEqual(
                manifest["recording_ids"],
                ["recording-a", "recording-b"],
            )
            self.assertEqual(len(manifest["blocked_next_steps"]), 5)

    @patch(
        "classical_conditioning.analysis.candidate_runner._verify_comparison",
        return_value="comparison-marker",
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner._verify_temporal",
        return_value="temporal-marker",
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner._verify_trial_outcomes",
        return_value="trial-marker",
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner._verify_movement",
        return_value="movement-marker",
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner._verify_metrics",
        return_value="metric-marker",
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner.build_candidate_metric_comparison"
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner.build_candidate_temporal_profiles"
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner.build_candidate_trial_outcomes"
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner.build_candidate_movement_state"
    )
    @patch(
        "classical_conditioning.analysis.candidate_runner.build_candidate_activity_metrics"
    )
    def test_resume_verifies_markers_without_rebuilding(
        self,
        mock_metrics,
        mock_movement,
        mock_trial,
        mock_temporal,
        mock_comparison,
        mock_verify_metrics,
        mock_verify_movement,
        mock_verify_trial,
        mock_verify_temporal,
        mock_verify_comparison,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            metadata = project_dir / "Metadata"
            metadata.mkdir()
            for name in (
                "recording-a_candidate-v1_complete.json",
                "recording-a_movement-candidate-v1_complete.json",
                "recording-a_candidate-temporal-outcomes-v2_complete.json",
                "recording-a_candidate-trial-outcomes-v1_complete.json",
                "candidate-a_candidate-metric-comparison-v1_complete.json",
            ):
                (metadata / name).write_text("{}", encoding="utf-8")

            result = run_candidate_development_pipeline(
                project_dir,
                ["recording-a"],
                analysis_id="candidate-a",
            )

            self.assertEqual(mock_metrics.call_count, 0)
            self.assertEqual(mock_movement.call_count, 0)
            self.assertEqual(mock_temporal.call_count, 0)
            self.assertEqual(mock_trial.call_count, 0)
            self.assertEqual(mock_comparison.call_count, 0)
            self.assertEqual(
                result.step_status["recording-a"],
                {
                    "activity-metrics": "existing",
                    "movement-state": "existing",
                    "temporal-profiles": "existing",
                    "trial-outcomes": "existing",
                },
            )

    def test_cli_exposes_candidate_runner(self) -> None:
        args = build_parser().parse_args(
            [
                "candidate-runner",
                "--project-dir",
                "paper",
                "--recording-id",
                "recording-a",
                "--analysis-id",
                "candidate-a",
            ]
        )

        self.assertEqual(args.recipe, RUNNER_RECIPE_ID)
        self.assertEqual(args.recording_id, ["recording-a"])

    def test_cli_exposes_corrected_candidate_runner(self) -> None:
        args = build_parser().parse_args(
            [
                "candidate-runner",
                "--project-dir",
                "paper",
                "--recording-id",
                "recording-a",
                "--analysis-id",
                "candidate-corrected",
                "--recipe",
                "candidate-corrected-runner-v1",
            ]
        )

        self.assertEqual(args.recipe, CORRECTED_RUNNER_RECIPE_ID)

    def test_resume_rejects_stale_cross_stage_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            recording_id = "recording-a"
            processed = project_dir / "Processed data" / recording_id
            quality = project_dir / "Quality checks" / recording_id
            metadata = project_dir / "Metadata"
            processed.mkdir(parents=True)
            quality.mkdir(parents=True)
            metadata.mkdir(parents=True)

            movement_path = processed / "movement_state_candidates-v1.parquet"
            movement_path.write_bytes(b"movement")
            movement_hash = sha256_file(movement_path)
            movement_summary = quality / "movement-candidate-v1_summary.json"
            movement_summary.write_text(
                json.dumps(
                    {
                        "recipe": "movement-candidate-v1",
                        "recording_id": recording_id,
                        "artifact": {"sha256": movement_hash},
                        "inputs": {
                            "candidate_metrics": {"sha256": "stale-metric"}
                        },
                    }
                ),
                encoding="utf-8",
            )
            (
                metadata / f"{recording_id}_movement-candidate-v1_complete.json"
            ).write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "recipe": "movement-candidate-v1",
                        "recording_id": recording_id,
                        "movement_sha256": movement_hash,
                        "summary_sha256": sha256_file(movement_summary),
                    }
                ),
                encoding="utf-8",
            )
            (
                metadata / f"{recording_id}_candidate-v1_complete.json"
            ).write_text(
                json.dumps({"metrics_sha256": "current-metric"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ArtifactIntegrityError, "stale"):
                _verify_movement(project_dir, recording_id)


if __name__ == "__main__":
    unittest.main()
