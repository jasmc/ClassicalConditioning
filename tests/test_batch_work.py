from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from classical_conditioning.analysis.batch_work import (
    RECIPE_ID,
    execute_batch_work,
    plan_batch_work,
    write_batch_work_manifest,
)
from classical_conditioning.cli import build_parser
from unittest.mock import patch


class BatchWorkTests(unittest.TestCase):
    def test_plan_marks_complete_and_pending_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            recording_id = "recording-a"
            metadata = project_dir / "Metadata"
            processed = project_dir / "Processed data" / recording_id
            metadata.mkdir(parents=True)
            processed.mkdir(parents=True)

            (
                metadata / f"{recording_id}_corrected-preprocess-v1_complete.json"
            ).write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "recipe": "corrected-preprocess-v1",
                        "recording_id": recording_id,
                    }
                ),
                encoding="utf-8",
            )
            (processed / "frame_preprocessed_corrected-v1.parquet").write_bytes(b"x")

            work = plan_batch_work(
                project_dir,
                [recording_id],
                metric_recipe="tail-candidate-corrected-v1",
            )
            statuses = dict(zip(work["stage"], work["status"]))
            self.assertEqual(statuses["corrected-preprocess"], "complete")
            self.assertEqual(statuses["activity-metrics"], "pending")
            self.assertEqual(len(work), 5)

            pending = plan_batch_work(
                project_dir,
                [recording_id],
                metric_recipe="tail-candidate-corrected-v1",
                selection="pending",
            )
            self.assertEqual(set(pending["stage"]), set(statuses) - {"corrected-preprocess"})

    def test_write_batch_manifest_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            result = write_batch_work_manifest(
                project_dir,
                ["recording-a", "recording-b"],
                batch_id="fixture-batch",
                metric_recipe="tail-candidate-corrected-v1",
            )
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["recipe"], RECIPE_ID)
            self.assertEqual(result.row_count, 10)
            self.assertEqual(result.pending_count, 10)
            self.assertFalse(summary["paper_approved"])

        args = build_parser().parse_args(
            [
                "plan-batch",
                "--project-dir",
                "paper",
                "--recording-id",
                "recording-a",
                "--batch-id",
                "fixture-batch",
            ]
        )
        self.assertEqual(args.metric_recipe, "tail-candidate-corrected-v1")
        self.assertEqual(args.selection, "all")

    @patch(
        "classical_conditioning.analysis.candidate_runner.run_candidate_development_pipeline"
    )
    def test_execute_pending_calls_runner_once(self, mock_runner) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            mock_runner.return_value.manifest_path = (
                project_dir / "Metadata" / "run_manifest.json"
            )
            result = execute_batch_work(
                project_dir,
                ["recording-a"],
                batch_id="fixture-batch",
                selection="pending",
                analysis_id="fixture-run",
            )
            mock_runner.assert_called_once()
            self.assertEqual(result.before_pending_count, 5)
            self.assertEqual(result.after_pending_count, 5)
            self.assertTrue(result.refreshed_manifest_path.is_file())

        args = build_parser().parse_args(
            [
                "execute-batch",
                "--project-dir",
                "paper",
                "--recording-id",
                "recording-a",
                "--batch-id",
                "fixture-batch",
                "--selection",
                "failed",
            ]
        )
        self.assertEqual(args.selection, "failed")
        self.assertFalse(args.no_overwrite_failed)


if __name__ == "__main__":
    unittest.main()
