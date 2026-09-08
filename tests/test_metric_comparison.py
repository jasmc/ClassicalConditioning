from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.metric_comparison import (
    OUTCOME_COLUMNS,
    RECIPE_ID,
    build_candidate_metric_comparison,
    summarize_candidate_metric_cohort,
    summarize_candidate_metric_windows,
)
from classical_conditioning.analysis.temporal_profiles import METRIC_IDS
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.cli import build_parser
from classical_conditioning.exceptions import SchemaValidationError


def temporal_fixture(recording_id: str) -> pd.DataFrame:
    rows = []
    for metric_index, metric_id in enumerate(METRIC_IDS.values()):
        for trial_number in (1, 2):
            for time, offset in ((-10.0, 0.0), (1.0, 1.0)):
                row = {
                    "Recording ID": recording_id,
                    "Trial type": "CS",
                    "Trial number": trial_number,
                    "Time bin center (s)": time,
                    "Metric ID": metric_id,
                    "Valid fraction": 0.9,
                    "Detector valid fraction": 0.8,
                    "Acquisition coverage": 1.0,
                }
                for column in OUTCOME_COLUMNS.values():
                    row[column] = metric_index + trial_number + offset
                rows.append(row)
    return pd.DataFrame(rows)


class MetricComparisonTests(unittest.TestCase):
    def test_identical_outcomes_are_summarized_for_all_five_metrics(self) -> None:
        profiles = pd.concat(
            [temporal_fixture("recording-a"), temporal_fixture("recording-b")],
            ignore_index=True,
        )

        recording = summarize_candidate_metric_windows(profiles)
        cohort = summarize_candidate_metric_cohort(recording)

        self.assertEqual(len(recording), 2 * len(METRIC_IDS) * len(OUTCOME_COLUMNS))
        self.assertEqual(len(cohort), len(METRIC_IDS) * len(OUTCOME_COLUMNS))
        self.assertEqual(set(recording["Metric ID"]), set(METRIC_IDS.values()))
        self.assertEqual(set(recording["Outcome ID"]), set(OUTCOME_COLUMNS))
        self.assertTrue(
            (recording["Response minus baseline"] == 1.0).all()
        )
        self.assertTrue((cohort["Recording count"] == 2).all())
        self.assertTrue((cohort["Cohort group"] == "all").all())
        self.assertTrue(
            (cohort["Contributing recordings Response minus baseline"] == 2).all()
        )

    def test_cohort_summary_splits_by_condition(self) -> None:
        recording = summarize_candidate_metric_windows(
            pd.concat(
                [temporal_fixture("recording-a"), temporal_fixture("recording-b")],
                ignore_index=True,
            )
        )
        recording["Condition ID"] = recording["Recording ID"].map(
            {"recording-a": "control", "recording-b": "fixedtrace"}
        )
        cohort = summarize_candidate_metric_cohort(recording)
        groups = set(cohort["Cohort group"])
        self.assertEqual(groups, {"all", "control", "fixedtrace"})
        self.assertTrue(
            (cohort.loc[cohort["Cohort group"] == "all", "Recording count"] == 2).all()
        )
        self.assertTrue(
            (
                cohort.loc[cohort["Cohort group"] == "control", "Recording count"]
                == 1
            ).all()
        )

    def test_difference_uses_only_trials_complete_in_both_windows(self) -> None:
        profiles = temporal_fixture("recording-a")
        incomplete_trial = (
            (profiles["Trial number"] == 2)
            & (profiles["Time bin center (s)"] > 0)
        )
        profiles.loc[
            incomplete_trial,
            list(OUTCOME_COLUMNS.values()),
        ] = float("nan")

        result = summarize_candidate_metric_windows(profiles)

        self.assertTrue((result["Baseline trial count"] == 2).all())
        self.assertTrue((result["Response trial count"] == 1).all())
        self.assertTrue((result["Complete trial count"] == 1).all())
        self.assertTrue((result["Response minus baseline"] == 1.0).all())

    def test_each_recording_must_contain_all_five_metrics(self) -> None:
        complete = temporal_fixture("recording-a")
        incomplete = temporal_fixture("recording-b")
        incomplete = incomplete.loc[
            incomplete["Metric ID"] != next(iter(METRIC_IDS.values()))
        ]

        with self.assertRaisesRegex(
            SchemaValidationError,
            "does not contain exactly the expected",
        ):
            summarize_candidate_metric_windows(
                pd.concat([complete, incomplete], ignore_index=True)
            )

    def test_builder_authenticates_input_and_blocks_selection_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            recording_id = "recording-a"
            processed_dir = project_dir / "Processed data" / recording_id
            quality_dir = project_dir / "Quality checks" / recording_id
            metadata_dir = project_dir / "Metadata"
            processed_dir.mkdir(parents=True)
            quality_dir.mkdir(parents=True)
            metadata_dir.mkdir(parents=True)

            profiles_path = (
                processed_dir / "candidate_temporal_outcomes-v3.parquet"
            )
            pq.write_table(
                pa.Table.from_pandas(
                    temporal_fixture(recording_id),
                    preserve_index=False,
                ),
                profiles_path,
                compression="zstd",
            )
            profiles_hash = sha256_file(profiles_path)
            summary_path = (
                quality_dir / "candidate-v3_temporal_outcomes_summary.json"
            )
            summary_path.write_text(
                json.dumps(
                    {
                        "recipe": "candidate-temporal-outcomes-v3",
                        "recording_id": recording_id,
                        "artifact": {"sha256": profiles_hash},
                    }
                ),
                encoding="utf-8",
            )
            marker_path = (
                metadata_dir
                / f"{recording_id}_candidate-temporal-outcomes-v3_complete.json"
            )
            marker_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "recipe": "candidate-temporal-outcomes-v3",
                        "recording_id": recording_id,
                        "profiles_sha256": profiles_hash,
                        "summary_sha256": sha256_file(summary_path),
                    }
                ),
                encoding="utf-8",
            )

            result = build_candidate_metric_comparison(
                project_dir,
                [recording_id],
                analysis_id="candidate-pilot",
            )

            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            marker = json.loads(
                result.completion_marker_path.read_text(encoding="utf-8")
            )
            self.assertFalse(summary["selection"]["performed"])
            self.assertFalse(summary["selection"]["paper_approved"])
            self.assertFalse(summary["inference"]["performed"])
            self.assertEqual(summary["metric_ids"], list(METRIC_IDS.values()))
            for name, path in result.artifact_paths.items():
                self.assertEqual(
                    sha256_file(path),
                    marker["artifact_sha256"][name],
                )

    def test_cli_exposes_candidate_metric_comparison(self) -> None:
        args = build_parser().parse_args(
            [
                "compare-candidate-metrics",
                "--project-dir",
                "paper",
                "--recording-id",
                "recording-a",
                "--analysis-id",
                "candidate-pilot",
            ]
        )

        self.assertEqual(args.recipe, RECIPE_ID)
        self.assertEqual(args.recording_id, ["recording-a"])


if __name__ == "__main__":
    unittest.main()
