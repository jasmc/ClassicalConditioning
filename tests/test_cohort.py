from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from classical_conditioning.artifacts import sha256_file
from classical_conditioning.cli import build_parser
from classical_conditioning.cohort import (
    apply_cohort,
    canonicalize_reviewed_cohort,
    freeze_cohort_manifest,
    load_cohort_manifest,
    logical_cohort_hash,
)
from classical_conditioning.exceptions import (
    ArtifactIntegrityError,
    SchemaValidationError,
    ScientificValidationError,
)


def reviewed_cohort() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": "allDelay",
                "recording_id": "recording-b",
                "fish_id": "fish-b",
                "condition_id": "control",
                "technical_valid": False,
                "technical_exclusion_reason": "protocol-incomplete",
                "behavioral_engagement": pd.NA,
                "behavioral_engagement_reason": pd.NA,
                "primary_included": False,
                "sensitivity_population_ids": [],
                "review_status": "approved",
                "reviewer": "reviewer-a",
                "reviewed_at": "2026-08-30T12:00:00+00:00",
                "source_qc_artifact_id": "qc-b",
            },
            {
                "experiment_id": "allDelay",
                "recording_id": "recording-a",
                "fish_id": "fish-a",
                "condition_id": "delay",
                "technical_valid": True,
                "technical_exclusion_reason": pd.NA,
                "behavioral_engagement": True,
                "behavioral_engagement_reason": "descriptive-only",
                "primary_included": True,
                "sensitivity_population_ids": ["strict", "all-technical"],
                "review_status": "approved",
                "reviewer": "reviewer-a",
                "reviewed_at": "2026-08-30T12:00:00Z",
                "source_qc_artifact_id": "qc-a",
            },
        ]
    )


class CohortManifestTests(unittest.TestCase):
    def test_canonical_hash_is_row_and_population_order_invariant(self) -> None:
        first = reviewed_cohort()
        second = first.iloc[::-1].reset_index(drop=True)
        fish_a_index = second.index[second["fish_id"] == "fish-a"][0]
        second.at[fish_a_index, "sensitivity_population_ids"] = [
            "all-technical",
            "strict",
        ]

        self.assertEqual(logical_cohort_hash(first), logical_cohort_hash(second))
        canonical = canonicalize_reviewed_cohort(first)
        self.assertEqual(list(canonical["recording_id"]), ["recording-b", "recording-a"])

    def test_freeze_requires_review_and_consistent_technical_inclusion(self) -> None:
        unreviewed = reviewed_cohort()
        unreviewed.loc[0, "review_status"] = "pending"
        with self.assertRaisesRegex(ScientificValidationError, "approved"):
            canonicalize_reviewed_cohort(unreviewed)

        invalid_primary = reviewed_cohort()
        invalid_primary.loc[0, "primary_included"] = True
        with self.assertRaisesRegex(ScientificValidationError, "Technically invalid"):
            canonicalize_reviewed_cohort(invalid_primary)

        empty = reviewed_cohort().iloc[0:0]
        with self.assertRaisesRegex(ScientificValidationError, "at least one"):
            canonicalize_reviewed_cohort(empty)

        no_primary = reviewed_cohort()
        no_primary["primary_included"] = False
        with self.assertRaisesRegex(ScientificValidationError, "include at least one"):
            canonicalize_reviewed_cohort(no_primary)

        naive_time = reviewed_cohort()
        naive_time["reviewed_at"] = "2026-08-30T12:00:00"
        with self.assertRaisesRegex(SchemaValidationError, "explicit timezone"):
            canonicalize_reviewed_cohort(naive_time)

    def test_apply_cohort_filters_excluded_fish_and_rejects_unknowns(self) -> None:
        data = pd.DataFrame(
            {
                "experiment_id": ["allDelay", "allDelay"],
                "fish_id": ["fish-a", "fish-b"],
                "value": [1.0, 2.0],
            }
        )

        included = apply_cohort(data, reviewed_cohort())

        self.assertEqual(list(included["fish_id"]), ["fish-a"])
        unknown = data.copy()
        unknown.loc[1, "fish_id"] = "fish-unknown"
        with self.assertRaisesRegex(SchemaValidationError, "absent"):
            apply_cohort(unknown, reviewed_cohort())
        with self.assertRaisesRegex(SchemaValidationError, "must be boolean"):
            apply_cohort(
                data,
                reviewed_cohort(),
                include_column="sensitivity_population_ids",
            )

    def test_freeze_publishes_immutable_authenticated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)

            result = freeze_cohort_manifest(
                project_dir,
                reviewed_cohort(),
                cohort_id="primary-v1",
                policy_id="technical-policy-v1",
            )

            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            marker = json.loads(
                result.completion_marker_path.read_text(encoding="utf-8")
            )
            self.assertEqual(summary["logical_content_sha256"], result.logical_content_sha256)
            self.assertEqual(sha256_file(result.manifest_path), marker["manifest_sha256"])
            self.assertEqual(result.row_count, 2)
            self.assertEqual(result.primary_count, 1)
            loaded = load_cohort_manifest(project_dir, "primary-v1")
            self.assertEqual(logical_cohort_hash(loaded), result.logical_content_sha256)
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                freeze_cohort_manifest(
                    project_dir,
                    reviewed_cohort(),
                    cohort_id="primary-v1",
                    policy_id="technical-policy-v1",
                )

            result.review_copy_path.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactIntegrityError, "byte lineage"):
                load_cohort_manifest(project_dir, "primary-v1")

    def test_cli_exposes_reviewed_cohort_freeze(self) -> None:
        args = build_parser().parse_args(
            [
                "freeze-cohort",
                "--project-dir",
                "paper",
                "--input",
                "reviewed.parquet",
                "--cohort-id",
                "primary-v1",
                "--policy-id",
                "technical-policy-v1",
            ]
        )

        self.assertEqual(args.recipe, "cohort-manifest-v1")
        self.assertEqual(args.cohort_id, "primary-v1")

    def test_cli_exposes_apply_cohort(self) -> None:
        args = build_parser().parse_args(
            [
                "apply-cohort",
                "--project-dir",
                "paper",
                "--cohort-id",
                "fixture-two-fish-v1",
                "--input",
                "data.parquet",
                "--output",
                "filtered.parquet",
            ]
        )
        self.assertEqual(args.include_column, "primary_included")
        self.assertEqual(args.cohort_id, "fixture-two-fish-v1")


if __name__ == "__main__":
    unittest.main()
