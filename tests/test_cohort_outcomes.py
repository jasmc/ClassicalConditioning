from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.cohort_outcomes import (
    build_analysis_eligibility,
    build_cohort_trial_outcomes,
    load_cohort_trial_outcomes,
)
from classical_conditioning.analysis.movement_state import (
    resolve_candidate_metric_source,
)
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.cohort import freeze_cohort_manifest


METRIC = "tail_length_weighted_angular_l1"


def reviewed_cohort() -> pd.DataFrame:
    common = {
        "behavioral_engagement": pd.NA,
        "behavioral_engagement_reason": pd.NA,
        "sensitivity_population_ids": [],
        "review_status": "approved",
        "reviewer": "reviewer",
        "reviewed_at": "2026-09-17T12:00:00Z",
    }
    return pd.DataFrame(
        [
            {
                **common,
                "experiment_id": "allDelay",
                "recording_id": "recording-a",
                "fish_id": "fish-a",
                "condition_id": "delay",
                "technical_valid": True,
                "technical_exclusion_reason": pd.NA,
                "primary_included": True,
                "source_qc_artifact_id": "qc-a",
            },
            {
                **common,
                "experiment_id": "allDelay",
                "recording_id": "recording-b",
                "fish_id": "fish-b",
                "condition_id": "control",
                "technical_valid": False,
                "technical_exclusion_reason": "tracking-failed",
                "primary_included": False,
                "source_qc_artifact_id": "qc-b",
            },
        ]
    )


def write_authenticated_trial_outcomes(
    project_dir: Path,
    recording_id: str,
    outcomes: pd.DataFrame,
) -> None:
    route = resolve_candidate_metric_source(
        metric_recipe="tail-candidate-corrected"
    )
    outcomes_path = (
        project_dir / "Processed data" / recording_id / route.trial_outcomes_name
    )
    summary_path = (
        project_dir / "Quality checks" / recording_id / route.trial_summary_name
    )
    marker_path = (
        project_dir / "Metadata" / f"{recording_id}_{route.trial_marker_suffix}"
    )
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(outcomes), outcomes_path)
    digest = sha256_file(outcomes_path)
    summary = {
        "recipe": route.trial_recipe,
        "recording_id": recording_id,
        "artifacts": {"outcomes": {"sha256": digest}},
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    marker = {
        "status": "complete",
        "recipe": route.trial_recipe,
        "recording_id": recording_id,
        "artifact_sha256": {"outcomes": digest},
        "summary_sha256": sha256_file(summary_path),
    }
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def cohort_outcomes_fixture() -> pd.DataFrame:
    rows = []
    for trial in range(5, 9):
        rows.append(
            {
                "cohort_id": "paper",
                "cohort_hash": "abc",
                "experiment_id": "allDelay",
                "recording_id": "recording-a",
                "fish_id": "fish-a",
                "condition_id": "delay",
                "trial_id": f"fish-a:CS:{trial:03d}",
                "alignment": "CS",
                "trial_number": trial,
                "block_10_name": "Pre-train",
                "metric_id": METRIC,
                "baseline_valid_sample_count": 10,
                "response_valid_sample_count": 10,
                "baseline_total_activity": 2.0,
                "response_total_activity": 1.0,
                "baseline_conditional_intensity": 2.0,
                "conditional_intensity": 1.0,
            }
        )
    return pd.DataFrame(rows)


class AnalysisEligibilityTests(unittest.TestCase):
    def test_cohort_boundary_excludes_fish_once_and_preserves_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            freeze_cohort_manifest(
                project_dir,
                reviewed_cohort(),
                cohort_id="paper",
                policy_id="technical-policy",
            )
            outcomes = cohort_outcomes_fixture().drop(
                columns=["cohort_id", "cohort_hash"]
            )
            write_authenticated_trial_outcomes(
                project_dir, "recording-a", outcomes
            )

            result = build_cohort_trial_outcomes(
                project_dir,
                cohort_id="paper",
            )
            loaded, summary = load_cohort_trial_outcomes(project_dir, "paper")
            flow = pq.read_table(result.sample_flow_path).to_pandas()

            self.assertEqual(set(loaded["fish_id"]), {"fish-a"})
            self.assertEqual(summary["fish_count"], 1)
            self.assertEqual(set(flow["fish_id"]), {"fish-a", "fish-b"})
            excluded = flow.loc[flow["fish_id"] == "fish-b"].iloc[0]
            self.assertEqual(excluded["disposition"], "excluded_without_outcomes")

    def test_eligibility_preserves_rows_and_names_reasons(self) -> None:
        outcomes = cohort_outcomes_fixture()
        outcomes.loc[0, "baseline_total_activity"] = 0.0
        outcomes.loc[1, "response_total_activity"] = np.nan
        outcomes.loc[2, "baseline_valid_sample_count"] = 0

        result = build_analysis_eligibility(
            outcomes,
            metric_id=METRIC,
            outcome_id="total-activity",
        )

        self.assertEqual(len(result), len(outcomes))
        self.assertEqual(
            result.loc[0, "ineligible_reason"], "nonpositive_ratio_baseline"
        )
        self.assertEqual(result.loc[1, "ineligible_reason"], "nonfinite_response")
        self.assertEqual(result.loc[2, "ineligible_reason"], "missing_baseline_window")
        self.assertTrue(result.loc[3, "eligible"])

    def test_conditional_intensity_has_named_no_bout_reason(self) -> None:
        outcomes = cohort_outcomes_fixture()
        outcomes.loc[0, "conditional_intensity"] = np.nan

        result = build_analysis_eligibility(
            outcomes,
            metric_id=METRIC,
            outcome_id="conditional-intensity",
        )

        self.assertEqual(
            result.loc[0, "ineligible_reason"],
            "conditional_intensity_undefined_no_bout",
        )

    def test_conditional_intensity_rejects_nonpositive_values(self) -> None:
        outcomes = cohort_outcomes_fixture()
        outcomes.loc[0, "conditional_intensity"] = 0.0

        result = build_analysis_eligibility(
            outcomes,
            metric_id=METRIC,
            outcome_id="conditional-intensity",
        )

        self.assertEqual(
            result.loc[0, "ineligible_reason"],
            "nonpositive_conditional_intensity",
        )


if __name__ == "__main__":
    unittest.main()
