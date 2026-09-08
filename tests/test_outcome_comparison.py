from __future__ import annotations

import unittest

import pandas as pd

from classical_conditioning.analysis.outcome_comparison import (
    RECIPE_ID,
    compare_legacy_candidate_outcomes,
)
from classical_conditioning.analysis.trial_outcomes import METRIC_IDS
from classical_conditioning.cli import build_parser
from classical_conditioning.exceptions import SchemaValidationError


class OutcomeComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.legacy = pd.DataFrame(
            {
                "Fish": ["fish-1", "fish-1"],
                "Trial type": ["CS", "CS"],
                "Trial number": [1, 2],
                "Mean 15 s before": [2.0, 4.0],
                "Mean CR": [4.0, 4.0],
                "Normalized vigor": [2.0, 1.0],
            }
        )
        rows = []
        for trial_number in (1, 2, 3):
            for metric_id in METRIC_IDS.values():
                rows.append(
                    {
                        "fish_id": "fish-1",
                        "alignment": "CS",
                        "trial_number": trial_number,
                        "metric_id": metric_id,
                        "baseline_total_activity": 2.0,
                        "response_total_activity": 6.0,
                    }
                )
        self.candidate = pd.DataFrame(rows)

    def test_reports_overlap_and_ratio_difference_for_every_metric(self) -> None:
        matched, coverage = compare_legacy_candidate_outcomes(
            self.legacy,
            self.candidate,
        )

        self.assertEqual(len(matched), 2 * len(METRIC_IDS))
        trial_one = matched[matched["trial_number"] == 1]
        self.assertTrue(
            (trial_one["candidate_response_baseline_ratio"] == 3.0).all()
        )
        self.assertTrue((trial_one["candidate_minus_legacy_ratio"] == 1.0).all())
        cs_coverage = coverage[coverage["alignment"] == "CS"]
        self.assertTrue((cs_coverage["legacy_trial_count"] == 2).all())
        self.assertTrue((cs_coverage["candidate_trial_count"] == 3).all())
        self.assertTrue((cs_coverage["overlap_trial_count"] == 2).all())
        self.assertTrue((cs_coverage["candidate_only_trial_count"] == 1).all())

    def test_rejects_duplicate_legacy_trial_identity(self) -> None:
        duplicated = pd.concat(
            [self.legacy, self.legacy.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaises(SchemaValidationError):
            compare_legacy_candidate_outcomes(duplicated, self.candidate)

    def test_matches_fish_identity_case_insensitively(self) -> None:
        candidate = self.candidate.copy()
        candidate["fish_id"] = candidate["fish_id"].str.upper()

        matched, _ = compare_legacy_candidate_outcomes(self.legacy, candidate)

        self.assertEqual(len(matched), 2 * len(METRIC_IDS))

    def test_rejects_nonempty_inputs_with_no_trial_overlap(self) -> None:
        candidate = self.candidate.copy()
        candidate["fish_id"] = "different-fish"

        with self.assertRaisesRegex(SchemaValidationError, "no overlapping"):
            compare_legacy_candidate_outcomes(self.legacy, candidate)

    def test_cli_exposes_versioned_comparison_recipe(self) -> None:
        args = build_parser().parse_args(
            [
                "legacy-candidate-outcome-comparison",
                "--project-dir",
                "paper",
                "--recording-id",
                "recording-a",
            ]
        )

        self.assertEqual(args.recipe, RECIPE_ID)
