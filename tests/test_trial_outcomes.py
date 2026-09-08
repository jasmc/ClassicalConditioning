from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from classical_conditioning.analysis.trial_outcomes import (
    METRIC_IDS,
    TrialOutcomeConfig,
    aggregate_trial_outcomes,
    verify_candidate_trial_outcomes,
)
from classical_conditioning.cli import build_parser
from classical_conditioning.exceptions import (
    ArtifactIntegrityError,
    SchemaValidationError,
)


class TrialOutcomeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = pd.DataFrame(
            {
                "FrameID": [0, 1, 2, 3, 4],
                "AbsoluteTime": [0, 500, 1_000, 1_500, 2_000],
                "FrameStep": [1] * 5,
                "DeltaTimeMs": [500.0] * 5,
                **{
                    column: [2.0, np.nan, 0.0, 4.0, 8.0]
                    for column in METRIC_IDS
                },
            }
        )
        self.movement = pd.DataFrame(
            {
                "FrameID": [0, 1, 2, 3, 4],
                "AbsoluteTime": [0, 500, 1_000, 1_500, 2_000],
                "valid": [True, False, True, True, True],
                "moving": [True, False, False, True, True],
                "bout_id": [0, 0, 0, 1, 1],
            }
        )
        self.protocol = pd.DataFrame(
            {"Type": ["Cycle"], "Beg": [1_000], "End": [1_100]}
        )
        self.identity = {
            "experiment_id": "allDelay",
            "recording_id": "20221115_04",
            "fish_id": "20221115_04",
            "condition_id": "delay-5s",
            "day": "20221115",
            "fish_number": "04",
        }
        self.config = TrialOutcomeConfig(
            baseline_window_s=(-1.0, 0.0),
            response_window_s=(0.0, 1.0),
        )

    def test_uses_exact_half_open_windows_and_preserves_rest_as_zero(self) -> None:
        outcomes, coverage = aggregate_trial_outcomes(
            self.frames,
            self.movement,
            self.protocol,
            identity=self.identity,
            experiment_name="allDelay",
            config=self.config,
        )

        metric_id = next(iter(METRIC_IDS.values()))
        outcome = outcomes[outcomes["metric_id"] == metric_id].iloc[0]
        trial_coverage = coverage[coverage["metric_id"] == metric_id].iloc[0]

        self.assertEqual(outcome["baseline_total_activity"], 2.0)
        self.assertEqual(outcome["response_total_activity"], 2.0)
        self.assertEqual(outcome["movement_probability"], 0.5)
        self.assertEqual(outcome["fraction_time_moving"], 0.5)
        self.assertEqual(outcome["conditional_intensity"], 4.0)
        self.assertEqual(outcome["bout_count"], 1)
        self.assertEqual(outcome["bout_rate_per_minute"], 60.0)
        self.assertEqual(outcome["mean_bout_duration_ms"], 1_000.0)
        self.assertEqual(trial_coverage["baseline_sample_count"], 2)
        self.assertEqual(trial_coverage["baseline_valid_sample_count"], 1)
        self.assertEqual(trial_coverage["baseline_valid_fraction"], 0.5)

    def test_rejects_duplicate_movement_rows(self) -> None:
        duplicated = pd.concat(
            [self.movement, self.movement.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaises(SchemaValidationError):
            aggregate_trial_outcomes(
                self.frames,
                duplicated,
                self.protocol,
                identity=self.identity,
                experiment_name="allDelay",
                config=self.config,
            )

    def test_cli_exposes_trial_outcome_recipe(self) -> None:
        args = build_parser().parse_args(
            [
                "candidate-trial-outcomes",
                "--project-dir",
                "paper",
                "--recording-id",
                "20221115_04",
            ]
        )

        self.assertEqual(args.recipe, "candidate-trial-outcomes-v1")

    @patch(
        "classical_conditioning.analysis.trial_outcomes._verify_inputs",
        return_value=(
            "recording",
            None,
            None,
            None,
            {
                "candidate_frames": "current-frame",
                "movement_state": "current-movement",
                "protocol": "current-protocol",
            },
            None,
        ),
    )
    @patch(
        "classical_conditioning.analysis.trial_outcomes.verify_completed_parquet_set"
    )
    def test_verifier_rejects_stale_upstream_lineage(
        self,
        mock_verify_completed,
        _mock_verify_inputs,
    ) -> None:
        mock_verify_completed.return_value = SimpleNamespace(
            summary={
                "inputs": {
                    "candidate_frames": "old-frame",
                    "movement_state": "current-movement",
                    "protocol": "current-protocol",
                }
            }
        )

        with self.assertRaisesRegex(ArtifactIntegrityError, "stale"):
            verify_candidate_trial_outcomes(
                Path("paper"),
                "recording-a",
            )
