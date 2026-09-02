from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from classical_conditioning.analysis.mixed_effects import (
    MixedEffectsConfig,
    build_mixed_effects_model_input,
    fit_candidate_mixed_effects,
)
from classical_conditioning.cli import build_parser


def _synthetic_outcomes(n_fish: int = 3, n_trials: int = 8) -> pd.DataFrame:
    rows = []
    metrics = [
        "segment_absolute_angular_speed_sum",
        "all_segment_angular_rms",
    ]
    for fish_index in range(n_fish):
        fish_id = f"fish-{fish_index}"
        for trial in range(1, n_trials + 1):
            for metric_id in metrics:
                baseline = 1.0 + 0.1 * fish_index
                response = baseline * (0.7 if trial > 4 else 1.0)
                rows.append(
                    {
                        "experiment_id": "allDelay",
                        "recording_id": fish_id,
                        "fish_id": fish_id,
                        "condition_id": "delay",
                        "trial_id": f"{fish_id}:CS:{trial:03d}",
                        "alignment": "CS",
                        "trial_number": trial,
                        "block_10_name": "Train 1" if trial <= 4 else "Train 2",
                        "metric_id": metric_id,
                        "baseline_total_activity": baseline,
                        "response_total_activity": response,
                        "conditional_intensity": response,
                        "bout_rate_per_minute": 10.0 * response,
                        "baseline_valid_sample_count": 100,
                        "response_valid_sample_count": 100,
                    }
                )
    return pd.DataFrame(rows)


class MixedEffectsTests(unittest.TestCase):
    def test_model_input_logs_and_requires_fish_grouping(self) -> None:
        outcomes = _synthetic_outcomes()
        model_input = build_mixed_effects_model_input(outcomes)
        self.assertIn("log_response", model_input.columns)
        self.assertEqual(set(model_input["outcome_id"]), set(
            ["total-activity", "conditional-intensity", "bout-rate"]
        ))
        with self.assertRaisesRegex(Exception, "fish_id"):
            MixedEffectsConfig(groups_column="recording_id")

    def test_fit_returns_diagnostics_per_metric_outcome(self) -> None:
        model_input = build_mixed_effects_model_input(_synthetic_outcomes())
        coefficients, diagnostics = fit_candidate_mixed_effects(model_input)
        self.assertGreater(len(diagnostics), 0)
        self.assertTrue(set(diagnostics["metric_id"]).issubset(set(model_input["metric_id"])))
        self.assertIn("converged", diagnostics.columns)
        if len(coefficients):
            self.assertIn("estimate", coefficients.columns)

    def test_cli_exposes_candidate_mixed_effects(self) -> None:
        args = build_parser().parse_args(
            [
                "candidate-mixed-effects",
                "--project-dir",
                "paper",
                "--recording-id",
                "recording-a",
                "--analysis-id",
                "mixed-a",
            ]
        )
        self.assertEqual(args.metric_recipe, "tail-candidate-corrected-v1")


if __name__ == "__main__":
    unittest.main()
