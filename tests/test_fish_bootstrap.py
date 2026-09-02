from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from classical_conditioning.analysis.fish_bootstrap import (
    FishBootstrapConfig,
    bootstrap_mean_effect,
)
from classical_conditioning.analysis.fish_permutation import (
    summarize_fish_learning_effects,
)
from classical_conditioning.analysis.model_input import build_candidate_model_input
from classical_conditioning.cli import build_parser


def _synthetic_outcomes(n_fish: int = 4, n_trials: int = 8) -> pd.DataFrame:
    rows = []
    for fish_index in range(n_fish):
        fish_id = f"fish-{fish_index}"
        for trial in range(1, n_trials + 1):
            baseline = 1.0
            response = 0.6 if trial > 4 else 1.0
            rows.append(
                {
                    "experiment_id": "allDelay",
                    "recording_id": fish_id,
                    "fish_id": fish_id,
                    "condition_id": "delay",
                    "trial_id": f"{fish_id}:CS:{trial:03d}",
                    "alignment": "CS",
                    "trial_number": trial,
                    "block_10_name": "Train 1" if trial <= 4 else "Train 5",
                    "metric_id": "segment_absolute_angular_speed_sum",
                    "baseline_total_activity": baseline,
                    "response_total_activity": response,
                    "conditional_intensity": response,
                    "bout_rate_per_minute": 10.0 * response,
                    "baseline_valid_sample_count": 100,
                    "response_valid_sample_count": 100,
                }
            )
    return pd.DataFrame(rows)


class FishBootstrapTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_fish_unit(self) -> None:
        model_input = build_candidate_model_input(_synthetic_outcomes())
        effects = summarize_fish_learning_effects(model_input)
        config = FishBootstrapConfig(n_bootstrap=199, seed=10)
        first = bootstrap_mean_effect(effects, config=config)
        second = bootstrap_mean_effect(effects, config=config)
        self.assertTrue(
            np.allclose(first["ci_lower"], second["ci_lower"], equal_nan=True)
        )
        self.assertEqual(set(first["resample_unit"]), {"fish_id"})
        self.assertTrue((first["ci_lower"] <= first["observed_mean_effect"]).all())
        self.assertTrue((first["observed_mean_effect"] <= first["ci_upper"]).all())

    def test_cli_exposes_fish_bootstrap(self) -> None:
        args = build_parser().parse_args(
            [
                "candidate-fish-bootstrap",
                "--project-dir",
                "paper",
                "--recording-id",
                "recording-a",
                "--analysis-id",
                "boot-a",
            ]
        )
        self.assertEqual(args.metric_recipe, "tail-candidate-corrected-v1")


if __name__ == "__main__":
    unittest.main()
