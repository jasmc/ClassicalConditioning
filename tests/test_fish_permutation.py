from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from classical_conditioning.analysis.fish_permutation import (
    FishPermutationConfig,
    permutation_test_mean_effect,
    summarize_fish_learning_effects,
)
from classical_conditioning.analysis.model_input import build_candidate_model_input
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
                        "block_10_name": "Train 1" if trial <= 4 else "Train 5",
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


class FishPermutationTests(unittest.TestCase):
    def test_fish_effects_are_late_minus_early(self) -> None:
        model_input = build_candidate_model_input(_synthetic_outcomes())
        effects = summarize_fish_learning_effects(model_input)
        self.assertTrue((effects["late_trial_count"] > 0).all())
        self.assertTrue((effects["early_trial_count"] > 0).all())
        self.assertTrue((effects["learning_effect"] < 0).all())

    def test_permutation_is_deterministic_and_two_sided(self) -> None:
        model_input = build_candidate_model_input(_synthetic_outcomes())
        effects = summarize_fish_learning_effects(model_input)
        config = FishPermutationConfig(n_permutations=199, seed=10)
        first = permutation_test_mean_effect(effects, config=config)
        second = permutation_test_mean_effect(effects, config=config)
        self.assertTrue(
            np.allclose(
                first["permutation_p_value"],
                second["permutation_p_value"],
                equal_nan=True,
            )
        )
        self.assertTrue((first["permutation_p_value"] > 0).all())
        self.assertTrue((first["permutation_p_value"] <= 1).all())

    def test_cli_exposes_fish_permutation(self) -> None:
        args = build_parser().parse_args(
            [
                "candidate-fish-permutation",
                "--project-dir",
                "paper",
                "--recording-id",
                "recording-a",
                "--analysis-id",
                "perm-a",
            ]
        )
        self.assertEqual(args.metric_recipe, "tail-candidate-corrected-v1")


if __name__ == "__main__":
    unittest.main()
