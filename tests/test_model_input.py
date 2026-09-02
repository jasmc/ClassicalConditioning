from __future__ import annotations

import unittest

import pandas as pd

from classical_conditioning.analysis.model_input import (
    MODEL_INPUT_COLUMNS,
    ModelInputConfig,
    build_candidate_model_input,
    model_input_coverage,
)
from classical_conditioning.cli import build_parser


def _synthetic_outcomes() -> pd.DataFrame:
    rows = []
    for fish_index in range(2):
        fish_id = f"fish-{fish_index}"
        for trial in range(1, 5):
                rows.append(
                    {
                        "experiment_id": "allDelay",
                        "recording_id": fish_id,
                        "fish_id": fish_id,
                        "condition_id": "delay",
                        "trial_id": f"{fish_id}:CS:{trial:03d}",
                        "alignment": "CS",
                        "trial_number": trial,
                        "block_10_name": "Train 1",
                        "metric_id": "segment_absolute_angular_speed_sum",
                        "baseline_total_activity": 1.0,
                        "response_total_activity": 0.8,
                        "conditional_intensity": 0.8,
                        "bout_rate_per_minute": 8.0,
                        "baseline_valid_sample_count": 100,
                        "response_valid_sample_count": 100,
                    }
                )
    return pd.DataFrame(rows)


class ModelInputTests(unittest.TestCase):
    def test_builder_emits_canonical_columns_and_coverage(self) -> None:
        model_input = build_candidate_model_input(_synthetic_outcomes())
        self.assertEqual(list(model_input.columns), list(MODEL_INPUT_COLUMNS))
        coverage = model_input_coverage(model_input)
        self.assertEqual(coverage["fish_count"], 2)
        self.assertIn("total-activity", coverage["outcome_ids"])

    def test_require_block_label_can_be_disabled(self) -> None:
        outcomes = _synthetic_outcomes()
        outcomes.loc[:, "block_10_name"] = None
        with self.assertRaisesRegex(Exception, "non-null block_10_name"):
            build_candidate_model_input(outcomes)
        kept = build_candidate_model_input(
            outcomes,
            config=ModelInputConfig(require_block_label=False),
        )
        self.assertEqual(len(kept), 2 * 4 * 3)

    def test_cli_exposes_candidate_model_input(self) -> None:
        args = build_parser().parse_args(
            [
                "candidate-model-input",
                "--project-dir",
                "paper",
                "--recording-id",
                "recording-a",
                "--analysis-id",
                "model-a",
            ]
        )
        self.assertEqual(args.metric_recipe, "tail-candidate-corrected-v1")


if __name__ == "__main__":
    unittest.main()
