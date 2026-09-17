from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import pandas as pd

from classical_conditioning.analysis.inference.learning_onset import (
    LearningOnsetConfig,
    _fit_mixed_model,
    _descriptive_trajectory_tables,
    _holm_adjust,
    build_learning_model_input,
    fish_level_robustness,
    localize_learning_onset,
    trial_contrasts,
)
from classical_conditioning.cli import build_parser


METRIC = "tail_length_weighted_angular_l1"


def model_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    eligibility = []
    for condition in ("control", "delay"):
        for fish_index in range(3):
            fish_id = f"{condition}-{fish_index}"
            for trial in range(5, 75):
                block = (
                    "Pre-train"
                    if trial < 15
                    else "Train 1"
                    if trial < 65
                    else "Test 1"
                )
                baseline = 2.0
                response = 2.0
                if condition == "delay" and trial >= 65:
                    response = 1.0
                trial_id = f"{fish_id}:CS:{trial:03d}"
                rows.append(
                    {
                        "cohort_id": "paper",
                        "cohort_hash": "abc",
                        "experiment_id": "allDelay",
                        "recording_id": fish_id,
                        "fish_id": fish_id,
                        "condition_id": condition,
                        "trial_id": trial_id,
                        "alignment": "CS",
                        "trial_number": trial,
                        "block_10_name": block,
                        "metric_id": METRIC,
                        "baseline_total_activity": baseline,
                        "response_total_activity": response,
                    }
                )
                eligibility.append(
                    {
                        "trial_id": trial_id,
                        "metric_id": METRIC,
                        "outcome_id": "total-activity",
                        "eligible": True,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(eligibility)


class LearningOnsetCoreTests(unittest.TestCase):
    def test_null_and_preexisting_difference_do_not_localize_onset(self) -> None:
        contrasts = pd.DataFrame(
            {
                "trial_number": np.arange(1, 11),
                "simultaneous_lower": np.zeros(10),
                "estimable": True,
            }
        )

        result = localize_learning_onset(
            contrasts,
            delta_min=0.0,
            persistence_trials=3,
        )

        self.assertFalse(result["localized"])
        self.assertEqual(result["failure_reason"], "threshold_never_exceeded")

    def test_gradual_persistent_effect_localizes_first_supported_run(self) -> None:
        contrasts = pd.DataFrame(
            {
                "trial_number": np.arange(1, 9),
                "simultaneous_lower": [-0.2, -0.1, 0.0, 0.05, 0.11, 0.14, 0.2, 0.3],
                "estimable": True,
            }
        )

        result = localize_learning_onset(
            contrasts,
            delta_min=0.1,
            persistence_trials=3,
        )

        self.assertTrue(result["localized"])
        self.assertEqual(result["onset_trial"], 5)

    def test_onset_requires_consecutive_scheduled_trials(self) -> None:
        contrasts = pd.DataFrame(
            {
                "trial_number": [65, 66, 67, 68],
                "simultaneous_lower": [0.2, 0.3, 0.4, -0.1],
                "estimable": [True, True, True, True],
            }
        )

        result = localize_learning_onset(
            contrasts,
            delta_min=0.1,
            persistence_trials=3,
        )

        self.assertTrue(result["localized"])
        self.assertEqual(result["onset_trial"], 65)

    def test_onset_rejects_transient_and_gapped_support(self) -> None:
        contrasts = pd.DataFrame(
            {
                "trial_number": [65, 66, 68, 69],
                "simultaneous_lower": [0.2, 0.3, 0.4, 0.5],
                "estimable": [True, True, True, True],
            }
        )

        result = localize_learning_onset(
            contrasts,
            delta_min=0.1,
            persistence_trials=3,
        )

        self.assertFalse(result["localized"])
        self.assertEqual(result["failure_reason"], "threshold_not_persistent")

    def test_model_input_and_robustness_preserve_condition(self) -> None:
        outcomes, eligibility = model_fixture()
        config = LearningOnsetConfig(
            metric_id=METRIC,
            late_blocks=("Test 1",),
            n_bootstrap=0,
            n_permutations=199,
        )

        model_input = build_learning_model_input(
            outcomes, eligibility, config=config
        )
        fish, result = fish_level_robustness(model_input, config=config)

        self.assertEqual(set(fish["condition_id"]), {"control", "delay"})
        self.assertGreater(result.iloc[0]["test_minus_control"], 0)
        self.assertTrue(np.isfinite(result.iloc[0]["permutation_p_value"]))

    def test_fish_identity_is_scoped_by_experiment(self) -> None:
        outcomes, eligibility = model_fixture()
        duplicate = outcomes.loc[outcomes["fish_id"] == "control-0"].copy()
        duplicate["experiment_id"] = "second-experiment"
        duplicate["recording_id"] = "second-recording"
        duplicate["trial_id"] = duplicate["trial_id"].astype(str) + ":second"
        duplicate_eligibility = eligibility.loc[
            eligibility["trial_id"].astype(str).str.startswith("control-0:")
        ].copy()
        duplicate_eligibility["trial_id"] = (
            duplicate_eligibility["trial_id"].astype(str) + ":second"
        )
        outcomes = pd.concat([outcomes, duplicate], ignore_index=True)
        eligibility = pd.concat(
            [eligibility, duplicate_eligibility], ignore_index=True
        )
        config = LearningOnsetConfig(
            metric_id=METRIC,
            late_blocks=("Test 1",),
            n_bootstrap=0,
            n_permutations=199,
        )

        model_input = build_learning_model_input(
            outcomes, eligibility, config=config
        )

        scoped = model_input.loc[
            model_input["fish_id"].astype(str) == "control-0", "fish_key"
        ]
        self.assertEqual(scoped.nunique(), 2)

    def test_model_input_is_row_order_invariant(self) -> None:
        outcomes, eligibility = model_fixture()
        config = LearningOnsetConfig(
            metric_id=METRIC,
            late_blocks=("Test 1",),
            n_bootstrap=0,
            n_permutations=199,
        )
        ordered = build_learning_model_input(outcomes, eligibility, config=config)
        shuffled = build_learning_model_input(
            outcomes.sample(frac=1.0, random_state=19),
            eligibility.sample(frac=1.0, random_state=23),
            config=config,
        )
        columns = ["trial_id", "fish_key", "cr_score", "trial_scaled"]

        pd.testing.assert_frame_equal(
            ordered.loc[:, columns].sort_values("trial_id").reset_index(drop=True),
            shuffled.loc[:, columns].sort_values("trial_id").reset_index(drop=True),
        )

    def test_duplicate_rows_do_not_add_fish_weight(self) -> None:
        outcomes, eligibility = model_fixture()
        config = LearningOnsetConfig(
            metric_id=METRIC,
            late_blocks=("Test 1",),
            n_bootstrap=0,
            n_permutations=199,
        )
        model_input = build_learning_model_input(
            outcomes, eligibility, config=config
        )
        _, original_group = _descriptive_trajectory_tables(model_input)
        duplicated = pd.concat(
            [
                model_input,
                model_input.loc[model_input["fish_key"] == "allDelay::control-0"],
            ],
            ignore_index=True,
        )
        _, duplicated_group = _descriptive_trajectory_tables(duplicated)

        pd.testing.assert_frame_equal(original_group, duplicated_group)

    def test_holm_adjustment_is_monotone_in_p_value_order(self) -> None:
        p_values = np.array([0.04, 0.001, 0.02, np.nan])
        adjusted = _holm_adjust(p_values)
        ordered = np.argsort(p_values[:3])

        self.assertTrue(np.all(np.diff(adjusted[:3][ordered]) >= 0))
        self.assertTrue(np.isnan(adjusted[3]))

    def test_cli_requires_explicit_test_condition_and_delta(self) -> None:
        args = build_parser().parse_args(
            [
                "learning-onset",
                "--project-dir",
                "paper",
                "--cohort-id",
                "paper-cohort",
                "--analysis-id",
                "learning",
                "--metric",
                METRIC,
                "--test-condition",
                "delay",
                "--delta-min",
                "0.1",
            ]
        )

        self.assertEqual(args.test_condition, "delay")
        self.assertEqual(args.delta_min, 0.1)

    def test_cli_can_disable_categorical_sensitivity(self) -> None:
        args = build_parser().parse_args(
            [
                "learning-onset",
                "--project-dir",
                "paper",
                "--cohort-id",
                "paper-cohort",
                "--analysis-id",
                "learning",
                "--metric",
                METRIC,
                "--test-condition",
                "delay",
                "--delta-min",
                "0.1",
                "--skip-categorical-sensitivity",
            ]
        )

        self.assertTrue(args.skip_categorical_sensitivity)

    def test_cli_exposes_residual_diagnostics_figure(self) -> None:
        args = build_parser().parse_args(
            [
                "figure-learning-diagnostics",
                "--project-dir",
                "paper",
                "--analysis-id",
                "learning",
            ]
        )

        self.assertEqual(args.analysis_id, "learning")

    @unittest.skipUnless(
        all(
            importlib.util.find_spec(module) is not None
            for module in ("statsmodels", "scipy", "patsy")
        ),
        "mixed-model analysis dependencies are unavailable",
    )
    def test_synthetic_condition_by_time_effect_is_recovered(self) -> None:
        rng = np.random.default_rng(27)
        rows = []
        for condition in ("control", "delay"):
            for fish_index in range(10):
                fish_key = f"experiment::{condition}-{fish_index}"
                fish_intercept = rng.normal(0.0, 0.2)
                for trial in range(1, 31):
                    log_baseline = 0.7 + rng.normal(0.0, 0.05)
                    learning = (
                        -0.5 * max(0, trial - 15) / 15
                        if condition == "delay"
                        else 0.0
                    )
                    rows.append(
                        {
                            "fish_key": fish_key,
                            "fish_id": f"{condition}-{fish_index}",
                            "condition_id": condition,
                            "trial_number": trial,
                            "trial_scaled": (trial - 15.5) / 8.655,
                            "trial_center": 15.5,
                            "trial_scale": 8.655,
                            "block_10_name": (
                                "Pre-train" if trial <= 10 else "Train 1"
                            ),
                            "log_baseline": log_baseline,
                            "log_response": (
                                0.3
                                + 0.6 * log_baseline
                                + fish_intercept
                                + learning
                                + rng.normal(0.0, 0.08)
                            ),
                        }
                    )
        data = pd.DataFrame(rows)
        data["condition_id"] = pd.Categorical(
            data["condition_id"], categories=["control", "delay"]
        )
        config = LearningOnsetConfig(
            metric_id=METRIC,
            random_effects_formula="1",
            allow_random_intercept_fallback=False,
            late_blocks=("Train 1",),
            run_categorical_sensitivity=False,
            sensitivity_optimizer=None,
            run_random_intercept_sensitivity=False,
            n_bootstrap=0,
            n_permutations=199,
        )
        formula = (
            "log_response ~ log_baseline + "
            "C(condition_id, Treatment(reference='control')) * "
            "bs(trial_scaled, df=4, degree=3, include_intercept=False)"
        )

        fitted, diagnostic = _fit_mixed_model(
            data,
            formula=formula,
            config=config,
        )

        self.assertEqual(diagnostic["diagnostic_status"], "ok")
        self.assertIsNotNone(fitted)
        contrasts, _ = trial_contrasts(fitted, data, config=config)
        early = contrasts.loc[contrasts["trial_number"] == 12, "learning_contrast"]
        late = contrasts.loc[contrasts["trial_number"] == 30, "learning_contrast"]
        self.assertGreater(float(late.iloc[0]), float(early.iloc[0]))


if __name__ == "__main__":
    unittest.main()
