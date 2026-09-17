from __future__ import annotations

import unittest

import pandas as pd

from classical_conditioning.figures.cohort_response import (
    DEFAULT_SELECTED_BLOCKS,
    summarize_event_aligned_ratios,
    summarize_selected_block_ratios,
    summarize_trial_ratios,
)


METRIC = "tail_length_weighted_angular_l1"


def trial_outcomes_fixture() -> pd.DataFrame:
    rows = []
    for fish_id, condition, multiplier in (
        ("fish-control", "control", 1.0),
        ("fish-delay", "delay", 2.0),
    ):
        for block in DEFAULT_SELECTED_BLOCKS:
            for trial in range(block.start_trial, block.end_trial + 1):
                rows.append(
                    {
                        "fish_id": fish_id,
                        "condition_id": condition,
                        "alignment": "CS",
                        "trial_number": trial,
                        "metric_id": METRIC,
                        "response_total_activity": 2.0 * multiplier,
                        "baseline_total_activity": 2.0,
                        "conditional_intensity": 2.0 * multiplier,
                        "baseline_conditional_intensity": 2.0,
                    }
                )
    return pd.DataFrame(rows)


def temporal_profiles_fixture() -> pd.DataFrame:
    rows = []
    for recording_id, multiplier in (("fish-control", 1.0), ("fish-delay", 2.0)):
        for trial in (5, 6):
            for time, value in ((-10.0, 2.0), (-5.0, 2.0), (1.0, 4.0 * multiplier)):
                rows.append(
                    {
                        "Recording ID": recording_id,
                        "Trial type": "CS",
                        "Trial number": trial,
                        "Time bin center (s)": time,
                        "Metric ID": METRIC,
                        "Total activity mean": value,
                        "Conditional intensity mean": value,
                    }
                )
    return pd.DataFrame(rows)


class CohortResponseFigureSummaryTests(unittest.TestCase):
    def test_selected_block_summary_uses_fish_medians_then_cohort_median(self) -> None:
        fish, cohort = summarize_selected_block_ratios(
            trial_outcomes_fixture(), metric_id=METRIC
        )

        self.assertEqual(len(fish), 2 * len(DEFAULT_SELECTED_BLOCKS))
        self.assertEqual(len(cohort), 2 * len(DEFAULT_SELECTED_BLOCKS))
        delay = cohort.loc[
            (cohort["condition_id"] == "delay")
            & (cohort["Selected block"] == "Early Test")
        ].iloc[0]
        self.assertEqual(delay["Fish count"], 1)
        self.assertEqual(delay["Cohort median response / baseline"], 2.0)

    def test_event_summary_normalizes_each_trial_before_fish_aggregation(self) -> None:
        fish, cohort = summarize_event_aligned_ratios(
            temporal_profiles_fixture(),
            metric_id=METRIC,
            condition_by_recording={"fish-control": "control", "fish-delay": "delay"},
        )

        self.assertEqual(fish["Recording ID"].nunique(), 2)
        delay_response = cohort.loc[
            (cohort["condition_id"] == "delay")
            & (cohort["Time bin center (s)"] == 1.0)
        ].iloc[0]
        self.assertEqual(delay_response["Cohort median response / baseline"], 4.0)
        control_baseline = cohort.loc[
            (cohort["condition_id"] == "control")
            & (cohort["Time bin center (s)"] == -10.0)
        ].iloc[0]
        self.assertEqual(control_baseline["Cohort median response / baseline"], 1.0)

    def test_trial_summary_uses_equal_fish_weight_with_unequal_rows(self) -> None:
        outcomes = trial_outcomes_fixture()
        one_row = outcomes.loc[
            (outcomes["fish_id"] == "fish-control")
            & (outcomes["trial_number"] == 5)
        ].iloc[0]
        duplicated = pd.concat(
            [outcomes, pd.DataFrame([one_row] * 20)], ignore_index=True
        )

        fish, cohort = summarize_trial_ratios(duplicated, metric_id=METRIC)

        trial_five = cohort.loc[cohort["trial_number"] == 5].set_index("condition_id")
        self.assertEqual(trial_five.loc["control", "Fish count"], 1)
        self.assertEqual(
            trial_five.loc["control", "Cohort median response / baseline"], 1.0
        )
        control_fish = fish.loc[
            (fish["fish_id"] == "fish-control") & (fish["trial_number"] == 5)
        ].iloc[0]
        self.assertEqual(control_fish["Contributing rows"], 21)

    def test_selected_block_keeps_ineligible_fish_in_coverage_table(self) -> None:
        outcomes = trial_outcomes_fixture()
        mask = (
            (outcomes["fish_id"] == "fish-delay")
            & outcomes["trial_number"].between(65, 69)
        )
        outcomes.loc[mask, "baseline_total_activity"] = 0.0

        fish, cohort = summarize_selected_block_ratios(outcomes, metric_id=METRIC)

        coverage = fish.loc[
            (fish["fish_id"] == "fish-delay")
            & (fish["Selected block"] == "Early Test")
        ].iloc[0]
        self.assertFalse(coverage["Eligible"])
        self.assertEqual(coverage["Contributing trials"], 0)
        self.assertEqual(
            coverage["Ineligible reason"],
            "fewer_than_minimum_finite_trial_ratios",
        )
        self.assertFalse(
            (
                (cohort["condition_id"] == "delay")
                & (cohort["Selected block"] == "Early Test")
            ).any()
        )


if __name__ == "__main__":
    unittest.main()
