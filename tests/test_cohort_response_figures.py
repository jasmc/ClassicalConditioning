from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.figures.cohort_response import (
    DEFAULT_SELECTED_BLOCKS,
    configured_catch_group,
    configured_cs_block_groups,
    summarize_event_aligned_ratios,
    summarize_scaled_activity_trial_groups,
    summarize_selected_block_ratios,
    summarize_trial_ratios,
)
from classical_conditioning.figures.population_heatmap import (
    build_population_heatmap_figure,
    summarize_population_heatmap,
)
from classical_conditioning.figures.export import FigureMode


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


def scaled_profiles_fixture() -> pd.DataFrame:
    rows = []
    for recording_id, fish_id, condition, value, available_trials in (
        ("control-1", "control-1", "control", 0.2, (25, 39, 53, 59, 65)),
        ("control-2", "control-2", "control", 0.8, (65,)),
        ("delay-1", "delay-1", "delay", 0.4, (25, 39, 53, 59, 65)),
    ):
        for trial in available_trials:
            for time in (-1.0, 1.0):
                rows.append(
                    {
                        "Recording ID": recording_id,
                        "Trial type": "CS",
                        "Trial number": trial,
                        "Time bin center (s)": time,
                        "Metric ID": METRIC,
                        "Scaled total activity": value,
                        "Valid expected fraction": 1.0,
                    }
                )
    return pd.DataFrame(rows)


class CohortResponseFigureSummaryTests(unittest.TestCase):
    def test_paper_selected_blocks_use_final_pretrain_and_test_fives(self) -> None:
        self.assertEqual(
            [(block.start_trial, block.end_trial) for block in DEFAULT_SELECTED_BLOCKS],
            [(10, 14), (65, 69), (90, 94)],
        )

    def test_configured_profile_groups_include_early_test_catch(self) -> None:
        catches = configured_catch_group("allDelay")
        self.assertEqual(catches[0].trial_numbers, (25, 39, 53, 59, 65))
        blocks = configured_cs_block_groups("allDelay")
        self.assertEqual(len(blocks), 9)
        self.assertEqual(blocks[6].label, "Test 1")
        self.assertIn(65, blocks[6].trial_numbers)

    def test_catch_scaled_activity_is_not_signed_vigor_change(self) -> None:
        fish, cohort = summarize_scaled_activity_trial_groups(
            scaled_profiles_fixture(),
            metric_id=METRIC,
            trial_groups=configured_catch_group("allDelay"),
            condition_by_recording={
                "control-1": "control", "control-2": "control", "delay-1": "delay"
            },
            fish_by_recording={
                "control-1": "control-1", "control-2": "control-2", "delay-1": "delay-1"
            },
        )
        self.assertTrue(fish["Fish median scaled total activity"].between(0, 1).all())
        self.assertTrue(cohort["Cohort median scaled total activity"].between(0, 1).all())
        self.assertNotIn("Signed vigor change", cohort.columns)

    def test_population_heatmap_uses_frozen_fish_weights_and_exposes_coverage(self) -> None:
        profiles = scaled_profiles_fixture()
        profiles.loc[
            (profiles["Recording ID"] == "control-2")
            & (profiles["Time bin center (s)"] == 1.0),
            "Valid expected fraction",
        ] = 0.5
        result = summarize_population_heatmap(
            profiles,
            metric_id=METRIC,
            condition_by_recording={
                "control-1": "control", "control-2": "control", "delay-1": "delay"
            },
            fish_by_recording={
                "control-1": "control-1", "control-2": "control-2", "delay-1": "delay-1"
            },
        )
        row = result.loc[
            (result["condition_id"] == "control")
            & (result["Trial number"] == 65)
            & (result["Time bin center (s)"] == 1.0)
        ].iloc[0]
        self.assertEqual(row["Contributing fish"], 1)
        self.assertEqual(row["Total cohort fish"], 2)
        self.assertAlmostEqual(row["Fish coverage fraction"], 0.5)
        self.assertAlmostEqual(row["Mean scaled total activity"], 0.2)
        self.assertEqual(row["Signal semantics"], "scaled_total_activity_all_valid_frames_including_zero")

    def test_population_heatmap_writes_panel_data_and_semantic_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            for recording_id, activity in (("control-fish", 0.2), ("delay-fish", 0.4)):
                path = (
                    project / "Processed data" / recording_id
                    / "candidate_temporal_outcomes-corrected.parquet"
                )
                path.parent.mkdir(parents=True)
                frame = pd.DataFrame([{
                    "Recording ID": recording_id,
                    "Trial type": "CS", "Trial number": trial,
                    "Time bin center (s)": time,
                    "Metric ID": METRIC,
                    "Scaled total activity": activity,
                    "Valid expected fraction": 1.0,
                } for trial in (10, 11) for time in (-0.25, 0.25)])
                pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
            cohort = SimpleNamespace(
                cohort_id="frozen", cohort_hash="cohort-hash",
                experiment_id="allDelay", metric_recipe="tail-candidate-corrected",
                recording_ids=("control-fish", "delay-fish"),
                condition_by_recording={"control-fish": "control", "delay-fish": "delay"},
                fish_by_recording={"control-fish": "control-fish", "delay-fish": "delay-fish"},
                input_artifacts=(),
            )
            with (
                patch("classical_conditioning.figures.population_heatmap._load_primary_cohort", return_value=cohort),
                patch("classical_conditioning.figures.population_heatmap._verify_temporal", return_value="marker-hash"),
                patch("classical_conditioning.figures.export._git_dirty", return_value=False),
            ):
                result = build_population_heatmap_figure(
                    project, cohort_id="frozen", analysis_id="paper",
                    metric_id=METRIC, mode=FigureMode.STATIC,
                )
                publication = build_population_heatmap_figure(
                    project, cohort_id="frozen", analysis_id="paper",
                    metric_id=METRIC, mode=FigureMode.PUBLICATION,
                    overwrite=True,
                )
            self.assertTrue(result.outputs[0].is_file())
            self.assertEqual({path.suffix for path in publication.outputs}, {".svg", ".pdf"})
            self.assertTrue(all(path.is_file() for path in publication.outputs))
            panel_data = (
                project / "Processed data" / "Analyses" / "paper"
                / "population-heatmap_tail-length-weighted-angular-l1.parquet"
            )
            self.assertTrue(panel_data.is_file())
            summary = json.loads(panel_data.with_suffix(".json").read_text(encoding="utf-8"))
            sidecar = json.loads(result.sidecar.read_text(encoding="utf-8"))
            self.assertEqual(summary["cohort_hash"], "cohort-hash")
            self.assertEqual(summary["aggregation"], "one bin per fish, then equal-fish mean")
            self.assertIn("heatmap__control__fish-coverage", sidecar["artist_registry"])
            self.assertEqual(
                sidecar["artist_registry"]["heatmap__delay__scaled-total-activity"]["units"],
                "0–1 scaled activity across all valid frames",
            )
            publication_sidecar = json.loads(publication.sidecar.read_text(encoding="utf-8"))
            self.assertEqual(
                sidecar["artist_registry"], publication_sidecar["artist_registry"]
            )

    def test_scaled_profile_pools_trials_within_fish_before_cohort(self) -> None:
        profiles = scaled_profiles_fixture()
        identities = {
            "control-1": "control",
            "control-2": "control",
            "delay-1": "delay",
        }
        fish_ids = {recording_id: recording_id for recording_id in identities}
        fish, cohort = summarize_scaled_activity_trial_groups(
            profiles,
            metric_id=METRIC,
            trial_groups=configured_catch_group("allDelay"),
            condition_by_recording=identities,
            fish_by_recording=fish_ids,
        )

        control = cohort.loc[
            (cohort["condition_id"] == "control")
            & (cohort["Time bin center (s)"] == 1.0)
        ].iloc[0]
        self.assertEqual(control["Fish count"], 2)
        self.assertAlmostEqual(
            control["Cohort median scaled total activity"], 0.5
        )
        contribution = fish.loc[
            (fish["fish_id"] == "control-1")
            & (fish["Time bin center (s)"] == 1.0),
            "Contributing trials",
        ].iloc[0]
        self.assertEqual(contribution, 5)

    def test_scaled_profile_masks_low_coverage_without_zero_filling(self) -> None:
        profiles = scaled_profiles_fixture()
        profiles.loc[
            (profiles["Recording ID"] == "control-2")
            & (profiles["Time bin center (s)"] == 1.0),
            "Valid expected fraction",
        ] = 0.5
        identities = {
            "control-1": "control",
            "control-2": "control",
            "delay-1": "delay",
        }
        fish_ids = {recording_id: recording_id for recording_id in identities}
        _, cohort = summarize_scaled_activity_trial_groups(
            profiles,
            metric_id=METRIC,
            trial_groups=configured_catch_group("allDelay"),
            condition_by_recording=identities,
            fish_by_recording=fish_ids,
        )

        control = cohort.loc[
            (cohort["condition_id"] == "control")
            & (cohort["Time bin center (s)"] == 1.0)
        ].iloc[0]
        self.assertEqual(control["Fish count"], 1)
        self.assertEqual(control["Cohort median scaled total activity"], 0.2)

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
            & (outcomes["trial_number"] == 10)
        ].iloc[0]
        duplicated = pd.concat(
            [outcomes, pd.DataFrame([one_row] * 20)], ignore_index=True
        )

        fish, cohort = summarize_trial_ratios(duplicated, metric_id=METRIC)

        trial_ten = cohort.loc[cohort["trial_number"] == 10].set_index("condition_id")
        self.assertEqual(trial_ten.loc["control", "Fish count"], 1)
        self.assertEqual(
            trial_ten.loc["control", "Cohort median response / baseline"], 1.0
        )
        control_fish = fish.loc[
            (fish["fish_id"] == "fish-control") & (fish["trial_number"] == 10)
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
