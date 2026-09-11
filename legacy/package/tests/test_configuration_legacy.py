from __future__ import annotations

import contextlib
import io
import json
import unittest
from dataclasses import replace

from classical_conditioning.analysis.temporal_profiles import TemporalProfileConfig
from classical_conditioning.config import (
    Alignment,
    ConditionRole,
    ConditionSpec,
    ConfigurationStage,
    ExperimentSpec,
    FishKey,
    Paradigm,
    Phase,
    ScientificStatus,
    TemporalOutcomeSettings,
    TimeWindow,
    TrialSpec,
    config_hash,
    config_to_json,
    get_experiment_spec,
    get_legacy_paper_config,
    get_trial_block_lookup,
    stage_config_hash,
)
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.preprocessing.legacy_v1 import (
    LegacyPreprocessingConfig,
)
from experiment_configuration import get_experiment_config
from general_configuration import config as legacy_general_config


class DomainConfigurationTests(unittest.TestCase):
    def test_fish_key_and_time_window_reject_invalid_identity(self) -> None:
        with self.assertRaises(ConfigurationError):
            FishKey(experiment_id="allDelay", day="", fish_number="04")
        with self.assertRaises(ConfigurationError):
            FishKey(experiment_id="allDelay", day="20221115", fish_number=4)  # type: ignore[arg-type]
        with self.assertRaises(ConfigurationError):
            TimeWindow(start_s=1.0, end_s=1.0)

    def test_experiment_rejects_duplicate_conditions_and_trials(self) -> None:
        condition = ConditionSpec(
            condition_id="control",
            display_name="Control",
            source_name="control",
            role=ConditionRole.CONTROL,
            color_rgb_255=(0, 0, 0),
        )
        trial = TrialSpec(
            alignment=Alignment.CS,
            trial_number=1,
            phase=Phase.PRE,
            block_10_id=1,
            block_10_name="Pre-train",
        )
        with self.assertRaises(ConfigurationError):
            ExperimentSpec(
                experiment_id="duplicate",
                paradigm=Paradigm.DELAY,
                conditions=(condition, condition),
                analysis_trials=(trial,),
                minimum_cs_trials=1,
                minimum_us_trials=1,
                cs_duration_s=10.0,
                conditioned_response_window=TimeWindow(0.0, 9.0),
            )
        with self.assertRaises(ConfigurationError):
            ExperimentSpec(
                experiment_id="duplicate",
                paradigm=Paradigm.DELAY,
                conditions=(condition,),
                analysis_trials=(trial, trial),
                minimum_cs_trials=1,
                minimum_us_trials=1,
                cs_duration_s=10.0,
                conditioned_response_window=TimeWindow(0.0, 9.0),
            )

    def test_invalid_enums_and_incomplete_trial_mapping_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            TrialSpec(
                alignment="CS",  # type: ignore[arg-type]
                trial_number=1,
                phase=Phase.PRE,
                block_10_id=1,
                block_10_name="Pre-train",
            )
        condition = ConditionSpec(
            condition_id="control",
            display_name="Control",
            source_name="control",
            role=ConditionRole.CONTROL,
            color_rgb_255=(0, 0, 0),
        )
        trials = tuple(
            TrialSpec(
                alignment=Alignment.CS,
                trial_number=number,
                phase=Phase.PRE,
                block_10_id=1,
                block_10_name="Pre-train",
            )
            for number in (1, 3)
        )
        with self.assertRaises(ConfigurationError):
            ExperimentSpec(
                experiment_id="incomplete",
                paradigm=Paradigm.DELAY,
                conditions=(condition,),
                analysis_trials=trials,
                minimum_cs_trials=1,
                minimum_us_trials=1,
                cs_duration_s=10.0,
                conditioned_response_window=TimeWindow(0.0, 9.0),
            )

    def test_all_delay_spec_reproduces_active_legacy_experiment_values(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            legacy = get_experiment_config("allDelay")
        migrated = get_experiment_spec("allDelay")

        self.assertEqual(migrated.minimum_cs_trials, legacy.min_number_cs_trials)
        self.assertEqual(migrated.minimum_us_trials, legacy.min_number_us_trials)
        self.assertEqual(migrated.cs_duration_s, legacy.cs_duration)
        self.assertEqual(
            [
                condition.condition_id for condition in migrated.conditions
            ],
            list(legacy.cond_dict),
        )
        self.assertEqual(
            [condition.color_rgb_255 for condition in migrated.conditions],
            [legacy.cond_dict[key]["color"] for key in legacy.cond_dict],
        )
        self.assertEqual(
            [condition.source_name for condition in migrated.conditions],
            [
                legacy.cond_dict[key]["name in original path"]
                for key in legacy.cond_dict
            ],
        )

    def test_all_delay_trial_lookup_reproduces_active_legacy_blocks(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            legacy = get_experiment_config("allDelay")
        expected: dict[tuple[str, int], str] = {}
        for alignment, blocks, names in (
            ("CS", legacy.trials_cs_blocks_10, legacy.names_cs_blocks_10),
            ("US", legacy.trials_us_blocks_10, legacy.names_us_blocks_10),
        ):
            for trials, name in zip(blocks, names, strict=True):
                expected.update(
                    {(alignment, int(trial_number)): name for trial_number in trials}
                )

        self.assertEqual(get_trial_block_lookup("allDelay"), expected)
        self.assertEqual(len(expected), 90 + 46)

    def test_fixed_vs_increasing_trace_uses_13s_cr_window(self) -> None:
        spec = get_experiment_spec("fixedVsIncreasingTrace")
        self.assertEqual(spec.paradigm, Paradigm.TRACE)
        self.assertEqual(spec.conditioned_response_window.start_s, 0.0)
        self.assertEqual(spec.conditioned_response_window.end_s, 13.0)
        self.assertEqual(
            [condition.condition_id for condition in spec.conditions],
            ["control", "fixedtrace"],
        )
        self.assertEqual(
            get_trial_block_lookup("fixedVsIncreasingTrace"),
            get_trial_block_lookup("allDelay"),
        )

    def test_unknown_experiment_uses_typed_configuration_error(self) -> None:
        with self.assertRaises(ConfigurationError):
            get_experiment_spec("not-migrated")


class ResolvedRecipeTests(unittest.TestCase):
    def test_legacy_recipe_reproduces_active_processing_defaults(self) -> None:
        resolved = get_legacy_paper_config()
        preprocessing = resolved.preprocessing
        legacy = LegacyPreprocessingConfig()

        shared_fields = (
            "expected_framerate_hz",
            "camera_rows_discarded_at_start",
            "maximum_interval_deviation_ms",
            "frame_loss_buffer_frames",
            "tracking_error_threshold_deg",
            "angle_point_count",
            "temporal_filter_frames",
            "bout_max_window_frames",
            "bout_min_window_frames",
            "bout_threshold_primary_deg_per_ms",
            "minimum_bout_duration_frames",
            "minimum_interbout_frames",
            "trial_start_frames",
            "trial_end_frames",
            "baseline_window_frames",
        )
        for field_name in shared_fields:
            with self.subTest(field=field_name):
                self.assertEqual(
                    getattr(preprocessing, field_name),
                    getattr(legacy, field_name),
                )
        self.assertEqual(
            preprocessing.spatial_filter_segments_configured_but_not_applied,
            legacy_general_config.filtering.space_bcf_window,
        )
        self.assertEqual(
            preprocessing.bout_threshold_secondary_configured_but_not_applied_deg_per_ms,
            legacy.bout_threshold_secondary_deg_per_ms,
        )
        self.assertEqual(resolved.scientific_status, ScientificStatus.LEGACY)

    def test_outcome_defaults_reproduce_active_temporal_profile_defaults(self) -> None:
        migrated = get_legacy_paper_config().outcomes
        active = TemporalProfileConfig()
        self.assertEqual(migrated.window_start_s, active.window_start_s)
        self.assertEqual(migrated.window_end_s, active.window_end_s)
        self.assertEqual(migrated.bin_width_s, active.bin_width_s)
        self.assertEqual(migrated.interval_closure, active.interval_closure)
        self.assertEqual(migrated.aggregation, active.aggregation)

    def test_serialization_and_hashes_are_stable(self) -> None:
        first = get_legacy_paper_config()
        second = get_legacy_paper_config()
        self.assertEqual(config_to_json(first), config_to_json(second))
        self.assertEqual(config_hash(first), config_hash(second))
        self.assertEqual(
            config_hash(first),
            "0c4f133a874d9d87cf9fda1e7be7b3e07b148eaaf86a2b5e647bc4c63d24d6e9",
        )
        decoded = json.loads(config_to_json(first))
        self.assertEqual(decoded["recipe_id"], "legacy-paper-v1")
        self.assertEqual(decoded["experiment"]["experiment_id"], "allDelay")
        self.assertEqual(
            {entry["section"] for entry in decoded["source_trace"]},
            {"experiment", "preprocessing", "outcomes"},
        )

    def test_stage_hash_changes_only_for_relevant_settings(self) -> None:
        baseline = get_legacy_paper_config()
        changed_outcomes = replace(
            baseline,
            outcomes=replace(baseline.outcomes, bin_width_s=1.0),
        )
        self.assertEqual(
            stage_config_hash(baseline, ConfigurationStage.PREPROCESSING),
            stage_config_hash(changed_outcomes, ConfigurationStage.PREPROCESSING),
        )
        self.assertNotEqual(
            stage_config_hash(baseline, ConfigurationStage.OUTCOMES),
            stage_config_hash(changed_outcomes, ConfigurationStage.OUTCOMES),
        )
        changed_preprocessing = replace(
            baseline,
            preprocessing=replace(
                baseline.preprocessing,
                temporal_filter_frames=11,
            ),
        )
        self.assertNotEqual(
            stage_config_hash(baseline, ConfigurationStage.PREPROCESSING),
            stage_config_hash(changed_preprocessing, ConfigurationStage.PREPROCESSING),
        )
        self.assertEqual(
            stage_config_hash(baseline, ConfigurationStage.OUTCOMES),
            stage_config_hash(changed_preprocessing, ConfigurationStage.OUTCOMES),
        )
        recolored_condition = replace(
            baseline.experiment.conditions[0],
            display_name="Control display only",
            color_rgb_255=(1, 2, 3),
        )
        display_only_change = replace(
            baseline,
            experiment=replace(
                baseline.experiment,
                conditions=(
                    recolored_condition,
                    *baseline.experiment.conditions[1:],
                ),
            ),
        )
        self.assertNotEqual(config_hash(baseline), config_hash(display_only_change))
        self.assertEqual(
            stage_config_hash(baseline, ConfigurationStage.PREPROCESSING),
            stage_config_hash(display_only_change, ConfigurationStage.PREPROCESSING),
        )
        self.assertEqual(
            stage_config_hash(baseline, ConfigurationStage.OUTCOMES),
            stage_config_hash(display_only_change, ConfigurationStage.OUTCOMES),
        )

    def test_invalid_outcome_settings_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            TemporalOutcomeSettings(
                window_start_s=-1.0,
                window_end_s=1.0,
                bin_width_s=0.0,
            )


if __name__ == "__main__":
    unittest.main()
