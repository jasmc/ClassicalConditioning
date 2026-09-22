from __future__ import annotations

import unittest

from classical_conditioning.config import (
    Alignment,
    ConditionRole,
    ConditionSpec,
    ExperimentSpec,
    FishKey,
    Paradigm,
    Phase,
    TimeWindow,
    TrialSpec,
    get_experiment_spec,
    get_trial_block_lookup,
)
from classical_conditioning.exceptions import ConfigurationError


class DomainConfigurationTests(unittest.TestCase):
    def test_identity_and_time_window_reject_invalid_values(self) -> None:
        with self.assertRaises(ConfigurationError):
            FishKey(experiment_id="allDelay", day="", fish_number="04")
        with self.assertRaises(ConfigurationError):
            TimeWindow(start_s=1.0, end_s=1.0)

    def test_experiment_rejects_duplicate_trials(self) -> None:
        condition = ConditionSpec("control", "Control", "control", ConditionRole.CONTROL, (0, 0, 0))
        trial = TrialSpec(Alignment.CS, 1, Phase.PRE, 1, "Pre-train")
        with self.assertRaises(ConfigurationError):
            ExperimentSpec("duplicate", Paradigm.DELAY, (condition,), (trial, trial), 1, 1, 10.0, TimeWindow(0.0, 9.0))

    def test_supported_experiments_expose_candidate_trial_maps(self) -> None:
        delay = get_experiment_spec("allDelay")
        trace_3s = get_experiment_spec("all3sTrace")
        trace_10s = get_experiment_spec("all10sTrace")
        self.assertEqual(delay.conditioned_response_window.end_s, 9.0)
        self.assertEqual(trace_3s.conditioned_response_window.end_s, 13.0)
        self.assertEqual(trace_10s.conditioned_response_window.end_s, 20.0)
        self.assertEqual(get_trial_block_lookup("allDelay"), get_trial_block_lookup("all3sTrace"))
        self.assertEqual(get_trial_block_lookup("allDelay"), get_trial_block_lookup("all10sTrace"))

    def test_supported_experiments_define_all_catches_including_early_test(self) -> None:
        for experiment_id in ("allDelay", "all3sTrace", "all10sTrace"):
            experiment = get_experiment_spec(experiment_id)
            catches = {
                trial.trial_number
                for trial in experiment.analysis_trials
                if trial.alignment is Alignment.CS and trial.catch
            }
            self.assertEqual(catches, {25, 39, 53, 59, 65})
            early_test = next(
                trial
                for trial in experiment.analysis_trials
                if trial.alignment is Alignment.CS and trial.trial_number == 65
            )
            self.assertEqual(early_test.phase, Phase.TEST)
            self.assertTrue(early_test.catch)

    def test_experiment_exposes_declared_cs_blocks_in_protocol_order(self) -> None:
        blocks = get_experiment_spec("allDelay").trial_blocks(Alignment.CS)
        self.assertEqual([name for name, _ in blocks], [
            "Pre-train",
            "Train 1",
            "Train 2",
            "Train 3",
            "Train 4",
            "Train 5",
            "Test 1",
            "Test 2",
            "Test 3",
        ])
        self.assertEqual(blocks[0][1], tuple(range(5, 15)))
        self.assertEqual(blocks[-1][1], tuple(range(85, 95)))


if __name__ == "__main__":
    unittest.main()
