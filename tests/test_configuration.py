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
        trace = get_experiment_spec("fixedVsIncreasingTrace")
        self.assertEqual(delay.conditioned_response_window.end_s, 9.0)
        self.assertEqual(trace.conditioned_response_window.end_s, 13.0)
        self.assertEqual(get_trial_block_lookup("allDelay"), get_trial_block_lookup("fixedVsIncreasingTrace"))


if __name__ == "__main__":
    unittest.main()
