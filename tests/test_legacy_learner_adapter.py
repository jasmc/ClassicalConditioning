from __future__ import annotations

import unittest

import pandas as pd

from classical_conditioning.analysis.legacy_learners import build_legacy_input
from classical_conditioning.analysis.legacy_learner_controls import control_flag_tables
from classical_conditioning.exceptions import SchemaValidationError


class LegacyLearnerAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cohort = pd.DataFrame([
            {"experiment_id": "allDelay", "condition_id": "delay", "fish_id": "d1", "primary_included": True},
            {"experiment_id": "allDelay", "condition_id": "control", "fish_id": "c1", "primary_included": True},
            {"experiment_id": "allDelay", "condition_id": "delay", "fish_id": "excluded", "primary_included": False},
        ])
        self.outcomes = pd.DataFrame([
            {"experiment_id": "allDelay", "condition_id": condition, "fish_id": fish,
             "alignment": "CS", "metric_id": "metric", "trial_number": 5,
             "baseline_total_activity": 2.0, "response_total_activity": 1.0}
            for condition, fish in (("delay", "d1"), ("control", "c1"), ("delay", "excluded"))
        ])

    def test_translation_keeps_only_primary_cs_metric_and_ratio(self) -> None:
        frame = build_legacy_input(self.cohort, self.outcomes, "metric")
        self.assertEqual(set(frame["Fish"]), {"d1", "c1"})
        self.assertEqual(set(frame["Normalized vigor"]), {0.5})
        self.assertEqual(set(frame["Mean CR"]), {1.0})

    def test_duplicate_trial_rejected(self) -> None:
        duplicated = pd.concat([self.outcomes, self.outcomes.iloc[[0]]])
        with self.assertRaises(SchemaValidationError):
            build_legacy_input(self.cohort, duplicated, "metric")

    def test_control_flag_rates_keep_fish_ids(self) -> None:
        a = pd.DataFrame({"Fish_ID": ["c1", "d1"], "Condition": ["control", "delay"],
                          "learner_primary": [True, True]})
        b = a.assign(learner_primary=[False, True])
        rates, fish = control_flag_tables({"a": a, "b": b})
        self.assertEqual(rates.loc[(rates.variant == "a") & (rates.condition == "control"),
                                   "flagged_fish_ids"].iloc[0], "c1")
        self.assertEqual(rates.loc[(rates.variant == "b") & (rates.condition == "control"),
                                   "flagged_count"].iloc[0], 0)
        self.assertEqual(len(fish), 4)


if __name__ == "__main__":
    unittest.main()
