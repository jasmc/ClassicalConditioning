from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from classical_conditioning.figures.per_trial_scaled_vigor import (
    scale_per_trial_conditional_vigor,
    summarize_equal_fish_scaled_vigor,
    summarize_legacy_pooled_scaled_vigor,
)


class PerTrialScaledVigorTests(unittest.TestCase):
    def test_each_trial_uses_its_own_baseline_and_missing_bins_stay_missing(self) -> None:
        rows = []
        for trial, factor in ((5, 1.0), (6, 10.0)):
            for time, value in ((-30, 1), (-29, 2), (-28, 3), (-27, 4), (1, 4), (2, 1)):
                rows.append({
                    "Recording ID": "fish", "Trial type": "CS", "Trial number": trial,
                    "Time bin center (s)": time, "Metric ID": "metric",
                    "Conditional intensity mean": value * factor,
                    "Valid expected fraction": 0.0 if time == 2 else 1.0,
                })
        scaled = scale_per_trial_conditional_vigor(pd.DataFrame(rows))
        for trial in (5, 6):
            subset = scaled.loc[scaled["Trial number"].eq(trial)]
            self.assertEqual(float(subset.loc[subset["Time bin center (s)"].eq(1), "Per-trial scaled vigor"].iloc[0]), 1.0)
            self.assertTrue(np.isnan(subset.loc[subset["Time bin center (s)"].eq(2), "Per-trial scaled vigor"].iloc[0]))
        unbounded = scale_per_trial_conditional_vigor(pd.DataFrame(rows), clip=False)
        self.assertGreater(float(unbounded.loc[
            unbounded["Trial number"].eq(5) & unbounded["Time bin center (s)"].eq(1),
            "Per-trial scaled vigor",
        ].iloc[0]), 1.0)

    def test_population_mean_weights_fish_equally_and_records_coverage(self) -> None:
        data = pd.DataFrame({
            "Recording ID": ["a", "b", "c"], "Metric ID": ["m"] * 3,
            "Trial number": [5] * 3, "Time bin center (s)": [0.25] * 3,
            "Per-trial scaled vigor": [0.0, 1.0, np.nan],
        })
        pooled = summarize_equal_fish_scaled_vigor(
            data, metric_id="m", condition_by_recording={"a": "delay", "b": "delay", "c": "delay"},
        )
        self.assertEqual(float(pooled["Mean per-trial scaled vigor"].iloc[0]), 0.5)
        self.assertEqual(int(pooled["Contributing fish"].iloc[0]), 2)
        self.assertEqual(int(pooled["Total cohort fish"].iloc[0]), 3)

    def test_legacy_pool_applies_second_scale_after_fish_mean(self) -> None:
        rows = []
        for fish, offset in (("a", 0), ("b", 2)):
            for time, value in ((-3.5, 0), (-2.5, 1), (-1.5, 2), (-0.5, 3), (0.5, 5)):
                rows.append({
                    "Recording ID": fish, "Metric ID": "m", "Trial number": 5,
                    "Time bin center (s)": time, "Per-trial scaled vigor": value + offset,
                })
        pooled = summarize_legacy_pooled_scaled_vigor(
            pd.DataFrame(rows), metric_id="m",
            condition_by_recording={"a": "delay", "b": "delay"},
        )
        before = pooled.loc[pooled["Time bin center (s)"].eq(-2.5)].iloc[0]
        cs = pooled.loc[pooled["Time bin center (s)"].eq(0.5)].iloc[0]
        self.assertAlmostEqual(float(before["Pooled pre-rescale vigor"]), 2.0)
        self.assertAlmostEqual(float(before["Pooled baseline P10"]), 1.3)
        self.assertAlmostEqual(float(before["Pooled baseline P90"]), 3.7)
        self.assertAlmostEqual(float(before["Mean per-trial scaled vigor"]), (2 - 1.3) / 2.4)
        self.assertEqual(float(cs["Mean per-trial scaled vigor"]), 1.0)


if __name__ == "__main__":
    unittest.main()
