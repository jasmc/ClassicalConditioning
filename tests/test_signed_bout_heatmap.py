from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from classical_conditioning.figures.signed_bout_heatmap import (
    calculate_fish_heatmaps,
    summarize_equal_fish_signed_log_vigor,
)
from classical_conditioning.figures.example_traces import METRIC_COLUMNS


METRIC = "tail_length_weighted_angular_l1"


class SignedBoutHeatmapTests(unittest.TestCase):
    def test_pool_accepts_one_pre15_baseline_and_rejects_mixed_windows(self) -> None:
        rows = pd.DataFrame({
            "Recording ID": ["fish-1", "fish-2"],
            "Trial number": [5, 5], "Time bin center (s)": [0.25, 0.25],
            "Metric ID": [METRIC, METRIC], "Signed log vigor": [1.0, 3.0],
            "Baseline start (s)": [-15.0, -15.0],
            "Baseline end (s)": [0.0, 0.0],
        })
        pooled = summarize_equal_fish_signed_log_vigor(
            rows, metric_id=METRIC,
            condition_by_recording={"fish-1": "delay", "fish-2": "delay"},
        )
        self.assertEqual(pooled.iloc[0]["Mean signed log vigor"], 2.0)
        self.assertEqual(pooled.iloc[0]["Baseline start (s)"], -15.0)
        self.assertEqual(pooled.iloc[0]["Signal semantics"],
                         "signed_bout_log_vigor_baseline_-15_to_0_median")
        rows.loc[1, "Baseline start (s)"] = -20.0
        with self.assertRaisesRegex(ValueError, "consistent pre-CS baseline"):
            summarize_equal_fish_signed_log_vigor(
                rows, metric_id=METRIC,
                condition_by_recording={"fish-1": "delay", "fish-2": "delay"},
            )

    def test_pre15_baseline_includes_start_excludes_cs_and_earlier_frames(self) -> None:
        times = (-19000, -15001, -15000, -1, 0, 1000)
        metrics = pd.DataFrame({
            "FrameID": range(6), "AbsoluteTime": times,
            METRIC_COLUMNS[METRIC]: np.exp([9., 9., 1., 3., 10., 4.]),
        })
        movement = pd.DataFrame({
            "FrameID": range(6), "AbsoluteTime": times,
            "valid": True, "moving": True, "bout_id": range(1, 7),
        })
        cycles = pd.DataFrame({"Beg": [1000000] * 4 + [0] + [1000000] * 89})
        bins = calculate_fish_heatmaps(
            metrics, movement, cycles, recording_id="example", metric_ids=(METRIC,),
            baseline_start_s=-15, baseline_end_s=0,
        )
        trial = bins.loc[bins["Trial number"].eq(5)].set_index("Time bin center (s)")
        self.assertAlmostEqual(trial.loc[1.25, "Signed log vigor"], 2.)
        self.assertAlmostEqual(trial.loc[-18.75, "Signed log vigor"], 7.)
        self.assertAlmostEqual(trial.loc[.25, "Signed log vigor"], 8.)
        self.assertTrue(bins["Baseline start (s)"].eq(-15).all())
        self.assertTrue(bins.loc[bins["Trial number"].eq(6), "Signed log vigor"].isna().all())

    def test_fish_bins_are_baseline_centered_and_pool_without_rescaling(self) -> None:
        values = (1.0, 1.0, 2.0, 4.0)
        times = (-19000, -18000, 1000, 2000)
        fish_frames = []
        for fish, factor in (("control-fish", 1.0), ("delay-fish", 2.0)):
            metrics = pd.DataFrame({
                "FrameID": range(4), "AbsoluteTime": times,
                METRIC_COLUMNS[METRIC]: np.array(values) * factor,
            })
            movement = pd.DataFrame({
                "FrameID": range(4), "AbsoluteTime": times,
                "valid": True, "moving": True, "bout_id": range(1, 5),
            })
            cycles = pd.DataFrame({"Beg": [1000000] * 4 + [0] + [1000000] * 89})
            fish_frames.append(calculate_fish_heatmaps(
                metrics, movement, cycles, recording_id=fish,
                metric_ids=(METRIC,),
            ))
        fish_bins = pd.concat(fish_frames, ignore_index=True)
        pooled = summarize_equal_fish_signed_log_vigor(
            fish_bins, metric_id=METRIC,
            condition_by_recording={"control-fish": "control", "delay-fish": "delay"},
        )
        for condition in ("control", "delay"):
            row = pooled.loc[
                pooled["condition_id"].eq(condition)
                & pooled["Trial number"].eq(5)
                & pooled["Time bin center (s)"].eq(1.25)
            ].iloc[0]
            self.assertAlmostEqual(row["Mean signed log vigor"], np.log(2.0))
            self.assertEqual(row["Contributing fish"], 1)
            self.assertEqual(row["Fish coverage fraction"], 1.0)
        self.assertTrue(np.isnan(pooled.loc[
            pooled["Trial number"].eq(6), "Mean signed log vigor"
        ]).all())


if __name__ == "__main__":
    unittest.main()
