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
