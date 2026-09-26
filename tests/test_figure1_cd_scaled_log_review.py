"""Check the explicit baseline boundaries and missing-bin behavior for D."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from render_figure1_cd_scaled_log_review import (  # noqa: E402
    half_second_bin_mean, scale_trial_log_vigor,
)
from classical_conditioning.figures.paper_panels import build_render_plan  # noqa: E402


class ScaledLogVigorReviewTests(unittest.TestCase):
    def test_paper_plan_uses_scaled_review_and_no_x_zoom(self) -> None:
        steps = build_render_plan(Path("project"), Path("review"),
                                  figure_set="figure1")
        traces = [step for step in steps if step.name.startswith("figure1-cd-")]
        self.assertEqual(len(traces), 2)
        for step in traces:
            self.assertEqual(step.panels, ("fig-1C", "fig-1D"))
            self.assertIn("scripts/render_figure1_cd_scaled_log_review.py", step.argv)
            self.assertNotIn("--window-start", step.argv)
            self.assertNotIn("--window-end", step.argv)

    def test_short_baseline_excludes_earlier_frames(self) -> None:
        time = np.array([-20.0, -16.0, -14.0, -10.0, -5.0, -1.0, 1.0])
        vigor = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        valid = np.ones(len(time), dtype=bool)
        moving = np.zeros(len(time), dtype=bool)
        bouts = np.zeros(len(time), dtype=int)
        short, _, _, short_count = scale_trial_log_vigor(
            vigor, time, valid, moving, bouts,
            baseline_start_s=-15, moving_bouts_only=False,
        )
        long, _, _, long_count = scale_trial_log_vigor(
            vigor, time, valid, moving, bouts,
            baseline_start_s=-20, moving_bouts_only=False,
        )
        self.assertEqual(short_count, 4)
        self.assertEqual(long_count, 6)
        self.assertFalse(np.allclose(short, long))

    def test_missing_bout_data_and_empty_bins_stay_missing(self) -> None:
        time = np.array([-15.0, -14.5, -10.0, -5.0, -1.0, 1.0])
        vigor = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        valid = np.ones(len(time), dtype=bool)
        moving = np.array([True, True, True, True, False, True])
        bouts = np.array([1, 2, 3, 4, 0, 5])
        scaled, _, _, count = scale_trial_log_vigor(
            vigor, time, valid, moving, bouts,
            baseline_start_s=-15, moving_bouts_only=True,
        )
        self.assertEqual(count, 4)
        self.assertTrue(np.isnan(scaled[4]))
        binned = half_second_bin_mean(time, scaled)
        self.assertTrue(np.isfinite(binned[10]))  # −15 to −14.5 s
        self.assertTrue(np.isnan(binned[12]))  # no frame in −14 to −13.5 s


if __name__ == "__main__":
    unittest.main()
