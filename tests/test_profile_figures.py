from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from classical_conditioning.figures.temporal_profiles import (
    OUTCOME_SPECS,
    _candidate_heatmap_figure,
    _outcome_cmap_name,
    _outcome_scale,
)
from classical_conditioning.figures.theme import (
    DEFAULT_THEME,
    DOUBLE_COLUMN_MM,
    heatmap_cmap,
    mm_to_in,
)


class ProfileFigureTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for metric in (
            "segment_absolute_angular_speed_sum",
            "all_segment_angular_rms",
            "whole_tail_xy_rms_speed",
            "whole_tail_xy_mean_speed",
            "curvature_change_rms",
        ):
            for trial in (1, 2):
                for time in (-0.25, 0.25):
                    rows.append(
                        {
                            "Recording ID": "fish-1",
                            "Trial type": "CS",
                            "Trial number": trial,
                            "Time bin center (s)": time,
                            "Metric ID": metric,
                            "Total activity mean": 1.0,
                            "Movement probability": 0.5,
                            "Fraction time moving": 0.5,
                            "Conditional intensity mean": 2.0,
                            "Bout rate per minute": 3.0,
                            "Valid expected fraction": 1.0,
                            "Detector valid fraction": 1.0,
                        }
                    )
        self.profiles = pd.DataFrame(rows)

    def test_all_outcome_modes_build_from_same_panel_table(self) -> None:
        for outcome_id in OUTCOME_SPECS:
            figure, panel_ids, mappings = _candidate_heatmap_figure(
                self.profiles,
                "CS",
                outcome_id,
            )
            try:
                heatmap_ids = [
                    key for key in mappings if key.startswith("heatmap__")
                ]
                stimulus_ids = [
                    key for key in mappings if key.startswith("stimulus__")
                ]
                self.assertEqual(len(panel_ids), 6)
                self.assertEqual(panel_ids[-1], "colorbar")
                self.assertEqual(len(heatmap_ids), 5)
                self.assertEqual(len(stimulus_ids), 5)
                self.assertIn("colorbar", mappings)
                colorbar_axes = [
                    axis for axis in figure.axes if not axis.images
                ]
                self.assertEqual(len(colorbar_axes), 1)
                self.assertLessEqual(
                    figure.get_size_inches()[0],
                    mm_to_in(DOUBLE_COLUMN_MM)[0] + 0.05,
                )
                expected_cmap = heatmap_cmap(
                    _outcome_cmap_name(OUTCOME_SPECS[outcome_id], DEFAULT_THEME),
                    DEFAULT_THEME,
                )
                image_cmaps = {
                    axis.images[0].get_cmap().name
                    for axis in figure.axes
                    if axis.images
                }
                self.assertEqual(image_cmaps, {expected_cmap.name})
                self.assertTrue(
                    all(
                        mapping.get("value_field")
                        == OUTCOME_SPECS[outcome_id]["column"]
                        for key, mapping in mappings.items()
                        if key.startswith("heatmap__")
                    )
                )
                self.assertTrue(
                    all("duration_s" in mappings[key] for key in stimulus_ids)
                )
            finally:
                __import__("matplotlib.pyplot").pyplot.close(figure)

    def test_low_coverage_is_masked_in_rendered_array(self) -> None:
        profiles = self.profiles.copy()
        profiles.loc[
            (profiles["Metric ID"] == "segment_absolute_angular_speed_sum")
            & (profiles["Trial number"] == 1)
            & np.isclose(profiles["Time bin center (s)"], -0.25),
            "Detector valid fraction",
        ] = 0.5
        figure, _, _ = _candidate_heatmap_figure(
            profiles,
            "CS",
            "movement-probability",
        )
        try:
            array = figure.axes[0].images[0].get_array()
            self.assertTrue(np.ma.getmaskarray(array)[0, 0])
        finally:
            __import__("matplotlib.pyplot").pyplot.close(figure)

    def test_probability_uses_fixed_zero_to_one_scale_and_metadata(self) -> None:
        figure, _, mappings = _candidate_heatmap_figure(
            self.profiles,
            "CS",
            "movement-probability",
        )
        try:
            image = figure.axes[0].images[0]
            self.assertEqual(image.get_clim(), (0.0, 1.0))
            heatmap_mapping = next(
                mapping
                for key, mapping in mappings.items()
                if key.startswith("heatmap__")
            )
            self.assertEqual(
                heatmap_mapping["display_scale"],
                "linear, fixed [0, 1]",
            )
        finally:
            __import__("matplotlib.pyplot").pyplot.close(figure)

    def test_quantile_scale_reports_actual_limits(self) -> None:
        values = np.array([0.0, 1.0, 2.0, 100.0])
        vmin, vmax, description = _outcome_scale(
            values,
            OUTCOME_SPECS["total-activity"],
        )
        self.assertEqual(vmin, 0.0)
        self.assertAlmostEqual(vmax, float(np.quantile(values, 0.99)))
        self.assertIn(f"vmax={vmax:g}", description)


if __name__ == "__main__":
    unittest.main()
