from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classical_conditioning.figures.temporal_profiles import (
    BOUT_OUTCOME_PANELS,
    FIGURE_SPECS,
    METRIC_LABELS,
    _candidate_heatmap_figure,
    _panel_cmap_name,
    _panel_scale,
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
        for index, metric in enumerate(METRIC_LABELS):
            for trial in (1, 2):
                for time in (-0.25, 0.25):
                    rows.append(
                        {
                            "Recording ID": "fish-1",
                            "Trial type": "CS",
                            "Trial number": trial,
                            "Time bin center (s)": time,
                            "Metric ID": metric,
                            # Raw intensity differs per metric; bout outcomes
                            # do not, because one detector produced them.
                            "Total activity mean": 1.0 + index,
                            "Scaled total activity": 0.5,
                            "Conditional intensity mean": 2.0 + index,
                            "Movement probability": 0.5,
                            "Fraction time moving": 0.5,
                            "Bout rate per minute": 3.0,
                            "Valid expected fraction": 1.0,
                            "Detector valid fraction": 1.0,
                        }
                    )
        self.profiles = pd.DataFrame(rows)

    def test_every_figure_builds_from_the_same_panel_table(self) -> None:
        for figure_id, spec in FIGURE_SPECS.items():
            figure, panel_ids, mappings = _candidate_heatmap_figure(
                self.profiles,
                "CS",
                figure_id,
            )
            try:
                heatmap_ids = [
                    key for key in mappings if key.startswith("heatmap__")
                ]
                stimulus_ids = [
                    key for key in mappings if key.startswith("stimulus__")
                ]
                self.assertEqual(len(heatmap_ids), len(spec.panels))
                self.assertEqual(len(stimulus_ids), len(spec.panels))
                self.assertLessEqual(
                    figure.get_size_inches()[0],
                    mm_to_in(DOUBLE_COLUMN_MM)[0] + 0.05,
                )
                self.assertTrue(
                    all("duration_s" in mappings[key] for key in stimulus_ids)
                )
                for panel in spec.panels:
                    mapping = next(
                        value
                        for key, value in mappings.items()
                        if key.startswith("heatmap__")
                        and key.endswith(f"{panel.key}__{figure_id}")
                    )
                    self.assertEqual(mapping["value_field"], panel.column)
                    expected_cmap = heatmap_cmap(
                        _panel_cmap_name(panel, DEFAULT_THEME),
                        DEFAULT_THEME,
                    )
                    self.assertEqual(
                        mapping["cmap"],
                        expected_cmap.name,
                    )
            finally:
                plt.close(figure)

    def test_intensity_figures_have_one_row_per_metric(self) -> None:
        for figure_id in (
            "total-activity-raw",
            "total-activity-scaled",
            "conditional-intensity-raw",
        ):
            spec = FIGURE_SPECS[figure_id]
            self.assertEqual(len(spec.panels), len(METRIC_LABELS))
            self.assertEqual(
                [panel.metric_id for panel in spec.panels],
                list(METRIC_LABELS),
            )

    def test_bout_figure_is_metric_free_with_three_outcome_rows(self) -> None:
        spec = FIGURE_SPECS["bout-outcomes"]
        self.assertEqual(len(spec.panels), 3)
        self.assertTrue(all(panel.metric_id is None for panel in spec.panels))
        self.assertEqual(
            [panel.column for panel in spec.panels],
            [
                "Movement probability",
                "Fraction time moving",
                "Bout rate per minute",
            ],
        )
        figure, _, mappings = _candidate_heatmap_figure(
            self.profiles,
            "CS",
            "bout-outcomes",
        )
        try:
            # Exactly three heatmaps, not three per metric.
            images = [axis.images[0] for axis in figure.axes if axis.images]
            self.assertEqual(len(images), 3)
            self.assertTrue(
                all(
                    mapping["shared_detector"] == "true"
                    for key, mapping in mappings.items()
                    if key.startswith("heatmap__")
                )
            )
        finally:
            plt.close(figure)

    def test_raw_figures_get_one_colorbar_per_row(self) -> None:
        figure, panel_ids, mappings = _candidate_heatmap_figure(
            self.profiles,
            "CS",
            "total-activity-raw",
        )
        try:
            colorbar_keys = [
                key for key in mappings if key.startswith("colorbar__")
            ]
            self.assertEqual(len(colorbar_keys), len(METRIC_LABELS))
            self.assertTrue(
                all(
                    mappings[key]["shared"] == "false" for key in colorbar_keys
                )
            )
            # Raw panels carry different units, so their limits differ.
            limits = {
                axis.images[0].get_clim() for axis in figure.axes if axis.images
            }
            self.assertGreater(len(limits), 1)
            self.assertNotIn("colorbar", panel_ids)
        finally:
            plt.close(figure)

    def test_scaled_figure_shares_one_fixed_zero_to_one_colorbar(self) -> None:
        figure, panel_ids, mappings = _candidate_heatmap_figure(
            self.profiles,
            "CS",
            "total-activity-scaled",
        )
        try:
            self.assertIn("colorbar", mappings)
            self.assertEqual(mappings["colorbar"]["shared"], "true")
            self.assertEqual(panel_ids[-1], "colorbar")
            for axis in figure.axes:
                if axis.images:
                    self.assertEqual(axis.images[0].get_clim(), (0.0, 1.0))
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
            plt.close(figure)

    def test_low_coverage_is_masked_in_rendered_array(self) -> None:
        profiles = self.profiles.copy()
        profiles.loc[
            (profiles["Trial number"] == 1)
            & np.isclose(profiles["Time bin center (s)"], -0.25),
            "Detector valid fraction",
        ] = 0.5
        figure, _, _ = _candidate_heatmap_figure(
            profiles,
            "CS",
            "bout-outcomes",
        )
        try:
            array = figure.axes[0].images[0].get_array()
            self.assertTrue(np.ma.getmaskarray(array)[0, 0])
        finally:
            plt.close(figure)

    def test_quantile_scale_reports_actual_limits(self) -> None:
        values = np.array([0.0, 1.0, 2.0, 100.0])
        vmin, vmax, description = _panel_scale(
            values,
            FIGURE_SPECS["total-activity-raw"].panels[0],
        )
        self.assertEqual(vmin, 0.0)
        self.assertAlmostEqual(vmax, float(np.quantile(values, 0.99)))
        self.assertIn(f"vmax={vmax:g}", description)

    def test_fixed_probability_scale_is_zero_to_one(self) -> None:
        vmin, vmax, description = _panel_scale(
            np.array([0.1, 0.2]),
            BOUT_OUTCOME_PANELS[0],
        )
        self.assertEqual((vmin, vmax), (0.0, 1.0))
        self.assertEqual(description, "linear, fixed [0, 1]")


if __name__ == "__main__":
    unittest.main()
