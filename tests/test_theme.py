from __future__ import annotations

import unittest

import matplotlib as mpl
import matplotlib.pyplot as plt

from classical_conditioning.config.domain import (
    Alignment,
    ConditionRole,
    ConditionSpec,
)
from classical_conditioning.figures.theme import (
    DEFAULT_THEME,
    DOUBLE_COLUMN_MM,
    apply_theme,
    condition_color,
    heatmap_cmap,
    mm_to_in,
    rgb_255_to_unit,
    style_axes,
    stimulus_duration_s,
)


class FigureThemeTests(unittest.TestCase):
    def test_rgb_converts_with_255(self) -> None:
        self.assertEqual(rgb_255_to_unit((0, 174, 239)), (0 / 255, 174 / 255, 239 / 255))
        self.assertNotEqual(rgb_255_to_unit((0, 174, 239))[1], 174 / 256)

    def test_condition_color_uses_spec_rgb_255(self) -> None:
        spec = ConditionSpec(
            condition_id="delay",
            display_name="Delay",
            source_name="delay",
            role=ConditionRole.CONDITIONED,
            color_rgb_255=(236, 0, 140),
        )
        self.assertEqual(condition_color(spec), rgb_255_to_unit((236, 0, 140)))

    def test_cs_and_us_tokens_match_legacy_rgb(self) -> None:
        self.assertEqual(DEFAULT_THEME.cs_color, rgb_255_to_unit((13, 129, 54)))
        self.assertEqual(DEFAULT_THEME.us_color, rgb_255_to_unit((112, 46, 120)))

    def test_march_single_fish_scaled_vigor_style(self) -> None:
        self.assertEqual(DEFAULT_THEME.single_fish_scaled_vigor_cmap, "managua_r")
        self.assertEqual(DEFAULT_THEME.single_fish_scaled_vigor_vmin, -0.25)
        self.assertEqual(DEFAULT_THEME.single_fish_scaled_vigor_vmax, 0.25)
        self.assertEqual(
            heatmap_cmap(DEFAULT_THEME.single_fish_scaled_vigor_cmap).name,
            "managua_r",
        )

    def test_stimulus_duration_defaults(self) -> None:
        self.assertEqual(stimulus_duration_s(Alignment.CS), 10.0)
        self.assertEqual(stimulus_duration_s("US"), 0.1)

    def test_apply_theme_sets_publication_rcparams(self) -> None:
        apply_theme()
        self.assertFalse(mpl.rcParams["axes.spines.top"])
        self.assertFalse(mpl.rcParams["axes.spines.right"])
        self.assertEqual(mpl.rcParams["svg.fonttype"], "none")
        self.assertEqual(mpl.rcParams["figure.dpi"], DEFAULT_THEME.figure_dpi)
        self.assertEqual(mpl.rcParams["font.sans-serif"], ["DejaVu Sans"])
        self.assertFalse(mpl.rcParams["axes.grid"])
        self.assertEqual(mpl.rcParams["xtick.direction"], "out")

    def test_double_column_width_in_inches(self) -> None:
        width_in, height_in = mm_to_in(DOUBLE_COLUMN_MM, 130.0)
        self.assertAlmostEqual(width_in, DOUBLE_COLUMN_MM / 25.4)
        self.assertAlmostEqual(height_in, 130.0 / 25.4)

    def test_axis_spacing_and_type_are_fixed_in_points(self) -> None:
        theme = apply_theme()
        figure, axis = plt.subplots(figsize=mm_to_in(DOUBLE_COLUMN_MM, 90))
        try:
            style_axes(axis, theme=theme, xlabel="Time (s)", ylabel="Activity")
            figure.canvas.draw()
            self.assertEqual(axis.spines["left"].get_position(), ("outward", 0.0))
            self.assertEqual(axis.spines["bottom"].get_position(), ("outward", 0.0))
            for tick in (*axis.xaxis.get_major_ticks(), *axis.yaxis.get_major_ticks()):
                self.assertEqual(tick.get_pad(), theme.tick_major_pad)
                self.assertEqual(tick.tick1line.get_markersize(), theme.tick_major_size)
                self.assertEqual(tick.label1.get_fontsize(), theme.tick_labelsize)
            self.assertEqual(axis.xaxis.labelpad, theme.axes_labelpad)
            self.assertEqual(axis.yaxis.labelpad, theme.axes_labelpad)
            self.assertEqual(axis.xaxis.label.get_fontsize(), theme.axes_labelsize)
            self.assertEqual(axis.yaxis.label.get_fontsize(), theme.axes_labelsize)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
