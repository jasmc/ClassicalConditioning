from __future__ import annotations

import unittest

import matplotlib as mpl

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
    mm_to_in,
    rgb_255_to_unit,
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

    def test_stimulus_duration_defaults(self) -> None:
        self.assertEqual(stimulus_duration_s(Alignment.CS), 10.0)
        self.assertEqual(stimulus_duration_s("US"), 0.1)

    def test_apply_theme_sets_publication_rcparams(self) -> None:
        apply_theme()
        self.assertFalse(mpl.rcParams["axes.spines.top"])
        self.assertFalse(mpl.rcParams["axes.spines.right"])
        self.assertEqual(mpl.rcParams["svg.fonttype"], "none")
        self.assertEqual(mpl.rcParams["figure.dpi"], DEFAULT_THEME.figure_dpi)
        self.assertFalse(mpl.rcParams["axes.grid"])
        self.assertEqual(mpl.rcParams["xtick.direction"], "out")

    def test_double_column_width_in_inches(self) -> None:
        width_in, height_in = mm_to_in(DOUBLE_COLUMN_MM, 130.0)
        self.assertAlmostEqual(width_in, DOUBLE_COLUMN_MM / 25.4)
        self.assertAlmostEqual(height_in, 130.0 / 25.4)


if __name__ == "__main__":
    unittest.main()
