"""Scientific annotation semantics: joint intervals, coverage and scheduled time."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('g_presentation', Path(__file__).resolve().parents[1] / 'scripts/paper/figure_02/panel_g/build_exploratory_presentation.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


class AsteriskSemantics(unittest.TestCase):
    def test_touching_zero_missing_bounds_and_inadequate_coverage_are_unmarked(self):
        curve = pd.DataFrame({'trial_number': [20, 21, 22, 23, 24, 25],
                              'simultaneous_lower': [.01, -.2, 0, np.nan, .02, -.1],
                              'simultaneous_upper': [.1, -.01, .1, .2, .1, 0],
                              'coverage_adequate': [True, True, True, True, False, True]})
        phase = pd.DataFrame({'contrast': ['late_training', 'early_testing'],
                              'simultaneous_lower': [.01, -.1], 'simultaneous_upper': [.2, .2]})
        self.assertEqual(g.star_decisions(curve, phase), ([20, 21], ['late_training']))

    def test_bracket_cannot_bridge_a_missing_scheduled_trial_or_phase_boundary(self):
        self.assertEqual(g.star_ranges([13, 14, 15, 16, 18, 63, 64, 65, 66]),
                         [[13, 14], [15, 16], [18, 18], [63, 64], [65, 66]])
        self.assertEqual(g.star_ranges([]), [])


if __name__ == '__main__':
    unittest.main()
