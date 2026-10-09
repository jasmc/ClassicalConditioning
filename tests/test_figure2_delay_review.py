"""Check fish-resampling semantics and declared multiplicity families."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location('delay_review', Path(__file__).resolve().parents[1] / 'scripts/render_figure2_delay_legacy_metric_lme.py')
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)


class BootstrapSemanticsTest(unittest.TestCase):
    def make_fish(self):
        rows = []
        for condition, offset in [('control', 0), ('delay', 10)]:
            for fish in range(3):
                for trial in range(5, 95):
                    ratio = float(offset + fish + 1) * (trial - 4)
                    if fish == 0 and trial == 80:
                        ratio = np.nan
                    rows.append(dict(condition_id=condition, fish_id=f'{condition}-{fish}', trial_number=trial, ratio=ratio))
        return pd.DataFrame(rows)

    def test_same_fish_selection_at_every_trial_and_missingness_retained(self):
        data = self.make_fish()
        summary, draws = review.bootstrap_trajectories(data, n_boot=200, seed=10)
        for condition, draw in draws.items():
            # Every selected fish doubles its value from trial5 to6; so must each
            # bootstrap median. Independent per-trial draws would break this.
            np.testing.assert_allclose(draw['medians'][:, 1], 2 * draw['medians'][:, 0])
            wide = data.loc[data.condition_id.eq(condition)].pivot(index='fish_id', columns='trial_number', values='ratio').sort_index()
            selected = wide.to_numpy()[draw['indices'][7]]
            np.testing.assert_allclose(draw['medians'][7], np.nanmedian(selected, axis=0), equal_nan=True)
            self.assertTrue(all(fid.startswith(condition) for fid in draw['fish_ids']))
            self.assertEqual(summary.loc[summary.condition_id.eq(condition) & summary.trial_number.eq(80), 'contributing_fish'].iloc[0], 2)

    def test_seed_and_row_order_reproducibility(self):
        data = self.make_fish()
        a, da = review.bootstrap_trajectories(data, n_boot=200, seed=10)
        b, db = review.bootstrap_trajectories(data.sample(frac=1, random_state=4), n_boot=200, seed=10)
        pd.testing.assert_frame_equal(a, b)
        for condition in da:
            np.testing.assert_array_equal(da[condition]['indices'], db[condition]['indices'])
            np.testing.assert_allclose(da[condition]['medians'], db[condition]['medians'], equal_nan=True)

    def test_failed_test_does_not_shrink_correction_family(self):
        frame = pd.DataFrame({'p': [.01, np.nan, .04]})
        review.adjust_family(frame, 'p', 'holm', 'holm')
        self.assertAlmostEqual(frame.holm.iloc[0], .03)
        self.assertTrue(np.isnan(frame.holm.iloc[1]))
        self.assertAlmostEqual(frame.holm.iloc[2], .08)


if __name__ == '__main__':
    unittest.main()
