import unittest
import numpy as np
from matplotlib import colormaps
from baseline_colour_mapping import baseline_centred_options, baseline_colour_norm


class BaselineColourTests(unittest.TestCase):
    def assert_midpoint(self, values, mask):
        result, meta = baseline_centred_options(values, mask)
        for name, half_range in [('log', .25), ('log', .34162552), ('C', 1), ('D', 1)]:
            signal = result[name]
            base = signal[np.asarray(mask, bool) & np.isfinite(signal)]
            if not len(base):
                continue
            norm = baseline_colour_norm(half_range)
            self.assertAlmostEqual(float(np.median(base)), 0)
            self.assertAlmostEqual(float(np.median(np.asarray(norm(base)))), .5)
            np.testing.assert_array_equal(colormaps['managua_r'](norm(0)),
                                          colormaps['managua_r'](.5))
        return result, meta

    def test_odd_and_even_skewed_baselines_after_clipping(self):
        for base in [[-3, -1, 1, 20], [-3, -.2, 0, .1, 20], [1, 4]]:
            values = np.r_[base, -100, 100, np.nan]
            mask = np.r_[np.ones(len(base), bool), False, False, False]
            result, _ = self.assert_midpoint(values, mask)
            self.assertTrue(np.isnan(result['C'][-1]))
            self.assertEqual(result['C'][-2], 1)
            self.assertEqual(result['D'][-3], -1)

    def test_boundary_selection_and_post_cs_values_do_not_change_reference(self):
        times = np.array([-15, -14.5, -.5, 0, .5])
        values = np.array([1., 2., 5., 1000., -1000.])
        mask = (times >= -15) & (times < 0)
        result, meta = self.assert_midpoint(values, mask)
        self.assertEqual(meta['baseline_count'], 3)
        self.assertEqual(meta['reference'], 2)
        self.assertEqual(result['log'][3], 998)

    def test_constant_and_single_baseline_have_log_reference_but_no_range(self):
        for values, mask in [([1, 1, 9], [1, 1, 0]), ([1, 9, np.nan], [1, 0, 0])]:
            result, meta = self.assert_midpoint(values, mask)
            self.assertEqual(meta['status'], 'undefined quantile range')
            self.assertTrue(np.isnan(result['C']).all() and np.isnan(result['D']).all())
            self.assertTrue(np.isfinite(result['log'][1]))

    def test_invalid_values_and_missing_baseline_remain_missing(self):
        result, meta = baseline_centred_options([np.inf, np.nan, 10], [1, 1, 0])
        self.assertEqual(meta['status'], 'missing baseline')
        self.assertTrue(all(np.isnan(v).all() for v in result.values()))
        result, _ = self.assert_midpoint([1, 2, 3, np.inf], [1, 1, 1, 0])
        self.assertTrue(all(np.isnan(v[-1]) for v in result.values()))

    def test_quantile_options_are_translation_and_positive_scale_invariant(self):
        values = np.array([-3., -1., 1., 20., 100.])
        mask = np.array([1, 1, 1, 1, 0], bool)
        r, _ = baseline_centred_options(values, mask)
        s, _ = baseline_centred_options(3*values+17, mask)
        for name in ['C', 'D']:
            np.testing.assert_allclose(r[name], s[name])

    def test_invalid_colour_ranges_are_rejected(self):
        for half_range in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                baseline_colour_norm(half_range)


if __name__ == '__main__':
    unittest.main()
