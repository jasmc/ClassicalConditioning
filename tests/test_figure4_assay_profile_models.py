"""Known-answer checks for assay bounds and whole-structure bootstrap restart."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from classical_conditioning.analysis.review import figure4_profiles as delay
from classical_conditioning.analysis.review import figure4_assay_profiles as assay


class AssayProfileTests(unittest.TestCase):
    def test_explicit_delay_bounds_reproduce_original_adapter(self):
        rng = np.random.default_rng(29)
        rows = []
        for g, group in enumerate(assay.GROUPS):
            for fish in range(6):
                offset = rng.normal(0, .2)
                for t in np.arange(-4.75, 12, .5):
                    rows.append(dict(fish_id=f'{g}-{fish}', display_group=group,
                                     condition_id='control' if g == 0 else 'trace',
                                     time_s=t, signed=offset + g*.03*t + rng.normal(0, .1)))
        frame = pd.DataFrame(rows)
        spec = assay.ProfileSpec(-5, 12)
        original, current = delay.model_data(frame), assay.model_data(frame, spec)
        pd.testing.assert_frame_equal(original, current)
        old, _ = delay.fit(original, random='1')
        new, history = assay.fit(current, random='1', spec=spec)
        self.assertIsNotNone(new, history)
        np.testing.assert_allclose(old.fe_params, new.fe_params, rtol=1e-9, atol=1e-10)
        times = np.arange(.25, 9, .5)
        noise = next(x['residual_variance'] for x in reversed(history) if x['diagnostic_status'] == 'ok')
        p1 = delay.individual_predictions(old, original, times, '1', noise)
        p2 = assay.individual_predictions(new, current, times, '1', noise, spec)
        for fish in p1:
            np.testing.assert_allclose(p1[fish], p2[fish], rtol=1e-8)

    def test_trace_grid_keeps_extra_support_and_missing_intensity(self):
        frame = pd.DataFrame({'fish_id': ['a']*5, 'display_group': [assay.GROUPS[0]]*5,
                              'time_s': [-5, 12.75, 13.25, 15.75, 16], 'signed': [1, np.nan, 0, .2, 1]})
        result = assay.model_data(frame)
        self.assertEqual(result.time_s.tolist(), [13.25, 15.75])
        np.testing.assert_allclose(result.trial_scaled, result.time_s/16)
        self.assertIn('upper_bound=16.0', assay.TRACE3.formula)
        with self.assertRaises(ValueError): assay.ProfileSpec(2, 1)

    def test_unstable_structure_restarts_seed_and_never_mixes_draws(self):
        data = pd.DataFrame({'fish_id': list('abcdef'),
                             'display_group': np.repeat(assay.GROUPS, 2)})
        calls = {'1 + trial_scaled': 0, '1': 0}
        weights = {'1 + trial_scaled': [], '1': []}
        def fitted(data, random, prepared=None, weights=None, spec=None):
            calls[random] += 1
            ok = not (random == '1 + trial_scaled' and calls[random] == 2)
            value = 11 if random == '1 + trial_scaled' else 22
            if weights is not None:
                saved_weights[random].append(weights.copy())
            return (SimpleNamespace(fe_params=pd.Series([value])) if ok else None), [{'diagnostic_status': 'ok' if ok else 'failed'}]
        saved_weights = weights
        with patch.object(assay, 'fit', side_effect=fitted), \
             patch.object(assay, 'prepare_fast', return_value=SimpleNamespace(keys=list('abcdef'))), \
             patch.object(assay, 'prediction_design', return_value=(None, np.ones((3, 1)))):
            _, _, estimates, draws, diagnostics = assay.fit_bootstrap(data, [.25], repetitions=10)
        self.assertEqual(calls, {'1 + trial_scaled': 11, '1': 11})
        self.assertEqual(diagnostics['random'], '1')
        self.assertEqual([s['accepted'] for s in diagnostics['structure_comparison']], [9, 10])
        self.assertTrue(np.all(draws == 22))
        self.assertTrue(np.all(estimates == 22))
        np.testing.assert_array_equal(saved_weights['1 + trial_scaled'], saved_weights['1'])
        self.assertTrue(all(s['full_fit_attempts'] for s in diagnostics['structure_comparison']))

    def test_trace_slope_predictions_match_dense_gaussian_conditioning(self):
        rows = [dict(fish_id=str(g), condition_id='control' if g == 0 else 'trace',
                     display_group=group, time_s=t, signed=y+g*.1)
                for g, group in enumerate(assay.GROUPS) for t, y in [(-2., .2), (14., -.3)]]
        data = assay.model_data(pd.DataFrame(rows))
        covariance = np.array([[.1, .02], [.02, .3]])
        result = SimpleNamespace(fe_params=pd.Series([0.]), cov_re=covariance,
                                 model=SimpleNamespace(exog=np.zeros((len(data), 1))))
        times = np.array([0., 8., 15.])
        with patch.object(assay, 'prediction_design', return_value=(None, np.zeros((9, 1)))):
            predicted = assay.individual_predictions(result, data, times, '1 + trial_scaled', .05)
        for fish, part in data.groupby('fish_id'):
            z = np.column_stack([np.ones(len(part)), part.time_s/16])
            gamma = covariance @ z.T @ np.linalg.solve(z @ covariance @ z.T + .05*np.eye(len(part)), part.signed)
            expected = np.column_stack([np.ones(len(times)), times/16]) @ gamma
            np.testing.assert_allclose(predicted[fish], expected, rtol=1e-12, atol=1e-12)


if __name__ == '__main__': unittest.main()
