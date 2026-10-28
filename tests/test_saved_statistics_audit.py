"""Audit checks detect wrong families, reference contrasts and displayed labels."""
import unittest

import numpy as np
import pandas as pd

from classical_conditioning.analysis.review.saved_statistics_audit import adjusted, phase_contrasts, annotation_labels, audit_tables, coefficient_rows, bootstrap_draws


class SavedStatisticsTests(unittest.TestCase):
    def test_unavailable_tests_keep_their_place_in_the_family(self):
        values = [.01, .04, .03, np.nan]
        np.testing.assert_allclose(adjusted(values, 'holm'), [.04, .09, .09, np.nan], equal_nan=True)
        np.testing.assert_allclose(adjusted(values, 'fdr_bh'), [.04, .0533333333333333, .0533333333333333, np.nan], equal_nan=True)

    def test_invalid_p_values_cannot_become_missing_results(self):
        for value in [np.inf, -np.inf, -.1, 1.1]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                adjusted([value], 'holm')

    def test_pre_reference_and_fixed_covariance_have_independent_known_answer(self):
        data = pd.DataFrame({'trial_number': list(range(5, 95)) * 2,
                             'condition_id': ['control'] * 90 + ['trace'] * 90})
        data['pre_trial'] = (data.trial_number - 9.5) / 9
        data['fit_phase'] = 'Pre'
        data['log_baseline'] = .7
        for prefix in ['train', 'test']:
            for i in range(4):
                data[prefix + '_b' + str(i)] = 0.
        terms = ['Intercept', "C(condition_id, Treatment(reference='control'))[T.trace]",
                 'log_baseline', 'pre_trial',
                 "C(condition_id, Treatment(reference='control'))[T.trace]:pre_trial"]
        coef = pd.DataFrame({'term': terms, 'estimate': [.5, .2, .7, .1, -.3]})
        cov = pd.DataFrame(np.eye(5) * .04, index=terms, columns=terms)
        spec = {'formula': "log_response ~ log_baseline + C(condition_id, Treatment(reference='control')) * pre_trial"}
        result = phase_contrasts(data, spec, coef, cov)
        u = (np.arange(5, 95) - 9.5) / 9
        np.testing.assert_allclose(result['condition_estimate'], -.2 + .3 * u)
        np.testing.assert_allclose(result['change_estimate'], .3 * u, atol=1e-15)
        np.testing.assert_allclose(result['condition_se'], np.sqrt(.04 * (1 + u * u)))
        np.testing.assert_allclose(result['change_se'], .2 * abs(u), atol=1e-15)
        with self.assertRaisesRegex(ValueError, 'covariance'):
            phase_contrasts(data, spec, coef, -cov)
        duplicate = data.copy()
        duplicate.loc[90, 'pre_trial'] = 99
        with self.assertRaisesRegex(ValueError, 'varies'):
            phase_contrasts(duplicate, spec, coef, cov)

    def test_duplicate_fish_trials_stop_before_statistics(self):
        data = pd.DataFrame({'fish_key': ['fish', 'fish'], 'trial_number': [5, 5]})
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            audit_tables(data, None, None, None, None, None, None, None, None)

    def test_saved_coefficient_rows_cannot_be_swapped_between_models(self):
        from scipy.stats import norm
        p = 2 * norm.sf(2)
        condition = "C(condition_id, Treatment(reference='control'))[T.trace]"
        d = pd.DataFrame({'block_10_name': ['Train 1'], 'estimate': [.2], 'p_raw': [p]})
        block = pd.DataFrame({'term': [condition + ':C(block_10_name)[T.Train 1]'],
                              'estimate': [.2], 'p_value': [p]})
        local = pd.DataFrame({'block_10_name': ['Train 1'], 'mean_estimate': [.2], 'mean_p_raw': [p],
                              'slope_estimate': [-.2], 'slope_p_raw': [p]})
        coefficients = pd.DataFrame({'term': [condition, condition + ':within_block_trial'],
                                     'estimate': [.2, -.2], 'standard_error': [.1, .1], 'p_value': [p, p]})
        self.assertEqual(coefficient_rows(d, local, block, {'Train 1': coefficients}), 3)
        coefficients.loc[1, 'estimate'] = .2
        with self.assertRaisesRegex(ValueError, 'Local row'):
            coefficient_rows(d, local, block, {'Train 1': coefficients})

    def test_bootstrap_replay_detects_changed_fish_indices_values_and_intervals(self):
        data = pd.DataFrame({'fish_id': ['a'] * 90 + ['b'] * 90,
                             'trial_number': list(range(5, 95)) * 2,
                             'condition_id': ['control'] * 180,
                             'log_baseline': np.zeros(180), 'log_response': [.2] * 90 + [.6] * 90})
        observed = pd.DataFrame({'condition_id': ['control'] * 90, 'trial_number': range(5, 95),
                                 'ci_lower': [.4] * 90, 'ci_upper': [.4] * 90})
        indices = np.tile([0, 1], (5000, 1))
        medians = np.full((5000, 90), .4)
        self.assertEqual(bootstrap_draws(data, observed, {'control': (['a', 'b'], indices, medians)})[0]['saved_resamples_checked'], 5000)
        for ids, ind, values, summary in [(['a', 'a'], indices, medians, observed),
                (['a', 'b'], indices[:100], medians, observed),
                (['a', 'b'], np.zeros_like(indices), medians, observed),
                (['a', 'b'], indices, medians + .1, observed),
                (['a', 'b'], indices, medians, observed.assign(ci_upper=.5))]:
            with self.subTest(ids=ids, shape=ind.shape), self.assertRaises(ValueError):
                bootstrap_draws(data, summary, {'control': (ids, ind, values)})

    def test_named_svg_symbols_and_missing_results_must_match_saved_values(self):
        d = pd.DataFrame({'block_10_name': ['Train 1'], 'p_holm': [.005]})
        local = pd.DataFrame({'block_10_name': ['Test 3'], 'mean_p_fdr': [np.nan],
                              'slope_p_raw': [np.nan], 'slope_p_fdr': [np.nan]})
        trials = pd.DataFrame({'trial_number': [19, 20], 'change_p_fdr': [.01, .5]})
        xml = '<svg xmlns="http://www.w3.org/2000/svg"><g id="panel-G">'
        for key, text in [('D-0', '**'), ('mean_p_fdr-0', 'n/a'), ('slope_p_raw-0', 'n/a'),
                          ('slope_p_fdr-0', 'n/a'), ('trial-19', '*')]:
            xml += '<g id="prefix-fig2__g__stats__' + key + '"><text>' + text + '</text></g>'
        xml += '</g></svg>'
        self.assertEqual(annotation_labels(xml.encode(), 'G', d, local, trials)['trial_markers'], 1)
        for changed in [xml.replace('>**<', '>*<'), xml.replace('>n/a<', '>0<', 1),
                        xml.replace('</g></svg>', '<g id="x-fig2__g__stats__trial-21"><text>*</text></g></g></svg>')]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                annotation_labels(changed.encode(), 'G', d, local, trials)


if __name__ == '__main__':
    unittest.main()
