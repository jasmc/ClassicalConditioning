"""Independent arithmetic checks on saved statistics; never fit or select models."""
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import kurtosis, norm
from statsmodels.stats.multitest import multipletests


def adjusted(values, method):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if np.isinf(values).any() or np.any((values[finite] < 0) | (values[finite] > 1)):
        raise ValueError('Invalid saved p-value')
    # Unavailable tests retain a place in the declared family, never a rejection.
    result = multipletests(np.where(finite, values, 1), method=method)[1]
    result[~finite] = np.nan
    return result


def phase_contrasts(data, diagnostics, coefficients, covariance):
    """Rebuild population design contrasts from saved bases/coefficients only."""
    columns = ['fit_phase', 'pre_trial', *['train_b' + str(i) for i in range(4)],
               *['test_b' + str(i) for i in range(4)]]
    for column in columns:
        if data.groupby('trial_number', observed=True)[column].nunique().max() != 1:
            raise ValueError('Saved phase basis varies within a scheduled trial')
    cases = data.sort_values('trial_number').drop_duplicates('trial_number').copy()
    if cases.trial_number.tolist() != list(range(5, 95)):
        raise ValueError('Incomplete scheduled phase design')
    conditions = sorted(data.condition_id.unique())
    if len(conditions) != 2 or conditions[0] != 'control':
        raise ValueError('Expected one control and one experimental condition')
    matrices = []
    terms = coefficients.term.tolist()
    for condition in conditions:
        rows = cases.copy()
        rows['condition_id'] = pd.Categorical([condition] * len(rows), categories=conditions)
        rows['log_baseline'] = data.log_baseline.mean()
        design = dmatrix(diagnostics['formula'].split('~', 1)[1], rows, return_type='dataframe')
        if set(design.columns) != set(terms):
            raise ValueError('Saved formula and coefficient terms disagree')
        matrices.append(design[terms].to_numpy())
    contrast = matrices[0] - matrices[1]
    change = contrast - contrast[:10].mean(axis=0)
    beta = coefficients.estimate.to_numpy()
    cov = covariance.loc[terms, terms].to_numpy()
    if not np.allclose(cov, cov.T) or np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError('Invalid saved fixed-effect covariance')
    output = {}
    for label, matrix in [('condition', contrast), ('change', change)]:
        estimate = matrix @ beta
        variance = np.einsum('ij,jk,ik->i', matrix, cov, matrix)
        if np.any(variance < -1e-12):
            raise ValueError('Negative contrast variance')
        se = np.sqrt(np.maximum(variance, 0))
        output.update({label + '_estimate': estimate, label + '_se': se,
                       label + '_lower': estimate - norm.ppf(.975) * se,
                       label + '_upper': estimate + norm.ppf(.975) * se,
                       label + '_p_raw': 2 * norm.sf(np.divide(abs(estimate), se,
                           out=np.full_like(se, np.nan), where=se > 0))})
    return output


def star_text(p):
    return 'n/a' if not np.isfinite(p) else '*' * sum(p < t for t in [.05, .01, .001])


def coefficient_rows(dtests, local, block_coefficients, local_coefficients):
    checked = 0
    for _, row in dtests.iterrows():
        term = block_coefficients.loc[block_coefficients.term.str.contains('condition_id', regex=False)
            & block_coefficients.term.str.contains('[T.' + row.block_10_name + ']', regex=False)
            & block_coefficients.term.str.contains(':', regex=False)]
        if len(term) != 1 or not np.allclose([row.estimate, row.p_raw], term[['estimate', 'p_value']].iloc[0], atol=1e-10):
            raise ValueError('D row differs from saved block coefficient')
        checked += 1
    for _, row in local.iterrows():
        if not np.isfinite(row.mean_p_raw):
            continue
        coef = local_coefficients[row.block_10_name]
        condition = coef.term.str.contains('condition_id', regex=False)
        interaction = coef.term.str.contains(':', regex=False)
        for prefix, subset in [('mean', coef.loc[condition & ~interaction]), ('slope', coef.loc[condition & interaction])]:
            if len(subset) != 1:
                raise ValueError('Ambiguous local condition coefficient')
            expected = subset[['estimate', 'p_value']].iloc[0].to_numpy()
            if not np.allclose([row[prefix + '_estimate'], row[prefix + '_p_raw']], expected, atol=1e-10):
                raise ValueError('Local row differs from saved coefficient')
            expected_p = 2 * norm.sf(abs(subset.estimate.iloc[0] / subset.standard_error.iloc[0]))
            if not np.isclose(expected_p, row[prefix + '_p_raw'], atol=1e-10):
                raise ValueError('Local coefficient p-value does not match Wald calculation')
            checked += 1
    return checked


def bootstrap_draws(data, observed, arrays):
    """Replay saved indices, never draw a new sample or fit a model."""
    checks = []
    for condition, (fish_ids, indices, medians) in arrays.items():
        selected = data.loc[data.condition_id.eq(condition)].copy()
        selected['difference'] = selected.log_response - selected.log_baseline
        matrix = selected.pivot(index='fish_id', columns='trial_number', values='difference')
        if len(set(fish_ids)) != len(fish_ids) or set(matrix.index) != set(fish_ids):
            raise ValueError('Bootstrap fish identities disagree with model input')
        matrix = matrix.reindex(index=fish_ids, columns=range(5, 95)).to_numpy()
        if indices.shape != (5000, len(fish_ids)) or medians.shape != (5000, 90) or not np.issubdtype(indices.dtype, np.integer) or indices.min() < 0 or indices.max() >= len(fish_ids):
            raise ValueError('Invalid saved bootstrap indices or family size')
        for start in range(0, len(indices), 100):
            values = np.nanmedian(matrix[indices[start:start + 100]], axis=1)
            if not np.allclose(values, medians[start:start + 100], atol=1e-12, equal_nan=True):
                raise ValueError('Bootstrap medians differ from replayed saved indices')
        expected = np.nanquantile(medians, [.025, .975], axis=0)
        summary = observed.loc[observed.condition_id.eq(condition)].sort_values('trial_number')
        if not np.allclose(summary[['ci_lower', 'ci_upper']].to_numpy().T, expected, atol=1e-12, equal_nan=True):
            raise ValueError('Observed interval differs from saved bootstrap percentiles')
        checks.append({'condition': condition, 'saved_resamples_checked': 5000,
                       'whole_fish_indices_and_missingness_preserved': True,
                       'pointwise_percentile_intervals': 'passed'})
    return checks


def annotation_labels(svg, panel, dtests, local, trials):
    """Check named SVG statistical labels against their documented saved rows.

    This establishes XML symbol content/association, not painted visibility.
    """
    root = ET.fromstring(svg)
    panels = [n for n in root.iter() if n.get('id') == 'panel-' + panel]
    if len(panels) != 1:
        raise ValueError('Missing or ambiguous selected panel')
    group = panels[0]
    expected = {}
    for index, row in dtests.iterrows():
        key = 'D-' + (str(index) if panel == 'G' else row.block_10_name.replace(' ', '-'))
        text = star_text(row.p_holm)
        if text:
            expected[key] = text
    for index, row in local.iterrows():
        for field, label in [('mean_p_fdr', 'M'), ('slope_p_raw', 'R-raw'), ('slope_p_fdr', 'R-FDR')]:
            text = star_text(row[field])
            if text:
                key = field + '-' + str(index) if panel == 'G' else label + '-' + row.block_10_name.replace(' ', '-')
                expected[key] = text
    significant = trials.loc[trials.change_p_fdr.lt(.05), 'trial_number'].astype(int).tolist()
    if panel == 'G':
        expected.update({'trial-' + str(t): '*' for t in significant})
    actual = {}
    for node in group.iter():
        identity = node.get('id', '')
        if panel == 'G' and 'fig2__g__stats__' in identity:
            key = identity.split('fig2__g__stats__', 1)[1]
            relevant = key.startswith(('D-', 'mean_p_fdr-', 'slope_p_raw-', 'slope_p_fdr-', 'trial-'))
        elif panel != 'G' and '__' + panel + '-' in identity:
            key = identity.split('__' + panel + '-', 1)[1]
            relevant = key.startswith(('D-', 'M-', 'R-raw-', 'R-FDR-'))
        else:
            continue
        if relevant:
            text = ''.join(''.join(n.itertext()) for n in node.iter() if n.tag.endswith('}text')).strip()
            if key in actual:
                raise ValueError('Duplicate statistical label: ' + key)
            actual[key] = text
    if actual != expected:
        raise ValueError('Displayed statistical labels differ: ' + str({'expected': expected, 'actual': actual}))
    if panel != 'G':
        scatter = [n for n in group.iter() if n.get('id', '').endswith('__' + panel + '-trial-stars')]
        if len(scatter) != 1 or sum(n.tag.endswith('}use') for n in scatter[0].iter()) != len(significant):
            raise ValueError('Trial marker count differs from saved BH90 decisions')
    return {'statistical_text_groups': len(actual), 'trial_markers': len(significant),
            'significant_trials': significant, 'coverage': 'named XML labels and marker counts; renderer visibility and exact marker positions require separate review'}


def audit_tables(data, observed, dtests, local, trials, diagnostics, coefficients, covariance, residuals):
    checks = []

    def equal(name, actual, expected, tolerance=1e-10):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if actual.shape != expected.shape or not np.allclose(actual, expected, atol=tolerance, rtol=1e-9, equal_nan=True):
            raise ValueError('Saved arithmetic mismatch: ' + name)
        finite = np.isfinite(actual) & np.isfinite(expected)
        checks.append({'check': name, 'status': 'passed', 'values': actual.size,
                       'max_absolute_error': float(np.max(abs(actual[finite] - expected[finite]))) if finite.any() else None})

    if data.duplicated(['fish_key', 'trial_number']).any():
        raise ValueError('Duplicate fish/trial keys')
    if not np.isfinite(data[['log_response', 'log_baseline']].to_numpy()).all():
        raise ValueError('Nonfinite fitted outcome/baseline')
    baseline = 'baseline_logmedian' if 'baseline_logmedian' in data else 'baseline_log_median'
    response = 'response_logmedian' if 'response_logmedian' in data else 'response_log_median'
    equal('model baseline uses saved window log median without second log', data.log_baseline, data[baseline])
    equal('model response uses saved window log median without second log', data.log_response, data[response])
    equal('scheduled trial scaling', data.trial_scaled, (data.trial_number - data.trial_center) / data.trial_scale)
    fish = data.assign(difference=data.log_response - data.log_baseline)
    summary = fish.groupby(['condition_id', 'trial_number'], observed=True).difference.agg(['median', 'count'])
    keys = pd.MultiIndex.from_frame(observed[['condition_id', 'trial_number']])
    if keys.has_duplicates or set(keys) != set(summary.index):
        raise ValueError('Observed trajectory keys differ from model-input coverage')
    equal('observed fish medians', observed['median'], summary.loc[keys, 'median'])
    equal('contributing fish', observed.contributing_fish, summary.loc[keys, 'count'], 0)
    if len(dtests) != 8 or len(local) != 9 or trials.trial_number.tolist() != list(range(5, 95)):
        raise ValueError('Declared multiplicity family sizes disagree')
    equal('Holm8', dtests.p_holm, adjusted(dtests.p_raw, 'holm'))
    for prefix in ['mean', 'slope']:
        equal(prefix + ' BH9', local[prefix + '_p_fdr'], adjusted(local[prefix + '_p_raw'], 'fdr_bh'))
    for prefix in ['change', 'condition']:
        equal(prefix + ' BH90', trials[prefix + '_p_fdr'], adjusted(trials[prefix + '_p_raw'], 'fdr_bh'))
    rebuilt = phase_contrasts(data, diagnostics, coefficients, covariance)
    for column, values in rebuilt.items():
        equal('fixed-effect phase ' + column, trials[column], values)
    if residuals.duplicated(['fish_key', 'trial_number']).any() or set(map(tuple, residuals[['fish_key', 'trial_number']].to_numpy())) != set(map(tuple, data[['fish_key', 'trial_number']].to_numpy())):
        raise ValueError('Residual coverage differs from model input')
    merged = fish.merge(residuals, on=['fish_key', 'trial_number'], validate='one_to_one', suffixes=('', '_residual'))
    equal('saved fitted values plus residual equal model response', merged.fitted + merged.residual, merged.log_response)
    return {'checks': checks, 'fish': data.groupby('condition_id', observed=True).fish_key.nunique().to_dict(),
            'fitted_observations': len(data), 'scheduled_observations': data.fish_key.nunique() * 90,
            'trial_center': float(data.trial_center.iloc[0]), 'trial_scale': float(data.trial_scale.iloc[0]),
            'residual_excess_kurtosis': float(kurtosis(residuals.residual)),
            'unavailable_local_blocks': local.loc[local.mean_p_raw.isna() | local.slope_p_raw.isna(), 'block_10_name'].tolist(),
            'significant_counts': {'D_Holm8': int(dtests.p_holm.lt(.05).sum()), 'M_BH9': int(local.mean_p_fdr.lt(.05).sum()),
                'R_raw': int(local.slope_p_raw.lt(.05).sum()), 'R_BH9': int(local.slope_p_fdr.lt(.05).sum()),
                'trial_BH90': int(trials.change_p_fdr.lt(.05).sum())}}
