"""Versioned Delay G mixed-effects review; fixed legacy metric, SSD outputs only.

Reuses the active condition-aware LME fit/contrast code. Descriptive whole-fish
bootstrap is separate from model-based inference. Never edits the assembly.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis, probplot
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from classical_conditioning.analysis.inference.learning_onset import (
    BLOCK_ORDER, LearningOnsetConfig, _fit_mixed_model, _fixed_covariance,
    _fixed_design_row, block_contrasts, block_global_interaction_test,
    model_coefficients, trial_contrasts,
)

METRIC = 'legacy_distal_angular_speed'
OUTCOME = 'conditional-intensity'  # Author correction: finite bout frames only.
ROOT = Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly')
SOURCE = ROOT / 'sources/20261008T200908002798Z/Fig2_PanelG_allDelay_pre15.figure.json'
BLOCK_FORMULA = "log_response ~ log_baseline + C(condition_id, Treatment(reference='control')) * C(block_10_name)"
TRIAL_FORMULA = "log_response ~ log_baseline + C(condition_id, Treatment(reference='control')) * bs(trial_scaled, df=5, degree=3, include_intercept=False)"
LOCAL_FORMULA = "log_response ~ log_baseline + C(condition_id, Treatment(reference='control')) * within_block_trial"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')


def say(message):
    print(message, flush=True)


def bootstrap_trajectories(fish, *, n_boot=5000, seed=10):
    """One fish draw carries all trials and NaNs; one matrix per condition."""
    rng = np.random.default_rng(seed)
    rows, draws = [], {}
    trials = np.arange(5, 95)
    for condition in ['control', 'delay']:
        d = fish.loc[fish.condition_id.eq(condition)]
        wide = d.pivot(index='fish_id', columns='trial_number', values='ratio').sort_index().reindex(columns=trials)
        values = wide.to_numpy(dtype=float)
        indices = rng.integers(0, len(wide), size=(n_boot, len(wide)))
        boot = np.empty((n_boot, len(trials)))
        for start in range(0, n_boot, 100):
            boot[start:start+100] = np.nanmedian(values[indices[start:start+100]], axis=1)
        low, high = np.nanquantile(boot, [.025, .975], axis=0)
        low_half, high_half = np.nanquantile(boot[:n_boot//2], [.025, .975], axis=0)
        median = np.nanmedian(values, axis=0)
        for i, trial in enumerate(trials):
            rows.append(dict(condition_id=condition, trial_number=int(trial), median=float(median[i]),
                ci_lower=float(low[i]), ci_upper=float(high[i]), q25=float(np.nanquantile(values[:, i], .25)),
                q75=float(np.nanquantile(values[:, i], .75)), contributing_fish=int(np.isfinite(values[:, i]).sum()),
                endpoint_change_2500_to_5000=float(max(abs(low[i]-low_half[i]), abs(high[i]-high_half[i])))))
        draws[condition] = {'fish_ids': wide.index.to_numpy(dtype=str), 'indices': indices, 'medians': boot}
    return pd.DataFrame(rows), draws


def adjust_family(frame, source, target, method):
    """Keep failed tests in the declared family as p=1; do not shrink families."""
    p = pd.to_numeric(frame[source], errors='coerce').to_numpy(dtype=float)
    ok = np.isfinite(p)
    adjusted = multipletests(np.where(ok, p, 1.0), method=method)[1]
    frame[target] = np.where(ok, adjusted, np.nan)


def fit(data, formula, config, name, out):
    result, diag = _fit_mixed_model(data, formula=formula, config=config)
    diag['model'] = name
    # Convergence alone is insufficient: require finite, positive fixed covariance.
    if result is not None:
        cov = _fixed_covariance(result)
        eig = np.linalg.eigvalsh((cov + cov.T)/2)
        diag['fixed_covariance_min_eigenvalue'] = float(eig.min())
        diag['fixed_covariance_finite'] = bool(np.isfinite(cov).all())
        diag['hessian_curvature_ok'] = bool(np.isfinite(diag['hessian_max_eigenvalue']) and diag['hessian_max_eigenvalue'] < 0)
        if not diag['fixed_covariance_finite'] or eig.min() <= 0 or not diag['hessian_curvature_ok']:
            diag['diagnostic_status'] = 'failed'
            diag['error'] = str(diag.get('error') or '') + '; invalid covariance or Hessian curvature'
            result = None
    write_json(out / (name + '-diagnostics.json'), diag)
    model_coefficients(result, model_name=name, confidence_level=.95).to_csv(out / (name + '-coefficients.csv'), index=False)
    if result is not None:
        pd.DataFrame(_fixed_covariance(result), index=result.fe_params.index, columns=result.fe_params.index).to_csv(out / (name + '-fixed-covariance.csv'))
        write_json(out / (name + '-random-covariance.json'), np.asarray(result.cov_re).tolist())
    say(f'{name}: {diag["diagnostic_status"]}; RE={diag.get("used_random_effects")}; {diag.get("error") or ""}')
    return result, diag


def residual_table(result, data, name):
    if result is None:
        return pd.DataFrame(), {'model': name, 'status': 'unavailable'}
    d = data[['fish_id', 'condition_id', 'trial_number', 'block_10_name']].copy()
    d['fitted'] = np.asarray(result.fittedvalues)
    d['residual'] = data.log_response.to_numpy() - d.fitted.to_numpy()
    lag = []
    for _, g in d.groupby('fish_id'):
        x = g.sort_values('trial_number').residual.to_numpy()
        lag.append(float(np.corrcoef(x[:-1], x[1:])[0, 1]))
    return d, {'model': name, 'status': 'review', 'residual_skewness': float(skew(d.residual)),
               'residual_excess_kurtosis': float(kurtosis(d.residual)), 'median_within_fish_lag1': float(np.nanmedian(lag)),
               'serial_dependence_caution': bool(abs(np.nanmedian(lag)) > .2)}


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


def make_panel(summary, d_tests, local, trials, global_test, diagnostics, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    fig, (marks, ax) = plt.subplots(2, 1, figsize=(10, 6.3), sharex=True, height_ratios=[1.35, 3.0], layout='constrained')
    marks.set_ylim(-.5, 4.6)
    marks.set_yticks([4, 3, 2, 1, 0], ['D · interaction (Holm)', 'M · block mean (FDR)', 'R · block slope (raw)', 'R · block slope (FDR)', 'Trial change · stars (FDR)'], fontsize=9)
    marks.tick_params(axis='both', length=0, labelbottom=False)
    marks.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    for _, r in d_tests.iterrows():
        if r.p_holm < .05:
            marks.text(r.center, 4, 'D' + stars(r.p_holm), color='#a97900', ha='center', va='center', fontsize=10)
    for _, r in local.iterrows():
        if r['status'] == 'ok':
            for column, lane, label, color in [('mean_p_fdr', 3, 'M', '#555555'), ('slope_p_raw', 2, 'R', '#a43c24'), ('slope_p_fdr', 1, 'R', '#a43c24')]:
                if r[column] < .05:
                    marks.text(r.center, lane, label + stars(r[column]), color=color, ha='center', va='center', fontsize=10)
        else:
            marks.text(r.center, 3, 'fit failed', rotation=35, color='.5', ha='center', fontsize=6)
    if not trials.empty:
        sig = trials.loc[trials.p_fdr < .05]
        marks.scatter(sig.trial_number, np.zeros(len(sig)), marker='*', color='black', s=30, linewidths=0)
    else:
        marks.text(49, 0, 'Trial model failed — no stars', ha='center', color='.45', fontsize=9)
    positions = [(name, 5 + i*10, 14 + i*10) for i, name in enumerate(BLOCK_ORDER)]
    for name, start, end in positions:
        marks.text((start+end)/2, 4.45, 'Pre' if name == 'Pre-train' else name.replace('Train ', 'Tr').replace('Test ', 'Te'), ha='center', fontsize=8, color='.4')
        if start > 5:
            for axis in [marks, ax]:
                axis.axvline(start-.5, color='.85', lw=.55, zorder=0)
    for condition, color, label in [('control', '#27aae1', 'Control (n=28)'), ('delay', '#f00098', 'Delay (n=29)')]:
        d = summary.loc[summary.condition_id.eq(condition)].sort_values('trial_number')
        ax.plot(d.trial_number, d['median'], color=color, lw=1.25, label=label)
        ax.fill_between(d.trial_number, d.ci_lower, d.ci_upper, color=color, alpha=.22, lw=0)
    for boundary in [14.5, 64.5]:
        ax.axvline(boundary, color='.4', linestyle=':', lw=.9)
    ax.axhline(1, color='.4', lw=.7)
    ax.set_xlim(4, 95)
    ax.set_ylim(.65, 1.30)
    ax.set_ylabel('Bout-only response mean / bout-only baseline mean\nmedian [pointwise 95% fish-bootstrap CI]')
    ax.set_xlabel('Global CS trial')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='lower right')
    p = float(global_test.iloc[0].p_value) if not global_test.empty else np.nan
    fig.suptitle('G | Delay — bout-only legacy metric; exploratory mixed-effects review\n' +
        f'Joint condition × block p={p:.3g} | 5,000 fish resamples, seed 10', fontsize=12)
    fig.text(.18, -.01, 'Stars test baseline-adjusted change versus Pre5–14, from one spline LME; no onset/extinction claim.\nD: Holm8; M/R: separate BH9 families; trial stars: BH90. Raw R is exploratory. Statistical approval pending.', fontsize=8)
    for ext in ['svg', 'png', 'pdf']:
        fig.savefig(out / ('Fig2_PanelG_delay_legacy_LME_bootstrap5000.' + ext), dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--assembly-root', type=Path, default=ROOT)
    parser.add_argument('--source-panel', type=Path, default=SOURCE)
    args = parser.parse_args()
    out = args.output_dir or args.assembly_root / 'row3-trial-ratio-review' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-delay-boutonly-lme')
    if args.assembly_root.resolve() not in out.resolve().parents:
        raise ValueError('Scientific outputs must stay under the designated SSD assembly root.')
    out.mkdir(parents=True, exist_ok=False)
    say('OUTPUT_DIRECTORY=' + str(out))
    old = json.loads(args.source_panel.read_text())
    assert old['analysis_identity']['metric_id'] == METRIC
    inputs = []
    for item in old['inputs'] + old['code_dependencies'] + [old['panel_data']] + old['outputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Source hash changed: ' + item['path'])
        inputs.append(item)
    paths = [x['path'] for x in old['inputs'] if 'trial-outcomes' in x['path'] and x['path'].endswith('.parquet')]
    data = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    data = data.loc[data.metric_id.eq(METRIC) & data.alignment.eq('CS') & data.trial_number.between(5, 94)].copy()
    b, r = data.baseline_conditional_intensity, data.conditional_intensity
    # Saved fields select valid adjacent moving/bout frames. No-bout windows
    # are NaN, not zero. Positive means are required by the logarithmic model.
    eligible = np.isfinite(b) & np.isfinite(r) & (b > 0) & (r > 0) & (data.response_moving_sample_count >= 1)
    data.loc[~eligible].to_parquet(out / 'excluded-outcomes.parquet', index=False)
    data = data.loc[eligible].sort_values(['condition_id', 'fish_id', 'trial_number']).reset_index(drop=True)
    assert data.fish_id.nunique() == 57
    assert not data.duplicated(['fish_id', 'trial_number']).any()
    data['ratio'] = data.conditional_intensity / data.baseline_conditional_intensity
    data['log_response'] = np.log(data.conditional_intensity)
    data['log_baseline'] = np.log(data.baseline_conditional_intensity)
    data['fish_key'] = data.fish_id.astype(str)
    data['trial_center'] = data.trial_number.mean()
    data['trial_scale'] = data.trial_number.std(ddof=0)
    data['trial_scaled'] = (data.trial_number - data.trial_center)/data.trial_scale
    data['condition_id'] = pd.Categorical(data.condition_id, categories=['control', 'delay'], ordered=True)
    data['block_10_name'] = pd.Categorical(data.block_10_name, categories=BLOCK_ORDER, ordered=True)
    data.to_parquet(out / 'model-input.parquet', index=False)
    fish = data[['fish_id', 'condition_id', 'trial_number', 'ratio']].copy()
    fish.to_parquet(out / 'fish-ratios.parquet', index=False)
    old_fish = pd.read_parquet(old['panel_data']['path'])
    match = fish.merge(old_fish, on=['fish_id', 'condition_id', 'trial_number'], validate='one_to_one')
    # This correction must differ from the preserved all-frame ratio panel.
    comparison = {'eligible_bout_only_rows': len(data), 'excluded_rows': int((~eligible).sum()),
                  'total_scheduled_fish_trials': len(eligible), 'cohort_fish': data.fish_id.nunique(),
                  'fraction_ratios_changed': float(np.mean(~np.isclose(match.ratio, match['Fish median response / baseline'])))}
    write_json(out / 'bout-only-comparison.json', comparison)
    summary, draws = bootstrap_trajectories(fish)
    summary.to_parquet(out / 'bootstrap-summary.parquet', index=False)
    summary.to_csv(out / 'bootstrap-summary.csv', index=False)
    for condition, item in draws.items():
        np.savez_compressed(out / (condition + '-bootstrap-draws.npz'), **item)
    say('5,000 whole-fish bootstrap draws completed; bout-only correction=' + json.dumps(comparison))
    config = LearningOnsetConfig(metric_id=METRIC, outcome_id=OUTCOME, n_bootstrap=5000, seed=10, run_categorical_sensitivity=False)
    spec = {'metric': METRIC, 'outcome_id': OUTCOME, 'baseline_s': [-15, 0], 'response_s': [0, 9], 'config': asdict(config),
            'ratio': 'conditional_intensity / baseline_conditional_intensity',
            'frame_mask': 'window & detector-valid & adjacent & moving; moving equals positive bout_id',
            'no_bout_windows': 'NaN; no zero replacement', 'model_transform': 'natural log of positive bout-only window means; no additive offset',
            'block_formula': BLOCK_FORMULA, 'trial_formula': TRIAL_FORMULA, 'local_formula': LOCAL_FORMULA,
            'local_random_effects': '1', 'D_family': 'Holm8 nonreference block interaction terms',
            'M_family': 'BH9 local block mean tests', 'R_family': 'BH9 local slope tests; raw shown separately',
            'trial_family': 'BH90 two-sided control-minus-Delay difference in change from average Pre5-14',
            'statistical_status': 'exploratory; no scientific freeze inferred', 'model_bootstrap_refits': 0,
            'descriptive_bootstrap': {'n': 5000, 'seed': 10, 'unit': 'whole fish within condition; NaNs preserved'},
            'onset': 'not estimated; no simultaneous onset calculation', 'scope': 'G only; no assembly modification'}
    write_json(out / 'prespecified-review-config.json', spec)
    with threadpool_limits(limits=1):
        block, diag_block = fit(data, BLOCK_FORMULA, config, 'block-model', out)
        longitudinal, diag_trial = fit(data, TRIAL_FORMULA, config, 'longitudinal-model', out)
        global_test = block_global_interaction_test(block) if block is not None else pd.DataFrame()
        global_test.to_csv(out / 'joint-interaction-test.csv', index=False)
        d_tests = []
        if block is not None:
            for term in block.fe_params.index:
                if 'condition_id' in term and 'block_10_name' in term and ':' in term:
                    name = next(n for n in BLOCK_ORDER if '[T.' + n + ']' in term)
                    center = float(data.loc[data.block_10_name.eq(name), 'trial_number'].mean())
                    d_tests.append(dict(block_10_name=name, term=term, center=center, estimate=float(block.fe_params[term]), p_raw=float(block.pvalues[term])))
        d_tests = pd.DataFrame(d_tests, columns=['block_10_name', 'term', 'center', 'estimate', 'p_raw'])
        if not d_tests.empty:
            adjust_family(d_tests, 'p_raw', 'p_holm', 'holm')
            block_contrasts(block, data, config=config).to_csv(out / 'block-learning-contrasts.csv', index=False)
        else:
            d_tests['p_holm'] = []
        d_tests.to_csv(out / 'D-interaction-tests.csv', index=False)
        trials = pd.DataFrame()
        if longitudinal is not None:
            trials, contrast_matrix = trial_contrasts(longitudinal, data, config=config)
            trials['p_raw'] = 2 * norm.sf(abs(trials.learning_contrast / trials.standard_error))
            adjust_family(trials, 'p_raw', 'p_fdr', 'fdr_bh')
            np.save(out / 'trial-contrast-matrix.npy', contrast_matrix)
        trials.to_csv(out / 'trial-tests-FDR.csv', index=False)
        local, diagnostics = [], [diag_block, diag_trial]
        for name in BLOCK_ORDER:
            sub = data.loc[data.block_10_name.eq(name)].copy()
            sub['within_block_trial'] = sub.trial_number - sub.trial_number.mean()
            result, diag = fit(sub, LOCAL_FORMULA, replace(config, random_effects_formula='1', optimizer='powell', allow_random_intercept_fallback=False), name.replace(' ', '-') + '-local', out)
            diagnostics.append(diag)
            row = {'block_10_name': name, 'center': float(sub.trial_number.mean()), 'status': diag['diagnostic_status'], 'mean_estimate': np.nan, 'mean_p_raw': np.nan, 'slope_estimate': np.nan, 'slope_p_raw': np.nan}
            if result is not None:
                for term in result.fe_params.index:
                    if 'condition_id' in term:
                        label = 'slope' if ':' in term else 'mean'
                        row[label + '_estimate'] = float(result.fe_params[term])
                        row[label + '_p_raw'] = float(result.pvalues[term])
            local.append(row)
        local = pd.DataFrame(local)
        for label in ['mean', 'slope']:
            adjust_family(local, label + '_p_raw', label + '_p_fdr', 'fdr_bh')
        local.to_csv(out / 'MR-local-block-tests.csv', index=False)
        residual_summaries = []
        residuals = []
        for result, name in [(block, 'block'), (longitudinal, 'longitudinal')]:
            table, info = residual_table(result, data, name)
            table['model'] = name
            residuals.append(table)
            residual_summaries.append(info)
        pd.concat(residuals, ignore_index=True).to_parquet(out / 'residuals.parquet', index=False)
        write_json(out / 'residual-summary.json', residual_summaries)
        # Optimizer and random-intercept sensitivity are specified before results.
        sensitivities = []
        for name, formula, main_result in [('block', BLOCK_FORMULA, block), ('longitudinal', TRIAL_FORMULA, longitudinal)]:
            for variant, sensitivity_config in [('powell', replace(config, optimizer='powell')), ('random-intercept', replace(config, random_effects_formula='1', allow_random_intercept_fallback=False, optimizer='powell'))]:
                result, diag = fit(data, formula, sensitivity_config, name + '-' + variant, out)
                diagnostics.append(diag)
                if result is not None and main_result is not None:
                    sensitivities.append({'model': name, 'variant': variant, 'max_fixed_coefficient_difference': float(np.max(abs(result.fe_params - main_result.fe_params)))})
        write_json(out / 'sensitivity-summary.json', sensitivities)
        # Leave-one-fish-out refits use the same formulas and do not select a model.
        influence = []
        for i, fish_id in enumerate(sorted(data.fish_id.unique())):
            sub = data.loc[~data.fish_id.eq(fish_id)].copy()
            b_fit, b_diag = _fit_mixed_model(sub, formula=BLOCK_FORMULA, config=config, collect_extended_diagnostics=False)
            t_fit, t_diag = _fit_mixed_model(sub, formula=TRIAL_FORMULA, config=config, collect_extended_diagnostics=False)
            row = {'omitted_fish': fish_id, 'block_status': b_diag['diagnostic_status'], 'trial_status': t_diag['diagnostic_status']}
            if b_fit is not None:
                bc = block_contrasts(b_fit, sub, config=config)
                row['mean_training_learning_contrast'] = float(bc.loc[bc.block_10_name.str.startswith('Train'), 'learning_contrast'].mean())
            if t_fit is not None and not trials.empty:
                tc, _ = trial_contrasts(t_fit, sub, config=config)
                row['max_trial_contrast_shift'] = float(np.max(abs(tc.learning_contrast.to_numpy() - trials.learning_contrast.to_numpy())))
            influence.append(row)
            if (i+1) % 5 == 0 or i == 56:
                pd.DataFrame(influence).to_csv(out / 'leave-one-fish-out.csv', index=False)
                say(f'Influence refits {i+1}/57 completed.')
        # Fish-level robustness: mean log ratio, all training versus pre (no chosen peak).
        effect_data = data.assign(log_ratio=np.log(data.ratio)).groupby(['fish_id', 'condition_id', 'phase'], observed=True).log_ratio.mean().unstack()
        effect_data['suppression_pre_minus_training'] = effect_data['Pre'] - effect_data['Train']
        effect_data.reset_index().to_csv(out / 'fish-training-effects.csv', index=False)
        control = effect_data.xs('control', level='condition_id').suppression_pre_minus_training.to_numpy()
        delay = effect_data.xs('delay', level='condition_id').suppression_pre_minus_training.to_numpy()
        rng = np.random.default_rng(10)
        boot_effect = delay[rng.integers(len(delay), size=(5000, len(delay)))].mean(axis=1) - control[rng.integers(len(control), size=(5000, len(control)))].mean(axis=1)
        robustness = {'estimand': 'Delay-minus-control suppression: mean fish log-ratio Pre5-14 minus Train15-64', 'estimate': float(delay.mean()-control.mean()), 'ci95': np.quantile(boot_effect, [.025, .975]).tolist(), 'n_boot': 5000, 'seed': 10}
        write_json(out / 'fish-robustness.json', robustness)
    pd.DataFrame(diagnostics).to_csv(out / 'all-fit-diagnostics.csv', index=False)
    make_panel(summary, d_tests, local, trials, global_test, diagnostics, out)
    # Plot residuals separately so model cautions are inspectable.
    if longitudinal is not None:
        residual = residuals[1]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
        axes[0].scatter(residual.fitted, residual.residual, s=3, alpha=.2)
        axes[0].axhline(0, color='black', lw=.7)
        axes[0].set(xlabel='Fitted log response', ylabel='Conditional residual')
        probplot(residual.residual, dist='norm', plot=axes[1])
        fig.savefig(out / 'residual-diagnostics.png', dpi=160)
        plt.close(fig)
    statistics = {'joint_interaction': global_test.to_dict('records'), 'D_significant': int((d_tests.p_holm < .05).sum()),
        'M_significant': int((local.mean_p_fdr < .05).sum()), 'R_raw_significant': int((local.slope_p_raw < .05).sum()),
        'R_FDR_significant': int((local.slope_p_fdr < .05).sum()), 'trial_FDR_significant': int((trials.p_fdr < .05).sum()) if not trials.empty else 0,
        'trial_significant_numbers': trials.loc[trials.p_fdr < .05, 'trial_number'].tolist() if not trials.empty else [],
        'failed_model_names': [x['model'] for x in diagnostics if x['diagnostic_status'] != 'ok'],
        'influence_failed_refits': sum(x['block_status'] != 'ok' or x['trial_status'] != 'ok' for x in influence),
        'residual_diagnostics': residual_summaries, 'fish_robustness': robustness,
        'bootstrap_max_endpoint_change': float(summary.endpoint_change_2500_to_5000.max()),
        'status': 'exploratory fitted results; scientific model/cohort/diagnostic review remains open'}
    write_json(out / 'result-summary.json', statistics)
    identity = {**old['analysis_identity'], 'outcome_id': OUTCOME,
                'ratio_formula': 'conditional_intensity / baseline_conditional_intensity; fish trial ratio then equal-fish condition median',
                'summary': 'median [pointwise95% whole-fish bootstrap CI];5000 draws;seed10',
                'significance_marks': 'fresh model-derived exploratory D/M/R/trial marks where significant',
                'frame_eligibility': spec['frame_mask']}
    (out / 'analysis-script.py').write_bytes(Path(__file__).read_bytes())
    write_json(out / 'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json', {
        'analysis_identity': identity, 'scientific_status': statistics['status'],
        'summary': 'median [pointwise95% whole-fish bootstrap CI];5000 draws;seed10', 'model_specification': spec,
        'inference': statistics, 'inputs': inputs, 'plotted_data': ['bootstrap-summary.parquet', 'D-interaction-tests.csv', 'MR-local-block-tests.csv', 'trial-tests-FDR.csv'],
        'code': [{'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}, {'path': str(Path(sys.modules['classical_conditioning.analysis.inference.learning_onset'].__file__)), 'sha256': sha(sys.modules['classical_conditioning.analysis.inference.learning_onset'].__file__)}],
        'plans': [{'path': str(Path('Plans')/n), 'sha256': sha(Path('Plans')/n)} for n in ['02_ANALYSIS_AND_STATISTICS.md', '03_LEARNING_ONSET_IMPLEMENTATION.md']],
        'outputs': [{'path': str(p), 'sha256': sha(p)} for p in sorted(out.iterdir()) if p.is_file()]})
    say(json.dumps({'output_directory': str(out), 'results': statistics}, indent=2))


if __name__ == '__main__':
    main()
