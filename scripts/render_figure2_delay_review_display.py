"""Render a corrected G preview from saved statistical results, without refits."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly')
SOURCE = ROOT / 'row3-trial-ratio-review/20261009T072921130316Z-delay-lme'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-dir', type=Path, default=SOURCE)
    parser.add_argument('--scale', choices=['ratio', 'log-ratio'], default='ratio')
    parser.add_argument('--assembly-root', type=Path, default=ROOT)
    args = parser.parse_args()
    source = args.analysis_dir
    out = args.assembly_root / 'row3-trial-ratio-review' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-delay-display')
    out.mkdir(parents=True, exist_ok=False)
    summary = pd.read_parquet(source / 'bootstrap-summary.parquet')
    dtests = pd.read_csv(source / 'D-interaction-tests.csv')
    local = pd.read_csv(source / 'MR-local-block-tests.csv')
    trials = pd.read_csv(source / 'trial-tests-FDR.csv')
    results = json.loads((source / 'result-summary.json').read_text())
    source_sidecar = json.loads((source / 'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json').read_text())
    bout_only = source_sidecar['analysis_identity'].get('outcome_id') == 'conditional-intensity'
    for item in source_sidecar['outputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Saved analysis artifact changed: ' + item['path'])
    assert sha(source / 'analysis-script.py') == source_sidecar['code'][0]['sha256']
    if args.scale == 'log-ratio':
        from render_figure2_delay_legacy_metric_lme import bootstrap_trajectories
        fish = pd.read_parquet(source / 'fish-ratios.parquet')
        fish['ratio'] = np.log(fish['ratio'])
        summary, draws = bootstrap_trajectories(fish, n_boot=5000, seed=10)
        summary.to_parquet(out / 'log-ratio-bootstrap-summary.parquet', index=False)
        for condition, draw in draws.items():
            np.savez_compressed(out / (condition + '-log-ratio-bootstrap-draws.npz'), **draw)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(10, 6.6), layout='constrained')
    grid = fig.add_gridspec(3, 1, height_ratios=[1.35, 3.0, .50])
    marks = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1], sharex=marks)
    note = fig.add_subplot(grid[2])
    marks.set_ylim(-.5, 4.7)
    marks.set_yticks([4, 3, 2, 1, 0], ['D · interaction (Holm)', 'M · block mean (FDR)', 'R · block slope (raw)', 'R · block slope (FDR)', 'Trial change · stars (FDR)'], fontsize=9)
    marks.tick_params(axis='both', length=0, labelbottom=False)
    marks.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    counts = {i: 0 for i in range(5)}
    for _, r in dtests.iterrows():
        if r.p_holm < .05:
            marks.text(r.center, 4, 'D' + stars(r.p_holm), color='#a97900', ha='center', va='center', fontsize=10)
            counts[4] += 1
    for _, r in local.loc[local.status.eq('ok')].iterrows():
        for column, lane, label, color in [('mean_p_fdr', 3, 'M', '#555555'), ('slope_p_raw', 2, 'R', '#a43c24'), ('slope_p_fdr', 1, 'R', '#a43c24')]:
            if r[column] < .05:
                marks.text(r.center, lane, label + stars(r[column]), color=color, ha='center', va='center', fontsize=10)
                counts[lane] += 1
    significant = trials.loc[trials.p_fdr < .05]
    marks.scatter(significant.trial_number, np.zeros(len(significant)), marker='*', color='black', s=30, linewidths=0)
    counts[0] = len(significant)
    for lane, count in counts.items():
        if count == 0:
            marks.text(50, lane, 'none below p = 0.05' if lane == 2 else 'none below adjusted p = 0.05', color='.55', ha='center', va='center', fontsize=8)
    blocks = ['Pre', 'Tr1', 'Tr2', 'Tr3', 'Tr4', 'Tr5', 'Te1', 'Te2', 'Te3']
    for i, label in enumerate(blocks):
        start = 5 + i*10
        marks.text(start + 4.5, 4.5, label, ha='center', fontsize=8, color='.4')
        if i:
            for axis in [marks, ax]:
                axis.axvline(start-.5, color='.85', lw=.55, zorder=0)
    for condition, color, label in [('control', '#27aae1', 'Control (n=28)'), ('delay', '#f00098', 'Delay (n=29)')]:
        d = summary.loc[summary.condition_id.eq(condition)].sort_values('trial_number')
        ax.plot(d.trial_number, d['median'], color=color, lw=1.25, label=label)
        ax.fill_between(d.trial_number, d.ci_lower, d.ci_upper, color=color, alpha=.22, lw=0)
    for boundary in [14.5, 64.5]:
        ax.axvline(boundary, color='.4', linestyle=':', lw=.9)
    is_log = args.scale == 'log-ratio'
    ax.axhline(0 if is_log else 1, color='.4', lw=.7)
    label = 'ln(bout-only response mean / baseline mean)' if is_log else ('Bout-only response mean / baseline mean' if bout_only else 'Response mean / pre-CS baseline mean')
    ax.set(xlim=(4, 95), ylim=(np.log(.65), np.log(1.30)) if is_log else (.65, 1.30), xlabel='Global CS trial',
           ylabel=label + '\nmedian [pointwise 95% fish-bootstrap CI]')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='lower right', fontsize=9)
    note.set_axis_off()
    note.text(0, .96, 'EXPLORATORY: heavy-tailed residuals; statistical model approval remains open.', va='top', fontsize=9, color='#a43c24', transform=note.transAxes)
    note.text(0, .55, 'Stars: adjusted change from Pre5–14, one spline LME. D: Holm8; M/R: separate BH9; trials: BH90.', va='top', fontsize=8, transform=note.transAxes)
    note.text(0, .18, 'Raw R is uncorrected. * p<.05, ** p<.01, *** p<.001. No onset or extinction claim.', va='top', fontsize=8, transform=note.transAxes)
    p = results['joint_interaction'][0]['p_value']
    fig.suptitle(f'G | Delay — tail bend angular speed; ' + ('bout-only ' if bout_only else '') + f'mixed-effects review\nJoint condition × block p={p:.3g} | 5,000 whole-fish resamples, seed 10', fontsize=12)
    for ext in ['svg', 'png', 'pdf']:
        fig.savefig(out / ('Fig2_PanelG_delay_legacy_LME_bootstrap5000.' + ext), dpi=220)
    plt.close(fig)
    coverage = summary.groupby('condition_id').contributing_fish.agg(['min', 'max']).to_dict('index')
    residual = results['residual_diagnostics'][1]
    caption = f"""# Delay panel G: bout-only mixed-effects review

Frozen metric: legacy_distal_angular_speed (tail bend angular speed, rad/ms).
Each fish-trial value is the arithmetic mean of finite, detector-valid, adjacent
bout frames in response [0,9), divided by the equivalent mean in baseline
[-15,0). Nonbout frames are ignored. No-bout windows remain missing. No smoothing
or imputation. Blue is control, magenta is Delay. Each line is the equal-fish
median of individual ratios. Reference 1 means unchanged bout intensity.
Cohort: 28 control and 29 Delay; contributing fish per trial: {coverage}.
There are 4,811 eligible fish-trial rows out of 5,130 scheduled.

Bands are pointwise 95% percentile confidence intervals of the condition median,
from 5,000 whole-fish resamples, seed 10. Draw the original number of fish with
replacement within each condition. Every selected fish brings its whole
90-trial trajectory and missing values, using the same draw across all trials.
Seed 10 makes draws reproducible. These bands measure uncertainty about the
median; they are not IQRs, individual spread, simultaneous bands or LME CIs.

Fresh results: joint condition-by-block Wald p={p:.8g};
D/Holm8: {results['D_significant']} blocks; M/BH9: {results['M_significant']} blocks;
R/raw: {results['R_raw_significant']} blocks; R/BH9: {results['R_FDR_significant']} blocks;
trial/BH90: {results['trial_FDR_significant']} trials. Full block labels, estimates
and adjusted p values are in saved tables. Black stars mark adjusted p<.05,
with one star symbol per trial; their shape does not encode p-value magnitude.
D/M/R asterisks use *<.05, **<.01, ***<.001 within their declared families.

Models fit natural log positive bout response, adjusted for natural log positive
bout baseline, without an offset. Block and spline5 models use fish random
intercepts and trial slopes. M/R use separate within-block random-intercept
models and centered trial. D tests block interactions relative to Pre5-14.
M tests a local condition difference at block center; R tests a local slope
difference. Trial stars test (control-Delay at trial) minus its average in
Pre5-14, from one longitudinal spline model. These are adjusted log-response
tests, not direct tests of displayed median ratios. Pre trial stars represent
departures from average Pre contrast; they cannot imply pre-training learning.

All main/local and optimizer/random-intercept sensitivity fits passed numerical
checks; all 57 leave-one-fish-out refits passed. Longitudinal residual skewness=
{residual['residual_skewness']:.3f}, excess kurtosis={residual['residual_excess_kurtosis']:.3f},
median fish lag1={residual['median_within_fish_lag1']:.3f}. Heavy tails remain,
so Gaussian Wald inference is exploratory. Missing bout windows can be
informative; retaining NaNs does not remove selection bias. This measures vigor
conditional on bouting, not movement probability or total activity.
Same-data fish training robustness (different estimand): {results['fish_robustness']}.
Maximum CI endpoint change from 2,500 to 5,000 draws:
{results['bootstrap_max_endpoint_change']:.5f} ratio units. No onset/extinction
claim or full assembly update. Panel I remains unavailable/inconclusive.
"""
    if is_log:
        caption = caption.replace('Each line is the equal-fish\nmedian of individual ratios. Reference 1 means unchanged bout intensity.', 'Each line is the equal-fish\nmedian of natural logs of individual ratios. Reference 0 means unchanged bout intensity.')
        caption += '\nLOG VERSION: natural log is applied to each individual fish ratio before taking condition medians and bootstrapping. The reference is zero, with negative values indicating lower response bout intensity. This is not the historical LogMedian outcome (median of frame-level log vigor in response minus median of frame-level log vigor in baseline). The statistical fits and annotations are identical across these two display versions.\n'
    (out / 'caption.md').write_text(caption, encoding='utf-8')
    inputs = ['bootstrap-summary.parquet', 'D-interaction-tests.csv', 'MR-local-block-tests.csv', 'trial-tests-FDR.csv', 'result-summary.json', 'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json', 'analysis-script.py']
    if is_log:
        inputs.append('fish-ratios.parquet')
    (out / 'display-script.py').write_bytes(Path(__file__).read_bytes())
    identity = {**source_sidecar['analysis_identity'],
                'display_scale': args.scale,
                'display_statistic': 'median of individual natural-log ratios' if is_log else 'median of individual unlogged ratios',
                'summary': 'median [pointwise95% whole-fish bootstrap CI];5000 draws;seed10',
                'significance_marks': 'fresh model-derived exploratory D/M/R/trial marks; counts recorded in inference'}
    sidecar = {'scientific_status': 'exploratory; heavy-tailed residuals; model approval pending',
               'analysis_identity': identity, 'summary': 'pointwise95% whole-fish bootstrap CI;5000;seed10',
               'inference': results, 'annotation_semantics': source_sidecar['model_specification'], 'analysis_directory': str(source),
               'code': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
               'inputs': [{'path': str(source / n), 'sha256': sha(source / n)} for n in inputs],
               'outputs': [{'path': str(p), 'sha256': sha(p)} for p in sorted(out.iterdir()) if p.is_file()]}
    (out / 'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json').write_text(json.dumps(sidecar, indent=2) + '\n')
    print('DISPLAY_DIRECTORY=' + str(out))


if __name__ == '__main__':
    main()
