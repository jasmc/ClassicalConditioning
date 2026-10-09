"""Render a corrected G preview from saved statistical results, without refits."""
from datetime import datetime, timezone
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
    out = ROOT / 'row3-trial-ratio-review' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-delay-display')
    out.mkdir(parents=True, exist_ok=False)
    summary = pd.read_parquet(SOURCE / 'bootstrap-summary.parquet')
    dtests = pd.read_csv(SOURCE / 'D-interaction-tests.csv')
    local = pd.read_csv(SOURCE / 'MR-local-block-tests.csv')
    trials = pd.read_csv(SOURCE / 'trial-tests-FDR.csv')
    results = json.loads((SOURCE / 'result-summary.json').read_text())
    source_sidecar = json.loads((SOURCE / 'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json').read_text())
    for item in source_sidecar['outputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Saved analysis artifact changed: ' + item['path'])
    assert sha(SOURCE / 'analysis-script.py') == source_sidecar['code'][0]['sha256']
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
            marks.text(50, lane, 'none below adjusted p = 0.05', color='.55', ha='center', va='center', fontsize=8)
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
    ax.axhline(1, color='.4', lw=.7)
    ax.set(xlim=(4, 95), ylim=(.65, 1.30), xlabel='Global CS trial',
           ylabel='Response mean / pre-CS baseline mean\nmedian [pointwise 95% fish-bootstrap CI]')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='lower right', fontsize=9)
    note.set_axis_off()
    note.text(0, .96, 'EXPLORATORY: heavy-tailed residuals; statistical model approval remains open.', va='top', fontsize=9, color='#a43c24', transform=note.transAxes)
    note.text(0, .55, 'Stars: adjusted change from Pre5–14, one spline LME. D: Holm8; M/R: separate BH9; trials: BH90.', va='top', fontsize=8, transform=note.transAxes)
    note.text(0, .18, 'Raw R is uncorrected. * p<.05, ** p<.01, *** p<.001. No onset or extinction claim.', va='top', fontsize=8, transform=note.transAxes)
    p = results['joint_interaction'][0]['p_value']
    fig.suptitle(f'G | Delay — tail bend angular speed; mixed-effects review\nJoint condition × block p={p:.3g} | 5,000 whole-fish resamples, seed 10', fontsize=12)
    for ext in ['svg', 'png', 'pdf']:
        fig.savefig(out / ('Fig2_PanelG_delay_legacy_LME_bootstrap5000.' + ext), dpi=220)
    plt.close(fig)
    caption = '''# Delay panel G: mixed-effects review

The metric is frozen to tail bend angular speed (legacy_distal_angular_speed, rad/ms). The displayed ratio is dimensionless: response mean[0,9) / baseline mean[-15,0). All 29 Delay and 28 control fish contribute at each unsmoothed global CS trial5–94. Lines are equal-fish medians. Shading is a pointwise95% percentile CI from5,000 bootstrap samples, seed10: fish are sampled with replacement separately within condition, and each selected fish carries all90 trials and any missing values. The draws are saved; missing values are preserved. Seed10 fixes the random draw sequence for reproducibility. This is not a simultaneous band or the CI of an LME contrast.

Numerical results: joint condition-by-block Wald p=.014085. No individual D interaction survives Holm8. M: local Train4 condition difference, p_FDR=.006939 (M**). R: local Train3 slope difference p_raw=.030317 (R*), p_FDR=.272856; no slope survives BH9. No trial change contrast survives BH90, so no black stars are present. The empty lanes are results, not missing calculations. FDR correction limits the expected false-discovery proportion among rejections under its assumptions; it is not a probability that each displayed star is false.

Models fit log(response+1e-6) with log(baseline+1e-6) as a covariate. The block and longitudinal spline5 models use fish random intercepts/slopes, with the specified fallback available but not used. M/R use separate within-block random-intercept LMEs at each block's centered trial. The trial stars test (control−Delay at trial)−average(control−Delay at Pre5–14), two-sided, from one condition-by-spline longitudinal model. They do not test the displayed median ratios directly. This replaces the legacy separate-per-trial LME and its unidentifiable fish random intercept; do not label it an exact legacy reproduction.

All11 main/local fits passed convergence, fixed covariance and Hessian curvature checks. Alternate Powell and intercept-only sensitivities fit. All57 leave-one-fish-out block/longitudinal refits completed without fit failures. Residual skewness is about−1.97 and excess kurtosis20.85, with strong tails and changing spread; Gaussian Wald p-values therefore remain exploratory. Numerical convergence is not validation. A supporting fish-level all-training-versus-pre log-ratio contrast is .13696 (Delay−control suppression),95% fish-bootstrap CI .08480–.19074; it uses a different estimand from the ANCOVA and trial tests and is not independent confirmation. Median CI endpoint maximum change from the first2,500 to all5,000 draws was .02350 ratio units, retained as a stability diagnostic.

Preserved analysis inputs, bootstrap draws, coefficients/covariances, full test tables, residuals, optimizer sensitivities, leave-one-fish-out outputs and hashes are in the linked source directory. This display only corrects layout; it does not refit, choose a more favorable test, or update the main assembly. Panel I remains unavailable/inconclusive.
'''
    (out / 'caption.md').write_text(caption, encoding='utf-8')
    inputs = ['bootstrap-summary.parquet', 'D-interaction-tests.csv', 'MR-local-block-tests.csv', 'trial-tests-FDR.csv', 'result-summary.json', 'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json', 'analysis-script.py']
    identity = {**source_sidecar['analysis_identity'],
                'summary': 'median [pointwise95% whole-fish bootstrap CI];5000 draws;seed10',
                'significance_marks': 'fresh model-derived exploratory M/R marks; no surviving D or trial FDR marks'}
    sidecar = {'scientific_status': 'exploratory; heavy-tailed residuals; model approval pending',
               'analysis_identity': identity, 'summary': 'pointwise95% whole-fish bootstrap CI;5000;seed10',
               'annotation_semantics': source_sidecar['model_specification'], 'analysis_directory': str(SOURCE),
               'code': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
               'inputs': [{'path': str(SOURCE / n), 'sha256': sha(SOURCE / n)} for n in inputs],
               'outputs': [{'path': str(p), 'sha256': sha(p)} for p in sorted(out.iterdir()) if p.is_file()]}
    (out / 'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json').write_text(json.dumps(sidecar, indent=2) + '\n')
    print('DISPLAY_DIRECTORY=' + str(out))


if __name__ == '__main__':
    main()
