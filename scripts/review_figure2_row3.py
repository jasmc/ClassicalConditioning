"""Read-only G/H audit; writes a unique row-three review on the SSD only."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly')
SRC = ROOT / 'sources/20261008T200908002798Z'
NAMES = ['Fig2_PanelG_allDelay_pre15', 'Fig2_PanelH_all3sTrace_pre15',
         'Fig2_PanelH_bootstrap', 'Fig2_PanelH_split-bootstrap']
VALUE = 'Fish median response / baseline'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = ROOT / 'row3-trial-ratio-review' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    audit = {'status': 'review only; scientific approval pending', 'panels': {},
             'I': {'status': 'placeholder; inconclusive', 'authenticated_inputs': None}}
    checks = []
    for name in NAMES:
        sidecar = SRC / (name + '.figure.json')
        j = json.loads(sidecar.read_text())
        checks.append({'path': str(sidecar), 'actual_sha256': digest(sidecar)})
        for item in j['inputs'] + j['code_dependencies'] + [j['panel_data']] + j['outputs']:
            p = Path(item['path'])
            actual = digest(p) if p.is_file() else None
            checks.append({**item, 'actual_sha256': actual, 'matches': actual == item['sha256']})
        audit['panels'][name] = {'identity': j['analysis_identity'], 'sidecar': str(sidecar),
                               'sidecar_sha256': digest(sidecar), 'inputs': j['inputs'],
                               'panel_data': j['panel_data'], 'outputs': j['outputs']}
    for name in NAMES[:2]:
        j = audit['panels'][name]
        d = pd.read_parquet(j['panel_data']['path'])
        coverage = d.groupby(['condition_id', 'trial_number']).agg(
            contributing_fish=('fish_id', 'nunique'), median=(VALUE, 'median'),
            q25=(VALUE, lambda x: x.quantile(.25)), q75=(VALUE, lambda x: x.quantile(.75))).reset_index()
        coverage.to_csv(out / (name + '-coverage.csv'), index=False)
        paths = [x['path'] for x in j['inputs'] if 'trial-outcomes' in x['path'] and x['path'].endswith('.parquet')]
        a = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        a = a.loc[a.metric_id.eq('legacy_distal_angular_speed') & a.alignment.eq('CS') & a.trial_number.between(5, 94)].copy()
        b, r = a.baseline_total_activity, a.response_total_activity
        eligible = np.isfinite(b) & np.isfinite(r) & (b > 0)
        excluded = a.loc[~eligible, ['fish_id', 'condition_id', 'trial_number', 'baseline_total_activity',
                                    'response_total_activity', 'baseline_valid_sample_count', 'response_valid_sample_count']]
        excluded.to_csv(out / (name + '-excluded.csv'), index=False)
        a['saved_ratio'] = np.where(eligible, r / b, np.nan)
        recalculated = a.loc[eligible].groupby(['fish_id', 'condition_id', 'trial_number']).saved_ratio.median().reset_index()
        merged = recalculated.merge(d, on=['fish_id', 'condition_id', 'trial_number'], validate='one_to_one', how='outer', indicator=True)
        numeric_match = bool(merged['_merge'].eq('both').all() and np.allclose(merged.saved_ratio, merged[VALUE], rtol=1e-12, atol=1e-12))
        j['audit'] = {'outcome_rows': len(a), 'ratio_rows': len(d), 'excluded_rows': len(excluded),
                      'recalculated_saved_ratio_match': numeric_match,
                      'contributing_rows_values': sorted(d['Contributing rows'].unique().tolist()),
                      'coverage': coverage.groupby('condition_id').contributing_fish.agg(['min', 'max']).to_dict('index'),
                      'baseline_range': [float(b.min()), float(b.max())],
                      'valid_sample_minimum': a[['baseline_valid_sample_count', 'response_valid_sample_count']].min().to_dict(),
                      'IQR_extent': [float(coverage.q25.min()), float(coverage.q75.max())],
                      'phase_trials': a.groupby('phase').trial_number.agg(['min', 'max']).to_dict('index')}
    prior = ROOT.parent / 'trace-plot-versions/20261006/figure-2H_fish-data.csv'
    old = pd.read_csv(prior)
    new = pd.read_parquet(SRC / (NAMES[1] + '.parquet'))
    m = old.merge(new, on=['fish_id', 'condition_id', 'trial_number'], suffixes=('_old', '_new'), validate='one_to_one', how='outer', indicator=True)
    audit['historical_H_numeric_match'] = bool(m['_merge'].eq('both').all() and np.allclose(m[VALUE + '_old'], m[VALUE + '_new'], rtol=1e-12, atol=1e-12))
    inventory = json.loads((ROOT / 'inventory.json').read_text())
    candidates = [x for x in inventory['candidates'] if ('figure-2G' in x['path'] or 'figure-2H' in x['path']) and Path(x['path']).suffix == '.png']
    for x in candidates:
        x['exists_now'] = Path(x['path']).is_file()
        x['actual_sha256'] = digest(x['path']) if x['exists_now'] else None
    audit['historical_inventory'] = candidates
    audit['hash_checks'] = checks
    audit['hash_check_failures'] = [x for x in checks if x.get('matches') is False]
    audit['review_script'] = {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)}
    plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True, layout='constrained')
    colors = {'control': '#27aae1', 'delay': '#f00098', 'trace': '#ff5733'}
    for ax, name, letter, title in zip(axes[:2], NAMES[:2], ['G', 'H'], ['Delay', '3sTrace']):
        d = pd.read_parquet(SRC / (name + '.parquet'))
        for condition, group in d.groupby('condition_id', sort=False):
            g = group.groupby('trial_number')[VALUE]
            median, q25, q75 = g.median(), g.quantile(.25), g.quantile(.75)
            label = condition.title() + ' (cohort n=' + str(group.fish_id.nunique()) + ')'
            ax.plot(median.index, median, color=colors[condition], label=label, lw=1.2)
            ax.fill_between(median.index, q25, q75, color=colors[condition], alpha=.2, lw=0)
        ax.axhline(1, color='.4', lw=.7)
        for boundary in [14.5, 64.5]:
            ax.axvline(boundary, color='.5', linestyle=':', lw=.7)
        ax.set_xlim(4, 95)
        ax.set_ylim(.65, 1.30)
        ax.set_title(letter + ' | ' + title + ' — proposed common axis', loc='left', fontsize=10)
        ax.set_xlabel('Global CS trial')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('Response mean / pre-CS baseline mean')
    axes[1].text(.5, -.23, 'Control n=18 at trials 81–94; otherwise 19', transform=axes[1].transAxes, ha='center', fontsize=8)
    axes[2].set_axis_off()
    axes[2].text(.05, .95, 'I | 10sTrace', fontsize=11, weight='bold', va='top', transform=axes[2].transAxes)
    axes[2].text(.05, .80, 'Placeholder\n\nAuthenticated cohort and\nprocessed ratios pending\n\nInterpretation: inconclusive', fontsize=11, va='top', transform=axes[2].transAxes)
    fig.suptitle('G–I display candidate — scientific approval pending | median and fish IQR | no smoothing or inference', fontsize=11)
    for ext in ['svg', 'png']:
        fig.savefig(out / ('row3-common-axis-candidate.' + ext), dpi=180)
    plt.close(fig)
    audit['display_candidate'] = {'status': 'proposed; not approved', 'y_limits': [.65, 1.30],
        'summary': 'equal-fish median with fish IQR', 'smoothing': 'none', 'inference': 'none',
        'sources': [audit['panels'][n]['panel_data'] for n in NAMES[:2]],
        'outputs': [{'path': str(out / ('row3-common-axis-candidate.' + ext)), 'sha256': digest(out / ('row3-common-axis-candidate.' + ext))} for ext in ['svg', 'png']]}
    (out / 'row3-common-axis-candidate.figure.json').write_text(json.dumps({
        **audit['display_candidate'], 'baseline_s': [-15, 0], 'metric_id': 'legacy_distal_angular_speed',
        'cohort_hashes': {n: audit['panels'][n]['identity']['cohort_hash'] for n in NAMES[:2]},
        'response_windows_s': {'G': [0, 9], 'H': [0, 13]}, 'script': audit['review_script'],
        'I': audit['I'], 'source_provenance': 'audit.json'}, indent=2) + '\n')
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    def rel(p):
        return html.escape(Path(os.path.relpath(p, out)).as_posix(), quote=True)
    catalog = json.loads((SRC / 'version-catalog.json').read_text())
    entries = [e for e in catalog['entries'] if e['title'] in ['Current G', 'Current H', 'H_bootstrap', 'H_split-bootstrap'] or e['title'].startswith('Original H')]
    cards = ''.join('<article><h2>' + html.escape(e['title']) + '</h2><p>' + html.escape(e['note']) +
                    '</p><a href="' + rel(e.get('svg') or e['original']) + '"><img src="' + rel(e['preview']) +
                    '" alt="' + html.escape(e['title']) + '"></a></article>' for e in entries)
    historical = ''.join('<li><a href="' + rel(x['path']) + '">' + html.escape(x['path']) + '</a></li>' for x in candidates if x['exists_now'] and x['path'].startswith('J:'))
    report = '''# Figure 2 G–I review — approval pending

This review preserves every source and does not change the assembly manifest or scientific registry. No new raw processing, inference or whole-figure rendering was run. A row-only common-axis candidate is saved beside this report, derived solely from the saved G/H plotted tables. No applicable AGENTS.md was found in the checked repository/SSD ancestor paths or repository file inventory.

| Available version | Scientific/display meaning | Status |
| --- | --- | --- |
| Current G | 29 Delay + 28 control; legacy metric; pre15; response [0,9); median/fish IQR | Current SVG/PNG/parquet/sidecar verified |
| Current H | 40 Trace + 19 control; legacy metric; pre15; response [0,13); median/fish IQR | Current SVG/PNG/parquet/sidecar verified |
| H bootstrap | Identical H fish ratios; median/pointwise 95% percentile bootstrap CI, 100 resamples, seed 10 | Combined condition axes; SVG available |
| H split bootstrap | Same inputs and bootstrap estimator | Separate condition plots with shared y scale; SVG available |
| Original H v1/v2/v3, 20261006 | Full59 legacy H; bootstrap / IQR / split bootstrap | Saved PNGs, not interchangeable band semantics; v2/v3 fixed y=.8–1.2 |
| Older full59 H | Saved PNG + fish-trial CSV and layout JSON | Historical presentation; no automatic selection |
| Older 53-fish H | Separate cohort (34 Trace + 19 controls), saved PNG | Scientific sensitivity, not a display-only alternative |
| G angular-L1 pre15/pre20 | Saved PNG/parquet/sidecars; distinct metric; pre20 also distinct baseline | Historical sensitivity only |
| I | No authenticated current cohort/processed ratio source recovered | Placeholder; 10sTrace inconclusive |

The inventory's saved paths and current existence/hash checks are in audit.json. It contains no saved G permutation/maxcontrast/LME plot PNG: those documented recipes are not available panel versions in this inventory. The old Delay LME failed influence diagnostics and did not localize simultaneous onset. No validated learning-onset or extinction claim is available; no extinction estimator is implemented. No Delay tests transfer to Trace. Uploaded stars have no authenticated inference provenance.

Current ratio = response mean / baseline mean, using finite metric values on detector-valid frames with FrameStep==1. It includes stationary valid frames; it does not use the moving-only conditional-intensity fields. It is a sample mean, not an integral or duration-weighted mean. The fields named total_activity are misleading. The rad/ms units cancel, leaving a dimensionless ratio. Finite numerator, finite strictly positive denominator are required; zero responses are permitted. No epsilon, winsorization, minimum valid-window fraction or additional fish threshold is applied by the ratio selector. Each fish/trial contributes once here (Contributing rows=1); repeated eligible outcome rows would otherwise be collapsed by their median. This aggregation is distinct from a ratio of pooled condition means.

Saved outcome ratios reproduce the plotted tables. G has 5,130 eligible fish/trial values and no exclusions, with 28 controls and 29 Delay fish at every trial. H has 5,296/5,310 eligible values: control fish 20230316_02 has both means missing and zero valid samples at trials 81–94; Trace coverage stays 40, control coverage falls from 19 to 18. Legends currently show cohort totals. Saved sample counts alone do not authenticate full-window coverage. G's minimum baseline count is 437; valid fraction/duration and upstream metric/event audits remain open.

The CS trial index comes from chronologically sorted Cycle events; the saved phase map is Pre5–14, Train15–64, Test65–94. Phase guides are 14.5/64.5; trials 5–94 are unsmoothed. Frame-level timing and protocol authentication remain separate gates. Catch identity is retained upstream; the selector does not filter catches separately.

Freeze decisions proposed for explicit user review:

1. **Cohort/input identity:** accept the specific saved G57 and exploratory H59 cohorts, or request changed eligibility. Full cohort hashes, input paths and hashes, source code hashes, panel-data/export hashes and verification results are in audit.json. File authentication is not cohort approval; upstream signal/alignment audit remains pending.
2. **Ratio/windows:** keep the already frozen [-15,0) baseline; approve [0,9) Delay and [0,13) Trace response windows, sample-mean ratio, valid stationary frames, and finite-positive denominator rule. Decide whether to add a minimum valid-window fraction or a small-baseline sensitivity analysis. Such changes require new versioned outcomes/panels, not relabeling existing files.
3. **Alignment/coverage:** approve global CS trials5–94, saved phase boundaries, and inclusion of available trials without a complete-case fish filter. Recommend documenting H's n=18 controls at81–94 and keeping a contributing-fish table alongside the panel. Approve coverage requirements only after valid fraction/duration and protocol audits.
4. **Summary/band:** proposed descriptive main row = equal-fish median with fish IQR for both G and H. IQR is spread, not a CI. If uncertainty is the intended message, choose bootstrap CI for both assays instead. Existing H bootstrap resamples observed fish values independently at each condition/trial (no units argument); it is pointwise, not simultaneous and does not preserve whole-fish trajectory draws across trials. A proposed alternative is resampling fish IDs once within condition, carrying complete trajectories and missingness through each draw, using 5,000 resamples/seed10 with a convergence check. That alternative is not yet generated or approved.
5. **Axes/layout:** propose combined conditions within each panel and a common G/H y range .65–1.30, ratio=1 reference. Current IQR extents are G .69653–1.24975 and H .89598–1.13246, so that proposed range shows both without clipping their bands. Current automatic G/H axes differ; this can exaggerate visual comparability. Inspect row3-common-axis-candidate.svg/png; it is rendered but not approved or installed in the main assembly. If separate condition plots are preferred, keep the same y range across all plots. All-fish extremes may exceed an IQR range and remain in plotted data.
6. **Smoothing:** propose none, retaining saved trial values and honest trial variation. Any smoother needs an explicit method/span/edge and missing-value rule, and a preserved unsmoothed version.
7. **Inference:** propose no significance, onset or extinction marks in this descriptive row. If inference is requested, prespecify the estimand, repeated-fish model/resampling, diagnostics, condition contrasts and multiplicity family across assays/trials. Pointwise bootstrap CI does not provide a simultaneous inference claim. Old failed Delay LME and historical reference annotations are not accepted results.
8. **Panel I:** retain an explicit unavailable placeholder and inconclusive interpretation until a reviewed cohort and processed inputs are authenticated. Do not use uploaded I as data.

These are proposed choices, not approval. Baseline and metric already have decisions; the rest remain pending. A display choice alone cannot close scientific gates. No scientific freeze or source selection has been recorded.
'''
    (out / 'review.md').write_text(report, encoding='utf-8')
    page = '<!doctype html><meta charset="utf-8"><title>Figure 2 G–I available versions</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;background:#f4f5f7;color:#222}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}article{background:white;padding:20px;border-radius:8px}img{width:100%}h2{font-size:20px}li{overflow-wrap:anywhere}</style><h1>Figure 2 G–I — review pending</h1><p>G: 29 Delay +28 control. H: 40 Trace +19 control (18 at trials81–94). Baseline [-15,0). I remains unavailable and inconclusive. No inference marks approved.</p><p><a href="review.md">Definitions and concrete freeze decisions</a> · <a href="audit.json">Hashes, numeric audit and inventory</a></p><main>' + cards + '<article><h2>I — 10sTrace</h2><p>Placeholder. Authenticated current cohort and processed ratio inputs unavailable. Interpretation: inconclusive.</p></article></main><h2>Historical saved alternatives</h2><p>Metric/cohort/baseline differences must remain explicit. These are historical references, not approved sources.</p><ul>' + historical + '</ul>'
    (out / 'comparison.html').write_text(page, encoding='utf-8')
    with (out / 'comparison.html').open('a', encoding='utf-8') as stream:
        stream.write('<h2>Proposed common-axis row — approval pending</h2><a href="row3-common-axis-candidate.svg"><img src="row3-common-axis-candidate.png" alt="Proposed G–I row with common G/H axes"></a>')
    print(json.dumps({'review_directory': str(out), 'hash_failures': audit['hash_check_failures'],
                      'historical_H_numeric_match': audit['historical_H_numeric_match'],
                      'audit': {n: audit['panels'][n]['audit'] for n in NAMES[:2]}}, indent=2))


if __name__ == '__main__':
    main()
