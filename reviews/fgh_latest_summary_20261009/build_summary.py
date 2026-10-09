"""A portable HTML review of all four latest F/G/H rows."""
from pathlib import Path
import base64, hashlib, html, json, os

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
CORRECTED=REPO/'reviews/fgh_full_bouts_baseline_samples_20261009'
DIRECT=REPO/'reviews/fgh_direct_bins_own_reference_20261009'
MEDIAN=REPO/'reviews/fgh_version4_quartersecond_means_20261009'
LAYOUT=REPO/'reviews/fgh_legacy_layout_20261009'
SELECTED=CORRECTED
COMPARISON=CORRECTED

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def data_uri(p,mime):
    return 'data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode('ascii')
def local_link(p):
    return html.escape(os.path.relpath(p,HERE).replace('\\','/'),quote=True)

variants=[
    {'id':'c-samples','title':'Version 1 · C · complete bout medians', 'status':'Corrected primary version',
     'description':'Each complete detected bout’s median log vigor is repeated on its eligible samples, before the plotting crop. No 0.5 s binning.',
     'baseline':'All eligible timepoints within [−15, 0) s, each carrying its complete bout’s median value.',
     'formula':'C = clip((x − m) / ((P90 − P10) / 2), −1, +1)',
     'validation':LAYOUT/'C_BoutSamples_validation.json'},
    {'id':'d-samples','title':'Version 1 · D · complete bout medians', 'status':'Corrected comparison',
     'description':'The same complete-bout medians and eligible sample intervals, with the median-preserving D denominator.',
     'baseline':'All eligible timepoints within [−15, 0) s, each carrying its complete bout’s median value.',
     'formula':'D = clip((x − m) / max(m − P10, P90 − m), −1, +1)',
     'validation':LAYOUT/'D_BoutSamples_validation.json'},
    {'id':'c-bins','title':'Version 2 · C · direct 0.5 s means', 'status':'Own bin baseline · zero median',
     'description':'Each displayed cell averages direct eligible framewise natural-log vigor, without bout-median substitution. Its baseline reference is calculated from these same unscaled bin means.',
     'baseline':'Finite unscaled 0.5 s baseline-bin means within [−15, 0) s; one vote per finite bin. The displayed baseline median is zero.',
     'formula':'C = clip((bin mean − m) / ((P90 − P10) / 2), −1, +1)',
     'validation':LAYOUT/'C_DirectBins_validation.json'},
    {'id':'median-bins','title':'Version 4 · direct 0.25 s means', 'status':'NaNs ignored · all 270 baselines defined',
     'description':'Each quarter-second cell averages direct eligible framewise natural-log vigor. Non-bout and invalid/ineligible samples remain NaN and are ignored. No minimum coverage threshold, no bout-median substitution, and no minimum-value replacement.',
     'baseline':'Arithmetic mean of finite unscaled 0.25 s bin means in [−20, 0) s; one vote per finite bin, with all-NaN bins ignored. All 270 trial references are defined. Subtract this mean without scaling.',
     'formula':'Δlog vigor = 0.25 s bin mean − mean(finite baseline-bin means in [−20, 0) s)',
     'validation':MEDIAN/'Version4_DirectBinMeans_validation.json'},
]
provenance={'summary_date_local':'2026-10-09','fish':{'F':'20221115_07','G':'20230310_08','H':'20221115_09'},
            'variants':[]}
cards=[]
for v in variants:
    validation=json.loads(v['validation'].read_text())
    files={Path(r['path']).suffix[1:]:Path(r['path']) for r in validation['outputs']}
    for r in validation['outputs']:
        assert digest(Path(r['path']))==r['sha256']
    vrecord={k:str(value) if isinstance(value,Path) else value for k,value in v.items()}
    vrecord['outputs']=validation['outputs'];provenance['variants'].append(vrecord)
    cards.append(f'''<section class="variant" id="{v['id']}" aria-labelledby="{v['id']}-title">
      <div class="variant-heading"><div><span class="badge {'selected' if v['id']=='c-samples' else ''}">{v['status']}</span>
      <h2 id="{v['id']}-title">{v['title']}</h2></div>
      <div class="downloads"><a download="{files['pdf'].name}" href="{data_uri(files['pdf'],'application/pdf')}">Download PDF</a>
      <a href="{local_link(files['svg'])}" target="_blank" rel="noopener">Open SVG</a></div></div>
      <p>{v['description']}</p><p class="baseline"><strong>Baseline:</strong> {v['baseline']}</p>
      <div class="formula">{v['formula']}</div>
      <a class="figure-link" href="{data_uri(files['png'],'image/png')}" download="{files['png'].name}" aria-label="Download {v['title']} as PNG">
      <img src="{data_uri(files['png'],'image/png')}" width="2700" height="1530" alt="{v['title']}: Delay 20221115_07, 3 s Trace 20230310_08, Control 20221115_09; one shared managua_r colourbar"></a>
      <div class="variant-foot"><span>Trials 5–94 · −20 to +20 s · shared managua_r {'colour range [−0.25, +0.25] log; data remain unclipped' if v['id']=='median-bins' else 'scale [−1, +1]'}</span>
      <span>{'Eligible sample intervals' if v['id'] in ['c-samples','d-samples'] else ('90 × 160 cells per fish' if v['id']=='median-bins' else '90 × 80 cells per fish')}</span></div></section>''')

confirmation=json.loads((CORRECTED/'numeric_verification.json').read_text())
assert confirmation['fish']['G']=='20230310_08'
provenance['numeric_verification']=confirmation
provenance['complete_bout_reconstruction']=json.loads((CORRECTED/'data_manifest.json').read_text())
provenance['current_direct_bin_reference']=json.loads((DIRECT/'data_manifest.json').read_text())
provenance['current_direct_bin_numeric_verification']=json.loads((DIRECT/'numeric_verification.json').read_text())
provenance['version4_bin_means']=json.loads((MEDIAN/'data_manifest.json').read_text())
provenance['version4_numeric_verification']=json.loads((MEDIAN/'numeric_verification.json').read_text())
provenance['previous_version4_balance_audit']=json.loads((REPO/'reviews/fgh_version4_baseline20_20261009/previous_v4_balance_summary.json').read_text())
provenance['current_example_trial_markers']=json.loads((LAYOUT/'example_trials.json').read_text())
provenance_path=HERE/'summary_provenance.json'
provenance_path.write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')

page='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Figure 1 F/G/H · latest heatmap versions</title>
<style>
:root{color-scheme:light;--ink:#202d38;--muted:#607080;--line:#dbe2e8;--paper:#fff;--bg:#f3f5f7}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1240px;margin:auto;padding:44px 28px 68px}h1,h2,h3{line-height:1.25;color:#182530}h1{font-size:34px;letter-spacing:-.8px;margin:8px 0 16px}h2{font-size:23px;margin:10px 0 14px}h3{font-size:18px;margin:0 0 12px}p{margin:0 0 14px}.eyebrow{font-size:12px;letter-spacing:1.8px;font-weight:700;color:var(--muted);text-transform:uppercase}
.lead{max-width:850px;color:var(--muted)}.fish-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}
.fish{background:var(--paper);border:1px solid var(--line);border-top:3px solid var(--color);border-radius:8px;padding:14px 18px}.fish strong{color:var(--color)}.fish code{display:block;color:var(--ink);margin-top:4px;font-size:15px}
nav{display:flex;flex-wrap:wrap;gap:10px;margin:26px 0}a{color:#28567b;text-underline-offset:3px}nav a,.downloads a{border:1px solid var(--line);border-radius:6px;padding:7px 12px;text-decoration:none;background:white;font-size:13px;font-weight:600}a:hover{color:#102f4c;background:#eef4f8}
.analysis,.comparison,.variant,.verification{background:var(--paper);border:1px solid var(--line);border-radius:10px;margin:22px 0;padding:26px}
ul{margin:8px 0 16px;padding-left:23px}li{padding-left:4px;margin:8px 0}.analysis-columns{display:grid;grid-template-columns:1fr 1fr;gap:26px}
.subtle{color:var(--muted);font-size:14px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;min-width:760px;font-size:14px}th,td{text-align:left;padding:12px 14px;vertical-align:top;border-bottom:1px solid var(--line)}th{background:#f3f6f8}td:first-child{font-weight:650}.formula{background:#f3f6f8;border-left:3px solid #8095a5;padding:12px 15px;font:14px/1.7 ui-monospace,Consolas,monospace;overflow-wrap:anywhere;margin:16px 0}
.variant-heading{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}.badge{display:inline-block;background:#f0f2f4;color:#536271;font-size:11px;font-weight:650;padding:3px 8px;border-radius:4px}.badge.selected{background:#e5f2ec;color:#236048}.downloads{display:flex;gap:8px;flex-wrap:wrap;margin-top:9px}.variant>p{font-size:15px}.baseline{color:var(--muted)}img{display:block;width:100%;height:auto}.figure-link{display:block;background:white}.variant-foot{display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:12px}.notice{border-left:3px solid #ae8a46;padding:10px 14px;background:#faf7ee;font-size:14px}.verified{color:#236048;font-weight:650}.verification code{overflow-wrap:anywhere}footer{font-size:12px;color:var(--muted);margin:28px 0}.print{cursor:pointer}
@media(max-width:720px){main{padding:24px 14px}h1{font-size:28px}.fish-grid,.analysis-columns{grid-template-columns:1fr}.analysis,.comparison,.variant,.verification{padding:18px}.variant-heading{display:block}.downloads{margin-bottom:16px}}
@media print{body{background:white}main{max-width:none;padding:0}nav,.downloads,.print{display:none}.variant{break-inside:avoid;break-before:page;border:0;padding:0}.fish-grid{grid-template-columns:repeat(3,1fr)}.analysis-columns{grid-template-columns:1fr 1fr}a{color:inherit}.analysis,.comparison,.verification{border:0;padding:12px 0}}
</style></head><body><main>
<header><div class="eyebrow">Figure 1 · F / G / H · 9 October 2026</div>
<h1>Latest single-fish heatmap versions</h1>
<p class="lead">Four current rows using the same fish, bout detector and display layout. The first three are unchanged. Version 4 now uses direct eligible log-vigor means in 0.25 s bins and a [−20, 0) s mean baseline, without scaling. Non-bout and invalid samples remain NaN. All 270 Version 4 baselines are defined.</p>
<div class="fish-grid"><div class="fish" style="--color:#d90072"><strong>F · Delay</strong><code>20221115_07</code></div>
<div class="fish" style="--color:#bb5b00"><strong>G · 3 s Trace</strong><code>20230310_08</code></div>
<div class="fish" style="--color:#087ea2"><strong>H · Control</strong><code>20221115_09</code></div></div>
<nav aria-label="Summary sections"><a href="#analysis">Analysis steps</a><a href="#comparison">Compare methods</a><a href="#c-samples">C · samples</a><a href="#d-samples">D · samples</a><a href="#c-bins">C · 0.5 s means</a><a href="#median-bins">V4 · 0.25 s means</a><a href="#verification">Data verification</a></nav></header>
<section class="analysis" id="analysis"><h2>Current analysis, step by step</h2>
<ul>
<li><strong>Read corrected tail angles and timing.</strong> Use each fish’s corrected angle frames, camera timing, stimulus protocol and angular-validity coverage. Check the source hashes. Frame times follow the constant camera cadence inferred from stable arrival runs.</li>
<li><strong>Calculate raw vigor before detecting bouts.</strong> Sum the 16 tail-angle components, calculate the wrapped change between consecutive valid frames, take its absolute value and divide by the inferred time step. The raw signal is angular speed in radians/ms; this is a framewise quantity, not a bout median.</li>
<li><strong>Detect bouts from raw vigor.</strong> Apply the shared median smoothing and rolling-envelope detector, then the amplitude, duration and interbout-gap rules. Bout detection occurs before calculating any bout summary; it is identical in all four versions.</li>
<li><strong>Keep eligible moving samples and take the natural logarithm.</strong> Require valid consecutive frames, a finite detector envelope, at least 80% angular tail coverage, membership in a detected moving bout, and finite positive raw vigor. Non-bout and invalid/ineligible samples remain NaN in every current version. Version 4 no longer replaces any missing value with −∞ or another floor.</li>
<li><strong>Construct the values for each version.</strong> C/D sample versions repeat the median of each complete bout’s eligible log samples on that bout’s eligible frames. Version 2 averages direct eligible log samples per 0.5 s bin. Version 4 averages direct eligible log samples per 0.25 s bin. Ignore NaNs; retain any bin with one or more eligible samples, regardless of missing coverage. All-NaN bins stay missing. Neither binned version substitutes bout medians.</li>
<li><strong>Calculate a separate matching baseline for each trial.</strong> C/D samples retain their [−15, 0) s timepoint P10/P50/P90; Version 2 retains its [−15, 0) s finite bin-mean P10/P50/P90. Version 4 uses the arithmetic mean of its finite quarter-second bin means in [−20, 0) s, one vote per finite bin. All-NaN baseline bins are ignored. It calculates no P10/P90.</li>
<li><strong>Align and crop the display to CS.</strong> Display [−20, +20) s, trials 5–94. Pre = 5–14; Train = 15–64; Test = 65–94. Version 2 has 80 half-second cells per trial. Version 4 has 160 quarter-second cells, with up to 80 baseline bins. All bins align to CS onset.</li>
<li><strong>Centre; scale only the C/D rows.</strong> C/D rows retain matching median subtraction and percentile scaling clipped to [−1, +1]. Version 4 subtracts its matching baseline mean, with no percentile calculation, division or numerical clipping. A Version 4 trial would be undefined only if it had no finite baseline-bin means. All 270 current references are defined and their centred baseline means are zero to numerical precision.</li>
<li><strong>Render the heatmaps.</strong> Use one shared managua_r colourbar per three-panel row, thin boxed axes, white phase boundaries, time ticks at −20/0/+20 s and phase names on the left of F. Broad green lines mark CS onset and offset at 0 and 10 s. Black left-pointing arrowheads to the right of G mark provisional panel E example trials 9, 17, 63, 66 and 93 in 20230310_08. The C/D rows show scaled “Vigor relative to baseline” with colour limits ±1. Version 4 shows “Log vigor relative to baseline” with fixed limits ±0.25: values beyond those limits use endpoint colours but retain their actual numeric values. Sample rectangles preserve eligible sample durations without filling gaps.</li>
</ul>
<details><summary>Detector settings used in all versions</summary><p class="subtle">Median window: 10 ms (7 samples). Rolling maximum: 28.57 ms (21 samples); rolling minimum: 571.43 ms (401 samples). Envelope threshold: 4°/ms; raw peak threshold: 1°/ms; minimum bout duration: 57.14 ms; maximum interbout gap: 14.29 ms. Rolling windows require finite support; invalid-frame gaps are not bridged.</p></details>
<div class="notice"><strong>Interpretation:</strong> the first three rows retain their median-zero baselines and existing undefined-scale rules. Current Version 4 has all 270 trial references defined and a zero baseline mean. Black Version 4 cells have no eligible contribution; they do not represent a measured vigor of zero. Mean centring does not guarantee a zero median or equal counts above and below zero.</div>
<p class="subtle" style="margin-top:14px"><strong>Full-bout consequence:</strong> some baseline bouts cross CS onset. Their complete medians therefore include post-onset eligible samples: 38 such bout/trial occurrences in Delay, 66 in 3 s Trace and 13 in Control. This follows the requested complete-bout treatment.</p>
</section>
<section class="comparison" id="comparison"><h2>What changes between the versions?</h2><div class="table-wrap"><table>
<thead><tr><th>Version</th><th>Value before centring</th><th>Baseline voting unit</th><th>Denominator</th></tr></thead>
<tbody><tr><td>Version 1 · C</td><td>Bout median log vigor on each eligible sample</td><td>Eligible sample</td><td>(P90 − P10) / 2</td></tr>
<tr><td>Version 1 · D</td><td>Same bout median and sample intervals</td><td>Eligible sample</td><td>max(m − P10, P90 − m)</td></tr>
<tr><td>Version 2 · C</td><td>Mean of direct eligible log samples per 0.5 s cell</td><td>Finite direct baseline-bin mean</td><td>(P90 − P10) / 2</td></tr>
<tr><td>Version 4</td><td>Mean of direct eligible log samples per 0.25 s cell; non-bout frames stay NaN</td><td>Finite quarter-second bin means in [−20, 0)</td><td>None: subtract baseline mean only; all 270 trial references defined</td></tr></tbody></table></div>
<p class="subtle" style="margin-top:14px">C/D sample rows share the same P10/P50/P90 per trial and differ only in their denominator. Version 2 uses its own baseline-bin statistics: it changes bout-median substitution, temporal aggregation and baseline weighting. D cannot have a larger absolute value than C for the same centred sample.</p>
<p class="subtle"><strong>Version 4:</strong> quarter-second bins give finer temporal resolution, with fewer eligible samples per bin and potentially more missing bins. The current version ignores non-bout NaNs and describes eligible moving vigor conditional on there being an eligible contribution. The baseline gives each finite bin one vote, regardless of its sample count. The ±0.25 limits affect colour saturation only; centred numeric values remain unclipped. The earlier −∞ experiment is preserved as history.</p>
<p class="subtle"><strong>Why this is not ordinary min–max:</strong> an affine transform mapping P10 to −1 and P90 to +1 centres their midpoint, not necessarily the median. Median subtraction cancels in that transform. C/D preserve the requested baseline-median zero instead; asymmetric percentile distances mean P10/P90 do not both necessarily map exactly to −1/+1.</p>
<p class="subtle"><strong>Why the earlier direct-bin row looked blue:</strong> direct log means were centred/scaled against a bout-median reference. In a typical 3 s Trace trial, its baseline-cell median was −1 and 70% of finite baseline cells clipped at −1. That mismatched reference did not meet the zero-centred display goal. Version 2 now uses its own baseline-bin means; its displayed baseline median is zero to numerical precision.</p>
<p class="subtle">Earlier exports remain preserved as history. C/D retain the full-bout and timepoint-baseline corrections; Version 2 uses the subsequently approved baseline-bin population. No raw-vigor reconstruction, detector, eligibility or direct cell mean was changed by the reference fix.</p></section>
__CARDS__
<section class="verification" id="verification"><h2>Confirmed source: 20230310_08</h2>
<p class="verified">The 3 s Trace panels use this fish’s actual data in all four rows.</p>
<ul><li>The frame table was reconstructed from <code>F:/Digested Data/all3sTrace-full-v1/Processed data/20230310_08/</code>, using its camera, protocol, corrected angle and coverage files.</li>
<li>Independent readback recalculated complete-bout medians and their repeated frame values, and checked the C/D timepoint quantiles. Version 2’s direct means and eligible counts are unchanged; its own baseline-bin quantiles and zero baseline medians were checked for every defined trial.</li>
<li>Version 4 reuses the same hash-verified direct eligible frame log values, with non-bout and invalid samples retained as NaN. Every quarter-second mean was independently verified with grouped finite frame values. All-NaN bin masks, frame counts, baseline means and trial flags were checked. All 270 baseline means are defined and centred to within 9.5 × 10<sup>−16</sup> log units.</li>
<li>The source table differs from the previous 20230307_12 fish. Delay and Control retain their selected frame data.</li>
<li>Exported SVG values, sample/bin intervals, row positions and colour fills were checked; the four PDFs were visually inspected.</li></ul>
<p class="subtle"><strong>Legacy layout:</strong> broad green CS boundaries, block names left of F, and G example arrows at trials 9, 17, 63, 66 and 93 are retained. The first three figures and their numeric calculations remain unchanged.</p>
<details open><summary>Per-trial baseline balance audit</summary><p>The original eligible-only median Version 4 had exactly equal above/below-zero counts in all 270 trials, excluding ties within 10<sup>−12</sup> log units. The earlier −∞ mean experiment had all 270 trials undefined. Current Version 4 removes that replacement and uses 0.25 s eligible-only means: all 270 trials are defined, and their baseline means are zero. Only 8 trials have equal above/below counts; the maximum difference is 33 bins. This follows mean centring, which does not enforce a 50/50 sign split.</p>
<p><a download="previous_v4_per_trial_balance.csv" href="__PREVIOUS_AUDIT__">Download original Version 4 per-trial balance audit</a> · <a download="current_v4_per_trial_baseline_balance.csv" href="__CURRENT_AUDIT__">Download current mean-based trial status audit</a></p></details>
<p style="margin-top:16px"><a download="summary_provenance.json" href="__PROVENANCE__">Download provenance and artifact hashes</a></p>
</section>
<footer>Figures and PDF downloads are embedded for offline viewing. SVG links refer to workspace exports. Complete-bout medians are retained for C/D; Versions 2 and 4 use their own baseline-bin references. Version 4 is unscaled. Earlier exports are preserved. <button class="print" onclick="window.print()">Print summary</button></footer>
</main></body></html>'''
page=page.replace('__CARDS__','\n'.join(cards)).replace('__PROVENANCE__',data_uri(provenance_path,'application/json'))
page=page.replace('__PREVIOUS_AUDIT__',data_uri(REPO/'reviews/fgh_version4_baseline20_20261009/previous_v4_per_trial_balance.csv','text/csv'))
page=page.replace('__CURRENT_AUDIT__',data_uri(MEDIAN/'per_trial_baseline_balance.csv','text/csv'))
target=HERE/'index.html'
target.write_text(page,encoding='utf-8')
assert page.count('<img ')==4
assert page.count('data:application/pdf;base64,')==4
assert '20230310_08' in page
print(json.dumps({'html':str(target),'bytes':target.stat().st_size,'variants':len(variants)},indent=2))
