"""Portable comparison preserving all four prior rows and adding candidates."""
from pathlib import Path
import base64,hashlib,html,json,re
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
OLD=REPO/'reviews/fgh_latest_summary_20261009'
V5=REPO/'reviews/fgh_version5_onesecond_means_20261009'
V6=REPO/'reviews/fgh_version6_trial_centralband_20261009'
V6FOUR=REPO/'reviews/fgh_version6_trial_quartiles_20261009'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def uri(p,mime):return 'data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()
def esc(s):return html.escape(str(s))
def download(p,mime,label):return f'<a download="{p.name}" href="{uri(p,mime)}">{label}</a>'

old=(OLD/'index.html').read_text(encoding='utf-8')
styles=re.search(r'<style>(.*?)</style>',old,re.S).group(1)
prior_cards=re.findall(r'<section class="variant".*?</section>',old,re.S)
assert len(prior_cards)==4
# Embed the old SVG downloads too, so every figure is portable.
for i,card in enumerate(prior_cards):
    def replace_svg(m):
        path=(OLD/m.group(1)).resolve()
        return 'href="'+uri(path,'image/svg+xml')+'"'
    prior_cards[i]=re.sub(r'href="([^"]+\.svg)"',replace_svg,card)

old_prov=json.loads((OLD/'summary_provenance.json').read_text())
preserved=json.loads((HERE/'preservation_before.json').read_text())
changed=[dict(r,current_sha256=digest(Path(r['path']))) for r in preserved if digest(Path(r['path']))!=r['sha256']]
assert all(Path(r['path'])==REPO/'configs/paper-figures/figure-elements.json' for r in changed)
preservation={'files_checked':len(preserved),'unchanged_files':len(preserved)-len(changed),
    'all_existing_figures_data_and_selection_records_unchanged':True,'inventory':preserved,
    'concurrent_workspace_changes':changed,
    'concurrent_change_note':'figure-elements.json changed during this run; none of the candidate scripts writes it. It was not restored or overwritten.'}
(HERE/'preservation_verification.json').write_text(json.dumps(preservation,indent=2)+'\n')
v5report=json.loads((V5/'numeric_verification.json').read_text())
v6report=json.loads((V6/'numeric_verification.json').read_text())
manifest6=json.loads((V6/'data_manifest.json').read_text())
mapping=manifest6['colour_mapping']
mapping4=json.loads((V6FOUR/'data_manifest.json').read_text())['colour_mapping']

specs=[
 ('v5','Version 5: 1 s means, original continuous colours','Version5_DirectBinMeans',
  ['Direct eligible framewise natural-log vigor; non-bout and invalid values stay NaN.',
   'Arithmetic means recalculated directly from finite frames in 1 s bins; any finite sample suffices. All-NaN bins remain black.',
   '40 bins per trial, with up to 20 baseline bins. Trial baseline = arithmetic mean of its finite new baseline-bin means in [-20,0) s, one vote per bin.',
   'Subtract this baseline mean. No percentile scaling or numerical clipping. Linear managua_r, fixed colour limits -0.25 to +0.25 log units; values outside retain their numeric values.']),
 ('v4-contrast','Version 4: separate continuous contrast candidate','Version4_Contrast',
  ['Uses the existing Version 4 quarter-second data and baseline mean without recalculation.',
   'Display coordinate u = 0.5 + 0.5 sign(d) sqrt(min(|d| / 0.25, 1)), where d is delta log vigor.',
   'Samples continuous managua_r at u. Zero colour and endpoint limits remain the same; small differences receive stronger colour separation.',
   'Only the colour mapping changes. The colourbar uses the inverse mapping and remains labelled in log units.']),
 ('v5-contrast','Version 5: separate continuous contrast candidate','Version5_Contrast',
  ['Uses exactly the new Version 5 1 s data and its baseline mean.',
   'Uses the same signed square-root colour mapping described above. No change to numeric values, masks or trial references.']),
 ('v6-four','Earlier Version 6: four colours (superseded candidate)','Version6_TrialQuartiles',
  ['Starts from current Version 4 finite quarter-second arithmetic-mean bins. All original numeric columns are preserved verbatim in the CSV exports.',
   'For each trial separately, calculate P25, P50 and P75 from finite unscaled bin means before CS in [-20,0) s, one vote per finite bin. Linear quantile interpolation.',
   'Subtract that trial baseline P50. Thus the baseline median and central boundary are zero; this changes mean centring to median centring for Version 6 only.',
   'Four classes: x < P25; P25 <= x < P50; P50 <= x < P75; x >= P75. Ties at boundaries enter the upper class. NaNs remain black and are never assigned a class.',
   'The common discrete scale is Q1-Q4, also exported as ordinal display scores [-1, -1/3, +1/3, +1]. These scores describe within-trial baseline rank, not a continuous amplitude normalization.',
   'Colours have the same rank meaning in every trial, but physical log thresholds differ. The shared colourbar therefore shows Q1-Q4; the exact 270 per-trial thresholds are available below.',
   'Sample four managua_r colours at positions '+str(mapping4['managua_r_sample_positions'])+': '+', '.join(mapping4['colours'])+'. No numerical clipping of log values.'])]
specs.insert(3,('v6','Version 6: five bands with a narrow dark managua_r centre','Version6_CentralBand',[
    'Uses the current Version 4 0.25 s mean bins; all original columns remain verbatim in the exports.',
    'For each trial, use finite baseline-bin means in [-20,0) s to calculate P25, P45, P50, P55 and P75 with linear quantile interpolation.',
    'Subtract baseline P50, so the baseline median is zero. The small central band spans P45 to P55, the middle 10% of the baseline distribution, and contains zero in every trial.',
    'Five classes: x < P25; P25 <= x < P45; P45 <= x < P55; P55 <= x < P75; x >= P75. Exact boundary ties enter the upper class. Black is reserved for missing values.',
    'All five colours follow managua_r at positions 0, 0.25, 0.5, 0.75 and 1. The dark purple midpoint marks the centre; pure black is reserved for missing cells.',
    'Common ordinal display scores [-1, -0.5, 0, +0.5, +1] put all trials on the same five-band scale. These describe within-trial baseline rank, not continuously normalized physical amplitude.',
    'The shared colourbar shows Low, Below, Centre, Above and High. Physical log thresholds vary by trial and are exported in the 270-row threshold table. Original mean-centred and new median-centred log values remain available; neither is numerically clipped.',
    'The earlier four-colour candidate is preserved below as superseded history.']))
records=[];cards=[]
for id,title,kind,bullets in specs:
    vp=HERE/f'{kind}_validation.json';v=json.loads(vp.read_text())
    for row in v['outputs']+v['source_numeric_files']:
        assert digest(Path(row['path']))==row['sha256']
    source_manifest=Path(v['source_data_manifest']);assert digest(source_manifest)==v['source_data_manifest_sha256']
    files={Path(r['path']).suffix:Path(r['path']) for r in v['outputs']}
    records.append({'id':id,'title':title,'processing_bullets':bullets,'validation_path':str(vp),'validation_sha256':digest(vp),'validation':v})
    cards.append(f'<section class="variant" id="{id}"><div class="variant-heading"><h2>{esc(title)}</h2><div class="downloads">'+
        download(files['.pdf'],'application/pdf','Download PDF')+download(files['.svg'],'image/svg+xml','Download SVG')+'</div></div><ul>'+''.join('<li>'+esc(b)+'</li>' for b in bullets)+
        '</ul><img src="'+uri(files['.png'],'image/png')+'" width="2700" height="1530" alt="'+esc(title)+'"></section>')

provenance={'date_local':'2026-10-09','status':'review candidates; no freeze or selection change',
    'previous_summary_sha256':digest(OLD/'index.html'),'previous_four_rows':old_prov,
    'new_candidates':records,'version5_manifest':json.loads((V5/'data_manifest.json').read_text()),
    'version5_numeric_verification':v5report,'version5_exported_verification':json.loads((HERE/'version5_exported_data_verification.json').read_text()),
    'version6_manifest':manifest6,'version6_numeric_verification':v6report,
    'earlier_four_colour_manifest':json.loads((V6FOUR/'data_manifest.json').read_text()),
    'renderer_revisions':[{'path':str(HERE/name),'sha256':digest(HERE/name)} for name in ['render_candidates_fourband.py','render_candidates.py']],
    'preservation':preservation,'builder_sha256':digest(Path(__file__))}
renderer_hashes={r['sha256'] for r in provenance['renderer_revisions']}
assert all(r['validation']['renderer_sha256'] in renderer_hashes for r in records)
prov=HERE/'summary_provenance.json';prov.write_text(json.dumps(provenance,indent=2)+'\n')
nav=''.join(f'<a href="#{id}">{esc(title.split(":")[0]+ (" contrast" if "contrast" in id else ""))}</a>' for id,title,_,_ in specs)
v5rows=''.join('<tr><td>'+p['panel']+'</td><td>'+str(p['finite_display_bins'])+'</td><td>'+str(p['empty_bins'])+'</td></tr>' for p in v5report['panels'])
swatches=''.join(f'<span style="display:inline-block;padding:12px 18px;background:{c};color:{"white" if i in [1,2,3] else "black"}">{mapping["band_labels"][i]}: {c}</span>' for i,c in enumerate(mapping['colours']))
page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F/G/H colour and binning comparison</title><style>'+styles+'</style></head><body><main>'
page+='''<header><div class="eyebrow">F / G / H · 9 October 2026 · review candidates</div><h1>Colour clarity and binning comparison</h1>
<p class="lead">The four existing rows are preserved below. New candidates compare 1 s bins, stronger continuous contrast, and five per-trial baseline bands with a small dark managua_r centre. The earlier four-colour candidate is preserved as history. No panel E or frozen figure has been changed.</p>
<div class="fish-grid"><div class="fish" style="--color:#d90072"><strong>F · Delay</strong><code>20221115_07</code></div><div class="fish" style="--color:#bb5b00"><strong>G · 3 s Trace</strong><code>20230310_08</code></div><div class="fish" style="--color:#087ea2"><strong>H · Control</strong><code>20221115_09</code></div></div><nav><a href="#analysis">Processing</a><a href="#median-bins">Current V4</a>'''+nav+'<a href="#verification">Verification</a></nav></header>'
page+='''<section class="analysis" id="analysis"><h2>Shared processing and comparisons</h2><ul>
<li>Raw vigor is the absolute wrapped consecutive-frame change in the sum of 16 corrected tail-angle components, divided by the inferred camera interval (rad/ms), calculated before bout detection.</li>
<li>Shared detector: median smoothing 10 ms (7 samples); rolling maximum 28.57 ms (21); minimum 571.43 ms (401); envelope threshold 4 degrees/ms; raw peak threshold 1 degree/ms; minimum duration 57.14 ms; maximum interbout gap 14.29 ms.</li>
<li>Eligible values require valid consecutive frames, finite envelope, at least 80% angular coverage, detected bout membership and finite positive raw vigor. Take the natural log. All other frames stay NaN. Each nonempty bin is retained, regardless of coverage fraction.</li>
<li>All rows display trials 5-94 over [-20,20) s. Pre = 5-14; Train = 15-64; Test = 65-94. Block names remain left of F; the six CS boundaries at 0 and 10 s retain #0d8136, 2.4 pt, alpha 0.8. G arrows remain at trials 9,17,63,66,93. Delay US is 9 s, Trace US is 13 s, during training; no Control guide.</li>
<li>Compare current V4 with V5 to inspect bin width and its matching mean baseline. Compare each continuous row with its contrast candidate to inspect colour mapping alone. V6 also changes the reference from baseline mean to baseline median and uses trial-specific percentile bands.</li></ul>
<p class="notice">P50 = zero in V6 because its baseline median is subtracted. Mean-zero V4/V5 does not imply median-zero. V6's colours compare ranks; equal colours across trials do not imply equal physical log-vigor differences.</p></section>'''
page+='<section class="comparison" id="comparison"><h2>Version 5 numerical check</h2><p>All 270 trial baselines are defined. Maximum absolute centred baseline mean: '+f'{max(p["maximum_absolute_baseline_mean"] for p in v5report["panels"]):.2e}'+ ' log units.</p><div class="table-wrap"><table><thead><tr><th>Panel</th><th>Finite 1 s bins / 3600</th><th>Missing bins</th></tr></thead><tbody>'+v5rows+'</tbody></table></div><p>Every new mean was recomputed from eligible frame values and independently checked. Exported means also match sample-count-weighted quarter-bin reconstruction. Unweighted averaging of old means can differ by up to 1.285 log units.</p></section>'
page+=''.join(cards)+'<section class="comparison"><h2>Four preserved existing rows</h2><p>Historical names and original processing definitions are retained: Version 1 C, Version 1 D, Version 2 C, and current Version 4. No separate Version 3 recipe is established.</p></section>'+''.join(prior_cards)
page+='<section class="verification" id="verification"><h2>Verification and provenance</h2><p class="verified">Source hashes, exported data, SVG cell geometry and colours, and scientific annotations verified. All original figure exports, data tables, freeze records and current selections retain their hashes; '+str(preservation['unchanged_files'])+' inventoried files are unchanged.</p><p>Version 6: '+str(v6report['defined_trials'])+' defined references; maximum absolute centred baseline median '+f'{v6report["max_abs_baseline_median"]:.2e}'+ ' log units; '+str(v6report['collapsed_band_trials'])+' trials with collapsed band boundaries. Zero lies in the central P45-P55 band in every trial. Exact ties belong to the upper class; classes are never forced to have equal counts.</p><div>'+swatches+'</div><p>'+download(V6/'per_trial_quartile_thresholds.csv','text/csv','Download all per-trial thresholds')+' · '+download(V5/'per_trial_baseline_balance.csv','text/csv','Download Version 5 baseline audit')+' · '+download(prov,'application/json','Download processing provenance and hashes')+'</p><details><summary>Concurrent workspace change</summary><p>The shared figure-elements.json style configuration changed during this run. These scripts do not write it; its before/after hashes are recorded in provenance. No existing figure was restyled or refrozen.</p></details></section><footer>All figure PNGs, SVGs and PDF downloads are embedded for offline viewing. Candidates remain unselected. Frozen artifacts and the original four-row summary are preserved.</footer></main></body></html>'
target=HERE/'index.html';target.write_text(page,encoding='utf-8')
assert page.count('<img ')==9 and page.count('data:application/pdf;base64,')==9
print(json.dumps({'html':str(target),'variants':9,'bytes':target.stat().st_size,'unchanged_files':preservation['unchanged_files']}))
