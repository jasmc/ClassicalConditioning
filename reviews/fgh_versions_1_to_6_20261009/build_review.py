"""Organize the existing verified F/G/H options without changing any figure."""
from pathlib import Path
import base64, hashlib, html, json

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
HERE.mkdir(exist_ok=True)
R=REPO/'reviews'
LAYOUT=R/'fgh_legacy_layout_20261009'
CANDIDATES=R/'fgh_colour_binning_candidates_20261009'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def uri(p,mime):return 'data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode('ascii')
def esc(s):return html.escape(str(s))
def load(p):return json.loads(p.read_text(encoding='utf-8'))

common=[
 'Use the same fish: F Delay 20221115_07, G 3 s Trace 20230310_08, H Control 20221115_09.',
 'Calculate raw framewise vigor before bout detection: sum 16 corrected tail-angle components, take the absolute wrapped consecutive-frame change, and divide by the inferred camera interval. Raw units are rad/ms.',
 'Use the same bout detector in every version: median smoothing 10 ms (7 samples); rolling maximum 28.57 ms (21 samples); minimum 571.43 ms (401 samples); envelope threshold 4 degrees/ms; peak threshold 1 degree/ms; minimum duration 57.14 ms; maximum gap 14.29 ms.',
 'Eligible samples require valid consecutive frames, finite detector envelope, at least 80% angular coverage, detected bout membership and finite positive raw vigor. Take the natural log. Non-bout and invalid/ineligible samples stay NaN.',
 'Binned versions average the finite eligible log values, not raw vigor followed by a log. Any single finite sample is sufficient; NaNs are ignored and all-NaN bins remain missing. There is no minimum coverage gate.',
 'Display trials 5-94 and [-20,20) s relative to CS. Pre = 5-14, Train = 15-64, Test = 65-94. Retain strong green CS boundaries at 0/10 s, phase names left of F, a shared colourbar and G arrows at trials 9/17/63/66/93.',
 'Missing cells are black. In V1 C/D and V2 C, Control trial 16 is black because its percentile scale is undefined. V6 has all 270 references defined.'
]

versions=[
 {'n':1,'title':'Complete-bout medians on samples','summary':'Summarize each bout, then preserve its eligible sample timing. Two percentile scales, C and D.',
  'bin':'No bins','baseline':'[-15,0) s; every eligible timepoint carrying its complete-bout median',
  'zero':'Baseline median','colours':'Continuous managua_r; scaled range [-1,+1]',
  'steps':[
   'For each complete detected bout, calculate the median of all its eligible natural-log vigor samples, including support beyond the display crop.',
   'Repeat that bout median on each of its eligible frame timepoints, preserving ineligible gaps; crop only afterwards.',
   'For each trial, use eligible timepoints in [-15,0) s to calculate P10, m=P50 and P90. Every timepoint votes; longer eligible bout duration contributes more votes.',
   'Subtract m, divide by the selected C or D denominator, then numerically clip to [-1,+1]. A nonpositive or missing denominator makes the trial undefined.'
  ],
  'difference':'C and D use exactly the same bout summaries and baseline population. Only the scaling denominator differs.',
  'consider':'Retains fine sample timing but removes variation within each bout. Complete medians of bouts crossing CS can include post-CS samples, even for baseline timepoints. C/D are median-centred percentile scales, not ordinary min-max scaling.',
  'options':[
   {'id':'v1-c','title':'Version 1 C','kind':'C_BoutSamples','validation':LAYOUT/'C_BoutSamples_validation.json',
    'formula':'C = clip((x - m) / ((P90 - P10) / 2), -1, +1)',
    'difference':'Uses half the P10-P90 range. Often gives stronger contrast and more saturation than D.'},
   {'id':'v1-d','title':'Version 1 D','kind':'D_BoutSamples','validation':LAYOUT/'D_BoutSamples_validation.json',
    'formula':'D = clip((x - m) / max(m - P10, P90 - m), -1, +1)',
    'difference':'Uses the larger distance from the median to either percentile. Absolute scaled values cannot exceed C for the same sample.'}
  ]},
 {'n':2,'title':'Direct log means in 0.5 s bins','summary':'Replace bout-median substitution with direct framewise log means, using a matching baseline-bin percentile scale.',
  'bin':'0.5 s; 80 bins/trial','baseline':'[-15,0) s; one vote per finite unscaled baseline-bin mean',
  'zero':'Median of baseline-bin means','colours':'Continuous managua_r; scaled range [-1,+1]',
  'steps':[
   'Average direct eligible framewise log vigor in each CS-aligned 0.5 s bin. Do not substitute bout medians.',
   'Within each trial, calculate P10, m=P50 and P90 from finite unscaled baseline-bin means in [-15,0) s. Each finite bin has one vote, regardless of its eligible frame count.',
   'Subtract m, divide by (P90-P10)/2 and numerically clip to [-1,+1]. Missing or zero percentile range makes the trial undefined.'
  ],
  'difference':'Compared with Version 1 C, changes the input value, temporal aggregation and baseline voting unit. Retains the same C formula and 15 s baseline interval.',
  'consider':'Describes direct eligible moving vigor, but the scale varies by trial. Equal colours across trials need not represent equal physical log differences. The older blue-offset row used a mismatched bout-median reference and is not this current Version 2.',
  'options':[{'id':'v2','title':'Version 2 C','kind':'C_DirectBins','validation':LAYOUT/'C_DirectBins_validation.json',
    'formula':'C = clip((bin mean - m) / ((P90 - P10) / 2), -1, +1)',
    'difference':'Displayed baseline-bin median is zero for all 269 defined fish/trials; Control trial 16 remains undefined.'}]},
 {'n':3,'title':'No established recipe','summary':'The historical numbering contains a gap.',
  'bin':'Not established','baseline':'Not established','zero':'Not established','colours':'No figure',
  'steps':[],
  'difference':'The recorded rows were Version 1 C, Version 1 D, Version 2 C and Version 4. Version 1 D is a second Version 1 option, not an approved Version 3.',
  'consider':'No formula or figure has been invented or renumbered to fill this gap.', 'options':[]},
 {'n':4,'title':'Direct log means in 0.25 s bins','summary':'Keep physical log differences: subtract a matching baseline mean, without percentile scaling.',
  'bin':'0.25 s; 160 bins/trial','baseline':'[-20,0) s; mean of finite baseline-bin means, one vote per bin',
  'zero':'Baseline mean; median may differ','colours':'Continuous managua_r; fixed limits +/-0.25 log units',
  'steps':[
   'Average direct eligible framewise log vigor in each 0.25 s bin.',
   'For each trial, take the arithmetic mean of its finite baseline-bin means in [-20,0) s: up to 80 baseline bins, one vote per finite bin.',
   'Subtract this baseline mean from each bin mean. No P10/P90 scale, division or numerical clipping.',
   'Render with fixed colour limits [-0.25,+0.25]. Values beyond the limits use endpoint colours while their full numeric values remain in exports.'
  ],
  'difference':'Compared with Version 2, uses finer bins, a 20 s baseline and mean subtraction; removes percentile scaling and numeric clipping.',
  'consider':'Preserves physical log-unit differences across trials and fish. Fine bins give more temporal detail and more missing cells. Mean-zero does not guarantee median-zero or equal positive/negative counts. Earlier median and -infinity experiments are historical, not this recipe.',
  'options':[
   {'id':'v4-linear','title':'Version 4 linear','kind':'Version4_DirectBinMeans','validation':R/'fgh_version4_quartersecond_means_20261009/Version4_DirectBinMeans_validation.json',
    'formula':'d = bin mean - mean(finite baseline-bin means)',
    'difference':'Linear colour mapping: u = (clip(d,-0.25,+0.25) + 0.25) / 0.5.'},
   {'id':'v4-contrast','title':'Version 4 contrast','kind':'Version4_Contrast','validation':CANDIDATES/'Version4_Contrast_validation.json',
    'formula':'Same d; colour coordinate u = 0.5 + 0.5 sign(d) sqrt(min(abs(d)/0.25,1))',
    'difference':'Same numeric data, baseline and masks as V4 linear. Only the colour mapping changes, increasing separation near zero. Colourbar remains in log units.'}
  ]},
 {'n':5,'title':'Direct log means in 1 s bins','summary':'The Version 4 recipe with coarser bins and its baseline recomputed from those new bins.',
  'bin':'1 s; 40 bins/trial','baseline':'[-20,0) s; mean of finite new baseline-bin means, one vote per bin',
  'zero':'Baseline mean; median may differ','colours':'Continuous managua_r; fixed limits +/-0.25 log units',
  'steps':[
   'Recalculate each 1 s mean directly from eligible framewise log values. Do not simply average the old 0.25 s means: eligible sample counts differ.',
   'For each trial, calculate the arithmetic mean of its finite new 1 s baseline-bin means in [-20,0) s, up to 20 bins. Each finite bin has one vote.',
   'Subtract that baseline mean. Retain physical log units, no percentile scaling and no numeric clipping.',
   'Use the same +/-0.25 colour limits as Version 4, with either linear or contrast rendering.'
  ],
  'difference':'Compared with Version 4, only the bin width and the matching recomputed baseline change. Linear versus contrast is a presentation choice, not another scientific recipe.',
  'consider':'More eligible samples contribute to each wider bin and fewer bins are missing, but short temporal features are averaged together. Physical log differences remain comparable across trials and fish; visual endpoint saturation still hides exact magnitudes beyond +/-0.25.',
  'options':[
   {'id':'v5-linear','title':'Version 5 linear','kind':'Version5_DirectBinMeans','validation':CANDIDATES/'Version5_DirectBinMeans_validation.json',
    'formula':'d = 1 s bin mean - mean(finite 1 s baseline-bin means)',
    'difference':'Linear continuous colours with the same physical scale as Version 4 linear.'},
   {'id':'v5-contrast','title':'Version 5 contrast','kind':'Version5_Contrast','validation':CANDIDATES/'Version5_Contrast_validation.json',
    'formula':'Same d; colour coordinate u = 0.5 + 0.5 sign(d) sqrt(min(abs(d)/0.25,1))',
    'difference':'Exactly the same numeric values as V5 linear; stronger visual separation of small differences near zero.'}
  ]},
 {'n':6,'title':'1 s means with five baseline percentile bands','summary':'Median-centre each trial, then show its position relative to its own baseline distribution.',
  'bin':'1 s; 40 bins/trial','baseline':'[-20,0) s; P25/P45/P50/P55/P75 of finite bin means',
  'zero':'Baseline median P50','colours':'Five managua_r colours; narrow dark P45-P55 centre',
  'steps':[
   'Start from the same directly frame-recomputed 1 s bin means as Version 5.',
   'For each trial separately, calculate P25/P45/P50/P55/P75 from its finite unscaled baseline-bin means in [-20,0) s, one vote per finite bin. Quantiles use linear interpolation.',
   'Subtract that trial P50 from each bin mean. The median-centred baseline has P50 = 0.',
   'Assign five bands using the uncentred boundaries P25/P45/P55/P75, or equivalently the centred boundaries P25-P50, P45-P50, P55-P50 and P75-P50.',
   'The bands are: x < P25; P25 <= x < P45; P45 <= x < P55; P55 <= x < P75; x >= P75. Exact boundary ties enter the upper band. Missing values remain black.',
   'Use managua_r positions 0/0.25/0.5/0.75/1: #81e7ff, #5775b3, #582948, #b26343, #ffcf67. The dark purple middle band contains zero and is distinct from missing black.',
   'Retain original mean-centred values and new median-centred log values. Export ordinal display scores [-1,-0.5,0,+0.5,+1]; these are category labels, not a continuous amplitude normalization.'
  ],
  'difference':'Compared with Version 5, retains the same 1 s means but changes mean centring to median centring and continuous physical colours to trial-specific percentile bands.',
  'consider':'Simplifies the display and identifies low/near-baseline/high activity relative to each trial. Equal colours across trials need not indicate equal physical log differences, and values within a band are merged visually. P45-P55 is a percentile interval, not necessarily exactly 10% of observed bins when there are few baseline bins or ties. The four-colour Version 6 is discarded; prior 0.25 s exports are historical.',
  'options':[{'id':'v6-five','title':'Version 6 five bands','kind':'Version6_OneSecondCentralBand','validation':CANDIDATES/'Version6_OneSecondCentralBand_validation.json',
    'formula':'z = 1 s bin mean - baseline P50; display band = interval between trial baseline percentile boundaries',
    'difference':'All 270 trial references are defined; all have zero inside the dark central P45-P55 band.'}]}
]

# Active choices satisfy the author's median-at-colour-midpoint criterion.
versions=[v for v in versions if v['n'] in [1,2,6]]
pros_cons={
 'v1-c':('Bout medians reduce the influence of extreme within-bout samples; continuous colours and precise eligible timing.', 'Suppresses within-bout variation; stronger clipping; baseline is duration weighted; complete bout medians can include post-CS values; one undefined trial.'),
 'v1-d':('Same bout summaries with a larger denominator: less or equal clipping than C and more retained graded colour variation.', 'Same loss of within-bout detail, duration-weighted baseline and crossing-CS support as C; lower contrast; one undefined trial.'),
 'v2':('Direct eligible log means; continuous detail; 0.5 s temporal resolution; median-centred reference matches displayed bin means.', 'Trial-specific scaling and numeric clipping hide absolute magnitudes; one undefined trial; sparse bins vote equally in the baseline.'),
 'v6-five':('Meets P50=0 in all 270 trials; a distinct dark centre; direct 1 s means; five colours make baseline-relative changes easy to read.', 'Coarser temporal and amplitude detail; colours describe each trial baseline rather than common physical magnitudes; at most 20 baseline bins estimate the percentiles.')
}
recommendation='Recommend Version 6 for the main heatmap under your stated goal: all 270 medians are defined and centred, zero is inside the dark P45-P55 band, and the five colours provide a simple common baseline-relative interpretation. Version 2 is the strongest alternative if finer timing and continuous colour gradations matter more; it retains 0.5 s bins but has one undefined trial.'
records=[];preserved={}; option_titles={};figure_cards={}
for version in versions:
    for option in version['options']:
        valpath=option['validation'];val=load(valpath)
        rows=val['outputs']+val['source_numeric_files']
        for row in rows:
            path=Path(row['path']);assert digest(path)==row['sha256'];preserved[str(path)]=row['sha256']
        manifest=Path(val['source_data_manifest']);assert digest(manifest)==val['source_data_manifest_sha256']
        preserved[str(manifest)]=val['source_data_manifest_sha256']
        preserved[str(valpath)]=digest(valpath)
        files={Path(row['path']).suffix:Path(row['path']) for row in val['outputs']}
        id=option['id'];option_titles[id]=option['title']
        controls=f'<label class="choice-label" for="choice-{id}">Your choice</label><select id="choice-{id}" data-choice="{id}"><option value="Undecided">Undecided</option><option value="Keep">Keep</option><option value="Discard">Discard</option></select>'
        figure_cards[id]=f'''<article class="option" id="{id}"><div class="option-head"><h3>{esc(option['title'])}</h3><div class="choice">{controls}</div></div><p>{esc(option['difference'])}</p><div class="formula">{esc(option['formula'])}</div><div class="downloads"><a download="{files['.pdf'].name}" href="{uri(files['.pdf'],'application/pdf')}">PDF</a><a download="{files['.svg'].name}" href="{uri(files['.svg'],'image/svg+xml')}">SVG</a></div><img src="{uri(files['.png'],'image/png')}" width="2700" height="1530" alt="{esc(option['title'])}: F Delay, G 3 s Trace, H Control"></article>'''
        records.append({'version_number':version['n'],'option':id,'title':option['title'],'validation_path':str(valpath),'validation_sha256':digest(valpath),'validation':val})

def text_guide():
    lines=['# F/G/H baseline-centred options - selection guide','',
        'Four available options meet baseline-median centring: V1 C, V1 D, V2 C and V6 five bands. Versions 4/5 are removed because mean centring does not ensure median-zero. Four-colour V6 remains discarded; Version 3 has no established recipe.','',
        '## Shared processing','']+[f'- {s}' for s in common]
    lines+=['','## Overview','', '| Option | How it is made | Baseline / scale | Pros | Cons |','|---|---|---|---|---|']
    lines += [f"| {o['title']} | {v['title']}; {v['bin']} | {v['baseline']}; {o['formula']} | {pros_cons[o['id']][0]} | {pros_cons[o['id']][1]} |" for v in versions for o in v['options']]
    for v in versions:
        lines+=['',f"## Version {v['n']}: {v['title']}",'',v['summary'],'','How it is made:',''] if v['steps'] else ['',f"## Version {v['n']}: {v['title']}",'',v['summary']]
        lines += [f'{i}. {s}' for i,s in enumerate(v['steps'],1)]
        lines += ['',f"Difference: {v['difference']}",'',f"When choosing: {v['consider']}"]
        for o in v['options']:
            lines+=['',f"### {o['title']}",'',o['difference'],'',f"`{o['formula']}`"]
    lines+=['','## Choosing between recipes','',
      '- Version 1: bout summaries on eligible sample timing; trial-relative percentile scaling.',
      '- Version 2: direct moving vigor in half-second means; trial-relative percentile scaling.',
      '- Versions 4 and 5: removed from active choices because baseline mean-zero does not ensure median-zero.',
      '- Version 6: one-second means shown as five bands relative to each trial baseline; absolute amplitude detail is reduced.',
      '- C versus D in Version 1 changes the numeric denominator; both preserve baseline median-zero when the scale is defined.', '', recommendation, '', 'Centring puts the baseline median at the middle colour; it does not make all baseline cells dark. V1/V2 have 269 defined references; V6 has 270.',
      '', 'All existing data, figures, scoped selections and frozen panels are preserved. No new recipe is assigned to the Version 3 gap.']
    return '\n'.join(lines)+'\n'

guide=HERE/'VERSION_GUIDE.md';guide.write_text(text_guide(),encoding='utf-8')
provenance={'date_local':'2026-10-09','purpose':'organized selection review of existing versions; no recomputation, deletion, freeze or scoped figure selection change',
 'versions':[{k:v for k,v in version.items() if k!='options'} for version in versions], 'options':records,
 'candidate_selection':load(CANDIDATES/'candidate_selection.json'),'baseline_filter_selection':load(HERE/'baseline_filter_selection.json'),'centring_audit':load(HERE/'baseline_centring_audit.json'),'recommendation':recommendation,
 'preserved_existing_files':[{'path':p,'sha256':h} for p,h in sorted(preserved.items())],
 'guide_sha256':digest(guide),'builder_sha256':digest(Path(__file__))}
prov=HERE/'provenance.json';prov.write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
table_text={
 'v1-c':('Complete-bout medians on eligible samples; no bins.', 'Continuous median-centred C scale.', 'Fine eligible timing; reduces influence of within-bout extremes.', 'Within-bout detail is lost; stronger clipping; whole bouts can cross CS; 269/270 defined.'),
 'v1-d':('Same complete-bout medians as V1 C; no bins.', 'Continuous median-centred D scale; larger denominator.', 'Fine timing; less or equal saturation than C.', 'Same bout-summary limitations; lower contrast; 269/270 defined.'),
 'v2':('Direct log means in 0.5 s bins.', 'Continuous median-centred C scale from baseline-bin values.', 'Finer timing and graded colours; no bout-median substitution.', 'Trial-specific scaling and clipping; 269/270 defined.'),
 'v6-five':('Direct log means in 1 s bins.', 'Five trial-baseline percentile bands; dark P45-P55 centre.', 'All 270 centred; simple and distinct baseline-relative bands.', 'Coarser timing; banding hides differences within bands; percentile estimates use at most 20 baseline bins.')
}
overview=''.join(f'<tr><td><a href="#{o["id"]}">{esc(o["title"])}</a></td>'+''.join('<td>'+esc(t)+'</td>' for t in table_text[o['id']])+'</tr>' for v in versions for o in v['options'])
sections=[]
for v in versions:
    steps='<h3>How it is made</h3><ol>'+''.join('<li>'+esc(s)+'</li>' for s in v['steps'])+'</ol>' if v['steps'] else ''
    sections.append(f'<section class="version" id="version-{v["n"]}"><div class="version-label">Version {v["n"]}</div><h2>{esc(v["title"])}</h2><p class="summary">{esc(v["summary"])}</p>{steps}<p class="difference"><strong>Difference:</strong> {esc(v["difference"])}</p><p class="consider"><strong>When choosing:</strong> {esc(v["consider"])}</p>'+''.join(figure_cards[o['id']] for o in v['options'])+'</section>')
css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f5f7;color:#25313c;font:16px/1.65 system-ui,Segoe UI,sans-serif}main{max-width:1240px;margin:auto;padding:38px 28px 60px}h1,h2,h3{line-height:1.25}h1{font-size:34px;margin:10px 0 15px}h2{font-size:26px;margin:8px 0 14px}h3{font-size:20px}p{margin:10px 0 15px}a{color:#28567b;text-underline-offset:3px}.eyebrow,.version-label{font-size:12px;text-transform:uppercase;letter-spacing:1.4px;font-weight:700;color:#607080}.lead{max-width:950px;color:#526576}.fish-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:22px 0}.fish{padding:12px 16px;border:1px solid #dbe2e8;border-top:3px solid var(--colour);background:white;border-radius:7px}.fish strong{color:var(--colour)}.fish code{display:block;color:#25313c}nav{display:flex;flex-wrap:wrap;gap:9px;margin:22px 0}nav a,.downloads a,button{border:1px solid #cad5df;border-radius:6px;padding:7px 12px;background:white;text-decoration:none;font:600 13px/1.5 system-ui;cursor:pointer}section.panel,.version{background:white;border:1px solid #dbe2e8;border-radius:10px;padding:26px;margin:22px 0}.summary{color:#526576;font-size:17px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;min-width:920px;font-size:14px}td,th{text-align:left;vertical-align:top;padding:11px 13px;border-bottom:1px solid #dbe2e8}th{background:#f1f5f8}.difference{border-left:3px solid #6d91ad;padding:12px 16px;background:#f1f6fa}.consider{color:#526576}.option{border-top:1px solid #dbe2e8;margin-top:26px;padding-top:16px}.option-head{display:flex;justify-content:space-between;align-items:center;gap:14px}.choice{display:flex;align-items:center;gap:8px}.choice-label{font-size:13px;color:#526576}select{font:14px system-ui;padding:7px;border:1px solid #cad5df;border-radius:5px;background:white}.formula{padding:12px 15px;background:#f3f6f8;border-left:3px solid #8095a5;font:14px/1.65 Consolas,monospace;overflow-wrap:anywhere}.downloads{display:flex;gap:9px;margin:15px 0}img{display:block;width:100%;height:auto}li{margin:8px 0}.notice{background:#faf7ee;border-left:3px solid #ae8a46;padding:12px 16px;font-size:14px}.choice-summary{font-size:14px}.status{color:#236048}.footer{color:#607080;font-size:13px}.tools{display:flex;flex-wrap:wrap;gap:10px;align-items:center}.tools a{font-size:14px}@media(max-width:720px){main{padding:22px 14px}h1{font-size:28px}.fish-grid{grid-template-columns:1fr}section.panel,.version{padding:18px}.option-head{align-items:flex-start;flex-direction:column}.choice{flex-wrap:wrap}}@media print{main{padding:0}body{background:white}nav,.choice,.downloads,.tools{display:none}.option{break-inside:avoid}.version{break-before:page;border:0}.fish-grid{grid-template-columns:repeat(3,1fr)}}'''
js='''const optionTitles=__TITLES__;const key='fgh-versions-1-to-6-20261009';let choices={};try{choices=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}const inputs=[...document.querySelectorAll('[data-choice]')];function update(){const groups=['Keep','Discard','Undecided'].map(state=>state+': '+inputs.filter(el=>el.value===state).map(el=>optionTitles[el.dataset.choice]).join(', '));document.getElementById('choice-summary').textContent=groups.join(' | ')}inputs.forEach(el=>{el.value=choices[el.dataset.choice]||'Undecided';el.addEventListener('change',()=>{choices[el.dataset.choice]=el.value;try{localStorage.setItem(key,JSON.stringify(choices))}catch(e){}update()})});update();document.getElementById('export-choices').addEventListener('click',()=>{const text=['F/G/H version choices - 9 October 2026','',...inputs.map(el=>optionTitles[el.dataset.choice]+': '+el.value),'','Four-colour Version 6: already discarded','Version 3: no established recipe','Versions 4 and 5: removed because mean centring does not guarantee median-zero'].join('\\n');const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([text],{type:'text/plain;charset=utf-8'}));link.download='FGH_version_choices.txt';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)})'''.replace('__TITLES__',json.dumps(option_titles))
page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F/G/H baseline-centred options</title><style>'+css+'</style></head><body><main>'
page+='''<header><div class="eyebrow">F / G / H - 9 October 2026</div><h1>Baseline-centred options: choose a recipe</h1><p class="lead">Four available options meet your baseline-median centring requirement. Each section explains how its data are made, what zero and colours mean, and what changes from the other recipes. Mark each option Keep, Discard or Undecided, then export your choices.</p><div class="fish-grid"><div class="fish" style="--colour:#d90072"><strong>F - Delay</strong><code>20221115_07</code></div><div class="fish" style="--colour:#bb5b00"><strong>G - 3 s Trace</strong><code>20230310_08</code></div><div class="fish" style="--colour:#087ea2"><strong>H - Control</strong><code>20221115_09</code></div></div><nav>'''+''.join(f'<a href="#version-{n}">Version {n}</a>' for n in [1,2,6])+'<a href="#overview">Overview</a><a href="#shared">Shared processing</a><a href="#your-choices">Your choices</a></nav><p class="notice">Versions 4 and 5 are removed from the active review: their mean-zero baseline does not guarantee median-zero. V1 C/D and V2 C centre all 269 defined trials; Control trial 16 is undefined. V6 centres all 270 trials. Version 3 has no established recipe; four-colour V6 remains discarded.</p></header>'
page+='<section class="panel" id="overview"><h2>Compare the recipes</h2><div class="table-wrap"><table><thead><tr><th>Option</th><th>What is displayed</th><th>Colour scale</th><th>Pros</th><th>Cons</th></tr></thead><tbody>'+overview+'</tbody></table></div><p>All available options centre the baseline median, P50, at the middle colour when defined. Centring does not make every baseline cell dark. All remaining colour scales are relative to each trial baseline; equal colours across trials need not mean equal physical log differences.</p></section>'
page+='<section class="panel" id="recommendation"><h2>Recommendation</h2><p>'+esc(recommendation)+'</p><p>The main limitation of V6 is that banding hides differences within each band. Use its underlying log-value exports to assess effect magnitude. This recommendation prioritizes your median-centred, clearly distinguished heatmap requirement.</p></section>'
page+='<section class="panel" id="shared"><h2>Shared processing</h2><ol>'+''.join('<li>'+esc(s)+'</li>' for s in common)+'</ol><p>All baseline intervals exclude CS onset. Sample/bin aggregation and baseline voting are separate steps. Versions 1/2 numerically clip the scaled values to [-1,+1]. Version 6 preserves the underlying log values but merges them into five colour bands.</p></section>'
page+='<section class="panel" id="additional-version"><h2>Additional Version 7: uploaded single-fish recipe</h2><p><a href="../fgh_version7_unpooled_recipe_20261009/index.html">Open Version 7 and its comparison with Version 6</a>. Within-window bout medians and 0.5 s means, centred over [-15,0), then separate P10/P90 scaling with a median-preserving centre. Baseline anchors are -0.7/+0.7 on continuous managua_r limits -1/+1. All 257 defined scales have median zero; 13 sparse Control trials are undefined. Physical log values remain exported.</p><p><a href="../fgh_version6_softer_high_20261009/index.html">Version 6: original versus softer High colour</a></p></section>'
page+='<section class="panel" id="version-8"><h2>Version 8: P10/P90 colour endpoints</h2><p><a href="../fgh_version8_percentile_endpoints_20261009/index.html">Open Version 8 and its comparison with Version 7</a>. Same Version 7 recipe, baseline references and median-preserving centre, with P10/P90 mapped to -1/+1. Colours saturate at and beyond these anchors. All 257 defined trials remain centred; the same 13 sparse Control trials are undefined.</p></section>'
page+=''.join(sections)
page+='<section class="panel" id="your-choices"><h2>Your choices</h2><p class="choice-summary" id="choice-summary"></p><div class="tools"><button id="export-choices">Export your choices</button><a download="VERSION_GUIDE.md" href="'+uri(guide,'text/markdown')+'">Download text guide</a><a download="provenance.json" href="'+uri(prov,'application/json')+'">Download provenance</a></div></section><p class="footer">All figures, PDFs and SVGs are embedded for offline review. Existing source files, figure exports, freezes and scoped selections are preserved. Earlier experiments remain in their historical folders.</p></main><script>'+js+'</script></body></html>'
target=HERE/'index.html';target.write_text(page,encoding='utf-8')
assert page.count('<img ')==4 and page.count('data:application/pdf;base64,')==4
assert all(digest(Path(p))==h for p,h in preserved.items())
print(json.dumps({'html':str(target),'bytes':target.stat().st_size,'figures':4,'numbered_sections':3,'existing_hashes_verified':len(preserved),'text_guide':str(guide)}))
