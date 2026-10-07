"""Assemble review notes and source-labelled gallery without touching prior assets."""
from pathlib import Path
import json, subprocess, html
import pandas as pd
import numpy as np
from hashlib import sha256
from PIL import Image, ImageOps, ImageDraw
REPO=Path(__file__).resolve().parents[2]
ROOT=Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly')
OUT=ROOT/'heatmap-reference-review-20261007'
summaries=json.loads((OUT/'summary.json').read_text())
stats=pd.read_csv(OUT/'baseline_trial_audit.csv');bins=pd.read_parquet(OUT/'candidate_bins.parquet')
inventory=pd.read_csv(OUT/'saved_alternative_inventory.csv')
lines=['# Figure 1 single-fish heatmap review — provisional, no selection frozen','',
'Current lettering: E = raw/signed vigor reference, F = Delay, G = 3 s Trace, H = Control. Historical E/G heatmap names map to current F/H by condition; historical C/D trace names map to current D/E by content. Panel E is included only to explain signal-to-bin correspondence; its design remains under separate review.','',
'The v5 acquisition-clock builder was reused verbatim with a read-only trial callback. All 21,600 saved F/G/H bins reproduced within 1e-12, including identical NaNs. This is a narrow figure review, not confirmation of the full legacy preprocessing pipeline. No population reruns or freezes were performed.','',
'## Saved alternative families and their semantics','',
'| Family | Metric / clock | Reference and operation | Support / binning | Palette and clipping |','|---|---|---|---|---|',
'| Historical per-trial linear conditional vigor | Multiple metrics, including tail-length-weighted angular L1; corrected arrival-clock profiles | P10–P90 of covered conditional-intensity bins before −15 s; divide by trial range | Detector-conditional bin means, coverage threshold; historical bounds, not proposed here | Original palette or managua_r; stored 0–1 clipping |',
'| Historical frame-first log P10–P90 | Multiple metrics; corrected arrival clock | ln(mean raw vigor per bout), reference [−20,0); divide by P90−P10 per trial | Moving positive valid frames; scale and clip frames before 0.5 s averaging; empty bins NaN | managua_r, stored 0–1 clipping |',
'| Historical all-frame candidate | Multiple metrics; corrected arrival clock | Mean raw vigor per bin; P10–P90 from covered bins before −15 s | All valid frames, not bout-conditional; 90% coverage flag; continuous display requires finite bins | managua_r, stored 0–1 clipping |',
'| Historical baseline-window signed-log review | tail_length_weighted_angular_l1; corrected arrival clock | Median moving-frame ln(vigor) subtraction; saved [−20,0) and [−15,0) versions | Bout median logs repeated on eligible frames; finite-only 0.5 s means | managua_r, ±0.25 display clipping only |',
'| Assembly legacy-vigor v1/v2 | legacy_distal_angular_speed; corrected arrival clock | Per-trial moving-frame median log subtraction: v1 [−20,0), v2 [−15,0) | Detector-conditional bout medians; saved v1/v2 differ mathematically through reference bounds | managua_r, ±0.25 display clipping; NaN black |',
'| Current cadence v5 | legacy_distal_angular_speed, rad/ms; reconstructed presumed acquisition cadence | SAME [−15,0) per-trial reference | Median bout log repeated only on finite positive valid moving frames, bout_id>0; finite-frame-weighted 0.5 s means | managua_r, ±0.25 display clipping; NaN black |','',
'Saved historical images are context, not controlled comparisons. Their clocks, metrics, references, order of operations and support can differ simultaneously. The complete file inventory includes paths and SHA-256 hashes; contact sheets preserve source filenames. A family with no located saved image is documented from its builder, not represented as a saved artifact.','',
'## Controlled candidates','',
'All candidates use the same reconstructed clock, legacy distal angular speed, detected-bout eligibility, trial windows [−20,20), and 80 half-second bins. Each bout median is calculated over its eligible portion within the trial window, then repeated on those same frames. Bin means weight bouts by their contributing eligible-frame counts. No unsupported value is filled with zero.','',
'1. **Trial centred:** subtract the trial’s median eligible frame log in [−15,0). This is translation in log units, equivalent to an amplitude ratio reference; there is no P10–P90 division.','2. **Fixed fish centred:** subtract one median pooled from those SAME [−15,0) eligible frames across trials 5–94, separately per fish. Longer bouts and trials with more moving frames contribute more reference weight. This uses the whole displayed session retrospectively, not a pre-training-only reference.','3. **Uncentred:** retain median ln(vigor / (1 rad/ms)); the numerical unit reference is explicit. This and fixed fish centring differ by a constant per fish. Uncentred values permit direct numerical comparison across these fish; separate fish references do not preserve between-fish amplitude levels.','',
'Columns 1–2 of candidate_gallery show IDENTICAL trial-centred stored values at ±0.25 and ±0.75. Columns 3 and 5 show IDENTICAL fixed-centred values and limits with managua_r versus RdBu_r. Column 4 uses viridis and a common 1st–99th percentile display range across all three fish. These quantiles set colour limits only; no data values are rescaled or clipped in storage. Black consistently means no eligible frame contribution, not zero vigor or proof of inactivity.','',
'## Baseline and saturation audit','',
'| Panel / fish | Baseline log range | Max/min geometric baseline ratio | Baseline frames min/median/max | Baseline bouts min/median/max | Current ±0.25 saturated | Trial ±0.75 saturated | Fixed ±0.75 saturated |','|---|---|---|---|---|---|---|---|']
for s in summaries:
    lines.append(f"| {s['panel']} / {s['fish']} | {s['baseline_log_range'][0]:.3f} to {s['baseline_log_range'][1]:.3f} | {s['baseline_median_amplitude_ratio_max_min']:.2f}× | {s['baseline_frames_min_median_max']} | {s['baseline_bouts_min_median_max']} | {s['trial_centred_saturation_025']:.1%} | {s['trial_centred_saturation_075']:.1%} | {s['fish_centred_saturation_075']:.1%} |")
lines+=['','Saturation percentages count finite bin values outside the displayed limits. They do not count black bins and do not imply stored-value clipping. Thousands of frames are temporally correlated; baseline bout counts and leave-one-bout sensitivity are more informative than treating frame count as an independent sample size.','']
for s in summaries:
    lines.append(f"{s['panel']}: fixed reference {s['fixed_log_reference']:.6f} ln units = {s['fixed_reference_rad_per_ms']:.6f} rad/ms; missing baseline trials {s['missing_baseline_trials']}; fewer than three baseline bouts {s['less_than_3_baseline_bouts_trials']}; largest leave-one-baseline-bout median shift {s['largest_leave_one_bout_shift']:.4f} ln units.")
lines+=['','With no eligible trial baseline, the current function returns ALL 80 bins NaN even if movement occurs elsewhere. Fixed/uncentred retain supported bins under that condition. With a sparse or unusual baseline, per-trial subtraction shifts every supported bin by the same offset; the leave-one-bout statistic exposes dependence on individual bouts. We did not change missing-baseline policy for the current candidate.','',
'## What centring removes or creates','',
'For every supported bin, `trial_centred − fish_centred = fish_reference − trial_reference`. Thus centring does not change within-trial contrasts or timing, but it changes comparisons between trials. If the whole-trial movement amplitude rises with learning, trial centring removes the shared rise. If CS amplitude stays constant while baseline amplitude changes, trial-centred CS values change despite constant absolute CS amplitude. These are exact algebraic consequences; these example fish alone cannot establish whether the changes are caused by learning.','',
'The next table compares mean supported-bin log amplitude before the paired US in early training (15–24) versus late training (55–64). Delay uses [0,9), Trace uses [0,13), Control uses [0,10) as a CS interval with no paired US. Equal trial weight, then equal finite-bin weight within each trial; it is descriptive, not an effect-size estimate or population inference.','',
'| Panel | Late−early uncentred / fixed | Late−early trial centred | Late−early trial baseline |','|---|---|---|---|']
for s in summaries:
    stop={'F':9,'G':13,'H':10}[s['panel']]
    part=bins.loc[bins.panel.eq(s['panel'])&bins.time_s.ge(0)&bins.time_s.lt(stop)]
    means=part.groupby('trial')[['uncentred_log','trial_centred','fish_centred']].mean()
    delta=means.loc[55:64].mean()-means.loc[15:24].mean()
    r=stats.loc[stats.panel.eq(s['panel'])].set_index('trial')
    bd=r.loc[55:64,'baseline_log_median'].mean()-r.loc[15:24,'baseline_log_median'].mean()
    lines.append(f"| {s['panel']} | {delta.uncentred_log:+.4f} | {delta.trial_centred:+.4f} | {bd:+.4f} |")
lines+=['','## Concrete bin examples','',
'| Panel / trial / interval | Eligible frames | Uncentred | Trial centred | Fixed fish centred | Trial−fish offset |','|---|---|---|---|---|---|']
example=[]
for s in summaries:
    for trial in [9,17,63,66,93,s['largest_offset_trial']]:
        p=bins.loc[bins.panel.eq(s['panel'])&bins.trial.eq(trial)&bins.contributing_frames.gt(0)]
        q=p.loc[(p.time_s-.25).abs().idxmin()]
        offset=q.trial_centred-q.fish_centred
        lines.append(f"| {s['panel']} / {trial} / [{q.time_s-.25:g},{q.time_s+.25:g}) | {q.contributing_frames} | {q.uncentred_log:+.5f} | {q.trial_centred:+.5f} | {q.fish_centred:+.5f} | {offset:+.5f} |")
        example.append(q)
pd.DataFrame(example).to_csv(OUT/'numerical_bin_examples.csv',index=False)
lines+=['','The historical P10–P90 alternative adds a trial-dependent slope as well as an offset. The diagnostic column in candidate_bins deliberately applies that division/clipping to the CURRENT bout-median signal on the SAME baseline/support; it isolates the mathematical effect and does not claim to reproduce historical log-bout-mean data. Baseline P90−P10 widths and clipped-frame fractions are recorded per trial. Narrow ranges amplify noise; clipping before binning permanently loses amplitude distinctions.','',
'## E–H visual review and provisional choice','',
'Current E demonstrates unbinned raw vigor and the signed bout signal on matched support. Its orange signed traces are small at assembly scale, while F–H colour bins are heavily influenced by their narrow ±0.25 range. A heatmap cell summarizes supported movement within 0.5 s; it does not assert movement throughout that interval. E should explain this relationship in its caption; no E redesign was performed here.','',
'Current F/G show broad patches of endpoint colours. Wider limits recover distinctions without changing the data. H has much more black support: its darker appearance partly reflects missing conditional movement samples, not simply weaker signed amplitude. All candidates retain actual chronological trial rows and phase boundaries; historical layouts include phase gutters, whereas the controlled gallery uses boundaries in a continuous grid. This layout difference is visual only.','',
'For a figure intended to show amplitude changes across trials, the fixed fish reference is the strongest provisional candidate: it retains between-trial changes while giving a readable zero reference. Uncentred log vigor is the reference check and is preferable if between-fish absolute amplitude comparisons are central. Trial centring remains appropriate for a explicitly baseline-relative within-trial response question. The measured offsets and saturation should inform selection; the claim that all per-trial normalization is inherently bad is too broad.','',
'Suggested selection for review: **fixed fish-centred bout median log vigor, ±0.75 shared display, explicit baseline/reference caption**. Compare managua_r and RdBu_r in the gallery; a diverging palette makes the zero reference clearer, while managua_r preserves the existing visual language. The wider range is a review choice, not an approved final range. No candidate has been frozen. The full preprocessing audit must be confirmed before broader reruns.','']
lines+=['## Range-division diagnostic measurements','',
'| Panel | P90−P10 ln range min/median/max | Maximum fraction of eligible bout-median frames clipped to 0/1 |','|---|---|---|']
for panel,r in stats.groupby('panel'):
    lines.append(f"| {panel} | {r.p90_p10_width.min():.3f} / {r.p90_p10_width.median():.3f} / {r.p90_p10_width.max():.3f} | {r.p10_p90_frames_clipped_fraction.max():.1%} |")
lines+=['','The diagnostic is generally unsaturated for bout medians: broad FRAME-log quantile ranges contain most BOUT medians. This does not validate historical mean-bout scaling or justify arbitrary range division. In G, one trial loses distinctions for 26.8% of eligible frames. H’s widths vary by a factor of 1.63, changing the relative gain between trials.','',
'Control trial 16 has one baseline bout, so leave-one-bout sensitivity cannot be estimated there: removing that bout leaves no baseline. The reported maximum sensitivities concern trials with remaining baseline support.','']
# Render source SVGs only into the new review directory, including original panels.
panels=[]
exe=Path('C:/Program Files/Inkscape/bin/inkscape.com')
for version,folder in [('v1',ROOT/'heatmaps'),('v2',ROOT/'heatmaps'),('v5',ROOT/'cadence-review-v5-20261007')]:
    for panel,name in [('F','Delay'),('G','3sTrace'),('H','Control')]:
        suffix=f'legacy-vigor_{version}' if version!='v5' else 'presumed-cadence_v5'
        svg=folder/f'Fig1_Panel{panel}_{name}_{suffix}.svg'
        png=OUT/f'saved_{panel}_{version}.png'
        side=json.loads(svg.with_suffix('.svg.json').read_text())
        assert sha256(svg.read_bytes()).hexdigest()==side['svg_sha256']
        if not png.exists():
            subprocess.run([str(exe),str(svg),'--export-type=png',f'--export-filename={png}','--export-width=530'],check=True,capture_output=True)
        panels.append((version,panel,png))
canvas=Image.new('RGB',(1650,1320),'white');d=ImageDraw.Draw(canvas)
for k,(version,panel,png) in enumerate(panels):
    x=(k%3)*550;y=(k//3)*440
    with Image.open(png) as im:canvas.paste(im.convert('RGB'),(x,y+30))
    d.text((x+15,y+8),f'{panel} / saved {version} / '+('arrival clock' if version!='v5' else 'presumed acquisition clock'),fill='black')
canvas.save(OUT/'saved_assembly_heatmaps.png')
with Image.open(ROOT/'cadence-review-v5-20261007/figure1-presumed-acquisition-review-v5.png') as im:
    im.crop((900,925,1800,1950)).save(OUT/'current_E_reference_GH_crop.png')
# Saved value differences distinguish actual math changes from file/style changes.
comparison=[]
for panel,name in [('F','Delay'),('G','3sTrace'),('H','Control')]:
    paths=[ROOT/f'heatmaps/Fig1_Panel{panel}_{name}_legacy-vigor_{v}.parquet' for v in ['v1','v2']]
    for p in paths:
        side=json.loads(p.with_suffix('.svg.json').read_text())
        assert sha256(p.read_bytes()).hexdigest()==side['panel_data_sha256']
    a,b=[pd.read_parquet(p) for p in paths]
    vals_a=a['Signed log vigor'].to_numpy();vals_b=b['Signed log vigor'].to_numpy()
    both=np.isfinite(vals_a)&np.isfinite(vals_b)
    comparison.append({'panel':panel,'v1_v2_equal_nan_values':bool(np.allclose(vals_a,vals_b,equal_nan=True)),
        'v1_v2_changed_support_bins':int((np.isfinite(vals_a)!=np.isfinite(vals_b)).sum()),
        'v1_v2_max_finite_difference':float(np.max(np.abs(vals_a[both]-vals_b[both]))) if both.any() else None})
(OUT/'saved_v1_v2_comparison.json').write_text(json.dumps(comparison,indent=2))
lines+=['## Saved v1 versus v2: measured mathematical differences','',
'Sidecars identify v1 [−20,0) and v2 [−15,0) trial baselines on the same arrival-clock metric/movement inputs. Their support is unchanged; these are actual value differences, not palette-only alternatives. Those historical bounds were inventoried, not recreated or proposed for the new candidates.','']
for c in comparison:
    lines.append(f"{c['panel']}: changed-support bins {c['v1_v2_changed_support_bins']}; maximum finite-bin difference {c['v1_v2_max_finite_difference']:.6f} ln units.")
(REPO/'reviews/figure1_heatmap_20261007/REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')
gallery=['candidate_gallery.png','baseline_diagnostics.png','saved_assembly_heatmaps.png']+[p.name for p in sorted(OUT.glob('saved_gallery_*.png'))]
body='<h1>Figure 1 heatmap review · provisional</h1><p>No candidate frozen. Current E is reference only; F Delay, G 3sTrace, H Control.</p>'
for name in gallery:body+=f'<h2>{html.escape(name)}</h2><img style="width:100%;max-width:1800px" src="{name}">'
body+='<h2>Review notes</h2><pre style="white-space:pre-wrap">'+html.escape('\n'.join(lines))+'</pre>'
(OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Heatmap reference review</title><body style="font:16px system-ui;margin:24px">'+body,encoding='utf-8')
manifest=json.loads((OUT/'review_manifest.json').read_text())
manifest['outputs']=[{'path':str(p),'sha256':sha256(p.read_bytes()).hexdigest()} for p in OUT.iterdir() if p.is_file() and p.name!='review_manifest.json']
manifest['review_notes']={'path':str(REPO/'reviews/figure1_heatmap_20261007/REVIEW.md'),'sha256':sha256((REPO/'reviews/figure1_heatmap_20261007/REVIEW.md').read_bytes()).hexdigest()}
(OUT/'review_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(comparison,indent=2));print('Review notes and galleries ready:',OUT)
