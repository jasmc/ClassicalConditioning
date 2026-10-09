from pathlib import Path
import base64,json
HERE=Path(__file__).resolve().parent
def uri(name,mime): return 'data:'+mime+';base64,'+base64.b64encode((HERE/name).read_bytes()).decode()
stem='FGH_Version7_UnpooledRecipe_legacy_layout'
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Version 7 — single-fish bout recipe</title>
<style>body{font:17px/1.6 system-ui;color:#25313c;max-width:1250px;margin:30px auto;padding:0 20px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #ccc}a{color:#28567b}.notice{background:#faf7ee;padding:15px} .scroll{overflow:auto}table{min-width:600px}</style>
<h1>Version 7: continuous baseline-relative percentile scaling</h1>
<p>Each fish is shown separately: F Delay 20221115_07, G 3 s Trace 20230310_08, H Control 20221115_09. Trials 5–94, aligned to CS, cover [−20,+20) seconds.</p>
<h2>How it is made</h2><ol>
<li>Use the original eligible natural-log vigor frames before bout summarization. Invalid frames and non-bout frames remain missing.</li>
<li>For each fish and trial, subtract the median of eligible original log frames during [−15,0) seconds. A missing baseline would make the whole trial missing.</li>
<li>Within the extracted [−20,+20) window, calculate each bout’s median baseline-centred log value. Repeat it only on that bout’s eligible frames. A crossing-window bout uses only its eligible frames within the window.</li>
<li>Average finite repeated bout values into 0.5 s bins. Bouts contribute according to their eligible frame count within each bin. Empty bins remain missing.</li>
<li>For each fish and trial, calculate the median of the finite 0.5 s bin values during [−15,0). Subtract this median from every bin in that trial. Each finite baseline bin has one vote, regardless of its eligible frame count.</li>
<li>Calculate baseline P10/P90 separately for each trial, using only finite [−15,0) bins and linear quantile interpolation. Require at least 10 finite baseline bins. Sparse or degenerate scales make the whole scaled trial undefined; physical values remain available.</li>
<li>Map P10 to −0.7, zero to 0, and P90 to +0.7. Use separate lower and upper slopes, connected by a short symmetric linear region containing the middle baseline observations. This preserves the calculated median even with an even number of bins. Apply the fitted transformation to every bin in the trial.</li>
<li>Use continuous managua_r with scaled limits −1 to +1, zero at its dark midpoint, and black for missing or undefined cells. Exported scaled values are not clipped; colours saturate outside ±1. No fish pooling is applied.</li></ol>
<p class="notice"><b>Verified:</b> 257 scaled trials have baseline median zero and P10/P90 at −0.7/+0.7 to numerical precision. Thirteen Control trials have fewer than 10 finite baseline bins and are undefined in this scaled display. They are trials 15, 16, 19, 23, 26, 64, 68, 69, 70, 72, 73, 77 and 88. No additional trials failed the degeneracy checks.</p>
<p>Colours now express position relative to each trial’s own baseline spread. Equal colours across fish can represent different physical log differences. Original physical values are retained in <code>delta_log_vigor</code>; display values are in <code>scaled_baseline_vigor</code>. Earlier physical and original frame-centred revisions are preserved under history.</p>
<p><b>Input provenance:</b> this candidate uses the same reconstructed angular-speed values and detected-bout eligibility masks as the current F/G/H candidates. It follows the uploaded aggregation recipe; it does not re-import the separate processed legacy metric and movement masks used by the pooled figure.</p>
<h2>Comparison with Version 6</h2><div class="scroll"><table><tr><th>Feature</th><th>Version 7</th><th>Version 6</th></tr>
<tr><td>Baseline</td><td>Median of finite 0.5 s bout-summary bin means, [−15,0)</td><td>Median of direct 1 s bin means, [−20,0)</td></tr>
<tr><td>Bout summary</td><td>Median within each extracted window, repeated over eligible frames</td><td>No bout-median substitution</td></tr>
<tr><td>Time bins</td><td>0.5 s means of repeated bout values</td><td>1 s means of direct log frames</td></tr>
<tr><td>Colours</td><td>Continuous scaled deviations; P10/P90 at ±0.7, range ±1</td><td>Five trial-specific baseline percentile bands</td></tr>
<tr><td>Displayed baseline median</td><td>Zero in all 257 defined scales</td><td>Guaranteed zero before band assignment; all 270 defined</td></tr></table></div>
<p><b>Pros:</b> centres the displayed baseline, matches its low and high percentile anchors across trials, retains continuous colours, and has finer time bins. <b>Cons:</b> equal colours no longer mean equal physical magnitudes; asymmetric scaling changes relative increases and decreases; 13 sparse trials are undefined. Ten bins are a screening rule rather than independent observations, because bins can share a bout. Within-bout variation is suppressed, and bout medians spanning CS can include post-CS frames.</p>
<details><summary>Exact scaling definition</summary><p>For centred baseline values, L = −P10 and R = P90. Let r be the maximum absolute value of the middle baseline order statistics. Set h = max(r, 0.05 min(L,R)), k = 0.7/max(L,R), and c = kh. In the central interval [−h,+h], z = kx. Below −h, z = −c + (0.7−c)/(L−h) × (x+h). Above +h, z = c + (0.7−c)/(R−h) × (x−h). Continue these outer slopes beyond P10/P90. No final subtraction is applied after scaling.</p><p>A scale is undefined if it has fewer than 10 finite baseline bins, a collapsed percentile side, overlap between the centre and an anchor, or transformed finite-sample quantiles that fail the P10/P50/P90 checks. Full parameters and reasons are exported in scaling_parameters.csv.</p></details>
<h2>Version 7 heatmaps</h2>'''
page+='<img alt="Version 7 F G H heatmaps" src="'+uri(stem+'.png','image/png')+'"><p><a download="'+stem+'.pdf" href="'+uri(stem+'.pdf','application/pdf')+'">Download PDF</a> · <a download="'+stem+'.svg" href="'+uri(stem+'.svg','image/svg+xml')+'">Download SVG</a></p>'
page+='<p>Numeric exports (physical and scaled columns): '+ ' · '.join('<a download="'+name+'" href="'+uri(name,'text/csv')+'">'+label+'</a>' for name,label in [('PanelF_mean_bins.csv','F bins'),('PanelG_mean_bins.csv','G bins'),('PanelH_mean_bins.csv','H bins'),('scaling_parameters.csv','Trial scaling parameters')])+'</p>'
page+='<p><a href="history/physical_bin_centred/index.html">Previous Version 7 with physical ±0.25 colours</a></p>'
page+='<p><a href="../fgh_versions_1_to_6_20261009/index.html">Previous versions</a> · <a href="../fgh_version6_softer_high_20261009/index.html">Version 6 softer High comparison</a></p><p>Version 7 updated at the author’s request. The earlier revision is preserved; other versions and frozen figures are unchanged.</p></html>'
(HERE/'index.html').write_text(page,encoding='utf-8')
assert page.count('<img ')==1 and '0.5 s' in page and '±0.7' in page
print('Built standalone Version 7 review')
