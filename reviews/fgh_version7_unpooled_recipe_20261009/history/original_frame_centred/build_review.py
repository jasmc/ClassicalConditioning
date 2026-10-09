from pathlib import Path
import base64,json
HERE=Path(__file__).resolve().parent
def uri(name,mime): return 'data:'+mime+';base64,'+base64.b64encode((HERE/name).read_bytes()).decode()
stem='FGH_Version7_UnpooledRecipe_legacy_layout'
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Version 7 — single-fish bout recipe</title>
<style>body{font:17px/1.6 system-ui;color:#25313c;max-width:1250px;margin:30px auto;padding:0 20px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #ccc}a{color:#28567b}.notice{background:#faf7ee;padding:15px} .scroll{overflow:auto}table{min-width:600px}</style>
<h1>Version 7: your recipe, without pooling</h1>
<p>Each fish is shown separately: F Delay 20221115_07, G 3 s Trace 20230310_08, H Control 20221115_09. Trials 5–94, aligned to CS, cover [−20,+20) seconds.</p>
<h2>How it is made</h2><ol>
<li>Use the original eligible natural-log vigor frames before bout summarization. Invalid frames and non-bout frames remain missing.</li>
<li>For each fish and trial, subtract the median of eligible original log frames during [−15,0) seconds. A missing baseline would make the whole trial missing.</li>
<li>Within the extracted [−20,+20) window, calculate each bout’s median baseline-centred log value. Repeat it only on that bout’s eligible frames. A crossing-window bout uses only its eligible frames within the window.</li>
<li>Average finite repeated bout values into 0.5 s bins. Bouts contribute according to their eligible frame count within each bin. Empty bins remain missing.</li>
<li>Use continuous managua_r with fixed −0.25 to +0.25 limits, zero at its dark midpoint, and black for missing cells. No percentile scaling, 0–1 scaling or fish pooling is applied. Exported numeric values are not clipped; values outside the colour limits receive endpoint colours.</li></ol>
<p class="notice"><b>Baseline distinction:</b> zero represents each trial’s original baseline-frame median. Bout summaries and bin averaging can shift the median of the displayed baseline cells away from zero. All 270 baseline references are defined. The largest absolute displayed baseline-bin median is 0.19408 log units.</p>
<p><b>Input provenance:</b> this candidate uses the same reconstructed angular-speed values and detected-bout eligibility masks as the current F/G/H candidates. It follows the uploaded aggregation recipe; it does not re-import the separate processed legacy metric and movement masks used by the pooled figure.</p>
<h2>Comparison with Version 6</h2><div class="scroll"><table><tr><th>Feature</th><th>Version 7</th><th>Version 6</th></tr>
<tr><td>Baseline</td><td>Original log-frame median, [−15,0)</td><td>Median of direct 1 s bin means, [−20,0)</td></tr>
<tr><td>Bout summary</td><td>Median within each extracted window, repeated over eligible frames</td><td>No bout-median substitution</td></tr>
<tr><td>Time bins</td><td>0.5 s means of repeated bout values</td><td>1 s means of direct log frames</td></tr>
<tr><td>Colours</td><td>Continuous physical log differences; fixed ±0.25</td><td>Five trial-specific baseline percentile bands</td></tr>
<tr><td>Displayed baseline median</td><td>Can differ from zero</td><td>Guaranteed zero before band assignment</td></tr></table></div>
<p><b>Pros:</b> matches the uploaded aggregation recipe, preserves a common physical log scale, and has finer time bins. <b>Cons:</b> suppresses within-bout variation, weights bouts by their eligible duration, and does not guarantee the displayed baseline median is zero. Bout medians spanning CS can also include post-CS frames.</p>
<h2>Version 7 heatmaps</h2>'''
page+='<img alt="Version 7 F G H heatmaps" src="'+uri(stem+'.png','image/png')+'"><p><a download="'+stem+'.pdf" href="'+uri(stem+'.pdf','application/pdf')+'">Download PDF</a> · <a download="'+stem+'.svg" href="'+uri(stem+'.svg','image/svg+xml')+'">Download SVG</a></p>'
page+='<p><a href="../fgh_versions_1_to_6_20261009/index.html">Previous versions</a> · <a href="../fgh_version6_softer_high_20261009/index.html">Version 6 softer High comparison</a></p><p>Additional review candidate requested by the author; no prior version or frozen figure was replaced.</p></html>'
(HERE/'index.html').write_text(page,encoding='utf-8')
assert page.count('<img ')==1 and '0.5 s' in page and '±0.25' in page
print('Built standalone Version 7 review')
