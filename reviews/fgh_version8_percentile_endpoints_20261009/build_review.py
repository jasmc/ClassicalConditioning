from pathlib import Path
import base64
HERE=Path(__file__).resolve().parent
def uri(name,mime): return 'data:'+mime+';base64,'+base64.b64encode((HERE/name).read_bytes()).decode()
stem='FGH_Version8_PercentileEndpoints_legacy_layout'
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Version 8: P10/P90 colour endpoints</title>
<style>body{font:17px/1.6 system-ui;color:#25313c;max-width:1250px;margin:30px auto;padding:0 20px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #ccc}a{color:#28567b}.notice{background:#faf7ee;padding:15px}.scroll{overflow:auto}table{min-width:600px}</style>
<h1>Version 8: P10 and P90 at the colour extremes</h1>
<p>Same fish, trials, bout summaries, 0.5 s bins, [−15,0) baseline and median-preserving centre as Version 7. P10 maps to −1, P50 to zero, and P90 to +1 on continuous managua_r. Values beyond P10/P90 receive saturated endpoint colours.</p>
<div class="scroll"><table><tr><th>Feature</th><th>Version 7</th><th>Version 8</th></tr>
<tr><td>Baseline P10 / P90</td><td>−0.7 / +0.7</td><td>−1 / +1</td></tr>
<tr><td>Baseline median</td><td>Zero</td><td>Zero</td></tr>
<tr><td>Colour saturation</td><td>Starts beyond the baseline P10/P90 anchors</td><td>Starts at P10/P90</td></tr>
<tr><td>Finite baseline requirement</td><td>At least 10 bins; valid two-sided spread</td><td>Unchanged</td></tr>
<tr><td>Colour interpretation</td><td>Continuous deviation relative to the trial baseline</td><td>Unchanged; expanded colour contrast</td></tr></table></div>
<p class="notice"><b>Verified:</b> all 257 defined trials have baseline median zero and P10/P90 at −1/+1 to numerical precision. The same 13 sparse Control trials are undefined. No physical log values or existing Version 7 files were changed.</p>
<h2>How it is made</h2><ol>
<li>Use Version 7’s physical log differences: within-window bout medians repeated on eligible frames, averaged in 0.5 s bins, then centred by the median of finite baseline bins during [−15,0).</li>
<li>Keep each trial’s baseline P10/P90 and the width of its symmetric centre. Multiply all Version 7 scaling slopes by 1/0.7. Equivalently, Version 8 scaled values equal Version 7 scaled values divided by 0.7.</li>
<li>Plot with managua_r and colour limits [−1,+1]. Values ≤P10 use the low endpoint colour; values ≥P90 use the high endpoint colour. Zero remains at the dark, nonblack midpoint. Missing or undefined values are black.</li>
<li>Retain physical values in delta_log_vigor and unclipped scaled values in scaled_baseline_vigor. Saturation applies to the displayed colours. Original Version 7 scaled values are also exported for comparison.</li></ol>
<p><b>Tradeoff:</b> Version 8 uses the full colour range for the baseline’s P10–P90 span. This increases contrast but merges large responses beyond those anchors into the same endpoint colours. Equal colours across trials describe baseline-relative position, rather than equal physical log magnitudes.</p>
<h2>Version 8 heatmaps</h2>'''
page+='<img alt="Version 8 F G H heatmaps" src="'+uri(stem+'.png','image/png')+'"><p><a download="'+stem+'.pdf" href="'+uri(stem+'.pdf','application/pdf')+'">PDF</a> · <a download="'+stem+'.svg" href="'+uri(stem+'.svg','image/svg+xml')+'">SVG</a></p>'
page+='<p>Numeric exports: '+' · '.join('<a download="'+name+'" href="'+uri(name,'text/csv')+'">'+label+'</a>' for name,label in [('PanelF_mean_bins.csv','F bins'),('PanelG_mean_bins.csv','G bins'),('PanelH_mean_bins.csv','H bins'),('scaling_parameters.csv','Scaling parameters')])+'</p>'
page+='<p><a href="../fgh_version7_unpooled_recipe_20261009/index.html">Compare with Version 7</a> · <a href="../fgh_versions_1_to_6_20261009/index.html">All previous recipes</a></p><p>Additional review candidate; no figure was frozen or replaced.</p></html>'
(HERE/'index.html').write_text(page,encoding='utf-8')
assert page.count('<img ')==1
print('Built Version 8 review')
