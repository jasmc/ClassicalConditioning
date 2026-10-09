"""Presentation-only V6 trial: soften High using the same managua_r palette."""
from pathlib import Path
import sys, json, copy
from matplotlib.colors import to_hex

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'reviews/fgh_colour_binning_candidates_20261009'))
import render_candidates as plotting

source = REPO / 'reviews/fgh_version6_onesecond_centralband_20261009'
original = REPO / 'reviews/fgh_colour_binning_candidates_20261009/FGH_Version6_OneSecondCentralBand_legacy_layout.png'
manifest = json.loads((source / 'data_manifest.json').read_text())
colours = copy.deepcopy(manifest['colour_mapping'])
before = colours['colours'][-1]
colours['colours'][-1] = to_hex(plotting.CMAP(.90))
colours['managua_r_sample_positions'][-1] = .90
tracked = [source / 'data_manifest.json', *source.glob('*.csv'), original]
hashes = {str(p): plotting.digest(p) for p in tracked}
plotting.HERE = HERE
plotting.render('Version6_SofterHigh', source, 'band_class', True, discrete=colours)
assert all(plotting.digest(Path(p)) == h for p, h in hashes.items())
(HERE / 'presentation_change.json').write_text(json.dumps({
    'status': 'exploratory colour candidate; not selected or frozen',
    'change': 'High only: managua_r sample 1.00 to 0.90',
    'original_high': before, 'candidate_high': colours['colours'][-1],
    'data_thresholds_binning_and_other_colours_unchanged': True,
    'preserved_source_hashes': hashes,
}, indent=2))
candidate = HERE / 'FGH_Version6_SofterHigh_legacy_layout.png'
page = '''<!doctype html><html><meta charset="utf-8"><title>Version 6: softer High</title>
<style>body{font:17px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#222}img{width:100%;height:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:850px){.pair{grid-template-columns:1fr}}a{color:#175b89}</style>
<h1>Version 6: softer High</h1><p>The High band now uses a less bright amber from <b>managua_r at 0.90</b> (#e09f57), instead of the endpoint at 1.00 (#ffcf67). The dark centre and other three colours are unchanged.</p>
<p>Both figures use identical 1 s bins, trial-specific baseline quartiles, and the central P45–P55 band. Only the High colour changes. This is a candidate for comparison.</p>
<div class="pair"><section><h2>Original High</h2><img src="../fgh_colour_binning_candidates_20261009/FGH_Version6_OneSecondCentralBand_legacy_layout.png"></section><section><h2>Softer High</h2><img src="FGH_Version6_SofterHigh_legacy_layout.png"><p><a href="FGH_Version6_SofterHigh_legacy_layout.pdf">PDF</a> · <a href="FGH_Version6_SofterHigh_legacy_layout.svg">SVG</a></p></section></div>
<p><a href="../fgh_versions_1_to_6_20261009/index.html#v6-five">All available versions</a></p></html>'''
(HERE / 'index.html').write_text(page, encoding='utf-8')
