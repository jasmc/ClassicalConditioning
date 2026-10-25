#!/usr/bin/env python3
"""Assemble a self-contained, unselected F–H history comparison without re-analysis."""
from __future__ import annotations

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]


import argparse
import base64
import copy
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = _COMMAND_REPOSITORY
sys.path.insert(0, str(ROOT / 'src'))
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot, file_hash, sha, state_lock, tag

ARCHIVE = Path('/Volumes/JOAQUIM/ClassicalConditioning-Archive/repository-cleanup/2026-10-10/payload')
REVIEWS = Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews')


def plain(markup):
    return ' '.join(html.unescape(re.sub(r'<[^>]*>', ' ', markup)).split())


from classical_conditioning.figures.svg_scope import scoped_working_svg


from classical_conditioning.figures.svg_scope import prefix_svg


def build(ticket_path, output, evidence):
    ticket = json.loads(ticket_path.read_text())
    state_path = Path(ticket['state_path'])
    for path, expected in ticket['starting_hashes'].items():
        if file_hash(path) != expected:
            raise ValueError('Starting input changed: ' + path)
    state = json.loads(state_path.read_text())
    inventory_path = ROOT / 'records/repository-archive/starting-inventory.json'
    inventory = {r['path']: r['sha256'] for r in json.loads(inventory_path.read_text())['files'] if r.get('sha256')}
    inputs = {str(Path(p).resolve()): h for p, h in ticket['starting_hashes'].items()}
    def load(path, expected=None):
        path = Path(path)
        raw = path.read_bytes()
        actual = sha(raw)
        if expected is not None and actual != expected:
            raise ValueError('Artifact authentication failed: ' + str(path))
        inputs[str(path.resolve())] = actual
        return raw
    load(inventory_path)
    load(ticket_path)
    load(Path(__file__))
    def archived(relative):
        if relative not in inventory:
            raise ValueError('Artifact absent from archive inventory: ' + relative)
        return load(ARCHIVE / relative, inventory[relative])
    selection = json.loads(load(ROOT / 'configs/paper-figures/selections/figure1-fgh-version12-freeze-20261009.json'))
    container_path = ARCHIVE / 'reviews/fgh_candidate_palette_versions_20261009/index.html'
    source = load(container_path, selection['container']['sha256']).decode()
    def embedded_json(name):
        return json.loads(re.search(r'<script[^>]*id="' + name + r'"[^>]*>(.*?)</script>', source, re.S)[1])
    catalogue = embedded_json('version-history-catalogue')
    frozen_archive = embedded_json('version12-frozen-archive')
    frozen_entry = frozen_archive['files']['frozen-candidate.svg']
    frozen_svg = base64.b64decode(frozen_entry['base64'])
    assert sha(frozen_svg) == frozen_entry['sha256'] == selection['exports'][0]['sha256']
    sections = {m[1]: m[2] for m in re.finditer(r'<section\b[^>]*id="([^"]+)"[^>]*>(.*?)</section>', source, re.S)}
    cards = []
    def add(key, title, family, difference, visual, reference, notes='', legend=''):
        cards.append(dict(id=key, title=title, family=family, difference=difference,
                          visual=visual, reference=reference, notes=notes, legend=legend))
    magma_evidence_path = ROOT / 'records/figure-reviews/figure-01/fgh-magma-comparison-2026-10-13.json'
    magma_evidence = json.loads(load(magma_evidence_path))
    magma_page = load(magma_evidence['review']['path'], magma_evidence['review']['sha256']).decode()
    magma_svg = re.search(r'<svg\b.*?</svg>', magma_page, re.S)[0].encode()
    current_raw = load(state['draft']['working_svg']['path'], state['draft']['working_svg']['sha256'])
    current_svg, current_scope_hash = scoped_working_svg(current_raw)
    legend_path = Path(state['draft']['legend']['path'])
    paragraphs = load(legend_path, state['draft']['legend']['sha256']).decode().split('\n\n')
    current_legend = '\n\n'.join(p for p in paragraphs if p.startswith('(F–H)') or p.startswith('Displayed trial numbers'))
    current_legend = current_legend.split(' In B,')[0]
    add('current', 'Current Figure 1 draft v16', 'latest',
        'Accepted F–H V12 values and bright blue–charcoal–red palette. Display rows 1–90; F arrows mark 5, 13, 59, 62, 89. Native DejaVu Sans text; G is labelled 3sTrace.',
        current_svg, dict(path=state['draft']['working_svg']['path'], sha256=sha(current_raw), scoped_svg_sha256=current_scope_hash),
        'Current presentation draft; no whole-figure formal freeze.', current_legend)
    # Bind the comparison to the same F–H geometry and heatmap pixels, even if
    # another chat has subsequently updated an unrelated full-figure panel.
    old_source = Path(magma_evidence['source_svg']['path'])
    assert old_source == Path(state['draft']['working_svg']['path'])
    assert all(next(n for n in ET.fromstring(current_svg).iter() if n.get('data-source-id') == f'fig1__{p.lower()}__heatmap__vigor-cells').get('{http://www.w3.org/1999/xlink}href') for p in 'FGH')
    add('magma', 'Magma alternative on current F–H values', 'latest',
        'Palette only: sequential magma over the same −1…+1 scale. Zero is the midpoint colour; missing bins stay black. Cell values, bounds, timing, guides, arrows and numbering match its authenticated v16 source.',
        magma_svg, dict(path=magma_evidence['review']['path'], sha256=magma_evidence['review']['sha256']),
        'Unselected alternative. Archived magma use confirmed; association with January remains unconfirmed.', current_legend)
    recipe_notes = {
        'v1-c': 'Eligible sample timing, with complete-bout log medians repeated on their eligible frames; [−15,0) sample baseline. Divide by half the P10–P90 width and numerically clip to ±1. 269/270 defined.',
        'v1-d': 'Same bout summaries and baseline as V1 C; denominator is max(P50−P10, P90−P50). Numerical clipping to ±1 remains; equal or less saturation. 269/270 defined.',
        'v2': 'Direct mean of eligible log frames in 0.5 s bins; equal finite-bin baseline median over [−15,0). Half P10–P90 width scaling and numerical clipping to ±1. 269/270 defined.',
        'v6-five': 'Direct 1 s log-frame means; [−20,0) finite-bin reference. Five percentile bands at P25/P45/P55/P75, with a dark central band. 270/270 defined.',
        'v6-soft': 'Same data and percentile bands as V6 five colours. The two views are visually identical under the common candidate palette; the earlier distinction was the High colour.',
        'v7-physical': '0.5 s means of repeated bout summaries, then median-centred over finite baseline bins [−15,0). Physical log difference, no percentile amplitude scaling; colours saturate at ±0.25. 270/270 defined.',
        'v7-scaled': 'Same V7 bout-summary bins, with separate lower/upper P10/P90 stretches; anchors at ±0.7 and colour limits ±1. Minimum 10 baseline bins; 257/270 defined.',
        'v8': 'V7 scaled values divided by 0.7: baseline anchors reach ±1, increasing contrast and colour saturation. The same 13 sparse Control trials remain undefined.',
        'v9': 'Unchanged V7 scaled values placed in five fixed bands at −0.5/−0.1/+0.1/+0.5. Within-band magnitudes are hidden; 257/270 defined.',
        'v10': 'V7 values in three bands with boundaries −0.1/+0.1; lower and upper categories merge moderate and extreme deviations. Same missing trials as V7/V9.',
        'v11': 'Direct medians of eligible log frames in 0.5 s bins; subtract median of finite baseline-bin medians [−15,0). One symmetric max(−P10,P90) denominator, without the 0.7 factor. 257/270 defined.',
        'v12': 'Same direct log-bin medians, baseline and missingness as V11. Multiply the symmetric scaled values by 0.7 for response headroom; continuous bright blue–charcoal–red, limits ±1. 257/270 defined.'}
    for recipe in catalogue['recipes']:
        key = recipe['id']
        section = sections[key]
        title = plain(re.search(r'<h2[^>]*>(.*?)</h2>', section, re.S)[1])
        title = title.replace('(current)', '(historical scaled option)')
        src = re.search(r'<img\b[^>]*src="([^"]+)"', section)[1]
        assert src.startswith('data:image/png;base64,') or (key == 'v12' and src.startswith('data:image/svg+xml;base64,'))
        image_bytes = base64.b64decode(src.split(',', 1)[1])
        visual = frozen_svg if key == 'v12' else src
        ref = dict(path=str(container_path), container_sha256=selection['container']['sha256'], entry=key,
                   image_sha256=sha(image_bytes), native_frozen_svg_sha256=sha(frozen_svg) if key == 'v12' else None)
        legend = ' '.join(plain(p) for p in re.findall(r'<p\b[^>]*>(.*?)</p>', section, re.S))
        add('recipe-' + key, 'F–H recipe ' + title, 'recipe', recipe_notes[key], visual, ref,
            'Accepted scoped scientific freeze.' if key == 'v12' else 'Historical recipe review. Original acquisition labels and saved annotations retained.', legend)
    extras = [
        ('v4-medians','V4 earlier: direct 0.5 s bin medians','fgh_version4_bin_medians_20261009/FGH_V4_direct_bin_medians_log025.svg','Eligible log-frame medians; subtract median of finite baseline bins [−15,0). No percentile scaling or numeric clipping; physical colour limits ±0.25. All 270 baselines defined.'),
        ('v4-floor-median','V4 intermediate: non-bout −∞ and medians','fgh_version4_baseline20_20261009/FGH_Version4_DirectBinMedians_legacy_layout.svg','Valid non-bout frames contribute −∞; bin and baseline medians over [−20,0). A −∞ baseline leaves the trial undefined. 107/270 defined: Delay 20, Trace 87, Control 0.'),
        ('v4-floor-mean','V4 intermediate: non-bout −∞ and means','fgh_version4_means_baseline20_20261009/FGH_Version4_DirectBinMeans_legacy_layout.svg','Arithmetic means replace medians. A single −∞ baseline contribution makes the reference −∞; all 270 trials are undefined. Black rows denote undefined arithmetic.'),
        ('v4-quarter','V4 revised: direct 0.25 s means','fgh_version4_quartersecond_means_20261009/FGH_Version4_DirectBinMeans_legacy_layout.svg','Ignore non-bout/invalid NaNs. Direct eligible log-frame means in 0.25 s bins; subtract the equal-bin baseline mean over [−20,0). No scaling; physical colour limits ±0.25. All 270 defined; mean-zero does not imply median-zero.'),
        ('v5','V5: direct 1 s means','fgh_colour_binning_candidates_20261009/FGH_Version5_DirectBinMeans_legacy_layout.svg','V4 revised recipe with 1 s bins instead of 0.25 s: equal-bin [−20,0) mean baseline, no percentile scaling or numeric clipping, physical colour limits ±0.25. Coarser timing.'),
        ('v7-original','V7 earlier: original frame-centred reference','fgh_version7_unpooled_recipe_20261009/history/original_frame_centred/FGH_Version7_UnpooledRecipe_legacy_layout.svg','Earlier bout-summary/mean-bin view centred using a frame baseline; its displayed baseline-bin median is not guaranteed to be zero. Superseded by the bin-centred V7 options.'),
    ]
    for key, title, relative, difference in extras:
        raw = archived('reviews/' + relative)
        add('history-' + key, title, 'history', difference, raw,
            dict(path=str(ARCHIVE / 'reviews' / relative), sha256=sha(raw)), 'Historical saved SVG, unchanged scientific contents.')
    # The later Figure 1 numbers are assembly/presentation drafts, not new
    # scientific recipes. Their F–H rows are compared independently of A–E.
    v10_record = json.loads(load(ROOT / 'records/figure-reviews/figure-01/working-version-selection-2026-10-13-v10.json'))
    draft_scope_hashes = {}
    for version in range(10, 16):
        if version == 10:
            path = Path(v10_record['kept']['path'])
            expected = v10_record['kept']['sha256']
        else:
            path = REVIEWS / f'figure-01-v{version}-editing.svg'
            audit_path = ROOT / f'records/figure-reviews/figure-01/figure-01-v{version}-editing-effective-presentation.json'
            expected = json.loads(load(audit_path))['svg_sha256']
        raw = load(path, expected)
        svg, scope_hash = scoped_working_svg(raw)
        draft_scope_hashes[str(version)] = scope_hash
        add('draft-' + str(version), f'Figure 1 presentation draft v{version} · F–H only', 'draft',
            'Same F–H source row as presentation drafts v10–v15, verified by scoped SVG hash. V12 scientific values and blue–charcoal–red colours; F arrows preserve acquisition IDs 9, 17, 63, 66, 93. Pre-v16 trial-label convention and condition typography retained.',
            svg, dict(path=str(path), sha256=sha(raw), scoped_svg_sha256=scope_hash),
            'v15 was approved, but its whole-figure freeze gate failed; approved candidate is preserved.' if version == 15 else 'Historical presentation draft. Whole-figure differences outside F–H are omitted.')
    assert len(set(draft_scope_hashes.values())) == 1
    add('draft-16', 'Figure 1 presentation draft v16 · current F–H', 'draft',
        'Same current row shown at the top: displayed trial = acquisition trial −4, rows 1–90; F examples 5, 13, 59, 62, 89; condition 3sTrace and embedded DejaVu Sans regular/bold. Heatmap pixel payloads preserve scientific V12.',
        current_svg, cards[0]['reference'], 'Current draft, duplicated here to complete the v10–v16 sequence.')
    # All visuals are embedded, with SVG ID references isolated per card.
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Figure 1 F–H · all versions</title>',
             '<style>body{margin:0;color:#17222b;background:#f5f6f7;font:16px/1.5 system-ui}main{max-width:1600px;margin:auto;padding:24px}h1{margin:0;font-size:27px}h2{font-size:18px;margin:0 0 8px}p{margin:8px 0}.intro{max-width:105ch}.controls{position:sticky;top:0;z-index:3;background:#f5f6f7;padding:10px 0;display:flex;flex-wrap:wrap;gap:8px;align-items:center}button,select{font:inherit;padding:6px 10px;border:1px solid #aab5bd;border-radius:5px;background:white}button{cursor:pointer}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.card{background:white;border:1px solid #cbd3d9;border-radius:7px;padding:16px;scroll-margin-top:90px}.card[hidden]{display:none}.tag{font-size:12px;font-weight:600;color:#405567;text-transform:uppercase;letter-spacing:.04em}.visual svg,.visual img{display:block;width:100%;height:auto}.difference{font-size:15px}.notes{font-size:13px;color:#576775}.pick{float:right;font-size:13px}details{font-size:14px;margin-top:8px}details p{white-space:pre-line}.coverage{background:white;padding:16px;border:1px solid #cbd3d9;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border:1px solid #d5dce1;text-align:left;vertical-align:top}a{color:#175b91}@media(max-width:850px){.grid{grid-template-columns:1fr}main{padding:12px}}@media print{.controls,.pick{display:none}.grid{display:block}.card{break-inside:avoid;margin-bottom:16px}.card[hidden]{display:block}}</style></head><body><main>',
             '<h1>Figure 1 F–H: all recorded versions</h1><p class="intro">Two numbering systems appear here: <b>F–H recipes V1–V12</b> change aggregation, baseline, scaling or palette; <b>Figure 1 drafts v10–v16</b> change the presentation of the assembled figure. Scientific V12 remains the accepted F–H selection. Equal colours across different recipes need not represent equal values.</p>',
             '<p class="intro">F: Delay fish 20221115_07 · G: 3sTrace fish 20230310_08 · H: Control fish 20221115_09. Historical figures retain their saved trial labels, condition names and annotations. Current displayed rows 1–90 correspond to acquisition trials 5–94. Black denotes missing/undefined values according to each recipe. No numerical analysis was rerun.</p>',
             '<div class="controls"><button id="all">Show all</button><button id="latest">Current vs magma</button><button id="chosen">Compare checked versions</button><select id="family" aria-label="Version family"><option value="all">All families</option><option value="latest">Current and magma</option><option value="recipe">Archived recipe gallery</option><option value="history">Additional historical revisions</option><option value="draft">Figure 1 drafts v10–v16</option></select><span id="count" aria-live="polite"></span></div>',
             '<div class="grid">']
    for card in cards:
        visual = prefix_svg(card['visual'], card['id']) if isinstance(card['visual'], bytes) else '<img alt="' + html.escape(card['title'], quote=True) + '" src="' + card['visual'] + '">'
        parts.append('<article class="card" id="' + card['id'] + '" data-family="' + card['family'] + '"><label class="pick"><input type="checkbox" aria-label="Compare ' + html.escape(card['title'], quote=True) + '"> Compare</label><div class="tag">' + html.escape(card['family']) + '</div><h2>' + html.escape(card['title']) + '</h2><p class="difference"><b>Key differences.</b> ' + html.escape(card['difference']) + '</p><div class="visual">' + visual + '</div><p class="notes">' + html.escape(card['notes']) + '</p>')
        if card['legend']:
            parts.append('<details><summary>Associated legend and recipe details</summary><p>' + html.escape(card['legend']) + '</p></details>')
        parts.append('</article>')
    parts.append('</div><section class="coverage"><h2>Documented gaps and earlier palette revisions</h2><p><b>V3:</b> no established scientific recipe or saved comparison view is recorded; no substitute is invented.</p><p><b>Earlier V11:</b> separate-side scaling preceded the symmetric denominator shown here. <b>Earlier V12 palette experiments:</b> ' + html.escape('; '.join(catalogue['version12_review_history'][:-1])) + '. These are documented revisions; distinct saved gallery images are not present in the authenticated selected container.</p><p>V4 median/−∞/mean/quarter-second revisions and V5 are included above from their preserved SVGs. Drafts v10–v15 have an identical F–H nested SVG; their whole-figure changes occurred outside this chat’s F–H scope. No whole-figure freeze, version selection, disposal or archival action is performed by this page.</p></section>')
    meta = dict(scope=['F', 'G', 'H'], status='read-only historical comparison; not a working-version selection',
                current_state_sha256=file_hash(state_path), sources=inputs,
                cards=[{k: v for k, v in c.items() if k not in ('visual',)} for c in cards],
                draft_v10_v15_identical_fgh_scope=draft_scope_hashes, missing_version_3=True,
                active_pair_updated=False, scientific_recalculation=False, standalone_svg_created=False)
    parts.append('<script type="application/json" id="comparison-provenance">' + json.dumps(meta).replace('<', '\\u003c') + '</script>')
    parts.append('<script>const cards=[...document.querySelectorAll(".card")],sel=document.getElementById("family"),count=document.getElementById("count");function show(test){cards.forEach(c=>c.hidden=!test(c));count.textContent=cards.filter(c=>!c.hidden).length+" / "+cards.length+" views"}document.getElementById("all").onclick=()=>{sel.value="all";show(()=>true)};document.getElementById("latest").onclick=()=>{sel.value="latest";show(c=>c.dataset.family==="latest")};document.getElementById("chosen").onclick=()=>show(c=>c.querySelector("input").checked);sel.onchange=()=>show(c=>sel.value==="all"||c.dataset.family===sel.value);show(()=>true);</script></main></body></html>')
    payload = ''.join(parts).encode()
    meta['html'] = dict(path=str(output), sha256=sha(payload), bytes=len(payload))
    meta['visual_review'] = 'pending exact saved-render inspection'
    evidence_bytes = (json.dumps(meta, indent=2) + '\n').encode()
    snap = Snapshot([*command_sources(__file__), *(inputs)])
    snap.hashes.update(inputs)
    made = []
    with state_lock(state_path):
        snap.check()
        try:
            for path, raw in ((output, payload), (evidence, evidence_bytes)):
                if not path.parent.is_dir():
                    raise FileNotFoundError(path.parent)
                with path.open('xb') as stream:
                    stream.write(raw)
                made.append(path)
                assert file_hash(path) == sha(raw)
            snap.check()
        except Exception:
            for path in made:
                path.unlink()
            raise
    print(json.dumps(dict(html=str(output), views=len(cards), bytes=len(payload), draft_v10_v15_identical=True), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    build(args.baseline, args.output, args.evidence)
