"""Recolour authenticated comparison heatmaps without redrawing other artists.

prepare stages bytes for visual review; publish rechecks the starting ticket and
uses the shared locked publisher. Source numerical caches are never modified.
"""
from pathlib import Path
import argparse
import base64
import copy
import hashlib
import io
import json
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import matplotlib as mpl
import numpy as np
import pandas as pd
from PIL import Image
from classical_conditioning.figure_workspace import Snapshot, publish_state

XLINK = '{http://www.w3.org/1999/xlink}href'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def packed(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()


def image_array(node):
    return np.array(Image.open(io.BytesIO(base64.b64decode(node.attrib[XLINK].split(',')[1]))).convert('RGBA'))


def local_data(record):
    path = record['path']
    return Path('/Volumes/JOAQUIM') / path[3:].replace('\\', '/') if path.startswith('J:\\') else Path(path)


def prepare(args):
    folder, stage = args.directory, args.stage
    baseline = json.loads(args.ticket.read_text())
    snapshot = Snapshot([])
    snapshot.hashes = baseline
    snapshot.check()
    raw = (folder / 'candidate-comparison.svg').read_text()
    root = ET.fromstring(raw)
    registry = json.loads((folder / 'candidate-comparison.figure.json').read_text())
    nodes, matrices, old_pixels = [], [], []
    for key, role in registry['artist_mappings'].items():
        if role.get('scientific_role') != 'heatmap.vigor':
            continue
        path = local_data(role['data'])
        if digest(path.read_bytes()) != role['data']['sha256']:
            raise ValueError('Unmatched heatmap source: ' + str(path))
        bins = pd.read_parquet(path)
        matrix = bins.pivot(index='acquisition_trial', columns='bin_index', values='delta_log_vigor').reindex(index=range(5, 95), columns=range(40)).to_numpy()
        node = next(n for n in root.iter() if n.attrib.get('id') == key)
        nodes.append(node)
        matrices.append(matrix[::-1])  # Saved SVG image uses a vertical flip.
        old_pixels.append(image_array(node))
    assert len(nodes) == 7
    cmap = mpl.colormaps['managua_r'].copy()
    cmap.set_bad('black')
    previous = {tuple(registry['artist_mappings'][n.attrib['id']]['limits']) for n in nodes}
    if len(previous) != 1:
        raise ValueError('Comparison does not have one shared scale')
    old_limits = previous.pop()
    values = np.ma.masked_invalid(np.stack(matrices))
    colors = cmap(mpl.colors.Normalize(*old_limits)(values), bytes=True)
    pixels = np.stack(old_pixels)
    _, height, width, _ = pixels.shape
    # Recover the original nearest-neighbour cell boundaries from all seven
    # authenticated images, then demand an exact RGBA match at EVERY pixel.
    # No numerical values are inferred from colour or changed by this mapping.
    xc = ((np.arange(40) + .5) * width / 40).astype(int)
    yc = ((np.arange(90) + .5) * height / 90).astype(int)
    row_map = []
    for y in range(height):
        nearest = int((y + .5) * 90 / height)
        choices = range(max(0, nearest - 1), min(90, nearest + 2))
        row_map.append(min(choices, key=lambda i: np.count_nonzero(pixels[:, y, xc] != colors[:, i])))
    col_map = []
    for x in range(width):
        nearest = int((x + .5) * 40 / width)
        choices = range(max(0, nearest - 1), min(40, nearest + 2))
        col_map.append(min(choices, key=lambda i: np.count_nonzero(pixels[:, yc, x] != colors[:, :, i])))
    if not np.array_equal(pixels, colors[:, np.array(row_map)[:, None], np.array(col_map)[None, :]]):
        raise ValueError('Source matrices do not exactly reconstruct saved raster pixels')
    new_colors = cmap(mpl.colors.Normalize(args.vmin, args.vmax)(values), bytes=True)
    new_pixels = new_colors[:, np.array(row_map)[:, None], np.array(col_map)[None, :]]
    for node, array in zip(nodes, new_pixels):
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format='PNG')
        old = node.attrib[XLINK]
        new = 'data:image/png;base64, ' + base64.b64encode(buffer.getvalue()).decode()
        if raw.count(old) != 1:
            raise ValueError('Ambiguous image binding')
        raw = raw.replace(old, new)
        registry['artist_mappings'][node.attrib['id']]['limits'] = [args.vmin, args.vmax]
    # Fixed palette colourbar pixels remain correct; only endpoint values change.
    for index, label in [('000', f'{args.vmin:.2f}'.replace('-', '−')), ('002', f'{args.vmax:.2f}')]:
        group = next(n for n in root.iter() if n.attrib.get('id') == f'tick-label__comparison-colourbar__y__{index}')
        text = next(n for n in group.iter() if n.tag.endswith('text'))
        pattern = r'(<g id="tick-label__comparison-colourbar__y__' + index + r'">\s*<text\b[^>]*>)[^<]*(</text>)'
        raw, count = re.subn(pattern, lambda m: m[1] + label + m[2], raw)
        if count != 1:
            raise ValueError('Unmatched colourbar label: ' + str(text.text))
    registry['presentation_edit'] = {'scope': 'candidate-comparison only', 'limits': [args.vmin, args.vmax], 'data_changed': False, 'source_svg_sha256': baseline[str((folder / 'candidate-comparison.svg').resolve())], 'generator': str(Path(__file__).resolve()), 'generator_sha256': digest(Path(__file__).read_bytes())}
    registry['layout_frames']['status'] = 'geometry-preserved; browser review unavailable'
    registry['layout_frames']['issues'] = ['Exact-browser refresh blocked by local-file URL policy; raster render reviewed separately']
    registry_bytes = packed(registry)
    old_registry_hash = digest((folder / 'candidate-comparison.figure.json').read_bytes())
    raw = raw.replace(old_registry_hash, digest(registry_bytes))
    legend = (folder / 'comparison-legend.md').read_text().replace('±0.25', f'±{args.vmax:g}')
    html = (folder / 'candidate-comparison.html').read_text()
    original_svg = (folder / 'candidate-comparison.svg').read_text()
    assert html.count(original_svg) == 1
    html = html.replace(original_svg, raw).replace('managua_r ±0.25', f'managua_r ±{args.vmax:g}')
    stage.mkdir(exist_ok=False)
    for name, content in [('candidate-comparison.svg', raw.encode()), ('candidate-comparison.html', html.encode()), ('candidate-comparison.figure.json', registry_bytes), ('comparison-legend.md', legend.encode())]:
        (stage / name).write_bytes(content)
    report = {'limits': [args.vmin, args.vmax], 'source_pixels_reconstructed_exactly': True, 'heatmaps_recolored': len(nodes), 'unchanged': ['numerical caches', 'missing-cell masks', 'cell placement', 'outcome artists', 'rankings', 'preferred-single-fish', 'main Figures 1 and 2'], 'staged_hashes': {p.name: digest(p.read_bytes()) for p in stage.iterdir()}, 'baseline': baseline, 'generator_sha256': digest(Path(__file__).read_bytes())}
    snapshot.check()
    (stage / 'verification.json').write_bytes(packed(report))
    print(json.dumps({k: v for k, v in report.items() if k not in ('baseline', 'staged_hashes')}))


def publish(args):
    folder, stage = args.directory, args.stage
    report = json.loads((stage / 'verification.json').read_text())
    if digest(Path(__file__).read_bytes()) != report['generator_sha256']:
        raise ValueError('Generator changed after staging')
    snapshot = Snapshot([])
    snapshot.hashes = report['baseline']
    snapshot.check()
    updates, archives = [], []
    history = folder / 'history' / ('colour-range-' + digest(packed(report))[:12])
    history.mkdir(exist_ok=False)
    for name, expected in report['staged_hashes'].items():
        data = (stage / name).read_bytes()
        if digest(data) != expected:
            raise ValueError('Staged artifact changed: ' + name)
        updates.append((folder / name, data))
        archives.append((history / name, (folder / name).read_bytes()))
    evidence = dict(report, visual_review=args.visual_review)
    evidence_path = history / 'colour-range-evidence.json'
    archives.append((evidence_path, packed(evidence)))
    state = {'scope': 'fig1-trace10-candidate-comparison', 'default_display': 'draft', 'draft': {key: {'path': str(folder / name), 'sha256': report['staged_hashes'][name]} for key, name in [('working_svg', 'candidate-comparison.svg'), ('review', 'candidate-comparison.html'), ('registry', 'candidate-comparison.figure.json'), ('legend', 'comparison-legend.md')]}, 'colour_limits': report['limits'], 'evidence': {'path': str(evidence_path), 'sha256': digest(packed(evidence))}, 'whole_figure_frozen': False, 'parent_figures_changed': False, 'scope_note': 'Supersedes comparison display references in 2026-10-20 completion records only; no preferred-fish or parent-figure state update.'}
    publish_state(folder / 'current-state.json', state, snapshot, artifacts=archives, evidence_updates=updates)
    print(folder / 'candidate-comparison.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'publish'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--ticket', type=Path)
    parser.add_argument('--vmin', type=float, default=-.1)
    parser.add_argument('--vmax', type=float, default=.1)
    parser.add_argument('--visual-review')
    args = parser.parse_args()
    if not np.isfinite([args.vmin, args.vmax]).all() or args.vmin >= 0 or args.vmax != -args.vmin:
        parser.error('This scoped adapter requires finite symmetric colour limits')
    if args.command == 'prepare' and args.ticket is None:
        parser.error('--ticket is required for prepare')
    if args.command == 'publish' and not args.visual_review:
        parser.error('--visual-review is required for publish')
    (prepare if args.command == 'prepare' else publish)(args)


if __name__ == '__main__':
    main()
