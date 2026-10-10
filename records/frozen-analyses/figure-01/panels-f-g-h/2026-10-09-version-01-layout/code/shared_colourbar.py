"""A shared colourbar for frozen Version 1 F/G/H; only presentation changes."""
from pathlib import Path
import json, re, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from freeze_and_style import snapshot, FISH, CMAP, NORM, COLOURS, digest, HERE

OUT = HERE / 'shared-colourbar'
SIZE = (9, 5.1)
LEFTS = [.055, .32, .585]
WIDTH = .22
LABEL = 'Vigor relative to baseline'

def verify(svg, tables):
    ns = {'s': 'http://www.w3.org/2000/svg'}
    root = ET.parse(svg).getroot()
    reports = []
    for i, (letter, runs) in enumerate(tables):
        group = next(g for g in root.findall('.//s:g', ns) if g.get('id') == f'frozen_samples_{letter}')
        paths = group.findall('s:path', ns)
        assert len(paths) == len(runs)
        xs = WIDTH * SIZE[0] * 72 / 40
        ys = .75 * SIZE[1] * 72 / 90
        for path, row in zip(paths, runs.itertuples()):
            points = np.array([float(v) for v in re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?', path.get('d'))]).reshape(-1, 2)
            np.testing.assert_allclose(np.min(points[:, 0]), LEFTS[i]*SIZE[0]*72+(row.start_s+20)*xs, atol=2e-6, rtol=0)
            np.testing.assert_allclose(np.ptp(points[:, 0]), (row.end_s-row.start_s)*xs, atol=2e-6, rtol=0)
            np.testing.assert_allclose(np.min(points[:, 1]), .12*SIZE[1]*72+(row.trial-5)*ys, atol=2e-6, rtol=0)
            np.testing.assert_allclose(np.ptp(points[:, 1]), ys, atol=2e-6, rtol=0)
            fill = re.search(r'fill:\s*(#[0-9a-f]+)', path.get('style', ''))
            assert (fill.group(1) if fill else '#000000') == to_hex(CMAP(NORM(row.C)))
        reports.append({'panel': letter, 'sample_runs': len(paths), 'values_intervals_positions_and_colours_verified': True})
    return reports

def main():
    frozen = snapshot()
    OUT.mkdir(exist_ok=True)
    fig = plt.figure(figsize=SIZE)
    tables = []
    for i, spec in enumerate(FISH):
        letter, name, fish, *_ = spec
        path = HERE / 'frozen-version1' / f'Panel{letter}_display_sample_runs.csv'
        runs = pd.read_csv(path)
        tables.append((letter, runs))
        ax = fig.add_axes([LEFTS[i], .13, WIDTH, .75])
        ax.set_facecolor('black')
        patches = [Rectangle((r.start_s, r.trial-.5), r.end_s-r.start_s, 1) for r in runs.itertuples()]
        collection = PatchCollection(patches, facecolors=CMAP(NORM(runs.C.to_numpy())), edgecolors='none', antialiaseds=False)
        collection.set_gid(f'frozen_samples_{letter}')
        ax.add_collection(collection)
        ax.set(xlim=(-20, 20), ylim=(94.5, 4.5), xticks=[-20, 0, 20], yticks=list(range(10, 91, 10)))
        ax.tick_params(direction='out', length=2.5, width=.6, labelsize=7, top=False, right=False, pad=2)
        ax.set_xlabel('Time from CS onset (s)', fontsize=8, labelpad=4)
        for spine in ax.spines.values():
            spine.set_linewidth(.6); spine.set_color('#333333')
        for t in [0, 10]:
            ax.axvline(t, color='#168241', linewidth=.65, alpha=.8)
        if spec[-1] is not None:
            ax.plot([spec[-1]]*2, [14.5, 64.5], color='#964bad', linewidth=.6, linestyle=':')
        for y in [14.5, 64.5]:
            ax.axhline(y, color='white', linewidth=.7)
        if i == 2:
            for text, y in [('Pre', 9.5), ('Train', 39.5), ('Test', 79.5)]:
                ax.text(1.045, y, text, transform=ax.get_yaxis_transform(), rotation=90,
                        va='center', ha='left', fontsize=7, color='#444444')
        fig.text(LEFTS[i]-.035, .947, letter, fontsize=14, weight='bold', va='center')
        fig.text(LEFTS[i]+WIDTH/2, .947, name, fontsize=10, ha='center', va='center', color=COLOURS[letter])
        fig.text(LEFTS[i]+WIDTH/2, .909, fish, fontsize=7, ha='center', va='center', color='#555555')
    cax = fig.add_axes([.875, .21, .012, .59])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, ticks=[-1, -.5, 0, .5, 1])
    bar.set_label(LABEL, fontsize=8, labelpad=4)
    bar.ax.tick_params(direction='out', length=2, width=.5, labelsize=7, pad=2)
    bar.outline.set_linewidth(.6)
    bar.solids.set_rasterized(False); bar.solids.set_edgecolor('face')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text()]
    assert all(b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.x1+1 and b.y1 <= fig.bbox.y1+1 for b in boxes)
    stem = OUT / 'FGH_C_BoutSamples_shared_colourbar'
    for ext in ['svg', 'png', 'pdf']:
        fig.savefig(stem.with_suffix('.'+ext), dpi=300, facecolor='white')
    plt.close(fig)
    reports = verify(stem.with_suffix('.svg'), tables)
    record = {'freeze_manifest': str(HERE/'frozen-version1/freeze.json'),
              'freeze_manifest_sha256': digest(HERE/'frozen-version1/freeze.json'),
              'layout': 'Three aligned panels; one managua_r colourbar at far right; phase names right of H; no y-axis title',
              'colourbar_label': LABEL, 'colourbar_count': 1, 'limits': [-1, 1],
              'panels': reports, 'frozen_numeric_values_unchanged': True,
              'outputs': [{'path': str(stem.with_suffix('.'+ext)), 'sha256': digest(stem.with_suffix('.'+ext))} for ext in ['svg','png','pdf']]}
    (OUT/'validation.json').write_text(json.dumps(record, indent=2)+'\n')
    config_path = HERE.parents[1]/'configs/paper-figures/figure1-fgh-version1-freeze-20261009.json'
    config = json.loads(config_path.read_text())
    config['current_layout'] = {'validation': str(OUT/'validation.json'), 'sha256': digest(OUT/'validation.json'),
                                'pdf': str(stem.with_suffix('.pdf')), 'colourbar_label': LABEL}
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(record, indent=2))

if __name__ == '__main__':
    main()
