"""Remake only F/G/H; audit the data median AND actual palette coordinate."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from PIL import Image, ImageOps
from baseline_colour_mapping import baseline_centred_options, baseline_colour_norm

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts'), str(REPO/'src')]
from build_figure1_legacy_vigor_heatmaps import ROOT, FISH, digest

SRC = ROOT/'fgh-trial-baseline-bin-centred-20261007'
OUT = ROOT/'fgh-all-options-baseline-colour-centred-20261007'
OUT.mkdir(parents=True, exist_ok=True)
source_manifest = json.loads((SRC/'build_manifest.json').read_text())
sources, parts, quantiles, legacy_checks = [], [], [], []
for spec in FISH:
    panel = spec[0]
    rec = next(r for r in source_manifest['panels'] if r['panel'] == panel)
    path = Path(rec['panel_data'])
    assert digest(path) == rec['panel_data_sha256']
    sources.append({'path': str(path), 'sha256': digest(path)})
    b = pd.read_parquet(path)
    b['panel'] = panel
    b['input_centred_log_bin'] = b.signed_bout_log_bin
    for col in ['colour_centred_log_bin', 'C_unclipped', 'C', 'D_unclipped', 'D']:
        b[col] = np.nan
    for trial, group in b.groupby('trial'):
        mask = (group.bin_center_s.ge(-15) & group.bin_center_s.lt(0)).to_numpy()
        values = group.signed_bout_log_bin.to_numpy()
        result, meta = baseline_centred_options(values, mask)
        b.loc[group.index, 'colour_centred_log_bin'] = result['log']
        for col in ['C_unclipped', 'C', 'D_unclipped', 'D']:
            b.loc[group.index, col] = result[col]
        quantiles.append({'panel': panel, 'trial': int(trial), **meta})
        if meta['C_scale'] > 0:
            old = np.clip((values-meta['p10'])/(2*meta['C_scale']), 0, 1)
            median = float(np.nanmedian(old[mask]))
            legacy_checks.append({'panel': panel, 'trial': int(trial),
                                  'old_C_palette_baseline_median': median,
                                  'old_C_center_error': median-.5})
        np.testing.assert_allclose(result['log'], values, atol=1e-12, rtol=0, equal_nan=True)
    parts.append(b)

bins = pd.concat(parts, ignore_index=True)
baseline = bins[bins.bin_center_s.ge(-15) & bins.bin_center_s.lt(0)]
p10, p90 = np.quantile(baseline.colour_centred_log_bin.dropna(), [.1, .9])
shared_half_range = max(abs(p10), abs(p90))
configs = [
    ('A', 'colour_centred_log_bin', .25, 'Log units · ±0.25', 'Trial-centred bout log-bin value'),
    ('B', 'colour_centred_log_bin', shared_half_range,
     f'Log units · shared baseline P10/P90\n±{shared_half_range:.3f}', 'Trial-centred bout log-bin value'),
    ('C', 'C', 1., 'Trial-centred · half P10–P90 width\nclipped to ±1', 'Trial-scaled log-bin value'),
    ('D', 'D', 1., 'Trial-centred · larger P10/P90 distance\nclipped to ±1', 'Trial-scaled log-bin value'),
]
cmap = plt.get_cmap('managua_r').copy()
cmap.set_bad('black')
midpoint_rgba = cmap(.5)
midpoint_hex = to_hex(midpoint_rgba)
audits = []
for letter, col, half_range, _, _ in configs:
    norm = baseline_colour_norm(half_range)
    assert norm(0.) == .5
    np.testing.assert_array_equal(cmap(norm(0.)), midpoint_rgba)
    for (panel, trial), group in bins.groupby(['panel', 'trial']):
        values = group[col].to_numpy()
        mask = (group.bin_center_s.ge(-15) & group.bin_center_s.lt(0)).to_numpy()
        base = values[mask & np.isfinite(values)]
        input_missing = group.input_centred_log_bin.isna().to_numpy()
        if len(base):
            median = float(np.median(base))
            coordinate = float(np.median(np.asarray(norm(base))))
            assert abs(median) < 1e-12 and abs(coordinate-.5) < 1e-12
            assert np.array_equal(np.isnan(values), input_missing)
        else:
            median, coordinate = np.nan, np.nan
            assert np.isnan(values).all()
        audits.append({'option': letter, 'panel': panel, 'trial': int(trial),
                       'finite_baseline_bins': len(base), 'baseline_median': median,
                       'median_palette_coordinate_after_clipping': coordinate,
                       'baseline_reference_palette_coordinate': float(norm(0.)),
                       'baseline_reference_hex': midpoint_hex,
                       'NaN_bins': int(np.isnan(values).sum()),
                       'input_NaN_bins': int(input_missing.sum()),
                       'baseline_endpoint_fraction': float(np.mean(np.abs(base) >= half_range)) if len(base) else np.nan,
                       'status': 'centred' if len(base) else 'undefined quantile range'})
audit = pd.DataFrame(audits)
audit.to_csv(OUT/'all_options_trial_colour_audit.csv', index=False)
pd.DataFrame(quantiles).to_csv(OUT/'trial_quantile_ranges.csv', index=False)
pd.DataFrame(legacy_checks).to_csv(OUT/'previous_C_failure_audit.csv', index=False)
bins.to_parquet(OUT/'all_options_bins.parquet', index=False)

plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(3, 4, figsize=(18, 12), layout='constrained')
for row, spec in enumerate(FISH):
    panel, name, fish, *_ = spec
    b = parts[row]
    for column, (letter, col, half_range, title, label) in enumerate(configs):
        ax = axes[row, column]
        norm = baseline_colour_norm(half_range)
        matrix = b.pivot(index='trial', columns='bin_center_s', values=col).to_numpy()
        im = ax.imshow(matrix, aspect='auto', interpolation='nearest',
                       extent=(-20, 20, 94.5, 4.5), cmap=cmap, norm=norm)
        ax.set_title(f'{letter} · {panel} {name} {fish}\n{title}', fontsize=9)
        for t in [0, 10]:
            ax.axvline(t, color='#0d7f3c', lw=.7)
        for y in [14.5, 64.5]:
            ax.axhline(y, color='white', lw=.8)
        if spec[-1] is not None:
            ax.plot([spec[-1]]*2, [14.5, 64.5], color='#78358c', ls=':', lw=.8)
        ax.set_xlabel('Seconds from measured CS onset')
        ax.set_ylabel('Trial')
        cb = fig.colorbar(im, ax=ax, shrink=.72, label=label,
                          ticks=[-half_range, 0, half_range], extend='both')
        cb.ax.axhline(0, color='white', lw=.5)
fig.suptitle('F/G/H · every option uses each trial’s baseline [−15,0) median as zero\n'
             f'managua_r: zero → palette midpoint 0.5 → {midpoint_hex}; black = NaN. '
             'C corrected; literal historical affine scaling excluded.', fontsize=12)
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(OUT/f'all_options_centred_comparison.{ext}', dpi=180)
plt.close(fig)

for letter, col, half_range, title, label in configs:
    thumbs = []
    norm = baseline_colour_norm(half_range)
    for spec, b in zip(FISH, parts):
        panel, name, fish, *_ = spec
        fig = plt.figure(figsize=(6.1, 5.5))
        grid = fig.add_gridspec(3, 1, height_ratios=[10, 50, 30], left=.17, right=.78,
                               bottom=.16, top=.80, hspace=.10)
        matrix = b.pivot(index='trial', columns='bin_center_s', values=col)
        for k, (phase, first, last) in enumerate([('Pre-Train', 5, 14), ('Train', 15, 64), ('Test', 65, 94)]):
            ax = fig.add_subplot(grid[k])
            ax.set_facecolor('black')
            ax.pcolormesh(np.arange(-20, 20.5, .5), np.arange(first-.5, last+1.5),
                          np.ma.masked_invalid(matrix.reindex(range(first, last+1)).to_numpy()),
                          cmap=cmap, norm=norm, shading='flat')
            ax.set_ylim(last+.5, first-.5)
            ax.set_yticks([first, last])
            ax.set_ylabel(phase)
            ax.set_xlim(-20, 20)
            ax.set_xticks([-20, -10, 0, 10, 20])
            if k < 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Time from measured CS onset (s)')
            for t in [0, 10]:
                ax.axvline(t, color='#0d7f3c', lw=.8, ls='--' if t == 10 else '-')
            if phase == 'Train' and spec[-1] is not None:
                ax.axvline(spec[-1], color='#78358c', ls=':', lw=.8)
        fig.text(.04, .955, panel, fontsize=19, weight='bold')
        fig.text(.17, .955, f'{name} · {fish}', fontsize=13, weight='bold')
        fig.text(.17, .88, title, fontsize=9)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          cax=fig.add_axes([.825, .16, .023, .64]),
                          ticks=[-half_range, 0, half_range], extend='both')
        cb.set_label(label)
        fig.text(.17, .055, '0.5 s bins · trial baseline [−15,0) · black = NaN', fontsize=8)
        fig.text(.17, .026, 'Trial baseline median = 0 → managua_r midpoint', fontsize=8, color=midpoint_hex)
        stem = OUT/f'{letter}_Panel{panel}_{name.replace(" ", "")}'
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(stem.with_suffix('.'+ext), dpi=220)
        plt.close(fig)
        with Image.open(stem.with_suffix('.png')) as image:
            thumbs.append(ImageOps.contain(image.convert('RGB'), (900, 820)))
    overview = Image.new('RGB', (2700, 820), 'white')
    for i, image in enumerate(thumbs):
        overview.paste(image, (900*i, 0))
    overview.save(OUT/f'{letter}_F-G-H.png')

fig, ax = plt.subplots(figsize=(9, 1.8), layout='constrained')
ax.imshow(np.linspace(-1, 1, 1024)[None, :], cmap=cmap, norm=baseline_colour_norm(1),
          aspect='auto', extent=(-1, 1, 0, 1))
ax.set_xticks([-1, 0, 1], ['Below baseline', f'Trial baseline median = 0\n{midpoint_hex}', 'Above baseline'])
ax.set_yticks([])
ax.axvline(0, color='white', lw=1)
ax.set_title('Verified colour contract: managua_r(0.5) is the trial baseline reference')
fig.savefig(OUT/'baseline_colour_key.png', dpi=180)
plt.close(fig)

summary = audit.groupby('option').agg(valid_trials=('baseline_median', 'count'),
                                     maximum_median_error=('baseline_median', lambda s: float(s.abs().max())),
                                     NaN_bins=('NaN_bins', 'sum')).to_dict('index')
report = {'scope': 'F/G/H only; review assets', 'baseline_s': [-15, 0],
          'reference_statistic': 'median of finite displayed baseline bins, each trial separately',
          'palette': 'managua_r', 'matplotlib': matplotlib.__version__,
          'palette_midpoint_rgba': list(midpoint_rgba), 'palette_midpoint_hex': midpoint_hex,
          'normalization': 'CenteredNorm(vcenter=0, symmetric halfrange, clip=True)',
          'options': {'A': 'log units, ±0.25', 'B': f'log units, ±{shared_half_range}',
                      'C': '(x-P50)/((P90-P10)/2), clip [-1,1]',
                      'D': '(x-P50)/max(P50-P10,P90-P50), clip [-1,1]'},
          'previous_C': {'valid_trials': len(legacy_checks),
                         'failed_midpoint_trials': int((pd.DataFrame(legacy_checks).old_C_center_error.abs() > 1e-12).sum())},
          'validation': summary, 'source_data': sources,
          'code': [{'path': str(p), 'sha256': digest(p)} for p in
                   [Path(__file__), Path(__file__).with_name('baseline_colour_mapping.py')]],
          'outputs': [{'path': str(p), 'sha256': digest(p)} for p in OUT.iterdir()
                      if p.is_file() and p.name != 'colour_review_manifest.json']}
(OUT/'colour_review_manifest.json').write_text(json.dumps(report, indent=2))
print(json.dumps({'palette_midpoint': midpoint_hex, 'validation': summary, 'previous_C': report['previous_C']}, indent=2))
print(OUT)
