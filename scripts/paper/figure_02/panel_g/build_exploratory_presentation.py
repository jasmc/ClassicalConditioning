"""Separate, authenticated Figure 2 G presentation of saved exploratory results.

No fitting, cohort selection, scientific freeze or main-figure state changes.
"""
from __future__ import annotations

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]


import argparse
import base64
import html
import io
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = _COMMAND_REPOSITORY
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot, publish_state, state_lock
from classical_conditioning.figures.layout_frames import POLICY, COLLECTOR, embed_spec, assess
from classical_conditioning.analysis.inference.panel_g_run import sha

STATE = ROOT / 'docs/paper/figures/figure-02/panel-g/exploratory/current-state.json'
NS = '{http://www.w3.org/2000/svg}'
COLOURS = {'control': '#27aae1', 'delay': '#f00098'}
EFFECT = '#63518c'
SIZE = (183, 122)
OUTPUTS = ('figure2-G-exploratory.svg', 'review.html', 'legend.md', 'draft-registry.json')


def save_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def record(path):
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def check_records(records):
    for item in records:
        p = Path(item['path'])
        actual = sha(p) if p.is_file() else None
        if actual != item['sha256']:
            raise ValueError('Bound source/state changed; preserve and reconcile: ' + str(p))


def dependencies():
    from matplotlib import get_data_path
    fonts = Path(get_data_path()) / 'fonts/ttf'
    return [Path(__file__), POLICY, COLLECTOR, COLLECTOR.with_suffix('.py'),
            ROOT / 'src/classical_conditioning/figures/layout_frames.py',
            ROOT / 'src/classical_conditioning/figure_workspace.py',
            fonts / 'DejaVuSans.ttf', fonts / 'DejaVuSans-Bold.ttf']


def load_inputs(out):
    ticket = json.loads((out / 'start-ticket.json').read_text())
    check_records(ticket['protected'] + ticket['inputs'])
    paths = [Path(e['path']) for e in ticket['inputs']]
    observed = pd.read_parquet(next(p for p in paths if p.name == 'bootstrap-summary.parquet'))
    curve = pd.read_csv(next(p for p in paths if p.name == 'population-trajectory.csv'))
    phase = pd.read_csv(next(p for p in paths if p.name == 'planned-phase-contrasts.csv'))
    result = json.loads(next(p for p in paths if p.name == 'result.json').read_text())
    assert len(curve) == 90 and curve.trial_number.tolist() == list(range(5, 95))
    assert len(observed) == 180 and not observed.duplicated(['condition_id', 'trial_number']).any()
    assert phase.contrast.tolist() == ['late_training', 'early_testing', 'late_testing']
    assert result['bootstrap_successful'] == result['bootstrap_requested'] == 999
    for condition in COLOURS:
        d = observed.loc[observed.condition_id.eq(condition)].sort_values('trial_number')
        np.testing.assert_array_equal(d.trial_number, curve.trial_number)
        np.testing.assert_array_equal(d.contributing_fish, curve[condition + '_fish'])
    assert (curve.simultaneous_lower <= curve.estimate).all()
    assert (curve.simultaneous_upper >= curve.estimate).all()
    return ticket, observed, curve, phase, result


def star_decisions(curve, phase):
    """One binary mark from the saved joint interval, never a p-value tier."""
    def excludes_zero(frame):
        finite = np.isfinite(frame.simultaneous_lower) & np.isfinite(frame.simultaneous_upper)
        return finite & ((frame.simultaneous_lower > 0) | (frame.simultaneous_upper < 0))
    adequate = curve.coverage_adequate.eq(True)
    return (curve.loc[adequate & excludes_zero(curve), 'trial_number'].astype(int).tolist(),
            phase.loc[excludes_zero(phase), 'contrast'].tolist())


def star_ranges(trials):
    """Group only adjacent scheduled trials, preserving phase boundaries."""
    ranges = []
    for trial in sorted(set(trials)):
        if not ranges or trial != ranges[-1][1] + 1 or ranges[-1][1] in (14, 64):
            ranges.append([trial, trial])
        else:
            ranges[-1][1] = trial
    return ranges


def render(out, asterisks=False):
    ticket, observed, curve, phase, result = load_inputs(out)
    if asterisks != (ticket.get('presentation_variant') == 'asterisks'):
        raise ValueError('Render option must match the pre-edit ticket variant')
    marked_trials, marked_phases = star_decisions(curve, phase) if asterisks else ([], [])
    sources = [record(p) for p in dependencies() if p.is_file()]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'g-exploratory-20261016',
                         'axes.linewidth': .5, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7})
    fig = plt.figure(figsize=(SIZE[0]/25.4, SIZE[1]/25.4))
    fig.patch.set_gid('g-canvas-background')

    def axes(identity, x, y, w, h):
        ax = fig.add_axes([x/SIZE[0], 1-(y+h)/SIZE[1], w/SIZE[0], h/SIZE[1]])
        ax.set_gid(identity)
        ax.patch.set_gid(identity + '-data-rectangle')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out', length=2, width=.5, pad=3)
        return ax

    def text(identity, x, y, content, size=8, **kwargs):
        artist = fig.text(x/SIZE[0], 1-y/SIZE[1], content, fontsize=size, **kwargs)
        artist.set_gid(identity)

    obs = axes('g-observed', 18, 19, 100, 25)
    change = axes('g-change', 18, 63, 100, 25)
    coverage = axes('g-coverage', 18, 103, 100, 6)
    effects = axes('g-phase-effects', 149, 26, 29, 33)

    text('g-letter', 2, 4.8, 'G', 10, weight='bold')
    text('g-status', 180, 4.8, 'Exploratory', 7, ha='right', color='#555555')
    text('g-observed-heading', 18, 11.5, 'Observed bout vigor', weight='bold')
    text('g-change-heading', 18, 56.5, 'Adjusted condition change', weight='bold')
    text('g-phase-heading', 124, 12, 'Planned phase changes', weight='bold')
    text('g-coverage-heading', 18, 99.5, 'Contributing fish', 7)

    for ax in (obs, change, coverage):
        ax.set_xlim(4, 95)
        ax.set_xticks([5, 15, 35, 55, 65, 85, 94])
        for boundary in (14.5, 64.5):
            ax.axvline(boundary, color='#aaaaaa', linestyle=':', linewidth=.6, zorder=0)
    for x, name in [(9.5, 'Pre'), (39.5, 'Training'), (79.5, 'Testing')]:
        obs.text(x, 1.07, name, transform=obs.get_xaxis_transform(), ha='center', fontsize=7)
    for condition, colour in COLOURS.items():
        d = observed.loc[observed.condition_id.eq(condition)].sort_values('trial_number')
        obs.fill_between(d.trial_number, d.ci_lower, d.ci_upper, color=colour, alpha=.18,
                         linewidth=0).set_gid('g-observed-' + condition + '-pointwise-band')
        obs.plot(d.trial_number, d['median'], color=colour, linewidth=1.2,
                 label='Control' if condition == 'control' else 'Delay')[0].set_gid('g-observed-' + condition + '-median')
        coverage.plot(curve.trial_number, curve[condition + '_fish'], color=colour, linewidth=.75)[0].set_gid('g-count-' + condition)
    obs.axhline(0, color='#888888', linewidth=.5, zorder=0)
    obs.set_ylim(float(np.floor(observed.ci_lower.min()/.05)*.05-.01),
                 float(np.ceil(observed.ci_upper.max()/.05)*.05+.01))
    obs.set_yticks([-.2, -.1, 0, .1])
    obs.set_ylabel('Median ln vigor\n(response − baseline)')
    obs.tick_params(labelbottom=False, bottom=False)
    key = obs.legend(loc='lower right', frameon=False, fontsize=7, ncol=2,
                     handlelength=1.4, columnspacing=.9, borderaxespad=.3)
    key.set_gid('g-condition-key')
    for name, d in curve.groupby('fit_phase', sort=False):
        change.fill_between(d.trial_number, d.simultaneous_lower, d.simultaneous_upper,
                            color=EFFECT, alpha=.18, linewidth=0).set_gid('g-adjusted-' + name + '-joint-band')
        change.plot(d.trial_number, d.estimate, color=EFFECT, linewidth=1.2)[0].set_gid('g-adjusted-' + name + '-estimate')
    change.axhline(0, color='#555555', linewidth=.6, zorder=0)
    change.set_ylim(-.13, .25)
    change.set_yticks([-.1, 0, .1, .2])
    change.set_ylabel('Change from Pre\n(control − Delay)')
    change.tick_params(labelbottom=False, bottom=False)
    if asterisks:
        change.text(4, -.19, 'Joint CI', transform=change.get_xaxis_transform(),
                    ha='left', va='center', fontsize=6).set_gid('g-joint-ci-row-label')
        for start, end in star_ranges(marked_trials):
            identity = f'g-joint-ci-range-{start}-{end}'
            if start != end:
                change.plot([start, end], [-.19, -.19], transform=change.get_xaxis_transform(),
                            color='black', linewidth=.5, clip_on=False)[0].set_gid(identity + '-bar')
                for side, trial in [('start', start), ('end', end)]:
                    change.plot([trial, trial], [-.15, -.23], transform=change.get_xaxis_transform(),
                                color='black', linewidth=.5, clip_on=False)[0].set_gid(identity + '-' + side)
            change.text((start + end)/2, -.09, '*', transform=change.get_xaxis_transform(),
                        ha='center', va='center', fontsize=6, color='black',
                        clip_on=False).set_gid(identity + '-asterisk')
    coverage.set_ylim(0, 31)
    coverage.set_yticks([0, 30])
    coverage.set_xlabel('Global CS trial')
    effects.axvline(0, color='#999999', linewidth=.6, zorder=0)
    for i, row in phase.iterrows():
        effects.errorbar(row.estimate, 2-i,
                         xerr=[[row.estimate-row.simultaneous_lower], [row.simultaneous_upper-row.estimate]],
                         fmt='o', color=EFFECT, markersize=3.4, capsize=2, elinewidth=.9,
                         capthick=.7).lines[0].set_gid('g-phase-' + row.contrast)
        if asterisks and row.contrast in marked_phases:
            effects.text(row.simultaneous_upper + .012, 2-i, '*', ha='center',
                         va='center', fontsize=6, color='black', clip_on=False).set_gid('g-joint-ci-star-phase-' + row.contrast)
    effects.set(xlim=(-.1, .2), ylim=(-.55, 2.55), yticks=[2, 1, 0],
                yticklabels=['Late\ntraining', 'Early\ntesting', 'Late\ntesting'], xticks=[-.1, 0, .1, .2])
    effects.set_xlabel('Change from Pre')
    effects.tick_params(axis='y', length=0, pad=4)
    effects.spines['left'].set_visible(False)
    text('g-direction', 124, 76, 'Positive = greater\nsuppression in Delay', 7, linespacing=1.4)
    text('g-acquisition', 124, 93, 'Acquisition timing\nunresolved', 8, weight='bold', linespacing=1.5)
    text('g-uncertainty-note', 124, 109, 'Model bands remain\nprovisional', 7, color='#555555', linespacing=1.4)
    if asterisks:
        text('g-star-key', 124, 119, '* Joint interval excludes 0', 6)

    stream = io.BytesIO()
    fig.savefig(stream, format='svg', metadata={'Date': None})
    plt.close(fig)
    root = ET.fromstring(stream.getvalue())
    group = root.find(NS + "g[@id='figure_1']")
    owners = {
        'g-header-owner': ['g-letter', 'g-status'],
        'g-observed-owner': ['g-observed', 'g-observed-heading'],
        'g-change-owner': ['g-change', 'g-change-heading'],
        'g-coverage-owner': ['g-coverage', 'g-coverage-heading'],
        'g-effects-owner': ['g-phase-effects', 'g-phase-heading', 'g-direction', 'g-acquisition', 'g-uncertainty-note'],
    }
    if asterisks:
        owners['g-effects-owner'].append('g-star-key')
    for owner, ids in owners.items():
        container = ET.SubElement(group, NS + 'g', {'id': owner})
        for identity in ids:
            node = group.find(NS + "*[@id='" + identity + "']")
            if node is None: raise ValueError('Missing owned artist: ' + identity)
            group.remove(node); container.append(node)
    font_css = ''
    for weight, file in [(400, 'DejaVuSans.ttf'), (700, 'DejaVuSans-Bold.ttf')]:
        font = next(p for p in dependencies() if p.name == file)
        font_css += "@font-face{font-family:'DejaVu Sans';font-weight:" + str(weight) + ";src:url(data:font/ttf;base64," + base64.b64encode(font.read_bytes()).decode() + ") format('truetype');}\n"
    style = ET.SubElement(root.find(NS + 'defs'), NS + 'style', {'type': 'text/css'})
    style.text = font_css
    mapping = {'observed': {'source': ticket['inputs'][0], 'summary': 'median across fish',
                           'x': 'trial_number', 'y': 'median', 'band': ['ci_lower', 'ci_upper'],
                           'interval': 'pointwise percentile, 5000 whole-fish descriptive resamples, seed 10'},
               'adjusted': {'source': ticket['inputs'][1], 'x': 'trial_number', 'y': 'estimate',
                            'band': ['simultaneous_lower', 'simultaneous_upper'],
                            'estimand': 'control-minus-Delay change relative to average Pre5-14 at common baseline'},
               'phase': {'source': ticket['inputs'][2], 'rows': phase.contrast.tolist(),
                         'windows': [[55, 64], [65, 74], [85, 94]],
                         'interval': 'same joint family as trial curve'},
               'coverage': {'source': ticket['inputs'][1], 'fields': ['control_fish', 'delay_fish'],
                            'definition': 'fish with eligible paired LogMedian outcome at each scheduled trial'}}
    if asterisks:
        mapping['asterisks'] = {'trial_source': ticket['inputs'][1], 'phase_source': ticket['inputs'][2],
                               'criterion': 'finite nominal 95% simultaneous interval excludes zero; trial coverage adequate',
                               'trial_numbers': marked_trials, 'phase_contrasts': marked_phases,
                               'display_ranges': star_ranges(marked_trials),
                               'range_meaning': 'brackets group individual qualifying scheduled trials; no pooled-range test or onset interval',
                               'multiplicity_family': '90 trial changes plus 3 planned phase changes',
                               'symbol': '*', 'binary_only': True, 'BH_p_values': False,
                               'onset_estimator': False, 'calibration_approved': False}
    metadata = ET.SubElement(root, NS + 'metadata', {'id': 'scientific-source-mapping'})
    metadata.text = json.dumps(mapping, separators=(',', ':'))
    allocations = [
        ('header', 'shared', [0, 0, 183, 7], 'g-header-owner'),
        ('observed', 'panel', [0, 7, 121, 45], 'g-observed-owner'),
        ('change', 'panel', [0, 52, 121, 44], 'g-change-owner'),
        ('coverage', 'panel', [0, 96, 121, 26], 'g-coverage-owner'),
        ('effects', 'panel', [121, 7, 62, 115], 'g-effects-owner')]
    spec = {'scope': 'figure', 'final_size_mm': list(SIZE), 'inset_mm': 1.,
            'allocations': [{'id': i, 'kind': kind, 'box_mm': box, 'group_ids': [owner]} for i, kind, box, owner in allocations],
            'background_ids': ['g-canvas-background', *[x + '-data-rectangle' for x in ['g-observed', 'g-change', 'g-coverage', 'g-phase-effects']]],
            'scientific_clips': [{'id': n.get('id'), 'evidence': 'Exact generated data-axis clip; no data marks outside axis limits'} for n in root.iter(NS + 'clipPath')],
            'data_axes': [{'id': name, 'element_ids': [identity + '-data-rectangle']} for name, identity in [('observed', 'g-observed'), ('change', 'g-change'), ('coverage', 'g-coverage')]],
            'alignments': [{'target': 'data_axes', 'property': prop, 'ids': ['observed', 'change', 'coverage']} for prop in ['left', 'right', 'width']]}
    svg = embed_spec(ET.tostring(root), spec)
    legend = """**G | Exploratory presentation of population change in the Delay experiment.**

Control, 28 fish; Delay, 29 fish. The cohort is unchanged. Of 5,130 scheduled fish/trial outcomes (CS5–94), 4,811 are eligible. Observed curves show medians across fish of response-minus-baseline LogMedian vigor. Each window uses the median natural log of positive valid detected-bout frame vigor: baseline [−15, 0) s and response [0, 9) s. Shading is the original pointwise percentile 95% interval from 5,000 whole-fish trajectory resamples within condition (seed 10). These descriptive medians differ from model means.

The adjusted curve models response LogMedian with baseline LogMedian as a covariate, condition-specific phase terms (linear Pre and separate cubic Train/Test splines) and fish random intercept/slope. It shows control-minus-Delay change from average Pre trials 5–14 at a common baseline; positive values indicate greater conditioned-direction suppression in Delay. The curve is segmented at phase boundaries. Shading and phase-effect error bars are nominal 95% bootstrap-t simultaneous intervals across one family of 90 scheduled trial contrasts and three planned phase averages, from 999 accepted whole-fish model refits. Phase windows are late training 55–64, early testing 65–74 and late testing 85–94. The lower strip reports contributing fish for each condition, without changing cohort membership or compressing scheduled time. An empty bout window has undefined conditional vigor, rather than zero activity.

This is an unfrozen exploratory version. Model bands remain provisional: shape simulations show inadequate full-curve coverage for abrupt/transient learning. No significance stars or acquisition marker are drawn. At the unapproved candidate 10% threshold sustained for three scheduled training trials, acquisition has no finite latest bound. The figure does not establish precise acquisition, extinction or an overall movement effect. The complete D/M/R and BH trial-test results remain preserved with the original G; they are not recomputed here.
"""
    if asterisks:
        range_text = ', '.join(f'CS{a}–{b}' if a != b else f'CS{a}' for a, b in star_ranges(marked_trials)) or 'no scheduled trials'
        phase_text = ', '.join(name.replace('_', ' ') for name in marked_phases) or 'no planned phases'
        legend = legend.replace('No significance stars or acquisition marker are drawn.',
            'An asterisk denotes a nominal 95% simultaneous interval excluding zero; trial marks also require adequate outcome coverage. Brackets group consecutive qualifying trial contrasts: each bracket’s asterisk applies to every included trial, not a pooled-range test or an onset interval. Qualifying results are ' + range_text + ' and ' + phase_text + '. The same binary criterion applies to the phase-effect asterisk. Asterisks are not BH-adjusted p-values or p<0.01/p<0.001 tiers. Unmarked estimates need not be zero. The bracket’s beginning is not an acquisition estimate, and no acquisition marker is drawn.')
    paragraphs = ''.join('<p>' + html.escape(p).replace('**', '') + '</p>' for p in legend.strip().split('\n\n'))
    page = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exploratory Figure 2 G</title><style>body{margin:20px auto;max-width:850px;padding:0 18px;background:white;color:#161616}svg{display:block;width:100%;height:auto}figcaption{font:14px/1.55 Georgia,serif}figure{margin:0}@media print{body{margin:0;padding:0;max-width:none}svg{width:183mm;height:122mm}}</style></head><body><figure>' + svg.decode() + '<figcaption>' + paragraphs + '</figcaption></figure></body></html>'
    check_records(ticket['protected'] + ticket['inputs']); check_records(sources)
    with state_lock(out / 'render'):
        for name in OUTPUTS:
            if (out / name).exists(): raise FileExistsError('Preserve previous candidate: ' + str(out / name))
        (out / OUTPUTS[0]).write_bytes(svg)
        (out / 'review.html').write_text(page)
        (out / 'legend.md').write_text(legend)
        save_json(out / 'draft-registry.json', {'scope': 'exploratory G only', 'scientific_status': 'unfrozen; no approved primary model or acquisition threshold', 'data_mappings': mapping, 'owners': owners, 'layout': spec, 'sources': sources, 'previous_versions_preserved': True})
        if asterisks:
            save_json(out / 'asterisk-decisions.json', mapping['asterisks'])
        save_json(out / 'render-outputs.json', [record(out / name) for name in OUTPUTS])
        save_json(out / 'render-sources.json', sources)
        snapshots = out / 'source-snapshots'; snapshots.mkdir()
        for i, item in enumerate(sources):
            (snapshots / f'{i:02d}-{Path(item["path"]).name}').write_bytes(Path(item['path']).read_bytes())
    print(out / 'review.html')


def publish(out):
    ticket = json.loads((out / 'start-ticket.json').read_text())
    check_records(ticket['protected'] + ticket['inputs'])
    check_records(json.loads((out / 'render-sources.json').read_text()))
    check_records(json.loads((out / 'render-outputs.json').read_text()))
    layout = json.loads((out / 'layout-evidence.json').read_text())
    actual = assess((out / OUTPUTS[0]).read_bytes(), layout['browser'])
    if actual['status'] != 'passed' or actual['artifact_sha256'] != layout['artifact_sha256']:
        raise ValueError('Exact-artifact layout evidence must pass')
    visual = json.loads((out / 'visual-review.json').read_text())
    if visual['svg_sha256'] != sha(out / OUTPUTS[0]) or not visual['passed']:
        raise ValueError('Bind actual visual review before publication')
    STATE.parent.mkdir(parents=True, exist_ok=True)
    extras = [out / n for n in ['asterisk-decisions.json', 'numerical-render-check.json', 'previous-exploratory-state.json'] if (out / n).is_file()]
    bound = [STATE, out / 'start-ticket.json', out / 'render-outputs.json', out / 'render-sources.json', out / 'layout-evidence.json', out / 'visual-review.json', *extras, *[out / n for n in OUTPUTS], *[Path(e['path']) for e in ticket['protected'] + ticket['inputs']], *[Path(e['path']) for e in json.loads((out / 'render-sources.json').read_text())]]
    snapshot = Snapshot([*command_sources(__file__), *(bound)])
    if STATE.exists():
        if not ticket.get('state_baseline'):
            raise FileExistsError('Existing exploratory state requires a new pre-edit ticket')
        check_records([ticket['state_baseline']])
        snapshot.hashes[str(STATE.resolve())] = ticket['state_baseline']['sha256']
    state = {'schema_version': 1, 'figure_id': 'fig2', 'panel_id': 'g', 'scope': 'separate exploratory presentation',
             'default_display': 'draft', 'default_display_format': 'html-svg-and-legend',
             'scientific_status': 'unfrozen exploratory; main Figure 2 and accepted G unchanged',
             'draft': {'working_svg': record(out / OUTPUTS[0]), 'review': record(out / 'review.html'), 'legend': {**record(out / 'legend.md'), 'panel_ids': ['G']}, 'semantic_registry': record(out / 'draft-registry.json'), 'layout_frames': record(out / 'layout-evidence.json'), 'visual_review': record(out / 'visual-review.json')},
             'parent_analysis': ticket['parent_stage'], 'starting_ticket': record(out / 'start-ticket.json'),
             'scientific_coverage': 'Explicit plotted source/estimand mappings; model calibration remains unresolved; no freeze'}
    if ticket.get('presentation_variant') == 'asterisks':
        state['presentation_variant'] = 'asterisks from joint interval exclusion'
        state['draft']['asterisk_decisions'] = record(out / 'asterisk-decisions.json')
        state['previous_versions'] = [{'variant': 'without asterisks', 'draft': ticket['previous_state']['draft'],
                                       'state_snapshot': record(out / 'previous-exploratory-state.json')}]
    with state_lock(out / 'publication'):
        check_records(ticket['protected'] + ticket['inputs'])
        publish_state(STATE, state, snapshot)
    print(STATE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['render', 'publish'])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--asterisks', action='store_true', help='Additional variant using binary joint-interval exclusion marks')
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to('/Volumes/JOAQUIM') or not Path('/Volumes/JOAQUIM').is_mount():
        raise ValueError('Use mounted JOAQUIM and a new explicit destination')
    if args.stage == 'render': render(args.output_dir.resolve(), asterisks=args.asterisks)
    else: publish(args.output_dir.resolve())
