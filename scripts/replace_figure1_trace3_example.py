"""Replace only the active Figure 1 H example, with guarded draft publication.

The historical F/G/H freeze remains a separate immutable scientific selection.
Preparation creates temporary review files; publication updates the sole active
SVG/HTML pair through figure_workspace.publish_state after source revalidation.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
from datetime import datetime, timezone
import io
import html as html_module
import json
from pathlib import Path
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import matplotlib as mpl
import numpy as np
import pandas as pd
from PIL import Image
from classical_conditioning.figure_workspace import Snapshot, publish_state, sha, tag, measured_presentation
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme
from classical_conditioning.figures.layout_frames import assess


def svg_editing_page(svg, **kwargs):
    # Reuse the supported display helper without importing the builder's
    # unrelated D/E scientific dependencies at module import on this host.
    source = ROOT / 'scripts/build_figure_legend_review.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('legend_html', 'svg_editing_page')]
    if len(functions) != 2: raise ValueError('Review helper interface changed')
    namespace = dict(ET=ET, tag=tag, html=html_module, re=re)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['svg_editing_page'](svg, **kwargs)

STATE = ROOT / 'docs/paper/figures/figure-01/current-state.json'
LEGEND = ROOT / 'docs/paper/figures/figure-01/legend.md'
REVIEW = Path(r'J:\ClassicalConditioning Outputs\current\reviews')
OUTPUT = REVIEW / 'figure-01-trace3-learner-selection-20261019'
CANDIDATE_REVIEW = OUTPUT / 'selection-report.json'
DATA = REVIEW / 'figure-02-3strace/fish'
TICKET = OUTPUT / 'replacement-start-ticket.json'
FISH, PREVIOUS = '20230802_08', '20230307_12'
PREFIX = 'edit-FGH-fig1__g__'
IMAGE = PREFIX + 'heatmap__vigor-cells'
HREF = '{http://www.w3.org/1999/xlink}href'
EVIDENCE = OUTPUT / 'figure-01-H-replacement-20261020.json'
MEASURES = OUTPUT / 'figure-01-H-effective-presentation-20261020.json'
BROWSER = OUTPUT / 'figure-01-H-browser-20261020.json'
LAYOUT = OUTPUT / 'figure-01-H-layout-20261020.json'
BACKUP = OUTPUT / 'figure-01-before-H-replacement-20261020.zip'
README = ROOT / 'docs/paper/figures/figure-01/README.md'


def configure(args):
    """Keep historical defaults while allowing a new explicit source-bound swap."""
    global OUTPUT, CANDIDATE_REVIEW, TICKET, FISH, PREVIOUS
    global EVIDENCE, MEASURES, BROWSER, LAYOUT, BACKUP
    FISH, PREVIOUS = args.fish, args.previous_fish
    OUTPUT, TICKET, CANDIDATE_REVIEW = args.output_dir.resolve(), args.ticket.resolve(), args.candidate_review.resolve()
    EVIDENCE = OUTPUT / 'figure-01-H-replacement-20261020.json'
    MEASURES = OUTPUT / 'figure-01-H-effective-presentation-20261020.json'
    BROWSER = OUTPUT / 'figure-01-H-browser-20261020.json'
    LAYOUT = OUTPUT / 'figure-01-H-layout-20261020.json'
    BACKUP = OUTPUT / 'figure-01-before-H-replacement-20261020.zip'


def request_parameters():
    return dict(fish=FISH, previous_fish=PREVIOUS, output_dir=str(OUTPUT), ticket=str(TICKET),
                candidate_review=str(CANDIDATE_REVIEW))


def encode(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relocated(path):
    """Resolve the recorded Mac checkout/JOAQUIM roots on this Windows host."""
    text = str(path).replace('\\', '/')
    if '/Volumes/JOAQUIM/' in text:
        return Path('J:/' + text.split('/Volumes/JOAQUIM/', 1)[1])
    if 'ClassicalConditioning/' in text and not Path(path).is_file():
        return ROOT / text.split('ClassicalConditioning/', 1)[1]
    return Path(path)


def bound(path, expected):
    actual = relocated(path)
    data = actual.read_bytes()
    # Git's Windows checkout changes LF to CRLF. Accept only this exact byte
    # conversion; never substitute a historical blob for changed working text.
    if sha(data) != expected and sha(data.replace(b'\r\n', b'\n')) != expected:
        raise ValueError('Bound source changed: ' + str(actual))
    return actual


def check_ticket(ignore=()):
    ticket = load(TICKET)
    if ticket['figure_id'] != 'fig1':
        raise ValueError('Wrong edit ticket scope')
    ignored = {str(Path(path).resolve()) for path in ignore}
    paths = []
    for path, expected in ticket['starting_hashes'].items():
        actual = relocated(path)
        if str(actual.resolve()) not in ignored:
            bound(path, expected)
        paths.append(actual)
    return ticket, paths


class EditSnapshot(Snapshot):
    def check(self, ignore=()):
        check_ticket(ignore=ignore)
        super().check(ignore=ignore)


def matrix(path):
    table = pd.read_parquet(path)
    values = table.pivot(index='display_trial', columns='bin_index', values='delta_log_vigor').sort_index().to_numpy()
    if values.shape != (90, 40):
        raise ValueError('Expected 90 trials by 40 direct one-second bins')
    if not np.array_equal(np.sort(table.display_trial.unique()), np.arange(1, 91)):
        raise ValueError('Unexpected trial indexing')
    return values


def ref(path, data=None):
    return dict(path=str(path), sha256=sha(path.read_bytes() if data is None else data))


def existing_sampling(old_image, old_values, colourmap):
    """Recover and verify exact nearest-cell pixel ownership from the old image.

    Matplotlib's SVG PNG is vertically reversed by the existing SVG transform.
    Keep its original 476 x 691 sampling, including fractional pixel boundaries.
    """
    height, width = old_image.shape[:2]
    colours = colourmap(mpl.colors.Normalize(-.25, .25)(old_values[::-1]), bytes=True)
    ys = np.floor((np.arange(90) + .5) * height / 90).astype(int)
    xs = np.floor((np.arange(40) + .5) * width / 40).astype(int)
    column_scores = np.all(old_image[ys, :, None, :] == colours[:, None, :, :], axis=3).sum(axis=0)
    columns = column_scores.argmax(axis=1)
    row_scores = np.all(old_image[:, xs, None, :] == colours.transpose(1, 0, 2)[None, :, :, :], axis=3).sum(axis=1)
    rows = row_scores.argmax(axis=1)
    if not (np.all(np.diff(columns) >= 0) and np.all(np.diff(rows) >= 0)):
        raise ValueError('Image sampling is not monotonic nearest-cell ownership')
    if not np.array_equal(old_image, colours[rows[:, None], columns[None, :]]):
        raise ValueError('Existing raster does not exactly match its source bins and palette')
    return rows, columns


def stripped(root):
    """Canonical artwork with only the authorized changes removed."""
    root = copy.deepcopy(root)
    for node in list(root):
        if node.get('id') in ('editable-copy-provenance', 'draft-trace3-example-selection'):
            root.remove(node)
    image = next(n for n in root.iter() if n.get('id') == IMAGE)
    image.set(HREF, 'AUTHORIZED-HEATMAP-PAYLOAD')
    labels = [n for n in root.iter() if n.tag == tag('text') and n.text in (FISH, PREVIOUS)]
    if len(labels) != 1:
        raise ValueError('Expected one scoped fish label')
    labels[0].text = 'AUTHORIZED-FISH-LABEL'
    return ET.tostring(root)


def prepare(work):
    ticket, ticket_paths = check_ticket()
    state = load(STATE)
    svg_path = bound(state['draft']['working_svg']['path'], state['draft']['working_svg']['sha256'])
    html_path = bound(state['draft']['review']['path'], state['draft']['review']['sha256'])
    bound(state['draft']['legend']['path'], state['draft']['legend']['sha256'])
    new_record, new_bins = DATA / (FISH + '.json'), DATA / (FISH + '-bins.parquet')
    old_bins = DATA / (PREVIOUS + '-bins.parquet')
    profile = load_presentation_profile('nature-main')
    paths = [*ticket_paths, TICKET, STATE, LEGEND, README, svg_path, html_path, new_record, new_bins, old_bins,
             CANDIDATE_REVIEW, Path(__file__),
             ROOT / 'scripts/build_figure_legend_review.py', ROOT / 'src/classical_conditioning/figure_workspace.py',
             ROOT / 'scripts/measure_figure1_trace3_replacement.cjs',
             ROOT / 'src/classical_conditioning/figures/layout_frames.py',
             ROOT / 'src/classical_conditioning/figures/layout_frames_browser.js',
             ROOT / 'configs/paper-figures/nature-main.json', ROOT / 'configs/paper-figures/figure-elements.json',
             ROOT / 'configs/paper-figures/layout-frames.json']
    snapshot = EditSnapshot(paths)
    snapshot.check()
    record = load(new_record)
    if record['recording_id'] != FISH or record['condition'] != 'trace' or record['heatmap'] != 'complete':
        raise ValueError('Wrong candidate source')
    output_hash = next(o['sha256'] for o in record['outputs'] if o['path'].endswith(FISH + '-bins.parquet'))
    bound(new_bins, output_hash)
    report = load(CANDIDATE_REVIEW)
    # Every candidate source is also independently bound to the earlier review.
    def find_hash(obj, key):
        if isinstance(obj, dict):
            if key in obj and isinstance(obj[key], str): return obj[key]
            for value in obj.values():
                found = find_hash(value, key)
                if found: return found
        return None
    for path in (new_record, new_bins, old_bins):
        expected = find_hash(report, str(path))
        if expected is None: raise ValueError('Candidate source absent from prior review: ' + str(path))
        bound(path, expected)
    values, old_values = matrix(new_bins), matrix(old_bins)
    raw_svg = svg_path.read_bytes()
    root = ET.fromstring(raw_svg)
    image = next(n for n in root.iter() if n.get('id') == IMAGE)
    png = base64.b64decode(image.get(HREF).split(',', 1)[1])
    original = np.asarray(Image.open(io.BytesIO(png)).convert('RGBA'))
    with mpl.rc_context():
        theme = apply_theme(profile=profile)
        if theme.intensity_cmap != 'managua_r':
            raise ValueError('Profile/current heatmap palette conflict')
        cmap = mpl.colormaps[theme.intensity_cmap].copy()
        cmap.set_bad(theme.heatmap_bad_color)
        rows, columns = existing_sampling(original, old_values, cmap)
        pixels = cmap(mpl.colors.Normalize(-.25, .25)(values[::-1]), bytes=True)[rows[:, None], columns[None, :]]
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format='PNG')
    image.set(HREF, 'data:image/png;base64,\n' + base64.b64encode(buffer.getvalue()).decode())
    labels = [n for n in root.iter() if n.tag == tag('text') and n.text == PREVIOUS]
    if len(labels) != 1: raise ValueError('Expected one previous fish label')
    labels[0].text = FISH
    selection = dict(schema_version=1, status='unfrozen author-selected scientific draft example',
        author_request=f'Replace the current 3sTrace example with the suggested {FISH} fish',
        figure_id='fig1', display_panel='H', source_identifier_panel='g', owner_group='layout-G',
        fish_id=FISH, previous_fish_id=PREVIOUS, condition='3sTrace', data=ref(new_bins),
        fish_record=ref(new_record), candidate_review=ref(CANDIDATE_REVIEW),
        predecessor_selection_evidence=state['draft'].get('trace3_example_selection'),
        source_provenance=record['heatmap_provenance'],
        scientific_recipe=dict(bin_width_s=1, frame_transform='natural log',
            bin_estimator='median of eligible original log frames', baseline_s=[-15, 0],
            baseline_estimator='median of finite baseline-bin medians', centre='bin log vigor minus trial baseline',
            scale='none', colour_limits=[-.25, .25], cmap=theme.intensity_cmap, missing='black',
            trial_rows='acquisition trials 5-94; display rows 1-90', nominal_expected_us_s=13,
            actual_us_s=record['heatmap_provenance']['expected_us_s']),
        raster=dict(shape=list(pixels.shape), vertical_order='reversed; existing SVG vertical transform retained',
            source_pixel_ownership_verified_exact=True, rgba_sha256=sha(pixels.tobytes())),
        generator=ref(Path(__file__)), presentation_profile_inputs=profile.inputs,
        prior_working_svg=ref(svg_path, raw_svg), whole_figure_frozen=False)
    meta = next(n for n in root if n.get('id') == 'editable-copy-provenance')
    provenance = json.loads(meta.text)
    registry = provenance['scientific_registry']
    for key, item in registry.items():
        if not key.startswith(PREFIX): continue
        context = item.setdefault('scientific_context', {})
        if context.get('fish_id') == PREVIOUS: context['fish_id'] = FISH
        context['active_example_source'] = ref(new_bins)
        item['display_panel_id'] = 'H'
        item['active_selection_status'] = 'unfrozen replacement; historical freeze is a predecessor'
        if 'source_frozen_svg' in item:
            item['inherited_geometry_source'] = item.pop('source_frozen_svg')
        if 'source_frozen_element_id' in item:
            item['inherited_geometry_element_id'] = item.pop('source_frozen_element_id')
        if key.endswith('heading__fish-id'):
            for field in ('text', 'rendered_text', 'source_text'):
                if field in item: item[field] = FISH
            if isinstance(item.get('geometry'), dict) and 'text' in item['geometry']:
                item['geometry']['text'] = FISH
        if key == IMAGE:
            context['data_mapping'] = 'Hash-authenticated per-fish cadence-reconstructed direct-log-bin cache; see draft-trace3-example-selection metadata'
            context['data_fields'] = ['acquisition_trial', 'display_trial', 'bin_index', 'bin_log_vigor', 'baseline_log_vigor', 'delta_log_vigor', 'eligible_frames']
            item['style_verification'] = dict(raster_content=dict(status='passed', evidence='All rendered cell colours and exact pixel ownership checked against authenticated 90x40 matrix; draft selection metadata binds inputs'))
            item['classification_evidence'] = ['Existing heatmap role retained; new authenticated source bins explicitly bound by replace_figure1_trace3_example.py']
    provenance['draft_trace3_example_selection'] = selection
    meta.text = json.dumps(provenance, ensure_ascii=False)
    nodes = [n for n in root if n.get('id') == 'draft-trace3-example-selection']
    if len(nodes) > 1: raise ValueError('Duplicate draft example selection metadata')
    node = nodes[0] if nodes else ET.SubElement(root, tag('metadata'), {'id': 'draft-trace3-example-selection'})
    node.text = json.dumps(selection, ensure_ascii=False)
    if stripped(root) != stripped(ET.fromstring(raw_svg)):
        raise ValueError('Artwork outside the authorized raster/label changed')
    new_svg = ET.tostring(root, encoding='utf-8')
    legend = LEGEND.read_text(encoding='utf-8')
    if legend.count('3sTrace (' + PREVIOUS + ')') != 1:
        raise ValueError('Expected one matching 3sTrace legend reference')
    legend = legend.replace('3sTrace (' + PREVIOUS + ')', '3sTrace (' + FISH + ')')
    readme = README.read_text(encoding='utf-8')
    if '**' + PREVIOUS + '**' in readme:
        readme = readme.replace('**' + PREVIOUS + '**', '**' + FISH + '**')
    html = svg_editing_page(new_svg, legend=legend).replace('main>svg{width:183mm;', 'main>svg{width:136mm;').encode('utf-8')
    evidence = dict(schema_version=1, scope='Figure 1 H only; unfrozen example replacement', selection=selection,
        predecessor_svg_sha256=sha(raw_svg), artifact_sha256=sha(new_svg), legend_sha256=sha(legend.encode()),
        checks=dict(all_other_artwork_identical=True, original_pixel_ownership_reproduced_exactly=True,
            shape=[90, 40], finite_cells=int(np.isfinite(values).sum()),
            trial_baselines_defined=int(np.any(np.isfinite(values), axis=1).sum()),
            expected_us_guide_unchanged_nominal_13s=True, historical_freeze_metadata_identical=True),
        coverage='Only requested example, source mappings and legend. Existing unresolved whole-figure production checks remain open.',
        starting_ticket=ref(TICKET), request=request_parameters(), bound_inputs=snapshot.hashes)
    measures = dict(artifact_sha256=sha(new_svg), scope='Figure 1 current draft',
        measured_presentation=measured_presentation(new_svg),
        scientific_replacement_evidence=str(EVIDENCE), inherited_geometry_preserved=True)
    snapshot.check()
    work.mkdir(parents=True, exist_ok=True)
    for name, data in [('candidate.svg', new_svg), ('candidate.html', html), ('legend.md', legend.encode()), ('README.md', readme.encode()),
                       ('evidence.json', encode(evidence)), ('measurements.json', encode(measures))]:
        with (work / name).open('xb') as stream: stream.write(data)
    manifest = dict(request=request_parameters(), inputs=snapshot.hashes, svg_path=str(svg_path), html_path=str(html_path),
                    staged={p.name:sha(p.read_bytes()) for p in work.iterdir() if p.is_file()})
    (work / 'manifest.json').write_bytes(encode(manifest))
    print(json.dumps(dict(stage=str(work), svg_sha256=sha(new_svg), finite_cells=int(np.isfinite(values).sum()))))


def publish(work):
    manifest = load(work / 'manifest.json')
    if manifest.get('request') != request_parameters():
        raise ValueError('Publication arguments differ from the prepared request')
    snapshot = EditSnapshot([])
    snapshot.hashes = manifest['inputs']
    snapshot.check()
    for name, expected in manifest['staged'].items(): bound(work / name, expected)
    svg, html = (work / 'candidate.svg').read_bytes(), (work / 'candidate.html').read_bytes()
    browser = load(work / 'browser.json')
    if browser['artifact_sha256'] != sha(svg) or browser['issues']:
        raise ValueError('Browser evidence missing, changed or failing')
    layout = assess(svg, browser)
    if layout['status'] != 'passed':
        raise ValueError('Scoped layout assessment failed: ' + str(layout['issues']))
    layout['browser'] = browser
    layout_bytes = encode(layout)
    if not (work / 'visual-review.json').is_file(): raise ValueError('Complete visual review first')
    visual = load(work / 'visual-review.json')
    if visual.get('artifact_sha256') != sha(svg) or visual.get('status') != 'passed':
        raise ValueError('Visual review must bind exact candidate')
    for name in ('browser.json', 'visual-review.json'):
        path = work / name
        snapshot.hashes[str(path.resolve())] = sha(path.read_bytes())
    state = load(STATE)
    svg_path, html_path = Path(manifest['svg_path']), Path(manifest['html_path'])
    evidence = load(work / 'evidence.json')
    evidence['browser_review'] = dict(path=str(BROWSER), sha256=sha((work / 'browser.json').read_bytes()))
    evidence['visual_review'] = visual
    evidence_bytes = encode(evidence)
    evidence_ref = ref(EVIDENCE, evidence_bytes)
    draft = state['draft']
    draft['status'] = f'Unfrozen Figure 1 draft; H uses author-selected {FISH}; inherited unresolved scopes retained'
    draft['working_svg'].update(path=str(svg_path), sha256=sha(svg))
    draft['review'] = dict(path=str(html_path), sha256=sha(html))
    draft['legend'] = dict(path=str(LEGEND), sha256=sha((work / 'legend.md').read_bytes()), panel_ids=list('ABCDEFGH'))
    draft['source_assembly_sha256'] = sha(svg)
    draft['previous_effective_presentation'] = draft['effective_presentation']
    draft['effective_presentation'] = ref(MEASURES, (work / 'measurements.json').read_bytes())
    draft['previous_layout_frames'] = draft['layout_frames']
    draft['layout_frames'] = ref(LAYOUT, layout_bytes)
    draft['trace3_example_selection'] = evidence_ref
    for item in draft['condition_order']:
        if item['display_panel'] == 'H':
            if item['fish_id'] != PREVIOUS: raise ValueError('H selection changed')
            item['fish_id'] = FISH
            item['source_bins'] = evidence['selection']['data']
            item['selection_status'] = 'unfrozen author-selected replacement'
    semantic = draft['semantic_registry']
    semantic['scope'] = semantic['scope'].split('; active H replaced with')[0] + f'; active H replaced with source-bound unfrozen {FISH} example'
    semantic['trace3_example_selection'] = evidence_ref
    mapping = semantic['condition_order_mapping']
    mapping['artifact_sha256'] = sha(svg)
    mapping['mapping'] = copy.deepcopy(draft['condition_order'])
    mapping['example_replacement_evidence'] = evidence_ref
    for axis in semantic['axis_inventory']['axes']:
        if axis.get('fish') == PREVIOUS:
            axis['fish'] = FISH
            axis['example_replacement_evidence'] = evidence_ref
    text_inventory = semantic['text_presentation_inventory']
    for item in text_inventory.get('text', []):
        for key in ('source_text', 'display_text'):
            if item.get(key) == PREVIOUS: item[key] = FISH
    for item in text_inventory.get('composition_changes', []):
        if item.get('text') == PREVIOUS: item['text'] = FISH
    draft['scientific_changes']['FGH'] = (f'Active display F Control 20221115_09, G Delay 20221115_07, H 3sTrace {FISH}. '
        'H is an unfrozen author-selected example replacement, using authenticated direct original log-frame medians in 1 s bins, '
        'trial baseline [-15,0) bin median, no amplitude scaling, managua_r +/-0.25 and black missing cells. '
        'Historical accepted F/G/H freeze remains separately preserved.')
    draft['scientific_changes']['H_example'] = evidence_ref
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    state['verification']['latest_scoped_example_edit'] = dict(evidence=evidence_ref, status='source, structure and rendered review passed; unfrozen')
    state['reproduction']['latest_scoped_example_edit'] = dict(generator=ref(Path(__file__)), starting_ticket=ref(TICKET), evidence=evidence_ref)
    backup = io.BytesIO()
    with zipfile.ZipFile(backup, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in (STATE, svg_path, html_path, LEGEND, README, TICKET): archive.writestr(path.name, path.read_bytes())
    artifacts = [(BACKUP, backup.getvalue()), (EVIDENCE, evidence_bytes),
                 (MEASURES, (work / 'measurements.json').read_bytes()), (BROWSER, (work / 'browser.json').read_bytes()),
                 (LAYOUT, layout_bytes)]
    updates = [(svg_path, svg), (html_path, html), (LEGEND, (work / 'legend.md').read_bytes()),
               (README, (work / 'README.md').read_bytes())]
    publish_state(STATE, state, snapshot, artifacts=artifacts, evidence_updates=updates)
    # Authenticate saved bytes after the state commit.
    for path, expected in [(svg_path,sha(svg)), (html_path,sha(html)), (LEGEND,sha((work/'legend.md').read_bytes()))]:
        bound(path, expected)
    print(json.dumps(dict(status='published', figure='1', panel='H', fish=FISH, svg=str(svg_path), html=str(html_path), evidence=str(EVIDENCE))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'publish'])
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--fish', default=FISH)
    parser.add_argument('--previous-fish', default=PREVIOUS)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--ticket', type=Path, default=TICKET)
    parser.add_argument('--candidate-review', type=Path, default=CANDIDATE_REVIEW)
    args = parser.parse_args()
    configure(args)
    (prepare if args.command == 'prepare' else publish)(args.work_dir.resolve())
