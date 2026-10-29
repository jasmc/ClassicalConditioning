"""Publish the exact two-canvas repair after fresh browser and visual review.

Uses repair_canvas's candidate, original snapshot and the shared atomic state
publisher. No geometry, captions, scientific selections or frozen bytes change.
"""
from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / 'pyproject.toml').is_file() and (p / 'src/classical_conditioning').is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / 'src')]

import argparse
import json
from pathlib import Path

from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot, publish_state, sha
from classical_conditioning.figures.layout_frames import assess
from scripts.paper.figure_02.assembly.repair_canvas import GROUPS, transparent_canvases
from scripts.paper.shared.collection import encode, reference, render
from scripts.paper.shared.reconcile_saved_draft import authenticated, verify_review


def preflight(directory, browser_report, note):
    directory = Path(directory).resolve()
    if directory.is_relative_to(_COMMAND_REPOSITORY) or not directory.is_dir():
        raise ValueError('Use an existing external candidate directory')
    if directory.is_relative_to('/Volumes') and not Path('/Volumes/JOAQUIM').is_mount():
        raise ValueError('JOAQUIM unavailable')
    if not note or not note.strip():
        raise ValueError('Actual saved-artifact visual review note required')
    snapshot = Snapshot(command_sources(__file__))
    manifest = json.loads(authenticated(reference(directory / 'manifest.json'), snapshot))
    for path, digest in manifest['source_hashes'].items():
        if snapshot.hashes.setdefault(str(Path(path).resolve()), digest) != digest:
            raise ValueError('Starting implementation changed: ' + path)
    snapshot.check()
    ticket = json.loads(authenticated(manifest['starting_ticket'], snapshot))
    if ticket['state_sha256'] != manifest['selected_state']['sha256']:
        raise ValueError('Candidate and starting state differ')
    for path, digest in ticket['starting_hashes'].items():
        if snapshot.hashes.setdefault(str(Path(path).resolve()), digest) != digest:
            raise ValueError('Starting source changed: ' + path)
    snapshot.check()
    state_ref = manifest['selected_state']
    original_state = authenticated(state_ref, snapshot)
    state = json.loads(original_state)
    if state['figure_id'] != 'fig2':
        raise ValueError('This publisher is scoped to Figure 2')
    draft = state['draft']
    source = authenticated(draft['working_svg'], snapshot)
    old_review = authenticated(draft['review'], snapshot)
    legend = authenticated(draft['legend'], snapshot)
    verify_review(source, old_review, legend, 'fig2')
    for ref in state.get('scientific_selection', {}).get('records', []):
        authenticated(ref, snapshot)
    for key in ('layout_frames', 'effective_presentation'):
        authenticated(draft[key], snapshot)
    registry = json.loads(authenticated(draft['semantic_registry'], snapshot))
    candidate = authenticated(manifest['outputs']['candidate.svg'], snapshot)
    expected, protection = transparent_canvases(source, GROUPS)
    if candidate != expected or protection != manifest['repair']:
        raise ValueError('Candidate differs from the exact two-canvas repair')
    candidate_review = authenticated(manifest['outputs']['review.html'], snapshot)
    verify_review(candidate, candidate_review, legend, 'fig2')
    browser = json.loads(authenticated(reference(Path(browser_report).resolve()), snapshot))
    if 'artifact_sha256' not in browser:
        browser = browser['candidate.svg']
    frame = assess(candidate, browser)
    if frame['status'] != 'passed':
        raise ValueError('Fresh browser frame checks failed: ' + str(frame['issues']))
    effective = authenticated(manifest['outputs']['effective-presentation.json'], snapshot)
    if json.loads(effective)['svg_sha256'] != sha(candidate):
        raise ValueError('Effective presentation belongs to different artwork')

    frame_raw = encode({**frame, 'visual_review': {'status': 'reviewed', 'note': note},
                        'protection': protection})
    frame_ref = reference(directory / 'verified-layout-evidence.json', frame_raw)
    evidence = {'schema_version': 1, 'scope': 'Figure 2 D/E/F canvas visibility',
                'source_svg': draft['working_svg'], 'candidate_sha256': sha(candidate),
                'candidate_manifest': reference(directory / 'manifest.json'),
                'browser_report': reference(Path(browser_report).resolve()),
                'layout_frames': frame_ref, 'protection': protection,
                'visual_review': note, 'scientific_selection_changed': False,
                'freeze_created': False, 'implementation': reference(Path(__file__)),
                'coverage': 'Two named white canvases only; inherited scientific and production limitations remain.'}
    evidence_raw = encode(evidence)
    evidence_ref = reference(directory / 'publication-evidence.json', evidence_raw)
    registry['canvas_visibility_repair'] = {'artifact_sha256': sha(candidate),
        'groups': list(GROUPS), 'evidence': evidence_ref,
        'status': 'verified transparent backgrounds; all other SVG bytes preserved'}
    registry_raw = encode(registry)
    draft['semantic_registry'] = reference(directory / 'semantic-registry.json', registry_raw)
    draft['working_svg'] = {**draft['working_svg'], 'sha256': sha(candidate)}
    review = render([{'id': 'fig2', 'svg_bytes': candidate, 'legend_text': legend.decode(),
                      'svg': draft['working_svg'], 'legend': draft['legend'],
                      'whole_figure_frozen': state.get('whole_figure_frozen', False)}])[0]
    verify_review(candidate, review, legend, 'fig2')
    draft['review'] = {**draft['review'], 'sha256': sha(review)}
    draft['layout_frames'] = frame_ref
    draft['effective_presentation'] = manifest['outputs']['effective-presentation.json']
    draft.setdefault('verification', {})['canvas_visibility_repair'] = evidence_ref
    artifacts = [(directory / 'before-state.json', original_state),
                 (directory / 'before.svg', source), (directory / 'before.html', old_review),
                 (directory / 'legend.md', legend),
                 (directory / 'verified-layout-evidence.json', frame_raw),
                 (directory / 'publication-evidence.json', evidence_raw),
                 (directory / 'semantic-registry.json', registry_raw)]
    for path, _ in artifacts:
        if path.exists():
            raise FileExistsError('Preserve existing publication artifact: ' + str(path))
    updates = [(Path(draft['working_svg']['path']), candidate),
               (Path(draft['review']['path']), review)]
    # Freeze/scientific authority and every other figure remain untouched.
    for key in ('scientific_selection', 'frozen_display', 'whole_figure_frozen'):
        if state.get(key) != json.loads(original_state).get(key):
            raise ValueError('Scientific authority changed: ' + key)
    snapshot.check()
    return Path(state_ref['path']), state, snapshot, artifacts, updates


def publish(directory, browser_report, note, *, check_only=False):
    path, state, snapshot, artifacts, updates = preflight(directory, browser_report, note)
    if not check_only:
        publish_state(path, state, snapshot, artifacts=artifacts, evidence_updates=updates)
    return {'figure_id': 'fig2', 'status': 'checked' if check_only else 'published',
            'svg_sha256': state['draft']['working_svg']['sha256'], 'freeze_created': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--browser-report', type=Path, required=True)
    parser.add_argument('--visual-review-note', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(publish(args.directory, args.browser_report, args.visual_review_note,
                             check_only=args.check_only), indent=2))


if __name__ == '__main__':
    main()
