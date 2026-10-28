"""Prepare an unmanaged Figure 2 canvas-overdraw candidate; never select it.

Preserves all geometry and scientific bytes. Fresh exact browser measurements
are required before a later guarded publication can replace the active pair.
"""
from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / 'pyproject.toml').is_file() and (p / 'src/classical_conditioning').is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / 'src')]

import argparse
import base64
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

from classical_conditioning.artifacts import artifact_staging, publish_transaction
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot, measured_presentation, sha, state_lock
from classical_conditioning.figures.layout_frames import POLICY, assess
from classical_conditioning.figures.svg_canvas import transparent_canvases
from scripts.paper.shared.collection import embedded_svg
from scripts.paper.shared.reconcile_saved_draft import ReviewContents, verify_review
from scripts.paper.shared.verify_layout_frames import external_destination, review_page

GROUPS = ('matched-E__canvas-E', 'trace10-F__canvas-F')


def prepare(ticket_path, output_dir):
    ticket_path, output_dir = Path(ticket_path).resolve(), Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise ValueError('Use an existing explicit external directory')
    raw_ticket = ticket_path.read_bytes()
    ticket = json.loads(raw_ticket)
    snapshot = Snapshot([*command_sources(__file__), POLICY])
    snapshot.hashes[str(ticket_path)] = sha(raw_ticket)
    for path, digest in ticket['starting_hashes'].items():
        path = str(Path(path).resolve())
        if snapshot.hashes.setdefault(path, digest) != digest:
            raise ValueError('Implementation differs from starting ticket: ' + path)
    snapshot.check()
    state_path = Path(ticket['state_path']).resolve()
    raw_state = state_path.read_bytes()
    if sha(raw_state) != ticket['state_sha256'] or snapshot.hashes.get(str(state_path)) != sha(raw_state):
        raise ValueError('State differs from starting ticket')
    state = json.loads(raw_state)
    if state['figure_id'] != 'fig2':
        raise ValueError('This adapter is scoped to Figure 2')
    draft = state['draft']

    def read(ref):
        path = Path(ref['path']).resolve()
        raw = path.read_bytes()
        if sha(raw) != ref['sha256'] or snapshot.hashes.setdefault(str(path), sha(raw)) != sha(raw):
            raise ValueError('Changed bound input: ' + str(path))
        return raw

    raw_svg, raw_html, caption = (read(draft[k]) for k in ('working_svg', 'review', 'legend'))
    for key in ('semantic_registry', 'layout_frames', 'effective_presentation'):
        read(draft[key])
    for ref in state.get('scientific_selection', {}).get('records', []):
        read(ref)
    verify_review(raw_svg, raw_html, caption, 'fig2')
    candidate, repair = transparent_canvases(raw_svg, GROUPS)
    html = raw_html.decode()
    contents = ReviewContents(html)
    contents.feed(html)
    fragment = contents.fragments[0]
    if ET.canonicalize(fragment) == ET.canonicalize(raw_svg.decode()):
        replacement = candidate.decode()
    else:
        replacement = embedded_svg(candidate, 'fig2--')
    html = html.replace(fragment, replacement, 1).encode()
    verify_review(candidate, html, caption, 'fig2')
    width = ET.fromstring(candidate).get('width', '')
    if not width.endswith('mm'):
        raise ValueError('Candidate requires an explicit native width in mm')
    width_mm = float(width[:-2])
    view = [float(v) for v in ET.fromstring(candidate).get('viewBox').split()]
    provenance = {'figures': [{'id': 'fig2',
        'svg': {'path': str(output_dir / 'candidate.svg'), 'sha256': sha(candidate)},
        'legend': draft['legend'], 'whole_figure_frozen': False,
        'embedded_svg_sha256': sha(replacement.encode()),
        'publication_size_mm': [width_mm, width_mm * view[3] / view[2]]}],
        'builder': {'path': str(Path(__file__).resolve()), 'sha256': snapshot.hashes[str(Path(__file__).resolve())]},
        'layout_policy': {'path': str(POLICY), 'sha256': snapshot.hashes[str(POLICY.resolve())]},
        'source_review': draft['review'],
        'scientific_status': 'Unselected presentation candidate; no scientific selection or freeze'}
    script = '<script type="application/json" id="paper-figure-provenance">' + json.dumps(provenance).replace('<', '\\u003c') + '</script>'
    html_text, count = re.subn(r'''<script\b(?=[^>]*\bid\s*=\s*["']paper-figure-provenance["'])[^>]*>.*?</script\s*>''',
                              lambda match: script, html.decode(), flags=re.S | re.I)
    if count > 1:
        raise ValueError('Ambiguous review provenance blocks')
    if not count:
        html_text = html_text.replace('</html>', script + '</html>') if '</html>' in html_text else html_text + script
    html = html_text.encode()
    verify_review(candidate, html, caption, 'fig2')
    payloads = {
        'candidate.svg': candidate,
        'review.html': html,
        'layout-review.html': review_page([{'name': 'candidate.svg',
                                            'svg_base64': base64.b64encode(candidate).decode()}]).encode(),
        'layout-evidence.json': (json.dumps(assess(candidate, None), indent=2) + '\n').encode(),
        'effective-presentation.json': (json.dumps(measured_presentation(candidate, width_mm=width_mm), indent=2) + '\n').encode(),
    }
    manifest = {'schema_version': 1, 'figure_id': 'fig2', 'status': 'unselected candidate',
                'starting_ticket': {'path': str(ticket_path), 'sha256': sha(raw_ticket)},
                'selected_state': {'path': str(state_path), 'sha256': sha(raw_state)},
                'repair': repair, 'scientific_selection_unchanged': True,
                'geometry_and_data_unchanged': True, 'new_freeze': False,
                'publication_blocker': 'Fresh exact browser frame measurements and rendered review required',
                'source_hashes': snapshot.hashes.copy(),
                'outputs': {name: {'path': str(output_dir / name), 'sha256': sha(raw)}
                            for name, raw in payloads.items()}}
    payloads['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    for name in payloads:
        external_destination(output_dir / name)
    with artifact_staging(output_dir, prefix='.canvas-repair-') as staging:
        units = []
        for name, raw in payloads.items():
            staged = staging / name
            staged.write_bytes(raw)
            units.append((staged, output_dir / name))

        def verify():
            snapshot.check()
            for name, raw in payloads.items():
                if (output_dir / name).read_bytes() != raw:
                    raise ValueError('Candidate publication byte mismatch: ' + name)

        with state_lock(state_path):
            snapshot.check()
            publish_transaction(units, staging, overwrite=False, verify_published=verify)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--starting-ticket', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.starting_ticket, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
