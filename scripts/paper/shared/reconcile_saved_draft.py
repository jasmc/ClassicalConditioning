"""Reconcile an authenticated saved draft without rewriting SVG/HTML or freezes.

The explicit manifest binds the observed conflict, preserved originals, source
evidence, proposed state and matching caption. Publication uses the existing
snapshot/lock/atomic-pointer guard; --check-only performs the same preflight.
"""
from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / 'pyproject.toml').is_file() and (p / 'src/classical_conditioning').is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / 'src')]

import argparse
import json
from pathlib import Path
from html.parser import HTMLParser
import xml.etree.ElementTree as ET
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot, publish_state, sha
from classical_conditioning.figures.evidence_portability import revalidate_layout
from classical_conditioning.figures.layout_frames import assess
from scripts.paper.shared.collection import embedded_svg


class ReviewContents(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.text = text
        self.offsets = [0]
        for line in text.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.svg_depth = 0
        self.fragments = []
        self.legend_depth = 0
        self.legend_tag = None
        self.legend = []

    def position(self):
        line, column = self.getpos()
        return self.offsets[line - 1] + column

    def handle_starttag(self, tag, attrs):
        if tag == 'svg':
            if not self.svg_depth:
                self.start = self.position()
            self.svg_depth += 1
        if self.legend_depth and tag == self.legend_tag:
            self.legend_depth += 1
        elif not self.legend_depth and (tag == 'figcaption' or
                (tag == 'div' and 'legend' in dict(attrs).get('class', '').split())):
            self.legend_tag = tag
            self.legend_depth = 1

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag == 'svg' and self.svg_depth:
            self.svg_depth -= 1
            if not self.svg_depth:
                self.fragments.append(self.text[self.start:self.text.index('>', self.position()) + 1])
        if tag == self.legend_tag and self.legend_depth:
            self.legend_depth -= 1

    def handle_data(self, data):
        if self.legend_depth:
            self.legend.append(data)


def verify_review(svg, html, caption, figure_id):
    contents = ReviewContents(html.decode())
    contents.feed(contents.text)
    if len(contents.fragments) != 1:
        raise ValueError('Review must contain exactly one scoped SVG')
    actual = ET.canonicalize(contents.fragments[0])
    if actual not in {ET.canonicalize(svg.decode()),
                      ET.canonicalize(embedded_svg(svg, figure_id + '--'))}:
        raise ValueError('Review SVG differs from working artwork')
    expected = ' '.join(p for p in caption.decode().strip().split('\n\n') if not p.startswith('#'))
    if ' '.join(expected.split()) != ' '.join(' '.join(contents.legend).split()):
        raise ValueError('Review caption differs from canonical caption')


def authenticated(ref, snapshot):
    path = Path(ref['path']).resolve()
    raw = path.read_bytes()
    if sha(raw) != ref['sha256']:
        raise ValueError('Reconciliation input hash mismatch: ' + str(path))
    previous = snapshot.hashes.setdefault(str(path), sha(raw))
    if previous != sha(raw):
        raise ValueError('Reconciliation input differs from starting snapshot: ' + str(path))
    return raw


def preflight(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    snapshot = Snapshot(command_sources(__file__))
    snapshot.hashes[str(manifest_path)] = sha(raw)
    for path, digest in manifest['starting_hashes'].items():
        existing = snapshot.hashes.setdefault(str(Path(path).resolve()), digest)
        if existing != digest:
            raise ValueError('Starting implementation/source changed: ' + path)
    snapshot.check()
    for ref in manifest['dependencies']:
        authenticated(ref, snapshot)
    old = json.loads(authenticated(manifest['previous_state'], snapshot))
    state = json.loads(authenticated(manifest['candidate_state'], snapshot))
    for key in ('figure_id', 'scientific_selection', 'frozen_display'):
        if state.get(key) != old.get(key):
            raise ValueError('Reconciliation cannot change frozen/scientific authority: ' + key)
    if state.get('whole_figure_frozen', False) != old.get('whole_figure_frozen', False):
        raise ValueError('Reconciliation cannot create a whole-figure freeze')
    for ref in old.get('scientific_selection', {}).get('records', []):
        authenticated(ref, snapshot)
    updates = []
    legend = authenticated(manifest['caption_source'], snapshot)
    draft = state['draft']
    if sha(legend) != draft['legend']['sha256']:
        raise ValueError('Candidate caption identity mismatch')
    old_caption = Path(old['draft']['legend']['path']).resolve()
    if Path(draft['legend']['path']).resolve() != old_caption:
        raise ValueError('Preserve the existing canonical caption path')
    updates.append((old_caption, legend))
    for role in ('working_svg', 'review'):
        observed = manifest['observed_artifacts'][role]
        if manifest['starting_hashes'].get(str(Path(observed['path']).resolve())) != observed['sha256']:
            raise ValueError('Observed artwork was not captured before reconciliation: ' + role)
        if draft[role]['path'] != observed['path'] or draft[role]['sha256'] != observed['sha256']:
            raise ValueError('Candidate must preserve authenticated observed artwork: ' + role)
        authenticated(observed, snapshot)
    svg = authenticated(draft['working_svg'], snapshot)
    verify_review(svg, authenticated(draft['review'], snapshot), legend, state['figure_id'])
    layout = json.loads(authenticated(draft['layout_frames'], snapshot))
    source = layout['measurement_provenance']['source_evidence']
    transported = revalidate_layout(svg, json.loads(authenticated(source, snapshot)), source)
    if transported != layout:
        raise ValueError('Layout transport receipt differs from authenticated original')
    current = assess(svg, layout['browser'])
    for key in ('artifact_sha256', 'spec_sha256', 'policy_sha256', 'collector_sha256', 'helper_sha256'):
        if current.get(key) != layout.get(key):
            raise ValueError('Candidate layout evidence is stale: ' + key)
    if current['status'] != 'passed':
        raise ValueError('Candidate layout checks failed: ' + str(current['issues']))
    effective = json.loads(authenticated(draft['effective_presentation'], snapshot))
    if effective.get('svg_sha256') != sha(svg):
        raise ValueError('Effective-presentation evidence belongs to different artwork')
    registry = draft['semantic_registry']
    if 'path' in registry:
        authenticated(registry, snapshot)
    snapshot.check()
    return manifest, state, snapshot, updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    manifest, state, snapshot, updates = preflight(args.manifest)
    if not args.check_only:
        publish_state(manifest['previous_state']['path'], state, snapshot,
                      evidence_updates=updates)
    print(json.dumps({'figure_id': state['figure_id'],
                      'status': 'checked' if args.check_only else 'published',
                      'artwork_unchanged': True, 'new_freeze': False}))


if __name__ == '__main__':
    main()
