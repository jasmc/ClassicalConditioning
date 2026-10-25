"""Saved scientific drafts cannot bypass state/source/artwork preservation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from classical_conditioning.figure_workspace import Snapshot, StaleFigureState, sha, publish_state
from classical_conditioning.figures.evidence_portability import revalidate_layout
from scripts.paper.shared.reconcile_saved_draft import preflight, verify_review
from test_evidence_portability import fixture


class SavedDraftReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        svg, saved = fixture()
        def write(name, value):
            data = value if isinstance(value, bytes) else (json.dumps(value) + '\n').encode()
            p = self.root / name
            p.write_bytes(data)
            return {'path': str(p.resolve()), 'sha256': sha(data)}
        self.write = write
        self.svg = write('working.svg', svg)
        self.html = write('review.html', b'<html><body>' + svg + b'<div class="legend"><p>New caption</p></div></body></html>')
        self.caption = write('legend.md', b'Old caption\n')
        source = write('saved-layout.json', saved)
        layout = write('layout.json', revalidate_layout(svg, saved, source))
        effect = write('effective.json', {'svg_sha256': sha(svg)})
        self.old = {'figure_id': 'fig1', 'scientific_selection': {'records': []}, 'frozen_display': {},
                    'draft': {'working_svg': self.svg, 'review': self.html, 'legend': self.caption}}
        self.state_ref = write('state.json', self.old)
        self.candidate = copy.deepcopy(self.old)
        self.candidate['draft'].update(layout_frames=layout, effective_presentation=effect, semantic_registry={})
        new_caption = write('matching-caption.md', b'New caption\n')
        self.candidate['draft']['legend']['sha256'] = new_caption['sha256']
        self.candidate_ref = write('candidate.json', self.candidate)
        self.manifest = {'starting_hashes': Snapshot([self.state_ref['path'], self.svg['path'], self.html['path'], self.caption['path']]).hashes,
                         'dependencies': [], 'previous_state': self.state_ref, 'candidate_state': self.candidate_ref,
                         'caption_source': new_caption, 'observed_artifacts': {'working_svg': self.svg, 'review': self.html}}
        self.manifest_path = Path(write('manifest.json', self.manifest)['path'])

    def test_guarded_publication_updates_caption_and_state_only(self):
        before = Path(self.svg['path']).read_bytes(), Path(self.html['path']).read_bytes()
        manifest, state, snapshot, updates = preflight(self.manifest_path)
        publish_state(manifest['previous_state']['path'], state, snapshot, evidence_updates=updates)
        self.assertEqual(Path(self.caption['path']).read_bytes(), b'New caption\n')
        self.assertEqual(json.loads(Path(self.state_ref['path']).read_text()), self.candidate)
        self.assertEqual(before, (Path(self.svg['path']).read_bytes(), Path(self.html['path']).read_bytes()))

    def test_concurrent_work_is_preserved(self):
        _, _, snapshot, _ = preflight(self.manifest_path)
        p = Path(self.svg['path'])
        p.write_bytes(b'Manual edit')
        with self.assertRaises(StaleFigureState):
            snapshot.check()
        self.assertEqual(p.read_bytes(), b'Manual edit')

    def test_scientific_selection_cannot_change(self):
        self.candidate['scientific_selection']['records'].append({'path': 'invented', 'sha256': '0' * 64})
        self.manifest['candidate_state'] = self.write('candidate.json', self.candidate)
        self.write('manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'scientific authority'):
            preflight(self.manifest_path)

    def test_unbound_observed_artwork_cannot_be_adopted(self):
        self.manifest['starting_hashes'].pop(self.svg['path'])
        self.write('manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'captured before'):
            preflight(self.manifest_path)

    def test_html_artwork_and_caption_must_match(self):
        svg = Path(self.svg['path']).read_bytes()
        html = Path(self.html['path']).read_bytes()
        with self.assertRaisesRegex(ValueError, 'caption differs'):
            verify_review(svg, html, b'Wrong caption', 'fig1')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            verify_review(svg, html + svg, b'New caption', 'fig1')
        with self.assertRaisesRegex(ValueError, 'differs from working artwork'):
            verify_review(svg.replace(b'<g ', b'<g opacity="0" '), html, b'New caption', 'fig1')


if __name__ == '__main__':
    unittest.main()
