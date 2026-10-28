"""Exercise unmanaged candidate publication, saved state preservation and conflicts."""
import json
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from classical_conditioning.figure_workspace import sha, StaleFigureState
from scripts.paper.figure_02.assembly import repair_canvas as repair


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="156mm" height="30mm" viewBox="0 0 30 10">'
        for group in repair.GROUPS:
            svg += f'<svg viewBox="0 0 10 10"><g id="{group}"><path d="M 0 10 L 10 10 L 10 0 L 0 0 z" style="fill: #ffffff" /></g></svg>'
        svg += '</svg>'
        self.draft = {}
        for key, raw in {'working_svg': svg.encode(), 'review': ('<html>' + svg + '<figcaption>Caption.</figcaption></html>').encode(),
                         'legend': b'# Figure 2\n\nCaption.\n', 'semantic_registry': b'{}',
                         'layout_frames': b'{}', 'effective_presentation': b'{}'}.items():
            path = self.root / key
            path.write_bytes(raw)
            self.draft[key] = {'path': str(path), 'sha256': sha(raw)}
        self.state = self.root / 'state.json'
        self.state.write_text(json.dumps({'figure_id': 'fig2', 'draft': self.draft,
                                         'scientific_selection': {'records': []}}))
        self.ticket = self.root / 'ticket.json'
        self.ticket.write_text(json.dumps({'state_path': str(self.state), 'state_sha256': sha(self.state.read_bytes()),
            'starting_hashes': {str(p): sha(p.read_bytes()) for p in self.root.iterdir()}}))
        self.before = {p: p.read_bytes() for p in self.root.iterdir()}
        self.out = self.root / 'candidate'
        self.out.mkdir()

    def test_matching_pair_is_saved_without_selecting_it(self):
        manifest = repair.prepare(self.ticket, self.out)
        self.assertEqual(manifest['status'], 'unselected candidate')
        for path, raw in self.before.items():
            self.assertEqual(path.read_bytes(), raw)
        for ref in manifest['outputs'].values():
            self.assertEqual(sha(Path(ref['path']).read_bytes()), ref['sha256'])
        evidence = json.loads((self.out / 'layout-evidence.json').read_bytes())
        self.assertNotEqual(evidence['status'], 'passed')
        repair.verify_review((self.out / 'candidate.svg').read_bytes(), (self.out / 'review.html').read_bytes(),
                             Path(self.draft['legend']['path']).read_bytes(), 'fig2')
        html = (self.out / 'review.html').read_text()
        provenance = json.loads(re.search(r'id="paper-figure-provenance">(.*?)</script>', html)[1])
        self.assertEqual(provenance['figures'][0]['svg']['sha256'], sha((self.out / 'candidate.svg').read_bytes()))
        self.assertEqual(provenance['figures'][0]['publication_size_mm'], [156, 52])
        self.assertIn('Unselected', provenance['scientific_status'])

    def test_existing_candidate_is_never_overwritten(self):
        saved = self.out / 'candidate.svg'
        saved.write_bytes(b'manual candidate')
        with self.assertRaises(FileExistsError):
            repair.prepare(self.ticket, self.out)
        self.assertEqual(saved.read_bytes(), b'manual candidate')
        self.assertEqual(list(self.out.iterdir()), [saved])

    def test_existing_review_provenance_is_rebound_to_candidate(self):
        html_path = Path(self.draft['review']['path'])
        html_path.write_text(html_path.read_text().replace('</html>',
            '<script id="paper-figure-provenance" type="application/json">{"stale":true}</script></html>'))
        state = json.loads(self.state.read_bytes())
        state['draft']['review']['sha256'] = sha(html_path.read_bytes())
        self.state.write_text(json.dumps(state))
        ticket = json.loads(self.ticket.read_bytes())
        ticket['state_sha256'] = sha(self.state.read_bytes())
        ticket['starting_hashes'] = {path: sha(Path(path).read_bytes()) for path in ticket['starting_hashes']}
        self.ticket.write_text(json.dumps(ticket))
        repair.prepare(self.ticket, self.out)
        html = (self.out / 'review.html').read_text()
        self.assertEqual(html.count('id="paper-figure-provenance"'), 1)
        self.assertNotIn('"stale":true', html)
        self.assertIn(sha((self.out / 'candidate.svg').read_bytes()), html)

    def test_concurrent_edit_survives_and_prevents_all_outputs(self):
        original = repair.transparent_canvases
        def conflict(*args):
            result = original(*args)
            Path(self.draft['working_svg']['path']).write_bytes(b'manual edit')
            return result
        with patch.object(repair, 'transparent_canvases', side_effect=conflict):
            with self.assertRaises(StaleFigureState):
                repair.prepare(self.ticket, self.out)
        self.assertEqual(list(self.out.iterdir()), [])
        self.assertEqual(self.state.read_bytes(), self.before[self.state])
        self.assertEqual(Path(self.draft['working_svg']['path']).read_bytes(), b'manual edit')


if __name__ == '__main__':
    unittest.main()
