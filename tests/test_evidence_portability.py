"""Only exact measurements and provable newline-only source transport may pass."""
import copy
import tempfile
from pathlib import Path
import unittest

from classical_conditioning.figure_workspace import sha
from classical_conditioning.figures import layout_frames as frames
from classical_conditioning.figures.evidence_portability import revalidate_layout, source_equivalence


def fixture():
    spec = dict(scope='figure', calibration=True, inset_mm=1.5,
                final_size_mm=[89., 30.], background_ids=[], alignments=[],
                allocations=[dict(id='A', kind='panel', box_mm=[0, 0, 89., 30.], group_ids=['A'])])
    data = frames.embed_spec(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 89 30"><g id="A"/></svg>', spec)
    browser = dict(artifact_sha256=sha(data), spec_sha256=frames.object_sha(spec),
                   collector_sha256=sha(frames.COLLECTOR.read_bytes()), final_size_mm=spec['final_size_mm'],
                   scope='figure', user_agent='synthetic test only', fonts_ready=True, images_ready=True,
                   raster_dpi=600, issues=[], allocations={'A': dict(painted_bounds_mm=[3, 3, 80, 20],
                                                                   unclipped_bounds_mm=[3, 3, 80, 20])})
    return data, frames.assess(data, browser)


class EvidencePortabilityTests(unittest.TestCase):
    def test_only_newlines_are_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'source.py'
            p.write_bytes(b'value = 1\n')
            result = source_equivalence(p, sha(b'value = 1\r\n'))
            self.assertEqual(result['equivalence'], 'LF/CRLF only')
            with self.assertRaisesRegex(ValueError, 'beyond line endings'):
                source_equivalence(p, sha(b'value = 2\r\n'))

    def test_transport_preserves_original_and_measured_arrays(self):
        data, saved = fixture()
        before = copy.deepcopy(saved)
        for key, path in [('policy_sha256', frames.POLICY), ('collector_sha256', frames.COLLECTOR),
                          ('helper_sha256', Path(frames.__file__))]:
            saved[key] = sha(path.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
        saved['browser']['collector_sha256'] = saved['collector_sha256']
        transported_input = copy.deepcopy(saved)
        result = revalidate_layout(data, saved, {'path': 'synthetic', 'sha256': 'fixture'})
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(saved, transported_input)
        self.assertEqual(result['browser']['allocations'], before['browser']['allocations'])
        self.assertFalse(result['measurement_provenance']['new_browser_measurements'])

    def test_artwork_changes_fail(self):
        data, saved = fixture()
        with self.assertRaisesRegex(ValueError, 'different SVG'):
            revalidate_layout(data + b' ', saved, {})

    def test_changed_source_or_collector_disagreement_fails(self):
        data, saved = fixture()
        saved['helper_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'beyond line endings'):
            revalidate_layout(data, saved, {})
        data, saved = fixture()
        saved['browser']['collector_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'identities disagree'):
            revalidate_layout(data, saved, {})

    def test_failed_evidence_and_outside_bounds_cannot_be_promoted(self):
        data, saved = fixture()
        saved['status'] = 'unresolved'
        with self.assertRaisesRegex(ValueError, 'already pass'):
            revalidate_layout(data, saved, {})
        data, saved = fixture()
        saved['browser']['allocations']['A']['unclipped_bounds_mm'] = [0, 0, 89, 30]
        with self.assertRaisesRegex(ValueError, 'fail current checks'):
            revalidate_layout(data, saved, {})


if __name__ == '__main__':
    unittest.main()
