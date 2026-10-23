"""Check manuscript plumbing without rendering guides or executing analyses."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from classical_conditioning.external_artifacts import sha256

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('g_study_builder', ROOT / 'scripts/build_figure2_g_analysis_guide.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class GuideManuscript(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'study.md'
        self.source.write_text('# Source\n\nScientific explanation, not a pointer.\n')

    def test_explicit_source_requires_authentication(self):
        with self.assertRaisesRegex(ValueError, 'requires'):
            builder.Guide(self.root, self.source)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            builder.Guide(self.root, self.source, '0' * 64)

    def test_authenticated_source_copied_as_exact_evidence(self):
        guide = builder.Guide(self.root, self.source, sha256(self.source))
        guide.source('guide-content', 'Manuscript', guide.manuscript, expected=guide.manuscript_sha256)
        self.assertEqual((self.root / 'sources/guide-content.md').read_bytes(), self.source.read_bytes())
        self.assertEqual(guide.sources['guide-content']['sha256'], sha256(self.source))

    def test_manuscript_changed_before_capture_fails(self):
        guide = builder.Guide(self.root, self.source, sha256(self.source))
        self.source.write_text('Concurrent revision')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            guide.source('guide-content', 'Manuscript', guide.manuscript, expected=guide.manuscript_sha256)


if __name__ == '__main__':
    unittest.main()
