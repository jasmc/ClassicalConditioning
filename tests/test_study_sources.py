"""Source-only migration must never substitute a pointer or an older manuscript."""
import json
from pathlib import Path
import tempfile
import unittest
from classical_conditioning.external_artifacts import resolve_artifact, sha256, check_study_sources


class StudySources(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.repo = base / 'repo'
        self.root = base / 'ssd'
        self.root.mkdir()
        (self.repo / 'docs/paper').mkdir(parents=True)
        self.source = self.root / 'study.md'
        self.source.write_text('# Original scientific discussion\n')
        self.entry = {'reference': 'docs/paper/review.md', 'path': 'study.md', 'sha256': sha256(self.source)}
        (self.repo / self.entry['reference']).write_text('Navigation pointer only')
        self.save()

    def save(self):
        (self.repo / 'docs/paper/study-sources.json').write_text(json.dumps({'root': str(self.root), 'entries': [self.entry]}))

    def test_current_source_wins_over_local_pointer(self):
        self.assertEqual(resolve_artifact(self.entry['reference'], repo=self.repo), self.source)
        self.assertEqual(check_study_sources(self.repo, external=True), [])

    def test_missing_drive_fails_without_pointer_fallback_but_offline_check_passes(self):
        self.source.unlink()
        self.assertEqual(check_study_sources(self.repo), [])
        with self.assertRaises(FileNotFoundError):
            resolve_artifact(self.entry['reference'], repo=self.repo)
        self.assertTrue(check_study_sources(self.repo, external=True))

    def test_changed_source_is_rejected(self):
        self.source.write_text('Manual revision requiring identity reconciliation')
        with self.assertRaises(ValueError):
            resolve_artifact(self.entry['reference'], repo=self.repo)

    def test_explicit_historical_identity_uses_verified_original(self):
        original = self.root / 'original.md'
        original.write_bytes(self.source.read_bytes())
        folder = self.repo / 'records/repository-archive'
        folder.mkdir(parents=True)
        (folder / 'verified-artifact-locations.json').write_text(json.dumps({'entries': [{'sha256': sha256(original), 'references': [self.entry['reference']], 'archive_root': str(self.root), 'relative_path': original.name}]}))
        self.source.write_text('Current revision')
        self.entry['sha256'] = sha256(self.source)
        self.save()
        self.assertEqual(resolve_artifact(self.entry['reference'], repo=self.repo), self.source)
        self.assertEqual(resolve_artifact(self.entry['reference'], expected_sha256=sha256(original), repo=self.repo), original)

    def test_traversal_rejected(self):
        outside = self.root.parent / 'outside.md'
        outside.write_text('outside')
        self.entry['path'] = '../outside.md'
        self.save()
        self.assertTrue(check_study_sources(self.repo))
        with self.assertRaises(ValueError):
            resolve_artifact(self.entry['reference'], repo=self.repo)

    def test_symlink_escape_rejected(self):
        outside = self.root.parent / 'outside.md'
        outside.write_text('outside')
        try:
            (self.root / 'escape.md').symlink_to(outside)
        except OSError as error:
            if getattr(error, 'winerror', None) == 1314:
                self.skipTest('Windows account lacks symlink creation privilege')
            raise
        self.entry['path'] = 'escape.md'
        self.save()
        self.assertTrue(check_study_sources(self.repo))
        with self.assertRaises(ValueError):
            resolve_artifact(self.entry['reference'], repo=self.repo)


if __name__ == '__main__':
    unittest.main()
