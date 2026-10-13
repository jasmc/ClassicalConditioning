"""Synthetic ownership and provenance failures; no real data or SSD required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('structure_check', ROOT / 'scripts/check_repository_structure.py')
structure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(structure)


class StructureChecks(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)
        for folder in ('docs/project-maintenance', 'docs/paper/figures', 'Plans', 'records/repository-archive', 'records/frozen-analyses', 'records/history'):
            (self.repo / folder).mkdir(parents=True)
        self.write('docs/project-maintenance/structure-policy.json', dict(
            active_doc_prefixes=['docs/paper/', 'docs/project-maintenance/'], compatibility_files=[],
            byte_sensitive_documents=[], forbidden_paths=['docs/analysis/learners'],
            migration_manifest='records/history/migration.json'))
        self.write('records/repository-archive/relocations.json', {'repository_moves': {}})
        self.write('records/frozen-analyses/index.json', {'files': []})
        self.write('records/history/migration.json', {'protected_files': []})

    def write(self, path, value):
        (self.repo / path).write_text(json.dumps(value))

    def test_minimal_layout_passes_without_external_access(self):
        report = structure.check(self.repo)
        self.assertEqual(report['issues'], [])
        self.assertEqual(report['external_verification'], 'not-requested')

    def test_historical_topic_folder_is_rejected(self):
        old = self.repo / 'docs/analysis/learners'
        old.mkdir(parents=True)
        (old / 'method.md').write_text('obsolete parallel guidance')
        report = structure.check(self.repo)
        self.assertTrue(any('Misplaced documentation' in issue for issue in report['issues']))
        self.assertTrue(any('Superseded layout' in issue for issue in report['issues']))

    def test_relocation_cycle_is_reported(self):
        self.write('records/repository-archive/relocations.json', {'repository_moves': {'a': 'b', 'b': 'a'}})
        self.assertTrue(any('Relocation cycle' in issue for issue in structure.check(self.repo)['issues']))

    def test_changed_preserved_evidence_fails(self):
        evidence = self.repo / 'records/history/exact.md'
        evidence.write_bytes(b'original')
        self.write('records/history/migration.json', {'protected_files': [
            {'path': 'records/history/exact.md', 'sha256': structure.sha256(evidence)}]})
        self.assertEqual(structure.check(self.repo)['issues'], [])
        evidence.write_bytes(b'changed')
        self.assertIn('Preserved bytes changed: records/history/exact.md', structure.check(self.repo)['issues'])

    def test_broken_active_link_is_reported(self):
        (self.repo / 'docs/paper/README.md').write_text('[Figure](figures/missing.md)')
        self.assertTrue(any('Broken active link' in issue for issue in structure.check(self.repo)['issues']))

    def test_existing_file_with_missing_heading_is_rejected(self):
        (self.repo / 'docs/paper/README.md').write_text('# Guide\n\n[Missing section](#removed-section)')
        self.assertTrue(any('Broken heading link' in issue for issue in structure.check(self.repo)['issues']))

    def test_operational_manual_cannot_be_a_future_progress_board(self):
        guide = self.repo / 'docs/running-analysis'
        guide.mkdir()
        (guide / 'build.md').write_text('# Build tools\n\n## Future command surface\n\nNot implemented.')
        policy_path = self.repo / 'docs/project-maintenance/structure-policy.json'
        policy = json.loads(policy_path.read_text())
        policy['active_doc_prefixes'].append('docs/running-analysis/')
        policy['future_work_headings'] = ['Future command surface']
        policy_path.write_text(json.dumps(policy))
        self.assertTrue(any('Future work/progress in operational guide' in issue for issue in structure.check(self.repo)['issues']))

    def test_joint_selection_and_local_state_evidence_are_authenticated(self):
        figure = self.repo / 'docs/paper/figures/figure-01'
        figure.mkdir()
        evidence = self.repo / 'records/history/joint.json'
        evidence.write_text('joint F/G/H')
        self.write('docs/paper/figures/figure-01/current-state.json', {
            'scientific_selection': {'records': [{'path': str(evidence), 'sha256': structure.sha256(evidence)}]},
            'frozen_display': {}, 'draft': {}})
        self.assertEqual(structure.check(self.repo)['issues'], [])
        evidence.write_text('other selection')
        self.assertTrue(any('Bound figure record changed' in issue for issue in structure.check(self.repo)['issues']))

    def test_explicit_current_selection_cannot_disagree_with_state(self):
        figure = self.repo / 'docs/paper/figures/figure-01'
        figure.mkdir()
        configs = self.repo / 'configs/paper-figures'
        configs.mkdir(parents=True)
        (configs / 'accepted.json').write_text('accepted')
        self.write('configs/paper-figures/behavior-paper.json', {'panels': {
            'fig-1F': {'scoped_selection_record': 'configs/paper-figures/accepted.json'}}})
        self.write('docs/paper/figures/figure-01/current-state.json', {
            'scientific_selection': {'records': []}, 'frozen_display': {}, 'draft': {}})
        self.assertIn('Scientific registry/state disagreement: fig-1F', structure.check(self.repo)['issues'])


if __name__ == '__main__':
    unittest.main()
