"""Migration safety uses synthetic external trees, never project artifacts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.maintenance import storage_migration as migration


class StorageMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.outputs = self.root / 'outputs'; self.outputs.mkdir()
        self.archive = self.root / 'archive'; self.archive.mkdir()
        self.source = self.outputs / 'old/review.svg'
        self.source.parent.mkdir(); self.source.write_bytes(b'<svg/>')
        self.destination = self.outputs / 'figures/figure-01/assembly/current/review.svg'
        self.row = dict(source=str(self.source), destination=str(self.destination),
                        bytes=6, sha256=migration.digest(self.source), action='move')
        self.plan = dict(roots=[str(self.outputs), str(self.archive)], excluded=[],
                         authorization='Author: implement storage consolidation', files=[self.row])
        self.manifest = self.root / 'manifest.json'
        self.receipt = self.root / 'publication.json'
        self.state = self.root / 'state.json'; self.state.write_text('{}')
        self.evidence = self.root / 'consumer-check.json'

    def publish(self):
        self.manifest.write_bytes(migration.packed(self.plan))
        self.evidence.write_bytes(migration.packed(dict(
            manifest_sha256=migration.digest(self.manifest), issues=[],
            bindings=[dict(path=str(self.state), bytes=self.state.stat().st_size,
                           sha256=migration.digest(self.state))])))
        migration.publish(self.plan, self.manifest, self.receipt, self.evidence)

    def test_copy_publish_and_prune(self):
        migration.copy(self.plan)
        self.assertTrue(self.source.exists())
        self.publish()
        result = migration.prune(self.plan, self.manifest, self.receipt)
        self.assertEqual(result['logical_source_bytes_removed'], 6)
        self.assertEqual(self.destination.read_bytes(), b'<svg/>')
        self.assertFalse(self.source.exists())
        self.assertEqual(migration.prune(self.plan, self.manifest, self.receipt)['removed'], [])

    def test_source_changed_before_copy(self):
        self.source.write_text('changed')
        with self.assertRaises(ValueError): migration.copy(self.plan)
        self.assertFalse(self.destination.exists())

    def test_source_changed_after_copy(self):
        migration.copy(self.plan); self.publish()
        self.source.write_text('changed')
        with self.assertRaises(ValueError): migration.prune(self.plan, self.manifest, self.receipt)
        self.assertTrue(self.source.exists())

    def test_destination_corruption_stops_prune(self):
        migration.copy(self.plan); self.publish()
        self.destination.write_text('bad')
        with self.assertRaises(ValueError): migration.prune(self.plan, self.manifest, self.receipt)
        self.assertTrue(self.source.exists())

    def test_direct_byte_comparison_is_required_before_removal(self):
        migration.copy(self.plan); self.publish()
        with patch.object(migration, 'equal_bytes', return_value=False):
            with self.assertRaises(ValueError): migration.prune(self.plan, self.manifest, self.receipt)
        self.assertTrue(self.source.exists())

    def test_publication_required(self):
        migration.copy(self.plan)
        with self.assertRaises(FileNotFoundError): migration.prune(self.plan, self.manifest, self.receipt)
        self.assertTrue(self.source.exists())

    def test_concurrent_state_edit_stops_prune(self):
        migration.copy(self.plan); self.publish(); self.state.write_text('{"new":true}')
        with self.assertRaises(ValueError): migration.prune(self.plan, self.manifest, self.receipt)
        self.assertTrue(self.source.exists())

    def test_casefold_collision(self):
        other = self.source.parent / 'other.svg'; other.write_bytes(b'other')
        self.plan['files'].append(dict(source=str(other), destination=str(self.destination).replace('review.svg', 'Review.svg'),
                                       bytes=5, sha256=migration.digest(other), action='move'))
        with self.assertRaises(ValueError): migration.copy(self.plan)

    def test_identical_duplicates_can_have_one_verified_destination(self):
        other = self.source.parent / 'other.svg'; other.write_bytes(b'<svg/>')
        self.plan['files'].append(dict(self.row, source=str(other)))
        migration.copy(self.plan); self.publish()
        result = migration.prune(self.plan, self.manifest, self.receipt)
        self.assertEqual(len(result['removed']), 2)

    def test_destination_collision_preserves_existing_bytes(self):
        self.destination.parent.mkdir(parents=True); self.destination.write_bytes(b'manual edit')
        with self.assertRaises(ValueError): migration.copy(self.plan)
        self.assertEqual(self.destination.read_bytes(), b'manual edit')

    def test_unexpected_file_prevents_copy(self):
        (self.source.parent / 'new.svg').write_text('new work')
        with self.assertRaises(ValueError): migration.copy(self.plan)

    def test_unavailable_storage_not_created(self):
        self.plan['roots'].append(str(self.root / 'missing'))
        with self.assertRaises(ValueError): migration.copy(self.plan)
        self.assertFalse((self.root / 'missing').exists())

    def test_interrupted_completed_copy_resumes(self):
        migration.copy(self.plan)
        self.assertEqual(migration.copy(self.plan)['copied_files'], 0)

    def test_partial_destination_preserved_for_reconciliation(self):
        self.destination.parent.mkdir(parents=True); self.destination.write_bytes(b'<s')
        with self.assertRaises(ValueError): migration.copy(self.plan)
        self.assertEqual(self.destination.read_bytes(), b'<s')

    def test_notes_gate_precedes_any_removal(self):
        migration.copy(self.plan)
        self.row['note_transfer_required'] = True
        self.publish()
        with self.assertRaises(ValueError): migration.prune(self.plan, self.manifest, self.receipt)
        self.assertTrue(self.source.exists())

    def test_symlink_and_traversal_rejected(self):
        for target in [str(self.outputs / '../escape'), str(self.root / 'elsewhere')]:
            with self.subTest(target=target):
                self.row['destination'] = target
                with self.assertRaises(ValueError): migration.copy(self.plan)

    def test_inventory_authenticates_embedded_payload(self):
        import hashlib
        html = self.source.parent / 'review.html'
        html.write_text('<script id="archive">' + json.dumps({'files':{'code':{
            'base64':'YQ==','sha256':hashlib.sha256(b'a').hexdigest()}}}) + '</script>')
        result = migration.inventory(self.plan['roots'])
        self.assertEqual(result['embedded'][0]['bytes'], 1)
        html.write_text(html.read_text().replace(hashlib.sha256(b'a').hexdigest(), 'bad'))
        with self.assertRaises(ValueError): migration.inventory(self.plan['roots'])

    def test_exact_historical_storage_alias_has_no_version_fallback(self):
        from classical_conditioning.external_artifacts import relocated_external_path
        repo = self.root / 'repo'
        index = repo / 'records/repository-archive/relocations.json'
        index.parent.mkdir(parents=True)
        migration.copy(self.plan)
        index.write_text(json.dumps({'external_moves':{str(self.source):dict(
            path=str(self.destination), bytes=6, sha256=self.row['sha256'])}}))
        self.source.unlink()
        self.assertEqual(relocated_external_path(self.source, repo=repo,
                         expected_sha256=self.row['sha256']), self.destination)
        with self.assertRaises(ValueError):
            relocated_external_path(self.source, repo=repo, expected_sha256='0'*64)
        self.destination.write_text('bad')
        with self.assertRaises(ValueError):
            relocated_external_path(self.source, repo=repo)

    def test_missing_alias_destination_fails_without_old_source_fallback(self):
        from classical_conditioning.external_artifacts import relocated_external_path
        repo = self.root / 'repo'
        index = repo / 'records/repository-archive/relocations.json'
        index.parent.mkdir(parents=True)
        index.write_text(json.dumps({'external_moves':{str(self.source):dict(
            path=str(self.destination), bytes=6, sha256=self.row['sha256'])}}))
        with self.assertRaises(FileNotFoundError): relocated_external_path(self.source, repo=repo)
        self.assertTrue(self.source.exists())

    def test_copy_action_preserves_notebook_original(self):
        self.row.update(action='copy', note_transfer_required=True)
        migration.copy(self.plan); self.publish()
        self.assertEqual(migration.prune(self.plan, self.manifest, self.receipt)['removed'], [])
        self.assertTrue(self.source.exists()); self.assertTrue(self.destination.exists())

    def control_plan(self):
        prepared = self.root / 'prepared.json'; prepared.write_text('{"relocated":true}')
        return dict(lock=str(self.root / 'migration'),
                    guards=[dict(path=str(self.state), bytes=2, sha256=migration.digest(self.state))],
                    controls=[dict(path=str(self.state), source=str(prepared), role='state',
                      expected_sha256=migration.digest(self.state), sha256=migration.digest(prepared),
                      bytes=prepared.stat().st_size)])

    def test_control_publication_checks_starting_state(self):
        plan = self.control_plan(); self.state.write_text('{"manual":true}')
        with self.assertRaises(ValueError): migration.publish_controls(plan)
        self.assertEqual(self.state.read_text(), '{"manual":true}')

    def test_duplicate_control_updates_are_rejected_before_writing(self):
        plan=self.control_plan();plan['controls'].append(dict(plan['controls'][0]))
        with self.assertRaises(ValueError):migration.publish_controls(plan)
        self.assertEqual(self.state.read_text(),'{}')

    def test_control_publication_uses_existing_state_lock(self):
        plan = self.control_plan(); lock = Path(str(self.state)+'.lock'); lock.write_text('other publisher')
        with self.assertRaises(RuntimeError): migration.publish_controls(plan)
        self.assertEqual(self.state.read_text(), '{}'); self.assertTrue(lock.exists())

    def test_control_publication_is_atomic_and_removes_own_locks(self):
        plan = self.control_plan(); migration.publish_controls(plan)
        self.assertEqual(self.state.read_text(), '{"relocated":true}')
        self.assertFalse(Path(str(self.state)+'.lock').exists())

    def test_publication_phases_recheck_before_pointer_and_roll_back(self):
        from unittest.mock import patch
        plan=self.control_plan();plan['guard_checkpoints']='publication-phases'
        source=self.root/'observed.svg';source.write_text('art')
        plan['guards'].append(dict(path=str(source),bytes=3,sha256=migration.digest(source)))
        target=self.root/'navigation.html';target.write_text('old')
        prepared=self.root/'prepared.html';prepared.write_text('new')
        plan['controls'].insert(0,dict(path=str(target),source=str(prepared),role='control',expected_sha256=migration.digest(target),sha256=migration.digest(prepared),bytes=3))
        original=migration.os.replace
        def change_after_control(a,b):
            result=original(a,b)
            if str(b)==str(target) and target.read_text()=='new':source.write_text('manual')
            return result
        with patch.object(migration.os,'replace',side_effect=change_after_control):
            with self.assertRaisesRegex(ValueError,'advanced'):migration.publish_controls(plan)
        self.assertEqual(target.read_text(),'old');self.assertEqual(self.state.read_text(),'{}');self.assertEqual(source.read_text(),'manual')

    def test_publication_phase_mode_requires_a_pointer(self):
        plan=self.control_plan();plan['guard_checkpoints']='publication-phases';plan['controls'][0]['role']='control'
        with self.assertRaisesRegex(ValueError,'pointers'):migration.publish_controls(plan)

    def test_legacy_windows_paths_follow_scoped_storage_directories(self):
        from scripts.workflows.macos.run_legacy_with_paths import relocate
        old = '/Volumes/JOAQUIM/ClassicalConditioning-Mac-20261009/pc-outputs/frozen'
        target = '/Volumes/JOAQUIM/ClassicalConditioning Outputs/figures/figure-02/panel-g/versions/frozen'
        value = 'F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/frozen/panel.svg'
        self.assertEqual(relocate(value, '/Volumes/JOAQUIM', self.root,
                                  [(old, target)]), target + '/panel.svg')
        self.assertEqual(relocate('J:/Raw Data/allDelay', '/Volumes/JOAQUIM', self.root),
                         '/Volumes/JOAQUIM/Raw Data/allDelay')


if __name__ == '__main__':
    unittest.main()
