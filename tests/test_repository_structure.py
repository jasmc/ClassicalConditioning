"""Synthetic ownership and provenance failures; no real data or SSD required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

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

    def test_nested_script_requires_its_full_catalogue_path(self):
        policy_path = self.repo / 'docs/project-maintenance/structure-policy.json'
        policy = json.loads(policy_path.read_text())
        policy['script_catalogue'] = 'scripts/README.md'
        policy_path.write_text(json.dumps(policy))
        nested = self.repo / 'scripts/paper/figure_01/panel_a'
        nested.mkdir(parents=True)
        (nested / 'build.py').write_text('print("synthetic")')
        catalogue = self.repo / 'scripts/README.md'
        catalogue.write_text('[build](build.py)')
        self.assertTrue(any('Uncatalogued script: paper/figure_01/panel_a/build.py' in issue for issue in structure.check(self.repo)['issues']))
        catalogue.write_text('[build](paper/figure_01/panel_a/build.py)')
        self.assertEqual(structure.check(self.repo)['issues'], [])

    def test_minimal_layout_passes_without_external_access(self):
        report = structure.check(self.repo)
        self.assertEqual(report['issues'], [])
        self.assertEqual(report['external_verification'], 'not-requested')

    def test_relocated_checkout_checks_local_science_and_draft_bytes(self):
        old = '/old/work/ClassicalConditioning'
        policy_path = self.repo/'docs/project-maintenance/structure-policy.json'
        policy = json.loads(policy_path.read_text())
        policy['recorded_repository_roots'] = [old]
        policy_path.write_text(json.dumps(policy))
        figure = self.repo/'docs/paper/figures/figure-01'
        figure.mkdir()
        selection = self.repo/'records/selection.json'
        selection.write_text('{}')
        legend = figure/'legend.md'
        legend.write_text('selected legend')
        configs = self.repo/'configs/paper-figures'
        configs.mkdir(parents=True)
        self.write('configs/paper-figures/behavior-paper.json', {'panels': {'fig-1F': {
            'scoped_selection_record': 'records/selection.json'}}})
        self.write('docs/paper/figures/figure-01/current-state.json', {
            'scientific_selection': {'records': [{'path': old+'/records/selection.json',
                                                'sha256': structure.sha256(selection)}]},
            'draft': {'legend': {'path': old+'/docs/paper/figures/figure-01/legend.md',
                                 'sha256': structure.sha256(legend)}}})
        (self.repo/'docs/paper/README.md').write_text('[Legend]('+old+'/docs/paper/figures/figure-01/legend.md)')
        self.assertEqual(structure.check(self.repo)['issues'], [])
        legend.write_text('unrecorded edit')
        self.assertIn('Bound figure record changed: '+str(legend.resolve()), structure.check(self.repo)['issues'])
        selection.write_text('changed scientific source')
        self.assertIn('Bound figure record changed: '+str(selection.resolve()), structure.check(self.repo)['issues'])

    def test_checkout_alias_is_exact_and_does_not_capture_external_storage(self):
        root = self.repo.resolve()
        aliases = ['/old/paper']
        self.assertEqual(structure.recorded_path('/old/paper/docs/guide.md', root, aliases), root/'docs/guide.md')
        self.assertEqual(structure.recorded_path('/old/paper-other/data.json', root, aliases), Path('/old/paper-other/data.json'))
        self.assertEqual(structure.recorded_path('/Volumes/JOAQUIM/data.json', root, aliases), Path('/Volumes/JOAQUIM/data.json'))
        with self.assertRaises(ValueError):
            structure.recorded_path('/old/paper/../other/data.json', root, aliases)

    def test_windows_external_bindings_are_not_local_mac_paths(self):
        panel = self.repo / 'docs/paper/figures/figure-01'
        panel.mkdir()
        for path in (r'J:\outputs\selected.svg', r'J:\\outputs\\selected.svg'):
            with self.subTest(path=path):
                self.write('docs/paper/figures/figure-01/current-state.json', {
                    'draft': {'working_svg': {'path': path, 'sha256': 'external'}}})
                self.assertEqual(structure.recorded_path(path, self.repo),
                                 Path('/Volumes/JOAQUIM/outputs/selected.svg'))
                self.assertEqual(structure.check(self.repo)['issues'], [])
                # This synthetic repository has no archive implementation;
                # never discover and audit a real host drive from the fixture.
                # The compatibility facade exports a function whose defining
                # globals belong to the maintained implementation module.
                with mock.patch.dict(structure.check.__globals__,
                        relocated_external_directory=mock.Mock(
                            side_effect=FileNotFoundError('Synthetic volume unavailable')),
                        resolve_artifact=mock.Mock(
                            side_effect=FileNotFoundError('Synthetic artifact unavailable'))):
                    report = structure.check(self.repo, external=True)
                self.assertTrue(report['unavailable_external_evidence'] or report['issues'])

    def test_doubled_windows_checkout_separators_use_local_alias(self):
        guide = self.repo / 'docs/paper/guide.md'
        guide.write_text('local source')
        alias = 'C:/Users/author/paper'
        recorded = r'C:\\Users\\author\\paper\\docs\\paper\\guide.md'
        self.assertEqual(structure.recorded_path(recorded, self.repo, [alias]), guide.resolve())
        with self.assertRaises(ValueError):
            structure.recorded_path(alias + '/../../escape', self.repo, [alias])

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

    def test_protected_external_links_resolve_exact_bytes_while_active_links_stay_direct(self):
        old = '/Volumes/JOAQUIM/retired-test-location/exact-source.py'
        retained = self.repo / 'records/retained.py'
        retained.write_bytes(b'exact historical source')
        historical = self.repo / 'docs/paper/historical.md'
        historical.write_text('[Source](' + old + ')')
        (self.repo / 'docs/paper/README.md').write_text('[Current source](' + old + ')')
        self.write('records/frozen-analyses/index.json', {'files': [
            {'path': 'docs/paper/historical.md', 'sha256': structure.sha256(historical)}]})
        self.write('records/repository-archive/relocations.json', {
            'repository_moves': {}, 'external_moves': {old: {
                'path': str(retained), 'bytes': retained.stat().st_size,
                'sha256': structure.sha256(retained)}}})
        # This fixture tests links without mounting or auditing real archives.
        with mock.patch.dict(structure.check.__globals__,
                joaquim_root=mock.Mock(return_value=self.repo/'unavailable-volume'),
                relocated_external_directory=mock.Mock(
                    side_effect=FileNotFoundError('Synthetic volume unavailable'))):
            report = structure.check(self.repo, external=True)
        self.assertEqual(report['issues'], [])
        self.assertNotIn('docs/paper/historical.md: ' + old, report['unavailable_external_evidence'])
        self.assertIn('docs/paper/README.md: ' + old, report['unavailable_external_evidence'])
        retained.write_bytes(b'corrupt historical source')
        with mock.patch.dict(structure.check.__globals__,
                joaquim_root=mock.Mock(return_value=self.repo/'unavailable-volume'),
                relocated_external_directory=mock.Mock(
                    side_effect=FileNotFoundError('Synthetic volume unavailable'))):
            report = structure.check(self.repo, external=True)
        self.assertTrue(any(issue.startswith('Historical external link changed: docs/paper/historical.md')
                            for issue in report['issues']))

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

    def test_nested_panel_legend_is_checked_without_whole_figure_selection_comparison(self):
        panel = self.repo / 'docs/paper/figures/figure-02/panel-g/exploratory'
        panel.mkdir(parents=True)
        legend = panel/'legend.md'
        legend.write_text('original legend')
        state = {'draft': {'legend': {'path': str(legend), 'sha256': structure.sha256(legend)}}}
        (panel/'current-state.json').write_text(json.dumps(state))
        self.assertEqual(structure.check(self.repo)['issues'], [])
        legend.write_text('manual edit')
        self.assertIn('Bound figure record changed: '+str(legend.resolve()), structure.check(self.repo)['issues'])

    def test_repository_skill_links_are_part_of_active_navigation(self):
        skill = self.repo/'.agents/skills/paper-figures'
        skill.mkdir(parents=True)
        (skill/'SKILL.md').write_text('[Workflow](../../../docs/paper/missing.md)')
        self.assertTrue(any('Broken active link: .agents/skills/' in issue
                            for issue in structure.check(self.repo)['issues']))

    def test_current_definition_and_display_mapping_must_match_bound_authority(self):
        figure = self.repo/'docs/paper/figures/figure-01'
        figure.mkdir()
        configs = self.repo/'configs/paper-figures'
        configs.mkdir(parents=True)
        selection = configs/'accepted.json'
        selection.write_text(json.dumps({'scientific_definition': {'bin_width_s': 1}}))
        self.write('configs/paper-figures/behavior-paper.json', {'panels': {'fig-1F': {
            'scoped_selection_record': 'configs/paper-figures/accepted.json',
            'scientific_definition': {'bin_width_s': 0.5},
            'presentation_mapping': {'display_panel': 'F', 'source_panel': 'F'}}}})
        self.write('docs/paper/figures/figure-01/current-state.json', {
            'scientific_selection': {'records': [{'path': str(selection), 'sha256': structure.sha256(selection)}]},
            'draft': {'condition_order': [{'display_panel': 'F', 'source_panel': 'H'}]}})
        issues = structure.check(self.repo)['issues']
        self.assertIn('Scientific definition/selection disagreement: fig-1F', issues)
        self.assertIn('Presentation mapping/state disagreement: fig-1F', issues)

    def test_mutable_index_keeps_exact_original_and_cannot_exempt_frozen_records(self):
        key = 'records/repository-archive/ARTIFACT_INDEX.md'
        live = self.repo/key
        live.write_text('old index')
        expected = structure.sha256(live)
        snapshot = self.repo/'records/repository-history/test/original.md'
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes(live.read_bytes())
        policy_path = self.repo/'docs/project-maintenance/structure-policy.json'
        policy = json.loads(policy_path.read_text())
        policy['mutable_index_originals'] = [{'path': key, 'sha256': expected,
            'preserved_original': str(snapshot.relative_to(self.repo))}]
        policy_path.write_text(json.dumps(policy))
        self.write('records/history/migration.json', {'protected_files': [{'path': key, 'sha256': expected}]})
        live.write_text('new authorized index entries')
        self.assertEqual(structure.check(self.repo)['issues'], [])
        self.write('records/frozen-analyses/index.json', {'files': [{'path': key, 'sha256': expected}]})
        self.assertIn('Preserved bytes changed: '+key, structure.check(self.repo)['issues'])
        self.write('records/frozen-analyses/index.json', {'files': []})
        snapshot.write_text('changed original')
        self.assertIn('Preserved index original changed: '+key, structure.check(self.repo)['issues'])

    def test_differing_protected_versions_are_not_silently_deduplicated(self):
        path = self.repo/'records/history/original.md'
        path.write_text('one version')
        self.write('records/history/migration.json', {'protected_files': [
            {'path': 'records/history/original.md', 'sha256': structure.sha256(path)},
            {'path': 'records/history/original.md', 'sha256': 'another-version'}]})
        self.assertIn('Preserved bytes changed: records/history/original.md', structure.check(self.repo)['issues'])

    def test_disposition_binding_is_checked_even_without_external_storage(self):
        path = self.repo/'records/repository-archive/disposition.json'
        path.write_text('{}')
        policy_path = self.repo/'docs/project-maintenance/structure-policy.json'
        policy = json.loads(policy_path.read_text())
        policy['archive_dispositions'] = [{'path': str(path.relative_to(self.repo)), 'sha256': structure.sha256(path)}]
        policy_path.write_text(json.dumps(policy))
        self.assertEqual(structure.check(self.repo)['issues'], [])
        path.write_text('{"altered":true}')
        self.assertTrue(any('Archive disposition changed' in issue for issue in structure.check(self.repo)['issues']))

    def test_exact_frozen_legend_recovery_cannot_hide_manual_draft_edits(self):
        panel = self.repo/'docs/paper/figures/figure-02/panel-a'
        panel.mkdir(parents=True)
        legend = panel/'legend.md'
        legend.write_text('frozen original legend')
        expected = structure.sha256(legend)
        original = self.repo/'records/history/legend.md'
        original.write_bytes(legend.read_bytes())
        migration = self.repo/'records/repository-history/test'
        migration.mkdir(parents=True)
        key = str(legend.relative_to(self.repo))
        (migration/'migration.json').write_text(json.dumps({'dispositions': [{
            'original_path': key, 'canonical_path': key, 'sha256': expected,
            'preserved_original': str(original.relative_to(self.repo))}]}))
        legend.write_text('current later legend')
        record = {'path': str(legend), 'sha256': expected}
        state_path = panel/'current-state.json'
        state_path.write_text(json.dumps({'frozen_display': {'legend': record}}))
        self.assertEqual(structure.check(self.repo)['issues'], [])
        state_path.write_text(json.dumps({'draft': {'legend': record}}))
        self.assertIn('Bound figure record changed: '+str(legend.resolve()), structure.check(self.repo)['issues'])
        # Even equal-valued records in the two scopes must be checked separately.
        state_path.write_text(json.dumps({'frozen_display': {'legend': record}, 'draft': {'legend': record}}))
        self.assertIn('Bound figure record changed: '+str(legend.resolve()), structure.check(self.repo)['issues'])

    def test_start_ticket_recovers_exact_history_without_hiding_draft_changes(self):
        panel = self.repo / 'docs/paper/figures/figure-01'
        panel.mkdir()
        ticket = self.repo / 'records/history/start.json'
        ticket.write_bytes(b'{"original": true}\r\n')
        expected = structure.sha256(ticket)
        original = self.repo / 'records/history/original-start.json'
        original.write_bytes(ticket.read_bytes())
        migration = self.repo / 'records/repository-history/ticket'
        migration.mkdir(parents=True)
        key = ticket.relative_to(self.repo).as_posix()
        (migration / 'migration.json').write_text(json.dumps({'dispositions': [{
            'original_path': key, 'canonical_path': key, 'sha256': expected,
            'preserved_original': original.relative_to(self.repo).as_posix()}]}))
        ticket.write_bytes(b'{"original": true}\n')
        record = {'path': key, 'sha256': expected}
        self.write('docs/paper/figures/figure-01/current-state.json', {'start_ticket': record})
        self.assertEqual(structure.check(self.repo)['issues'], [])
        self.write('docs/paper/figures/figure-01/current-state.json', {
            'start_ticket': record, 'draft': {'legend': record}})
        self.assertIn('Bound figure record changed: ' + str(ticket.resolve()), structure.check(self.repo)['issues'])
        original.write_bytes(b'changed history')
        self.assertTrue(structure.check(self.repo)['issues'])

    def test_script_catalogue_must_link_every_retained_tool(self):
        scripts = self.repo/'scripts'
        scripts.mkdir()
        (scripts/'review.py').write_text('"""retained review"""')
        (scripts/'README.md').write_text('# Tools')
        policy_path = self.repo/'docs/project-maintenance/structure-policy.json'
        policy = json.loads(policy_path.read_text())
        policy['script_catalogue'] = 'scripts/README.md'
        policy_path.write_text(json.dumps(policy))
        self.assertIn('Uncatalogued script: review.py', structure.check(self.repo)['issues'])
        (scripts/'README.md').write_text('[Review](review.py)')
        self.assertEqual(structure.check(self.repo)['issues'], [])


if __name__ == '__main__':
    unittest.main()
