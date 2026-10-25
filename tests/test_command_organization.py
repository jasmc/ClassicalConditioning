"""Maintenance contracts: no scientific datasets or browser sessions required."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from classical_conditioning.analysis.review import metric_comparison_storage as storage
from classical_conditioning.reviews.embedded_downloads import with_embedded_downloads

ROOT = Path(__file__).resolve().parents[1]


class MetricStorage(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_one_external_workspace_and_repository_rejection(self):
        target = self.root / 'review'
        with patch.object(storage, 'external_output', return_value=target):
            self.assertEqual(storage.workspace(), storage.workspace(target))
        with self.assertRaisesRegex(ValueError, 'outside'):
            storage.workspace(ROOT / 'outputs/review')
        with self.assertRaises(FileNotFoundError):
            storage.workspace(self.root / 'missing/review')
        self.assertFalse((self.root / 'missing').exists())

    def test_unmounted_volume_is_not_created(self):
        with patch.object(storage.os.path, 'ismount', return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, 'volume unavailable'):
                storage.workspace('/Volumes/JOAQUIM/unavailable/review')

    def test_consumer_authenticates_producer_receipt_and_preserves_manual_edits(self):
        for name in ['fish_window_summary.csv', 'fish_paired_change.csv']:
            (self.root / name).write_text('fish_id,value\nf1,1\n')
        storage.bind_inputs(self.root)
        self.assertEqual(storage.verified_input(self.root), self.root / 'fish_paired_change.csv')
        (self.root / 'fish_paired_change.csv').write_text('manual revision')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            storage.verified_input(self.root)
        with self.assertRaises(FileExistsError):
            storage.require_new_outputs(self.root, ['fish_paired_change.csv'])
        self.assertEqual((self.root / 'fish_paired_change.csv').read_text(), 'manual revision')

    def test_older_input_needs_explicit_identity(self):
        p = self.root / 'fish_paired_change.csv'
        p.write_bytes(b'old saved input')
        self.assertEqual(storage.verified_input(self.root, hashlib.sha256(p.read_bytes()).hexdigest()), p)
        with self.assertRaises(ValueError):
            storage.verified_input(self.root, '0' * 64)


class ScopedCommands(unittest.TestCase):
    def test_reusable_helpers_import_without_external_storage(self):
        modules = ['figures.review_html', 'figures.svg_scope', 'figures.layout_primitives',
                   'figures.figure_01.legacy_layout', 'figures.figure_01.legacy_display',
                   'figures.figure_01.selected_fgh', 'reviews.figure2_g_sources',
                   'analysis.review.bout_block_review']
        code = 'import importlib\n' + '\n'.join('importlib.import_module(' + repr('classical_conditioning.' + m) + ')' for m in modules)
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), CLASSICAL_CONDITIONING_ARTIFACT_ROOT='/unavailable-not-a-real-root')
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_guard_tracks_extracted_helper_changes(self):
        from classical_conditioning import command_provenance
        from classical_conditioning.figure_workspace import Snapshot, StaleFigureState
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / 'src/classical_conditioning'
            package.mkdir(parents=True)
            (package / '__init__.py').write_text('')
            helper = package / 'helper.py'
            helper.write_text('def value(): return 1\n')
            command = root / 'command.py'
            command.write_text('from classical_conditioning.helper import value\n')
            with patch.object(command_provenance, 'REPOSITORY', root):
                command_provenance.command_sources.cache_clear()
                sources = command_provenance.command_sources(command)
            self.assertIn(helper.resolve(), sources)
            snapshot = Snapshot(sources)
            helper.write_text('def value(): return 2\n')
            with self.assertRaises(StaleFigureState):
                snapshot.check()
        command_provenance.command_sources.cache_clear()

    def test_local_script_imports_resolve_after_command_moves(self):
        for home in ['src', 'scripts', 'tests']:
            for path in (ROOT / home).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text())):
                    names = (['scripts.' + a.name for a in node.names] if isinstance(node, ast.ImportFrom) and node.module == 'scripts'
                             else [node.module] if isinstance(node, ast.ImportFrom) and node.module
                             else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                    for name in names:
                        if name.startswith('scripts.'):
                            target = ROOT / name.replace('.', '/')
                            self.assertTrue(target.with_suffix('.py').is_file() or target.is_dir(),
                                            str(path.relative_to(ROOT)) + ': missing ' + name)

    def test_three_script_import_cycles_are_gone(self):
        paths = list((ROOT / 'scripts').rglob('*.py'))
        modules = {p.relative_to(ROOT).as_posix()[:-3].replace('/', '.'): p for p in paths}
        graph = {m: set() for m in modules}
        for m, p in modules.items():
            for n in ast.walk(ast.parse(p.read_text())):
                if isinstance(n, ast.ImportFrom) and n.module:
                    if n.module in modules:
                        graph[m].add(n.module)
                    for alias in n.names:
                        candidate = n.module + '.' + alias.name
                        if candidate in modules:
                            graph[m].add(candidate)
                elif isinstance(n, ast.Import):
                    graph[m].update(a.name for a in n.names if a.name in modules)
        visiting, done = set(), set()
        def visit(m):
            self.assertNotIn(m, visiting, 'Circular script import: ' + m)
            if m in done:
                return
            visiting.add(m)
            for child in graph[m]:
                visit(child)
            visiting.remove(m)
            done.add(m)
        for m in graph:
            visit(m)

    def test_core_and_metric_help_have_no_storage_side_effects(self):
        commands = ['scripts/check_repository_structure.py', 'scripts/repository_archive.py',
                    'scripts/freeze_figure.py', 'scripts/main_paper_figures.py',
                    'scripts/workflows/metrics/compare_train_test_windows.py',
                    'scripts/workflows/metrics/compare_response_magnitude.py',
                    'scripts/paper/figure_03/run_blup_comparison.py',
                    'scripts/paper/figure_03/build_blup_study.py']
        env = dict(os.environ, CLASSICAL_CONDITIONING_ARTIFACT_ROOT='/unavailable-not-a-real-root')
        for command in commands:
            result = subprocess.run([sys.executable, str(ROOT / command), '--help'], cwd=ROOT.parent,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, (command, result.stderr))

    def test_embedded_download_handler_is_idempotent_and_preserves_notebook_and_payload(self):
        raw = '<html><body><a download="x" href="data:text/plain;base64,YQ==">x</a><script>const storageKey="unchanged";</script></body></html>'
        result = with_embedded_downloads(raw)
        self.assertIn('storageKey="unchanged"', result)
        self.assertIn('href="data:text/plain;base64,YQ=="', result)
        self.assertEqual(with_embedded_downloads(result), result)
        self.assertEqual(result.count('id="embedded-source-downloads"'), 1)

    def test_extracted_bout_calculation_preserves_empty_window_and_units(self):
        from classical_conditioning.analysis.review.bout_block_review import validate_calculation
        validate_calculation()

    def test_browser_edit_modes_fail_before_any_browser_launch(self):
        node = shutil.which('node')
        bundled = Path('/Users/joaquim/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if node is None and bundled.is_file():
            node = str(bundled)
        if node is None:
            self.skipTest('Node is unavailable')
        module = ROOT / 'src/classical_conditioning/figures/browser/command_options.cjs'
        code = "const a=require('assert'),m=require(" + json.dumps(str(module)) + ");a.equal(m.parseMode([]).verify,true);a.equal(m.parseMode(['--apply']).verify,false);a.throws(()=>m.parseMode(['--verify','--apply']));a.throws(()=>m.parseMode(['--center-labels']));a.equal(m.parseMode(['--center-labels'],{allowCenter:true}).centerOnly,true);a.throws(()=>m.parseMode(['--aply']));"
        result = subprocess.run([node, '-e', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ['verify_nature_presentation.cjs', 'verify_annotations.cjs', 'verify_axis_geometry.cjs']:
            result = subprocess.run([node, str(ROOT / 'scripts/paper/shared' / name), 'unused', 'unused', '--verify', '--apply'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Verification cannot be combined', result.stderr)


if __name__ == '__main__':
    unittest.main()
