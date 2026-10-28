import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from classical_conditioning.external_artifacts import relocated_external_directory, resolve_artifact
from classical_conditioning import external_artifacts as artifacts


class ExternalVolumePortabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.volume = self.root / 'volume'
        self.volume.mkdir()
        self.repo = self.root / 'repo'
        self.index = self.repo / 'records/repository-archive/relocations.json'
        self.index.parent.mkdir(parents=True)
        self.config = patch.dict(os.environ, {'CLASSICAL_CONDITIONING_JOAQUIM_ROOT': str(self.volume)})
        self.config.start()
        self.addCleanup(self.config.stop)

    def write_index(self, **value):
        self.index.write_text(json.dumps(value), encoding='utf-8')
        artifacts._external_moves.cache_clear()
        artifacts._external_directories.cache_clear()

    def test_mac_windows_and_configured_physical_references_share_exact_move(self):
        dest = self.volume / 'figures/selected.svg'
        dest.parent.mkdir()
        dest.write_bytes(b'exact')
        digest = artifacts.sha256(dest)
        self.write_index(external_moves={'/Volumes/JOAQUIM/old.svg': {
            'path': '/Volumes/JOAQUIM/figures/selected.svg', 'bytes': 5, 'sha256': digest}})
        for ref in ('/Volumes/JOAQUIM/old.svg', r'J:\old.svg', str(self.volume / 'old.svg')):
            self.assertEqual(resolve_artifact(ref, repo=self.repo, expected_sha256=digest), dest)
        dest.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            resolve_artifact(r'J:\old.svg', repo=self.repo, expected_sha256=digest)

    def test_directory_only_chains_and_historical_pc_alias(self):
        dest = self.volume / 'figures/current/file.svg'
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b'exact')
        self.write_index(external_directory_moves={
            '/Volumes/JOAQUIM/old': '/Volumes/JOAQUIM/intermediate',
            '/Volumes/JOAQUIM/intermediate': '/Volumes/JOAQUIM/figures/current',
            '/Volumes/JOAQUIM/ClassicalConditioning-Mac-20261009/pc-outputs/test': '/Volumes/JOAQUIM/figures/current'})
        for ref in ('/Volumes/JOAQUIM/old/file.svg', r'J:\old\file.svg',
                    'F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/test/file.svg'):
            self.assertEqual(resolve_artifact(ref, repo=self.repo, expected_sha256=artifacts.sha256(dest)), dest)

    def test_exact_chain_verifies_final_bytes_without_requiring_retired_intermediate(self):
        dest = self.volume / 'final.svg'
        dest.write_bytes(b'exact')
        row = {'bytes': 5, 'sha256': artifacts.sha256(dest)}
        self.write_index(external_moves={
            '/Volumes/JOAQUIM/old.svg': dict(row, path='/Volumes/JOAQUIM/middle.svg'),
            '/Volumes/JOAQUIM/middle.svg': dict(row, path='/Volumes/JOAQUIM/final.svg')})
        self.assertEqual(resolve_artifact('/Volumes/JOAQUIM/old.svg', repo=self.repo), dest)
        self.write_index(external_moves={
            '/Volumes/JOAQUIM/old.svg': dict(row, path='/Volumes/JOAQUIM/middle.svg'),
            '/Volumes/JOAQUIM/middle.svg': dict(row, path='/Volumes/JOAQUIM/old.svg')})
        with self.assertRaisesRegex(ValueError, 'cycle'):
            resolve_artifact('/Volumes/JOAQUIM/old.svg', repo=self.repo)

    def test_traversal_directory_cycles_and_symlink_escape(self):
        for ref in ('/Volumes/JOAQUIM/../escape', r'J:\..\escape', '../escape', r'..\escape'):
            with self.assertRaises(ValueError):
                resolve_artifact(ref, repo=self.repo)
        self.write_index(external_directory_moves={'/Volumes/JOAQUIM/old': '/Volumes/JOAQUIM/old/child'})
        with self.assertRaisesRegex(ValueError, 'cycle'):
            relocated_external_directory('/Volumes/JOAQUIM/old', repo=self.repo)
        self.write_index()
        outside = self.root / 'outside'
        outside.mkdir()
        try:
            (self.volume / 'escape').symlink_to(outside, target_is_directory=True)
        except OSError as error:
            if getattr(error, 'winerror', None) == 1314:
                self.skipTest('Windows account lacks symlink creation privilege; traversal/cycle assertions passed')
            raise
        with self.assertRaisesRegex(ValueError, 'escapes'):
            resolve_artifact('/Volumes/JOAQUIM/escape/file', repo=self.repo)

    def test_detection_requires_unique_mounted_identity_and_honors_explicit_root(self):
        other = self.root / 'second-volume'
        other.mkdir()
        with patch.dict(os.environ, {}, clear=True), patch.object(artifacts, '_joaquim_roots', return_value=[]):
            with self.assertRaises(FileNotFoundError):
                artifacts.joaquim_root()
        with patch.dict(os.environ, {}, clear=True), patch.object(artifacts, '_joaquim_roots', return_value=[self.volume, other]), patch('os.path.ismount', return_value=True):
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                artifacts.joaquim_root()
        with patch.object(artifacts, '_joaquim_roots', side_effect=AssertionError('Explicit configuration must take precedence')):
            self.assertEqual(artifacts.joaquim_root(), self.volume)

    @unittest.skipUnless(os.name=='nt','Windows drive-letter discovery')
    def test_detected_alternate_letter_round_trips_without_configuration(self):
        dest=self.volume/'selected.svg';dest.write_bytes(b'exact')
        digest=artifacts.sha256(dest)
        self.write_index(external_moves={'/Volumes/JOAQUIM/old.svg':{
            'path':'/Volumes/JOAQUIM/selected.svg','sha256':digest,'bytes':5}})
        with patch.dict(os.environ,{},clear=True),patch.object(artifacts,'_windows_joaquim_roots',return_value=[Path('K:/')]),patch.object(artifacts,'joaquim_root',return_value=self.volume):
            self.assertEqual(resolve_artifact('K:/old.svg',repo=self.repo,expected_sha256=digest),dest)

    def test_missing_archived_reference_requires_hash_when_versions_are_ambiguous(self):
        for name, raw in [('older', b'old'), ('newer', b'new')]:
            directory = self.index.parent / name
            directory.mkdir()
            payload = self.root / name / 'payload/missing.txt'
            payload.parent.mkdir(parents=True)
            payload.write_bytes(raw)
            (directory / 'transfer-inventory.json').write_text(json.dumps({
                'archive_root': str(self.root / name), 'files': [{'path': 'missing.txt',
                    'action': 'archive_to_ssd', 'sha256': artifacts.sha256(payload), 'bytes': len(raw)}]}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            resolve_artifact('missing.txt', repo=self.repo)
        expected = hashlib.sha256(b'old').hexdigest()
        self.assertEqual(resolve_artifact('missing.txt', repo=self.repo, expected_sha256=expected),
                         self.root / 'older/payload/missing.txt')

    def test_active_draft_cannot_hide_manual_edit_with_historical_recovery(self):
        active = self.repo / 'working.svg'
        active.write_bytes(b'manual edit')
        original = self.volume / 'original.svg'
        original.write_bytes(b'previous version')
        expected = artifacts.sha256(original)
        (self.index.parent / 'verified-artifact-locations.json').write_text(json.dumps({'entries': [{
            'references': ['working.svg'], 'sha256': expected,
            'archive_root': str(self.volume), 'relative_path': 'original.svg'}]}), encoding='utf-8')
        self.assertEqual(resolve_artifact('working.svg', expected_sha256=expected, repo=self.repo), original)
        with self.assertRaisesRegex(ValueError, 'Active artifact'):
            resolve_artifact('working.svg', expected_sha256=expected, repo=self.repo, allow_historical=False)
        self.assertEqual(active.read_bytes(), b'manual edit')

    def test_windows_reference_can_request_exact_historical_mac_identity(self):
        original = self.volume / 'original.svg'
        original.write_bytes(b'previous version')
        expected = artifacts.sha256(original)
        (self.index.parent / 'verified-artifact-locations.json').write_text(json.dumps({'entries': [{
            'references': ['/Volumes/JOAQUIM/selected.svg'], 'sha256': expected,
            'archive_root': '/Volumes/JOAQUIM', 'relative_path': 'original.svg'}]}), encoding='utf-8')
        self.assertEqual(resolve_artifact(r'J:\selected.svg', expected_sha256=expected, repo=self.repo), original)

    def test_relocated_exact_bytes_and_missing_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume = root / 'volume'
            volume.mkdir()
            repo = root / 'repo'
            index = repo / 'records/repository-archive/relocations.json'
            index.parent.mkdir(parents=True)
            dest = volume / 'figures/current.svg'
            dest.parent.mkdir()
            dest.write_bytes(b'artwork')
            digest = hashlib.sha256(b'artwork').hexdigest()
            old = '/Volumes/JOAQUIM/old/artwork.svg'
            index.write_text(json.dumps({
                'external_moves': {old: {'path': '/Volumes/JOAQUIM/figures/current.svg',
                                         'sha256': digest, 'bytes': 7}},
                'external_directory_moves': {'/Volumes/JOAQUIM/old': '/Volumes/JOAQUIM/figures'},
            }), encoding='utf-8')
            with patch.dict(os.environ, {'CLASSICAL_CONDITIONING_JOAQUIM_ROOT': str(volume)}):
                self.assertEqual(resolve_artifact(old, expected_sha256=digest, repo=repo), dest)
                self.assertEqual(relocated_external_directory('/Volumes/JOAQUIM/old', repo=repo), dest.parent)
                with self.assertRaises(ValueError):
                    resolve_artifact(old, expected_sha256='0' * 64, repo=repo)
            with patch.dict(os.environ, {'CLASSICAL_CONDITIONING_JOAQUIM_ROOT': str(root / 'missing')}):
                with self.assertRaises(FileNotFoundError):
                    resolve_artifact(old, expected_sha256=digest, repo=repo)

    def test_authenticated_repository_paths_disable_checkout_conversion(self):
        repo = Path(__file__).resolve().parents[1]
        policy = json.loads((repo / 'docs/project-maintenance/structure-policy.json').read_text(encoding='utf-8'))
        paths = {row['path'] for row in json.loads((repo / 'records/frozen-analyses/index.json').read_text(encoding='utf-8'))['files']}
        manifests = [repo / policy['migration_manifest'], *(repo / 'records/repository-history').glob('*/migration.json')]
        for manifest in manifests:
            if manifest.is_file():
                paths.update(row['path'] for row in json.loads(manifest.read_text(encoding='utf-8')).get('protected_files', []))
        result = subprocess.run(['git', 'check-attr', '-z', '--stdin', 'text'], cwd=repo,
                                input=b''.join(path.encode('utf-8') + b'\0' for path in sorted(paths)),
                                capture_output=True, check=True)
        fields = result.stdout.split(b'\0')[:-1]
        failures = [fields[i].decode('utf-8') for i in range(0, len(fields), 3) if fields[i + 2] != b'unset']
        self.assertEqual(failures, [])

    def test_metric_workspace_uses_directory_moves_for_both_host_references(self):
        from classical_conditioning.analysis.review import metric_comparison_storage as storage
        self.write_index(external_directory_moves={
            '/Volumes/JOAQUIM/old-review':'/Volumes/JOAQUIM/figures/metric-review'})
        expected=self.volume/'figures/metric-review';expected.parent.mkdir()
        def relocated(value):
            return artifacts.relocated_external_directory(value,repo=self.repo)
        with patch.object(storage,'relocated_external_directory',side_effect=relocated):
            for reference in ('/Volumes/JOAQUIM/old-review',r'J:\old-review'):
                self.assertEqual(storage.workspace(reference),expected.resolve())


class CheckoutAndLockPortabilityTests(unittest.TestCase):
    def test_all_protected_index_blobs_survive_both_checkout_settings(self):
        repo = Path(__file__).resolve().parents[1]
        attrs = (repo / '.gitattributes').read_bytes()
        paths = [json.loads(line.rsplit(' -text', 1)[0]).lstrip('/')
                 for line in attrs.decode().splitlines() if line and not line.startswith('#')]
        def git(*args, cwd=repo, data=None):
            return subprocess.run(['git', '-c', 'core.fsmonitor=false', *args], cwd=cwd,
                                  input=data, capture_output=True, check=True, timeout=90).stdout
        initial_index = git('ls-files', '--stage', '-z')
        indexed = {p.decode('utf-8') for p in git('ls-files', '-z').split(b'\0') if p}
        indexed_paths = [p for p in paths if p in indexed]
        batch = git('cat-file', '--batch', data=b''.join((':' + p + '\n').encode() for p in indexed_paths))
        offset, blobs = 0, {}
        for path in indexed_paths:
            end = batch.index(b'\n', offset)
            header = batch[offset:end]
            self.assertFalse(header.endswith(b' missing'), path)
            size = int(header.rsplit(b' ', 1)[1])
            blobs[path] = batch[end + 1:end + 1 + size]
            offset = end + 2 + size
        # New authenticated inputs need the same protection before their first
        # commit. Never replace an indexed blob with a dirty working-file copy.
        for path in paths:
            if path not in indexed:
                blobs[path] = (repo / path).read_bytes()
        for autocrlf in ('true', 'false'):
            with self.subTest(autocrlf=autocrlf), tempfile.TemporaryDirectory() as folder:
                checkout = Path(folder)
                git('init', '--quiet', cwd=checkout)
                git('config', 'core.autocrlf', autocrlf, cwd=checkout)
                git('config', 'core.eol', 'crlf' if autocrlf == 'true' else 'lf', cwd=checkout)
                (checkout / '.gitattributes').write_bytes(attrs)
                for path, raw in blobs.items():
                    target = checkout / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                git('-c', 'core.safecrlf=false', 'add', '--all', cwd=checkout)
                before = git('ls-files', '--stage', '-z', cwd=checkout)
                output = checkout / 'export'
                output.mkdir()
                git('checkout-index', '--all', '--prefix=' + str(output) + os.sep, cwd=checkout)
                for path, raw in blobs.items():
                    self.assertEqual((output / path).read_bytes(), raw, path)
                self.assertEqual(git('ls-files', '--stage', '-z', cwd=checkout), before)
        self.assertEqual(git('ls-files', '--stage', '-z'), initial_index)

    def test_independent_run_lock_contends_across_processes_and_releases_after_error(self):
        from classical_conditioning.analysis.inference.panel_g_run import run_lock
        code = '''import sys
from pathlib import Path
from classical_conditioning.analysis.inference.panel_g_run import run_lock
try:
    with run_lock(Path(sys.argv[1])): pass
except ValueError: sys.exit(19)
'''
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def child():
                return subprocess.run([sys.executable, '-c', code, folder], env=env,
                                      capture_output=True, timeout=30)
            with self.assertRaisesRegex(RuntimeError, 'test cleanup'):
                with run_lock(root):
                    result = child()
                    self.assertEqual(result.returncode, 19, result.stderr.decode(errors='replace'))
                    raise RuntimeError('test cleanup')
            result = child()
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))


if __name__ == '__main__':
    unittest.main()
