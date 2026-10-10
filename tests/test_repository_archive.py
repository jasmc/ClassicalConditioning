"""Archive transactions use small synthetic payloads, never real scientific data."""
import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from classical_conditioning.external_artifacts import resolve_artifact, sha256, external_output

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('repository_archive', ROOT/'scripts/repository_archive.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root/'repo'; self.repo.mkdir()
        self.source = self.repo/'reviews/result.csv';self.source.parent.mkdir();self.source.write_bytes(b'fish,value\nA,1\n')
        self.dest = self.root/'ssd/archive'
        self.manifest = dict(schema_version=1, archive_root=str(self.dest), files=[dict(path='reviews/result.csv',bytes=self.source.stat().st_size,sha256=sha256(self.source),action='archive_to_ssd',state='pending_transfer')])

    def transfer(self):
        return archive.transfer(self.manifest,self.repo,self.dest,check_mount=False)

    def test_missing_drive_changes_nothing(self):
        with patch.object(archive.os.path,'ismount',return_value=False):
            with self.assertRaises(FileNotFoundError): archive.transfer(self.manifest,self.repo,self.dest)
        self.assertTrue(self.source.exists()); self.assertFalse(self.dest.parent.exists())

    def test_copy_verifies_but_does_not_prune(self):
        self.assertEqual(self.transfer(),1);self.assertTrue(self.source.exists())
        self.assertEqual(archive.verify_payload(self.manifest,self.dest),1)
        result=archive.prune(self.manifest,self.repo,self.dest,check_mount=False)
        self.assertEqual(result['removed_bytes'],self.manifest['files'][0]['bytes']);self.assertFalse(self.source.exists())

    def test_changed_source_before_transfer_is_rejected(self):
        self.source.write_text('different')
        with self.assertRaises(ValueError):self.transfer()
        self.assertFalse(self.dest.exists())

    def test_changed_source_after_copy_is_not_deleted(self):
        self.transfer();self.source.write_text('different')
        with self.assertRaises(ValueError):archive.prune(self.manifest,self.repo,self.dest,check_mount=False)
        self.assertTrue(self.source.exists())

    def test_archive_corruption_prevents_prune(self):
        self.transfer();(self.dest/'payload/reviews/result.csv').write_text('bad')
        with self.assertRaises(ValueError):archive.prune(self.manifest,self.repo,self.dest,check_mount=False)
        self.assertTrue(self.source.exists())

    def test_destination_collision_is_not_overwritten(self):
        self.dest.mkdir(parents=True);(self.dest/'keep').write_text('existing')
        with self.assertRaises(FileExistsError):self.transfer()
        self.assertEqual((self.dest/'keep').read_text(),'existing')

    def test_interrupted_copy_resumes(self):
        real_copy=archive.shutil.copyfile
        with patch.object(archive.shutil,'copyfile',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):self.transfer()
        self.assertTrue(self.dest.with_name('archive.staging').exists())
        self.assertEqual(self.transfer(),1);self.assertTrue(self.source.exists())

    def test_changed_inventory_cannot_resume_staging(self):
        with patch.object(archive.shutil,'copyfile',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):self.transfer()
        self.manifest['date']='changed'
        with self.assertRaises(ValueError):self.transfer()

    def test_parent_traversal_rejected(self):
        self.manifest['files'][0]['path']='../escape'
        with self.assertRaises(ValueError):self.transfer()

    def test_embedded_archive_hash_mismatch(self):
        html=self.root/'review.html';html.write_text('<script id="frozen-archive">'+json.dumps({'files':{'code.py':{'base64':base64.b64encode(b'exact').decode(),'sha256':'incorrect'}}})+'</script>')
        with self.assertRaises(ValueError):archive.embedded_checks(html)

    def test_historical_windows_reference_resolves_and_checks_hash(self):
        metadata=self.repo/'docs/maintenance/archive';metadata.mkdir(parents=True)
        (metadata/'transfer-inventory.json').write_text(json.dumps(self.manifest))
        self.assertEqual(resolve_artifact(r'C:\Users\joaquim\Documents\ClassicalConditioning\reviews\result.csv',repo=self.repo),self.source)
        self.transfer();self.source.unlink()
        self.assertEqual(resolve_artifact('reviews/result.csv',repo=self.repo),self.dest/'payload/reviews/result.csv')
        (self.dest/'payload/reviews/result.csv').write_text('bad')
        with self.assertRaises(ValueError):resolve_artifact('reviews/result.csv',repo=self.repo)

    def test_followup_inventory_resolves_without_rewriting_original_checkpoint(self):
        metadata = self.repo/'docs/maintenance/archive'
        metadata.mkdir(parents=True)
        original = metadata/'transfer-inventory.json'
        original.write_text(json.dumps(dict(archive_root='original-archive', files=[
            dict(path='reviews/result.csv', sha256='old-retained-version', action='retain')
        ])))
        original_bytes = original.read_bytes()
        followup = metadata/'scripts-cleanup-20261010'
        followup.mkdir()
        (followup/'transfer-inventory.json').write_text(json.dumps(self.manifest))
        self.transfer()
        self.source.unlink()
        resolved = resolve_artifact('reviews/result.csv', repo=self.repo)
        self.assertEqual(resolved, self.dest/'payload/reviews/result.csv')
        self.assertEqual(original.read_bytes(), original_bytes)
        resolved.write_text('corrupted follow-up')
        with self.assertRaises(ValueError):
            resolve_artifact('reviews/result.csv', repo=self.repo)

    def test_new_output_does_not_fabricate_volume(self):
        with patch.dict('os.environ',{},clear=True),patch('os.path.ismount',return_value=False):
            with self.assertRaises(FileNotFoundError):external_output('reviews/new.html')
        with patch.dict('os.environ',{'CLASSICAL_CONDITIONING_ARTIFACT_ROOT':str(ROOT)}):
            with self.assertRaises(ValueError):external_output('reviews/new.html')

    def test_prune_without_publication_preserves_source(self):
        with self.assertRaises(FileNotFoundError):archive.prune(self.manifest,self.repo,self.dest,check_mount=False)
        self.assertTrue(self.source.exists())


if __name__=='__main__':unittest.main()
