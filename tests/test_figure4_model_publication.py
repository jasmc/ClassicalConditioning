"""Publication guard invariants for immutable sources and editable outputs."""
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('trace_profile_builder', Path(__file__).resolve().parents[1] / 'scripts/build_figure4_3strace_model_draft.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PublicationGuardTests(unittest.TestCase):
    def fixture(self, directory):
        original, working = Path(directory) / 'original.parquet', Path(directory) / 'working.svg'
        for path in (original, working): path.write_bytes(b'first')
        expected = hashlib.sha256(b'first').hexdigest()
        snapshot = builder.PublicationSnapshot([original])
        snapshot.hashes = {str(p.resolve()): expected for p in (original, working)}
        return original, working, snapshot

    def test_original_is_hashed_in_each_attempt_and_changes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            original, working, snapshot = self.fixture(directory)
            with patch.object(builder, 'verify', wraps=builder.verify) as verifier:
                snapshot.check(); snapshot.check()
            calls = [c.args[0]['path'] for c in verifier.call_args_list]
            self.assertEqual(calls.count(str(original.resolve())), 1)
            self.assertEqual(calls.count(str(working.resolve())), 2)
            original.write_bytes(b'other')
            stamp = original.stat()
            os.utime(original, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1000000000))
            with self.assertRaises(builder.StaleFigureState): snapshot.check()
            fresh = builder.PublicationSnapshot([original])
            fresh.hashes = snapshot.hashes.copy()
            with self.assertRaises(ValueError): fresh.check()

    def test_manual_output_edits_fail_even_with_restored_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            original, working, snapshot = self.fixture(directory)
            snapshot.check()
            stamp = working.stat()
            working.write_bytes(b'other')
            os.utime(working, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            with self.assertRaises(ValueError): snapshot.check()


if __name__ == '__main__': unittest.main()
