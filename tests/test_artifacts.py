from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from classical_conditioning.artifacts import (
    publish_transaction,
    sha256_file,
    verify_completed_analysis_parquet_set,
    write_json_atomic,
)
from classical_conditioning.exceptions import ArtifactIntegrityError


class ArtifactHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_atomic_json_replaces_content_without_incomplete_files(self) -> None:
        output = self.root / "Metadata" / "report.json"
        write_json_atomic(output, {"version": 1})
        write_json_atomic(output, {"version": 2})

        self.assertEqual(
            json.loads(output.read_text(encoding="utf-8")),
            {"version": 2},
        )
        self.assertEqual(
            list(output.parent.glob(f".{output.name}.*.incomplete")),
            [],
        )

    def test_atomic_json_cleans_temporary_file_when_publish_fails(self) -> None:
        output = self.root / "Metadata" / "report.json"
        output.parent.mkdir()

        with patch(
            "classical_conditioning.artifacts.os.replace",
            side_effect=OSError("simulated failure"),
        ):
            with self.assertRaisesRegex(OSError, "simulated failure"):
                write_json_atomic(output, {"version": 1})

        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.iterdir()), [])

    def test_transaction_rejects_duplicate_final_paths_before_mutation(self) -> None:
        staging = self.root / "staging"
        staging.mkdir()
        first = staging / "first"
        second = staging / "second"
        first.write_text("first", encoding="utf-8")
        second.write_text("second", encoding="utf-8")
        final = self.root / "final"
        final.write_text("existing", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "duplicate final"):
            publish_transaction(
                ((first, final), (second, final)),
                staging,
                overwrite=True,
            )

        self.assertEqual(final.read_text(encoding="utf-8"), "existing")
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())

    def test_transaction_rejects_missing_staged_path_before_backup(self) -> None:
        staging = self.root / "staging"
        staging.mkdir()
        missing = staging / "missing"
        final = self.root / "final"
        final.write_text("existing", encoding="utf-8")

        with self.assertRaisesRegex(FileNotFoundError, "missing"):
            publish_transaction(
                ((missing, final),),
                staging,
                overwrite=True,
            )

        self.assertEqual(final.read_text(encoding="utf-8"), "existing")
        self.assertFalse(
            staging.with_name(f"{staging.name}-backups").exists()
        )

    def test_transaction_rejects_same_staged_and_final_path(self) -> None:
        staging = self.root / "staging"
        staging.mkdir()
        artifact = staging / "artifact"
        artifact.write_text("content", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "must be distinct"):
            publish_transaction(
                ((artifact, artifact),),
                staging,
                overwrite=True,
            )

        self.assertEqual(artifact.read_text(encoding="utf-8"), "content")

    def test_transaction_rejects_empty_publication(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            publish_transaction((), self.root / "staging", overwrite=False)

    def test_transaction_removes_stale_artifact_with_publication(self) -> None:
        staging = self.root / "staging"
        staging.mkdir()
        staged = staging / "new"
        staged.write_text("new", encoding="utf-8")
        final = self.root / "final"
        final.write_text("old", encoding="utf-8")
        stale = self.root / "stale"
        stale.write_text("stale", encoding="utf-8")

        publish_transaction(
            ((staged, final),),
            staging,
            overwrite=True,
            removals=(stale,),
        )

        self.assertEqual(final.read_text(encoding="utf-8"), "new")
        self.assertFalse(stale.exists())

    def test_transaction_restores_removal_when_publication_fails(self) -> None:
        staging = self.root / "staging"
        staging.mkdir()
        staged = staging / "new"
        staged.write_text("new", encoding="utf-8")
        final = self.root / "final"
        stale = self.root / "stale"
        stale.write_text("stale", encoding="utf-8")

        real_replace = __import__("os").replace

        def fail_publication(source: Path, destination: Path) -> None:
            if Path(source) == staged:
                raise OSError("simulated publication failure")
            real_replace(source, destination)

        with patch(
            "classical_conditioning.artifacts.os.replace",
            side_effect=fail_publication,
        ):
            with self.assertRaisesRegex(OSError, "simulated publication failure"):
                publish_transaction(
                    ((staged, final),),
                    staging,
                    overwrite=True,
                    removals=(stale,),
                )

        self.assertFalse(final.exists())
        self.assertEqual(stale.read_text(encoding="utf-8"), "stale")

    def test_analysis_artifact_verification_authenticates_cohort_identity(
        self,
    ) -> None:
        artifact = self.root / "result.parquet"
        artifact.write_bytes(b"lossless artifact")
        digest = sha256_file(artifact)
        summary = self.root / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "recipe": "analysis-v1",
                    "analysis_id": "cohort-a",
                    "recording_ids": ["recording-a", "recording-b"],
                    "alignment": "CS",
                    "artifacts": {
                        "result": {
                            "path": str(artifact),
                            "sha256": digest,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        marker = self.root / "marker.json"
        marker.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "recipe": "analysis-v1",
                    "analysis_id": "cohort-a",
                    "recording_ids": ["recording-a", "recording-b"],
                    "alignment": "CS",
                    "artifact_sha256": {"result": digest},
                    "summary_sha256": sha256_file(summary),
                }
            ),
            encoding="utf-8",
        )

        verified = verify_completed_analysis_parquet_set(
            {"result": artifact},
            summary,
            marker,
            recipe="analysis-v1",
            analysis_id="cohort-a",
            recording_ids=("recording-a", "recording-b"),
            alignment="CS",
        )

        self.assertEqual(verified.data_paths, {"result": artifact.resolve()})
        with self.assertRaisesRegex(ArtifactIntegrityError, "lineage is invalid"):
            verify_completed_analysis_parquet_set(
                {"result": artifact},
                summary,
                marker,
                recipe="analysis-v1",
                analysis_id="cohort-a",
                recording_ids=("recording-b", "recording-a"),
                alignment="CS",
            )


if __name__ == "__main__":
    unittest.main()
